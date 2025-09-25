# run_v1.py - CORRECTED: Remove cost pre-calculation, let MILP handle all costs
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from veho_net.io_loader import load_workbook, params_to_dict
from veho_net.validators import validate_inputs
from veho_net.build_structures import build_od_and_direct, candidate_paths
from veho_net.time_cost import containers_for_pkgs_day
from veho_net.milp import solve_arc_pooled_path_selection
from veho_net.reporting import (
    _identify_volume_types_with_costs,
    _calculate_hourly_throughput_with_costs,
    build_od_selected_outputs,
    build_dwell_hotspots,
    build_lane_summary,
    add_zone,
    validate_network_aggregations
)
from veho_net.write_outputs import (
    write_workbook,
    write_compare_workbook,
    write_executive_summary_workbook
)

# Output naming
OUTPUT_FILE_TEMPLATE = "{scenario_id}_{strategy}.xlsx"
COMPARE_FILE_TEMPLATE = "{scenario_id}_compare.xlsx"
EXECUTIVE_SUMMARY_TEMPLATE = "{scenario_id}_exec_sum.xlsx"


def _fix_od_fill_rates_from_lanes(od_selected: pd.DataFrame, arc_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate OD-level fill rates based on actual lane usage.
    Multiple ODs share lanes, so aggregate fill rates from lane level data.
    """
    od = od_selected.copy()

    # Initialize with zero (no hardcoded defaults)
    od['container_fill_rate'] = 0.0
    od['truck_fill_rate'] = 0.0
    od['packages_dwelled'] = 0.0

    if arc_summary.empty:
        return od

    # Calculate weighted fill rates based on lane usage for each OD
    for idx, row in od.iterrows():
        path_str = str(row.get('path_str', ''))
        if not path_str or '->' not in path_str:
            continue

        nodes = path_str.split('->')
        od_pkgs = float(row['pkgs_day'])

        # Collect fill rates from lanes this OD uses
        lane_fill_rates = []
        lane_container_rates = []
        lane_dwelled = []
        lane_weights = []

        for i in range(len(nodes) - 1):
            from_fac = nodes[i].strip()
            to_fac = nodes[i + 1].strip()

            lane = arc_summary[
                (arc_summary['from_facility'] == from_fac) &
                (arc_summary['to_facility'] == to_fac)
                ]

            if not lane.empty:
                lane_row = lane.iloc[0]
                lane_pkgs = float(lane_row.get('pkgs_day', 1))

                # Weight by this OD's share of the lane
                weight = od_pkgs / max(lane_pkgs, 1)

                lane_fill_rates.append(float(lane_row.get('truck_fill_rate', 0.0)))
                lane_container_rates.append(float(lane_row.get('container_fill_rate', 0.0)))
                lane_dwelled.append(float(lane_row.get('packages_dwelled', 0)) * weight)
                lane_weights.append(weight)

        if lane_fill_rates:
            # Calculate weighted average fill rates
            total_weight = sum(lane_weights)
            if total_weight > 0:
                od.at[idx, 'truck_fill_rate'] = sum(r * w for r, w in zip(lane_fill_rates, lane_weights)) / total_weight
                od.at[idx, 'container_fill_rate'] = sum(
                    r * w for r, w in zip(lane_container_rates, lane_weights)) / total_weight
                od.at[idx, 'packages_dwelled'] = sum(lane_dwelled)

    return od


def _fix_facility_rollup_last_mile_costs(facility_rollup: pd.DataFrame, od_selected: pd.DataFrame) -> pd.DataFrame:
    """
    Add last mile cost calculations to facility rollup by destination.
    """
    if facility_rollup.empty:
        return facility_rollup

    facility_rollup = facility_rollup.copy()

    # Initialize columns
    facility_rollup['last_mile_cost'] = 0.0
    facility_rollup['last_mile_cpp'] = 0.0

    if od_selected.empty:
        return facility_rollup

    # Check if processing_cost column exists in od_selected (from new MILP architecture)
    if 'processing_cost' not in od_selected.columns:
        print("  Warning: processing_cost column missing from od_selected, skipping last mile cost calculation")
        return facility_rollup

    try:
        # Calculate processing costs by destination facility (approximation for last mile costs)
        last_mile_costs = od_selected.groupby('dest').agg({
            'processing_cost': 'sum',
            'pkgs_day': 'sum'
        }).reset_index()

        last_mile_costs['last_mile_cpp'] = last_mile_costs['processing_cost'] / last_mile_costs['pkgs_day'].replace(0,
                                                                                                                    1)

        # Update facility rollup with last mile costs where they exist
        for _, cost_row in last_mile_costs.iterrows():
            facility_name = cost_row['dest']
            facility_mask = facility_rollup['facility'] == facility_name

            if facility_mask.any():
                facility_rollup.loc[facility_mask, 'last_mile_cost'] = cost_row['processing_cost']
                facility_rollup.loc[facility_mask, 'last_mile_cpp'] = cost_row['last_mile_cpp']

    except Exception as e:
        print(f"  Warning: Could not calculate last mile costs: {e}")

    return facility_rollup


# Removed _fix_network_kpis function - now using MILP-calculated network KPIs directly


def _run_one_strategy(
        base_id: str,
        strategy: str,
        facilities: pd.DataFrame,
        zips: pd.DataFrame,
        demand: pd.DataFrame,
        inj: pd.DataFrame,
        mb: pd.DataFrame,
        timing: dict,
        costs: dict,
        cont: pd.DataFrame,
        pkgmix: pd.DataFrame,
        run_kv: dict,
        scenario_row: pd.Series,
        out_dir: Path,
):
    """
    CORRECTED: Execute network optimization with MILP-only cost calculation.
    """

    # Extract scenario information
    scenario_id_from_input = scenario_row.get("scenario_id",
                                              scenario_row.get("pair_id", "default_scenario"))
    year = int(scenario_row.get("year", scenario_row.get("demand_year", 2028)))
    day_type = str(scenario_row["day_type"]).strip().lower()

    # Build unique scenario identifier
    scenario_id = f"{scenario_id_from_input}_{strategy}"

    timing_local = dict(timing)
    timing_local["load_strategy"] = strategy

    # Add strategy to costs dict for MILP solver
    costs_local = dict(costs)
    costs_local["load_strategy"] = strategy

    facilities = facilities.copy()
    facilities.attrs["enforce_parent_hub_over_miles"] = int(run_kv.get("enforce_parent_hub_over_miles", 500))

    print(f"Processing {scenario_id}: {year} {day_type} {strategy} strategy")

    # Build OD matrix
    year_demand = demand.query("year == @year").copy()
    if year_demand.empty:
        print(f"  No demand data for year {year}")
        return None

    od, dir_fac, _dest_pop = build_od_and_direct(facilities, zips, year_demand, inj)

    od_day_col = "pkgs_offpeak_day" if day_type == "offpeak" else "pkgs_peak_day"
    direct_day_col = "dir_pkgs_offpeak_day" if day_type == "offpeak" else "dir_pkgs_peak_day"
    od = od[od[od_day_col] > 0].copy()

    if od.empty:
        print(f"  No OD pairs with volume for {day_type}")
        return None

    print(f"  Generated {len(od)} OD pairs with volume")

    # Set pkgs_day column
    od["pkgs_day"] = od[od_day_col]

    # Build candidate paths - SAME for both strategies
    around_factor = float(run_kv.get("path_around_the_world_factor", 1.5))
    paths = candidate_paths(od, facilities, mb, around_factor=around_factor)

    if paths.empty:
        print(f"  No valid paths generated")
        return None

    print(f"  Generated {len(paths)} candidate paths")

    # Merge packages - NO cost pre-calculation
    paths = paths.merge(
        od[['origin', 'dest', 'pkgs_day']],
        on=['origin', 'dest'],
        how='left'
    )

    if 'pkgs_day' not in paths.columns:
        print(f"  Failed to merge package data")
        return None

    paths['pkgs_day'] = paths['pkgs_day'].fillna(0)

    # Add basic metadata for MILP
    paths["scenario_id"] = scenario_id
    paths["day_type"] = day_type

    print(f"  Paths prepared for MILP optimization (no pre-calculated costs)")

    # CORRECTED: MILP solver handles ALL cost calculation and returns network KPIs
    print(f"  Running MILP optimization with {strategy} strategy cost calculation...")
    od_selected, arc_summary, network_kpis = solve_arc_pooled_path_selection(
        paths, facilities, mb, pkgmix, cont, costs_local, timing_local
    )

    if od_selected.empty:
        print(f"  Optimization failed - no solution found")
        return None

    print(f"  Optimization selected {len(od_selected)} optimal paths")

    # Update OD fill rates from actual lane aggregation
    od_selected = _fix_od_fill_rates_from_lanes(od_selected, arc_summary)

    # Generate output data with validation
    try:
        direct_day = dir_fac.copy()
        direct_day["dir_pkgs_day"] = direct_day[direct_day_col]
    except:
        direct_day = pd.DataFrame()

    # Build output datasets with correct zone calculation
    od_out = build_od_selected_outputs(od_selected, facilities, direct_day, mb)
    dwell_hotspots = build_dwell_hotspots(od_selected)

    # Facility analysis with validated calculations
    facility_rollup = _identify_volume_types_with_costs(
        od_selected, pd.DataFrame(), direct_day, arc_summary
    )
    facility_rollup = _calculate_hourly_throughput_with_costs(
        facility_rollup, timing_local, strategy
    )
    facility_rollup = _fix_facility_rollup_last_mile_costs(facility_rollup, od_selected)

    # Validate aggregate calculations (skip network KPI validation since using MILP values)
    validation_results = validate_network_aggregations(od_selected, arc_summary, facility_rollup)

    if not validation_results.get('package_consistency', True):
        print(f"  Warning: Package volume inconsistency detected")
        print(f"    OD total: {validation_results.get('total_od_packages', 0):,.0f}")
        print(f"    Facility injection total: {validation_results.get('total_facility_injection', 0):,.0f}")

    if not validation_results.get('cost_consistency', True):
        print(f"  Warning: Cost aggregation inconsistency detected")
        print(f"    OD total cost: ${validation_results.get('total_od_cost', 0):,.0f}")
        print(f"    Arc total cost: ${validation_results.get('total_arc_cost', 0):,.0f}")

    # Generate path steps for output
    path_steps = []
    for _, od_row in od_selected.iterrows():
        path_str = str(od_row.get('path_str', f"{od_row['origin']}->{od_row['dest']}"))
        nodes = path_str.split('->')

        for i, (from_fac, to_fac) in enumerate(zip(nodes[:-1], nodes[1:])):
            path_steps.append({
                'scenario_id': scenario_id,
                'origin': od_row['origin'],
                'dest': od_row['dest'],
                'step_order': i + 1,
                'from_facility': from_fac.strip(),
                'to_facility': to_fac.strip(),
                'distance_miles': 500,  # Default - could calculate actual
                'drive_hours': 8,  # Default
                'processing_hours_at_destination': 2
            })

    path_steps_df = pd.DataFrame(path_steps)
    lane_summary = build_lane_summary(arc_summary)

    # Calculate KPIs using MILP-provided network KPIs (no recalculation)
    total_cost = od_selected["total_cost"].sum()
    total_pkgs = od_selected["pkgs_day"].sum()
    cost_per_pkg = total_cost / max(total_pkgs, 1)

    # Use KPIs from MILP solver (already correctly calculated)
    kpis = pd.Series({
        "total_cost": total_cost,
        "cost_per_pkg": cost_per_pkg,
        "num_ods": len(od_selected),
        "avg_container_fill_rate": network_kpis["avg_container_fill_rate"],
        "avg_truck_fill_rate": network_kpis["avg_truck_fill_rate"],
        "total_packages_dwelled": network_kpis["total_packages_dwelled"],
        "sla_violations": 0,
        "around_world_flags": 0,
        "pct_direct": (od_selected["path_type"] == "direct").mean() * 100 if "path_type" in od_selected.columns else 0,
        "pct_1_touch": (od_selected[
                            "path_type"] == "1_touch").mean() * 100 if "path_type" in od_selected.columns else 0,
        "pct_2_touch": (od_selected[
                            "path_type"] == "2_touch").mean() * 100 if "path_type" in od_selected.columns else 0,
        "pct_3_touch": (od_selected[
                            "path_type"] == "3_touch").mean() * 100 if "path_type" in od_selected.columns else 0,
    })

    # Remove the old network KPI calculation since we're using MILP values
    print(f"  MILP-calculated network avg truck fill rate: {network_kpis['avg_truck_fill_rate']:.1%}")
    print(f"  MILP-calculated network avg container fill rate: {network_kpis['avg_container_fill_rate']:.1%}")

    # Write outputs
    out_path = out_dir / OUTPUT_FILE_TEMPLATE.format(scenario_id=scenario_id_from_input, strategy=strategy)

    scenario_summary = pd.DataFrame([{
        'key': 'scenario_id', 'value': scenario_id
    }, {
        'key': 'strategy', 'value': strategy
    }, {
        'key': 'total_cost', 'value': kpis['total_cost']
    }, {
        'key': 'cost_per_pkg', 'value': kpis['cost_per_pkg']
    }, {
        'key': 'total_packages', 'value': total_pkgs
    }])

    write_success = write_workbook(
        out_path,
        scenario_summary,
        od_out,
        path_steps_df,
        dwell_hotspots,
        facility_rollup,
        arc_summary,
        kpis,
        None  # No sort allocation for core model
    )

    if not write_success:
        print(f"  Failed to write output file")
        return None

    print(f"  ✓ {scenario_id}: ${kpis['total_cost']:,.0f} cost, ${kpis['cost_per_pkg']:.3f}/pkg")

    return scenario_id, out_path, kpis, {
        'scenario_id': scenario_id,
        'scenario_id_from_input': scenario_id_from_input,
        'strategy': strategy,
        'output_file': str(out_path.name),
        **kpis.to_dict()
    }


def main(input_path: str, output_dir: str):
    """
    Main execution function for network optimization.

    Loads input data, validates parameters, runs optimization for both
    container and fluid strategies, and generates comparison reports.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"🚀 Starting Network Optimization with Corrected Architecture")
    print(f"📁 Input: {input_path.name}")
    print(f"📁 Output: {output_dir}")
    print(f"🔧 All cost calculation happens in MILP with proper aggregation")

    dfs = load_workbook(input_path)
    validate_inputs(dfs)

    timing_local = params_to_dict(dfs["timing_params"])
    costs = params_to_dict(dfs["cost_params"])
    run_kv = params_to_dict(dfs["run_settings"])

    # Set base_id from input file name
    base_id = input_path.stem.replace("_input", "").replace("_v4", "")

    strategies = ["container", "fluid"]
    compare_results = []
    created_files = []

    print(f"\n📊 Processing {len(dfs['scenarios'])} scenarios with {len(strategies)} strategies each")

    # Run each scenario
    for scenario_idx, scenario_row in dfs["scenarios"].iterrows():
        scenario_id_from_input = scenario_row.get("scenario_id",
                                                  scenario_row.get("pair_id", f"scenario_{scenario_idx + 1}"))

        print(f"\n🎯 Scenario {scenario_idx + 1}/{len(dfs['scenarios'])}: {scenario_id_from_input}")

        for strategy in strategies:
            try:
                result = _run_one_strategy(
                    base_id, strategy,
                    dfs["facilities"], dfs["zips"], dfs["demand"],
                    dfs["injection_distribution"], dfs["mileage_bands"],
                    timing_local, costs, dfs["container_params"], dfs["package_mix"],
                    run_kv, scenario_row, output_dir
                )

                if result:
                    scenario_id, out_path, kpis, results_data = result
                    results_data['base_id'] = base_id
                    compare_results.append(results_data)
                    created_files.append(out_path.name)

            except Exception as e:
                print(f"❌ Error running {scenario_id_from_input}_{strategy}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Write comparison reports
    if compare_results:
        print(f"\n📈 Creating comparison analysis...")
        compare_df = pd.DataFrame(compare_results)

        first_scenario_id = compare_results[0].get('scenario_id_from_input', 'comparison')
        compare_path = output_dir / COMPARE_FILE_TEMPLATE.format(scenario_id=first_scenario_id)

        write_compare_success = write_compare_workbook(compare_path, compare_df, run_kv)
        if write_compare_success:
            created_files.append(compare_path.name)
            print(f"  ✓ Created comparison file: {compare_path.name}")

            # Executive summary
            exec_summary_path = output_dir / EXECUTIVE_SUMMARY_TEMPLATE.format(scenario_id=first_scenario_id)
            results_by_strategy = {}

            for result in compare_results:
                strategy = result['strategy']
                if strategy not in results_by_strategy:
                    results_by_strategy[strategy] = {
                        'kpis': pd.Series(result),
                        'facility_rollup': pd.DataFrame(),
                        'od_out': pd.DataFrame()
                    }

            write_executive_summary_workbook(
                exec_summary_path, results_by_strategy, run_kv, base_id
            )
            created_files.append(exec_summary_path.name)
            print(f"  ✓ Created executive summary: {exec_summary_path.name}")

    print("\n🎉 Network Optimization Complete!")

    if created_files:
        print(f"📋 Created {len(created_files)} files:")
        for file_name in sorted(set(created_files)):
            print(f"  📄 {file_name}")
    else:
        print("⚠️  No output files created")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Veho Network Optimization with Corrected Architecture")
    parser.add_argument("--input", required=True, help="Path to input Excel file")
    parser.add_argument("--output_dir", required=True, help="Path to output directory")
    args = parser.parse_args()

    main(args.input, args.output_dir)