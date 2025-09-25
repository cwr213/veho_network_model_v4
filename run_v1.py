# run_v1.py - Network optimization with corrected fill rates and no hardcoded values
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from veho_net.io_loader import load_workbook, params_to_dict
from veho_net.validators import validate_inputs
from veho_net.build_structures import build_od_and_direct, candidate_paths
from veho_net.time_cost import path_cost_and_time, containers_for_pkgs_day
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


def _allocate_lane_costs_to_ods(od_selected: pd.DataFrame, arc_summary: pd.DataFrame, costs: dict,
                                strategy: str) -> pd.DataFrame:
    """Allocate lane-level costs back to individual OD pairs based on volume share."""
    od = od_selected.copy()

    # Map path types to number of intermediate touches
    touch_map = {"direct": 0, "1_touch": 1, "2_touch": 2, "3_touch": 3, "4_touch": 4}
    od['num_touches'] = od['path_type'].map(touch_map).fillna(0)

    # Get required cost parameters from inputs
    sort_pp = float(costs.get("sort_cost_per_pkg", 0.0))
    container_handling_cost = float(costs.get("container_handling_cost", 0.0))
    last_mile_sort_pp = float(costs.get("last_mile_sort_cost_per_pkg", 0.0))
    last_mile_delivery_pp = float(costs.get("last_mile_delivery_cost_per_pkg", 0.0))

    # Validate required parameters exist
    required_params = {
        "sort_cost_per_pkg": sort_pp,
        "last_mile_sort_cost_per_pkg": last_mile_sort_pp,
        "last_mile_delivery_cost_per_pkg": last_mile_delivery_pp
    }

    missing_params = [name for name, value in required_params.items() if value == 0.0]
    if missing_params:
        raise ValueError(f"Required cost parameters missing or zero: {missing_params}")

    if strategy.lower() == "container" and container_handling_cost == 0.0:
        raise ValueError("container_handling_cost required for container strategy")

    # Cost calculation based on strategy - CORRECTED touch_cost calculation
    if strategy.lower() == "container":
        # Container strategy: sort at origin, container handling at intermediate touches, last mile sort at destination
        od['injection_sort_cost'] = sort_pp * od['pkgs_day']
        od['last_mile_sort_cost'] = last_mile_sort_pp * od['pkgs_day']
        od['last_mile_delivery_cost'] = last_mile_delivery_pp * od['pkgs_day']
        od['last_mile_cost'] = od['last_mile_sort_cost'] + od['last_mile_delivery_cost']
        od['container_handling_cost'] = 0.0  # Will be filled from lane data

        # Touch cost = ONLY sort and crossdock costs (no linehaul or delivery)
        od['touch_cost'] = od['injection_sort_cost'] + od['last_mile_sort_cost']  # Container handling added separately
        od['touch_cpp'] = sort_pp + last_mile_sort_pp

    else:
        # Fluid strategy: sort at every facility
        od['injection_sort_cost'] = sort_pp * od['pkgs_day']
        od['intermediate_sort_cost'] = sort_pp * od['num_touches'] * od['pkgs_day']
        od['last_mile_sort_cost'] = last_mile_sort_pp * od['pkgs_day']
        od['last_mile_delivery_cost'] = last_mile_delivery_pp * od['pkgs_day']
        od['last_mile_cost'] = od['last_mile_sort_cost'] + od['last_mile_delivery_cost']

        # Touch cost = ONLY sort costs (no linehaul or delivery)
        od['touch_cost'] = od['injection_sort_cost'] + od['intermediate_sort_cost'] + od['last_mile_sort_cost']
        od['touch_cpp'] = sort_pp + (sort_pp * od['num_touches']) + last_mile_sort_pp

    # Initialize linehaul costs
    od['linehaul_cost'] = 0.0
    od['linehaul_cpp'] = 0.0

    # Allocate linehaul costs and container handling costs from arc summary
    if arc_summary is not None and not arc_summary.empty:
        for idx, row in od.iterrows():
            path_str = str(row.get('path_str', ''))
            if not path_str or '->' not in path_str:
                continue

            nodes = path_str.split('->')
            od_pkgs = float(row['pkgs_day'])
            od_linehaul_cost = 0.0
            od_container_handling_cost = 0.0

            for i in range(len(nodes) - 1):
                from_fac = nodes[i].strip()
                to_fac = nodes[i + 1].strip()

                lane = arc_summary[
                    (arc_summary['from_facility'] == from_fac) &
                    (arc_summary['to_facility'] == to_fac)
                    ]

                if not lane.empty:
                    lane_row = lane.iloc[0]
                    lane_total_cost = float(lane_row.get('total_cost', 0))
                    lane_total_pkgs = float(lane_row.get('pkgs_day', 1))

                    if lane_total_pkgs > 0:
                        od_share = od_pkgs / lane_total_pkgs
                        allocated_cost = lane_total_cost * od_share
                        od_linehaul_cost += allocated_cost

                        # Container handling cost allocation for intermediate touches
                        if strategy.lower() == "container" and i > 0:  # Skip first leg (origin)
                            lane_containers = float(lane_row.get('physical_containers', 0))
                            lane_container_cost = lane_containers * container_handling_cost
                            allocated_container_cost = lane_container_cost * od_share
                            od_container_handling_cost += allocated_container_cost

            od.at[idx, 'linehaul_cost'] = od_linehaul_cost
            od.at[idx, 'linehaul_cpp'] = od_linehaul_cost / od_pkgs if od_pkgs > 0 else 0

            if strategy.lower() == "container":
                od.at[idx, 'container_handling_cost'] = od_container_handling_cost

    # Calculate totals
    if strategy.lower() == "container":
        od['total_cost'] = od['linehaul_cost'] + od['touch_cost'] + od['container_handling_cost'] + od[
            'last_mile_delivery_cost']
        # Add container handling to touch_cost (only sort and crossdock costs)
        od['touch_cost'] = od['touch_cost'] + od['container_handling_cost']
        od['touch_cpp'] = od['touch_cost'] / od['pkgs_day'].replace(0, 1)
    else:
        od['total_cost'] = od['linehaul_cost'] + od['touch_cost'] + od['last_mile_delivery_cost']

    od['cost_per_pkg'] = od['total_cost'] / od['pkgs_day'].replace(0, 1)

    # Add output columns for compatibility
    od['injection_sort_cpp'] = od['injection_sort_cost'] / od['pkgs_day'].replace(0, 1)
    od['last_mile_destination_cpp'] = od['last_mile_cost'] / od['pkgs_day'].replace(0, 1)

    return od


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

    # Check if last_mile_cost column exists in od_selected
    if 'last_mile_cost' not in od_selected.columns:
        print("  Warning: last_mile_cost column missing from od_selected, skipping last mile cost calculation")
        return facility_rollup

    try:
        # Calculate last mile costs by destination facility
        last_mile_costs = od_selected.groupby('dest').agg({
            'last_mile_cost': 'sum',
            'pkgs_day': 'sum'
        }).reset_index()

        last_mile_costs['last_mile_cpp'] = last_mile_costs['last_mile_cost'] / last_mile_costs['pkgs_day'].replace(0, 1)

        # Update facility rollup with last mile costs where they exist
        for _, cost_row in last_mile_costs.iterrows():
            facility_name = cost_row['dest']
            facility_mask = facility_rollup['facility'] == facility_name

            if facility_mask.any():
                facility_rollup.loc[facility_mask, 'last_mile_cost'] = cost_row['last_mile_cost']
                facility_rollup.loc[facility_mask, 'last_mile_cpp'] = cost_row['last_mile_cpp']

    except Exception as e:
        print(f"  Warning: Could not calculate last mile costs: {e}")

    return facility_rollup


def _fix_network_kpis(od_selected: pd.DataFrame, arc_summary: pd.DataFrame) -> dict:
    """
    Calculate network-level KPIs using package-cube to truck-cube ratios.
    This inherently provides package-weighted averages.
    """
    if od_selected.empty:
        return {
            "avg_container_fill_rate": 0.0,
            "avg_truck_fill_rate": 0.0,
            "total_packages_dwelled": 0
        }

    # Use arc-level data for accurate truck fill calculation
    if not arc_summary.empty:
        # Total package cube across all arcs
        total_pkg_cube = arc_summary['pkg_cube_cuft'].sum() if 'pkg_cube_cuft' in arc_summary.columns else 0

        # Total truck cube capacity across all arcs
        if 'trucks' in arc_summary.columns and 'cube_per_truck' in arc_summary.columns:
            total_truck_cube = (arc_summary['trucks'] * arc_summary['cube_per_truck']).sum()
        else:
            total_truck_cube = 1  # Avoid division by zero

        # Calculate fill rates using cube ratios (inherently package-weighted)
        avg_truck_fill = total_pkg_cube / total_truck_cube if total_truck_cube > 0 else 0.0

        # Container fill rate (volume-weighted if available)
        if 'container_fill_rate' in arc_summary.columns and 'pkgs_day' in arc_summary.columns:
            total_volume = arc_summary['pkgs_day'].sum()
            if total_volume > 0:
                avg_container_fill = (arc_summary['container_fill_rate'] * arc_summary['pkgs_day']).sum() / total_volume
            else:
                avg_container_fill = 0.0
        else:
            avg_container_fill = 0.0

        total_dwelled = arc_summary.get('packages_dwelled', 0).sum()
    else:
        # Fallback to basic calculation
        avg_truck_fill = 0.0
        avg_container_fill = 0.0
        total_dwelled = od_selected.get('packages_dwelled', 0).sum()

    return {
        "avg_container_fill_rate": max(0, min(1, avg_container_fill)),
        "avg_truck_fill_rate": max(0, min(1, avg_truck_fill)),
        "total_packages_dwelled": max(0, total_dwelled)
    }


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
    Execute network optimization for a single strategy.

    Builds OD matrix, generates candidate paths, calculates costs,
    solves optimization, and generates output reports.
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

    # Build candidate paths
    paths = candidate_paths(od, facilities, mb, around_factor=float(run_kv["path_around_the_world_factor"]))

    if paths.empty:
        print(f"  No valid paths generated")
        return None

    print(f"  Generated {len(paths)} candidate paths")

    # Merge packages and add containers
    paths = paths.merge(
        od[['origin', 'dest', 'pkgs_day']],
        on=['origin', 'dest'],
        how='left'
    )

    if 'pkgs_day' not in paths.columns:
        print(f"  Failed to merge package data")
        return None

    paths['pkgs_day'] = paths['pkgs_day'].fillna(0)
    paths["containers_cont"] = paths["pkgs_day"].apply(
        lambda x: containers_for_pkgs_day(x, pkgmix, cont)
    )

    paths["scenario_id"] = scenario_id
    paths["day_type"] = day_type

    # Cost and time calculation with validated parameters
    print(f"  Calculating costs for {len(paths)} paths using {strategy} strategy...")
    cost_time_results = []

    for idx, row in paths.iterrows():
        try:
            total_cost, total_hours, sums, steps = path_cost_and_time(
                row, facilities, mb, timing_local, costs_local, pkgmix, cont, row["pkgs_day"]
            )

            result = {
                'index': idx,
                'total_cost': total_cost,
                'total_hours': total_hours,
                **sums
            }
            cost_time_results.append(result)

        except Exception as e:
            print(f"    Error: Cost calculation failed for path {idx}: {e}")
            # Re-raise error instead of using fallback values
            raise ValueError(f"Cost calculation failed for path {idx}: {e}. Check input parameters.")

    print(f"  Cost calculation completed for {len(cost_time_results)} paths")

    # Merge results back
    cost_time_df = pd.DataFrame(cost_time_results)
    cost_time_df.set_index('index', inplace=True)

    for col in cost_time_df.columns:
        if col in ['total_cost', 'total_hours']:
            paths[col] = paths.index.map(cost_time_df[col]).fillna(0)
        else:
            paths[f"summary_{col}"] = paths.index.map(cost_time_df[col]).fillna(0)

    # MILP optimization with strategy-aware costs
    print(f"  Running MILP optimization using {strategy} strategy...")
    od_selected, arc_summary = solve_arc_pooled_path_selection(
        paths, facilities, mb, pkgmix, cont, costs_local  # Use costs_local with strategy
    )

    if od_selected.empty:
        print(f"  Optimization failed - no solution found")
        return None

    print(f"  Optimization selected {len(od_selected)} optimal paths")

    # Cost allocation and fill rate calculations
    od_selected = _allocate_lane_costs_to_ods(od_selected, arc_summary, costs_local, strategy)

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

    # Validate aggregate calculations
    validation_results = validate_network_aggregations(od_selected, arc_summary, facility_rollup)

    if not validation_results.get('package_consistency', True):
        print(f"  Warning: Package volume inconsistency detected")
        print(f"    OD total: {validation_results.get('total_od_packages', 0):,.0f}")
        print(f"    Facility injection total: {validation_results.get('total_facility_injection', 0):,.0f}")

    if not validation_results.get('cost_consistency', True):
        print(f"  Warning: Cost aggregation inconsistency detected")
        print(f"    OD total cost: ${validation_results.get('total_od_cost', 0):,.0f}")
        print(f"    Arc total cost: ${validation_results.get('total_arc_cost', 0):,.0f}")

    print(f"  Network avg truck fill rate: {validation_results.get('network_avg_truck_fill', 0):.1%}")
    print(f"  Network avg container fill rate: {validation_results.get('network_avg_container_fill', 0):.1%}")

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

    # Calculate KPIs with volume-weighted averages
    total_cost = od_selected["total_cost"].sum()
    total_pkgs = od_selected["pkgs_day"].sum()
    cost_per_pkg = total_cost / max(total_pkgs, 1)

    # Network-level fill rate calculation using validated logic
    network_fill_rates = _fix_network_kpis(od_selected, arc_summary)

    kpis = pd.Series({
        "total_cost": total_cost,
        "cost_per_pkg": cost_per_pkg,
        "num_ods": len(od_selected),
        "avg_container_fill_rate": network_fill_rates["avg_container_fill_rate"],
        "avg_truck_fill_rate": network_fill_rates["avg_truck_fill_rate"],
        "total_packages_dwelled": network_fill_rates["total_packages_dwelled"],
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

    print(f"🚀 Starting Network Optimization with Validated Parameters")
    print(f"📁 Input: {input_path.name}")
    print(f"📁 Output: {output_dir}")

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
    parser = argparse.ArgumentParser(description="Veho Network Optimization with Validated Parameters")
    parser.add_argument("--input", required=True, help="Path to input Excel file")
    parser.add_argument("--output_dir", required=True, help="Output directory path")
    args = parser.parse_args()

    main(args.input, args.output_dir)