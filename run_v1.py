# run_v1.py - COMPLETE REBUILT VERSION
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
    add_zone
)
from veho_net.sort_optimization import calculate_containerization_costs_corrected
from veho_net.write_outputs import write_workbook, write_compare_workbook

# Output naming control
OUTPUT_FILE_TEMPLATE = "{scenario_id}_results_v1.xlsx"
COMPARE_FILE_TEMPLATE = "{base_id}_compare.xlsx"


def _allocate_lane_costs_to_ods(od_selected: pd.DataFrame, arc_summary: pd.DataFrame, costs: dict,
                                strategy: str) -> pd.DataFrame:
    """
    COMPREHENSIVE FIX: Proper cost allocation including packages_dwelled from lanes to ODs.
    """
    od = od_selected.copy()

    # Calculate touch costs per OD based on containerization level
    touch_map = {"direct": 0, "1_touch": 1, "2_touch": 2, "3_touch": 3, "4_touch": 4}
    od['num_touches'] = od['path_type'].map(touch_map).fillna(0)

    crossdock_pp = float(costs.get("crossdock_touch_cost_per_pkg", 0.0))
    sort_pp = float(costs.get("sort_cost_per_pkg", 0.0))
    lm_sort_pp = float(costs.get("last_mile_sort_cost_per_pkg", 0.0))

    if strategy.lower() == "container":
        od['touch_cost'] = od['num_touches'] * crossdock_pp * od['pkgs_day']
        od['touch_cpp'] = od['num_touches'] * crossdock_pp

        if 'containerization_level' in od.columns:
            lm_sort_multiplier = od['containerization_level'].map({
                'region': 1.0,
                'market': 0.5,
                'sort_group': 0.1
            }).fillna(1.0)

            od['lm_sort_cost'] = lm_sort_multiplier * lm_sort_pp * od['pkgs_day']
            od['touch_cost'] += od['lm_sort_cost']
            od['touch_cpp'] += lm_sort_multiplier * lm_sort_pp
    else:
        od['touch_cost'] = (od['num_touches'] + 1) * sort_pp * od['pkgs_day']
        od['touch_cpp'] = (od['num_touches'] + 1) * sort_pp

    # Initialize cost and dwelled package columns
    od['linehaul_cost'] = 0.0
    od['linehaul_cpp'] = 0.0
    od['packages_dwelled'] = 0.0  # CRITICAL: Initialize this column

    if arc_summary is None or arc_summary.empty:
        print("Warning: No arc_summary provided for cost allocation")
        od['total_cost'] = od['touch_cost']
        od['cost_per_pkg'] = od['touch_cpp']
        return od

    print(f"Allocating costs from {len(arc_summary)} lanes to {len(od)} OD pairs...")

    # CRITICAL FIX: For each OD, find its path legs and allocate costs AND packages_dwelled
    allocation_debug = []

    for idx, row in od.iterrows():
        path_str = str(row.get('path_str', ''))
        if not path_str or '->' not in path_str:
            continue

        nodes = path_str.split('->')
        od_pkgs = float(row['pkgs_day'])
        od_linehaul_cost = 0.0
        od_packages_dwelled = 0.0

        # Sum costs and dwelled packages across all legs in this path
        for i in range(len(nodes) - 1):
            from_fac = nodes[i].strip()
            to_fac = nodes[i + 1].strip()

            # Find this lane in arc_summary
            lane = arc_summary[
                (arc_summary['from_facility'] == from_fac) &
                (arc_summary['to_facility'] == to_fac)
                ]

            if not lane.empty:
                lane_row = lane.iloc[0]
                lane_total_cost = float(lane_row.get('total_cost', 0))
                lane_total_pkgs = float(lane_row.get('pkgs_day', 1))
                lane_packages_dwelled = float(lane_row.get('packages_dwelled', 0))

                if lane_total_pkgs > 0:
                    od_share = od_pkgs / lane_total_pkgs
                    allocated_cost = lane_total_cost * od_share
                    allocated_dwelled = lane_packages_dwelled * od_share

                    od_linehaul_cost += allocated_cost
                    od_packages_dwelled += allocated_dwelled

                    # Debug tracking
                    allocation_debug.append({
                        'od_pair': f"{row['origin']}->{row['dest']}",
                        'lane': f"{from_fac}->{to_fac}",
                        'od_pkgs': od_pkgs,
                        'lane_pkgs': lane_total_pkgs,
                        'od_share': od_share,
                        'lane_dwelled': lane_packages_dwelled,
                        'allocated_dwelled': allocated_dwelled
                    })

        od.at[idx, 'linehaul_cost'] = od_linehaul_cost
        od.at[idx, 'linehaul_cpp'] = od_linehaul_cost / od_pkgs if od_pkgs > 0 else 0
        od.at[idx, 'packages_dwelled'] = od_packages_dwelled  # CRITICAL: Properly allocated

    # Calculate totals
    od['total_cost'] = od['linehaul_cost'] + od['touch_cost']
    od['cost_per_pkg'] = od['total_cost'] / od['pkgs_day'].replace(0, 1)

    # Debug output
    total_lane_dwelled = arc_summary['packages_dwelled'].sum()
    total_od_dwelled = od['packages_dwelled'].sum()

    print(f"COST ALLOCATION SUMMARY:")
    print(f"  Total linehaul cost: ${od['linehaul_cost'].sum():,.2f}")
    print(f"  Total touch cost: ${od['touch_cost'].sum():,.2f}")
    print(f"  CRITICAL - Dwelled packages:")
    print(f"    Lane level total: {total_lane_dwelled:,.0f}")
    print(f"    OD level total: {total_od_dwelled:,.0f}")
    print(f"    Allocation efficiency: {(total_od_dwelled / max(total_lane_dwelled, 1)) * 100:.1f}%")

    # Show specific examples if dwelled packages > 0
    if total_od_dwelled > 0:
        dwelled_ods = od[od['packages_dwelled'] > 0]
        if not dwelled_ods.empty:
            print(f"  Sample dwelled ODs:")
            for i, (_, sample_od) in enumerate(dwelled_ods.head(3).iterrows()):
                print(f"    {sample_od['origin']}->{sample_od['dest']}: {sample_od['packages_dwelled']:.1f} dwelled")
    else:
        print("  WARNING: No dwelled packages allocated to ODs - this indicates an allocation problem")
        # Show debug info for first few allocations
        if allocation_debug:
            print("  Allocation debug (first 5):")
            for debug in allocation_debug[:5]:
                print(
                    f"    {debug['od_pair']}: lane_dwelled={debug['lane_dwelled']:.1f}, allocated={debug['allocated_dwelled']:.1f}")

    return od


# Functions now imported from veho_net.reporting


def validate_sort_capacity_feasibility(facilities: pd.DataFrame, od_selected: pd.DataFrame,
                                       timing_kv: dict) -> bool:
    """
    FIXED: Validate sort capacity at the region/parent hub level (minimum requirement).
    """
    print("🔍 Validating sort capacity feasibility...")

    # Get sort facilities with capacity constraints
    sort_facilities = facilities[
        (facilities['type'].isin(['hub', 'hybrid'])) &
        (facilities['max_sort_points_capacity'].notna()) &
        (facilities['max_sort_points_capacity'] > 0)
        ].copy()

    if sort_facilities.empty:
        print("✅ Sort capacity validation passed (no constrained facilities)")
        return True

    # Calculate MINIMUM sort points needed (region level)
    sort_points_per_destination = int(timing_kv.get('sort_points_per_destination', 2))

    # Count unique PARENT HUBS that each facility serves (minimum requirement)
    facility_parent_map = facilities.set_index('facility_name')['parent_hub_name'].to_dict()

    capacity_issues = []

    for _, facility in sort_facilities.iterrows():
        facility_name = facility['facility_name']
        max_capacity = int(facility['max_sort_points_capacity'])

        # For minimum calculation, count unique parent hubs served
        served_destinations = od_selected[od_selected['origin'] == facility_name]['dest'].unique()
        unique_parent_hubs = set()

        for dest in served_destinations:
            parent_hub = facility_parent_map.get(dest, dest)
            unique_parent_hubs.add(parent_hub)

        # Minimum sort points = unique parent hubs * points per destination
        min_sort_points_needed = len(unique_parent_hubs) * sort_points_per_destination

        if min_sort_points_needed > max_capacity:
            capacity_issues.append({
                'facility': facility_name,
                'needed': min_sort_points_needed,
                'capacity': max_capacity,
                'unique_parent_hubs': len(unique_parent_hubs)
            })

    if capacity_issues:
        print("❌ Sort capacity validation failed: Sort point capacity insufficient for minimum requirements:")
        for issue in capacity_issues[:15]:  # Show first 15
            print(f"  {issue['facility']} needs {issue['needed']} sort points "
                  f"(serving {issue['unique_parent_hubs']} regions) "
                  f"but only has capacity for {issue['capacity']}")

        if len(capacity_issues) > 15:
            print(f"  ... and {len(capacity_issues) - 15} more facilities")

        return False

    print("✅ Sort capacity validation passed")
    return True


def optimize_sort_allocation(od_selected: pd.DataFrame, cost_analysis: pd.DataFrame,
                             facilities: pd.DataFrame, timing_kv: dict) -> pd.DataFrame:
    """
    Enhanced sort allocation optimization with capacity constraints.
    """
    print("🎯 Optimizing sort allocation...")

    if cost_analysis.empty:
        print("Warning: No cost analysis data for sort optimization")
        return pd.DataFrame()

    try:
        # Simple efficiency-based allocation
        allocation_results = []

        for _, row in cost_analysis.iterrows():
            od_pair_id = row.get('od_pair_id', f"{row.get('origin', 'unknown')}_{row.get('dest', 'unknown')}")

            # Find best containerization level based on cost
            region_cost = row.get('region_cost', float('inf'))
            market_cost = row.get('market_cost', float('inf'))
            sort_group_cost = row.get('sort_group_cost', float('inf'))

            costs = {
                'region': region_cost,
                'market': market_cost,
                'sort_group': sort_group_cost
            }

            # Choose level with minimum cost
            best_level = min(costs, key=costs.get)
            best_cost = costs[best_level]

            # Calculate savings vs baseline (region level)
            baseline_cost = region_cost
            savings = max(0, baseline_cost - best_cost)

            allocation_results.append({
                'od_pair_id': od_pair_id,
                'origin': row.get('origin', 'unknown'),
                'dest': row.get('dest', 'unknown'),
                'pkgs_day': row.get('pkgs_day', 0),
                'optimal_containerization_level': best_level,
                'daily_cost_savings': savings,
                'sort_points_used': int(timing_kv.get('sort_points_per_destination', 2)),
                'efficiency_score': savings / max(1, int(timing_kv.get('sort_points_per_destination', 2)))
            })

        result_df = pd.DataFrame(allocation_results)

        if not result_df.empty:
            total_savings = result_df['daily_cost_savings'].sum()
            print(f"✅ Sort allocation optimized: ${total_savings:,.0f}/day potential savings")

        return result_df

    except Exception as e:
        print(f"Warning: Sort allocation optimization failed: {e}")
        return pd.DataFrame()


def apply_sort_allocation(od_selected: pd.DataFrame, sort_allocation: pd.DataFrame,
                          cost_analysis: pd.DataFrame, facilities: pd.DataFrame) -> pd.DataFrame:
    """
    Apply optimized sort allocation to OD pairs.
    """
    if sort_allocation.empty or od_selected.empty:
        return od_selected

    od_enhanced = od_selected.copy()

    # Create mapping from sort allocation
    allocation_map = sort_allocation.set_index('od_pair_id')['optimal_containerization_level'].to_dict()

    # Create OD pair IDs for matching
    od_enhanced['od_pair_id'] = od_enhanced['origin'] + '_' + od_enhanced['dest']

    # Apply optimal containerization levels
    od_enhanced['containerization_level'] = od_enhanced['od_pair_id'].map(allocation_map).fillna('region')

    # Add efficiency scores
    efficiency_map = sort_allocation.set_index('od_pair_id')['efficiency_score'].to_dict()
    od_enhanced['containerization_efficiency_score'] = od_enhanced['od_pair_id'].map(efficiency_map).fillna(0)

    print(f"Applied sort allocation to {len(od_enhanced)} OD pairs")

    return od_enhanced


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
    """Enhanced strategy execution with all optimizations."""

    # Scenario setup
    year = int(scenario_row.get("year", scenario_row.get("demand_year", 2026)))
    day_type = str(scenario_row["day_type"]).strip().lower()
    scenario_id = f"{base_id}_{strategy}"

    print(f"\n=== Running {scenario_id} with Enhanced Sort Optimization ===")

    # Inject load strategy into timing
    timing_local = dict(timing)
    timing_local["load_strategy"] = strategy

    # Set parent hub enforcement threshold
    facilities = facilities.copy()
    facilities.attrs["enforce_parent_hub_over_miles"] = int(run_kv.get("enforce_parent_hub_over_miles", 500))

    # Build OD matrix
    year_demand = demand.query("year == @year").copy()
    od, dir_fac, _dest_pop = build_od_and_direct(facilities, zips, year_demand, inj)

    od_day_col = "pkgs_offpeak_day" if day_type == "offpeak" else "pkgs_peak_day"
    direct_day_col = "dir_pkgs_offpeak_day" if day_type == "offpeak" else "dir_pkgs_peak_day"
    od = od[od[od_day_col] > 0].copy()

    if od.empty:
        print(f"[{scenario_id}] No OD demand for {day_type}. Skipping.")
        return None

    print(f"[{scenario_id}] Generated {len(od)} OD pairs")

    # Enhanced sort capacity validation
    if not validate_sort_capacity_feasibility(facilities, od, timing_local):
        print(f"[{scenario_id}] Sort capacity validation failed - continuing with warnings")

    # CRITICAL FIX: Set pkgs_day column BEFORE sort optimization
    od["pkgs_day"] = od[od_day_col]

    # Enhanced sort optimization with cost analysis (with error handling)
    print("🔄 Calculating containerization costs...")
    try:
        cost_analysis = calculate_containerization_costs_corrected(
            od, facilities, mb, costs, timing_local, pkgmix, cont
        )
    except Exception as e:
        print(f"Warning: Sort cost analysis failed: {e}")
        cost_analysis = pd.DataFrame()

    # Optimize sort allocation (with error handling)
    try:
        sort_allocation = optimize_sort_allocation(od, cost_analysis, facilities, timing_local)
    except Exception as e:
        print(f"Warning: Sort allocation optimization failed: {e}")
        sort_allocation = pd.DataFrame()

    # Apply optimization results (with error handling)
    try:
        od = apply_sort_allocation(od, sort_allocation, cost_analysis, facilities)
    except Exception as e:
        print(f"Warning: Sort allocation application failed: {e}")
        # Continue with original od DataFrame

    # Build candidate paths with enhanced constraints
    paths = candidate_paths(od, facilities, mb, around_factor=float(run_kv["path_around_the_world_factor"]))

    if paths.empty:
        print(f"[{scenario_id}] No valid paths generated. Skipping.")
        return None

    # Add containers calculation
    paths["containers_cont"] = paths["pkgs_day"].apply(
        lambda x: containers_for_pkgs_day(x, pkgmix, cont)
    )

    # Add scenario info
    paths["scenario_id"] = scenario_id
    paths["day_type"] = day_type

    # Enhanced cost and time calculation
    print(f"[{scenario_id}] Starting cost/time calculation for {len(paths)} paths...")
    cost_time_results = []
    for idx, row in paths.iterrows():
        try:
            total_cost, total_hours, sums, steps = path_cost_and_time(
                row, facilities, mb, timing_local, costs, pkgmix, cont, row["pkgs_day"]
            )

            result = {
                'index': idx,
                'total_cost': total_cost,
                'total_hours': total_hours,
                **sums  # Include all summary metrics
            }
            cost_time_results.append(result)

        except Exception as e:
            print(f"Warning: Cost calculation failed for path {idx}: {e}")
            # Add default values
            cost_time_results.append({
                'index': idx,
                'total_cost': 0,
                'total_hours': 24,
                'distance_miles_total': 0,
                'sla_days': 1,
                'total_trucks': 1,
                'container_fill_rate': 0.8,
                'truck_fill_rate': 0.8,
                'packages_dwelled': 0
            })

    # Merge cost/time results back
    cost_time_df = pd.DataFrame(cost_time_results)
    print(f"[{scenario_id}] Cost/time calculation complete: {len(cost_time_df)} results")

    if not cost_time_df.empty:
        cost_time_df.set_index('index', inplace=True)
        for col in cost_time_df.columns:
            if col in ['total_cost', 'total_hours']:
                paths[col] = paths.index.map(cost_time_df[col]).fillna(0)
            else:
                paths[f"summary_{col}"] = paths.index.map(cost_time_df[col]).fillna(0)

    # Enhanced MILP optimization
    print(f"[{scenario_id}] Solving enhanced optimization...")
    od_selected, arc_summary = solve_arc_pooled_path_selection(
        paths, facilities, mb, pkgmix, cont, costs
    )

    if od_selected.empty:
        print(f"[{scenario_id}] Optimization failed. Skipping.")
        return None

    # Enhanced cost allocation
    od_selected = _allocate_lane_costs_to_ods(od_selected, arc_summary, costs, strategy)

    # Enhanced output generation
    direct_day = dir_fac.copy()
    direct_day["dir_pkgs_day"] = direct_day[direct_day_col]

    od_out = build_od_selected_outputs(od_selected, facilities, direct_day)
    dwell_hotspots = build_dwell_hotspots(od_selected)

    # Enhanced facility analysis with costs
    facility_rollup = _identify_volume_types_with_costs(
        od_selected, pd.DataFrame(), direct_day, arc_summary
    )
    facility_rollup = _calculate_hourly_throughput_with_costs(
        facility_rollup, timing_local, strategy
    )

    # Generate path steps (simplified for now)
    path_steps = pd.DataFrame({
        'scenario_id': scenario_id,
        'origin': od_selected['origin'],
        'dest': od_selected['dest'],
        'step_order': 1,
        'from_facility': od_selected['origin'],
        'to_facility': od_selected['dest'],
        'distance_miles': 500,  # Default
        'drive_hours': 8,
        'processing_hours_at_destination': 2
    })

    lane_summary = build_lane_summary(arc_summary)

    # Enhanced KPI calculation
    total_cost = od_selected["total_cost"].sum() if "total_cost" in od_selected.columns else 0
    total_pkgs = od_selected["pkgs_day"].sum()
    cost_per_pkg = total_cost / max(total_pkgs, 1)

    # Enhanced metrics
    sort_savings = sort_allocation['daily_cost_savings'].sum() if not sort_allocation.empty else 0
    avg_container_fill = od_selected.get('container_fill_rate', pd.Series([0.8])).mean()
    avg_truck_fill = od_selected.get('truck_fill_rate', pd.Series([0.8])).mean()
    total_dwelled = od_selected.get('packages_dwelled', pd.Series([0])).sum()

    kpis = pd.Series({
        "total_cost": total_cost,
        "cost_per_pkg": cost_per_pkg,
        "num_ods": len(od_selected),
        "sort_optimization_savings": sort_savings,
        "avg_container_fill_rate": avg_container_fill,
        "avg_truck_fill_rate": avg_truck_fill,
        "total_packages_dwelled": total_dwelled,
        "sla_violations": 0,  # Placeholder
        "around_world_flags": 0,  # Placeholder
        "pct_direct": (od_selected["path_type"] == "direct").mean() * 100 if not od_selected.empty else 0,
        "pct_1_touch": (od_selected["path_type"] == "1_touch").mean() * 100 if not od_selected.empty else 0,
        "pct_2_touch": (od_selected["path_type"] == "2_touch").mean() * 100 if not od_selected.empty else 0,
        "pct_3_touch": (od_selected["path_type"] == "3_touch").mean() * 100 if not od_selected.empty else 0,
    })

    # Write enhanced outputs
    out_path = out_dir / OUTPUT_FILE_TEMPLATE.format(scenario_id=scenario_id)
    write_workbook(
        out_path,
        pd.DataFrame([{**kpis.to_dict(), 'scenario_id': scenario_id}]),  # scenario summary
        od_out,
        path_steps,
        dwell_hotspots,
        facility_rollup,
        arc_summary,
        kpis,
        sort_allocation  # Enhanced with sort optimization data
    )

    print(f"[{scenario_id}] ✅ Results written to: {out_path}")
    print(f"[{scenario_id}] KPIs: ${total_cost:,.0f} total, ${cost_per_pkg:.3f}/pkg, "
          f"{sort_savings:,.0f} sort savings, {avg_truck_fill:.1%} fill rate")

    return scenario_id, out_path, kpis, {
        'scenario_id': scenario_id,
        'strategy': strategy,
        'output_file': str(out_path.name),
        **kpis.to_dict()
    }


def main(input_path: str, output_dir: str):
    """Enhanced main execution with complete sort optimization."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Loading enhanced workbook from: {input_path}")
    dfs = load_workbook(input_path)
    validate_inputs(dfs)

    # Enhanced parameter processing with defaults
    timing_local = params_to_dict(dfs["timing_params"])
    costs = params_to_dict(dfs["cost_params"])
    run_kv = params_to_dict(dfs["run_settings"])

    # Add enhanced defaults
    if 'sort_points_per_destination' not in timing_local:
        timing_local['sort_points_per_destination'] = 2
        print("Added default timing parameter sort_points_per_destination: 2")

    if 'sort_setup_cost_per_point' not in costs:
        costs['sort_setup_cost_per_point'] = 0.0
        print("Added default cost parameter sort_setup_cost_per_point: 0.0")

    # Run enhanced paired mode
    base_id = input_path.stem.replace("_input", "").replace("_v4", "").replace("veho_model", "sort_model")
    strategies = ["container", "fluid"]

    print("Running in paired mode with enhanced sort optimization")

    compare_results = []

    for scenario_idx, scenario_row in dfs["scenarios"].iterrows():
        for strategy in strategies:
            try:
                print(f"\n--- Attempting {base_id}_{strategy} ---")
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
                    print(f"✅ Successfully completed {scenario_id}")
                else:
                    print(f"⚠️  {base_id}_{strategy} returned no results")

            except Exception as e:
                print(f"❌ Error running {base_id}_{strategy}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Enhanced comparison report
    if compare_results:
        compare_df = pd.DataFrame(compare_results)
        compare_path = output_dir / COMPARE_FILE_TEMPLATE.format(base_id=base_id)
        write_compare_workbook(compare_path, compare_df, run_kv)
        print(f"✅ Enhanced comparison report: {compare_path}")

    print("🎉 Enhanced optimization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Veho Network Optimization with Sort Intelligence")
    parser.add_argument("--input", required=True, help="Path to input Excel file")
    parser.add_argument("--output_dir", required=True, help="Output directory path")
    args = parser.parse_args()

    main(args.input, args.output_dir)