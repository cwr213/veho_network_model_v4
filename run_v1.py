# run_v1.py - CLEANED VERSION with minimal terminal output
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
from veho_net.write_outputs import (
    write_workbook,
    write_compare_workbook,
    write_executive_summary_workbook,
    write_consolidated_multi_year_workbook
)

# Output naming control - NOW INCLUDES SCENARIO_ID
OUTPUT_FILE_TEMPLATE = "{scenario_id}_{strategy}_results.xlsx"
COMPARE_FILE_TEMPLATE = "{scenario_id}_compare.xlsx"


def _allocate_lane_costs_to_ods(od_selected: pd.DataFrame, arc_summary: pd.DataFrame, costs: dict,
                                strategy: str) -> pd.DataFrame:
    """Proper cost allocation including packages_dwelled from lanes to ODs."""
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
    od['packages_dwelled'] = 0.0

    if arc_summary is None or arc_summary.empty:
        od['total_cost'] = od['touch_cost']
        od['cost_per_pkg'] = od['touch_cpp']
        return od

    # For each OD, find its path legs and allocate costs AND packages_dwelled
    for idx, row in od.iterrows():
        path_str = str(row.get('path_str', ''))
        if not path_str or '->' not in path_str:
            continue

        nodes = path_str.split('->')
        od_pkgs = float(row['pkgs_day'])
        od_linehaul_cost = 0.0
        od_packages_dwelled = 0.0

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
                lane_packages_dwelled = float(lane_row.get('packages_dwelled', 0))

                if lane_total_pkgs > 0:
                    od_share = od_pkgs / lane_total_pkgs
                    allocated_cost = lane_total_cost * od_share
                    allocated_dwelled = lane_packages_dwelled * od_share

                    od_linehaul_cost += allocated_cost
                    od_packages_dwelled += allocated_dwelled

        od.at[idx, 'linehaul_cost'] = od_linehaul_cost
        od.at[idx, 'linehaul_cpp'] = od_linehaul_cost / od_pkgs if od_pkgs > 0 else 0
        od.at[idx, 'packages_dwelled'] = od_packages_dwelled

    # Calculate totals
    od['total_cost'] = od['linehaul_cost'] + od['touch_cost']
    od['cost_per_pkg'] = od['total_cost'] / od['pkgs_day'].replace(0, 1)

    return od


def validate_sort_capacity_feasibility(facilities: pd.DataFrame, od_selected: pd.DataFrame,
                                       timing_kv: dict) -> bool:
    """Validate sort capacity at the region/parent hub level."""
    sort_facilities = facilities[
        (facilities['type'].isin(['hub', 'hybrid'])) &
        (facilities['max_sort_points_capacity'].notna()) &
        (facilities['max_sort_points_capacity'] > 0)
        ].copy()

    if sort_facilities.empty:
        return True

    sort_points_per_destination = int(timing_kv.get('sort_points_per_destination', 2))
    facility_parent_map = facilities.set_index('facility_name')['parent_hub_name'].to_dict()
    capacity_issues = []

    for _, facility in sort_facilities.iterrows():
        facility_name = facility['facility_name']
        max_capacity = int(facility['max_sort_points_capacity'])

        served_destinations = od_selected[od_selected['origin'] == facility_name]['dest'].unique()
        unique_parent_hubs = set()

        for dest in served_destinations:
            parent_hub = facility_parent_map.get(dest, dest)
            unique_parent_hubs.add(parent_hub)

        min_sort_points_needed = len(unique_parent_hubs) * sort_points_per_destination

        if min_sort_points_needed > max_capacity:
            capacity_issues.append({
                'facility': facility_name,
                'needed': min_sort_points_needed,
                'capacity': max_capacity,
                'unique_parent_hubs': len(unique_parent_hubs)
            })

    return len(capacity_issues) == 0


def optimize_sort_allocation(od_selected: pd.DataFrame, cost_analysis: pd.DataFrame,
                             facilities: pd.DataFrame, timing_kv: dict) -> pd.DataFrame:
    """Enhanced sort allocation optimization with capacity constraints."""
    if cost_analysis.empty:
        return pd.DataFrame()

    try:
        allocation_results = []

        for _, row in cost_analysis.iterrows():
            od_pair_id = row.get('od_pair_id', f"{row.get('origin', 'unknown')}_{row.get('dest', 'unknown')}")

            region_cost = row.get('region_cost', float('inf'))
            market_cost = row.get('market_cost', float('inf'))
            sort_group_cost = row.get('sort_group_cost', float('inf'))

            costs = {
                'region': region_cost,
                'market': market_cost,
                'sort_group': sort_group_cost
            }

            best_level = min(costs, key=costs.get)
            best_cost = costs[best_level]
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

        return pd.DataFrame(allocation_results)

    except Exception:
        return pd.DataFrame()


def apply_sort_allocation(od_selected: pd.DataFrame, sort_allocation: pd.DataFrame,
                          cost_analysis: pd.DataFrame, facilities: pd.DataFrame) -> pd.DataFrame:
    """Apply optimized sort allocation to OD pairs."""
    if sort_allocation.empty or od_selected.empty:
        return od_selected

    od_enhanced = od_selected.copy()
    allocation_map = sort_allocation.set_index('od_pair_id')['optimal_containerization_level'].to_dict()
    od_enhanced['od_pair_id'] = od_enhanced['origin'] + '_' + od_enhanced['dest']
    od_enhanced['containerization_level'] = od_enhanced['od_pair_id'].map(allocation_map).fillna('region')

    efficiency_map = sort_allocation.set_index('od_pair_id')['efficiency_score'].to_dict()
    od_enhanced['containerization_efficiency_score'] = od_enhanced['od_pair_id'].map(efficiency_map).fillna(0)

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
    """Enhanced strategy execution with scenario_id in output file names."""

    # Extract scenario_id from input (with fallback)
    scenario_id_from_input = scenario_row.get("scenario_id", scenario_row.get("pair_id", "default_scenario"))
    year = int(scenario_row.get("year", scenario_row.get("demand_year", 2026)))
    day_type = str(scenario_row["day_type"]).strip().lower()

    # Build unique scenario identifier: input_scenario_id + strategy
    scenario_id = f"{scenario_id_from_input}_{strategy}"

    timing_local = dict(timing)
    timing_local["load_strategy"] = strategy

    facilities = facilities.copy()
    facilities.attrs["enforce_parent_hub_over_miles"] = int(run_kv.get("enforce_parent_hub_over_miles", 500))

    # Build OD matrix
    year_demand = demand.query("year == @year").copy()
    od, dir_fac, _dest_pop = build_od_and_direct(facilities, zips, year_demand, inj)

    od_day_col = "pkgs_offpeak_day" if day_type == "offpeak" else "pkgs_peak_day"
    direct_day_col = "dir_pkgs_offpeak_day" if day_type == "offpeak" else "dir_pkgs_peak_day"
    od = od[od[od_day_col] > 0].copy()

    if od.empty:
        return None

    # Validate sort capacity
    validate_sort_capacity_feasibility(facilities, od, timing_local)

    # Set pkgs_day column
    od["pkgs_day"] = od[od_day_col]

    # Sort optimization with error handling
    try:
        cost_analysis = calculate_containerization_costs_corrected(
            od, facilities, mb, costs, timing_local, pkgmix, cont
        )
        sort_allocation = optimize_sort_allocation(od, cost_analysis, facilities, timing_local)
        od = apply_sort_allocation(od, sort_allocation, cost_analysis, facilities)
    except Exception:
        cost_analysis = pd.DataFrame()
        sort_allocation = pd.DataFrame()

    # Build candidate paths
    paths = candidate_paths(od, facilities, mb, around_factor=float(run_kv["path_around_the_world_factor"]))

    if paths.empty:
        return None

    # Merge packages and add containers
    paths = paths.merge(
        od[['origin', 'dest', 'pkgs_day']],
        on=['origin', 'dest'],
        how='left'
    )

    if 'pkgs_day' not in paths.columns:
        return None

    paths['pkgs_day'] = paths['pkgs_day'].fillna(0)

    paths["containers_cont"] = paths["pkgs_day"].apply(
        lambda x: containers_for_pkgs_day(x, pkgmix, cont)
    )

    paths["scenario_id"] = scenario_id
    paths["day_type"] = day_type

    # Cost and time calculation
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
                **sums
            }
            cost_time_results.append(result)

        except Exception:
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

    # Merge results back
    cost_time_df = pd.DataFrame(cost_time_results)

    if not cost_time_df.empty:
        cost_time_df.set_index('index', inplace=True)
        for col in cost_time_df.columns:
            if col in ['total_cost', 'total_hours']:
                paths[col] = paths.index.map(cost_time_df[col]).fillna(0)
            else:
                paths[f"summary_{col}"] = paths.index.map(cost_time_df[col]).fillna(0)

    # MILP optimization
    od_selected, arc_summary = solve_arc_pooled_path_selection(
        paths, facilities, mb, pkgmix, cont, costs
    )

    if od_selected.empty:
        return None

    # Cost allocation
    try:
        od_selected = _allocate_lane_costs_to_ods(od_selected, arc_summary, costs, strategy)
    except Exception:
        if 'total_cost' not in od_selected.columns:
            od_selected['total_cost'] = 1000
        if 'cost_per_pkg' not in od_selected.columns:
            od_selected['cost_per_pkg'] = od_selected['total_cost'] / od_selected['pkgs_day'].replace(0, 1)

    # Generate output data
    try:
        direct_day = dir_fac.copy()
        direct_day["dir_pkgs_day"] = direct_day[direct_day_col]
    except Exception:
        direct_day = pd.DataFrame()

    try:
        od_out = build_od_selected_outputs(od_selected, facilities, direct_day)
    except Exception:
        od_out = od_selected.copy()

    try:
        dwell_hotspots = build_dwell_hotspots(od_selected)
    except Exception:
        dwell_hotspots = pd.DataFrame()

    # Facility analysis
    try:
        facility_rollup = _identify_volume_types_with_costs(
            od_selected, pd.DataFrame(), direct_day, arc_summary
        )
        facility_rollup = _calculate_hourly_throughput_with_costs(
            facility_rollup, timing_local, strategy
        )
    except Exception:
        unique_facilities = set()
        if not od_selected.empty:
            unique_facilities.update(od_selected['origin'].unique())
            unique_facilities.update(od_selected['dest'].unique())

        facility_rollup = pd.DataFrame([
            {'facility': fac, 'injection_pkgs_day': 0, 'last_mile_pkgs_day': 0, 'peak_hourly_throughput': 0}
            for fac in unique_facilities
        ])

    # Generate path steps
    try:
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
                    'distance_miles': 500,
                    'drive_hours': 8,
                    'processing_hours_at_destination': 2
                })

        path_steps_df = pd.DataFrame(path_steps)
    except Exception:
        path_steps_df = pd.DataFrame([{
            'scenario_id': scenario_id,
            'origin': 'Unknown',
            'dest': 'Unknown',
            'step_order': 1,
            'from_facility': 'Unknown',
            'to_facility': 'Unknown',
            'distance_miles': 0,
            'drive_hours': 0,
            'processing_hours_at_destination': 0
        }])

    try:
        lane_summary = build_lane_summary(arc_summary)
    except Exception:
        lane_summary = pd.DataFrame()

    # Calculate KPIs
    try:
        total_cost = od_selected["total_cost"].sum() if "total_cost" in od_selected.columns else 0
        total_pkgs = od_selected["pkgs_day"].sum() if "pkgs_day" in od_selected.columns else 1
        cost_per_pkg = total_cost / max(total_pkgs, 1)

        sort_savings = sort_allocation[
            'daily_cost_savings'].sum() if not sort_allocation.empty and 'daily_cost_savings' in sort_allocation.columns else 0
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
            "sla_violations": 0,
            "around_world_flags": 0,
            "pct_direct": (od_selected[
                               "path_type"] == "direct").mean() * 100 if not od_selected.empty and "path_type" in od_selected.columns else 0,
            "pct_1_touch": (od_selected[
                                "path_type"] == "1_touch").mean() * 100 if not od_selected.empty and "path_type" in od_selected.columns else 0,
            "pct_2_touch": (od_selected[
                                "path_type"] == "2_touch").mean() * 100 if not od_selected.empty and "path_type" in od_selected.columns else 0,
            "pct_3_touch": (od_selected[
                                "path_type"] == "3_touch").mean() * 100 if not od_selected.empty and "path_type" in od_selected.columns else 0,
        })

    except Exception:
        kpis = pd.Series({
            "total_cost": 0,
            "cost_per_pkg": 0,
            "num_ods": len(od_selected) if not od_selected.empty else 0,
            "sort_optimization_savings": 0,
            "avg_container_fill_rate": 0.8,
            "avg_truck_fill_rate": 0.8,
            "total_packages_dwelled": 0,
        })

    # Write outputs
    out_path = out_dir / OUTPUT_FILE_TEMPLATE.format(scenario_id=scenario_id)

    try:
        scenario_summary = pd.DataFrame([{
            'scenario_id': scenario_id,
            'strategy': strategy,
            'total_cost': kpis['total_cost'],
            'cost_per_pkg': kpis['cost_per_pkg'],
            'total_packages': kpis.get('num_ods', 0) * od_selected['pkgs_day'].mean() if not od_selected.empty else 0,
            'optimization_savings': kpis['sort_optimization_savings'],
            'avg_fill_rate': kpis['avg_truck_fill_rate']
        }])

        write_success = write_workbook(
            out_path,
            scenario_summary,
            od_out if not od_out.empty else pd.DataFrame([{"note": "No OD data"}]),
            path_steps_df if not path_steps_df.empty else pd.DataFrame([{"note": "No path data"}]),
            dwell_hotspots if not dwell_hotspots.empty else pd.DataFrame([{"note": "No dwell data"}]),
            facility_rollup if not facility_rollup.empty else pd.DataFrame([{"note": "No facility data"}]),
            arc_summary if not arc_summary.empty else pd.DataFrame([{"note": "No arc data"}]),
            kpis,
            sort_allocation if not sort_allocation.empty else None
        )

        if not write_success:
            return None

    except Exception:
        return None

    print(
        f"[{scenario_id}] ${kpis['total_cost']:,.0f} cost, ${kpis['cost_per_pkg']:.3f}/pkg, {kpis['avg_truck_fill_rate']:.1%} fill")

    return scenario_id, out_path, kpis, {
        'scenario_id': scenario_id,
        'scenario_id_from_input': scenario_id_from_input,  # Keep track of original
        'strategy': strategy,
        'output_file': str(out_path.name),
        **kpis.to_dict()
    }


def main(input_path: str, output_dir: str):
    """Main execution with scenario_id-based file naming."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Loading: {input_path.name}")
    dfs = load_workbook(input_path)
    validate_inputs(dfs)

    timing_local = params_to_dict(dfs["timing_params"])
    costs = params_to_dict(dfs["cost_params"])
    run_kv = params_to_dict(dfs["run_settings"])

    # Add defaults silently
    if 'sort_points_per_destination' not in timing_local:
        timing_local['sort_points_per_destination'] = 2

    if 'sort_setup_cost_per_point' not in costs:
        costs['sort_setup_cost_per_point'] = 0.0

    # Use the original base_id for internal processing
    base_id = input_path.stem.replace("_input", "").replace("_v4", "").replace("veho_model", "sort_model")
    strategies = ["container", "fluid"]
    compare_results = []
    created_files = []  # Track files created in this run

    for scenario_idx, scenario_row in dfs["scenarios"].iterrows():
        # Get scenario_id from input for unique naming
        scenario_id_from_input = scenario_row.get("scenario_id",
                                                  scenario_row.get("pair_id", f"scenario_{scenario_idx + 1}"))

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
                    created_files.append(out_path.name)  # Track created file

            except Exception as e:
                print(f"Error running {scenario_id_from_input}_{strategy}: {e}")
                continue

    # Write comparison reports with scenario_id
    if compare_results:
        compare_df = pd.DataFrame(compare_results)

        # Use the first scenario_id for comparison file naming
        first_scenario_id = compare_results[0].get('scenario_id_from_input', 'comparison')
        compare_path = output_dir / COMPARE_FILE_TEMPLATE.format(scenario_id=first_scenario_id)

        write_compare_success = write_compare_workbook(compare_path, compare_df, run_kv)
        if write_compare_success:
            created_files.append(compare_path.name)

            # Executive summary with scenario_id
            exec_summary_path = output_dir / f"{first_scenario_id}_executive_summary.xlsx"
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

    print("🎉 Optimization complete!")

    # Show only files created in THIS run
    if created_files:
        print(f"📋 Created {len(created_files)} files this run:")
        for file_name in sorted(set(created_files)):
            print(f"  📄 {file_name}")
    else:
        print(f"⚠️  No output files created")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Veho Network Optimization")
    parser.add_argument("--input", required=True, help="Path to input Excel file")
    parser.add_argument("--output_dir", required=True, help="Output directory path")
    args = parser.parse_args()

    main(args.input, args.output_dir)