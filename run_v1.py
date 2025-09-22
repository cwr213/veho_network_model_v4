# run_v1.py - FIXED with Proper Lane-Level Cost Allocation
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
    build_od_selected_outputs,
    build_dwell_hotspots,
    build_facility_rollup,
    add_zone,
    build_lane_summary,
)
from veho_net.write_outputs import write_workbook, write_compare_workbook

# Output naming control
OUTPUT_FILE_TEMPLATE = "{scenario_id}_results_v1.xlsx"
COMPARE_FILE_TEMPLATE = "{base_id}_compare.xlsx"
EXECUTIVE_SUMMARY_TEMPLATE = "{base_id}_executive_summary.xlsx"
CONSOLIDATED_OUTPUT = "Network_Analysis_All_Scenarios.xlsx"


def _allocate_lane_costs_to_ods(od_selected: pd.DataFrame, arc_summary: pd.DataFrame, costs: dict,
                                strategy: str) -> pd.DataFrame:
    """
    Properly allocate costs to OD pairs:
    - Linehaul costs: From arc_summary (pooled lane costs) allocated proportionally
    - Touch costs: Calculated per OD based on number of touches

    CRITICAL: Touch costs should NOT include origin sort (that's injection_sort_cpp)
    - Container: intermediate touches only (crossdock)
    - Fluid: intermediate touches + destination sort (NOT origin)
    """
    od = od_selected.copy()

    # Calculate touch costs per OD (independent of consolidation)
    touch_map = {"direct": 0, "1_touch": 1, "2_touch": 2, "3_touch": 3, "4_touch": 4}
    od['num_touches'] = od['path_type'].map(touch_map).fillna(0)

    crossdock_pp = float(costs.get("crossdock_touch_cost_per_pkg", 0.0))
    sort_pp = float(costs.get("sort_cost_per_pkg", 0.0))

    if strategy.lower() == "container":
        # Container: crossdock touches only (intermediate hubs)
        od['touch_cost'] = od['num_touches'] * crossdock_pp * od['pkgs_day']
        od['touch_cpp'] = od['num_touches'] * crossdock_pp
    else:
        # Fluid: intermediate touches + destination sort (NOT origin - that's injection_sort)
        od['touch_cost'] = (od['num_touches'] + 1) * sort_pp * od['pkgs_day']  # touches + dest
        od['touch_cpp'] = (od['num_touches'] + 1) * sort_pp

    # Allocate linehaul costs from arc_summary
    od['linehaul_cost'] = 0.0
    od['linehaul_cpp'] = 0.0

    if arc_summary is None or arc_summary.empty:
        od['total_cost'] = od['touch_cost']
        od['cost_per_pkg'] = od['touch_cpp']
        return od

    # For each OD, find its path legs and allocate costs
    for idx, row in od.iterrows():
        path_str = str(row.get('path_str', ''))
        if not path_str or '->' not in path_str:
            continue

        nodes = path_str.split('->')
        od_pkgs = float(row['pkgs_day'])
        od_linehaul_cost = 0.0

        # Sum costs across all legs in this path
        for i in range(len(nodes) - 1):
            from_fac = nodes[i].strip()
            to_fac = nodes[i + 1].strip()

            # Find this lane in arc_summary
            lane = arc_summary[
                (arc_summary['from_facility'] == from_fac) &
                (arc_summary['to_facility'] == to_fac)
                ]

            if not lane.empty:
                lane_total_cost = float(lane.iloc[0].get('total_cost', 0))
                lane_total_pkgs = float(lane.iloc[0].get('pkgs_day', 1))

                # Allocate proportionally to this OD's share of lane volume
                if lane_total_pkgs > 0:
                    od_share = od_pkgs / lane_total_pkgs
                    od_linehaul_cost += lane_total_cost * od_share

        od.at[idx, 'linehaul_cost'] = od_linehaul_cost
        od.at[idx, 'linehaul_cpp'] = od_linehaul_cost / od_pkgs if od_pkgs > 0 else 0

    # Calculate totals
    od['total_cost'] = od['linehaul_cost'] + od['touch_cost']
    od['cost_per_pkg'] = od['total_cost'] / od['pkgs_day'].replace(0, 1)

    return od


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
    Enhanced strategy execution with proper cost allocation.
    """
    # Scenario setup
    year = int(scenario_row["year"]) if "year" in scenario_row else int(scenario_row["demand_year"])
    day_type = str(scenario_row["day_type"]).strip().lower()
    scenario_id = f"{base_id}_{strategy}"

    print(f"\n=== Running {scenario_id} ===")

    # Inject load strategy into timing
    timing_local = dict(timing)
    timing_local["load_strategy"] = strategy

    # Set parent hub enforcement threshold
    facilities = facilities.copy()
    facilities.attrs["enforce_parent_hub_over_miles"] = int(run_kv.get("enforce_parent_hub_over_miles", 500))

    # Build OD matrix (only hub/hybrid origins)
    year_demand = demand.query("year == @year").copy()
    od, dir_fac, _dest_pop = build_od_and_direct(facilities, zips, year_demand, inj)

    od_day_col = "pkgs_offpeak_day" if day_type == "offpeak" else "pkgs_peak_day"
    direct_day_col = "dir_pkgs_offpeak_day" if day_type == "offpeak" else "dir_pkgs_peak_day"
    od = od[od[od_day_col] > 0].copy()

    if od.empty:
        print(f"[{scenario_id}] No OD demand for {day_type}. Skipping.")
        return scenario_id, None, pd.Series(dtype=float), None

    # Generate candidate paths with hub hierarchy
    around_factor = float(run_kv.get("path_around_the_world_factor", 2.0))
    cands = candidate_paths(od, facilities, mb, around_factor=around_factor)

    if cands.empty:
        print(f"[{scenario_id}] No candidate paths generated. Skipping.")
        return scenario_id, None, pd.Series(dtype=float), None

    print(f"[{scenario_id}] Generated {len(cands)} candidate paths")

    # FIXED: Build candidate table with ONE row per unique (origin, dest, path_str)
    # Calculate metrics for candidate paths (for comparison/validation)
    cand_rows, detail_rows, direct_dist = [], [], {}

    for _, r in cands.iterrows():
        sub = od[(od["origin"] == r["origin"]) & (od["dest"] == r["dest"])]
        if sub.empty:
            continue
        pkgs_day = float(sub.iloc[0][od_day_col])

        try:
            cost, hours, sums, steps = path_cost_and_time(
                r, facilities, mb, timing_local, costs, pkgmix, cont, pkgs_day
            )
            conts = containers_for_pkgs_day(pkgs_day, pkgmix, cont)

            cand_rows.append({
                "scenario_id": scenario_id,
                "origin": r["origin"],
                "dest": r["dest"],
                "day_type": day_type,
                "path_type": r["path_type"],
                "path_nodes": r.get("path_nodes", None),
                "path_str": r.get("path_str",
                                  "->".join(r["path_nodes"]) if isinstance(r.get("path_nodes"), list) else None),
                "pkgs_day": pkgs_day,
                "containers_cont": conts,
                "time_hours": hours,
                "distance_miles": sums["distance_miles_total"],
                "linehaul_hours": sums["linehaul_hours_total"],
                "handling_hours": sums["handling_hours_total"],
                "dwell_hours": sums["dwell_hours_total"],
                "destination_dwell_hours": sums["destination_dwell_hours"],
                "sla_days": sums["sla_days"],
                "cost_candidate_path": cost,  # Note: This is standalone cost, not pooled
                "total_facilities_touched": sums.get("total_facilities_touched", len(r.get("path_nodes", []))),
            })

            for st in steps:
                step = {
                    "scenario_id": scenario_id,
                    "origin": r["origin"],
                    "dest": r["dest"],
                    "day_type": day_type,
                    "path_type": r["path_type"],
                    "path_str": r.get("path_str"),
                    "pkgs_day": pkgs_day,
                    "containers_cont": conts,
                }
                step.update(st)
                detail_rows.append(step)

            if r["path_type"] == "direct":
                key = (scenario_id, r["origin"], r["dest"], day_type)
                direct_dist[key] = sums["distance_miles_total"]

        except Exception as e:
            print(f"[{scenario_id}] Error processing path {r['path_str']}: {e}")
            continue

    cand_tbl = pd.DataFrame(cand_rows)
    detail_all = pd.DataFrame(detail_rows)

    if cand_tbl.empty:
        print(f"[{scenario_id}] No valid candidate paths for {day_type}. Skipping.")
        return scenario_id, None, pd.Series(dtype=float), None

    print(f"[{scenario_id}] Processed {len(cand_tbl)} valid paths")

    # MILP path selection (uses pooled trucks for actual optimization)
    selected_basic, arc_summary = solve_arc_pooled_path_selection(
        cand_tbl[["scenario_id", "origin", "dest", "day_type", "path_type", "pkgs_day", "path_nodes", "path_str",
                  "containers_cont"]],
        facilities, mb, pkgmix, cont, costs,
    )

    # FIXED: Merge candidate metrics WITHOUT duplicating rows
    merge_keys = ["scenario_id", "origin", "dest", "day_type", "path_str"]

    # Ensure no duplicates in cand_tbl before merging
    cand_tbl_unique = cand_tbl.drop_duplicates(subset=merge_keys)

    selected = selected_basic.merge(cand_tbl_unique, on=merge_keys, how="left", suffixes=("", "_cand"))

    # Normalize column names
    cand_rename = {
        "distance_miles_cand": "distance_miles",
        "linehaul_hours_cand": "linehaul_hours",
        "handling_hours_cand": "handling_hours",
        "dwell_hours_cand": "dwell_hours",
        "destination_dwell_hours_cand": "destination_dwell_hours",
        "sla_days_cand": "sla_days",
        "time_hours_cand": "time_hours",
        "pkgs_day_cand": "pkgs_day_ref",  # Rename to avoid confusion
    }
    for old, new in cand_rename.items():
        if old in selected.columns and new not in selected.columns:
            selected = selected.rename(columns={old: new})

    # Build output flags
    dd = pd.Series(direct_dist, name="distance_miles")
    if not dd.empty:
        dd.index = pd.MultiIndex.from_tuples(dd.index, names=["scenario_id", "origin", "dest", "day_type"])

    flags = {
        "path_around_the_world_factor": float(run_kv.get("path_around_the_world_factor", 2.0)),
        "sla_target_days": int(run_kv.get("sla_target_days", 3)),
    }
    od_out = build_od_selected_outputs(selected, dd, flags)
    od_out["scenario_id"] = scenario_id

    # Add zones
    od_out = add_zone(od_out, facilities)

    # CRITICAL FIX: Properly allocate costs using lane-level pooling
    od_out = _allocate_lane_costs_to_ods(od_out, arc_summary, costs, strategy)

    # CRITICAL FIX 2: Add injection sort and last mile costs to get TOTAL cost per package
    # OD-level costs currently only have: linehaul + intermediate/dest touches
    # Still need: injection sort (origin) + last mile (destination)
    sort_cost = float(costs.get("sort_cost_per_pkg", 0.0))
    lm_cost = float(costs.get("last_mile_cpp", 0.0))

    # Add origin injection sort cost
    od_out['injection_sort_cost'] = sort_cost * od_out['pkgs_day']
    od_out['injection_sort_cpp'] = sort_cost

    # Add destination last mile cost
    od_out['last_mile_cost'] = lm_cost * od_out['pkgs_day']
    od_out['last_mile_destination_cpp'] = lm_cost

    # Recalculate total cost to include ALL components
    od_out['total_cost'] = (
            od_out['injection_sort_cost'] +  # Origin sort
            od_out['linehaul_cost'] +  # Transportation (allocated from pooled lanes)
            od_out['touch_cost'] +  # Intermediate + dest touches
            od_out['last_mile_cost']  # Destination last mile
    )
    od_out['cost_per_pkg'] = od_out['total_cost'] / od_out['pkgs_day'].replace(0, 1)

    # Filter detail steps to selected paths only
    key_cols = ["scenario_id", "origin", "dest", "day_type", "path_type", "path_str"]
    detail_sel = detail_all.merge(od_out[key_cols].drop_duplicates(), on=key_cols, how="inner")

    # Direct injection volumes
    direct_day = (
        dir_fac[dir_fac["year"] == year][["dest", direct_day_col]]
        .groupby("dest", as_index=False)[direct_day_col]
        .sum()
        .rename(columns={direct_day_col: "dir_pkgs_day"})
    )

    # Enhanced facility rollup with VA-based throughput
    facility_rollup = build_facility_rollup(
        facilities=facilities,
        zips=zips,
        od_selected=od_out,
        path_steps_selected=detail_sel,
        direct_day=direct_day,
        arc_summary=arc_summary,
        costs=costs,
        load_strategy=strategy,
        timing_kv=timing_local
    )

    # Add year and strategy columns for consolidation
    facility_rollup['year'] = year
    facility_rollup['day_type'] = day_type
    facility_rollup['strategy'] = strategy

    lane_summary = build_lane_summary(arc_summary)
    if not lane_summary.empty:
        lane_summary['year'] = year
        lane_summary['day_type'] = day_type
        lane_summary['strategy'] = strategy
        lane_summary['scenario_id'] = scenario_id

    scen_sum = _scenario_summary(run_kv, scenario_id, year, day_type, mb, timing_local, cont)
    dwell_hotspots = build_dwell_hotspots(detail_sel)
    kpis = _network_kpis(od_out)

    # Write individual scenario results
    out_path = out_dir / OUTPUT_FILE_TEMPLATE.format(scenario_id=scenario_id)
    write_workbook(out_path, scen_sum, od_out, detail_sel, dwell_hotspots, facility_rollup, lane_summary, kpis)

    # Validation output
    annual_total = float(demand.query("year == @year")["annual_pkgs"].sum())
    daily_off = float(demand.query("year == @year")["offpeak_pct_of_annual"].iloc[0])
    daily_peak = float(demand.query("year == @year")["peak_pct_of_annual"].iloc[0])
    expected_daily_total = annual_total * (daily_peak if day_type == "peak" else daily_off)
    actual_daily_total = direct_day["dir_pkgs_day"].sum() + od_out["pkgs_day"].sum()

    print(f"[{scenario_id}] Volume check: expected={expected_daily_total:,.0f}, actual={actual_daily_total:,.0f}")
    print(
        f"[{scenario_id}] Hub hierarchy: {(facility_rollup['hub_tier'] == 'primary').sum()} primary, {(facility_rollup['hub_tier'] == 'secondary').sum()} secondary hubs")
    print(
        f"[{scenario_id}] Lane summary: {len(lane_summary)} active lanes, avg {lane_summary['packages_per_truck'].mean():.0f} pkgs/truck")
    print(
        f"[{scenario_id}] Cost allocation: Total=${od_out['total_cost'].sum():,.0f} (Linehaul=${od_out['linehaul_cost'].sum():,.0f}, Touch=${od_out['touch_cost'].sum():,.0f})")

    # Return enhanced results data for consolidation
    results_data = {
        'facility_rollup': facility_rollup,
        'lane_summary': lane_summary,
        'od_selected': od_out,
        'total_cost': kpis.get('total_cost', 0),
        'cost_per_package': kpis.get('cost_per_pkg', 0),
        'arc_summary': arc_summary,
        'kpis': kpis,
        'primary_hubs': facility_rollup[facility_rollup['hub_tier'] == 'primary']['facility'].tolist(),
        'secondary_hubs': facility_rollup[facility_rollup['hub_tier'] == 'secondary']['facility'].tolist(),
        'year': year,
        'day_type': day_type,
        'strategy': strategy,
        'scenario_id': scenario_id,
    }

    return scenario_id, out_path, kpis, results_data


def _create_monday_executive_summary(base_id: str, results_by_strategy: dict, out_dir: Path):
    """
    Create executive summary with clear answers to Monday's 4 key questions.
    Enhanced with VA-based hourly throughput analysis.
    """
    if len(results_by_strategy) < 2:
        return

    # Get results for both strategies
    container_results = results_by_strategy.get('container')
    fluid_results = results_by_strategy.get('fluid')

    if not container_results or not fluid_results:
        return

    # Monday Question 1: Optimal containerization strategy
    container_cost = container_results['total_cost']
    fluid_cost = fluid_results['total_cost']
    optimal_strategy = 'container' if container_cost <= fluid_cost else 'fluid'
    cost_difference = abs(container_cost - fluid_cost)

    # Monday Question 2: Enhanced Hourly throughput by hub with VA analysis
    container_rollup = container_results['facility_rollup']
    fluid_rollup = fluid_results['facility_rollup']

    # Filter to hub/hybrid facilities only and get enhanced throughput data
    container_hubs = container_rollup[container_rollup['type'].isin(['hub', 'hybrid'])].copy()
    fluid_hubs = fluid_rollup[fluid_rollup['type'].isin(['hub', 'hybrid'])].copy()

    # Enhanced throughput analysis using the new VA-based calculations
    throughput_cols = ['injection_hourly_throughput', 'intermediate_hourly_throughput', 'lm_hourly_throughput',
                       'peak_hourly_throughput']

    # Ensure all throughput columns exist
    for col in throughput_cols:
        if col not in container_hubs.columns:
            container_hubs[col] = 0
        if col not in fluid_hubs.columns:
            fluid_hubs[col] = 0

    # Monday Question 3: Inbound/outbound trailers per facility

    # Monday Question 4: Zone mix and paths by origin facility
    container_od = container_results['od_selected']
    path_type_summary = container_od.groupby('origin')['path_type'].value_counts().unstack(fill_value=0).reset_index()

    if 'origin' in path_type_summary.columns:
        path_type_summary = path_type_summary.rename(columns={'origin': 'facility'})

    # Create executive summary sheets
    summary_data = {}

    # Sheet 1: Containerization Strategy Recommendation
    strategy_comparison = pd.DataFrame([
        {
            'strategy': 'container',
            'total_daily_cost': container_cost,
            'primary_hubs': len(container_results['primary_hubs']),
            'secondary_hubs': len(container_results['secondary_hubs']),
            'avg_packages_per_truck': container_results['lane_summary']['packages_per_truck'].mean() if not
            container_results['lane_summary'].empty else 0,
            'total_active_lanes': len(container_results['lane_summary']),
        },
        {
            'strategy': 'fluid',
            'total_daily_cost': fluid_cost,
            'primary_hubs': len(fluid_results['primary_hubs']),
            'secondary_hubs': len(fluid_results['secondary_hubs']),
            'avg_packages_per_truck': fluid_results['lane_summary']['packages_per_truck'].mean() if not fluid_results[
                'lane_summary'].empty else 0,
            'total_active_lanes': len(fluid_results['lane_summary']),
        }
    ])

    summary_data['Strategy_Comparison'] = strategy_comparison

    # Sheet 2: Hub Hourly Throughput
    hub_throughput = container_hubs[['facility', 'type', 'hub_tier'] + throughput_cols].copy()

    fluid_throughput_data = fluid_hubs.set_index('facility')[throughput_cols].add_suffix('_fluid')
    hub_throughput = hub_throughput.set_index('facility').join(fluid_throughput_data, how='left').reset_index()

    for col in hub_throughput.columns:
        if hub_throughput[col].dtype in ['float64', 'int64']:
            hub_throughput[col] = hub_throughput[col].fillna(0)

    hub_throughput = hub_throughput.sort_values('peak_hourly_throughput', ascending=False)
    summary_data['Hub_Hourly_Throughput'] = hub_throughput

    # Sheet 3: Facility Truck Requirements
    truck_requirements = container_hubs[[
        'facility', 'type', 'hub_tier', 'outbound_lane_count', 'outbound_trucks_total',
        'inbound_lane_count', 'inbound_trucks_total', 'total_trucks_per_day'
    ]].copy()
    summary_data['Facility_Truck_Requirements'] = truck_requirements

    # Sheet 4: Path Type Analysis
    if not path_type_summary.empty:
        path_analysis = path_type_summary.copy()

        numeric_cols = [col for col in path_analysis.columns if col != 'facility']
        if numeric_cols:
            path_analysis['total_paths'] = path_analysis[numeric_cols].sum(axis=1)

            for col in numeric_cols:
                if col in path_analysis.columns:
                    path_analysis[f'pct_{col}'] = (path_analysis[col] / path_analysis['total_paths'] * 100).round(1)
                    path_analysis[f'pct_{col}'] = path_analysis[f'pct_{col}'].fillna(0)

        summary_data['Path_Type_Analysis'] = path_analysis

    # Sheet 5: Monday Key Answers
    monday_answers = pd.DataFrame([
        {
            'question': '1. Optimal Containerization Strategy',
            'answer': f'{optimal_strategy.upper()} strategy',
            'detail': f'Cost difference: ${cost_difference:,.0f}/day in favor of {optimal_strategy}',
            'packages_per_truck': f"{strategy_comparison[strategy_comparison['strategy'] == optimal_strategy]['avg_packages_per_truck'].iloc[0]:.0f} packages/truck average"
        },
        {
            'question': '2. Hourly Throughput by Hub (VA-based)',
            'answer': f'Enhanced VA-based requirements calculated for {len(hub_throughput)} hubs',
            'detail': f"Peak range: {hub_throughput['peak_hourly_throughput'].min():.0f} - {hub_throughput['peak_hourly_throughput'].max():.0f} packages/hour",
            'packages_per_truck': f"Top hub: {hub_throughput.iloc[0]['facility']} at {hub_throughput.iloc[0]['peak_hourly_throughput']:.0f} pkgs/hr peak"
        },
        {
            'question': '3. Inbound/Outbound Trailers per Facility',
            'answer': f'Truck requirements calculated for all {len(truck_requirements)} facilities',
            'detail': f"Total network: {truck_requirements['total_trucks_per_day'].sum():.0f} trucks/day",
            'packages_per_truck': f"Avg outbound: {truck_requirements['outbound_trucks_total'].mean():.1f} trucks/facility"
        },
        {
            'question': '4. Zone Mix and Touch Patterns by Origin',
            'answer': 'Path analysis by origin facility completed',
            'detail': f"Network average: {container_od['path_type'].value_counts(normalize=True).round(3).to_dict()}",
            'packages_per_truck': f"Total origin facilities: {len(path_type_summary) if not path_type_summary.empty else 0}"
        }
    ])

    summary_data['Monday_Key_Answers'] = monday_answers

    # Write executive summary
    exec_summary_path = out_dir / EXECUTIVE_SUMMARY_TEMPLATE.format(base_id=base_id)
    with pd.ExcelWriter(exec_summary_path, engine="xlsxwriter") as writer:
        for sheet_name, df in summary_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"[{base_id}] Executive summary written: {exec_summary_path}")

    # Console output
    print(f"\n=== MONDAY EXECUTIVE SUMMARY ===")
    print(f"Optimal Strategy: {optimal_strategy.upper()}")
    print(f"Cost Advantage: ${cost_difference:,.0f}/day")
    print(
        f"Hub Count: {len(container_results['primary_hubs'])} primary, {len(container_results['secondary_hubs'])} secondary")
    if not hub_throughput.empty:
        print(
            f"Top Hub Throughput: {hub_throughput.iloc[0]['facility']} at {hub_throughput.iloc[0]['peak_hourly_throughput']:.0f} pkgs/hr")
    print(f"Network Truck Requirement: {truck_requirements['total_trucks_per_day'].sum():.0f} trucks/day")


def _create_consolidated_output(all_results: list, out_dir: Path):
    """
    Create consolidated multi-year, multi-strategy output file.
    """
    print(f"\n=== Creating Consolidated Multi-Year Analysis ===")

    all_facility_rollups = []
    all_lane_summaries = []
    all_od_selected = []
    all_strategy_comparisons = []

    for result in all_results:
        if result is None:
            continue

        if 'facility_rollup' in result and result['facility_rollup'] is not None:
            all_facility_rollups.append(result['facility_rollup'])

        if 'lane_summary' in result and result['lane_summary'] is not None:
            all_lane_summaries.append(result['lane_summary'])

        if 'od_selected' in result and result['od_selected'] is not None:
            od = result['od_selected'].copy()
            od['year'] = result['year']
            od['strategy'] = result['strategy']
            all_od_selected.append(od)

        all_strategy_comparisons.append({
            'year': result['year'],
            'day_type': result['day_type'],
            'strategy': result['strategy'],
            'scenario_id': result['scenario_id'],
            'total_cost': result.get('total_cost', 0),
            'cost_per_pkg': result.get('cost_per_package', 0),
            'primary_hubs': len(result.get('primary_hubs', [])),
            'secondary_hubs': len(result.get('secondary_hubs', [])),
        })

    consolidated_data = {}

    if all_facility_rollups:
        consolidated_data['Facility_Rollup'] = pd.concat(all_facility_rollups, ignore_index=True)

        hub_throughput = consolidated_data['Facility_Rollup'][
            consolidated_data['Facility_Rollup']['type'].isin(['hub', 'hybrid'])
        ][['facility', 'market', 'region', 'type', 'hub_tier', 'year', 'day_type', 'strategy',
           'injection_hourly_throughput', 'intermediate_hourly_throughput',
           'lm_hourly_throughput', 'peak_hourly_throughput']].copy()
        consolidated_data['Hub_Hourly_Throughput'] = hub_throughput

        truck_reqs = consolidated_data['Facility_Rollup'][
            ['facility', 'type', 'hub_tier', 'year', 'day_type', 'strategy',
             'outbound_lane_count', 'outbound_trucks_total',
             'inbound_lane_count', 'inbound_trucks_total', 'total_trucks_per_day']
        ].copy()
        consolidated_data['Facility_Truck_Requirements'] = truck_reqs

    if all_lane_summaries:
        consolidated_data['Lane_Summary'] = pd.concat(all_lane_summaries, ignore_index=True)

    if all_od_selected:
        od_combined = pd.concat(all_od_selected, ignore_index=True)
        consolidated_data['OD_Selected_Paths'] = od_combined

        path_analysis = od_combined.groupby(['year', 'day_type', 'strategy', 'origin'])[
            'path_type'].value_counts().unstack(fill_value=0).reset_index()
        path_analysis = path_analysis.rename(columns={'origin': 'facility'})
        consolidated_data['Path_Type_Analysis'] = path_analysis

    if all_strategy_comparisons:
        consolidated_data['Strategy_Comparison'] = pd.DataFrame(all_strategy_comparisons)

    consolidated_path = out_dir / CONSOLIDATED_OUTPUT
    with pd.ExcelWriter(consolidated_path, engine="xlsxwriter") as writer:
        for sheet_name, df in consolidated_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  ✓ {sheet_name}: {len(df):,} rows")

    print(f"✅ Consolidated analysis written: {consolidated_path}")
    print(f"   Total sheets: {len(consolidated_data)}")


def main(input_path: str, output_dir: str | None):
    inp = Path(input_path)
    out_dir = Path(output_dir) if output_dir else Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading workbook from: {inp}")
    dfs = load_workbook(inp)
    validate_inputs(dfs)

    facilities = dfs["facilities"].copy()
    zips = dfs["zips"].copy()
    demand = dfs["demand"].copy()
    inj = dfs["injection_distribution"].copy()
    mb = dfs["mileage_bands"].sort_values("mileage_band_min").reset_index(drop=True)
    timing = params_to_dict(dfs["timing_params"])
    costs = params_to_dict(dfs["cost_params"])
    cont = dfs["container_params"].copy()
    pkgmix = dfs["package_mix"].copy()
    run_kv = params_to_dict(dfs["run_settings"])
    scenarios = dfs["scenarios"].copy()

    # Add default parameters
    if "hours_per_touch" not in timing:
        timing["hours_per_touch"] = 6.0
        print("Added default hours_per_touch: 6.0")

    if "sla_penalty_per_touch_per_pkg" not in costs:
        costs["sla_penalty_per_touch_per_pkg"] = 0.25
        print("Added default sla_penalty_per_touch_per_pkg: 0.25")

    if "injection_va_hours" not in timing:
        timing["injection_va_hours"] = 8.0
        print("Added default injection_va_hours: 8.0")

    if "middle_mile_va_hours" not in timing:
        timing["middle_mile_va_hours"] = 16.0
        print("Added default middle_mile_va_hours: 16.0")

    if "last_mile_va_hours" not in timing:
        timing["last_mile_va_hours"] = 4.0
        print("Added default last_mile_va_hours: 4.0")

    compare_mode = str(run_kv.get("compare_mode", "single")).strip().lower()
    if compare_mode not in {"single", "paired"}:
        raise ValueError(f"run_settings.compare_mode must be 'single' or 'paired', got '{compare_mode}'")

    print(f"Running in {compare_mode} mode")

    all_results = []

    for _, s in scenarios.iterrows():
        base_id = str(s.get("pair_id", f"{int(s['year'])}_{str(s['day_type']).strip().lower()}")).strip()

        if compare_mode == "single":
            strategy = str(run_kv.get("load_strategy", "container")).strip().lower()
            scenario_id, out_path, _kpis, results_data = _run_one_strategy(
                base_id, strategy, facilities, zips, demand, inj, mb, timing, costs, cont, pkgmix, run_kv, s, out_dir
            )
            if out_path:
                print(f"✅ Wrote {out_path}")
                all_results.append(results_data)
        else:
            results_by_strategy = {}
            per_base = []

            for strategy in ["container", "fluid"]:
                scenario_id, out_path, kpis, results_data = _run_one_strategy(
                    base_id, strategy, facilities, zips, demand, inj, mb, timing, costs, cont, pkgmix, run_kv, s,
                    out_dir
                )
                if out_path is not None and not kpis.empty:
                    results_by_strategy[strategy] = results_data
                    all_results.append(results_data)

                    rec = kpis.to_dict()
                    rec.update({
                        "base_id": base_id,
                        "scenario_id": scenario_id,
                        "strategy": strategy,
                        "output_file": str(out_path),
                    })
                    per_base.append(rec)
                    print(f"✅ Completed {scenario_id}")

            if per_base:
                compare_df = pd.DataFrame(per_base)
                compare_path = out_dir / COMPARE_FILE_TEMPLATE.format(base_id=base_id)
                write_compare_workbook(compare_path, compare_df, run_kv)
                print(f"✅ Comparison written: {compare_path}")

                _create_monday_executive_summary(base_id, results_by_strategy, out_dir)

    if all_results:
        _create_consolidated_output(all_results, out_dir)

    print(f"\n🎉 All analysis complete! Results in: {out_dir.resolve()}")


def _scenario_summary(run_kv, scenario_id, year, day_type, mb, timing, cont):
    g = cont[cont["container_type"].str.lower() == "gaylord"].iloc[0]
    rows = [
        {"key": "scenario_id", "value": scenario_id},
        {"key": "demand_year", "value": year},
        {"key": "day_type", "value": day_type},
        {"key": "load_strategy", "value": str(timing.get("load_strategy", "container"))},
        {"key": "hours_per_touch", "value": float(timing.get("hours_per_touch", 6.0))},
        {"key": "injection_va_hours", "value": float(timing.get("injection_va_hours", 8.0))},
        {"key": "middle_mile_va_hours", "value": float(timing.get("middle_mile_va_hours", 16.0))},
        {"key": "last_mile_va_hours", "value": float(timing.get("last_mile_va_hours", 4.0))},
        {"key": "usable_cube_cuft", "value": float(g["usable_cube_cuft"])},
        {"key": "pack_utilization_container", "value": float(g["pack_utilization_container"])},
        {"key": "containers_per_truck", "value": int(g["containers_per_truck"])},
        {"key": "trailer_air_cube_cuft", "value": float(g.get("trailer_air_cube_cuft", 4060.0))},
        {"key": "pack_utilization_fluid", "value": float(g.get("pack_utilization_fluid", 0.85))},
        {"key": "sla_target_days", "value": int(run_kv.get("sla_target_days", 3))},
        {"key": "path_around_the_world_factor", "value": float(run_kv.get("path_around_the_world_factor", 2.0))},
        {"key": "enforce_parent_hub_over_miles", "value": int(run_kv.get("enforce_parent_hub_over_miles", 500))},
    ]
    return pd.DataFrame(rows)


def _network_kpis(od_selected: pd.DataFrame) -> pd.Series:
    tot_cost = od_selected.get("total_cost", od_selected.get("cost_candidate_path", pd.Series([0]))).sum()
    tot_pkgs = od_selected["pkgs_day"].sum()
    return pd.Series({
        "total_cost": tot_cost,
        "cost_per_pkg": (tot_cost / tot_pkgs) if tot_pkgs > 0 else np.nan,
        "num_ods": len(od_selected),
        "sla_violations": int((od_selected["end_to_end_sla_flag"] == 1).sum()),
        "around_world_flags": int((od_selected["around_world_flag"] == 1).sum()),
        "pct_direct": round(100 * (od_selected["path_type"] == "direct").mean(), 2),
        "pct_1_touch": round(100 * (od_selected["path_type"] == "1_touch").mean(), 2),
        "pct_2_touch": round(100 * (od_selected["path_type"] == "2_touch").mean(), 2),
        "pct_3_touch": round(100 * (od_selected["path_type"] == "3_touch").mean(), 2),
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input workbook (.xlsx)")
    ap.add_argument("--output_dir", default=None, help="Output directory (default: ./outputs)")
    args = ap.parse_args()
    main(args.input, args.output_dir)