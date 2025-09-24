# run_v1.py - ENHANCED with Sort Point Optimization, Realistic Fill Rates, and Updated File Naming
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from veho_net.io_loader import load_workbook, params_to_dict
from veho_net.validators import validate_inputs
from veho_net.build_structures import build_od_and_direct, candidate_paths
from veho_net.time_cost import path_cost_and_time, containers_for_pkgs_day, format_realistic_fill_summary, \
    print_enhanced_fill_summary
from veho_net.milp import solve_arc_pooled_path_selection
from veho_net.reporting import (
    build_od_selected_outputs,
    build_dwell_hotspots,
    build_facility_rollup,
    add_zone,
    build_lane_summary,
)
from veho_net.write_outputs import write_workbook, write_compare_workbook

# NEW: Import sort optimization module
from veho_net.sort_optimization import (
    validate_sort_capacity_feasibility,
    calculate_containerization_costs,
    optimize_sort_allocation,
    apply_sort_allocation,
    summarize_sort_allocation
)

# UPDATED: Shorter output naming templates with scenario_id preservation
OUTPUT_FILE_TEMPLATE = "{scenario_id}_results.xlsx"  # Removed "_v1"
COMPARE_FILE_TEMPLATE = "{base_id}_compare.xlsx"  # Kept as-is
EXECUTIVE_SUMMARY_TEMPLATE = "{base_id}_summary.xlsx"  # "executive_summary" → "summary"
CONSOLIDATED_OUTPUT = "Network_Analysis_Consolidated.xlsx"  # "All_Scenarios" → "Consolidated"


def safe_scenario_filename(scenario_id: str, suffix: str = "results") -> str:
    """
    Create safe filenames that preserve scenario_id but prevent length issues.

    Examples:
      safe_scenario_filename("2025_peak_container", "results")
      → "2025_peak_container_results.xlsx"

      safe_scenario_filename("Very_Long_Scenario_Name_2025_peak_container", "results")
      → "VeryLong2025_peak_container_results.xlsx"  # Abbreviated but preserves key info
    """
    # Windows filename limit is ~255 chars, but let's keep under 50 for safety
    max_length = 45  # Leave room for suffix + ".xlsx"

    if len(f"{scenario_id}_{suffix}.xlsx") <= max_length:
        return f"{scenario_id}_{suffix}.xlsx"

    # If too long, intelligently abbreviate while preserving year_daytype_strategy
    parts = scenario_id.split('_')

    if len(parts) >= 3:
        # Try to preserve last 3 parts (likely year_daytype_strategy)
        abbreviated = ''.join([p[:4] for p in parts[:-3]]) + '_' + '_'.join(parts[-3:])
        if len(f"{abbreviated}_{suffix}.xlsx") <= max_length:
            return f"{abbreviated}_{suffix}.xlsx"

    # Last resort: truncate but ensure scenario_id is still recognizable
    truncated = scenario_id[:max_length - len(suffix) - 6]  # -6 for "_{suffix}.xlsx"
    return f"{truncated}_{suffix}.xlsx"


def get_consolidated_filename() -> str:
    """Generate timestamped consolidated filename to avoid overwrites."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"Network_Analysis_{timestamp}.xlsx"


def _add_default_parameters(timing: dict, costs: dict) -> tuple[dict, dict]:
    """Add default parameters for backwards compatibility and new features."""

    # Enhanced timing parameters
    timing_defaults = {
        "hours_per_touch": 6.0,
        "injection_va_hours": 8.0,
        "middle_mile_va_hours": 16.0,
        "last_mile_va_hours": 4.0,
        "sort_points_per_destination": 2,
    }

    # Enhanced cost parameters
    cost_defaults = {
        "sla_penalty_per_touch_per_pkg": 0.25,
        "sort_setup_cost_per_point": 0.0,  # Set to 0 initially
        "last_mile_sort_cost_per_pkg": 0.5,  # Default estimate
        "premium_economy_dwell_threshold": 0.10,
        "allow_premium_economy_dwell": True,
        "dwell_cost_per_pkg_per_day": 0.05,
    }

    # Add defaults for missing parameters
    for key, default_val in timing_defaults.items():
        if key not in timing:
            timing[key] = default_val
            print(f"Added default timing parameter {key}: {default_val}")

    for key, default_val in cost_defaults.items():
        if key not in costs:
            costs[key] = default_val
            print(f"Added default cost parameter {key}: {default_val}")

    # Handle parameter renames
    if "last_mile_cpp" in costs and "last_mile_delivery_cost_per_pkg" not in costs:
        costs["last_mile_delivery_cost_per_pkg"] = costs["last_mile_cpp"]
        print("Renamed last_mile_cpp → last_mile_delivery_cost_per_pkg")

    return timing, costs


def _allocate_lane_costs_to_ods(od_selected: pd.DataFrame, arc_summary: pd.DataFrame, costs: dict,
                                strategy: str) -> pd.DataFrame:
    """
    Enhanced cost allocation with containerization level awareness.
    """
    od = od_selected.copy()

    # Calculate touch costs per OD based on containerization level
    touch_map = {"direct": 0, "1_touch": 1, "2_touch": 2, "3_touch": 3, "4_touch": 4}
    od['num_touches'] = od['path_type'].map(touch_map).fillna(0)

    crossdock_pp = float(costs.get("crossdock_touch_cost_per_pkg", 0.0))
    sort_pp = float(costs.get("sort_cost_per_pkg", 0.0))
    lm_sort_pp = float(costs.get("last_mile_sort_cost_per_pkg", 0.0))

    if strategy.lower() == "container":
        # Container: crossdock touches only (intermediate hubs)
        od['touch_cost'] = od['num_touches'] * crossdock_pp * od['pkgs_day']
        od['touch_cpp'] = od['num_touches'] * crossdock_pp

        # Add last mile sort cost based on containerization level
        if 'containerization_level' in od.columns:
            # Enhanced: Variable last mile sort based on containerization level
            lm_sort_multiplier = od['containerization_level'].map({
                'region': 1.0,  # Full sort at region level
                'market': 0.5,  # Reduced sort at market level
                'sort_group': 0.1  # Minimal sort for pre-sorted containers
            }).fillna(1.0)

            od['lm_sort_cost'] = lm_sort_multiplier * lm_sort_pp * od['pkgs_day']
            od['touch_cost'] += od['lm_sort_cost']
            od['touch_cpp'] += lm_sort_multiplier * lm_sort_pp

    else:
        # Fluid: intermediate touches + destination sort (NOT origin - that's injection_sort)
        od['touch_cost'] = (od['num_touches'] + 1) * sort_pp * od['pkgs_day']  # touches + dest
        od['touch_cpp'] = (od['num_touches'] + 1) * sort_pp

    # Allocate linehaul costs from arc_summary (unchanged logic)
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
    UPDATED: Enhanced strategy execution with realistic fill rate reporting and updated file naming.
    """
    # Scenario setup
    year = int(scenario_row["year"]) if "year" in scenario_row else int(scenario_row["demand_year"])
    day_type = str(scenario_row["day_type"]).strip().lower()
    scenario_id = f"{base_id}_{strategy}"

    print(f"\n=== Running {scenario_id} with Enhanced Sort Optimization & Realistic Fill Rates ===")

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

    # Rename column for consistency
    od = od.rename(columns={od_day_col: "pkgs_day"})

    print(f"[{scenario_id}] Generated {len(od)} OD pairs")

    # NEW: Sort Point Capacity Validation
    try:
        validate_sort_capacity_feasibility(facilities, od, timing_local)
        print("✅ Sort capacity validation passed")
    except ValueError as e:
        print(f"❌ Sort capacity validation failed: {e}")
        return scenario_id, None, pd.Series(dtype=float), None

    # NEW: Calculate containerization costs for different levels
    if strategy.lower() == "container":
        print("🔄 Calculating containerization costs...")
        cost_analysis = calculate_containerization_costs(od, facilities, mb, costs, timing_local)

        print("🎯 Optimizing sort allocation...")
        sort_allocation = optimize_sort_allocation(cost_analysis, facilities, timing_local)

        # Apply sort allocation to OD data with all required parameters
        od = apply_sort_allocation(od, sort_allocation, cost_analysis, facilities)

        # Create allocation summary
        allocation_summary = summarize_sort_allocation(od, cost_analysis, sort_allocation)

        print(f"✅ Sort optimization complete - {len(allocation_summary)} allocations made")
    else:
        # Fluid strategy: no containerization optimization needed
        od['containerization_level'] = 'market'  # Default for fluid
        allocation_summary = pd.DataFrame()

    # Generate candidate paths with hub hierarchy
    around_factor = float(run_kv.get("path_around_the_world_factor", 2.0))
    cands = candidate_paths(od, facilities, mb, around_factor=around_factor)

    if cands.empty:
        print(f"[{scenario_id}] No candidate paths generated. Skipping.")
        return scenario_id, None, pd.Series(dtype=float), None

    print(f"[{scenario_id}] Generated {len(cands)} candidate paths")

    # Enhanced path costing with containerization awareness and realistic fill rates
    cand_rows, detail_rows, direct_dist = [], [], {}

    for _, r in cands.iterrows():
        sub = od[(od["origin"] == r["origin"]) & (od["dest"] == r["dest"])]
        if sub.empty:
            continue
        pkgs_day = float(sub.iloc[0]["pkgs_day"])

        try:
            cost, hours, sums, steps = path_cost_and_time(
                r, facilities, mb, timing_local, costs, pkgmix, cont, pkgs_day
            )
            conts = containers_for_pkgs_day(pkgs_day, pkgmix, cont)

            # Enhanced candidate data with realistic fill rates
            cand_data = {
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
                "cost_candidate_path": cost,
                "total_facilities_touched": sums.get("total_facilities_touched", len(r.get("path_nodes", []))),
                # UPDATED: Enhanced realistic fill rate metrics
                "container_fill_rate": sums.get("container_fill_rate", 0.0),  # Now realistic
                "truck_fill_rate": sums.get("truck_fill_rate", 0.0),  # Now realistic
                "packages_dwelled": sums.get("packages_dwelled", 0),
                # NEW: Add debugging info for fill rate validation
                "raw_container_cube": sums.get("raw_container_cube", 0),
                "raw_truck_cube": sums.get("raw_truck_cube", 0),
                "effective_utilization_factor": sums.get("effective_utilization_factor", 0.85),
            }

            # Add containerization level if available
            if 'containerization_level' in sub.columns:
                cand_data["containerization_level"] = sub.iloc[0]['containerization_level']

            cand_rows.append(cand_data)

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
            print(f"[{scenario_id}] Error processing path {r.get('path_str', 'unknown')}: {e}")
            continue

    cand_tbl = pd.DataFrame(cand_rows)
    detail_all = pd.DataFrame(detail_rows)

    if cand_tbl.empty:
        print(f"[{scenario_id}] No valid candidate paths for {day_type}. Skipping.")
        return scenario_id, None, pd.Series(dtype=float), None

    print(f"[{scenario_id}] Processed {len(cand_tbl)} valid paths with realistic fill rate metrics")

    # Enhanced MILP path selection
    selected_basic, arc_summary = solve_arc_pooled_path_selection(
        cand_tbl[["scenario_id", "origin", "dest", "day_type", "path_type", "pkgs_day", "path_nodes", "path_str",
                  "containers_cont"]],
        facilities, mb, pkgmix, cont, costs,
    )

    # Enhanced merge with candidate metrics
    merge_keys = ["scenario_id", "origin", "dest", "day_type", "path_str"]
    cand_tbl_unique = cand_tbl.drop_duplicates(subset=merge_keys)
    selected = selected_basic.merge(cand_tbl_unique, on=merge_keys, how="left", suffixes=("", "_cand"))

    # Normalize column names and handle enhanced metrics
    cand_rename = {
        "distance_miles_cand": "distance_miles",
        "linehaul_hours_cand": "linehaul_hours",
        "handling_hours_cand": "handling_hours",
        "dwell_hours_cand": "dwell_hours",
        "destination_dwell_hours_cand": "destination_dwell_hours",
        "sla_days_cand": "sla_days",
        "time_hours_cand": "time_hours",
        "pkgs_day_cand": "pkgs_day_ref",
        "container_fill_rate_cand": "container_fill_rate",  # Realistic fill rate
        "truck_fill_rate_cand": "truck_fill_rate",  # Realistic fill rate
        "packages_dwelled_cand": "packages_dwelled",
        "containerization_level_cand": "containerization_level",
        # NEW: Debugging columns for fill rate validation
        "raw_container_cube_cand": "raw_container_cube",
        "raw_truck_cube_cand": "raw_truck_cube",
        "effective_utilization_factor_cand": "effective_utilization_factor",
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

    # Enhanced cost allocation with containerization awareness
    od_out = _allocate_lane_costs_to_ods(od_out, arc_summary, costs, strategy)

    # Add complete cost structure
    sort_cost = float(costs.get("sort_cost_per_pkg", 0.0))
    lm_delivery_cost = float(costs.get("last_mile_delivery_cost_per_pkg", 0.0))

    # Add origin injection sort cost
    od_out['injection_sort_cost'] = sort_cost * od_out['pkgs_day']
    od_out['injection_sort_cpp'] = sort_cost

    # Add destination last mile delivery cost
    od_out['last_mile_delivery_cost'] = lm_delivery_cost * od_out['pkgs_day']
    od_out['last_mile_delivery_cpp'] = lm_delivery_cost

    # Recalculate total cost with all components
    od_out['total_cost'] = (
            od_out['injection_sort_cost'] +  # Origin sort
            od_out['linehaul_cost'] +  # Transportation
            od_out['touch_cost'] +  # Intermediate + destination processing
            od_out['last_mile_delivery_cost']  # Final delivery
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

    # Enhanced facility rollup
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

    # Add scenario identification columns
    facility_rollup['year'] = year
    facility_rollup['day_type'] = day_type
    facility_rollup['strategy'] = strategy

    # Enhanced lane summary
    lane_summary = build_lane_summary(arc_summary)
    if not lane_summary.empty:
        lane_summary['year'] = year
        lane_summary['day_type'] = day_type
        lane_summary['strategy'] = strategy
        lane_summary['scenario_id'] = scenario_id

    # Scenario summary with enhanced metrics
    scen_sum = _scenario_summary(run_kv, scenario_id, year, day_type, mb, timing_local, cont)
    dwell_hotspots = build_dwell_hotspots(detail_sel)
    kpis = _network_kpis(od_out)

    # Add enhanced KPIs with realistic fill rates
    enhanced_kpis = pd.Series({
        "avg_container_fill_rate": od_out.get('container_fill_rate', pd.Series([0])).mean(),  # Realistic
        "avg_truck_fill_rate": od_out.get('truck_fill_rate', pd.Series([0])).mean(),  # Realistic
        "total_packages_dwelled": od_out.get('packages_dwelled', pd.Series([0])).sum(),
        "sort_optimization_savings": allocation_summary[
            'daily_cost_savings'].sum() if not allocation_summary.empty else 0,
        # NEW: Add theoretical fill rates for comparison
        "theoretical_container_fill_rate": od_out.get('container_fill_rate', pd.Series([0])).mean() /
                                           od_out.get('effective_utilization_factor', pd.Series([0.85])).mean()
        if strategy.lower() == "container" else 0,
        "theoretical_truck_fill_rate": od_out.get('truck_fill_rate', pd.Series([0])).mean() /
                                       od_out.get('effective_utilization_factor', pd.Series([0.85])).mean(),
    })
    kpis = pd.concat([kpis, enhanced_kpis])

    # UPDATED: Write individual scenario results with safe filename
    out_filename = safe_scenario_filename(scenario_id, "results")
    out_path = out_dir / out_filename

    # Always log the filename mapping for traceability
    print(f"[{scenario_id}] Writing to: {out_filename}")

    write_workbook(out_path, scen_sum, od_out, detail_sel, dwell_hotspots, facility_rollup, lane_summary, kpis)

    # Add sort allocation summary to output if available
    if not allocation_summary.empty:
        with pd.ExcelWriter(out_path, mode='a', engine="openpyxl") as writer:
            allocation_summary.to_excel(writer, sheet_name='sort_allocation_summary', index=False)

    # UPDATED: Enhanced validation output with realistic fill rate reporting
    annual_total = float(demand.query("year == @year")["annual_pkgs"].sum())
    daily_off = float(demand.query("year == @year")["offpeak_pct_of_annual"].iloc[0])
    daily_peak = float(demand.query("year == @year")["peak_pct_of_annual"].iloc[0])
    expected_daily_total = annual_total * (daily_peak if day_type == "peak" else daily_off)
    actual_daily_total = direct_day["dir_pkgs_day"].sum() + od_out["pkgs_day"].sum()

    print(f"[{scenario_id}] Volume check: expected={expected_daily_total:,.0f}, actual={actual_daily_total:,.0f}")
    print(
        f"[{scenario_id}] Hub hierarchy: {(facility_rollup['hub_tier'] == 'primary').sum()} primary, {(facility_rollup['hub_tier'] == 'secondary').sum()} secondary hubs")
    print(f"[{scenario_id}] Lane summary: {len(lane_summary)} active lanes")

    # UPDATED: Realistic fill rate reporting
    if not lane_summary.empty and not od_out.empty:
        avg_truck_fill_realistic = od_out.get('truck_fill_rate', pd.Series([0])).mean()
        avg_container_fill_realistic = od_out.get('container_fill_rate', pd.Series([0])).mean()

        # Get pack utilization factors for comparison
        if strategy.lower() == "container":
            gaylord_row = cont[cont["container_type"].str.lower() == "gaylord"].iloc[0]
            pack_util_container = float(gaylord_row.get("pack_utilization_container", 0.85))
            pack_util_truck = pack_util_container  # Same factor for container truck capacity

            # Calculate what these would be as theoretical fill rates
            theoretical_container_fill = avg_container_fill_realistic / pack_util_container if pack_util_container > 0 else 0
            theoretical_truck_fill = avg_truck_fill_realistic / pack_util_truck if pack_util_truck > 0 else 0

            print(f"[{scenario_id}]   Realistic fill rates:")
            print(
                f"[{scenario_id}]     Container: {avg_container_fill_realistic:.1%} actual ({theoretical_container_fill:.1%} usable)")
            print(
                f"[{scenario_id}]     Truck: {avg_truck_fill_realistic:.1%} actual ({theoretical_truck_fill:.1%} usable)")
            print(f"[{scenario_id}]     → Based on {pack_util_container:.0%} container pack factor")

        else:
            # Fluid strategy
            pack_util_fluid = float(cont.get("pack_utilization_fluid", pd.Series([0.85])).iloc[0])
            theoretical_truck_fill = avg_truck_fill_realistic / pack_util_fluid if pack_util_fluid > 0 else 0

            print(f"[{scenario_id}]   Realistic fill rates:")
            print(
                f"[{scenario_id}]     Truck: {avg_truck_fill_realistic:.1%} actual ({theoretical_truck_fill:.1%} usable)")
            print(f"[{scenario_id}]     → Based on {pack_util_fluid:.0%} fluid pack factor")

    print(f"[{scenario_id}] Cost allocation: Total=${od_out['total_cost'].sum():,.0f}")
    if not allocation_summary.empty:
        print(f"[{scenario_id}] Sort optimization: ${allocation_summary['daily_cost_savings'].sum():,.0f}/day savings")

    # Return enhanced results data for consolidation
    results_data = {
        'facility_rollup': facility_rollup,
        'lane_summary': lane_summary,
        'od_selected': od_out,
        'sort_allocation_summary': allocation_summary,
        'total_cost': kpis.get('total_cost', 0),
        'cost_per_package': kpis.get('cost_per_pkg', 0),
        'sort_optimization_savings': enhanced_kpis.get('sort_optimization_savings', 0),
        'avg_container_fill_rate': enhanced_kpis.get('avg_container_fill_rate', 0),  # Realistic
        'avg_truck_fill_rate': enhanced_kpis.get('avg_truck_fill_rate', 0),  # Realistic
        'theoretical_container_fill_rate': enhanced_kpis.get('theoretical_container_fill_rate', 0),
        'theoretical_truck_fill_rate': enhanced_kpis.get('theoretical_truck_fill_rate', 0),
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
    UPDATED: Enhanced executive summary with realistic fill rate insights and updated file naming.
    """
    if len(results_by_strategy) < 2:
        return

    container_results = results_by_strategy.get('container')
    fluid_results = results_by_strategy.get('fluid')

    if not container_results or not fluid_results:
        return

    # Enhanced strategy comparison
    container_cost = container_results['total_cost']
    fluid_cost = fluid_results['total_cost']
    optimal_strategy = 'container' if container_cost <= fluid_cost else 'fluid'
    cost_difference = abs(container_cost - fluid_cost)

    # Sort optimization impact
    sort_savings = container_results.get('sort_optimization_savings', 0)

    # UPDATED: Realistic fill rate analysis
    container_truck_fill = container_results.get('avg_truck_fill_rate', 0)
    container_container_fill = container_results.get('avg_container_fill_rate', 0)
    fluid_truck_fill = fluid_results.get('avg_truck_fill_rate', 0)

    # Get theoretical fill rates for comparison
    container_truck_theoretical = container_results.get('theoretical_truck_fill_rate', 0)
    container_container_theoretical = container_results.get('theoretical_container_fill_rate', 0)
    fluid_truck_theoretical = fluid_results.get('theoretical_truck_fill_rate', 0)

    # Create enhanced summary data
    summary_data = {}

    # Sheet 1: Enhanced Strategy Comparison with realistic fill rates
    strategy_comparison = pd.DataFrame([
        {
            'strategy': 'container',
            'total_daily_cost': container_cost,
            'sort_optimization_savings': container_results.get('sort_optimization_savings', 0),
            'net_daily_cost': container_cost - container_results.get('sort_optimization_savings', 0),
            'realistic_container_fill_rate': container_container_fill,  # Realistic
            'realistic_truck_fill_rate': container_truck_fill,  # Realistic
            'theoretical_container_fill_rate': container_container_theoretical,  # Theoretical
            'theoretical_truck_fill_rate': container_truck_theoretical,  # Theoretical
            'primary_hubs': len(container_results['primary_hubs']),
            'secondary_hubs': len(container_results['secondary_hubs']),
        },
        {
            'strategy': 'fluid',
            'total_daily_cost': fluid_cost,
            'sort_optimization_savings': 0,  # Fluid doesn't have sort optimization
            'net_daily_cost': fluid_cost,
            'realistic_container_fill_rate': 0,  # Fluid doesn't use containers
            'realistic_truck_fill_rate': fluid_truck_fill,  # Realistic
            'theoretical_container_fill_rate': 0,
            'theoretical_truck_fill_rate': fluid_truck_theoretical,  # Theoretical
            'primary_hubs': len(fluid_results['primary_hubs']),
            'secondary_hubs': len(fluid_results['secondary_hubs']),
        }
    ])

    summary_data['Enhanced_Strategy_Comparison'] = strategy_comparison

    # Add other sheets from original function...
    container_rollup = container_results['facility_rollup']
    fluid_rollup = fluid_results['facility_rollup']

    container_hubs = container_rollup[container_rollup['type'].isin(['hub', 'hybrid'])].copy()
    fluid_hubs = fluid_rollup[fluid_rollup['type'].isin(['hub', 'hybrid'])].copy()

    throughput_cols = ['injection_hourly_throughput', 'intermediate_hourly_throughput', 'lm_hourly_throughput',
                       'peak_hourly_throughput']

    for col in throughput_cols:
        if col not in container_hubs.columns:
            container_hubs[col] = 0
        if col not in fluid_hubs.columns:
            fluid_hubs[col] = 0

    hub_throughput = container_hubs[['facility', 'type', 'hub_tier'] + throughput_cols].copy()
    fluid_throughput_data = fluid_hubs.set_index('facility')[throughput_cols].add_suffix('_fluid')
    hub_throughput = hub_throughput.set_index('facility').join(fluid_throughput_data, how='left').reset_index()

    for col in hub_throughput.columns:
        if hub_throughput[col].dtype in ['float64', 'int64']:
            hub_throughput[col] = hub_throughput[col].fillna(0)

    hub_throughput = hub_throughput.sort_values('peak_hourly_throughput', ascending=False)
    summary_data['Hub_Hourly_Throughput'] = hub_throughput

    # UPDATED: Enhanced Monday answers with realistic fill rate insights
    monday_answers = pd.DataFrame([
        {
            'question': '1. Optimal Containerization Strategy',
            'answer': f'{optimal_strategy.upper()} strategy',
            'detail': f'Base cost difference: ${cost_difference:,.0f}/day. Sort optimization saves additional ${sort_savings:,.0f}/day',
            'realistic_metrics': f"Realistic fill rates - Container: {container_container_fill:.1%}, Truck: {container_truck_fill:.1%} of total cubic capacity"
        },
        {
            'question': '2. Realistic vs Theoretical Fill Rate Analysis',
            'answer': f'Container strategy uses {container_truck_fill:.1%} of total truck cubic capacity (vs {container_truck_theoretical:.1%} theoretical)',
            'detail': f'This is actual physical space utilization, not theoretical maximum after pack factors',
            'realistic_metrics': f'Compare to fluid: {fluid_truck_fill:.1%} actual vs {fluid_truck_theoretical:.1%} theoretical truck fill rate'
        },
        {
            'question': '3. Sort Point Optimization Impact',
            'answer': f'${sort_savings:,.0f}/day potential savings from optimal containerization',
            'detail': f'Hybrid allocation optimizes {len(container_results.get("sort_allocation_summary", pd.DataFrame()))} OD pairs',
            'realistic_metrics': f'Improved consolidation efficiency through targeted deeper containerization'
        },
        {
            'question': '4. Operational Reality Check',
            'answer': f'Realistic fill rates provide true operational picture vs misleading theoretical rates',
            'detail': f'Previous "high fill rates" were inflated due to pack factor calculations',
            'realistic_metrics': f"Container: {container_container_fill:.1%} real vs {container_container_theoretical:.1%} theoretical, Truck: {container_truck_fill:.1%} real vs {container_truck_theoretical:.1%} theoretical"
        }
    ])

    summary_data['Enhanced_Monday_Answers'] = monday_answers

    # Add sort allocation details if available
    if 'sort_allocation_summary' in container_results and not container_results['sort_allocation_summary'].empty:
        summary_data['Sort_Allocation_Details'] = container_results['sort_allocation_summary']

    # UPDATED: Write enhanced executive summary with updated filename template
    exec_summary_filename = safe_scenario_filename(base_id, "summary")
    exec_summary_path = out_dir / exec_summary_filename

    with pd.ExcelWriter(exec_summary_path, engine="xlsxwriter") as writer:
        for sheet_name, df in summary_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"[{base_id}] Enhanced executive summary written: {exec_summary_filename}")
    print(f"\n=== ENHANCED MONDAY EXECUTIVE SUMMARY ===")
    print(f"Optimal Strategy: {optimal_strategy.upper()}")
    print(f"Base Cost Advantage: ${cost_difference:,.0f}/day")
    print(f"Sort Optimization Bonus: ${sort_savings:,.0f}/day")
    print(f"Combined Daily Savings: ${cost_difference + sort_savings:,.0f}/day")
    print(f"REALISTIC Fill Rates:")
    print(
        f"  Container: {container_container_fill:.1%} of total container space ({container_container_theoretical:.1%} theoretical)")
    print(
        f"  Truck: {container_truck_fill:.1%} of total truck space (container strategy) ({container_truck_theoretical:.1%} theoretical)")
    print(
        f"  Truck: {fluid_truck_fill:.1%} of total truck space (fluid strategy) ({fluid_truck_theoretical:.1%} theoretical)")
    print(f"  → These are actual physical space utilization rates vs theoretical usable space rates")


def _create_consolidated_output(all_results: list, out_dir: Path):
    """UPDATED: Enhanced consolidated output with realistic fill rate metrics and updated filename."""
    print(f"\n=== Creating Enhanced Consolidated Multi-Year Analysis ===")

    all_facility_rollups = []
    all_lane_summaries = []
    all_od_selected = []
    all_strategy_comparisons = []
    all_sort_allocations = []

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

        if 'sort_allocation_summary' in result and not result['sort_allocation_summary'].empty:
            sort_alloc = result['sort_allocation_summary'].copy()
            sort_alloc['year'] = result['year']
            sort_alloc['day_type'] = result['day_type']
            sort_alloc['strategy'] = result['strategy']
            all_sort_allocations.append(sort_alloc)

        # UPDATED: Enhanced strategy comparison with realistic fill rates
        all_strategy_comparisons.append({
            'year': result['year'],
            'day_type': result['day_type'],
            'strategy': result['strategy'],
            'scenario_id': result['scenario_id'],
            'total_cost': result.get('total_cost', 0),
            'cost_per_pkg': result.get('cost_per_package', 0),
            'sort_optimization_savings': result.get('sort_optimization_savings', 0),
            'realistic_container_fill_rate': result.get('avg_container_fill_rate', 0),  # Realistic
            'realistic_truck_fill_rate': result.get('avg_truck_fill_rate', 0),  # Realistic
            'theoretical_container_fill_rate': result.get('theoretical_container_fill_rate', 0),  # Theoretical
            'theoretical_truck_fill_rate': result.get('theoretical_truck_fill_rate', 0),  # Theoretical
            'primary_hubs': len(result.get('primary_hubs', [])),
            'secondary_hubs': len(result.get('secondary_hubs', [])),
        })

    consolidated_data = {}

    if all_facility_rollups:
        consolidated_data['Facility_Rollup'] = pd.concat(all_facility_rollups, ignore_index=True)

    if all_lane_summaries:
        consolidated_data['Lane_Summary'] = pd.concat(all_lane_summaries, ignore_index=True)

    if all_od_selected:
        od_combined = pd.concat(all_od_selected, ignore_index=True)
        consolidated_data['OD_Selected_Paths'] = od_combined

    if all_sort_allocations:
        consolidated_data['Sort_Allocation_Summary'] = pd.concat(all_sort_allocations, ignore_index=True)

    if all_strategy_comparisons:
        consolidated_data['Enhanced_Strategy_Comparison'] = pd.DataFrame(all_strategy_comparisons)

    # UPDATED: Use updated consolidated filename
    consolidated_path = out_dir / CONSOLIDATED_OUTPUT
    with pd.ExcelWriter(consolidated_path, engine="xlsxwriter") as writer:
        for sheet_name, df in consolidated_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  ✓ {sheet_name}: {len(df):,} rows")

    print(f"✅ Enhanced consolidated analysis written: {CONSOLIDATED_OUTPUT}")
    print(f"   Total sheets: {len(consolidated_data)}")


def main(input_path: str, output_dir: str | None):
    inp = Path(input_path)
    out_dir = Path(output_dir) if output_dir else Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading enhanced workbook from: {inp}")
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

    # Enhanced parameter defaults
    timing, costs = _add_default_parameters(timing, costs)

    compare_mode = str(run_kv.get("compare_mode", "single")).strip().lower()
    if compare_mode not in {"single", "paired"}:
        raise ValueError(f"run_settings.compare_mode must be 'single' or 'paired', got '{compare_mode}'")

    print(
        f"Running in {compare_mode} mode with enhanced sort optimization, realistic fill rate reporting, and updated file naming")

    all_results = []

    for _, s in scenarios.iterrows():
        base_id = str(s.get("pair_id", f"{int(s['year'])}_{str(s['day_type']).strip().lower()}")).strip()

        if compare_mode == "single":
            strategy = str(run_kv.get("load_strategy", "container")).strip().lower()
            scenario_id, out_path, _kpis, results_data = _run_one_strategy(
                base_id, strategy, facilities, zips, demand, inj, mb, timing, costs, cont, pkgmix, run_kv, s, out_dir
            )
            if out_path:
                print(f"✅ Enhanced results written: {out_path.name}")
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
                        "output_file": str(out_path.name),  # Just filename, not full path
                    })
                    per_base.append(rec)
                    print(f"✅ Enhanced {scenario_id} completed")

            if per_base:
                compare_df = pd.DataFrame(per_base)
                compare_filename = safe_scenario_filename(base_id, "compare")
                compare_path = out_dir / compare_filename
                write_compare_workbook(compare_path, compare_df, run_kv)
                print(f"✅ Enhanced comparison written: {compare_filename}")

                _create_monday_executive_summary(base_id, results_by_strategy, out_dir)

    if all_results:
        _create_consolidated_output(all_results, out_dir)

    print(
        f"\n🎉 Enhanced analysis complete with realistic fill rates, sort optimization, and updated file naming! Results in: {out_dir.resolve()}")


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
        {"key": "sort_points_per_destination", "value": int(timing.get("sort_points_per_destination", 2))},
        {"key": "premium_economy_dwell_threshold", "value": float(timing.get("premium_economy_dwell_threshold", 0.10))},
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

    enhanced_kpis = pd.Series({
        "total_cost": tot_cost,
        "cost_per_pkg": (tot_cost / tot_pkgs) if tot_pkgs > 0 else np.nan,
        "num_ods": len(od_selected),
        "sla_violations": int((od_selected.get("end_to_end_sla_flag", pd.Series([0])) == 1).sum()),
        "around_world_flags": int((od_selected.get("around_world_flag", pd.Series([0])) == 1).sum()),
        "pct_direct": round(100 * (od_selected["path_type"] == "direct").mean(), 2),
        "pct_1_touch": round(100 * (od_selected["path_type"] == "1_touch").mean(), 2),
        "pct_2_touch": round(100 * (od_selected["path_type"] == "2_touch").mean(), 2),
        "pct_3_touch": round(100 * (od_selected["path_type"] == "3_touch").mean(), 2),
        # UPDATED: Enhanced metrics with realistic fill rates
        "avg_container_fill_rate": od_selected.get('container_fill_rate', pd.Series([0])).mean(),  # Realistic
        "avg_truck_fill_rate": od_selected.get('truck_fill_rate', pd.Series([0])).mean(),  # Realistic
        "total_packages_dwelled": od_selected.get('packages_dwelled', pd.Series([0])).sum(),
    })

    return enhanced_kpis


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input workbook (.xlsx)")
    ap.add_argument("--output_dir", default=None, help="Output directory (default: ./outputs)")
    args = ap.parse_args()
    main(args.input, args.output_dir)