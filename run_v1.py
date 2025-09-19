# run_v1.py - COMPLETE FINAL VERSION
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
    Enhanced strategy execution with all fixes applied.
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

    # Calculate cost/time for each candidate path
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
                "cost_candidate_path": cost,
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

    # MILP path selection
    selected_basic, arc_summary = solve_arc_pooled_path_selection(
        cand_tbl[["scenario_id", "origin", "dest", "day_type", "path_type", "pkgs_day", "path_nodes", "path_str",
                  "containers_cont"]],
        facilities, mb, pkgmix, cont, costs,
    )

    # Merge candidate metrics
    merge_keys = ["scenario_id", "origin", "dest", "day_type", "path_str"]
    selected = selected_basic.merge(cand_tbl, on=merge_keys, how="left", suffixes=("", "_cand"))

    # Normalize column names
    cand_rename = {
        "distance_miles_cand": "distance_miles",
        "linehaul_hours_cand": "linehaul_hours",
        "handling_hours_cand": "handling_hours",
        "dwell_hours_cand": "dwell_hours",
        "destination_dwell_hours_cand": "destination_dwell_hours",
        "sla_days_cand": "sla_days",
        "cost_candidate_path_cand": "cost_candidate_path",
        "time_hours_cand": "time_hours",
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

    # Enhanced facility rollup with clear lane/truck metrics
    facility_rollup = build_facility_rollup(
        facilities=facilities,
        zips=zips,
        od_selected=od_out,
        path_steps_selected=detail_sel,
        direct_day=direct_day,
        arc_summary=arc_summary,
        costs=costs,
        load_strategy=strategy
    )

    lane_summary = build_lane_summary(arc_summary)
    scen_sum = _scenario_summary(run_kv, scenario_id, year, day_type, mb, timing_local, cont)
    dwell_hotspots = build_dwell_hotspots(detail_sel)
    kpis = _network_kpis(od_out)

    # Write results
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

    # Return enhanced results data
    results_data = {
        'facility_rollup': facility_rollup,
        'total_cost': kpis.get('total_cost', 0),
        'cost_per_package': kpis.get('cost_per_pkg', 0),
        'od_selected': od_out,
        'lane_summary': lane_summary,
        'arc_summary': arc_summary,
        'kpis': kpis,
        'primary_hubs': facility_rollup[facility_rollup['hub_tier'] == 'primary']['facility'].tolist(),
        'secondary_hubs': facility_rollup[facility_rollup['hub_tier'] == 'secondary']['facility'].tolist(),
    }

    return scenario_id, out_path, kpis, results_data


def _create_monday_executive_summary(base_id: str, results_by_strategy: dict, out_dir: Path):
    """
    Create executive summary with clear answers to Monday's 4 key questions.
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

    # Monday Question 2: Hourly throughput by hub
    container_rollup = container_results['facility_rollup']
    fluid_rollup = fluid_results['facility_rollup']

    # Filter to hub/hybrid facilities only
    container_hubs = container_rollup[container_rollup['type'].isin(['hub', 'hybrid'])].copy()
    fluid_hubs = fluid_rollup[fluid_rollup['type'].isin(['hub', 'hybrid'])].copy()

    # Calculate hourly throughput (assuming 16-hour operation)
    operating_hours = 16
    peak_hour_factor = 1.3

    container_hubs['hourly_throughput_avg'] = container_hubs['origin_pkgs_day'] / operating_hours
    container_hubs['hourly_throughput_peak'] = container_hubs['hourly_throughput_avg'] * peak_hour_factor

    fluid_hubs['hourly_throughput_avg'] = fluid_hubs['origin_pkgs_day'] / operating_hours
    fluid_hubs['hourly_throughput_peak'] = fluid_hubs['hourly_throughput_avg'] * peak_hour_factor

    # Monday Question 3: Inbound/outbound trailers per facility
    # This is already calculated in the enhanced facility_rollup as outbound_trucks_total and inbound_trucks_total

    # Monday Question 4: Zone mix and paths by origin facility
    container_od = container_results['od_selected']
    path_type_summary = container_od.groupby('origin')['path_type'].value_counts().unstack(fill_value=0)

    # Create executive summary sheets
    summary_data = {}

    # Sheet 1: Containerization Strategy Recommendation
    strategy_comparison = pd.DataFrame([
        {
            'strategy': 'container',
            'total_daily_cost': container_cost,
            'primary_hubs': len(container_results['primary_hubs']),
            'secondary_hubs': len(container_results['secondary_hubs']),
            'avg_packages_per_truck': container_results['lane_summary']['packages_per_truck'].mean(),
            'total_active_lanes': len(container_results['lane_summary']),
        },
        {
            'strategy': 'fluid',
            'total_daily_cost': fluid_cost,
            'primary_hubs': len(fluid_results['primary_hubs']),
            'secondary_hubs': len(fluid_results['secondary_hubs']),
            'avg_packages_per_truck': fluid_results['lane_summary']['packages_per_truck'].mean(),
            'total_active_lanes': len(fluid_results['lane_summary']),
        }
    ])

    summary_data['Strategy_Comparison'] = strategy_comparison

    # Sheet 2: Hub Hourly Throughput Requirements
    hub_throughput = container_hubs[['facility', 'type', 'hub_tier', 'hourly_throughput_peak']].copy()
    hub_throughput['fluid_hourly_throughput_peak'] = hub_throughput['facility'].map(
        fluid_hubs.set_index('facility')['hourly_throughput_peak']
    )
    hub_throughput = hub_throughput.sort_values('hourly_throughput_peak', ascending=False)
    summary_data['Hub_Hourly_Throughput'] = hub_throughput

    # Sheet 3: Facility Truck Requirements
    truck_requirements = container_hubs[[
        'facility', 'type', 'hub_tier', 'outbound_lane_count', 'outbound_trucks_total',
        'inbound_lane_count', 'inbound_trucks_total', 'total_trucks_per_day'
    ]].copy()
    summary_data['Facility_Truck_Requirements'] = truck_requirements

    # Sheet 4: Path Type Analysis by Origin
    if not path_type_summary.empty:
        path_analysis = path_type_summary.copy()
        path_analysis['total_paths'] = path_analysis.sum(axis=1)
        for col in ['direct', '1_touch', '2_touch', '3_touch']:
            if col in path_analysis.columns:
                path_analysis[f'pct_{col}'] = (path_analysis[col] / path_analysis['total_paths'] * 100).round(1)
        summary_data['Path_Type_Analysis'] = path_analysis

    # Sheet 5: Monday Key Answers Summary
    monday_answers = pd.DataFrame([
        {
            'question': '1. Optimal Containerization Strategy',
            'answer': f'{optimal_strategy.upper()} strategy',
            'detail': f'Cost difference: ${cost_difference:,.0f}/day in favor of {optimal_strategy}',
            'packages_per_truck': f"{strategy_comparison[strategy_comparison['strategy'] == optimal_strategy]['avg_packages_per_truck'].iloc[0]:.0f} packages/truck average"
        },
        {
            'question': '2. Hourly Throughput by Hub',
            'answer': f'Peak hourly requirements calculated for {len(hub_throughput)} hubs',
            'detail': f"Range: {hub_throughput['hourly_throughput_peak'].min():.0f} - {hub_throughput['hourly_throughput_peak'].max():.0f} packages/hour",
            'packages_per_truck': f"Top hub: {hub_throughput.iloc[0]['facility']} at {hub_throughput.iloc[0]['hourly_throughput_peak']:.0f} pkgs/hr"
        },
        {
            'question': '3. Inbound/Outbound Trailers per Facility',
            'answer': f'Truck requirements calculated for all {len(truck_requirements)} facilities',
            'detail': f"Total network: {truck_requirements['total_trucks_per_day'].sum():.0f} trucks/day",
            'packages_per_truck': f"Avg outbound: {truck_requirements['outbound_trucks_total'].mean():.1f} trucks/facility"
        },
        {
            'question': '4. Zone Mix and Touch Patterns',
            'answer': 'Path analysis by origin facility completed',
            'detail': f"Network average: {container_od['path_type'].value_counts(normalize=True).round(3).to_dict()}",
            'packages_per_truck': f"Total origin facilities: {len(path_type_summary) if not path_type_summary.empty else 0}"
        }
    ])

    summary_data['Monday_Key_Answers'] = monday_answers

    # Write executive summary workbook
    exec_summary_path = out_dir / EXECUTIVE_SUMMARY_TEMPLATE.format(base_id=base_id)
    with pd.ExcelWriter(exec_summary_path, engine="xlsxwriter") as writer:
        for sheet_name, df in summary_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"[{base_id}] Executive summary written: {exec_summary_path}")

    # Console output for immediate review
    print(f"\n=== MONDAY EXECUTIVE SUMMARY ===")
    print(f"Optimal Strategy: {optimal_strategy.upper()}")
    print(f"Cost Advantage: ${cost_difference:,.0f}/day")
    print(
        f"Hub Count: {len(container_results['primary_hubs'])} primary, {len(container_results['secondary_hubs'])} secondary")
    print(
        f"Top Hub Throughput: {hub_throughput.iloc[0]['facility']} at {hub_throughput.iloc[0]['hourly_throughput_peak']:.0f} pkgs/hr")
    print(f"Network Truck Requirement: {truck_requirements['total_trucks_per_day'].sum():.0f} trucks/day")


def main(input_path: str, output_dir: str | None):
    inp = Path(input_path)
    out_dir = Path(output_dir) if output_dir else Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading workbook from: {inp}")
    dfs = load_workbook(inp)
    validate_inputs(dfs)

    # Load all required data
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

    # Add default parameters if missing
    if "hours_per_touch" not in timing:
        timing["hours_per_touch"] = 6.0
        print("Added default hours_per_touch: 6.0")

    if "sla_penalty_per_touch_per_pkg" not in costs:
        costs["sla_penalty_per_touch_per_pkg"] = 0.25
        print("Added default sla_penalty_per_touch_per_pkg: 0.25")

    compare_mode = str(run_kv.get("compare_mode", "single")).strip().lower()
    if compare_mode not in {"single", "paired"}:
        raise ValueError(f"run_settings.compare_mode must be 'single' or 'paired', got '{compare_mode}'")

    print(f"Running in {compare_mode} mode")

    for _, s in scenarios.iterrows():
        base_id = str(s.get("pair_id", f"{int(s['year'])}_{str(s['day_type']).strip().lower()}")).strip()

        if compare_mode == "single":
            strategy = str(run_kv.get("load_strategy", "container")).strip().lower()
            scenario_id, out_path, _kpis, _results_data = _run_one_strategy(
                base_id, strategy, facilities, zips, demand, inj, mb, timing, costs, cont, pkgmix, run_kv, s, out_dir
            )
            if out_path:
                print(f"✅ Wrote {out_path}")
        else:
            # Paired mode: run both strategies
            results_by_strategy = {}
            per_base = []

            for strategy in ["container", "fluid"]:
                scenario_id, out_path, kpis, results_data = _run_one_strategy(
                    base_id, strategy, facilities, zips, demand, inj, mb, timing, costs, cont, pkgmix, run_kv, s,
                    out_dir
                )
                if out_path is not None and not kpis.empty:
                    results_by_strategy[strategy] = results_data

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
                # Write standard comparison
                compare_df = pd.DataFrame(per_base)
                compare_path = out_dir / COMPARE_FILE_TEMPLATE.format(base_id=base_id)
                write_compare_workbook(compare_path, compare_df, run_kv)
                print(f"✅ Comparison written: {compare_path}")

                # Create Monday executive summary
                _create_monday_executive_summary(base_id, results_by_strategy, out_dir)

    print(f"\n🎉 All analysis complete! Results in: {out_dir.resolve()}")


def _scenario_summary(run_kv, scenario_id, year, day_type, mb, timing, cont):
    g = cont[cont["container_type"].str.lower() == "gaylord"].iloc[0]
    rows = [
        {"key": "scenario_id", "value": scenario_id},
        {"key": "demand_year", "value": year},
        {"key": "day_type", "value": day_type},
        {"key": "load_strategy", "value": str(timing.get("load_strategy", "container"))},
        {"key": "hours_per_touch", "value": float(timing.get("hours_per_touch", 6.0))},
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
    tot_cost = od_selected.get("total_cost", od_selected["cost_candidate_path"]).sum()
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