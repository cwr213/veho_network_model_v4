# run_v1.py
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
)
from veho_net.write_outputs import write_workbook, write_compare_workbook
from veho_net.config import OUTPUT_DIR


# ------------------------------------------------------------
# File naming control (edit these to change output filenames)
OUTPUT_FILE_TEMPLATE = "{scenario_id}_results_v1.xlsx"
COMPARE_FILE_TEMPLATE = "{base_id}_compare.xlsx"
# ------------------------------------------------------------


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
    Executes a single strategy (container or fluid) for one scenarios row.
    Returns:
        scenario_id, out_path, kpis (pd.Series)
    """
    # containers per truck (for reporting)
    containers_per_truck = int(
        cont[cont["container_type"].str.lower() == "gaylord"].iloc[0]["containers_per_truck"]
    )

    year = int(scenario_row["year"]) if "year" in scenario_row else int(scenario_row["demand_year"])
    day_type = str(scenario_row["day_type"]).strip().lower()

    # Scenario ID: base_id + strategy suffix
    scenario_id = f"{base_id}_{strategy}"

    # Timing: inject strategy for time/cost functions
    timing_local = dict(timing)
    timing_local["load_strategy"] = strategy

    # Build OD + DIRECT (day-type MM shares)
    year_demand = demand.query("year == @year").copy()
    od, dir_fac, _dest_pop = build_od_and_direct(facilities, zips, year_demand, inj)

    od_day_col = "pkgs_offpeak_day" if day_type == "offpeak" else "pkgs_peak_day"
    direct_day_col = "dir_pkgs_offpeak_day" if day_type == "offpeak" else "dir_pkgs_peak_day"
    od = od[od[od_day_col] > 0].copy()

    # Candidate paths (shortest direct/1-touch/2-touch); pass bands & guard factor
    around_factor = float(run_kv.get("path_around_the_world_factor", 2.0))
    cands = candidate_paths(od, facilities, mb, around_factor=around_factor)

    # Candidate timing/cost + steps
    cand_rows, detail_rows, direct_dist = [], [], {}
    for _, r in cands.iterrows():
        sub = od[(od["origin"] == r["origin"]) & (od["dest"] == r["dest"])]
        if sub.empty:
            continue
        pkgs_day = float(sub.iloc[0][od_day_col])

        path_str = r["path_str"]

        cost, hours, sums, steps = path_cost_and_time(
            r, facilities, mb, timing_local, costs, pkgmix, cont, pkgs_day
        )
        conts = containers_for_pkgs_day(pkgs_day, pkgmix, cont)

        cand_rows.append(
            {
                "scenario_id": scenario_id,
                "origin": r["origin"],
                "dest": r["dest"],
                "year": year,
                "day_type": day_type,
                "path_type": r["path_type"],
                "path_nodes": r.get("path_nodes", None),
                "path_str": path_str,
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
            }
        )

        for st in steps:
            step = {
                "scenario_id": scenario_id,
                "origin": r["origin"],
                "dest": r["dest"],
                "day_type": day_type,
                "path_type": r["path_type"],
                "path_str": path_str,
                "pkgs_day": pkgs_day,
                "containers_cont": conts,
            }
            step.update(st)
            detail_rows.append(step)

        if r["path_type"] == "direct":
            key = (scenario_id, r["origin"], r["dest"], day_type)
            direct_dist[key] = sums["distance_miles_total"]

    cand_tbl = pd.DataFrame(cand_rows)
    if cand_tbl.empty:
        print(f"[{scenario_id}] No OD volume for {day_type}. Skipping.")
        return scenario_id, None, pd.Series(dtype=float)

    # MILP path selection with arc pooling
    selected_basic, arc_summary = solve_arc_pooled_path_selection(
        cand_tbl[
            [
                "scenario_id",
                "origin",
                "dest",
                "day_type",
                "path_type",
                "pkgs_day",
                "path_nodes",
                "path_str",
                "containers_cont",
            ]
        ],
        facilities,
        mb,
        pkgmix,
        cont,
        costs,
    )

    # Merge back timing/candidate metrics for chosen rows (avoid list keys)
    merge_keys = ["scenario_id", "origin", "dest", "day_type", "path_str"]
    selected = selected_basic.merge(
        cand_tbl.drop(columns=["year"]),
        on=merge_keys,
        how="left",
        suffixes=("", "_cand"),
    )

    # ---- Normalize candidate metric column names (avoid *_cand surprises) ----
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

    # Flags (around-the-world, SLA target days from run_settings)
    dd = pd.Series(direct_dist, name="distance_miles")
    dd.index = pd.MultiIndex.from_tuples(dd.index, names=["scenario_id", "origin", "dest", "day_type"])
    flags = {
        "path_around_the_world_factor": float(run_kv.get("path_around_the_world_factor", 2.0)),
        "sla_target_days": int(run_kv.get("sla_target_days", 3)),
    }
    od_out = build_od_selected_outputs(selected, dd, flags)
    od_out["scenario_id"] = scenario_id
    od_out["containers_per_truck"] = containers_per_truck

    # Zones (add if not present)
    od_out = add_zone(od_out, facilities)

    # Keep only selected steps for detail
    detail_all = pd.DataFrame(detail_rows)
    key_cols = ["scenario_id", "origin", "dest", "day_type", "path_type", "path_str"]
    detail_sel = detail_all.merge(od_out[key_cols].drop_duplicates(), on=key_cols, how="inner")

    # Facility-level direct day for this year/day type (origin lens needs direct at O==D; direct_day is dest-based)
    direct_day = (
        dir_fac[dir_fac["year"] == year][["dest", direct_day_col]]
        .groupby("dest", as_index=False)[direct_day_col]
        .sum()
        .rename(columns={direct_day_col: "dir_pkgs_day"})
    )

    # Facility rollup (origin-based CPPs + zones)
    facility_rollup = build_facility_rollup(
        facilities=facilities,
        zips=zips,
        od_selected=od_out,
        path_steps_selected=detail_sel,
        direct_day=direct_day,
        arc_summary=arc_summary,
        costs=costs,
        load_strategy=strategy,
    )

    scen_sum = _scenario_summary(run_kv, scenario_id, year, day_type, mb, timing_local, cont)
    dwell_hotspots = build_dwell_hotspots(detail_sel)
    kpis = _network_kpis(od_out)

    # Write workbook
    out_path = out_dir / OUTPUT_FILE_TEMPLATE.format(scenario_id=scenario_id)
    write_workbook(
        out_path,
        scen_sum,
        od_out,
        detail_sel,
        dwell_hotspots,
        facility_rollup,
        arc_summary,
        kpis,
    )

    # Console reconciliation
    annual_total = float(demand.query("year == @year")["annual_pkgs"].sum())
    daily_off = float(demand.query("year == @year")["offpeak_pct_of_annual"].iloc[0])
    daily_peak = float(demand.query("year == @year")["peak_pct_of_annual"].iloc[0])
    expected_daily_total = annual_total * (daily_peak if day_type == "peak" else daily_off)
    actual_daily_total = direct_day["dir_pkgs_day"].sum() + od_out["pkgs_day"].sum()
    print(
        f"[{scenario_id}] expected daily={expected_daily_total:,.0f}; actual (direct+MMdest)={actual_daily_total:,.0f}"
    )

    return scenario_id, out_path, kpis


def main(input_path: str, output_dir: str | None):
    inp = Path(input_path)
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

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

    compare_mode = str(run_kv.get("compare_mode", "single")).strip().lower()
    valid_modes = {"single", "paired"}
    if compare_mode not in valid_modes:
        raise ValueError(f"run_settings.compare_mode must be one of {valid_modes}, got '{compare_mode}'")

    for _, s in scenarios.iterrows():
        # base id: prefer pair_id if present, else year_dayType
        if "pair_id" in s and pd.notna(s["pair_id"]) and str(s["pair_id"]).strip():
            base_id = str(s["pair_id"]).strip()
        else:
            base_id = f"{int(s['year'])}_{str(s['day_type']).strip().lower()}"

        if compare_mode == "single":
            strategy = str(run_kv.get("load_strategy", "container")).strip().lower()
            scenario_id, out_path, _kpis = _run_one_strategy(
                base_id, strategy, facilities, zips, demand, inj, mb, timing, costs, cont, pkgmix, run_kv, s, out_dir
            )
            print(f"Wrote {out_path}")
        else:
            per_base = []
            for strategy in ["container", "fluid"]:
                scenario_id, out_path, kpis = _run_one_strategy(
                    base_id, strategy, facilities, zips, demand, inj, mb, timing, costs, cont, pkgmix, run_kv, s, out_dir
                )
                if out_path is not None and not kpis.empty:
                    rec = kpis.to_dict()
                    rec.update({
                        "base_id": base_id,
                        "scenario_id": scenario_id,
                        "strategy": strategy,
                        "output_file": str(out_path),
                    })
                    per_base.append(rec)

            if per_base:
                compare_df = pd.DataFrame(per_base)
                compare_path = out_dir / COMPARE_FILE_TEMPLATE.format(base_id=base_id)
                write_compare_workbook(compare_path, compare_df, run_kv)
                print(f"[{base_id}] comparison written: {compare_path}")

    print(f"Done. Wrote outputs to {out_dir.resolve()}")


def _scenario_summary(run_kv, scenario_id, year, day_type, mb, timing, cont):
    g = cont[cont["container_type"].str.lower() == "gaylord"].iloc[0]
    rows = [
        {"key": "scenario_id", "value": scenario_id},
        {"key": "demand_year", "value": year},
        {"key": "day_type", "value": day_type},
        {"key": "load_strategy", "value": str(timing.get("load_strategy", "container"))},
        {"key": "usable_cube_cuft", "value": float(g["usable_cube_cuft"])},
        {"key": "pack_utilization_container", "value": float(g["pack_utilization_container"])},
        {"key": "containers_per_truck", "value": int(g["containers_per_truck"])},
        {"key": "trailer_air_cube_cuft", "value": float(g.get("trailer_air_cube_cuft", 4060.0))},
        {"key": "pack_utilization_fluid", "value": float(g.get("pack_utilization_fluid", 0.85))},
        {"key": "cpt_hours_local", "value": str(timing.get("cpt_hours_local"))},
        {"key": "delivery_day_cutoff_local", "value": str(timing.get("delivery_day_cutoff_local", "04:00"))},
        {"key": "load_hours", "value": float(timing.get("load_hours", 1.0))},
        {"key": "unload_hours", "value": float(timing.get("unload_hours", 1.0))},
        {"key": "sort_hours_per_touch", "value": float(timing.get("sort_hours_per_touch", 2.0))},
        {"key": "crossdock_hours_per_touch", "value": float(timing.get("crossdock_hours_per_touch", 1.0))},
        {"key": "departure_cutoff_hours_per_move", "value": float(timing.get("departure_cutoff_hours_per_move", 1.0))},
        {"key": "sla_target_days", "value": int(run_kv.get("sla_target_days", 3))},
        {"key": "path_around_the_world_factor", "value": float(run_kv.get("path_around_the_world_factor", 2.0))},
        {"key": "mileage_bands_json", "value": mb.to_json(orient="records")},
    ]
    return pd.DataFrame(rows)


def _network_kpis(od_selected: pd.DataFrame) -> pd.Series:
    tot_cost = od_selected.get("total_cost", od_selected["cost_candidate_path"]).sum()
    tot_pkgs = od_selected["pkgs_day"].sum()
    return pd.Series(
        {
            "total_cost": tot_cost,
            "cost_per_pkg": (tot_cost / tot_pkgs) if tot_pkgs > 0 else np.nan,
            "num_ods": len(od_selected),
            "sla_violations": int((od_selected["end_to_end_sla_flag"] == 1).sum()),
            "around_world_flags": int((od_selected["around_world_flag"] == 1).sum()),
            "pct_direct": round(100 * (od_selected["path_type"] == "direct").mean(), 2),
            "pct_1_touch": round(100 * (od_selected["path_type"] == "1_touch").mean(), 2),
            "pct_2_touch": round(100 * (od_selected["path_type"] == "2_touch").mean(), 2),
        }
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input workbook")
    ap.add_argument("--output_dir", default=None, help="Output directory, default ./outputs")
    args = ap.parse_args()
    main(args.input, args.output_dir)
