import pandas as pd

def write_workbook(path, scen_sum, od_out, path_detail, dwell_hotspots, facility_rollup, arc_summary, kpis):
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        scen_sum.to_excel(xw, sheet_name="scenario_summary", index=False)
        od_out.to_excel(xw, sheet_name="od_selected_paths", index=False)
        path_detail.to_excel(xw, sheet_name="path_steps_selected", index=False)
        dwell_hotspots.to_excel(xw, sheet_name="dwell_hotspots", index=False)
        facility_rollup.to_excel(xw, sheet_name="facility_rollup", index=False)
        arc_summary.to_excel(xw, sheet_name="arc_summary", index=False)
        kpis.to_frame("value").to_excel(xw, sheet_name="kpis")

def write_compare_workbook(path, compare_df: pd.DataFrame, run_kv: dict):
    """
    compare_df columns expected (built in runner):
      base_id, scenario_id, strategy, output_file, plus all KPI fields:
        total_cost, cost_per_pkg, num_ods, sla_violations, around_world_flags,
        pct_direct, pct_1_touch, pct_2_touch
    """
    # tidy order
    front = ["base_id", "strategy", "scenario_id", "output_file"]
    kpi_cols = [
        "total_cost", "cost_per_pkg", "num_ods",
        "sla_violations", "around_world_flags",
        "pct_direct", "pct_1_touch", "pct_2_touch",
    ]
    cols = front + [c for c in kpi_cols if c in compare_df.columns]
    df = compare_df[cols].copy()

    # a wide view: one row per base_id, columns per strategy for the headline KPIs
    wide = df.pivot(index="base_id", columns="strategy", values="cost_per_pkg")
    wide = wide.rename_axis(None, axis=1).reset_index()

    # settings snapshot for traceability
    settings = pd.DataFrame(
        [{"key": k, "value": v} for k, v in run_kv.items()]
    )

    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        df.to_excel(xw, sheet_name="kpi_compare_long", index=False)
        wide.to_excel(xw, sheet_name="kpi_compare_wide", index=False)
        settings.to_excel(xw, sheet_name="run_settings", index=False)
