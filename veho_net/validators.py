# veho_net/validators.py

import pandas as pd

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _fail(msg: str, df: pd.DataFrame | None = None):
    if df is not None:
        raise ValueError(f"{msg}\nFound columns: {list(df.columns)}")
    raise ValueError(msg)

# ---------------- container_params ----------------

def _check_container_params(df_raw: pd.DataFrame):
    df = _norm_cols(df_raw)
    required = {
        "container_type",
        "usable_cube_cuft",
        "pack_utilization_container",
        "containers_per_truck",
        "trailer_air_cube_cuft",
        "pack_utilization_fluid",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        _fail(f"container_params missing required columns: {missing}", df)
    if df.empty:
        _fail("container_params has no rows", df)
    if not (df["container_type"].str.lower() == "gaylord").any():
        _fail("container_params must include a row where container_type == 'gaylord'", df)

# ---------------- facilities / zips ----------------

def _check_facilities(df_raw: pd.DataFrame):
    df = _norm_cols(df_raw)
    required = {
        "facility_name", "type", "market", "region",
        "lat", "lon", "timezone", "parent_hub_name", "is_injection_node"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        _fail(f"facilities missing required columns: {missing}", df)
    dups = df["facility_name"][df["facility_name"].duplicated()].unique()
    if len(dups) > 0:
        _fail(f"facilities has duplicate facility_name values: {list(dups)}", df)

def _check_zips(df_raw: pd.DataFrame):
    df = _norm_cols(df_raw)
    required = {"zip", "facility_name_assigned", "market", "population"}
    missing = sorted(required - set(df.columns))
    if missing:
        _fail(f"zips missing required columns: {missing}", df)
    dups = df["zip"][df["zip"].duplicated()].unique()
    if len(dups) > 0:
        _fail(f"zips has duplicate ZIP codes: {list(dups)}", df)

# ---------------- demand (STRICT) ----------------

def _check_demand(df_raw: pd.DataFrame):
    df = _norm_cols(df_raw)
    required = {
        "year",
        "annual_pkgs",
        "offpeak_pct_of_annual",
        "peak_pct_of_annual",
        "middle_mile_share_offpeak",
        "middle_mile_share_peak",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        _fail(f"demand missing required columns: {missing}", df)
    # sanity: values should be within [0,1] for % shares
    for col in ["offpeak_pct_of_annual", "peak_pct_of_annual",
                "middle_mile_share_offpeak", "middle_mile_share_peak"]:
        bad = df[~df[col].between(0, 1, inclusive="both")]
        if not bad.empty:
            raise ValueError(f"demand: column '{col}' has values outside [0,1]. Offenders (first 5 rows):\n{bad.head(5)}")

def _check_injection_distribution(df: pd.DataFrame):
    required = {"facility_name", "absolute_share"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"injection_distribution missing required columns: {sorted(missing)}")
    # Enabled flag optional, but if present must be 0/1
    if "enabled_for_injection" in df.columns:
        bad = ~pd.to_numeric(df["enabled_for_injection"], errors="coerce").isin([0, 1])
        if bad.any():
            raise ValueError("injection_distribution.enabled_for_injection must be 0/1 when present")
    # absolute_share must sum > 0 across enabled rows
    tmp = df.copy()
    if "enabled_for_injection" in tmp.columns:
        tmp = tmp[tmp["enabled_for_injection"].astype(int) == 1]
    w = pd.to_numeric(tmp["absolute_share"], errors="coerce").fillna(0.0)
    if float(w.sum()) <= 0:
        raise ValueError("injection_distribution.absolute_share must sum > 0 over enabled rows")

def _check_mileage_bands(df_raw: pd.DataFrame):
    df = _norm_cols(df_raw)
    required = {"mileage_band_min", "mileage_band_max", "fixed_cost_per_truck", "variable_cost_per_mile",
                "circuity_factor", "mph"}
    missing = sorted(required - set(df.columns))
    if missing:
        _fail(f"mileage_bands missing required columns: {missing}", df)

def _check_timing_params(df_raw: pd.DataFrame):
    df = _norm_cols(df_raw)
    required_keys = {"cpt_hours_local", "delivery_day_cutoff_local", "load_hours", "unload_hours",
                     "sort_hours_per_touch", "crossdock_hours_per_touch", "departure_cutoff_hours_per_move"}
    missing = sorted(required_keys - set(df["key"]))
    if missing:
        raise ValueError(f"timing_params missing required keys: {missing}")

def _check_cost_params(df_raw: pd.DataFrame):
    df = _norm_cols(df_raw)
    required_keys = {"sort_cost_per_pkg", "crossdock_touch_cost_per_pkg", "last_mile_cpp"}
    missing = sorted(required_keys - set(df["key"]))
    if missing:
        raise ValueError(f"cost_params missing required keys: {missing}")

def _check_package_mix(df_raw: pd.DataFrame):
    df = _norm_cols(df_raw)
    required = {"package_type", "share_of_pkgs", "avg_cube_cuft"}
    missing = sorted(required - set(df.columns))
    if missing:
        _fail(f"package_mix missing required columns: {missing}", df)
    s = float(df["share_of_pkgs"].sum())
    if abs(s - 1.0) > 1e-6:
        _fail(f"package_mix share_of_pkgs must sum to 1.0 (found {s})", df)

def _check_run_settings(df_raw: pd.DataFrame):
    df = _norm_cols(df_raw)
    required_keys = {"load_strategy", "sla_target_days", "path_around_the_world_factor"}
    missing = sorted(required_keys - set(df["key"]))
    if missing:
        raise ValueError(f"run_settings missing required keys: {missing}")

def _check_scenarios(df_raw: pd.DataFrame):
    df = _norm_cols(df_raw)
    required = {"year", "day_type"}  # pair_id optional in this runner
    missing = sorted(required - set(df.columns))
    if missing:
        _fail(f"scenarios missing required columns: {missing}", df)

def validate_inputs(dfs: dict):
    _check_container_params(dfs["container_params"])
    _check_facilities(dfs["facilities"])
    _check_zips(dfs["zips"])
    _check_demand(dfs["demand"])
    _check_injection_distribution(dfs["injection_distribution"])
    _check_mileage_bands(dfs["mileage_bands"])
    _check_timing_params(dfs["timing_params"])
    _check_cost_params(dfs["cost_params"])
    _check_package_mix(dfs["package_mix"])
    _check_run_settings(dfs["run_settings"])
    _check_scenarios(dfs["scenarios"])
