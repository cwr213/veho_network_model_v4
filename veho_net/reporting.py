# veho_net/reporting.py
import pandas as pd
import numpy as np
from .geo import haversine_miles

# ---------------- Zones ----------------

def add_zone(od_selected: pd.DataFrame, facilities: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'zone' (one of {0,'1-2','3','4','5','6','7','8'}) using RAW haversine between O and D.
    Zone 0 means O==D (direct injection case).
    """
    if "zone" in od_selected.columns and od_selected["zone"].notna().any():
        return od_selected

    fac = facilities.set_index("facility_name")

    def raw_m(o, d):
        return haversine_miles(float(fac.at[o, "lat"]), float(fac.at[o, "lon"]),
                               float(fac.at[d, "lat"]), float(fac.at[d, "lon"]))

    def zlabel(miles: float, o: str, d: str) -> object:
        if o == d:
            return 0
        m = float(miles)
        if m <= 150: return "1-2"
        if m <= 300: return "3"
        if m <= 600: return "4"
        if m <= 1000: return "5"
        if m <= 1400: return "6"
        if m <= 1800: return "7"
        return "8"

    df = od_selected.copy()
    if "distance_direct_raw_miles" not in df.columns:
        df["distance_direct_raw_miles"] = df.apply(lambda r: raw_m(r["origin"], r["dest"]), axis=1)

    df["zone"] = df.apply(lambda r: zlabel(r["distance_direct_raw_miles"], r["origin"], r["dest"]), axis=1)
    return df


# ---------------- Dwell hotspots ----------------

def build_dwell_hotspots(path_steps_selected: pd.DataFrame) -> pd.DataFrame:
    """
    Sum dwell_hours by the 'from' facility of each leg.
    Robust to different column namings in steps: tries a few variants.
    """
    if path_steps_selected is None or path_steps_selected.empty:
        return pd.DataFrame(columns=["facility", "total_dwell_hours"])

    df = path_steps_selected.copy()

    # Find the 'from' facility column
    from_candidates = ["facility_from", "from_facility", "from", "node_from", "origin_facility"]
    fcol = next((c for c in from_candidates if c in df.columns), None)
    if fcol is None:
        # If we have a single 'facility' column for steps, use that
        if "facility" in df.columns:
            fcol = "facility"
        else:
            # Nothing to group on — return empty but consistent
            return pd.DataFrame(columns=["facility", "total_dwell_hours"])

    # Find a dwell column
    dwell_candidates = ["dwell_hours", "dwell_hours_total", "dwell"]
    dcol = next((c for c in dwell_candidates if c in df.columns), None)
    if dcol is None:
        # If dwell isn't present, infer zero dwell
        df["dwell_hours"] = 0.0
        dcol = "dwell_hours"

    out = (
        df.groupby(fcol, as_index=False)[dcol]
          .sum()
          .rename(columns={fcol: "facility", dcol: "total_dwell_hours"})
          .sort_values("total_dwell_hours", ascending=False)
    )
    return out


# ---------------- OD selected outputs flags ----------------

def build_od_selected_outputs(selected: pd.DataFrame, direct_dist_series: pd.Series, flags: dict) -> pd.DataFrame:
    df = selected.copy()
    factor = float(flags.get("path_around_the_world_factor", 2.0))
    sla_target_days = int(flags.get("sla_target_days", 3))

    # Prepare direct distance distinctly to avoid clobbering path columns
    direct_df = (
        direct_dist_series
        .reset_index()
        .rename(columns={"distance_miles": "distance_miles_direct"})
    )

    # Merge direct distance
    df = df.merge(
        direct_df,
        on=["scenario_id", "origin", "dest", "day_type"],
        how="left",
    )

    # Pick which path-distance column to use
    if "distance_miles" in df.columns:
        dist_col = "distance_miles"
    elif "distance_miles_cand" in df.columns:
        dist_col = "distance_miles_cand"
    else:
        raise KeyError(
            "Selected paths are missing a distance column. Expected 'distance_miles' or 'distance_miles_cand'."
        )

    # Around-the-world flag
    df["around_world_flag"] = (
        (df[dist_col] > factor * df["distance_miles_direct"]).astype(int)
    ).fillna(0)

    # SLA flag
    if "sla_days" not in df.columns:
        raise KeyError("Selected paths are missing 'sla_days' needed for SLA flagging.")
    df["end_to_end_sla_flag"] = (df["sla_days"] > sla_target_days).astype(int)

    # Label the candidate family that won
    df["shortest_family"] = df["path_type"].replace(
        {"direct": "direct", "1_touch": "shortest_1", "2_touch": "shortest_2"}
    )

    return df


# ---------------- Facility rollup (origin lens with CPPs & zones) ----------------

def _per_touch_cost(load_strategy: str, costs: dict) -> float:
    if load_strategy == "container":
        return float(costs["crossdock_touch_cost_per_pkg"])
    else:
        return float(costs["sort_cost_per_pkg"])

def _touches_for_path(path_type: str) -> int:
    # direct: 0 intermediates; 1_touch: 1; 2_touch: 2
    return 0 if path_type == "direct" else (1 if path_type == "1_touch" else 2)

def _zones_wide_origin(od_selected: pd.DataFrame) -> pd.DataFrame:
    zcols = [
        "zone_0_pkgs","zone_1-2_pkgs","zone_3_pkgs","zone_4_pkgs",
        "zone_5_pkgs","zone_6_pkgs","zone_7_pkgs","zone_8_pkgs"
    ]
    out = od_selected.copy()
    def zcol(z):
        if z == 0:
            return "zone_0_pkgs"
        elif z == "1-2":
            return "zone_1-2_pkgs"
        else:
            return f"zone_{z}_pkgs"
    out["zone_col"] = out["zone"].apply(zcol)
    ztab = (out.groupby(["origin","zone_col"])["pkgs_day"].sum()
              .unstack(fill_value=0.0)
              .reindex(columns=zcols, fill_value=0.0)
              .reset_index()
              .rename(columns={"origin":"facility"}))
    return ztab

def build_facility_rollup(facilities: pd.DataFrame,
                          zips: pd.DataFrame,
                          od_selected: pd.DataFrame,
                          path_steps_selected: pd.DataFrame,
                          direct_day: pd.DataFrame,
                          arc_summary: pd.DataFrame,
                          costs: dict,
                          load_strategy: str) -> pd.DataFrame:
    """
    Origin-based lens:
      origin_pkgs_day = direct_injection (O==D) + origin middle-mile (O!=D)
      CPPs per origin package:
        - injection_sort_cpp = sort_cost_per_pkg
        - mm_processing_cpp  = touches * (crossdock or sort) on intermediate nodes only (exclude origin and last-mile)
        - mm_linehaul_cpp    = (total path cost per pkg - mm_processing_cpp_od) averaged by pkgs_day over ODs that originate at O
        - last_mile_cpp      = required cost_params['last_mile_cpp'] (constant per origin pkg)
        - total_variable_cpp = sum of above
      Zones: origin-side zone buckets (using raw haversine)
    """
    fac_meta = facilities[["facility_name","market","region"]].rename(columns={"facility_name":"facility"})

    # Origin middle-mile (O!=D)
    origin_mm = od_selected.groupby("origin", as_index=False)["pkgs_day"].sum().rename(columns={"origin":"facility","pkgs_day":"origin_mm_pkgs"})

    # Direct injection (O==D): direct_day is dest-based; for origin lens, O==D appears as facility == dest
    direct_origin = direct_day.rename(columns={"dest":"facility","dir_pkgs_day":"direct_injection_pkgs"})

    # Merge volumes
    vols = fac_meta.merge(origin_mm, on="facility", how="left").merge(direct_origin, on="facility", how="left")
    for c in ["origin_mm_pkgs","direct_injection_pkgs"]:
        if c not in vols.columns:
            vols[c] = 0.0
    vols[["origin_mm_pkgs","direct_injection_pkgs"]] = vols[["origin_mm_pkgs","direct_injection_pkgs"]].fillna(0.0)
    vols["origin_pkgs_day"] = vols["origin_mm_pkgs"] + vols["direct_injection_pkgs"]

    # Zones (origin)
    zwide = _zones_wide_origin(od_selected)
    vols = vols.merge(zwide, on="facility", how="left")
    zcols = ["zone_0_pkgs","zone_1-2_pkgs","zone_3_pkgs","zone_4_pkgs","zone_5_pkgs","zone_6_pkgs","zone_7_pkgs","zone_8_pkgs"]
    for c in zcols:
        if c not in vols.columns:
            vols[c] = 0.0
    vols[zcols] = vols[zcols].fillna(0.0)

    # Costs
    sort_cost = float(costs["sort_cost_per_pkg"])
    lm_cpp = float(costs["last_mile_cpp"])  # required by validator
    per_touch = _per_touch_cost(load_strategy, costs)

    # Per-OD processing & linehaul CPP (weighted averages by pkgs_day)
    tmp = od_selected.copy()
    tmp["touches"] = tmp["path_type"].map(_touches_for_path).astype(int)
    tmp["mm_processing_cpp_od"] = tmp["touches"] * per_touch
    tmp["mm_linehaul_cpp_od"] = (tmp["cost_candidate_path"] / tmp["pkgs_day"]) - tmp["mm_processing_cpp_od"]
    tmp.loc[~np.isfinite(tmp["mm_linehaul_cpp_od"]), "mm_linehaul_cpp_od"] = 0.0

    def _weighted_cpp(group: pd.DataFrame) -> pd.Series:
        w = group["pkgs_day"].sum()
        if w <= 0:
            return pd.Series({"mm_processing_cpp": 0.0, "mm_linehaul_cpp": 0.0})
        return pd.Series({
            "mm_processing_cpp": float(np.average(group["mm_processing_cpp_od"], weights=group["pkgs_day"])),
            "mm_linehaul_cpp":   float(np.average(group["mm_linehaul_cpp_od"],   weights=group["pkgs_day"])),
        })

    agg = tmp.groupby("origin", as_index=False).apply(_weighted_cpp, include_groups=False)
    agg = agg.rename(columns={"origin": "facility"})

    vols = vols.merge(agg, on="facility", how="left").fillna({"mm_processing_cpp":0.0,"mm_linehaul_cpp":0.0})

    # Constant CPPs
    vols["injection_sort_cpp"] = sort_cost
    vols["last_mile_cpp"] = lm_cpp

    # Total
    vols["total_variable_cpp"] = vols["injection_sort_cpp"] + vols["mm_processing_cpp"] + vols["mm_linehaul_cpp"] + vols["last_mile_cpp"]

    # Order columns
    ordered = [
        "facility","market","region",
        "direct_injection_pkgs","origin_mm_pkgs","origin_pkgs_day",
        "zone_0_pkgs","zone_1-2_pkgs","zone_3_pkgs","zone_4_pkgs","zone_5_pkgs","zone_6_pkgs","zone_7_pkgs","zone_8_pkgs",
        "injection_sort_cpp","mm_processing_cpp","mm_linehaul_cpp","last_mile_cpp","total_variable_cpp",
    ]
    for c in ordered:
        if c not in vols.columns:
            vols[c] = np.nan
    return vols[ordered].sort_values(["facility"])
