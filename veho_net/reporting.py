# veho_net/reporting.py
import pandas as pd
import numpy as np
from .geo import haversine_miles

# ---------------- Zones ----------------

def add_zone(od_selected: pd.DataFrame, facilities: pd.DataFrame) -> pd.DataFrame:
    if "zone" in od_selected.columns and od_selected["zone"].notna().any():
        return od_selected
    fac = facilities.set_index("facility_name")
    def raw_m(o, d):
        return haversine_miles(float(fac.at[o,"lat"]), float(fac.at[o,"lon"]),
                               float(fac.at[d,"lat"]), float(fac.at[d,"lon"]))
    def zlabel(miles: float, o: str, d: str):
        if o == d: return 0
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
    if path_steps_selected is None or path_steps_selected.empty:
        return pd.DataFrame(columns=["facility", "total_dwell_hours"])
    df = path_steps_selected.copy()
    from_candidates = ["facility_from", "from_facility", "from", "node_from", "origin_facility", "facility"]
    fcol = next((c for c in from_candidates if c in df.columns), None)
    dwell_candidates = ["dwell_hours", "dwell_hours_total", "dwell"]
    dcol = next((c for c in dwell_candidates if c in df.columns), None)
    if fcol is None:
        return pd.DataFrame(columns=["facility", "total_dwell_hours"])
    if dcol is None:
        df["dwell_hours"] = 0.0
        dcol = "dwell_hours"
    out = (df.groupby(fcol, as_index=False)[dcol].sum()
             .rename(columns={fcol:"facility", dcol:"total_dwell_hours"})
             .sort_values("total_dwell_hours", ascending=False))
    return out


# ---------------- OD selected outputs (flags) ----------------

def build_od_selected_outputs(selected: pd.DataFrame, direct_dist_series: pd.Series, flags: dict) -> pd.DataFrame:
    df = selected.copy()
    factor = float(flags.get("path_around_the_world_factor", 2.0))
    sla_target_days = int(flags.get("sla_target_days", 3))

    direct_df = (direct_dist_series.reset_index()
                 .rename(columns={"distance_miles":"distance_miles_direct"}))

    df = df.merge(direct_df, on=["scenario_id","origin","dest","day_type"], how="left")

    if "distance_miles" in df.columns:
        dist_col = "distance_miles"
    elif "distance_miles_cand" in df.columns:
        dist_col = "distance_miles_cand"
    else:
        raise KeyError("Selected paths missing distance column.")

    df["around_world_flag"] = ((df[dist_col] > factor * df["distance_miles_direct"]).astype(int)).fillna(0)
    if "sla_days" not in df.columns:
        raise KeyError("Selected paths missing 'sla_days'.")
    df["end_to_end_sla_flag"] = (df["sla_days"] > sla_target_days).astype(int)
    df["shortest_family"] = df["path_type"].replace({"direct":"direct","1_touch":"shortest_1","2_touch":"shortest_2"})
    return df


# ---------------- Facility rollup (origin lens + zones + CPP + lanes/containers) ----------------

def _per_touch_cost(load_strategy: str, costs: dict) -> float:
    return float(costs["crossdock_touch_cost_per_pkg"]) if load_strategy == "container" else float(costs["sort_cost_per_pkg"])

def _touches_for_path(path_type: str) -> int:
    return 0 if path_type == "direct" else (1 if path_type == "1_touch" else 2)

def _zones_wide_origin(od_selected: pd.DataFrame) -> pd.DataFrame:
    zcols = ["zone_0_pkgs","zone_1-2_pkgs","zone_3_pkgs","zone_4_pkgs","zone_5_pkgs","zone_6_pkgs","zone_7_pkgs","zone_8_pkgs"]
    out = od_selected.copy()
    def zcol(z): return "zone_0_pkgs" if z == 0 else ("zone_1-2_pkgs" if z == "1-2" else f"zone_{z}_pkgs")
    out["zone_col"] = out["zone"].apply(zcol)
    ztab = (out.groupby(["origin","zone_col"])["pkgs_day"].sum()
              .unstack(fill_value=0.0)
              .reindex(columns=zcols, fill_value=0.0)
              .reset_index()
              .rename(columns={"origin":"facility"}))
    return ztab

def _reconstruct_first_last_arcs(od_selected: pd.DataFrame):
    """
    From chosen path_str, build the set of first-leg arcs (origin->n1) and final-leg arcs (n_last->dest).
    Returns:
      first_legs: DataFrame ['from_fac','to_fac','pkgs_day'] aggregated
      last_legs:  DataFrame ['from_fac','to_fac','pkgs_day'] aggregated
    """
    recs_first, recs_last = [], []
    for _, r in od_selected.iterrows():
        try:
            nodes = str(r["path_str"]).split("->")
            if len(nodes) >= 2:
                # first leg
                recs_first.append({"from_fac": nodes[0], "to_fac": nodes[1], "pkgs_day": float(r["pkgs_day"])})
                # last leg
                recs_last.append({"from_fac": nodes[-2], "to_fac": nodes[-1], "pkgs_day": float(r["pkgs_day"])})
        except Exception:
            continue
    fl = pd.DataFrame(recs_first)
    ll = pd.DataFrame(recs_last)
    if not fl.empty:
        fl = fl.groupby(["from_fac","to_fac"], as_index=False)["pkgs_day"].sum()
    if not ll.empty:
        ll = ll.groupby(["from_fac","to_fac"], as_index=False)["pkgs_day"].sum()
    return fl, ll

def _classify_arc_roles(arc_summary: pd.DataFrame, od_selected: pd.DataFrame):
    """
    Classify each arc as 'origin', 'destination', or 'intermediate' using reconstructed first/last legs.
    """
    if arc_summary is None or arc_summary.empty:
        return pd.DataFrame(columns=["from_fac","to_fac","role","pkgs_day","containers","trucks"])

    arcs = arc_summary.copy()

    # Standardize column names that might vary
    from_cands = ["from_facility","from","origin_facility","facility_from","from_fac"]
    to_cands   = ["to_facility","to","dest_facility","facility_to","to_fac"]
    cont_cands = ["containers","containers_cont","containers_total"]
    trk_cands  = ["trucks","truck_count","trucks_total"]
    pkg_cands  = ["pkgs_day","packages_day","pkgs"]

    def pick(cols, df):
        for c in cols:
            if c in df.columns: return c
        return None

    fc = pick(from_cands, arcs)
    tc = pick(to_cands, arcs)
    cc = pick(cont_cands, arcs)
    tc2 = pick(trk_cands, arcs)
    pc = pick(pkg_cands, arcs)

    # Minimal required: from/to
    if fc is None or tc is None:
        return pd.DataFrame(columns=["from_fac","to_fac","role","pkgs_day","containers","trucks"])

    arcs = arcs.rename(columns={fc:"from_fac", tc:"to_fac"})
    if pc and pc != "pkgs_day": arcs = arcs.rename(columns={pc:"pkgs_day"})
    if cc and cc != "containers": arcs = arcs.rename(columns={cc:"containers"})
    if tc2 and tc2 != "trucks": arcs = arcs.rename(columns={tc2:"trucks"})

    # Build first/last sets
    fl, ll = _reconstruct_first_last_arcs(od_selected)
    fl["role"] = "origin"
    ll["role"] = "destination"

    # Merge roles; anything else becomes 'intermediate'
    arcs = arcs.merge(fl.assign(flag_origin=1), on=["from_fac","to_fac"], how="left")
    arcs = arcs.merge(ll.assign(flag_dest=1), on=["from_fac","to_fac"], how="left")
    arcs["role"] = np.where(arcs["flag_origin"] == 1, "origin",
                    np.where(arcs["flag_dest"] == 1, "destination", "intermediate"))
    arcs = arcs.drop(columns=[c for c in ["flag_origin","flag_dest"] if c in arcs.columns])

    # Ensure numeric columns
    for c in ["pkgs_day","containers","trucks"]:
        if c not in arcs.columns: arcs[c] = 0.0
        arcs[c] = arcs[c].fillna(0.0).astype(float)

    return arcs[["from_fac","to_fac","role","pkgs_day","containers","trucks"]]

def build_facility_rollup(facilities: pd.DataFrame,
                          zips: pd.DataFrame,
                          od_selected: pd.DataFrame,
                          path_steps_selected: pd.DataFrame,
                          direct_day: pd.DataFrame,
                          arc_summary: pd.DataFrame,
                          costs: dict,
                          load_strategy: str) -> pd.DataFrame:
    """
    Origin-based lens with restored lane/container summaries + zones + CPP.
    """
    fac_meta = facilities[["facility_name","market","region"]].rename(columns={"facility_name":"facility"})

    # Volumes (origin lens)
    origin_mm = od_selected.groupby("origin", as_index=False)["pkgs_day"].sum() \
                           .rename(columns={"origin":"facility","pkgs_day":"origin_mm_pkgs"})
    direct_origin = direct_day.rename(columns={"dest":"facility","dir_pkgs_day":"direct_injection_pkgs"})

    vols = fac_meta.merge(origin_mm, on="facility", how="left").merge(direct_origin, on="facility", how="left")
    vols[["origin_mm_pkgs","direct_injection_pkgs"]] = vols[["origin_mm_pkgs","direct_injection_pkgs"]].fillna(0.0)
    vols["origin_pkgs_day"] = vols["origin_mm_pkgs"] + vols["direct_injection_pkgs"]

    # Zones
    zwide = _zones_wide_origin(od_selected)
    vols = vols.merge(zwide, on="facility", how="left")
    for c in ["zone_0_pkgs","zone_1-2_pkgs","zone_3_pkgs","zone_4_pkgs","zone_5_pkgs","zone_6_pkgs","zone_7_pkgs","zone_8_pkgs"]:
        if c not in vols.columns: vols[c] = 0.0
        vols[c] = vols[c].fillna(0.0)

    # CPPs
    sort_cost = float(costs["sort_cost_per_pkg"])
    lm_cpp = float(costs["last_mile_cpp"])
    per_touch = _per_touch_cost(load_strategy, costs)

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

    cpp = tmp.groupby("origin", as_index=False).apply(_weighted_cpp, include_groups=False).rename(columns={"origin":"facility"})
    vols = vols.merge(cpp, on="facility", how="left").fillna({"mm_processing_cpp":0.0,"mm_linehaul_cpp":0.0})
    vols["injection_sort_cpp"] = sort_cost
    vols["last_mile_cpp"] = lm_cpp
    vols["total_variable_cpp"] = vols["injection_sort_cpp"] + vols["mm_processing_cpp"] + vols["mm_linehaul_cpp"] + vols["last_mile_cpp"]

    # ---------- Lanes & Containers/Trucks (restored) ----------
    arcs = _classify_arc_roles(arc_summary, od_selected)

    # Lane counts per facility (unique counterparties)
    ob_counts = arcs.groupby("from_fac")["to_fac"].nunique().reset_index().rename(columns={"from_fac":"facility","to_fac":"outbound_lane_count"})
    ib_counts = arcs.groupby("to_fac")["from_fac"].nunique().reset_index().rename(columns={"to_fac":"facility","from_fac":"inbound_lane_count"})
    vols = vols.merge(ob_counts, on="facility", how="left").merge(ib_counts, on="facility", how="left")
    vols["outbound_lane_count"] = vols["outbound_lane_count"].fillna(0).astype(int)
    vols["inbound_lane_count"]  = vols["inbound_lane_count"].fillna(0).astype(int)

    # Containers & trucks by role
    role_agg = arcs.groupby(["from_fac","role"], as_index=False)[["containers","trucks","pkgs_day"]].sum()
    # Outbound by role (origin + intermediate)
    ob = role_agg[role_agg["role"].isin(["origin","intermediate"])].groupby("from_fac", as_index=False)[["containers","trucks"]].sum() \
            .rename(columns={"from_fac":"facility","containers":"outbound_containers","trucks":"outbound_trucks"})
    vols = vols.merge(ob, on="facility", how="left")
    vols[["outbound_containers","outbound_trucks"]] = vols[["outbound_containers","outbound_trucks"]].fillna(0.0)

    # Inbound (destination + intermediate incoming) — sum where facility is 'to_fac'
    role_agg_ib = arcs.groupby(["to_fac","role"], as_index=False)[["containers","trucks"]].sum()
    ib = role_agg_ib.groupby("to_fac", as_index=False)[["containers","trucks"]].sum() \
            .rename(columns={"to_fac":"facility","containers":"inbound_containers","trucks":"inbound_trucks"})
    vols = vols.merge(ib, on="facility", how="left")
    vols[["inbound_containers","inbound_trucks"]] = vols[["inbound_containers","inbound_trucks"]].fillna(0.0)

    # Split origin vs intermediate vs destination containers using arc roles
    origin_cont = arcs[arcs["role"]=="origin"].groupby("from_fac", as_index=False)[["containers","trucks"]].sum() \
                    .rename(columns={"from_fac":"facility","containers":"origin_containers","trucks":"origin_trucks"})
    interm_cont = arcs[arcs["role"]=="intermediate"].groupby("from_fac", as_index=False)[["containers","trucks"]].sum() \
                    .rename(columns={"from_fac":"facility","containers":"intermediate_containers","trucks":"intermediate_trucks"})
    dest_cont   = arcs[arcs["role"]=="destination"].groupby("to_fac", as_index=False)[["containers","trucks"]].sum() \
                    .rename(columns={"to_fac":"facility","containers":"destination_containers","trucks":"destination_trucks"})

    vols = vols.merge(origin_cont, on="facility", how="left") \
               .merge(interm_cont, on="facility", how="left") \
               .merge(dest_cont, on="facility", how="left")

    for c in ["origin_containers","intermediate_containers","destination_containers",
              "origin_trucks","intermediate_trucks","destination_trucks"]:
        if c not in vols.columns: vols[c] = 0.0
        vols[c] = vols[c].fillna(0.0)

    # Final column ordering (includes restored metrics)
    ordered = [
        "facility","market","region",
        "direct_injection_pkgs","origin_mm_pkgs","origin_pkgs_day",
        "zone_0_pkgs","zone_1-2_pkgs","zone_3_pkgs","zone_4_pkgs","zone_5_pkgs","zone_6_pkgs","zone_7_pkgs","zone_8_pkgs",
        "injection_sort_cpp","mm_processing_cpp","mm_linehaul_cpp","last_mile_cpp","total_variable_cpp",
        "outbound_lane_count","inbound_lane_count",
        "outbound_containers","inbound_containers","outbound_trucks","inbound_trucks",
        "origin_containers","intermediate_containers","destination_containers",
        "origin_trucks","intermediate_trucks","destination_trucks",
    ]
    for c in ordered:
        if c not in vols.columns: vols[c] = np.nan
    return vols[ordered].sort_values(["facility"])

# ---------------- Lane summary (compat shim around arc_summary) ----------------

def build_lane_summary(arc_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Compatibility wrapper so run_v1 can write a 'lane_summary' sheet.
    If arc_summary already contains lane-level rows, we just normalize column
    names and add a few optional metrics when source columns exist.

    Expected (but not strictly required) columns in arc_summary:
      - from_fac / from_facility / origin_facility / facility_from
      - to_fac   / to_facility   / dest_facility   / facility_to
      - pkgs_day (or packages_day / pkgs)
      - containers (or containers_cont / containers_total)
      - trucks (or truck_count / trucks_total)
      - cost_trucks or total_cost (optional)

    Returns a DataFrame with normalized columns:
      ['from_fac','to_fac','pkgs_day','containers','trucks', ... optional metrics ...]
    """
    if arc_summary is None or arc_summary.empty:
        return pd.DataFrame(columns=["from_fac","to_fac","pkgs_day","containers","trucks"])

    df = arc_summary.copy()

    # Column resolution helpers
    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    from_c = pick(["from_fac","from_facility","origin_facility","facility_from","from"])
    to_c   = pick(["to_fac","to_facility","dest_facility","facility_to","to"])
    pk_c   = pick(["pkgs_day","packages_day","pkgs"])
    cont_c = pick(["containers","containers_cont","containers_total"])
    trk_c  = pick(["trucks","truck_count","trucks_total"])
    cost_c = pick(["cost_trucks","total_cost","cost"])

    # Minimal required: endpoints
    if from_c is None or to_c is None:
        # Return something predictable even if input is non-standard
        return pd.DataFrame(columns=["from_fac","to_fac","pkgs_day","containers","trucks"])

    # Normalize names
    rename_map = {from_c: "from_fac", to_c: "to_fac"}
    if pk_c and pk_c != "pkgs_day": rename_map[pk_c] = "pkgs_day"
    if cont_c and cont_c != "containers": rename_map[cont_c] = "containers"
    if trk_c and trk_c != "trucks": rename_map[trk_c] = "trucks"
    if cost_c and cost_c != "lane_cost": rename_map[cost_c] = "lane_cost"
    df = df.rename(columns=rename_map)

    # Ensure numeric columns exist
    for c in ["pkgs_day","containers","trucks","lane_cost"]:
        if c not in df.columns:
            df[c] = 0.0
    for c in ["pkgs_day","containers","trucks","lane_cost"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Optional metrics (only when inputs exist)
    # Cost per pkg
    df["cpp"] = np.where(df["pkgs_day"] > 0, df["lane_cost"] / df["pkgs_day"], np.nan)

    # Containers per truck (can highlight under/over-utilization at a glance)
    df["containers_per_truck"] = np.where(df["trucks"] > 0, df["containers"] / df["trucks"], np.nan)

    # Aggregate to unique lanes (some MILP pipelines emit multiple rows per lane)
    out = (df.groupby(["from_fac","to_fac"], as_index=False)
             .agg({"pkgs_day":"sum","containers":"sum","trucks":"sum","lane_cost":"sum"})
          )

    # Recompute derived metrics post-aggregation
    out["cpp"] = np.where(out["pkgs_day"] > 0, out["lane_cost"] / out["pkgs_day"], np.nan)
    out["containers_per_truck"] = np.where(out["trucks"] > 0, out["containers"] / out["trucks"], np.nan)

    # Sort for readability
    out = out.sort_values(["from_fac","to_fac"]).reset_index(drop=True)
    return out
