# veho_net/reporting.py - ENHANCED VERSION with VA-based hourly throughput
import pandas as pd
import numpy as np
from .geo import haversine_miles


# ---------------- Zones ----------------

def add_zone(od_selected: pd.DataFrame, facilities: pd.DataFrame) -> pd.DataFrame:
    if "zone" in od_selected.columns and od_selected["zone"].notna().any():
        return od_selected
    fac = facilities.set_index("facility_name")

    def raw_m(o, d):
        return haversine_miles(float(fac.at[o, "lat"]), float(fac.at[o, "lon"]),
                               float(fac.at[d, "lat"]), float(fac.at[d, "lon"]))

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
           .rename(columns={fcol: "facility", dcol: "total_dwell_hours"})
           .sort_values("total_dwell_hours", ascending=False))
    return out


# ---------------- OD selected outputs ----------------

def build_od_selected_outputs(selected: pd.DataFrame, direct_dist_series: pd.Series, flags: dict) -> pd.DataFrame:
    df = selected.copy()
    factor = float(flags.get("path_around_the_world_factor", 2.0))
    sla_target_days = int(flags.get("sla_target_days", 3))

    direct_df = (direct_dist_series.reset_index()
                 .rename(columns={"distance_miles": "distance_miles_direct"}))

    df = df.merge(direct_df, on=["scenario_id", "origin", "dest", "day_type"], how="left")

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
    df["shortest_family"] = df["path_type"].replace(
        {"direct": "direct", "1_touch": "shortest_1", "2_touch": "shortest_2"})
    return df


# ---------------- Enhanced Volume Identification ----------------

def _identify_volume_types(od_selected: pd.DataFrame, path_steps_selected: pd.DataFrame,
                           direct_day: pd.DataFrame) -> pd.DataFrame:
    """
    Identify injection, intermediate, and last mile volumes by facility.

    Returns DataFrame with columns:
    - facility
    - injection_pkgs_day: Packages originating at facility
    - intermediate_pkgs_day: Packages passing through facility
    - last_mile_pkgs_day: Packages ending at facility
    """
    volume_data = []

    # Get all facilities from various sources
    all_facilities = set()
    if not od_selected.empty:
        all_facilities.update(od_selected['origin'].unique())
        all_facilities.update(od_selected['dest'].unique())
    if not direct_day.empty:
        all_facilities.update(direct_day['dest'].unique())
    if not path_steps_selected.empty and 'from_facility' in path_steps_selected.columns:
        all_facilities.update(path_steps_selected['from_facility'].unique())
        all_facilities.update(path_steps_selected['to_facility'].unique())

    for facility in all_facilities:
        # 1. Injection volume: packages originating at facility (middle-mile origins)
        injection_pkgs = od_selected[od_selected['origin'] == facility]['pkgs_day'].sum()

        # 2. Last mile volume: direct injection packages ending at facility
        last_mile_pkgs = direct_day[direct_day['dest'] == facility]['dir_pkgs_day'].sum() if not direct_day.empty else 0

        # 3. Intermediate volume: packages passing through facility (not origin, not final dest)
        intermediate_pkgs = 0

        if not path_steps_selected.empty and 'to_facility' in path_steps_selected.columns:
            # Find all path steps where this facility is the destination
            facility_steps = path_steps_selected[path_steps_selected['to_facility'] == facility].copy()

            if not facility_steps.empty:
                # For each OD pair, check if this facility is intermediate (not the final destination)
                for _, step in facility_steps.iterrows():
                    origin_od = step.get('origin', '')
                    dest_od = step.get('dest', '')

                    # If this facility is not the final OD destination, it's intermediate
                    if facility != dest_od and origin_od and dest_od:
                        # Find the package volume for this OD pair
                        od_volume = od_selected[
                            (od_selected['origin'] == origin_od) &
                            (od_selected['dest'] == dest_od)
                            ]['pkgs_day'].sum()

                        intermediate_pkgs += od_volume

        volume_data.append({
            'facility': facility,
            'injection_pkgs_day': injection_pkgs,
            'intermediate_pkgs_day': intermediate_pkgs,
            'last_mile_pkgs_day': last_mile_pkgs
        })

    return pd.DataFrame(volume_data)


def _calculate_hourly_throughput(volume_df: pd.DataFrame, timing_kv: dict, load_strategy: str) -> pd.DataFrame:
    """
    Calculate hourly throughput requirements based on volume availability windows.
    """
    df = volume_df.copy()

    # Get VA hours from timing parameters
    injection_va_hours = float(timing_kv.get('injection_va_hours', 8.0))
    middle_mile_va_hours = float(timing_kv.get('middle_mile_va_hours', 16.0))
    last_mile_va_hours = float(timing_kv.get('last_mile_va_hours', 4.0))

    # Calculate hourly throughputs
    df['injection_hourly_throughput'] = df['injection_pkgs_day'] / injection_va_hours

    # Intermediate throughput depends on load strategy
    if load_strategy.lower() == 'container':
        # Container strategy: intermediate volume just crossdocked, minimal sorting
        df['intermediate_hourly_throughput'] = 0.0
    else:
        # Fluid strategy: intermediate volume needs full sorting
        df['intermediate_hourly_throughput'] = df['intermediate_pkgs_day'] / middle_mile_va_hours

    df['lm_hourly_throughput'] = df['last_mile_pkgs_day'] / last_mile_va_hours

    # Peak hourly throughput is the maximum of all types
    df['peak_hourly_throughput'] = df[
        ['injection_hourly_throughput', 'intermediate_hourly_throughput', 'lm_hourly_throughput']].max(axis=1)

    # Round for presentation
    throughput_cols = ['injection_hourly_throughput', 'intermediate_hourly_throughput', 'lm_hourly_throughput',
                       'peak_hourly_throughput']
    for col in throughput_cols:
        df[col] = df[col].round(0).astype(int)

    return df


# ---------------- Enhanced Facility rollup with VA-based throughput ----------------

def _per_touch_cost(load_strategy: str, costs: dict) -> float:
    return float(costs.get("crossdock_touch_cost_per_pkg", 0.0)) if load_strategy == "container" else float(
        costs.get("sort_cost_per_pkg", 0.0))


def _touches_for_path(path_type: str) -> int:
    touch_map = {"direct": 0, "1_touch": 1, "2_touch": 2, "3_touch": 3, "4_touch": 4}
    return touch_map.get(path_type, 0)


def _zones_wide_origin(od_selected: pd.DataFrame) -> pd.DataFrame:
    zcols = ["zone_0_pkgs", "zone_1-2_pkgs", "zone_3_pkgs", "zone_4_pkgs", "zone_5_pkgs", "zone_6_pkgs", "zone_7_pkgs",
             "zone_8_pkgs"]
    out = od_selected.copy()

    def zcol(z): return "zone_0_pkgs" if z == 0 else ("zone_1-2_pkgs" if z == "1-2" else f"zone_{z}_pkgs")

    out["zone_col"] = out["zone"].apply(zcol)
    ztab = (out.groupby(["origin", "zone_col"])["pkgs_day"].sum()
            .unstack(fill_value=0.0)
            .reindex(columns=zcols, fill_value=0.0)
            .reset_index()
            .rename(columns={"origin": "facility"}))
    return ztab


def _reconstruct_lanes_from_paths(od_selected: pd.DataFrame):
    """
    Enhanced lane reconstruction from path strings with detailed tracking.
    Returns outbound and inbound lane details per facility.
    """
    outbound_lanes = []  # Lanes originating from each facility
    inbound_lanes = []  # Lanes terminating at each facility

    for _, r in od_selected.iterrows():
        try:
            nodes = str(r["path_str"]).split("->")
            if len(nodes) >= 2:
                pkgs = float(r["pkgs_day"])

                # Extract all facility-to-facility legs in the path
                for i in range(len(nodes) - 1):
                    from_fac = nodes[i].strip()
                    to_fac = nodes[i + 1].strip()

                    outbound_lanes.append({
                        "from_facility": from_fac,
                        "to_facility": to_fac,
                        "pkgs_day": pkgs,
                        "origin_od": r["origin"],
                        "dest_od": r["dest"],
                        "path_type": r["path_type"],
                        "leg_position": i + 1,  # 1st leg, 2nd leg, etc.
                        "is_origin_leg": (i == 0),
                        "is_final_leg": (i == len(nodes) - 2)
                    })

                    inbound_lanes.append({
                        "from_facility": from_fac,
                        "to_facility": to_fac,
                        "pkgs_day": pkgs,
                        "origin_od": r["origin"],
                        "dest_od": r["dest"],
                        "path_type": r["path_type"],
                        "leg_position": i + 1,
                        "is_origin_leg": (i == 0),
                        "is_final_leg": (i == len(nodes) - 2)
                    })
        except Exception as e:
            continue

    outbound_df = pd.DataFrame(outbound_lanes)
    inbound_df = pd.DataFrame(inbound_lanes)

    return outbound_df, inbound_df


def _calculate_lane_metrics(lane_df: pd.DataFrame, arc_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate enhanced lane metrics including truck and container details.
    """
    if lane_df.empty:
        return pd.DataFrame(columns=["from_facility", "to_facility", "lane_packages", "lane_trucks", "lane_containers"])

    # Aggregate lanes (same from->to facility pair)
    lane_summary = lane_df.groupby(["from_facility", "to_facility"]).agg({
        "pkgs_day": "sum",
        "origin_od": "nunique",  # Number of unique OD pairs using this lane
        "path_type": lambda x: x.value_counts().to_dict(),
        "is_origin_leg": "sum",
        "is_final_leg": "sum"
    }).reset_index()

    lane_summary.rename(columns={
        "pkgs_day": "lane_packages",
        "origin_od": "unique_od_pairs"
    }, inplace=True)

    # Add truck and container data from arc_summary if available
    if arc_summary is not None and not arc_summary.empty:
        # Standardize arc_summary column names
        arc_cols = arc_summary.columns.tolist()

        # Find correct column names in arc_summary
        from_col = next((c for c in ["from_facility", "from_fac", "origin_facility"] if c in arc_cols), None)
        to_col = next((c for c in ["to_facility", "to_fac", "dest_facility"] if c in arc_cols), None)
        trucks_col = next((c for c in ["trucks", "truck_count", "trucks_total"] if c in arc_cols), None)
        containers_col = next((c for c in ["containers", "containers_cont", "containers_total"] if c in arc_cols), None)

        if from_col and to_col:
            arc_renamed = arc_summary.rename(columns={
                from_col: "from_facility",
                to_col: "to_facility"
            })

            if trucks_col:
                arc_renamed["lane_trucks"] = arc_renamed[trucks_col]
            if containers_col:
                arc_renamed["lane_containers"] = arc_renamed[containers_col]

            # Merge with lane summary
            merge_cols = ["from_facility", "to_facility"]
            if trucks_col:
                merge_cols.append("lane_trucks")
            if containers_col:
                merge_cols.append("lane_containers")

            lane_summary = lane_summary.merge(
                arc_renamed[merge_cols],
                on=["from_facility", "to_facility"],
                how="left"
            )

    # Fill missing values
    for col in ["lane_trucks", "lane_containers"]:
        if col not in lane_summary.columns:
            lane_summary[col] = 0.0
        else:
            lane_summary[col] = lane_summary[col].fillna(0.0)

    # Calculate efficiency metrics
    lane_summary["packages_per_truck"] = np.where(
        lane_summary["lane_trucks"] > 0,
        lane_summary["lane_packages"] / lane_summary["lane_trucks"],
        0
    )

    lane_summary["containers_per_truck"] = np.where(
        lane_summary["lane_trucks"] > 0,
        lane_summary["lane_containers"] / lane_summary["lane_trucks"],
        0
    )

    return lane_summary.round(2)


def build_facility_rollup(facilities: pd.DataFrame,
                          zips: pd.DataFrame,
                          od_selected: pd.DataFrame,
                          path_steps_selected: pd.DataFrame,
                          direct_day: pd.DataFrame,
                          arc_summary: pd.DataFrame,
                          costs: dict,
                          load_strategy: str,
                          timing_kv: dict = None) -> pd.DataFrame:
    """
    Enhanced facility rollup with VA-based hourly throughput calculations.
    """
    if timing_kv is None:
        timing_kv = {}

    fac_meta = facilities[["facility_name", "market", "region", "type", "parent_hub_name"]].rename(
        columns={"facility_name": "facility"})

    # Enhanced volume identification and hourly throughput calculation
    volume_types = _identify_volume_types(od_selected, path_steps_selected, direct_day)
    hourly_throughput = _calculate_hourly_throughput(volume_types, timing_kv, load_strategy)

    # Merge facility metadata with volume data
    vols = fac_meta.merge(volume_types, on="facility", how="left")
    vols = vols.merge(hourly_throughput[['facility', 'injection_hourly_throughput', 'intermediate_hourly_throughput',
                                         'lm_hourly_throughput', 'peak_hourly_throughput']], on="facility", how="left")

    # Fill missing values
    volume_cols = ['injection_pkgs_day', 'intermediate_pkgs_day', 'last_mile_pkgs_day']
    throughput_cols = ['injection_hourly_throughput', 'intermediate_hourly_throughput', 'lm_hourly_throughput',
                       'peak_hourly_throughput']

    for col in volume_cols + throughput_cols:
        if col not in vols.columns:
            vols[col] = 0
        vols[col] = vols[col].fillna(0)

    # Legacy compatibility: calculate origin_pkgs_day for existing logic
    vols["origin_pkgs_day"] = vols["injection_pkgs_day"] + vols["last_mile_pkgs_day"]

    # Zone distribution (keep existing logic)
    zwide = _zones_wide_origin(od_selected)
    vols = vols.merge(zwide, on="facility", how="left")
    for c in ["zone_0_pkgs", "zone_1-2_pkgs", "zone_3_pkgs", "zone_4_pkgs", "zone_5_pkgs", "zone_6_pkgs", "zone_7_pkgs",
              "zone_8_pkgs"]:
        if c not in vols.columns: vols[c] = 0.0
        vols[c] = vols[c].fillna(0.0)

    # Cost per package calculations (keep existing logic)
    sort_cost = float(costs.get("sort_cost_per_pkg", 0.0))
    lm_cpp = float(costs.get("last_mile_cpp", 0.0))
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
            "mm_linehaul_cpp": float(np.average(group["mm_linehaul_cpp_od"], weights=group["pkgs_day"])),
        })

    cpp = tmp.groupby("origin", as_index=False).apply(_weighted_cpp, include_groups=False).rename(
        columns={"origin": "facility"})
    vols = vols.merge(cpp, on="facility", how="left").fillna({"mm_processing_cpp": 0.0, "mm_linehaul_cpp": 0.0})
    vols["injection_sort_cpp"] = sort_cost
    vols["last_mile_cpp"] = lm_cpp
    vols["total_variable_cpp"] = vols["injection_sort_cpp"] + vols["mm_processing_cpp"] + vols["mm_linehaul_cpp"] + \
                                 vols["last_mile_cpp"]

    # ENHANCED: Detailed lane and truck analysis (keep existing logic)
    outbound_lanes_df, inbound_lanes_df = _reconstruct_lanes_from_paths(od_selected)

    # Outbound lane metrics
    outbound_summary = _calculate_lane_metrics(outbound_lanes_df, arc_summary)
    outbound_facility_metrics = outbound_summary.groupby("from_facility").agg({
        "to_facility": "nunique",  # Number of unique destinations
        "lane_packages": "sum",
        "lane_trucks": "sum",
        "lane_containers": "sum",
        "unique_od_pairs": "sum"
    }).reset_index().rename(columns={
        "from_facility": "facility",
        "to_facility": "outbound_lane_count",
        "lane_packages": "outbound_packages_total",
        "lane_trucks": "outbound_trucks_total",
        "lane_containers": "outbound_containers_total",
        "unique_od_pairs": "outbound_od_pairs_served"
    })

    # Inbound lane metrics
    inbound_summary = _calculate_lane_metrics(inbound_lanes_df, arc_summary)
    inbound_facility_metrics = inbound_summary.groupby("to_facility").agg({
        "from_facility": "nunique",  # Number of unique origins
        "lane_packages": "sum",
        "lane_trucks": "sum",
        "lane_containers": "sum",
        "unique_od_pairs": "sum"
    }).reset_index().rename(columns={
        "to_facility": "facility",
        "from_facility": "inbound_lane_count",
        "lane_packages": "inbound_packages_total",
        "lane_trucks": "inbound_trucks_total",
        "lane_containers": "inbound_containers_total",
        "unique_od_pairs": "inbound_od_pairs_served"
    })

    # Merge lane metrics with volume data
    vols = vols.merge(outbound_facility_metrics, on="facility", how="left")
    vols = vols.merge(inbound_facility_metrics, on="facility", how="left")

    # Fill missing values
    lane_cols = [
        "outbound_lane_count", "outbound_packages_total", "outbound_trucks_total", "outbound_containers_total",
        "outbound_od_pairs_served",
        "inbound_lane_count", "inbound_packages_total", "inbound_trucks_total", "inbound_containers_total",
        "inbound_od_pairs_served"
    ]
    for col in lane_cols:
        if col not in vols.columns:
            vols[col] = 0
        vols[col] = vols[col].fillna(0).astype(int)

    # Calculate hub tier (primary vs secondary)
    vols["hub_tier"] = vols.apply(
        lambda row: "primary" if row["parent_hub_name"] == row["facility"]
        else "secondary" if row["type"] in ["hub", "hybrid"]
        else "launch", axis=1
    )

    # Calculate efficiency metrics
    vols["outbound_packages_per_truck"] = np.where(
        vols["outbound_trucks_total"] > 0,
        vols["outbound_packages_total"] / vols["outbound_trucks_total"],
        0
    ).round(1)

    vols["inbound_packages_per_truck"] = np.where(
        vols["inbound_trucks_total"] > 0,
        vols["inbound_packages_total"] / vols["inbound_trucks_total"],
        0
    ).round(1)

    vols["total_trucks_per_day"] = vols["outbound_trucks_total"] + vols["inbound_trucks_total"]
    vols["total_lane_count"] = vols["outbound_lane_count"] + vols["inbound_lane_count"]

    # Enhanced column ordering with new throughput metrics
    ordered = [
        "facility", "market", "region", "type", "hub_tier", "parent_hub_name",

        # Enhanced volume metrics
        "injection_pkgs_day", "intermediate_pkgs_day", "last_mile_pkgs_day", "origin_pkgs_day",

        # NEW: VA-based hourly throughput metrics
        "injection_hourly_throughput", "intermediate_hourly_throughput", "lm_hourly_throughput",
        "peak_hourly_throughput",

        # Zone distribution
        "zone_0_pkgs", "zone_1-2_pkgs", "zone_3_pkgs", "zone_4_pkgs",
        "zone_5_pkgs", "zone_6_pkgs", "zone_7_pkgs", "zone_8_pkgs",

        # CLEAR OUTBOUND METRICS (for Monday question #3)
        "outbound_lane_count", "outbound_trucks_total", "outbound_packages_total",
        "outbound_containers_total", "outbound_packages_per_truck", "outbound_od_pairs_served",

        # CLEAR INBOUND METRICS (for Monday question #3)
        "inbound_lane_count", "inbound_trucks_total", "inbound_packages_total",
        "inbound_containers_total", "inbound_packages_per_truck", "inbound_od_pairs_served",

        # Summary metrics
        "total_lane_count", "total_trucks_per_day",

        # Cost metrics
        "injection_sort_cpp", "mm_processing_cpp", "mm_linehaul_cpp", "last_mile_cpp", "total_variable_cpp",
    ]

    for c in ordered:
        if c not in vols.columns:
            vols[c] = np.nan

    return vols[ordered].sort_values(["hub_tier", "peak_hourly_throughput"], ascending=[True, False])


# ---------------- Enhanced Lane summary ----------------

def build_lane_summary(arc_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced lane summary showing facility-to-facility connections with clear metrics.
    """
    if arc_summary is None or arc_summary.empty:
        return pd.DataFrame(
            columns=["from_facility", "to_facility", "packages_per_day", "trucks_per_day", "containers_per_day"])

    df = arc_summary.copy()

    # Column resolution
    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    from_c = pick(["from_facility", "from", "origin_facility", "facility_from", "from_fac"])
    to_c = pick(["to_facility", "to", "dest_facility", "facility_to", "to_fac"])
    pk_c = pick(["pkgs_day", "packages_day", "pkgs"])
    cont_c = pick(["containers", "containers_cont", "containers_total"])
    trk_c = pick(["trucks", "truck_count", "trucks_total"])
    cost_c = pick(["total_cost", "lane_cost", "cost"])

    if from_c is None or to_c is None:
        return pd.DataFrame(
            columns=["from_facility", "to_facility", "packages_per_day", "trucks_per_day", "containers_per_day"])

    # Standardize names
    rename_map = {from_c: "from_facility", to_c: "to_facility"}
    if pk_c and pk_c != "packages_per_day": rename_map[pk_c] = "packages_per_day"
    if cont_c and cont_c != "containers_per_day": rename_map[cont_c] = "containers_per_day"
    if trk_c and trk_c != "trucks_per_day": rename_map[trk_c] = "trucks_per_day"
    if cost_c and cost_c != "total_lane_cost": rename_map[cost_c] = "total_lane_cost"
    df = df.rename(columns=rename_map)

    # Ensure required columns
    for c in ["packages_per_day", "containers_per_day", "trucks_per_day", "total_lane_cost"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Aggregate to unique lanes
    out = df.groupby(["from_facility", "to_facility"], as_index=False).agg({
        "packages_per_day": "sum",
        "containers_per_day": "sum",
        "trucks_per_day": "sum",
        "total_lane_cost": "sum"
    })

    # Calculate efficiency metrics
    out["cost_per_package"] = np.where(out["packages_per_day"] > 0,
                                       out["total_lane_cost"] / out["packages_per_day"], 0)
    out["packages_per_truck"] = np.where(out["trucks_per_day"] > 0,
                                         out["packages_per_day"] / out["trucks_per_day"], 0)
    out["containers_per_truck"] = np.where(out["trucks_per_day"] > 0,
                                           out["containers_per_day"] / out["trucks_per_day"], 0)

    # Add utilization flags
    out["high_volume_lane"] = (out["packages_per_day"] >= 1000).astype(int)
    out["underutilized_truck"] = (out["packages_per_truck"] < 1000).astype(int)
    out["full_truck_utilization"] = (out["packages_per_truck"] >= 2000).astype(int)

    # Sort by volume for easy analysis
    out = out.sort_values("packages_per_day", ascending=False).reset_index(drop=True)

    # Round for presentation
    numeric_cols = ["cost_per_package", "packages_per_truck", "containers_per_truck"]
    for col in numeric_cols:
        out[col] = out[col].round(2)

    return out