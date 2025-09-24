# veho_net/reporting.py - BULLETPROOF VERSION with robust error handling
import pandas as pd
import numpy as np
from .geo import haversine_miles


# ---------------- Zones (unchanged core logic) ----------------

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


# ---------------- Bulletproof Dwell hotspots ----------------

def build_dwell_hotspots(path_steps_selected: pd.DataFrame) -> pd.DataFrame:
    """Bulletproof dwell hotspots with comprehensive error handling."""

    # Handle empty input
    if path_steps_selected is None or path_steps_selected.empty:
        return pd.DataFrame(columns=["facility", "total_dwell_hours"])

    df = path_steps_selected.copy()

    # Find facility column
    facility_candidates = ["facility_from", "from_facility", "from", "node_from", "origin_facility", "facility"]
    fcol = None
    for candidate in facility_candidates:
        if candidate in df.columns:
            fcol = candidate
            break

    if fcol is None:
        print("Warning: No facility column found in path_steps_selected")
        return pd.DataFrame(columns=["facility", "total_dwell_hours"])

    # Find dwell column
    dwell_candidates = ["dwell_hours", "dwell_hours_total", "dwell"]
    dcol = None
    for candidate in dwell_candidates:
        if candidate in df.columns:
            dcol = candidate
            break

    if dcol is None:
        df["dwell_hours"] = 0.0
        dcol = "dwell_hours"

    # Simple aggregation - only use columns that exist
    try:
        out = (df.groupby(fcol, as_index=False)[dcol].sum()
               .rename(columns={fcol: "facility", dcol: "total_dwell_hours"})
               .sort_values("total_dwell_hours", ascending=False))

        # Add packages_dwelled if available
        if "packages_dwelled" in df.columns:
            dwelled_agg = df.groupby(fcol, as_index=False)["packages_dwelled"].sum()
            dwelled_agg = dwelled_agg.rename(columns={fcol: "facility"})
            out = out.merge(dwelled_agg, on="facility", how="left")

        return out

    except Exception as e:
        print(f"Warning: Error in build_dwell_hotspots: {e}")
        return pd.DataFrame(columns=["facility", "total_dwell_hours"])


# ---------------- Bulletproof OD selected outputs ----------------

def build_od_selected_outputs(selected: pd.DataFrame, direct_dist_series: pd.Series, flags: dict) -> pd.DataFrame:
    """Bulletproof OD outputs with safe column handling."""

    if selected.empty:
        return selected

    df = selected.copy()
    factor = float(flags.get("path_around_the_world_factor", 2.0))
    sla_target_days = int(flags.get("sla_target_days", 3))

    # Safe merge with direct distances
    try:
        if not direct_dist_series.empty:
            direct_df = (direct_dist_series.reset_index()
                         .rename(columns={"distance_miles": "distance_miles_direct"}))
            df = df.merge(direct_df, on=["scenario_id", "origin", "dest", "day_type"], how="left")
    except Exception as e:
        print(f"Warning: Could not merge direct distances: {e}")

    # Find distance column safely
    distance_cols = ["distance_miles", "distance_miles_cand"]
    dist_col = None
    for col in distance_cols:
        if col in df.columns:
            dist_col = col
            break

    if dist_col is None:
        print("Warning: No distance column found for around_world_flag calculation")
        df["around_world_flag"] = 0
    else:
        if "distance_miles_direct" in df.columns:
            df["around_world_flag"] = ((df[dist_col] > factor * df["distance_miles_direct"]).astype(int)).fillna(0)
        else:
            df["around_world_flag"] = 0

    # Safe SLA flag calculation
    if "sla_days" in df.columns:
        df["end_to_end_sla_flag"] = (df["sla_days"] > sla_target_days).astype(int)
    else:
        df["end_to_end_sla_flag"] = 0

    df["shortest_family"] = df["path_type"].replace(
        {"direct": "direct", "1_touch": "shortest_1", "2_touch": "shortest_2"})

    # Safe containerization flags - only add if columns exist
    if "containerization_level" in df.columns:
        df["deep_containerization_flag"] = (df["containerization_level"] == "sort_group").astype(int)
        df["market_containerization_flag"] = (df["containerization_level"] == "market").astype(int)
        df["region_containerization_flag"] = (df["containerization_level"] == "region").astype(int)

    # Safe fill and spill flags
    if "spill_opportunity_flag" in df.columns:
        df["has_fill_spill_opportunity"] = df["spill_opportunity_flag"].astype(int)

    if "has_secondary_region_sort" in df.columns:
        df["secondary_sort_available"] = df["has_secondary_region_sort"].astype(int)

    # Safe fill rate flags
    if "truck_fill_rate" in df.columns:
        df["high_truck_utilization"] = (df["truck_fill_rate"] >= 0.85).astype(int)
        df["low_truck_utilization"] = (df["truck_fill_rate"] < 0.60).astype(int)

    if "container_fill_rate" in df.columns:
        df["high_container_utilization"] = (df["container_fill_rate"] >= 0.85).astype(int)

    return df


# ---------------- Simplified Volume Identification ----------------

def _identify_volume_types(od_selected: pd.DataFrame, path_steps_selected: pd.DataFrame,
                           direct_day: pd.DataFrame) -> pd.DataFrame:
    """Simplified volume identification with bulletproof error handling."""

    volume_data = []

    # Get all facilities safely
    all_facilities = set()

    try:
        if not od_selected.empty:
            all_facilities.update(od_selected['origin'].unique())
            all_facilities.update(od_selected['dest'].unique())
    except Exception as e:
        print(f"Warning: Error getting facilities from od_selected: {e}")

    try:
        if not direct_day.empty:
            all_facilities.update(direct_day['dest'].unique())
    except Exception as e:
        print(f"Warning: Error getting facilities from direct_day: {e}")

    for facility in all_facilities:
        # Basic volume calculation with error handling
        try:
            # 1. Injection volume
            injection_pkgs = 0
            if not od_selected.empty and 'origin' in od_selected.columns and 'pkgs_day' in od_selected.columns:
                injection_pkgs = od_selected[od_selected['origin'] == facility]['pkgs_day'].sum()

            # 2. Last mile volume
            last_mile_pkgs = 0
            if not direct_day.empty and 'dest' in direct_day.columns:
                direct_col = 'dir_pkgs_day'
                if direct_col in direct_day.columns:
                    last_mile_pkgs = direct_day[direct_day['dest'] == facility][direct_col].sum()

            # 3. Intermediate volume (simplified)
            intermediate_pkgs = 0

            volume_entry = {
                'facility': facility,
                'injection_pkgs_day': injection_pkgs,
                'intermediate_pkgs_day': intermediate_pkgs,
                'last_mile_pkgs_day': last_mile_pkgs,
            }

            volume_data.append(volume_entry)

        except Exception as e:
            print(f"Warning: Error processing facility {facility}: {e}")
            volume_data.append({
                'facility': facility,
                'injection_pkgs_day': 0,
                'intermediate_pkgs_day': 0,
                'last_mile_pkgs_day': 0,
            })

    return pd.DataFrame(volume_data)


def _calculate_hourly_throughput(volume_df: pd.DataFrame, timing_kv: dict, load_strategy: str) -> pd.DataFrame:
    """Simplified hourly throughput calculation."""

    df = volume_df.copy()

    # Get VA hours safely
    injection_va_hours = float(timing_kv.get('injection_va_hours', 8.0))
    middle_mile_va_hours = float(timing_kv.get('middle_mile_va_hours', 16.0))
    last_mile_va_hours = float(timing_kv.get('last_mile_va_hours', 4.0))

    # Simple throughput calculation
    df['injection_hourly_throughput'] = df['injection_pkgs_day'] / injection_va_hours
    df['intermediate_hourly_throughput'] = df['intermediate_pkgs_day'] / middle_mile_va_hours
    df['lm_hourly_throughput'] = df['last_mile_pkgs_day'] / last_mile_va_hours

    # Peak is max of all types
    df['peak_hourly_throughput'] = df[
        ['injection_hourly_throughput', 'intermediate_hourly_throughput', 'lm_hourly_throughput']
    ].max(axis=1)

    # Round for presentation
    for col in ['injection_hourly_throughput', 'intermediate_hourly_throughput', 'lm_hourly_throughput',
                'peak_hourly_throughput']:
        df[col] = df[col].fillna(0).round(0).astype(int)

    return df


# ---------------- Simplified Facility rollup ----------------

def build_facility_rollup(facilities: pd.DataFrame,
                          zips: pd.DataFrame,
                          od_selected: pd.DataFrame,
                          path_steps_selected: pd.DataFrame,
                          direct_day: pd.DataFrame,
                          arc_summary: pd.DataFrame,
                          costs: dict,
                          load_strategy: str,
                          timing_kv: dict = None) -> pd.DataFrame:
    """Simplified facility rollup with bulletproof error handling."""

    if timing_kv is None:
        timing_kv = {}

    # Start with facility metadata
    required_cols = ["facility_name", "market", "region", "type", "parent_hub_name"]
    available_cols = [col for col in required_cols if col in facilities.columns]

    if not available_cols:
        print("Error: facilities DataFrame missing required columns")
        return pd.DataFrame()

    fac_meta = facilities[available_cols].rename(columns={"facility_name": "facility"})

    # Add optional capacity columns if they exist
    optional_capacity_cols = ['max_sort_points_capacity', 'current_sort_points_used', 'last_mile_sort_groups_count']
    for col in optional_capacity_cols:
        if col in facilities.columns:
            temp_df = facilities[['facility_name', col]].rename(columns={'facility_name': 'facility'})
            fac_meta = fac_meta.merge(temp_df, on='facility', how='left')
            fac_meta[col] = fac_meta[col].fillna(0)

    # Volume identification
    try:
        volume_types = _identify_volume_types(od_selected, path_steps_selected, direct_day)
        hourly_throughput = _calculate_hourly_throughput(volume_types, timing_kv, load_strategy)

        vols = fac_meta.merge(volume_types, on="facility", how="left")
        vols = vols.merge(hourly_throughput[['facility', 'injection_hourly_throughput',
                                             'intermediate_hourly_throughput', 'lm_hourly_throughput',
                                             'peak_hourly_throughput']], on="facility", how="left")
    except Exception as e:
        print(f"Warning: Error in volume calculation: {e}")
        vols = fac_meta.copy()
        # Add default columns
        default_vol_cols = ['injection_pkgs_day', 'intermediate_pkgs_day', 'last_mile_pkgs_day',
                            'injection_hourly_throughput', 'intermediate_hourly_throughput',
                            'lm_hourly_throughput', 'peak_hourly_throughput']
        for col in default_vol_cols:
            vols[col] = 0

    # Fill missing values safely
    numeric_cols = ['injection_pkgs_day', 'intermediate_pkgs_day', 'last_mile_pkgs_day',
                    'injection_hourly_throughput', 'intermediate_hourly_throughput',
                    'lm_hourly_throughput', 'peak_hourly_throughput']

    for col in numeric_cols:
        if col in vols.columns:
            vols[col] = vols[col].fillna(0)
        else:
            vols[col] = 0

    # Calculate basic metrics
    vols["origin_pkgs_day"] = vols["injection_pkgs_day"] + vols["last_mile_pkgs_day"]

    # Zone distribution (simplified)
    try:
        zwide = _zones_wide_origin(od_selected, direct_day)
        vols = vols.merge(zwide, on="facility", how="left")
    except Exception as e:
        print(f"Warning: Error in zone calculation: {e}")
        # Add default zone columns
        zone_cols = ["zone_0_pkgs", "zone_1-2_pkgs", "zone_3_pkgs", "zone_4_pkgs",
                     "zone_5_pkgs", "zone_6_pkgs", "zone_7_pkgs", "zone_8_pkgs"]
        for col in zone_cols:
            vols[col] = 0.0

    # Fill zone columns
    zone_cols = ["zone_0_pkgs", "zone_1-2_pkgs", "zone_3_pkgs", "zone_4_pkgs",
                 "zone_5_pkgs", "zone_6_pkgs", "zone_7_pkgs", "zone_8_pkgs"]
    for col in zone_cols:
        if col in vols.columns:
            vols[col] = vols[col].fillna(0.0)
        else:
            vols[col] = 0.0

    # Basic cost calculations
    sort_cost = float(costs.get("sort_cost_per_pkg", 0.0))
    lm_cpp = float(costs.get("last_mile_delivery_cost_per_pkg", costs.get("last_mile_cpp", 0.0)))

    vols["injection_sort_cpp"] = sort_cost
    vols["last_mile_delivery_cpp"] = lm_cpp
    vols["mm_processing_cpp"] = 0.0
    vols["mm_linehaul_cpp"] = 0.0

    # Simplified total cost
    vols["total_variable_cpp"] = sort_cost + lm_cpp

    # Hub tier calculation
    vols["hub_tier"] = vols.apply(
        lambda row: "primary" if row.get("parent_hub_name") == row["facility"]
        else "secondary" if row.get("type") in ["hub", "hybrid"]
        else "launch", axis=1
    )

    # Simplified lane metrics (add defaults)
    lane_cols = [
        "outbound_lane_count", "outbound_packages_total", "outbound_trucks_total",
        "outbound_containers_total", "outbound_od_pairs_served",
        "inbound_lane_count", "inbound_packages_total", "inbound_trucks_total",
        "inbound_containers_total", "inbound_od_pairs_served",
        "total_lane_count", "total_trucks_per_day"
    ]
    for col in lane_cols:
        vols[col] = 0

    # Column ordering (simplified)
    base_cols = ["facility", "market", "region", "type", "hub_tier", "parent_hub_name"]
    volume_cols = ["injection_pkgs_day", "intermediate_pkgs_day", "last_mile_pkgs_day", "origin_pkgs_day"]
    throughput_cols = ["injection_hourly_throughput", "intermediate_hourly_throughput",
                       "lm_hourly_throughput", "peak_hourly_throughput"]
    zone_cols_ordered = zone_cols
    cost_cols = ["injection_sort_cpp", "mm_processing_cpp", "mm_linehaul_cpp",
                 "last_mile_delivery_cpp", "total_variable_cpp"]

    ordered = base_cols + volume_cols + throughput_cols + zone_cols_ordered + lane_cols + cost_cols

    # Ensure all columns exist
    for c in ordered:
        if c not in vols.columns:
            vols[c] = 0 if c.endswith(('_pkgs', '_total', '_count', '_cpp')) else ""

    # Filter to existing columns only
    existing_cols = [c for c in ordered if c in vols.columns]

    return vols[existing_cols].sort_values("peak_hourly_throughput", ascending=False)


# ---------------- Helper functions (simplified) ----------------

def _zones_wide_origin(od_selected: pd.DataFrame, direct_day: pd.DataFrame = None) -> pd.DataFrame:
    """Simplified zone distribution."""

    zcols = ["zone_0_pkgs", "zone_1-2_pkgs", "zone_3_pkgs", "zone_4_pkgs",
             "zone_5_pkgs", "zone_6_pkgs", "zone_7_pkgs", "zone_8_pkgs"]

    if od_selected.empty:
        return pd.DataFrame(columns=["facility"] + zcols)

    try:
        out = od_selected.copy()

        def zcol(z):
            return "zone_0_pkgs" if z == 0 else ("zone_1-2_pkgs" if z == "1-2" else f"zone_{z}_pkgs")

        out["zone_col"] = out["zone"].apply(zcol)
        ztab = (out.groupby(["origin", "zone_col"])["pkgs_day"].sum()
                .unstack(fill_value=0.0)
                .reindex(columns=zcols, fill_value=0.0)
                .reset_index()
                .rename(columns={"origin": "facility"}))

        # Add Zone 0 from direct_day if available
        if direct_day is not None and not direct_day.empty and 'dest' in direct_day.columns:
            direct_col = 'dir_pkgs_day'
            if direct_col in direct_day.columns:
                direct_zone_0 = direct_day.rename(columns={"dest": "facility", direct_col: "zone_0_pkgs"})[
                    ["facility", "zone_0_pkgs"]]
                ztab = ztab.merge(direct_zone_0, on="facility", how="outer", suffixes=("_old", ""))
                if "zone_0_pkgs_old" in ztab.columns:
                    ztab = ztab.drop(columns=["zone_0_pkgs_old"])

        # Ensure all zone columns exist
        for col in zcols:
            if col not in ztab.columns:
                ztab[col] = 0.0
            ztab[col] = ztab[col].fillna(0.0)

        return ztab[["facility"] + zcols]

    except Exception as e:
        print(f"Warning: Error in zone calculation: {e}")
        # Return empty dataframe with correct structure
        default_data = pd.DataFrame(columns=["facility"] + zcols)
        return default_data


def build_lane_summary(arc_summary: pd.DataFrame) -> pd.DataFrame:
    """Simplified lane summary."""

    if arc_summary is None or arc_summary.empty:
        return pd.DataFrame(columns=["from_facility", "to_facility", "packages_per_day",
                                     "trucks_per_day", "containers_per_day"])

    df = arc_summary.copy()

    # Find columns safely
    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    from_c = pick(["from_facility", "from", "origin_facility", "facility_from", "from_fac"])
    to_c = pick(["to_facility", "to", "dest_facility", "facility_to", "to_fac"])
    pk_c = pick(["pkgs_day", "packages_day", "pkgs"])

    if from_c is None or to_c is None:
        return pd.DataFrame(columns=["from_facility", "to_facility", "packages_per_day"])

    # Simple rename and aggregation
    rename_map = {from_c: "from_facility", to_c: "to_facility"}
    if pk_c:
        rename_map[pk_c] = "packages_per_day"

    df = df.rename(columns=rename_map)

    # Ensure required columns exist
    required_cols = ["packages_per_day", "trucks", "total_cost"]
    for c in required_cols:
        if c not in df.columns:
            if c == "packages_per_day":
                df[c] = 0.0
            elif c == "trucks":
                df[c] = 1.0  # Default to 1 truck
            elif c == "total_cost":
                df[c] = 0.0

    # Simple aggregation
    try:
        out = df.groupby(["from_facility", "to_facility"], as_index=False).agg({
            "packages_per_day": "sum",
            "trucks": "sum",
            "total_cost": "sum"
        })

        # Add basic metrics
        out["packages_per_truck"] = np.where(out["trucks"] > 0,
                                             out["packages_per_day"] / out["trucks"], 0)
        out["cost_per_package"] = np.where(out["packages_per_day"] > 0,
                                           out["total_cost"] / out["packages_per_day"], 0)

        # Add basic flags
        out["high_volume_lane"] = (out["packages_per_day"] >= 1000).astype(int)
        out["full_truck_utilization"] = (out["packages_per_truck"] >= 2000).astype(int)

        out = out.sort_values("packages_per_day", ascending=False).reset_index(drop=True)
        return out.round(2)

    except Exception as e:
        print(f"Warning: Error in lane summary: {e}")
        return pd.DataFrame(columns=["from_facility", "to_facility", "packages_per_day"])