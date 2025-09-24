# veho_net/reporting.py - COMPLETE with Sort Optimization and Fill & Spill Analysis
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


# ---------------- Enhanced Dwell hotspots ----------------

def build_dwell_hotspots(path_steps_selected: pd.DataFrame) -> pd.DataFrame:
    if path_steps_selected is None or path_steps_selected.empty:
        return pd.DataFrame(columns=["facility", "total_dwell_hours", "packages_dwelled"])

    df = path_steps_selected.copy()
    from_candidates = ["facility_from", "from_facility", "from", "node_from", "origin_facility", "facility"]
    fcol = next((c for c in from_candidates if c in df.columns), None)
    dwell_candidates = ["dwell_hours", "dwell_hours_total", "dwell"]
    dcol = next((c for c in dwell_candidates if c in df.columns), None)

    if fcol is None:
        return pd.DataFrame(columns=["facility", "total_dwell_hours", "packages_dwelled"])

    if dcol is None:
        df["dwell_hours"] = 0.0
        dcol = "dwell_hours"

    # Enhanced dwell analysis with package counts
    agg_dict = {dcol: "sum"}
    if "packages_dwelled" in df.columns:
        agg_dict["packages_dwelled"] = "sum"
    if "pkgs_day" in df.columns:
        agg_dict["total_packages"] = "sum"

    out = (df.groupby(fcol, as_index=False).agg(agg_dict)
           .rename(columns={fcol: "facility", dcol: "total_dwell_hours"})
           .sort_values("total_dwell_hours", ascending=False))

    # Calculate dwell rate if data available
    if "packages_dwelled" in out.columns and "total_packages" in out.columns:
        out["dwell_rate"] = np.where(out["total_packages"] > 0,
                                     out["packages_dwelled"] / out["total_packages"],
                                     0.0)
        out["dwell_rate"] = out["dwell_rate"].round(3)

    return out


# ---------------- Enhanced OD selected outputs with Fill & Spill ----------------

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

    # Enhanced: Add containerization efficiency flags
    if "containerization_level" in df.columns:
        df["deep_containerization_flag"] = (df["containerization_level"] == "sort_group").astype(int)
        df["market_containerization_flag"] = (df["containerization_level"] == "market").astype(int)
        df["region_containerization_flag"] = (df["containerization_level"] == "region").astype(int)

    # Enhanced: Add fill and spill flags for operational insights
    if "spill_opportunity_flag" in df.columns:
        df["has_fill_spill_opportunity"] = df["spill_opportunity_flag"].astype(int)

    if "has_secondary_region_sort" in df.columns:
        df["secondary_sort_available"] = df["has_secondary_region_sort"].astype(int)

    # Enhanced: Add fill rate efficiency flags
    if "truck_fill_rate" in df.columns:
        df["high_truck_utilization"] = (df["truck_fill_rate"] >= 0.85).astype(int)
        df["low_truck_utilization"] = (df["truck_fill_rate"] < 0.60).astype(int)

    if "container_fill_rate" in df.columns:
        df["high_container_utilization"] = (df["container_fill_rate"] >= 0.85).astype(int)

    # Enhanced: Containerization priority scoring for operational planning
    if "containerization_efficiency_score" in df.columns:
        df["high_efficiency_route"] = (df["containerization_efficiency_score"] >= 100).astype(
            int)  # >$100/day per sort point
        df["optimization_priority"] = pd.cut(
            df["containerization_efficiency_score"],
            bins=[-float('inf'), 25, 75, 150, float('inf')],
            labels=['Low', 'Medium', 'High', 'Critical']
        ).astype(str)

    return df


# ---------------- Enhanced Volume Identification with Containerization ----------------

def _identify_volume_types(od_selected: pd.DataFrame, path_steps_selected: pd.DataFrame,
                           direct_day: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced volume identification with containerization level awareness and fill/spill tracking.
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

        # Enhanced: Break down injection by containerization level
        injection_by_level = {}
        if 'containerization_level' in od_selected.columns:
            level_breakdown = od_selected[od_selected['origin'] == facility].groupby('containerization_level')[
                'pkgs_day'].sum()
            injection_by_level = {
                'injection_region_pkgs': level_breakdown.get('region', 0),
                'injection_market_pkgs': level_breakdown.get('market', 0),
                'injection_sort_group_pkgs': level_breakdown.get('sort_group', 0),
            }

        # Enhanced: Fill and spill volume tracking
        spill_metrics = {}
        if 'spill_opportunity_flag' in od_selected.columns:
            facility_ods = od_selected[od_selected['origin'] == facility]
            spill_volume = facility_ods[facility_ods['spill_opportunity_flag'] == True]['pkgs_day'].sum()
            spill_destinations = facility_ods[facility_ods['spill_opportunity_flag'] == True]['dest'].nunique()

            spill_metrics = {
                'spillable_volume_pkgs_day': spill_volume,
                'spill_destinations_count': spill_destinations,
                'spill_capability_pct': (spill_volume / max(injection_pkgs, 1)) * 100 if injection_pkgs > 0 else 0,
            }

        # 2. Last mile volume: direct injection packages ending at facility
        last_mile_pkgs = direct_day[direct_day['dest'] == facility]['dir_pkgs_day'].sum() if not direct_day.empty else 0

        # 3. Intermediate volume: packages passing through facility (not origin, not final dest)
        intermediate_pkgs = 0
        if not path_steps_selected.empty and 'to_facility' in path_steps_selected.columns:
            facility_steps = path_steps_selected[path_steps_selected['to_facility'] == facility].copy()

            if not facility_steps.empty:
                for _, step in facility_steps.iterrows():
                    origin_od = step.get('origin', '')
                    dest_od = step.get('dest', '')

                    if facility != dest_od and origin_od and dest_od:
                        od_volume = od_selected[
                            (od_selected['origin'] == origin_od) &
                            (od_selected['dest'] == dest_od)
                            ]['pkgs_day'].sum()
                        intermediate_pkgs += od_volume

        # Enhanced: Calculate fill rate metrics for this facility's outbound lanes
        facility_fill_rates = {}
        if not od_selected.empty:
            facility_ods = od_selected[od_selected['origin'] == facility]
            if not facility_ods.empty and 'truck_fill_rate' in facility_ods.columns:
                facility_fill_rates = {
                    'avg_truck_fill_rate': facility_ods['truck_fill_rate'].mean(),
                    'avg_container_fill_rate': facility_ods.get('container_fill_rate', pd.Series([0])).mean(),
                    'packages_dwelled': facility_ods.get('packages_dwelled', pd.Series([0])).sum(),
                }

        # Enhanced: Sort point utilization for this facility
        sort_utilization = {}
        if 'sort_points_used' in od_selected.columns:
            facility_sort_points = od_selected[od_selected['origin'] == facility]['sort_points_used'].sum()
            sort_utilization = {
                'sort_points_allocated': facility_sort_points,
                'avg_efficiency_score': od_selected[od_selected['origin'] == facility].get(
                    'containerization_efficiency_score', pd.Series([0])).mean(),
            }

        volume_entry = {
            'facility': facility,
            'injection_pkgs_day': injection_pkgs,
            'intermediate_pkgs_day': intermediate_pkgs,
            'last_mile_pkgs_day': last_mile_pkgs,
            **injection_by_level,
            **spill_metrics,
            **facility_fill_rates,
            **sort_utilization
        }

        # Fill missing values with 0
        default_cols = [
            'injection_region_pkgs', 'injection_market_pkgs', 'injection_sort_group_pkgs',
            'spillable_volume_pkgs_day', 'spill_destinations_count', 'spill_capability_pct',
            'avg_truck_fill_rate', 'avg_container_fill_rate', 'packages_dwelled',
            'sort_points_allocated', 'avg_efficiency_score'
        ]
        for key in default_cols:
            if key not in volume_entry:
                volume_entry[key] = 0

        volume_data.append(volume_entry)

    return pd.DataFrame(volume_data)


def _calculate_hourly_throughput(volume_df: pd.DataFrame, timing_kv: dict, load_strategy: str) -> pd.DataFrame:
    """
    Enhanced hourly throughput calculation with containerization level awareness.
    """
    df = volume_df.copy()

    # Get VA hours from timing parameters
    injection_va_hours = float(timing_kv.get('injection_va_hours', 8.0))
    middle_mile_va_hours = float(timing_kv.get('middle_mile_va_hours', 16.0))
    last_mile_va_hours = float(timing_kv.get('last_mile_va_hours', 4.0))

    # Enhanced: Calculate throughput by containerization level
    if load_strategy.lower() == 'container':
        # Region level: full sorting required
        df['region_sort_throughput'] = df.get('injection_region_pkgs', 0) / injection_va_hours

        # Market level: reduced sorting (crossdock style)
        df['market_crossdock_throughput'] = df.get('injection_market_pkgs', 0) / (injection_va_hours * 0.5)

        # Sort group level: minimal handling
        df['sort_group_throughput'] = df.get('injection_sort_group_pkgs', 0) / (injection_va_hours * 0.2)

        # Total injection throughput
        df['injection_hourly_throughput'] = (
                df['region_sort_throughput'] + df['market_crossdock_throughput'] + df['sort_group_throughput']
        )

        # Intermediate throughput: crossdock only
        df['intermediate_hourly_throughput'] = 0.0

    else:
        # Fluid strategy: standard calculation
        df['injection_hourly_throughput'] = df['injection_pkgs_day'] / injection_va_hours
        df['intermediate_hourly_throughput'] = df['intermediate_pkgs_day'] / middle_mile_va_hours

    df['lm_hourly_throughput'] = df['last_mile_pkgs_day'] / last_mile_va_hours

    # Enhanced: Add spill handling throughput if applicable
    if 'spillable_volume_pkgs_day' in df.columns:
        df['spill_handling_throughput'] = df['spillable_volume_pkgs_day'] / injection_va_hours

    # Peak hourly throughput is the maximum of all types
    throughput_cols = ['injection_hourly_throughput', 'intermediate_hourly_throughput', 'lm_hourly_throughput']
    if 'spill_handling_throughput' in df.columns:
        throughput_cols.append('spill_handling_throughput')

    df['peak_hourly_throughput'] = df[throughput_cols].max(axis=1)

    # Round for presentation
    all_throughput_cols = throughput_cols + ['region_sort_throughput', 'market_crossdock_throughput',
                                             'sort_group_throughput', 'peak_hourly_throughput']

    for col in all_throughput_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).round(0).astype(int)

    return df


# ---------------- Enhanced Facility rollup with Sort Optimization ----------------

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
    Complete facility rollup with sort optimization insights, fill rate analysis, and spill capabilities.
    """
    if timing_kv is None:
        timing_kv = {}

    fac_meta = facilities[["facility_name", "market", "region", "type", "parent_hub_name"]].rename(
        columns={"facility_name": "facility"})

    # Add sort capacity information if available
    capacity_cols = ['max_sort_points_capacity', 'current_sort_points_used', 'last_mile_sort_groups_count']
    for col in capacity_cols:
        if col in facilities.columns:
            fac_meta = fac_meta.merge(
                facilities[['facility_name', col]].rename(columns={'facility_name': 'facility'}),
                on='facility', how='left'
            )
            fac_meta[col] = fac_meta[col].fillna(0)

    # Enhanced volume identification and hourly throughput calculation
    volume_types = _identify_volume_types(od_selected, path_steps_selected, direct_day)
    hourly_throughput = _calculate_hourly_throughput(volume_types, timing_kv, load_strategy)

    # Merge facility metadata with enhanced volume data
    vols = fac_meta.merge(volume_types, on="facility", how="left")

    # Merge enhanced throughput data
    throughput_cols_to_merge = [col for col in hourly_throughput.columns
                                if col.endswith('_throughput') and col != 'facility']
    if throughput_cols_to_merge:
        throughput_cols_to_merge = ['facility'] + throughput_cols_to_merge
        vols = vols.merge(hourly_throughput[throughput_cols_to_merge], on="facility", how="left")

    # Fill missing values for all enhanced columns
    enhanced_cols = [
        'injection_pkgs_day', 'intermediate_pkgs_day', 'last_mile_pkgs_day',
        'injection_region_pkgs', 'injection_market_pkgs', 'injection_sort_group_pkgs',
        'spillable_volume_pkgs_day', 'spill_destinations_count', 'spill_capability_pct',
        'avg_truck_fill_rate', 'avg_container_fill_rate', 'packages_dwelled',
        'sort_points_allocated', 'avg_efficiency_score',
        'injection_hourly_throughput', 'intermediate_hourly_throughput', 'lm_hourly_throughput',
        'peak_hourly_throughput', 'region_sort_throughput', 'market_crossdock_throughput',
        'sort_group_throughput'
    ]

    for col in enhanced_cols:
        if col not in vols.columns:
            vols[col] = 0
        vols[col] = vols[col].fillna(0)

    # Total packages handled at facility
    vols["origin_pkgs_day"] = vols["injection_pkgs_day"] + vols["last_mile_pkgs_day"]

    # Enhanced zone distribution with direct injection as zone 0
    zwide = _zones_wide_origin(od_selected, direct_day)
    vols = vols.merge(zwide, on="facility", how="left")
    zone_cols = ["zone_0_pkgs", "zone_1-2_pkgs", "zone_3_pkgs", "zone_4_pkgs", "zone_5_pkgs",
                 "zone_6_pkgs", "zone_7_pkgs", "zone_8_pkgs"]
    for c in zone_cols:
        if c not in vols.columns:
            vols[c] = 0.0
        vols[c] = vols[c].fillna(0.0)

    # Enhanced cost calculations with containerization awareness
    sort_cost = float(costs.get("sort_cost_per_pkg", 0.0))
    lm_cpp = float(costs.get("last_mile_delivery_cost_per_pkg", costs.get("last_mile_cpp", 0.0)))
    lm_sort_cpp = float(costs.get("last_mile_sort_cost_per_pkg", 0.0))

    # Calculate weighted CPP by containerization level
    tmp = od_selected.copy()
    if 'linehaul_cpp' in tmp.columns and 'touch_cpp' in tmp.columns:
        # Use pre-calculated values with containerization awareness
        def _weighted_cpp_enhanced(group: pd.DataFrame) -> pd.Series:
            w = group["pkgs_day"].sum()
            if w <= 0:
                return pd.Series({
                    "mm_processing_cpp": 0.0,
                    "mm_linehaul_cpp": 0.0,
                    "containerization_efficiency": 0.0
                })

            # Calculate efficiency based on containerization levels
            container_efficiency = 0.0
            if 'containerization_level' in group.columns:
                level_weights = {'sort_group': 3, 'market': 2, 'region': 1}
                container_efficiency = sum(
                    level_weights.get(level, 1) * weight
                    for level, weight in group['containerization_level'].value_counts(normalize=True).items()
                )

            return pd.Series({
                "mm_processing_cpp": float(np.average(group["touch_cpp"], weights=group["pkgs_day"])),
                "mm_linehaul_cpp": float(np.average(group["linehaul_cpp"], weights=group["pkgs_day"])),
                "containerization_efficiency": container_efficiency,
            })

        cpp = tmp.groupby("origin", as_index=False).apply(_weighted_cpp_enhanced, include_groups=False).rename(
            columns={"origin": "facility"})
    else:
        # Fallback calculation
        cpp = pd.DataFrame({'facility': vols['facility'].unique(),
                            'mm_processing_cpp': 0.0, 'mm_linehaul_cpp': 0.0, 'containerization_efficiency': 0.0})

    vols = vols.merge(cpp, on="facility", how="left").fillna({
        "mm_processing_cpp": 0.0, "mm_linehaul_cpp": 0.0, "containerization_efficiency": 0.0
    })

    # Base costs
    vols["injection_sort_cpp"] = sort_cost
    vols["last_mile_delivery_cpp"] = lm_cpp
    vols["last_mile_sort_cpp"] = lm_sort_cpp

    # Enhanced total variable CPP with containerization efficiency
    vols["total_variable_cpp"] = np.where(
        vols["origin_pkgs_day"] > 0,
        (
            # Direct injection packages (injection sort + last mile delivery)
                ((vols["injection_sort_cpp"] + vols["last_mile_delivery_cpp"]) * vols["last_mile_pkgs_day"]) +
                # MM injection packages (injection sort + processing + linehaul + last mile sort + delivery)
                ((vols["injection_sort_cpp"] + vols["mm_processing_cpp"] + vols["mm_linehaul_cpp"] +
                  vols["last_mile_sort_cpp"] + vols["last_mile_delivery_cpp"]) * vols["injection_pkgs_day"])
        ) / vols["origin_pkgs_day"],
        0.0
    )

    # Enhanced lane and truck analysis
    outbound_lanes_df, inbound_lanes_df = _reconstruct_lanes_from_paths(od_selected)

    # Enhanced outbound metrics with fill rates
    outbound_summary = _calculate_lane_metrics(outbound_lanes_df, arc_summary)
    if not outbound_summary.empty:
        outbound_facility_metrics = outbound_summary.groupby("from_facility").agg({
            "to_facility": "nunique",
            "lane_packages": "sum",
            "lane_trucks": "sum",
            "lane_containers": "sum",
            "unique_od_pairs": "sum",
            "packages_per_truck": "mean",
            "containers_per_truck": "mean",
        }).reset_index().rename(columns={
            "from_facility": "facility",
            "to_facility": "outbound_lane_count",
            "lane_packages": "outbound_packages_total",
            "lane_trucks": "outbound_trucks_total",
            "lane_containers": "outbound_containers_total",
            "unique_od_pairs": "outbound_od_pairs_served",
            "packages_per_truck": "avg_outbound_packages_per_truck",
            "containers_per_truck": "avg_outbound_containers_per_truck",
        })
    else:
        outbound_facility_metrics = pd.DataFrame(columns=[
            'facility', 'outbound_lane_count', 'outbound_packages_total', 'outbound_trucks_total',
            'outbound_containers_total', 'outbound_od_pairs_served', 'avg_outbound_packages_per_truck',
            'avg_outbound_containers_per_truck'
        ])

    # Enhanced inbound metrics
    inbound_summary = _calculate_lane_metrics(inbound_lanes_df, arc_summary)
    if not inbound_summary.empty:
        inbound_facility_metrics = inbound_summary.groupby("to_facility").agg({
            "from_facility": "nunique",
            "lane_packages": "sum",
            "lane_trucks": "sum",
            "lane_containers": "sum",
            "unique_od_pairs": "sum",
            "packages_per_truck": "mean",
        }).reset_index().rename(columns={
            "to_facility": "facility",
            "from_facility": "inbound_lane_count",
            "lane_packages": "inbound_packages_total",
            "lane_trucks": "inbound_trucks_total",
            "lane_containers": "inbound_containers_total",
            "unique_od_pairs": "inbound_od_pairs_served",
            "packages_per_truck": "avg_inbound_packages_per_truck",
        })
    else:
        inbound_facility_metrics = pd.DataFrame(columns=[
            'facility', 'inbound_lane_count', 'inbound_packages_total', 'inbound_trucks_total',
            'inbound_containers_total', 'inbound_od_pairs_served', 'avg_inbound_packages_per_truck'
        ])

    # Merge enhanced lane metrics
    vols = vols.merge(outbound_facility_metrics, on="facility", how="left")
    vols = vols.merge(inbound_facility_metrics, on="facility", how="left")

    # Fill missing enhanced lane values
    enhanced_lane_cols = [
        "outbound_lane_count", "outbound_packages_total", "outbound_trucks_total", "outbound_containers_total",
        "outbound_od_pairs_served", "avg_outbound_packages_per_truck", "avg_outbound_containers_per_truck",
        "inbound_lane_count", "inbound_packages_total", "inbound_trucks_total", "inbound_containers_total",
        "inbound_od_pairs_served", "avg_inbound_packages_per_truck"
    ]
    for col in enhanced_lane_cols:
        if col not in vols.columns:
            vols[col] = 0
        vols[col] = vols[col].fillna(0)

    # Calculate hub tier with enhanced logic
    vols["hub_tier"] = vols.apply(
        lambda row: "primary" if row["parent_hub_name"] == row["facility"]
        else "secondary" if row["type"] in ["hub", "hybrid"]
        else "launch", axis=1
    )

    # Enhanced efficiency and capacity metrics
    vols["total_trucks_per_day"] = vols["outbound_trucks_total"] + vols["inbound_trucks_total"]
    vols["total_lane_count"] = vols["outbound_lane_count"] + vols["inbound_lane_count"]

    # Calculate sort utilization if capacity data available
    if 'max_sort_points_capacity' in vols.columns:
        vols['sort_utilization_rate'] = np.where(
            vols['max_sort_points_capacity'] > 0,
            vols.get('sort_points_allocated', 0) / vols['max_sort_points_capacity'],
            0.0
        )

        vols['available_sort_capacity'] = vols['max_sort_points_capacity'] - vols.get('sort_points_allocated', 0)

    # Enhanced: Spill handling capacity assessment
    if 'spillable_volume_pkgs_day' in vols.columns and 'available_sort_capacity' in vols.columns:
        vols['spill_handling_capability'] = np.where(
            (vols['spillable_volume_pkgs_day'] > 0) & (vols['available_sort_capacity'] > 0),
            'Available',
            'Constrained' if vols['spillable_volume_pkgs_day'] > 0 else 'Not Required'
        )

    # Enhanced column ordering with new metrics
    base_cols = [
        "facility", "market", "region", "type", "hub_tier", "parent_hub_name",
    ]

    # Sort capacity columns (if available)
    capacity_cols_ordered = [col for col in ['max_sort_points_capacity', 'sort_points_allocated',
                                             'sort_utilization_rate', 'available_sort_capacity',
                                             'last_mile_sort_groups_count']
                             if col in vols.columns]

    # Enhanced volume metrics with containerization breakdown
    volume_cols = [
        "injection_pkgs_day", "intermediate_pkgs_day", "last_mile_pkgs_day", "origin_pkgs_day",
        "injection_region_pkgs", "injection_market_pkgs", "injection_sort_group_pkgs",
    ]

    # Fill and spill metrics
    spill_cols = [col for col in ['spillable_volume_pkgs_day', 'spill_destinations_count',
                                  'spill_capability_pct', 'spill_handling_capability'] if col in vols.columns]

    # Enhanced throughput metrics
    throughput_cols_ordered = [col for col in [
        "injection_hourly_throughput", "intermediate_hourly_throughput", "lm_hourly_throughput",
        "peak_hourly_throughput", "region_sort_throughput", "market_crossdock_throughput",
        "sort_group_throughput", "spill_handling_throughput"
    ] if col in vols.columns]

    # Fill rate and efficiency metrics
    efficiency_cols = [col for col in [
        "avg_truck_fill_rate", "avg_container_fill_rate", "packages_dwelled",
        "containerization_efficiency", "avg_efficiency_score"
    ] if col in vols.columns]

    # Zone distribution
    zone_cols_ordered = zone_cols

    # Enhanced lane metrics
    lane_cols = [
        "outbound_lane_count", "outbound_trucks_total", "outbound_packages_total",
        "outbound_containers_total", "avg_outbound_packages_per_truck", "avg_outbound_containers_per_truck",
        "outbound_od_pairs_served",
        "inbound_lane_count", "inbound_trucks_total", "inbound_packages_total",
        "inbound_containers_total", "avg_inbound_packages_per_truck", "inbound_od_pairs_served",
        "total_lane_count", "total_trucks_per_day",
    ]

    # Cost metrics
    cost_cols = [
        "injection_sort_cpp", "mm_processing_cpp", "mm_linehaul_cpp", "last_mile_sort_cpp",
        "last_mile_delivery_cpp", "total_variable_cpp",
    ]

    ordered = (base_cols + capacity_cols_ordered + volume_cols + spill_cols + throughput_cols_ordered +
               efficiency_cols + zone_cols_ordered + lane_cols + cost_cols)

    # Ensure all columns exist
    for c in ordered:
        if c not in vols.columns:
            vols[c] = 0 if c.endswith(('_pkgs', '_total', '_count', '_rate', '_pct')) else np.nan

    return vols[ordered].sort_values(["hub_tier", "peak_hourly_throughput"], ascending=[True, False])


# ---------------- Helper functions (enhanced with containerization awareness) ----------------

def _per_touch_cost(load_strategy: str, costs: dict) -> float:
    return float(costs.get("crossdock_touch_cost_per_pkg", 0.0)) if load_strategy == "container" else float(
        costs.get("sort_cost_per_pkg", 0.0))


def _touches_for_path(path_type: str) -> int:
    touch_map = {"direct": 0, "1_touch": 1, "2_touch": 2, "3_touch": 3, "4_touch": 4}
    return touch_map.get(path_type, 0)


def _zones_wide_origin(od_selected: pd.DataFrame, direct_day: pd.DataFrame = None) -> pd.DataFrame:
    """Create wide-format zone distribution by origin facility."""
    zcols = ["zone_0_pkgs", "zone_1-2_pkgs", "zone_3_pkgs", "zone_4_pkgs", "zone_5_pkgs", "zone_6_pkgs", "zone_7_pkgs",
             "zone_8_pkgs"]

    if od_selected.empty:
        return pd.DataFrame(columns=["facility"] + zcols)

    out = od_selected.copy()

    def zcol(z):
        return "zone_0_pkgs" if z == 0 else ("zone_1-2_pkgs" if z == "1-2" else f"zone_{z}_pkgs")

    out["zone_col"] = out["zone"].apply(zcol)
    ztab = (out.groupby(["origin", "zone_col"])["pkgs_day"].sum()
            .unstack(fill_value=0.0)
            .reindex(columns=zcols, fill_value=0.0)
            .reset_index()
            .rename(columns={"origin": "facility"}))

    # Add Zone 0 (direct injection) from direct_day
    if direct_day is not None and not direct_day.empty:
        direct_zone_0 = direct_day.rename(columns={"dest": "facility", "dir_pkgs_day": "zone_0_pkgs"})[
            ["facility", "zone_0_pkgs"]]
        ztab = ztab.merge(direct_zone_0, on="facility", how="outer", suffixes=("_old", ""))
        if "zone_0_pkgs_old" in ztab.columns:
            ztab = ztab.drop(columns=["zone_0_pkgs_old"])
        ztab["zone_0_pkgs"] = ztab["zone_0_pkgs"].fillna(0.0)

    # Ensure all zone columns exist and are filled
    for col in zcols:
        if col not in ztab.columns:
            ztab[col] = 0.0
        ztab[col] = ztab[col].fillna(0.0)

    return ztab[["facility"] + zcols]


def _reconstruct_lanes_from_paths(od_selected: pd.DataFrame):
    """Enhanced lane reconstruction with containerization and fill rate tracking."""
    outbound_lanes = []
    inbound_lanes = []

    for _, r in od_selected.iterrows():
        try:
            nodes = str(r["path_str"]).split("->")
            if len(nodes) >= 2:
                pkgs = float(r["pkgs_day"])

                # Enhanced: Extract containerization and fill rate metrics
                truck_fill = r.get('truck_fill_rate', 0.8)
                container_fill = r.get('container_fill_rate', 0.8)
                containerization_level = r.get('containerization_level', 'region')
                sort_points_used = r.get('sort_points_used', 2)
                efficiency_score = r.get('containerization_efficiency_score', 0)
                spill_opportunity = r.get('spill_opportunity_flag', False)

                for i in range(len(nodes) - 1):
                    from_fac = nodes[i].strip()
                    to_fac = nodes[i + 1].strip()

                    base_lane_data = {
                        "from_facility": from_fac,
                        "to_facility": to_fac,
                        "pkgs_day": pkgs,
                        "origin_od": r["origin"],
                        "dest_od": r["dest"],
                        "path_type": r["path_type"],
                        "leg_position": i + 1,
                        "is_origin_leg": (i == 0),
                        "is_final_leg": (i == len(nodes) - 2),
                        # Enhanced metrics
                        "truck_fill_rate": truck_fill,
                        "container_fill_rate": container_fill,
                        "containerization_level": containerization_level,
                        "sort_points_used": sort_points_used,
                        "efficiency_score": efficiency_score,
                        "spill_opportunity": spill_opportunity,
                        "packages_dwelled": r.get('packages_dwelled', 0),
                    }

                    outbound_lanes.append(base_lane_data.copy())
                    inbound_lanes.append(base_lane_data.copy())

        except Exception as e:
            continue

    return pd.DataFrame(outbound_lanes), pd.DataFrame(inbound_lanes)


def _calculate_lane_metrics(lane_df: pd.DataFrame, arc_summary: pd.DataFrame) -> pd.DataFrame:
    """Enhanced lane metrics with containerization and fill rate analysis."""
    if lane_df.empty:
        return pd.DataFrame(columns=["from_facility", "to_facility", "lane_packages", "lane_trucks",
                                     "lane_containers", "avg_truck_fill_rate"])

    # Enhanced aggregation with containerization and fill rate metrics
    agg_dict = {
        "pkgs_day": "sum",
        "origin_od": "nunique",
        "path_type": lambda x: x.value_counts().to_dict(),
        "is_origin_leg": "sum",
        "is_final_leg": "sum",
    }

    # Add fill rate aggregations if available
    if 'truck_fill_rate' in lane_df.columns:
        agg_dict["truck_fill_rate"] = lambda x: np.average(x, weights=lane_df.loc[x.index, "pkgs_day"])
    if 'container_fill_rate' in lane_df.columns:
        agg_dict["container_fill_rate"] = lambda x: np.average(x, weights=lane_df.loc[x.index, "pkgs_day"])
    if 'packages_dwelled' in lane_df.columns:
        agg_dict["packages_dwelled"] = "sum"
    if 'containerization_level' in lane_df.columns:
        agg_dict["containerization_level"] = lambda x: x.mode().iloc[0] if not x.empty else 'region'
    if 'efficiency_score' in lane_df.columns:
        agg_dict["efficiency_score"] = lambda x: np.average(x, weights=lane_df.loc[x.index, "pkgs_day"])
    if 'sort_points_used' in lane_df.columns:
        agg_dict["sort_points_used"] = "sum"
    if 'spill_opportunity' in lane_df.columns:
        agg_dict["spill_opportunities"] = "sum"

    lane_summary = lane_df.groupby(["from_facility", "to_facility"]).agg(agg_dict).reset_index()

    # Rename columns
    rename_dict = {
        "pkgs_day": "lane_packages",
        "origin_od": "unique_od_pairs",
        "truck_fill_rate": "avg_truck_fill_rate",
        "container_fill_rate": "avg_container_fill_rate",
        "efficiency_score": "avg_efficiency_score",
    }
    lane_summary.rename(columns=rename_dict, inplace=True)

    # Add truck and container data from arc_summary if available
    if arc_summary is not None and not arc_summary.empty:
        arc_cols = arc_summary.columns.tolist()
        from_col = next((c for c in ["from_facility", "from_fac", "origin_facility"] if c in arc_cols), None)
        to_col = next((c for c in ["to_facility", "to_fac", "dest_facility"] if c in arc_cols), None)
        trucks_col = next((c for c in ["trucks", "truck_count", "trucks_total"] if c in arc_cols), None)
        containers_col = next(
            (c for c in ["containers", "containers_cont", "containers_total", "physical_containers"] if c in arc_cols),
            None)

        if from_col and to_col:
            arc_renamed = arc_summary.rename(columns={from_col: "from_facility", to_col: "to_facility"})
            merge_cols = ["from_facility", "to_facility"]

            if trucks_col:
                arc_renamed["lane_trucks"] = arc_renamed[trucks_col]
                merge_cols.append("lane_trucks")
            if containers_col:
                arc_renamed["lane_containers"] = arc_renamed[containers_col]
                merge_cols.append("lane_containers")

            lane_summary = lane_summary.merge(arc_renamed[merge_cols], on=["from_facility", "to_facility"], how="left")

    # Fill missing values and calculate efficiency metrics
    default_cols = ["lane_trucks", "lane_containers", "avg_truck_fill_rate", "avg_container_fill_rate",
                    "packages_dwelled", "avg_efficiency_score", "sort_points_used", "spill_opportunities"]
    for col in default_cols:
        if col not in lane_summary.columns:
            lane_summary[col] = 0.0
        else:
            lane_summary[col] = lane_summary[col].fillna(0.0)

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

    # Enhanced efficiency flags with containerization awareness
    lane_summary["high_efficiency_lane"] = (
            (lane_summary["avg_truck_fill_rate"] >= 0.85) &
            (lane_summary["packages_per_truck"] >= 1500) &
            (lane_summary.get("avg_efficiency_score", 0) >= 75)
    ).astype(int)

    lane_summary["optimization_opportunity"] = (
            (lane_summary["avg_truck_fill_rate"] < 0.60) |
            (lane_summary["packages_per_truck"] < 1000) |
            (lane_summary.get("avg_efficiency_score", 0) < 25)
    ).astype(int)

    lane_summary["spill_enabled_lane"] = (lane_summary.get("spill_opportunities", 0) > 0).astype(int)

    return lane_summary.round(3)


# ---------------- Enhanced Lane summary ----------------

def build_lane_summary(arc_summary: pd.DataFrame) -> pd.DataFrame:
    """Enhanced lane summary with containerization and fill rate insights."""
    if arc_summary is None or arc_summary.empty:
        return pd.DataFrame(columns=["from_facility", "to_facility", "packages_per_day",
                                     "trucks_per_day", "containers_per_day"])

    df = arc_summary.copy()

    # Enhanced column resolution for new metrics
    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    from_c = pick(["from_facility", "from", "origin_facility", "facility_from", "from_fac"])
    to_c = pick(["to_facility", "to", "dest_facility", "facility_to", "to_fac"])
    pk_c = pick(["pkgs_day", "packages_day", "pkgs"])
    cont_c = pick(["containers", "containers_cont", "containers_total", "physical_containers"])
    trk_c = pick(["trucks", "truck_count", "trucks_total"])
    cost_c = pick(["total_cost", "lane_cost", "cost"])

    # Enhanced: Fill rate columns
    truck_fill_c = pick(["truck_fill_rate", "truck_utilization", "truck_efficiency"])
    container_fill_c = pick(["container_fill_rate", "container_utilization", "container_efficiency"])
    dwelled_c = pick(["packages_dwelled", "dwelled_packages", "dwell_volume"])

    if from_c is None or to_c is None:
        return pd.DataFrame(columns=["from_facility", "to_facility", "packages_per_day",
                                     "trucks_per_day", "containers_per_day"])

    # Enhanced rename map
    rename_map = {from_c: "from_facility", to_c: "to_facility"}
    if pk_c and pk_c != "packages_per_day":
        rename_map[pk_c] = "packages_per_day"
    if cont_c and cont_c != "containers_per_day":
        rename_map[cont_c] = "containers_per_day"
    if trk_c and trk_c != "trucks_per_day":
        rename_map[trk_c] = "trucks_per_day"
    if cost_c and cost_c != "total_lane_cost":
        rename_map[cost_c] = "total_lane_cost"
    if truck_fill_c:
        rename_map[truck_fill_c] = "avg_truck_fill_rate"
    if container_fill_c:
        rename_map[container_fill_c] = "avg_container_fill_rate"
    if dwelled_c:
        rename_map[dwelled_c] = "packages_dwelled"

    df = df.rename(columns=rename_map)

    # Enhanced ensure required columns
    required_cols = [
        "packages_per_day", "containers_per_day", "trucks_per_day", "total_lane_cost",
        "avg_truck_fill_rate", "avg_container_fill_rate", "packages_dwelled"
    ]
    for c in required_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Enhanced aggregation to unique lanes
    agg_dict = {
        "packages_per_day": "sum",
        "containers_per_day": "sum",
        "trucks_per_day": "sum",
        "total_lane_cost": "sum",
        "packages_dwelled": "sum",
    }

    # Weight-average the fill rates
    if "avg_truck_fill_rate" in df.columns:
        agg_dict["avg_truck_fill_rate"] = lambda x: np.average(x, weights=df.loc[
            x.index, "packages_per_day"]) if x.sum() > 0 else 0
    if "avg_container_fill_rate" in df.columns:
        agg_dict["avg_container_fill_rate"] = lambda x: np.average(x, weights=df.loc[
            x.index, "packages_per_day"]) if x.sum() > 0 else 0

    out = df.groupby(["from_facility", "to_facility"], as_index=False).agg(agg_dict)

    # Enhanced efficiency metrics
    out["cost_per_package"] = np.where(out["packages_per_day"] > 0,
                                       out["total_lane_cost"] / out["packages_per_day"], 0)
    out["packages_per_truck"] = np.where(out["trucks_per_day"] > 0,
                                         out["packages_per_day"] / out["trucks_per_day"], 0)
    out["containers_per_truck"] = np.where(out["trucks_per_day"] > 0,
                                           out["containers_per_day"] / out["trucks_per_day"], 0)

    # Enhanced utilization and opportunity flags
    out["high_volume_lane"] = (out["packages_per_day"] >= 1000).astype(int)
    out["high_efficiency_lane"] = (
            (out["avg_truck_fill_rate"] >= 0.85) & (out["packages_per_truck"] >= 1500)
    ).astype(int)
    out["underutilized_lane"] = (
            (out["avg_truck_fill_rate"] < 0.60) | (out["packages_per_truck"] < 1000)
    ).astype(int)
    out["full_truck_utilization"] = (out["packages_per_truck"] >= 2000).astype(int)
    out["significant_dwell"] = (out["packages_dwelled"] >= 100).astype(int)

    # Enhanced: Calculate dwell rate
    out["dwell_rate"] = np.where(out["packages_per_day"] > 0,
                                 out["packages_dwelled"] / out["packages_per_day"], 0)

    # Sort by volume for easy analysis
    out = out.sort_values("packages_per_day", ascending=False).reset_index(drop=True)

    # Round for presentation
    numeric_cols = ["cost_per_package", "packages_per_truck", "containers_per_truck",
                    "avg_truck_fill_rate", "avg_container_fill_rate", "dwell_rate"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = out[col].round(3)

    return out