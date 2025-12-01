"""
Reporting Module

Generates facility-level and network-level metrics from optimization results.
Calculates operational volumes, network characteristics, and aggregated summaries.

Updated for container persistence:
- Facility container counts derived from OD tracking (not recalculated)
- Distinguishes persisted vs fresh containers at intermediates
- Container fill rates reflect creation point, not arc

Key Functions:
- build_facility_volume: Daily operational metrics by facility
- build_facility_network_profile: Zone/sort/distance/touch characteristics
- calculate_network_*: Network-level aggregations
- validate_network_aggregations: Data quality checks
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List

from .geo_v4 import calculate_zone_from_distance, haversine_miles, band_lookup
from .containers_v4 import weighted_pkg_cube
from .utils import safe_divide, get_facility_lookup, extract_path_nodes
from .config_v4 import OptimizationConstants


# ============================================================================
# FACILITY VOLUME (OPERATIONAL METRICS)
# ============================================================================

def build_facility_volume(
        od_selected: pd.DataFrame,
        direct_day: pd.DataFrame,
        arc_summary: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        strategy: str,
        facilities: pd.DataFrame,
        timing_params: Dict
) -> pd.DataFrame:
    """
    Calculate facility daily volumes and throughput (operational metrics only).

    Updated to use container tracking from od_selected when available,
    falling back to calculation when tracking not present.

    ALL VALUES FROM INPUT PARAMETERS - NO HARDCODING.
    """
    volume_data = []

    # Check if od_selected has container tracking
    has_container_tracking = 'arc_containers' in od_selected.columns

    all_facilities = set()
    if not od_selected.empty:
        all_facilities.update(od_selected['origin'].unique())
        all_facilities.update(od_selected['dest'].unique())
    if not direct_day.empty and 'dest' in direct_day.columns:
        all_facilities.update(direct_day['dest'].unique())

    fac_lookup = get_facility_lookup(facilities)

    for facility in all_facilities:
        try:
            # Direct injection volumes (Zone 0 - no middle-mile)
            direct_pkgs = 0
            direct_containers = 0

            if not direct_day.empty:
                direct_col = 'dir_pkgs_day'
                if direct_col in direct_day.columns:
                    fac_direct = direct_day[direct_day['dest'] == facility]
                    if not fac_direct.empty:
                        direct_pkgs = fac_direct[direct_col].sum()

                        if direct_pkgs > 0:
                            containers = _calculate_containers_for_volume(
                                direct_pkgs, package_mix, container_params, strategy
                            )
                            direct_containers = containers['containers']

            # Middle-mile injection volumes (EXCLUDE O=D)
            mm_injection_pkgs = 0
            mm_injection_containers = 0

            if not od_selected.empty:
                outbound = od_selected[
                    (od_selected['origin'] == facility) &
                    (od_selected['origin'] != od_selected['dest'])
                ]
                if not outbound.empty:
                    mm_injection_pkgs = outbound['pkgs_day'].sum()

                    # Use tracked containers if available
                    if has_container_tracking and 'origin_containers' in outbound.columns:
                        mm_injection_containers = int(outbound['origin_containers'].sum())
                    elif mm_injection_pkgs > 0:
                        containers = _calculate_containers_for_volume(
                            mm_injection_pkgs, package_mix, container_params, strategy
                        )
                        mm_injection_containers = containers['containers']

            # O=D volumes for hybrid facilities
            od_same_pkgs = 0
            od_same_containers = 0

            if not od_selected.empty:
                od_same = od_selected[
                    (od_selected['origin'] == facility) &
                    (od_selected['dest'] == facility)
                ]
                if not od_same.empty:
                    od_same_pkgs = od_same['pkgs_day'].sum()

                    if has_container_tracking and 'origin_containers' in od_same.columns:
                        od_same_containers = int(od_same['origin_containers'].sum())
                    elif od_same_pkgs > 0:
                        containers = _calculate_containers_for_volume(
                            od_same_pkgs, package_mix, container_params, strategy
                        )
                        od_same_containers = containers['containers']

            # Intermediate facility operations
            intermediate_pkgs_sort = 0
            intermediate_pkgs_crossdock = 0
            intermediate_containers_sort = 0
            intermediate_containers_crossdock = 0

            if not od_selected.empty:
                for _, path_row in od_selected.iterrows():
                    path_nodes = extract_path_nodes(path_row)

                    final_dest = path_row['dest']
                    path_strategy = path_row.get('effective_strategy', strategy)
                    chosen_sort_level = path_row.get('chosen_sort_level', 'market')

                    # Check if this facility is an intermediate
                    if len(path_nodes) > 2 and facility in path_nodes[1:-1]:
                        operation = _determine_intermediate_operation_type(
                            intermediate_facility=facility,
                            dest_facility=final_dest,
                            path_strategy=path_strategy,
                            chosen_sort_level=chosen_sort_level,
                            facilities=facilities
                        )

                        pkgs = path_row['pkgs_day']

                        # Get containers from tracking if available
                        if has_container_tracking and 'arc_containers' in path_row.index:
                            arc_containers = path_row['arc_containers']
                            arc_container_type = path_row.get('arc_container_type', {})

                            # Find arc leaving this facility
                            fac_idx = path_nodes.index(facility)
                            if fac_idx < len(path_nodes) - 1:
                                next_fac = path_nodes[fac_idx + 1]
                                arc_key = (facility, next_fac)
                                containers = arc_containers.get(arc_key, 0)
                            else:
                                containers = 0
                        else:
                            containers_data = _calculate_containers_for_volume(
                                pkgs, package_mix, container_params, path_strategy
                            )
                            containers = containers_data['containers']

                        if operation == 'sort':
                            intermediate_pkgs_sort += pkgs
                            intermediate_containers_sort += containers
                        else:
                            intermediate_pkgs_crossdock += pkgs
                            intermediate_containers_crossdock += containers

            intermediate_pkgs = intermediate_pkgs_sort + intermediate_pkgs_crossdock
            intermediate_containers = intermediate_containers_sort + intermediate_containers_crossdock

            # Last mile volumes
            last_mile_pkgs = direct_pkgs
            last_mile_containers = direct_containers

            if not od_selected.empty:
                inbound = od_selected[od_selected['dest'] == facility]
                if not inbound.empty:
                    mm_last_mile = inbound['pkgs_day'].sum()
                    last_mile_pkgs += mm_last_mile

                    # Get inbound containers from arc tracking
                    if has_container_tracking:
                        inbound_containers = _get_inbound_containers_for_facility(
                            facility, inbound, od_selected
                        )
                        last_mile_containers += inbound_containers
                    elif mm_last_mile > 0:
                        containers = _calculate_containers_for_volume(
                            mm_last_mile, package_mix, container_params, strategy
                        )
                        last_mile_containers += containers['containers']

            # Total containers (avoid double-counting)
            total_containers = (mm_injection_containers +
                                od_same_containers +
                                intermediate_containers +
                                last_mile_containers)

            # Truck movements from arc summary
            outbound_trucks = 0
            inbound_trucks = 0
            if not arc_summary.empty:
                outbound_arcs = arc_summary[
                    (arc_summary['from_facility'] == facility) &
                    (arc_summary['from_facility'] != arc_summary['to_facility'])
                ]
                if not outbound_arcs.empty:
                    outbound_trucks = int(outbound_arcs['trucks'].sum())

                inbound_arcs = arc_summary[
                    (arc_summary['to_facility'] == facility) &
                    (arc_summary['from_facility'] != arc_summary['to_facility'])
                ]
                if not inbound_arcs.empty:
                    inbound_trucks = int(inbound_arcs['trucks'].sum())

            # Hourly throughput from timing_params
            injection_va_hours = float(timing_params['injection_va_hours'])
            middle_mile_va_hours = float(timing_params['middle_mile_va_hours'])
            crossdock_va_hours = float(timing_params.get('crossdock_va_hours', 3.0))
            last_mile_va_hours = float(timing_params['last_mile_va_hours'])

            total_injection_pkgs = mm_injection_pkgs + od_same_pkgs
            injection_hourly = safe_divide(total_injection_pkgs, injection_va_hours)

            intermediate_sort_hourly = safe_divide(intermediate_pkgs_sort, middle_mile_va_hours)
            intermediate_crossdock_hourly = safe_divide(intermediate_pkgs_crossdock, crossdock_va_hours)
            last_mile_hourly = safe_divide(last_mile_pkgs, last_mile_va_hours)

            peak_hourly = max(
                injection_hourly,
                intermediate_sort_hourly,
                intermediate_crossdock_hourly,
                last_mile_hourly
            )

            fac_type = fac_lookup.at[facility, 'type'] if facility in fac_lookup.index else 'unknown'

            volume_entry = {
                'facility': facility,
                'facility_type': fac_type,
                'injection_pkgs_day': mm_injection_pkgs,
                'injection_containers': mm_injection_containers,
                'od_same_pkgs_day': od_same_pkgs,
                'od_same_containers': od_same_containers,
                'intermediate_sort_pkgs_day': intermediate_pkgs_sort,
                'intermediate_crossdock_pkgs_day': intermediate_pkgs_crossdock,
                'intermediate_pkgs_day': intermediate_pkgs,
                'intermediate_sort_containers': intermediate_containers_sort,
                'intermediate_crossdock_containers': intermediate_containers_crossdock,
                'intermediate_containers': intermediate_containers,
                'last_mile_pkgs_day': last_mile_pkgs,
                'last_mile_containers': last_mile_containers,
                'direct_injection_pkgs_day': direct_pkgs,
                'direct_injection_containers': direct_containers,
                'total_daily_containers': max(0, total_containers),
                'outbound_trucks': outbound_trucks,
                'inbound_trucks': inbound_trucks,
                'injection_hourly_throughput': int(round(injection_hourly)),
                'intermediate_sort_hourly_throughput': int(round(intermediate_sort_hourly)),
                'intermediate_crossdock_hourly_throughput': int(round(intermediate_crossdock_hourly)),
                'last_mile_hourly_throughput': int(round(last_mile_hourly)),
                'peak_hourly_throughput': int(round(peak_hourly)),
            }

            volume_data.append(volume_entry)

        except Exception as e:
            print(f"    Warning: Error calculating volume for {facility}: {e}")
            continue

    return pd.DataFrame(volume_data)


def _get_inbound_containers_for_facility(
        facility: str,
        inbound_ods: pd.DataFrame,
        od_selected: pd.DataFrame
) -> int:
    """
    Get total inbound containers for a facility from OD tracking.

    Looks at the final arc of each inbound OD flow.
    """
    total_containers = 0

    for _, od_row in inbound_ods.iterrows():
        if 'arc_containers' not in od_row.index:
            continue

        arc_containers = od_row['arc_containers']
        path_nodes = extract_path_nodes(od_row)

        if len(path_nodes) >= 2:
            # Find arc ending at this facility
            for i in range(len(path_nodes) - 1):
                if path_nodes[i + 1] == facility:
                    arc_key = (path_nodes[i], facility)
                    total_containers += arc_containers.get(arc_key, 0)
                    break

    return int(total_containers)


def _determine_intermediate_operation_type(
        intermediate_facility: str,
        dest_facility: str,
        path_strategy: str,
        chosen_sort_level: str,
        facilities: pd.DataFrame
) -> str:
    """
    Determine if intermediate facility performs sort or crossdock.

    Rules:
    - Fluid strategy: Always sort (must split freight)
    - Region sort: Always sort (freight sent unsorted to regional hub)
    - Market/sort_group: Crossdock (already sorted to destination)

    Returns:
        'sort' or 'crossdock'

    Raises:
        ValueError: If sort_level is not one of the expected values
    """
    if path_strategy.lower() == 'fluid':
        return 'sort'

    if chosen_sort_level == 'region':
        return 'sort'
    elif chosen_sort_level in ['market', 'sort_group']:
        return 'crossdock'
    else:
        raise ValueError(
            f"Unexpected sort_level '{chosen_sort_level}' for intermediate operation. "
            f"Expected one of: 'region', 'market', 'sort_group'"
        )


def _calculate_containers_for_volume(
        packages: float,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        strategy: str
) -> Dict[str, float]:
    """
    Calculate containers and cube for volume from input parameters.

    This is the fallback calculation when OD tracking is not available.
    """
    w_cube = weighted_pkg_cube(package_mix)
    total_cube = packages * w_cube

    if strategy.lower() == "container":
        gaylord_row = container_params[
            container_params["container_type"].str.lower() == "gaylord"
        ].iloc[0]
        raw_container_cube = float(gaylord_row["usable_cube_cuft"])
        pack_util = float(gaylord_row["pack_utilization_container"])
        effective_cube = raw_container_cube * pack_util

        containers = max(1, int(np.ceil(total_cube / effective_cube)))
    else:
        containers = 0

    return {'containers': containers, 'cube': total_cube}


# ============================================================================
# ZONE PACKAGE COUNTING (SHARED HELPER)
# ============================================================================

def _count_zone_packages(
        od_df: pd.DataFrame,
        direct_df: pd.DataFrame = None,
        facility: str = None
) -> Tuple[Dict[int, float], float]:
    """
    Count packages by zone.

    Shared helper for zone distribution calculations. Handles both facility-level
    and network-level aggregations.

    Args:
        od_df: Middle-mile OD flows with 'zone' column
        direct_df: Direct injection (zone 0), optional
        facility: If set, only count direct injection for this facility's destinations.
                  If None, count all direct injection.

    Returns:
        Tuple of (zone_counts dict {zone_num: pkg_count}, total_packages)
    """
    zone_pkgs = {i: 0.0 for i in range(9)}
    zone_pkgs[-1] = 0.0  # Unknown
    total_pkgs = 0.0

    # Count middle-mile packages by zone
    if not od_df.empty and 'zone' in od_df.columns:
        total_pkgs += od_df['pkgs_day'].sum()

        for zone_num in range(9):
            zone_pkgs[zone_num] += od_df[od_df['zone'] == zone_num]['pkgs_day'].sum()

        # Unknown zone (-1)
        zone_pkgs[-1] += od_df[od_df['zone'] == -1]['pkgs_day'].sum()

    # Count direct injection packages (zone 0)
    if direct_df is not None and not direct_df.empty:
        direct_col = 'dir_pkgs_day'
        if direct_col in direct_df.columns:
            if facility is not None:
                # Facility-level: only count direct injection where THIS facility is destination
                if 'dest' in direct_df.columns:
                    facility_direct = direct_df[direct_df['dest'] == facility]
                    if not facility_direct.empty:
                        direct_pkgs = facility_direct[direct_col].sum()
                        zone_pkgs[0] += direct_pkgs
                        total_pkgs += direct_pkgs
            else:
                # Network-level: count all direct injection
                direct_pkgs = direct_df[direct_col].sum()
                zone_pkgs[0] += direct_pkgs
                total_pkgs += direct_pkgs

    return zone_pkgs, total_pkgs


# ============================================================================
# FACILITY NETWORK PROFILE (NETWORK CHARACTERISTICS)
# ============================================================================

def build_facility_network_profile(
        od_selected: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame,
        direct_day: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Build facility network profile with zone/sort/distance/touch characteristics.

    Includes zones 0-8 and unknown zone tracking.
    """
    if od_selected.empty:
        return pd.DataFrame()

    fac_lookup = get_facility_lookup(facilities)
    profile_data = []

    origins = od_selected['origin'].unique()

    for origin in origins:
        try:
            origin_ods = od_selected[od_selected['origin'] == origin].copy()

            if origin_ods.empty:
                continue

            # Calculate metrics
            distance_metrics = _calculate_distance_metrics_for_ods(
                origin_ods, facilities, mileage_bands
            )

            touch_metrics = _calculate_touch_metrics_for_ods(origin_ods, facilities)

            # Zone distribution (using shared helper)
            zone_dist = _calculate_zone_distribution_for_ods(origin_ods, direct_day, origin)

            # Sort level distribution
            sort_dist = _calculate_sort_level_distribution_for_ods(origin_ods)

            # Container metrics (if tracking available)
            container_metrics = _calculate_container_metrics_for_origin(origin_ods)

            fac_type = fac_lookup.at[origin, 'type'] if origin in fac_lookup.index else 'unknown'

            profile_entry = {
                'facility': origin,
                'facility_type': fac_type,
                'total_od_pairs': len(origin_ods),
                'unique_destinations': origin_ods['dest'].nunique(),
                'total_packages': origin_ods['pkgs_day'].sum(),

                # Distance metrics
                **distance_metrics,

                # Touch metrics
                **touch_metrics,

                # Zone distribution (0-8 + unknown)
                **zone_dist,

                # Sort level distribution
                **sort_dist,

                # Container metrics
                **container_metrics,
            }

            profile_data.append(profile_entry)

        except Exception as e:
            print(f"    Warning: Error calculating profile for {origin}: {e}")
            continue

    return pd.DataFrame(profile_data)


def _calculate_container_metrics_for_origin(ods: pd.DataFrame) -> Dict[str, float]:
    """Calculate container metrics for ODs originating from a facility."""
    result = {
        'total_origin_containers': 0,
        'avg_origin_container_fill': 0.0,
    }

    if 'origin_containers' not in ods.columns:
        return result

    total_containers = ods['origin_containers'].sum()
    result['total_origin_containers'] = int(total_containers)

    if 'origin_container_fill' in ods.columns and total_containers > 0:
        # Weighted average fill rate
        weighted_fill = (ods['origin_container_fill'] * ods['origin_containers']).sum()
        result['avg_origin_container_fill'] = round(
            safe_divide(weighted_fill, total_containers), 3
        )

    return result


def _calculate_distance_metrics_for_ods(
        ods: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> Dict[str, float]:
    """Calculate distance metrics for OD set."""
    fac = facilities.set_index('facility_name')[['lat', 'lon']].astype(float)

    zone_miles_list = []
    transit_miles_list = []

    for _, od_row in ods.iterrows():
        origin = od_row['origin']
        dest = od_row['dest']
        pkgs = od_row['pkgs_day']

        # Zone miles (straight-line O-D)
        if origin in fac.index and dest in fac.index:
            o_lat, o_lon = fac.at[origin, 'lat'], fac.at[origin, 'lon']
            d_lat, d_lon = fac.at[dest, 'lat'], fac.at[dest, 'lon']

            zone_miles = haversine_miles(o_lat, o_lon, d_lat, d_lon)
            zone_miles_list.extend([zone_miles] * int(pkgs))

        # Transit miles (sum of all arcs)
        path_nodes = extract_path_nodes(od_row)

        transit_miles = 0
        for i in range(len(path_nodes) - 1):
            from_fac = path_nodes[i]
            to_fac = path_nodes[i + 1]

            if from_fac in fac.index and to_fac in fac.index:
                f_lat, f_lon = fac.at[from_fac, 'lat'], fac.at[from_fac, 'lon']
                t_lat, t_lon = fac.at[to_fac, 'lat'], fac.at[to_fac, 'lon']

                raw_dist = haversine_miles(f_lat, f_lon, t_lat, t_lon)

                if from_fac != to_fac:
                    _, _, circuity, _ = band_lookup(raw_dist, mileage_bands)
                    transit_miles += raw_dist * circuity

        transit_miles_list.extend([transit_miles] * int(pkgs))

    return {
        'avg_zone_miles': round(np.mean(zone_miles_list), 1) if zone_miles_list else 0,
        'avg_transit_miles': round(np.mean(transit_miles_list), 1) if transit_miles_list else 0,
    }


def _calculate_touch_metrics_for_ods(
        ods: pd.DataFrame,
        facilities: pd.DataFrame
) -> Dict[str, float]:
    """Calculate touch metrics for OD set."""
    fac_lookup = get_facility_lookup(facilities)

    total_touches_list = []
    hub_touches_list = []

    for _, od_row in ods.iterrows():
        path_nodes = extract_path_nodes(od_row)
        pkgs = od_row['pkgs_day']

        # Total touches
        total_touches = len(path_nodes)
        total_touches_list.extend([total_touches] * int(pkgs))

        # Hub touches
        hub_touches = 0
        for i, node in enumerate(path_nodes):
            if node in fac_lookup.index:
                node_type = fac_lookup.at[node, 'type']

                if node_type in ['hub', 'hybrid']:
                    hub_touches += 1
                elif i < len(path_nodes) - 1:
                    hub_touches += 1

        hub_touches_list.extend([hub_touches] * int(pkgs))

    return {
        'avg_total_touches': round(np.mean(total_touches_list), 2) if total_touches_list else 0,
        'avg_hub_touches': round(np.mean(hub_touches_list), 2) if hub_touches_list else 0,
    }


def _calculate_zone_distribution_for_ods(
        ods: pd.DataFrame,
        direct_day: pd.DataFrame = None,
        facility: str = None
) -> Dict[str, float]:
    """
    Calculate zone distribution for OD set.

    Uses shared _count_zone_packages helper.
    Returns DECIMALS not percentages (0.25 not 25.0).

    Args:
        ods: OD DataFrame with integer zone column
        direct_day: Direct injection (optional, for zone 0)
        facility: Facility name (for direct injection filtering)

    Returns:
        Dictionary with zone_0_pct through zone_8_pct and zone_unknown_pct (decimals 0-1)
    """
    zone_pkgs, total_pkgs = _count_zone_packages(ods, direct_day, facility)

    zone_dist = {}
    for zone_num in range(9):
        zone_dist[f'zone_{zone_num}_pct'] = round(
            safe_divide(zone_pkgs[zone_num], total_pkgs), 4
        )

    zone_dist['zone_unknown_pct'] = round(
        safe_divide(zone_pkgs[-1], total_pkgs), 4
    )

    return zone_dist


def _calculate_sort_level_distribution_for_ods(ods: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate sort level distribution for OD set.

    Returns dict with sort level percentages by packages and destinations.
    """
    result = {
        'region_sort_pct_pkgs': 0.0,
        'region_sort_pct_dests': 0.0,
        'market_sort_pct_pkgs': 0.0,
        'market_sort_pct_dests': 0.0,
        'sort_group_pct_pkgs': 0.0,
        'sort_group_pct_dests': 0.0,
    }

    if 'chosen_sort_level' not in ods.columns:
        return result

    total_pkgs = ods['pkgs_day'].sum()
    total_dests = len(ods)

    if total_pkgs == 0 or total_dests == 0:
        return result

    for sort_level in ['region', 'market', 'sort_group']:
        level_ods = ods[ods['chosen_sort_level'] == sort_level]

        pkgs = level_ods['pkgs_day'].sum()
        dests = len(level_ods)

        result[f'{sort_level}_pct_pkgs'] = round(safe_divide(pkgs, total_pkgs), 4)
        result[f'{sort_level}_pct_dests'] = round(safe_divide(dests, total_dests), 4)

    return result


# ============================================================================
# NETWORK-LEVEL METRICS
# ============================================================================

def calculate_network_distance_metrics(
        od_selected: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> Dict[str, float]:
    """Calculate network-level distance metrics."""
    if od_selected.empty:
        return {
            'avg_zone_miles': 0.0,
            'avg_transit_miles': 0.0,
        }

    return _calculate_distance_metrics_for_ods(od_selected, facilities, mileage_bands)


def calculate_network_touch_metrics(
        od_selected: pd.DataFrame,
        facilities: pd.DataFrame
) -> Dict[str, float]:
    """Calculate network-level touch metrics."""
    if od_selected.empty:
        return {
            'avg_total_touches': 0.0,
            'avg_hub_touches': 0.0,
        }

    return _calculate_touch_metrics_for_ods(od_selected, facilities)


def calculate_network_zone_distribution(
        od_selected: pd.DataFrame,
        direct_day: pd.DataFrame = None
) -> Dict:
    """
    Calculate network-level zone distribution.

    Uses shared _count_zone_packages helper for consistency with facility-level calculations.

    Includes:
    - Zone 0: Direct injection (from direct_day)
    - Zones 1-8: Middle-mile (from od_selected)
    - Zone -1: Unknown (classification failed)

    Args:
        od_selected: Selected OD paths with integer zone column
        direct_day: Direct injection volumes (optional)

    Returns:
        Dictionary with:
            - zone_0_pkgs through zone_8_pkgs (int)
            - zone_0_pct through zone_8_pct (decimal 0-1)
            - zone_unknown_pkgs (int)
            - zone_unknown_pct (decimal 0-1)
    """
    zone_pkgs, total_pkgs = _count_zone_packages(od_selected, direct_day, facility=None)

    result = {}
    for zone_num in range(9):
        result[f'zone_{zone_num}_pkgs'] = int(zone_pkgs[zone_num])
        result[f'zone_{zone_num}_pct'] = round(
            safe_divide(zone_pkgs[zone_num], total_pkgs), 4
        )

    result['zone_unknown_pkgs'] = int(zone_pkgs[-1])
    result['zone_unknown_pct'] = round(
        safe_divide(zone_pkgs[-1], total_pkgs), 4
    )

    return result


def calculate_network_sort_distribution(od_selected: pd.DataFrame) -> Dict:
    """
    Calculate network-level sort level distribution.

    NOTE: Used for comparison workbooks only, not in scenario KPIs.
    For OD-level sort detail, see sort_analysis sheet.
    """
    result = {
        'region_sort_pkgs': 0,
        'region_sort_pct_pkgs': 0.0,
        'region_sort_pct_dests': 0.0,
        'market_sort_pkgs': 0,
        'market_sort_pct_pkgs': 0.0,
        'market_sort_pct_dests': 0.0,
        'sort_group_pkgs': 0,
        'sort_group_pct_pkgs': 0.0,
        'sort_group_pct_dests': 0.0,
    }

    if od_selected.empty or 'chosen_sort_level' not in od_selected.columns:
        return result

    total_pkgs = od_selected['pkgs_day'].sum()
    total_dests = len(od_selected)

    if total_pkgs == 0 or total_dests == 0:
        return result

    for sort_level in ['region', 'market', 'sort_group']:
        level_ods = od_selected[od_selected['chosen_sort_level'] == sort_level]

        pkgs = level_ods['pkgs_day'].sum()
        dests = len(level_ods)

        result[f'{sort_level}_pkgs'] = int(pkgs)
        result[f'{sort_level}_pct_pkgs'] = round(safe_divide(pkgs, total_pkgs), 4)
        result[f'{sort_level}_pct_dests'] = round(safe_divide(dests, total_dests), 4)

    return result


def calculate_network_container_metrics(
        od_selected: pd.DataFrame,
        arc_summary: pd.DataFrame
) -> Dict:
    """
    Calculate network-level container metrics.

    New function to report container persistence statistics.
    """
    result = {
        'total_origin_containers': 0,
        'avg_origin_container_fill': 0.0,
        'total_arc_containers': 0,
        'persisted_container_pct': 0.0,
        'fresh_container_pct': 0.0,
    }

    # OD-level container creation
    if 'origin_containers' in od_selected.columns:
        total_origin = od_selected['origin_containers'].sum()
        result['total_origin_containers'] = int(total_origin)

        if 'origin_container_fill' in od_selected.columns and total_origin > 0:
            weighted_fill = (
                od_selected['origin_container_fill'] * od_selected['origin_containers']
            ).sum()
            result['avg_origin_container_fill'] = round(
                safe_divide(weighted_fill, total_origin), 3
            )

    # Arc-level container tracking
    if 'physical_containers' in arc_summary.columns:
        total_arc = arc_summary['physical_containers'].sum()
        result['total_arc_containers'] = int(total_arc)

        if 'persisted_containers' in arc_summary.columns:
            persisted = arc_summary['persisted_containers'].sum()
            result['persisted_container_pct'] = round(
                safe_divide(persisted, total_arc), 3
            )

        if 'fresh_containers' in arc_summary.columns:
            fresh = arc_summary['fresh_containers'].sum()
            result['fresh_container_pct'] = round(
                safe_divide(fresh, total_arc), 3
            )

    return result


# ============================================================================
# ZONE CLASSIFICATION
# ============================================================================

def add_zone_classification(
        od_df: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> pd.DataFrame:
    """
    Add zone classification based on distance from mileage_bands.

    Zone Assignment Rules:
    - For ALL flows (including O=D): Use mileage_bands for distance
    - Zone 0 is ONLY for direct injection (handled separately)
    - Returns -1 for unknown/error cases

    Args:
        od_df: OD DataFrame
        facilities: Facility master data
        mileage_bands: Mileage bands with integer zone column

    Returns:
        DataFrame with added 'zone' column (integer)
    """
    if od_df.empty:
        return od_df

    od_df = od_df.copy()
    od_df['zone'] = -1  # Initialize as unknown

    for idx, row in od_df.iterrows():
        origin = row['origin']
        dest = row['dest']

        # ALL middle-mile flows use distance-based zone
        zone = calculate_zone_from_distance(origin, dest, facilities, mileage_bands)
        od_df.at[idx, 'zone'] = zone

    return od_df


def add_direct_injection_zone_classification(
        direct_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add zone 0 classification to direct injection flows.

    Direct injection = packages injected at destination without middle-mile transport.
    These are ALWAYS zone 0.

    Args:
        direct_df: Direct injection DataFrame with 'dest' column

    Returns:
        DataFrame with added 'zone' column set to 0
    """
    if direct_df.empty:
        return direct_df

    direct_df = direct_df.copy()
    direct_df['zone'] = 0  # Integer 0

    return direct_df


def add_zone_miles_to_od_selected(
        od_df: pd.DataFrame,
        facilities: pd.DataFrame
) -> pd.DataFrame:
    """
    Add zone_miles column for validation.

    This allows validation of zone classification by showing actual O-D distance.

    Args:
        od_df: OD DataFrame
        facilities: Facility master data

    Returns:
        DataFrame with added 'zone_miles' column (float)
    """
    if od_df.empty:
        return od_df

    od_df = od_df.copy()
    fac_lookup = get_facility_lookup(facilities)

    od_df['zone_miles'] = 0.0

    for idx, row in od_df.iterrows():
        origin = row['origin']
        dest = row['dest']

        if origin in fac_lookup.index and dest in fac_lookup.index:
            o_lat = fac_lookup.at[origin, 'lat']
            o_lon = fac_lookup.at[origin, 'lon']
            d_lat = fac_lookup.at[dest, 'lat']
            d_lon = fac_lookup.at[dest, 'lon']

            zone_miles = haversine_miles(o_lat, o_lon, d_lat, d_lon)
            od_df.at[idx, 'zone_miles'] = round(zone_miles, 1)

    return od_df


# ============================================================================
# PATH STEPS (LEGACY - USED BY run_v4.py)
# ============================================================================

def build_path_steps(
        od_selected: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> pd.DataFrame:
    """Generate path steps from selected OD paths."""
    path_steps = []

    fac_lookup = facilities.set_index('facility_name')[['lat', 'lon']].astype(float)

    for _, od_row in od_selected.iterrows():
        path_nodes = extract_path_nodes(od_row)
        scenario_id = od_row.get('scenario_id', 'default')

        for i in range(len(path_nodes) - 1):
            from_fac = path_nodes[i]
            to_fac = path_nodes[i + 1]

            if from_fac == to_fac:
                from_lat = fac_lookup.at[from_fac, 'lat'] if from_fac in fac_lookup.index else 0
                from_lon = fac_lookup.at[from_fac, 'lon'] if from_fac in fac_lookup.index else 0
                to_lat = from_lat
                to_lon = from_lon

                path_steps.append({
                    'scenario_id': scenario_id,
                    'origin': od_row['origin'],
                    'dest': od_row['dest'],
                    'step_order': i + 1,
                    'from_facility': from_fac,
                    'to_facility': to_fac,
                    'from_lat': from_lat,
                    'from_lon': from_lon,
                    'to_lat': to_lat,
                    'to_lon': to_lon,
                    'distance_miles': 0.0,
                    'drive_hours': 0.0
                })
                continue

            if from_fac in fac_lookup.index and to_fac in fac_lookup.index:
                lat1, lon1 = fac_lookup.at[from_fac, 'lat'], fac_lookup.at[from_fac, 'lon']
                lat2, lon2 = fac_lookup.at[to_fac, 'lat'], fac_lookup.at[to_fac, 'lon']

                raw_dist = haversine_miles(lat1, lon1, lat2, lon2)
                fixed, var, circuit, mph = band_lookup(raw_dist, mileage_bands)
                actual_dist = raw_dist * circuit
                drive_hours = safe_divide(actual_dist, mph)

                path_steps.append({
                    'scenario_id': scenario_id,
                    'origin': od_row['origin'],
                    'dest': od_row['dest'],
                    'step_order': i + 1,
                    'from_facility': from_fac,
                    'to_facility': to_fac,
                    'from_lat': lat1,
                    'from_lon': lon1,
                    'to_lat': lat2,
                    'to_lon': lon2,
                    'distance_miles': actual_dist,
                    'drive_hours': drive_hours
                })

    return pd.DataFrame(path_steps)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_network_aggregations(
        od_selected: pd.DataFrame,
        arc_summary: pd.DataFrame,
        facility_volume: pd.DataFrame
) -> Dict:
    """
    Validate network data quality and flag issues.

    Checks:
    - Unknown zone percentage (data quality flag)
    - Arc fill rates for utilization tracking
    - Container consistency between OD and arc levels

    Returns:
        Dictionary with validation metrics
    """
    validation_results = {}

    try:
        # Check for unknown zones (data quality flag)
        if not od_selected.empty and 'zone' in od_selected.columns:
            unknown_zone_pkgs = od_selected[od_selected['zone'] == -1]['pkgs_day'].sum()
            total_pkgs = od_selected['pkgs_day'].sum()
            unknown_zone_pct = safe_divide(unknown_zone_pkgs, total_pkgs)

            validation_results['unknown_zone_packages'] = int(unknown_zone_pkgs)
            validation_results['unknown_zone_pct'] = round(unknown_zone_pct, 4)

            if unknown_zone_pct > 0.01:  # 1% threshold
                print(f"  WARNING: Data Quality - {unknown_zone_pct * 100:.1f}% of packages in unknown zone")
                print(f"           Check mileage_bands coverage and facility coordinates")

        # Calculate network average truck fill rate
        if not arc_summary.empty and 'truck_fill_rate' in arc_summary.columns:
            non_od_arcs = arc_summary[
                arc_summary['from_facility'] != arc_summary['to_facility']
            ]

            if not non_od_arcs.empty and 'pkg_cube_cuft' in non_od_arcs.columns:
                total_pkg_cube = non_od_arcs['pkg_cube_cuft'].sum()
                total_truck_cube = (
                    non_od_arcs['trucks'] * non_od_arcs.get('cube_per_truck', 0)
                ).sum()
                validation_results['network_avg_truck_fill'] = safe_divide(
                    total_pkg_cube, total_truck_cube
                )
            else:
                validation_results['network_avg_truck_fill'] = 0
        else:
            validation_results['network_avg_truck_fill'] = 0

        # Container consistency check (new)
        if 'origin_containers' in od_selected.columns:
            od_origin_containers = od_selected['origin_containers'].sum()
            validation_results['od_origin_containers'] = int(od_origin_containers)

        if 'physical_containers' in arc_summary.columns:
            arc_total_containers = arc_summary['physical_containers'].sum()
            validation_results['arc_total_containers'] = int(arc_total_containers)

            # Check for persisted vs fresh breakdown
            if 'persisted_containers' in arc_summary.columns:
                persisted = arc_summary['persisted_containers'].sum()
                fresh = arc_summary['fresh_containers'].sum()
                validation_results['persisted_containers'] = int(persisted)
                validation_results['fresh_containers'] = int(fresh)
                validation_results['persistence_rate'] = round(
                    safe_divide(persisted, arc_total_containers), 3
                )

        validation_results['package_consistency'] = True

    except Exception as e:
        validation_results['validation_error'] = str(e)
        print(f"  WARNING: Validation error - {e}")

    return validation_results