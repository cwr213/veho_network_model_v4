"""
Reporting Module

Generates facility-level aggregations, network metrics, and analytics outputs.

Key Outputs:
    - facility_volume: Daily operational volumes and throughput
    - facility_network_profile: Network characteristics per facility
    - Network-level KPIs with zone/sort/distance/touch distributions

Major Changes in v4:
    - Removed cost allocation columns (accuracy issues)
    - Split facility_rollup into volume and network profile
    - Added zone distribution (Zones 1-8)
    - Added sort level distribution (region/market/sort_group)
    - Added distance metrics (zone_miles, transit_miles)
    - Added touch metrics (total_touches, hub_touches)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

from .geo_v4 import calculate_zone_from_distance, haversine_miles, band_lookup
from .containers_v4 import weighted_pkg_cube
from .utils import safe_divide, get_facility_lookup
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

    NO COST COLUMNS - focus on volumes, containers, trucks, and throughput.

    Args:
        od_selected: Selected OD paths
        direct_day: Direct injection volumes
        arc_summary: Arc-level aggregations
        package_mix: Package distribution
        container_params: Container parameters
        strategy: Loading strategy
        facilities: Facility master data
        timing_params: Timing parameters dict

    Returns:
        DataFrame with operational metrics per facility
    """
    volume_data = []

    # Get all facilities
    all_facilities = set()
    if not od_selected.empty:
        all_facilities.update(od_selected['origin'].unique())
        all_facilities.update(od_selected['dest'].unique())
    if not direct_day.empty and 'dest' in direct_day.columns:
        all_facilities.update(direct_day['dest'].unique())

    fac_lookup = get_facility_lookup(facilities)

    for facility in all_facilities:
        try:
            # Direct injection volumes
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

            # Middle-mile injection volumes
            mm_injection_pkgs = 0
            mm_injection_containers = 0

            if not od_selected.empty:
                outbound = od_selected[od_selected['origin'] == facility]
                if not outbound.empty:
                    mm_injection_pkgs = outbound['pkgs_day'].sum()

                    if mm_injection_pkgs > 0:
                        containers = _calculate_containers_for_volume(
                            mm_injection_pkgs, package_mix, container_params, strategy
                        )
                        mm_injection_containers = containers['containers']

            # Intermediate facility operations (sort vs crossdock)
            intermediate_pkgs_sort = 0
            intermediate_pkgs_crossdock = 0
            intermediate_containers_sort = 0
            intermediate_containers_crossdock = 0

            if not od_selected.empty:
                for _, path_row in od_selected.iterrows():
                    path_nodes = path_row.get('path_nodes', [])
                    if not isinstance(path_nodes, list):
                        continue

                    final_dest = path_row['dest']
                    path_strategy = path_row.get('effective_strategy', strategy)
                    chosen_sort_level = path_row.get('chosen_sort_level', 'market')

                    for node in path_nodes[1:-1]:
                        if node == facility:
                            operation = _determine_intermediate_operation_type(
                                intermediate_facility=facility,
                                dest_facility=final_dest,
                                path_strategy=path_strategy,
                                chosen_sort_level=chosen_sort_level,
                                facilities=facilities
                            )

                            pkgs = path_row['pkgs_day']

                            if operation == 'sort':
                                intermediate_pkgs_sort += pkgs
                            else:
                                intermediate_pkgs_crossdock += pkgs

                            containers = _calculate_containers_for_volume(
                                pkgs, package_mix, container_params, path_strategy
                            )

                            if operation == 'sort':
                                intermediate_containers_sort += containers['containers']
                            else:
                                intermediate_containers_crossdock += containers['containers']

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

                    if mm_last_mile > 0:
                        containers = _calculate_containers_for_volume(
                            mm_last_mile, package_mix, container_params, strategy
                        )
                        last_mile_containers += containers['containers']

            # Total containers (avoiding double-count for O=D)
            total_containers = mm_injection_containers + intermediate_containers + last_mile_containers

            if not od_selected.empty:
                od_paths = od_selected[od_selected['origin'] == facility]
                for _, od_row in od_paths.iterrows():
                    if od_row['origin'] == od_row['dest']:
                        od_containers = _calculate_containers_for_volume(
                            od_row['pkgs_day'], package_mix, container_params, strategy
                        )['containers']
                        total_containers -= od_containers

            # Truck movements
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

            # Hourly throughput calculations
            injection_va_hours = float(timing_params['injection_va_hours'])
            middle_mile_va_hours = float(timing_params['middle_mile_va_hours'])
            crossdock_va_hours = float(timing_params.get('crossdock_va_hours', 3.0))
            last_mile_va_hours = float(timing_params['last_mile_va_hours'])

            injection_hourly = safe_divide(mm_injection_pkgs, injection_va_hours)
            intermediate_sort_hourly = safe_divide(intermediate_pkgs_sort, middle_mile_va_hours)
            intermediate_crossdock_hourly = safe_divide(intermediate_pkgs_crossdock, crossdock_va_hours)
            last_mile_hourly = safe_divide(last_mile_pkgs, last_mile_va_hours)

            peak_hourly = max(
                injection_hourly,
                intermediate_sort_hourly,
                intermediate_crossdock_hourly,
                last_mile_hourly
            )

            # Get facility type
            fac_type = fac_lookup.at[facility, 'type'] if facility in fac_lookup.index else 'unknown'

            volume_entry = {
                'facility': facility,
                'facility_type': fac_type,

                # Injection volumes
                'injection_pkgs_day': mm_injection_pkgs,
                'injection_containers': mm_injection_containers,

                # Intermediate volumes (split by operation type)
                'intermediate_sort_pkgs_day': intermediate_pkgs_sort,
                'intermediate_crossdock_pkgs_day': intermediate_pkgs_crossdock,
                'intermediate_pkgs_day': intermediate_pkgs,
                'intermediate_sort_containers': intermediate_containers_sort,
                'intermediate_crossdock_containers': intermediate_containers_crossdock,
                'intermediate_containers': intermediate_containers,

                # Last mile volumes
                'last_mile_pkgs_day': last_mile_pkgs,
                'last_mile_containers': last_mile_containers,

                # Direct injection
                'direct_injection_pkgs_day': direct_pkgs,
                'direct_injection_containers': direct_containers,

                # Total containers
                'total_daily_containers': max(0, total_containers),

                # Truck movements
                'outbound_trucks': outbound_trucks,
                'inbound_trucks': inbound_trucks,

                # Hourly throughput
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


def _determine_intermediate_operation_type(
        intermediate_facility: str,
        dest_facility: str,
        path_strategy: str,
        chosen_sort_level: str,
        facilities: pd.DataFrame
) -> str:
    """Determine if intermediate facility needs sort or crossdock."""
    if path_strategy.lower() == 'fluid':
        return 'sort'

    if path_strategy.lower() == 'container':
        if chosen_sort_level == 'region':
            fac_lookup = get_facility_lookup(facilities)

            if dest_facility in fac_lookup.index:
                dest_hub = fac_lookup.at[dest_facility, 'regional_sort_hub']
                if pd.isna(dest_hub) or dest_hub == '':
                    dest_hub = dest_facility

                if intermediate_facility == dest_hub:
                    return 'sort'
                else:
                    return 'crossdock'

        elif chosen_sort_level in ['market', 'sort_group']:
            return 'crossdock'

    return 'sort'


def _calculate_containers_for_volume(
        packages: float,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        strategy: str
) -> Dict[str, float]:
    """Calculate containers and cube for volume."""
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
# FACILITY NETWORK PROFILE (NETWORK CHARACTERISTICS)
# ============================================================================

def build_facility_network_profile(
        od_selected: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> pd.DataFrame:
    """
    Build facility network profile with zone/sort/distance/touch characteristics.

    For each facility, calculate network metrics for OD pairs originating there:
    - Zone distribution
    - Sort level distribution (if enabled)
    - Distance metrics (zone_miles, transit_miles)
    - Touch metrics (total_touches, hub_touches)

    Args:
        od_selected: Selected OD paths with chosen routes
        facilities: Facility master data
        mileage_bands: Mileage bands for distance calculations

    Returns:
        DataFrame with network profile per facility
    """
    if od_selected.empty:
        return pd.DataFrame()

    fac_lookup = get_facility_lookup(facilities)
    profile_data = []

    # Get unique origin facilities
    origins = od_selected['origin'].unique()

    for origin in origins:
        try:
            # Filter to ODs originating from this facility
            origin_ods = od_selected[od_selected['origin'] == origin].copy()

            if origin_ods.empty:
                continue

            # Calculate distance metrics
            distance_metrics = _calculate_distance_metrics_for_ods(
                origin_ods, facilities, mileage_bands
            )

            # Calculate touch metrics
            touch_metrics = _calculate_touch_metrics_for_ods(origin_ods, facilities)

            # Calculate zone distribution
            zone_dist = _calculate_zone_distribution_for_ods(origin_ods)

            # Calculate sort level distribution (if sort level data exists)
            sort_dist = _calculate_sort_level_distribution_for_ods(origin_ods)

            # Get facility type
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

                # Zone distribution
                **zone_dist,

                # Sort level distribution
                **sort_dist,
            }

            profile_data.append(profile_entry)

        except Exception as e:
            print(f"    Warning: Error calculating profile for {origin}: {e}")
            continue

    return pd.DataFrame(profile_data)


def _calculate_distance_metrics_for_ods(
        ods: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate distance metrics for a set of OD pairs.

    Returns:
        - avg_zone_miles: Average straight-line O-D distance
        - avg_transit_miles: Average actual path miles (with circuity)
    """
    fac = facilities.set_index('facility_name')[['lat', 'lon']].astype(float)

    zone_miles_list = []
    transit_miles_list = []

    for _, od_row in ods.iterrows():
        origin = od_row['origin']
        dest = od_row['dest']
        pkgs = od_row['pkgs_day']

        # Calculate zone miles (straight-line O-D)
        if origin in fac.index and dest in fac.index:
            o_lat, o_lon = fac.at[origin, 'lat'], fac.at[origin, 'lon']
            d_lat, d_lon = fac.at[dest, 'lat'], fac.at[dest, 'lon']

            zone_miles = haversine_miles(o_lat, o_lon, d_lat, d_lon)
            zone_miles_list.extend([zone_miles] * int(pkgs))

        # Calculate transit miles (sum of all arcs)
        path_nodes = od_row.get('path_nodes', [origin, dest])
        if not isinstance(path_nodes, list):
            path_nodes = [origin, dest]

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
        'avg_zone_miles': np.mean(zone_miles_list) if zone_miles_list else 0,
        'avg_transit_miles': np.mean(transit_miles_list) if transit_miles_list else 0,
    }


def _calculate_touch_metrics_for_ods(
        ods: pd.DataFrame,
        facilities: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate touch metrics for a set of OD pairs.

    Returns:
        - avg_total_touches: Average facilities touched (origin + intermediates + dest)
        - avg_hub_touches: Average hub/hybrid facilities touched (excludes launch dest)
    """
    fac_lookup = get_facility_lookup(facilities)

    total_touches_list = []
    hub_touches_list = []

    for _, od_row in ods.iterrows():
        path_nodes = od_row.get('path_nodes', [od_row['origin'], od_row['dest']])
        if not isinstance(path_nodes, list):
            path_nodes = [od_row['origin'], od_row['dest']]

        pkgs = od_row['pkgs_day']

        # Total touches = all facilities in path
        total_touches = len(path_nodes)
        total_touches_list.extend([total_touches] * int(pkgs))

        # Hub touches = hubs/hybrids in path (exclude launch destination)
        hub_touches = 0
        for i, node in enumerate(path_nodes):
            if node in fac_lookup.index:
                node_type = fac_lookup.at[node, 'type']

                # Count hub/hybrid, or last facility if it's not launch
                if node_type in ['hub', 'hybrid']:
                    hub_touches += 1
                elif i < len(path_nodes) - 1:  # Not the destination
                    hub_touches += 1

        hub_touches_list.extend([hub_touches] * int(pkgs))

    return {
        'avg_total_touches': np.mean(total_touches_list) if total_touches_list else 0,
        'avg_hub_touches': np.mean(hub_touches_list) if hub_touches_list else 0,
    }


def _calculate_zone_distribution_for_ods(ods: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate zone distribution for a set of OD pairs.

    Returns dict with zone_1_pct through zone_8_pct.
    """
    if 'zone' not in ods.columns:
        return {f'zone_{i}_pct': 0.0 for i in range(1, 9)}

    total_pkgs = ods['pkgs_day'].sum()

    if total_pkgs == 0:
        return {f'zone_{i}_pct': 0.0 for i in range(1, 9)}

    zone_dist = {}

    for zone_num in range(1, 9):
        zone_str = str(zone_num)
        zone_pkgs = ods[ods['zone'].astype(str) == zone_str]['pkgs_day'].sum()
        zone_dist[f'zone_{zone_num}_pct'] = safe_divide(zone_pkgs, total_pkgs) * 100

    return zone_dist


def _calculate_sort_level_distribution_for_ods(ods: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate sort level distribution for a set of OD pairs.

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

        result[f'{sort_level}_pct_pkgs'] = safe_divide(pkgs, total_pkgs) * 100
        result[f'{sort_level}_pct_dests'] = safe_divide(dests, total_dests) * 100

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


def calculate_network_zone_distribution(od_selected: pd.DataFrame) -> Dict:
    """
    Calculate network-level zone distribution.

    Returns dict with both package counts and percentages for zones 1-8.
    """
    result = {}

    if od_selected.empty or 'zone' not in od_selected.columns:
        for i in range(1, 9):
            result[f'zone_{i}_pkgs'] = 0
            result[f'zone_{i}_pct'] = 0.0
        return result

    total_pkgs = od_selected['pkgs_day'].sum()

    for zone_num in range(1, 9):
        zone_str = str(zone_num)
        zone_pkgs = od_selected[od_selected['zone'].astype(str) == zone_str]['pkgs_day'].sum()

        result[f'zone_{zone_num}_pkgs'] = zone_pkgs
        result[f'zone_{zone_num}_pct'] = safe_divide(zone_pkgs, total_pkgs) * 100

    return result


def calculate_network_sort_distribution(od_selected: pd.DataFrame) -> Dict:
    """
    Calculate network-level sort level distribution.

    Returns dict with packages, % packages, and % destinations for each sort level.
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

        result[f'{sort_level}_pkgs'] = pkgs
        result[f'{sort_level}_pct_pkgs'] = safe_divide(pkgs, total_pkgs) * 100
        result[f'{sort_level}_pct_dests'] = safe_divide(dests, total_dests) * 100

    return result


# ============================================================================
# ZONE CLASSIFICATION
# ============================================================================

def add_zone_classification(
        od_df: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> pd.DataFrame:
    """Add zone classification based on O-D straight-line distance."""
    if od_df.empty:
        return od_df

    od_df = od_df.copy()
    od_df['zone'] = 'unknown'

    for idx, row in od_df.iterrows():
        origin = row['origin']
        dest = row['dest']

        if origin == dest:
            od_df.at[idx, 'zone'] = '1'  # O=D is always local
        else:
            zone = calculate_zone_from_distance(origin, dest, facilities, mileage_bands)
            od_df.at[idx, 'zone'] = zone

    return od_df


# ============================================================================
# PATH STEPS (LEGACY - KEEP FOR COMPATIBILITY)
# ============================================================================

def build_path_steps(
        od_selected: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame,
        timing_params: Dict
) -> pd.DataFrame:
    """
    Generate path steps from selected OD paths.

    Legacy function - maintained for backward compatibility.
    """
    path_steps = []

    fac_lookup = facilities.set_index('facility_name')[['lat', 'lon']].astype(float)
    hours_per_touch = float(timing_params['hours_per_touch'])

    for _, od_row in od_selected.iterrows():
        path_str = str(od_row.get('path_str', f"{od_row['origin']}->{od_row['dest']}"))
        scenario_id = od_row.get('scenario_id', 'default')

        if '->' not in path_str:
            continue

        nodes = [n.strip() for n in path_str.split('->')]

        for i, (from_fac, to_fac) in enumerate(zip(nodes[:-1], nodes[1:])):
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
                    'drive_hours': 0.0,
                    'processing_hours_at_destination': hours_per_touch
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
                    'drive_hours': drive_hours,
                    'processing_hours_at_destination': hours_per_touch
                })

    return pd.DataFrame(path_steps)


# ============================================================================
# SORT SUMMARY (LEGACY - KEEP FOR COMPATIBILITY)
# ============================================================================

def build_sort_summary(
        selected_paths: pd.DataFrame,
        sort_decisions: Dict,
        facilities: pd.DataFrame
) -> pd.DataFrame:
    """
    Build sort decision summary with regional hub info.

    Legacy function - maintained for backward compatibility.
    """
    summary_data = []

    fac_lookup = get_facility_lookup(facilities)

    for _, od_row in selected_paths.iterrows():
        group_key = (od_row['scenario_id'], od_row['origin'],
                     od_row['dest'], od_row['day_type'])
        chosen_sort_level = sort_decisions.get(group_key, 'market')

        origin_region_hub = ''
        dest_region_hub = ''

        if od_row['origin'] in fac_lookup.index:
            origin_region_hub = fac_lookup.at[od_row['origin'], 'regional_sort_hub']
            if pd.isna(origin_region_hub):
                origin_region_hub = od_row['origin']

        if od_row['dest'] in fac_lookup.index:
            dest_region_hub = fac_lookup.at[od_row['dest'], 'regional_sort_hub']
            if pd.isna(dest_region_hub):
                dest_region_hub = od_row['dest']

        summary_data.append({
            'origin': od_row['origin'],
            'origin_region_hub': origin_region_hub,
            'dest': od_row['dest'],
            'dest_region_hub': dest_region_hub,
            'pkgs_day': od_row['pkgs_day'],
            'chosen_sort_level': chosen_sort_level,
            'total_cost': od_row['total_cost'],
            'processing_cost': od_row['processing_cost']
        })

    return pd.DataFrame(summary_data)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_network_aggregations(
        od_selected: pd.DataFrame,
        arc_summary: pd.DataFrame,
        facility_volume: pd.DataFrame
) -> Dict:
    """
    Validate aggregate calculations with sort/crossdock consistency checks.

    Args:
        od_selected: Selected OD paths
        arc_summary: Arc-level summary
        facility_volume: Facility volume rollup

    Returns:
        Dictionary with validation results
    """
    validation_results = {}

    try:
        total_od_pkgs = od_selected['pkgs_day'].sum() if not od_selected.empty else 0
        total_facility_mm_injection = facility_volume[
            'injection_pkgs_day'].sum() if not facility_volume.empty else 0

        validation_results['total_od_packages'] = total_od_pkgs
        validation_results['total_facility_mm_injection'] = total_facility_mm_injection
        validation_results['package_consistency'] = abs(total_od_pkgs - total_facility_mm_injection) < 0.01

        # Check intermediate consistency (sort + crossdock = total)
        if not facility_volume.empty:
            for _, row in facility_volume.iterrows():
                sort_pkgs = row.get('intermediate_sort_pkgs_day', 0)
                crossdock_pkgs = row.get('intermediate_crossdock_pkgs_day', 0)
                total_intermediate = row.get('intermediate_pkgs_day', 0)

                if abs((sort_pkgs + crossdock_pkgs) - total_intermediate) > 0.01:
                    facility_name = row.get('facility', 'unknown')
                    print(f"  ⚠️  Intermediate package mismatch at {facility_name}")
                    validation_results['intermediate_consistency'] = False
                    break
            else:
                validation_results['intermediate_consistency'] = True

        # Check arc fill rates
        if not arc_summary.empty and 'truck_fill_rate' in arc_summary.columns:
            non_od_arcs = arc_summary[arc_summary['from_facility'] != arc_summary['to_facility']]

            if not non_od_arcs.empty:
                total_pkg_cube = non_od_arcs['pkg_cube_cuft'].sum() if 'pkg_cube_cuft' in non_od_arcs.columns else 0
                total_truck_cube = (non_od_arcs['trucks'] * non_od_arcs.get('cube_per_truck', 0)).sum()

                validation_results['network_avg_truck_fill'] = safe_divide(total_pkg_cube, total_truck_cube)
            else:
                validation_results['network_avg_truck_fill'] = 0
        else:
            validation_results['network_avg_truck_fill'] = 0

    except Exception as e:
        validation_results['validation_error'] = str(e)

    return validation_results


# ============================================================================
# LEGACY FUNCTION (DEPRECATED BUT KEPT FOR COMPATIBILITY)
# ============================================================================

def build_facility_rollup(
        od_selected: pd.DataFrame,
        direct_day: pd.DataFrame,
        arc_summary: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        strategy: str,
        facilities: pd.DataFrame,
        cost_params
) -> pd.DataFrame:
    """
    DEPRECATED: Legacy function that included cost columns.

    Use build_facility_volume() instead for operational metrics only.

    This function is kept for backward compatibility with existing code
    but will be removed in future versions.
    """
    print("  ⚠️  Warning: build_facility_rollup is deprecated, use build_facility_volume instead")

    # Return facility volume (without cost columns)
    timing_params = {
        'injection_va_hours': 8,
        'middle_mile_va_hours': 16,
        'crossdock_va_hours': 3,
        'last_mile_va_hours': 4,
    }

    return build_facility_volume(
        od_selected,
        direct_day,
        arc_summary,
        package_mix,
        container_params,
        strategy,
        facilities,
        timing_params
    )


def calculate_hourly_throughput(
        facility_rollup: pd.DataFrame,
        timing_params: Dict
) -> pd.DataFrame:
    """
    DEPRECATED: Throughput now calculated in build_facility_volume().

    This function is a no-op for backward compatibility.
    """
    return facility_rollup