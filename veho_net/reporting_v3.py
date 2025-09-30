"""
Reporting Module

Generates facility-level aggregations, zone classifications, and network metrics.
Properly handles direct injection vs. middle-mile injection separation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from .geo_v3 import calculate_zone_from_distance
from .containers_v3 import weighted_pkg_cube


def build_facility_rollup(
        od_selected: pd.DataFrame,
        direct_day: pd.DataFrame,
        arc_summary: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        strategy: str
) -> pd.DataFrame:
    """
    Calculate facility volume and cost breakdowns by role.

    Separates:
    - Direct injection (Zone 0) - shipper direct to facility
    - Middle-mile injection (Zones 1-5) - origin of network flows
    - Intermediate (pass-through for crossdock/sort)
    - Last mile (direct + middle-mile arrivals for final delivery)

    Args:
        od_selected: Selected OD paths with costs
        direct_day: Direct injection volumes (Zone 0)
        arc_summary: Arc-level aggregations
        package_mix: Package distribution
        container_params: Container parameters
        strategy: Loading strategy

    Returns:
        DataFrame with facility-level metrics by role
    """
    volume_data = []

    # Collect all facilities
    all_facilities = set()
    if not od_selected.empty:
        all_facilities.update(od_selected['origin'].unique())
        all_facilities.update(od_selected['dest'].unique())
    if not direct_day.empty and 'dest' in direct_day.columns:
        all_facilities.update(direct_day['dest'].unique())

    for facility in all_facilities:
        try:
            # === DIRECT INJECTION (Zone 0) ===
            direct_injection_pkgs = 0
            direct_injection_containers = 0
            direct_injection_cube = 0.0

            if not direct_day.empty:
                direct_col = 'dir_pkgs_day'
                if direct_col in direct_day.columns:
                    facility_direct = direct_day[direct_day['dest'] == facility]
                    if not facility_direct.empty:
                        direct_injection_pkgs = facility_direct[direct_col].sum()

                        # Calculate containers for direct injection
                        if direct_injection_pkgs > 0:
                            containers_calc = _calculate_containers_for_volume(
                                direct_injection_pkgs, package_mix, container_params, strategy
                            )
                            direct_injection_containers = containers_calc['containers']
                            direct_injection_cube = containers_calc['cube']

            # === MIDDLE-MILE INJECTION (Origin) ===
            mm_injection_pkgs = 0
            mm_injection_containers = 0
            mm_injection_cube = 0.0
            mm_injection_cost = 0.0

            if not od_selected.empty:
                outbound_ods = od_selected[od_selected['origin'] == facility]
                if not outbound_ods.empty:
                    mm_injection_pkgs = outbound_ods['pkgs_day'].sum()
                    mm_injection_cost = outbound_ods['total_cost'].sum()

                    # Calculate containers for middle-mile injection
                    if mm_injection_pkgs > 0:
                        containers_calc = _calculate_containers_for_volume(
                            mm_injection_pkgs, package_mix, container_params, strategy
                        )
                        mm_injection_containers = containers_calc['containers']
                        mm_injection_cube = containers_calc['cube']

            # === INTERMEDIATE (Pass-through) ===
            intermediate_pkgs = 0
            intermediate_containers = 0
            intermediate_cube = 0.0

            if not arc_summary.empty and not od_selected.empty:
                # Packages arriving at this facility
                inbound_arcs = arc_summary[
                    (arc_summary['to_facility'] == facility) &
                    (arc_summary['from_facility'] != facility)
                    ]

                if not inbound_arcs.empty:
                    total_inbound = inbound_arcs['pkgs_day'].sum()

                    # Packages destined for this facility (final delivery)
                    destined_here = od_selected[od_selected['dest'] == facility]['pkgs_day'].sum()

                    # Intermediate = inbound - final destination
                    intermediate_pkgs = max(0, total_inbound - destined_here)

                    # Calculate containers for intermediate
                    if intermediate_pkgs > 0:
                        containers_calc = _calculate_containers_for_volume(
                            intermediate_pkgs, package_mix, container_params, strategy
                        )
                        intermediate_containers = containers_calc['containers']
                        intermediate_cube = containers_calc['cube']

            # === LAST MILE (All arriving for delivery) ===
            last_mile_pkgs = 0
            last_mile_containers = 0
            last_mile_cube = 0.0

            # Start with direct injection
            last_mile_pkgs = direct_injection_pkgs
            last_mile_containers = direct_injection_containers
            last_mile_cube = direct_injection_cube

            # Add middle-mile arrivals for final delivery
            if not od_selected.empty:
                inbound_ods = od_selected[od_selected['dest'] == facility]
                if not inbound_ods.empty:
                    mm_last_mile_pkgs = inbound_ods['pkgs_day'].sum()
                    last_mile_pkgs += mm_last_mile_pkgs

                    if mm_last_mile_pkgs > 0:
                        containers_calc = _calculate_containers_for_volume(
                            mm_last_mile_pkgs, package_mix, container_params, strategy
                        )
                        last_mile_containers += containers_calc['containers']
                        last_mile_cube += containers_calc['cube']

            # Calculate per-package costs
            mm_injection_cpp = mm_injection_cost / mm_injection_pkgs if mm_injection_pkgs > 0 else 0

            volume_entry = {
                'facility': facility,

                # Direct injection (Zone 0)
                'direct_injection_pkgs_day': direct_injection_pkgs,
                'direct_injection_containers': direct_injection_containers,
                'direct_injection_cube_cuft': direct_injection_cube,

                # Middle-mile injection (network origin)
                'middle_mile_injection_pkgs_day': mm_injection_pkgs,
                'middle_mile_injection_containers': mm_injection_containers,
                'middle_mile_injection_cube_cuft': mm_injection_cube,
                'middle_mile_injection_cost': mm_injection_cost,
                'middle_mile_injection_cpp': mm_injection_cpp,

                # Intermediate (pass-through)
                'middle_mile_intermediate_pkgs_day': intermediate_pkgs,
                'middle_mile_intermediate_containers': intermediate_containers,
                'middle_mile_intermediate_cube_cuft': intermediate_cube,

                # Last mile (all delivery)
                'last_mile_pkgs_day': last_mile_pkgs,
                'last_mile_containers': last_mile_containers,
                'last_mile_cube_cuft': last_mile_cube,
            }

            volume_data.append(volume_entry)

        except Exception as e:
            print(f"    Warning: Error calculating volume for facility {facility}: {e}")
            continue

    return pd.DataFrame(volume_data)


def _calculate_containers_for_volume(
        packages: float,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        strategy: str
) -> Dict[str, float]:
    """Calculate containers and cube for given package volume."""
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


def calculate_hourly_throughput(
        facility_rollup: pd.DataFrame,
        timing_params: Dict
) -> pd.DataFrame:
    """
    Calculate hourly throughput requirements for capacity planning.

    Uses value-added hours from timing_params:
    - injection_va_hours: For origin processing
    - middle_mile_va_hours: For intermediate crossdock/sort
    - last_mile_va_hours: For final delivery processing

    Args:
        facility_rollup: Facility volume data
        timing_params: Timing parameters dict

    Returns:
        facility_rollup with added throughput columns
    """
    df = facility_rollup.copy()

    # Get VA hours from parameters
    injection_va_hours = float(timing_params['injection_va_hours'])
    middle_mile_va_hours = float(timing_params['middle_mile_va_hours'])
    last_mile_va_hours = float(timing_params['last_mile_va_hours'])

    # Calculate throughput for each role
    df['injection_hourly_throughput'] = df['middle_mile_injection_pkgs_day'] / injection_va_hours
    df['intermediate_hourly_throughput'] = df['middle_mile_intermediate_pkgs_day'] / middle_mile_va_hours
    df['last_mile_hourly_throughput'] = df['last_mile_pkgs_day'] / last_mile_va_hours

    # Peak throughput is maximum across all roles
    df['peak_hourly_throughput'] = df[[
        'injection_hourly_throughput',
        'intermediate_hourly_throughput',
        'last_mile_hourly_throughput'
    ]].max(axis=1)

    # Round for practical planning
    throughput_cols = [
        'injection_hourly_throughput',
        'intermediate_hourly_throughput',
        'last_mile_hourly_throughput',
        'peak_hourly_throughput'
    ]

    for col in throughput_cols:
        df[col] = df[col].fillna(0).round(0).astype(int)

    return df


def add_zone_classification(
        od_df: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> pd.DataFrame:
    """
    Add zone classification based on O-D straight-line distance.

    Zone Rules:
    - Direct injection (not in this df): Zone 0
    - O=D middle-mile: Zone 2 (distance=0, but priced as network volume)
    - O≠D: Lookup zone from mileage_bands by straight-line distance

    Args:
        od_df: OD selected paths
        facilities: Facility master with coordinates
        mileage_bands: Mileage bands with zone mapping

    Returns:
        od_df with added 'zone' column
    """
    if od_df.empty:
        return od_df

    od_df = od_df.copy()
    od_df['zone'] = 'unknown'

    for idx, row in od_df.iterrows():
        origin = row['origin']
        dest = row['dest']

        if origin == dest:
            # O=D middle-mile: Always Zone 2
            od_df.at[idx, 'zone'] = 'Zone 2'
        else:
            # O≠D: Calculate zone by distance
            zone = calculate_zone_from_distance(origin, dest, facilities, mileage_bands)
            od_df.at[idx, 'zone'] = zone

    return od_df


def build_path_steps(
        od_selected: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame,
        timing_params: Dict
) -> pd.DataFrame:
    """
    Generate path steps from selected OD paths.

    Each step represents one leg of the journey with actual calculated distances.

    Args:
        od_selected: Selected OD paths
        facilities: Facility data with coordinates
        mileage_bands: Mileage bands for distance/timing
        timing_params: Timing parameters

    Returns:
        DataFrame with path steps
    """
    from .geo_v3 import haversine_miles, band_lookup

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
            # Check for O=D leg
            if from_fac == to_fac:
                path_steps.append({
                    'scenario_id': scenario_id,
                    'origin': od_row['origin'],
                    'dest': od_row['dest'],
                    'step_order': i + 1,
                    'from_facility': from_fac,
                    'to_facility': to_fac,
                    'distance_miles': 0.0,
                    'drive_hours': 0.0,
                    'processing_hours_at_destination': hours_per_touch
                })
                continue

            # Calculate actual distance and timing
            if from_fac in fac_lookup.index and to_fac in fac_lookup.index:
                lat1, lon1 = fac_lookup.at[from_fac, 'lat'], fac_lookup.at[from_fac, 'lon']
                lat2, lon2 = fac_lookup.at[to_fac, 'lat'], fac_lookup.at[to_fac, 'lon']

                raw_dist = haversine_miles(lat1, lon1, lat2, lon2)
                fixed, var, circuit, mph = band_lookup(raw_dist, mileage_bands)
                actual_dist = raw_dist * circuit
                drive_hours = actual_dist / mph

                path_steps.append({
                    'scenario_id': scenario_id,
                    'origin': od_row['origin'],
                    'dest': od_row['dest'],
                    'step_order': i + 1,
                    'from_facility': from_fac,
                    'to_facility': to_fac,
                    'distance_miles': actual_dist,
                    'drive_hours': drive_hours,
                    'processing_hours_at_destination': hours_per_touch
                })
            else:
                print(f"    Warning: Missing coordinates for {from_fac} or {to_fac}")

    return pd.DataFrame(path_steps)


def validate_network_aggregations(
        od_selected: pd.DataFrame,
        arc_summary: pd.DataFrame,
        facility_rollup: pd.DataFrame
) -> Dict:
    """
    Validate that aggregate calculations are mathematically consistent.

    Checks:
    - Total OD packages = Total facility injection
    - Package volumes consistent across levels
    - Costs aggregate properly

    Args:
        od_selected: Selected OD paths
        arc_summary: Arc aggregations
        facility_rollup: Facility rollup

    Returns:
        Dict with validation results and metrics
    """
    validation_results = {}

    try:
        # Package validation
        total_od_pkgs = od_selected['pkgs_day'].sum() if not od_selected.empty else 0
        total_facility_mm_injection = facility_rollup[
            'middle_mile_injection_pkgs_day'].sum() if not facility_rollup.empty else 0

        validation_results['total_od_packages'] = total_od_pkgs
        validation_results['total_facility_mm_injection'] = total_facility_mm_injection
        validation_results['package_consistency'] = abs(total_od_pkgs - total_facility_mm_injection) < 0.01

        # Cost validation
        total_od_cost = od_selected['total_cost'].sum() if 'total_cost' in od_selected.columns else 0
        total_arc_cost = arc_summary[
            'total_cost'].sum() if not arc_summary.empty and 'total_cost' in arc_summary.columns else 0

        validation_results['total_od_cost'] = total_od_cost
        validation_results['total_arc_cost'] = total_arc_cost
        validation_results['cost_consistency'] = abs(total_od_cost - total_arc_cost) / max(total_od_cost, 1) < 0.05

        # Fill rates
        if not arc_summary.empty and 'truck_fill_rate' in arc_summary.columns:
            non_od_arcs = arc_summary[arc_summary['from_facility'] != arc_summary['to_facility']]

            if not non_od_arcs.empty:
                total_pkg_cube = non_od_arcs['pkg_cube_cuft'].sum() if 'pkg_cube_cuft' in non_od_arcs.columns else 0
                total_truck_cube = (non_od_arcs['trucks'] * non_od_arcs.get('cube_per_truck',
                                                                            0)).sum() if 'trucks' in non_od_arcs.columns else 1

                if total_truck_cube > 0:
                    validation_results['network_avg_truck_fill'] = total_pkg_cube / total_truck_cube
                else:
                    validation_results['network_avg_truck_fill'] = 0
            else:
                validation_results['network_avg_truck_fill'] = 0
        else:
            validation_results['network_avg_truck_fill'] = 0

    except Exception as e:
        validation_results['validation_error'] = str(e)

    return validation_results