"""
Network Reporting Module

Generates facility-level, arc-level, and network-level aggregations and KPIs
from optimization results. Provides executive-ready metrics and operational insights.

Key Functions:
- Facility volume analysis by role (injection, intermediate, last-mile)
- Hourly throughput calculations for capacity planning
- Zone classification for pricing analysis
- Dwell hotspot identification for operational attention
- Network-wide KPI aggregation and validation
"""

import pandas as pd
import numpy as np
from .geo import haversine_miles


def _identify_volume_types_with_costs(
        od_selected: pd.DataFrame,
        path_steps_selected: pd.DataFrame,
        direct_day: pd.DataFrame,
        arc_summary: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate facility volume and cost breakdowns by role type.

    Each facility shows costs for packages that ORIGINATE there,
    including the full cost chain through to final delivery. This
    attribution approach supports hub network development by showing
    end-to-end costs by injection point.

    Facility Roles:
    - Injection: Packages originating at this facility (full cost chain)
    - Intermediate: Packages passing through (processing cost only)
    - Last Mile: Packages delivered here (from direct or middle-mile)

    Args:
        od_selected: Selected OD paths with costs
        path_steps_selected: Path step details (currently unused)
        direct_day: Direct injection volumes
        arc_summary: Arc-level aggregated metrics

    Returns:
        DataFrame with facility-level volume and cost metrics by role
    """
    volume_data = []

    # Collect all facilities from various data sources
    all_facilities = set()

    if not od_selected.empty:
        all_facilities.update(od_selected['origin'].unique())
        all_facilities.update(od_selected['dest'].unique())

    if not direct_day.empty and 'dest' in direct_day.columns:
        all_facilities.update(direct_day['dest'].unique())

    for facility in all_facilities:
        try:
            # Injection role: facility as origin - show FULL cost chain for originating packages
            injection_pkgs = 0
            total_injection_cost = 0

            if not od_selected.empty:
                outbound_ods = od_selected[od_selected['origin'] == facility]
                if not outbound_ods.empty:
                    injection_pkgs = outbound_ods['pkgs_day'].sum()
                    # Full cost for packages originating at this facility
                    total_injection_cost = outbound_ods['total_cost'].sum()

            # Intermediate role: packages passing through (NOT originating here)
            intermediate_pkgs = 0
            intermediate_processing_cost = 0

            if not arc_summary.empty:
                # Packages arriving for processing (excluding packages that originated here)
                inbound_arcs = arc_summary[arc_summary['to_facility'] == facility]

                if not inbound_arcs.empty:
                    inbound_pkgs = inbound_arcs['pkgs_day'].sum()
                    # Intermediate packages = inbound packages not originating at this facility
                    intermediate_pkgs = inbound_pkgs - injection_pkgs
                    intermediate_pkgs = max(0, intermediate_pkgs)

                    # Processing cost for intermediate packages only
                    if intermediate_pkgs > 0:
                        # This would be crossdock/sort costs for pass-through packages
                        intermediate_processing_cost = 0  # Simplified for now

            # Destination role: packages delivered here (from direct injection)
            last_mile_pkgs = 0
            last_mile_delivery_cost = 0

            # Direct injection packages (bypass middle mile)
            if not direct_day.empty:
                direct_col = 'dir_pkgs_day'
                if direct_col in direct_day.columns:
                    facility_direct = direct_day[direct_day['dest'] == facility]
                    if not facility_direct.empty:
                        last_mile_pkgs = facility_direct[direct_col].sum()

            # Middle mile packages arriving for final delivery
            if not od_selected.empty:
                inbound_ods = od_selected[od_selected['dest'] == facility]
                if not inbound_ods.empty:
                    last_mile_pkgs += inbound_ods['pkgs_day'].sum()

            # Calculate unit costs for packages ORIGINATING at this facility
            injection_cost_per_pkg = (total_injection_cost / injection_pkgs) if injection_pkgs > 0 else 0
            intermediate_cost_per_pkg = (
                        intermediate_processing_cost / intermediate_pkgs) if intermediate_pkgs > 0 else 0

            volume_entry = {
                'facility': facility,
                'injection_pkgs_day': injection_pkgs,
                'intermediate_pkgs_day': intermediate_pkgs,
                'last_mile_pkgs_day': last_mile_pkgs,
                # For injection: show full cost chain for originating packages
                'injection_total_cost': total_injection_cost,
                'injection_cost_per_pkg': injection_cost_per_pkg,
                # For intermediate: show processing cost for pass-through packages
                'intermediate_processing_cost': intermediate_processing_cost,
                'intermediate_cost_per_pkg': intermediate_cost_per_pkg,
                # Legacy columns for compatibility
                'injection_linehaul_cost': 0,
                'injection_processing_cost': 0,
                'intermediate_linehaul_cost': 0,
                'last_mile_delivery_cost': 0,
                'injection_sort_cpp': 0,
                'mm_linehaul_cpp': 0,
                'mm_processing_cpp': 0,
                'last_mile_delivery_cpp': 0,
                'last_mile_cost': 0,
                'last_mile_cpp': 0,
                'total_variable_cpp': injection_cost_per_pkg
            }

            volume_data.append(volume_entry)

        except Exception as e:
            print(f"    Warning: Could not calculate volume for facility {facility}: {e}")
            # Add default entry to maintain facility in output
            volume_data.append({
                'facility': facility,
                'injection_pkgs_day': 0,
                'intermediate_pkgs_day': 0,
                'last_mile_pkgs_day': 0,
                'injection_total_cost': 0,
                'injection_cost_per_pkg': 0,
                'intermediate_processing_cost': 0,
                'intermediate_cost_per_pkg': 0,
                'injection_linehaul_cost': 0,
                'injection_processing_cost': 0,
                'intermediate_linehaul_cost': 0,
                'last_mile_delivery_cost': 0,
                'injection_sort_cpp': 0,
                'mm_linehaul_cpp': 0,
                'mm_processing_cpp': 0,
                'last_mile_delivery_cpp': 0,
                'last_mile_cost': 0,
                'last_mile_cpp': 0,
                'total_variable_cpp': 0
            })

    return pd.DataFrame(volume_data)


def _calculate_hourly_throughput_with_costs(
        volume_df: pd.DataFrame,
        timing_kv: dict,
        load_strategy: str
) -> pd.DataFrame:
    """
    Calculate facility throughput requirements based on value-added hours.

    Uses timing parameters to convert daily volumes into hourly throughput
    requirements for capacity planning and labor forecasting.

    Args:
        volume_df: Facility volume data by role
        timing_kv: Timing parameters with VA hours by facility role
        load_strategy: Loading strategy (used for context, not calculation)

    Returns:
        volume_df with added hourly throughput columns
    """
    df = volume_df.copy()

    # Get VA hours from timing parameters
    injection_va_hours = float(timing_kv.get('injection_va_hours',
                                             timing_kv.get('sort_hours_per_touch', 8.0)))
    middle_mile_va_hours = float(timing_kv.get('middle_mile_va_hours',
                                               timing_kv.get('crossdock_hours_per_touch', 16.0)))
    last_mile_va_hours = float(timing_kv.get('last_mile_va_hours', 4.0))

    # Calculate throughput for each facility role
    df['injection_hourly_throughput'] = df['injection_pkgs_day'] / injection_va_hours
    df['intermediate_hourly_throughput'] = df['intermediate_pkgs_day'] / middle_mile_va_hours
    df['lm_hourly_throughput'] = df['last_mile_pkgs_day'] / last_mile_va_hours

    # Peak throughput is maximum across all roles
    df['peak_hourly_throughput'] = df[
        ['injection_hourly_throughput', 'intermediate_hourly_throughput', 'lm_hourly_throughput']
    ].max(axis=1)

    # Round throughput values for practical planning
    throughput_cols = ['injection_hourly_throughput', 'intermediate_hourly_throughput',
                       'lm_hourly_throughput', 'peak_hourly_throughput']
    for col in throughput_cols:
        df[col] = df[col].fillna(0).round(0).astype(int)

    return df


def calculate_zone_from_distance(
        origin: str,
        dest: str,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> str:
    """
    Calculate zone based on straight-line haversine distance between origin and destination.

    Zone classification is based on O-D distance regardless of routing, as zones
    drive shipper pricing and should not depend on carrier routing decisions.

    Args:
        origin: Origin facility name
        dest: Destination facility name
        facilities: Facility master with coordinates
        mileage_bands: Mileage bands with zone mappings

    Returns:
        Zone classification string (e.g., "Zone 1", "Zone 2")
    """
    try:
        # Get facility coordinates
        fac_lookup = facilities.set_index('facility_name')[['lat', 'lon']]

        if origin not in fac_lookup.index or dest not in fac_lookup.index:
            return 'unknown'

        o_lat, o_lon = fac_lookup.at[origin, 'lat'], fac_lookup.at[origin, 'lon']
        d_lat, d_lon = fac_lookup.at[dest, 'lat'], fac_lookup.at[dest, 'lon']

        # Calculate straight-line distance (no circuity factor applied for zone classification)
        raw_distance = haversine_miles(o_lat, o_lon, d_lat, d_lon)

        # Look up zone from mileage bands
        if 'zone' in mileage_bands.columns:
            matching_band = mileage_bands[
                (mileage_bands['mileage_band_min'] <= raw_distance) &
                (raw_distance <= mileage_bands['mileage_band_max'])
                ]

            if not matching_band.empty:
                return str(matching_band.iloc[0]['zone'])

        return 'unknown'

    except Exception as e:
        print(f"    Warning: Could not calculate zone for {origin}->{dest}: {e}")
        return 'unknown'


def add_zone(
        df: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Add zone classification based on origin-destination straight-line distance.

    Direct injection (no middle-mile routing) is always classified as "Zone 0".
    All other flows classified by O-D distance using mileage_bands zone mapping.

    Args:
        df: DataFrame with origin, dest, and optionally path_type columns
        facilities: Facility master data
        mileage_bands: Mileage bands with zone mappings (optional)

    Returns:
        df with added 'zone' column
    """
    if df.empty:
        return df

    df = df.copy()

    # Check for direct injection (Zone 0)
    if 'path_type' in df.columns:
        df['zone'] = df['path_type'].apply(lambda x: 'Zone 0' if x == 'direct' else 'unknown')
    else:
        df['zone'] = 'unknown'

    # Calculate zones for non-direct paths
    if mileage_bands is not None and 'origin' in df.columns and 'dest' in df.columns:
        non_direct_mask = df['zone'] == 'unknown'

        for idx in df[non_direct_mask].index:
            origin = df.at[idx, 'origin']
            dest = df.at[idx, 'dest']
            zone = calculate_zone_from_distance(origin, dest, facilities, mileage_bands)
            df.at[idx, 'zone'] = zone

    return df


def build_od_selected_outputs(
        od_selected: pd.DataFrame,
        facilities: pd.DataFrame,
        direct_day: pd.DataFrame,
        mileage_bands: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Build OD output table with correct zone information.

    Adds zone classification based on O-D straight-line distance for
    pricing and analysis purposes.

    Args:
        od_selected: Selected OD paths from optimization
        facilities: Facility master data
        direct_day: Direct injection volumes
        mileage_bands: Mileage bands with zone mappings

    Returns:
        od_selected with added zone column
    """
    if od_selected.empty:
        return od_selected

    od_out = od_selected.copy()

    # Add correct zone calculation based on O-D distance
    od_out = add_zone(od_out, facilities, mileage_bands)

    return od_out


def build_dwell_hotspots(od_selected: pd.DataFrame) -> pd.DataFrame:
    """
    Identify facilities with significant package dwell for operational attention.

    Flags origin facilities where packages are being held for next-day
    dispatch due to premium economy threshold rounding.

    Args:
        od_selected: Selected OD paths with packages_dwelled column

    Returns:
        DataFrame of dwell hotspots sorted by total dwelled packages
    """
    if od_selected.empty or 'packages_dwelled' not in od_selected.columns:
        return pd.DataFrame()

    # Filter to ODs with meaningful dwell volumes (>10 packages)
    dwelled = od_selected[od_selected['packages_dwelled'] > 10].copy()

    if dwelled.empty:
        return pd.DataFrame()

    # Aggregate dwell by origin facility
    hotspots = dwelled.groupby('origin').agg({
        'packages_dwelled': 'sum',
        'pkgs_day': 'sum',
        'dest': 'count'
    }).reset_index()

    hotspots['dwell_rate'] = hotspots['packages_dwelled'] / hotspots['pkgs_day']
    hotspots = hotspots.sort_values('packages_dwelled', ascending=False)

    return hotspots


def build_lane_summary(arc_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Create lane-level summary aggregating across scenarios and day types.

    Useful for identifying high-volume or high-cost lanes for
    operational focus and carrier negotiations.

    Args:
        arc_summary: Arc-level data with volumes and costs

    Returns:
        DataFrame of lane summaries sorted by total cost
    """
    if arc_summary.empty:
        return pd.DataFrame()

    # Aggregate by lane (from-to facility pair)
    lane_summary = arc_summary.groupby(['from_facility', 'to_facility']).agg({
        'pkgs_day': 'sum',
        'trucks': 'mean',
        'total_cost': 'sum',
        'distance_miles': 'first',
        'packages_dwelled': 'sum'
    }).reset_index()

    lane_summary['cost_per_pkg'] = lane_summary['total_cost'] / lane_summary['pkgs_day'].replace(0, 1)

    return lane_summary.sort_values('total_cost', ascending=False)


def validate_network_aggregations(
        od_selected: pd.DataFrame,
        arc_summary: pd.DataFrame,
        facility_rollup: pd.DataFrame
) -> dict:
    """
    Validate that aggregate calculations are mathematically consistent.

    Checks package volume consistency across OD, arc, and facility levels.
    Validates cost aggregations between OD and arc levels. Uses package-weighted
    averages for fill rate calculations.

    Validation Checks:
    1. Total OD packages = Total facility injection packages
    2. Total OD packages = Total arc packages
    3. Total OD cost ≈ Total arc cost (within 5%)
    4. Network fill rates calculated consistently

    Args:
        od_selected: Selected OD paths
        arc_summary: Arc-level aggregations
        facility_rollup: Facility-level aggregations

    Returns:
        Dictionary with validation results and metrics
    """
    validation_results = {}

    try:
        # Total package validation
        total_od_pkgs = od_selected['pkgs_day'].sum() if not od_selected.empty else 0
        total_arc_pkgs = arc_summary['pkgs_day'].sum() if not arc_summary.empty else 0
        total_facility_injection = facility_rollup['injection_pkgs_day'].sum() if not facility_rollup.empty else 0

        validation_results['total_od_packages'] = total_od_pkgs
        validation_results['total_arc_packages'] = total_arc_pkgs
        validation_results['total_facility_injection'] = total_facility_injection
        validation_results['package_consistency'] = abs(total_od_pkgs - total_facility_injection) < 0.01
        validation_results['arc_package_consistency'] = abs(total_od_pkgs - total_arc_pkgs) < 0.01

        # Cost validation
        total_od_cost = od_selected['total_cost'].sum() if 'total_cost' in od_selected.columns else 0
        total_arc_cost = arc_summary['total_cost'].sum() if 'total_cost' in arc_summary.columns else 0

        validation_results['total_od_cost'] = total_od_cost
        validation_results['total_arc_cost'] = total_arc_cost
        validation_results['cost_consistency'] = abs(total_od_cost - total_arc_cost) / max(total_od_cost, 1) < 0.05

        # Package-weighted fill rates from arc data
        if not arc_summary.empty and 'truck_fill_rate' in arc_summary.columns:
            # Get total package cube and total truck cube for inherent weighting
            total_pkg_cube = arc_summary['pkg_cube_cuft'].sum() if 'pkg_cube_cuft' in arc_summary.columns else 0
            total_truck_cube = (arc_summary['trucks'] * arc_summary.get('cube_per_truck',
                                                                        0)).sum() if 'trucks' in arc_summary.columns else 1

            if total_truck_cube > 0:
                validation_results['network_avg_truck_fill'] = total_pkg_cube / total_truck_cube
            else:
                validation_results['network_avg_truck_fill'] = 0

            # Container fill rate calculation
            if 'container_fill_rate' in arc_summary.columns:
                total_volume = arc_summary['pkgs_day'].sum()
                if total_volume > 0:
                    validation_results['network_avg_container_fill'] = (
                                                                               arc_summary['container_fill_rate'] *
                                                                               arc_summary['pkgs_day']
                                                                       ).sum() / total_volume
                else:
                    validation_results['network_avg_container_fill'] = 0
            else:
                validation_results['network_avg_container_fill'] = 0
        else:
            validation_results['network_avg_truck_fill'] = 0
            validation_results['network_avg_container_fill'] = 0

    except Exception as e:
        validation_results['validation_error'] = str(e)

    return validation_results