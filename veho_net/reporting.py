import pandas as pd
import numpy as np


def _identify_volume_types_with_costs(od_selected: pd.DataFrame, path_steps_selected: pd.DataFrame,
                                      direct_day: pd.DataFrame, arc_summary: pd.DataFrame) -> pd.DataFrame:
    """
    COMPLETELY REWRITTEN: Proper facility cost calculation based on actual role in network.

    Key insight: Facilities play different roles and costs should reflect actual operations:
    - Injection facilities: Sort and send packages (injection costs)
    - Intermediate facilities: Receive, process, and forward packages (crossdock costs)
    - Destination facilities: Receive packages for final delivery (last mile costs)
    """

    volume_data = []

    # Get all facilities from multiple sources
    all_facilities = set()

    if not od_selected.empty:
        all_facilities.update(od_selected['origin'].unique())
        all_facilities.update(od_selected['dest'].unique())

    if not direct_day.empty and 'dest' in direct_day.columns:
        all_facilities.update(direct_day['dest'].unique())

    for facility in all_facilities:
        try:
            # 1. INJECTION ROLE: This facility as origin (sending packages out)
            injection_pkgs = 0
            injection_linehaul_cost = 0
            injection_processing_cost = 0

            if not od_selected.empty:
                outbound_ods = od_selected[od_selected['origin'] == facility]
                if not outbound_ods.empty:
                    injection_pkgs = outbound_ods['pkgs_day'].sum()

                    # Injection costs: sort cost + outbound linehaul cost
                    if 'injection_sort_cost' in outbound_ods.columns:
                        injection_processing_cost = outbound_ods['injection_sort_cost'].sum()
                    if 'linehaul_cost' in outbound_ods.columns:
                        injection_linehaul_cost = outbound_ods['linehaul_cost'].sum()

            # 2. INTERMEDIATE ROLE: This facility as pass-through (crossdock operations)
            intermediate_pkgs = 0
            intermediate_linehaul_cost = 0
            intermediate_processing_cost = 0

            if not arc_summary.empty:
                # As destination of inbound lanes (packages arriving for processing)
                inbound_arcs = arc_summary[arc_summary['to_facility'] == facility]
                # As origin of outbound lanes (packages leaving after processing)
                outbound_arcs = arc_summary[arc_summary['from_facility'] == facility]

                if not inbound_arcs.empty and not outbound_arcs.empty:
                    # This facility has both inbound and outbound flows = intermediate role
                    inbound_pkgs = inbound_arcs['pkgs_day'].sum()
                    outbound_pkgs = outbound_arcs['pkgs_day'].sum()

                    # Intermediate packages = packages that flow through (not originating here)
                    # Use inbound volume minus any packages that originate at this facility
                    intermediate_pkgs = inbound_pkgs - injection_pkgs
                    intermediate_pkgs = max(0, intermediate_pkgs)  # Can't be negative

                    if intermediate_pkgs > 0:
                        # Processing cost for intermediate packages (crossdock operations)
                        crossdock_cost_per_pkg = 0.30  # $0.30 per package for crossdock
                        intermediate_processing_cost = intermediate_pkgs * crossdock_cost_per_pkg

                        # Linehaul cost allocation for inbound packages
                        if not inbound_arcs.empty:
                            total_inbound_cost = inbound_arcs['total_cost'].sum()
                            if inbound_pkgs > 0:
                                # Allocate inbound linehaul cost proportionally to intermediate packages
                                intermediate_linehaul_cost = total_inbound_cost * (intermediate_pkgs / inbound_pkgs)

            # 3. DESTINATION ROLE: This facility as final destination
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

                    # Last mile delivery costs
                    if 'last_mile_delivery_cost' in inbound_ods.columns:
                        last_mile_delivery_cost = inbound_ods['last_mile_delivery_cost'].sum()

            # CRITICAL FIX: Calculate costs per package correctly
            injection_sort_cpp = (injection_processing_cost / injection_pkgs) if injection_pkgs > 0 else 0
            mm_linehaul_cpp = (injection_linehaul_cost + intermediate_linehaul_cost) / (
                    injection_pkgs + intermediate_pkgs) if (injection_pkgs + intermediate_pkgs) > 0 else 0
            mm_processing_cpp = (intermediate_processing_cost / intermediate_pkgs) if intermediate_pkgs > 0 else 0
            last_mile_delivery_cpp = (last_mile_delivery_cost / last_mile_pkgs) if last_mile_pkgs > 0 else 0

            volume_entry = {
                'facility': facility,
                'injection_pkgs_day': injection_pkgs,
                'intermediate_pkgs_day': intermediate_pkgs,
                'last_mile_pkgs_day': last_mile_pkgs,
                # Cost totals
                'injection_linehaul_cost': injection_linehaul_cost,
                'injection_processing_cost': injection_processing_cost,
                'intermediate_linehaul_cost': intermediate_linehaul_cost,
                'intermediate_processing_cost': intermediate_processing_cost,
                'last_mile_delivery_cost': last_mile_delivery_cost,
                # CRITICAL: Cost per package calculations
                'injection_sort_cpp': injection_sort_cpp,
                'mm_linehaul_cpp': mm_linehaul_cpp,
                'mm_processing_cpp': mm_processing_cpp,
                'last_mile_delivery_cpp': last_mile_delivery_cpp,
                'total_variable_cpp': injection_sort_cpp + mm_linehaul_cpp + mm_processing_cpp + last_mile_delivery_cpp
            }

            volume_data.append(volume_entry)

        except Exception as e:
            # Add default entry
            volume_data.append({
                'facility': facility,
                'injection_pkgs_day': 0,
                'intermediate_pkgs_day': 0,
                'last_mile_pkgs_day': 0,
                'injection_linehaul_cost': 0,
                'injection_processing_cost': 0,
                'intermediate_linehaul_cost': 0,
                'intermediate_processing_cost': 0,
                'last_mile_delivery_cost': 0,
                'injection_sort_cpp': 0,
                'mm_linehaul_cpp': 0,
                'mm_processing_cpp': 0,
                'last_mile_delivery_cpp': 0,
                'total_variable_cpp': 0
            })

    result_df = pd.DataFrame(volume_data)
    return result_df


def _calculate_hourly_throughput_with_costs(volume_df: pd.DataFrame, timing_kv: dict,
                                            load_strategy: str) -> pd.DataFrame:
    """SIMPLIFIED: Just add throughput calculations, costs already calculated above."""

    df = volume_df.copy()

    # Get VA hours safely
    injection_va_hours = float(timing_kv.get('injection_va_hours', 8.0))
    middle_mile_va_hours = float(timing_kv.get('middle_mile_va_hours', 16.0))
    last_mile_va_hours = float(timing_kv.get('last_mile_va_hours', 4.0))

    # Throughput calculation
    df['injection_hourly_throughput'] = df['injection_pkgs_day'] / injection_va_hours
    df['intermediate_hourly_throughput'] = df['intermediate_pkgs_day'] / middle_mile_va_hours
    df['lm_hourly_throughput'] = df['last_mile_pkgs_day'] / last_mile_va_hours

    # Peak is max of all types
    df['peak_hourly_throughput'] = df[
        ['injection_hourly_throughput', 'intermediate_hourly_throughput', 'lm_hourly_throughput']
    ].max(axis=1)

    # Round throughput columns
    throughput_cols = ['injection_hourly_throughput', 'intermediate_hourly_throughput', 'lm_hourly_throughput',
                       'peak_hourly_throughput']
    for col in throughput_cols:
        df[col] = df[col].fillna(0).round(0).astype(int)

    # Round cost columns
    cost_cols = ['injection_sort_cpp', 'mm_linehaul_cpp', 'mm_processing_cpp', 'last_mile_delivery_cpp',
                 'total_variable_cpp']
    for col in cost_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).round(3)

    return df


def add_zone(df: pd.DataFrame, facilities: pd.DataFrame) -> pd.DataFrame:
    """Add zone classification based on distance or region."""
    if df.empty:
        return df

    # Simple zone logic - can be enhanced based on your needs
    df = df.copy()
    df['zone'] = 'unknown'

    # Create facility lookup
    fac_regions = facilities.set_index('facility_name')['region'].to_dict()

    # Add zones based on origin regions
    if 'origin' in df.columns:
        df['zone'] = df['origin'].map(fac_regions).fillna('unknown')

    return df


def build_od_selected_outputs(od_selected: pd.DataFrame, facilities: pd.DataFrame,
                              direct_day: pd.DataFrame) -> pd.DataFrame:
    """Build enhanced OD outputs with zone information."""
    if od_selected.empty:
        return od_selected

    od_out = od_selected.copy()

    # Add zone information for analysis
    od_out = add_zone(od_out, facilities)

    return od_out


def build_dwell_hotspots(od_selected: pd.DataFrame) -> pd.DataFrame:
    """Build dwell hotspots analysis."""
    if od_selected.empty or 'packages_dwelled' not in od_selected.columns:
        return pd.DataFrame()

    # Filter to ODs with significant dwell
    dwelled = od_selected[od_selected['packages_dwelled'] > 10].copy()

    if dwelled.empty:
        return pd.DataFrame()

    # Aggregate by origin
    hotspots = dwelled.groupby('origin').agg({
        'packages_dwelled': 'sum',
        'pkgs_day': 'sum',
        'dest': 'count'
    }).reset_index()

    hotspots['dwell_rate'] = hotspots['packages_dwelled'] / hotspots['pkgs_day']
    hotspots = hotspots.sort_values('packages_dwelled', ascending=False)

    return hotspots


def build_lane_summary(arc_summary: pd.DataFrame) -> pd.DataFrame:
    """Build lane summary from arc data."""
    if arc_summary.empty:
        return pd.DataFrame()

    # Group by lane (from-to pair) aggregating across scenarios/day types
    lane_summary = arc_summary.groupby(['from_facility', 'to_facility']).agg({
        'pkgs_day': 'sum',
        'trucks': 'mean',
        'total_cost': 'sum',
        'distance_miles': 'first',
        'packages_dwelled': 'sum'
    }).reset_index()

    lane_summary['cost_per_pkg'] = lane_summary['total_cost'] / lane_summary['pkgs_day'].replace(0, 1)

    return lane_summary.sort_values('total_cost', ascending=False)