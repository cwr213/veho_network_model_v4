"""
Zone Cost Analysis Module

Complete zone tracking:
- Zone 0: Direct injection (no middle-mile)
- Zone 1: Middle-mile O=D (has injection sort)
- Zones 2-8: Distance-based from mileage bands
- Unknown: Unclassified (data quality flag)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

from .geo_v4 import haversine_miles, band_lookup
from .utils import safe_divide, get_facility_lookup


def calculate_zone_cost_analysis(
        od_selected: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame,
        direct_day: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Calculate comprehensive cost analysis by zone.

    Provides executive-level view of network economics by distance tier.

    Includes:
    - Zone 0: Direct injection packages (if direct_day provided)
    - Zones 1-8: Middle-mile packages by distance
    - Unknown: Unclassified packages (data quality flag)

    Args:
        od_selected: Selected OD paths with cost breakdown (middle-mile flows)
        facilities: Facility master data
        mileage_bands: Mileage bands for distance calculations
        direct_day: Direct injection volumes (optional, for zone 0)

    Returns:
        DataFrame with columns:
            - zone: Zone identifier (0-8, unknown)
            - total_packages: Package volume
            - pct_of_total_packages: Volume percentage
            - total_cost: Aggregate cost
            - pct_of_total_cost: Cost percentage
            - cost_per_pkg: Average cost per package
            - linehaul_cost_per_pkg: Transport cost component
            - processing_cost_per_pkg: Handling cost component
            - avg_zone_miles: Average straight-line O-D distance
            - avg_transit_miles: Average actual routing distance
            - circuity_factor: Ratio of transit to zone miles
            - avg_total_touches: Average facilities in path
            - avg_hub_touches: Average hub/hybrid touches
            - cost_per_mile: Cost per transit mile
    """
    if od_selected.empty and (direct_day is None or direct_day.empty):
        return pd.DataFrame()

    fac_lookup = get_facility_lookup(facilities)
    zone_analysis = []

    # Calculate network totals (middle-mile + direct injection)
    total_network_pkgs = 0
    total_network_cost = 0

    if not od_selected.empty:
        total_network_pkgs += od_selected['pkgs_day'].sum()
        total_network_cost += od_selected['total_cost'].sum()

    # Add direct injection to totals (if provided)
    if direct_day is not None and not direct_day.empty:
        direct_col = 'dir_pkgs_day'
        if direct_col in direct_day.columns:
            total_network_pkgs += direct_day[direct_col].sum()

    # Process Zone 0: Direct Injection (if data provided)
    if direct_day is not None and not direct_day.empty:
        direct_col = 'dir_pkgs_day'
        if direct_col in direct_day.columns:
            direct_pkgs = direct_day[direct_col].sum()

            if direct_pkgs > 0:
                zone_analysis.append({
                    'zone': 'Zone 0',
                    'zone_number': 0,
                    'total_packages': int(direct_pkgs),
                    'pct_of_total_packages': round(safe_divide(direct_pkgs, total_network_pkgs), 4),
                    'total_cost': 0.0,
                    'pct_of_total_cost': 0.0,
                    'cost_per_pkg': 0.0,
                    'linehaul_cost_per_pkg': 0.0,
                    'processing_cost_per_pkg': 0.0,
                    'avg_zone_miles': 0.0,
                    'avg_transit_miles': 0.0,
                    'circuity_factor': 0.0,
                    'avg_total_touches': 1.0,
                    'avg_hub_touches': 0.0,
                    'cost_per_mile': 0.0,
                })

    # Process Zones 1-8 from middle mile flows
    if not od_selected.empty:
        for zone_num in range(1, 9):
            zone_mask = od_selected['zone'] == zone_num
            zone_ods = od_selected[zone_mask].copy()

            if zone_ods.empty:
                continue

            # Volume metrics
            total_pkgs = zone_ods['pkgs_day'].sum()
            pct_pkgs = safe_divide(total_pkgs, total_network_pkgs)

            # Cost metrics
            total_cost = zone_ods['total_cost'].sum()
            pct_cost = safe_divide(total_cost, total_network_cost)
            cost_per_pkg = safe_divide(total_cost, total_pkgs)

            # Cost breakdown
            linehaul_cost = zone_ods['linehaul_cost'].sum()
            processing_cost = zone_ods['processing_cost'].sum()
            linehaul_per_pkg = safe_divide(linehaul_cost, total_pkgs)
            processing_per_pkg = safe_divide(processing_cost, total_pkgs)

            # Distance metrics
            zone_miles_weighted = []
            transit_miles_weighted = []

            for _, row in zone_ods.iterrows():
                pkgs = row['pkgs_day']

                # Zone miles (straight-line O-D)
                if row['origin'] in fac_lookup.index and row['dest'] in fac_lookup.index:
                    o_lat = fac_lookup.at[row['origin'], 'lat']
                    o_lon = fac_lookup.at[row['origin'], 'lon']
                    d_lat = fac_lookup.at[row['dest'], 'lat']
                    d_lon = fac_lookup.at[row['dest'], 'lon']

                    zone_miles = haversine_miles(o_lat, o_lon, d_lat, d_lon)
                    zone_miles_weighted.extend([zone_miles] * int(pkgs))

                # Transit miles (sum of all arcs in path)
                transit_miles = _calculate_path_transit_miles(
                    row, fac_lookup, mileage_bands
                )
                transit_miles_weighted.extend([transit_miles] * int(pkgs))

            avg_zone_miles = np.mean(zone_miles_weighted) if zone_miles_weighted else 0
            avg_transit_miles = np.mean(transit_miles_weighted) if transit_miles_weighted else 0
            circuity = safe_divide(avg_transit_miles, avg_zone_miles, default=1.0)

            # Touch metrics
            touch_weighted = []
            hub_touch_weighted = []

            for _, row in zone_ods.iterrows():
                pkgs = row['pkgs_day']
                path_nodes = row.get('path_nodes', [row['origin'], row['dest']])

                # Total touches
                total_touches = len(path_nodes)
                touch_weighted.extend([total_touches] * int(pkgs))

                # Hub touches (exclude launch destinations)
                hub_touches = 0
                for i, node in enumerate(path_nodes):
                    if node in fac_lookup.index:
                        node_type = fac_lookup.at[node, 'type']
                        if node_type in ['hub', 'hybrid']:
                            hub_touches += 1
                        elif i < len(path_nodes) - 1:  # Not destination
                            hub_touches += 1

                hub_touch_weighted.extend([hub_touches] * int(pkgs))

            avg_touches = np.mean(touch_weighted) if touch_weighted else 0
            avg_hub_touches = np.mean(hub_touch_weighted) if hub_touch_weighted else 0

            # Cost efficiency
            cost_per_mile = safe_divide(linehaul_cost, avg_transit_miles * total_pkgs)

            zone_analysis.append({
                'zone': f'Zone {zone_num}',
                'zone_number': zone_num,
                'total_packages': int(total_pkgs),
                'pct_of_total_packages': round(pct_pkgs, 4),
                'total_cost': round(total_cost, 2),
                'pct_of_total_cost': round(pct_cost, 4),
                'cost_per_pkg': round(cost_per_pkg, 3),
                'linehaul_cost_per_pkg': round(linehaul_per_pkg, 3),
                'processing_cost_per_pkg': round(processing_per_pkg, 3),
                'avg_zone_miles': round(avg_zone_miles, 1),
                'avg_transit_miles': round(avg_transit_miles, 1),
                'circuity_factor': round(circuity, 3),
                'avg_total_touches': round(avg_touches, 2),
                'avg_hub_touches': round(avg_hub_touches, 2),
                'cost_per_mile': round(cost_per_mile, 4),
            })

        # Process Unknown Zone (data quality flag)
        unknown_mask = od_selected['zone'] == -1
        unknown_ods = od_selected[unknown_mask].copy()

        if not unknown_ods.empty:
            total_pkgs = unknown_ods['pkgs_day'].sum()
            pct_pkgs = safe_divide(total_pkgs, total_network_pkgs)

            total_cost = unknown_ods['total_cost'].sum()
            pct_cost = safe_divide(total_cost, total_network_cost)
            cost_per_pkg = safe_divide(total_cost, total_pkgs)

            linehaul_cost = unknown_ods['linehaul_cost'].sum()
            processing_cost = unknown_ods['processing_cost'].sum()
            linehaul_per_pkg = safe_divide(linehaul_cost, total_pkgs)
            processing_per_pkg = safe_divide(processing_cost, total_pkgs)

            # Calculate metrics for unknown zone
            zone_miles_weighted = []
            transit_miles_weighted = []
            touch_weighted = []
            hub_touch_weighted = []

            for _, row in unknown_ods.iterrows():
                pkgs = row['pkgs_day']

                # Zone miles
                if row['origin'] in fac_lookup.index and row['dest'] in fac_lookup.index:
                    o_lat = fac_lookup.at[row['origin'], 'lat']
                    o_lon = fac_lookup.at[row['origin'], 'lon']
                    d_lat = fac_lookup.at[row['dest'], 'lat']
                    d_lon = fac_lookup.at[row['dest'], 'lon']

                    zone_miles = haversine_miles(o_lat, o_lon, d_lat, d_lon)
                    zone_miles_weighted.extend([zone_miles] * int(pkgs))

                # Transit miles
                transit_miles = _calculate_path_transit_miles(row, fac_lookup, mileage_bands)
                transit_miles_weighted.extend([transit_miles] * int(pkgs))

                # Touches
                path_nodes = row.get('path_nodes', [row['origin'], row['dest']])

                total_touches = len(path_nodes)
                touch_weighted.extend([total_touches] * int(pkgs))

                hub_touches = sum(
                    1 for i, node in enumerate(path_nodes)
                    if node in fac_lookup.index and
                    (fac_lookup.at[node, 'type'] in ['hub', 'hybrid'] or i < len(path_nodes) - 1)
                )
                hub_touch_weighted.extend([hub_touches] * int(pkgs))

            avg_zone_miles = np.mean(zone_miles_weighted) if zone_miles_weighted else 0
            avg_transit_miles = np.mean(transit_miles_weighted) if transit_miles_weighted else 0
            circuity = safe_divide(avg_transit_miles, avg_zone_miles, default=1.0)
            avg_touches = np.mean(touch_weighted) if touch_weighted else 0
            avg_hub_touches = np.mean(hub_touch_weighted) if hub_touch_weighted else 0
            cost_per_mile = safe_divide(linehaul_cost, avg_transit_miles * total_pkgs)

            zone_analysis.append({
                'zone': 'Unknown',
                'zone_number': 999,
                'total_packages': int(total_pkgs),
                'pct_of_total_packages': round(pct_pkgs, 4),
                'total_cost': round(total_cost, 2),
                'pct_of_total_cost': round(pct_cost, 4),
                'cost_per_pkg': round(cost_per_pkg, 3),
                'linehaul_cost_per_pkg': round(linehaul_per_pkg, 3),
                'processing_cost_per_pkg': round(processing_per_pkg, 3),
                'avg_zone_miles': round(avg_zone_miles, 1),
                'avg_transit_miles': round(avg_transit_miles, 1),
                'circuity_factor': round(circuity, 3),
                'avg_total_touches': round(avg_touches, 2),
                'avg_hub_touches': round(avg_hub_touches, 2),
                'cost_per_mile': round(cost_per_mile, 4),
            })

            print(f"  WARNING: {pct_pkgs * 100:.1f}% of packages in 'Unknown' zone")
            print(f"     Check mileage_bands coverage and facility coordinates")

    df = pd.DataFrame(zone_analysis)
    if df.empty:
        return df

    return df.sort_values('zone_number').reset_index(drop=True)


def _calculate_path_transit_miles(
        od_row: pd.Series,
        fac_lookup: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> float:
    """Calculate total transit miles for a path (sum of all arcs)."""
    path_nodes = od_row.get('path_nodes', [od_row['origin'], od_row['dest']])

    total_miles = 0.0

    for i in range(len(path_nodes) - 1):
        from_fac = path_nodes[i]
        to_fac = path_nodes[i + 1]

        if from_fac == to_fac:
            continue

        if from_fac in fac_lookup.index and to_fac in fac_lookup.index:
            f_lat = fac_lookup.at[from_fac, 'lat']
            f_lon = fac_lookup.at[from_fac, 'lon']
            t_lat = fac_lookup.at[to_fac, 'lat']
            t_lon = fac_lookup.at[to_fac, 'lon']

            raw_dist = haversine_miles(f_lat, f_lon, t_lat, t_lon)

            if raw_dist > 0:
                _, _, circuity, _ = band_lookup(raw_dist, mileage_bands)
                total_miles += raw_dist * circuity

    return total_miles


def create_zone_cost_summary_table(zone_analysis: pd.DataFrame) -> str:
    """
    Create formatted text summary of zone cost analysis.

    Useful for console output or executive summary documents.
    Includes zones 0-8 and unknown zone.
    """
    if zone_analysis.empty:
        return "No zone data available"

    lines = []
    lines.append("=" * 110)
    lines.append("ZONE COST ANALYSIS SUMMARY")
    lines.append("=" * 110)
    lines.append("")

    # Header
    lines.append(
        f"{'Zone':<12} {'Packages':<12} {'% Vol':<8} {'Cost/Pkg':<10} "
        f"{'Zone Mi':<10} {'Transit Mi':<12} {'Touches':<10}"
    )
    lines.append("-" * 110)

    # Data rows
    for _, row in zone_analysis.iterrows():
        zone_display = row['zone']

        # Special formatting for Zone 0 and Unknown
        if row['zone_number'] == 0:
            zone_display = "Zone 0 (DI)"  # Direct Injection
        elif row['zone_number'] == 999:
            zone_display = "Unknown ⚠️"

        lines.append(
            f"{zone_display:<12} "
            f"{row['total_packages']:>10,}  "
            f"{row['pct_of_total_packages']:>6.1f}%  "
            f"${row['cost_per_pkg']:>8.3f}  "
            f"{row['avg_zone_miles']:>8.1f}  "
            f"{row['avg_transit_miles']:>10.1f}  "
            f"{row['avg_total_touches']:>8.2f}"
        )

    lines.append("=" * 110)

    # Summary statistics
    total_pkgs = zone_analysis['total_packages'].sum()

    # Calculate weighted CPP (excluding zone 0 which has 0 cost)
    cost_zones = zone_analysis[zone_analysis['total_cost'] > 0]
    if not cost_zones.empty:
        weighted_cpp = (
                               cost_zones['cost_per_pkg'] * cost_zones['total_packages']
                       ).sum() / cost_zones['total_packages'].sum()
    else:
        weighted_cpp = 0

    lines.append("")
    lines.append(f"Network Total Packages: {total_pkgs:,}")
    lines.append(f"Middle-Mile Avg Cost/Pkg: ${weighted_cpp:.3f}")

    # Flag unknown zones
    unknown_row = zone_analysis[zone_analysis['zone_number'] == 999]
    if not unknown_row.empty:
        unknown_pct = unknown_row.iloc[0]['pct_of_total_packages']
        lines.append("")
        lines.append(f"⚠️  Data Quality Alert: {unknown_pct:.1f}% in Unknown zone")
        lines.append(f"   → Review mileage_bands coverage and facility coordinates")

    lines.append("")

    return "\n".join(lines)