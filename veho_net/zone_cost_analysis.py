"""
Zone Cost Analysis Module

Add this to veho_net/ directory to calculate per-zone cost metrics.
Integrates with existing reporting pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

from .geo_v4 import haversine_miles, band_lookup
from .utils import safe_divide, get_facility_lookup


def calculate_zone_cost_analysis(
        od_selected: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate comprehensive cost analysis by zone.

    Provides executive-level view of network economics by distance tier.

    Args:
        od_selected: Selected OD paths with cost breakdown
        facilities: Facility master data
        mileage_bands: Mileage bands for distance calculations

    Returns:
        DataFrame with columns:
            - zone: Zone identifier
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
    if od_selected.empty:
        return pd.DataFrame()

    fac_lookup = get_facility_lookup(facilities)
    zone_analysis = []

    # Network totals for percentage calculations
    total_network_pkgs = od_selected['pkgs_day'].sum()
    total_network_cost = od_selected['total_cost'].sum()

    for zone_num in range(1, 9):
        zone_str = str(zone_num)
        zone_ods = od_selected[od_selected['zone'] == zone_str].copy()

        if zone_ods.empty:
            continue

        # Volume metrics
        total_pkgs = zone_ods['pkgs_day'].sum()
        pct_pkgs = safe_divide(total_pkgs, total_network_pkgs) * 100

        # Cost metrics
        total_cost = zone_ods['total_cost'].sum()
        pct_cost = safe_divide(total_cost, total_network_cost) * 100
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

            if not isinstance(path_nodes, (list, tuple)):
                path_nodes = [row['origin'], row['dest']]

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
            'pct_of_total_packages': round(pct_pkgs, 2),
            'total_cost': round(total_cost, 2),
            'pct_of_total_cost': round(pct_cost, 2),
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

    return pd.DataFrame(zone_analysis).sort_values('zone_number')


def _calculate_path_transit_miles(
        od_row: pd.Series,
        fac_lookup: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> float:
    """Calculate total transit miles for a path (sum of all arcs)."""
    path_nodes = od_row.get('path_nodes', [od_row['origin'], od_row['dest']])

    if not isinstance(path_nodes, (list, tuple)):
        path_nodes = [od_row['origin'], od_row['dest']]

    if isinstance(path_nodes, tuple):
        path_nodes = list(path_nodes)

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
    """
    if zone_analysis.empty:
        return "No zone data available"

    lines = []
    lines.append("=" * 100)
    lines.append("ZONE COST ANALYSIS SUMMARY")
    lines.append("=" * 100)
    lines.append("")

    # Header
    lines.append(
        f"{'Zone':<10} {'Packages':<12} {'% Vol':<8} {'Cost/Pkg':<10} "
        f"{'Zone Mi':<10} {'Transit Mi':<12} {'Touches':<10}"
    )
    lines.append("-" * 100)

    # Data rows
    for _, row in zone_analysis.iterrows():
        lines.append(
            f"{row['zone']:<10} "
            f"{row['total_packages']:>10,}  "
            f"{row['pct_of_total_packages']:>6.1f}%  "
            f"${row['cost_per_pkg']:>8.3f}  "
            f"{row['avg_zone_miles']:>8.1f}  "
            f"{row['avg_transit_miles']:>10.1f}  "
            f"{row['avg_total_touches']:>8.2f}"
        )

    lines.append("=" * 100)

    # Summary statistics
    total_pkgs = zone_analysis['total_packages'].sum()
    weighted_cpp = (
                           zone_analysis['cost_per_pkg'] * zone_analysis['total_packages']
                   ).sum() / total_pkgs

    lines.append("")
    lines.append(f"Network Total Packages: {total_pkgs:,}")
    lines.append(f"Network Avg Cost/Pkg: ${weighted_cpp:.3f}")
    lines.append("")

    return "\n".join(lines)