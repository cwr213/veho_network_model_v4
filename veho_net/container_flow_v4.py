"""
Container Flow Tracking Module v4

CRITICAL FIX: Track containers from origin through network to get accurate fill rates.

Problem: Current model recalculates containers at each arc, ignoring that containers
are physical objects created once at origin based on sort level.

Impact: More granular sorting creates MORE containers, leading to lower fill rates.
- Region sort: Consolidates to 1 destination → fewer containers
- Market sort: 1 container set per destination → moderate containers
- Sort_group sort: Multiple sets per destination → most containers

This module provides corrected calculations for MILP and reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from .containers_v4 import (
    weighted_pkg_cube,
    get_raw_trailer_cube,
    get_containers_per_truck
)
from .utils import safe_divide, get_facility_lookup
from .config_v4 import OptimizationConstants


def calculate_origin_containers_by_sort_level(
        origin: str,
        dest: str,
        packages: float,
        sort_level: str,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        facilities: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate containers created at origin based on chosen sort level.

    KEY INSIGHT: Containers are created ONCE at origin and persist through network.
    Sort level determines container multiplication factor.

    Args:
        origin: Origin facility name
        dest: Destination facility name
        packages: Package volume
        sort_level: Sort level chosen (region/market/sort_group)
        package_mix: Package distribution
        container_params: Container parameters
        facilities: Facility master

    Returns:
        Dict with:
            - containers: Physical container count
            - sort_destinations: Number of sort destinations created
            - containers_per_destination: Containers per destination
            - avg_pkgs_per_container: Average packages per container
            - container_fill_rate: Fill rate of containers
            - total_cube: Total cube
    """
    w_cube = weighted_pkg_cube(package_mix)
    total_cube = packages * w_cube

    gaylord_row = container_params[
        container_params["container_type"].str.lower() == "gaylord"
        ].iloc[0]

    raw_container_cube = float(gaylord_row["usable_cube_cuft"])
    pack_util = float(gaylord_row["pack_utilization_container"])
    effective_container_cube = raw_container_cube * pack_util

    fac_lookup = get_facility_lookup(facilities)

    # Determine sort destinations based on sort level
    if sort_level == 'region':
        # Sorting to regional hub - one set of containers for the region
        sort_destinations = 1

    elif sort_level == 'market':
        # Sorting to individual destination - one set per market
        sort_destinations = 1

    elif sort_level == 'sort_group':
        # Sorting to granular sort groups within destination
        if dest in fac_lookup.index:
            groups = fac_lookup.at[dest, 'last_mile_sort_groups_count']
            if pd.isna(groups) or groups <= 0:
                groups = OptimizationConstants.DEFAULT_SORT_GROUPS
            sort_destinations = int(groups)
        else:
            sort_destinations = OptimizationConstants.DEFAULT_SORT_GROUPS
    else:
        sort_destinations = 1

    # Calculate containers PER sort destination
    cube_per_destination = total_cube / sort_destinations
    containers_per_destination = max(1, int(np.ceil(
        cube_per_destination / effective_container_cube
    )))

    # Total physical containers
    total_containers = containers_per_destination * sort_destinations

    # Average fill rate across containers
    avg_fill_rate = safe_divide(
        cube_per_destination,
        containers_per_destination * raw_container_cube
    )

    return {
        'containers': total_containers,
        'sort_destinations': sort_destinations,
        'containers_per_destination': containers_per_destination,
        'avg_pkgs_per_container': safe_divide(packages, total_containers),
        'container_fill_rate': min(1.0, avg_fill_rate),
        'total_cube': total_cube
    }


def calculate_arc_containers_from_flows(
        arc_od_flows: List[Dict],
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        facilities: pd.DataFrame
) -> Dict:
    """
    Calculate containers on an arc by summing from all OD flows.

    CORRECT METHOD: Sum containers already created at origins.
    WRONG METHOD: Recalculate based on aggregate packages (current model).

    Args:
        arc_od_flows: List of dicts with keys:
            - origin, dest, packages, sort_level
        package_mix: Package distribution
        container_params: Container parameters
        facilities: Facility master

    Returns:
        Dict with:
            - total_containers: Sum of containers from all OD flows
            - trucks_needed: Trucks required for containers
            - truck_fill_rate: Actual truck utilization
            - container_fill_rate: Average container utilization
            - total_packages: Total packages on arc
            - total_cube: Total cube on arc
    """
    containers_per_truck = get_containers_per_truck(container_params)
    raw_trailer_cube = get_raw_trailer_cube(container_params)
    w_cube = weighted_pkg_cube(package_mix)

    total_containers = 0
    total_cube = 0
    total_packages = 0
    container_fill_rates = []

    # Sum containers from each OD flow
    for flow in arc_od_flows:
        # Calculate containers created at origin for this OD
        container_data = calculate_origin_containers_by_sort_level(
            origin=flow['origin'],
            dest=flow['dest'],
            packages=flow['packages'],
            sort_level=flow['sort_level'],
            package_mix=package_mix,
            container_params=container_params,
            facilities=facilities
        )

        # These containers flow through this arc
        total_containers += container_data['containers']
        total_cube += flow['packages'] * w_cube
        total_packages += flow['packages']
        container_fill_rates.append(container_data['container_fill_rate'])

    # Trucks needed based on physical container count
    trucks_needed = max(1, int(np.ceil(total_containers / containers_per_truck)))

    # True truck fill rate (cube vs. available trailer space)
    truck_fill_rate = safe_divide(total_cube, trucks_needed * raw_trailer_cube)

    # Average container fill rate
    avg_container_fill = np.mean(container_fill_rates) if container_fill_rates else 0

    return {
        'total_containers': total_containers,
        'trucks_needed': trucks_needed,
        'truck_fill_rate': min(1.0, truck_fill_rate),
        'container_fill_rate': avg_container_fill,
        'total_packages': total_packages,
        'total_cube': total_cube,
        'avg_containers_per_truck': safe_divide(total_containers, trucks_needed)
    }


def build_od_container_map(
        od_selected: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        facilities: pd.DataFrame
) -> pd.DataFrame:
    """
    Build container counts for all OD pairs.

    Returns od_selected with additional columns:
        - origin_containers: Containers created at origin
        - sort_destinations: Number of sort destinations
        - container_fill_rate: Fill rate of containers
    """
    od_with_containers = od_selected.copy()

    container_data = []

    for _, row in od_selected.iterrows():
        containers = calculate_origin_containers_by_sort_level(
            origin=row['origin'],
            dest=row['dest'],
            packages=row['pkgs_day'],
            sort_level=row.get('chosen_sort_level', 'market'),
            package_mix=package_mix,
            container_params=container_params,
            facilities=facilities
        )

        container_data.append(containers)

    od_with_containers['origin_containers'] = [d['containers'] for d in container_data]
    od_with_containers['sort_destinations'] = [d['sort_destinations'] for d in container_data]
    od_with_containers['container_fill_rate'] = [d['container_fill_rate'] for d in container_data]

    return od_with_containers


def recalculate_arc_summary_with_container_flow(
        od_selected: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        facilities: pd.DataFrame
) -> pd.DataFrame:
    """
    Rebuild arc summary with correct container flow tracking.

    This REPLACES the flawed arc summary from MILP output.

    Args:
        od_selected: Selected OD paths with sort levels
        package_mix: Package distribution
        container_params: Container parameters
        facilities: Facility master

    Returns:
        Corrected arc summary DataFrame with accurate container counts and fill rates
    """
    # First, add container data to OD selected
    od_with_containers = build_od_container_map(
        od_selected, package_mix, container_params, facilities
    )

    # Group OD flows by arc
    arc_flows = {}  # (from, to) -> [flow_dicts]

    for _, od_row in od_with_containers.iterrows():
        path_nodes = od_row.get('path_nodes', [od_row['origin'], od_row['dest']])

        if not isinstance(path_nodes, (list, tuple)):
            path_nodes = [od_row['origin'], od_row['dest']]

        if isinstance(path_nodes, tuple):
            path_nodes = list(path_nodes)

        # Each arc in path gets this OD's flow
        for i in range(len(path_nodes) - 1):
            from_fac = path_nodes[i]
            to_fac = path_nodes[i + 1]

            arc_key = (from_fac, to_fac)

            if arc_key not in arc_flows:
                arc_flows[arc_key] = []

            arc_flows[arc_key].append({
                'origin': od_row['origin'],
                'dest': od_row['dest'],
                'packages': od_row['pkgs_day'],
                'sort_level': od_row.get('chosen_sort_level', 'market')
            })

    # Calculate corrected metrics for each arc
    corrected_arcs = []

    for arc_key, od_flows in arc_flows.items():
        from_fac, to_fac = arc_key

        # Calculate with container flow
        arc_metrics = calculate_arc_containers_from_flows(
            od_flows, package_mix, container_params, facilities
        )

        corrected_arcs.append({
            'from_facility': from_fac,
            'to_facility': to_fac,
            'total_packages': int(arc_metrics['total_packages']),
            'total_containers': arc_metrics['total_containers'],
            'trucks_needed': arc_metrics['trucks_needed'],
            'truck_fill_rate': round(arc_metrics['truck_fill_rate'], 3),
            'container_fill_rate': round(arc_metrics['container_fill_rate'], 3),
            'total_cube': round(arc_metrics['total_cube'], 2),
            'avg_containers_per_truck': round(arc_metrics['avg_containers_per_truck'], 1),
            'num_od_flows': len(od_flows)
        })

    return pd.DataFrame(corrected_arcs)


def analyze_sort_level_container_impact(
        od_selected: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        facilities: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze how sort level choice impacts container count and fill rates.

    Reveals the hidden cost of granular sorting.

    Returns:
        DataFrame showing per sort level:
            - num_od_pairs: Count of OD pairs
            - total_packages: Total package volume
            - total_containers: Total containers created
            - avg_containers_per_od: Average containers per OD
            - avg_container_fill_rate: Average container utilization
            - packages_per_container: Efficiency metric
    """
    analysis = []

    for sort_level in ['region', 'market', 'sort_group']:
        level_ods = od_selected[
            od_selected.get('chosen_sort_level', 'market') == sort_level
            ]

        if level_ods.empty:
            continue

        total_pkgs = 0
        total_containers = 0
        fill_rates = []

        for _, od_row in level_ods.iterrows():
            container_data = calculate_origin_containers_by_sort_level(
                origin=od_row['origin'],
                dest=od_row['dest'],
                packages=od_row['pkgs_day'],
                sort_level=sort_level,
                package_mix=package_mix,
                container_params=container_params,
                facilities=facilities
            )

            total_pkgs += od_row['pkgs_day']
            total_containers += container_data['containers']
            fill_rates.append(container_data['container_fill_rate'])

        num_ods = len(level_ods)

        analysis.append({
            'sort_level': sort_level,
            'num_od_pairs': num_ods,
            'total_packages': int(total_pkgs),
            'total_containers': total_containers,
            'avg_containers_per_od': round(safe_divide(total_containers, num_ods), 1),
            'avg_container_fill_rate': round(np.mean(fill_rates) if fill_rates else 0, 3),
            'packages_per_container': round(safe_divide(total_pkgs, total_containers), 1)
        })

    return pd.DataFrame(analysis)


def create_container_flow_diagnostic(
        od_selected: pd.DataFrame,
        arc_summary_original: pd.DataFrame,
        arc_summary_corrected: pd.DataFrame
) -> str:
    """
    Create diagnostic report comparing original vs corrected fill rates.

    Shows impact of container flow fix.
    """
    lines = []
    lines.append("=" * 100)
    lines.append("CONTAINER FLOW DIAGNOSTIC - Original vs Corrected Fill Rates")
    lines.append("=" * 100)
    lines.append("")

    # Overall comparison
    orig_avg_fill = arc_summary_original[
        'truck_fill_rate'].mean() if 'truck_fill_rate' in arc_summary_original.columns else 0
    corr_avg_fill = arc_summary_corrected['truck_fill_rate'].mean()

    lines.append(f"Network Average Truck Fill Rate:")
    lines.append(f"  Original (WRONG): {orig_avg_fill:.1%}")
    lines.append(f"  Corrected:        {corr_avg_fill:.1%}")
    lines.append(f"  Difference:       {(corr_avg_fill - orig_avg_fill):.1%} ⚠️")
    lines.append("")

    # Sort level impact
    lines.append("Sort Level Impact on Containers:")
    lines.append("")

    for sort_level in ['region', 'market', 'sort_group']:
        level_ods = od_selected[od_selected.get('chosen_sort_level', 'market') == sort_level]
        if not level_ods.empty:
            total_pkgs = level_ods['pkgs_day'].sum()
            total_containers = level_ods['origin_containers'].sum() if 'origin_containers' in level_ods.columns else 0

            lines.append(f"  {sort_level.title()}:")
            lines.append(f"    Packages:    {total_pkgs:>10,.0f}")
            lines.append(f"    Containers:  {total_containers:>10,.0f}")
            lines.append(f"    Pkgs/Cont:   {safe_divide(total_pkgs, total_containers):>10.1f}")
            lines.append("")

    lines.append("=" * 100)
    lines.append("")
    lines.append("💡 KEY INSIGHT: Sort_group creates significantly more containers than region sort,")
    lines.append("   leading to lower truck fill rates and higher linehaul costs.")
    lines.append("")
    lines.append("=" * 100)

    return "\n".join(lines)