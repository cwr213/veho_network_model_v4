"""
Container Flow Tracking Module v4.1

Updates:
1. Arc summary now uses 'trucks' column name (not 'trucks_needed')
2. Added all required columns for downstream compatibility
"""

import pandas as pd
import numpy as np
from typing import Dict, List

from .containers_v4 import (
    weighted_pkg_cube,
    get_raw_trailer_cube,
    get_containers_per_truck
)
from .geo_v4 import haversine_miles, band_lookup
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

    if sort_level == 'region':
        sort_destinations = 1
    elif sort_level == 'market':
        sort_destinations = 1
    elif sort_level == 'sort_group':
        if dest in fac_lookup.index:
            groups = fac_lookup.at[dest, 'last_mile_sort_groups_count']
            if pd.isna(groups) or groups <= 0:
                groups = OptimizationConstants.DEFAULT_SORT_GROUPS
            sort_destinations = int(groups)
        else:
            sort_destinations = OptimizationConstants.DEFAULT_SORT_GROUPS
    else:
        sort_destinations = 1

    cube_per_destination = total_cube / sort_destinations
    containers_per_destination = max(1, int(np.ceil(
        cube_per_destination / effective_container_cube
    )))

    total_containers = containers_per_destination * sort_destinations

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


def build_od_container_map(
        od_selected: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        facilities: pd.DataFrame
) -> pd.DataFrame:
    """Add container counts to all OD pairs."""
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


def calculate_arc_containers_from_flows(
        arc_od_flows: List[Dict],
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        facilities: pd.DataFrame
) -> Dict:
    """Calculate containers on an arc by summing from all OD flows."""
    containers_per_truck = get_containers_per_truck(container_params)
    raw_trailer_cube = get_raw_trailer_cube(container_params)
    w_cube = weighted_pkg_cube(package_mix)

    total_containers = 0
    total_cube = 0
    total_packages = 0
    container_fill_rates = []

    for flow in arc_od_flows:
        container_data = calculate_origin_containers_by_sort_level(
            origin=flow['origin'],
            dest=flow['dest'],
            packages=flow['packages'],
            sort_level=flow['sort_level'],
            package_mix=package_mix,
            container_params=container_params,
            facilities=facilities
        )

        total_containers += container_data['containers']
        total_cube += flow['packages'] * w_cube
        total_packages += flow['packages']
        container_fill_rates.append(container_data['container_fill_rate'])

    trucks_needed = max(1, int(np.ceil(total_containers / containers_per_truck)))
    truck_fill_rate = safe_divide(total_cube, trucks_needed * raw_trailer_cube)
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


def recalculate_arc_summary_with_container_flow(
        od_selected: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> pd.DataFrame:
    """
    Rebuild arc summary with correct container flow tracking.

    FIXED: Now includes all required columns with correct names.
    """
    od_with_containers = build_od_container_map(
        od_selected, package_mix, container_params, facilities
    )

    # Get facility coordinates for distance calculations
    fac = facilities.set_index('facility_name')[['lat', 'lon']].astype(float)

    arc_flows = {}

    for _, od_row in od_with_containers.iterrows():
        path_nodes = od_row.get('path_nodes', [od_row['origin'], od_row['dest']])

        if not isinstance(path_nodes, (list, tuple)):
            path_nodes = [od_row['origin'], od_row['dest']]
        if isinstance(path_nodes, tuple):
            path_nodes = list(path_nodes)

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

    corrected_arcs = []

    for arc_key, od_flows_list in arc_flows.items():
        from_fac, to_fac = arc_key

        # Calculate arc metrics
        arc_metrics = calculate_arc_containers_from_flows(
            od_flows_list, package_mix, container_params, facilities
        )

        # Calculate distance and cost
        if from_fac == to_fac:
            distance_miles = 0.0
            cost_per_truck = 0.0
        elif from_fac in fac.index and to_fac in fac.index:
            lat1, lon1 = fac.at[from_fac, 'lat'], fac.at[from_fac, 'lon']
            lat2, lon2 = fac.at[to_fac, 'lat'], fac.at[to_fac, 'lon']

            raw_dist = haversine_miles(lat1, lon1, lat2, lon2)
            fixed, var, circuity, _ = band_lookup(raw_dist, mileage_bands)
            distance_miles = raw_dist * circuity
            cost_per_truck = fixed + var * distance_miles
        else:
            distance_miles = 0.0
            cost_per_truck = 0.0

        total_cost = arc_metrics['trucks_needed'] * cost_per_truck
        packages_per_truck = safe_divide(
            arc_metrics['total_packages'],
            arc_metrics['trucks_needed']
        )
        cube_per_truck = safe_divide(
            arc_metrics['total_cube'],
            arc_metrics['trucks_needed']
        )

        corrected_arcs.append({
            'from_facility': from_fac,
            'to_facility': to_fac,
            'distance_miles': round(distance_miles, 1),
            'pkgs_day': int(arc_metrics['total_packages']),
            'pkg_cube_cuft': round(arc_metrics['total_cube'], 2),
            'trucks': arc_metrics['trucks_needed'],  # FIXED: Use 'trucks' not 'trucks_needed'
            'physical_containers': arc_metrics['total_containers'],
            'packages_per_truck': round(packages_per_truck, 1),
            'cube_per_truck': round(cube_per_truck, 1),
            'container_fill_rate': round(arc_metrics['container_fill_rate'], 3),
            'truck_fill_rate': round(arc_metrics['truck_fill_rate'], 3),
            'cost_per_truck': round(cost_per_truck, 2),
            'total_cost': round(total_cost, 2),
            'CPP': round(safe_divide(total_cost, arc_metrics['total_packages']), 4),
            'num_od_flows': len(od_flows_list)
        })

    return pd.DataFrame(corrected_arcs)


def analyze_sort_level_container_impact(
        od_selected: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        facilities: pd.DataFrame
) -> pd.DataFrame:
    """Analyze how sort level choice impacts container count and fill rates."""
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
    """Create diagnostic report comparing original vs corrected fill rates."""
    lines = []
    lines.append("=" * 100)
    lines.append("CONTAINER FLOW DIAGNOSTIC - Original vs Corrected Fill Rates")
    lines.append("=" * 100)
    lines.append("")

    orig_avg_fill = arc_summary_original[
        'truck_fill_rate'].mean() if 'truck_fill_rate' in arc_summary_original.columns else 0
    corr_avg_fill = arc_summary_corrected['truck_fill_rate'].mean()

    lines.append(f"Network Average Truck Fill Rate:")
    lines.append(f"  Original (WRONG): {orig_avg_fill:.1%}")
    lines.append(f"  Corrected:        {corr_avg_fill:.1%}")
    lines.append(f"  Difference:       {(corr_avg_fill - orig_avg_fill):.1%}")
    lines.append("")

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

    return "\n".join(lines)