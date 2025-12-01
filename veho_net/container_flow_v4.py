"""
Container Flow Tracking Module

Tracks physical container flows through the network with proper persistence logic.
Containers are created at origin and sort operations, persisted through crossdock.

Key Concepts:
- Container Creation Point: Facility where containers are built (origin or sort hub)
- Persisted Containers: Pass through crossdock unchanged
- Fresh Containers: Created at sort operations (may consolidate cross-OD)

Business Rules:
- Origin always creates containers based on sort level
- Crossdock: containers pass through unchanged (partials remain partial)
- Sort (region level at intermediate): containers broken down, packages re-sorted
- Cross-OD consolidation only occurs at sort operations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict

from .containers_v4 import (
    weighted_pkg_cube,
    get_raw_trailer_cube,
    get_containers_per_truck,
    get_container_capacity
)
from .geo_v4 import haversine_miles, band_lookup
from .utils import safe_divide, get_facility_lookup, extract_path_nodes
from .config_v4 import OptimizationConstants


# ============================================================================
# CONTAINER CREATION POINT IDENTIFICATION
# ============================================================================

def identify_container_creation_points(
        path_nodes: List[str],
        sort_level: str,
        facilities: pd.DataFrame
) -> List[str]:
    """
    Identify facilities where containers are created or recreated.

    Rules:
    - Origin ALWAYS creates containers
    - Region sort: intermediate hubs perform sort → create new containers
    - Market/sort_group: intermediates are crossdock → containers persist

    Args:
        path_nodes: List of facility names in path order
        sort_level: 'region', 'market', or 'sort_group'
        facilities: Facility master data

    Returns:
        List of facility names where containers are created
    """
    if not path_nodes:
        return []

    creation_points = [path_nodes[0]]  # Origin always creates

    if sort_level == 'region' and len(path_nodes) > 2:
        # Region sort: each intermediate hub performs sort operation
        # Containers are broken down and rebuilt for next leg
        for node in path_nodes[1:-1]:
            creation_points.append(node)

    # Market and sort_group: only origin creates
    # All intermediates are crossdock - containers persist

    return creation_points


def get_operation_type_at_facility(
        facility: str,
        path_nodes: List[str],
        sort_level: str,
        origin: str,
        dest: str
) -> str:
    """
    Determine what operation type occurs at a facility for a given OD flow.

    Returns:
        'origin' - container creation at injection point
        'sort' - container breakdown and recreation
        'crossdock' - container pass-through (no breakdown)
        'destination' - final delivery point
    """
    if facility == origin and facility == path_nodes[0]:
        return 'origin'

    if facility == dest and facility == path_nodes[-1]:
        return 'destination'

    # Intermediate facility
    if facility in path_nodes[1:-1]:
        if sort_level == 'region':
            return 'sort'
        else:
            return 'crossdock'

    return 'unknown'


# ============================================================================
# CONTAINER CALCULATION AT CREATION POINTS
# ============================================================================

def calculate_containers_at_creation(
        packages: float,
        sort_level: str,
        dest: str,
        creation_facility: str,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        facilities: pd.DataFrame
) -> Dict:
    """
    Calculate containers created at a specific creation point.

    At origin: containers based on sort level (region/market/sort_group)
    At sort hub: containers based on downstream destination

    Args:
        packages: Package volume
        sort_level: Sort level for this OD flow
        dest: Final destination facility
        creation_facility: Where containers are being created
        package_mix: Package mix distribution
        container_params: Container parameters
        facilities: Facility master data

    Returns:
        Dict with container metrics
    """
    w_cube = weighted_pkg_cube(package_mix)
    total_cube = packages * w_cube

    gaylord_row = container_params[
        container_params["container_type"].str.lower() == "gaylord"
    ].iloc[0]

    raw_container_cube = float(gaylord_row["usable_cube_cuft"])
    pack_util_container = float(gaylord_row["pack_utilization_container"])
    effective_container_cube = raw_container_cube * pack_util_container

    fac_lookup = get_facility_lookup(facilities)

    # Determine sort destinations based on sort level and creation point
    if sort_level == 'region':
        # At origin: sorting to regional hub (1 destination)
        # At intermediate sort: sorting to next downstream (1 destination)
        sort_destinations = 1

    elif sort_level == 'market':
        # Sorting to specific destination facility (1 destination)
        sort_destinations = 1

    elif sort_level == 'sort_group':
        # Sorting to route groups within destination
        if dest not in fac_lookup.index:
            raise ValueError(f"Destination '{dest}' not found in facilities")

        groups = fac_lookup.at[dest, 'last_mile_sort_groups_count']
        if pd.isna(groups) or groups <= 0:
            raise ValueError(f"Destination '{dest}' missing last_mile_sort_groups_count")
        sort_destinations = int(groups)

    else:
        raise ValueError(f"Invalid sort_level: {sort_level}")

    # Calculate containers
    cube_per_destination = total_cube / sort_destinations
    containers_per_destination = max(1, int(np.ceil(
        cube_per_destination / effective_container_cube
    )))
    total_containers = containers_per_destination * sort_destinations

    # Fill rate (actual cube / raw capacity)
    container_fill_rate = safe_divide(
        cube_per_destination,
        containers_per_destination * raw_container_cube
    )

    return {
        'containers': total_containers,
        'sort_destinations': sort_destinations,
        'container_fill_rate': min(1.0, container_fill_rate),
        'total_cube': total_cube,
        'creation_facility': creation_facility,
        'avg_pkgs_per_container': safe_divide(packages, total_containers)
    }


# ============================================================================
# OD-LEVEL CONTAINER TRACKING WITH PERSISTENCE
# ============================================================================

def build_od_container_map_with_persistence(
        od_selected: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        facilities: pd.DataFrame
) -> pd.DataFrame:
    """
    Build container tracking for all OD pairs with persistence through crossdock.

    For each OD, tracks:
    - Container creation points
    - Containers on each arc (persisted or fresh)
    - Container fill rates at creation

    Args:
        od_selected: Selected OD paths from optimization
        package_mix: Package mix distribution
        container_params: Container parameters
        facilities: Facility master data

    Returns:
        DataFrame with added container tracking columns:
        - creation_points: List of facilities where containers created
        - origin_containers: Containers created at origin
        - origin_container_fill: Fill rate at origin
        - arc_containers: Dict mapping (from, to) -> container count
        - arc_container_type: Dict mapping (from, to) -> 'persisted' or 'fresh'
    """
    od_with_containers = od_selected.copy()

    # Initialize tracking columns
    creation_points_list = []
    origin_containers_list = []
    origin_fill_list = []
    arc_containers_list = []
    arc_container_type_list = []

    for idx, row in od_selected.iterrows():
        path_nodes = extract_path_nodes(row)
        sort_level = row.get('chosen_sort_level', 'market')
        origin = row['origin']
        dest = row['dest']
        packages = row['pkgs_day']

        # Identify where containers are created
        creation_points = identify_container_creation_points(
            path_nodes, sort_level, facilities
        )

        # Calculate containers at origin
        origin_container_data = calculate_containers_at_creation(
            packages=packages,
            sort_level=sort_level,
            dest=dest,
            creation_facility=origin,
            package_mix=package_mix,
            container_params=container_params,
            facilities=facilities
        )

        # Track containers per arc
        arc_containers = {}
        arc_container_type = {}

        current_containers = origin_container_data['containers']
        current_fill = origin_container_data['container_fill_rate']
        last_creation_point = origin

        for i in range(len(path_nodes) - 1):
            from_fac = path_nodes[i]
            to_fac = path_nodes[i + 1]
            arc_key = (from_fac, to_fac)

            # Check if this is a new creation point (sort operation)
            if from_fac in creation_points and from_fac != origin:
                # Sort operation: recalculate containers
                # For region sort, we're now sorting to next downstream
                sort_data = calculate_containers_at_creation(
                    packages=packages,
                    sort_level='market',  # After region sort, it's effectively market-sorted
                    dest=dest,
                    creation_facility=from_fac,
                    package_mix=package_mix,
                    container_params=container_params,
                    facilities=facilities
                )
                current_containers = sort_data['containers']
                current_fill = sort_data['container_fill_rate']
                last_creation_point = from_fac
                arc_container_type[arc_key] = 'fresh'
            else:
                # Crossdock or origin: containers persist
                if from_fac == origin:
                    arc_container_type[arc_key] = 'origin'
                else:
                    arc_container_type[arc_key] = 'persisted'

            arc_containers[arc_key] = current_containers

        creation_points_list.append(creation_points)
        origin_containers_list.append(origin_container_data['containers'])
        origin_fill_list.append(origin_container_data['container_fill_rate'])
        arc_containers_list.append(arc_containers)
        arc_container_type_list.append(arc_container_type)

    od_with_containers['creation_points'] = creation_points_list
    od_with_containers['origin_containers'] = origin_containers_list
    od_with_containers['origin_container_fill'] = origin_fill_list
    od_with_containers['arc_containers'] = arc_containers_list
    od_with_containers['arc_container_type'] = arc_container_type_list

    return od_with_containers


# ============================================================================
# ARC-LEVEL AGGREGATION WITH CROSS-OD CONSOLIDATION
# ============================================================================

def aggregate_arc_containers_with_consolidation(
        od_with_containers: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        facilities: pd.DataFrame
) -> Dict[Tuple[str, str], Dict]:
    """
    Aggregate containers per arc, handling cross-OD consolidation at sort operations.

    Key logic:
    - Persisted containers: sum from individual OD flows (no consolidation)
    - Fresh containers at sort: aggregate packages first, then calculate containers
      (enables cross-OD consolidation)

    Args:
        od_with_containers: OD data with container tracking
        package_mix: Package mix distribution
        container_params: Container parameters
        facilities: Facility master data

    Returns:
        Dict mapping (from_fac, to_fac) -> arc metrics dict
    """
    # Group flows by arc
    arc_flows = defaultdict(list)

    for idx, row in od_with_containers.iterrows():
        arc_containers = row['arc_containers']
        arc_container_type = row['arc_container_type']

        for arc_key, containers in arc_containers.items():
            arc_flows[arc_key].append({
                'origin': row['origin'],
                'dest': row['dest'],
                'packages': row['pkgs_day'],
                'sort_level': row.get('chosen_sort_level', 'market'),
                'containers': containers,
                'container_type': arc_container_type.get(arc_key, 'unknown'),
                'container_fill': row['origin_container_fill']
            })

    # Aggregate each arc
    arc_aggregates = {}
    w_cube = weighted_pkg_cube(package_mix)

    for arc_key, flows in arc_flows.items():
        from_fac, to_fac = arc_key

        # Separate persisted vs fresh flows
        persisted_flows = [f for f in flows if f['container_type'] in ('origin', 'persisted')]
        fresh_flows = [f for f in flows if f['container_type'] == 'fresh']

        # Persisted containers: sum directly (no consolidation possible)
        persisted_containers = sum(f['containers'] for f in persisted_flows)
        persisted_packages = sum(f['packages'] for f in persisted_flows)
        persisted_cube = persisted_packages * w_cube

        # Weighted average fill rate for persisted
        if persisted_containers > 0:
            persisted_fill = sum(
                f['container_fill'] * f['containers'] for f in persisted_flows
            ) / persisted_containers
        else:
            persisted_fill = 0.0

        # Fresh containers at sort: consolidate by destination, then calculate
        # This enables cross-OD consolidation
        fresh_containers = 0
        fresh_packages = 0
        fresh_fills = []

        if fresh_flows:
            # Group fresh flows by destination (can consolidate same-dest packages)
            dest_packages = defaultdict(float)
            for f in fresh_flows:
                dest_packages[f['dest']] += f['packages']
                fresh_packages += f['packages']

            # Calculate consolidated containers per destination
            for dest, pkgs in dest_packages.items():
                container_data = calculate_containers_at_creation(
                    packages=pkgs,
                    sort_level='market',  # Post-sort is effectively market-sorted
                    dest=dest,
                    creation_facility=from_fac,
                    package_mix=package_mix,
                    container_params=container_params,
                    facilities=facilities
                )
                fresh_containers += container_data['containers']
                fresh_fills.append(container_data['container_fill_rate'])

        fresh_cube = fresh_packages * w_cube
        fresh_fill = np.mean(fresh_fills) if fresh_fills else 0.0

        # Total arc metrics
        total_containers = persisted_containers + fresh_containers
        total_packages = persisted_packages + fresh_packages
        total_cube = persisted_cube + fresh_cube

        arc_aggregates[arc_key] = {
            'from_facility': from_fac,
            'to_facility': to_fac,
            'total_containers': total_containers,
            'persisted_containers': persisted_containers,
            'fresh_containers': fresh_containers,
            'total_packages': total_packages,
            'total_cube': total_cube,
            'persisted_container_fill': persisted_fill,
            'fresh_container_fill': fresh_fill,
            'num_od_flows': len(flows)
        }

    return arc_aggregates


# ============================================================================
# ARC SUMMARY RECALCULATION
# ============================================================================

def recalculate_arc_summary_with_container_flow(
        od_selected: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> pd.DataFrame:
    """
    Rebuild arc summary with CORRECT container persistence logic.

    Key changes from original:
    - Containers persist through crossdock (not recalculated)
    - Fresh containers only at sort operations
    - Cross-OD consolidation at sort points
    - Truck fill rate based on physical container count

    Args:
        od_selected: Selected OD paths from optimization
        package_mix: Package mix distribution
        container_params: Container parameters
        facilities: Facility master data
        mileage_bands: Mileage bands for cost calculation

    Returns:
        DataFrame with arc-level summary including container tracking
    """
    # Step 1: Build OD-level container tracking
    print("    Building OD container map with persistence...")
    od_with_containers = build_od_container_map_with_persistence(
        od_selected, package_mix, container_params, facilities
    )

    # Step 2: Aggregate to arc level with consolidation logic
    print("    Aggregating containers per arc...")
    arc_aggregates = aggregate_arc_containers_with_consolidation(
        od_with_containers, package_mix, container_params, facilities
    )

    print(f"    Aggregated {len(arc_aggregates)} unique arcs")

    # Step 3: Build arc summary with truck calculations
    fac = facilities.set_index('facility_name')[['lat', 'lon']].astype(float)
    containers_per_truck = get_containers_per_truck(container_params)
    raw_trailer_cube = get_raw_trailer_cube(container_params)

    arc_rows = []

    for arc_key, arc_data in arc_aggregates.items():
        from_fac = arc_data['from_facility']
        to_fac = arc_data['to_facility']

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

        # Calculate trucks needed based on PHYSICAL containers
        total_containers = arc_data['total_containers']
        trucks_needed = max(1, int(np.ceil(total_containers / containers_per_truck)))

        # Truck fill rate = actual cube / truck capacity
        total_cube = arc_data['total_cube']
        truck_fill_rate = safe_divide(
            total_cube,
            trucks_needed * raw_trailer_cube
        )

        # Weighted average container fill rate
        persisted = arc_data['persisted_containers']
        fresh = arc_data['fresh_containers']
        if total_containers > 0:
            avg_container_fill = (
                arc_data['persisted_container_fill'] * persisted +
                arc_data['fresh_container_fill'] * fresh
            ) / total_containers
        else:
            avg_container_fill = 0.0

        total_cost = trucks_needed * cost_per_truck
        total_packages = arc_data['total_packages']

        arc_rows.append({
            'from_facility': from_fac,
            'to_facility': to_fac,
            'distance_miles': round(distance_miles, 1),
            'pkgs_day': int(total_packages),
            'pkg_cube_cuft': round(total_cube, 2),
            'trucks': trucks_needed,
            'physical_containers': total_containers,
            'persisted_containers': persisted,
            'fresh_containers': fresh,
            'packages_per_truck': round(safe_divide(total_packages, trucks_needed), 1),
            'cube_per_truck': round(safe_divide(total_cube, trucks_needed), 1),
            'container_fill_rate': round(avg_container_fill, 3),
            'truck_fill_rate': round(min(1.0, truck_fill_rate), 3),
            'cost_per_truck': round(cost_per_truck, 2),
            'total_cost': round(total_cost, 2),
            'CPP': round(safe_divide(total_cost, total_packages), 4),
            'num_od_flows': arc_data['num_od_flows']
        })

    result_df = pd.DataFrame(arc_rows)

    if result_df.empty:
        return result_df

    # Sort for consistent output
    result_df = result_df.sort_values(
        ['from_facility', 'to_facility']
    ).reset_index(drop=True)

    # Diagnostic output
    print("\n    Arc Summary (top 5 by packages):")
    if not result_df.empty:
        top_arcs = result_df.nlargest(5, 'pkgs_day')
        for _, arc in top_arcs.iterrows():
            persist_pct = safe_divide(arc['persisted_containers'], arc['physical_containers']) * 100
            print(f"      {arc['from_facility']}→{arc['to_facility']}: "
                  f"{arc['pkgs_day']:>6,} pkgs, "
                  f"{arc['physical_containers']:>3} containers ({persist_pct:.0f}% persisted), "
                  f"{arc['trucks']:>2} trucks, {arc['truck_fill_rate']:.1%} fill")

    return result_df


# ============================================================================
# LEGACY FUNCTION - UPDATED FOR COMPATIBILITY
# ============================================================================

def build_od_container_map(
        od_selected: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        facilities: pd.DataFrame
) -> pd.DataFrame:
    """
    Legacy wrapper - calls new persistence-aware function.

    Maintains backward compatibility while using new logic.
    """
    return build_od_container_map_with_persistence(
        od_selected, package_mix, container_params, facilities
    )


# ============================================================================
# SORT LEVEL CONTAINER IMPACT ANALYSIS
# ============================================================================

def analyze_sort_level_container_impact(
        od_selected: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        facilities: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze how sort level choice impacts container count and fill rates.

    Updated to show persistence vs fresh container breakdown.
    """
    # Build full container tracking
    od_with_containers = build_od_container_map_with_persistence(
        od_selected, package_mix, container_params, facilities
    )

    analysis = []

    for sort_level in ['region', 'market', 'sort_group']:
        level_mask = od_with_containers['chosen_sort_level'] == sort_level
        level_ods = od_with_containers[level_mask]

        if level_ods.empty:
            continue

        total_pkgs = level_ods['pkgs_day'].sum()
        total_origin_containers = level_ods['origin_containers'].sum()
        avg_origin_fill = level_ods['origin_container_fill'].mean()

        # Count creation points (indicates sort operations)
        total_creation_points = sum(
            len(cp) for cp in level_ods['creation_points']
        )
        avg_creation_points = total_creation_points / len(level_ods)

        analysis.append({
            'sort_level': sort_level,
            'num_od_pairs': len(level_ods),
            'total_packages': int(total_pkgs),
            'origin_containers': int(total_origin_containers),
            'avg_origin_container_fill': round(avg_origin_fill, 3),
            'avg_creation_points_per_od': round(avg_creation_points, 2),
            'packages_per_container': round(safe_divide(total_pkgs, total_origin_containers), 1),
            'note': _get_sort_level_note(sort_level)
        })

    return pd.DataFrame(analysis)


def _get_sort_level_note(sort_level: str) -> str:
    """Get explanatory note for sort level behavior."""
    notes = {
        'region': 'Containers recreated at each intermediate hub (sort operation)',
        'market': 'Containers persist through intermediates (crossdock only)',
        'sort_group': 'Most containers at origin, persist through network'
    }
    return notes.get(sort_level, '')


# ============================================================================
# DIAGNOSTIC COMPARISON
# ============================================================================

def create_container_flow_diagnostic(
        od_selected: pd.DataFrame,
        arc_summary_original: pd.DataFrame,
        arc_summary_corrected: pd.DataFrame
) -> str:
    """
    Create diagnostic report comparing original vs corrected container tracking.

    Shows impact of persistence logic on fill rates and container counts.
    """
    lines = []
    lines.append("=" * 100)
    lines.append("CONTAINER FLOW DIAGNOSTIC - Persistence Logic Applied")
    lines.append("=" * 100)
    lines.append("")

    # Network-level comparison
    if 'truck_fill_rate' in arc_summary_original.columns:
        orig_avg_fill = arc_summary_original['truck_fill_rate'].mean()
    else:
        orig_avg_fill = 0

    corr_avg_fill = arc_summary_corrected['truck_fill_rate'].mean()

    lines.append("NETWORK SUMMARY:")
    lines.append(f"  Avg Truck Fill Rate:")
    lines.append(f"    Original (recalc each arc): {orig_avg_fill:.1%}")
    lines.append(f"    Corrected (persistence):    {corr_avg_fill:.1%}")
    lines.append(f"    Delta:                      {(corr_avg_fill - orig_avg_fill):+.1%}")
    lines.append("")

    # Container breakdown
    if 'physical_containers' in arc_summary_corrected.columns:
        total_containers = arc_summary_corrected['physical_containers'].sum()
        persisted = arc_summary_corrected['persisted_containers'].sum()
        fresh = arc_summary_corrected['fresh_containers'].sum()

        lines.append("CONTAINER BREAKDOWN:")
        lines.append(f"  Total physical containers: {total_containers:,}")
        lines.append(f"    Persisted (crossdock):   {persisted:,} ({safe_divide(persisted, total_containers)*100:.1f}%)")
        lines.append(f"    Fresh (sort operations): {fresh:,} ({safe_divide(fresh, total_containers)*100:.1f}%)")
        lines.append("")

    # Per-arc comparison (major arcs only)
    lines.append("ARC-LEVEL COMPARISON (top 10 by volume change):")
    lines.append("")

    if not arc_summary_original.empty and not arc_summary_corrected.empty:
        # Merge for comparison
        orig_cols = ['from_facility', 'to_facility', 'pkgs_day', 'trucks', 'truck_fill_rate']
        orig_subset = arc_summary_original[[c for c in orig_cols if c in arc_summary_original.columns]].copy()
        orig_subset = orig_subset.rename(columns={
            'trucks': 'trucks_orig',
            'truck_fill_rate': 'fill_orig'
        })

        corr_cols = ['from_facility', 'to_facility', 'trucks', 'truck_fill_rate',
                     'physical_containers', 'persisted_containers', 'fresh_containers']
        corr_subset = arc_summary_corrected[[c for c in corr_cols if c in arc_summary_corrected.columns]].copy()
        corr_subset = corr_subset.rename(columns={
            'trucks': 'trucks_corr',
            'truck_fill_rate': 'fill_corr'
        })

        merged = orig_subset.merge(
            corr_subset,
            on=['from_facility', 'to_facility'],
            how='outer'
        )

        if 'fill_orig' in merged.columns and 'fill_corr' in merged.columns:
            merged['fill_delta'] = merged['fill_corr'] - merged['fill_orig']
            merged = merged.sort_values('fill_delta', ascending=True)

            lines.append(f"{'Arc':<25} {'Orig Fill':<12} {'Corr Fill':<12} {'Delta':<10} {'Containers':<15}")
            lines.append("-" * 80)

            for _, row in merged.head(10).iterrows():
                arc_name = f"{row['from_facility']}→{row['to_facility']}"
                orig_fill = row.get('fill_orig', 0) or 0
                corr_fill = row.get('fill_corr', 0) or 0
                delta = row.get('fill_delta', 0) or 0
                containers = row.get('physical_containers', 0) or 0
                persisted = row.get('persisted_containers', 0) or 0

                container_str = f"{int(containers)} ({int(persisted)} pers)"

                lines.append(
                    f"{arc_name:<25} {orig_fill:>10.1%}  {corr_fill:>10.1%}  "
                    f"{delta:>+8.1%}  {container_str:<15}"
                )

    lines.append("")
    lines.append("=" * 100)
    lines.append("")
    lines.append("INTERPRETATION:")
    lines.append("  - Lower fill rates reflect reality of partial containers persisting")
    lines.append("  - Persisted containers = passed through crossdock unchanged")
    lines.append("  - Fresh containers = created at sort operations (may consolidate)")
    lines.append("  - Higher fresh % = more sort operations in network")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_container_summary_for_od(
        od_row: pd.Series,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        facilities: pd.DataFrame
) -> Dict:
    """
    Get container summary for a single OD flow.

    Useful for detailed inspection and debugging.
    """
    path_nodes = extract_path_nodes(od_row)
    sort_level = od_row.get('chosen_sort_level', 'market')

    creation_points = identify_container_creation_points(
        path_nodes, sort_level, facilities
    )

    origin_data = calculate_containers_at_creation(
        packages=od_row['pkgs_day'],
        sort_level=sort_level,
        dest=od_row['dest'],
        creation_facility=od_row['origin'],
        package_mix=package_mix,
        container_params=container_params,
        facilities=facilities
    )

    return {
        'origin': od_row['origin'],
        'dest': od_row['dest'],
        'packages': od_row['pkgs_day'],
        'sort_level': sort_level,
        'path': path_nodes,
        'creation_points': creation_points,
        'origin_containers': origin_data['containers'],
        'origin_fill_rate': origin_data['container_fill_rate'],
        'has_intermediate_sort': len(creation_points) > 1
    }