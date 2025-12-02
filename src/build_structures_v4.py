"""
Network Structure Generation Module

Creates OD matrix and generates candidate paths through the network.
Generates MULTIPLE candidate paths per OD pair (direct, 1-touch, 2-touch, 3-touch)
to enable MILP optimization of path selection.

Key Functions:
    - build_od_and_direct: Create OD matrix from demand forecast
    - candidate_paths: Generate multiple feasible routing alternatives per OD
    - get_facility_lookup: Indexed facility reference data

Business Rules:
    - O=D paths allowed only for hybrid facilities
    - Launch facilities cannot be intermediate stops
    - path_around_the_world_factor filters unreasonably circuitous paths
    - Regional hub hierarchy is a preference, not a hard constraint
"""

from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np

from .config_v4 import OptimizationConstants
from .geo_v4 import haversine_miles

# Progress indicator support
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def get_facility_lookup(facilities: pd.DataFrame) -> pd.DataFrame:
    """
    Create indexed facility lookup for efficient access.

    Returns DataFrame indexed by facility_name with key attributes.
    """
    return facilities.set_index("facility_name")


def build_od_and_direct(
        facilities: pd.DataFrame,
        zips: pd.DataFrame,
        demand: pd.DataFrame,
        injection_distribution: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build OD matrix and direct injection volumes from demand data.

    Args:
        facilities: Facility master data
        zips: Zip code to facility assignments with population
        demand: Demand data (already filtered by year in caller)
        injection_distribution: Origin injection shares by facility

    Returns:
        od: DataFrame with origin, dest, and volume columns (pkgs_peak_day, pkgs_offpeak_day)
        direct_day: DataFrame with direct injection (Zone 0) volumes
        dest_pop: DataFrame with destination population for zone distribution
    """
    fac = get_facility_lookup(facilities)

    # Calculate destination shares by population
    dest_shares = _calculate_destination_shares(zips, fac)

    if not dest_shares:
        raise ValueError("No destination shares calculated - check zips data has valid facility_name_assigned values")

    # Build destination population DataFrame
    if not zips.empty:
        dest_pop = zips.groupby('facility_name_assigned')['population'].sum().reset_index()
        dest_pop.columns = ['dest', 'population']
    else:
        dest_pop = pd.DataFrame(columns=['dest', 'population'])

    # Get demand parameters
    if demand.empty:
        raise ValueError("Demand data is empty")

    demand_row = demand.iloc[0]

    # Required columns - matches input file structure
    required_demand_cols = {
        'annual_pkgs',
        'offpeak_pct_of_annual',
        'peak_pct_of_annual',
        'middle_mile_share_peak',
        'middle_mile_share_offpeak'
    }
    missing = required_demand_cols - set(demand.columns)
    if missing:
        raise ValueError(f"Demand sheet missing required columns: {sorted(missing)}")

    # Calculate daily packages from annual totals
    annual_pkgs = float(demand_row['annual_pkgs'])
    offpeak_pct = float(demand_row['offpeak_pct_of_annual'])
    peak_pct = float(demand_row['peak_pct_of_annual'])
    mm_share_peak = float(demand_row['middle_mile_share_peak'])
    mm_share_offpeak = float(demand_row['middle_mile_share_offpeak'])

    peak_pkgs = annual_pkgs * peak_pct
    offpeak_pkgs = annual_pkgs * offpeak_pct

    if annual_pkgs == 0:
        raise ValueError("No package volume found in demand data (annual_pkgs = 0)")

    if injection_distribution.empty:
        raise ValueError("injection_distribution is empty")

    od_rows = []
    direct_rows = []

    # Process each origin from injection distribution
    for _, inj_row in injection_distribution.iterrows():
        origin = inj_row['facility_name']
        inj_share = float(inj_row['absolute_share'])

        if inj_share < 0.0001:
            continue

        # Calculate origin's share of middle-mile volume
        origin_peak_mm = peak_pkgs * mm_share_peak * inj_share
        origin_offpeak_mm = offpeak_pkgs * mm_share_offpeak * inj_share

        # Distribute to destinations by population share
        for dest, dest_share in dest_shares.items():
            peak_vol = origin_peak_mm * dest_share
            offpeak_vol = origin_offpeak_mm * dest_share

            if peak_vol < 0.01 and offpeak_vol < 0.01:
                continue

            # Skip O=D unless hybrid facility
            if origin == dest:
                if origin in fac.index:
                    fac_type = str(fac.at[origin, 'type']).lower()
                    if fac_type != 'hybrid':
                        continue
                else:
                    continue

            od_rows.append({
                'origin': origin,
                'dest': dest,
                'pkgs_peak_day': peak_vol,
                'pkgs_offpeak_day': offpeak_vol
            })

    # Direct injection (Zone 0) - distributed by population to destinations
    direct_peak = peak_pkgs * (1 - mm_share_peak)
    direct_offpeak = offpeak_pkgs * (1 - mm_share_offpeak)

    for dest, dest_share in dest_shares.items():
        peak_vol = direct_peak * dest_share
        offpeak_vol = direct_offpeak * dest_share

        if peak_vol < 0.01 and offpeak_vol < 0.01:
            continue

        direct_rows.append({
            'dest': dest,
            'dir_pkgs_peak_day': peak_vol,
            'dir_pkgs_offpeak_day': offpeak_vol
        })

    od = pd.DataFrame(od_rows)
    direct_day = pd.DataFrame(direct_rows)

    # Ensure expected columns exist even if empty
    if od.empty:
        od = pd.DataFrame(columns=['origin', 'dest', 'pkgs_peak_day', 'pkgs_offpeak_day'])

    if direct_day.empty:
        direct_day = pd.DataFrame(columns=['dest', 'dir_pkgs_peak_day', 'dir_pkgs_offpeak_day'])

    # Aggregate duplicate OD pairs (can happen with multiple injection sources)
    if not od.empty:
        od = od.groupby(['origin', 'dest'], as_index=False).agg({
            'pkgs_peak_day': 'sum',
            'pkgs_offpeak_day': 'sum'
        })

    if not direct_day.empty:
        direct_day = direct_day.groupby('dest', as_index=False).agg({
            'dir_pkgs_peak_day': 'sum',
            'dir_pkgs_offpeak_day': 'sum'
        })

    print(f"  OD matrix built: {len(od)} pairs from {len(injection_distribution)} origins to {len(dest_shares)} destinations")

    return od, direct_day, dest_pop


def _calculate_destination_shares(
        zips: pd.DataFrame,
        fac: pd.DataFrame
) -> Dict[str, float]:
    """Calculate destination share by population for each delivery facility."""
    if zips.empty:
        # Fallback: equal distribution to all facilities with last-mile capability
        delivery_facs = [f for f in fac.index
                        if str(fac.at[f, 'type']).lower() in ('hybrid', 'launch')]
        n = len(delivery_facs)
        return {f: 1.0/n for f in delivery_facs} if n > 0 else {}

    # Group population by assigned facility
    pop_by_fac = zips.groupby('facility_name_assigned')['population'].sum()
    total_pop = pop_by_fac.sum()

    if total_pop == 0:
        return {}

    return (pop_by_fac / total_pop).to_dict()


def candidate_paths(
        od: pd.DataFrame,
        facilities: pd.DataFrame,
        around_factor: float
) -> pd.DataFrame:
    """
    Generate MULTIPLE candidate paths per OD pair for MILP optimization.

    This is the critical function for network optimization. It generates:
    - Direct paths (O → D)
    - 1-touch paths (O → H → D) through each valid intermediate hub
    - 2-touch paths (O → H1 → H2 → D) through valid hub combinations
    - 3-touch paths (O → H1 → H2 → H3 → D) for transcontinental routes

    The MILP then selects the optimal path for each OD considering:
    - Arc consolidation benefits (multiple ODs sharing arcs)
    - Total network cost
    - Hub capacity constraints

    Args:
        od: OD matrix with origin, dest, pkgs_day columns
        facilities: Facility master data
        around_factor: Maximum path distance as multiple of direct distance.
            E.g., 1.5 means path can be at most 50% longer than direct route.
            Paths exceeding this are filtered out as "around the world" routes.
            Must come from run_settings.path_around_the_world_factor.

    Returns:
        DataFrame with columns:
            - origin, dest: OD pair identifiers
            - path_type: Classification (direct, 1_touch, 2_touch, 3_touch)
            - path_nodes: List of facilities in path
            - path_str: String representation "O->H1->H2->D"
            - strategy_hint: Optional strategy override (None = use global)
            - path_miles: Total path distance
            - direct_miles: Direct O→D distance
            - circuity_ratio: path_miles / direct_miles
    """
    fac = get_facility_lookup(facilities)

    # Get valid intermediate hubs (hub or hybrid, not launch-only)
    valid_hubs = _get_valid_intermediate_hubs(fac)

    # Pre-compute distances between all facility pairs for efficiency
    distance_cache = _build_distance_cache(fac)

    rows = []

    # Get unique OD pairs
    od_pairs = od[['origin', 'dest']].drop_duplicates()

    # Progress indicator
    if HAS_TQDM:
        od_iterator = tqdm(od_pairs.iterrows(), total=len(od_pairs),
                          desc="Generating candidate paths")
    else:
        od_iterator = od_pairs.iterrows()

    for _, od_row in od_iterator:
        origin = od_row['origin']
        dest = od_row['dest']

        # Generate all candidate paths for this OD
        paths = _generate_all_paths_for_od(
            origin=origin,
            dest=dest,
            fac=fac,
            valid_hubs=valid_hubs,
            distance_cache=distance_cache,
            around_factor=around_factor
        )

        for path_data in paths:
            rows.append({
                'origin': origin,
                'dest': dest,
                **path_data
            })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Deduplicate paths (same O, D, path_str)
    df = df.drop_duplicates(subset=['origin', 'dest', 'path_str']).reset_index(drop=True)

    # Log path generation statistics
    paths_per_od = df.groupby(['origin', 'dest']).size()
    print(f"Path generation complete:")
    print(f"  Total OD pairs: {len(paths_per_od)}")
    print(f"  Total candidate paths: {len(df)}")
    print(f"  Avg paths per OD: {paths_per_od.mean():.1f}")
    print(f"  Max paths per OD: {paths_per_od.max()}")
    print(f"  Path type distribution:")
    for ptype, count in df['path_type'].value_counts().items():
        print(f"    {ptype}: {count} ({100*count/len(df):.1f}%)")

    return df


def _get_valid_intermediate_hubs(fac: pd.DataFrame) -> Set[str]:
    """
    Get set of facilities that can serve as intermediate hubs.

    Rules:
    - Must be 'hub' or 'hybrid' facility type
    - Launch-only facilities cannot be intermediates
    """
    valid = set()
    for facility_name in fac.index:
        fac_type = str(fac.at[facility_name, 'type']).lower()
        if fac_type in ('hub', 'hybrid'):
            valid.add(facility_name)
    return valid


def _build_distance_cache(fac: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """
    Pre-compute haversine distances between all facility pairs.

    Returns dict mapping (fac1, fac2) -> distance_miles
    """
    cache = {}
    facilities_list = list(fac.index)

    for i, f1 in enumerate(facilities_list):
        lat1, lon1 = fac.at[f1, 'lat'], fac.at[f1, 'lon']
        for f2 in facilities_list[i:]:  # Only compute upper triangle
            lat2, lon2 = fac.at[f2, 'lat'], fac.at[f2, 'lon']
            dist = haversine_miles(lat1, lon1, lat2, lon2)
            cache[(f1, f2)] = dist
            cache[(f2, f1)] = dist  # Symmetric

    return cache


def _get_distance(
        f1: str,
        f2: str,
        distance_cache: Dict[Tuple[str, str], float]
) -> float:
    """Get distance between two facilities from cache."""
    if f1 == f2:
        return 0.0
    return distance_cache.get((f1, f2), float('inf'))


def _calculate_path_distance(
        path: List[str],
        distance_cache: Dict[Tuple[str, str], float]
) -> float:
    """Calculate total distance of a path."""
    if len(path) < 2:
        return 0.0

    total = 0.0
    for i in range(len(path) - 1):
        total += _get_distance(path[i], path[i+1], distance_cache)
    return total


def _generate_all_paths_for_od(
        origin: str,
        dest: str,
        fac: pd.DataFrame,
        valid_hubs: Set[str],
        distance_cache: Dict[Tuple[str, str], float],
        around_factor: float
) -> List[Dict]:
    """
    Generate candidate paths for a single OD pair - SHORTEST of each path type.

    Returns at most 4 paths:
        1. Direct (O → D)
        2. Best 1-touch (shortest O → H → D)
        3. Best 2-touch (shortest O → H1 → H2 → D) if direct > 300mi
        4. Best 3-touch (shortest O → H1 → H2 → H3 → D) if direct > 1000mi

    This gives MILP real choices between consolidation strategies without
    combinatorial explosion.
    """
    paths = []

    direct_miles = _get_distance(origin, dest, distance_cache)
    max_path_miles = direct_miles * around_factor if direct_miles > 0 else float('inf')

    # Special case: O=D
    if origin == dest:
        paths.append(_create_path_data(
            path_nodes=[origin],
            path_type='direct',
            direct_miles=0.0,
            path_miles=0.0
        ))
        return paths

    # 1. Direct path (always included)
    paths.append(_create_path_data(
        path_nodes=[origin, dest],
        path_type='direct',
        direct_miles=direct_miles,
        path_miles=direct_miles
    ))

    # 2. Best 1-touch path (O → H → D)
    best_1touch = None
    best_1touch_miles = float('inf')

    for hub in valid_hubs:
        if hub == origin or hub == dest:
            continue

        path_miles = _get_distance(origin, hub, distance_cache) + \
                     _get_distance(hub, dest, distance_cache)

        if path_miles <= max_path_miles and path_miles < best_1touch_miles:
            best_1touch = [origin, hub, dest]
            best_1touch_miles = path_miles

    if best_1touch:
        paths.append(_create_path_data(
            path_nodes=best_1touch,
            path_type='1_touch',
            direct_miles=direct_miles,
            path_miles=best_1touch_miles
        ))

    # 3. Best 2-touch path (O → H1 → H2 → D)
    best_2touch = None
    best_2touch_miles = float('inf')

    for hub1 in valid_hubs:
        if hub1 == origin or hub1 == dest:
            continue

        leg1 = _get_distance(origin, hub1, distance_cache)
        if leg1 >= best_2touch_miles:
            continue

        for hub2 in valid_hubs:
            if hub2 == origin or hub2 == dest or hub2 == hub1:
                continue

            path_miles = leg1 + \
                         _get_distance(hub1, hub2, distance_cache) + \
                         _get_distance(hub2, dest, distance_cache)

            if path_miles <= max_path_miles and path_miles < best_2touch_miles:
                best_2touch = [origin, hub1, hub2, dest]
                best_2touch_miles = path_miles

    if best_2touch:
        paths.append(_create_path_data(
            path_nodes=best_2touch,
            path_type='2_touch',
            direct_miles=direct_miles,
            path_miles=best_2touch_miles
        ))

    # 4. Best 3-touch path (O → H1 → H2 → H3 → D)
    best_3touch = None
    best_3touch_miles = float('inf')

    for hub1 in valid_hubs:
        if hub1 == origin or hub1 == dest:
            continue

        leg1 = _get_distance(origin, hub1, distance_cache)
        if leg1 >= best_3touch_miles:
            continue

        for hub2 in valid_hubs:
            if hub2 in (origin, dest, hub1):
                continue

            leg12 = leg1 + _get_distance(hub1, hub2, distance_cache)
            if leg12 >= best_3touch_miles:
                continue

            for hub3 in valid_hubs:
                if hub3 in (origin, dest, hub1, hub2):
                    continue

                path_miles = leg12 + \
                             _get_distance(hub2, hub3, distance_cache) + \
                             _get_distance(hub3, dest, distance_cache)

                if path_miles <= max_path_miles and path_miles < best_3touch_miles:
                    best_3touch = [origin, hub1, hub2, hub3, dest]
                    best_3touch_miles = path_miles

    if best_3touch:
        paths.append(_create_path_data(
            path_nodes=best_3touch,
            path_type='3_touch',
            direct_miles=direct_miles,
            path_miles=best_3touch_miles
        ))

    return paths


def _create_path_data(
        path_nodes: List[str],
        path_type: str,
        direct_miles: float,
        path_miles: float
) -> Dict:
    """Create standardized path data dictionary."""
    circuity = path_miles / direct_miles if direct_miles > 0 else 1.0

    return {
        'path_type': path_type,
        'path_nodes': list(path_nodes),  # Ensure list type
        'path_str': '->'.join(path_nodes),
        'strategy_hint': None,
        'path_miles': round(path_miles, 1),
        'direct_miles': round(direct_miles, 1),
        'circuity_ratio': round(circuity, 3)
    }


def _get_regional_hub(facility: str, fac: pd.DataFrame) -> Optional[str]:
    """Get regional sort hub for a facility."""
    if facility not in fac.index:
        return None

    regional = fac.at[facility, 'regional_sort_hub']
    if pd.isna(regional) or regional == '':
        return facility  # Self is regional hub
    return regional


def _get_all_regional_hubs(fac: pd.DataFrame) -> Set[str]:
    """Get set of all unique regional sort hubs."""
    hubs = set()
    for facility in fac.index:
        regional = fac.at[facility, 'regional_sort_hub']
        if pd.notna(regional) and regional != '':
            hubs.add(regional)
        # Also add facilities that are their own regional hub
        if str(fac.at[facility, 'type']).lower() in ('hub', 'hybrid'):
            hubs.add(facility)
    return hubs


def validate_paths(
        candidates: pd.DataFrame,
        facilities: pd.DataFrame
) -> pd.DataFrame:
    """
    Validate candidate paths against business rules.

    Removes paths that:
    - Have launch facilities as intermediates
    - Have duplicate nodes (loops)
    - Reference non-existent facilities

    Returns filtered DataFrame.
    """
    fac = get_facility_lookup(facilities)
    valid_facilities = set(fac.index)

    def is_valid_path(row) -> bool:
        nodes = row['path_nodes']

        if not isinstance(nodes, list):
            return False

        # Check for duplicates
        if len(nodes) != len(set(nodes)):
            return False

        # Check all facilities exist
        if not all(n in valid_facilities for n in nodes):
            return False

        # Check intermediates are not launch-only
        if len(nodes) > 2:
            for intermediate in nodes[1:-1]:
                fac_type = str(fac.at[intermediate, 'type']).lower()
                if fac_type == 'launch':
                    return False

        return True

    valid_mask = candidates.apply(is_valid_path, axis=1)
    filtered = candidates[valid_mask].reset_index(drop=True)

    removed = len(candidates) - len(filtered)
    if removed > 0:
        print(f"Validation removed {removed} invalid paths")

    return filtered


def add_od_volumes_to_candidates(
        candidates: pd.DataFrame,
        od: pd.DataFrame
) -> pd.DataFrame:
    """
    Join OD volume data to candidate paths.

    Each candidate path gets the pkgs_day from its corresponding OD pair.
    """
    # Get volume by OD pair
    od_vols = od.groupby(['origin', 'dest'])['pkgs_day'].sum().reset_index()

    # Merge to candidates
    merged = candidates.merge(
        od_vols,
        on=['origin', 'dest'],
        how='left'
    )

    # Fill missing volumes with 0
    merged['pkgs_day'] = merged['pkgs_day'].fillna(0)

    return merged