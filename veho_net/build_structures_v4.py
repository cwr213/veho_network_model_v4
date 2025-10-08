"""
Network Structure Generation Module

Builds OD matrices and generates candidate paths with proper O=D handling.

Key Functions:
    - build_od_and_direct: Create OD matrix from demand forecast
    - candidate_paths: Generate feasible routing alternatives

Business Rules:
    - O=D paths allowed only for hybrid facilities
    - Launch facilities cannot be intermediate stops
    - Regional hub hierarchy respected in path construction
    - Around-the-world factor limits path circuity
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Set, Dict

from .geo_v4 import haversine_miles, cached_haversine
from .utils import (
    get_facility_lookup,
    ensure_columns_exist,
    validate_shares_sum_to_one,
    check_for_duplicates,
)
from .config_v4 import OptimizationConstants

# Progress indicator support
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ============================================================================
# GEOGRAPHIC FILTERING HELPERS
# ============================================================================

def _in_bounding_box(
        hub: str,
        start: str,
        end: str,
        fac_lookup: pd.DataFrame,
        buffer_degrees: float = 0.5
) -> bool:
    """Fast bounding box check for geographic filtering."""
    hub_lat = float(fac_lookup.at[hub, 'lat'])
    hub_lon = float(fac_lookup.at[hub, 'lon'])
    start_lat = float(fac_lookup.at[start, 'lat'])
    start_lon = float(fac_lookup.at[start, 'lon'])
    end_lat = float(fac_lookup.at[end, 'lat'])
    end_lon = float(fac_lookup.at[end, 'lon'])

    min_lat = min(start_lat, end_lat) - buffer_degrees
    max_lat = max(start_lat, end_lat) + buffer_degrees
    min_lon = min(start_lon, end_lon) - buffer_degrees
    max_lon = max(start_lon, end_lon) + buffer_degrees

    return (min_lat <= hub_lat <= max_lat and
            min_lon <= hub_lon <= max_lon)


def _get_viable_intermediates(
        start_node: str,
        end_node: str,
        candidate_hubs: List[str],
        fac_lookup: pd.DataFrame,
        max_detour_factor: float
) -> List[str]:
    """Pre-filter intermediates by geographic proximity."""
    if start_node == end_node:
        return []

    direct_dist = cached_haversine(
        float(fac_lookup.at[start_node, 'lat']),
        float(fac_lookup.at[start_node, 'lon']),
        float(fac_lookup.at[end_node, 'lat']),
        float(fac_lookup.at[end_node, 'lon'])
    )

    max_total_dist = direct_dist * max_detour_factor
    viable = []

    for hub in candidate_hubs:
        # Fast bounding box check first
        if not _in_bounding_box(hub, start_node, end_node, fac_lookup):
            continue

        # Then precise distance check
        dist_to_hub = cached_haversine(
            float(fac_lookup.at[start_node, 'lat']),
            float(fac_lookup.at[start_node, 'lon']),
            float(fac_lookup.at[hub, 'lat']),
            float(fac_lookup.at[hub, 'lon'])
        )
        dist_from_hub = cached_haversine(
            float(fac_lookup.at[hub, 'lat']),
            float(fac_lookup.at[hub, 'lon']),
            float(fac_lookup.at[end_node, 'lat']),
            float(fac_lookup.at[end_node, 'lon'])
        )

        if dist_to_hub + dist_from_hub <= max_total_dist:
            viable.append(hub)

    return viable


# ============================================================================
# OD MATRIX GENERATION
# ============================================================================

def build_od_and_direct(
        facilities: pd.DataFrame,
        zips: pd.DataFrame,
        year_demand: pd.DataFrame,
        injection_distribution: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate OD matrix for middle-mile flows and direct injection volumes.

    This function creates the demand structure for the network optimization:
    1. Middle-mile OD pairs: Packages flowing between facilities
    2. Direct injection: Packages injected directly at destination

    Business Logic:
    ---------------
    - Injection origins: Only hub/hybrid facilities with is_injection_node=1
    - Destinations: All facilities weighted by population served
    - O=D flows: Allowed only for hybrid facilities (can both inject and deliver)

    Args:
        facilities: Facility master data with required columns:
            - facility_name, type, lat, lon, parent_hub_name
            - regional_sort_hub, is_injection_node
        zips: ZIP code assignments with required columns:
            - zip, facility_name_assigned, population
        year_demand: Annual demand forecast with required columns:
            - year, annual_pkgs, offpeak_pct_of_annual, peak_pct_of_annual
            - middle_mile_share_offpeak, middle_mile_share_peak
        injection_distribution: Package injection distribution with columns:
            - facility_name, absolute_share

    Returns:
        Tuple of:
            - od: OD matrix with columns [origin, dest, pkgs_offpeak_day, pkgs_peak_day]
            - direct: Direct injection volumes [dest, dir_pkgs_offpeak_day, dir_pkgs_peak_day]
            - dest_pop: Destination population shares [dest, dest_pop_share]

    Raises:
        ValueError: If required columns missing or validation fails
    """
    # Validate required columns
    ensure_columns_exist(
        facilities,
        ["facility_name", "type", "lat", "lon", "parent_hub_name",
         "regional_sort_hub", "is_injection_node"],
        context="facilities"
    )

    ensure_columns_exist(
        zips,
        ["zip", "facility_name_assigned", "population"],
        context="zips"
    )

    ensure_columns_exist(
        year_demand,
        ["year", "annual_pkgs", "offpeak_pct_of_annual", "peak_pct_of_annual",
         "middle_mile_share_offpeak", "middle_mile_share_peak"],
        context="year_demand"
    )

    # Prepare facilities lookup
    fac = facilities.drop_duplicates(subset=["facility_name"]).reset_index(drop=True)
    fac["type"] = fac["type"].str.lower()

    # Calculate destination population shares
    z = zips[["zip", "facility_name_assigned", "population"]].drop_duplicates(subset=["zip"]).copy()
    pop_by_dest = z.groupby("facility_name_assigned", as_index=False)["population"].sum()
    pop_by_dest["dest_pop_share"] = pop_by_dest["population"] / pop_by_dest["population"].sum()

    # Extract demand parameters
    year_val = int(year_demand["year"].iloc[0])
    annual_total = float(year_demand["annual_pkgs"].sum())
    off_pct = float(year_demand["offpeak_pct_of_annual"].iloc[0])
    peak_pct = float(year_demand["peak_pct_of_annual"].iloc[0])
    mm_off = float(year_demand["middle_mile_share_offpeak"].iloc[0])
    mm_peak = float(year_demand["middle_mile_share_peak"].iloc[0])

    # Build direct injection volumes
    direct = pop_by_dest.rename(columns={"facility_name_assigned": "dest"}).copy()
    direct["year"] = year_val
    direct["dir_pkgs_offpeak_day"] = annual_total * off_pct * (1.0 - mm_off) * direct["dest_pop_share"]
    direct["dir_pkgs_peak_day"] = annual_total * peak_pct * (1.0 - mm_peak) * direct["dest_pop_share"]

    # Build injection distribution (origin shares)
    inj = injection_distribution[["facility_name", "absolute_share"]].copy()
    inj = inj.merge(
        fac[["facility_name", "is_injection_node", "type", "parent_hub_name"]],
        on="facility_name",
        how="left",
        validate="many_to_one"
    )

    # Validate injection facilities exist
    if inj["is_injection_node"].isna().any():
        missing = inj.loc[inj["is_injection_node"].isna(), "facility_name"].unique().tolist()
        raise ValueError(f"injection_distribution facilities not found in facilities: {missing}")

    # Filter to valid injection origins (hub/hybrid with is_injection_node=1)
    valid_origin_types = ["hub", "hybrid"]
    inj = inj[
        (inj["is_injection_node"].astype(int) == 1) &
        (inj["type"].isin(valid_origin_types))
        ].copy()

    if inj.empty:
        raise ValueError(
            "No valid injection origins found. "
            "Ensure hub/hybrid facilities have is_injection_node=1"
        )

    # Normalize injection shares
    inj["abs_w"] = pd.to_numeric(inj["absolute_share"], errors="coerce").fillna(0.0)
    total_w = float(inj["abs_w"].sum())

    if total_w <= OptimizationConstants.EPSILON:
        raise ValueError("injection_distribution.absolute_share must sum > 0")

    inj["inj_share"] = inj["abs_w"] / total_w
    inj = inj[["facility_name", "inj_share"]].rename(columns={"facility_name": "origin"})

    # Create OD grid (all origin-destination combinations)
    dest2 = pop_by_dest.rename(columns={"facility_name_assigned": "dest"})[["dest", "dest_pop_share"]]
    grid = inj.assign(_k=1).merge(dest2.assign(_k=1), on="_k").drop(columns="_k")

    # Calculate OD volumes
    od = grid.copy()
    od["pkgs_offpeak_day"] = annual_total * off_pct * mm_off * od["inj_share"] * od["dest_pop_share"]
    od["pkgs_peak_day"] = annual_total * peak_pct * mm_peak * od["inj_share"] * od["dest_pop_share"]

    # Apply O=D business rules
    fac_types = fac.set_index("facility_name")["type"].to_dict()

    od_filtered = []
    for _, row in od.iterrows():
        origin = row["origin"]
        dest = row["dest"]

        if origin == dest:
            # O=D: Only allowed for hybrid facilities
            origin_type = fac_types.get(origin, "").lower()
            if origin_type == "hybrid":
                od_filtered.append(row)
            # Else: skip O=D for hub/launch facilities
        else:
            # O≠D: Always allowed
            od_filtered.append(row)

    od = pd.DataFrame(od_filtered).reset_index(drop=True)

    return od, direct, pop_by_dest


# ============================================================================
# PATH GENERATION
# ============================================================================

def candidate_paths(
        od: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame,
        around_factor: float
) -> pd.DataFrame:
    """
    Generate candidate paths with routing constraints.

    Path Generation Algorithm:
    --------------------------
    For each OD pair, generate feasible routing alternatives:

    1. Direct path: O → D
    2. One-touch path: O → intermediate → D
       - Intermediate must be hub/hybrid (not launch)
       - Distance constraint: dist(O→I→D) ≤ around_factor × dist(O→D)

    Special Handling:
    -----------------
    - O=D paths: Single-facility path (O → D where O=D)
    - Secondary hubs: Paths must route through parent hub first
    - Launch facilities: Cannot be intermediate stops

    Args:
        od: OD matrix with origin, dest, and volume columns
        facilities: Facility master data
        mileage_bands: Mileage bands for distance calculations
        around_factor: Maximum circuity multiplier (e.g., 1.3 = 30% longer than direct)

    Returns:
        DataFrame with columns:
            - origin, dest: OD pair identifiers
            - path_type: Classification (direct, 1_touch, 2_touch, etc.)
            - path_nodes: Tuple of facilities in path
            - path_str: String representation "O->I->D"
            - strategy_hint: Optional strategy override (None = use global)

    Raises:
        ValueError: If facilities data invalid

    Example:
        >>> paths = candidate_paths(od, facilities, mileage_bands, around_factor=1.3)
        >>> # Returns paths like:
        >>> # [HUB1] -> [LAUNCH5]: Direct
        >>> # [HUB1] -> [HUB2] -> [LAUNCH5]: 1-touch via HUB2
    """
    fac = get_facility_lookup(facilities)

    # Identify valid intermediate facilities (hub/hybrid only)
    valid_intermediate_types = ["hub", "hybrid"]
    hubs_enabled = fac.index[fac["type"].isin(valid_intermediate_types)]

    # Build facility relationship maps
    parent_map = {}
    primary_hubs = set()
    secondary_hubs = set()
    launch_facilities = set()

    for facility_name in fac.index:
        facility_type = fac.at[facility_name, "type"]
        parent_hub = fac.at[facility_name, "parent_hub_name"]

        # Default to self if no parent specified
        if pd.isna(parent_hub) or parent_hub == "":
            parent_hub = facility_name

        parent_map[facility_name] = parent_hub

        # Classify facilities
        if facility_type in ["hub", "hybrid"]:
            if parent_hub == facility_name:
                primary_hubs.add(facility_name)
            else:
                secondary_hubs.add(facility_name)
        elif facility_type == "launch":
            launch_facilities.add(facility_name)

    def calculate_raw_distance(o: str, d: str) -> float:
        """Calculate straight-line distance between facilities."""
        return cached_haversine(
            float(fac.at[o, "lat"]), float(fac.at[o, "lon"]),
            float(fac.at[d, "lat"]), float(fac.at[d, "lon"])
        )

    def build_path_with_constraints(origin: str, dest: str) -> List[List[str]]:
        """
        Generate feasible paths for OD pair with business constraints.

        Returns list of paths, where each path is a list of facility names.
        """
        paths = []

        # Special case: O=D (same facility)
        if origin == dest:
            paths.append([origin, dest])
            return paths

        # Determine required path segments based on facility types
        if dest in launch_facilities:
            dest_parent = parent_map.get(dest, dest)
            if dest_parent == dest:
                required_end = [dest]
            else:
                required_end = [dest_parent, dest]
        else:
            required_end = [dest]

        # Check if origin already in required end segment
        if origin in required_end:
            paths.append([origin, dest])
            return paths

        # Handle secondary hub origins (must route through parent first)
        if origin in secondary_hubs:
            origin_parent = parent_map[origin]
            if origin_parent != origin and origin_parent != dest:
                required_start = [origin, origin_parent]
            else:
                required_start = [origin]
        else:
            required_start = [origin]

        start_node = required_start[-1]
        end_node = required_end[0]

        # Case 1: Start and end nodes are the same
        if start_node == end_node:
            combined = required_start + required_end[1:]
            if len(combined) == len(set(combined)):  # No duplicates
                paths.append(combined)
        else:
            # Case 2: Direct path (no intermediate)
            combined = required_start + required_end
            if len(combined) == len(set(combined)):
                paths.append(combined)

            # Case 3: One-touch path (with intermediate hub)
            # PRE-FILTER intermediates by geography
            candidate_intermediates = [
                h for h in hubs_enabled
                if h not in required_start and h not in required_end
            ]

            # Geographic filtering before distance calculation
            viable_intermediates = _get_viable_intermediates(
                start_node,
                end_node,
                candidate_intermediates,
                fac,
                around_factor
            )

            if viable_intermediates:
                # Find best intermediate hub (shortest total distance)
                best_intermediate = min(
                    viable_intermediates,
                    key=lambda h: calculate_raw_distance(start_node, h) +
                                  calculate_raw_distance(h, end_node)
                )

                intermediate_path = required_start + [best_intermediate] + required_end
                if len(intermediate_path) == len(set(intermediate_path)):
                    paths.append(intermediate_path)

        return paths

    # Generate paths for all OD pairs
    rows = []

    # Progress indicator
    if HAS_TQDM:
        od_iterator = tqdm(od.iterrows(), total=len(od), desc="Generating paths")
    else:
        od_iterator = od.iterrows()

    for _, od_row in od_iterator:
        origin = od_row["origin"]
        dest = od_row["dest"]

        candidate_paths_list = build_path_with_constraints(origin, dest)

        for path_nodes in candidate_paths_list:
            # Classify path by number of facilities
            num_facilities = len(path_nodes)
            if num_facilities == 2:
                path_type = "direct"
            elif num_facilities == 3:
                path_type = "1_touch"
            elif num_facilities == 4:
                path_type = "2_touch"
            elif num_facilities == 5:
                path_type = "3_touch"
            else:
                path_type = "4_touch"

            rows.append({
                "origin": origin,
                "dest": dest,
                "path_type": path_type,
                "path_nodes": tuple(path_nodes),  # Tuple for memory efficiency
                "path_str": "->".join(path_nodes),
                "strategy_hint": None  # Will use global strategy
            })

    # Create DataFrame from collected rows
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Validate paths (no launch facilities as intermediates)
    def validate_path_nodes(path_nodes: tuple) -> bool:
        """Ensure no launch facilities in intermediate positions."""
        if len(path_nodes) <= 2:
            return True

        for node in path_nodes[1:-1]:  # Check only intermediate nodes
            if node in fac.index and fac.at[node, "type"] == "launch":
                return False
        return True

    initial_count = len(df)
    df = df[df["path_nodes"].apply(validate_path_nodes)].reset_index(drop=True)

    removed_count = initial_count - len(df)
    if removed_count > 0:
        print(f"  Removed {removed_count} invalid paths (launch facility violations)")

    # Remove duplicate paths (path_nodes is already a tuple)
    df = df.drop_duplicates(subset=["origin", "dest", "path_nodes"]).reset_index(drop=True)

    return df


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def summarize_od_matrix(od: pd.DataFrame) -> pd.Series:
    """
    Generate summary statistics for OD matrix.

    Args:
        od: OD matrix with volume columns

    Returns:
        Series with summary metrics
    """
    summary = pd.Series({
        'total_od_pairs': len(od),
        'unique_origins': od['origin'].nunique(),
        'unique_destinations': od['dest'].nunique(),
        'total_offpeak_pkgs': od['pkgs_offpeak_day'].sum(),
        'total_peak_pkgs': od['pkgs_peak_day'].sum(),
        'avg_pkgs_per_od_offpeak': od['pkgs_offpeak_day'].mean(),
        'avg_pkgs_per_od_peak': od['pkgs_peak_day'].mean(),
    })

    return summary


def summarize_paths(paths: pd.DataFrame) -> pd.Series:
    """
    Generate summary statistics for candidate paths.

    Args:
        paths: Candidate paths DataFrame

    Returns:
        Series with summary metrics
    """
    if paths.empty:
        return pd.Series()

    path_type_counts = paths['path_type'].value_counts().to_dict()

    summary = pd.Series({
        'total_paths': len(paths),
        'unique_od_pairs': paths.groupby(['origin', 'dest']).ngroups,
        'avg_paths_per_od': len(paths) / max(paths.groupby(['origin', 'dest']).ngroups, 1),
        **{f'paths_{k}': v for k, v in path_type_counts.items()}
    })

    return summary


def validate_path_structure(
        paths: pd.DataFrame,
        facilities: pd.DataFrame
) -> Dict[str, bool]:
    """
    Validate path structure and identify issues.

    Args:
        paths: Candidate paths DataFrame
        facilities: Facility master data

    Returns:
        Dictionary with validation results
    """
    fac = get_facility_lookup(facilities)

    results = {
        'all_facilities_exist': True,
        'no_launch_intermediates': True,
        'valid_path_nodes': True,
    }

    # Check all facilities in paths exist
    all_facilities_in_paths = set()
    for _, row in paths.iterrows():
        path_nodes = row['path_nodes']
        if isinstance(path_nodes, (list, tuple)):
            all_facilities_in_paths.update(path_nodes)

    missing_facilities = all_facilities_in_paths - set(fac.index)
    if missing_facilities:
        results['all_facilities_exist'] = False
        print(f"Warning: Facilities in paths not found in master: {missing_facilities}")

    # Check for launch facilities as intermediates
    for _, row in paths.iterrows():
        path_nodes = row['path_nodes']
        if isinstance(path_nodes, (list, tuple)) and len(path_nodes) > 2:
            for node in path_nodes[1:-1]:
                if node in fac.index and fac.at[node, 'type'] == 'launch':
                    results['no_launch_intermediates'] = False
                    print(f"Warning: Launch facility {node} used as intermediate in path")

    return results