"""
Network Structure Generation Module - v4.6

CRITICAL FIX: Ensures similar OD pairs take common paths through regional hubs.
- PHL→LAX1, PHL→LAX2, PHL→LAX3 all route via same hub (e.g., ONT)
- Prevents inconsistent routing (e.g., PHL→IND→LAX3)
- Uses regional hub hierarchy to enforce network topology

Key Functions:
    - build_od_and_direct: Create OD matrix from demand forecast
    - candidate_paths: Generate feasible routing alternatives with consistency

Business Rules:
    - O=D paths allowed only for hybrid facilities
    - Launch facilities cannot be intermediate stops
    - Regional hub hierarchy respected in path construction
    - All destinations in same region route through same hub from any origin
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
# OD MATRIX GENERATION (UNCHANGED)
# ============================================================================

def build_od_and_direct(
        facilities: pd.DataFrame,
        zips: pd.DataFrame,
        year_demand: pd.DataFrame,
        injection_distribution: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate OD matrix for middle-mile flows and direct injection volumes.

    [Documentation unchanged from original]
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
# PATH GENERATION - COMPLETE REWRITE FOR CONSISTENCY
# ============================================================================

def candidate_paths(
        od: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame,
        around_factor: float
) -> pd.DataFrame:
    """
    Generate candidate paths with REGIONAL HUB CONSISTENCY.

    KEY PRINCIPLE: All destinations in same region should route through
    their regional hub from any given origin.

    This ensures:
    - PHL→LAX1, PHL→LAX2, PHL→LAX3 all route via same hub (e.g., ONT)
    - No inconsistent routing (e.g., PHL→IND→LAX3)
    - Network topology is respected

    Path Generation Algorithm:
    --------------------------
    1. Build regional hub hierarchy
    2. For each origin:
        a. Identify regional hubs that serve destinations
        b. Determine THE canonical path from origin to each regional hub
        c. For all destinations in that region, use same path to hub
        d. Extend path to specific destination if needed

    Special Handling:
    -----------------
    - O=D paths: Single-facility path (O → D where O=D)
    - Secondary hubs: Must route through parent hub first
    - Launch facilities: Cannot be intermediate stops

    Args:
        od: OD matrix with origin, dest, and volume columns
        facilities: Facility master data
        mileage_bands: Mileage bands for distance calculations (unused in v4.6)
        around_factor: Maximum circuity multiplier (unused in v4.6 - topology-driven)

    Returns:
        DataFrame with columns:
            - origin, dest: OD pair identifiers
            - path_type: Classification (direct, 1_touch, 2_touch, etc.)
            - path_nodes: Tuple of facilities in path
            - path_str: String representation "O->I->D"
            - strategy_hint: Optional strategy override (None = use global)
    """
    fac = get_facility_lookup(facilities)

    # Build regional hub hierarchy
    region_hierarchy = _build_region_hierarchy(fac)

    # Build facility relationships
    parent_map = {}
    for facility_name in fac.index:
        parent_hub = fac.at[facility_name, "parent_hub_name"]
        if pd.isna(parent_hub) or parent_hub == "":
            parent_hub = facility_name
        parent_map[facility_name] = parent_hub

    rows = []

    # Progress indicator
    if HAS_TQDM:
        od_iterator = tqdm(od.iterrows(), total=len(od), desc="Generating paths")
    else:
        od_iterator = od.iterrows()

    # Process each OD pair
    for _, od_row in od_iterator:
        origin = od_row["origin"]
        dest = od_row["dest"]

        # Generate path(s) for this OD pair
        paths = _generate_paths_for_od(
            origin, dest, fac, region_hierarchy, parent_map
        )

        for path_nodes in paths:
            # Classify path by number of facilities
            num_facilities = len(path_nodes)
            if num_facilities == 1:
                path_type = "direct"  # O=D
            elif num_facilities == 2:
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
                "path_nodes": tuple(path_nodes),
                "path_str": "->".join(path_nodes),
                "strategy_hint": None
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

    # Remove duplicate paths
    df = df.drop_duplicates(subset=["origin", "dest", "path_nodes"]).reset_index(drop=True)

    return df


def _build_region_hierarchy(fac: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build mapping: regional_hub → [facilities in region]

    Returns:
        Dictionary where keys are regional hubs and values are lists of
        facilities that belong to that region.
    """
    hierarchy = {}

    for facility in fac.index:
        region_hub = fac.at[facility, 'regional_sort_hub']
        if pd.isna(region_hub) or region_hub == '':
            region_hub = facility

        if region_hub not in hierarchy:
            hierarchy[region_hub] = []
        hierarchy[region_hub].append(facility)

    return hierarchy


def _generate_paths_for_od(
        origin: str,
        dest: str,
        fac: pd.DataFrame,
        region_hierarchy: Dict[str, List[str]],
        parent_map: Dict[str, str]
) -> List[List[str]]:
    """
    Generate path(s) for a single OD pair with regional hub consistency.

    Returns:
        List of paths, where each path is a list of facility names.
        Typically returns 1 path (the canonical path through regional hubs).
    """
    # Special case: O=D (same facility)
    if origin == dest:
        return [[origin]]

    # Determine destination's regional hub
    dest_region_hub = fac.at[dest, 'regional_sort_hub']
    if pd.isna(dest_region_hub) or dest_region_hub == '':
        dest_region_hub = dest

    # Build canonical path from origin to destination's regional hub
    hub_path = _get_origin_to_region_path(origin, dest_region_hub, fac, parent_map)

    # Extend path to specific destination if needed
    if dest == dest_region_hub:
        # Destination IS the regional hub
        full_path = hub_path
    else:
        # Destination is downstream of regional hub
        full_path = hub_path + [dest]

    # Validate no duplicates
    if len(full_path) != len(set(full_path)):
        # Remove duplicates while preserving order
        seen = set()
        full_path = [x for x in full_path if not (x in seen or seen.add(x))]

    return [full_path]


def _get_origin_to_region_path(
        origin: str,
        region_hub: str,
        fac: pd.DataFrame,
        parent_map: Dict[str, str]
) -> List[str]:
    """
    Determine THE canonical path from origin to regional hub.

    Rules:
    1. If origin == region_hub: [origin]
    2. If origin is secondary hub: [origin, parent, region_hub]
    3. If origin is in different region: [origin, origin_region, region_hub]
    4. Otherwise: [origin, region_hub]

    Returns:
        List of facility names representing the canonical path.
    """
    if origin == region_hub:
        return [origin]

    path = [origin]

    # Check if origin needs to route through its parent first
    origin_parent = parent_map.get(origin, origin)
    if origin_parent != origin and origin_parent != region_hub:
        path.append(origin_parent)

    # Check if origin needs to route through its regional hub first
    origin_region = fac.at[origin, 'regional_sort_hub']
    if pd.isna(origin_region) or origin_region == '':
        origin_region = origin

    if origin_region != origin and origin_region != region_hub:
        if origin_region not in path:
            path.append(origin_region)

    # Finally add destination regional hub if not already there
    if region_hub not in path:
        path.append(region_hub)

    return path


# ============================================================================
# HELPER FUNCTIONS (UNCHANGED)
# ============================================================================

def summarize_od_matrix(od: pd.DataFrame) -> pd.Series:
    """Generate summary statistics for OD matrix."""
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
    """Generate summary statistics for candidate paths."""
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
    """Validate path structure and identify issues."""
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