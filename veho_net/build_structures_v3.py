"""
Network Structure Generation Module

Builds OD matrices and generates candidate paths with proper O=D handling.

Key Business Rules:
- Direct injection: Zone 0, no middle-mile routing
- Middle-mile O=D: Zone 2, injection sort + last mile (no linehaul)
- Only hub/hybrid facilities can originate middle-mile flows
- Launch facilities receive only, never send
- Secondary hubs route outbound through parent
- No circular routing (facility cannot appear twice in path)
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from .geo_v3 import haversine_miles


def build_od_and_direct(
        facilities: pd.DataFrame,
        zips: pd.DataFrame,
        year_demand: pd.DataFrame,
        injection_distribution: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate OD matrix for middle-mile flows and direct injection volumes.

    Two flow types:
    1. Direct injection (Zone 0): Shipper → Last mile facility (bypasses middle-mile)
    2. Middle-mile (Zones 1-5): Hub → Hub/Launch (through network)

    Middle-mile O=D handling:
    - If origin = dest and facility is hybrid: Include in middle-mile OD
    - Gets injection sort + last mile costs, but no linehaul
    - Classified as Zone 2 (not Zone 0)

    Args:
        facilities: Facility master with type, coordinates, injection flags
        zips: ZIP code assignments with population
        year_demand: Annual demand forecast with peak/offpeak splits
        injection_distribution: Injection share allocation

    Returns:
        Tuple of (od_matrix, direct_injection, destination_population)
    """
    # Validate required columns
    req_fac = {"facility_name", "type", "lat", "lon", "parent_hub_name",
               "regional_sort_hub", "is_injection_node"}
    if not req_fac.issubset(facilities.columns):
        missing = req_fac - set(facilities.columns)
        raise ValueError(f"facilities missing columns: {sorted(missing)}")

    req_zips = {"zip", "facility_name_assigned", "population"}
    if not req_zips.issubset(zips.columns):
        missing = req_zips - set(zips.columns)
        raise ValueError(f"zips missing columns: {sorted(missing)}")

    req_dem = {"year", "annual_pkgs", "offpeak_pct_of_annual", "peak_pct_of_annual",
               "middle_mile_share_offpeak", "middle_mile_share_peak"}
    if not req_dem.issubset(year_demand.columns):
        missing = req_dem - set(year_demand.columns)
        raise ValueError(f"demand missing columns: {sorted(missing)}")

    # Prepare data
    fac = facilities.drop_duplicates(subset=["facility_name"]).reset_index(drop=True)
    fac["type"] = fac["type"].str.lower()

    z = zips[["zip", "facility_name_assigned", "population"]].drop_duplicates(subset=["zip"]).copy()

    # Calculate destination population shares
    pop_by_dest = z.groupby("facility_name_assigned", as_index=False)["population"].sum()
    pop_by_dest["dest_pop_share"] = pop_by_dest["population"] / pop_by_dest["population"].sum()

    # Extract demand parameters
    year_val = int(year_demand["year"].iloc[0])
    annual_total = float(year_demand["annual_pkgs"].sum())
    off_pct = float(year_demand["offpeak_pct_of_annual"].iloc[0])
    peak_pct = float(year_demand["peak_pct_of_annual"].iloc[0])
    mm_off = float(year_demand["middle_mile_share_offpeak"].iloc[0])
    mm_peak = float(year_demand["middle_mile_share_peak"].iloc[0])

    # Direct injection volumes (Zone 0 - all facilities receive)
    direct = pop_by_dest.rename(columns={"facility_name_assigned": "dest"}).copy()
    direct["year"] = year_val
    direct["dir_pkgs_offpeak_day"] = annual_total * off_pct * (1.0 - mm_off) * direct["dest_pop_share"]
    direct["dir_pkgs_peak_day"] = annual_total * peak_pct * (1.0 - mm_peak) * direct["dest_pop_share"]

    # Middle-mile injection: ONLY hub/hybrid facilities can originate
    inj = injection_distribution[["facility_name", "absolute_share"]].copy()
    inj = inj.merge(
        fac[["facility_name", "is_injection_node", "type", "parent_hub_name"]],
        on="facility_name",
        how="left",
        validate="many_to_one"
    )

    if inj["is_injection_node"].isna().any():
        missing = inj.loc[inj["is_injection_node"].isna(), "facility_name"].unique().tolist()
        raise ValueError(f"injection_distribution facilities not found: {missing}")

    # Filter to valid injection origins (hub/hybrid with injection enabled)
    valid_origin_types = ["hub", "hybrid"]
    inj = inj[
        (inj["is_injection_node"].astype(int) == 1) &
        (inj["type"].isin(valid_origin_types))
        ].copy()

    if inj.empty:
        raise ValueError("No valid injection origins found (hub/hybrid with is_injection_node=1)")

    # Normalize injection shares
    inj["abs_w"] = pd.to_numeric(inj["absolute_share"], errors="coerce").fillna(0.0)
    total_w = float(inj["abs_w"].sum())
    if total_w <= 0:
        raise ValueError("injection_distribution.absolute_share must sum > 0")

    inj["inj_share"] = inj["abs_w"] / total_w
    inj = inj[["facility_name", "inj_share"]].rename(columns={"facility_name": "origin"})

    # Build middle-mile OD matrix
    dest2 = pop_by_dest.rename(columns={"facility_name_assigned": "dest"})[["dest", "dest_pop_share"]]
    grid = inj.assign(_k=1).merge(dest2.assign(_k=1), on="_k").drop(columns="_k")

    od = grid.copy()
    od["pkgs_offpeak_day"] = annual_total * off_pct * mm_off * od["inj_share"] * od["dest_pop_share"]
    od["pkgs_peak_day"] = annual_total * peak_pct * mm_peak * od["inj_share"] * od["dest_pop_share"]

    # O=D handling: Keep for hybrid facilities (they do own delivery)
    # Remove for pure hub facilities (they don't do delivery)
    fac_types = fac.set_index("facility_name")["type"].to_dict()

    od_filtered = []
    for _, row in od.iterrows():
        origin = row["origin"]
        dest = row["dest"]

        if origin == dest:
            # Keep O=D only if origin is hybrid (has delivery capability)
            origin_type = fac_types.get(origin, "").lower()
            if origin_type == "hybrid":
                od_filtered.append(row)
        else:
            # Keep all O≠D pairs
            od_filtered.append(row)

    od = pd.DataFrame(od_filtered).reset_index(drop=True)

    return od, direct, pop_by_dest


def candidate_paths(
        od: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame,
        around_factor: float
) -> pd.DataFrame:
    """
    Generate candidate paths with routing constraints and strategy hints.

    Generates both standard paths and fluid-optimized variants for multi-hop ODs.
    MILP will choose optimal paths based on costs.

    Routing Rules:
    1. Secondary hubs MUST route outbound through parent
    2. Launch destinations MUST have parent hub as second-to-last stop
    3. Only hub/hybrid can be intermediate stops
    4. No circular routing
    5. Path distance ≤ around_factor × straight-line distance

    Args:
        od: OD matrix with origin, dest, package volumes
        facilities: Facility master
        mileage_bands: Distance parameters
        around_factor: Max path distance multiple (from run_settings)

    Returns:
        DataFrame of candidate paths with path_nodes, path_str, strategy_hint
    """
    fac = facilities.set_index("facility_name").copy()
    fac["type"] = fac["type"].astype(str).str.lower()

    # Only hub/hybrid can be intermediate stops
    valid_intermediate_types = ["hub", "hybrid"]
    hubs_enabled = fac.index[fac["type"].isin(valid_intermediate_types)]

    def raw_distance(o, d):
        """Calculate straight-line distance."""
        return haversine_miles(
            float(fac.at[o, "lat"]), float(fac.at[o, "lon"]),
            float(fac.at[d, "lat"]), float(fac.at[d, "lon"])
        )

    # Build facility relationships
    parent_map = {}
    primary_hubs = set()
    secondary_hubs = set()
    launch_facilities = set()

    for facility_name in fac.index:
        facility_type = fac.at[facility_name, "type"]
        parent_hub = fac.at[facility_name, "parent_hub_name"]

        if pd.isna(parent_hub) or parent_hub == "":
            parent_hub = facility_name
        parent_map[facility_name] = parent_hub

        if facility_type in ["hub", "hybrid"]:
            if parent_hub == facility_name:
                primary_hubs.add(facility_name)
            else:
                secondary_hubs.add(facility_name)
        elif facility_type == "launch":
            launch_facilities.add(facility_name)

    def build_path_with_constraints(origin, dest):
        """Build valid paths respecting routing rules."""
        paths = []

        # Check for O=D
        if origin == dest:
            # O=D: Direct path only (no routing needed)
            paths.append([origin, dest])
            return paths

        # Determine destination constraint (launch parent rule)
        if dest in launch_facilities:
            dest_parent = parent_map.get(dest, dest)
            if dest_parent == dest:
                required_end = [dest]
            else:
                required_end = [dest_parent, dest]
        else:
            required_end = [dest]

        # Check circular routing
        if origin in required_end:
            paths.append([origin, dest])
            return paths

        # Determine origin constraint (secondary hub parent rule)
        if origin in secondary_hubs:
            origin_parent = parent_map[origin]
            if origin_parent == origin:
                required_start = [origin]
            else:
                if origin_parent in required_end or origin in required_end:
                    required_start = [origin]
                else:
                    required_start = [origin, origin_parent]
        else:
            required_start = [origin]

        # Build paths
        start_node = required_start[-1]
        end_node = required_end[0]

        if start_node == end_node:
            combined = required_start + required_end[1:]
            if len(combined) == len(set(combined)):
                paths.append(combined)
        else:
            direct_distance = raw_distance(start_node, end_node)

            # Option 1: Direct connection
            combined = required_start + required_end
            if len(combined) == len(set(combined)):
                paths.append(combined)

            # Option 2: One intermediate hub (for consolidation)
            candidate_intermediates = [
                h for h in hubs_enabled
                if h not in required_start and h not in required_end
            ]

            if candidate_intermediates:
                best_intermediate = min(
                    candidate_intermediates,
                    key=lambda h: raw_distance(start_node, h) + raw_distance(h, end_node)
                )

                intermediate_distance = (
                        raw_distance(start_node, best_intermediate) +
                        raw_distance(best_intermediate, end_node)
                )

                if intermediate_distance <= around_factor * direct_distance:
                    intermediate_path = required_start + [best_intermediate] + required_end
                    if len(intermediate_path) == len(set(intermediate_path)):
                        paths.append(intermediate_path)

        return paths

    # Generate all candidate paths
    rows = []

    for _, od_row in od.iterrows():
        origin = od_row["origin"]
        dest = od_row["dest"]
        od_volume = od_row.get("pkgs_peak_day", od_row.get("pkgs_offpeak_day", 0))

        # Generate standard paths
        candidate_paths_list = build_path_with_constraints(origin, dest)

        for path_nodes in candidate_paths_list:
            # Classify path type
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
                "path_nodes": path_nodes,
                "path_str": "->".join(path_nodes),
                "strategy_hint": None  # Use global strategy
            })

        # For multi-hop paths, generate fluid-optimized direct variant
        # (No threshold - let MILP decide if cost-effective)
        multi_hop_exists = any(len(p) > 2 for p in candidate_paths_list)

        if multi_hop_exists and origin != dest:
            # Offer direct long-haul as fluid option
            rows.append({
                "origin": origin,
                "dest": dest,
                "path_type": "direct",
                "path_nodes": [origin, dest],
                "path_str": f"{origin}->{dest}",
                "strategy_hint": "fluid"  # Suggest fluid may be better
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Remove invalid paths (launch facilities as intermediate)
    def validate_path_nodes(path_nodes):
        for i, node in enumerate(path_nodes[1:-1], 1):
            if node in fac.index and fac.at[node, "type"] == "launch":
                return False
        return True

    initial_count = len(df)
    df = df[df["path_nodes"].apply(validate_path_nodes)].reset_index(drop=True)

    removed_count = initial_count - len(df)
    if removed_count > 0:
        print(f"  Removed {removed_count} invalid paths (launch facility violations)")

    # Remove exact duplicates
    df["path_nodes_tuple"] = df["path_nodes"].apply(tuple)
    df = df.drop_duplicates(subset=["origin", "dest", "path_nodes_tuple"]).drop(
        columns=["path_nodes_tuple"]
    ).reset_index(drop=True)

    return df