"""
Sort Optimization Module

Multi-level sort optimization logic (OPTIONAL FEATURE).
Implements region/market/sort_group sorting with capacity constraints.

Only used when enable_sort_optimization=True in run_settings.

TODO: This is a placeholder for future implementation.
The full sort optimization logic from the sort model needs to be:
1. Simplified based on our design discussions
2. Integrated into the MILP as binary decision variables
3. Tested with capacity constraints

For Phase 1, this module contains helper functions but sort optimization
is not yet integrated into the main MILP.
"""

import pandas as pd
from typing import Dict, List, Set
from .config_v3 import SortLevel


def build_facility_relationships(facilities: pd.DataFrame) -> Dict:
    """
    Build facility relationship mapping for sort decisions.

    Returns dict with:
    - parent_hub_map: Routing hierarchy (parent_hub_name)
    - regional_sort_hub_map: Sort consolidation hierarchy (regional_sort_hub)
    - facility_types: hub/hybrid/launch classification
    - hub_facilities: Set of hub/hybrid facilities
    - launch_facilities: Set of launch facilities

    Args:
        facilities: Facility master data

    Returns:
        Dictionary with facility relationships
    """
    relationships = {}

    parent_hub_map = {}
    regional_sort_hub_map = {}
    facility_types = {}
    launch_facilities = set()
    hub_facilities = set()

    for _, row in facilities.iterrows():
        facility_name = row['facility_name']
        facility_type = str(row['type']).lower()
        parent_hub = row.get('parent_hub_name', facility_name)
        regional_sort_hub = row.get('regional_sort_hub', parent_hub)

        # Default to self if blank
        if pd.isna(parent_hub) or parent_hub == "":
            parent_hub = facility_name
        if pd.isna(regional_sort_hub) or regional_sort_hub == "":
            regional_sort_hub = parent_hub

        parent_hub_map[facility_name] = parent_hub
        regional_sort_hub_map[facility_name] = regional_sort_hub
        facility_types[facility_name] = facility_type

        if facility_type == 'launch':
            launch_facilities.add(facility_name)
        elif facility_type in ['hub', 'hybrid']:
            hub_facilities.add(facility_name)

    relationships = {
        'parent_hub_map': parent_hub_map,
        'regional_sort_hub_map': regional_sort_hub_map,
        'facility_types': facility_types,
        'launch_facilities': launch_facilities,
        'hub_facilities': hub_facilities
    }

    return relationships


def get_sort_level_options(
        od_row: pd.Series,
        facilities: pd.DataFrame,
        relationships: Dict
) -> List[str]:
    """
    Determine valid sort levels for an OD pair.

    Business Rules:
    - O=D (self-destination): MUST use sort_group
    - Regional hub → own children: Cannot use region (already in region)
    - All others: Can use region, market, or sort_group

    Args:
        od_row: OD pair data
        facilities: Facility master
        relationships: Facility relationships from build_facility_relationships()

    Returns:
        List of valid sort level strings
    """
    origin = od_row['origin']
    dest = od_row['dest']

    # O=D special case
    if origin == dest:
        return ['sort_group']

    # Check if origin is regional hub for destination
    regional_sort_hub_map = relationships['regional_sort_hub_map']
    dest_regional_hub = regional_sort_hub_map.get(dest, dest)

    if origin == dest_regional_hub:
        # Origin is regional hub for destination - cannot use region level
        return ['market', 'sort_group']

    # All other cases: all three levels available
    return ['region', 'market', 'sort_group']


def calculate_sort_capacity_required(
        origin: str,
        destinations: List[str],
        sort_level: str,
        sort_points_per_dest: float,
        facilities: pd.DataFrame,
        relationships: Dict
) -> float:
    """
    Calculate sort points required at origin for given destinations and level.

    Formulas:
    - Region: unique_regional_hubs × sort_points_per_dest
    - Market: num_destinations × sort_points_per_dest
    - Sort_group: Σ(dest_sort_groups × sort_points_per_dest)

    Args:
        origin: Origin facility
        destinations: List of destination facilities
        sort_level: Sort level choice
        sort_points_per_dest: Base sort points per destination
        facilities: Facility master
        relationships: Facility relationships

    Returns:
        Total sort points required
    """
    regional_sort_hub_map = relationships['regional_sort_hub_map']
    facility_lookup = facilities.set_index('facility_name')

    if sort_level == 'region':
        # Unique regional sort hubs
        unique_hubs = set(regional_sort_hub_map.get(d, d) for d in destinations)
        return len(unique_hubs) * sort_points_per_dest

    elif sort_level == 'market':
        # One per destination facility
        return len(destinations) * sort_points_per_dest

    elif sort_level == 'sort_group':
        # Multiply by sort groups per destination
        total_points = 0
        for dest in destinations:
            if dest in facility_lookup.index:
                sort_groups = facility_lookup.at[dest, 'last_mile_sort_groups_count']
                if pd.isna(sort_groups) or sort_groups <= 0:
                    sort_groups = 4  # Conservative fallback
                total_points += sort_points_per_dest * float(sort_groups)
        return total_points

    return 0


# TODO: The following functions need to be implemented for full sort optimization

def add_sort_optimization_to_milp(model, candidates, facilities, cost_params, timing_params):
    """
    Add sort level decision variables and constraints to MILP.

    TODO: Implement this to integrate sort optimization into main MILP.

    Would add:
    - Binary variables for sort level choice per OD
    - Capacity constraints per facility
    - Cost terms dependent on sort level choice
    - Regional hub → children constraints
    """
    raise NotImplementedError("Sort optimization not yet integrated into MILP")


def build_sort_decision_summary(od_selected, sort_decisions, cost_params, facilities):
    """
    Build summary of sort level decisions with cost breakdown.

    TODO: Implement after sort optimization integrated into MILP.

    Would return DataFrame with:
    - Origin, dest, chosen_sort_level
    - Sort capacity consumed
    - Cost by sort level component
    - Savings vs. market level baseline
    """
    raise NotImplementedError("Sort optimization not yet integrated")


# Placeholder for future full implementation
def solve_with_sort_optimization(*args, **kwargs):
    """
    Extended MILP solver with sort optimization.

    TODO: This is where the full sort optimization logic would go.

    Would need to:
    1. Add sort level binary variables to MILP
    2. Add capacity constraints
    3. Link sort decisions to processing costs
    4. Enforce regional hub → children rules
    5. Return sort_summary with decisions
    """
    raise NotImplementedError(
        "Sort optimization not yet implemented in v3. "
        "Set enable_sort_optimization=False in run_settings."
    )