# veho_net/sort_optimization.py - NEW MODULE for containerization optimization
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from .geo import haversine_miles, band_lookup


def validate_sort_capacity_feasibility(facilities: pd.DataFrame, od_selected: pd.DataFrame,
                                       timing_kv: dict) -> None:
    """
    Validate that minimum operational requirements can be met before optimization.

    Raises ValueError if sort point capacity is insufficient for minimum requirements.
    Prints warnings for sort throughput constraints (if populated).
    """
    capacity_warnings = []
    throughput_warnings = []

    sort_points_per_dest = int(timing_kv.get('sort_points_per_destination', 2))

    # Check each facility that can be an origin
    origin_facilities = od_selected['origin'].unique()

    for facility_name in origin_facilities:
        facility_row = facilities[facilities['facility_name'] == facility_name]
        if facility_row.empty:
            continue

        facility = facility_row.iloc[0]

        # Skip if not hub/hybrid (shouldn't be origins anyway, but safety check)
        if facility['type'] not in ['hub', 'hybrid']:
            continue

        # Calculate minimum sort points needed (region level minimum)
        facility_ods = od_selected[od_selected['origin'] == facility_name]
        unique_regions = facility_ods['dest'].apply(lambda x: get_region_for_facility(x, facilities)).nunique()
        min_required = unique_regions * sort_points_per_dest

        max_capacity = facility.get('max_sort_points_capacity', 0)
        if pd.isna(max_capacity):
            max_capacity = 0
        max_capacity = int(max_capacity)

        if max_capacity < min_required:
            capacity_warnings.append(
                f"{facility_name} needs {min_required} sort points "
                f"(regions={unique_regions} × {sort_points_per_dest} points/dest) "
                f"but only has capacity for {max_capacity}"
            )

        # Sort throughput check (if populated)
        throughput_capacity = facility.get('sort_throughput_pkgs_per_hour', None)
        if not pd.isna(throughput_capacity) and throughput_capacity > 0:
            daily_volume = facility_ods['pkgs_day'].sum()
            # Assuming 16-hour sort window per day (can be made configurable)
            daily_capacity = float(throughput_capacity) * 16

            if daily_volume > daily_capacity:
                throughput_warnings.append(
                    f"{facility_name} has {daily_volume:,.0f} pkgs/day volume "
                    f"but only {daily_capacity:,.0f} pkgs/day capacity"
                )

    if capacity_warnings:
        raise ValueError("Sort point capacity insufficient for minimum requirements:\n" +
                         "\n".join(capacity_warnings))

    if throughput_warnings:
        print("WARNING: Sort throughput constraints detected:\n" + "\n".join(throughput_warnings))


def get_region_for_facility(facility_name: str, facilities: pd.DataFrame) -> str:
    """Get the region for a facility, using parent hub's region if it's a launch facility."""
    facility_row = facilities[facilities['facility_name'] == facility_name]
    if facility_row.empty:
        return facility_name

    facility = facility_row.iloc[0]

    # If it's a launch facility, use parent hub's region
    if facility['type'] == 'launch':
        parent_hub = facility['parent_hub_name']
        if not pd.isna(parent_hub) and parent_hub != facility_name:
            parent_row = facilities[facilities['facility_name'] == parent_hub]
            if not parent_row.empty:
                return parent_row.iloc[0]['region']

    return facility['region']


def calculate_containerization_costs(od_selected: pd.DataFrame, facilities: pd.DataFrame,
                                     mileage_bands: pd.DataFrame, costs: dict,
                                     timing_kv: dict) -> pd.DataFrame:
    """
    Calculate costs for each OD pair at different containerization levels.

    Returns DataFrame with columns:
    - od_pair_id, origin, dest, pkgs_day
    - region_cost, market_cost, sort_group_cost
    - region_sort_points, market_sort_points, sort_group_sort_points
    """
    results = []

    sort_points_per_dest = int(timing_kv.get('sort_points_per_destination', 2))

    for _, od in od_selected.iterrows():
        origin_name = od['origin']
        dest_name = od['dest']
        pkgs_day = float(od['pkgs_day'])

        # Get facility info
        origin_fac = facilities[facilities['facility_name'] == origin_name].iloc[0]
        dest_fac = facilities[facilities['facility_name'] == dest_name].iloc[0]

        # Calculate costs for each containerization level
        cost_data = {
            'od_pair_id': f"{origin_name}_{dest_name}",
            'origin': origin_name,
            'dest': dest_name,
            'pkgs_day': pkgs_day,
        }

        for level in ['region', 'market', 'sort_group']:
            cost, sort_points = calculate_level_cost(
                origin_fac, dest_fac, pkgs_day, level,
                facilities, mileage_bands, costs, timing_kv, sort_points_per_dest
            )
            cost_data[f'{level}_cost'] = cost
            cost_data[f'{level}_sort_points'] = sort_points

        results.append(cost_data)

    return pd.DataFrame(results)


def calculate_level_cost(origin_fac: pd.Series, dest_fac: pd.Series, pkgs_day: float,
                         level: str, facilities: pd.DataFrame, mileage_bands: pd.DataFrame,
                         costs: dict, timing_kv: dict, sort_points_per_dest: int) -> Tuple[float, int]:
    """
    Calculate total cost for this OD pair at the specified containerization level.

    Returns (total_cost, sort_points_needed)
    """

    # Determine containerization approach
    if level == 'region':
        # Container goes to destination region (parent hub), then gets sorted locally
        if dest_fac['type'] == 'launch':
            container_dest = dest_fac['parent_hub_name']
        else:
            container_dest = dest_fac['facility_name']
        sort_points_needed = sort_points_per_dest  # 1 destination region

    elif level == 'market':
        # Container goes directly to destination market
        container_dest = dest_fac['facility_name']
        sort_points_needed = sort_points_per_dest  # 1 destination market

    else:  # sort_group
        # Container is pre-sorted to sort groups
        container_dest = dest_fac['facility_name']
        # Get number of sort groups for this destination facility
        sort_groups = dest_fac.get('last_mile_sort_groups_count', 4)  # Default 4
        sort_points_needed = int(sort_groups) * sort_points_per_dest

    # Calculate linehaul cost (origin to container destination)
    linehaul_cost = calculate_linehaul_cost(
        origin_fac, container_dest, pkgs_day, facilities, mileage_bands
    )

    # Calculate sorting costs
    origin_sort_cost = pkgs_day * float(costs.get('sort_cost_per_pkg', 0.0))

    if level == 'region':
        # Additional sorting at destination parent hub (region level)
        dest_sort_cost = pkgs_day * float(costs.get('last_mile_sort_cost_per_pkg', 0.0))
    elif level == 'market':
        # Crossdock at destination market
        dest_sort_cost = pkgs_day * float(costs.get('crossdock_touch_cost_per_pkg', 0.0))
    else:  # sort_group
        # Pre-sorted, minimal handling at destination
        dest_sort_cost = pkgs_day * float(costs.get('crossdock_touch_cost_per_pkg', 0.0)) * 0.5

    # Setup costs (if any)
    setup_cost = sort_points_needed * float(costs.get('sort_setup_cost_per_point', 0.0))

    # Last mile delivery cost (same regardless of containerization level)
    delivery_cost = pkgs_day * float(costs.get('last_mile_delivery_cost_per_pkg', 0.0))

    total_cost = linehaul_cost + origin_sort_cost + dest_sort_cost + setup_cost + delivery_cost

    return total_cost, sort_points_needed


def calculate_linehaul_cost(origin_fac: pd.Series, dest_facility_name: str, pkgs_day: float,
                            facilities: pd.DataFrame, mileage_bands: pd.DataFrame) -> float:
    """Calculate linehaul transportation cost between two facilities."""

    # Get destination facility info
    dest_fac = facilities[facilities['facility_name'] == dest_facility_name].iloc[0]

    # Calculate distance
    distance = haversine_miles(
        float(origin_fac['lat']), float(origin_fac['lon']),
        float(dest_fac['lat']), float(dest_fac['lon'])
    )

    # Get mileage band costs
    fixed_cost, var_cost, circuity, mph = band_lookup(distance, mileage_bands)
    actual_distance = distance * circuity

    # Estimate trucks needed (simplified - could use container logic)
    # Assume 2000 packages per truck on average
    trucks_needed = max(1, pkgs_day / 2000.0)

    return trucks_needed * (fixed_cost + var_cost * actual_distance)


def optimize_sort_allocation(cost_analysis: pd.DataFrame, facilities: pd.DataFrame,
                             timing_kv: dict) -> Dict[str, str]:
    """
    Hybrid optimization: Allocate sort capacity to maximize cost savings per sort point.

    Returns dictionary mapping od_pair_id to optimal containerization level.
    """

    # Step 1: Calculate savings and efficiency for each opportunity
    opportunities = []

    for _, row in cost_analysis.iterrows():
        baseline_cost = row['region_cost']  # Region level is baseline

        for level in ['market', 'sort_group']:
            cost = row[f'{level}_cost']
            sort_points = row[f'{level}_sort_points']
            baseline_points = row['region_sort_points']

            incremental_sort_points = sort_points - baseline_points
            cost_savings = baseline_cost - cost

            if incremental_sort_points > 0 and cost_savings > 0:
                efficiency = cost_savings / incremental_sort_points

                opportunities.append({
                    'od_pair_id': row['od_pair_id'],
                    'origin': row['origin'],
                    'level': level,
                    'cost_savings': cost_savings,
                    'incremental_sort_points': incremental_sort_points,
                    'efficiency': efficiency,
                    'pkgs_day': row['pkgs_day']
                })

    # Step 2: Sort by efficiency (cost savings per incremental sort point)
    opportunities.sort(key=lambda x: x['efficiency'], reverse=True)

    # Step 3: Calculate available capacity per facility
    facility_capacity = {}
    sort_points_per_dest = int(timing_kv.get('sort_points_per_destination', 2))

    for _, facility in facilities[facilities['type'].isin(['hub', 'hybrid'])].iterrows():
        facility_name = facility['facility_name']
        max_capacity = int(facility.get('max_sort_points_capacity', 0))

        # Calculate minimum required (region level for all destinations)
        facility_ods = cost_analysis[cost_analysis['origin'] == facility_name]
        min_required = len(facility_ods) * sort_points_per_dest

        available = max_capacity - min_required
        facility_capacity[facility_name] = max(0, available)

    # Step 4: Greedy allocation based on efficiency
    allocation = {row['od_pair_id']: 'region' for _, row in cost_analysis.iterrows()}

    for opp in opportunities:
        facility = opp['origin']
        needed = opp['incremental_sort_points']

        if facility_capacity.get(facility, 0) >= needed:
            allocation[opp['od_pair_id']] = opp['level']
            facility_capacity[facility] -= needed

            print(f"Allocated {opp['level']} to {opp['od_pair_id']}: "
                  f"saves ${opp['cost_savings']:.0f}/day with {needed} sort points "
                  f"(${opp['efficiency']:.0f}/point/day)")

    return allocation


def apply_sort_allocation(od_selected: pd.DataFrame, allocation: Dict[str, str],
                          cost_analysis: pd.DataFrame, facilities: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced sort allocation application with fill and spill analysis.

    Adds containerization columns plus fill/spill opportunity analysis.
    """
    od_result = od_selected.copy()

    # Create od_pair_id for matching
    od_result['od_pair_id'] = od_result['origin'] + '_' + od_result['dest']

    # Apply allocation decisions
    od_result['containerization_level'] = od_result['od_pair_id'].map(allocation)
    od_result['containerization_level'].fillna('region', inplace=True)

    # Add sort points and efficiency data
    cost_lookup = cost_analysis.set_index('od_pair_id')

    od_result['sort_points_used'] = od_result.apply(
        lambda row: cost_lookup.loc[row['od_pair_id'], f"{row['containerization_level']}_sort_points"]
        if row['od_pair_id'] in cost_lookup.index else 2, axis=1
    )

    # Calculate efficiency score (cost savings vs region baseline)
    od_result['containerization_efficiency_score'] = od_result.apply(
        lambda row: _calculate_efficiency_score(row, cost_lookup), axis=1
    )

    # Enhanced: Fill and Spill Analysis
    od_result = _add_fill_spill_analysis(od_result, facilities, allocation)

    return od_result


def _calculate_efficiency_score(row: pd.Series, cost_lookup: pd.DataFrame) -> float:
    """Calculate cost savings per sort point for this OD pair."""
    od_pair_id = row['od_pair_id']
    level = row['containerization_level']

    if od_pair_id not in cost_lookup.index:
        return 0.0

    cost_row = cost_lookup.loc[od_pair_id]
    region_cost = cost_row.get('region_cost', 0)
    level_cost = cost_row.get(f'{level}_cost', region_cost)
    sort_points = cost_row.get(f'{level}_sort_points', 1)

    cost_savings = region_cost - level_cost
    return cost_savings / max(sort_points, 1) if sort_points > 0 else 0.0


def _add_fill_spill_analysis(od_selected: pd.DataFrame, facilities: pd.DataFrame,
                             allocation: Dict[str, str]) -> pd.DataFrame:
    """
    Add fill and spill opportunity analysis to OD data.

    Key logic: If origin does market-level to facility A, but region-level to facility A's parent hub,
    then market containers can "spill" into region containers when needed.
    """
    od_result = od_selected.copy()

    # Create facility lookup
    fac_lookup = facilities.set_index('facility_name')[['parent_hub_name', 'type']].to_dict('index')

    # Add destination parent hub info
    od_result['destination_parent_hub'] = od_result['dest'].map(
        lambda dest: fac_lookup.get(dest, {}).get('parent_hub_name', dest)
    )

    # Determine if origin has region-level containerization to destination's parent
    od_result['has_secondary_region_sort'] = False
    od_result['spill_opportunity_flag'] = False
    od_result['spill_parent_hub'] = None

    # Group by origin to analyze all destinations from each origin
    for origin, origin_group in od_result.groupby('origin'):
        origin_allocations = {}

        # Build allocation map for this origin
        for _, row in origin_group.iterrows():
            dest = row['dest']
            parent = row['destination_parent_hub']
            level = row['containerization_level']

            origin_allocations[dest] = level
            origin_allocations[parent] = origin_allocations.get(parent, 'region')  # Default assumption

        # Check actual allocation data for parent hubs
        for od_pair_id, allocated_level in allocation.items():
            if od_pair_id.startswith(f"{origin}_"):
                dest = od_pair_id.split('_', 1)[1]
                if dest in fac_lookup and fac_lookup[dest]['type'] in ['hub', 'hybrid']:
                    origin_allocations[dest] = allocated_level

        # Now determine fill and spill opportunities
        origin_indices = origin_group.index
        for idx in origin_indices:
            row = od_result.loc[idx]
            dest = row['dest']
            parent = row['destination_parent_hub']
            dest_level = row['containerization_level']

            # Check if parent hub gets region-level containerization
            parent_level = origin_allocations.get(parent, 'region')

            od_result.at[idx, 'has_secondary_region_sort'] = (parent_level == 'region' and parent != dest)

            # Spill opportunity exists if:
            # 1. This destination gets market/sort_group level containerization
            # 2. Parent hub gets region-level containerization
            # 3. Parent hub is different from destination
            if (dest_level in ['market', 'sort_group'] and
                    parent_level == 'region' and
                    parent != dest):
                od_result.at[idx, 'spill_opportunity_flag'] = True
                od_result.at[idx, 'spill_parent_hub'] = parent

    return od_result


def summarize_sort_allocation(od_selected: pd.DataFrame, cost_analysis: pd.DataFrame,
                              allocation: Dict[str, str]) -> pd.DataFrame:
    """
    Enhanced summary of sort allocation decisions with fill and spill insights.
    """
    summary_data = []

    for od_pair_id, level in allocation.items():
        cost_row = cost_analysis[cost_analysis['od_pair_id'] == od_pair_id]
        if cost_row.empty:
            continue

        cost_row = cost_row.iloc[0]

        baseline_cost = cost_row['region_cost']
        actual_cost = cost_row[f'{level}_cost']
        cost_savings = baseline_cost - actual_cost
        sort_points = cost_row[f'{level}_sort_points']
        efficiency_score = cost_savings / max(sort_points, 1)

        # Get fill and spill data from OD selected if available
        od_row = od_selected[od_selected['od_pair_id'] == od_pair_id]
        spill_data = {}
        if not od_row.empty:
            row = od_row.iloc[0]
            spill_data = {
                'has_spill_opportunity': row.get('spill_opportunity_flag', False),
                'spill_parent_hub': row.get('spill_parent_hub', ''),
                'has_secondary_region_sort': row.get('has_secondary_region_sort', False),
                'destination_parent_hub': row.get('destination_parent_hub', ''),
            }

        summary_data.append({
            'od_pair_id': od_pair_id,
            'origin': cost_row['origin'],
            'dest': cost_row['dest'],
            'pkgs_day': cost_row['pkgs_day'],
            'containerization_level': level,
            'daily_cost_savings': cost_savings,
            'sort_points_used': sort_points,
            'efficiency_score': efficiency_score,
            **spill_data
        })

    summary = pd.DataFrame(summary_data)

    # Enhanced summary stats with fill and spill insights
    print("\n=== ENHANCED SORT ALLOCATION SUMMARY ===")
    print(f"Total daily cost savings: ${summary['daily_cost_savings'].sum():,.0f}")
    print(f"Total sort points allocated: {summary['sort_points_used'].sum():,.0f}")
    print(f"Average efficiency score: ${summary['efficiency_score'].mean():.0f}/point/day")

    print("\nAllocation by level:")
    level_summary = summary.groupby('containerization_level').agg({
        'od_pair_id': 'count',
        'pkgs_day': 'sum',
        'daily_cost_savings': 'sum',
        'sort_points_used': 'sum',
        'efficiency_score': 'mean'
    }).round(1)
    print(level_summary)

    # Fill and spill insights
    if 'has_spill_opportunity' in summary.columns:
        spill_routes = summary[summary['has_spill_opportunity'] == True]
        if not spill_routes.empty:
            print(f"\nFill & Spill Opportunities:")
            print(f"  Routes with spill capability: {len(spill_routes)}")
            print(f"  Spillable volume: {spill_routes['pkgs_day'].sum():,.0f} pkgs/day")
            print(f"  Affected parent hubs: {spill_routes['spill_parent_hub'].nunique()}")

    return summary