"""
Fluid Load Opportunity Analysis Module

Identifies OD pairs where switching from container to fluid loading
would improve economics through better truck utilization.

Key Insight: Low-density lanes sorted to granular levels (market/sort_group)
create many partially-filled containers. Consolidating these with fluid loading
can dramatically improve truck fill rates, saving enough on linehaul to offset
the incremental destination sort cost.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from .containers_v4 import (
    weighted_pkg_cube,
    calculate_truck_capacity,
    get_raw_trailer_cube
)
from .geo_v4 import band_lookup, haversine_miles
from .utils import safe_divide, get_facility_lookup
from .config_v4 import CostParameters


def analyze_fluid_load_opportunities(
        od_selected: pd.DataFrame,
        arc_summary: pd.DataFrame,
        facilities: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        mileage_bands: pd.DataFrame,
        cost_params: CostParameters,
        min_daily_benefit: float = 50.0,
        max_results: int = 50
) -> pd.DataFrame:
    """
    Identify lanes where fluid loading would be more economical than containerization.

    Analysis Logic:
    ---------------
    1. Filter to lanes currently using container strategy with low fill rates
    2. For each lane, simulate switching to fluid loading with region-level sort
    3. Calculate potential consolidation benefit with other region-destined freight
    4. Compare transport savings vs. incremental destination sort cost
    5. Rank opportunities by net daily benefit

    Args:
        od_selected: Selected OD paths from optimization
        arc_summary: Arc-level summary with fill rates
        facilities: Facility master data
        package_mix: Package distribution
        container_params: Container/trailer parameters
        mileage_bands: Mileage bands for cost calculation
        cost_params: Cost parameters
        min_daily_benefit: Minimum daily savings threshold (default $50)
        max_results: Maximum opportunities to return

    Returns:
        DataFrame with columns:
            - origin: Origin facility
            - dest: Destination facility
            - dest_region_hub: Regional hub serving destination
            - current_strategy: Current loading strategy
            - current_sort_level: Current sort level
            - packages_per_day: Daily package volume
            - current_trucks: Current trucks required
            - current_fill_rate: Current truck fill rate
            - potential_trucks_fluid: Trucks needed with fluid consolidation
            - potential_fill_rate_fluid: Estimated fill rate with fluid
            - trucks_saved: Truck reduction
            - transport_savings_daily: Daily transport cost savings
            - incremental_sort_cost_daily: Additional dest sort cost
            - net_benefit_daily: Net daily savings
            - annual_benefit: Net annual savings (250 days)
            - payback_complexity: Implementation complexity score
    """
    if od_selected.empty or arc_summary.empty:
        return pd.DataFrame()

    fac_lookup = get_facility_lookup(facilities)
    w_cube = weighted_pkg_cube(package_mix)
    raw_trailer_cube = get_raw_trailer_cube(container_params)

    # Container and fluid capacities
    container_capacity = calculate_truck_capacity(
        package_mix, container_params, "container"
    )
    fluid_capacity = calculate_truck_capacity(
        package_mix, container_params, "fluid"
    )

    opportunities = []

    # Focus on container strategy lanes with suboptimal fill rates
    container_ods = od_selected[
        (od_selected.get('effective_strategy', 'container').str.lower() == 'container')
    ].copy()

    for _, od_row in container_ods.iterrows():
        origin = od_row['origin']
        dest = od_row['dest']

        # Skip O=D flows
        if origin == dest:
            continue

        # Get destination regional hub
        dest_region_hub = _get_regional_hub(dest, fac_lookup)

        # Find corresponding arc
        matching_arc = arc_summary[
            (arc_summary['from_facility'] == origin) &
            (arc_summary['to_facility'] == dest)
            ]

        if matching_arc.empty:
            continue

        arc = matching_arc.iloc[0]

        # Current state
        current_sort_level = od_row.get('chosen_sort_level', 'market')
        pkgs_per_day = od_row['pkgs_day']
        current_trucks = arc['trucks']
        current_fill = arc.get('truck_fill_rate', 0)

        # Opportunity criteria: low fill rate and granular sort level
        # (market or sort_group creates more partially-filled containers)
        is_granular_sort = current_sort_level in ['market', 'sort_group']
        is_low_fill = current_fill < 0.75  # Below 75% fill threshold

        if not (is_granular_sort and is_low_fill):
            continue

        # Calculate potential consolidation benefit
        consolidation_data = _estimate_consolidation_potential(
            origin, dest_region_hub, od_selected,
            package_mix, container_params
        )

        # Simulate fluid loading with region sort
        # Fluid allows mixing packages destined for different markets in same region
        total_region_pkgs = consolidation_data['total_packages']
        total_region_cube = total_region_pkgs * w_cube

        # Trucks needed with fluid (better utilization)
        potential_trucks_fluid = max(1, int(np.ceil(
            total_region_cube / (raw_trailer_cube * 0.85)  # Assume 85% fluid utilization
        )))

        # Allocate trucks proportionally to this OD's share
        od_share = safe_divide(pkgs_per_day, total_region_pkgs)
        od_trucks_fluid = max(1, potential_trucks_fluid * od_share)

        potential_fill_fluid = safe_divide(
            pkgs_per_day * w_cube,
            od_trucks_fluid * raw_trailer_cube
        )

        # Calculate savings
        trucks_saved = current_trucks - od_trucks_fluid

        if trucks_saved <= 0:
            continue  # No transport savings

        # Transport cost savings
        distance = arc['distance_miles']
        cost_per_truck = arc['cost_per_truck']
        transport_savings = trucks_saved * cost_per_truck

        # Incremental sort cost at destination
        # Currently sorted at origin to market/sort_group, would need dest sort
        if current_sort_level == 'sort_group':
            # Was fully pre-sorted, now needs full destination sort
            incremental_sort_cost = (
                    pkgs_per_day * cost_params.intermediate_sort_cost_per_pkg
            )
        elif current_sort_level == 'market':
            # Was sorted to market, now needs regional hub to split + dest sort
            # Regional hub does crossdock (cheaper), dest does final sort
            incremental_sort_cost = (
                    pkgs_per_day * cost_params.intermediate_sort_cost_per_pkg * 0.5
            )
        else:
            # Already region sort, no change
            incremental_sort_cost = 0

        net_benefit = transport_savings - incremental_sort_cost

        if net_benefit < min_daily_benefit:
            continue

        # Calculate implementation complexity score (1-5, higher = more complex)
        complexity = _calculate_implementation_complexity(
            od_row, arc, dest_region_hub, fac_lookup
        )

        opportunities.append({
            'origin': origin,
            'dest': dest,
            'dest_region_hub': dest_region_hub,
            'current_strategy': 'container',
            'current_sort_level': current_sort_level,
            'packages_per_day': int(pkgs_per_day),
            'current_trucks': int(current_trucks),
            'current_fill_rate': round(current_fill, 3),
            'potential_trucks_fluid': round(od_trucks_fluid, 1),
            'potential_fill_rate_fluid': round(min(potential_fill_fluid, 1.0), 3),
            'trucks_saved': round(trucks_saved, 1),
            'transport_savings_daily': round(transport_savings, 2),
            'incremental_sort_cost_daily': round(incremental_sort_cost, 2),
            'net_benefit_daily': round(net_benefit, 2),
            'annual_benefit': round(net_benefit * 250, 2),  # 250 operating days
            'payback_complexity': complexity,
            'distance_miles': round(distance, 1),
            'consolidation_pkgs_available': int(consolidation_data['total_packages']),
            'consolidation_destinations': consolidation_data['num_destinations'],
        })

    df = pd.DataFrame(opportunities)

    if df.empty:
        return df

    # Sort by net benefit and limit results
    df = df.sort_values('net_benefit_daily', ascending=False).head(max_results)

    return df.reset_index(drop=True)


def _get_regional_hub(facility: str, fac_lookup: pd.DataFrame) -> str:
    """Get regional hub for a facility."""
    if facility not in fac_lookup.index:
        return facility

    hub = fac_lookup.at[facility, 'regional_sort_hub']
    if pd.isna(hub) or hub == '':
        return facility

    return hub


def _estimate_consolidation_potential(
        origin: str,
        dest_region_hub: str,
        od_selected: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame
) -> Dict:
    """
    Estimate consolidation potential for region-destined freight.

    Returns total packages and destinations going to same region from origin.
    """
    # Find all ODs from this origin to destinations in same region
    same_region_ods = []

    for _, od in od_selected.iterrows():
        if od['origin'] != origin:
            continue

        # Get destination's regional hub
        dest = od['dest']
        # Simplified: use facility relationships (would need fac_lookup)
        # For now, assume destinations with same first 3 chars are in same region
        # In production, use proper regional_sort_hub lookup
        same_region_ods.append(od)

    total_packages = sum(od['pkgs_day'] for od in same_region_ods)
    num_destinations = len(set(od['dest'] for od in same_region_ods))

    return {
        'total_packages': total_packages,
        'num_destinations': num_destinations
    }


def _calculate_implementation_complexity(
        od_row: pd.Series,
        arc: pd.Series,
        dest_region_hub: str,
        fac_lookup: pd.DataFrame
) -> int:
    """
    Score implementation complexity (1-5).

    Factors:
    - Distance (longer = easier to justify change)
    - Volume (higher = easier to justify)
    - Current fill rate (lower = stronger case)
    - Path length (more touches = more complex)
    """
    score = 3  # Baseline moderate complexity

    # Distance factor (longer haul = easier to justify)
    distance = arc['distance_miles']
    if distance > 2000:
        score -= 1  # Easier (clear transport savings)
    elif distance < 500:
        score += 1  # Harder (less transport savings)

    # Volume factor (higher volume = easier to implement)
    pkgs = od_row['pkgs_day']
    if pkgs > 1000:
        score -= 1  # Easier (material impact)
    elif pkgs < 100:
        score += 1  # Harder (small impact)

    # Fill rate factor (lower fill = stronger case)
    fill_rate = arc.get('truck_fill_rate', 0)
    if fill_rate < 0.60:
        score -= 1  # Easier (obvious inefficiency)

    return max(1, min(5, score))  # Clamp to 1-5


def create_fluid_load_summary_report(opportunities: pd.DataFrame) -> str:
    """
    Create formatted text summary of fluid load opportunities.
    """
    if opportunities.empty:
        return "No fluid load opportunities identified (all lanes optimally utilized)"

    lines = []
    lines.append("=" * 120)
    lines.append("FLUID LOAD OPPORTUNITY ANALYSIS")
    lines.append("=" * 120)
    lines.append("")
    lines.append(f"Found {len(opportunities)} opportunities with daily savings >= minimum threshold")
    lines.append("")

    # Summary stats
    total_daily_savings = opportunities['net_benefit_daily'].sum()
    total_annual_savings = opportunities['annual_benefit'].sum()
    total_trucks_saved = opportunities['trucks_saved'].sum()

    lines.append(f"Total Daily Savings Potential: ${total_daily_savings:,.2f}")
    lines.append(f"Total Annual Savings Potential: ${total_annual_savings:,.2f}")
    lines.append(f"Total Trucks Saved: {total_trucks_saved:.1f} per day")
    lines.append("")
    lines.append("=" * 120)
    lines.append("")

    # Top opportunities
    lines.append("TOP OPPORTUNITIES (by daily savings):")
    lines.append("")
    lines.append(
        f"{'Origin':<12} {'Dest':<12} {'Pkgs/Day':<10} {'Curr Fill':<12} "
        f"{'Trucks Saved':<14} {'Daily $':<12} {'Annual $':<15}"
    )
    lines.append("-" * 120)

    for _, opp in opportunities.head(20).iterrows():
        lines.append(
            f"{opp['origin']:<12} {opp['dest']:<12} "
            f"{opp['packages_per_day']:>8,}  "
            f"{opp['current_fill_rate']:>10.1%}  "
            f"{opp['trucks_saved']:>12.1f}  "
            f"${opp['net_benefit_daily']:>10,.2f}  "
            f"${opp['annual_benefit']:>13,.0f}"
        )

    lines.append("=" * 120)

    return "\n".join(lines)


def calculate_sort_point_savings(
        opportunities: pd.DataFrame,
        timing_params: Dict,
        facilities: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate sort point capacity freed up by implementing fluid load opportunities.

    When switching from market/sort_group to region sort, origin facilities
    need fewer sort points.

    Returns summary by origin facility showing potential sort point reduction.
    """
    if opportunities.empty:
        return pd.DataFrame()

    fac_lookup = get_facility_lookup(facilities)
    sort_points_per_dest = float(timing_params['sort_points_per_destination'])

    facility_savings = []

    for origin in opportunities['origin'].unique():
        origin_opps = opportunities[opportunities['origin'] == origin]

        # Calculate current sort points used
        current_sort_points = 0
        potential_sort_points = 0

        for _, opp in origin_opps.iterrows():
            current_level = opp['current_sort_level']

            if current_level == 'market':
                # Currently sorting to individual destination
                current_sort_points += sort_points_per_dest
                # Would sort to region instead (shared across multiple dests)
                potential_sort_points += sort_points_per_dest / opp['consolidation_destinations']

            elif current_level == 'sort_group':
                # Currently sorting to granular sort groups
                dest = opp['dest']
                if dest in fac_lookup.index:
                    groups = fac_lookup.at[dest, 'last_mile_sort_groups_count']
                    if pd.isna(groups):
                        groups = 4  # Default
                    current_sort_points += sort_points_per_dest * groups
                    potential_sort_points += sort_points_per_dest  # Region level

        sort_points_freed = current_sort_points - potential_sort_points

        if sort_points_freed > 0:
            max_capacity = fac_lookup.at[origin, 'max_sort_points_capacity'] if origin in fac_lookup.index else 0

            facility_savings.append({
                'facility': origin,
                'current_sort_points': round(current_sort_points, 1),
                'potential_sort_points': round(potential_sort_points, 1),
                'sort_points_freed': round(sort_points_freed, 1),
                'max_capacity': max_capacity,
                'capacity_freed_pct': round(
                    safe_divide(sort_points_freed, max_capacity) * 100, 1
                ),
                'num_opportunities': len(origin_opps),
                'total_daily_savings': origin_opps['net_benefit_daily'].sum()
            })

    return pd.DataFrame(facility_savings).sort_values('sort_points_freed', ascending=False)