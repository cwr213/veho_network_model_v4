"""
Fluid Load Opportunity Analysis Module

Identifies arcs where switching from container to fluid loading would reduce costs.
Analyzes actual planned arcs only, comparing transport savings against incremental sort costs.
"""

import pandas as pd
import numpy as np
from typing import Dict

from .containers_v4 import (
    weighted_pkg_cube,
    get_raw_trailer_cube
)
from .utils import safe_divide, get_facility_lookup
from .config_v4 import CostParameters


def analyze_fluid_load_opportunities(
        od_selected: pd.DataFrame,
        arc_summary: pd.DataFrame,
        facilities: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        mileage_bands: pd.DataFrame,
        cost_params: CostParameters
) -> pd.DataFrame:
    """
    Identify planned arcs where fluid loading would be more economical.

    SIMPLIFIED APPROACH:
    - Only analyze arcs in arc_summary (actual planned flows)
    - Direct comparison: container vs fluid for SAME arc
    - Calculate: trucks_fluid = ceil(arc_cube / (trailer_cube * pack_util_fluid))
    - Compare transport cost vs incremental sort cost
    - Returns ALL opportunities with positive net benefit (no thresholds)

    Args:
        od_selected: Selected OD paths (used for sort level info)
        arc_summary: Arc-level summary with current fill rates
        facilities: Facility master data
        package_mix: Package distribution
        container_params: Container/trailer parameters
        mileage_bands: Mileage bands (unused, kept for API compatibility)
        cost_params: Cost parameters

    Returns:
        DataFrame with ALL arcs that have positive net benefit from fluid loading.
        User can filter/sort in Excel as needed.
    """
    if arc_summary.empty:
        return pd.DataFrame()

    fac_lookup = get_facility_lookup(facilities)
    w_cube = weighted_pkg_cube(package_mix)
    raw_trailer_cube = get_raw_trailer_cube(container_params)

    # Get fluid pack utilization from input parameters
    pack_util_fluid = float(container_params["pack_utilization_fluid"].iloc[0])

    opportunities = []

    # Analyze each arc in the optimized solution
    for _, arc in arc_summary.iterrows():
        from_fac = arc['from_facility']
        to_fac = arc['to_facility']

        # Skip O=D arcs (no linehaul benefit)
        if from_fac == to_fac:
            continue

        # Only consider arcs currently using container strategy
        current_strategy = arc.get('effective_strategy', 'container')
        if current_strategy.lower() != 'container':
            continue

        # Only consider arcs with low fill rates (opportunity for improvement)
        current_fill = arc.get('truck_fill_rate', 0)
        if current_fill >= 0.75:  # Already well-utilized
            continue

        # Current state
        pkgs_per_day = arc['pkgs_day']
        current_trucks = arc['trucks']
        total_cube = arc['pkg_cube_cuft']
        cost_per_truck = arc['cost_per_truck']

        # Calculate trucks needed with fluid strategy
        # Uses input parameter pack_util_fluid (no hardcoding)
        potential_trucks_fluid = max(1, int(np.ceil(
            total_cube / (raw_trailer_cube * pack_util_fluid)
        )))

        # Calculate fluid fill rate (using RAW trailer cube for reporting)
        potential_fill_fluid = safe_divide(
            total_cube,
            potential_trucks_fluid * raw_trailer_cube
        )

        # Calculate transport savings
        trucks_saved = current_trucks - potential_trucks_fluid

        if trucks_saved <= 0:
            continue  # No transport savings

        transport_savings = trucks_saved * cost_per_truck

        # Calculate incremental sort cost
        # Need to determine what sort level(s) flow on this arc
        incremental_sort_cost = _calculate_incremental_sort_cost(
            from_fac, to_fac, pkgs_per_day, od_selected,
            facilities, cost_params
        )

        # Calculate net benefit
        net_benefit = transport_savings - incremental_sort_cost

        # Return ALL opportunities with positive benefit (no threshold)
        if net_benefit <= 0:
            continue

        opportunities.append({
            'from_facility': from_fac,
            'to_facility': to_fac,
            'packages_per_day': int(pkgs_per_day),
            'current_strategy': 'container',
            'current_trucks': int(current_trucks),
            'current_fill_rate': round(current_fill, 3),
            'potential_trucks_fluid': int(potential_trucks_fluid),
            'potential_fill_rate_fluid': round(min(potential_fill_fluid, 1.0), 3),
            'trucks_saved': int(trucks_saved),
            'transport_savings_daily': round(transport_savings, 2),
            'incremental_sort_cost_daily': round(incremental_sort_cost, 2),
            'net_benefit_daily': round(net_benefit, 2),
            'annual_benefit': round(net_benefit * 250, 2),
            'distance_miles': round(arc['distance_miles'], 1),
            'current_cost_per_pkg': round(arc.get('CPP', 0), 4),
        })

    df = pd.DataFrame(opportunities)

    if df.empty:
        return df

    # Sort by net benefit (highest first) - user can re-sort in Excel
    df = df.sort_values('net_benefit_daily', ascending=False)

    return df.reset_index(drop=True)


def _calculate_incremental_sort_cost(
        from_fac: str,
        to_fac: str,
        arc_packages: float,
        od_selected: pd.DataFrame,
        facilities: pd.DataFrame,
        cost_params: CostParameters
) -> float:
    """
    Calculate incremental sort cost when switching arc to fluid strategy.

    Logic:
    - Container strategy typically uses pre-sorted freight (market or sort_group)
    - Fluid strategy requires destination to do more sorting (can't pre-sort mixed freight)
    - Cost difference = additional destination sort operations needed

    Approach:
    1. Find all OD flows using this arc
    2. Determine current sort levels
    3. Calculate incremental sort cost based on sort level degradation

    Args:
        from_fac: Arc origin
        to_fac: Arc destination
        arc_packages: Total packages on arc
        od_selected: OD paths with sort level info
        facilities: Facility master data
        cost_params: Cost parameters

    Returns:
        Daily incremental sort cost (can be 0 if no degradation needed)
    """
    if od_selected.empty:
        return 0.0

    # Find OD flows that use this arc
    arc_ods = []
    for _, od_row in od_selected.iterrows():
        path_nodes = od_row.get('path_nodes', [])
        if not isinstance(path_nodes, (list, tuple)):
            continue

        # Check if this arc is in the path
        for i in range(len(path_nodes) - 1):
            if path_nodes[i] == from_fac and path_nodes[i + 1] == to_fac:
                arc_ods.append(od_row)
                break

    if not arc_ods:
        # No OD info available, make conservative estimate
        # Assume market sort → needs intermediate sort at destination
        return arc_packages * cost_params.intermediate_sort_cost_per_pkg * 0.5

    # Calculate weighted average incremental cost based on current sort levels
    total_incremental = 0.0

    for od_row in arc_ods:
        pkgs = od_row['pkgs_day']
        current_sort = od_row.get('chosen_sort_level', 'market')

        # Determine sort cost impact
        if current_sort == 'sort_group':
            # Currently pre-sorted to granular level
            # Fluid would require full destination sort
            incremental = cost_params.intermediate_sort_cost_per_pkg
        elif current_sort == 'market':
            # Currently sorted to market level
            # Fluid would require some destination sorting
            # Use 50% of full sort cost as estimate
            incremental = cost_params.intermediate_sort_cost_per_pkg * 0.5
        else:
            # Region sort or other - minimal incremental cost
            incremental = 0.0

        total_incremental += pkgs * incremental

    return total_incremental


def create_fluid_load_summary_report(opportunities: pd.DataFrame) -> str:
    """
    Create formatted text summary of fluid load opportunities.
    """
    if opportunities.empty:
        return "No fluid load opportunities identified (all arcs optimally utilized)"

    lines = []
    lines.append("")
    lines.append(f"Found {len(opportunities)} arc opportunities with positive net benefit")
    lines.append("")

    # Summary stats
    total_daily_savings = opportunities['net_benefit_daily'].sum()
    total_annual_savings = opportunities['annual_benefit'].sum()
    total_trucks_saved = opportunities['trucks_saved'].sum()

    lines.append(f"Total Daily Savings Potential: ${total_daily_savings:,.2f}")
    lines.append(f"Total Annual Savings Potential: ${total_annual_savings:,.2f}")
    lines.append(f"Total Trucks Saved: {total_trucks_saved:.0f} per day")
    lines.append("")
    lines.append("=" * 120)
    lines.append("")

    # Top opportunities
    lines.append("TOP OPPORTUNITIES (by daily savings):")
    lines.append("")
    lines.append(
        f"{'From':<12} {'To':<12} {'Pkgs/Day':<10} {'Curr Fill':<12} "
        f"{'Trucks Saved':<14} {'Daily $':<12} {'Annual $':<15}"
    )
    lines.append("-" * 120)

    for _, opp in opportunities.head(20).iterrows():
        lines.append(
            f"{opp['from_facility']:<12} {opp['to_facility']:<12} "
            f"{opp['packages_per_day']:>8,}  "
            f"{opp['current_fill_rate']:>10.1%}  "
            f"{opp['trucks_saved']:>12}  "
            f"${opp['net_benefit_daily']:>10,.2f}  "
            f"${opp['annual_benefit']:>13,.0f}"
        )

    lines.append("=" * 120)
    lines.append("")
    lines.append(f"Note: Full list of {len(opportunities)} opportunities saved to Excel for filtering/sorting")

    return "\n".join(lines)


def calculate_sort_point_savings(
        opportunities: pd.DataFrame,
        timing_params: Dict,
        facilities: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate sort point capacity freed up by fluid loading opportunities.

    Note: With simplified arc-based analysis, sort point savings are less
    directly calculable since we're not tracking origin facility sort decisions.
    This function is kept for API compatibility but returns empty DataFrame.

    For detailed sort point analysis, use sort_strategy_comparison module.

    Args:
        opportunities: Fluid load opportunities (arc-based)
        timing_params: Timing parameters
        facilities: Facility master data

    Returns:
        Empty DataFrame (feature not applicable to simplified arc analysis)
    """
    # Not applicable to simplified arc-based analysis
    # Sort point savings require OD-level sort decision tracking
    return pd.DataFrame()