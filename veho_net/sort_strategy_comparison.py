"""
Sort Strategy Comparison Module

Runs two optimization scenarios:
1. Baseline: All destinations sorted to market level
2. Optimized: Model freely chooses optimal sort level per OD

"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from copy import deepcopy

from .milp_v4 import solve_network_optimization
from .utils import safe_divide


def run_sort_strategy_comparison(
        candidates: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        cost_params,
        timing_params: Dict,
        global_strategy,
        scenario_id: str = "comparison"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[Tuple]]:
    """
    Run baseline (market sort) vs. optimized (free choice) comparison.

    CRITICAL FIX: Now returns optimized results as 4th tuple element
    for reuse in main optimization flow.

    Args:
        candidates: Candidate paths
        facilities: Facility master
        mileage_bands: Mileage bands
        package_mix: Package mix
        container_params: Container params
        cost_params: Cost parameters
        timing_params: Timing parameters
        global_strategy: Load strategy
        scenario_id: Identifier for comparison

    Returns:
        Tuple of:
            - comparison_summary: High-level comparison metrics
            - detailed_comparison: OD-level differences
            - facility_comparison: Facility-level sort point comparison
            - optimized_results: (od_selected, arc_summary, kpis, sort_summary)
                                 from optimized run for reuse
    """
    print(f"\n{'=' * 70}")
    print("SORT STRATEGY COMPARISON")
    print("=" * 70)

    # ========== RUN 1: BASELINE (Market Sort) ==========
    print("\n1. Running BASELINE optimization (market sort only)...")

    baseline_results = solve_network_optimization(
        candidates=candidates,
        facilities=facilities,
        mileage_bands=mileage_bands,
        package_mix=package_mix,
        container_params=container_params,
        cost_params=cost_params,
        timing_params=timing_params,
        global_strategy=global_strategy,
        enable_sort_optimization=False  # Forces market sort
    )

    baseline_od, baseline_arc, baseline_kpis, _ = baseline_results

    if baseline_od.empty:
        print("  ❌ Baseline optimization failed")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

    baseline_cost = baseline_kpis.get('total_cost', 0)
    baseline_pkgs = baseline_kpis.get('total_packages', 0)
    baseline_cpp = safe_divide(baseline_cost, baseline_pkgs)

    print(f"  ✓ Baseline: ${baseline_cost:,.0f} (${baseline_cpp:.3f}/pkg)")

    # ========== RUN 2: OPTIMIZED (Free Choice) ==========
    print("\n2. Running OPTIMIZED optimization (sort level optimization)...")

    optimized_results = solve_network_optimization(
        candidates=candidates,
        facilities=facilities,
        mileage_bands=mileage_bands,
        package_mix=package_mix,
        container_params=container_params,
        cost_params=cost_params,
        timing_params=timing_params,
        global_strategy=global_strategy,
        enable_sort_optimization=True  # Allows optimization
    )

    optimized_od, optimized_arc, optimized_kpis, optimized_sort = optimized_results

    if optimized_od.empty:
        print("  ❌ Optimized optimization failed")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

    optimized_cost = optimized_kpis.get('total_cost', 0)
    optimized_pkgs = optimized_kpis.get('total_packages', 0)
    optimized_cpp = safe_divide(optimized_cost, optimized_pkgs)

    print(f"  ✓ Optimized: ${optimized_cost:,.0f} (${optimized_cpp:.3f}/pkg)")

    # ========== COMPARISON ANALYSIS ==========
    print("\n3. Analyzing differences...")

    cost_savings = baseline_cost - optimized_cost
    cost_savings_pct = safe_divide(cost_savings, baseline_cost) * 100
    cpp_delta = baseline_cpp - optimized_cpp

    print(f"\n  💰 Cost Savings: ${cost_savings:,.0f} ({cost_savings_pct:.1f}%)")
    print(f"  📦 CPP Delta: ${cpp_delta:.4f}")

    # High-level comparison summary
    comparison_summary = pd.DataFrame([
        {
            'metric': 'Total Daily Cost',
            'baseline_market': f"${baseline_cost:,.2f}",
            'optimized': f"${optimized_cost:,.2f}",
            'delta': f"${cost_savings:,.2f}",
            'delta_pct': f"{cost_savings_pct:.2f}%"
        },
        {
            'metric': 'Cost per Package',
            'baseline_market': f"${baseline_cpp:.4f}",
            'optimized': f"${optimized_cpp:.4f}",
            'delta': f"${cpp_delta:.4f}",
            'delta_pct': f"{safe_divide(cpp_delta, baseline_cpp) * 100:.2f}%"
        },
        {
            'metric': 'Avg Truck Fill Rate',
            'baseline_market': f"{baseline_kpis.get('avg_truck_fill_rate', 0):.1%}",
            'optimized': f"{optimized_kpis.get('avg_truck_fill_rate', 0):.1%}",
            'delta': f"{(optimized_kpis.get('avg_truck_fill_rate', 0) - baseline_kpis.get('avg_truck_fill_rate', 0)):.2%}",
            'delta_pct': ''
        }
    ])

    # OD-level detailed comparison (where sort level changed)
    detailed_comparison = _build_od_comparison(
        baseline_od, optimized_od, facilities, timing_params
    )

    # Facility-level sort point comparison
    facility_comparison = _build_facility_sort_comparison(
        baseline_od, optimized_od, facilities, timing_params
    )

    # Sort level distribution comparison
    sort_dist_baseline = _calculate_sort_distribution(baseline_od)
    sort_dist_optimized = _calculate_sort_distribution(optimized_od)

    print("\n  Sort Level Distribution:")
    print(f"    {'Level':<15} {'Baseline %':<15} {'Optimized %':<15} {'Delta':<15}")
    print(f"    {'-' * 60}")
    for level in ['region', 'market', 'sort_group']:
        base_pct = sort_dist_baseline.get(f'{level}_pct', 0)
        opt_pct = sort_dist_optimized.get(f'{level}_pct', 0)
        delta = opt_pct - base_pct
        print(f"    {level.title():<15} {base_pct:>12.1f}%  {opt_pct:>12.1f}%  {delta:>+12.1f}%")

    print(f"\n{'=' * 70}")
    print("COMPARISON COMPLETE")
    print("=" * 70)

    # CRITICAL FIX: Return optimized results for reuse
    return (
        comparison_summary,
        detailed_comparison,
        facility_comparison,
        optimized_results  # This avoids duplicate optimization!
    )


def _build_od_comparison(
        baseline_od: pd.DataFrame,
        optimized_od: pd.DataFrame,
        facilities: pd.DataFrame,
        timing_params: Dict
) -> pd.DataFrame:
    """
    Build OD-level comparison showing where sort levels changed.
    """
    comparison_rows = []

    # Merge baseline and optimized on OD pair
    merged = baseline_od.merge(
        optimized_od,
        on=['origin', 'dest'],
        suffixes=('_base', '_opt'),
        how='outer'
    )

    for _, row in merged.iterrows():
        # Skip if same in both
        base_sort = row.get('chosen_sort_level_base', 'market')
        opt_sort = row.get('chosen_sort_level_opt', 'market')

        if base_sort == opt_sort:
            continue  # No change

        base_cost = row.get('total_cost_base', 0)
        opt_cost = row.get('total_cost_opt', 0)
        cost_delta = opt_cost - base_cost

        pkgs = row.get('pkgs_day_base', row.get('pkgs_day_opt', 0))

        comparison_rows.append({
            'origin': row['origin'],
            'dest': row['dest'],
            'packages_per_day': pkgs,
            'baseline_sort_level': base_sort,
            'optimized_sort_level': opt_sort,
            'baseline_cost': base_cost,
            'optimized_cost': opt_cost,
            'cost_delta': cost_delta,
            'cost_delta_per_pkg': safe_divide(cost_delta, pkgs),
            'reason': _infer_change_reason(base_sort, opt_sort, cost_delta)
        })

    df = pd.DataFrame(comparison_rows)

    if df.empty:
        return df

    return df.sort_values('cost_delta')


def _build_facility_sort_comparison(
        baseline_od: pd.DataFrame,
        optimized_od: pd.DataFrame,
        facilities: pd.DataFrame,
        timing_params: Dict
) -> pd.DataFrame:
    """
    Build facility-level sort point comparison.
    """
    from .utils import get_facility_lookup

    fac_lookup = get_facility_lookup(facilities)
    sort_points_per_dest = float(timing_params['sort_points_per_destination'])

    # Get hub/hybrid facilities
    hub_facilities = facilities[
        facilities['type'].str.lower().isin(['hub', 'hybrid'])
    ]['facility_name'].unique()

    facility_rows = []

    for facility in hub_facilities:
        # Baseline sort points
        baseline_points = _calculate_facility_sort_points(
            facility, baseline_od, fac_lookup, sort_points_per_dest
        )

        # Optimized sort points
        optimized_points = _calculate_facility_sort_points(
            facility, optimized_od, fac_lookup, sort_points_per_dest
        )

        points_saved = baseline_points - optimized_points

        max_capacity = fac_lookup.at[facility, 'max_sort_points_capacity'] if facility in fac_lookup.index else 0

        baseline_util = safe_divide(baseline_points, max_capacity) * 100
        optimized_util = safe_divide(optimized_points, max_capacity) * 100

        facility_rows.append({
            'facility': facility,
            'max_capacity': max_capacity,
            'baseline_points_used': round(baseline_points, 1),
            'baseline_utilization_pct': round(baseline_util, 1),
            'optimized_points_used': round(optimized_points, 1),
            'optimized_utilization_pct': round(optimized_util, 1),
            'points_freed': round(points_saved, 1),
            'capacity_freed_pct': round(safe_divide(points_saved, max_capacity) * 100, 1)
        })

    df = pd.DataFrame(facility_rows)

    if df.empty:
        return df

    return df.sort_values('points_freed', ascending=False)


def _calculate_facility_sort_points(
        facility: str,
        od_df: pd.DataFrame,
        fac_lookup: pd.DataFrame,
        sort_points_per_dest: float
) -> float:
    """
    Calculate total sort points used at a facility.
    """
    origin_ods = od_df[od_df['origin'] == facility]

    if origin_ods.empty:
        return 0.0

    total_points = 0.0

    # Track unique regions and markets
    regions_served = set()
    markets_served = set()

    for _, od_row in origin_ods.iterrows():
        dest = od_row['dest']
        sort_level = od_row.get('chosen_sort_level', 'market')

        if sort_level == 'region':
            # Get destination's regional hub
            if dest in fac_lookup.index:
                hub = fac_lookup.at[dest, 'regional_sort_hub']
                if pd.isna(hub) or hub == '':
                    hub = dest
                regions_served.add(hub)

        elif sort_level == 'market':
            markets_served.add(dest)

        elif sort_level == 'sort_group':
            # Sort groups per destination
            if dest in fac_lookup.index:
                groups = fac_lookup.at[dest, 'last_mile_sort_groups_count']
                if pd.isna(groups) or groups <= 0:
                    groups = 4
                total_points += sort_points_per_dest * groups
            else:
                total_points += sort_points_per_dest * 4

    # Add points for regions and markets
    total_points += len(regions_served) * sort_points_per_dest
    total_points += len(markets_served) * sort_points_per_dest

    return total_points


def _calculate_sort_distribution(od_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate sort level distribution percentages."""
    if od_df.empty or 'chosen_sort_level' not in od_df.columns:
        return {
            'region_pct': 0.0,
            'market_pct': 100.0,  # Baseline assumption
            'sort_group_pct': 0.0
        }

    total_pkgs = od_df['pkgs_day'].sum()

    dist = {}
    for level in ['region', 'market', 'sort_group']:
        level_pkgs = od_df[od_df['chosen_sort_level'] == level]['pkgs_day'].sum()
        dist[f'{level}_pct'] = safe_divide(level_pkgs, total_pkgs) * 100

    return dist


def _infer_change_reason(
        base_level: str,
        opt_level: str,
        cost_delta: float
) -> str:
    """Infer business reason for sort level change."""
    if cost_delta < 0:
        # Optimized is cheaper
        if base_level == 'market' and opt_level == 'region':
            return "Region sort improves linehaul consolidation"
        elif base_level == 'market' and opt_level == 'sort_group':
            return "Pre-sort reduces downstream handling cost"
        elif base_level == 'region' and opt_level == 'market':
            return "Market sort reduces dest processing"
        else:
            return "Cost optimization"
    else:
        # Optimized is more expensive (shouldn't happen often)
        return "Capacity constraint relief"


def create_comparison_summary_report(
        comparison_summary: pd.DataFrame,
        facility_comparison: pd.DataFrame
) -> str:
    """Create formatted text summary."""
    if comparison_summary.empty:
        return "No comparison data available"

    lines = []
    lines.append("=" * 100)
    lines.append("SORT STRATEGY COMPARISON SUMMARY")
    lines.append("=" * 100)
    lines.append("")

    # High-level metrics
    for _, row in comparison_summary.iterrows():
        lines.append(f"{row['metric']:<30} Baseline: {row['baseline_market']:<15} "
                     f"Optimized: {row['optimized']:<15} Delta: {row['delta']:<15}")

    lines.append("")
    lines.append("=" * 100)
    lines.append("")

    # Facility-level sort point savings
    if not facility_comparison.empty:
        lines.append("FACILITY SORT POINT COMPARISON:")
        lines.append("")
        lines.append(f"{'Facility':<15} {'Baseline Util%':<18} {'Optimized Util%':<18} "
                     f"{'Points Freed':<15}")
        lines.append("-" * 100)

        for _, row in facility_comparison.head(10).iterrows():
            lines.append(
                f"{row['facility']:<15} "
                f"{row['baseline_utilization_pct']:>16.1f}%  "
                f"{row['optimized_utilization_pct']:>16.1f}%  "
                f"{row['points_freed']:>13.1f}"
            )

        total_freed = facility_comparison['points_freed'].sum()
        lines.append("-" * 100)
        lines.append(f"{'TOTAL':<15} {'':>16}  {'':>16}  {total_freed:>13.1f}")

    lines.append("")
    lines.append("=" * 100)

    return "\n".join(lines)