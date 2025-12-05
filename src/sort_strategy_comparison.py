"""
Sort Strategy Comparison Module

Compares constrained baseline (market sort only) against unrestricted optimization
(free sort level choice per OD). Quantifies cost and capacity impacts of sort decisions.
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
        scenario_id: str = "comparison",
        scenario_row: pd.Series = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[Tuple]]:
    """
    Run baseline (market sort) vs. optimized (free choice) comparison.
    """
    print(f"\n{'=' * 70}")
    print("SORT STRATEGY COMPARISON")
    print("=" * 70)

    print(f"\n{'-' * 70}")
    print("[1/2] BASELINE: Market Sort (CONSTRAINED)")
    print("-" * 70)

    baseline_results = solve_network_optimization(
        candidates=candidates,
        facilities=facilities,
        mileage_bands=mileage_bands,
        package_mix=package_mix,
        container_params=container_params,
        cost_params=cost_params,
        timing_params=timing_params,
        global_strategy=global_strategy,
        enable_sort_optimization=False,
        scenario_row=scenario_row
    )

    baseline_od, baseline_arc, baseline_kpis, _ = baseline_results

    if baseline_od.empty:
        print("  ❌ Baseline optimization failed")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

    baseline_cost = baseline_kpis.get('total_cost', 0)
    baseline_pkgs = baseline_kpis.get('total_packages', 0)
    baseline_cpp = safe_divide(baseline_cost, baseline_pkgs)

    print(f"\n  Baseline result:")
    print(f"    Total cost: ${baseline_cost:,.0f}")
    print(f"    Cost/pkg: ${baseline_cpp:.3f}")
    print(f"    Sort strategy: Market sort for all {len(baseline_od)} OD pairs")

    print(f"\n{'-' * 70}")
    print("[2/2] OPTIMIZED: Free Sort Choice (UNRESTRICTED)")
    print("-" * 70)

    optimized_results = solve_network_optimization(
        candidates=candidates,
        facilities=facilities,
        mileage_bands=mileage_bands,
        package_mix=package_mix,
        container_params=container_params,
        cost_params=cost_params,
        timing_params=timing_params,
        global_strategy=global_strategy,
        enable_sort_optimization=True,
        scenario_row=scenario_row
    )

    optimized_od, optimized_arc, optimized_kpis, optimized_sort = optimized_results

    if optimized_od.empty:
        print("  ❌ Optimized optimization failed")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

    optimized_cost = optimized_kpis.get('total_cost', 0)
    optimized_pkgs = optimized_kpis.get('total_packages', 0)
    optimized_cpp = safe_divide(optimized_cost, optimized_pkgs)

    print(f"\n  Optimized result:")
    print(f"    Total cost: ${optimized_cost:,.0f}")
    print(f"    Cost/pkg: ${optimized_cpp:.3f}")

    print(f"\n{'=' * 70}")
    print("COMPARISON RESULTS")
    print("═" * 70)

    cost_savings = baseline_cost - optimized_cost
    cost_savings_pct = safe_divide(cost_savings, baseline_cost) * 100
    cpp_delta = baseline_cpp - optimized_cpp
    cpp_delta_pct = safe_divide(cpp_delta, baseline_cpp) * 100

    print(f"\n  Baseline (market sort):      ${baseline_cost:>12,.0f}  @ ${baseline_cpp:.4f}/pkg")
    print(f"  Optimized (free choice):     ${optimized_cost:>12,.0f}  @ ${optimized_cpp:.4f}/pkg")
    print(f"  {'-' * 70}")
    print(f"  Savings from optimization:   ${cost_savings:>12,.0f}  (${cpp_delta:+.4f}/pkg)")
    print(f"  Improvement:                 {cost_savings_pct:>11.2f}%")

    if cost_savings < 0:
        print(f"\n  WARNING: Optimized cost exceeds baseline (possible model issue)")
    elif abs(cost_savings) < baseline_cost * 0.001:
        print(f"\n  Baseline already near-optimal (< 0.1% improvement)")
    else:
        print(f"\n  Optimization identified cost reduction opportunities")

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
            'delta_pct': f"{cpp_delta_pct:.2f}%"
        },
        {
            'metric': 'Avg Truck Fill Rate',
            'baseline_market': f"{baseline_kpis.get('avg_truck_fill_rate', 0):.1%}",
            'optimized': f"{optimized_kpis.get('avg_truck_fill_rate', 0):.1%}",
            'delta': f"{(optimized_kpis.get('avg_truck_fill_rate', 0) - baseline_kpis.get('avg_truck_fill_rate', 0)):.2%}",
            'delta_pct': ''
        }
    ])

    print(f"\n{'-' * 70}")
    print("Analyzing OD-level sort decisions...")
    detailed_comparison = _build_od_comparison(
        baseline_od, optimized_od, facilities, timing_params
    )

    if not detailed_comparison.empty:
        print(f"  Found {len(detailed_comparison)} OD pairs where sort level changed")
    else:
        print(f"  All OD pairs use same sort level in both scenarios")

    facility_comparison = _build_facility_sort_comparison(
        baseline_od, optimized_od, facilities, timing_params, scenario_row
    )

    sort_dist_baseline = _calculate_sort_distribution(baseline_od)
    sort_dist_optimized = _calculate_sort_distribution(optimized_od)

    print(f"\nSort Level Distribution:")
    print(f"  {'Level':<15} {'Baseline':<15} {'Optimized':<15} {'Change':<15}")
    print(f"  {'-' * 60}")
    for level in ['region', 'market', 'sort_group']:
        base_pct = sort_dist_baseline.get(f'{level}_pct', 0) * 100
        opt_pct = sort_dist_optimized.get(f'{level}_pct', 0) * 100
        delta = (opt_pct - base_pct)
        print(f"  {level.title():<15} {base_pct:>12.1f}%  {opt_pct:>12.1f}%  {delta:>+12.1f}%")

    print(f"\n{'=' * 70}")
    print("COMPARISON COMPLETE")
    print("=" * 70)

    return (
        comparison_summary,
        detailed_comparison,
        facility_comparison,
        optimized_results
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

    merged = baseline_od.merge(
        optimized_od,
        on=['origin', 'dest'],
        suffixes=('_base', '_opt'),
        how='outer'
    )

    for _, row in merged.iterrows():
        base_sort = row.get('chosen_sort_level_base', 'market')
        opt_sort = row.get('chosen_sort_level_opt', 'market')

        if base_sort == opt_sort:
            continue

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
        timing_params: Dict,
        scenario_row: pd.Series = None
) -> pd.DataFrame:
    """
    Build facility-level sort point utilization for optimized scenario.

    Shows actual sort points used vs capacity constraint.
    Baseline is unconstrained so not meaningful to compare utilization.
    """
    from .utils import get_facility_lookup

    fac_lookup = get_facility_lookup(facilities)
    sort_points_per_dest = float(timing_params['sort_points_per_destination'])

    # Check for scenario-level capacity override
    scenario_capacity_override = None
    if scenario_row is not None and 'max_sort_points_capacity' in scenario_row.index:
        override_val = scenario_row['max_sort_points_capacity']
        if pd.notna(override_val) and override_val > 0:
            scenario_capacity_override = float(override_val)

    hub_facilities = facilities[
        facilities['type'].str.lower().isin(['hub', 'hybrid'])
    ]['facility_name'].unique()

    facility_rows = []

    for facility in hub_facilities:
        if facility not in fac_lookup.index:
            continue

        # Calculate optimized points with breakdown
        points_breakdown = _calculate_facility_sort_points_breakdown(
            facility, optimized_od, fac_lookup, sort_points_per_dest
        )

        # Use scenario override if available, else facility-level
        if scenario_capacity_override is not None:
            max_capacity = scenario_capacity_override
        else:
            max_capacity = fac_lookup.at[facility, 'max_sort_points_capacity']
            if pd.isna(max_capacity) or max_capacity <= 0:
                continue  # Skip facilities without capacity constraint
            max_capacity = float(max_capacity)

        total_points = points_breakdown['total']
        optimized_util = safe_divide(total_points, max_capacity)

        facility_rows.append({
            'facility': facility,
            'max_sort_capacity': max_capacity,
            'sort_points_used': round(total_points, 1),
            'utilization_pct': round(optimized_util, 4),
            'headroom': round(max_capacity - total_points, 1),
            'region_pts': round(points_breakdown['region_pts'], 1),
            'market_pts': round(points_breakdown['market_pts'], 1),
            'sort_group_pts': round(points_breakdown['sort_group_pts'], 1),
            'num_region_hubs': points_breakdown['num_region_hubs'],
            'num_markets': points_breakdown['num_markets'],
            'num_sort_group_ods': points_breakdown['num_sort_group_ods'],
            'total_ods': points_breakdown['total_ods']
        })

    df = pd.DataFrame(facility_rows)

    if df.empty:
        return df

    return df.sort_values('utilization_pct', ascending=False)


def _calculate_facility_sort_points_breakdown(
        facility: str,
        od_df: pd.DataFrame,
        fac_lookup: pd.DataFrame,
        sort_points_per_dest: float
) -> Dict:
    """
    Calculate sort points with full breakdown.

    Returns dict with points and counts for each sort level.
    """
    origin_ods = od_df[od_df['origin'] == facility]

    result = {
        'region_pts': 0.0,
        'market_pts': 0.0,
        'sort_group_pts': 0.0,
        'total': 0.0,
        'num_region_hubs': 0,
        'num_markets': 0,
        'num_sort_group_ods': 0,
        'total_ods': len(origin_ods)
    }

    if origin_ods.empty:
        return result

    regions_served = set()
    markets_served = set()
    sort_group_ods = 0

    for _, od_row in origin_ods.iterrows():
        dest = od_row['dest']
        sort_level = od_row.get('chosen_sort_level', 'market')

        if sort_level == 'region':
            if dest in fac_lookup.index:
                hub = fac_lookup.at[dest, 'regional_sort_hub']
                if pd.isna(hub) or hub == '':
                    hub = dest
                regions_served.add(hub)

        elif sort_level == 'market':
            markets_served.add(dest)

        elif sort_level == 'sort_group':
            sort_group_ods += 1
            if dest not in fac_lookup.index:
                raise ValueError(
                    f"Destination facility '{dest}' not found in facilities master data"
                )

            groups = fac_lookup.at[dest, 'last_mile_sort_groups_count']
            if pd.isna(groups) or groups <= 0:
                raise ValueError(
                    f"Destination facility '{dest}' missing valid last_mile_sort_groups_count."
                )
            result['sort_group_pts'] += sort_points_per_dest * groups

    result['region_pts'] = len(regions_served) * sort_points_per_dest
    result['market_pts'] = len(markets_served) * sort_points_per_dest
    result['num_region_hubs'] = len(regions_served)
    result['num_markets'] = len(markets_served)
    result['num_sort_group_ods'] = sort_group_ods

    result['total'] = result['region_pts'] + result['market_pts'] + result['sort_group_pts']

    return result


def _calculate_facility_sort_points(
        facility: str,
        od_df: pd.DataFrame,
        fac_lookup: pd.DataFrame,
        sort_points_per_dest: float
) -> float:
    """
    Calculate total sort points used at a facility.

    Sort points by level:
    - region: 1 point per unique regional_sort_hub
    - market: 1 point per unique destination
    - sort_group: last_mile_sort_groups_count points per destination

    All multiplied by sort_points_per_dest from config.
    """
    breakdown = _calculate_facility_sort_points_breakdown(
        facility, od_df, fac_lookup, sort_points_per_dest
    )
    return breakdown['total']


def _calculate_sort_distribution(od_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate sort level distribution by package volume."""
    if od_df.empty or 'chosen_sort_level' not in od_df.columns:
        return {
            'region_pct': 0.0,
            'market_pct': 1.0,
            'sort_group_pct': 0.0
        }

    total_pkgs = od_df['pkgs_day'].sum()

    dist = {}
    for level in ['region', 'market', 'sort_group']:
        level_pkgs = od_df[od_df['chosen_sort_level'] == level]['pkgs_day'].sum()
        dist[f'{level}_pct'] = safe_divide(level_pkgs, total_pkgs)

    return dist


def _infer_change_reason(
        base_level: str,
        opt_level: str,
        cost_delta: float
) -> str:
    """Infer business reason for sort level change."""
    if cost_delta < 0:
        if base_level == 'market' and opt_level == 'region':
            return "Region sort improves linehaul consolidation"
        elif base_level == 'market' and opt_level == 'sort_group':
            return "Pre-sort reduces downstream handling cost"
        elif base_level == 'region' and opt_level == 'market':
            return "Market sort reduces dest processing"
        else:
            return "Cost optimization"
    else:
        return "Capacity constraint relief"


def create_comparison_summary_report(
        comparison_summary: pd.DataFrame,
        facility_comparison: pd.DataFrame
) -> str:
    """Create formatted text summary."""
    if comparison_summary.empty:
        return "No comparison data available"

    lines = []
    lines.append("=" * 140)
    lines.append("SORT STRATEGY COMPARISON SUMMARY")
    lines.append("=" * 140)
    lines.append("")

    for _, row in comparison_summary.iterrows():
        lines.append(f"{row['metric']:<30} Baseline: {row['baseline_market']:<15} "
                     f"Optimized: {row['optimized']:<15} Delta: {row['delta']:<15}")

    lines.append("")
    lines.append("=" * 140)
    lines.append("")

    if not facility_comparison.empty:
        lines.append("FACILITY SORT POINT UTILIZATION (Optimized Scenario):")
        lines.append("")
        lines.append(f"{'Facility':<12} {'Max Cap':<10} {'Pts Used':<10} {'Util%':<10} "
                     f"{'Headroom':<10} {'Rgn Pts':<10} {'Mkt Pts':<10} {'SG Pts':<10} {'ODs':<6}")
        lines.append("-" * 140)

        for _, row in facility_comparison.iterrows():
            lines.append(
                f"{row['facility']:<12} "
                f"{row['max_sort_capacity']:>8.0f}  "
                f"{row['sort_points_used']:>8.1f}  "
                f"{row['utilization_pct']*100:>8.1f}%  "
                f"{row['headroom']:>8.1f}  "
                f"{row['region_pts']:>8.1f}  "
                f"{row['market_pts']:>8.1f}  "
                f"{row['sort_group_pts']:>8.1f}  "
                f"{row['total_ods']:>4}"
            )

        lines.append("-" * 140)

        # Summary stats
        avg_util = facility_comparison['utilization_pct'].mean() * 100
        max_util = facility_comparison['utilization_pct'].max() * 100
        total_headroom = facility_comparison['headroom'].sum()
        total_region_pts = facility_comparison['region_pts'].sum()
        total_market_pts = facility_comparison['market_pts'].sum()
        total_sg_pts = facility_comparison['sort_group_pts'].sum()

        lines.append(f"")
        lines.append(f"  Average utilization: {avg_util:.1f}%")
        lines.append(f"  Max utilization: {max_util:.1f}%")
        lines.append(f"  Total headroom: {total_headroom:.0f} points")
        lines.append(f"")
        lines.append(f"  Network sort points breakdown:")
        lines.append(f"    Region:     {total_region_pts:>8.0f} pts")
        lines.append(f"    Market:     {total_market_pts:>8.0f} pts")
        lines.append(f"    Sort Group: {total_sg_pts:>8.0f} pts")

    lines.append("")
    lines.append("=" * 140)

    return "\n".join(lines)