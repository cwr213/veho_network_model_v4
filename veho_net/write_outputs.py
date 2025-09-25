import pandas as pd
import numpy as np
from pathlib import Path


def safe_sheet_name(name: str) -> str:
    """Create Excel-safe sheet names that are <= 31 characters."""
    invalid_chars = ['[', ']', ':', '*', '?', '/', '\\']
    clean_name = name
    for char in invalid_chars:
        clean_name = clean_name.replace(char, '_')

    if len(clean_name) <= 31:
        return clean_name

    # Intelligent truncation
    if '_comparison' in clean_name:
        base = clean_name.replace('_comparison', '')
        return f"{base[:20]}_comparison" if len(base) <= 20 else f"{base[:20]}_comparison"
    elif '_analysis' in clean_name:
        base = clean_name.replace('_analysis', '')
        return f"{base[:22]}_analysis" if len(base) <= 22 else f"{base[:22]}_analysis"
    elif '_summary' in clean_name:
        base = clean_name.replace('_summary', '')
        return f"{base[:23]}_summary" if len(base) <= 23 else f"{base[:23]}_summary"

    return clean_name[:31]


def write_workbook(path, scen_sum, od_out, path_detail, dwell_hotspots, facility_rollup, arc_summary, kpis,
                   sort_allocation_summary=None):
    """Write comprehensive workbook with all sheets."""
    try:
        with pd.ExcelWriter(path, engine="xlsxwriter") as xw:

            # Core sheets
            try:
                if not scen_sum.empty:
                    scen_sum.to_excel(xw, sheet_name=safe_sheet_name("scenario_summary"), index=False)
                else:
                    pd.DataFrame([{"note": "No scenario data"}]).to_excel(xw, sheet_name=safe_sheet_name(
                        "scenario_summary"), index=False)

                if not od_out.empty:
                    od_out.to_excel(xw, sheet_name=safe_sheet_name("od_selected_paths"), index=False)
                else:
                    pd.DataFrame([{"note": "No OD path data"}]).to_excel(xw, sheet_name=safe_sheet_name(
                        "od_selected_paths"), index=False)

                if not path_detail.empty:
                    path_detail.to_excel(xw, sheet_name=safe_sheet_name("path_steps_selected"), index=False)
                else:
                    pd.DataFrame([{"note": "No path detail data"}]).to_excel(xw, sheet_name=safe_sheet_name(
                        "path_steps_selected"), index=False)

                if not dwell_hotspots.empty:
                    dwell_hotspots.to_excel(xw, sheet_name=safe_sheet_name("dwell_hotspots"), index=False)
                else:
                    pd.DataFrame([{"note": "No dwell hotspot data"}]).to_excel(xw, sheet_name=safe_sheet_name(
                        "dwell_hotspots"), index=False)

                if not facility_rollup.empty:
                    facility_rollup.to_excel(xw, sheet_name=safe_sheet_name("facility_rollup"), index=False)
                else:
                    pd.DataFrame([{"note": "No facility data"}]).to_excel(xw,
                                                                          sheet_name=safe_sheet_name("facility_rollup"),
                                                                          index=False)

                if not arc_summary.empty:
                    arc_summary.to_excel(xw, sheet_name=safe_sheet_name("arc_summary"), index=False)
                else:
                    pd.DataFrame([{"note": "No arc/lane data"}]).to_excel(xw, sheet_name=safe_sheet_name("arc_summary"),
                                                                          index=False)

                if kpis is not None and not kpis.empty:
                    kpis.to_frame("value").to_excel(xw, sheet_name=safe_sheet_name("kpis"))
                else:
                    pd.Series([0], index=["total_cost"]).to_frame("value").to_excel(xw,
                                                                                    sheet_name=safe_sheet_name("kpis"))

            except Exception as e:
                raise e

            # Enhanced sort optimization sheets
            try:
                if sort_allocation_summary is not None and not sort_allocation_summary.empty:
                    sort_allocation_summary.to_excel(xw, sheet_name=safe_sheet_name("sort_allocation_summary"),
                                                     index=False)
                    create_containerization_summary_sheet(xw, od_out, sort_allocation_summary)
                    create_origin_strategy_analysis_sheet(xw, od_out, sort_allocation_summary)
                    create_route_efficiency_analysis_sheet(xw, sort_allocation_summary)
                    create_actionable_insights_sheet(xw, sort_allocation_summary, od_out)
                else:
                    pd.DataFrame([{"note": "No sort optimization data"}]).to_excel(xw, sheet_name=safe_sheet_name(
                        "sort_allocation_summary"), index=False)

            except Exception:
                pass

            # Fill rate & utilization analysis
            try:
                if not od_out.empty:
                    create_fill_rate_analysis_sheet(xw, od_out)
                    create_lane_utilization_analysis_sheet(xw, arc_summary)
            except Exception:
                pass

            # Advanced analysis sheets
            try:
                if not od_out.empty and 'containerization_level' in od_out.columns:
                    create_fill_spill_analysis_sheet(xw, od_out)
                    create_zone_strategy_analysis_sheet(xw, od_out)
                    create_network_strategy_summary_sheet(xw, od_out)
            except Exception:
                pass

            # Sort capacity analysis
            try:
                if not facility_rollup.empty:
                    create_sort_capacity_analysis_sheet(xw, facility_rollup)
            except Exception:
                pass

        return True

    except Exception as e:
        return False


def create_containerization_summary_sheet(writer, od_out, sort_allocation_summary):
    """Create containerization summary by level."""
    try:
        if 'containerization_level' in od_out.columns and not od_out.empty:
            level_summary = od_out.groupby('containerization_level').agg({
                'pkgs_day': 'sum',
                'total_cost': 'sum',
                'cost_per_pkg': 'mean',
                'origin': 'nunique'
            }).reset_index()

            level_summary.columns = ['containerization_level', 'total_packages', 'total_cost',
                                     'avg_cost_per_pkg', 'unique_origins']

            if not sort_allocation_summary.empty and 'efficiency_score' in sort_allocation_summary.columns:
                efficiency_by_level = sort_allocation_summary.groupby('optimal_containerization_level')[
                    'efficiency_score'].mean()
                level_summary = level_summary.set_index('containerization_level').join(
                    efficiency_by_level).reset_index()
                level_summary = level_summary.rename(columns={'efficiency_score': 'avg_efficiency_score'})

            if 'daily_cost_savings' in sort_allocation_summary.columns:
                savings_by_level = sort_allocation_summary.groupby('optimal_containerization_level')[
                    'daily_cost_savings'].sum()
                level_summary = level_summary.set_index('containerization_level').join(savings_by_level).reset_index()
                level_summary = level_summary.rename(columns={'daily_cost_savings': 'total_daily_savings'})

            level_summary.to_excel(writer, sheet_name=safe_sheet_name("containerization_summary"), index=False)

    except Exception:
        pass


def create_origin_strategy_analysis_sheet(writer, od_out, sort_allocation_summary):
    """Create origin-level strategy analysis."""
    try:
        if not od_out.empty:
            origin_strategy = od_out.groupby('origin').agg({
                'pkgs_day': 'sum',
                'total_cost': 'sum',
                'dest': 'count'
            }).reset_index()

            origin_strategy.columns = ['origin_facility', 'total_packages_day', 'total_daily_cost',
                                       'destinations_served']
            origin_strategy['cost_per_pkg'] = origin_strategy['total_daily_cost'] / origin_strategy[
                'total_packages_day']

            if 'containerization_level' in od_out.columns:
                strategy_mix = od_out.groupby('origin')['containerization_level'].apply(
                    lambda x: x.value_counts().to_dict()
                ).reset_index()
                strategy_mix.columns = ['origin_facility', 'containerization_strategy_mix']

                origin_strategy = origin_strategy.merge(strategy_mix, on='origin_facility', how='left')

            if not sort_allocation_summary.empty and 'origin' in sort_allocation_summary.columns:
                sort_metrics = sort_allocation_summary.groupby('origin').agg({
                    'daily_cost_savings': 'sum',
                    'efficiency_score': 'mean',
                    'sort_points_used': 'sum'
                }).reset_index()
                sort_metrics.columns = ['origin_facility', 'total_sort_savings', 'avg_efficiency_score',
                                        'total_sort_points']

                origin_strategy = origin_strategy.merge(sort_metrics, on='origin_facility', how='left')

            origin_strategy = origin_strategy.sort_values('total_daily_cost', ascending=False)
            origin_strategy.to_excel(writer, sheet_name=safe_sheet_name("origin_strategy_analysis"), index=False)

    except Exception:
        pass


def create_route_efficiency_analysis_sheet(writer, sort_allocation_summary):
    """Create route efficiency analysis."""
    try:
        if not sort_allocation_summary.empty:
            efficiency_analysis = sort_allocation_summary.copy()

            if 'daily_cost_savings' in efficiency_analysis.columns and 'pkgs_day' in efficiency_analysis.columns:
                efficiency_analysis['savings_per_package'] = (
                        efficiency_analysis['daily_cost_savings'] / efficiency_analysis['pkgs_day'].replace(0, 1)
                ).round(3)

            if 'daily_cost_savings' in efficiency_analysis.columns and 'sort_points_used' in efficiency_analysis.columns:
                efficiency_analysis['efficiency_score'] = (
                        efficiency_analysis['daily_cost_savings'] / efficiency_analysis['sort_points_used'].replace(0,
                                                                                                                    1)
                ).round(2)

                efficiency_analysis['implementation_priority'] = pd.cut(
                    efficiency_analysis['efficiency_score'],
                    bins=[-float('inf'), 25, 75, 150, float('inf')],
                    labels=['Low Priority', 'Medium Priority', 'High Priority', 'Critical Priority']
                ).astype(str)

            efficiency_analysis = efficiency_analysis.sort_values(
                'efficiency_score' if 'efficiency_score' in efficiency_analysis.columns else 'daily_cost_savings',
                ascending=False
            )

            efficiency_analysis.to_excel(writer, sheet_name=safe_sheet_name("route_efficiency_analysis"), index=False)

    except Exception:
        pass


def create_actionable_insights_sheet(writer, sort_allocation_summary, od_out):
    """Generate actionable business insights."""
    try:
        insights = []

        if not sort_allocation_summary.empty and 'efficiency_score' in sort_allocation_summary.columns:
            high_priority_routes = sort_allocation_summary[sort_allocation_summary['efficiency_score'] > 100]
            if not high_priority_routes.empty:
                total_savings = high_priority_routes['daily_cost_savings'].sum()
                insights.append({
                    'category': 'High-Impact Routes',
                    'insight': f'{len(high_priority_routes)} routes with >$100/day savings per sort point',
                    'action': 'Prioritize for immediate deeper containerization',
                    'potential_savings': f"${total_savings:,.0f}/day",
                    'implementation_effort': 'Medium',
                    'routes': ', '.join(high_priority_routes['od_pair_id'].head(5).tolist()
                                        if 'od_pair_id' in high_priority_routes.columns else [
                        'See route_efficiency_analysis'])
                })

        if not od_out.empty and 'truck_fill_rate' in od_out.columns:
            low_fill_routes = od_out[od_out['truck_fill_rate'] < 0.60]
            if not low_fill_routes.empty:
                avg_fill_improvement = (0.75 - low_fill_routes['truck_fill_rate'].mean()) * 100
                insights.append({
                    'category': 'Fill Rate Opportunities',
                    'insight': f'{len(low_fill_routes)} routes with truck fill rates below 60%',
                    'action': 'Review consolidation patterns or adjust service levels',
                    'potential_impact': f'{avg_fill_improvement:.1f}% average fill rate improvement possible',
                    'implementation_effort': 'Low',
                    'affected_volume': f"{low_fill_routes['pkgs_day'].sum():,.0f} pkgs/day"
                })

        if not od_out.empty and 'containerization_level' in od_out.columns:
            region_level_routes = od_out[od_out['containerization_level'] == 'region']
            if len(region_level_routes) > 0:
                insights.append({
                    'category': 'Sort Optimization',
                    'insight': f'{len(region_level_routes)} routes using region-level containerization',
                    'action': 'Evaluate opportunities for market or sort-group level optimization',
                    'potential_impact': 'Reduced processing costs through granular containerization',
                    'implementation_effort': 'High',
                    'affected_volume': f"{region_level_routes['pkgs_day'].sum():,.0f} pkgs/day"
                })

        insights_df = pd.DataFrame(insights)
        if not insights_df.empty:
            priority_scores = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
            if 'implementation_effort' in insights_df.columns:
                effort_scores = {'Low': 1, 'Medium': 2, 'High': 3}
                insights_df['effort_score'] = insights_df['implementation_effort'].map(effort_scores).fillna(2)

            insights_df.to_excel(writer, sheet_name=safe_sheet_name("actionable_insights"), index=False)

    except Exception:
        pass


def create_fill_rate_analysis_sheet(writer, od_out):
    """Create comprehensive fill rate and utilization analysis."""
    try:
        if 'truck_fill_rate' in od_out.columns and not od_out.empty:
            fill_analysis = od_out.groupby(['origin', 'path_type']).agg({
                'truck_fill_rate': 'mean',
                'pkgs_day': 'sum',
                'total_cost': 'sum'
            }).reset_index()

            if 'container_fill_rate' in od_out.columns:
                container_fill = od_out.groupby(['origin', 'path_type'])['container_fill_rate'].mean()
                fill_analysis['container_fill_rate'] = fill_analysis.set_index(['origin', 'path_type']).index.map(
                    container_fill).fillna(0)

            fill_analysis['truck_efficiency_category'] = pd.cut(
                fill_analysis['truck_fill_rate'],
                bins=[0, 0.6, 0.8, 1.0],
                labels=['Low', 'Medium', 'High']
            ).astype(str)

            if 'container_fill_rate' in fill_analysis.columns:
                fill_analysis['container_efficiency_category'] = pd.cut(
                    fill_analysis['container_fill_rate'],
                    bins=[0, 0.6, 0.8, 1.0],
                    labels=['Low', 'Medium', 'High']
                ).astype(str)

            fill_analysis = fill_analysis.round(3)
            fill_analysis.to_excel(writer, sheet_name=safe_sheet_name("fill_rate_analysis"), index=False)

    except Exception:
        pass


def create_lane_utilization_analysis_sheet(writer, arc_summary):
    """Create lane-level utilization analysis."""
    try:
        if arc_summary is not None and not arc_summary.empty:
            required_cols = ['from_facility', 'to_facility', 'pkgs_day', 'trucks']
            if all(col in arc_summary.columns for col in required_cols):

                lane_utilization = arc_summary[required_cols].copy()

                optional_cols = ['truck_fill_rate', 'container_fill_rate', 'packages_per_truck', 'total_cost', 'CPP']
                for col in optional_cols:
                    if col in arc_summary.columns:
                        lane_utilization[col] = arc_summary[col]

                if 'truck_fill_rate' in lane_utilization.columns:
                    lane_utilization['utilization_category'] = pd.cut(
                        lane_utilization['truck_fill_rate'],
                        bins=[0, 0.6, 0.8, 1.0],
                        labels=['Low (<60%)', 'Medium (60-80%)', 'High (80%+)']
                    ).astype(str)

                    lane_utilization['fill_improvement_potential'] = np.where(
                        lane_utilization['truck_fill_rate'] < 0.85,
                        (0.85 - lane_utilization['truck_fill_rate']) * lane_utilization['pkgs_day'],
                        0
                    ).round(0)

                lane_utilization['volume_category'] = pd.cut(
                    lane_utilization['pkgs_day'],
                    bins=[0, 500, 2000, 10000, float('inf')],
                    labels=['Low Volume', 'Medium Volume', 'High Volume', 'Very High Volume']
                ).astype(str)

                lane_utilization = lane_utilization.sort_values('pkgs_day', ascending=False)
                lane_utilization.to_excel(writer, sheet_name=safe_sheet_name("lane_utilization_analysis"), index=False)

    except Exception:
        pass


def create_fill_spill_analysis_sheet(writer, od_out):
    """Create fill and spill analysis."""
    try:
        if 'spill_opportunity_flag' in od_out.columns:
            spill_analysis = od_out[od_out['spill_opportunity_flag'] == True].copy()

            if not spill_analysis.empty:
                spill_summary = spill_analysis.groupby(['origin', 'spill_parent_hub']).agg({
                    'pkgs_day': 'sum',
                    'dest': 'count',
                    'total_cost': 'sum'
                }).reset_index()

                spill_summary.columns = ['origin_facility', 'spill_parent_hub', 'spillable_packages_day',
                                         'affected_destinations', 'total_affected_cost']

                spill_summary['operational_flexibility'] = pd.cut(
                    spill_summary['spillable_packages_day'],
                    bins=[0, 500, 2000, 10000, float('inf')],
                    labels=['Low', 'Medium', 'High', 'Critical']
                ).astype(str)

                spill_summary.to_excel(writer, sheet_name=safe_sheet_name("fill_spill_analysis"), index=False)
        else:
            pd.DataFrame([{"note": "No spill opportunity data available"}]).to_excel(
                writer, sheet_name=safe_sheet_name("fill_spill_analysis"), index=False)

    except Exception:
        pass


def create_zone_strategy_analysis_sheet(writer, od_out):
    """Create zone-level strategy analysis."""
    try:
        if 'zone' in od_out.columns and 'containerization_level' in od_out.columns:
            zone_strategy = od_out.groupby(['zone', 'containerization_level']).agg({
                'pkgs_day': 'sum',
                'total_cost': 'sum',
                'cost_per_pkg': 'mean',
                'origin': 'nunique'
            }).reset_index()

            zone_strategy['volume_share'] = zone_strategy.groupby('zone')['pkgs_day'].transform(lambda x: x / x.sum())

            zone_strategy_pivot = zone_strategy.pivot_table(
                index='zone',
                columns='containerization_level',
                values=['pkgs_day', 'volume_share'],
                fill_value=0
            )

            zone_strategy_pivot.to_excel(writer, sheet_name=safe_sheet_name("zone_strategy_analysis"))
        else:
            pd.DataFrame([{"note": "No zone or containerization data available"}]).to_excel(
                writer, sheet_name=safe_sheet_name("zone_strategy_analysis"), index=False)

    except Exception:
        pass


def create_network_strategy_summary_sheet(writer, od_out):
    """Create network-level strategy summary."""
    try:
        if 'containerization_level' in od_out.columns:
            network_summary = []

            for level in ['region', 'market', 'sort_group']:
                level_data = od_out[od_out['containerization_level'] == level]
                volume = level_data['pkgs_day'].sum() if not level_data.empty else 0
                count = len(level_data)

                network_summary.append({
                    'containerization_level': level,
                    'daily_volume': volume,
                    'route_count': count
                })

            network_summary_df = pd.DataFrame(network_summary)

            total_volume = network_summary_df['daily_volume'].sum()
            if total_volume > 0:
                network_summary_df['volume_percentage'] = (
                        network_summary_df['daily_volume'] / total_volume * 100).round(1)
            else:
                network_summary_df['volume_percentage'] = 0.0

            network_summary_df.to_excel(writer, sheet_name=safe_sheet_name("network_strategy_summary"), index=False)
        else:
            pd.DataFrame([{"note": "No containerization level data available"}]).to_excel(
                writer, sheet_name=safe_sheet_name("network_strategy_summary"), index=False)

    except Exception:
        pass


def create_sort_capacity_analysis_sheet(writer, facility_rollup):
    """Create sort capacity analysis."""
    try:
        if not facility_rollup.empty:
            sort_facilities = facility_rollup[
                (facility_rollup.get('type', '').isin(['hub', 'hybrid'])) |
                (facility_rollup.get('max_sort_points_capacity', 0) > 0)
                ].copy()

            if not sort_facilities.empty:
                capacity_cols = ['facility', 'type']
                optional_cols = ['max_sort_points_capacity', 'sort_points_allocated', 'sort_utilization_rate',
                                 'available_sort_capacity', 'injection_pkgs_day', 'peak_hourly_throughput']

                available_cols = capacity_cols + [col for col in optional_cols if col in sort_facilities.columns]
                sort_utilization = sort_facilities[available_cols].copy()

                if 'max_sort_points_capacity' in sort_utilization.columns and 'sort_points_allocated' in sort_utilization.columns:
                    sort_utilization['capacity_utilization_pct'] = (
                            sort_utilization['sort_points_allocated'] / sort_utilization[
                        'max_sort_points_capacity'].replace(0, 1) * 100
                    ).round(1)

                sort_utilization.to_excel(writer, sheet_name=safe_sheet_name("sort_capacity_analysis"), index=False)
            else:
                pd.DataFrame([{"note": "No sort capacity data available"}]).to_excel(
                    writer, sheet_name=safe_sheet_name("sort_capacity_analysis"), index=False)
        else:
            pd.DataFrame([{"note": "No facility data available"}]).to_excel(
                writer, sheet_name=safe_sheet_name("sort_capacity_analysis"), index=False)

    except Exception:
        pass


def write_compare_workbook(path, compare_df: pd.DataFrame, run_kv: dict):
    """Enhanced comparison workbook with clean output."""
    try:
        # Enhanced column ordering
        front = ["base_id", "strategy", "scenario_id", "output_file"]
        kpi_cols = [
            "total_cost", "cost_per_pkg", "num_ods",
            "sla_violations", "around_world_flags",
            "pct_direct", "pct_1_touch", "pct_2_touch", "pct_3_touch",
            "sort_optimization_savings", "avg_container_fill_rate",
            "avg_truck_fill_rate", "total_packages_dwelled"
        ]
        cols = front + [c for c in kpi_cols if c in compare_df.columns]
        df = compare_df[cols].copy()

        # Enhanced wide views
        metrics_to_pivot = ['cost_per_pkg', 'sort_optimization_savings', 'avg_truck_fill_rate',
                            'avg_container_fill_rate']
        wide_views = {}

        for metric in metrics_to_pivot:
            if metric in df.columns:
                wide = df.pivot(index="base_id", columns="strategy", values=metric)
                wide = wide.rename_axis(None, axis=1).reset_index()

                if 'container' in wide.columns and 'fluid' in wide.columns:
                    wide[f'{metric}_advantage'] = wide['container'] - wide['fluid']
                    wide[f'{metric}_advantage_pct'] = (
                            wide[f'{metric}_advantage'] / wide['fluid'].replace(0, 1) * 100).round(1)

                safe_name = safe_sheet_name(f"{metric}_comparison")
                wide_views[safe_name] = wide

        # Enhanced strategy efficiency analysis
        if 'sort_optimization_savings' in df.columns and 'total_cost' in df.columns:
            df['net_cost'] = df['total_cost'] - df.get('sort_optimization_savings', 0)
            df['optimization_impact_pct'] = np.where(
                df['total_cost'] > 0,
                (df.get('sort_optimization_savings', 0) / df['total_cost'] * 100).round(2),
                0
            )

        settings = pd.DataFrame([{"key": k, "value": v} for k, v in run_kv.items()])

        with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
            df.to_excel(xw, sheet_name=safe_sheet_name("enhanced_kpi_compare"), index=False)

            for sheet_name, wide_df in wide_views.items():
                wide_df.to_excel(xw, sheet_name=sheet_name, index=False)

            settings.to_excel(xw, sheet_name=safe_sheet_name("run_settings"), index=False)

            # Enhanced fill rate comparison
            if 'avg_truck_fill_rate' in df.columns:
                fill_comparison = df.pivot_table(
                    index='base_id',
                    columns='strategy',
                    values=['avg_truck_fill_rate', 'avg_container_fill_rate', 'total_packages_dwelled'],
                    aggfunc='first',
                    fill_value=0
                ).round(3)
                fill_comparison.to_excel(xw, sheet_name=safe_sheet_name("fill_rate_comparison"))

            # Optimization impact summary
            if 'sort_optimization_savings' in df.columns:
                opt_summary = df.groupby('strategy').agg({
                    'sort_optimization_savings': ['sum', 'mean', 'count'],
                    'optimization_impact_pct': 'mean',
                    'total_cost': 'sum'
                }).round(2)
                opt_summary.to_excel(xw, sheet_name=safe_sheet_name("optimization_impact"))

            # Strategic recommendations
            if len(df) >= 2:
                create_strategic_recommendations_sheet(xw, df)

        return True

    except Exception:
        return False


def create_strategic_recommendations_sheet(writer, compare_df):
    """Generate strategic recommendations based on comparison analysis."""
    try:
        recommendations = []

        container_data = compare_df[compare_df['strategy'] == 'container']
        fluid_data = compare_df[compare_df['strategy'] == 'fluid']

        if not container_data.empty and not fluid_data.empty:
            container_cost = container_data['total_cost'].mean()
            fluid_cost = fluid_data['total_cost'].mean()
            cost_advantage = abs(container_cost - fluid_cost)
            optimal_strategy = 'container' if container_cost <= fluid_cost else 'fluid'

            recommendations.append({
                'category': 'Strategic Direction',
                'recommendation': f'Implement {optimal_strategy} strategy network-wide',
                'rationale': f'${cost_advantage:,.0f}/day cost advantage over alternative',
                'priority': 'Critical',
                'implementation_timeline': '6-12 months'
            })

            if 'sort_optimization_savings' in container_data.columns:
                sort_savings = container_data['sort_optimization_savings'].mean()
                if sort_savings > 1000:
                    recommendations.append({
                        'category': 'Sort Optimization',
                        'recommendation': 'Prioritize containerization level optimization',
                        'rationale': f'Additional ${sort_savings:,.0f}/day savings potential',
                        'priority': 'High',
                        'implementation_timeline': '3-6 months'
                    })

            avg_fill = container_data.get('avg_truck_fill_rate', pd.Series([0])).mean()
            if avg_fill < 0.75:
                potential_improvement = (0.85 - avg_fill) * 100
                recommendations.append({
                    'category': 'Operational Efficiency',
                    'recommendation': 'Focus on truck utilization improvement programs',
                    'rationale': f'Current {avg_fill:.1%} fill rate has {potential_improvement:.1f}% improvement potential',
                    'priority': 'Medium',
                    'implementation_timeline': '1-3 months'
                })

        recommendations_df = pd.DataFrame(recommendations)
        if not recommendations_df.empty:
            recommendations_df.to_excel(writer, sheet_name=safe_sheet_name("strategic_recommendations"), index=False)

    except Exception:
        pass


def write_executive_summary_workbook(path, results_by_strategy: dict, run_kv: dict, base_id: str):
    """Write executive summary workbook with key business deliverables."""
    try:
        summary_data = {}

        container_results = results_by_strategy.get('container', {})
        fluid_results = results_by_strategy.get('fluid', {})

        # Monday Key Answers
        monday_answers = create_monday_key_answers(container_results, fluid_results)
        summary_data['Monday_Key_Answers'] = monday_answers

        # Containerization Strategy Comparison
        containerization_comparison = create_containerization_strategy_comparison(container_results, fluid_results)
        summary_data['Containerization_Strategy'] = containerization_comparison

        # Hub Hourly Throughput
        hub_throughput = create_hub_hourly_throughput_analysis(container_results, fluid_results)
        summary_data['Hub_Hourly_Throughput'] = hub_throughput

        # Facility Comparison
        facility_comparison = create_all_facilities_comparison(container_results, fluid_results)
        summary_data['All_Facilities_Comparison'] = facility_comparison

        # Zone Flow Analysis
        zone_flow = create_zone_flow_analysis(container_results, fluid_results)
        if not zone_flow.empty:
            summary_data['Zone_Flow_Analysis'] = zone_flow

        # Path Type Summary
        path_type_summary = create_path_type_summary(container_results, fluid_results)
        if not path_type_summary.empty:
            summary_data['Path_Type_Summary'] = path_type_summary

        # Run Settings
        settings_df = pd.DataFrame([{"key": k, "value": v} for k, v in run_kv.items()])
        summary_data['Run_Settings'] = settings_df

        # Write all sheets
        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            for sheet_name, df in summary_data.items():
                safe_name = safe_sheet_name(sheet_name)
                df.to_excel(writer, sheet_name=safe_name, index=False)

        return True

    except Exception:
        return False


def create_monday_key_answers(container_results: dict, fluid_results: dict) -> pd.DataFrame:
    """Create Monday key answers based on results comparison."""
    try:
        container_kpis = container_results.get('kpis', pd.Series())
        fluid_kpis = fluid_results.get('kpis', pd.Series())

        container_cost = container_kpis.get('total_cost', 0)
        fluid_cost = fluid_kpis.get('total_cost', 0)
        optimal_strategy = 'container' if container_cost <= fluid_cost else 'fluid'
        cost_difference = abs(container_cost - fluid_cost)

        sort_savings = container_kpis.get('sort_optimization_savings', 0)
        container_fill = container_kpis.get('avg_truck_fill_rate', 0)
        fluid_fill = fluid_kpis.get('avg_truck_fill_rate', 0)

        monday_answers = pd.DataFrame([
            {
                'question': '1. Optimal Containerization Strategy',
                'answer': f'{optimal_strategy.upper()} strategy',
                'detail': f'Base cost difference: ${cost_difference:,.0f}/day. Sort optimization saves additional ${sort_savings:,.0f}/day',
                'realistic_metrics': f"Fill rates - Container: {container_fill:.1%}, Fluid: {fluid_fill:.1%}"
            },
            {
                'question': '2. Fill Rate Analysis',
                'answer': f'Container strategy achieves {container_fill:.1%} truck fill rate',
                'detail': f'Compare to fluid: {fluid_fill:.1%} truck fill rate',
                'realistic_metrics': f'Actual physical space utilization (not theoretical maximum)'
            },
            {
                'question': '3. Sort Point Optimization Impact',
                'answer': f'${sort_savings:,.0f}/day potential savings from optimal containerization',
                'detail': f'Enhanced containerization level optimization across network',
                'realistic_metrics': f'Improved consolidation efficiency through targeted deeper containerization'
            },
            {
                'question': '4. Network Performance Summary',
                'answer': f'Total daily cost: ${container_cost if optimal_strategy == "container" else fluid_cost:,.0f}',
                'detail': f'Cost per package: ${container_kpis.get("cost_per_pkg", 0):.3f} (container) vs ${fluid_kpis.get("cost_per_pkg", 0):.3f} (fluid)',
                'realistic_metrics': f'Network processes {container_kpis.get("num_ods", 0)} origin-destination pairs'
            }
        ])

        return monday_answers

    except Exception as e:
        return pd.DataFrame([{"question": "Error", "answer": "Could not generate answers", "detail": str(e)}])


def create_containerization_strategy_comparison(container_results: dict, fluid_results: dict) -> pd.DataFrame:
    """Create strategy comparison summary."""
    try:
        container_kpis = container_results.get('kpis', pd.Series())
        fluid_kpis = fluid_results.get('kpis', pd.Series())

        comparison = pd.DataFrame([
            {
                'strategy': 'container',
                'total_cost': container_kpis.get('total_cost', 0),
                'cost_per_pkg': container_kpis.get('cost_per_pkg', 0),
                'sort_optimization_savings': container_kpis.get('sort_optimization_savings', 0),
                'avg_truck_fill_rate': container_kpis.get('avg_truck_fill_rate', 0),
                'avg_container_fill_rate': container_kpis.get('avg_container_fill_rate', 0),
                'total_packages_dwelled': container_kpis.get('total_packages_dwelled', 0)
            },
            {
                'strategy': 'fluid',
                'total_cost': fluid_kpis.get('total_cost', 0),
                'cost_per_pkg': fluid_kpis.get('cost_per_pkg', 0),
                'sort_optimization_savings': fluid_kpis.get('sort_optimization_savings', 0),
                'avg_truck_fill_rate': fluid_kpis.get('avg_truck_fill_rate', 0),
                'avg_container_fill_rate': fluid_kpis.get('avg_container_fill_rate', 0),
                'total_packages_dwelled': fluid_kpis.get('total_packages_dwelled', 0)
            }
        ])

        return comparison

    except Exception:
        return pd.DataFrame()


def create_hub_hourly_throughput_analysis(container_results: dict, fluid_results: dict) -> pd.DataFrame:
    """Create hub hourly throughput analysis."""
    try:
        container_facilities = container_results.get('facility_rollup', pd.DataFrame())

        if container_facilities.empty:
            return pd.DataFrame([{"note": "No facility data available"}])

        container_hubs = container_facilities[
            container_facilities.get('type', '').isin(['hub', 'hybrid'])
        ].copy()

        if container_hubs.empty:
            return pd.DataFrame([{"note": "No hub data available"}])

        throughput_cols = ['peak_hourly_throughput', 'injection_hourly_throughput', 'intermediate_hourly_throughput']
        base_cols = ['facility', 'type']

        available_cols = base_cols + [col for col in throughput_cols if col in container_hubs.columns]
        hub_throughput = container_hubs[available_cols].copy()

        fluid_facilities = fluid_results.get('facility_rollup', pd.DataFrame())
        if not fluid_facilities.empty:
            fluid_hubs = fluid_facilities[fluid_facilities.get('type', '').isin(['hub', 'hybrid'])].copy()
            if not fluid_hubs.empty:
                fluid_throughput = fluid_hubs.set_index('facility')[
                    [col for col in throughput_cols if col in fluid_hubs.columns]
                ].add_suffix('_fluid')

                hub_throughput = hub_throughput.set_index('facility').join(
                    fluid_throughput, how='left'
                ).reset_index().fillna(0)

        hub_throughput = hub_throughput.sort_values(
            'peak_hourly_throughput' if 'peak_hourly_throughput' in hub_throughput.columns else 'facility'
        )

        return hub_throughput

    except Exception:
        return pd.DataFrame([{"note": "Error creating hub analysis"}])


def create_all_facilities_comparison(container_results: dict, fluid_results: dict) -> pd.DataFrame:
    """Create all facilities comparison."""
    try:
        container_facilities = container_results.get('facility_rollup', pd.DataFrame())

        if container_facilities.empty:
            return pd.DataFrame([{"note": "No facility data available"}])

        facility_comparison = container_facilities.copy()
        facility_comparison['strategy'] = 'container'

        fluid_facilities = fluid_results.get('facility_rollup', pd.DataFrame())
        if not fluid_facilities.empty:
            fluid_comparison = fluid_facilities.copy()
            fluid_comparison['strategy'] = 'fluid'
            facility_comparison = pd.concat([facility_comparison, fluid_comparison], ignore_index=True)

        return facility_comparison

    except Exception:
        return pd.DataFrame([{"note": "Error creating facilities comparison"}])


def create_zone_flow_analysis(container_results: dict, fluid_results: dict) -> pd.DataFrame:
    """Create zone flow analysis."""
    try:
        container_od = container_results.get('od_out', pd.DataFrame())

        if container_od.empty or 'zone' not in container_od.columns:
            return pd.DataFrame()

        zone_flow = container_od.groupby('zone').agg({
            'pkgs_day': 'sum',
            'total_cost': 'sum',
            'origin': 'nunique',
            'dest': 'nunique'
        }).reset_index()

        zone_flow.columns = ['zone', 'total_packages', 'total_cost', 'unique_origins', 'unique_destinations']
        zone_flow['cost_per_pkg'] = zone_flow['total_cost'] / zone_flow['total_packages'].replace(0, 1)

        return zone_flow

    except Exception:
        return pd.DataFrame()


def create_path_type_summary(container_results: dict, fluid_results: dict) -> pd.DataFrame:
    """Create path type summary."""
    try:
        container_od = container_results.get('od_out', pd.DataFrame())

        if container_od.empty or 'path_type' not in container_od.columns:
            return pd.DataFrame()

        path_summary = container_od.groupby(['origin', 'path_type']).agg({
            'pkgs_day': 'sum',
            'dest': 'count'
        }).reset_index()

        path_summary.columns = ['origin_facility', 'path_type', 'total_packages', 'route_count']

        path_summary['pct_of_origin_volume'] = path_summary.groupby('origin_facility')['total_packages'].transform(
            lambda x: (path_summary.loc[x.index, 'total_packages'] / x.sum() * 100).round(1)
        )

        return path_summary

    except Exception:
        return pd.DataFrame()


def write_consolidated_multi_year_workbook(path: Path, all_results: list, run_kv: dict):
    """Write consolidated workbook with stacked data across all years and strategies."""
    try:
        consolidated_data = {}

        # Stack facility rollup data
        facility_data = []
        for result in all_results:
            facility_rollup = result.get('facility_rollup', pd.DataFrame())
            if not facility_rollup.empty:
                facility_rollup['scenario_id'] = result.get('scenario_id', 'unknown')
                facility_rollup['strategy'] = result.get('strategy', 'unknown')
                facility_rollup['year'] = result.get('year', 2025)
                facility_data.append(facility_rollup)

        if facility_data:
            consolidated_data['Facility_Rollup'] = pd.concat(facility_data, ignore_index=True)

        # Write consolidated file
        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            for sheet_name, df in consolidated_data.items():
                safe_name = safe_sheet_name(sheet_name)
                df.to_excel(writer, sheet_name=safe_name, index=False)

        return True

    except Exception:
        return False