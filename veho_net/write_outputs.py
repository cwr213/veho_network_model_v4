import pandas as pd
import numpy as np


def write_workbook(path, scen_sum, od_out, path_detail, dwell_hotspots, facility_rollup, arc_summary, kpis,
                   sort_allocation_summary=None):
    """Enhanced workbook writer with complete sort optimization and fill/spill insights."""
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        # Core sheets
        scen_sum.to_excel(xw, sheet_name="scenario_summary", index=False)
        od_out.to_excel(xw, sheet_name="od_selected_paths", index=False)
        path_detail.to_excel(xw, sheet_name="path_steps_selected", index=False)
        dwell_hotspots.to_excel(xw, sheet_name="dwell_hotspots", index=False)
        facility_rollup.to_excel(xw, sheet_name="facility_rollup", index=False)
        arc_summary.to_excel(xw, sheet_name="arc_summary", index=False)
        kpis.to_frame("value").to_excel(xw, sheet_name="kpis")

        # Enhanced: Add sort optimization summary if available
        if sort_allocation_summary is not None and not sort_allocation_summary.empty:
            sort_allocation_summary.to_excel(xw, sheet_name="sort_allocation_summary", index=False)

            # Create comprehensive sort optimization insights
            create_sort_insights_sheet(xw, sort_allocation_summary, od_out, facility_rollup)

        # Enhanced: Add fill rate analysis sheet
        if not od_out.empty:
            create_fill_rate_analysis_sheet(xw, od_out, arc_summary)

        # Enhanced: Add containerization strategy analysis
        if 'containerization_level' in od_out.columns:
            create_containerization_strategy_sheet(xw, od_out, facility_rollup)


def create_sort_insights_sheet(writer, sort_allocation_summary, od_out, facility_rollup):
    """Create comprehensive sort optimization insights with fill and spill analysis."""

    # Summary by containerization level
    if 'containerization_level' in od_out.columns:
        level_summary = od_out.groupby('containerization_level').agg({
            'pkgs_day': 'sum',
            'total_cost': 'sum',
            'cost_per_pkg': 'mean',
            'origin': 'nunique',
            'sort_points_used': 'sum',
            'containerization_efficiency_score': 'mean'
        }).reset_index()
        level_summary.columns = ['containerization_level', 'total_packages', 'total_cost',
                                 'avg_cost_per_pkg', 'unique_origins', 'total_sort_points', 'avg_efficiency_score']
        level_summary['cost_savings_estimate'] = (
                    level_summary['avg_efficiency_score'] * level_summary['total_sort_points']).round(0)
        level_summary.to_excel(writer, sheet_name="containerization_summary", index=False)

    # Enhanced: Fill and Spill Analysis
    if 'spill_opportunity_flag' in od_out.columns:
        spill_analysis = od_out[od_out['spill_opportunity_flag'] == True].copy()

        if not spill_analysis.empty:
            spill_summary = spill_analysis.groupby(['origin', 'spill_parent_hub']).agg({
                'pkgs_day': 'sum',
                'dest': 'count',
                'containerization_level': lambda x: x.value_counts().to_dict(),
                'total_cost': 'sum',
                'containerization_efficiency_score': 'mean'
            }).reset_index()
            spill_summary.columns = ['origin_facility', 'spill_parent_hub', 'spillable_packages_day',
                                     'affected_destinations', 'containerization_mix', 'total_affected_cost',
                                     'avg_spill_efficiency']

            spill_summary['spill_potential_pct'] = (
                    spill_summary['spillable_packages_day'] / spill_summary['spillable_packages_day'].sum() * 100
            ).round(1)

            # Add operational flexibility scoring
            spill_summary['operational_flexibility'] = pd.cut(
                spill_summary['spillable_packages_day'],
                bins=[0, 500, 2000, 10000, float('inf')],
                labels=['Low', 'Medium', 'High', 'Critical']
            ).astype(str)

            spill_summary.to_excel(writer, sheet_name="fill_spill_analysis", index=False)

    # Enhanced: Origin-level containerization strategy analysis
    if not od_out.empty and 'containerization_level' in od_out.columns:
        origin_strategy = od_out.groupby('origin').agg({
            'containerization_level': lambda x: x.value_counts().to_dict(),
            'pkgs_day': 'sum',
            'sort_points_used': 'sum',
            'containerization_efficiency_score': 'mean',
            'spill_opportunity_flag': 'sum',
            'has_secondary_region_sort': 'any',
            'total_cost': 'sum'
        }).reset_index()

        origin_strategy.columns = ['origin_facility', 'containerization_strategy_mix', 'total_packages_day',
                                   'total_sort_points_used', 'avg_efficiency_score', 'spill_opportunities',
                                   'has_region_backup', 'total_daily_cost']

        # Calculate strategy efficiency metrics per origin
        origin_strategy['sort_points_per_1000_pkgs'] = (
                origin_strategy['total_sort_points_used'] / (origin_strategy['total_packages_day'] / 1000)
        ).round(2)

        origin_strategy['cost_per_pkg'] = (
                origin_strategy['total_daily_cost'] / origin_strategy['total_packages_day']
        ).round(3)

        origin_strategy['operational_flexibility'] = np.where(
            origin_strategy['has_region_backup'] & (origin_strategy['spill_opportunities'] > 0),
            'High',
            'Medium' if origin_strategy['has_region_backup'] or (origin_strategy['spill_opportunities'] > 0) else 'Low'
        )

        # Add investment priority scoring
        origin_strategy['investment_priority'] = pd.cut(
            origin_strategy['avg_efficiency_score'],
            bins=[-float('inf'), 25, 75, 150, float('inf')],
            labels=['Low', 'Medium', 'High', 'Critical']
        ).astype(str)

        origin_strategy.to_excel(writer, sheet_name="origin_strategy_analysis", index=False)

    # Enhanced: Facility sort utilization analysis with spill handling
    if not facility_rollup.empty:
        sort_facilities = facility_rollup[
            (facility_rollup['type'].isin(['hub', 'hybrid'])) &
            (facility_rollup.get('max_sort_points_capacity', 0) > 0)
            ].copy()

        if not sort_facilities.empty:
            capacity_cols = ['facility', 'type', 'hub_tier', 'max_sort_points_capacity',
                             'sort_points_allocated', 'sort_utilization_rate', 'available_sort_capacity',
                             'injection_pkgs_day', 'peak_hourly_throughput']

            # Filter to existing columns
            available_cols = [col for col in capacity_cols if col in sort_facilities.columns]
            sort_utilization = sort_facilities[available_cols].copy()

            # Add calculated fields
            if 'available_sort_capacity' not in sort_utilization.columns:
                sort_utilization['available_sort_capacity'] = (
                        sort_utilization.get('max_sort_points_capacity', 0) -
                        sort_utilization.get('sort_points_allocated', 0)
                )

            # Enhanced: Add spill handling capacity analysis
            if 'spill_opportunity_flag' in od_out.columns:
                spill_volume_by_facility = od_out[od_out['spill_opportunity_flag'] == True].groupby(
                    'spill_parent_hub')['pkgs_day'].sum().to_dict()

                sort_utilization['potential_spill_volume'] = sort_utilization['facility'].map(
                    spill_volume_by_facility).fillna(0)

                sort_utilization['spill_handling_capacity'] = np.where(
                    sort_utilization['available_sort_capacity'] > 0,
                    'Available' if sort_utilization['potential_spill_volume'] > 0 else 'Available (No Demand)',
                    'Constrained' if sort_utilization['potential_spill_volume'] > 0 else 'Not Required'
                )

                # Calculate spill efficiency potential
                sort_utilization['spill_efficiency_potential'] = np.where(
                    (sort_utilization['available_sort_capacity'] > 0) & (
                                sort_utilization['potential_spill_volume'] > 0),
                    sort_utilization['potential_spill_volume'] / sort_utilization['available_sort_capacity'],
                    0
                ).round(1)

            # Add capacity utilization flags
            sort_utilization['capacity_status'] = pd.cut(
                sort_utilization.get('sort_utilization_rate', 0),
                bins=[0, 0.5, 0.8, 0.95, 1.1],
                labels=['Underutilized', 'Optimal', 'Near Capacity', 'Over Capacity']
            ).astype(str)

            sort_utilization.to_excel(writer, sheet_name="sort_capacity_analysis", index=False)

    # Enhanced: Route efficiency analysis with actionable insights
    if not sort_allocation_summary.empty:
        efficiency_analysis = sort_allocation_summary.copy()

        # Calculate comprehensive efficiency metrics
        efficiency_analysis['savings_per_package'] = (
                efficiency_analysis['daily_cost_savings'] / efficiency_analysis['pkgs_day']
        ).round(3)
        efficiency_analysis['efficiency_score'] = (
                efficiency_analysis['daily_cost_savings'] / efficiency_analysis['sort_points_used']
        ).round(2)

        # Add ROI metrics
        if 'total_cost' in sort_allocation_summary.columns:
            efficiency_analysis['roi_percentage'] = (
                    efficiency_analysis['daily_cost_savings'] / efficiency_analysis.get('total_cost', 1) * 100
            ).round(2)

        # Add implementation priority
        efficiency_analysis['implementation_priority'] = pd.cut(
            efficiency_analysis['efficiency_score'],
            bins=[-float('inf'), 25, 75, 150, float('inf')],
            labels=['Low Priority', 'Medium Priority', 'High Priority', 'Critical Priority']
        ).astype(str)

        # Add spill context if available
        if 'has_spill_opportunity' in efficiency_analysis.columns:
            efficiency_analysis['spill_context'] = np.where(
                efficiency_analysis['has_spill_opportunity'] == True,
                'Spillable',
                'Fixed'
            )

        efficiency_analysis = efficiency_analysis.sort_values('efficiency_score', ascending=False)
        efficiency_analysis.to_excel(writer, sheet_name="route_efficiency_analysis", index=False)

        # Create actionable insights summary
        create_actionable_insights_sheet(writer, efficiency_analysis, od_out, facility_rollup)


def create_actionable_insights_sheet(writer, efficiency_analysis, od_out, facility_rollup):
    """Generate actionable business insights for immediate implementation."""
    insights = []

    # High-priority route optimizations
    if not efficiency_analysis.empty:
        high_priority_routes = efficiency_analysis[efficiency_analysis['efficiency_score'] > 100]
        if not high_priority_routes.empty:
            total_savings = high_priority_routes['daily_cost_savings'].sum()
            insights.append({
                'category': 'High-Impact Routes',
                'insight': f'{len(high_priority_routes)} routes with >$100/day savings per sort point',
                'action': 'Prioritize for immediate deeper containerization',
                'potential_savings': f"${total_savings:,.0f}/day",
                'implementation_effort': 'Medium',
                'routes': ', '.join(high_priority_routes['od_pair_id'].head(5).tolist())
            })

    # Sort capacity constraints
    if not facility_rollup.empty and 'sort_utilization_rate' in facility_rollup.columns:
        over_capacity = facility_rollup[facility_rollup['sort_utilization_rate'] > 0.90]
        if not over_capacity.empty:
            insights.append({
                'category': 'Capacity Constraints',
                'insight': f'{len(over_capacity)} facilities operating above 90% sort capacity',
                'action': 'Review capacity expansion or load balancing opportunities',
                'potential_impact': f'{over_capacity["injection_pkgs_day"].sum():,.0f} pkgs/day affected',
                'implementation_effort': 'High',
                'facilities': ', '.join(over_capacity['facility'].tolist())
            })

    # Fill rate optimization opportunities
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

    # Spill opportunity insights
    if 'spill_opportunity_flag' in od_out.columns:
        spill_routes = od_out[od_out['spill_opportunity_flag'] == True]
        if not spill_routes.empty:
            spill_hubs = spill_routes['spill_parent_hub'].nunique()
            insights.append({
                'category': 'Operational Flexibility',
                'insight': f'{len(spill_routes)} routes have fill & spill capabilities',
                'action': 'Develop operational procedures for capacity management',
                'potential_impact': f'{spill_hubs} parent hubs can handle overflow',
                'implementation_effort': 'Medium',
                'spillable_volume': f"{spill_routes['pkgs_day'].sum():,.0f} pkgs/day"
            })

    # Underutilized sort capacity
    if not facility_rollup.empty and 'sort_utilization_rate' in facility_rollup.columns:
        underutilized = facility_rollup[facility_rollup['sort_utilization_rate'] < 0.40]
        if not underutilized.empty:
            insights.append({
                'category': 'Underutilized Capacity',
                'insight': f'{len(underutilized)} facilities using <40% of sort capacity',
                'action': 'Consider consolidating volumes or reallocating capacity',
                'potential_impact': 'Reduce fixed costs and improve efficiency',
                'implementation_effort': 'High',
                'facilities': ', '.join(underutilized['facility'].tolist())
            })

    insights_df = pd.DataFrame(insights)
    if not insights_df.empty:
        # Add priority scoring
        effort_scores = {'Low': 1, 'Medium': 2, 'High': 3}
        insights_df['effort_score'] = insights_df['implementation_effort'].map(effort_scores)

        # Extract potential savings numbers for ranking
        insights_df['savings_numeric'] = insights_df['potential_savings'].fillna('$0').str.extract(
            r'\$([0-9,]+)').fillna('0').astype(str).str.replace(',', '').astype(float)

        insights_df['priority_score'] = (insights_df['savings_numeric'] / 1000) / insights_df['effort_score']
        insights_df = insights_df.sort_values('priority_score', ascending=False)

        insights_df.to_excel(writer, sheet_name="actionable_insights", index=False)


def create_fill_rate_analysis_sheet(writer, od_out, arc_summary):
    """Create comprehensive fill rate and utilization analysis."""

    # OD-level fill rate analysis
    if 'truck_fill_rate' in od_out.columns:
        fill_analysis = od_out.groupby(['origin', 'path_type']).agg({
            'truck_fill_rate': 'mean',
            'container_fill_rate': 'mean',
            'packages_dwelled': 'sum',
            'pkgs_day': 'sum',
            'total_cost': 'sum',
            'containerization_level': lambda x: x.mode().iloc[0] if not x.empty else 'region'
        }).reset_index()

        fill_analysis['dwell_rate'] = np.where(
            fill_analysis['pkgs_day'] > 0,
            fill_analysis['packages_dwelled'] / fill_analysis['pkgs_day'],
            0
        )

        # Add efficiency scoring
        fill_analysis['fill_efficiency_score'] = (
                fill_analysis['truck_fill_rate'] * 0.6 +
                fill_analysis['container_fill_rate'] * 0.4
        ).round(3)

        fill_analysis['efficiency_category'] = pd.cut(
            fill_analysis['fill_efficiency_score'],
            bins=[0, 0.6, 0.8, 1.0],
            labels=['Low', 'Medium', 'High']
        ).astype(str)

        fill_analysis = fill_analysis.round(3)
        fill_analysis.to_excel(writer, sheet_name="fill_rate_analysis", index=False)

    # Lane-level utilization summary with enhanced metrics
    if arc_summary is not None and not arc_summary.empty:
        if 'truck_fill_rate' in arc_summary.columns:
            lane_utilization = arc_summary[[
                'from_facility', 'to_facility', 'pkgs_day', 'trucks',
                'truck_fill_rate', 'container_fill_rate', 'packages_per_truck',
                'total_cost', 'CPP'
            ]].copy()

            # Add utilization categories
            lane_utilization['utilization_category'] = pd.cut(
                lane_utilization['truck_fill_rate'],
                bins=[0, 0.6, 0.8, 1.0],
                labels=['Low (<60%)', 'Medium (60-80%)', 'High (80%+)']
            ).astype(str)

            # Add volume categories
            lane_utilization['volume_category'] = pd.cut(
                lane_utilization['pkgs_day'],
                bins=[0, 500, 2000, 10000, float('inf')],
                labels=['Low Volume', 'Medium Volume', 'High Volume', 'Very High Volume']
            ).astype(str)

            # Calculate improvement potential
            lane_utilization['fill_improvement_potential'] = np.where(
                lane_utilization['truck_fill_rate'] < 0.85,
                (0.85 - lane_utilization['truck_fill_rate']) * lane_utilization['pkgs_day'],
                0
            ).round(0)

            lane_utilization = lane_utilization.sort_values('truck_fill_rate', ascending=False)
            lane_utilization.to_excel(writer, sheet_name="lane_utilization_analysis", index=False)


def create_containerization_strategy_sheet(writer, od_out, facility_rollup):
    """Create strategic containerization analysis for network planning."""

    # Strategy effectiveness by distance zones
    if 'zone' in od_out.columns and 'containerization_level' in od_out.columns:
        zone_strategy = od_out.groupby(['zone', 'containerization_level']).agg({
            'pkgs_day': 'sum',
            'total_cost': 'sum',
            'cost_per_pkg': 'mean',
            'containerization_efficiency_score': 'mean',
            'origin': 'nunique'
        }).reset_index()

        zone_strategy['volume_share'] = zone_strategy.groupby('zone')['pkgs_day'].transform(lambda x: x / x.sum())
        zone_strategy['avg_savings_per_point'] = zone_strategy['containerization_efficiency_score']

        zone_strategy_pivot = zone_strategy.pivot_table(
            index='zone',
            columns='containerization_level',
            values=['pkgs_day', 'avg_savings_per_point'],
            fill_value=0
        )

        zone_strategy_pivot.to_excel(writer, sheet_name="zone_strategy_analysis")

    # Network-level containerization summary
    if 'containerization_level' in od_out.columns:
        network_summary = pd.DataFrame([{
            'metric': 'Total Daily Volume',
            'region_level': od_out[od_out['containerization_level'] == 'region']['pkgs_day'].sum(),
            'market_level': od_out[od_out['containerization_level'] == 'market']['pkgs_day'].sum(),
            'sort_group_level': od_out[od_out['containerization_level'] == 'sort_group']['pkgs_day'].sum(),
        }, {
            'metric': 'Average Efficiency Score',
            'region_level': od_out[od_out['containerization_level'] == 'region'][
                'containerization_efficiency_score'].mean(),
            'market_level': od_out[od_out['containerization_level'] == 'market'][
                'containerization_efficiency_score'].mean(),
            'sort_group_level': od_out[od_out['containerization_level'] == 'sort_group'][
                'containerization_efficiency_score'].mean(),
        }, {
            'metric': 'Route Count',
            'region_level': (od_out['containerization_level'] == 'region').sum(),
            'market_level': (od_out['containerization_level'] == 'market').sum(),
            'sort_group_level': (od_out['containerization_level'] == 'sort_group').sum(),
        }])

        network_summary.to_excel(writer, sheet_name="network_strategy_summary", index=False)


def write_compare_workbook(path, compare_df: pd.DataFrame, run_kv: dict):
    """Enhanced comparison workbook with comprehensive sort optimization metrics."""

    # Enhanced column ordering with new metrics
    front = ["base_id", "strategy", "scenario_id", "output_file"]
    kpi_cols = [
        "total_cost", "cost_per_pkg", "num_ods",
        "sla_violations", "around_world_flags",
        "pct_direct", "pct_1_touch", "pct_2_touch", "pct_3_touch",
        # Enhanced metrics
        "sort_optimization_savings", "avg_container_fill_rate",
        "avg_truck_fill_rate", "total_packages_dwelled"
    ]
    cols = front + [c for c in kpi_cols if c in compare_df.columns]
    df = compare_df[cols].copy()

    # Enhanced wide views with new metrics
    metrics_to_pivot = ['cost_per_pkg', 'sort_optimization_savings', 'avg_truck_fill_rate', 'avg_container_fill_rate']
    wide_views = {}

    for metric in metrics_to_pivot:
        if metric in df.columns:
            wide = df.pivot(index="base_id", columns="strategy", values=metric)
            wide = wide.rename_axis(None, axis=1).reset_index()

            # Add comparison calculations
            if 'container' in wide.columns and 'fluid' in wide.columns:
                wide[f'{metric}_advantage'] = wide['container'] - wide['fluid']
                wide[f'{metric}_advantage_pct'] = (wide[f'{metric}_advantage'] / wide['fluid'] * 100).round(1)

            wide_views[f"{metric}_comparison"] = wide

    # Enhanced strategy efficiency analysis
    if 'sort_optimization_savings' in df.columns and 'total_cost' in df.columns:
        df['net_cost'] = df['total_cost'] - df.get('sort_optimization_savings', 0)
        df['optimization_impact_pct'] = np.where(
            df['total_cost'] > 0,
            (df.get('sort_optimization_savings', 0) / df['total_cost'] * 100).round(2),
            0
        )
        df['total_efficiency_score'] = (
                df['avg_truck_fill_rate'] * 0.4 +
                df['avg_container_fill_rate'] * 0.3 +
                df['optimization_impact_pct'] * 0.3
        ).round(2)

    # Settings snapshot for traceability
    settings = pd.DataFrame([{"key": k, "value": v} for k, v in run_kv.items()])

    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        df.to_excel(xw, sheet_name="enhanced_kpi_compare", index=False)

        # Write multiple comparison views
        for sheet_name, wide_df in wide_views.items():
            wide_df.to_excel(xw, sheet_name=sheet_name, index=False)

        settings.to_excel(xw, sheet_name="run_settings", index=False)

        # Enhanced: Add comprehensive fill rate comparison
        if 'avg_truck_fill_rate' in df.columns:
            fill_comparison = df.pivot_table(
                index='base_id',
                columns='strategy',
                values=['avg_truck_fill_rate', 'avg_container_fill_rate', 'total_packages_dwelled'],
                aggfunc='first',
                fill_value=0
            ).round(3)
            fill_comparison.to_excel(xw, sheet_name="fill_rate_comparison")

        # Enhanced: Add optimization impact summary
        if 'sort_optimization_savings' in df.columns:
            opt_summary = df.groupby('strategy').agg({
                'sort_optimization_savings': ['sum', 'mean', 'count'],
                'optimization_impact_pct': 'mean',
                'total_cost': 'sum',
                'total_efficiency_score': 'mean'
            }).round(2)
            opt_summary.to_excel(xw, sheet_name="optimization_impact_summary")

        # Enhanced: Strategic recommendations
        if len(df) >= 2:
            create_strategic_recommendations_sheet(xw, df)


def create_strategic_recommendations_sheet(writer, compare_df):
    """Generate strategic recommendations based on comparison analysis."""
    recommendations = []

    # Determine optimal strategy
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
            'implementation_timeline': '6-12 months',
            'key_metrics': f"Fill rates: {container_data.get('avg_truck_fill_rate', pd.Series([0])).mean():.1%} (container) vs {fluid_data.get('avg_truck_fill_rate', pd.Series([0])).mean():.1%} (fluid)"
        })

        # Sort optimization impact
        if 'sort_optimization_savings' in container_data.columns:
            sort_savings = container_data['sort_optimization_savings'].mean()
            if sort_savings > 1000:
                recommendations.append({
                    'category': 'Sort Optimization',
                    'recommendation': 'Prioritize containerization level optimization',
                    'rationale': f'Additional ${sort_savings:,.0f}/day savings potential through smart containerization',
                    'priority': 'High',
                    'implementation_timeline': '3-6 months',
                    'key_metrics': f"Efficiency-based allocation shows {sort_savings / container_cost * 100:.1f}% cost improvement"
                })

        # Fill rate improvement opportunities
        avg_fill = container_data.get('avg_truck_fill_rate', pd.Series([0])).mean()
        if avg_fill < 0.75:
            potential_improvement = (0.85 - avg_fill) * 100
            recommendations.append({
                'category': 'Operational Efficiency',
                'recommendation': 'Focus on truck utilization improvement programs',
                'rationale': f'Current {avg_fill:.1%} fill rate has {potential_improvement:.1f}% improvement potential',
                'priority': 'Medium',
                'implementation_timeline': '1-3 months',
                'key_metrics': f"Target 85% fill rate through better consolidation"
            })

    recommendations_df = pd.DataFrame(recommendations)
    if not recommendations_df.empty:
        # Add implementation difficulty scoring
        priority_scores = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        recommendations_df['priority_score'] = recommendations_df['priority'].map(priority_scores)
        recommendations_df = recommendations_df.sort_values('priority_score', ascending=False)

        recommendations_df.to_excel(writer, sheet_name="strategic_recommendations", index=False)


def write_enhanced_executive_summary(path, strategy_comparison, hub_analysis, optimization_insights,
                                     facility_requirements, key_findings):
    """Write comprehensive executive summary with sort optimization insights."""

    summary_data = {
        'Strategy_Comparison': strategy_comparison,
        'Hub_Analysis': hub_analysis,
        'Optimization_Insights': optimization_insights,
        'Facility_Requirements': facility_requirements,
        'Key_Findings': key_findings
    }

    # Add derived insights
    if not strategy_comparison.empty:
        # Calculate comprehensive ROI analysis
        container_row = strategy_comparison[strategy_comparison['strategy'] == 'container']
        if not container_row.empty:
            base_cost = container_row.iloc[0].get('total_daily_cost', 0)
            optimization_savings = container_row.iloc[0].get('sort_optimization_savings', 0)

            optimization_roi = (optimization_savings / max(base_cost, 1)) * 100 if base_cost > 0 else 0

            roi_analysis = pd.DataFrame([{
                'metric': 'Sort Optimization ROI',
                'value': f"{optimization_roi:.2f}%",
                'description': 'Daily cost savings as % of total daily cost',
                'impact_category': 'High' if optimization_roi > 5 else 'Medium' if optimization_roi > 2 else 'Low',
                'annual_value': f"${optimization_savings * 365:,.0f}"
            }, {
                'metric': 'Fill Rate Efficiency',
                'value': f"{container_row.iloc[0].get('avg_truck_fill_rate', 0):.1%}",
                'description': 'Average truck utilization across network',
                'impact_category': 'High' if container_row.iloc[0].get('avg_truck_fill_rate', 0) > 0.8 else 'Medium',
                'annual_value': f"Baseline established"
            }])

            summary_data['ROI_Analysis'] = roi_analysis

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for sheet_name, df in summary_data.items():
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=False)


# Enhanced utility functions for complex analysis
def calculate_optimization_metrics(od_selected, sort_allocation_summary):
    """Calculate comprehensive optimization metrics with fill and spill insights."""
    metrics = {}

    if not sort_allocation_summary.empty:
        metrics['total_daily_savings'] = sort_allocation_summary['daily_cost_savings'].sum()
        metrics['total_sort_points_used'] = sort_allocation_summary['sort_points_used'].sum()
        metrics['avg_efficiency'] = metrics['total_daily_savings'] / max(metrics['total_sort_points_used'], 1)
        metrics['routes_optimized'] = len(sort_allocation_summary)

        # Enhanced: Spill opportunity metrics
        if 'has_spill_opportunity' in sort_allocation_summary.columns:
            spill_routes = sort_allocation_summary[sort_allocation_summary['has_spill_opportunity'] == True]
            metrics['spill_routes_count'] = len(spill_routes)
            metrics['spill_volume'] = spill_routes['pkgs_day'].sum() if not spill_routes.empty else 0

    if not od_selected.empty and 'containerization_level' in od_selected.columns:
        level_dist = od_selected['containerization_level'].value_counts(normalize=True) * 100
        metrics['pct_region_level'] = level_dist.get('region', 0)
        metrics['pct_market_level'] = level_dist.get('market', 0)
        metrics['pct_sort_group_level'] = level_dist.get('sort_group', 0)

    if not od_selected.empty and 'truck_fill_rate' in od_selected.columns:
        metrics['avg_truck_fill_rate'] = od_selected['truck_fill_rate'].mean()
        metrics['high_fill_routes'] = (od_selected['truck_fill_rate'] >= 0.85).sum()
        metrics['optimization_opportunities'] = (od_selected['truck_fill_rate'] < 0.60).sum()

    return metrics


def generate_actionable_insights(facility_rollup, sort_allocation_summary, od_selected):
    """Generate actionable business insights from optimization analysis."""
    insights = []

    # Sort capacity insights
    if not facility_rollup.empty and 'sort_utilization_rate' in facility_rollup.columns:
        over_capacity = facility_rollup[facility_rollup['sort_utilization_rate'] > 0.90]
        if not over_capacity.empty:
            insights.append({
                'category': 'Capacity Constraint',
                'insight': f'{len(over_capacity)} facilities operating above 90% sort capacity',
                'recommendation': 'Consider capacity expansion or load balancing',
                'priority': 'High',
                'facilities': ', '.join(over_capacity['facility'].tolist()),
                'affected_volume': f"{over_capacity['injection_pkgs_day'].sum():,.0f} pkgs/day"
            })

    # Fill rate insights
    if not od_selected.empty and 'truck_fill_rate' in od_selected.columns:
        low_fill_routes = od_selected[od_selected['truck_fill_rate'] < 0.60]
        if not low_fill_routes.empty:
            insights.append({
                'category': 'Utilization Opportunity',
                'insight': f'{len(low_fill_routes)} routes with truck fill rates below 60%',
                'recommendation': 'Review consolidation opportunities or adjust service levels',
                'priority': 'Medium',
                'potential_savings': f"${(low_fill_routes['pkgs_day'] * 0.50).sum():,.0f}/day estimated",
                'affected_volume': f"{low_fill_routes['pkgs_day'].sum():,.0f} pkgs/day"
            })

    # Optimization impact insights
    if not sort_allocation_summary.empty:
        high_efficiency_routes = sort_allocation_summary[
            sort_allocation_summary.get('efficiency_score', 0) > 100
            ]
        if not high_efficiency_routes.empty:
            insights.append({
                'category': 'High-Impact Routes',
                'insight': f'{len(high_efficiency_routes)} routes with >$100/day savings per sort point',
                'recommendation': 'Prioritize these routes for deeper containerization',
                'priority': 'Critical',
                'total_savings': f"${high_efficiency_routes['daily_cost_savings'].sum():,.0f}/day",
                'implementation_effort': 'Medium'
            })

    # Spill opportunity insights
    if 'spill_opportunity_flag' in od_selected.columns:
        spill_routes = od_selected[od_selected['spill_opportunity_flag'] == True]
        if not spill_routes.empty:
            insights.append({
                'category': 'Operational Flexibility',
                'insight': f'{len(spill_routes)} routes have fill & spill capabilities',
                'recommendation': 'Develop operational procedures for dynamic capacity management',
                'priority': 'Medium',
                'spill_volume': f"{spill_routes['pkgs_day'].sum():,.0f} pkgs/day spillable",
                'affected_hubs': f"{spill_routes['spill_parent_hub'].nunique()} parent hubs"
            })

    return pd.DataFrame(insights)