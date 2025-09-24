import pandas as pd
import numpy as np


def safe_sheet_name(name: str) -> str:
    """
    Create Excel-safe sheet names that are <= 31 characters.

    Excel sheet name rules:
    - Max 31 characters
    - No invalid characters: [ ] : * ? / \
    - Cannot be blank

    Examples:
        safe_sheet_name("sort_optimization_savings_comparison")
        → "sort_optimization_comparison"

        safe_sheet_name("very_long_sheet_name_that_exceeds_31_characters")
        → "very_long_sheet_name_exceeds"
    """
    # Remove invalid characters
    invalid_chars = ['[', ']', ':', '*', '?', '/', '\\']
    clean_name = name
    for char in invalid_chars:
        clean_name = clean_name.replace(char, '_')

    # Truncate to 31 characters if needed
    if len(clean_name) <= 31:
        return clean_name

    # Intelligent truncation - try to preserve meaningful parts
    if '_comparison' in clean_name:
        # For comparison sheets, keep "_comparison" suffix
        base = clean_name.replace('_comparison', '')
        if len(base) <= 20:  # 20 + 11 ("_comparison") = 31
            return f"{base}_comparison"
        else:
            return f"{base[:20]}_comparison"

    elif '_analysis' in clean_name:
        # For analysis sheets, keep "_analysis" suffix
        base = clean_name.replace('_analysis', '')
        if len(base) <= 22:  # 22 + 9 ("_analysis") = 31
            return f"{base}_analysis"
        else:
            return f"{base[:22]}_analysis"

    elif '_summary' in clean_name:
        # For summary sheets, keep "_summary" suffix
        base = clean_name.replace('_summary', '')
        if len(base) <= 23:  # 23 + 8 ("_summary") = 31
            return f"{base}_summary"
        else:
            return f"{base[:23]}_summary"

    # Default truncation - just cut at 31 characters
    return clean_name[:31]


def write_workbook(path, scen_sum, od_out, path_detail, dwell_hotspots, facility_rollup, arc_summary, kpis,
                   sort_allocation_summary=None):
    """Enhanced workbook writer with complete sort optimization and fill/spill insights."""
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        # Core sheets with safe names
        scen_sum.to_excel(xw, sheet_name=safe_sheet_name("scenario_summary"), index=False)
        od_out.to_excel(xw, sheet_name=safe_sheet_name("od_selected_paths"), index=False)
        path_detail.to_excel(xw, sheet_name=safe_sheet_name("path_steps_selected"), index=False)
        dwell_hotspots.to_excel(xw, sheet_name=safe_sheet_name("dwell_hotspots"), index=False)
        facility_rollup.to_excel(xw, sheet_name=safe_sheet_name("facility_rollup"), index=False)
        arc_summary.to_excel(xw, sheet_name=safe_sheet_name("arc_summary"), index=False)
        kpis.to_frame("value").to_excel(xw, sheet_name=safe_sheet_name("kpis"))

        # Enhanced: Add sort optimization summary if available
        if sort_allocation_summary is not None and not sort_allocation_summary.empty:
            sort_allocation_summary.to_excel(xw, sheet_name=safe_sheet_name("sort_allocation_summary"), index=False)

            # Create comprehensive sort optimization insights
            create_sort_insights_sheet(xw, sort_allocation_summary, od_out, facility_rollup)

        # Enhanced: Add fill rate analysis sheet
        if not od_out.empty:
            create_fill_rate_analysis_sheet(xw, od_out, arc_summary)

        # Enhanced: Add containerization strategy analysis (SAFE VERSION)
        if 'containerization_level' in od_out.columns:
            create_containerization_strategy_sheet_safe(xw, od_out, facility_rollup)


def create_containerization_strategy_sheet_safe(writer, od_out, facility_rollup):
    """SAFE: Create strategic containerization analysis for network planning with missing column handling."""

    # Strategy effectiveness by distance zones (SAFE VERSION)
    if 'zone' in od_out.columns and 'containerization_level' in od_out.columns:
        # Build aggregation dict with only available columns
        agg_dict = {
            'pkgs_day': 'sum',
            'total_cost': 'sum',
            'cost_per_pkg': 'mean',
            'origin': 'nunique'
        }

        # Only include containerization_efficiency_score if it exists
        if 'containerization_efficiency_score' in od_out.columns:
            agg_dict['containerization_efficiency_score'] = 'mean'

        try:
            zone_strategy = od_out.groupby(['zone', 'containerization_level']).agg(agg_dict).reset_index()

            zone_strategy['volume_share'] = zone_strategy.groupby('zone')['pkgs_day'].transform(lambda x: x / x.sum())

            # Safe handling of efficiency score
            if 'containerization_efficiency_score' in zone_strategy.columns:
                zone_strategy['avg_savings_per_point'] = zone_strategy['containerization_efficiency_score']
            else:
                zone_strategy['avg_savings_per_point'] = 0.0

            # Create pivot table with available columns
            pivot_columns = ['pkgs_day', 'avg_savings_per_point']
            zone_strategy_pivot = zone_strategy.pivot_table(
                index='zone',
                columns='containerization_level',
                values=pivot_columns,
                fill_value=0
            )

            zone_strategy_pivot.to_excel(writer, sheet_name=safe_sheet_name("zone_strategy_analysis"))

        except Exception as e:
            print(f"Warning: Could not create zone strategy analysis: {e}")

    # Network-level containerization summary (SAFE VERSION)
    if 'containerization_level' in od_out.columns:
        try:
            network_summary_data = []

            # Basic metrics
            for level in ['region', 'market', 'sort_group']:
                level_data = od_out[od_out['containerization_level'] == level]
                volume = level_data['pkgs_day'].sum() if not level_data.empty else 0
                count = len(level_data)
                avg_efficiency = 0.0

                # Safe efficiency calculation
                if 'containerization_efficiency_score' in level_data.columns and not level_data.empty:
                    avg_efficiency = level_data['containerization_efficiency_score'].mean()

                network_summary_data.append({
                    'containerization_level': level,
                    'daily_volume': volume,
                    'route_count': count,
                    'avg_efficiency_score': avg_efficiency
                })

            network_summary = pd.DataFrame(network_summary_data)

            # Add percentages
            total_volume = network_summary['daily_volume'].sum()
            if total_volume > 0:
                network_summary['volume_percentage'] = (network_summary['daily_volume'] / total_volume * 100).round(1)
            else:
                network_summary['volume_percentage'] = 0.0

            network_summary.to_excel(writer, sheet_name=safe_sheet_name("network_strategy_summary"), index=False)

        except Exception as e:
            print(f"Warning: Could not create network strategy summary: {e}")


def create_sort_insights_sheet(writer, sort_allocation_summary, od_out, facility_rollup):
    """Create comprehensive sort optimization insights with safe column handling."""

    # Summary by containerization level (SAFE VERSION)
    if 'containerization_level' in od_out.columns:
        try:
            # Build aggregation with safe column handling
            agg_dict = {
                'pkgs_day': 'sum',
                'total_cost': 'sum',
                'cost_per_pkg': 'mean',
                'origin': 'nunique'
            }

            # Add optional columns if they exist
            if 'sort_points_used' in od_out.columns:
                agg_dict['sort_points_used'] = 'sum'
            if 'containerization_efficiency_score' in od_out.columns:
                agg_dict['containerization_efficiency_score'] = 'mean'

            level_summary = od_out.groupby('containerization_level').agg(agg_dict).reset_index()

            level_summary.columns = ['containerization_level', 'total_packages', 'total_cost',
                                     'avg_cost_per_pkg', 'unique_origins'] + \
                                    (['total_sort_points'] if 'sort_points_used' in agg_dict else []) + \
                                    (
                                        ['avg_efficiency_score'] if 'containerization_efficiency_score' in agg_dict else [])

            # Safe cost savings calculation
            if 'avg_efficiency_score' in level_summary.columns and 'total_sort_points' in level_summary.columns:
                level_summary['cost_savings_estimate'] = (
                        level_summary['avg_efficiency_score'] * level_summary['total_sort_points']
                ).round(0)
            else:
                level_summary['cost_savings_estimate'] = 0

            level_summary.to_excel(writer, sheet_name=safe_sheet_name("containerization_summary"), index=False)

        except Exception as e:
            print(f"Warning: Could not create containerization summary: {e}")

    # Enhanced: Fill and Spill Analysis (SAFE VERSION)
    if 'spill_opportunity_flag' in od_out.columns:
        try:
            spill_analysis = od_out[od_out['spill_opportunity_flag'] == True].copy()

            if not spill_analysis.empty:
                # Safe aggregation for spill analysis
                spill_agg_dict = {
                    'pkgs_day': 'sum',
                    'dest': 'count',
                    'total_cost': 'sum'
                }

                if 'containerization_level' in spill_analysis.columns:
                    spill_agg_dict['containerization_level'] = lambda x: x.value_counts().to_dict()
                if 'containerization_efficiency_score' in spill_analysis.columns:
                    spill_agg_dict['containerization_efficiency_score'] = 'mean'

                spill_summary = spill_analysis.groupby(['origin', 'spill_parent_hub']).agg(spill_agg_dict).reset_index()

                spill_summary.columns = ['origin_facility', 'spill_parent_hub', 'spillable_packages_day',
                                         'affected_destinations', 'total_affected_cost'] + \
                                        (
                                            ['containerization_mix'] if 'containerization_level' in spill_agg_dict else []) + \
                                        (
                                            ['avg_spill_efficiency'] if 'containerization_efficiency_score' in spill_agg_dict else [])

                # Safe percentage calculation
                total_spillable = spill_summary['spillable_packages_day'].sum()
                if total_spillable > 0:
                    spill_summary['spill_potential_pct'] = (
                            spill_summary['spillable_packages_day'] / total_spillable * 100
                    ).round(1)
                else:
                    spill_summary['spill_potential_pct'] = 0.0

                # Add operational flexibility scoring
                spill_summary['operational_flexibility'] = pd.cut(
                    spill_summary['spillable_packages_day'],
                    bins=[0, 500, 2000, 10000, float('inf')],
                    labels=['Low', 'Medium', 'High', 'Critical']
                ).astype(str)

                spill_summary.to_excel(writer, sheet_name=safe_sheet_name("fill_spill_analysis"), index=False)

        except Exception as e:
            print(f"Warning: Could not create fill spill analysis: {e}")

    # Enhanced: Origin-level containerization strategy analysis (SAFE VERSION)
    if not od_out.empty and 'containerization_level' in od_out.columns:
        try:
            # Safe aggregation for origin strategy
            origin_agg_dict = {
                'containerization_level': lambda x: x.value_counts().to_dict(),
                'pkgs_day': 'sum',
                'total_cost': 'sum'
            }

            # Add optional columns
            optional_cols = ['sort_points_used', 'containerization_efficiency_score',
                             'spill_opportunity_flag', 'has_secondary_region_sort']
            for col in optional_cols:
                if col in od_out.columns:
                    if col in ['sort_points_used']:
                        origin_agg_dict[col] = 'sum'
                    elif col in ['containerization_efficiency_score']:
                        origin_agg_dict[col] = 'mean'
                    elif col in ['spill_opportunity_flag']:
                        origin_agg_dict[col] = 'sum'
                    elif col in ['has_secondary_region_sort']:
                        origin_agg_dict[col] = 'any'

            origin_strategy = od_out.groupby('origin').agg(origin_agg_dict).reset_index()

            # Build column names dynamically
            base_cols = ['origin_facility', 'containerization_strategy_mix', 'total_packages_day', 'total_daily_cost']
            dynamic_cols = []

            for col in optional_cols:
                if col in origin_agg_dict:
                    if col == 'sort_points_used':
                        dynamic_cols.append('total_sort_points_used')
                    elif col == 'containerization_efficiency_score':
                        dynamic_cols.append('avg_efficiency_score')
                    elif col == 'spill_opportunity_flag':
                        dynamic_cols.append('spill_opportunities')
                    elif col == 'has_secondary_region_sort':
                        dynamic_cols.append('has_region_backup')

            origin_strategy.columns = ['origin'] + list(origin_strategy.columns[1:])
            origin_strategy = origin_strategy.rename(columns={'origin': 'origin_facility'})

            # Safe metric calculations
            if 'total_sort_points_used' in [c.replace('total_sort_points_used', 'sort_points_used') for c in
                                            origin_strategy.columns] and 'total_packages_day' in origin_strategy.columns:
                origin_strategy['sort_points_per_1000_pkgs'] = (
                        origin_strategy.get('total_sort_points_used', 0) / (
                            origin_strategy['total_packages_day'] / 1000)
                ).round(2)

            if 'total_daily_cost' in origin_strategy.columns and 'total_packages_day' in origin_strategy.columns:
                origin_strategy['cost_per_pkg'] = (
                        origin_strategy['total_daily_cost'] / origin_strategy['total_packages_day']
                ).round(3)

            origin_strategy.to_excel(writer, sheet_name=safe_sheet_name("origin_strategy_analysis"), index=False)

        except Exception as e:
            print(f"Warning: Could not create origin strategy analysis: {e}")

    # Continue with other analyses but with similar safe handling...
    _create_remaining_safe_analyses(writer, sort_allocation_summary, od_out, facility_rollup)


def _create_remaining_safe_analyses(writer, sort_allocation_summary, od_out, facility_rollup):
    """Create remaining analyses with safe error handling."""

    # Enhanced: Facility sort utilization analysis with safe handling
    try:
        if not facility_rollup.empty:
            required_cols = ['facility', 'type']
            available_cols = [col for col in required_cols if col in facility_rollup.columns]

            if len(available_cols) == len(required_cols):
                sort_facilities = facility_rollup[
                    (facility_rollup['type'].isin(['hub', 'hybrid'])) &
                    (facility_rollup.get('max_sort_points_capacity', 0) > 0)
                    ].copy()

                if not sort_facilities.empty:
                    # Build capacity analysis with available columns
                    base_capacity_cols = ['facility', 'type', 'hub_tier']
                    optional_capacity_cols = ['max_sort_points_capacity', 'sort_points_allocated',
                                              'sort_utilization_rate', 'available_sort_capacity',
                                              'injection_pkgs_day', 'peak_hourly_throughput']

                    available_capacity_cols = base_capacity_cols + [col for col in optional_capacity_cols if
                                                                    col in sort_facilities.columns]
                    sort_utilization = sort_facilities[available_capacity_cols].copy()

                    # Safe calculations
                    if 'available_sort_capacity' not in sort_utilization.columns:
                        if 'max_sort_points_capacity' in sort_utilization.columns and 'sort_points_allocated' in sort_utilization.columns:
                            sort_utilization['available_sort_capacity'] = (
                                    sort_utilization['max_sort_points_capacity'] -
                                    sort_utilization['sort_points_allocated']
                            )
                        else:
                            sort_utilization['available_sort_capacity'] = 0

                    sort_utilization.to_excel(writer, sheet_name=safe_sheet_name("sort_capacity_analysis"), index=False)

    except Exception as e:
        print(f"Warning: Could not create sort capacity analysis: {e}")

    # Enhanced: Route efficiency analysis (SAFE VERSION)
    try:
        if not sort_allocation_summary.empty:
            efficiency_analysis = sort_allocation_summary.copy()

            # Safe efficiency calculations
            if 'daily_cost_savings' in efficiency_analysis.columns and 'pkgs_day' in efficiency_analysis.columns:
                efficiency_analysis['savings_per_package'] = (
                        efficiency_analysis['daily_cost_savings'] / efficiency_analysis['pkgs_day'].replace(0, 1)
                ).round(3)

            if 'daily_cost_savings' in efficiency_analysis.columns and 'sort_points_used' in efficiency_analysis.columns:
                efficiency_analysis['efficiency_score'] = (
                        efficiency_analysis['daily_cost_savings'] / efficiency_analysis['sort_points_used'].replace(0,
                                                                                                                    1)
                ).round(2)

                # Add implementation priority based on efficiency score
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

            # Create actionable insights if we have the required data
            if 'efficiency_score' in efficiency_analysis.columns:
                create_actionable_insights_sheet_safe(writer, efficiency_analysis, od_out, facility_rollup)

    except Exception as e:
        print(f"Warning: Could not create route efficiency analysis: {e}")


def create_actionable_insights_sheet_safe(writer, efficiency_analysis, od_out, facility_rollup):
    """Generate actionable business insights with safe error handling."""
    try:
        insights = []

        # High-priority route optimizations (SAFE)
        if 'efficiency_score' in efficiency_analysis.columns and 'daily_cost_savings' in efficiency_analysis.columns:
            high_priority_routes = efficiency_analysis[efficiency_analysis['efficiency_score'] > 100]
            if not high_priority_routes.empty:
                total_savings = high_priority_routes['daily_cost_savings'].sum()
                insights.append({
                    'category': 'High-Impact Routes',
                    'insight': f'{len(high_priority_routes)} routes with >$100/day savings per sort point',
                    'action': 'Prioritize for immediate deeper containerization',
                    'potential_savings': f"${total_savings:,.0f}/day",
                    'implementation_effort': 'Medium',
                    'routes': ', '.join(high_priority_routes['od_pair_id'].head(
                        5).tolist()) if 'od_pair_id' in high_priority_routes.columns else 'See route_efficiency_analysis sheet'
                })

        # Fill rate optimization opportunities (SAFE)
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

        insights_df = pd.DataFrame(insights)
        if not insights_df.empty:
            insights_df.to_excel(writer, sheet_name=safe_sheet_name("actionable_insights"), index=False)

    except Exception as e:
        print(f"Warning: Could not create actionable insights: {e}")


def create_fill_rate_analysis_sheet(writer, od_out, arc_summary):
    """Create comprehensive fill rate and utilization analysis with safe handling."""

    try:
        # OD-level fill rate analysis (SAFE)
        if 'truck_fill_rate' in od_out.columns:
            # Build aggregation with available columns
            agg_dict = {
                'truck_fill_rate': 'mean',
                'pkgs_day': 'sum',
                'total_cost': 'sum'
            }

            # Add optional columns
            optional_fill_cols = ['container_fill_rate', 'packages_dwelled', 'containerization_level']
            for col in optional_fill_cols:
                if col in od_out.columns:
                    if col == 'containerization_level':
                        agg_dict[col] = lambda x: x.mode().iloc[0] if not x.empty else 'region'
                    else:
                        agg_dict[col] = 'mean' if col == 'container_fill_rate' else 'sum'

            fill_analysis = od_out.groupby(['origin', 'path_type']).agg(agg_dict).reset_index()

            # Safe calculations
            if 'packages_dwelled' in fill_analysis.columns:
                fill_analysis['dwell_rate'] = np.where(
                    fill_analysis['pkgs_day'] > 0,
                    fill_analysis['packages_dwelled'] / fill_analysis['pkgs_day'],
                    0
                )

            # Add efficiency scoring
            container_weight = 0.4 if 'container_fill_rate' in fill_analysis.columns else 0
            truck_weight = 0.6 if container_weight > 0 else 1.0

            fill_analysis['fill_efficiency_score'] = (
                    fill_analysis['truck_fill_rate'] * truck_weight +
                    (fill_analysis.get('container_fill_rate', 0) * container_weight)
            ).round(3)

            fill_analysis['efficiency_category'] = pd.cut(
                fill_analysis['fill_efficiency_score'],
                bins=[0, 0.6, 0.8, 1.0],
                labels=['Low', 'Medium', 'High']
            ).astype(str)

            fill_analysis = fill_analysis.round(3)
            fill_analysis.to_excel(writer, sheet_name=safe_sheet_name("fill_rate_analysis"), index=False)

    except Exception as e:
        print(f"Warning: Could not create fill rate analysis: {e}")

    try:
        # Lane-level utilization summary (SAFE)
        if arc_summary is not None and not arc_summary.empty:
            required_lane_cols = ['from_facility', 'to_facility', 'pkgs_day', 'trucks']
            if all(col in arc_summary.columns for col in required_lane_cols):

                # Build safe column list
                safe_lane_cols = required_lane_cols.copy()
                optional_lane_cols = ['truck_fill_rate', 'container_fill_rate', 'packages_per_truck', 'total_cost',
                                      'CPP']
                safe_lane_cols.extend([col for col in optional_lane_cols if col in arc_summary.columns])

                lane_utilization = arc_summary[safe_lane_cols].copy()

                # Safe categorizations
                if 'truck_fill_rate' in lane_utilization.columns:
                    lane_utilization['utilization_category'] = pd.cut(
                        lane_utilization['truck_fill_rate'],
                        bins=[0, 0.6, 0.8, 1.0],
                        labels=['Low (<60%)', 'Medium (60-80%)', 'High (80%+)']
                    ).astype(str)

                    # Calculate improvement potential
                    lane_utilization['fill_improvement_potential'] = np.where(
                        lane_utilization['truck_fill_rate'] < 0.85,
                        (0.85 - lane_utilization['truck_fill_rate']) * lane_utilization['pkgs_day'],
                        0
                    ).round(0)

                # Add volume categories
                lane_utilization['volume_category'] = pd.cut(
                    lane_utilization['pkgs_day'],
                    bins=[0, 500, 2000, 10000, float('inf')],
                    labels=['Low Volume', 'Medium Volume', 'High Volume', 'Very High Volume']
                ).astype(str)

                sort_col = 'truck_fill_rate' if 'truck_fill_rate' in lane_utilization.columns else 'pkgs_day'
                lane_utilization = lane_utilization.sort_values(sort_col, ascending=False)
                lane_utilization.to_excel(writer, sheet_name=safe_sheet_name("lane_utilization_analysis"), index=False)

    except Exception as e:
        print(f"Warning: Could not create lane utilization analysis: {e}")


def write_compare_workbook(path, compare_df: pd.DataFrame, run_kv: dict):
    """FIXED: Enhanced comparison workbook with safe sheet names."""

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

    # Enhanced wide views with new metrics and SAFE SHEET NAMES
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

            # FIXED: Use safe sheet names
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
        df['total_efficiency_score'] = (
                df['avg_truck_fill_rate'] * 0.4 +
                df['avg_container_fill_rate'] * 0.3 +
                df['optimization_impact_pct'] * 0.3
        ).round(2)

    # Settings snapshot for traceability
    settings = pd.DataFrame([{"key": k, "value": v} for k, v in run_kv.items()])

    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        # FIXED: Use safe sheet names throughout
        df.to_excel(xw, sheet_name=safe_sheet_name("enhanced_kpi_compare"), index=False)

        # Write multiple comparison views with safe names
        for sheet_name, wide_df in wide_views.items():
            wide_df.to_excel(xw, sheet_name=sheet_name, index=False)

        settings.to_excel(xw, sheet_name=safe_sheet_name("run_settings"), index=False)

        # Enhanced: Add comprehensive fill rate comparison
        if 'avg_truck_fill_rate' in df.columns:
            fill_comparison = df.pivot_table(
                index='base_id',
                columns='strategy',
                values=['avg_truck_fill_rate', 'avg_container_fill_rate', 'total_packages_dwelled'],
                aggfunc='first',
                fill_value=0
            ).round(3)
            fill_comparison.to_excel(xw, sheet_name=safe_sheet_name("fill_rate_comparison"))

        # Enhanced: Add optimization impact summary
        if 'sort_optimization_savings' in df.columns:
            opt_summary = df.groupby('strategy').agg({
                'sort_optimization_savings': ['sum', 'mean', 'count'],
                'optimization_impact_pct': 'mean',
                'total_cost': 'sum',
                'total_efficiency_score': 'mean'
            }).round(2)
            opt_summary.to_excel(xw, sheet_name=safe_sheet_name("optimization_impact_summary"))

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

        recommendations_df.to_excel(writer, sheet_name=safe_sheet_name("strategic_recommendations"), index=False)