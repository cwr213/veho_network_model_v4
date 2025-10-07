"""
Output Writing Module

Generates Excel workbooks with optimization results using v4 reporting structure.

Major Changes in v4:
    - Two facility sheets: facility_volume and facility_network_profile
    - Added network-level zone/sort/distance/touch metrics
    - Enhanced comparison and executive summary
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict


def write_workbook(
        path: Path,
        scenario_summary: pd.DataFrame,
        od_selected: pd.DataFrame,
        path_steps: pd.DataFrame,
        facility_volume: pd.DataFrame,
        facility_network_profile: pd.DataFrame,
        arc_summary: pd.DataFrame,
        kpis: pd.Series,
        sort_summary: Optional[pd.DataFrame] = None
) -> bool:
    """
    Write main scenario output workbook with v4 structure.

    Sheets:
    1. scenario_summary - Key metadata
    2. od_selected_paths - Optimal OD paths with costs
    3. path_steps_selected - Leg-by-leg breakdown
    4. facility_volume - Daily operational volumes (NEW: no cost columns)
    5. facility_network_profile - Network characteristics (NEW)
    6. arc_summary - Lane-level aggregations
    7. kpis - Network performance metrics (ENHANCED: zone/sort/distance/touch)
    8. sort_analysis - Sort level decisions (if provided)

    Args:
        path: Output file path
        scenario_summary: Scenario metadata
        od_selected: Selected OD paths
        path_steps: Path step details
        facility_volume: Facility operational metrics (NEW)
        facility_network_profile: Facility network characteristics (NEW)
        arc_summary: Arc/lane summary
        kpis: KPI metrics (ENHANCED)
        sort_summary: Sort analysis (optional)

    Returns:
        True if successful, False otherwise
    """
    try:
        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:

            # 1. Scenario Summary
            if not scenario_summary.empty:
                scenario_summary.to_excel(writer, sheet_name="scenario_summary", index=False)
            else:
                pd.DataFrame([{"key": "scenario_id", "value": "unknown"}]).to_excel(
                    writer, sheet_name="scenario_summary", index=False
                )

            # 2. OD Selected Paths
            if not od_selected.empty:
                od_selected.to_excel(writer, sheet_name="od_selected_paths", index=False)
            else:
                pd.DataFrame([{"note": "No OD paths"}]).to_excel(
                    writer, sheet_name="od_selected_paths", index=False
                )

            # 3. Path Steps
            if not path_steps.empty:
                path_steps.to_excel(writer, sheet_name="path_steps_selected", index=False)
            else:
                pd.DataFrame([{"note": "No path steps"}]).to_excel(
                    writer, sheet_name="path_steps_selected", index=False
                )

            # 4. Facility Volume (NEW - operational metrics only)
            if not facility_volume.empty:
                facility_volume.to_excel(writer, sheet_name="facility_volume", index=False)
            else:
                pd.DataFrame([{"note": "No facility data"}]).to_excel(
                    writer, sheet_name="facility_volume", index=False
                )

            # 5. Facility Network Profile (NEW - zone/sort/distance/touch)
            if not facility_network_profile.empty:
                facility_network_profile.to_excel(writer, sheet_name="facility_network_profile", index=False)
            else:
                pd.DataFrame([{"note": "No network profile data"}]).to_excel(
                    writer, sheet_name="facility_network_profile", index=False
                )

            # 6. Arc Summary
            if not arc_summary.empty:
                arc_summary.to_excel(writer, sheet_name="arc_summary", index=False)
            else:
                pd.DataFrame([{"note": "No arc data"}]).to_excel(
                    writer, sheet_name="arc_summary", index=False
                )

            # 7. KPIs (ENHANCED with zone/sort/distance/touch)
            if kpis is not None and not kpis.empty:
                kpis.to_frame("value").to_excel(writer, sheet_name="kpis")
            else:
                pd.Series([0], index=["total_cost"]).to_frame("value").to_excel(
                    writer, sheet_name="kpis"
                )

            # 8. Sort Analysis (optional)
            if sort_summary is not None and not sort_summary.empty:
                sort_summary.to_excel(writer, sheet_name="sort_analysis", index=False)

        return True

    except Exception as e:
        print(f"Error writing workbook {path}: {e}")
        return False


def write_comparison_workbook(
        path: Path,
        all_results: List[Dict],
        run_settings: Dict
) -> bool:
    """
    Write comparison workbook with ENHANCED metrics.

    Sheets:
    1. kpi_compare_long - All results in long format (ENHANCED)
    2. kpi_compare_wide - Pivoted by strategy (ENHANCED)
    3. zone_distribution - Zone mix across scenarios (NEW)
    4. sort_distribution - Sort level mix across scenarios (NEW)
    5. run_settings - Configuration used

    Args:
        path: Output file path
        all_results: List of result dictionaries with enhanced metrics
        run_settings: Run configuration

    Returns:
        True if successful, False otherwise
    """
    try:
        compare_df = pd.DataFrame(all_results)

        if compare_df.empty:
            print("No results to compare")
            return False

        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:

            # 1. Long format comparison (ENHANCED with new metrics)
            compare_df.to_excel(writer, sheet_name="kpi_compare_long", index=False)

            # 2. Wide format comparison (pivot by strategy)
            try:
                if 'strategy' in compare_df.columns and len(compare_df['strategy'].unique()) > 1:
                    # Core metrics to pivot
                    metrics_to_pivot = [
                        'total_cost', 'cost_per_pkg',
                        'avg_truck_fill_rate', 'avg_container_fill_rate',
                        'avg_zone_miles', 'avg_transit_miles',
                        'avg_total_touches', 'avg_hub_touches'
                    ]
                    available_metrics = [col for col in metrics_to_pivot if col in compare_df.columns]

                    if available_metrics:
                        pivot_index = 'scenario_id' if 'scenario_id' in compare_df.columns else compare_df.index

                        wide_df = compare_df.pivot_table(
                            index=pivot_index,
                            columns='strategy',
                            values=available_metrics,
                            aggfunc='first'
                        )
                        wide_df.to_excel(writer, sheet_name="kpi_compare_wide")
                    else:
                        compare_df.to_excel(writer, sheet_name="kpi_compare_wide", index=False)
                else:
                    compare_df.to_excel(writer, sheet_name="kpi_compare_wide", index=False)

            except Exception as e:
                print(f"Warning: Could not create wide format: {e}")
                compare_df.to_excel(writer, sheet_name="kpi_compare_wide", index=False)

            # 3. Zone Distribution (NEW)
            zone_cols = [col for col in compare_df.columns if col.startswith('zone_') and col.endswith('_pct')]
            if zone_cols:
                zone_df = compare_df[['scenario_id', 'strategy'] + zone_cols].copy()
                zone_df.to_excel(writer, sheet_name="zone_distribution", index=False)

            # 4. Sort Level Distribution (NEW)
            sort_cols = [col for col in compare_df.columns if
                         'sort' in col and ('pct_pkgs' in col or 'pct_dests' in col)]
            if sort_cols:
                sort_df = compare_df[['scenario_id', 'strategy'] + sort_cols].copy()
                sort_df.to_excel(writer, sheet_name="sort_distribution", index=False)

            # 5. Run Settings
            settings_df = pd.DataFrame([
                {"key": k, "value": v} for k, v in run_settings.items()
            ])
            settings_df.to_excel(writer, sheet_name="run_settings", index=False)

        return True

    except Exception as e:
        print(f"Error writing comparison workbook {path}: {e}")
        return False


def write_executive_summary(
        path: Path,
        all_results: List[Dict],
        run_settings: Dict
) -> bool:
    """
    Write executive summary with ENHANCED business insights.

    Sheets:
    1. Executive_Summary - Key takeaways (ENHANCED)
    2. Network_Efficiency - Distance and touch metrics (NEW)
    3. Zone_Analysis - Zone distribution analysis (NEW)
    4. Sort_Strategy - Sort level analysis (NEW)
    5. Cost_Analysis - Cost breakdown and drivers
    6. Network_Utilization - Fill rates and efficiency metrics

    Args:
        path: Output file path
        all_results: List of result dictionaries
        run_settings: Run configuration

    Returns:
        True if successful, False otherwise
    """
    try:
        if not all_results:
            return False

        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:

            # 1. Executive Summary (ENHANCED)
            summary_data = []
            for result in all_results:
                summary_data.append({
                    'Scenario': result.get('scenario_id', 'Unknown'),
                    'Strategy': result.get('strategy', 'Unknown'),
                    'Total Daily Cost': f"${result.get('total_cost', 0):,.0f}",
                    'Cost per Package': f"${result.get('cost_per_pkg', 0):.3f}",
                    'Avg Zone Miles': f"{result.get('avg_zone_miles', 0):.1f}",
                    'Avg Transit Miles': f"{result.get('avg_transit_miles', 0):.1f}",
                    'Avg Total Touches': f"{result.get('avg_total_touches', 0):.2f}",
                    'Avg Hub Touches': f"{result.get('avg_hub_touches', 0):.2f}",
                    'Truck Fill Rate': f"{result.get('avg_truck_fill_rate', 0):.1%}",
                })

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Executive_Summary", index=False)

            # 2. Network Efficiency (NEW)
            efficiency_data = []
            for result in all_results:
                efficiency_data.append({
                    'Scenario': result.get('scenario_id', 'Unknown'),
                    'Avg Zone Miles': result.get('avg_zone_miles', 0),
                    'Avg Transit Miles': result.get('avg_transit_miles', 0),
                    'Circuity Factor': result.get('avg_transit_miles', 0) / max(result.get('avg_zone_miles', 1), 1),
                    'Avg Total Touches': result.get('avg_total_touches', 0),
                    'Avg Hub Touches': result.get('avg_hub_touches', 0),
                })

            efficiency_df = pd.DataFrame(efficiency_data)
            efficiency_df.to_excel(writer, sheet_name="Network_Efficiency", index=False)

            # 3. Zone Analysis (NEW)
            zone_data = []
            for result in all_results:
                for zone_num in range(1, 9):
                    zone_pct = result.get(f'zone_{zone_num}_pct', 0)
                    if zone_pct > 0:
                        zone_data.append({
                            'Scenario': result.get('scenario_id', 'Unknown'),
                            'Zone': f'Zone {zone_num}',
                            'Percentage': zone_pct,
                            'Packages': result.get(f'zone_{zone_num}_pkgs', 0),
                        })

            if zone_data:
                zone_df = pd.DataFrame(zone_data)
                zone_df.to_excel(writer, sheet_name="Zone_Analysis", index=False)

            # 4. Sort Strategy (NEW)
            sort_data = []
            for result in all_results:
                for sort_level in ['region', 'market', 'sort_group']:
                    pct_pkgs = result.get(f'{sort_level}_pct_pkgs', 0)
                    pct_dests = result.get(f'{sort_level}_pct_dests', 0)

                    if pct_pkgs > 0 or pct_dests > 0:
                        sort_data.append({
                            'Scenario': result.get('scenario_id', 'Unknown'),
                            'Sort Level': sort_level.title(),
                            '% of Packages': pct_pkgs,
                            '% of Destinations': pct_dests,
                            'Packages': result.get(f'{sort_level}_pkgs', 0),
                        })

            if sort_data:
                sort_df = pd.DataFrame(sort_data)
                sort_df.to_excel(writer, sheet_name="Sort_Strategy", index=False)

            # 5. Cost Analysis
            cost_data = []
            for result in all_results:
                total_cost = result.get('total_cost', 0)

                cost_data.append({
                    'Scenario': result.get('scenario_id', 'Unknown'),
                    'Total Cost': total_cost,
                    'Cost per Package': result.get('cost_per_pkg', 0),
                    'Total Packages': result.get('total_packages', 0),
                })

            cost_df = pd.DataFrame(cost_data)
            cost_df.to_excel(writer, sheet_name="Cost_Analysis", index=False)

            # 6. Network Utilization
            util_data = []
            for result in all_results:
                util_data.append({
                    'Scenario': result.get('scenario_id', 'Unknown'),
                    'Truck Fill Rate': result.get('avg_truck_fill_rate', 0),
                    'Container Fill Rate': result.get('avg_container_fill_rate', 0),
                })

            util_df = pd.DataFrame(util_data)
            util_df.to_excel(writer, sheet_name="Network_Utilization", index=False)

        return True

    except Exception as e:
        print(f"Error writing executive summary {path}: {e}")
        return False