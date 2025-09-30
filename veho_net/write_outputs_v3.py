"""
Output Writing Module

Generates Excel workbooks with optimization results.
All sheets formatted for executive review and operational use.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict


def write_workbook(
        path: Path,
        scenario_summary: pd.DataFrame,
        od_selected: pd.DataFrame,
        path_steps: pd.DataFrame,
        facility_rollup: pd.DataFrame,
        arc_summary: pd.DataFrame,
        kpis: pd.Series,
        sort_summary: Optional[pd.DataFrame] = None
) -> bool:
    """
    Write main scenario output workbook.

    Sheets:
    1. scenario_summary - Key metadata (key-value format)
    2. od_selected_paths - Optimal OD paths with costs
    3. path_steps_selected - Leg-by-leg breakdown
    4. facility_rollup - Facility volume/cost by role
    5. arc_summary - Lane-level aggregations
    6. kpis - Network performance metrics
    7. sort_analysis - Sort level decisions (if provided)

    Args:
        path: Output file path
        scenario_summary: Scenario metadata
        od_selected: Selected OD paths
        path_steps: Path step details
        facility_rollup: Facility aggregations
        arc_summary: Arc/lane summary
        kpis: KPI metrics
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

            # 4. Facility Rollup
            if not facility_rollup.empty:
                facility_rollup.to_excel(writer, sheet_name="facility_rollup", index=False)
            else:
                pd.DataFrame([{"note": "No facility data"}]).to_excel(
                    writer, sheet_name="facility_rollup", index=False
                )

            # 5. Arc Summary
            if not arc_summary.empty:
                arc_summary.to_excel(writer, sheet_name="arc_summary", index=False)
            else:
                pd.DataFrame([{"note": "No arc data"}]).to_excel(
                    writer, sheet_name="arc_summary", index=False
                )

            # 6. KPIs
            if kpis is not None and not kpis.empty:
                kpis.to_frame("value").to_excel(writer, sheet_name="kpis")
            else:
                pd.Series([0], index=["total_cost"]).to_frame("value").to_excel(
                    writer, sheet_name="kpis"
                )

            # 7. Sort Analysis (optional)
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
    Write comparison workbook across scenarios/strategies.

    Sheets:
    1. kpi_compare_long - All results in long format
    2. kpi_compare_wide - Pivoted by strategy
    3. run_settings - Configuration used

    Args:
        path: Output file path
        all_results: List of result dictionaries
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

            # 1. Long format comparison
            compare_df.to_excel(writer, sheet_name="kpi_compare_long", index=False)

            # 2. Wide format comparison (pivot by strategy if applicable)
            try:
                if 'strategy' in compare_df.columns and len(compare_df['strategy'].unique()) > 1:
                    # Metrics to pivot
                    metrics_to_pivot = ['total_cost', 'cost_per_pkg', 'avg_truck_fill_rate']
                    available_metrics = [col for col in metrics_to_pivot if col in compare_df.columns]

                    if available_metrics:
                        # Determine pivot index
                        if 'scenario_id' in compare_df.columns:
                            pivot_index = 'scenario_id'
                        else:
                            pivot_index = compare_df.index

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

            # 3. Run settings
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
    Write executive summary with key business insights.

    Sheets:
    1. Executive_Summary - Key takeaways and recommendations
    2. Cost_Analysis - Cost breakdown and drivers
    3. Network_Utilization - Fill rates and efficiency metrics

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

            # 1. Executive Summary
            summary_data = []
            for result in all_results:
                summary_data.append({
                    'Scenario': result.get('scenario_id', 'Unknown'),
                    'Strategy': result.get('strategy', 'Unknown'),
                    'Total Daily Cost': f"${result.get('total_cost', 0):,.0f}",
                    'Cost per Package': f"${result.get('cost_per_pkg', 0):.3f}",
                    'Truck Fill Rate': f"{result.get('avg_truck_fill_rate', 0):.1%}",
                    'Container Fill Rate': f"{result.get('avg_container_fill_rate', 0):.1%}",
                })

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Executive_Summary", index=False)

            # 2. Cost Analysis
            cost_data = []
            for result in all_results:
                total_cost = result.get('total_cost', 0)
                total_pkgs = result.get('num_ods', 1) * 1000  # Approximate

                cost_data.append({
                    'Scenario': result.get('scenario_id', 'Unknown'),
                    'Total Cost': total_cost,
                    'Transportation Cost (est)': total_cost * 0.6,  # Approximate
                    'Processing Cost (est)': total_cost * 0.4,
                    'Cost per Package': result.get('cost_per_pkg', 0),
                })

            cost_df = pd.DataFrame(cost_data)
            cost_df.to_excel(writer, sheet_name="Cost_Analysis", index=False)

            # 3. Network Utilization
            util_data = []
            for result in all_results:
                util_data.append({
                    'Scenario': result.get('scenario_id', 'Unknown'),
                    'Truck Fill Rate': result.get('avg_truck_fill_rate', 0),
                    'Container Fill Rate': result.get('avg_container_fill_rate', 0),
                    'Packages Dwelled': result.get('total_packages_dwelled', 0),
                })

            util_df = pd.DataFrame(util_data)
            util_df.to_excel(writer, sheet_name="Network_Utilization", index=False)

        return True

    except Exception as e:
        print(f"Error writing executive summary {path}: {e}")
        return False