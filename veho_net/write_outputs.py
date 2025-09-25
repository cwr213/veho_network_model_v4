import pandas as pd
import numpy as np
from pathlib import Path


def safe_sheet_name(name: str) -> str:
    """Create Excel-safe sheet names that are <= 31 characters."""
    invalid_chars = ['[', ']', ':', '*', '?', '/', '\\']
    clean_name = name
    for char in invalid_chars:
        clean_name = clean_name.replace(char, '_')
    return clean_name[:31]


def write_workbook(path, scenario_summary, od_out, path_detail, dwell_hotspots, facility_rollup, arc_summary, kpis,
                   sort_allocation_summary=None):
    """Write simplified workbook with core sheets only."""
    try:
        with pd.ExcelWriter(path, engine="xlsxwriter") as xw:

            # Core sheets matching expected output structure

            # 1. Scenario Summary (key-value format)
            if not scenario_summary.empty:
                scenario_summary.to_excel(xw, sheet_name="scenario_summary", index=False)
            else:
                pd.DataFrame([{"key": "scenario_id", "value": "unknown"}]).to_excel(
                    xw, sheet_name="scenario_summary", index=False)

            # 2. OD Selected Paths
            if not od_out.empty:
                od_out.to_excel(xw, sheet_name="od_selected_paths", index=False)
            else:
                pd.DataFrame([{"note": "No OD path data"}]).to_excel(
                    xw, sheet_name="od_selected_paths", index=False)

            # 3. Path Steps Selected
            if not path_detail.empty:
                path_detail.to_excel(xw, sheet_name="path_steps_selected", index=False)
            else:
                pd.DataFrame([{"note": "No path detail data"}]).to_excel(
                    xw, sheet_name="path_steps_selected", index=False)

            # 4. Dwell Hotspots
            if not dwell_hotspots.empty:
                dwell_hotspots.to_excel(xw, sheet_name="dwell_hotspots", index=False)
            else:
                pd.DataFrame([{"note": "No dwell hotspot data"}]).to_excel(
                    xw, sheet_name="dwell_hotspots", index=False)

            # 5. Facility Rollup
            if not facility_rollup.empty:
                facility_rollup.to_excel(xw, sheet_name="facility_rollup", index=False)
            else:
                pd.DataFrame([{"note": "No facility data"}]).to_excel(
                    xw, sheet_name="facility_rollup", index=False)

            # 6. Arc Summary
            if not arc_summary.empty:
                arc_summary.to_excel(xw, sheet_name="arc_summary", index=False)
            else:
                pd.DataFrame([{"note": "No arc/lane data"}]).to_excel(
                    xw, sheet_name="arc_summary", index=False)

            # 7. KPIs (key-value format)
            if kpis is not None and not kpis.empty:
                kpis.to_frame("value").to_excel(xw, sheet_name="kpis")
            else:
                pd.Series([0], index=["total_cost"]).to_frame("value").to_excel(
                    xw, sheet_name="kpis")

        return True

    except Exception as e:
        print(f"Error writing workbook {path}: {e}")
        return False


def write_compare_workbook(path, compare_df: pd.DataFrame, run_kv: dict):
    """Write comparison workbook with expected sheets."""
    try:
        # Ensure required columns exist
        required_cols = ['scenario_id', 'strategy', 'total_cost', 'cost_per_pkg']
        missing_cols = [col for col in required_cols if col not in compare_df.columns]
        if missing_cols:
            print(f"Missing columns in compare_df: {missing_cols}")
            return False

        with pd.ExcelWriter(path, engine="xlsxwriter") as xw:

            # Long format comparison
            compare_df.to_excel(xw, sheet_name="kpi_compare_long", index=False)

            # Wide format comparison (pivot by strategy)
            try:
                metrics_to_pivot = ['total_cost', 'cost_per_pkg']
                available_metrics = [col for col in metrics_to_pivot if col in compare_df.columns]

                if available_metrics and 'strategy' in compare_df.columns:
                    # Create identifier for pivoting
                    if 'scenario_id_from_input' in compare_df.columns:
                        pivot_id = compare_df['scenario_id_from_input']
                    elif 'base_id' in compare_df.columns:
                        pivot_id = compare_df['base_id']
                    else:
                        pivot_id = compare_df['scenario_id']

                    wide_df = compare_df.pivot_table(
                        index=pivot_id.name if hasattr(pivot_id, 'name') else 'id',
                        columns='strategy',
                        values=available_metrics,
                        aggfunc='first'
                    )
                    wide_df.to_excel(xw, sheet_name="kpi_compare_wide")
                else:
                    # Fallback: duplicate long format
                    compare_df.to_excel(xw, sheet_name="kpi_compare_wide", index=False)

            except Exception as e:
                print(f"Warning: Could not create wide format comparison: {e}")
                # Use long format as fallback
                compare_df.to_excel(xw, sheet_name="kpi_compare_wide", index=False)

            # Run settings
            settings = pd.DataFrame([{"key": k, "value": v} for k, v in run_kv.items()])
            settings.to_excel(xw, sheet_name="run_settings", index=False)

        return True

    except Exception as e:
        print(f"Error writing comparison workbook {path}: {e}")
        return False


def write_executive_summary_workbook(path, results_by_strategy: dict, run_kv: dict, base_id: str):
    """Write executive summary with key business insights."""
    try:
        with pd.ExcelWriter(path, engine="xlsxwriter") as xw:

            # Strategy Comparison
            strategy_comparison = create_strategy_comparison(results_by_strategy)
            strategy_comparison.to_excel(xw, sheet_name="Strategy_Comparison", index=False)

            # Hub Hourly Throughput
            hub_throughput = create_hub_throughput_analysis(results_by_strategy)
            if not hub_throughput.empty:
                hub_throughput.to_excel(xw, sheet_name="Hub_Hourly_Throughput", index=False)

            # Facility Truck Requirements
            truck_requirements = create_facility_truck_requirements(results_by_strategy)
            if not truck_requirements.empty:
                truck_requirements.to_excel(xw, sheet_name="Facility_Truck_Requirements", index=False)

            # Path Type Analysis
            path_analysis = create_path_type_analysis(results_by_strategy)
            if not path_analysis.empty:
                path_analysis.to_excel(xw, sheet_name="Path_Type_Analysis", index=False)

            # Key Answers
            key_answers = create_key_answers(results_by_strategy)
            key_answers.to_excel(xw, sheet_name="Key_Answers", index=False)

        return True

    except Exception as e:
        print(f"Error writing executive summary {path}: {e}")
        return False


def create_strategy_comparison(results_by_strategy: dict) -> pd.DataFrame:
    """Create strategy comparison table."""
    try:
        comparison_data = []

        for strategy, data in results_by_strategy.items():
            kpis = data.get('kpis', pd.Series())

            comparison_data.append({
                'strategy': strategy,
                'total_cost': kpis.get('total_cost', 0),
                'cost_per_pkg': kpis.get('cost_per_pkg', 0),
                'num_ods': kpis.get('num_ods', 0),
                'avg_truck_fill_rate': kpis.get('avg_truck_fill_rate', 0),
                'avg_container_fill_rate': kpis.get('avg_container_fill_rate', 0),
                'pct_direct': kpis.get('pct_direct', 0),
                'pct_1_touch': kpis.get('pct_1_touch', 0),
                'pct_2_touch': kpis.get('pct_2_touch', 0)
            })

        return pd.DataFrame(comparison_data)

    except Exception as e:
        return pd.DataFrame([{"error": f"Could not create strategy comparison: {e}"}])


def create_hub_throughput_analysis(results_by_strategy: dict) -> pd.DataFrame:
    """Create hub throughput analysis."""
    try:
        # Get container strategy facility data
        container_data = results_by_strategy.get('container', {})
        facility_rollup = container_data.get('facility_rollup', pd.DataFrame())

        if facility_rollup.empty:
            return pd.DataFrame()

        # Filter to hubs and include throughput metrics
        hub_data = facility_rollup[
            facility_rollup.get('type', '').isin(['hub', 'hybrid'])
        ].copy()

        if hub_data.empty:
            return pd.DataFrame()

        throughput_cols = ['facility', 'peak_hourly_throughput', 'injection_pkgs_day', 'last_mile_pkgs_day']
        available_cols = [col for col in throughput_cols if col in hub_data.columns]

        return hub_data[available_cols]

    except Exception:
        return pd.DataFrame()


def create_facility_truck_requirements(results_by_strategy: dict) -> pd.DataFrame:
    """Create facility truck requirements analysis."""
    try:
        truck_data = []

        for strategy, data in results_by_strategy.items():
            facility_rollup = data.get('facility_rollup', pd.DataFrame())

            if not facility_rollup.empty:
                for _, facility in facility_rollup.iterrows():
                    truck_data.append({
                        'facility': facility.get('facility', 'unknown'),
                        'strategy': strategy,
                        'peak_throughput': facility.get('peak_hourly_throughput', 0),
                        'estimated_trucks_needed': max(1, int(facility.get('peak_hourly_throughput', 0) / 100))
                    })

        return pd.DataFrame(truck_data)

    except Exception:
        return pd.DataFrame()


def create_path_type_analysis(results_by_strategy: dict) -> pd.DataFrame:
    """Create path type analysis."""
    try:
        path_analysis = []

        for strategy, data in results_by_strategy.items():
            kpis = data.get('kpis', pd.Series())

            path_analysis.append({
                'strategy': strategy,
                'pct_direct': kpis.get('pct_direct', 0),
                'pct_1_touch': kpis.get('pct_1_touch', 0),
                'pct_2_touch': kpis.get('pct_2_touch', 0),
                'pct_3_touch': kpis.get('pct_3_touch', 0)
            })

        return pd.DataFrame(path_analysis)

    except Exception:
        return pd.DataFrame()


def create_key_answers(results_by_strategy: dict) -> pd.DataFrame:
    """Create key answers."""
    try:
        container_kpis = results_by_strategy.get('container', {}).get('kpis', pd.Series())
        fluid_kpis = results_by_strategy.get('fluid', {}).get('kpis', pd.Series())

        container_cost = container_kpis.get('total_cost', 0)
        fluid_cost = fluid_kpis.get('total_cost', 0)

        optimal_strategy = 'container' if container_cost <= fluid_cost else 'fluid'
        cost_difference = abs(container_cost - fluid_cost)

        key_answers = pd.DataFrame([
            {
                'question': '1. Optimal Strategy',
                'answer': f'{optimal_strategy.upper()} strategy recommended',
                'detail': f'Daily cost difference: ${cost_difference:,.0f}',
                'metric': f'Container: ${container_cost:,.0f}, Fluid: ${fluid_cost:,.0f}'
            },
            {
                'question': '2. Network Cost',
                'answer': f'Total daily cost: ${min(container_cost, fluid_cost):,.0f}',
                'detail': f'Cost per package: ${min(container_kpis.get("cost_per_pkg", 0), fluid_kpis.get("cost_per_pkg", 0)):.3f}',
                'metric': f'Serving {max(container_kpis.get("num_ods", 0), fluid_kpis.get("num_ods", 0))} OD pairs'
            },
            {
                'question': '3. Fill Rates',
                'answer': f'Truck utilization: {container_kpis.get("avg_truck_fill_rate", 0):.1%} (container), {fluid_kpis.get("avg_truck_fill_rate", 0):.1%} (fluid)',
                'detail': 'Both strategies show reasonable utilization',
                'metric': 'Target: >75% truck fill rate'
            }
        ])

        return key_answers

    except Exception as e:
        return pd.DataFrame([{"question": "Error", "answer": "Could not generate answers", "detail": str(e)}])


def write_consolidated_multi_year_workbook(path: Path, all_results: list, run_kv: dict):
    """Write consolidated workbook - simplified version."""
    try:
        consolidated_data = {}

        # Just collect key data for now
        for result in all_results:
            scenario_id = result.get('scenario_id', 'unknown')
            consolidated_data[scenario_id] = {
                'total_cost': result.get('total_cost', 0),
                'strategy': result.get('strategy', 'unknown')
            }

        summary_df = pd.DataFrame([
            {'scenario_id': k, **v} for k, v in consolidated_data.items()
        ])

        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            summary_df.to_excel(writer, sheet_name="Consolidated_Summary", index=False)

        return True

    except Exception:
        return False