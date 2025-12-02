"""
Main Execution Script

Orchestrates end-to-end network optimization:
- Loads and validates input data
- Generates candidate paths through network
- Solves MILP optimization per scenario
- Produces facility-level and network-level outputs
- Supports sort strategy comparison and fluid load analysis
"""

import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
import sys

from src.config_v4 import (
    CostParameters, TimingParameters,
    LoadStrategy, OUTPUT_FILE_TEMPLATE
)
from src.io_loader_v4 import load_workbook, params_to_dict
from src.validators_v4 import validate_inputs
from src.build_structures_v4 import build_od_and_direct, candidate_paths
from src.milp_v4 import solve_network_optimization
from src.reporting_v4 import (
    build_facility_volume,
    build_facility_network_profile,
    calculate_network_distance_metrics,
    calculate_network_touch_metrics,
    calculate_network_zone_distribution,
    calculate_network_sort_distribution,
    add_zone_classification,
    add_direct_injection_zone_classification,
    add_zone_miles_to_od_selected,
    build_path_steps,
    validate_network_aggregations
)
from src.write_outputs_v4 import (
    write_workbook,
    write_comparison_workbook,
    write_executive_summary
)
from src.zone_cost_analysis import (
    calculate_zone_cost_analysis,
    create_zone_cost_summary_table
)
from src.fluid_load_analysis import (
    analyze_fluid_load_opportunities,
    create_fluid_load_summary_report
)
from src.sort_strategy_comparison import (
    run_sort_strategy_comparison,
    create_comparison_summary_report
)
from src.container_flow_v4 import (
    build_od_container_map,
    build_od_container_map_with_persistence,
    recalculate_arc_summary_with_container_flow,
    analyze_sort_level_container_impact,
    create_container_flow_diagnostic
)


def generate_run_id(scenarios_df: pd.DataFrame) -> str:
    """Generate descriptive run_id from scenario characteristics."""

    years = scenarios_df['year'].unique()

    if 'max_sort_points_capacity' in scenarios_df.columns:
        capacities = scenarios_df['max_sort_points_capacity'].dropna()
        if len(capacities) > 0:
            cap_values = sorted(capacities.unique())
            if len(cap_values) == 1:
                cap_suffix = f"_cap{int(cap_values[0])}"
            else:
                cap_suffix = f"_cap{int(min(cap_values))}-{int(max(cap_values))}"
        else:
            cap_suffix = ""
    else:
        cap_suffix = ""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    year_str = f"{min(years)}" if len(years) == 1 else f"{min(years)}-{max(years)}"

    return f"{year_str}_{timestamp}{cap_suffix}"


def validate_scenario_data(scenario_row: pd.Series, dfs: dict) -> tuple:
    """
    Validate scenario has required data before processing.
    """
    year = int(scenario_row["year"])

    year_demand = dfs["demand"].query("year == @year")
    if year_demand.empty:
        return False, f"No demand data for year {year}"

    if dfs["facilities"].empty:
        return False, "No facilities defined"

    if dfs["package_mix"].empty:
        return False, "No package mix defined"

    if dfs["mileage_bands"].empty:
        return False, "No mileage bands defined"

    return True, "OK"


def validate_arc_summary(arc_summary: pd.DataFrame, context: str = "") -> bool:
    """
    Validate arc summary has required columns.
    """
    if arc_summary.empty:
        print(f"  WARNING:  {context}: Arc summary is empty")
        return False

    required_cols = ['from_facility', 'to_facility', 'distance_miles',
                     'pkgs_day', 'trucks', 'truck_fill_rate']
    missing = [col for col in required_cols if col not in arc_summary.columns]

    if missing:
        print(f"  WARNING:  {context}: Arc summary missing columns: {missing}")
        print(f"     Available: {list(arc_summary.columns)}")
        return False

    return True


def main(input_path: str, output_dir: str):
    """Main execution with comprehensive error handling and validation."""
    start_time = datetime.now()

    print("=" * 70)
    print("VEHO NETWORK OPTIMIZATION")
    print("=" * 70)

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n Input: {input_path.name}")
    print(f" Output: {output_dir}")

    print(f"\n{'=' * 70}")
    print("LOADING INPUTS")
    print("=" * 70)

    try:
        dfs = load_workbook(input_path)
        print("Workbook loaded successfully")
    except Exception as e:
        print(f"\n ERROR FATAL: Could not load workbook")
        print(f"   Error: {e}")
        return 1

    try:
        validate_inputs(dfs)
        print("Input validation passed")
    except Exception as e:
        print(f"\n ERROR FATAL: Input validation failed")
        print(f"   Error: {e}")
        return 1

    print(f"\n{'=' * 70}")
    print("PARSING PARAMETERS")
    print("=" * 70)

    try:
        timing_params_dict = params_to_dict(dfs["timing_params"])
        cost_params_dict = params_to_dict(dfs["cost_params"])
        run_settings_dict = params_to_dict(dfs["run_settings"])

        print("Parameters parsed successfully")
    except Exception as e:
        print(f"\n ERROR FATAL: Could not parse parameters")
        print(f"   Error: {e}")
        return 1

    try:
        cost_params = CostParameters(
            injection_sort_cost_per_pkg=float(cost_params_dict["injection_sort_cost_per_pkg"]),
            intermediate_sort_cost_per_pkg=float(cost_params_dict["intermediate_sort_cost_per_pkg"]),
            last_mile_sort_cost_per_pkg=float(cost_params_dict["last_mile_sort_cost_per_pkg"]),
            last_mile_delivery_cost_per_pkg=float(cost_params_dict["last_mile_delivery_cost_per_pkg"]),
            container_handling_cost=float(cost_params_dict["container_handling_cost"]),
        )
        print("Cost parameters created")
    except Exception as e:
        print(f"\n ERROR FATAL: Invalid cost parameters")
        print(f"   Error: {e}")
        return 1

    # Always use container strategy as baseline
    # Fluid opportunities are identified post-optimization via fluid_load_analysis
    global_strategy = LoadStrategy.CONTAINER

    enable_sort_opt = bool(run_settings_dict.get("enable_sort_optimization", False))

    user_run_id = run_settings_dict.get("run_id", None)
    if user_run_id and str(user_run_id).strip():
        run_id = str(user_run_id).strip()
        print(f"\nUsing provided run_id: {run_id}")
    else:
        run_id = generate_run_id(dfs["scenarios"])
        print(f"\nAuto-generated run_id: {run_id}")

    print(f"\nConfiguration:")
    print(f"  Strategy: Container (fluid opportunities analyzed post-optimization)")
    print(f"  Sort optimization: {'ENABLED' if enable_sort_opt else 'DISABLED'}")
    print(f"  Run ID: {run_id}")

    all_results = []
    created_files = []
    scenario_data_store = {}

    print(f"\n{'=' * 70}")
    print(f"PROCESSING {len(dfs['scenarios'])} SCENARIOS")
    print("=" * 70)

    for scenario_idx, scenario_row in dfs["scenarios"].iterrows():
        scenario_id = scenario_row.get("scenario_id", f"scenario_{scenario_idx + 1}")
        year = int(scenario_row["year"])
        day_type = str(scenario_row["day_type"]).strip().lower()

        print(f"\n{'-' * 70}")
        print(f"SCENARIO {scenario_idx + 1}/{len(dfs['scenarios'])}: {scenario_id}")
        print(f"  Year: {year}, Day Type: {day_type}")

        if 'max_sort_points_capacity' in scenario_row.index and pd.notna(scenario_row['max_sort_points_capacity']):
            capacity_override = int(scenario_row['max_sort_points_capacity'])
            print(f"  Capacity Override: {capacity_override} sort points")
        else:
            print(f"  Capacity: Using facilities defaults")

        print("-" * 70)

        is_valid, error_msg = validate_scenario_data(scenario_row, dfs)
        if not is_valid:
            print(f"  ERROR: Skipping scenario: {error_msg}")
            continue

        try:
            print("\n1. Building OD matrix...")
            year_demand = dfs["demand"].query("year == @year").copy()

            if year_demand.empty:
                print(f"  ERROR: No demand data for year {year}")
                continue

            od, direct_day, dest_pop = build_od_and_direct(
                dfs["facilities"],
                dfs["zips"],
                year_demand,
                dfs["injection_distribution"]
            )

            od_col = "pkgs_offpeak_day" if day_type == "offpeak" else "pkgs_peak_day"
            direct_col = "dir_pkgs_offpeak_day" if day_type == "offpeak" else "dir_pkgs_peak_day"

            od["pkgs_day"] = od[od_col]
            od = od[od["pkgs_day"] > 0].copy()

            if od.empty:
                print(f"  ERROR: No OD pairs with volume for {day_type}")
                continue

            print(f"  Generated {len(od)} OD pairs")

            direct_day["dir_pkgs_day"] = direct_day[direct_col]

            direct_day = add_direct_injection_zone_classification(direct_day)
            direct_pkgs_total = direct_day["dir_pkgs_day"].sum()
            print(f"  Direct injection (Zone 0): {direct_pkgs_total:,.0f} packages")

            print("\n2. Generating candidate paths...")

            try:
                around_factor = float(run_settings_dict["path_around_the_world_factor"])

                paths = candidate_paths(
                    od,
                    dfs["facilities"],
                    around_factor
                )
            except Exception as e:
                print(f"  ERROR: Path generation failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            if paths.empty:
                print(f"  ERROR: No valid paths generated")
                continue

            paths = paths.merge(
                od[['origin', 'dest', 'pkgs_day']],
                on=['origin', 'dest'],
                how='left'
            )

            paths['pkgs_day'] = paths['pkgs_day'].fillna(0)
            paths["scenario_id"] = scenario_id
            paths["day_type"] = day_type

            print(f"  Generated {len(paths)} candidate paths")

            if enable_sort_opt:
                print(f"\n{'-' * 70}")
                print(" RUNNING BASELINE COMPARISON")
                print("   (Market Sort vs. Optimized Sort Level)")
                print("-" * 70)

                try:
                    comparison_summary, detailed_comparison, facility_comparison, optimized_results = (
                        run_sort_strategy_comparison(
                            candidates=paths,
                            facilities=dfs["facilities"],
                            mileage_bands=dfs["mileage_bands"],
                            package_mix=dfs["package_mix"],
                            container_params=dfs["container_params"],
                            cost_params=cost_params,
                            timing_params=timing_params_dict,
                            global_strategy=global_strategy,
                            scenario_id=scenario_id,
                            scenario_row=scenario_row
                        )
                    )

                    if not comparison_summary.empty:
                        comp_output = output_dir / f"sort_comparison_{scenario_id}_{global_strategy.value}.xlsx"
                        with pd.ExcelWriter(comp_output, engine='xlsxwriter') as writer:
                            comparison_summary.to_excel(writer, sheet_name='summary', index=False)
                            if not detailed_comparison.empty:
                                detailed_comparison.to_excel(writer, sheet_name='od_changes', index=False)
                            if not facility_comparison.empty:
                                facility_comparison.to_excel(writer, sheet_name='facility_comparison', index=False)

                        print(f"\n  Saved: {comp_output.name}")
                        created_files.append(comp_output.name)

                        print("\n" + create_comparison_summary_report(
                            comparison_summary, facility_comparison
                        ))

                    if optimized_results is not None:
                        print(f"\nUsing optimized results from comparison")

                        od_selected, arc_summary_original, network_kpis, sort_summary = optimized_results

                    else:
                        print(f"\nWARNING: Comparison didn't return results, solving again...")
                        print("\n3. Running MILP optimization...")

                        od_selected, arc_summary_original, network_kpis, sort_summary = (
                            solve_network_optimization(
                                paths,
                                dfs["facilities"],
                                dfs["mileage_bands"],
                                dfs["package_mix"],
                                dfs["container_params"],
                                cost_params,
                                timing_params_dict,
                                global_strategy,
                                True,
                                scenario_row
                            )
                        )

                except Exception as e:
                    print(f"  WARNING:  Sort comparison failed: {e}")
                    import traceback
                    traceback.print_exc()

                    print(f"\n{'─' * 70}")
                    print("Running optimization without comparison...")
                    print("-" * 70)
                    print("\n3. Running MILP optimization...")

                    od_selected, arc_summary_original, network_kpis, sort_summary = (
                        solve_network_optimization(
                            paths,
                            dfs["facilities"],
                            dfs["mileage_bands"],
                            dfs["package_mix"],
                            dfs["container_params"],
                            cost_params,
                            timing_params_dict,
                            global_strategy,
                            True,
                            scenario_row
                        )
                    )

            else:
                print("\n3. Running MILP optimization...")

                try:
                    od_selected, arc_summary_original, network_kpis, sort_summary = solve_network_optimization(
                        paths,
                        dfs["facilities"],
                        dfs["mileage_bands"],
                        dfs["package_mix"],
                        dfs["container_params"],
                        cost_params,
                        timing_params_dict,
                        global_strategy,
                        False,
                        scenario_row
                    )
                except Exception as e:
                    print(f"  ERROR: Optimization failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            if od_selected.empty:
                print(f"  ERROR: Optimization returned no paths")
                continue

            print(f"  Selected {len(od_selected)} optimal paths")

            # path_nodes already validated as lists from candidate_paths()

            print("\n4. Adding zone classification...")

            od_selected = add_zone_classification(
                od_selected,
                dfs["facilities"],
                dfs["mileage_bands"]
            )

            od_selected = add_zone_miles_to_od_selected(
                od_selected,
                dfs["facilities"]
            )

            unknown_zones = od_selected[od_selected['zone'] == -1]
            if not unknown_zones.empty:
                unknown_pkgs = unknown_zones['pkgs_day'].sum()
                unknown_pct = (unknown_pkgs / od_selected['pkgs_day'].sum()) * 100
                print(f"  WARNING: {unknown_pct:.1f}% of packages in unknown zone")
                print(f"     {len(unknown_zones)} OD pairs affected")

                print(f"\n     Example unknown zone ODs:")
                for _, row in unknown_zones.head(5).iterrows():
                    print(f"     {row['origin']} -> {row['dest']}: {row['zone_miles']:.1f} miles")
            else:
                print(f"  All OD pairs successfully classified")

            print("\n5. Building container flow tracking (with persistence)...")

            if not validate_arc_summary(arc_summary_original, "Original"):
                print("  WARNING:  Original arc summary invalid, proceeding with correction anyway...")

            try:
                od_selected = build_od_container_map_with_persistence(
                    od_selected,
                    dfs["package_mix"],
                    dfs["container_params"],
                    dfs["facilities"]
                )
                print("  Container map built with persistence tracking")

                arc_summary_corrected = recalculate_arc_summary_with_container_flow(
                    od_selected,
                    dfs["package_mix"],
                    dfs["container_params"],
                    dfs["facilities"],
                    dfs["mileage_bands"]
                )
                print("  Arc summary recalculated")

                if validate_arc_summary(arc_summary_corrected, "Corrected"):
                    print("  Corrected arc summary validated")
                    # Log persistence statistics
                    if 'persisted_containers' in arc_summary_corrected.columns:
                        total_cont = arc_summary_corrected['physical_containers'].sum()
                        persisted = arc_summary_corrected['persisted_containers'].sum()
                        persist_pct = (persisted / total_cont * 100) if total_cont > 0 else 0
                        print(
                            f"  Container persistence: {persist_pct:.1f}% of containers persisted through crossdock")
                else:
                    print("  WARNING:  Corrected arc summary validation failed, using original")
                    arc_summary_corrected = arc_summary_original

                sort_container_impact = analyze_sort_level_container_impact(
                    od_selected,
                    dfs["package_mix"],
                    dfs["container_params"],
                    dfs["facilities"]
                )

                diagnostic = create_container_flow_diagnostic(
                    od_selected,
                    arc_summary_original,
                    arc_summary_corrected
                )

                print(diagnostic)

                if not sort_container_impact.empty:
                    print("\n Sort Level Container Analysis:")
                    print(sort_container_impact.to_string(index=False))

                arc_summary = arc_summary_corrected

            except Exception as e:
                print(f"  WARNING:  Container flow correction failed: {e}")
                import traceback
                traceback.print_exc()
                print("  Using original arc summary")
                arc_summary = arc_summary_original
                sort_container_impact = pd.DataFrame()

            print("\n6. Generating outputs...")

            zone_cost_analysis = pd.DataFrame()
            try:
                print("\n6a. Calculating zone cost analysis (including Zone 0)...")
                zone_cost_analysis = calculate_zone_cost_analysis(
                    od_selected,
                    dfs["facilities"],
                    dfs["mileage_bands"],
                    direct_day
                )

                if not zone_cost_analysis.empty:
                    print(create_zone_cost_summary_table(zone_cost_analysis))
            except Exception as e:
                print(f"  WARNING:  Zone cost analysis failed: {e}")

            try:
                path_steps = build_path_steps(
                    od_selected,
                    dfs["facilities"],
                    dfs["mileage_bands"],
                )
                print("  Path steps built")
            except Exception as e:
                print(f"  WARNING:  Path steps generation failed: {e}")
                path_steps = pd.DataFrame()

            try:
                facility_volume = build_facility_volume(
                    od_selected,
                    direct_day,
                    arc_summary,
                    dfs["package_mix"],
                    dfs["container_params"],
                    global_strategy.value,
                    dfs["facilities"],
                    timing_params_dict
                )
                print("  Facility volume calculated")
            except Exception as e:
                print(f"  WARNING:  Facility volume calculation failed: {e}")
                import traceback
                traceback.print_exc()
                facility_volume = pd.DataFrame()

            try:
                facility_network_profile = build_facility_network_profile(
                    od_selected,
                    dfs["facilities"],
                    dfs["mileage_bands"],
                    direct_day
                )
                print("  Facility network profile built")
            except Exception as e:
                print(f"  WARNING:  Facility network profile failed: {e}")
                facility_network_profile = pd.DataFrame()

            try:
                distance_metrics = calculate_network_distance_metrics(
                    od_selected,
                    dfs["facilities"],
                    dfs["mileage_bands"]
                )
                touch_metrics = calculate_network_touch_metrics(
                    od_selected,
                    dfs["facilities"]
                )

                print("  Network metrics calculated")
            except Exception as e:
                print(f"  WARNING: Network metrics calculation failed - {e}")
                distance_metrics = {}
                touch_metrics = {}

            try:
                validation_results = validate_network_aggregations(
                    od_selected,
                    arc_summary,
                    facility_volume
                )

                if not validation_results.get('package_consistency', True):
                    print("  WARNING:  Warning: Package volume inconsistency detected")

                if validation_results.get('unknown_zone_pct', 0) > 0:
                    print(f"  WARNING:  {validation_results['unknown_zone_pct']:.1f}% of packages in unknown zone")

            except Exception as e:
                print(f"  WARNING:  Validation check failed: {e}")

            total_cost = od_selected["total_cost"].sum()
            total_pkgs = od_selected["pkgs_day"].sum()
            cost_per_pkg = total_cost / max(total_pkgs, 1)

            kpis = pd.Series({
                "scenario_id": scenario_id,
                "year": year,
                "day_type": day_type,
                "strategy": global_strategy.value,
                "total_cost": total_cost,
                "cost_per_pkg": cost_per_pkg,
                "total_packages": total_pkgs,
                "direct_injection_packages": direct_pkgs_total,
                "middle_mile_packages": total_pkgs,
                "num_ods": len(od_selected),
                **network_kpis,
                **distance_metrics,
                **touch_metrics
            })

            scenario_summary = pd.DataFrame([
                {"key": "scenario_id", "value": scenario_id},
                {"key": "year", "value": year},
                {"key": "day_type", "value": day_type},
                {"key": "strategy", "value": global_strategy.value},
                {"key": "total_cost", "value": total_cost},
                {"key": "cost_per_pkg", "value": cost_per_pkg},
                {"key": "middle_mile_packages", "value": total_pkgs},
                {"key": "direct_injection_packages", "value": direct_pkgs_total},
                {"key": "total_packages", "value": total_pkgs + direct_pkgs_total},
                {"key": "container_flow_corrected", "value": True},
                {"key": "zone_tracking", "value": "0-8 + Unknown"},
                {"key": "zone_miles_validation", "value": True}
            ])

            if 'max_sort_points_capacity' in scenario_row.index and pd.notna(scenario_row['max_sort_points_capacity']):
                capacity_used = int(scenario_row['max_sort_points_capacity'])
                scenario_summary = pd.concat([
                    scenario_summary,
                    pd.DataFrame([{"key": "max_sort_capacity_override", "value": capacity_used}])
                ], ignore_index=True)

            output_filename = OUTPUT_FILE_TEMPLATE.format(
                scenario_id=scenario_id,
                strategy=global_strategy.value
            )
            output_path = output_dir / output_filename

            sort_analysis = sort_summary

            try:
                write_success = write_workbook(
                    output_path,
                    scenario_summary,
                    od_selected,
                    path_steps,
                    facility_volume,
                    facility_network_profile,
                    arc_summary,
                    kpis,
                    sort_analysis,
                    zone_cost_analysis
                )

                if write_success:
                    created_files.append(output_filename)
                    print(f"  Wrote: {output_filename}")
                    print(f"  Total cost: ${total_cost:,.0f} (${cost_per_pkg:.3f}/pkg)")
                else:
                    print(f"  ERROR: Failed to write output file")
                    continue
            except Exception as e:
                print(f"  ERROR: Error writing output file: {e}")
                import traceback
                traceback.print_exc()
                continue

            if not sort_container_impact.empty:
                try:
                    container_impact_path = output_dir / f"container_impact_{scenario_id}.xlsx"
                    with pd.ExcelWriter(container_impact_path, engine='xlsxwriter') as writer:
                        # Sort container impact analysis
                        sort_container_impact.to_excel(
                            writer, sheet_name='sort_container_impact', index=False
                        )

                        # Container flow diagnostic as structured data
                        diagnostic_data = []

                        # Network fill rates
                        orig_fill = arc_summary_original[
                            'truck_fill_rate'].mean() if 'truck_fill_rate' in arc_summary_original.columns else 0
                        corr_fill = arc_summary['truck_fill_rate'].mean()

                        diagnostic_data.append({
                            'metric': 'Network Avg Truck Fill (Original)',
                            'value': round(orig_fill, 4)
                        })
                        diagnostic_data.append({
                            'metric': 'Network Avg Truck Fill (Corrected)',
                            'value': round(corr_fill, 4)
                        })
                        diagnostic_data.append({
                            'metric': 'Fill Rate Difference',
                            'value': round(corr_fill - orig_fill, 4)
                        })

                        # Sort level impacts
                        for sort_level in ['region', 'market', 'sort_group']:
                            level_ods = od_selected[od_selected.get('chosen_sort_level', 'market') == sort_level]
                            if not level_ods.empty:
                                total_pkgs = level_ods['pkgs_day'].sum()
                                total_containers = level_ods[
                                    'origin_containers'].sum() if 'origin_containers' in level_ods.columns else 0

                                diagnostic_data.append({
                                    'metric': f'{sort_level.title()} - Packages',
                                    'value': int(total_pkgs)
                                })
                                diagnostic_data.append({
                                    'metric': f'{sort_level.title()} - Containers',
                                    'value': int(total_containers)
                                })
                                diagnostic_data.append({
                                    'metric': f'{sort_level.title()} - Pkgs per Container',
                                    'value': round(total_pkgs / max(total_containers, 1), 1)
                                })

                        diagnostic_df = pd.DataFrame(diagnostic_data)
                        diagnostic_df.to_excel(
                            writer, sheet_name='diagnostic', index=False
                        )

                    created_files.append(f"container_impact_{scenario_id}.xlsx")
                    print(f"  Saved container impact to: container_impact_{scenario_id}.xlsx")
                except Exception as e:
                    print(f"  WARNING: Could not save container impact - {e}")
                    import traceback
                    traceback.print_exc()

            print("\n7. Running fluid load analysis for this scenario...")

            try:
                required_for_fluid = ['pkgs_day', 'chosen_sort_level', 'path_nodes']
                missing_cols = [col for col in required_for_fluid if col not in od_selected.columns]

                if missing_cols:
                    print(f"  WARNING:  Skipping fluid analysis - missing columns: {missing_cols}")
                else:
                    fluid_fill_threshold = float(run_settings_dict.get("fluid_opportunity_fill_threshold", 0.75))

                    fluid_opportunities = analyze_fluid_load_opportunities(
                        od_selected=od_selected,
                        arc_summary=arc_summary,
                        facilities=dfs["facilities"],
                        package_mix=dfs["package_mix"],
                        container_params=dfs["container_params"],
                        mileage_bands=dfs["mileage_bands"],
                        cost_params=cost_params,
                        fluid_fill_threshold=fluid_fill_threshold
                    )

                    if not fluid_opportunities.empty:
                        print(create_fluid_load_summary_report(fluid_opportunities))

                        try:
                            fluid_output = output_dir / f"fluid_opportunities_{scenario_id}.xlsx"
                            with pd.ExcelWriter(fluid_output, engine='xlsxwriter') as writer:
                                # Detailed opportunities
                                fluid_opportunities.to_excel(
                                    writer,
                                    sheet_name='opportunities',
                                    index=False
                                )

                                # Summary metrics
                                fluid_summary = pd.DataFrame([{
                                    'metric': 'Total Arcs Analyzed',
                                    'value': len(fluid_opportunities)
                                }, {
                                    'metric': 'Total Daily Savings Potential',
                                    'value': fluid_opportunities['net_benefit_daily'].sum()
                                }, {
                                    'metric': 'Total Annual Savings Potential',
                                    'value': fluid_opportunities['annual_benefit'].sum()
                                }, {
                                    'metric': 'Total Trucks Saved Per Day',
                                    'value': fluid_opportunities['trucks_saved'].sum()
                                }, {
                                    'metric': 'Avg Savings Per Arc',
                                    'value': fluid_opportunities['net_benefit_daily'].mean()
                                }])

                                fluid_summary.to_excel(
                                    writer,
                                    sheet_name='summary',
                                    index=False
                                )

                            print(f"\n  Saved fluid opportunities to: fluid_opportunities_{scenario_id}.xlsx")
                            print(f"    ({len(fluid_opportunities)} arcs analyzed)")
                            created_files.append(f"fluid_opportunities_{scenario_id}.xlsx")
                        except Exception as e:
                            print(f"  WARNING: Could not save fluid opportunities - {e}")
                    else:
                        print("All arcs optimally utilized - no fluid opportunities found")

            except Exception as e:
                print(f"  WARNING:  Fluid load analysis failed: {e}")
                import traceback
                traceback.print_exc()

            all_results.append(kpis.to_dict())

            scenario_data_store[scenario_id] = {
                'od_selected': od_selected.copy(),
                'arc_summary': arc_summary.copy(),
                'kpis': kpis.to_dict()
            }

        except Exception as e:
            print(f"\n ERROR: Error processing {scenario_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("GENERATING COMPARISON REPORTS")
        print("=" * 70)

        try:
            compare_path = output_dir / f"comparison_{run_id}.xlsx"
            compare_success = write_comparison_workbook(
                compare_path,
                all_results,
                run_settings_dict
            )

            if compare_success:
                created_files.append(f"comparison_{run_id}.xlsx")
                print(f"  Created: comparison_{run_id}.xlsx")
        except Exception as e:
            print(f"  WARNING:  Could not create comparison workbook: {e}")

        try:
            exec_path = output_dir / f"executive_summary_{run_id}.xlsx"
            exec_success = write_executive_summary(
                exec_path,
                all_results,
                run_settings_dict
            )

            if exec_success:
                created_files.append(f"executive_summary_{run_id}.xlsx")
                print(f"  Created: executive_summary_{run_id}.xlsx")
        except Exception as e:
            print(f"  WARNING:  Could not create executive summary: {e}")

    elapsed = datetime.now() - start_time

    print(f"\n{'=' * 70}")
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Elapsed time: {elapsed}")
    print(f"Processed: {len(all_results)} scenarios")
    print(f"Created: {len(created_files)} output files")

    if created_files:
        print(f"\nOutput files in {output_dir}:")
        for filename in sorted(created_files):
            print(f"  📄 {filename}")

    print("=" * 70)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Veho Network Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_v4.py --input data/input.xlsx --output_dir outputs/
  python run_v4.py --input data/input.xlsx --output_dir outputs/test_run/

Zone Classification:
  - Zone 0: Direct injection (no middle-mile transport)
  - Zone 1-8: Distance-based from mileage_bands (includes O=D)
  - Unknown: Classification failed (data quality flag)

Scenario Capacity Override:
  - Add optional max_sort_points_capacity column to scenarios sheet
  - Overrides facilities.max_sort_points_capacity for that scenario
  - Leave blank to use facility defaults

Smart run_id Generation:
  - Leave run_id blank in run_settings for auto-generation
  - Auto format: {year}_{timestamp}_cap{min}-{max}
  - Or provide custom run_id to override

Fluid Load Analysis:
  - Arc-based: Analyzes actual planned arcs only
  - Per-scenario: Each scenario gets its own analysis
  - No consolidation assumptions: Direct container vs fluid comparison
  - Returns ALL opportunities: Filter/sort in Excel as needed

For input file validation, run the diagnostic script first:
  python diagnostic_runner.py data/input.xlsx
        """
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input Excel file"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to output directory"
    )

    args = parser.parse_args()

    try:
        exit_code = main(args.input, args.output_dir)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nWARNING:  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: FATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)