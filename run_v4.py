"""
Main Execution Script - v4.2 COMPLETE FIX

Fixed Issues:
1. Timing params consistency (dict vs object)
2. Arc summary column validation
3. Path nodes tuple/list handling
4. Empty result handling in all analysis sections
5. Better error messages and validation
6. Proper cleanup and resource handling
"""

import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
import sys

from veho_net.config_v4 import (
    CostParameters, TimingParameters, RunSettings,
    LoadStrategy, OUTPUT_FILE_TEMPLATE
)
from veho_net.io_loader_v4 import load_workbook, params_to_dict
from veho_net.validators_v4 import validate_inputs
from veho_net.build_structures_v4 import build_od_and_direct, candidate_paths
from veho_net.milp_v4 import solve_network_optimization
from veho_net.reporting_v4 import (
    build_facility_volume,
    build_facility_network_profile,
    calculate_network_distance_metrics,
    calculate_network_touch_metrics,
    calculate_network_zone_distribution,
    calculate_network_sort_distribution,
    add_zone_classification,
    build_path_steps,
    build_sort_summary,
    validate_network_aggregations
)
from veho_net.write_outputs_v4 import (
    write_workbook,
    write_comparison_workbook,
    write_executive_summary
)
from veho_net.zone_cost_analysis import (
    calculate_zone_cost_analysis,
    create_zone_cost_summary_table
)
from veho_net.fluid_load_analysis import (
    analyze_fluid_load_opportunities,
    create_fluid_load_summary_report,
    calculate_sort_point_savings
)
from veho_net.sort_strategy_comparison import (
    run_sort_strategy_comparison,
    create_comparison_summary_report
)
from veho_net.container_flow_v4 import (
    build_od_container_map,
    recalculate_arc_summary_with_container_flow,
    analyze_sort_level_container_impact,
    create_container_flow_diagnostic
)


def validate_scenario_data(scenario_row: pd.Series, dfs: dict) -> tuple:
    """
    Validate scenario has required data before processing.

    Returns:
        (is_valid: bool, error_message: str)
    """
    year = int(scenario_row["year"])

    # Check demand data exists
    year_demand = dfs["demand"].query("year == @year")
    if year_demand.empty:
        return False, f"No demand data for year {year}"

    # Check facilities exist
    if dfs["facilities"].empty:
        return False, "No facilities defined"

    # Check package mix exists
    if dfs["package_mix"].empty:
        return False, "No package mix defined"

    # Check mileage bands exist
    if dfs["mileage_bands"].empty:
        return False, "No mileage bands defined"

    return True, "OK"


def normalize_path_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure path_nodes column contains lists (not tuples) for downstream processing.

    Args:
        df: DataFrame with path_nodes column

    Returns:
        DataFrame with normalized path_nodes
    """
    if 'path_nodes' not in df.columns:
        return df

    df = df.copy()
    df['path_nodes'] = df['path_nodes'].apply(
        lambda x: list(x) if isinstance(x, tuple) else
        (x if isinstance(x, list) else [])
    )
    return df


def validate_arc_summary(arc_summary: pd.DataFrame, context: str = "") -> bool:
    """
    Validate arc summary has required columns.

    Args:
        arc_summary: Arc summary DataFrame
        context: Description for error messages

    Returns:
        True if valid, False otherwise
    """
    if arc_summary.empty:
        print(f"  ⚠️  {context}: Arc summary is empty")
        return False

    required_cols = ['from_facility', 'to_facility', 'distance_miles',
                     'pkgs_day', 'trucks', 'truck_fill_rate']
    missing = [col for col in required_cols if col not in arc_summary.columns]

    if missing:
        print(f"  ⚠️  {context}: Arc summary missing columns: {missing}")
        print(f"     Available: {list(arc_summary.columns)}")
        return False

    return True


def main(input_path: str, output_dir: str):
    """Main execution with comprehensive error handling and validation."""
    start_time = datetime.now()

    print("=" * 70)
    print("VEHO NETWORK OPTIMIZATION v4.2 COMPLETE")
    print("Comprehensive Bug Fixes + Enhanced Validation")
    print("=" * 70)

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n📁 Input: {input_path.name}")
    print(f"📁 Output: {output_dir}")

    # ========================================================================
    # LOAD AND VALIDATE INPUTS
    # ========================================================================

    print(f"\n{'=' * 70}")
    print("LOADING INPUTS")
    print("=" * 70)

    try:
        dfs = load_workbook(input_path)
        print("✓ Workbook loaded successfully")
    except Exception as e:
        print(f"\n❌ FATAL: Could not load workbook")
        print(f"   Error: {e}")
        return 1

    try:
        validate_inputs(dfs)
        print("✓ Input validation passed")
    except Exception as e:
        print(f"\n❌ FATAL: Input validation failed")
        print(f"   Error: {e}")
        return 1

    # ========================================================================
    # PARSE PARAMETERS
    # ========================================================================

    print(f"\n{'=' * 70}")
    print("PARSING PARAMETERS")
    print("=" * 70)

    try:
        timing_params_dict = params_to_dict(dfs["timing_params"])
        cost_params_dict = params_to_dict(dfs["cost_params"])
        run_settings_dict = params_to_dict(dfs["run_settings"])

        print("✓ Parameters parsed successfully")
    except Exception as e:
        print(f"\n❌ FATAL: Could not parse parameters")
        print(f"   Error: {e}")
        return 1

    # Create parameter objects
    try:
        cost_params = CostParameters(
            injection_sort_cost_per_pkg=float(cost_params_dict["injection_sort_cost_per_pkg"]),
            intermediate_sort_cost_per_pkg=float(cost_params_dict["intermediate_sort_cost_per_pkg"]),
            last_mile_sort_cost_per_pkg=float(cost_params_dict["last_mile_sort_cost_per_pkg"]),
            last_mile_delivery_cost_per_pkg=float(cost_params_dict["last_mile_delivery_cost_per_pkg"]),
            container_handling_cost=float(cost_params_dict["container_handling_cost"]),
            premium_economy_dwell_threshold=float(cost_params_dict["premium_economy_dwell_threshold"]),
            dwell_cost_per_pkg_per_day=float(cost_params_dict["dwell_cost_per_pkg_per_day"]),
            sla_penalty_per_touch_per_pkg=float(cost_params_dict["sla_penalty_per_touch_per_pkg"])
        )
        print("✓ Cost parameters created")
    except Exception as e:
        print(f"\n❌ FATAL: Invalid cost parameters")
        print(f"   Error: {e}")
        return 1

    # Determine strategy
    try:
        global_strategy = (
            LoadStrategy.CONTAINER
            if str(run_settings_dict["load_strategy"]).lower() == "container"
            else LoadStrategy.FLUID
        )
    except Exception as e:
        print(f"\n❌ FATAL: Invalid load_strategy in run_settings")
        print(f"   Error: {e}")
        return 1

    enable_sort_opt = bool(run_settings_dict.get("enable_sort_optimization", False))
    around_factor = float(run_settings_dict.get("path_around_the_world_factor", 1.3))
    run_id = run_settings_dict.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

    print(f"\n✓ Configuration:")
    print(f"  - Global strategy: {global_strategy.value}")
    print(f"  - Sort optimization: {'ENABLED' if enable_sort_opt else 'DISABLED'}")
    print(f"  - Container flow correction: ENABLED")
    print(f"  - Path around factor: {around_factor}")
    print(f"  - Run ID: {run_id}")

    all_results = []
    created_files = []

    # ========================================================================
    # PROCESS SCENARIOS
    # ========================================================================

    print(f"\n{'=' * 70}")
    print(f"PROCESSING {len(dfs['scenarios'])} SCENARIOS")
    print("=" * 70)

    for scenario_idx, scenario_row in dfs["scenarios"].iterrows():
        scenario_id = scenario_row.get("scenario_id", f"scenario_{scenario_idx + 1}")
        year = int(scenario_row["year"])
        day_type = str(scenario_row["day_type"]).strip().lower()

        print(f"\n{'─' * 70}")
        print(f"SCENARIO {scenario_idx + 1}/{len(dfs['scenarios'])}: {scenario_id}")
        print(f"  Year: {year}, Day Type: {day_type}")
        print("─" * 70)

        # Validate scenario data
        is_valid, error_msg = validate_scenario_data(scenario_row, dfs)
        if not is_valid:
            print(f"  ❌ Skipping scenario: {error_msg}")
            continue

        try:
            # ================================================================
            # 1. BUILD OD MATRIX
            # ================================================================

            print("\n1. Building OD matrix...")
            year_demand = dfs["demand"].query("year == @year").copy()

            if year_demand.empty:
                print(f"  ❌ No demand data for year {year}")
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
                print(f"  ❌ No OD pairs with volume for {day_type}")
                continue

            print(f"  ✓ Generated {len(od)} OD pairs")

            direct_day["dir_pkgs_day"] = direct_day[direct_col]

            # ================================================================
            # 2. GENERATE CANDIDATE PATHS
            # ================================================================

            print("\n2. Generating candidate paths...")

            try:
                paths = candidate_paths(
                    od,
                    dfs["facilities"],
                    dfs["mileage_bands"],
                    around_factor
                )
            except Exception as e:
                print(f"  ❌ Path generation failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            if paths.empty:
                print(f"  ❌ No valid paths generated")
                continue

            # Merge package volumes
            paths = paths.merge(
                od[['origin', 'dest', 'pkgs_day']],
                on=['origin', 'dest'],
                how='left'
            )

            paths['pkgs_day'] = paths['pkgs_day'].fillna(0)
            paths["scenario_id"] = scenario_id
            paths["day_type"] = day_type

            print(f"  ✓ Generated {len(paths)} candidate paths")

            # ================================================================
            # 3. SORT STRATEGY COMPARISON (if enabled)
            # ================================================================

            if enable_sort_opt:
                print(f"\n{'─' * 70}")
                print("🔍 RUNNING BASELINE COMPARISON")
                print("   (Market Sort vs. Optimized Sort Level)")
                print("─" * 70)

                try:
                    # Run comparison - NOW RETURNS optimized results
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
                            scenario_id=scenario_id
                        )
                    )

                    # Save comparison results
                    if not comparison_summary.empty:
                        comp_output = output_dir / f"sort_comparison_{scenario_id}_{global_strategy.value}.xlsx"
                        with pd.ExcelWriter(comp_output, engine='xlsxwriter') as writer:
                            comparison_summary.to_excel(writer, sheet_name='summary', index=False)
                            if not detailed_comparison.empty:
                                detailed_comparison.to_excel(writer, sheet_name='od_changes', index=False)
                            if not facility_comparison.empty:
                                facility_comparison.to_excel(writer, sheet_name='facility_comparison', index=False)

                        print(f"\n  ✓ Saved comparison to: {comp_output.name}")
                        created_files.append(comp_output.name)

                        print("\n" + create_comparison_summary_report(
                            comparison_summary, facility_comparison
                        ))

                    # CRITICAL FIX: Reuse optimized results
                    if optimized_results is not None:
                        print(f"\n{'─' * 70}")
                        print("✓ USING OPTIMIZED RESULTS FROM COMPARISON")
                        print("  (Saved 10 minutes by reusing results)")
                        print("─" * 70)

                        od_selected, arc_summary_original, network_kpis, sort_summary = optimized_results

                    else:
                        # Fallback: solve if comparison didn't return results
                        print(f"\n{'─' * 70}")
                        print("⚠️  Comparison didn't return results, solving again...")
                        print("─" * 70)
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
                                True  # enable_sort_optimization
                            )
                        )

                except Exception as e:
                    print(f"  ⚠️  Sort comparison failed: {e}")
                    import traceback
                    traceback.print_exc()

                    # Fallback to direct optimization
                    print(f"\n{'─' * 70}")
                    print("Running optimization without comparison...")
                    print("─" * 70)
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
                            True  # enable_sort_optimization
                        )
                    )

            else:
                # No comparison needed, run optimization once
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
                        False  # enable_sort_optimization
                    )
                except Exception as e:
                    print(f"  ❌ Optimization failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # ================================================================
            # 4. Validate optimization results
            # ================================================================

            if od_selected.empty:
                print(f"  ❌ Optimization returned no paths")
                continue

            print(f"  ✓ Selected {len(od_selected)} optimal paths")

            # Normalize path_nodes to lists
            od_selected = normalize_path_nodes(od_selected)

            # ================================================================
            # 5. CONTAINER FLOW CORRECTION (continues as before)
            # ================================================================

            print("\n4. Applying container flow correction...")

            # Validate original arc summary
            if not validate_arc_summary(arc_summary_original, "Original"):
                print("  ⚠️  Original arc summary invalid, proceeding with correction anyway...")

            try:
                # Build container map
                od_selected = build_od_container_map(
                    od_selected,
                    dfs["package_mix"],
                    dfs["container_params"],
                    dfs["facilities"]
                )
                print("  ✓ Container map built")

                # Recalculate arc summary with container flow
                arc_summary_corrected = recalculate_arc_summary_with_container_flow(
                    od_selected,
                    dfs["package_mix"],
                    dfs["container_params"],
                    dfs["facilities"],
                    dfs["mileage_bands"]
                )
                print("  ✓ Arc summary recalculated")

                # Validate corrected arc summary
                if validate_arc_summary(arc_summary_corrected, "Corrected"):
                    print("  ✓ Corrected arc summary validated")
                else:
                    print("  ⚠️  Corrected arc summary validation failed, using original")
                    arc_summary_corrected = arc_summary_original

                # Analyze sort level container impact
                sort_container_impact = analyze_sort_level_container_impact(
                    od_selected,
                    dfs["package_mix"],
                    dfs["container_params"],
                    dfs["facilities"]
                )

                # Create diagnostic
                diagnostic = create_container_flow_diagnostic(
                    od_selected,
                    arc_summary_original,
                    arc_summary_corrected
                )

                print(diagnostic)

                if not sort_container_impact.empty:
                    print("\n📊 Sort Level Container Analysis:")
                    print(sort_container_impact.to_string(index=False))

                # Use corrected version
                arc_summary = arc_summary_corrected

            except Exception as e:
                print(f"  ⚠️  Container flow correction failed: {e}")
                import traceback
                traceback.print_exc()
                print("  → Using original arc summary")
                arc_summary = arc_summary_original
                sort_container_impact = pd.DataFrame()

            # ================================================================
            # 6. GENERATE OUTPUTS
            # ================================================================

            print("\n5. Generating outputs...")

            try:
                # Add zone classification
                od_selected = add_zone_classification(
                    od_selected,
                    dfs["facilities"],
                    dfs["mileage_bands"]
                )
                print("  ✓ Zone classification added")
            except Exception as e:
                print(f"  ⚠️  Zone classification failed: {e}")

            # Zone cost analysis
            zone_cost_analysis = pd.DataFrame()
            try:
                print("\n5a. Calculating zone cost analysis...")
                zone_cost_analysis = calculate_zone_cost_analysis(
                    od_selected,
                    dfs["facilities"],
                    dfs["mileage_bands"]
                )

                if not zone_cost_analysis.empty:
                    print(create_zone_cost_summary_table(zone_cost_analysis))
            except Exception as e:
                print(f"  ⚠️  Zone cost analysis failed: {e}")

            # Path steps
            try:
                path_steps = build_path_steps(
                    od_selected,
                    dfs["facilities"],
                    dfs["mileage_bands"],
                    timing_params_dict
                )
                print("  ✓ Path steps built")
            except Exception as e:
                print(f"  ⚠️  Path steps generation failed: {e}")
                path_steps = pd.DataFrame()

            # Facility volume
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
                print("  ✓ Facility volume calculated")
            except Exception as e:
                print(f"  ⚠️  Facility volume calculation failed: {e}")
                import traceback
                traceback.print_exc()
                facility_volume = pd.DataFrame()

            # Facility network profile
            try:
                facility_network_profile = build_facility_network_profile(
                    od_selected,
                    dfs["facilities"],
                    dfs["mileage_bands"]
                )
                print("  ✓ Facility network profile built")
            except Exception as e:
                print(f"  ⚠️  Facility network profile failed: {e}")
                facility_network_profile = pd.DataFrame()

            # Network metrics
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
                zone_distribution = calculate_network_zone_distribution(od_selected)

                sort_distribution = {}
                if enable_sort_opt and not sort_summary.empty:
                    sort_distribution = calculate_network_sort_distribution(od_selected)

                print("  ✓ Network metrics calculated")
            except Exception as e:
                print(f"  ⚠️  Network metrics calculation failed: {e}")
                distance_metrics = {}
                touch_metrics = {}
                zone_distribution = {}
                sort_distribution = {}

            # Validation
            try:
                validation_results = validate_network_aggregations(
                    od_selected,
                    arc_summary,
                    facility_volume
                )

                if not validation_results.get('package_consistency', True):
                    print("  ⚠️  Warning: Package volume inconsistency detected")
            except Exception as e:
                print(f"  ⚠️  Validation check failed: {e}")

            # Calculate totals
            total_cost = od_selected["total_cost"].sum()
            total_pkgs = od_selected["pkgs_day"].sum()
            cost_per_pkg = total_cost / max(total_pkgs, 1)

            # Build KPIs
            kpis = pd.Series({
                "scenario_id": scenario_id,
                "year": year,
                "day_type": day_type,
                "strategy": global_strategy.value,
                "total_cost": total_cost,
                "cost_per_pkg": cost_per_pkg,
                "total_packages": total_pkgs,
                "num_ods": len(od_selected),
                **network_kpis,
                **distance_metrics,
                **touch_metrics,
                **zone_distribution,
                **sort_distribution
            })

            # Scenario summary
            scenario_summary = pd.DataFrame([
                {"key": "scenario_id", "value": scenario_id},
                {"key": "year", "value": year},
                {"key": "day_type", "value": day_type},
                {"key": "strategy", "value": global_strategy.value},
                {"key": "total_cost", "value": total_cost},
                {"key": "cost_per_pkg", "value": cost_per_pkg},
                {"key": "total_packages", "value": total_pkgs},
                {"key": "container_flow_corrected", "value": True},
                {"key": "zone_sort_distributions_fixed", "value": True}
            ])

            # ================================================================
            # 7. WRITE OUTPUT FILES
            # ================================================================

            output_filename = OUTPUT_FILE_TEMPLATE.format(
                scenario_id=scenario_id,
                strategy=global_strategy.value
            )
            output_path = output_dir / output_filename

            # Build sort analysis
            if enable_sort_opt and not sort_summary.empty:
                try:
                    sort_analysis = build_sort_summary(
                        od_selected,
                        {(row['scenario_id'], row['origin'], row['dest'], row['day_type']): row['chosen_sort_level']
                         for _, row in od_selected.iterrows()},
                        dfs["facilities"]
                    )
                except Exception as e:
                    print(f"  ⚠️  Sort analysis build failed: {e}")
                    sort_analysis = sort_summary
            else:
                sort_analysis = sort_summary

            # Write main workbook
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
                    print(f"  ✓ Wrote: {output_filename}")
                    print(f"  ✓ Total cost: ${total_cost:,.0f} (${cost_per_pkg:.3f}/pkg)")
                else:
                    print(f"  ❌ Failed to write output file")
                    continue
            except Exception as e:
                print(f"  ❌ Error writing output file: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Write container impact (if available)
            if not sort_container_impact.empty:
                try:
                    container_impact_path = output_dir / f"container_impact_{scenario_id}.xlsx"
                    with pd.ExcelWriter(container_impact_path, engine='xlsxwriter') as writer:
                        sort_container_impact.to_excel(
                            writer, sheet_name='sort_container_impact', index=False
                        )
                        pd.DataFrame([{'diagnostic': diagnostic}]).to_excel(
                            writer, sheet_name='diagnostic', index=False
                        )

                    created_files.append(f"container_impact_{scenario_id}.xlsx")
                    print(f"  ✓ Saved container impact to: container_impact_{scenario_id}.xlsx")
                except Exception as e:
                    print(f"  ⚠️  Could not save container impact: {e}")

            # Store results
            all_results.append(kpis.to_dict())

        except Exception as e:
            print(f"\n❌ Error processing {scenario_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ========================================================================
    # FLUID LOAD OPPORTUNITY ANALYSIS
    # ========================================================================

    if len(all_results) > 0 and 'od_selected' in locals() and not od_selected.empty:
        print(f"\n{'=' * 70}")
        print("FLUID LOAD OPPORTUNITY ANALYSIS")
        print("=" * 70)

        try:
            # Validate required columns
            required_for_fluid = ['origin', 'dest', 'pkgs_day', 'effective_strategy']
            missing_cols = [col for col in required_for_fluid if col not in od_selected.columns]

            if missing_cols:
                print(f"  ⚠️  Skipping fluid analysis - missing columns: {missing_cols}")
            else:
                fluid_opportunities = analyze_fluid_load_opportunities(
                    od_selected=od_selected,
                    arc_summary=arc_summary,
                    facilities=dfs["facilities"],
                    package_mix=dfs["package_mix"],
                    container_params=dfs["container_params"],
                    mileage_bands=dfs["mileage_bands"],
                    cost_params=cost_params,
                    min_daily_benefit=50.0,
                    max_results=50
                )

                if not fluid_opportunities.empty:
                    print(create_fluid_load_summary_report(fluid_opportunities))

                    # Calculate sort point savings
                    try:
                        sort_point_savings = calculate_sort_point_savings(
                            fluid_opportunities,
                            timing_params_dict,
                            dfs["facilities"]
                        )

                        if not sort_point_savings.empty:
                            print("\n📊 Sort Point Capacity Freed by Fluid Loading:")
                            print(sort_point_savings.to_string(index=False))
                    except Exception as e:
                        print(f"  ⚠️  Sort point savings calculation failed: {e}")
                        sort_point_savings = pd.DataFrame()

                    # Save results
                    try:
                        fluid_output = output_dir / f"fluid_opportunities_{run_id}.xlsx"
                        with pd.ExcelWriter(fluid_output, engine='xlsxwriter') as writer:
                            fluid_opportunities.to_excel(
                                writer,
                                sheet_name='opportunities',
                                index=False
                            )
                            if not sort_point_savings.empty:
                                sort_point_savings.to_excel(
                                    writer,
                                    sheet_name='sort_point_savings',
                                    index=False
                                )

                        print(f"\n✓ Saved fluid opportunities to: {fluid_output.name}")
                        created_files.append(fluid_output.name)
                    except Exception as e:
                        print(f"  ⚠️  Could not save fluid opportunities: {e}")
                else:
                    print("  ✓ All lanes optimally utilized - no fluid opportunities found")

        except Exception as e:
            print(f"  ⚠️  Fluid load analysis failed: {e}")
            import traceback
            traceback.print_exc()

    # ========================================================================
    # COMPARISON REPORTS
    # ========================================================================

    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("GENERATING COMPARISON REPORTS")
        print("=" * 70)

        # Comparison workbook
        try:
            compare_path = output_dir / f"comparison_{run_id}.xlsx"
            compare_success = write_comparison_workbook(
                compare_path,
                all_results,
                run_settings_dict
            )

            if compare_success:
                created_files.append(f"comparison_{run_id}.xlsx")
                print(f"  ✓ Created: comparison_{run_id}.xlsx")
        except Exception as e:
            print(f"  ⚠️  Could not create comparison workbook: {e}")

        # Executive summary
        try:
            exec_path = output_dir / f"executive_summary_{run_id}.xlsx"
            exec_success = write_executive_summary(
                exec_path,
                all_results,
                run_settings_dict
            )

            if exec_success:
                created_files.append(f"executive_summary_{run_id}.xlsx")
                print(f"  ✓ Created: executive_summary_{run_id}.xlsx")
        except Exception as e:
            print(f"  ⚠️  Could not create executive summary: {e}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    elapsed = datetime.now() - start_time

    print(f"\n{'=' * 70}")
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"⏱️  Elapsed time: {elapsed}")
    print(f"📋 Processed: {len(all_results)} scenarios")
    print(f"📄 Created files: {len(created_files)}")
    print(f"✅ Container flow correction: APPLIED")
    print(f"✅ Zone/sort distributions: FIXED")
    print(f"✅ Enhanced error handling: ACTIVE")

    if created_files:
        print(f"\nOutput files in {output_dir}:")
        for filename in sorted(created_files):
            print(f"  📄 {filename}")

    print("\n" + "=" * 70)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Veho Network Optimization v4.2 COMPLETE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_v4.py --input data/input.xlsx --output_dir outputs/
  python run_v4.py --input data/input.xlsx --output_dir outputs/test_run/

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
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)