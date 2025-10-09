"""
Main Execution Script for Network Optimization v4
Enhanced with automatic baseline comparison and zone cost analysis
"""

import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

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


def main(input_path: str, output_dir: str):
    """Main execution function for network optimization."""
    start_time = datetime.now()

    print("=" * 70)
    print("VEHO NETWORK OPTIMIZATION v4")
    print("=" * 70)

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n📁 Input: {input_path.name}")
    print(f"📁 Output: {output_dir}")

    print(f"\n{'=' * 70}")
    print("LOADING INPUTS")
    print("=" * 70)

    dfs = load_workbook(input_path)
    validate_inputs(dfs)

    print(f"\n{'=' * 70}")
    print("PARSING PARAMETERS")
    print("=" * 70)

    timing_params_dict = params_to_dict(dfs["timing_params"])
    cost_params_dict = params_to_dict(dfs["cost_params"])
    run_settings_dict = params_to_dict(dfs["run_settings"])

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

    global_strategy = (
        LoadStrategy.CONTAINER
        if str(run_settings_dict["load_strategy"]).lower() == "container"
        else LoadStrategy.FLUID
    )

    enable_sort_opt = bool(run_settings_dict.get("enable_sort_optimization", False))
    around_factor = float(run_settings_dict["path_around_the_world_factor"])
    run_id = run_settings_dict.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

    print(f"✓ Global strategy: {global_strategy.value}")
    print(f"✓ Sort optimization: {'ENABLED' if enable_sort_opt else 'DISABLED'}")
    print(f"✓ Path around factor: {around_factor}")
    print(f"✓ Run ID: {run_id}")

    if enable_sort_opt:
        print("\n✅ Sort level optimization (region/market/sort_group) ENABLED")
        print("   Model will optimize sort level per OD pair with capacity constraints")

    print(f"\n{'=' * 70}")
    print(f"PROCESSING {len(dfs['scenarios'])} SCENARIOS")
    print("=" * 70)

    all_results = []
    created_files = []

    for scenario_idx, scenario_row in dfs["scenarios"].iterrows():
        scenario_id = scenario_row.get("scenario_id", f"scenario_{scenario_idx + 1}")
        year = int(scenario_row["year"])
        day_type = str(scenario_row["day_type"]).strip().lower()

        print(f"\n{'─' * 70}")
        print(f"SCENARIO {scenario_idx + 1}/{len(dfs['scenarios'])}: {scenario_id}")
        print(f"  Year: {year}, Day Type: {day_type}")
        print("─" * 70)

        try:
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

            print("\n2. Generating candidate paths...")
            paths = candidate_paths(
                od,
                dfs["facilities"],
                dfs["mileage_bands"],
                around_factor
            )

            if paths.empty:
                print(f"  ❌ No valid paths generated")
                continue

            paths = paths.merge(
                od[['origin', 'dest', 'pkgs_day']],
                on=['origin', 'dest'],
                how='left'
            )

            paths['pkgs_day'] = paths['pkgs_day'].fillna(0)
            paths["scenario_id"] = scenario_id
            paths["day_type"] = day_type

            print(f"  ✓ Generated {len(paths)} candidate paths")

            # ========== BASELINE VS OPTIMIZED COMPARISON ==========
            # If sort optimization is enabled, run baseline comparison automatically
            if enable_sort_opt:
                print(f"\n{'─' * 70}")
                print("🔍 RUNNING BASELINE COMPARISON")
                print("   (Market Sort vs. Optimized Sort Level)")
                print("─" * 70)

                comparison_summary, detailed_comparison, facility_comparison = (
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

                if not comparison_summary.empty:
                    # Save comparison results
                    comp_output = output_dir / f"sort_comparison_{scenario_id}_{global_strategy.value}.xlsx"
                    with pd.ExcelWriter(comp_output, engine='xlsxwriter') as writer:
                        comparison_summary.to_excel(
                            writer, sheet_name='summary', index=False
                        )
                        if not detailed_comparison.empty:
                            detailed_comparison.to_excel(
                                writer, sheet_name='od_changes', index=False
                            )
                        if not facility_comparison.empty:
                            facility_comparison.to_excel(
                                writer, sheet_name='facility_comparison', index=False
                            )

                    print(f"\n  ✓ Saved comparison to: {comp_output.name}")
                    created_files.append(comp_output.name)

                    # Print summary
                    print("\n" + create_comparison_summary_report(
                        comparison_summary, facility_comparison
                    ))

                print(f"\n{'─' * 70}")
                print("CONTINUING WITH OPTIMIZED RUN")
                print("─" * 70)
            # ======================================================

            print("\n3. Running MILP optimization...")
            od_selected, arc_summary, network_kpis, sort_summary = solve_network_optimization(
                paths,
                dfs["facilities"],
                dfs["mileage_bands"],
                dfs["package_mix"],
                dfs["container_params"],
                cost_params,
                timing_params_dict,
                global_strategy,
                enable_sort_opt
            )

            if od_selected.empty:
                print(f"  ❌ Optimization failed")
                continue

            print(f"  ✓ Selected {len(od_selected)} optimal paths")

            print("\n4. Generating outputs...")

            od_selected = add_zone_classification(
                od_selected,
                dfs["facilities"],
                dfs["mileage_bands"]
            )

            # ========== ZONE COST ANALYSIS ==========
            print("\n4a. Calculating zone cost analysis...")
            zone_cost_analysis = calculate_zone_cost_analysis(
                od_selected,
                dfs["facilities"],
                dfs["mileage_bands"]
            )

            if not zone_cost_analysis.empty:
                print(create_zone_cost_summary_table(zone_cost_analysis))
            else:
                print("  ⚠️  No zone data to analyze")
            # ========================================

            path_steps = build_path_steps(
                od_selected,
                dfs["facilities"],
                dfs["mileage_bands"],
                timing_params_dict
            )

            # Build facility volume (operational metrics)
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

            # Build facility network profile (zone/sort/distance/touch)
            facility_network_profile = build_facility_network_profile(
                od_selected,
                dfs["facilities"],
                dfs["mileage_bands"]
            )

            # Calculate network-level metrics
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

            validation_results = validate_network_aggregations(
                od_selected,
                arc_summary,
                facility_volume
            )

            if not validation_results.get('package_consistency', True):
                print("  ⚠️  Warning: Package volume inconsistency detected")

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
                "num_ods": len(od_selected),
                **network_kpis,
                **distance_metrics,
                **touch_metrics,
                **zone_distribution,
                **sort_distribution
            })

            scenario_summary = pd.DataFrame([
                {"key": "scenario_id", "value": scenario_id},
                {"key": "year", "value": year},
                {"key": "day_type", "value": day_type},
                {"key": "strategy", "value": global_strategy.value},
                {"key": "total_cost", "value": total_cost},
                {"key": "cost_per_pkg", "value": cost_per_pkg},
                {"key": "total_packages", "value": total_pkgs}
            ])

            output_filename = OUTPUT_FILE_TEMPLATE.format(
                scenario_id=scenario_id,
                strategy=global_strategy.value
            )
            output_path = output_dir / output_filename

            if enable_sort_opt and not sort_summary.empty:
                sort_analysis = build_sort_summary(
                    od_selected,
                    {(row['scenario_id'], row['origin'], row['dest'], row['day_type']): row['chosen_sort_level']
                     for _, row in od_selected.iterrows()},
                    dfs["facilities"]
                )
            else:
                sort_analysis = sort_summary

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
                zone_cost_analysis  # NEW: Pass zone cost analysis
            )

            if write_success:
                created_files.append(output_filename)
                print(f"  ✓ Wrote: {output_filename}")
                print(f"  ✓ Total cost: ${total_cost:,.0f} (${cost_per_pkg:.3f}/pkg)")
            else:
                print(f"  ❌ Failed to write output file")
                continue

            all_results.append(kpis.to_dict())

        except Exception as e:
            print(f"\n❌ Error processing {scenario_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ========== FLUID LOAD OPPORTUNITY ANALYSIS ==========
    # Run after all scenarios complete to analyze optimization results
    if len(all_results) > 0 and 'od_selected' in locals() and not od_selected.empty:
        print(f"\n{'=' * 70}")
        print("FLUID LOAD OPPORTUNITY ANALYSIS")
        print("=" * 70)

        try:
            fluid_opportunities = analyze_fluid_load_opportunities(
                od_selected=od_selected,
                arc_summary=arc_summary,
                facilities=dfs["facilities"],
                package_mix=dfs["package_mix"],
                container_params=dfs["container_params"],
                mileage_bands=dfs["mileage_bands"],
                cost_params=cost_params,
                min_daily_benefit=50.0,  # $50/day minimum
                max_results=50
            )

            if not fluid_opportunities.empty:
                print(create_fluid_load_summary_report(fluid_opportunities))

                # Calculate sort point savings
                sort_point_savings = calculate_sort_point_savings(
                    fluid_opportunities,
                    timing_params_dict,
                    dfs["facilities"]
                )

                if not sort_point_savings.empty:
                    print("\n📊 Sort Point Capacity Freed by Fluid Loading:")
                    print(sort_point_savings.to_string(index=False))

                # Save fluid opportunities to file
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
            else:
                print("  ✓ All lanes optimally utilized - no fluid opportunities found")
        except Exception as e:
            print(f"  ⚠️  Fluid load analysis failed: {e}")
    # =====================================================

    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("GENERATING COMPARISON REPORTS")
        print("=" * 70)

        compare_path = output_dir / f"comparison_{run_id}.xlsx"
        compare_success = write_comparison_workbook(
            compare_path,
            all_results,
            run_settings_dict
        )

        if compare_success:
            created_files.append(f"comparison_{run_id}.xlsx")
            print(f"  ✓ Created: comparison_{run_id}.xlsx")

        exec_path = output_dir / f"executive_summary_{run_id}.xlsx"
        exec_success = write_executive_summary(
            exec_path,
            all_results,
            run_settings_dict
        )

        if exec_success:
            created_files.append(f"executive_summary_{run_id}.xlsx")
            print(f"  ✓ Created: executive_summary_{run_id}.xlsx")

    elapsed = datetime.now() - start_time

    print(f"\n{'=' * 70}")
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"⏱️  Elapsed time: {elapsed}")
    print(f"📋 Processed: {len(all_results)} scenarios")
    print(f"📄 Created files: {len(created_files)}")

    if created_files:
        print(f"\nOutput files in {output_dir}:")
        for filename in sorted(created_files):
            print(f"  📄 {filename}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Veho Network Optimization v4 - Enhanced with Baseline Comparison"
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
        main(args.input, args.output_dir)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        exit(1)