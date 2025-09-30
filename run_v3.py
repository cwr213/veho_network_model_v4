"""
Main Execution Script for Network Optimization v3
"""

import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

from veho_net.config_v3 import (
    CostParameters, TimingParameters, RunSettings,
    LoadStrategy, OUTPUT_FILE_TEMPLATE
)
from veho_net.io_loader_v3 import load_workbook, params_to_dict
from veho_net.validators_v3 import validate_inputs
from veho_net.build_structures_v3 import build_od_and_direct, candidate_paths
from veho_net.milp_v3 import solve_network_optimization
from veho_net.reporting_v3 import (
    build_facility_rollup,
    calculate_hourly_throughput,
    add_zone_classification,
    build_path_steps,
    build_sort_summary,
    validate_network_aggregations
)
from veho_net.write_outputs_v3 import (
    write_workbook,
    write_comparison_workbook,
    write_executive_summary
)


def main(input_path: str, output_dir: str):
    """Main execution function for network optimization."""
    start_time = datetime.now()

    print("=" * 70)
    print("VEHO NETWORK OPTIMIZATION v3")
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

    print(f"✓ Global strategy: {global_strategy.value}")
    print(f"✓ Sort optimization: {'ENABLED' if enable_sort_opt else 'DISABLED'}")
    print(f"✓ Path around factor: {around_factor}")

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

            path_steps = build_path_steps(
                od_selected,
                dfs["facilities"],
                dfs["mileage_bands"],
                timing_params_dict
            )

            facility_rollup = build_facility_rollup(
                od_selected,
                direct_day,
                arc_summary,
                dfs["package_mix"],
                dfs["container_params"],
                global_strategy.value,
                dfs["facilities"],
                cost_params
            )

            facility_rollup = calculate_hourly_throughput(
                facility_rollup,
                timing_params_dict
            )

            validation_results = validate_network_aggregations(
                od_selected,
                arc_summary,
                facility_rollup
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
                **network_kpis
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
                facility_rollup,
                arc_summary,
                kpis,
                sort_analysis
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

    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("GENERATING COMPARISON REPORTS")
        print("=" * 70)

        compare_path = output_dir / "comparison.xlsx"
        compare_success = write_comparison_workbook(
            compare_path,
            all_results,
            run_settings_dict
        )

        if compare_success:
            created_files.append("comparison.xlsx")
            print(f"  ✓ Created: comparison.xlsx")

        exec_path = output_dir / "executive_summary.xlsx"
        exec_success = write_executive_summary(
            exec_path,
            all_results,
            run_settings_dict
        )

        if exec_success:
            created_files.append("executive_summary.xlsx")
            print(f"  ✓ Created: executive_summary.xlsx")

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
        description="Veho Network Optimization v3 - Consolidated Model"
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