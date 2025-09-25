# test_strategy_costs.py - Quick test to verify strategy differentiation
import pandas as pd
from pathlib import Path
from veho_net.io_loader import load_workbook, params_to_dict
from veho_net.time_cost import path_cost_and_time
from veho_net.build_structures import build_od_and_direct, candidate_paths


def test_strategy_costs():
    """Test if strategy differentiation works in cost calculation"""

    # Load your input data
    input_path = Path("data/veho_model_input_v4.xlsx")
    dfs = load_workbook(input_path)

    timing_base = params_to_dict(dfs["timing_params"])
    costs_base = params_to_dict(dfs["cost_params"])

    print("=== LOADING ACTUAL DATA ===")

    # Get actual facilities and create real OD pairs
    facilities = dfs["facilities"]
    print(f"Available facilities: {facilities['facility_name'].head(10).tolist()}...")

    # Build actual OD matrix
    year_demand = dfs["demand"].query("year == 2028")
    od, _, _ = build_od_and_direct(facilities, dfs["zips"], year_demand, dfs["injection_distribution"])
    od["pkgs_day"] = od["pkgs_peak_day"]
    od = od[od["pkgs_day"] > 0].head(5)  # Take first 5 OD pairs for testing

    print(f"Testing with {len(od)} actual OD pairs")

    # Generate actual candidate paths
    around_factor = 1.5
    paths = candidate_paths(od, facilities, dfs["mileage_bands"], around_factor=around_factor)
    paths = paths.merge(od[['origin', 'dest', 'pkgs_day']], on=['origin', 'dest'], how='left')

    # Take the first multi-touch path for testing
    multi_touch_paths = paths[paths['path_type'] != 'direct']
    if multi_touch_paths.empty:
        print("No multi-touch paths found, using direct path")
        test_path = paths.iloc[0]
    else:
        test_path = multi_touch_paths.iloc[0]

    print(f"Testing path: {test_path['path_str']}")
    print(f"Path type: {test_path['path_type']}")
    print(f"Packages: {test_path['pkgs_day']}")

    test_packages = float(test_path['pkgs_day'])

    test_packages = 1000  # Test with 1000 packages

    # Test Container Strategy
    timing_container = timing_base.copy()
    timing_container['load_strategy'] = 'container'
    costs_container = costs_base.copy()
    costs_container['load_strategy'] = 'container'

    print("=== CONTAINER STRATEGY TEST ===")
    print(f"sort_cost_per_pkg: ${costs_container.get('sort_cost_per_pkg', 'MISSING')}")
    print(f"container_handling_cost: ${costs_container.get('container_handling_cost', 'MISSING')}")

    try:
        container_cost, _, _, _ = path_cost_and_time(
            test_path, dfs["facilities"], dfs["mileage_bands"],
            timing_container, costs_container,
            dfs["package_mix"], dfs["container_params"], test_packages
        )
        print(f"Container total cost: ${container_cost:.2f}")
        print(f"Container cost per package: ${container_cost / test_packages:.4f}")
    except Exception as e:
        print(f"Container calculation failed: {e}")
        return

    # Test Fluid Strategy
    timing_fluid = timing_base.copy()
    timing_fluid['load_strategy'] = 'fluid'
    costs_fluid = costs_base.copy()
    costs_fluid['load_strategy'] = 'fluid'

    print("\n=== FLUID STRATEGY TEST ===")
    print(f"sort_cost_per_pkg: ${costs_fluid.get('sort_cost_per_pkg', 'MISSING')}")
    print(f"container_handling_cost: ${costs_fluid.get('container_handling_cost', 'MISSING')}")

    try:
        fluid_cost, _, _, _ = path_cost_and_time(
            test_path, dfs["facilities"], dfs["mileage_bands"],
            timing_fluid, costs_fluid,
            dfs["package_mix"], dfs["container_params"], test_packages
        )
        print(f"Fluid total cost: ${fluid_cost:.2f}")
        print(f"Fluid cost per package: ${fluid_cost / test_packages:.4f}")
    except Exception as e:
        print(f"Fluid calculation failed: {e}")
        return

    # Compare Results
    print(f"\n=== COMPARISON ===")
    print(f"Cost difference: ${abs(container_cost - fluid_cost):.2f}")
    print(f"Relative difference: {abs(container_cost - fluid_cost) / min(container_cost, fluid_cost) * 100:.1f}%")

    if abs(container_cost - fluid_cost) < 0.01:
        print("❌ PROBLEM: Strategies produce identical costs!")
        print("This means the strategy differentiation logic is not working.")
    else:
        print("✅ SUCCESS: Strategies produce different costs!")
        print("The strategy differentiation logic is working.")

    # Test with extreme cost parameters to amplify differences
    print(f"\n=== EXTREME TEST (High Sort Cost) ===")
    costs_extreme = costs_base.copy()
    costs_extreme['sort_cost_per_pkg'] = 5.0  # Very high sort cost
    costs_extreme['container_handling_cost'] = 0.01  # Very low container handling

    # Container with extreme costs
    costs_container_extreme = costs_extreme.copy()
    costs_container_extreme['load_strategy'] = 'container'
    container_extreme_cost, _, _, _ = path_cost_and_time(
        test_path, dfs["facilities"], dfs["mileage_bands"],
        timing_container, costs_container_extreme,
        dfs["package_mix"], dfs["container_params"], test_packages
    )

    # Fluid with extreme costs
    costs_fluid_extreme = costs_extreme.copy()
    costs_fluid_extreme['load_strategy'] = 'fluid'
    fluid_extreme_cost, _, _, _ = path_cost_and_time(
        test_path, dfs["facilities"], dfs["mileage_bands"],
        timing_fluid, costs_fluid_extreme,
        dfs["package_mix"], dfs["container_params"], test_packages
    )

    print(f"Container extreme cost: ${container_extreme_cost:.2f}")
    print(f"Fluid extreme cost: ${fluid_extreme_cost:.2f}")
    print(f"Extreme cost difference: ${abs(container_extreme_cost - fluid_extreme_cost):.2f}")

    if abs(container_extreme_cost - fluid_extreme_cost) < 0.01:
        print("❌ CRITICAL: Even with extreme parameters, strategies are identical!")
    else:
        print("✅ Extreme test shows strategy differentiation works!")


if __name__ == "__main__":
    test_strategy_costs()