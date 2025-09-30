"""
Container and Truck Capacity Calculations

All calculations use input parameters - NO HARDCODED VALUES.
Handles both container and fluid loading strategies with proper fill rate calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def weighted_pkg_cube(package_mix: pd.DataFrame) -> float:
    """
    Calculate weighted average cube per package across package mix.

    Args:
        package_mix: DataFrame with columns:
            - share_of_pkgs (must sum to 1.0)
            - avg_cube_cuft

    Returns:
        Weighted average cubic feet per package
    """
    return float((package_mix["share_of_pkgs"] * package_mix["avg_cube_cuft"]).sum())


def calculate_truck_capacity(
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        strategy: str
) -> float:
    """
    Calculate packages per truck capacity based on loading strategy.

    Container strategy:
        capacity = ((usable_cube × pack_util_container) ÷ weighted_pkg_cube) × containers_per_truck

    Fluid strategy:
        capacity = (trailer_cube × pack_util_fluid) ÷ weighted_pkg_cube

    Args:
        package_mix: Package distribution data
        container_params: Container/trailer capacity parameters
        strategy: "container" or "fluid"

    Returns:
        Packages per truck capacity (effective capacity with utilization)
    """
    weighted_avg_pkg_cube = weighted_pkg_cube(package_mix)

    if strategy.lower() == "container":
        # Container strategy: packages → gaylords → trucks
        gaylord_row = container_params[
            container_params["container_type"].str.lower() == "gaylord"
            ].iloc[0]

        usable_cube = float(gaylord_row["usable_cube_cuft"])
        pack_util = float(gaylord_row["pack_utilization_container"])
        containers_per_truck_val = int(gaylord_row["containers_per_truck"])

        effective_container_cube = usable_cube * pack_util
        packages_per_truck = (effective_container_cube / weighted_avg_pkg_cube) * containers_per_truck_val

    else:  # fluid
        # Fluid strategy: packages → trucks directly
        trailer_cube = float(container_params["trailer_air_cube_cuft"].iloc[0])
        pack_util = float(container_params["pack_utilization_fluid"].iloc[0])

        effective_trailer_cube = trailer_cube * pack_util
        packages_per_truck = effective_trailer_cube / weighted_avg_pkg_cube

    return packages_per_truck


def calculate_trucks_and_fill_rates(
        total_packages: float,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        strategy: str,
        dwell_threshold: float
) -> Dict[str, float]:
    """
    Calculate truck requirements and fill rates with premium economy dwell logic.

    Premium Economy Dwell Logic:
    - Calculate exact trucks needed based on effective cube capacity
    - If fractional truck < dwell_threshold: round down, dwell excess packages
    - Always use minimum 1 truck (never round to 0)

    Fill Rate Calculation:
    - Uses RAW capacity (not effective) per executive reporting standard
    - Measures actual cube utilization against theoretical maximum

    Args:
        total_packages: Total package volume
        package_mix: Package distribution
        container_params: Container/trailer parameters
        strategy: Loading strategy
        dwell_threshold: Fractional truck threshold from cost_params

    Returns:
        Dict with:
            - physical_containers: Container count (container strategy only)
            - trucks_needed: Number of trucks required
            - container_fill_rate: Container utilization (0-1)
            - truck_fill_rate: Truck utilization (0-1)
            - packages_dwelled: Packages delayed to next day
            - total_cube_cuft: Total package cube
            - cube_per_truck: Average cube per truck
    """
    if total_packages <= 0:
        return {
            'physical_containers': 0,
            'trucks_needed': 1,
            'container_fill_rate': 0.0,
            'truck_fill_rate': 0.0,
            'packages_dwelled': 0,
            'total_cube_cuft': 0.0,
            'cube_per_truck': 0.0
        }

    # Calculate total cube
    weighted_cube = weighted_pkg_cube(package_mix)
    total_cube = total_packages * weighted_cube

    # Get raw capacities for fill rate calculations
    raw_trailer_cube = float(container_params["trailer_air_cube_cuft"].iloc[0])

    # Get truck capacity from parameters
    packages_per_truck_capacity = calculate_truck_capacity(package_mix, container_params, strategy)

    if strategy.lower() == "container":
        # Container strategy: packages → gaylords → trucks
        gaylord_row = container_params[
            container_params["container_type"].str.lower() == "gaylord"
            ].iloc[0]

        raw_container_cube = float(gaylord_row["usable_cube_cuft"])
        pack_util_container = float(gaylord_row["pack_utilization_container"])
        effective_container_cube = raw_container_cube * pack_util_container
        containers_per_truck_val = int(gaylord_row["containers_per_truck"])

        # Calculate containers needed (based on effective capacity)
        exact_containers = total_cube / effective_container_cube
        physical_containers = max(1, int(np.ceil(exact_containers)))

        # Calculate trucks needed
        raw_trucks = total_packages / packages_per_truck_capacity

        # Apply dwell logic
        if raw_trucks <= 1.0:
            final_trucks = 1
            packages_dwelled = 0
        else:
            fractional_part = raw_trucks - int(raw_trucks)
            if fractional_part < dwell_threshold:
                # Round down, dwell excess
                final_trucks = int(raw_trucks)
                missing_capacity = (raw_trucks - final_trucks) * packages_per_truck_capacity
                packages_dwelled = missing_capacity
            else:
                # Add the partial truck
                final_trucks = int(np.ceil(raw_trucks))
                packages_dwelled = 0

        # Fill rates use raw capacities
        container_fill_rate = min(1.0, total_cube / (physical_containers * raw_container_cube))
        truck_fill_rate = min(1.0, total_cube / (final_trucks * raw_trailer_cube))

    else:  # fluid
        # Fluid strategy: packages → trucks directly
        raw_trucks = total_packages / packages_per_truck_capacity

        # Apply dwell logic
        if raw_trucks <= 1.0:
            final_trucks = 1
            packages_dwelled = 0
        else:
            fractional_part = raw_trucks - int(raw_trucks)
            if fractional_part < dwell_threshold:
                # Round down, dwell excess
                final_trucks = int(raw_trucks)
                missing_capacity = (raw_trucks - final_trucks) * packages_per_truck_capacity
                packages_dwelled = missing_capacity
            else:
                # Add the partial truck
                final_trucks = int(np.ceil(raw_trucks))
                packages_dwelled = 0

        # No containers in fluid strategy
        physical_containers = 0
        container_fill_rate = 0.0

        # Truck fill rate uses raw capacity
        truck_fill_rate = min(1.0, total_cube / (final_trucks * raw_trailer_cube))

    # Calculate cube per truck
    cube_per_truck = total_cube / final_trucks if final_trucks > 0 else 0

    return {
        'physical_containers': physical_containers,
        'trucks_needed': final_trucks,
        'container_fill_rate': container_fill_rate,
        'truck_fill_rate': truck_fill_rate,
        'packages_dwelled': packages_dwelled,
        'total_cube_cuft': total_cube,
        'cube_per_truck': cube_per_truck,
    }