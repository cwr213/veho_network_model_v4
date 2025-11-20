"""
Container and Truck Capacity Calculations

Handles both container and fluid loading strategies with proper fill rate calculations.

Key Functions:
    - weighted_pkg_cube: Calculate weighted average package cube
    - calculate_truck_capacity: Determine packages per truck by strategy
    - calculate_trucks_and_fill_rates: Full truck calculation with dwell logic
    - calculate_containers_per_package: NEW - Fraction of container per package

Business Logic:
    Container Strategy: packages → gaylords → trucks
    Fluid Strategy: packages → trucks directly
    Premium Economy Dwell: Round down fractional trucks below threshold
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from .config_v4 import OptimizationConstants
from .utils import safe_divide


# ============================================================================
# PACKAGE CUBE CALCULATIONS
# ============================================================================

def weighted_pkg_cube(package_mix: pd.DataFrame) -> float:
    """
    Calculate weighted average cubic feet per package.

    Args:
        package_mix: Package distribution with share_of_pkgs and avg_cube_cuft columns

    Returns:
        Weighted average cube per package
    """
    return float((package_mix["share_of_pkgs"] * package_mix["avg_cube_cuft"]).sum())


# ============================================================================
# CONTAINER PER PACKAGE CALCULATION (NEW)
# ============================================================================

def calculate_containers_per_package(
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame
) -> float:
    """
    Calculate fraction of container per package for cost allocation.

    Returns inverse of packages per container (e.g., 0.02 if 50 pkgs fit per container).
    Used to allocate container handling costs on per-package basis.

    Args:
        package_mix: Package distribution
        container_params: Container parameters

    Returns:
        Containers per package (decimal fraction)
    """
    weighted_cube = weighted_pkg_cube(package_mix)

    if weighted_cube < OptimizationConstants.EPSILON:
        raise ValueError(f"Weighted package cube must be positive, got {weighted_cube}")

    # Get effective container capacity
    gaylord_row = container_params[
        container_params["container_type"].str.lower() == "gaylord"
        ].iloc[0]

    raw_container_cube = float(gaylord_row["usable_cube_cuft"])
    pack_util_container = float(gaylord_row["pack_utilization_container"])
    effective_container_cube = raw_container_cube * pack_util_container

    # Calculate packages per container
    packages_per_container = effective_container_cube / weighted_cube

    # Return inverse (containers per package)
    return 1.0 / packages_per_container


# ============================================================================
# TRUCK CAPACITY CALCULATIONS
# ============================================================================

def calculate_truck_capacity(
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        strategy: str
) -> float:
    """
    Calculate packages per truck capacity based on loading strategy.

    Container: capacity through gaylords with pack utilization
    Fluid: direct trailer capacity with pack utilization

    Args:
        package_mix: Package distribution
        container_params: Container/trailer parameters
        strategy: 'container' or 'fluid'

    Returns:
        Effective packages per truck capacity
    """
    weighted_avg_pkg_cube = weighted_pkg_cube(package_mix)

    if weighted_avg_pkg_cube < OptimizationConstants.EPSILON:
        raise ValueError(f"Weighted package cube must be positive, got {weighted_avg_pkg_cube}")

    strategy_lower = strategy.lower()

    if strategy_lower == "container":
        # Container strategy: packages → gaylords → trucks
        gaylord_row = container_params[
            container_params["container_type"].str.lower() == "gaylord"
            ].iloc[0]

        usable_cube = float(gaylord_row["usable_cube_cuft"])
        pack_util = float(gaylord_row["pack_utilization_container"])
        containers_per_truck_val = int(gaylord_row["containers_per_truck"])

        effective_container_cube = usable_cube * pack_util
        packages_per_truck = (effective_container_cube / weighted_avg_pkg_cube) * containers_per_truck_val

    elif strategy_lower == "fluid":
        # Fluid strategy: packages → trucks directly
        trailer_cube = float(container_params["trailer_air_cube_cuft"].iloc[0])
        pack_util = float(container_params["pack_utilization_fluid"].iloc[0])

        effective_trailer_cube = trailer_cube * pack_util
        packages_per_truck = effective_trailer_cube / weighted_avg_pkg_cube

    else:
        raise ValueError(f"Invalid strategy '{strategy}'. Must be 'container' or 'fluid'")

    return packages_per_truck


# ============================================================================
# PREMIUM ECONOMY DWELL LOGIC
# ============================================================================

def calculate_trucks_and_fill_rates(
        total_packages: float,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        strategy: str,
        dwell_threshold: float
) -> Dict[str, float]:
    """
    Calculate truck requirements with premium economy dwell logic.

    Premium economy dwell: If fractional truck usage is below threshold, round down
    and dwell excess packages to next day rather than dispatch nearly-empty truck.

    Fill rates use raw capacity (not effective) for executive reporting standard.

    Args:
        total_packages: Package volume
        package_mix: Package distribution
        container_params: Container/trailer parameters
        strategy: 'container' or 'fluid'
        dwell_threshold: Fractional truck threshold for rounding decision

    Returns:
        Dict with trucks_needed, fill rates, packages_dwelled, and cube metrics
    """
    # Handle zero-package case
    if total_packages <= 0:
        return {
            'physical_containers': 0,
            'trucks_needed': 1,  # Always minimum 1 truck
            'container_fill_rate': 0.0,
            'truck_fill_rate': 0.0,
            'packages_dwelled': 0,
            'total_cube_cuft': 0.0,
            'cube_per_truck': 0.0
        }

    # Calculate total cube and get raw trailer capacity
    weighted_cube = weighted_pkg_cube(package_mix)
    total_cube = total_packages * weighted_cube
    raw_trailer_cube = float(container_params["trailer_air_cube_cuft"].iloc[0])

    # Get truck capacity from parameters
    packages_per_truck_capacity = calculate_truck_capacity(package_mix, container_params, strategy)

    if strategy.lower() == "container":
        # ===== CONTAINER STRATEGY =====
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

        # Calculate raw trucks needed
        raw_trucks = total_packages / packages_per_truck_capacity

        # Apply premium economy dwell logic
        if raw_trucks <= 1.0:
            # Always use at least 1 truck (never round to 0)
            final_trucks = 1
            packages_dwelled = 0
        else:
            fractional_part = raw_trucks - int(raw_trucks)

            if fractional_part < dwell_threshold:
                # Round down and dwell excess packages
                final_trucks = int(raw_trucks)
                missing_capacity = (raw_trucks - final_trucks) * packages_per_truck_capacity
                packages_dwelled = missing_capacity
            else:
                # Round up and dispatch partial truck
                final_trucks = int(np.ceil(raw_trucks))
                packages_dwelled = 0

        # Fill rates use raw capacities (executive reporting standard)
        container_fill_rate = min(1.0, total_cube / (physical_containers * raw_container_cube))
        truck_fill_rate = min(1.0, total_cube / (final_trucks * raw_trailer_cube))

    else:
        # ===== FLUID STRATEGY =====
        # Calculate raw trucks needed
        raw_trucks = total_packages / packages_per_truck_capacity

        # Apply premium economy dwell logic
        if raw_trucks <= 1.0:
            # Always use at least 1 truck (never round to 0)
            final_trucks = 1
            packages_dwelled = 0
        else:
            fractional_part = raw_trucks - int(raw_trucks)

            if fractional_part < dwell_threshold:
                # Round down and dwell excess packages
                final_trucks = int(raw_trucks)
                missing_capacity = (raw_trucks - final_trucks) * packages_per_truck_capacity
                packages_dwelled = missing_capacity
            else:
                # Round up and dispatch partial truck
                final_trucks = int(np.ceil(raw_trucks))
                packages_dwelled = 0

        # No containers in fluid strategy
        physical_containers = 0
        container_fill_rate = 0.0

        # Truck fill rate uses raw capacity
        truck_fill_rate = min(1.0, total_cube / (final_trucks * raw_trailer_cube))

    # Calculate cube per truck
    cube_per_truck = safe_divide(total_cube, final_trucks, default=0.0)

    return {
        'physical_containers': physical_containers,
        'trucks_needed': final_trucks,
        'container_fill_rate': container_fill_rate,
        'truck_fill_rate': truck_fill_rate,
        'packages_dwelled': packages_dwelled,
        'total_cube_cuft': total_cube,
        'cube_per_truck': cube_per_truck,
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_container_capacity(container_params: pd.DataFrame) -> float:
    """
    Get effective gaylord container capacity with pack utilization.

    Args:
        container_params: Container parameters

    Returns:
        Effective cube capacity in cubic feet
    """
    gaylord_row = container_params[
        container_params["container_type"].str.lower() == "gaylord"
        ].iloc[0]

    usable_cube = float(gaylord_row["usable_cube_cuft"])
    pack_util = float(gaylord_row["pack_utilization_container"])

    return usable_cube * pack_util


def get_containers_per_truck(container_params: pd.DataFrame) -> int:
    """
    Get number of gaylord containers per truck.

    Args:
        container_params: Container parameters

    Returns:
        Integer count of containers per truck
    """
    gaylord_row = container_params[
        container_params["container_type"].str.lower() == "gaylord"
        ].iloc[0]

    return int(gaylord_row["containers_per_truck"])


def get_trailer_capacity(container_params: pd.DataFrame) -> float:
    """
    Get effective trailer capacity for fluid strategy.

    Args:
        container_params: Container parameters

    Returns:
        Effective trailer cube in cubic feet
    """
    trailer_cube = float(container_params["trailer_air_cube_cuft"].iloc[0])
    pack_util = float(container_params["pack_utilization_fluid"].iloc[0])

    return trailer_cube * pack_util


def get_raw_trailer_cube(container_params: pd.DataFrame) -> float:
    """
    Get raw trailer cube capacity (for fill rate calculations).

    Args:
        container_params: Container parameters

    Returns:
        Raw trailer cube in cubic feet
    """
    return float(container_params["trailer_air_cube_cuft"].iloc[0])


def estimate_containers_for_packages(
        packages: float,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame
) -> int:
    """
    Estimate number of containers needed for given package volume.

    Args:
        packages: Package count
        package_mix: Package mix distribution
        container_params: Container parameters

    Returns:
        Number of containers required
    """
    if packages <= 0:
        return 0

    weighted_cube = weighted_pkg_cube(package_mix)
    total_cube = packages * weighted_cube

    effective_container_cube = get_container_capacity(container_params)

    return max(1, int(np.ceil(total_cube / effective_container_cube)))


def calculate_trucks_vectorized(
        packages_array: np.ndarray,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        strategy: str,
        dwell_threshold: float
) -> np.ndarray:
    """
    Vectorized truck calculation for batch processing.

    Args:
        packages_array: Array of package volumes
        package_mix: Package distribution
        container_params: Container parameters
        strategy: Loading strategy
        dwell_threshold: Premium economy threshold

    Returns:
        Array of truck counts (integers)
    """
    packages_per_truck_capacity = calculate_truck_capacity(
        package_mix, container_params, strategy
    )

    raw_trucks = packages_array / packages_per_truck_capacity

    # Vectorized premium economy logic
    fractional_part = raw_trucks - np.floor(raw_trucks)

    final_trucks = np.where(
        raw_trucks <= 1.0,
        1,  # Always at least 1 truck
        np.where(
            fractional_part < dwell_threshold,
            np.floor(raw_trucks),  # Round down
            np.ceil(raw_trucks)  # Round up
        )
    )

    return final_trucks.astype(int)