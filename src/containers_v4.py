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
from typing import Dict

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
        strategy: 'container' or 'fluid' (fluid used for opportunity analysis)

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
