"""
Configuration Module - Constants, Enums, and Data Classes

Defines all type-safe enumerations and data structures used throughout the model.
NO DEFAULT VALUES - all parameters must come from input files.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

EARTH_RADIUS_MILES = 3958.756
HOURS_PER_DAY = 24.0
MINUTES_PER_HOUR = 60.0


# ============================================================================
# OPTIMIZATION CONSTANTS
# ============================================================================

class OptimizationConstants:
    """
    MILP solver configuration constants.

    These values control solver behavior and constraint formulation.
    Adjust these for performance tuning on large networks.
    """
    # Constraint formulation
    BIG_M: int = 1_000_000  # Arc activation constraint upper bound
    EPSILON: float = 1e-9  # Numerical tolerance for near-zero comparisons

    # Solver configuration
    MAX_SOLVER_TIME_SECONDS: float = 600.0  # 10 minutes default timeout
    NUM_SOLVER_WORKERS: int = 8  # Parallel threads for branch-and-bound

    # Scaling factors (for converting floats to integers in MILP)
    CUBE_SCALE_FACTOR: int = 1000  # Scale cube values by 1000x for integer arithmetic
    COST_SCALE_FACTOR: int = 1  # Keep costs at original scale

    # Default strategy settings
    DEFAULT_SORT_LEVEL: str = 'market'  # Baseline sort level when optimization disabled


class ValidationTolerances:
    """
    Validation threshold constants for input checking and result verification.
    """
    # Input validation
    SHARE_SUM_TOLERANCE: float = 1e-6  # Tolerance for shares summing to 1.0
    PERCENTAGE_TOLERANCE: float = 1e-6  # Tolerance for percentage validations

    # Result validation
    COST_MATCH_TOLERANCE: float = 0.01  # $0.01 tolerance for cost reconciliation
    PACKAGE_MATCH_TOLERANCE: float = 0.01  # Package count matching tolerance
    VOLUME_MATCH_TOLERANCE: float = 0.01  # Volume conservation check tolerance


class PerformanceThresholds:
    """
    Performance warning thresholds for operational planning.
    """
    # Utilization warnings
    LOW_TRUCK_FILL_THRESHOLD: float = 0.50  # Warn if truck fill < 50%
    HIGH_TRUCK_FILL_THRESHOLD: float = 0.95  # Warn if truck fill > 95%

    LOW_CONTAINER_FILL_THRESHOLD: float = 0.60  # Warn if container fill < 60%

    # Analysis thresholds
    FLUID_OPPORTUNITY_FILL_THRESHOLD: float = 0.75  # Only analyze arcs below this fill rate

    # Capacity warnings
    SORT_CAPACITY_WARNING_THRESHOLD: float = 0.85  # Warn at 85% of max capacity
    SORT_CAPACITY_CRITICAL_THRESHOLD: float = 0.95  # Critical at 95% of max capacity

    # Network complexity
    MAX_PATH_LENGTH: int = 5  # Maximum facilities in a single path
    MAX_OD_PAIRS_WARNING: int = 10000  # Warn if OD matrix exceeds this size


# ============================================================================
# ENUMERATIONS
# ============================================================================

class FlowType(Enum):
    """Package flow types through network."""
    DIRECT_INJECTION = "direct_injection"
    MIDDLE_MILE = "middle_mile"
    MIDDLE_MILE_OD = "middle_mile_od"


class FacilityType(Enum):
    """Facility operational capabilities."""
    HUB = "hub"
    HYBRID = "hybrid"
    LAUNCH = "launch"


class SortLevel(Enum):
    """Sort granularity levels at origin facility."""
    REGION = "region"
    MARKET = "market"
    SORT_GROUP = "sort_group"


class LoadStrategy(Enum):
    """Loading strategies for linehaul transportation."""
    CONTAINER = "container"
    FLUID = "fluid"


class PathType(Enum):
    """Path classification by number of facilities."""
    DIRECT = "direct"
    ONE_TOUCH = "1_touch"
    TWO_TOUCH = "2_touch"
    THREE_TOUCH = "3_touch"
    FOUR_TOUCH = "4_touch"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass(frozen=True)
class CostParameters:
    """
    Granular cost structure - ALL REQUIRED from cost_params input sheet.
    NO DEFAULTS ALLOWED.
    """
    injection_sort_cost_per_pkg: float
    intermediate_sort_cost_per_pkg: float
    last_mile_sort_cost_per_pkg: float
    last_mile_delivery_cost_per_pkg: float
    container_handling_cost: float
    premium_economy_dwell_threshold: float
    dwell_cost_per_pkg_per_day: float

    def __post_init__(self):
        """Validate all costs are non-negative."""
        for field_name in self.__dataclass_fields__:
            field_value = getattr(self, field_name)
            if field_value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {field_value}")


@dataclass(frozen=True)
class TimingParameters:
    """
    Timing parameters - ALL REQUIRED from timing_params input sheet.
    NO DEFAULTS ALLOWED.

    Processing Windows by Operation Type:
    - injection_va_hours: Origin facility sort/processing (typically 8 hours)
    - middle_mile_va_hours: Intermediate facility full sort operation (typically 16 hours)
    - crossdock_va_hours: Intermediate facility crossdock operation (typically 3 hours)
    - last_mile_va_hours: Destination facility final sort/staging (typically 4 hours)

    The distinction between middle_mile_va_hours and crossdock_va_hours enables
    accurate capacity planning:
    - Sort operations require extensive conveyor/automation infrastructure
    - Crossdock operations only need dock space and material handling equipment
    """
    hours_per_touch: float
    load_hours: float
    unload_hours: float
    injection_va_hours: float
    middle_mile_va_hours: float
    crossdock_va_hours: float
    last_mile_va_hours: float
    sort_points_per_destination: float

    def __post_init__(self):
        """Validate all timing values are positive."""
        for field_name in self.__dataclass_fields__:
            field_value = getattr(self, field_name)
            if field_value <= 0:
                raise ValueError(f"{field_name} must be positive, got {field_value}")


@dataclass(frozen=True)
class RunSettings:
    """
    Run configuration - ALL REQUIRED from run_settings input sheet.
    NO DEFAULTS ALLOWED.
    """
    load_strategy: LoadStrategy
    sla_target_days: int
    path_around_the_world_factor: float
    enable_sort_optimization: bool

    def __post_init__(self):
        """Validate run settings."""
        if self.path_around_the_world_factor < 1.0:
            raise ValueError(
                f"path_around_the_world_factor must be >= 1.0, "
                f"got {self.path_around_the_world_factor}"
            )

        if self.sla_target_days < 1:
            raise ValueError(f"sla_target_days must be >= 1, got {self.sla_target_days}")


# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

DAY_TYPES = {"offpeak", "peak"}

OUTPUT_DIR = Path("outputs")
OUTPUT_FILE_TEMPLATE = "{scenario_id}_{strategy}.xlsx"
COMPARE_FILE_TEMPLATE = "{scenario_id}_compare.xlsx"
EXECUTIVE_SUMMARY_TEMPLATE = "{scenario_id}_exec_sum.xlsx"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_optimization_config() -> dict:
    """
    Get optimization configuration as dictionary for solver setup.

    Returns:
        Dictionary with solver configuration parameters
    """
    return {
        'max_time_seconds': OptimizationConstants.MAX_SOLVER_TIME_SECONDS,
        'num_workers': OptimizationConstants.NUM_SOLVER_WORKERS,
        'big_m': OptimizationConstants.BIG_M,
        'epsilon': OptimizationConstants.EPSILON,
    }


def get_validation_config() -> dict:
    """
    Get validation configuration as dictionary.

    Returns:
        Dictionary with validation tolerance parameters
    """
    return {
        'share_sum_tolerance': ValidationTolerances.SHARE_SUM_TOLERANCE,
        'cost_match_tolerance': ValidationTolerances.COST_MATCH_TOLERANCE,
        'package_match_tolerance': ValidationTolerances.PACKAGE_MATCH_TOLERANCE,
    }