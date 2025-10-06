"""
Configuration Module - Constants, Enums, and Data Classes

Defines all type-safe enumerations and data structures used throughout the model.
NO DEFAULT VALUES - all parameters must come from input files.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

EARTH_RADIUS_MILES = 3958.756
HOURS_PER_DAY = 24.0
MINUTES_PER_HOUR = 60.0


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
    sla_penalty_per_touch_per_pkg: float

    def __post_init__(self):
        """Validate all costs are non-negative."""
        for field_name, value in self.__dataclass_fields__.items():
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
        for field_name, value in self.__dataclass_fields__.items():
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
            raise ValueError(f"path_around_the_world_factor must be >= 1.0, got {self.path_around_the_world_factor}")

        if self.sla_target_days < 1:
            raise ValueError(f"sla_target_days must be >= 1, got {self.sla_target_days}")


DAY_TYPES = {"offpeak", "peak"}

OUTPUT_DIR = Path("outputs")
OUTPUT_FILE_TEMPLATE = "{scenario_id}_{strategy}.xlsx"
COMPARE_FILE_TEMPLATE = "{scenario_id}_compare.xlsx"
EXECUTIVE_SUMMARY_TEMPLATE = "{scenario_id}_exec_sum.xlsx"