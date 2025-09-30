"""
Configuration Module - Constants, Enums, and Data Classes

Defines all type-safe enumerations and data structures used throughout the model.
NO DEFAULT VALUES - all parameters must come from input files.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

# Physical and mathematical constants (ONLY acceptable hardcoded values)
EARTH_RADIUS_MILES = 3958.756
HOURS_PER_DAY = 24.0
MINUTES_PER_HOUR = 60.0


class FlowType(Enum):
    """Package flow types through network."""
    DIRECT_INJECTION = "direct_injection"  # Zone 0 - shipper direct to last mile
    MIDDLE_MILE = "middle_mile"  # Zones 1-5 - hub network routing
    MIDDLE_MILE_OD = "middle_mile_od"  # Zone 2 - origin=destination middle-mile


class FacilityType(Enum):
    """Facility operational capabilities."""
    HUB = "hub"  # Sort and ship only (no last-mile delivery)
    HYBRID = "hybrid"  # Sort, ship, AND last-mile delivery
    LAUNCH = "launch"  # Last-mile delivery only (no middle-mile)


class SortLevel(Enum):
    """Sort granularity levels at origin facility."""
    REGION = "region"  # Sort to regional_sort_hub (multiple facilities consolidated)
    MARKET = "market"  # Sort to destination facility (one per facility)
    SORT_GROUP = "sort_group"  # Sort to route groups (multiple per facility)


class LoadStrategy(Enum):
    """Loading strategies for linehaul transportation."""
    CONTAINER = "container"  # Gaylords/pallets in trailers (lower fill, crossdock-capable)
    FLUID = "fluid"  # Floor loaded (higher fill, requires full sort at destination)


class PathType(Enum):
    """Path classification by number of facilities."""
    DIRECT = "direct"  # 2 facilities (O→D)
    ONE_TOUCH = "1_touch"  # 3 facilities (O→I→D)
    TWO_TOUCH = "2_touch"  # 4 facilities
    THREE_TOUCH = "3_touch"  # 5 facilities
    FOUR_TOUCH = "4_touch"  # 6+ facilities


@dataclass(frozen=True)
class CostParameters:
    """
    Granular cost structure - ALL REQUIRED from cost_params input sheet.
    NO DEFAULTS ALLOWED.
    """
    # Sort/processing costs (per package)
    injection_sort_cost_per_pkg: float
    intermediate_sort_cost_per_pkg: float
    last_mile_sort_cost_per_pkg: float
    last_mile_delivery_cost_per_pkg: float

    # Container handling (per container)
    container_handling_cost: float

    # Dwell parameters
    premium_economy_dwell_threshold: float  # Fractional truck threshold (e.g., 0.10)
    dwell_cost_per_pkg_per_day: float  # Cost for dwelling packages 24 hours

    # SLA penalties
    sla_penalty_per_touch_per_pkg: float  # Penalty per intermediate touch

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
    """
    # Facility processing hours
    hours_per_touch: float  # Generic facility processing time
    load_hours: float  # Loading time
    unload_hours: float  # Unloading time

    # Value-added hours by facility role
    injection_va_hours: float  # VA hours at injection facilities
    middle_mile_va_hours: float  # VA hours at intermediate hubs
    last_mile_va_hours: float  # VA hours at last-mile facilities

    # Sort optimization parameters
    sort_points_per_destination: float  # Sort points per destination (base unit)

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
    load_strategy: LoadStrategy  # Global loading strategy
    sla_target_days: int  # Target service level (days)
    path_around_the_world_factor: float  # Max path distance vs. straight-line
    enable_sort_optimization: bool  # Enable multi-level sort optimization

    def __post_init__(self):
        """Validate run settings."""
        if self.path_around_the_world_factor < 1.0:
            raise ValueError(f"path_around_the_world_factor must be >= 1.0, got {self.path_around_the_world_factor}")

        if self.sla_target_days < 1:
            raise ValueError(f"sla_target_days must be >= 1, got {self.sla_target_days}")


# Valid day types
DAY_TYPES = {"offpeak", "peak"}

# Output directory structure
OUTPUT_DIR = Path("outputs")
OUTPUT_FILE_TEMPLATE = "{scenario_id}_{strategy}.xlsx"
COMPARE_FILE_TEMPLATE = "{scenario_id}_compare.xlsx"
EXECUTIVE_SUMMARY_TEMPLATE = "{scenario_id}_exec_sum.xlsx"