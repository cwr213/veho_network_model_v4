"""
Configuration Module v3 - Bridge to v4

This is a temporary bridge file to maintain backward compatibility
during the v3 → v4 migration. All new functionality is in config_v4.py.

This file simply re-exports everything from config_v4 so existing
imports like `from .config_v3 import ...` continue to work.

TODO: After full migration, remove this file and update all imports
to use config_v4 directly.
"""

# Re-export everything from config_v4
from .config_v4 import (
    # Constants
    EARTH_RADIUS_MILES,
    HOURS_PER_DAY,
    MINUTES_PER_HOUR,

    # Configuration classes
    OptimizationConstants,
    ValidationTolerances,
    PerformanceThresholds,

    # Enumerations
    FlowType,
    FacilityType,
    SortLevel,
    LoadStrategy,
    PathType,

    # Data classes
    CostParameters,
    TimingParameters,
    RunSettings,

    # Output configuration
    DAY_TYPES,
    OUTPUT_DIR,
    OUTPUT_FILE_TEMPLATE,
    COMPARE_FILE_TEMPLATE,
    EXECUTIVE_SUMMARY_TEMPLATE,

    # Helper functions
    get_optimization_config,
    get_validation_config,
)

__all__ = [
    # Constants
    'EARTH_RADIUS_MILES',
    'HOURS_PER_DAY',
    'MINUTES_PER_HOUR',

    # Configuration classes
    'OptimizationConstants',
    'ValidationTolerances',
    'PerformanceThresholds',

    # Enumerations
    'FlowType',
    'FacilityType',
    'SortLevel',
    'LoadStrategy',
    'PathType',

    # Data classes
    'CostParameters',
    'TimingParameters',
    'RunSettings',

    # Output configuration
    'DAY_TYPES',
    'OUTPUT_DIR',
    'OUTPUT_FILE_TEMPLATE',
    'COMPARE_FILE_TEMPLATE',
    'EXECUTIVE_SUMMARY_TEMPLATE',

    # Helper functions
    'get_optimization_config',
    'get_validation_config',
]