"""
Reporting v3 - Bridge to v4

Backward compatibility bridge during v3 → v4 migration.
All functionality is in reporting_v4.py.

TODO: After migration, remove this file and update imports to reporting_v4.
"""

from .reporting_v4 import (
    # New v4 functions
    build_facility_volume,
    build_facility_network_profile,
    calculate_network_distance_metrics,
    calculate_network_touch_metrics,
    calculate_network_zone_distribution,
    calculate_network_sort_distribution,

    # Legacy functions (for backward compatibility)
    build_facility_rollup,
    calculate_hourly_throughput,
    add_zone_classification,
    build_path_steps,
    build_sort_summary,
    validate_network_aggregations,
)

__all__ = [
    # New v4 functions
    'build_facility_volume',
    'build_facility_network_profile',
    'calculate_network_distance_metrics',
    'calculate_network_touch_metrics',
    'calculate_network_zone_distribution',
    'calculate_network_sort_distribution',

    # Legacy functions
    'build_facility_rollup',
    'calculate_hourly_throughput',
    'add_zone_classification',
    'build_path_steps',
    'build_sort_summary',
    'validate_network_aggregations',
]