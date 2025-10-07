"""
Geographic Utilities v3 - Bridge to v4

This is a temporary bridge file to maintain backward compatibility
during the v3 → v4 migration. All functionality is in geo_v4.py.

This file simply re-exports everything from geo_v4 so existing
imports like `from .geo_v3 import ...` continue to work.

TODO: After full migration, remove this file and update all imports
to use geo_v4 directly.
"""

# Re-export everything from geo_v4
from .geo_v4 import (
    # Core distance functions
    haversine_miles,
    calculate_distance_with_circuity,

    # Mileage band lookups
    band_lookup,
    get_mileage_band_for_distance,

    # Zone classification
    calculate_zone_from_distance,
    calculate_zone_from_coordinates,

    # Validation and helpers
    validate_mileage_bands,
    estimate_driving_time,
)

__all__ = [
    # Core distance functions
    'haversine_miles',
    'calculate_distance_with_circuity',

    # Mileage band lookups
    'band_lookup',
    'get_mileage_band_for_distance',

    # Zone classification
    'calculate_zone_from_distance',
    'calculate_zone_from_coordinates',

    # Validation and helpers
    'validate_mileage_bands',
    'estimate_driving_time',
]