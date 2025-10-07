"""
Container Calculations v3 - Bridge to v4

This is a temporary bridge file to maintain backward compatibility
during the v3 → v4 migration. All functionality is in containers_v4.py.

This file simply re-exports everything from containers_v4 so existing
imports like `from .containers_v3 import ...` continue to work.

TODO: After full migration, remove this file and update all imports
to use containers_v4 directly.
"""

# Re-export everything from containers_v4
from .containers_v4 import (
    # Core functions
    weighted_pkg_cube,
    calculate_truck_capacity,
    calculate_trucks_and_fill_rates,

    # Helper functions
    get_container_capacity,
    get_containers_per_truck,
    get_trailer_capacity,
    get_raw_trailer_cube,
    estimate_containers_for_packages,
)

__all__ = [
    # Core functions
    'weighted_pkg_cube',
    'calculate_truck_capacity',
    'calculate_trucks_and_fill_rates',

    # Helper functions
    'get_container_capacity',
    'get_containers_per_truck',
    'get_trailer_capacity',
    'get_raw_trailer_cube',
    'estimate_containers_for_packages',
]