"""
MILP Optimization v3 - Bridge to v4

Backward compatibility bridge during v3 → v4 migration.
All functionality is in milp_v4.py.

TODO: After migration, remove this file and update imports to milp_v4.
"""

from .milp_v4 import (
    solve_network_optimization,
)

__all__ = [
    'solve_network_optimization',
]