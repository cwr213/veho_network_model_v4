"""
Build Structures v3 - Bridge to v4

This is a temporary bridge file to maintain backward compatibility
during the v3 → v4 migration. All functionality is in build_structures_v4.py.

This file simply re-exports everything from build_structures_v4 so existing
imports like `from .build_structures_v3 import ...` continue to work.

TODO: After full migration, remove this file and update all imports
to use build_structures_v4 directly.
"""

# Re-export everything from build_structures_v4
from .build_structures_v4 import (
    # Core functions
    build_od_and_direct,
    candidate_paths,

    # Helper functions
    summarize_od_matrix,
    summarize_paths,
    validate_path_structure,
)

__all__ = [
    # Core functions
    'build_od_and_direct',
    'candidate_paths',

    # Helper functions
    'summarize_od_matrix',
    'summarize_paths',
    'validate_path_structure',
]