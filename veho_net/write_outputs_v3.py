"""
Write Outputs v3 - Bridge to v4

Backward compatibility bridge during v3 → v4 migration.

TODO: After migration, remove this file and update imports to write_outputs_v4.
"""

from .write_outputs_v4 import (
    write_workbook,
    write_comparison_workbook,
    write_executive_summary,
)

__all__ = [
    'write_workbook',
    'write_comparison_workbook',
    'write_executive_summary',
]