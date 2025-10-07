"""
Veho Network Optimization Package v4

A comprehensive transportation network optimization framework for middle-mile
and last-mile delivery network design.
"""

__version__ = "4.0.0"

# Expose key utilities at package level for convenience
from .utils import (
    # Facility helpers
    get_facility_lookup,
    normalize_facility_types,
    get_facility_coordinates,

    # Math helpers
    safe_divide,
    safe_percentage,
    clamp,

    # Formatting
    format_currency,
    format_percentage,
    format_number,
    format_distance,

    # Validation
    validate_shares_sum_to_one,
    validate_non_negative,
    validate_percentages,

    # Data quality
    check_for_duplicates,
    check_for_missing_values,

    # DataFrame helpers
    ensure_columns_exist,
    add_missing_columns,
)

__all__ = [
    # Facility helpers
    "get_facility_lookup",
    "normalize_facility_types",
    "get_facility_coordinates",

    # Math helpers
    "safe_divide",
    "safe_percentage",
    "clamp",

    # Formatting
    "format_currency",
    "format_percentage",
    "format_number",
    "format_distance",

    # Validation
    "validate_shares_sum_to_one",
    "validate_non_negative",
    "validate_percentages",

    # Data quality
    "check_for_duplicates",
    "check_for_missing_values",

    # DataFrame helpers
    "ensure_columns_exist",
    "add_missing_columns",
]