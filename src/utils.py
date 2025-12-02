"""
Utility Functions Module

Shared utility functions used across the network optimization model.
Consolidates common patterns to reduce code duplication and improve maintainability.

Key Functions:
    - Facility lookup and indexing helpers
    - Safe mathematical operations
    - Formatting utilities for reporting
    - Data validation helpers
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List, Tuple
from functools import lru_cache

from .config_v4 import (
    OptimizationConstants,
    ValidationTolerances,
)


# ============================================================================
# FACILITY LOOKUP HELPERS
# ============================================================================

def get_facility_lookup(facilities: pd.DataFrame) -> pd.DataFrame:
    """
    Create indexed facility lookup with normalized types.

    This is a common pattern used throughout the codebase. Centralizing
    it here ensures consistency and makes it easier to add caching later.

    Args:
        facilities: Facility master data with required columns:
            - facility_name (unique identifier)
            - type (hub/hybrid/launch)
            - lat, lon (coordinates)

    Returns:
        DataFrame indexed by facility_name with normalized 'type' field

    Example:
        >>> fac_lookup = get_facility_lookup(facilities)
        >>> hub_type = fac_lookup.at['HUB1', 'type']  # Returns 'hub'
    """
    fac = facilities.copy()
    fac['type'] = fac['type'].astype(str).str.lower()
    return fac.set_index('facility_name')


def normalize_facility_types(facilities: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize facility type field to lowercase.

    Args:
        facilities: Facility master data

    Returns:
        DataFrame with normalized 'type' column
    """
    df = facilities.copy()
    df['type'] = df['type'].astype(str).str.lower()
    return df


def get_facility_coordinates(
        facilities: pd.DataFrame,
        facility_name: str
) -> Tuple[float, float]:
    """
    Get lat/lon coordinates for a facility.

    Args:
        facilities: Facility master data
        facility_name: Name of facility to look up

    Returns:
        Tuple of (latitude, longitude)

    Raises:
        KeyError: If facility not found
        ValueError: If coordinates are invalid
    """
    fac_lookup = get_facility_lookup(facilities)

    if facility_name not in fac_lookup.index:
        raise KeyError(f"Facility '{facility_name}' not found in master data")

    lat = float(fac_lookup.at[facility_name, 'lat'])
    lon = float(fac_lookup.at[facility_name, 'lon'])

    if pd.isna(lat) or pd.isna(lon):
        raise ValueError(f"Invalid coordinates for facility '{facility_name}'")

    return lat, lon


# ============================================================================
# SAFE MATHEMATICAL OPERATIONS
# ============================================================================

def safe_divide(
        numerator: float,
        denominator: float,
        default: float = 0.0,
        epsilon: Optional[float] = None
) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Uses EPSILON from OptimizationConstants for near-zero detection.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Value to return if denominator is zero or near-zero
        epsilon: Custom epsilon value (defaults to OptimizationConstants.EPSILON)

    Returns:
        Result of division or default value

    Example:
        >>> safe_divide(100, 0)  # Returns 0.0
        >>> safe_divide(100, 50)  # Returns 2.0
        >>> safe_divide(100, 0, default=999)  # Returns 999
    """
    if epsilon is None:
        epsilon = OptimizationConstants.EPSILON

    if abs(denominator) < epsilon:
        return default
    return numerator / denominator


def safe_percentage(
        part: float,
        total: float,
        default: float = 0.0
) -> float:
    """
    Safely calculate percentage, handling zero totals.

    Args:
        part: Partial value
        total: Total value
        default: Value to return if total is zero

    Returns:
        Percentage as decimal (0-1 scale)
    """
    return safe_divide(part, total, default=default)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value between min and max bounds.

    Args:
        value: Value to clamp
        min_val: Minimum bound
        max_val: Maximum bound

    Returns:
        Clamped value

    Example:
        >>> clamp(0.95, 0.0, 1.0)  # Returns 0.95
        >>> clamp(1.5, 0.0, 1.0)   # Returns 1.0
        >>> clamp(-0.5, 0.0, 1.0)  # Returns 0.0
    """
    return max(min_val, min(max_val, value))


# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def format_currency(value: float, decimals: int = 0) -> str:
    """
    Format numeric value as currency string.

    Args:
        value: Numeric value to format
        decimals: Number of decimal places

    Returns:
        Formatted currency string (e.g., "$1,234.56")

    Example:
        >>> format_currency(1234.567, decimals=2)
        '$1,234.57'
    """
    return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format numeric value as percentage string.

    Args:
        value: Numeric value (0-1 scale)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string (e.g., "85.3%")

    Example:
        >>> format_percentage(0.8534, decimals=1)
        '85.3%'
    """
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 0) -> str:
    """
    Format number with thousands separators.

    Args:
        value: Numeric value
        decimals: Number of decimal places

    Returns:
        Formatted number string

    Example:
        >>> format_number(1234567)
        '1,234,567'
    """
    return f"{value:,.{decimals}f}"


def format_distance(miles: float, decimals: int = 1) -> str:
    """
    Format distance with units.

    Args:
        miles: Distance in miles
        decimals: Number of decimal places

    Returns:
        Formatted distance string

    Example:
        >>> format_distance(123.456)
        '123.5 mi'
    """
    return f"{miles:.{decimals}f} mi"


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_shares_sum_to_one(
        shares: pd.Series,
        tolerance: Optional[float] = None,
        raise_error: bool = True
) -> bool:
    """
    Validate that share values sum to 1.0 within tolerance.

    Args:
        shares: Series of share values
        tolerance: Acceptable deviation from 1.0 (uses ValidationTolerances default)
        raise_error: Whether to raise ValueError on failure

    Returns:
        True if valid, False otherwise

    Raises:
        ValueError: If shares don't sum to 1.0 and raise_error=True
    """
    if tolerance is None:
        tolerance = ValidationTolerances.SHARE_SUM_TOLERANCE

    total = float(shares.sum())
    is_valid = abs(total - 1.0) < tolerance

    if not is_valid and raise_error:
        raise ValueError(
            f"Shares must sum to 1.0 (±{tolerance}), got {total:.6f}"
        )

    return is_valid


def validate_non_negative(
        df: pd.DataFrame,
        columns: List[str],
        raise_error: bool = True
) -> bool:
    """
    Validate that specified columns contain no negative values.

    Args:
        df: DataFrame to validate
        columns: List of column names to check
        raise_error: Whether to raise ValueError on failure

    Returns:
        True if all values non-negative, False otherwise

    Raises:
        ValueError: If negative values found and raise_error=True
    """
    for col in columns:
        if col not in df.columns:
            if raise_error:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            return False

        if (df[col] < 0).any():
            if raise_error:
                negative_rows = df[df[col] < 0]
                raise ValueError(
                    f"Column '{col}' contains negative values:\n"
                    f"{negative_rows[[col]].head()}"
                )
            return False

    return True


def validate_percentages(
        df: pd.DataFrame,
        columns: List[str],
        raise_error: bool = True
) -> bool:
    """
    Validate that columns contain values in [0, 1] range.

    Args:
        df: DataFrame to validate
        columns: List of column names to check
        raise_error: Whether to raise ValueError on failure

    Returns:
        True if all values in valid range, False otherwise

    Raises:
        ValueError: If values outside [0,1] found and raise_error=True
    """
    for col in columns:
        if col not in df.columns:
            if raise_error:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            return False

        if not df[col].between(0, 1, inclusive='both').all():
            if raise_error:
                invalid_rows = df[~df[col].between(0, 1, inclusive='both')]
                raise ValueError(
                    f"Column '{col}' contains values outside [0,1]:\n"
                    f"{invalid_rows[[col]].head()}"
                )
            return False

    return True


# ============================================================================
# DATA QUALITY HELPERS
# ============================================================================

def check_for_duplicates(
        df: pd.DataFrame,
        columns: List[str],
        raise_error: bool = True
) -> pd.DataFrame:
    """
    Check for duplicate rows based on specified columns.

    Args:
        df: DataFrame to check
        columns: Columns to check for duplicates
        raise_error: Whether to raise ValueError on duplicates

    Returns:
        DataFrame of duplicate rows (empty if no duplicates)

    Raises:
        ValueError: If duplicates found and raise_error=True
    """
    duplicates = df[df.duplicated(subset=columns, keep=False)]

    if not duplicates.empty and raise_error:
        raise ValueError(
            f"Found {len(duplicates)} duplicate rows on columns {columns}:\n"
            f"{duplicates[columns].head()}"
        )

    return duplicates


def check_for_missing_values(
        df: pd.DataFrame,
        required_columns: List[str],
        raise_error: bool = True
) -> Dict[str, int]:
    """
    Check for missing values in required columns.

    Args:
        df: DataFrame to check
        required_columns: Columns that should not have missing values
        raise_error: Whether to raise ValueError on missing values

    Returns:
        Dictionary mapping column names to count of missing values

    Raises:
        ValueError: If missing values found and raise_error=True
    """
    missing_counts = {}

    for col in required_columns:
        if col not in df.columns:
            if raise_error:
                raise ValueError(f"Required column '{col}' not found in DataFrame")
            missing_counts[col] = len(df)
            continue

        missing = df[col].isna().sum()
        if missing > 0:
            missing_counts[col] = missing

    if missing_counts and raise_error:
        raise ValueError(
            f"Missing values found in required columns:\n"
            f"{missing_counts}"
        )

    return missing_counts


# ============================================================================
# DATAFRAME HELPERS
# ============================================================================

def ensure_columns_exist(
        df: pd.DataFrame,
        required_columns: List[str],
        context: str = ""
) -> None:
    """
    Verify that DataFrame contains all required columns.

    Args:
        df: DataFrame to check
        required_columns: List of required column names
        context: Description of context for better error messages

    Raises:
        ValueError: If any required columns are missing
    """
    missing = set(required_columns) - set(df.columns)

    if missing:
        context_str = f" in {context}" if context else ""
        raise ValueError(
            f"Missing required columns{context_str}: {sorted(missing)}\n"
            f"Required: {sorted(required_columns)}\n"
            f"Present: {sorted(df.columns)}"
        )


def add_missing_columns(
        df: pd.DataFrame,
        columns: Dict[str, Any]
) -> pd.DataFrame:
    """
    Add missing columns with default values.

    Args:
        df: DataFrame to modify
        columns: Dictionary mapping column names to default values

    Returns:
        DataFrame with added columns

    Example:
        >>> df = add_missing_columns(df, {'new_col': 0, 'flag': False})
    """
    df = df.copy()
    for col, default_value in columns.items():
        if col not in df.columns:
            df[col] = default_value
    return df


# ============================================================================
# PATH NODE EXTRACTION
# ============================================================================

def extract_path_nodes(row: pd.Series) -> List[str]:
    """
    Extract path nodes from DataFrame row.

    Path nodes should be lists from build_structures_v4. Falls back to path_str parsing
    if path_nodes is missing or invalid.
    """
    nodes = row.get("path_nodes", None)

    # Primary: use path_nodes list directly
    if isinstance(nodes, list) and len(nodes) >= 2:
        return nodes

    # Fallback: Parse from path_str
    path_str = row.get("path_str", "")
    if isinstance(path_str, str) and "->" in path_str:
        parsed = [n.strip() for n in path_str.split("->")]
        if len(parsed) >= 2:
            return parsed

    # Last resort: direct path
    return [row.get("origin", ""), row.get("dest", "")]

# ============================================================================
# CONTAINER FLOW HELPERS
# ============================================================================

def is_container_creation_point(
        facility: str,
        path_nodes: List[str],
        sort_level: str,
        origin: str
) -> bool:
    """
    Determine if containers are created/recreated at this facility.

    Rules:
    - Origin always creates containers
    - Region sort: intermediate facilities also create (after sort)
    - Market/sort_group: only origin creates

    Args:
        facility: Facility to check
        path_nodes: Full path
        sort_level: Sort level for this OD
        origin: Origin facility

    Returns:
        True if containers are created at this facility
    """
    if facility == origin:
        return True

    if sort_level == 'region' and facility in path_nodes[1:-1]:
        return True

    return False


def get_intermediate_operation_type(
        facility: str,
        path_nodes: List[str],
        sort_level: str,
        origin: str,
        dest: str
) -> str:
    """
    Get operation type at intermediate facility.

    Returns:
        'sort', 'crossdock', 'origin', or 'destination'
    """
    if facility == origin:
        return 'origin'
    if facility == dest:
        return 'destination'

    if facility in path_nodes[1:-1]:
        if sort_level == 'region':
            return 'sort'
        else:
            return 'crossdock'

    return 'unknown'