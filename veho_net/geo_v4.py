"""
Geographic Utilities Module

Provides distance calculations and mileage band lookups for transportation network modeling.
Uses only physical constants - all parameters from input data.

Key Functions:
    - haversine_miles: Great-circle distance between coordinates
    - band_lookup: Retrieve mileage band parameters for distance
    - calculate_zone_from_distance: Zone classification for pricing

Distance Calculation:
    Uses Haversine formula for great-circle distance on a sphere.
    Suitable for transportation network planning where accuracy within
    0.5% is acceptable (ignores Earth's ellipsoid shape).

CRITICAL CHANGE v4.7: Zones are now integers 0-8, or -1 for unknown.
No more string parsing. Input must have integer zone column.
"""

import math
import pandas as pd
from typing import Tuple, Optional
from functools import lru_cache

from .config_v4 import EARTH_RADIUS_MILES, OptimizationConstants
from .utils import get_facility_lookup


# ============================================================================
# DISTANCE CALCULATIONS
# ============================================================================

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points using Haversine formula.

    The Haversine formula calculates the shortest distance over the Earth's
    surface, giving an "as-the-crow-flies" distance. For transportation
    networks, this is multiplied by a circuity factor to account for actual
    road routing.

    Formula:
        a = sin²(Δφ/2) + cos(φ1) × cos(φ2) × sin²(Δλ/2)
        c = 2 × atan2(√a, √(1-a))
        d = R × c

    Where:
        φ = latitude, λ = longitude, R = Earth radius

    Args:
        lat1, lon1: Origin coordinates in decimal degrees
        lat2, lon2: Destination coordinates in decimal degrees

    Returns:
        Distance in miles (great-circle distance)

    Raises:
        ValueError: If coordinates are invalid (not in valid lat/lon ranges)

    Example:
        >>> # New York to Los Angeles
        >>> haversine_miles(40.7128, -74.0060, 34.0522, -118.2437)
        2451.1  # miles

    Notes:
        - Accuracy is within 0.5% for most transportation applications
        - Assumes spherical Earth (ignores ellipsoid corrections)
        - Returns 0 for identical coordinates
    """
    # Validate coordinate ranges
    if not (-90 <= lat1 <= 90) or not (-90 <= lat2 <= 90):
        raise ValueError(
            f"Latitude must be in range [-90, 90]. "
            f"Got: lat1={lat1}, lat2={lat2}"
        )

    if not (-180 <= lon1 <= 180) or not (-180 <= lon2 <= 180):
        raise ValueError(
            f"Longitude must be in range [-180, 180]. "
            f"Got: lon1={lon1}, lon2={lon2}"
        )

    # Handle identical coordinates
    if abs(lat1 - lat2) < OptimizationConstants.EPSILON and abs(lon1 - lon2) < OptimizationConstants.EPSILON:
        return 0.0

    # Convert to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = (math.sin(dphi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2)

    # Numerical stability fix for antipodal points
    a = min(1.0, a)  # Clamp to prevent sqrt of >1 due to floating point errors

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = EARTH_RADIUS_MILES * c

    # Validation
    if not math.isfinite(distance):
        raise ValueError(
            f"Invalid distance calculation for coordinates: "
            f"({lat1}, {lon1}) to ({lat2}, {lon2})"
        )

    return distance


@lru_cache(maxsize=10000)
def cached_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Cached haversine calculation for repeated lookups."""
    return haversine_miles(lat1, lon1, lat2, lon2)


def calculate_distance_with_circuity(
        lat1: float, lon1: float,
        lat2: float, lon2: float,
        mileage_bands: pd.DataFrame
) -> Tuple[float, float]:
    """
    Calculate both straight-line and actual driving distance.

    Args:
        lat1, lon1: Origin coordinates
        lat2, lon2: Destination coordinates
        mileage_bands: Mileage bands with circuity factors

    Returns:
        Tuple of (straight_line_miles, actual_driving_miles)
    """
    straight_line = haversine_miles(lat1, lon1, lat2, lon2)
    _, _, circuity, _ = band_lookup(straight_line, mileage_bands)
    actual_distance = straight_line * circuity

    return straight_line, actual_distance


# ============================================================================
# MILEAGE BAND LOOKUPS
# ============================================================================

def band_lookup(
        raw_haversine_miles: float,
        mileage_bands: pd.DataFrame
) -> Tuple[float, float, float, float]:
    """
    Look up mileage band parameters for a given distance.

    Mileage bands define cost and routing parameters that vary by distance.
    For example, short distances may have higher fixed costs and lower speeds
    (urban routing), while long distances have lower fixed costs and higher
    speeds (highway routing).

    Args:
        raw_haversine_miles: Straight-line distance in miles (no circuity)
        mileage_bands: DataFrame with columns:
            - mileage_band_min: Lower bound of distance range
            - mileage_band_max: Upper bound of distance range
            - fixed_cost_per_truck: Fixed cost component
            - variable_cost_per_mile: Variable cost per mile
            - circuity_factor: Multiplier for actual vs straight-line distance
            - mph: Average speed for this distance range

    Returns:
        Tuple of (fixed_cost_per_truck, variable_cost_per_mile, circuity_factor, mph)

    Raises:
        ValueError: If distance not found in any band and exceeds maximum range

    Example:
        >>> # 500 mile distance lookup
        >>> fixed, variable, circuity, mph = band_lookup(500, mileage_bands)
        >>> # Returns parameters for 500-mile distance band
        >>> actual_distance = 500 * circuity  # Apply circuity
        >>> cost = fixed + variable * actual_distance

    Notes:
        - Uses first matching band if distance falls in multiple ranges
        - For distances exceeding all bands, uses last (longest) band parameters
        - Circuity typically ranges from 1.1 (rural) to 1.4 (urban)
    """
    distance = float(raw_haversine_miles)

    # Validate input
    if distance < 0:
        raise ValueError(f"Distance must be non-negative, got {distance}")

    # Find matching band
    matching_bands = mileage_bands[
        (mileage_bands["mileage_band_min"] <= distance) &
        (distance <= mileage_bands["mileage_band_max"])
        ]

    if matching_bands.empty:
        # Use last band if distance exceeds maximum
        if distance > mileage_bands["mileage_band_max"].max():
            band = mileage_bands.iloc[-1]
        else:
            # This should not happen if bands are properly configured
            raise ValueError(
                f"Distance {distance:.1f} miles not found in mileage bands. "
                f"Check that mileage_band_min and mileage_band_max cover full range."
            )
    else:
        # Use first matching band
        band = matching_bands.iloc[0]

    return (
        float(band["fixed_cost_per_truck"]),
        float(band["variable_cost_per_mile"]),
        float(band["circuity_factor"]),
        float(band["mph"])
    )


def get_mileage_band_for_distance(
        distance_miles: float,
        mileage_bands: pd.DataFrame
) -> pd.Series:
    """
    Get complete mileage band row for a given distance.

    Args:
        distance_miles: Distance in miles
        mileage_bands: Mileage bands DataFrame

    Returns:
        Series containing all mileage band parameters
    """
    matching_bands = mileage_bands[
        (mileage_bands["mileage_band_min"] <= distance_miles) &
        (distance_miles <= mileage_bands["mileage_band_max"])
        ]

    if matching_bands.empty:
        # Use last band for distances exceeding maximum
        return mileage_bands.iloc[-1]

    return matching_bands.iloc[0]


# ============================================================================
# ZONE CLASSIFICATION - FIXED v4.5
# ============================================================================

def calculate_zone_from_distance(
        origin: str,
        dest: str,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> int:
    """
    Calculate zone classification based on straight-line distance.

    SIMPLIFIED v4.7: Returns integer zone 0-8, or -1 for unknown.

    Zone Assignment Logic:
    ----------------------
    - Calculate haversine distance (O=D returns 0.0)
    - Look up zone in mileage_bands based on distance
    - Return integer zone directly from mileage_bands.zone column
    - Zone 0 is ONLY for direct injection (handled separately)
    - Middle-mile O=D uses mileage_bands for distance=0

    Input Requirements:
    -------------------
    - mileage_bands.zone must be integer 0-8
    - mileage_bands must have band starting at 0 for O=D flows

    Args:
        origin: Origin facility name
        dest: Destination facility name
        facilities: Facility master data with lat/lon
        mileage_bands: Mileage bands with integer 'zone' column

    Returns:
        Integer zone 0-8, or -1 for unknown/error

    Example:
        >>> # O=D flow
        >>> zone = calculate_zone_from_distance('DFW02', 'DFW02', facilities, bands)
        >>> # Returns 1 if mileage_bands has band starting at 0 with zone=1

        >>> # Long-haul flow
        >>> zone = calculate_zone_from_distance('DFW02', 'SEA01', facilities, bands)
        >>> # Returns 8 for ~2000 mile distance

        >>> # Error case
        >>> zone = calculate_zone_from_distance('INVALID', 'SEA01', facilities, bands)
        >>> # Returns -1
    """
    try:
        from .utils import get_facility_lookup

        fac_lookup = get_facility_lookup(facilities)

        # Validate facilities exist
        if origin not in fac_lookup.index:
            print(f"Warning: Origin facility '{origin}' not found")
            return -1

        if dest not in fac_lookup.index:
            print(f"Warning: Destination facility '{dest}' not found")
            return -1

        # Get coordinates
        o_lat = float(fac_lookup.at[origin, 'lat'])
        o_lon = float(fac_lookup.at[origin, 'lon'])
        d_lat = float(fac_lookup.at[dest, 'lat'])
        d_lon = float(fac_lookup.at[dest, 'lon'])

        if any(pd.isna([o_lat, o_lon, d_lat, d_lon])):
            print(f"Warning: Invalid coordinates for {origin} or {dest}")
            return -1

        # Calculate straight-line distance (NO circuity for zone classification)
        raw_distance = haversine_miles(o_lat, o_lon, d_lat, d_lon)

        # Validate mileage_bands has zone column
        if 'zone' not in mileage_bands.columns:
            print("Warning: 'zone' column not found in mileage_bands")
            return -1

        # Find matching band by distance
        matching_band = mileage_bands[
            (mileage_bands['mileage_band_min'] <= raw_distance) &
            (raw_distance <= mileage_bands['mileage_band_max'])
            ]

        if not matching_band.empty:
            zone_val = matching_band.iloc[0]['zone']

            # Convert to integer
            try:
                zone_int = int(zone_val)

                # Validate range
                if 0 <= zone_int <= 8:
                    return zone_int
                else:
                    print(f"Warning: Zone {zone_int} outside valid range (0-8) for {origin}→{dest}")
                    return -1

            except (ValueError, TypeError):
                print(f"Warning: Invalid zone value '{zone_val}' for {origin}→{dest}")
                return -1

        # Distance exceeds all bands - use last band's zone
        if raw_distance > mileage_bands['mileage_band_max'].max():
            zone_val = mileage_bands.iloc[-1]['zone']

            try:
                zone_int = int(zone_val)
                if 0 <= zone_int <= 8:
                    return zone_int
            except (ValueError, TypeError):
                pass

            return -1

        # Distance below all bands - should not happen if bands start at 0
        min_band = mileage_bands['mileage_band_min'].min()
        print(f"Warning: Distance {raw_distance:.1f} miles below all bands for {origin}→{dest}")
        print(f"  Minimum band starts at: {min_band}")
        print(f"  Ensure mileage_bands has a band starting at 0 for O=D flows")
        return -1

    except Exception as e:
        print(f"Warning: Could not calculate zone for {origin}→{dest}: {e}")
        return -1


def calculate_zone_from_coordinates(
        lat1: float, lon1: float,
        lat2: float, lon2: float,
        mileage_bands: pd.DataFrame
) -> int:
    """
    Calculate zone classification from coordinates directly.

    SIMPLIFIED v4.7: Returns integer zone 0-8, or -1 for unknown.

    Args:
        lat1, lon1: Origin coordinates
        lat2, lon2: Destination coordinates
        mileage_bands: Mileage bands with integer zone column

    Returns:
        Integer zone 0-8, or -1 for unknown
    """
    try:
        raw_distance = haversine_miles(lat1, lon1, lat2, lon2)

        if 'zone' not in mileage_bands.columns:
            return -1

        matching_band = mileage_bands[
            (mileage_bands['mileage_band_min'] <= raw_distance) &
            (raw_distance <= mileage_bands['mileage_band_max'])
            ]

        if not matching_band.empty:
            zone_val = matching_band.iloc[0]['zone']

            try:
                zone_int = int(zone_val)
                if 0 <= zone_int <= 8:
                    return zone_int
            except (ValueError, TypeError):
                pass

        return -1

    except Exception:
        return -1


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_mileage_bands(mileage_bands: pd.DataFrame) -> bool:
    """
    Validate mileage bands configuration.

    Checks:
        - Required columns present
        - No gaps in distance ranges
        - No overlapping ranges (except at boundaries)
        - min < max for all bands
        - Positive cost/speed values
        - Band starts at 0 for O=D support

    Args:
        mileage_bands: Mileage bands DataFrame

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails with detailed error message
    """
    required_cols = {
        'mileage_band_min', 'mileage_band_max',
        'fixed_cost_per_truck', 'variable_cost_per_mile',
        'circuity_factor', 'mph'
    }

    missing = required_cols - set(mileage_bands.columns)
    if missing:
        raise ValueError(f"Mileage bands missing required columns: {sorted(missing)}")

    # Check min < max
    invalid_ranges = mileage_bands[
        mileage_bands['mileage_band_min'] >= mileage_bands['mileage_band_max']
        ]
    if not invalid_ranges.empty:
        raise ValueError(
            f"Invalid mileage band ranges (min >= max):\n"
            f"{invalid_ranges[['mileage_band_min', 'mileage_band_max']]}"
        )

    # Check first band starts at 0 (for O=D support)
    min_band = mileage_bands['mileage_band_min'].min()
    if min_band > OptimizationConstants.EPSILON:
        print(
            f"Warning: First mileage band starts at {min_band} miles, not 0.\n"
            f"  O=D flows (0 miles) may not be classified correctly.\n"
            f"  Consider adding a band starting at 0."
        )

    # Check for gaps (optional - warn only)
    sorted_bands = mileage_bands.sort_values('mileage_band_min')
    for i in range(len(sorted_bands) - 1):
        current_max = sorted_bands.iloc[i]['mileage_band_max']
        next_min = sorted_bands.iloc[i + 1]['mileage_band_min']

        if current_max < next_min - OptimizationConstants.EPSILON:
            print(
                f"Warning: Gap in mileage bands between "
                f"{current_max:.1f} and {next_min:.1f} miles"
            )

    # Check positive values
    if (mileage_bands['fixed_cost_per_truck'] < 0).any():
        raise ValueError("Fixed costs must be non-negative")

    if (mileage_bands['variable_cost_per_mile'] < 0).any():
        raise ValueError("Variable costs must be non-negative")

    if (mileage_bands['circuity_factor'] < 1.0).any():
        raise ValueError("Circuity factor must be >= 1.0")

    if (mileage_bands['mph'] <= 0).any():
        raise ValueError("Speed (mph) must be positive")

    return True


def estimate_driving_time(
        distance_miles: float,
        mileage_bands: pd.DataFrame
) -> float:
    """
    Estimate driving time for a given distance.

    Args:
        distance_miles: Distance in miles
        mileage_bands: Mileage bands with mph data

    Returns:
        Driving time in hours
    """
    _, _, circuity, mph = band_lookup(distance_miles, mileage_bands)
    actual_distance = distance_miles * circuity

    return actual_distance / mph if mph > 0 else 0.0