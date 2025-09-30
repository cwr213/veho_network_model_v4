"""
Geographic Utilities Module

Provides distance calculations and mileage band lookups.
Uses only physical constants - all parameters from input data.
"""

import math
import pandas as pd
from typing import Tuple
from .config_v3 import EARTH_RADIUS_MILES


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points using Haversine formula.

    Args:
        lat1, lon1: Origin coordinates (degrees)
        lat2, lon2: Destination coordinates (degrees)

    Returns:
        Distance in miles
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS_MILES * c


def band_lookup(
        raw_haversine_miles: float,
        mileage_bands: pd.DataFrame
) -> Tuple[float, float, float, float]:
    """
    Look up mileage band parameters for a given distance.

    Args:
        raw_haversine_miles: Straight-line distance in miles
        mileage_bands: DataFrame with columns:
            - mileage_band_min
            - mileage_band_max
            - fixed_cost_per_truck
            - variable_cost_per_mile
            - circuity_factor
            - mph

    Returns:
        Tuple of (fixed_cost_per_truck, variable_cost_per_mile, circuity_factor, mph)

    Raises:
        ValueError: If distance not found in any band
    """
    distance = float(raw_haversine_miles)

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
            raise ValueError(f"Distance {distance} miles not found in mileage bands")
    else:
        band = matching_bands.iloc[0]

    return (
        float(band["fixed_cost_per_truck"]),
        float(band["variable_cost_per_mile"]),
        float(band["circuity_factor"]),
        float(band["mph"])
    )


def calculate_zone_from_distance(
        origin: str,
        dest: str,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> str:
    """
    Calculate zone based on straight-line distance between facilities.

    Zone classification uses O-D straight-line distance (no circuity) as zones
    drive shipper pricing and should not depend on carrier routing decisions.

    Args:
        origin: Origin facility name
        dest: Destination facility name
        facilities: Facility data with lat/lon
        mileage_bands: Mileage bands with zone column

    Returns:
        Zone string (e.g., "Zone 2")
    """
    try:
        fac_lookup = facilities.set_index('facility_name')[['lat', 'lon']]

        if origin not in fac_lookup.index or dest not in fac_lookup.index:
            return 'unknown'

        o_lat = float(fac_lookup.at[origin, 'lat'])
        o_lon = float(fac_lookup.at[origin, 'lon'])
        d_lat = float(fac_lookup.at[dest, 'lat'])
        d_lon = float(fac_lookup.at[dest, 'lon'])

        # Calculate straight-line distance (no circuity for zone classification)
        raw_distance = haversine_miles(o_lat, o_lon, d_lat, d_lon)

        # Look up zone from mileage bands
        if 'zone' in mileage_bands.columns:
            matching_band = mileage_bands[
                (mileage_bands['mileage_band_min'] <= raw_distance) &
                (raw_distance <= mileage_bands['mileage_band_max'])
                ]

            if not matching_band.empty:
                return str(matching_band.iloc[0]['zone'])

        return 'unknown'

    except Exception as e:
        print(f"Warning: Could not calculate zone for {origin}->{dest}: {e}")
        return 'unknown'