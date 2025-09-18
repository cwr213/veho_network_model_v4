import math
import pandas as pd

# Haversine distance (miles)
def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    R = 3958.756  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def band_lookup(raw_haversine_miles: float, bands: pd.DataFrame):
    """
    Look up the mileage band row for a raw haversine distance and return:
      fixed_cost_per_truck, variable_cost_per_mile, circuity_factor, mph

    Expects bands to have the following exact columns (matching your validator):
      - mileage_band_min
      - mileage_band_max
      - fixed_cost_per_truck
      - variable_cost_per_mile
      - circuity_factor
      - mph
    """
    # ensure numeric
    m = float(raw_haversine_miles)
    r = bands[(bands["mileage_band_min"] <= m) & (m <= bands["mileage_band_max"])]

    if r.empty:
        # If not found (e.g., above max band), take the last band
        r = bands.iloc[[-1]]

    r = r.iloc[0]
    return (
        float(r["fixed_cost_per_truck"]),
        float(r["variable_cost_per_mile"]),
        float(r["circuity_factor"]),
        float(r["mph"]),
    )
