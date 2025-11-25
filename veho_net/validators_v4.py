"""
Input Validation Module

STRICT VALIDATION - NO FALLBACKS ALLOWED
All required fields must be present and valid.
Raises descriptive errors for any issues.
"""

import pandas as pd
import numpy as np
from typing import Set


def validate_referential_integrity(dfs: dict) -> None:
    """
    Validate foreign key relationships across sheets.

    - Check injection facilities have is_injection_node=1
    - Check injection facilities are hub/hybrid
    - Verify mileage_bands start at 0
    """
    all_facilities = set(dfs["facilities"]["facility_name"])

    # 1. Check injection_distribution facilities exist
    if not dfs["injection_distribution"].empty:
        inj_facilities = set(dfs["injection_distribution"]["facility_name"])
        missing = inj_facilities - all_facilities
        if missing:
            raise ValueError(
                f"injection_distribution references non-existent facilities: {sorted(missing)}\n"
                f"Fix: Add these facilities to facilities sheet or remove from injection_distribution"
            )

    # 2. Check zips reference valid facilities
    if not dfs["zips"].empty:
        zip_facilities = set(dfs["zips"]["facility_name_assigned"])
        missing = zip_facilities - all_facilities
        if missing:
            raise ValueError(
                f"zips references non-existent facilities: {sorted(missing)}\n"
                f"Fix: Add these facilities to facilities sheet or update zip assignments"
            )

    # 3. Check parent_hub_name references
    parent_hubs = set(dfs["facilities"]["parent_hub_name"].dropna())
    parent_hubs.discard("")
    invalid_parents = parent_hubs - all_facilities
    if invalid_parents:
        raise ValueError(
            f"facilities.parent_hub_name references non-existent facilities: {sorted(invalid_parents)}\n"
            f"Fix: Update parent_hub_name to reference valid facilities"
        )

    # 4. Check regional_sort_hub references
    regional_hubs = set(dfs["facilities"]["regional_sort_hub"].dropna())
    regional_hubs.discard("")
    invalid_regional = regional_hubs - all_facilities
    if invalid_regional:
        raise ValueError(
            f"facilities.regional_sort_hub references non-existent facilities: {sorted(invalid_regional)}\n"
            f"Fix: Update regional_sort_hub to reference valid facilities"
        )

    # 5. NEW: Verify injection distribution facilities exist and are hub/hybrid
    if not dfs["injection_distribution"].empty:
        injection_facs = set(dfs["injection_distribution"]["facility_name"])

        for fac in injection_facs:
            fac_row = dfs["facilities"][
                dfs["facilities"]["facility_name"] == fac
                ].iloc[0]

            fac_type = str(fac_row['type']).lower()

            # Must be hub or hybrid (can receive middle-mile volume)
            if fac_type not in ['hub', 'hybrid']:
                raise ValueError(
                    f"injection_distribution facility '{fac}' must be hub or hybrid.\n"
                    f"Found type: {fac_type}\n"
                    f"Fix: Change type to 'hub' or 'hybrid', or remove from injection_distribution\n"
                    f"Note: Launch facilities cannot receive middle-mile injection"
                )

            # NOTE: is_injection_node=1 is NOT required here
            # injection_distribution allocates middle-mile volume to hubs
            # is_injection_node=1 means the facility accepts client injections
            # A facility can receive allocated volume without being a client injection point

    # 6. NEW: Verify mileage_bands start at 0
    min_band = dfs["mileage_bands"]["mileage_band_min"].min()
    if min_band > 0.001:
        raise ValueError(
            f"mileage_bands MUST include band starting at 0 (for O=D flows).\n"
            f"Current minimum: {min_band}\n"
            f"Fix: Add row with mileage_band_min=0"
        )

    # 7. NEW: Verify mileage_bands have integer zones
    try:
        zones = dfs["mileage_bands"]['zone'].astype(int)
        invalid = zones[(zones < 0) | (zones > 8)]
        if len(invalid) > 0:
            raise ValueError(
                f"mileage_bands.zone must be integers 0-8.\n"
                f"Found invalid: {invalid.unique()}"
            )
    except Exception as e:
        raise ValueError(
            f"mileage_bands.zone must be integers 0-8.\n"
            f"Error: {e}"
        )

    print("Referential integrity validated")

def validate_inputs(dfs: dict) -> None:
    """
    Comprehensive input validation across all sheets.

    Args:
        dfs: Dictionary of sheet_name -> DataFrame

    Raises:
        ValueError: If any validation fails
    """
    print("=" * 60)
    print("STRICT INPUT VALIDATION - NO FALLBACKS ALLOWED")
    print("=" * 60)

    _validate_facilities(dfs["facilities"])
    print("facilities validated")

    _validate_zips(dfs["zips"])
    print("zips validated")

    _validate_demand(dfs["demand"])
    print("demand validated")

    _validate_injection_distribution(dfs["injection_distribution"])
    print("injection_distribution validated")

    _validate_mileage_bands(dfs["mileage_bands"])
    print("mileage_bands validated")

    _validate_timing_params(dfs["timing_params"])
    print("timing_params validated")

    _validate_cost_params(dfs["cost_params"])
    print("cost_params validated")

    _validate_container_params(dfs["container_params"])
    print("container_params validated")

    _validate_package_mix(dfs["package_mix"])
    print("package_mix validated")

    _validate_run_settings(dfs["run_settings"])
    print("run_settings validated")

    _validate_scenarios(dfs["scenarios"])
    print("scenarios validated")

    validate_referential_integrity(dfs)

    print("=" * 60)
    print("VALIDATION COMPLETE - ALL REQUIRED FIELDS PRESENT")
    print("=" * 60)


def _validate_facilities(df: pd.DataFrame) -> None:
    """
    Validate facilities sheet.

    Required columns:
    - facility_name, type, market, region
    - lat, lon, timezone
    - parent_hub_name, regional_sort_hub
    - is_injection_node
    - max_sort_points_capacity (for hub/hybrid)
    - last_mile_sort_groups_count (for launch/hybrid)
    """
    required_cols = {
        "facility_name", "type", "market", "region",
        "lat", "lon", "timezone",
        "parent_hub_name", "regional_sort_hub",
        "is_injection_node",
        "max_sort_points_capacity",
        "last_mile_sort_groups_count"
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"facilities sheet missing required columns: {sorted(missing)}\n"
            f"Required: {sorted(required_cols)}"
        )

    duplicates = df["facility_name"][df["facility_name"].duplicated()].unique()
    if len(duplicates) > 0:
        raise ValueError(f"facilities has duplicate facility_name values: {list(duplicates)}")

    valid_types = {"hub", "hybrid", "launch"}
    invalid_types = df[~df["type"].str.lower().isin(valid_types)]
    if not invalid_types.empty:
        raise ValueError(
            f"facilities has invalid type values. Must be one of {valid_types}.\n"
            f"Invalid rows: {invalid_types[['facility_name', 'type']].to_dict('records')}"
        )

    missing_coords = df[pd.isna(df["lat"]) | pd.isna(df["lon"])]
    if not missing_coords.empty:
        raise ValueError(
            f"facilities missing lat/lon coordinates for: "
            f"{missing_coords['facility_name'].tolist()}"
        )

    hub_hybrid = df[df["type"].str.lower().isin(["hub", "hybrid"])]
    missing_capacity = hub_hybrid[
        pd.isna(hub_hybrid["max_sort_points_capacity"]) |
        (hub_hybrid["max_sort_points_capacity"] <= 0)
        ]

    if not missing_capacity.empty:
        raise ValueError(
            f"Hub/hybrid facilities MUST have max_sort_points_capacity > 0.\n"
            f"Missing/invalid for: {missing_capacity['facility_name'].tolist()}\n"
            f"Update facilities sheet with valid capacity values."
        )

    delivery = df[df["type"].str.lower().isin(["launch", "hybrid"])]
    missing_groups = delivery[
        pd.isna(delivery["last_mile_sort_groups_count"]) |
        (delivery["last_mile_sort_groups_count"] <= 0)
        ]

    if not missing_groups.empty:
        raise ValueError(
            f"Launch/hybrid facilities MUST have last_mile_sort_groups_count > 0.\n"
            f"Missing/invalid for: {missing_groups['facility_name'].tolist()}\n"
            f"Update facilities sheet with valid sort group counts."
        )

    invalid_injection = df[
        (df["type"].str.lower() == "launch") &
        (df["is_injection_node"].astype(int) == 1)
        ]

    if not invalid_injection.empty:
        raise ValueError(
            f"Launch facilities CANNOT be injection nodes.\n"
            f"Set is_injection_node=0 for: {invalid_injection['facility_name'].tolist()}"
        )


def _validate_zips(df: pd.DataFrame) -> None:
    """Validate ZIP code assignment data."""
    required_cols = {"zip", "facility_name_assigned", "market", "population"}

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"zips sheet missing required columns: {sorted(missing)}")

    duplicates = df["zip"][df["zip"].duplicated()].unique()
    if len(duplicates) > 0:
        raise ValueError(f"zips has duplicate ZIP codes: {list(duplicates)}")

    invalid_pop = df[df["population"] < 0]
    if not invalid_pop.empty:
        raise ValueError(f"zips has negative population values for ZIPs: {invalid_pop['zip'].tolist()}")


def _validate_demand(df: pd.DataFrame) -> None:
    """Validate demand forecast parameters."""
    required_cols = {
        "year", "annual_pkgs",
        "offpeak_pct_of_annual", "peak_pct_of_annual",
        "middle_mile_share_offpeak", "middle_mile_share_peak"
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"demand sheet missing required columns: {sorted(missing)}")

    pct_cols = [
        "offpeak_pct_of_annual", "peak_pct_of_annual",
        "middle_mile_share_offpeak", "middle_mile_share_peak"
    ]

    for col in pct_cols:
        invalid = df[~df[col].between(0, 1, inclusive="both")]
        if not invalid.empty:
            raise ValueError(
                f"demand column '{col}' has values outside [0,1].\n"
                f"Invalid rows: {invalid[[col]].to_dict('records')}"
            )


def _validate_injection_distribution(df: pd.DataFrame) -> None:
    """Validate injection distribution parameters."""
    required_cols = {"facility_name", "absolute_share"}

    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"injection_distribution missing columns: {sorted(missing)}")

    total_share = pd.to_numeric(df["absolute_share"], errors="coerce").fillna(0.0).sum()
    if total_share <= 0:
        raise ValueError("injection_distribution.absolute_share must sum > 0")


def _validate_mileage_bands(df: pd.DataFrame) -> None:
    """
    Validate mileage band cost structure and zone mapping.

    - Zone must be integer 0-8 (no strings allowed)
    - Must have band starting at 0 for O=D flows
    """
    required_cols = {
        "mileage_band_min", "mileage_band_max",
        "fixed_cost_per_truck", "variable_cost_per_mile",
        "circuity_factor", "mph", "zone"
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"mileage_bands sheet missing columns: {sorted(missing)}")

    # STRICT: zone must be integer 0-8
    try:
        zones = df['zone'].astype(int)
    except:
        raise ValueError(
            f"mileage_bands.zone must be integer 0-8.\n"
            f"Found non-integer values: {df['zone'].unique()}\n"
            f"Fix: Ensure zone column contains only integers (no text, no decimals)"
        )

    invalid_zones = zones[(zones < 0) | (zones > 8)]
    if len(invalid_zones) > 0:
        raise ValueError(
            f"mileage_bands.zone must be 0-8.\n"
            f"Found invalid zones: {invalid_zones.unique()}\n"
            f"Valid zones: 0 (direct injection), 1-8 (distance-based)"
        )

    # Require band starting at 0 for O=D flows
    min_start = df['mileage_band_min'].min()
    if min_start > 0.001:
        raise ValueError(
            f"mileage_bands MUST have band starting at 0 for O=D flows.\n"
            f"Current minimum mileage_band_min: {min_start}\n"
            f"Fix: Add row with mileage_band_min=0, mileage_band_max=0 (or small value)"
        )

    # Check for invalid ranges
    invalid_ranges = df[df["mileage_band_min"] >= df["mileage_band_max"]]
    if not invalid_ranges.empty:
        raise ValueError(
            f"mileage_bands has invalid ranges (min >= max):\n"
            f"{invalid_ranges[['mileage_band_min', 'mileage_band_max']].to_dict('records')}"
        )

    # Check for negative costs/speeds
    if (df["fixed_cost_per_truck"] < 0).any():
        raise ValueError("mileage_bands: fixed_cost_per_truck must be non-negative")

    if (df["variable_cost_per_mile"] < 0).any():
        raise ValueError("mileage_bands: variable_cost_per_mile must be non-negative")

    if (df["circuity_factor"] < 1.0).any():
        raise ValueError("mileage_bands: circuity_factor must be >= 1.0")

    if (df["mph"] <= 0).any():
        raise ValueError("mileage_bands: mph must be positive")

def _validate_timing_params(df: pd.DataFrame) -> None:
    """
    Validate timing parameters - ALL REQUIRED, NO DEFAULTS.

    Enhanced validation includes crossdock_va_hours for sort vs. crossdock distinction.
    """
    required_keys = {
        "hours_per_touch",
        "injection_va_hours",
        "middle_mile_va_hours",
        "crossdock_va_hours",
        "last_mile_va_hours",
        "sort_points_per_destination"
    }

    missing = required_keys - set(df["key"])
    if missing:
        raise ValueError(
            f"timing_params missing REQUIRED keys: {sorted(missing)}\n"
            f"All timing parameters must be specified - NO DEFAULTS.\n"
            f"Note: crossdock_va_hours is required for sort vs. crossdock operations."
        )

    for _, row in df.iterrows():
        key = row["key"]
        if key in required_keys:
            try:
                value = float(row["value"])
                if value <= 0:
                    raise ValueError(f"timing_params: {key} must be positive (found: {value})")
            except (ValueError, TypeError):
                raise ValueError(f"timing_params: {key} must be numeric (found: {row['value']})")

    crossdock_hours = float(df[df["key"] == "crossdock_va_hours"]["value"].iloc[0])
    middle_mile_hours = float(df[df["key"] == "middle_mile_va_hours"]["value"].iloc[0])

    if crossdock_hours >= middle_mile_hours:
        print(f"Warning: crossdock_va_hours ({crossdock_hours}) should typically be less than "
              f"middle_mile_va_hours ({middle_mile_hours})")


def _validate_cost_params(df: pd.DataFrame) -> None:
    """
    Validate cost parameters - ALL REQUIRED, NO DEFAULTS.
    """
    required_keys = {
        "injection_sort_cost_per_pkg",
        "intermediate_sort_cost_per_pkg",
        "last_mile_sort_cost_per_pkg",
        "last_mile_delivery_cost_per_pkg",
        "container_handling_cost",
        "premium_economy_dwell_threshold",
        "dwell_cost_per_pkg_per_day",
        "sla_penalty_per_touch_per_pkg"
    }

    missing = required_keys - set(df["key"])
    if missing:
        raise ValueError(
            f"cost_params missing REQUIRED keys: {sorted(missing)}\n"
            f"All cost parameters must be specified - NO DEFAULTS."
        )

    for _, row in df.iterrows():
        key = row["key"]
        if key in required_keys:
            try:
                value = float(row["value"])
                if value < 0:
                    raise ValueError(f"cost_params: {key} must be non-negative (found: {value})")
            except (ValueError, TypeError):
                raise ValueError(f"cost_params: {key} must be numeric (found: {row['value']})")

    dwell_row = df[df["key"] == "premium_economy_dwell_threshold"]
    if not dwell_row.empty:
        dwell_val = float(dwell_row.iloc[0]["value"])
        if not (0 <= dwell_val <= 1):
            raise ValueError(
                f"premium_economy_dwell_threshold must be in [0,1] (found: {dwell_val})"
            )


def _validate_container_params(df: pd.DataFrame) -> None:
    """Validate container parameters."""
    required_cols = {
        "container_type", "usable_cube_cuft",
        "pack_utilization_container", "containers_per_truck",
        "trailer_air_cube_cuft", "pack_utilization_fluid"
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"container_params missing columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("container_params has no rows")

    if not (df["container_type"].str.lower() == "gaylord").any():
        raise ValueError("container_params must include row with container_type='gaylord'")

    gaylord = df[df["container_type"].str.lower() == "gaylord"].iloc[0]

    pack_util_container = float(gaylord["pack_utilization_container"])
    if not (0 < pack_util_container <= 1):
        raise ValueError(
            f"pack_utilization_container must be in (0,1] (found: {pack_util_container})"
        )

    pack_util_fluid = float(df["pack_utilization_fluid"].iloc[0])
    if not (0 < pack_util_fluid <= 1):
        raise ValueError(f"pack_utilization_fluid must be in (0,1] (found: {pack_util_fluid})")


def _validate_package_mix(df: pd.DataFrame) -> None:
    """Validate package mix distribution."""
    required_cols = {"package_type", "share_of_pkgs", "avg_cube_cuft"}

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"package_mix missing columns: {sorted(missing)}")

    total_share = float(df["share_of_pkgs"].sum())
    if abs(total_share - 1.0) > 1e-6:
        raise ValueError(
            f"package_mix share_of_pkgs must sum to 1.0 (found: {total_share})"
        )

    invalid_cube = df[df["avg_cube_cuft"] <= 0]
    if not invalid_cube.empty:
        raise ValueError(
            f"package_mix avg_cube_cuft must be positive.\n"
            f"Invalid rows: {invalid_cube[['package_type', 'avg_cube_cuft']].to_dict('records')}"
        )


def _validate_run_settings(df: pd.DataFrame) -> None:
    """
    Validate run settings parameters.

    Note: load_strategy parameter removed - model always uses container baseline
    with fluid opportunities identified via post-optimization analysis.
    """
    required_keys = {
        "path_around_the_world_factor",
        "enable_sort_optimization"
    }

    missing = required_keys - set(df["key"])
    if missing:
        raise ValueError(
            f"run_settings missing REQUIRED keys: {sorted(missing)}\n"
            f"All run settings must be specified."
        )

    around_row = df[df["key"] == "path_around_the_world_factor"]
    if not around_row.empty:
        around_val = float(around_row.iloc[0]["value"])
        if around_val < 1.0 or around_val > 5.0:
            raise ValueError(
                f"path_around_the_world_factor should be in [1.0, 5.0] (found: {around_val})"
            )


def _validate_scenarios(df: pd.DataFrame) -> None:
    """Validate scenario definitions."""
    required_cols = {"year", "day_type"}

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"scenarios sheet missing columns: {sorted(missing)}")

    valid_day_types = {"peak", "offpeak"}
    invalid_days = df[~df["day_type"].str.lower().isin(valid_day_types)]
    if not invalid_days.empty:
        raise ValueError(
            f"scenarios day_type must be 'peak' or 'offpeak'.\n"
            f"Invalid rows: {invalid_days[['day_type']].to_dict('records')}"
        )