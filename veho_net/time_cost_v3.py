"""
Time and Cost Calculation Module

Provides utilities for calculating transportation timing, container requirements,
and truck capacity. Also contains legacy path cost calculation function retained
for testing purposes only.

Key Functions:
- weighted_pkg_cube(): Calculate weighted average package cube
- calculate_truck_capacity(): Determine packages per truck by strategy
- enhanced_container_truck_calculation(): Truck requirements with dwell logic
- containers_for_pkgs_day(): Container count for daily volume

Legacy Functions (deprecated):
- path_cost_and_time(): Old path-level cost calculation (replaced by MILP)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import pytz
from typing import Tuple, Dict, Optional, List
from .geo_v3 import haversine_miles, band_lookup


def _parse_hhmm(val) -> time:
    """
    Parse time values from various formats into time objects.

    Handles:
    - Python time objects
    - Pandas datetime objects
    - Excel decimal time values (e.g., 0.5 = 12:00)
    - String formats (HH:MM or HH:MM:SS)

    Args:
        val: Time value in any supported format

    Returns:
        time object

    Raises:
        ValueError: If format cannot be parsed
    """
    from datetime import time as _t

    if isinstance(val, _t):
        return val.replace(tzinfo=None)

    if hasattr(val, "to_pydatetime"):
        dt = val.to_pydatetime()
        return time(dt.hour, dt.minute, dt.second)

    # Excel stores times as fractions of a day (0.5 = noon)
    if isinstance(val, (int, float)) and not pd.isna(val):
        total_minutes = int(round(float(val) * 24 * 60)) % (24 * 60)
        return time(total_minutes // 60, total_minutes % 60)

    raw = str(val).strip()
    if "," in raw:
        raw = raw.split(",")[0].strip()

    parts = raw.split(":")
    if len(parts) == 2:
        hh, mm = parts
        return time(int(hh), int(mm))
    if len(parts) == 3:
        hh, mm, ss = parts
        return time(int(hh), int(mm), int(ss))

    raise ValueError(f"Unable to parse time '{val}'. Use HH:MM or HH:MM:SS.")


def _facility_lookup(facility_df: pd.DataFrame) -> Dict[str, dict]:
    """
    Create facility lookup dictionary with key attributes.

    Args:
        facility_df: Facility master data

    Returns:
        Dictionary mapping facility_name to attributes dict
    """
    lookup_dict = {}
    for _, r in facility_df.iterrows():
        lookup_dict[r["facility_name"]] = {
            "type": r["type"],
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "tz": str(r["timezone"]),
            "parent": str(r["parent_hub_name"]),
        }
    return lookup_dict


def effective_gaylord_cube(container_params: pd.DataFrame) -> float:
    """
    Calculate effective gaylord container cube capacity with pack utilization.

    Args:
        container_params: Container parameters with pack_utilization_container

    Returns:
        Effective cube capacity in cubic feet
    """
    gaylord_row = container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0]
    return float(gaylord_row["usable_cube_cuft"]) * float(gaylord_row["pack_utilization_container"])


def containers_per_truck(container_params: pd.DataFrame) -> int:
    """
    Get number of gaylord containers that fit per truck.

    Args:
        container_params: Container parameters

    Returns:
        Integer count of containers per truck
    """
    gaylord_row = container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0]
    return int(gaylord_row["containers_per_truck"])


def weighted_pkg_cube(package_mix: pd.DataFrame) -> float:
    """
    Calculate weighted average cube per package across package mix.

    Args:
        package_mix: Package distribution with share_of_pkgs and avg_cube_cuft

    Returns:
        Weighted average cube in cubic feet per package
    """
    return float((package_mix["share_of_pkgs"] * package_mix["avg_cube_cuft"]).sum())


def trailer_effective_cube(container_params: pd.DataFrame) -> float:
    """
    Calculate effective trailer cube capacity for fluid strategy.

    Args:
        container_params: Container parameters with pack_utilization_fluid

    Returns:
        Effective trailer cube in cubic feet
    """
    base_cube = float(container_params["trailer_air_cube_cuft"].iloc[0])
    utilization = float(container_params["pack_utilization_fluid"].iloc[0])
    return base_cube * utilization


def trailer_raw_cube(container_params: pd.DataFrame) -> float:
    """
    Get raw trailer cube capacity for fill rate calculations.

    Args:
        container_params: Container parameters

    Returns:
        Raw trailer cube in cubic feet
    """
    return float(container_params["trailer_air_cube_cuft"].iloc[0])


def compute_leg_metrics(
        from_facility: str,
        to_facility: str,
        facility_lookup: dict,
        bands: pd.DataFrame
) -> dict:
    """
    Calculate distance, cost, and timing metrics for a single leg.

    Args:
        from_facility: Origin facility name
        to_facility: Destination facility name
        facility_lookup: Facility attribute dictionary
        bands: Mileage bands with cost/timing parameters

    Returns:
        Dictionary with distance_miles, fixed cost, variable cost, mph
    """
    lat1, lon1 = facility_lookup[from_facility]["lat"], facility_lookup[from_facility]["lon"]
    lat2, lon2 = facility_lookup[to_facility]["lat"], facility_lookup[to_facility]["lon"]

    raw_distance = haversine_miles(lat1, lon1, lat2, lon2)
    fixed_cost, variable_cost, circuity_factor, mph = band_lookup(raw_distance, bands)
    actual_distance = raw_distance * circuity_factor

    return {
        "distance_miles": actual_distance,
        "fixed": fixed_cost,
        "var": variable_cost,
        "mph": mph
    }


def calculate_truck_capacity(
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        strategy: str
) -> float:
    """
    Calculate packages per truck capacity based on loading strategy.

    Container strategy formula:
        packages_per_truck = ((usable_cube × pack_util_container) ÷ weighted_pkg_cube) × containers_per_truck

    Fluid strategy formula:
        packages_per_truck = (trailer_cube × pack_util_fluid) ÷ weighted_pkg_cube

    Args:
        package_mix: Package distribution with cube factors
        container_params: Container and trailer capacity parameters
        strategy: Loading strategy ('container' or 'fluid')

    Returns:
        Packages per truck capacity
    """
    weighted_avg_pkg_cube = weighted_pkg_cube(package_mix)

    if strategy.lower() == "container":
        gaylord_row = container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0]
        usable_cube = float(gaylord_row["usable_cube_cuft"])
        pack_util = float(gaylord_row["pack_utilization_container"])
        containers_per_truck_val = int(gaylord_row["containers_per_truck"])

        packages_per_truck = ((usable_cube * pack_util) / weighted_avg_pkg_cube) * containers_per_truck_val
    else:
        # Fluid strategy
        trailer_cube = float(container_params["trailer_air_cube_cuft"].iloc[0])
        pack_util = float(container_params["pack_utilization_fluid"].iloc[0])

        packages_per_truck = (trailer_cube * pack_util) / weighted_avg_pkg_cube

    return packages_per_truck


def enhanced_container_truck_calculation(
        lane_od_pairs: List[Tuple],
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        cost_kv: dict,
        strategy: str = "container"
) -> dict:
    """
    Calculate truck requirements and fill rates with premium economy dwell logic.

    Premium Economy Dwell Logic:
    - Calculate exact trucks needed based on effective cube capacity
    - If fractional truck < dwell_threshold: round down, dwell excess packages
    - Always use minimum 1 truck (never round to 0)
    - Dwelled packages incur dwell cost penalty

    Fill Rate Calculation:
    - Uses RAW capacity (not effective) per executive reporting standard
    - Measures actual cube utilization against theoretical maximum

    Args:
        lane_od_pairs: List of (od_dict, packages) tuples for aggregation
        package_mix: Package mix distribution
        container_params: Container and trailer capacity parameters
        cost_kv: Cost parameters including premium_economy_dwell_threshold
        strategy: Loading strategy ('container' or 'fluid')

    Returns:
        Dictionary with:
        - physical_containers: Container count (container strategy only)
        - trucks_needed: Number of trucks required
        - container_fill_rate: Container utilization (0-1)
        - truck_fill_rate: Truck utilization (0-1)
        - packages_dwelled: Packages delayed to next day
        - total_cube_cuft: Total package cube
        - cube_per_truck: Average cube per truck
    """
    total_pkgs = sum(pkgs for _, pkgs in lane_od_pairs)

    if total_pkgs <= 0:
        return {
            'physical_containers': 0,
            'trucks_needed': 1,
            'container_fill_rate': 0.0,
            'truck_fill_rate': 0.0,
            'packages_dwelled': 0,
            'total_cube_cuft': 0.0,
            'cube_per_truck': 0.0
        }

    weighted_cube = weighted_pkg_cube(package_mix)
    total_cube = total_pkgs * weighted_cube
    raw_trailer_cube = trailer_raw_cube(container_params)

    # Get truck capacity from input parameters
    packages_per_truck_capacity = calculate_truck_capacity(package_mix, container_params, strategy)

    # Get dwell threshold from cost parameters
    dwell_threshold = float(cost_kv.get('premium_economy_dwell_threshold', 0.10))

    if strategy.lower() == "container":
        # Container strategy: packages → containers → trucks
        gaylord_row = container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0]
        raw_container_cube = float(gaylord_row["usable_cube_cuft"])
        pack_util_container = float(gaylord_row["pack_utilization_container"])
        effective_container_cube = raw_container_cube * pack_util_container
        containers_per_truck_val = int(gaylord_row["containers_per_truck"])

        # Calculate containers needed based on effective space
        exact_containers = total_cube / effective_container_cube
        physical_containers = max(1, int(np.ceil(exact_containers)))

        # Calculate raw trucks needed
        raw_trucks = total_pkgs / packages_per_truck_capacity

        # Apply premium economy dwell logic
        if raw_trucks <= 1.0:
            # Always use at least 1 truck (never round to 0)
            final_trucks = 1
            packages_dwelled = 0
        else:
            fractional_part = raw_trucks - int(raw_trucks)
            if fractional_part < dwell_threshold:
                # Round down and dwell excess packages
                final_trucks = int(raw_trucks)
                missing_capacity = (raw_trucks - final_trucks) * packages_per_truck_capacity
                packages_dwelled = missing_capacity
            else:
                # Add the partial truck (fractional part >= threshold)
                final_trucks = int(np.ceil(raw_trucks))
                packages_dwelled = 0

        # Fill rates use raw capacities (executive reporting standard)
        container_fill_rate = min(1.0, total_cube / (physical_containers * raw_container_cube))
        truck_fill_rate = min(1.0, total_cube / (final_trucks * raw_trailer_cube))

    else:
        # Fluid strategy: packages → trucks directly
        pack_util_fluid = float(container_params["pack_utilization_fluid"].iloc[0])
        effective_trailer_cube = raw_trailer_cube * pack_util_fluid

        # Calculate raw trucks needed
        raw_trucks = total_pkgs / packages_per_truck_capacity

        # Apply premium economy dwell logic
        if raw_trucks <= 1.0:
            # Always use at least 1 truck (never round to 0)
            final_trucks = 1
            packages_dwelled = 0
        else:
            fractional_part = raw_trucks - int(raw_trucks)
            if fractional_part < dwell_threshold:
                # Round down and dwell excess packages
                final_trucks = int(raw_trucks)
                missing_capacity = (raw_trucks - final_trucks) * packages_per_truck_capacity
                packages_dwelled = missing_capacity
            else:
                # Add the partial truck (fractional part >= threshold)
                final_trucks = int(np.ceil(raw_trucks))
                packages_dwelled = 0

        # No containers in fluid strategy
        physical_containers = 0
        container_fill_rate = 0.0

        # Truck fill rate uses raw capacity
        truck_fill_rate = min(1.0, total_cube / (final_trucks * raw_trailer_cube))

    # Calculate cube per truck
    cube_per_truck = total_cube / final_trucks if final_trucks > 0 else 0

    return {
        'physical_containers': physical_containers,
        'trucks_needed': final_trucks,
        'container_fill_rate': container_fill_rate,
        'truck_fill_rate': truck_fill_rate,
        'packages_dwelled': packages_dwelled,
        'total_cube_cuft': total_cube,
        'cube_per_truck': cube_per_truck,
    }


def resolve_legs_for_path(
        origin: str,
        dest: str,
        path_type: str,
        facilities: pd.DataFrame,
        bands: pd.DataFrame
) -> List[Tuple[str, str, float]]:
    """
    Generate legs for a path based on destination parent hub rules.

    DEPRECATED: This function is retained for backward compatibility but should
    not be used in production. Use build_structures.candidate_paths() instead.

    Args:
        origin: Origin facility name
        dest: Destination facility name
        path_type: Path classification (direct, 1_touch, etc.)
        facilities: Facility master data
        bands: Mileage bands

    Returns:
        List of tuples: (from_facility, to_facility, distance_miles)
    """
    facility_lookup = _facility_lookup(facilities)
    o, d = origin, dest

    parent_hub = facility_lookup.get(d, {}).get("parent", d)
    if pd.isna(parent_hub) or parent_hub == "":
        parent_hub = d

    def _leg(from_fac, to_fac):
        leg_metrics = compute_leg_metrics(from_fac, to_fac, facility_lookup, bands)
        return (from_fac, to_fac, leg_metrics["distance_miles"])

    if path_type == "direct":
        legs = [(o, d)]
    elif path_type == "1_touch":
        legs = [(o, d)] if parent_hub == d else [(o, parent_hub), (parent_hub, d)]
    else:
        if parent_hub == d:
            legs = [(o, d)]
        else:
            legs = [(o, parent_hub), (parent_hub, d)]

    return [_leg(from_fac, to_fac) for from_fac, to_fac in legs]


def containers_for_pkgs_day(
        pkgs_day: float,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame
) -> float:
    """
    Calculate containers needed for daily package volume.

    Args:
        pkgs_day: Daily package volume
        package_mix: Package mix distribution
        container_params: Container capacity parameters

    Returns:
        Number of containers required
    """
    weighted_cube = weighted_pkg_cube(package_mix)
    effective_cube = effective_gaylord_cube(container_params)
    return (pkgs_day * weighted_cube) / max(effective_cube, 1e-9)


def path_cost_and_time(
        row: pd.Series,
        facilities: pd.DataFrame,
        bands: pd.DataFrame,
        timing_kv: dict,
        cost_kv: dict,
        pkg_mix: pd.DataFrame,
        cont_params: pd.DataFrame,
        day_pkgs: float,
) -> Tuple[float, float, dict, list]:
    """
    DEPRECATED: Legacy path-level cost and time calculation.

    This function is retained ONLY for testing purposes .
    Production code uses MILP solver (solve_arc_pooled_path_selection) for all
    cost calculations with proper arc-level aggregation.

    DO NOT USE THIS FUNCTION IN PRODUCTION CODE.

    Args:
        row: Path data with origin, dest, path_nodes
        facilities: Facility master data
        bands: Mileage bands
        timing_kv: Timing parameters
        cost_kv: Cost parameters
        pkg_mix: Package mix distribution
        cont_params: Container parameters
        day_pkgs: Daily package volume

    Returns:
        Tuple of (total_cost, total_hours, summary_dict, steps_list)
    """
    facility_lookup = _facility_lookup(facilities)

    # Get timing parameters
    hours_per_touch = float(timing_kv["hours_per_touch"])
    load_hours = float(timing_kv["load_hours"])
    unload_hours = float(timing_kv["unload_hours"])
    strategy = str(timing_kv.get("load_strategy", cost_kv.get("load_strategy", "container"))).lower()

    # Get cost parameters
    sort_pp = float(cost_kv["sort_cost_per_pkg"])
    container_handling_cost = float(cost_kv["container_handling_cost"])
    last_mile_sort_pp = float(cost_kv["last_mile_sort_cost_per_pkg"])
    last_mile_delivery_pp = float(cost_kv["last_mile_delivery_cost_per_pkg"])
    dwell_cost_pp = float(cost_kv.get("dwell_cost_per_pkg_per_day", 0.0))
    sla_penalty_pp = float(cost_kv.get("sla_penalty_per_touch_per_pkg", 0.0))

    origin = row["origin"]
    dest = row["dest"]

    # Build path from nodes or resolve from path type
    nodes = row.get("path_nodes", None)
    if isinstance(nodes, list) and len(nodes) >= 2:
        # Validate no launch facilities as intermediate stops
        for intermediate_node in nodes[1:-1]:
            if intermediate_node in facility_lookup and facility_lookup[intermediate_node]["type"] == "launch":
                raise ValueError(f"Launch facility {intermediate_node} cannot be intermediate stop in path")
        pairs = list(zip(nodes[:-1], nodes[1:]))
        legs = pairs
    else:
        legs = [(from_fac, to_fac) for from_fac, to_fac, _ in
                resolve_legs_for_path(origin, dest, row["path_type"], facilities, bands)]

    # Calculate leg metrics
    leg_metrics = []
    total_distance = 0.0
    for from_fac, to_fac in legs:
        leg_data = compute_leg_metrics(from_fac, to_fac, facility_lookup, bands)
        leg_metrics.append(leg_data)
        total_distance += leg_data["distance_miles"]

    # Calculate truck requirements
    lane_od_pairs = [(row.to_dict(), day_pkgs)]
    truck_calc = enhanced_container_truck_calculation(
        lane_od_pairs, pkg_mix, cont_params, cost_kv, strategy
    )

    trucks_per_leg = truck_calc['trucks_needed']
    container_fill = truck_calc['container_fill_rate']
    truck_fill = truck_calc['truck_fill_rate']
    packages_dwelled = truck_calc['packages_dwelled']
    total_cube_cuft = truck_calc['total_cube_cuft']
    cube_per_truck = truck_calc['cube_per_truck']
    containers_needed = truck_calc['physical_containers']

    # Transportation cost
    base_trucking_cost = sum([leg["fixed"] + leg["var"] * leg["distance_miles"] for leg in leg_metrics])
    trucking_cost = base_trucking_cost * trucks_per_leg

    # Processing costs with strategy differentiation
    num_intermediate = max(len(legs) - 1, 0)

    if strategy == "container":
        # Container strategy
        origin_sort_cost = sort_pp * day_pkgs
        container_handling_touch_cost = num_intermediate * container_handling_cost * containers_needed
        last_mile_sort_cost = last_mile_sort_pp * day_pkgs
        last_mile_delivery_cost = last_mile_delivery_pp * day_pkgs
        total_processing_cost = origin_sort_cost + container_handling_touch_cost + last_mile_sort_cost + last_mile_delivery_cost
    else:
        # Fluid strategy
        origin_sort_cost = sort_pp * day_pkgs
        intermediate_sort_cost = num_intermediate * sort_pp * day_pkgs
        last_mile_sort_cost = last_mile_sort_pp * day_pkgs
        last_mile_delivery_cost = last_mile_delivery_pp * day_pkgs
        total_processing_cost = origin_sort_cost + intermediate_sort_cost + last_mile_sort_cost + last_mile_delivery_cost

    # Additional costs
    dwell_cost = packages_dwelled * dwell_cost_pp
    sla_penalty_cost = sla_penalty_pp * num_intermediate * day_pkgs if num_intermediate > 0 else 0.0

    total_cost = trucking_cost + total_processing_cost + dwell_cost + sla_penalty_cost

    # Timing calculations
    total_drive_hours = sum(leg["distance_miles"] / max(leg["mph"], 1e-6) for leg in leg_metrics)
    total_facilities = len(legs) + 1
    total_processing_hours = total_facilities * hours_per_touch
    dwell_hours = packages_dwelled / max(day_pkgs, 1e-9) * 24.0
    total_hours = total_drive_hours + total_processing_hours + dwell_hours
    sla_days = max(1, int(np.ceil(total_hours / 24.0)))

    # Generate step details
    steps = []
    for idx, ((from_fac, to_fac), leg) in enumerate(zip(legs, leg_metrics)):
        drive_hours = leg["distance_miles"] / max(leg["mph"], 1e-6)
        steps.append({
            "step_order": idx + 1,
            "from_facility": from_fac,
            "to_facility": to_fac,
            "distance_miles": leg["distance_miles"],
            "drive_hours": drive_hours,
            "processing_hours_at_destination": hours_per_touch,
            "facility_type": facility_lookup.get(to_fac, {}).get("type", "unknown"),
            "leg_cost": (leg["fixed"] + leg["var"] * leg["distance_miles"]) * trucks_per_leg,
            "trucks_on_leg": trucks_per_leg,
            "container_fill_rate": container_fill,
            "truck_fill_rate": truck_fill,
            "cube_cuft": total_cube_cuft / len(legs),
            "packages_dwelled": packages_dwelled / len(legs),
        })

    # Summary metrics
    summary_dict = {
        "distance_miles_total": total_distance,
        "linehaul_hours_total": total_drive_hours,
        "handling_hours_total": total_processing_hours,
        "dwell_hours_total": dwell_hours,
        "destination_dwell_hours": 0.0,
        "sla_days": sla_days,
        "total_trucks": trucks_per_leg * len(legs),
        "total_facilities_touched": total_facilities,
        "packages_dwelled": packages_dwelled,
        "physical_containers": containers_needed,
        "total_cube_cuft": total_cube_cuft,
        "cube_per_truck": cube_per_truck,
    }

    return total_cost, total_hours, summary_dict, steps