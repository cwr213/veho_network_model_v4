# veho_net/time_cost.py - COMPLETE UPDATED VERSION with fixes
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import pytz
from typing import Tuple, Dict, Optional, List, Tuple as Tup
from .geo import haversine_miles, band_lookup


# ---------- helpers ----------

def _parse_hhmm(val) -> time:
    from datetime import time as _t
    if isinstance(val, _t): return val.replace(tzinfo=None)
    if hasattr(val, "to_pydatetime"):
        dt = val.to_pydatetime()
        return time(dt.hour, dt.minute, dt.second)
    if isinstance(val, (int, float)) and not pd.isna(val):
        total_minutes = int(round(float(val) * 24 * 60)) % (24 * 60)
        return time(total_minutes // 60, total_minutes % 60)
    raw = str(val).strip()
    if "," in raw: raw = raw.split(",")[0].strip()
    parts = raw.split(":")
    if len(parts) == 2:
        hh, mm = parts;
        return time(int(hh), int(mm))
    if len(parts) == 3:
        hh, mm, ss = parts;
        return time(int(hh), int(mm), int(ss))
    raise ValueError(f"Unable to parse time '{val}'. Use HH:MM or HH:MM:SS.")


def _facility_lookup(fac_df: pd.DataFrame) -> Dict[str, dict]:
    d = {}
    for _, r in fac_df.iterrows():
        d[r["facility_name"]] = {
            "type": r["type"],
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "tz": str(r["timezone"]),
            "parent": str(r["parent_hub_name"]),
        }
    return d


def effective_gaylord_cube(container_params: pd.DataFrame, timing_kv: dict) -> float:
    g = container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0]
    return float(g["usable_cube_cuft"]) * float(g["pack_utilization_container"])


def containers_per_truck(container_params: pd.DataFrame) -> int:
    g = container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0]
    return int(g["containers_per_truck"])


def weighted_pkg_cube(package_mix: pd.DataFrame) -> float:
    return float((package_mix["share_of_pkgs"] * package_mix["avg_cube_cuft"]).sum())


def trailer_effective_cube(container_params: pd.DataFrame) -> float:
    if "trailer_air_cube_cuft" in container_params.columns:
        base = float(container_params["trailer_air_cube_cuft"].iloc[0])
    else:
        base = 4060.0
    util_col = container_params.get("pack_utilization_fluid", pd.Series([0.85]))
    util = float(util_col.iloc[0] if len(util_col) > 0 else 0.85)
    return base * util


def compute_leg_metrics(fr: str, to: str, facL: dict, bands: pd.DataFrame) -> dict:
    lat1, lon1 = facL[fr]["lat"], facL[fr]["lon"]
    lat2, lon2 = facL[to]["lat"], facL[to]["lon"]
    raw = haversine_miles(lat1, lon1, lat2, lon2)
    fixed, var, circuit, mph = band_lookup(raw, bands)
    dist = raw * circuit
    return {"distance_miles": dist, "fixed": fixed, "var": var, "mph": mph}


def enhanced_container_truck_calculation(lane_od_pairs: List[Tuple], package_mix: pd.DataFrame,
                                         container_params: pd.DataFrame, cost_kv: dict,
                                         strategy: str = "container") -> dict:
    """
    FIXED: Proper container/truck calculation with strategy differentiation and realistic fill rates.

    Key fixes:
    1. Container vs Fluid strategies have different calculations
    2. Fill rates calculated against RAW capacity (not effective/theoretical)
    3. Proper dwell logic application
    4. Realistic truck requirements
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

    w_cube = weighted_pkg_cube(package_mix)
    total_cube = total_pkgs * w_cube

    if strategy.lower() == "container":
        # Container strategy: Uses gaylord containers in trucks
        gaylord_row = container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0]
        raw_container_cube = float(gaylord_row["usable_cube_cuft"])
        pack_util_container = float(gaylord_row["pack_utilization_container"])
        effective_container_cube = raw_container_cube * pack_util_container
        cpt = int(gaylord_row["containers_per_truck"])

        # Calculate containers needed based on effective space
        exact_containers = total_cube / effective_container_cube
        physical_containers = max(1, int(np.ceil(exact_containers)))

        # Calculate raw trucks needed
        raw_trucks = physical_containers / cpt

        # Apply dwell logic
        dwell_threshold = float(cost_kv.get('premium_economy_dwell_threshold', 0.10))
        allow_dwell = bool(cost_kv.get('allow_premium_economy_dwell', True))

        if raw_trucks <= 1.0:
            final_trucks = 1
            packages_dwelled = 0
        elif allow_dwell and raw_trucks > 1.0:
            fractional_part = raw_trucks - int(raw_trucks)
            if fractional_part < dwell_threshold:
                final_trucks = int(raw_trucks)
                missing_capacity = (raw_trucks - final_trucks) * cpt
                packages_dwelled = missing_capacity * effective_container_cube / w_cube
            else:
                final_trucks = int(np.ceil(raw_trucks))
                packages_dwelled = 0
        else:
            final_trucks = max(1, int(np.ceil(raw_trucks)))
            packages_dwelled = 0

        # CRITICAL FIX: Calculate fill rates against RAW capacity (not effective)
        # This gives realistic fill rates that operators would see
        container_fill_rate = min(1.0, total_cube / (physical_containers * raw_container_cube))
        raw_truck_cube = raw_container_cube * cpt
        truck_fill_rate = min(1.0, total_cube / (final_trucks * raw_truck_cube))

    else:
        # Fluid strategy: Direct loading into truck trailers
        if "trailer_air_cube_cuft" in container_params.columns:
            raw_truck_cube = float(container_params["trailer_air_cube_cuft"].iloc[0])
        else:
            raw_truck_cube = 4060.0  # Standard truck trailer cubic feet

        pack_util_fluid = float(container_params.get("pack_utilization_fluid", pd.Series([0.85])).iloc[0])
        effective_truck_cube = raw_truck_cube * pack_util_fluid

        raw_trucks = total_cube / effective_truck_cube

        # Apply dwell logic
        dwell_threshold = float(cost_kv.get('premium_economy_dwell_threshold', 0.10))
        allow_dwell = bool(cost_kv.get('allow_premium_economy_dwell', True))

        if raw_trucks <= 1.0:
            final_trucks = 1
            packages_dwelled = 0
        elif allow_dwell and raw_trucks > 1.0:
            fractional_part = raw_trucks - int(raw_trucks)
            if fractional_part < dwell_threshold:
                final_trucks = int(raw_trucks)
                missing_capacity = (raw_trucks - final_trucks) * effective_truck_cube
                packages_dwelled = missing_capacity / w_cube
            else:
                final_trucks = int(np.ceil(raw_trucks))
                packages_dwelled = 0
        else:
            final_trucks = max(1, int(np.ceil(raw_trucks)))
            packages_dwelled = 0

        # No containers in fluid strategy
        physical_containers = 0
        container_fill_rate = 0.0

        # CRITICAL FIX: Calculate truck fill rate against RAW capacity
        truck_fill_rate = min(1.0, total_cube / (final_trucks * raw_truck_cube))

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
        'total_packages': total_pkgs,
        'raw_container_cube': raw_container_cube if strategy.lower() == "container" else 0,
        'raw_truck_cube': raw_truck_cube,
        'effective_utilization_factor': pack_util_container if strategy.lower() == "container" else pack_util_fluid
    }


def calculate_fill_rate_comparison(total_cube: float, containers: int, trucks: int,
                                   container_params: pd.DataFrame, strategy: str) -> dict:
    """Calculate both realistic and theoretical fill rates for comparison."""

    if strategy.lower() == "container":
        gaylord_row = container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0]
        raw_container_cube = float(gaylord_row["usable_cube_cuft"])
        pack_util_container = float(gaylord_row["pack_utilization_container"])
        effective_container_cube = raw_container_cube * pack_util_container
        cpt = int(gaylord_row["containers_per_truck"])

        container_fill_realistic = (total_cube / containers) / raw_container_cube if containers > 0 else 0
        container_fill_theoretical = (total_cube / containers) / effective_container_cube if containers > 0 else 0

        raw_truck_cube = raw_container_cube * cpt
        effective_truck_cube = effective_container_cube * cpt

        truck_fill_realistic = total_cube / (trucks * raw_truck_cube) if trucks > 0 else 0
        truck_fill_theoretical = containers / (trucks * cpt) if trucks > 0 else 0

        return {
            'container_fill_realistic': min(1.0, container_fill_realistic),
            'container_fill_theoretical': min(1.0, container_fill_theoretical),
            'truck_fill_realistic': min(1.0, truck_fill_realistic),
            'truck_fill_theoretical': min(1.0, truck_fill_theoretical),
            'raw_container_cube': raw_container_cube,
            'effective_container_cube': effective_container_cube,
            'pack_utilization_factor': pack_util_container
        }
    else:
        raw_truck_cube = float(container_params.get("trailer_air_cube_cuft", pd.Series([4060.0])).iloc[0])
        pack_util_fluid = float(container_params.get("pack_utilization_fluid", pd.Series([0.85])).iloc[0])
        effective_truck_cube = raw_truck_cube * pack_util_fluid

        truck_fill_realistic = total_cube / (trucks * raw_truck_cube) if trucks > 0 else 0
        truck_fill_theoretical = total_cube / (trucks * effective_truck_cube) if trucks > 0 else 0

        return {
            'container_fill_realistic': 0.0,
            'container_fill_theoretical': 0.0,
            'truck_fill_realistic': min(1.0, truck_fill_realistic),
            'truck_fill_theoretical': min(1.0, truck_fill_theoretical),
            'raw_truck_cube': raw_truck_cube,
            'effective_truck_cube': effective_truck_cube,
            'pack_utilization_factor': pack_util_fluid
        }


def smart_truck_rounding(volume_or_cube: float, truck_capacity: float,
                         dwell_threshold: float = 0.10) -> int:
    """DEPRECATED: Use enhanced_container_truck_calculation instead."""
    if truck_capacity <= 0:
        return 1

    raw_trucks = volume_or_cube / truck_capacity
    if raw_trucks <= 1.0:
        return 1

    fractional_part = raw_trucks - int(raw_trucks)
    if fractional_part < dwell_threshold:
        return max(1, int(raw_trucks))
    else:
        return int(raw_trucks) + 1


def resolve_legs_for_path(origin: str, dest: str, path_type: str, facilities: pd.DataFrame, bands: pd.DataFrame) -> \
        List[Tup[str, str, float]]:
    facL = _facility_lookup(facilities)
    o, d = origin, dest

    ph = facL.get(d, {}).get("parent", d)
    if pd.isna(ph) or ph == "":
        ph = d

    def _leg(fr, to):
        m = compute_leg_metrics(fr, to, facL, bands)
        return (fr, to, m["distance_miles"])

    if path_type == "direct":
        legs = [(o, d)]
    elif path_type == "1_touch":
        legs = [(o, d)] if ph == d else [(o, ph), (ph, d)]
    else:
        if ph == d:
            legs = [(o, d)]
        else:
            legs = [(o, ph), (ph, d)]
    return [_leg(fr, to) for fr, to in legs]


def containers_for_pkgs_day(pkgs_day: float, package_mix: pd.DataFrame, container_params: pd.DataFrame) -> float:
    w_cube = weighted_pkg_cube(package_mix)
    eff_cube = effective_gaylord_cube(container_params, {})
    return (pkgs_day * w_cube) / max(eff_cube, 1e-9)


# ---------- main cost/time calculator ----------

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
    """Enhanced cost/time calculation with container handling logic."""
    facL = _facility_lookup(facilities)

    # Enhanced timing parameters
    hours_per_touch = float(timing_kv.get("hours_per_touch", 6.0))
    load_h = float(timing_kv.get("load_hours", 0.5))
    unload_h = float(timing_kv.get("unload_hours", 0.5))
    strategy = str(timing_kv.get("load_strategy", "container")).lower()

    # Get cost parameters - NO HARDCODED VALUES
    sort_pp = float(cost_kv.get("sort_cost_per_pkg", 0.0))
    container_handling_cost = float(cost_kv.get("container_handling_cost", 0.0))
    last_mile_sort_pp = float(cost_kv.get("last_mile_sort_cost_per_pkg", 0.0))
    last_mile_delivery_pp = float(cost_kv.get("last_mile_delivery_cost_per_pkg", 0.0))
    dwell_cost_pp = float(cost_kv.get("dwell_cost_per_pkg_per_day", 0.0))
    sla_penalty_pp = float(cost_kv.get("sla_penalty_per_touch_per_pkg", 0.0))

    o = row["origin"]
    d = row["dest"]

    # Validate and build path
    nodes = row.get("path_nodes", None)
    if isinstance(nodes, list) and len(nodes) >= 2:
        # Enforce launch facilities cannot be intermediate stops
        for intermediate_node in nodes[1:-1]:
            if intermediate_node in facL and facL[intermediate_node]["type"] == "launch":
                raise ValueError(f"Launch facility {intermediate_node} cannot be intermediate stop in path")
        pairs = list(zip(nodes[:-1], nodes[1:]))
        legs = pairs
    else:
        legs = [(fr, to) for fr, to, _ in resolve_legs_for_path(o, d, row["path_type"], facilities, bands)]

    # Calculate leg metrics
    leg_metrics = []
    total_distance = 0.0
    for fr, to in legs:
        m = compute_leg_metrics(fr, to, facL, bands)
        leg_metrics.append(m)
        total_distance += m["distance_miles"]

    # Enhanced truck calculation using realistic fill rates
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

    # Transportation cost with actual truck utilization
    base_trucking_cost = sum([m["fixed"] + m["var"] * m["distance_miles"] for m in leg_metrics])
    trucking_cost = base_trucking_cost * trucks_per_leg

    # Processing costs with NEW CONTAINER HANDLING LOGIC
    num_intermediate = max(len(legs) - 1, 0)

    if strategy == "container":
        # Container strategy: sort at origin, container handling at intermediate touches, last mile at destination

        # Origin sort cost
        origin_sort_cost = sort_pp * day_pkgs

        # Container handling cost at intermediate touches
        container_handling_touch_cost = num_intermediate * container_handling_cost * containers_needed

        # Last mile costs
        last_mile_sort_cost = last_mile_sort_pp * day_pkgs
        last_mile_delivery_cost = last_mile_delivery_pp * day_pkgs

        total_processing_cost = origin_sort_cost + container_handling_touch_cost + last_mile_sort_cost + last_mile_delivery_cost

    else:
        # Fluid strategy: sort at every facility

        # Origin sort
        origin_sort_cost = sort_pp * day_pkgs

        # Intermediate sort costs
        intermediate_sort_cost = num_intermediate * sort_pp * day_pkgs

        # Last mile costs
        last_mile_sort_cost = last_mile_sort_pp * day_pkgs
        last_mile_delivery_cost = last_mile_delivery_pp * day_pkgs

        total_processing_cost = origin_sort_cost + intermediate_sort_cost + last_mile_sort_cost + last_mile_delivery_cost

    # Dwell cost for packages that wait
    dwell_cost = packages_dwelled * dwell_cost_pp

    # SLA penalty for extra touches
    if num_intermediate > 0:
        sla_penalty_cost = sla_penalty_pp * num_intermediate * day_pkgs
    else:
        sla_penalty_cost = 0.0

    total_cost = trucking_cost + total_processing_cost + dwell_cost + sla_penalty_cost

    # Enhanced timing calculation
    total_drive_hours = sum(m["distance_miles"] / max(m["mph"], 1e-6) for m in leg_metrics)
    total_facilities = len(legs) + 1
    total_processing_hours = total_facilities * hours_per_touch

    # Add dwell time impact on SLA
    dwell_hours = packages_dwelled / max(day_pkgs, 1e-9) * 24.0
    total_hours = total_drive_hours + total_processing_hours + dwell_hours

    sla_days = max(1, int(np.ceil(total_hours / 24.0)))

    # Generate step details
    steps = []
    for idx, ((fr, to), m) in enumerate(zip(legs, leg_metrics)):
        drive_h = m["distance_miles"] / max(m["mph"], 1e-6)

        steps.append({
            "step_order": idx + 1,
            "from_facility": fr,
            "to_facility": to,
            "distance_miles": m["distance_miles"],
            "drive_hours": drive_h,
            "processing_hours_at_destination": hours_per_touch,
            "facility_type": facL.get(to, {}).get("type", "unknown"),
            "leg_cost": (m["fixed"] + m["var"] * m["distance_miles"]) * trucks_per_leg,
            "trucks_on_leg": trucks_per_leg,
            "container_fill_rate": container_fill,
            "truck_fill_rate": truck_fill,
            "cube_cuft": total_cube_cuft / len(legs),
            "packages_dwelled": packages_dwelled / len(legs),
        })

    # Enhanced summary metrics with realistic fill rates and cubic metrics
    sums = {
        "distance_miles_total": total_distance,
        "linehaul_hours_total": total_drive_hours,
        "handling_hours_total": total_processing_hours,
        "dwell_hours_total": dwell_hours,
        "destination_dwell_hours": 0.0,
        "sla_days": sla_days,
        "total_trucks": trucks_per_leg * len(legs),
        "total_facilities_touched": total_facilities,
        "container_fill_rate": container_fill,
        "truck_fill_rate": truck_fill,
        "packages_dwelled": packages_dwelled,
        "physical_containers": containers_needed,
        "total_cube_cuft": total_cube_cuft,
        "cube_per_truck": cube_per_truck,
        "raw_container_cube": truck_calc.get('raw_container_cube', 0),
        "raw_truck_cube": truck_calc.get('raw_truck_cube', 0),
        "effective_utilization_factor": truck_calc.get('effective_utilization_factor', 0.85),
    }

    return total_cost, total_hours, sums, steps