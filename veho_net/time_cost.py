# veho_net/time_cost.py - ENHANCED with improved container/truck fill logic
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
    Enhanced container/truck calculation with proper partial container handling.

    Args:
        lane_od_pairs: List of (od_dict, pkgs_day) for this lane
        package_mix: Package mix DataFrame
        container_params: Container parameters DataFrame
        cost_kv: Cost parameters dictionary
        strategy: 'container' or 'fluid'

    Returns:
        dict with truck calculation results including fill rates
    """

    # Calculate total volume for this lane
    total_pkgs = sum(pkgs for _, pkgs in lane_od_pairs)
    if total_pkgs <= 0:
        return {
            'physical_containers': 0,
            'trucks_needed': 0,
            'container_fill_rate': 0.0,
            'truck_fill_rate': 0.0,
            'packages_dwelled': 0,
            'total_cube': 0.0
        }

    w_cube = weighted_pkg_cube(package_mix)
    total_cube = total_pkgs * w_cube

    if strategy.lower() == "container":
        # Container strategy calculations
        eff_g_cube = effective_gaylord_cube(container_params, {})
        cpt = containers_per_truck(container_params)

        # Step 1: Calculate exact containers needed (no rounding)
        exact_containers = total_cube / max(eff_g_cube, 1e-9)

        # Step 2: Build actual containers (always round UP for physical containers)
        physical_containers = max(1, int(np.ceil(exact_containers)))
        container_fill_rate = exact_containers / physical_containers

        # Step 3: Calculate trucks with dwell logic
        raw_trucks = physical_containers / cpt

        # Get dwell parameters
        dwell_threshold = float(cost_kv.get('premium_economy_dwell_threshold', 0.10))
        allow_dwell = bool(cost_kv.get('allow_premium_economy_dwell', True))

        # Apply dwell logic
        if allow_dwell and (raw_trucks - int(raw_trucks)) < dwell_threshold:
            final_trucks = max(1, int(raw_trucks))  # Round down, packages dwell
            packages_dwelled = total_pkgs * max(0, (raw_trucks - final_trucks) / raw_trucks)
        else:
            final_trucks = max(1, int(np.ceil(raw_trucks)))  # Round up, send partial truck
            packages_dwelled = 0

        # Calculate truck fill rate
        truck_fill_rate = physical_containers / max(final_trucks * cpt, 1e-9)
        truck_fill_rate = min(1.0, truck_fill_rate)  # Cap at 100%

    else:
        # Fluid strategy calculations
        trailer_eff = trailer_effective_cube(container_params)

        # Direct truck calculation
        raw_trucks = total_cube / max(trailer_eff, 1e-9)

        # Get dwell parameters (same as container)
        dwell_threshold = float(cost_kv.get('premium_economy_dwell_threshold', 0.10))
        allow_dwell = bool(cost_kv.get('allow_premium_economy_dwell', True))

        if allow_dwell and (raw_trucks - int(raw_trucks)) < dwell_threshold:
            final_trucks = max(1, int(raw_trucks))
            packages_dwelled = total_pkgs * max(0, (raw_trucks - final_trucks) / raw_trucks)
        else:
            final_trucks = max(1, int(np.ceil(raw_trucks)))
            packages_dwelled = 0

        # Fluid doesn't use physical containers
        physical_containers = 0
        container_fill_rate = total_cube / max(final_trucks * trailer_eff, 1e-9)
        container_fill_rate = min(1.0, container_fill_rate)  # This is actually trailer fill rate
        truck_fill_rate = container_fill_rate  # Same thing for fluid

    return {
        'physical_containers': physical_containers,
        'trucks_needed': final_trucks,
        'container_fill_rate': container_fill_rate,
        'truck_fill_rate': truck_fill_rate,
        'packages_dwelled': packages_dwelled,
        'total_cube': total_cube,
        'total_packages': total_pkgs
    }


def smart_truck_rounding(volume_or_cube: float, truck_capacity: float,
                         dwell_threshold: float = 0.10) -> int:
    """
    DEPRECATED: Use enhanced_container_truck_calculation instead.
    Kept for backwards compatibility.
    """
    if truck_capacity <= 0:
        return 1  # Minimum 1 truck

    raw_trucks = volume_or_cube / truck_capacity
    if raw_trucks <= 1.0:
        return 1  # Always need at least 1 truck

    fractional_part = raw_trucks - int(raw_trucks)
    if fractional_part < dwell_threshold:
        return int(raw_trucks)  # Round down if under threshold
    else:
        return int(raw_trucks) + 1  # Round up


def resolve_legs_for_path(origin: str, dest: str, path_type: str, facilities: pd.DataFrame, bands: pd.DataFrame) -> \
        List[Tup[str, str, float]]:
    facL = _facility_lookup(facilities)
    o, d = origin, dest

    # Get parent hub safely
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
    """
    Enhanced cost/time calculation with improved container/truck logic:
    1. Uses enhanced container/truck calculations with proper fill rates
    2. Configurable dwell thresholds
    3. Proper cost allocation based on actual truck utilization
    4. Improved SLA calculation using hours per touch
    """
    facL = _facility_lookup(facilities)

    # Enhanced timing parameters
    hours_per_touch = float(timing_kv.get("hours_per_touch", 6.0))
    load_h = float(timing_kv.get("load_hours", 0.5))
    unload_h = float(timing_kv.get("unload_hours", 0.5))
    strategy = str(timing_kv.get("load_strategy", "container")).lower()

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

    # Enhanced truck calculation using new logic
    lane_od_pairs = [(row.to_dict(), day_pkgs)]  # Single OD for this path
    truck_calc = enhanced_container_truck_calculation(
        lane_od_pairs, pkg_mix, cont_params, cost_kv, strategy
    )

    trucks_per_leg = truck_calc['trucks_needed']
    container_fill = truck_calc['container_fill_rate']
    truck_fill = truck_calc['truck_fill_rate']
    packages_dwelled = truck_calc['packages_dwelled']

    # Transportation cost with actual truck utilization
    base_trucking_cost = sum([m["fixed"] + m["var"] * m["distance_miles"] for m in leg_metrics])
    trucking_cost = base_trucking_cost * trucks_per_leg

    # Processing costs with containerization awareness
    num_intermediate = max(len(legs) - 1, 0)

    if strategy == "container":
        # Container strategy: crossdock touches for intermediate, sort at origin/dest
        crossdock_touches = num_intermediate
        sort_touches = 2  # Origin sort + destination sort

        crossdock_cost = float(cost_kv.get("crossdock_touch_cost_per_pkg", 0.0)) * crossdock_touches * day_pkgs
        sort_cost = float(cost_kv.get("sort_cost_per_pkg", 0.0)) * sort_touches * day_pkgs

    else:
        # Fluid strategy: sort at every touch point
        sort_touches = len(legs) + 1  # Origin + each intermediate + destination
        crossdock_cost = 0.0
        sort_cost = float(cost_kv.get("sort_cost_per_pkg", 0.0)) * sort_touches * day_pkgs

    # Dwell cost for packages that wait
    dwell_cost = packages_dwelled * float(cost_kv.get("dwell_cost_per_pkg_per_day", 0.0))

    # SLA penalty for extra touches (discourages over-consolidation)
    sla_penalty_per_touch = float(cost_kv.get("sla_penalty_per_touch_per_pkg", 0.25))
    if num_intermediate > 0:
        sla_penalty_cost = sla_penalty_per_touch * num_intermediate * day_pkgs
    else:
        sla_penalty_cost = 0.0

    total_cost = trucking_cost + crossdock_cost + sort_cost + dwell_cost + sla_penalty_cost

    # Enhanced timing calculation
    total_drive_hours = sum(m["distance_miles"] / max(m["mph"], 1e-6) for m in leg_metrics)
    total_facilities = len(legs) + 1  # Number of facilities touched
    total_processing_hours = total_facilities * hours_per_touch

    # Add dwell time impact on SLA
    dwell_hours = packages_dwelled / max(day_pkgs, 1e-9) * 24.0  # Proportional dwell impact
    total_hours = total_drive_hours + total_processing_hours + dwell_hours

    # Simple SLA calculation in days
    sla_days = max(1, int(np.ceil(total_hours / 24.0)))

    # Generate enhanced step details
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
        })

    # Enhanced summary metrics
    sums = {
        "distance_miles_total": total_distance,
        "linehaul_hours_total": total_drive_hours,
        "handling_hours_total": total_processing_hours,
        "dwell_hours_total": dwell_hours,
        "destination_dwell_hours": 0.0,  # Could be enhanced later
        "sla_days": sla_days,
        "total_trucks": trucks_per_leg * len(legs),
        "total_facilities_touched": total_facilities,
        "container_fill_rate": container_fill,
        "truck_fill_rate": truck_fill,
        "packages_dwelled": packages_dwelled,
        "physical_containers": truck_calc['physical_containers'],
    }

    return total_cost, total_hours, sums, steps