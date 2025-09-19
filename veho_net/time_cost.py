# veho_net/time_cost.py - COMPLETE FIXED VERSION
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


def smart_truck_rounding(volume_or_cube: float, truck_capacity: float, threshold: float = 0.10) -> int:
    """
    Smart truck rounding - if <10% of new truck is used, round down.
    Reflects operational reality where some packages can be delayed for better fill.
    """
    if truck_capacity <= 0:
        return 1  # Minimum 1 truck

    raw_trucks = volume_or_cube / truck_capacity
    if raw_trucks <= 1.0:
        return 1  # Always need at least 1 truck

    fractional_part = raw_trucks - int(raw_trucks)
    if fractional_part < threshold:
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
    Enhanced cost/time calculation with all fixes:
    1. Simplified SLA calculation using hours per touch
    2. Smart truck rounding to avoid marginal consolidation
    3. Launch facility constraints enforced
    4. SLA penalty for extra touches
    """
    facL = _facility_lookup(facilities)

    # Simplified timing parameters
    hours_per_touch = float(timing_kv.get("hours_per_touch", 6.0))
    load_h = float(timing_kv.get("load_hours", 0.5))
    unload_h = float(timing_kv.get("unload_hours", 0.5))
    strategy = str(timing_kv.get("load_strategy", "container")).lower()

    # Container/cube calculations
    w_cube = weighted_pkg_cube(pkg_mix)
    eff_g_cube = effective_gaylord_cube(cont_params, timing_kv)
    cpt = containers_per_truck(cont_params)
    trailer_eff = trailer_effective_cube(cont_params)

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

    # Smart truck calculation with rounding
    total_cube = max(day_pkgs, 0.0) * w_cube
    if strategy == "container":
        gaylords = max(1, total_cube / max(eff_g_cube, 1e-9))
        trucks_per_leg = smart_truck_rounding(gaylords, cpt)
    else:
        trucks_per_leg = smart_truck_rounding(total_cube, trailer_eff)

    # Transportation cost
    trucking_cost = sum([m["fixed"] + m["var"] * m["distance_miles"] for m in leg_metrics]) * trucks_per_leg

    # Processing costs
    num_intermediate = max(len(legs) - 1, 0)
    crossdock_touches = num_intermediate if strategy == "container" else 0
    sort_touches = 1 + (num_intermediate if strategy == "fluid" else 0) + 1  # origin + intermediate + destination

    crossdock_cost = float(cost_kv.get("crossdock_touch_cost_per_pkg", 0.0)) * crossdock_touches * day_pkgs
    sort_cost = float(cost_kv.get("sort_cost_per_pkg", 0.0)) * sort_touches * day_pkgs

    # SLA penalty for extra touches (discourages over-consolidation)
    sla_penalty_per_touch = float(cost_kv.get("sla_penalty_per_touch_per_pkg", 0.25))
    if num_intermediate > 0:
        sla_penalty_cost = sla_penalty_per_touch * num_intermediate * day_pkgs
    else:
        sla_penalty_cost = 0.0

    total_cost = trucking_cost + crossdock_cost + sort_cost + sla_penalty_cost

    # Simplified timing calculation
    total_drive_hours = sum(m["distance_miles"] / max(m["mph"], 1e-6) for m in leg_metrics)
    total_facilities = len(legs) + 1  # Number of facilities touched
    total_processing_hours = total_facilities * hours_per_touch
    total_hours = total_drive_hours + total_processing_hours

    # Simple SLA calculation in days
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
        })

    # Summary metrics
    sums = {
        "distance_miles_total": total_distance,
        "linehaul_hours_total": total_drive_hours,
        "handling_hours_total": total_processing_hours,
        "dwell_hours_total": 0.0,
        "destination_dwell_hours": 0.0,
        "sla_days": sla_days,
        "total_trucks": trucks_per_leg * len(legs),
        "total_facilities_touched": total_facilities,
    }

    return total_cost, total_hours, sums, steps