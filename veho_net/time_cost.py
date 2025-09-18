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
        hh, mm = parts; return time(int(hh), int(mm))
    if len(parts) == 3:
        hh, mm, ss = parts; return time(int(hh), int(mm), int(ss))
    raise ValueError(f"Unable to parse time '{val}'. Use HH:MM or HH:MM:SS.")

def _next_cpt(after_local_dt: datetime, cpt_t: time) -> datetime:
    cpt_today = after_local_dt.replace(hour=cpt_t.hour, minute=cpt_t.minute, second=cpt_t.second, microsecond=0)
    return cpt_today if after_local_dt <= cpt_today else cpt_today + timedelta(days=1)

def _safe_tz(name: Optional[str]) -> pytz.BaseTzInfo:
    if name is None or (isinstance(name, float) and pd.isna(name)): return pytz.UTC
    s = str(name).strip()
    try: return pytz.timezone(s)
    except Exception: return pytz.UTC

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
    util = float(container_params.get("pack_utilization_fluid", pd.Series([0.85])).iloc[0])
    return base * util

def compute_leg_metrics(fr: str, to: str, facL: dict, bands: pd.DataFrame) -> dict:
    lat1, lon1 = facL[fr]["lat"], facL[fr]["lon"]
    lat2, lon2 = facL[to]["lat"], facL[to]["lon"]
    raw = haversine_miles(lat1, lon1, lat2, lon2)
    fixed, var, circuit, mph = band_lookup(raw, bands)
    dist = raw * circuit
    return {"distance_miles": dist, "fixed": fixed, "var": var, "mph": mph}

def resolve_legs_for_path(origin: str, dest: str, path_type: str, facilities: pd.DataFrame, bands: pd.DataFrame) -> List[Tup[str, str, float]]:
    facL = _facility_lookup(facilities)
    o, d = origin, dest
    ph = facL[d]["parent"]

    def _leg(fr, to):
        m = compute_leg_metrics(fr, to, facL, bands)
        return (fr, to, m["distance_miles"])

    if path_type == "direct":
        legs = [(o, d)]
    elif path_type == "1_touch":
        legs = [(o, d)] if (pd.isna(ph) or ph == d) else [(o, ph), (ph, d)]
    else:
        if pd.isna(ph) or ph == d:
            # choose a nearby H1 elsewhere; simplified
            legs = [(o, d)]  # guarded upstream; in practice we pass nodes for 2_touch
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
    Returns: (total_cost_candidate, total_hours, detail_sums, step_rows)
    detail_sums includes:
      distance_miles_total, linehaul_hours_total, handling_hours_total,
      dwell_hours_total, destination_dwell_hours, sla_days
    """
    facL = _facility_lookup(facilities)

    # required inputs
    cpt_t = _parse_hhmm(timing_kv.get("cpt_hours_local", "02:00"))
    deliv_cutoff_t = _parse_hhmm(timing_kv.get("delivery_day_cutoff_local", "04:00"))
    load_h = float(timing_kv["load_hours"])
    unload_h = float(timing_kv["unload_hours"])
    sort_h = float(timing_kv["sort_hours_per_touch"])
    xdock_h = float(timing_kv["crossdock_hours_per_touch"])
    cutoff_h = float(timing_kv["departure_cutoff_hours_per_move"])
    strategy = str(timing_kv.get("load_strategy", "container")).lower()

    # cube + containerization
    w_cube = weighted_pkg_cube(pkg_mix)
    eff_g_cube = effective_gaylord_cube(cont_params, timing_kv)
    cpt = containers_per_truck(cont_params)
    trailer_eff = trailer_effective_cube(cont_params)

    o = row["origin"]; d = row["dest"]

    # legs: honor explicit node chain if provided
    nodes = row.get("path_nodes", None)
    if isinstance(nodes, list) and len(nodes) >= 2:
        pairs = list(zip(nodes[:-1], nodes[1:]))
        legs = pairs
    else:
        legs = [(fr, to) for fr, to, _ in resolve_legs_for_path(o, d, row["path_type"], facilities, bands)]

    # per-leg metrics
    leg_metrics = []
    total_distance = 0.0
    for fr, to in legs:
        m = compute_leg_metrics(fr, to, facL, bands)
        leg_metrics.append(m)
        total_distance += m["distance_miles"]

    # trucking cost candidate (per-OD ceiling; MILP will pool arcs)
    total_cube = max(day_pkgs, 0.0) * w_cube
    if strategy == "container":
        gaylords = int(np.ceil(total_cube / max(eff_g_cube, 1e-9)))
        trucks_per_leg = int(np.ceil(gaylords / max(cpt, 1)))
    else:
        trucks_per_leg = int(np.ceil(total_cube / max(trailer_eff, 1e-9)))
    trucking_cost = sum([m["fixed"] + m["var"] * m["distance_miles"] for m in leg_metrics]) * trucks_per_leg

    # touches cost candidate (OD-level)
    # touch type per node:
    # - origin: sort
    # - intermediate: container->crossdock; fluid->sort
    # - destination: sort
    num_intermediate = max(len(legs) - 1, 0)
    crossdock_touches = num_intermediate if strategy == "container" else 0
    sort_touches = 1 + (num_intermediate if strategy == "fluid" else 0) + 1  # origin + (maybe inter) + destination

    crossdock_cost = float(cost_kv.get("crossdock_touch_cost_per_pkg", 0.0)) * crossdock_touches * day_pkgs
    sort_cost = float(cost_kv.get("sort_cost_per_pkg", 0.0)) * sort_touches * day_pkgs
    total_cost = trucking_cost + crossdock_cost + sort_cost

    # timing with single CPT cadence + destination delivery cutoff
    linehaul_hours_total = 0.0
    handling_hours_total = 0.0
    dwell_hours_total = 0.0
    destination_dwell_hours = 0.0
    steps = []

    tz_o = _safe_tz(facL[o]["tz"])
    # Day 1 reference date (departures happen at CPT on this date)
    day1_date = datetime(2030, 1, 1, 0, 0, 0)
    depart_local = tz_o.localize(day1_date.replace(hour=cpt_t.hour, minute=cpt_t.minute, second=cpt_t.second))

    for idx, ((fr, to), m) in enumerate(zip(legs, leg_metrics)):
        tz_from = _safe_tz(facL[fr]["tz"])
        tz_to = _safe_tz(facL[to]["tz"])

        depart_local = depart_local.astimezone(tz_from)
        drive_h = m["distance_miles"] / max(m["mph"], 1e-6)
        arrive_local = depart_local + timedelta(hours=drive_h)
        linehaul_hours_total += drive_h

        is_dest = (to == d)
        # handling at arrival node
        if is_dest:
            unload = unload_h
            sort_here = sort_h  # always sort at D
            handling_here = unload + sort_here
            # destination dwell to delivery cutoff
            # if arrival <= cutoff -> same-day delivery (dwell 0), else to next day's cutoff
            cutoff_today = arrive_local.replace(hour=deliv_cutoff_t.hour, minute=deliv_cutoff_t.minute, second=deliv_cutoff_t.second, microsecond=0)
            if arrive_local <= cutoff_today:
                # deliver same day at cutoff (or we could treat as immediate; we’ll set dwell to time until cutoff to reflect staging)
                dest_dwell = max((cutoff_today - arrive_local).total_seconds() / 3600.0, 0.0)
                delivery_dt = cutoff_today
            else:
                next_cutoff = cutoff_today + timedelta(days=1)
                dest_dwell = (next_cutoff - arrive_local).total_seconds() / 3600.0
                delivery_dt = next_cutoff

            destination_dwell_hours += dest_dwell
            dwell = dest_dwell
            next_depart = None  # no depart from destination
        else:
            unload = unload_h
            xdock = xdock_h if strategy == "container" else 0.0
            sort_here = 0.0 if strategy == "container" else sort_h
            handling_here = unload + xdock + sort_here
            ready_time = arrive_local + timedelta(hours=handling_here + cutoff_h)
            next_depart = _next_cpt(ready_time, cpt_t)
            dwell = (next_depart - ready_time).total_seconds() / 3600.0

        handling_hours_total += handling_here
        dwell_hours_total += max(dwell, 0.0)

        steps.append({
            "step_order": idx + 1,
            "from_facility": fr, "to_facility": to,
            "band_mph": m["mph"], "distance_miles": m["distance_miles"], "drive_hours": drive_h,
            "arrive_local_ts": arrive_local.strftime("%Y-%m-%d %H:%M"),
            "unload_hours": unload,
            "crossdock_hours": (xdock_h if (not is_dest and strategy == "container") else 0.0),
            "sort_hours": (sort_h if (is_dest or (not is_dest and strategy == "fluid")) else 0.0),
            "cutoff_hours": float(cutoff_h if not is_dest else 0.0),
            "next_cpt_local_ts": (next_depart.strftime("%Y-%m-%d %H:%M") if next_depart is not None else ""),
            "dwell_hours_at_node": dwell,
        })

        if not is_dest:
            depart_local = next_depart.astimezone(tz_to)

    # SLA in days relative to Day 1 (cpt departure date)
    # find final delivery datetime from last step
    last = steps[-1]
    last_arrive = datetime.strptime(last["arrive_local_ts"], "%Y-%m-%d %H:%M")
    tz_d = _safe_tz(facL[d]["tz"])
    last_arrive = tz_d.localize(last_arrive).astimezone(tz_d)
    cutoff_today = last_arrive.replace(hour=deliv_cutoff_t.hour, minute=deliv_cutoff_t.minute, second=deliv_cutoff_t.second, microsecond=0)
    if last_arrive <= cutoff_today:
        delivery_dt = cutoff_today
    else:
        delivery_dt = cutoff_today + timedelta(days=1)
    sla_days = (delivery_dt.date() - day1_date.date()).days

    total_hours = linehaul_hours_total + handling_hours_total + dwell_hours_total
    sums = {
        "distance_miles_total": total_distance,
        "linehaul_hours_total": linehaul_hours_total,
        "handling_hours_total": handling_hours_total,
        "dwell_hours_total": dwell_hours_total,
        "destination_dwell_hours": destination_dwell_hours,
        "sla_days": int(sla_days),
    }
    return total_cost, total_hours, sums, steps
