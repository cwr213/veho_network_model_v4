from ortools.sat.python import cp_model
import pandas as pd
from typing import Dict, Tuple, List
from .time_cost import resolve_legs_for_path, containers_for_pkgs_day, weighted_pkg_cube
from .geo import haversine_miles, band_lookup

def _arc_cost_per_truck(u: str, v: str, facilities: pd.DataFrame, mileage_bands: pd.DataFrame) -> Tuple[float, float]:
    """Return (distance_miles, cost_per_truck) for arc u->v."""
    fac = facilities.set_index("facility_name")[["lat","lon"]].astype(float)
    lat1, lon1 = fac.at[u, "lat"], fac.at[u, "lon"]
    lat2, lon2 = fac.at[v, "lat"], fac.at[v, "lon"]
    raw = haversine_miles(lat1, lon1, lat2, lon2)
    fixed, var, circuit, _mph = band_lookup(raw, mileage_bands)
    dist = raw * circuit
    return dist, fixed + var * dist

def _legs_for_candidate(row: pd.Series, facilities: pd.DataFrame, mileage_bands: pd.DataFrame):
    """Yield (u,v,dist_miles) legs for a candidate. Honor path_nodes if provided; otherwise resolve by path_type."""
    nodes = row.get("path_nodes", None)
    if isinstance(nodes, list) and len(nodes) >= 2:
        pairs = list(zip(nodes[:-1], nodes[1:]))
        legs = []
        for u, v in pairs:
            dist, _ = _arc_cost_per_truck(u, v, facilities, mileage_bands)
            legs.append((u, v, dist))
        return legs
    # fallback: compute from path_type
    return resolve_legs_for_path(row["origin"], row["dest"], row["path_type"], facilities, mileage_bands)

def solve_arc_pooled_path_selection(
    candidates: pd.DataFrame,
    facilities: pd.DataFrame,
    mileage_bands: pd.DataFrame,
    package_mix: pd.DataFrame,
    container_params: pd.DataFrame,
    cost_kv: Dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inputs:
      candidates: rows with [scenario_id, origin, dest, day_type, path_type, pkgs_day, path_nodes, path_str, containers_cont]
    Returns:
      selected_paths: chosen one per (scenario_id,origin,dest,day_type) with the same cols as input
      arc_summary: per-arc totals: [scenario_id, day_type, from_facility, to_facility,
                                    distance_miles, pkgs_day, pkg_cube_cuft, containers,
                                    trucks, avg_containers_per_truck, fill_rate,
                                    cost_per_truck, total_cost, CPP]
    """
    cand = candidates.reset_index(drop=True).copy()

    # ensure containers_cont exists
    if "containers_cont" not in cand.columns:
        cand["containers_cont"] = cand["pkgs_day"].apply(lambda x: containers_for_pkgs_day(x, package_mix, container_params))

    # index of paths
    path_keys = list(cand.index)

    # enumerate arcs per path
    arc_index_map: Dict[Tuple[str, str], int] = {}
    arc_meta: List[Dict] = []     # idx -> {from,to,distance_miles,cost_per_truck}
    path_arcs: Dict[int, List[int]] = {}   # path_idx -> [arc_idx,...]

    for i in path_keys:
        r = cand.loc[i]
        legs = _legs_for_candidate(r, facilities, mileage_bands)
        ids = []
        for (u, v, dist) in legs:
            key = (u, v)
            if key not in arc_index_map:
                d_mi, cpt = _arc_cost_per_truck(u, v, facilities, mileage_bands)
                arc_index_map[key] = len(arc_meta)
                arc_meta.append({"from": u, "to": v, "distance_miles": d_mi, "cost_per_truck": cpt})
            ids.append(arc_index_map[key])
        path_arcs[i] = ids

    # CP-SAT model
    model = cp_model.CpModel()

    # choose exactly one path per OD group
    groups = cand.groupby(["scenario_id", "origin", "dest", "day_type"]).indices
    x = {i: model.NewBoolVar(f"x_{i}") for i in path_keys}
    for _, idxs in groups.items():
        model.Add(sum(x[i] for i in idxs) == 1)

    # pooled trucks per arc (integer)
    cpt_capacity = int(container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0]["containers_per_truck"])
    y = {a_idx: model.NewIntVar(0, 10**9, f"y_{a_idx}") for a_idx in range(len(arc_meta))}

    # containers demand per arc (scaled to integers)
    SCALE = 1000
    cont_scaled = {i: int(round(float(cand.at[i, "containers_cont"]) * SCALE)) for i in path_keys}
    for a_idx in range(len(arc_meta)):
        terms = []
        for i in path_keys:
            if a_idx in path_arcs[i]:
                terms.append(cont_scaled[i] * x[i])
        if terms:
            model.Add(y[a_idx] * cpt_capacity * SCALE >= sum(terms))
        else:
            model.Add(y[a_idx] == 0)

    # objective: trucks cost on arcs + per-path touches/sort
    crossdock_pp = float(cost_kv.get("crossdock_touch_cost_per_pkg", 0.0))
    sort_pp = float(cost_kv.get("sort_cost_per_pkg", 0.0))
    touch_map = {"direct": 0, "1_touch": 1, "2_touch": 2}
    touch_cost = {i: crossdock_pp * float(cand.at[i, "pkgs_day"]) * touch_map[str(cand.at[i, "path_type"])] for i in path_keys}
    sort_cost_map = {i: sort_pp * float(cand.at[i, "pkgs_day"]) for i in path_keys}
    arc_cost = {a_idx: float(arc_meta[a_idx]["cost_per_truck"]) for a_idx in range(len(arc_meta))}

    model.Minimize(
        sum(y[a_idx] * arc_cost[a_idx] for a_idx in range(len(arc_meta)))
        + sum(x[i] * (touch_cost[i] + sort_cost_map[i]) for i in path_keys)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 180.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"Solver failed: status {status}")

    # extract chosen paths
    chosen_idx = [i for i in path_keys if solver.Value(x[i]) == 1]
    selected_paths = cand.loc[chosen_idx].reset_index(drop=True)

    # arc aggregation per (arc, scenario, day)
    w_cube = weighted_pkg_cube(package_mix)
    trailer_air_cube = float(container_params["trailer_air_cube_cuft"].iloc[0]) if "trailer_air_cube_cuft" in container_params.columns else 4060.0

    agg = {}  # (u,v,scen,day) -> metrics
    for i, r in selected_paths.iterrows():
        cont = float(r["containers_cont"])
        pkgs = float(r["pkgs_day"])
        cube = pkgs * w_cube
        for a_idx in path_arcs[chosen_idx[i]]:  # map back to original arc indices
            meta = arc_meta[a_idx]
            key = (meta["from"], meta["to"], r["scenario_id"], r["day_type"])
            if key not in agg:
                agg[key] = {
                    "pkgs_day": 0.0,
                    "containers": 0.0,
                    "pkg_cube_cuft": 0.0,
                    "distance_miles": meta["distance_miles"],
                    "cost_per_truck": meta["cost_per_truck"],
                }
            agg[key]["pkgs_day"] += pkgs
            agg[key]["containers"] += cont
            agg[key]["pkg_cube_cuft"] += cube

    rows = []
    for (u, v, scen, day), val in agg.items():
        trucks = val["containers"] / max(cpt_capacity, 1)
        avg_cont_per_truck = val["containers"] / max(trucks, 1e-9)
        fill_rate = val["pkg_cube_cuft"] / max(trucks * trailer_air_cube, 1e-9)
        total_cost = trucks * val["cost_per_truck"]
        cpp = (total_cost / val["pkgs_day"]) if val["pkgs_day"] > 0 else 0.0

        rows.append({
            "scenario_id": scen,
            "day_type": day,
            "from_facility": u,
            "to_facility": v,
            "distance_miles": val["distance_miles"],
            "pkgs_day": val["pkgs_day"],
            "pkg_cube_cuft": val["pkg_cube_cuft"],
            "containers": val["containers"],
            "trucks": trucks,
            "avg_containers_per_truck": avg_cont_per_truck,
            "fill_rate": fill_rate,
            "cost_per_truck": val["cost_per_truck"],
            "total_cost": total_cost,
            "CPP": cpp,
        })

    arc_summary = pd.DataFrame(rows).sort_values(
        ["scenario_id", "day_type", "from_facility", "to_facility"]
    ).reset_index(drop=True)

    return selected_paths, arc_summary
