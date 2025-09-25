# veho_net/milp.py - UPDATED VERSION with enhanced strategy differentiation
from ortools.sat.python import cp_model
import pandas as pd
from typing import Dict, Tuple, List
from .time_cost import enhanced_container_truck_calculation, weighted_pkg_cube
from .geo import haversine_miles, band_lookup


def _arc_cost_per_truck(u: str, v: str, facilities: pd.DataFrame, mileage_bands: pd.DataFrame) -> Tuple[float, float]:
    """Return (distance_miles, cost_per_truck) for arc u->v."""
    fac = facilities.set_index("facility_name")[["lat", "lon"]].astype(float)
    lat1, lon1 = fac.at[u, "lat"], fac.at[u, "lon"]
    lat2, lon2 = fac.at[v, "lat"], fac.at[v, "lon"]
    raw = haversine_miles(lat1, lon1, lat2, lon2)
    fixed, var, circuit, _mph = band_lookup(raw, mileage_bands)
    dist = raw * circuit
    return dist, fixed + var * dist


def _legs_for_candidate(row: pd.Series, facilities: pd.DataFrame, mileage_bands: pd.DataFrame):
    """Yield (u,v,dist_miles) legs for a candidate."""
    nodes = row.get("path_nodes", None)
    if isinstance(nodes, list) and len(nodes) >= 2:
        pairs = list(zip(nodes[:-1], nodes[1:]))
        legs = []
        for u, v in pairs:
            dist, _ = _arc_cost_per_truck(u, v, facilities, mileage_bands)
            legs.append((u, v, dist))
        return legs

    from .time_cost import resolve_legs_for_path
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
    FIXED: Enhanced MILP solver with proper strategy differentiation.
    """
    cand = candidates.reset_index(drop=True).copy()

    # CRITICAL FIX: Get strategy from cost_kv (now passed properly from run_v1.py)
    strategy = str(cost_kv.get("load_strategy", "container")).lower()
    print(f"    MILP solver using {strategy} strategy")

    # Ensure containers_cont exists
    if "containers_cont" not in cand.columns:
        from .time_cost import containers_for_pkgs_day
        cand["containers_cont"] = cand["pkgs_day"].apply(
            lambda x: containers_for_pkgs_day(x, package_mix, container_params))

    path_keys = list(cand.index)

    # Enumerate arcs per path and build lane-level data with STRATEGY AWARENESS
    arc_index_map: Dict[Tuple[str, str], int] = {}
    arc_meta: List[Dict] = []
    path_arcs: Dict[int, List[int]] = {}
    lane_od_data: Dict[Tuple[str, str], List[Tuple]] = {}

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
                lane_od_data[key] = []

            lane_od_data[key].append((r.to_dict(), float(r["pkgs_day"])))
            ids.append(arc_index_map[key])

        path_arcs[i] = ids

    # Calculate ENHANCED truck requirements per lane with STRATEGY DIFFERENTIATION
    lane_truck_calc = {}

    for (u, v), od_list in lane_od_data.items():
        # CRITICAL: Use the enhanced truck calculation with proper strategy
        truck_calc = enhanced_container_truck_calculation(
            od_list, package_mix, container_params, cost_kv, strategy
        )
        lane_truck_calc[(u, v)] = truck_calc

    # CP-SAT model
    model = cp_model.CpModel()

    # Choose exactly one path per OD group
    groups = cand.groupby(["scenario_id", "origin", "dest", "day_type"]).indices
    x = {i: model.NewBoolVar(f"x_{i}") for i in path_keys}
    for _, idxs in groups.items():
        model.Add(sum(x[i] for i in idxs) == 1)

    # Enhanced truck variables per arc
    SCALE = 1000
    y = {a_idx: model.NewIntVar(0, 10 ** 9, f"y_{a_idx}") for a_idx in range(len(arc_meta))}

    # ENHANCED volume constraints per arc with strategy-specific logic
    for a_idx in range(len(arc_meta)):
        arc = arc_meta[a_idx]
        lane_key = (arc["from"], arc["to"])

        terms = []
        for i in path_keys:
            if a_idx in path_arcs[i]:
                pkgs_scaled = int(round(float(cand.at[i, "pkgs_day"]) * SCALE))
                terms.append(pkgs_scaled * x[i])

        if terms:
            if lane_key in lane_truck_calc:
                truck_calc = lane_truck_calc[lane_key]
                required_trucks_scaled = int(round(truck_calc['trucks_needed'] * SCALE))
                model.Add(y[a_idx] >= required_trucks_scaled)
            else:
                # Fallback constraint based on strategy
                if strategy == "container":
                    # Container strategy: assume 2000 packages per truck
                    model.Add(y[a_idx] * 2000 * SCALE >= sum(terms))
                else:
                    # Fluid strategy: assume 1800 packages per truck (less efficient)
                    model.Add(y[a_idx] * 1800 * SCALE >= sum(terms))
        else:
            model.Add(y[a_idx] == 0)

    # Enhanced objective function - now using pre-calculated costs
    crossdock_pp = float(cost_kv.get("crossdock_touch_cost_per_pkg", 0.0))
    sort_pp = float(cost_kv.get("sort_cost_per_pkg", 0.0))
    dwell_cost_pp = float(cost_kv.get("dwell_cost_per_pkg_per_day", 0.0))

    # CRITICAL FIX: Use pre-calculated strategy-specific costs instead of recalculating
    # This ensures the MILP uses the exact costs calculated by path_cost_and_time()
    path_costs = {}
    for i in path_keys:
        # Use the total_cost calculated by path_cost_and_time() which is strategy-aware
        path_costs[i] = int(float(cand.at[i, 'total_cost']) * SCALE)

    print(f"    Using pre-calculated {strategy} strategy costs for path selection")

    # Simple objective: minimize total pre-calculated path costs
    model.Minimize(sum(x[i] * path_costs[i] for i in path_keys))

    # Solve with enhanced parameters
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return pd.DataFrame(), pd.DataFrame()

    # Extract chosen paths
    chosen_idx = [i for i in path_keys if solver.Value(x[i]) == 1]
    selected_paths = cand.loc[chosen_idx].reset_index(drop=True)

    # ENHANCED arc aggregation with strategy-specific metrics
    w_cube = weighted_pkg_cube(package_mix)

    agg = {}
    for i, r in selected_paths.iterrows():
        pkgs = float(r["pkgs_day"])
        cube = pkgs * w_cube

        original_idx = chosen_idx[i]

        for a_idx in path_arcs[original_idx]:
            meta = arc_meta[a_idx]
            lane_key = (meta["from"], meta["to"])
            key = (meta["from"], meta["to"], r["scenario_id"], r["day_type"])

            if key not in agg:
                truck_calc = lane_truck_calc.get(lane_key, {
                    'trucks_needed': 1,
                    'container_fill_rate': 0.5 if strategy == "container" else 0.0,
                    'truck_fill_rate': 0.6 if strategy == "container" else 0.7,  # Different defaults
                    'physical_containers': 1 if strategy == "container" else 0,
                    'packages_dwelled': 0,
                    'total_cube_cuft': 0,
                    'cube_per_truck': 0
                })

                agg[key] = {
                    "pkgs_day": 0.0,
                    "pkg_cube_cuft": 0.0,
                    "distance_miles": meta["distance_miles"],
                    "cost_per_truck": meta["cost_per_truck"],
                    "trucks_needed": truck_calc['trucks_needed'],
                    "container_fill_rate": truck_calc['container_fill_rate'],
                    "truck_fill_rate": truck_calc['truck_fill_rate'],
                    "physical_containers": truck_calc['physical_containers'],
                    "packages_dwelled": truck_calc['packages_dwelled'],
                    "total_cube_cuft": truck_calc['total_cube_cuft'],
                    "cube_per_truck": truck_calc['cube_per_truck'],
                }

            agg[key]["pkgs_day"] += pkgs
            agg[key]["pkg_cube_cuft"] += cube

    # Build ENHANCED arc summary with strategy-specific metrics
    rows = []
    for (u, v, scen, day), val in agg.items():
        trucks = val["trucks_needed"]
        total_cost = trucks * val["cost_per_truck"]
        cpp = (total_cost / val["pkgs_day"]) if val["pkgs_day"] > 0 else 0.0

        packages_per_truck = val["pkgs_day"] / max(trucks, 1e-9)
        actual_cube_per_truck = val["pkg_cube_cuft"] / max(trucks, 1e-9)

        rows.append({
            "scenario_id": scen,
            "day_type": day,
            "from_facility": u,
            "to_facility": v,
            "distance_miles": val["distance_miles"],
            "pkgs_day": val["pkgs_day"],
            "pkg_cube_cuft": val["pkg_cube_cuft"],
            "trucks": trucks,
            "physical_containers": val["physical_containers"],
            "packages_per_truck": packages_per_truck,
            "cube_per_truck": actual_cube_per_truck,
            "container_fill_rate": val["container_fill_rate"],
            "truck_fill_rate": val["truck_fill_rate"],
            "packages_dwelled": val["packages_dwelled"],
            "cost_per_truck": val["cost_per_truck"],
            "total_cost": total_cost,
            "CPP": cpp,
        })

    arc_summary = pd.DataFrame(rows).sort_values(
        ["scenario_id", "day_type", "from_facility", "to_facility"]
    ).reset_index(drop=True)

    return selected_paths, arc_summary