# veho_net/milp.py - ENHANCED with lane-level truck optimization and improved fill logic
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
    """Yield (u,v,dist_miles) legs for a candidate. Honor path_nodes if provided; otherwise resolve by path_type."""
    nodes = row.get("path_nodes", None)
    if isinstance(nodes, list) and len(nodes) >= 2:
        pairs = list(zip(nodes[:-1], nodes[1:]))
        legs = []
        for u, v in pairs:
            dist, _ = _arc_cost_per_truck(u, v, facilities, mileage_bands)
            legs.append((u, v, dist))
        return legs
    # fallback: compute from path_type using enhanced logic
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
    Enhanced MILP solver with lane-level truck optimization and improved fill logic.

    Key improvements:
    1. Lane-level truck calculations with proper consolidation
    2. Enhanced container/truck fill rate modeling
    3. Dwell cost integration
    4. Support for all path types including 3_touch and 4_touch

    Inputs:
      candidates: rows with [scenario_id, origin, dest, day_type, path_type, pkgs_day, path_nodes, path_str, containers_cont]
    Returns:
      selected_paths: chosen one per (scenario_id,origin,dest,day_type) with the same cols as input
      arc_summary: per-arc totals with enhanced metrics including fill rates
    """
    cand = candidates.reset_index(drop=True).copy()

    # ensure containers_cont exists (backwards compatibility)
    if "containers_cont" not in cand.columns:
        from .time_cost import containers_for_pkgs_day
        cand["containers_cont"] = cand["pkgs_day"].apply(
            lambda x: containers_for_pkgs_day(x, package_mix, container_params))

    # index of paths
    path_keys = list(cand.index)

    # enumerate arcs per path and build lane-level data
    arc_index_map: Dict[Tuple[str, str], int] = {}
    arc_meta: List[Dict] = []  # idx -> {from,to,distance_miles,cost_per_truck}
    path_arcs: Dict[int, List[int]] = {}  # path_idx -> [arc_idx,...]
    lane_od_data: Dict[Tuple[str, str], List[Tuple]] = {}  # (u,v) -> [(od_dict, pkgs_day), ...]

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

            # Add this OD to the lane data for proper consolidation calculations
            lane_od_data[key].append((r.to_dict(), float(r["pkgs_day"])))
            ids.append(arc_index_map[key])

        path_arcs[i] = ids

    # Calculate enhanced truck requirements per lane
    lane_truck_calc = {}
    strategy = str(cost_kv.get("load_strategy", "container")).lower()

    for (u, v), od_list in lane_od_data.items():
        truck_calc = enhanced_container_truck_calculation(
            od_list, package_mix, container_params, cost_kv, strategy
        )
        lane_truck_calc[(u, v)] = truck_calc

    # CP-SAT model
    model = cp_model.CpModel()

    # choose exactly one path per OD group
    groups = cand.groupby(["scenario_id", "origin", "dest", "day_type"]).indices
    x = {i: model.NewBoolVar(f"x_{i}") for i in path_keys}
    for _, idxs in groups.items():
        model.Add(sum(x[i] for i in idxs) == 1)

    # Enhanced truck variables per arc (continuous approximation scaled to integers)
    SCALE = 1000
    y = {a_idx: model.NewIntVar(0, 10 ** 9, f"y_{a_idx}") for a_idx in range(len(arc_meta))}

    # Volume constraints per arc with enhanced logic
    for a_idx in range(len(arc_meta)):
        arc = arc_meta[a_idx]
        lane_key = (arc["from"], arc["to"])

        # Get all OD pairs that could use this lane
        terms = []
        for i in path_keys:
            if a_idx in path_arcs[i]:
                pkgs_scaled = int(round(float(cand.at[i, "pkgs_day"]) * SCALE))
                terms.append(pkgs_scaled * x[i])

        if terms:
            # Use the enhanced truck calculation for this lane
            if lane_key in lane_truck_calc:
                # Calculate truck capacity based on strategy and fill rates
                truck_calc = lane_truck_calc[lane_key]

                if strategy == "container":
                    cpt = int(container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0][
                                  "containers_per_truck"])
                    eff_cube = float(
                        container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0][
                            "usable_cube_cuft"])
                    util = float(container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0][
                                     "pack_utilization_container"])
                    capacity_per_truck = (eff_cube * util * cpt) / weighted_pkg_cube(package_mix)
                else:
                    trailer_cube = float(container_params.get("trailer_air_cube_cuft", pd.Series([4060.0])).iloc[0])
                    util = float(container_params.get("pack_utilization_fluid", pd.Series([0.85])).iloc[0])
                    capacity_per_truck = (trailer_cube * util) / weighted_pkg_cube(package_mix)

                model.Add(y[a_idx] * int(capacity_per_truck * SCALE) >= sum(terms))
            else:
                # Fallback to simple capacity constraint
                model.Add(y[a_idx] * 2000 * SCALE >= sum(terms))  # 2000 pkgs per truck default
        else:
            model.Add(y[a_idx] == 0)

    # Enhanced objective function with dwell and fill rate considerations
    crossdock_pp = float(cost_kv.get("crossdock_touch_cost_per_pkg", 0.0))
    sort_pp = float(cost_kv.get("sort_cost_per_pkg", 0.0))
    dwell_cost_pp = float(cost_kv.get("dwell_cost_per_pkg_per_day", 0.0))

    # Enhanced touch mapping to handle all path types
    touch_map = {
        "direct": 0,
        "1_touch": 1,
        "2_touch": 2,
        "3_touch": 3,
        "4_touch": 4
    }

    # Calculate processing costs with strategy awareness
    processing_cost = {}
    for i in path_keys:
        path_type = str(cand.at[i, "path_type"])
        pkgs_day = float(cand.at[i, "pkgs_day"])

        # Get number of touches with fallback for unknown path types
        if path_type in touch_map:
            num_touches = touch_map[path_type]
        else:
            # Fallback: estimate touches from path_nodes if available
            path_nodes = cand.at[i, "path_nodes"]
            if isinstance(path_nodes, list) and len(path_nodes) >= 2:
                num_touches = len(path_nodes) - 2  # Intermediate facilities
            else:
                print(f"Warning: Unknown path type '{path_type}', assuming 0 touches")
                num_touches = 0

        if strategy == "container":
            # Container: crossdock at intermediate, sort at origin/dest
            touch_cost = crossdock_pp * pkgs_day * num_touches
            sort_cost = sort_pp * pkgs_day * 2  # Origin + destination sort
        else:
            # Fluid: sort at every touch point
            sort_cost = sort_pp * pkgs_day * (num_touches + 2)  # Intermediate + origin + dest
            touch_cost = 0.0

        processing_cost[i] = touch_cost + sort_cost

    # Add dwell costs based on truck calculations
    dwell_cost_total = 0
    for lane_key, truck_calc in lane_truck_calc.items():
        dwell_cost_total += truck_calc['packages_dwelled'] * dwell_cost_pp

    # Minimize total cost: transportation + processing + dwell
    arc_cost = {a_idx: float(arc_meta[a_idx]["cost_per_truck"]) for a_idx in range(len(arc_meta))}

    model.Minimize(
        sum(y[a_idx] * arc_cost[a_idx] for a_idx in range(len(arc_meta)))
        + sum(x[i] * processing_cost[i] for i in path_keys)
        + int(dwell_cost_total * SCALE)  # Add dwell cost to objective
    )

    # Solve with enhanced parameters
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300.0  # 5 minute limit
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"MILP solver failed with status: {status}")
        # Return empty results rather than crashing
        return pd.DataFrame(), pd.DataFrame()

    # Map status to readable names
    status_names = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN"
    }
    status_name = status_names.get(status, f"UNKNOWN_STATUS_{status}")
    print(f"MILP solver completed with status: {status_name}")

    # extract chosen paths
    chosen_idx = [i for i in path_keys if solver.Value(x[i]) == 1]
    selected_paths = cand.loc[chosen_idx].reset_index(drop=True)

    print(f"Selected {len(selected_paths)} paths from {len(cand)} candidates")

    # Enhanced arc aggregation with fill rate metrics
    w_cube = weighted_pkg_cube(package_mix)
    trailer_air_cube = float(container_params.get("trailer_air_cube_cuft", pd.Series([4060.0])).iloc[0])

    agg = {}  # (u,v,scen,day) -> metrics
    for i, r in selected_paths.iterrows():
        pkgs = float(r["pkgs_day"])
        cube = pkgs * w_cube

        # Get the original index to map back to path_arcs
        original_idx = chosen_idx[i]

        for a_idx in path_arcs[original_idx]:
            meta = arc_meta[a_idx]
            lane_key = (meta["from"], meta["to"])
            key = (meta["from"], meta["to"], r["scenario_id"], r["day_type"])

            if key not in agg:
                # Get enhanced truck calculation for this lane
                truck_calc = lane_truck_calc.get(lane_key, {
                    'trucks_needed': 1,
                    'container_fill_rate': 0.8,
                    'truck_fill_rate': 0.8,
                    'physical_containers': 1,
                    'packages_dwelled': 0
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
                }

            agg[key]["pkgs_day"] += pkgs
            agg[key]["pkg_cube_cuft"] += cube

    # Build enhanced arc summary
    rows = []
    for (u, v, scen, day), val in agg.items():
        trucks = val["trucks_needed"]
        total_cost = trucks * val["cost_per_truck"]
        cpp = (total_cost / val["pkgs_day"]) if val["pkgs_day"] > 0 else 0.0

        # Enhanced metrics
        packages_per_truck = val["pkgs_day"] / max(trucks, 1e-9)

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

    # Validation output with enhanced metrics
    if not selected_paths.empty:
        path_type_counts = selected_paths["path_type"].value_counts()
        print("Selected path types:")
        for path_type, count in path_type_counts.items():
            print(f"  {path_type}: {count}")

    if not arc_summary.empty:
        total_trucks = arc_summary["trucks"].sum()
        avg_container_fill = arc_summary["container_fill_rate"].mean()
        avg_truck_fill = arc_summary["truck_fill_rate"].mean()
        total_dwelled = arc_summary["packages_dwelled"].sum()

        print(f"Enhanced arc summary: {len(arc_summary)} arcs")
        print(f"  Total trucks: {total_trucks:.1f}")
        print(f"  Avg container fill: {avg_container_fill:.1%}")
        print(f"  Avg truck fill: {avg_truck_fill:.1%}")
        print(f"  Total packages dwelled: {total_dwelled:,.0f}")

    return selected_paths, arc_summary