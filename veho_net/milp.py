# veho_net/milp.py - TARGETED FIX for cost recalculation in MILP solver
from ortools.sat.python import cp_model
import pandas as pd
from typing import Dict, Tuple, List
from .time_cost import enhanced_container_truck_calculation, weighted_pkg_cube
from .geo import haversine_miles, band_lookup


def _arc_cost_per_truck(u: str, v: str, facilities: pd.DataFrame, mileage_bands: pd.DataFrame) -> Tuple[float, float]:
    """Calculate distance and transportation cost for arc between two facilities."""
    fac = facilities.set_index("facility_name")[["lat", "lon"]].astype(float)
    lat1, lon1 = fac.at[u, "lat"], fac.at[u, "lon"]
    lat2, lon2 = fac.at[v, "lat"], fac.at[v, "lon"]
    raw = haversine_miles(lat1, lon1, lat2, lon2)
    fixed, var, circuit, _mph = band_lookup(raw, mileage_bands)
    dist = raw * circuit
    return dist, fixed + var * dist


def _legs_for_candidate(row: pd.Series, facilities: pd.DataFrame, mileage_bands: pd.DataFrame):
    """Extract leg information from candidate path for arc analysis."""
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
        timing_kv: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    FIXED: Solve path selection with proper strategy-specific cost recalculation.
    """
    cand = candidates.reset_index(drop=True).copy()
    strategy = str(cost_kv.get("load_strategy", "container")).lower()

    print(f"    MILP solver using {strategy} strategy with recalculated costs")

    # CRITICAL FIX: Recalculate costs for current strategy using proper parameter flow
    if "total_cost" in cand.columns:
        from .time_cost import path_cost_and_time

        # FIXED: Ensure strategy parameters flow correctly
        timing_kv_for_recalc = timing_kv.copy()
        timing_kv_for_recalc['load_strategy'] = strategy

        cost_kv_for_recalc = cost_kv.copy()
        cost_kv_for_recalc['load_strategy'] = strategy

        print(f"    Recalculating costs for {len(cand)} paths using {strategy} strategy...")
        print(f"    DEBUG: strategy={strategy}, sort_cost=${cost_kv_for_recalc.get('sort_cost_per_pkg', 'MISSING')}")

        # Track cost statistics for debugging
        original_costs = cand['total_cost'].copy()

        for idx, row in cand.iterrows():
            try:
                # FIXED: Recalculate cost with properly configured parameters
                total_cost, total_hours, sums, steps = path_cost_and_time(
                    row, facilities, mileage_bands, timing_kv_for_recalc, cost_kv_for_recalc,
                    package_mix, container_params, float(row["pkgs_day"])
                )

                # Update the path cost for current strategy
                cand.at[idx, 'total_cost'] = total_cost

                # Debug first few recalculations
                if idx < 3:
                    original_cost = original_costs.iloc[idx]
                    print(
                        f"    DEBUG path {idx}: {row.get('path_str', 'unknown')} - original=${original_cost:.0f}, recalc=${total_cost:.0f}")

            except Exception as e:
                print(f"    Error: Cost recalculation failed for path {idx}: {e}")
                raise ValueError(f"Cost recalculation failed for path {idx}: {e}")

        # Verify cost recalculation worked
        recalc_costs = cand['total_cost']
        cost_changed = not original_costs.equals(recalc_costs)
        print(f"    Cost recalculation completed. Costs changed: {cost_changed}")

        if not cost_changed:
            print(f"    WARNING: Costs did not change during recalculation!")
            print(f"    This suggests the strategy parameter is not flowing correctly.")

    # Validate required columns exist
    if "containers_cont" not in cand.columns:
        from .time_cost import containers_for_pkgs_day
        cand["containers_cont"] = cand["pkgs_day"].apply(
            lambda x: containers_for_pkgs_day(x, package_mix, container_params))

    path_keys = list(cand.index)

    # Build arc metadata and lane-level package aggregations
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

    # Calculate truck requirements per lane using input parameters only
    lane_truck_calc = {}
    for (u, v), od_list in lane_od_data.items():
        truck_calc = enhanced_container_truck_calculation(
            od_list, package_mix, container_params, cost_kv, strategy
        )
        lane_truck_calc[(u, v)] = truck_calc

    # Initialize CP-SAT optimization model
    model = cp_model.CpModel()

    # Path selection variables: choose exactly one path per OD group
    groups = cand.groupby(["scenario_id", "origin", "dest", "day_type"]).indices
    x = {i: model.NewBoolVar(f"x_{i}") for i in path_keys}
    for _, idxs in groups.items():
        model.Add(sum(x[i] for i in idxs) == 1)

    # Truck variables per arc
    SCALE = 1000
    y = {a_idx: model.NewIntVar(0, 10 ** 9, f"y_{a_idx}") for a_idx in range(len(arc_meta))}

    # Volume-to-truck constraints per arc
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
                print(f"    Warning: No truck calculation for lane {lane_key}")
                model.Add(y[a_idx] >= 1 * SCALE)  # Minimum 1 truck
        else:
            model.Add(y[a_idx] == 0)

    # Objective: minimize total recalculated path costs
    path_costs = {}
    for i in path_keys:
        path_costs[i] = int(float(cand.at[i, 'total_cost']) * SCALE)

    model.Minimize(sum(x[i] * path_costs[i] for i in path_keys))

    # Debug output to verify cost differences between strategies
    if len(cand) > 0:
        cost_range = f"${cand['total_cost'].min():.0f} - ${cand['total_cost'].max():.0f}"
        avg_cost = f"${cand['total_cost'].mean():.0f}"
        print(f"    Strategy-specific cost range: {cost_range} (avg: {avg_cost})")

    # Solve with reasonable time limit
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"    MILP solver failed with status: {status}")
        return pd.DataFrame(), pd.DataFrame()

    # Extract solution: selected paths
    chosen_idx = [i for i in path_keys if solver.Value(x[i]) == 1]
    selected_paths = cand.loc[chosen_idx].reset_index(drop=True)

    # Build arc summary with strategy-specific metrics
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
                    'container_fill_rate': 0.0,
                    'truck_fill_rate': 0.0,
                    'physical_containers': 0,
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

    # Create arc summary DataFrame
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