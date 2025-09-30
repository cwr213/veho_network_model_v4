"""
MILP Optimization Module

Solves arc-pooled path selection with optional sort optimization.
All cost calculations happen within MILP with proper arc aggregation.

Key Features:
- Path selection: Choose optimal path for each OD pair
- Arc aggregation: Pool volumes across paths sharing lanes
- Truck calculation: Based on effective cube capacity
- Strategy handling: Respects strategy_hint or uses global strategy
- Optional: Multi-level sort optimization with capacity constraints
"""

from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from .config_v3 import CostParameters, LoadStrategy
from .containers_v3 import weighted_pkg_cube, calculate_truck_capacity
from .geo_v3 import haversine_miles, band_lookup


def solve_network_optimization(
        candidates: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        cost_params: CostParameters,
        timing_params: Dict,
        global_strategy: LoadStrategy,
        enable_sort_optimization: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Optional[pd.DataFrame]]:
    """
    Solve network optimization using arc-pooled MILP.

    Args:
        candidates: Candidate paths with origin, dest, path_nodes, strategy_hint
        facilities: Facility master data
        mileage_bands: Distance-based parameters
        package_mix: Package distribution
        container_params: Container/trailer capacity
        cost_params: Cost parameters (CostParameters dataclass)
        timing_params: Timing parameters dict
        global_strategy: Default loading strategy
        enable_sort_optimization: Enable multi-level sort optimization

    Returns:
        Tuple of (selected_paths, arc_summary, network_kpis, sort_summary)
    """
    cand = candidates.reset_index(drop=True).copy()

    print(f"    MILP optimization: {global_strategy.value} strategy (global)")
    print(f"    Sort optimization: {'ENABLED' if enable_sort_optimization else 'DISABLED'}")
    print(f"    Candidate paths: {len(cand)}")

    path_keys = list(cand.index)
    w_cube = weighted_pkg_cube(package_mix)

    # Build arc metadata and path-to-arc mapping
    arc_index_map: Dict[Tuple[str, str], int] = {}
    arc_meta: List[Dict] = []
    path_arcs: Dict[int, List[int]] = {}
    path_od_data: Dict[int, Dict] = {}

    for i in path_keys:
        r = cand.loc[i]
        legs = _extract_legs_from_path(r, facilities, mileage_bands)

        path_nodes = r.get("path_nodes", [r["origin"], r["dest"]])
        if not isinstance(path_nodes, list):
            path_nodes = [r["origin"], r["dest"]]

        # Determine effective strategy for this path
        effective_strategy = r.get("strategy_hint") if r.get("strategy_hint") else global_strategy.value

        # Store path metadata
        path_od_data[i] = {
            'origin': r["origin"],
            'dest': r["dest"],
            'pkgs_day': float(r["pkgs_day"]),
            'scenario_id': r.get("scenario_id", "default"),
            'day_type': r.get("day_type", "peak"),
            'path_str': r.get("path_str", f"{r['origin']}->{r['dest']}"),
            'path_type': r.get("path_type", "direct"),
            'path_nodes': path_nodes,
            'effective_strategy': effective_strategy
        }

        # Map path to arcs
        arc_ids = []
        for (u, v, dist, cost_per_truck, mph) in legs:
            key = (u, v)
            if key not in arc_index_map:
                arc_index_map[key] = len(arc_meta)
                arc_meta.append({
                    "from": u,
                    "to": v,
                    "distance_miles": dist,
                    "cost_per_truck": cost_per_truck,
                    "mph": mph
                })
            arc_ids.append(arc_index_map[key])

        path_arcs[i] = arc_ids

    print(f"    Generated {len(arc_meta)} unique arcs from {len(cand)} paths")

    # Initialize CP-SAT model
    model = cp_model.CpModel()

    # Path selection variables: exactly one path per OD group
    groups = cand.groupby(["scenario_id", "origin", "dest", "day_type"]).indices
    x = {i: model.NewBoolVar(f"x_{i}") for i in path_keys}

    for group_name, idxs in groups.items():
        model.Add(sum(x[i] for i in idxs) == 1)

    print(f"    Created {len(groups)} OD selection constraints")

    # Arc volume variables
    arc_pkgs = {a_idx: model.NewIntVar(0, 1000000, f"arc_pkgs_{a_idx}")
                for a_idx in range(len(arc_meta))}

    # Link path selection to arc volumes
    for a_idx in range(len(arc_meta)):
        terms = []
        for i in path_keys:
            if a_idx in path_arcs[i]:
                pkgs = int(round(path_od_data[i]['pkgs_day']))
                terms.append(pkgs * x[i])

        if terms:
            model.Add(arc_pkgs[a_idx] == sum(terms))
        else:
            model.Add(arc_pkgs[a_idx] == 0)

    # Calculate truck requirements per arc based on effective strategy
    # For now, use global strategy for truck calculations
    # (Per-arc strategy would require more complex formulation)

    raw_trailer_cube = float(container_params["trailer_air_cube_cuft"].iloc[0])

    if global_strategy == LoadStrategy.CONTAINER:
        gaylord_row = container_params[
            container_params["container_type"].str.lower() == "gaylord"
            ].iloc[0]
        raw_container_cube = float(gaylord_row["usable_cube_cuft"])
        pack_util_container = float(gaylord_row["pack_utilization_container"])
        effective_container_cube = raw_container_cube * pack_util_container
        containers_per_truck = int(gaylord_row["containers_per_truck"])
        effective_truck_cube = containers_per_truck * effective_container_cube
    else:  # FLUID
        pack_util_fluid = float(container_params["pack_utilization_fluid"].iloc[0])
        effective_truck_cube = raw_trailer_cube * pack_util_fluid

    effective_truck_cube_scaled = int(effective_truck_cube * 1000)
    w_cube_scaled = int(w_cube * 1000)

    # Arc truck variables
    arc_trucks = {a_idx: model.NewIntVar(0, 1000, f"arc_trucks_{a_idx}")
                  for a_idx in range(len(arc_meta))}

    for a_idx in range(len(arc_meta)):
        # Truck capacity constraint
        model.Add(arc_trucks[a_idx] * effective_truck_cube_scaled >=
                  arc_pkgs[a_idx] * w_cube_scaled)

        # Minimum 1 truck if any packages
        arc_has_pkgs = model.NewBoolVar(f"arc_has_pkgs_{a_idx}")
        BIG_M = 1000000
        model.Add(arc_pkgs[a_idx] <= BIG_M * arc_has_pkgs)
        model.Add(arc_pkgs[a_idx] >= 1 * arc_has_pkgs)
        model.Add(arc_trucks[a_idx] >= arc_has_pkgs)

    # Objective: Minimize total cost (transportation + processing)
    cost_terms = []

    # 1. Transportation costs (arc-level)
    for a_idx in range(len(arc_meta)):
        arc = arc_meta[a_idx]
        truck_cost = int(arc["cost_per_truck"])
        cost_terms.append(arc_trucks[a_idx] * truck_cost)

    # 2. Processing costs (path-level, strategy-dependent)
    for i in path_keys:
        path_data = path_od_data[i]
        volume = int(path_data['pkgs_day'])
        effective_strategy = path_data['effective_strategy']

        # Calculate processing cost based on effective strategy
        processing_cost = _calculate_processing_cost(
            path_data['path_nodes'],
            effective_strategy,
            cost_params,
            facilities,
            package_mix,
            container_params
        )

        cost_terms.append(x[i] * int(processing_cost * volume))

    model.Minimize(sum(cost_terms))

    print(f"    Objective: {len([t for t in cost_terms if 'arc_trucks' in str(t)])} transportation + "
          f"{len([t for t in cost_terms if 'x[' in str(t)])} processing terms")

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300.0
    solver.parameters.num_search_workers = 8

    print(f"    Starting MILP solver...")
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"    ❌ MILP solver failed with status: {status}")
        return pd.DataFrame(), pd.DataFrame(), {}, None

    print(f"    ✅ MILP solver completed: {status}")
    print(f"    Total optimized cost: ${solver.ObjectiveValue():,.0f}")

    # Extract solution
    chosen_idx = [i for i in path_keys if solver.Value(x[i]) == 1]

    # Build selected paths dataframe
    selected_paths_data = []

    for i in chosen_idx:
        path_data = path_od_data[i]

        # Calculate costs for this path
        processing_cost = _calculate_processing_cost(
            path_data['path_nodes'],
            path_data['effective_strategy'],
            cost_params,
            facilities,
            package_mix,
            container_params
        ) * path_data['pkgs_day']

        # Calculate transportation cost (allocate based on package share)
        transport_cost = 0
        for a_idx in path_arcs[i]:
            arc = arc_meta[a_idx]
            trucks = solver.Value(arc_trucks[a_idx])
            if trucks > 0:
                arc_total_pkgs = solver.Value(arc_pkgs[a_idx])
                path_share = path_data['pkgs_day'] / max(arc_total_pkgs, 1e-9)
                transport_cost += trucks * arc['cost_per_truck'] * path_share

        total_cost = transport_cost + processing_cost

        selected_paths_data.append({
            **path_data,
            'total_cost': total_cost,
            'linehaul_cost': transport_cost,
            'processing_cost': processing_cost,
            'cost_per_pkg': total_cost / path_data['pkgs_day']
        })

    selected_paths = pd.DataFrame(selected_paths_data)

    # Build arc summary with fill rates
    arc_summary_data = []
    dwell_threshold = cost_params.premium_economy_dwell_threshold

    for a_idx in range(len(arc_meta)):
        arc = arc_meta[a_idx]
        pkgs = solver.Value(arc_pkgs[a_idx])
        trucks = solver.Value(arc_trucks[a_idx])

        if pkgs > 0:
            total_cost = trucks * arc['cost_per_truck']
            cube = pkgs * w_cube

            # Calculate fill rates using raw capacities
            if global_strategy == LoadStrategy.CONTAINER:
                gaylord_row = container_params[
                    container_params["container_type"].str.lower() == "gaylord"
                    ].iloc[0]
                raw_container_cube = float(gaylord_row["usable_cube_cuft"])
                pack_util = float(gaylord_row["pack_utilization_container"])
                effective_container_cube = raw_container_cube * pack_util
                containers_per_truck_val = int(gaylord_row["containers_per_truck"])

                actual_containers = max(1, int(np.ceil(cube / effective_container_cube)))
                container_fill_rate = cube / (actual_containers * raw_container_cube)
                truck_fill_rate = cube / (trucks * raw_trailer_cube)
            else:  # FLUID
                container_fill_rate = 0.0
                actual_containers = 0
                truck_fill_rate = cube / (trucks * raw_trailer_cube)

            # Calculate dwell based on premium economy threshold
            exact_trucks_needed = cube / effective_truck_cube

            if exact_trucks_needed <= 1.0:
                packages_dwelled = 0
            else:
                fractional_part = exact_trucks_needed - int(exact_trucks_needed)
                if fractional_part > 0 and fractional_part < dwell_threshold:
                    optimal_trucks = int(exact_trucks_needed)
                    if trucks == optimal_trucks:
                        excess_cube = cube - (trucks * effective_truck_cube)
                        packages_dwelled = max(0, excess_cube / w_cube)
                    else:
                        packages_dwelled = 0
                else:
                    packages_dwelled = 0

            # Get scenario info from first path using this arc
            scenario_id = "default"
            day_type = "peak"
            for i in chosen_idx:
                if a_idx in path_arcs[i]:
                    scenario_id = path_od_data[i]['scenario_id']
                    day_type = path_od_data[i]['day_type']
                    break

            arc_summary_data.append({
                "scenario_id": scenario_id,
                "day_type": day_type,
                "from_facility": arc["from"],
                "to_facility": arc["to"],
                "distance_miles": arc["distance_miles"],
                "pkgs_day": pkgs,
                "pkg_cube_cuft": cube,
                "trucks": trucks,
                "physical_containers": actual_containers,
                "packages_per_truck": pkgs / trucks,
                "cube_per_truck": cube / trucks,
                "container_fill_rate": container_fill_rate,
                "truck_fill_rate": truck_fill_rate,
                "packages_dwelled": packages_dwelled,
                "cost_per_truck": arc["cost_per_truck"],
                "total_cost": total_cost,
                "CPP": total_cost / pkgs,
            })

    arc_summary = pd.DataFrame(arc_summary_data).sort_values(
        ["scenario_id", "day_type", "from_facility", "to_facility"]
    ).reset_index(drop=True)

    print(f"    Selected {len(selected_paths)} optimal paths using {len(arc_summary)} arcs")

    # Calculate network KPIs
    network_kpis = _calculate_network_kpis(arc_summary, raw_trailer_cube)

    # Sort summary (placeholder - to be implemented if sort optimization enabled)
    sort_summary = None
    if enable_sort_optimization:
        print("    Sort optimization not yet implemented in v3")
        # TODO: Implement sort optimization logic

    return selected_paths, arc_summary, network_kpis, sort_summary


def _extract_legs_from_path(
        row: pd.Series,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> List[Tuple[str, str, float, float, float]]:
    """Extract leg information from path."""
    nodes = row.get("path_nodes", None)

    if isinstance(nodes, list) and len(nodes) >= 2:
        pairs = list(zip(nodes[:-1], nodes[1:]))
        legs = []
        for u, v in pairs:
            dist, cost_per_truck, mph = _arc_cost_per_truck(u, v, facilities, mileage_bands)
            legs.append((u, v, dist, cost_per_truck, mph))
        return legs

    # Fallback
    origin, dest = row["origin"], row["dest"]
    dist, cost_per_truck, mph = _arc_cost_per_truck(origin, dest, facilities, mileage_bands)
    return [(origin, dest, dist, cost_per_truck, mph)]


def _arc_cost_per_truck(
        u: str,
        v: str,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> Tuple[float, float, float]:
    """Calculate distance and cost for arc."""
    fac = facilities.set_index("facility_name")[["lat", "lon"]].astype(float)
    lat1, lon1 = fac.at[u, "lat"], fac.at[u, "lon"]
    lat2, lon2 = fac.at[v, "lat"], fac.at[v, "lon"]

    raw = haversine_miles(lat1, lon1, lat2, lon2)
    fixed, var, circuit, mph = band_lookup(raw, mileage_bands)
    dist = raw * circuit

    return dist, fixed + var * dist, mph


def _calculate_processing_cost(
        path_nodes: List[str],
        strategy: str,
        cost_params: CostParameters,
        facilities: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame
) -> float:
    """
    Calculate per-package processing cost for path based on strategy.

    Container: injection sort + container handling at intermediate + last mile
    Fluid: injection sort + full sort at intermediate + last mile
    """
    # Origin pays injection sort
    processing_cost_pp = cost_params.injection_sort_cost_per_pkg

    # Intermediate facilities (excluding origin and destination)
    intermediate_facilities = path_nodes[1:-1] if len(path_nodes) > 2 else []
    num_intermediate = len(intermediate_facilities)

    if strategy.lower() == "container":
        # Container handling cost per container at intermediate
        if num_intermediate > 0:
            w_cube = weighted_pkg_cube(package_mix)
            gaylord_row = container_params[
                container_params["container_type"].str.lower() == "gaylord"
                ].iloc[0]
            usable_cube = float(gaylord_row["usable_cube_cuft"])
            pack_util = float(gaylord_row["pack_utilization_container"])
            effective_cube = usable_cube * pack_util

            containers_per_pkg = w_cube / effective_cube
            processing_cost_pp += num_intermediate * cost_params.container_handling_cost * containers_per_pkg
    else:  # fluid
        # Full sort at every intermediate
        processing_cost_pp += num_intermediate * cost_params.intermediate_sort_cost_per_pkg

    # Destination pays last mile costs
    processing_cost_pp += cost_params.last_mile_sort_cost_per_pkg
    processing_cost_pp += cost_params.last_mile_delivery_cost_per_pkg

    return processing_cost_pp


def _calculate_network_kpis(arc_summary: pd.DataFrame, raw_trailer_cube: float) -> Dict[str, float]:
    """Calculate network-level KPIs."""
    if arc_summary.empty:
        return {
            "avg_truck_fill_rate": 0.0,
            "avg_container_fill_rate": 0.0,
            "total_packages_dwelled": 0
        }

    total_cube_used = arc_summary['pkg_cube_cuft'].sum()
    total_cube_capacity = (arc_summary['trucks'] * raw_trailer_cube).sum()
    network_truck_fill = total_cube_used / total_cube_capacity if total_cube_capacity > 0 else 0

    total_volume = arc_summary['pkgs_day'].sum()
    if total_volume > 0:
        network_container_fill = (
                                         arc_summary['container_fill_rate'] * arc_summary['pkgs_day']
                                 ).sum() / total_volume
    else:
        network_container_fill = 0.0

    total_dwelled = arc_summary['packages_dwelled'].sum()

    return {
        "avg_truck_fill_rate": max(0.0, min(1.0, network_truck_fill)),
        "avg_container_fill_rate": max(0.0, min(1.0, network_container_fill)),
        "total_packages_dwelled": max(0, total_dwelled)
    }