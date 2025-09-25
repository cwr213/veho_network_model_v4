# veho_net/milp.py - CORRECTED ARCHITECTURE: All cost calculation in MILP with proper aggregation
from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from .time_cost import weighted_pkg_cube, calculate_truck_capacity
from .geo import haversine_miles, band_lookup


def _arc_cost_per_truck(u: str, v: str, facilities: pd.DataFrame, mileage_bands: pd.DataFrame) -> Tuple[
    float, float, float]:
    """Calculate distance and transportation cost for arc between two facilities."""
    fac = facilities.set_index("facility_name")[["lat", "lon"]].astype(float)
    lat1, lon1 = fac.at[u, "lat"], fac.at[u, "lon"]
    lat2, lon2 = fac.at[v, "lat"], fac.at[v, "lon"]
    raw = haversine_miles(lat1, lon1, lat2, lon2)
    fixed, var, circuit, mph = band_lookup(raw, mileage_bands)
    dist = raw * circuit
    return dist, fixed + var * dist, mph


def _legs_for_candidate(row: pd.Series, facilities: pd.DataFrame, mileage_bands: pd.DataFrame):
    """Extract leg information from candidate path for arc analysis."""
    nodes = row.get("path_nodes", None)
    if isinstance(nodes, list) and len(nodes) >= 2:
        pairs = list(zip(nodes[:-1], nodes[1:]))
        legs = []
        for u, v in pairs:
            dist, cost_per_truck, mph = _arc_cost_per_truck(u, v, facilities, mileage_bands)
            legs.append((u, v, dist, cost_per_truck, mph))
        return legs

    # Fallback to basic origin->dest
    o, d = row["origin"], row["dest"]
    dist, cost_per_truck, mph = _arc_cost_per_truck(o, d, facilities, mileage_bands)
    return [(o, d, dist, cost_per_truck, mph)]


def _calculate_processing_costs_per_package(path_nodes: List[str], strategy: str, cost_kv: Dict,
                                            facilities: pd.DataFrame, package_mix: pd.DataFrame,
                                            container_params: pd.DataFrame) -> float:
    """
    Calculate per-package processing costs for a path based on strategy using only input parameters.
    """
    # Get cost parameters
    injection_sort_pp = float(cost_kv.get("injection_sort_cost_per_pkg", cost_kv.get("sort_cost_per_pkg", 0.0)))
    intermediate_sort_pp = float(cost_kv.get("intermediate_sort_cost_per_pkg", cost_kv.get("sort_cost_per_pkg", 0.0)))
    last_mile_sort_pp = float(cost_kv.get("last_mile_sort_cost_per_pkg", 0.0))
    last_mile_delivery_pp = float(cost_kv.get("last_mile_delivery_cost_per_pkg", 0.0))
    container_handling_cost = float(cost_kv.get("container_handling_cost", 0.0))

    # Always pay injection sort at origin
    processing_cost_pp = injection_sort_pp

    # Count intermediate facilities (excluding origin and destination)
    intermediate_facilities = path_nodes[1:-1] if len(path_nodes) > 2 else []
    num_intermediate = len(intermediate_facilities)

    if strategy.lower() == "container":
        # Container strategy: pay container handling cost per container at intermediate facilities
        if num_intermediate > 0:
            # Calculate containers per package using input parameters only
            from .time_cost import weighted_pkg_cube
            w_cube = weighted_pkg_cube(package_mix)

            # Get container parameters
            gaylord_row = container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0]
            usable_cube = float(gaylord_row["usable_cube_cuft"])
            pack_util = float(gaylord_row["pack_utilization_container"])
            effective_container_cube = usable_cube * pack_util

            # Containers per package = package_cube / container_effective_cube
            containers_per_pkg = w_cube / effective_container_cube

            # Container handling cost per package = containers_per_pkg * container_handling_cost * num_intermediate
            processing_cost_pp += num_intermediate * container_handling_cost * containers_per_pkg
    else:
        # Fluid strategy: pay full sort cost at every intermediate facility
        processing_cost_pp += num_intermediate * intermediate_sort_pp

    # Always pay last mile costs at destination
    processing_cost_pp += last_mile_sort_pp + last_mile_delivery_pp

    return processing_cost_pp


def solve_arc_pooled_path_selection(
        candidates: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        cost_kv: Dict[str, float],
        timing_kv: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    CORRECTED ARCHITECTURE: All cost calculation happens in MILP with proper arc-level aggregation.
    """
    cand = candidates.reset_index(drop=True).copy()
    strategy = str(cost_kv.get("load_strategy", "container")).lower()

    print(f"    MILP solver with corrected architecture: {strategy} strategy")
    print(f"    All costs calculated with proper aggregation (no pre-calculation)")

    # Get cost parameters for debugging
    injection_sort_pp = float(cost_kv.get("injection_sort_cost_per_pkg", cost_kv.get("sort_cost_per_pkg", 0.0)))
    intermediate_sort_pp = float(cost_kv.get("intermediate_sort_cost_per_pkg", cost_kv.get("sort_cost_per_pkg", 0.0)))
    container_handling_cost = float(cost_kv.get("container_handling_cost", 0.0))

    # Show strategy-specific cost parameters being used
    if strategy.lower() == "container":
        print(
            f"    Cost params: injection_sort=${injection_sort_pp:.3f}, container_handling=${container_handling_cost:.3f}")
    else:
        print(
            f"    Cost params: injection_sort=${injection_sort_pp:.3f}, intermediate_sort=${intermediate_sort_pp:.3f}")

    path_keys = list(cand.index)
    w_cube = weighted_pkg_cube(package_mix)

    # Build arc metadata and path-to-arc mapping
    arc_index_map: Dict[Tuple[str, str], int] = {}
    arc_meta: List[Dict] = []
    path_arcs: Dict[int, List[int]] = {}
    path_od_data: Dict[int, Dict] = {}

    for i in path_keys:
        r = cand.loc[i]
        legs = _legs_for_candidate(r, facilities, mileage_bands)

        path_nodes = r.get("path_nodes", [r["origin"], r["dest"]])
        if not isinstance(path_nodes, list):
            path_nodes = [r["origin"], r["dest"]]

        # Calculate per-package processing cost for this path
        processing_cost_pp = _calculate_processing_costs_per_package(path_nodes, strategy, cost_kv, facilities,
                                                                     package_mix, container_params)

        # Store path metadata
        path_od_data[i] = {
            'origin': r["origin"],
            'dest': r["dest"],
            'pkgs_day': float(r["pkgs_day"]),
            'processing_cost_pp': processing_cost_pp,
            'scenario_id': r.get("scenario_id", "default"),
            'day_type': r.get("day_type", "peak"),
            'path_str': r.get("path_str", f"{r['origin']}->{r['dest']}"),
            'path_type': r.get("path_type", "direct")
        }

        # Map path to arcs
        ids = []
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
            ids.append(arc_index_map[key])

        path_arcs[i] = ids

    print(f"    Generated {len(arc_meta)} unique arcs from {len(cand)} candidate paths")

    # Initialize CP-SAT optimization model
    model = cp_model.CpModel()

    # Path selection variables: choose exactly one path per OD group
    groups = cand.groupby(["scenario_id", "origin", "dest", "day_type"]).indices
    x = {i: model.NewBoolVar(f"x_{i}") for i in path_keys}

    for group_name, idxs in groups.items():
        model.Add(sum(x[i] for i in idxs) == 1)

    print(f"    Created {len(groups)} OD selection constraints")

    # Arc volume variables - track packages per arc (unscaled for simplicity)
    arc_pkgs = {a_idx: model.NewIntVar(0, 1000000, f"arc_pkgs_{a_idx}") for a_idx in range(len(arc_meta))}

    # Link path selection to arc volumes
    for a_idx in range(len(arc_meta)):
        # Sum packages from all paths that use this arc
        terms = []
        for i in path_keys:
            if a_idx in path_arcs[i]:
                pkgs = int(round(path_od_data[i]['pkgs_day']))  # Use unscaled packages
                terms.append(pkgs * x[i])

        if terms:
            model.Add(arc_pkgs[a_idx] == sum(terms))
        else:
            model.Add(arc_pkgs[a_idx] == 0)

    # Calculate truck requirements per arc based on cube capacity with proper pack utilization
    w_cube = weighted_pkg_cube(package_mix)
    raw_trailer_cube = float(container_params["trailer_air_cube_cuft"].iloc[0])

    if strategy.lower() == "container":
        # Container strategy: use effective container capacity for planning
        gaylord_row = container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0]
        raw_container_cube = float(gaylord_row["usable_cube_cuft"])
        pack_util_container = float(gaylord_row["pack_utilization_container"])
        effective_container_cube = raw_container_cube * pack_util_container
        containers_per_truck = int(gaylord_row["containers_per_truck"])

        # Effective truck capacity = containers per truck * effective container capacity
        effective_truck_cube = containers_per_truck * effective_container_cube

        print(f"    Container capacity: {effective_container_cube:.1f} cuft/container (effective)")
        print(f"    Truck capacity: {effective_truck_cube:.1f} cuft/truck ({containers_per_truck} containers)")

    else:
        # Fluid strategy: use effective trailer capacity for planning
        pack_util_fluid = float(container_params["pack_utilization_fluid"].iloc[0])
        effective_truck_cube = raw_trailer_cube * pack_util_fluid

        print(f"    Truck capacity: {effective_truck_cube:.1f} cuft/truck (effective)")

    # Convert to integer for MILP (cube * 1000 to handle decimals)
    effective_truck_cube_scaled = int(effective_truck_cube * 1000)
    w_cube_scaled = int(w_cube * 1000)

    print(f"    Package cube: {w_cube:.3f} cuft/package")

    # Arc truck variables
    arc_trucks = {a_idx: model.NewIntVar(0, 1000, f"arc_trucks_{a_idx}") for a_idx in range(len(arc_meta))}

    # Truck requirement constraints based on EFFECTIVE cube capacity
    for a_idx in range(len(arc_meta)):
        # Trucks needed based on effective cube: trucks * effective_truck_cube >= packages * w_cube
        model.Add(arc_trucks[a_idx] * effective_truck_cube_scaled >= arc_pkgs[a_idx] * w_cube_scaled)

        # Ensure minimum 1 truck if any packages
        arc_has_pkgs = model.NewBoolVar(f"arc_has_pkgs_{a_idx}")
        BIG_M = 1000000
        model.Add(arc_pkgs[a_idx] <= BIG_M * arc_has_pkgs)
        model.Add(arc_pkgs[a_idx] >= 1 * arc_has_pkgs)
        model.Add(arc_trucks[a_idx] >= arc_has_pkgs)

    # Objective: minimize total cost (unscaled)
    cost_terms = []

    # 1. Transportation costs (truck-based)
    for a_idx in range(len(arc_meta)):
        arc = arc_meta[a_idx]
        truck_cost = int(arc["cost_per_truck"])
        cost_terms.append(arc_trucks[a_idx] * truck_cost)

    # 2. Processing costs (package-based)
    for i in path_keys:
        path_data = path_od_data[i]
        processing_cost = int(path_data['processing_cost_pp'] * path_data['pkgs_day'])
        cost_terms.append(x[i] * processing_cost)

    model.Minimize(sum(cost_terms))

    print(
        f"    Objective includes {len([t for t in cost_terms if 'arc_trucks' in str(t)])} transportation + {len([t for t in cost_terms if 'x[' in str(t)])} processing cost terms")

    # Solve with reasonable time limit
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"    MILP solver failed with status: {status}")
        return pd.DataFrame(), pd.DataFrame()

    print(f"    MILP solver completed with status: {status}")

    # Extract solution
    chosen_idx = [i for i in path_keys if solver.Value(x[i]) == 1]
    selected_paths_data = []

    total_cost_unscaled = solver.ObjectiveValue()
    print(f"    Total optimized cost: ${total_cost_unscaled:,.0f}")

    # Build selected paths dataframe
    for i in chosen_idx:
        path_data = path_od_data[i]

        # Calculate actual costs for this path
        total_processing_cost = path_data['processing_cost_pp'] * path_data['pkgs_day']

        # Calculate transportation cost (sum across arcs this path uses)
        total_transport_cost = 0
        for a_idx in path_arcs[i]:
            arc = arc_meta[a_idx]
            trucks_on_arc = solver.Value(arc_trucks[a_idx])
            if trucks_on_arc > 0:
                # Allocate transportation cost based on package share
                arc_total_pkgs = solver.Value(arc_pkgs[a_idx])
                path_share = path_data['pkgs_day'] / max(arc_total_pkgs, 1e-9)
                allocated_transport_cost = trucks_on_arc * arc['cost_per_truck'] * path_share
                total_transport_cost += allocated_transport_cost

        total_path_cost = total_transport_cost + total_processing_cost

        selected_paths_data.append({
            **path_data,
            'total_cost': total_path_cost,
            'linehaul_cost': total_transport_cost,
            'processing_cost': total_processing_cost,
            'cost_per_pkg': total_path_cost / path_data['pkgs_day']
        })

    selected_paths = pd.DataFrame(selected_paths_data)

    # Build arc summary
    arc_summary_data = []
    for a_idx in range(len(arc_meta)):
        arc = arc_meta[a_idx]
        pkgs = solver.Value(arc_pkgs[a_idx])
        trucks = solver.Value(arc_trucks[a_idx])

        if pkgs > 0:
            total_cost = trucks * arc['cost_per_truck']
            cube = pkgs * w_cube

            # Calculate fill rates using RAW capacities (for performance measurement)
            if strategy.lower() == "container":
                # Container strategy: calculate containers needed and fill rates
                gaylord_row = container_params[container_params["container_type"].str.lower() == "gaylord"].iloc[0]
                raw_container_cube = float(gaylord_row["usable_cube_cuft"])
                pack_util_container = float(gaylord_row["pack_utilization_container"])
                effective_container_cube = raw_container_cube * pack_util_container
                containers_per_truck = int(gaylord_row["containers_per_truck"])

                # Calculate containers needed based on EFFECTIVE container capacity (realistic planning)
                actual_containers = max(1, int(np.ceil(cube / effective_container_cube)))

                # Container fill rate = actual cube / (containers * raw container cube)
                # This measures performance against raw capacity, not effective capacity
                container_fill_rate = cube / (actual_containers * raw_container_cube)

                # Truck fill rate = actual cube / (trucks * raw trailer cube)
                truck_fill_rate = cube / (trucks * raw_trailer_cube)

            else:
                # Fluid strategy: no containers
                container_fill_rate = 0.0
                actual_containers = 0

                # Truck fill rate = actual cube / (trucks * raw trailer cube)
                truck_fill_rate = cube / (trucks * raw_trailer_cube)

            # Dwell calculation based on effective capacity constraints
            if strategy.lower() == "container":
                max_effective_capacity = trucks * effective_truck_cube
                if cube > max_effective_capacity:
                    excess_cube = cube - max_effective_capacity
                    packages_dwelled = excess_cube / w_cube
                else:
                    packages_dwelled = 0
            else:
                max_effective_capacity = trucks * effective_truck_cube
                if cube > max_effective_capacity:
                    excess_cube = cube - max_effective_capacity
                    packages_dwelled = excess_cube / w_cube
                else:
                    packages_dwelled = 0

            packages_dwelled = max(0, packages_dwelled)

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

    # Calculate correct network-level KPIs
    network_kpis = {}
    if not arc_summary.empty:
        total_cube_used = arc_summary['pkg_cube_cuft'].sum()
        total_cube_capacity = (arc_summary['trucks'] * raw_trailer_cube).sum()
        network_truck_fill = total_cube_used / total_cube_capacity if total_cube_capacity > 0 else 0

        # Volume-weighted container fill rate
        total_volume = arc_summary['pkgs_day'].sum()
        if total_volume > 0:
            network_container_fill = (arc_summary['container_fill_rate'] * arc_summary['pkgs_day']).sum() / total_volume
        else:
            network_container_fill = 0.0

        total_dwelled = arc_summary['packages_dwelled'].sum()

        network_kpis = {
            "avg_truck_fill_rate": max(0.0, min(1.0, network_truck_fill)),
            "avg_container_fill_rate": max(0.0, min(1.0, network_container_fill)),
            "total_packages_dwelled": max(0, total_dwelled)
        }

        print(f"    Network average truck fill rate: {network_truck_fill:.1%}")
        print(f"    Network average container fill rate: {network_container_fill:.1%}")
    else:
        network_kpis = {
            "avg_truck_fill_rate": 0.0,
            "avg_container_fill_rate": 0.0,
            "total_packages_dwelled": 0
        }

    return selected_paths, arc_summary, network_kpis