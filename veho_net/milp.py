"""
MILP Optimization Module

Solves arc-pooled path selection using OR-Tools CP-SAT solver.
All cost calculations (transportation + processing) happen within this module
with proper arc-level aggregation for accurate network-wide metrics.

Key Features:
- Path selection: Chooses optimal path for each OD pair
- Arc aggregation: Pools volumes across paths sharing the same lane
- Truck calculation: Determines truck requirements based on effective cube capacity
- Cost optimization: Minimizes total network cost (transportation + processing)
- Strategy support: Differentiates between container and fluid loading strategies
"""

from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from .time_cost import weighted_pkg_cube
from .geo import haversine_miles, band_lookup


def _arc_cost_per_truck(
        u: str,
        v: str,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> Tuple[float, float, float]:
    """
    Calculate distance and transportation cost for arc between two facilities.

    Args:
        u: Origin facility name
        v: Destination facility name
        facilities: Facility data with lat/lon coordinates
        mileage_bands: Mileage-based cost and timing parameters

    Returns:
        Tuple of (distance_miles, cost_per_truck, mph)
    """
    facility_lookup = facilities.set_index("facility_name")[["lat", "lon"]].astype(float)
    lat1, lon1 = facility_lookup.at[u, "lat"], facility_lookup.at[u, "lon"]
    lat2, lon2 = facility_lookup.at[v, "lat"], facility_lookup.at[v, "lon"]

    raw_distance = haversine_miles(lat1, lon1, lat2, lon2)
    fixed_cost, variable_cost, circuity_factor, mph = band_lookup(raw_distance, mileage_bands)
    actual_distance = raw_distance * circuity_factor

    return actual_distance, fixed_cost + variable_cost * actual_distance, mph


def _legs_for_candidate(
        row: pd.Series,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> List[Tuple[str, str, float, float, float]]:
    """
    Extract leg information from candidate path for arc analysis.

    Args:
        row: Candidate path data including path_nodes
        facilities: Facility data
        mileage_bands: Mileage-based parameters

    Returns:
        List of tuples: (from_facility, to_facility, distance, cost_per_truck, mph)
    """
    nodes = row.get("path_nodes", None)

    if isinstance(nodes, list) and len(nodes) >= 2:
        pairs = list(zip(nodes[:-1], nodes[1:]))
        legs = []
        for u, v in pairs:
            dist, cost_per_truck, mph = _arc_cost_per_truck(u, v, facilities, mileage_bands)
            legs.append((u, v, dist, cost_per_truck, mph))
        return legs

    # Fallback to basic origin->dest
    origin, dest = row["origin"], row["dest"]
    dist, cost_per_truck, mph = _arc_cost_per_truck(origin, dest, facilities, mileage_bands)
    return [(origin, dest, dist, cost_per_truck, mph)]


def _calculate_processing_costs_per_package(
        path_nodes: List[str],
        strategy: str,
        cost_kv: Dict,
        facilities: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame
) -> float:
    """
    Calculate per-package processing costs for a path based on loading strategy.

    Container Strategy:
        - Origin: Injection sort cost
        - Intermediate facilities: Container handling cost per container
        - Destination: Last mile sort + delivery

    Fluid Strategy:
        - Origin: Injection sort cost
        - Intermediate facilities: Full sort cost per package
        - Destination: Last mile sort + delivery

    Args:
        path_nodes: List of facility names in path order
        strategy: Loading strategy ('container' or 'fluid')
        cost_kv: Cost parameters dictionary
        facilities: Facility data (unused but kept for interface consistency)
        package_mix: Package mix distribution
        container_params: Container and trailer parameters

    Returns:
        Per-package processing cost for the entire path
    """
    # Get cost parameters
    injection_sort_pp = float(cost_kv.get("injection_sort_cost_per_pkg",
                                          cost_kv.get("sort_cost_per_pkg", 0.0)))
    intermediate_sort_pp = float(cost_kv.get("intermediate_sort_cost_per_pkg",
                                             cost_kv.get("sort_cost_per_pkg", 0.0)))
    last_mile_sort_pp = float(cost_kv.get("last_mile_sort_cost_per_pkg", 0.0))
    last_mile_delivery_pp = float(cost_kv.get("last_mile_delivery_cost_per_pkg", 0.0))
    container_handling_cost = float(cost_kv.get("container_handling_cost", 0.0))

    # Origin always pays injection sort
    processing_cost_pp = injection_sort_pp

    # Intermediate facilities (excluding origin and destination)
    intermediate_facilities = path_nodes[1:-1] if len(path_nodes) > 2 else []
    num_intermediate = len(intermediate_facilities)

    if strategy.lower() == "container":
        # Container strategy: Pay container handling cost per container at intermediate facilities
        if num_intermediate > 0:
            # Calculate containers per package
            weighted_cube = weighted_pkg_cube(package_mix)

            gaylord_row = container_params[
                container_params["container_type"].str.lower() == "gaylord"
                ].iloc[0]
            usable_cube = float(gaylord_row["usable_cube_cuft"])
            pack_util = float(gaylord_row["pack_utilization_container"])
            effective_container_cube = usable_cube * pack_util

            containers_per_pkg = weighted_cube / effective_container_cube

            # Container handling cost scales with number of touches and containers per package
            processing_cost_pp += num_intermediate * container_handling_cost * containers_per_pkg
    else:
        # Fluid strategy: Pay full sort cost at every intermediate facility
        processing_cost_pp += num_intermediate * intermediate_sort_pp

    # Destination always pays last mile costs
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
    Solve network optimization using arc-pooled MILP approach.

    This function:
    1. Selects optimal path for each OD pair
    2. Aggregates volumes across paths sharing lanes (arcs)
    3. Calculates truck requirements based on effective cube capacity
    4. Computes transportation and processing costs
    5. Returns selected paths, arc summary, and network KPIs

    Args:
        candidates: Candidate paths with origin, dest, path_nodes, pkgs_day
        facilities: Facility master data
        mileage_bands: Distance-based cost and timing parameters
        package_mix: Package type distribution and cube factors
        container_params: Container and trailer capacity parameters
        cost_kv: Cost parameters including strategy
        timing_kv: Timing parameters (currently unused in MILP)

    Returns:
        Tuple of (selected_paths_df, arc_summary_df, network_kpis_dict)
    """
    cand = candidates.reset_index(drop=True).copy()
    strategy = str(cost_kv.get("load_strategy", "container")).lower()

    print(f"    MILP optimization: {strategy} strategy")

    # Log key cost parameters for transparency
    injection_sort_pp = float(cost_kv.get("injection_sort_cost_per_pkg",
                                          cost_kv.get("sort_cost_per_pkg", 0.0)))
    intermediate_sort_pp = float(cost_kv.get("intermediate_sort_cost_per_pkg",
                                             cost_kv.get("sort_cost_per_pkg", 0.0)))
    container_handling_cost = float(cost_kv.get("container_handling_cost", 0.0))

    if strategy == "container":
        print(f"    Cost params: injection_sort=${injection_sort_pp:.3f}, "
              f"container_handling=${container_handling_cost:.3f}")
    else:
        print(f"    Cost params: injection_sort=${injection_sort_pp:.3f}, "
              f"intermediate_sort=${intermediate_sort_pp:.3f}")

    path_keys = list(cand.index)
    weighted_cube = weighted_pkg_cube(package_mix)

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
        processing_cost_pp = _calculate_processing_costs_per_package(
            path_nodes, strategy, cost_kv, facilities, package_mix, container_params
        )

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

    print(f"    Generated {len(arc_meta)} unique arcs from {len(cand)} candidate paths")

    # Initialize CP-SAT optimization model
    model = cp_model.CpModel()

    # Path selection variables: choose exactly one path per OD group
    groups = cand.groupby(["scenario_id", "origin", "dest", "day_type"]).indices
    x = {i: model.NewBoolVar(f"x_{i}") for i in path_keys}

    for group_name, idxs in groups.items():
        model.Add(sum(x[i] for i in idxs) == 1)

    print(f"    Created {len(groups)} OD selection constraints")

    # Arc volume variables - track packages per arc
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

    # Calculate truck requirements per arc based on effective cube capacity
    raw_trailer_cube = float(container_params["trailer_air_cube_cuft"].iloc[0])

    if strategy == "container":
        # Container strategy: Use effective container capacity
        gaylord_row = container_params[
            container_params["container_type"].str.lower() == "gaylord"
            ].iloc[0]
        raw_container_cube = float(gaylord_row["usable_cube_cuft"])
        pack_util_container = float(gaylord_row["pack_utilization_container"])
        effective_container_cube = raw_container_cube * pack_util_container
        containers_per_truck = int(gaylord_row["containers_per_truck"])

        effective_truck_cube = containers_per_truck * effective_container_cube

        print(f"    Container capacity: {effective_container_cube:.1f} cuft/container (effective)")
        print(f"    Truck capacity: {effective_truck_cube:.1f} cuft/truck ({containers_per_truck} containers)")
    else:
        # Fluid strategy: Use effective trailer capacity
        pack_util_fluid = float(container_params["pack_utilization_fluid"].iloc[0])
        effective_truck_cube = raw_trailer_cube * pack_util_fluid

        print(f"    Truck capacity: {effective_truck_cube:.1f} cuft/truck (effective)")

    # Convert to integer for MILP (scale by 1000 to handle decimals)
    effective_truck_cube_scaled = int(effective_truck_cube * 1000)
    weighted_cube_scaled = int(weighted_cube * 1000)

    print(f"    Package cube: {weighted_cube:.3f} cuft/package")

    # Arc truck variables
    arc_trucks = {a_idx: model.NewIntVar(0, 1000, f"arc_trucks_{a_idx}")
                  for a_idx in range(len(arc_meta))}

    # Truck requirement constraints based on effective cube capacity
    # Business rule: Trucks * effective_capacity >= packages * package_cube
    for a_idx in range(len(arc_meta)):
        model.Add(arc_trucks[a_idx] * effective_truck_cube_scaled >=
                  arc_pkgs[a_idx] * weighted_cube_scaled)

        # Ensure minimum 1 truck if any packages (never round to 0)
        arc_has_pkgs = model.NewBoolVar(f"arc_has_pkgs_{a_idx}")
        BIG_M = 1000000  # Standard MILP big-M constraint value
        model.Add(arc_pkgs[a_idx] <= BIG_M * arc_has_pkgs)
        model.Add(arc_pkgs[a_idx] >= 1 * arc_has_pkgs)
        model.Add(arc_trucks[a_idx] >= arc_has_pkgs)

    # Objective: Minimize total cost (transportation + processing)
    cost_terms = []

    # Transportation costs (truck-based, aggregated at arc level)
    for a_idx in range(len(arc_meta)):
        arc = arc_meta[a_idx]
        truck_cost = int(arc["cost_per_truck"])
        cost_terms.append(arc_trucks[a_idx] * truck_cost)

    # Processing costs (package-based, path-specific)
    for i in path_keys:
        path_data = path_od_data[i]
        processing_cost = int(path_data['processing_cost_pp'] * path_data['pkgs_day'])
        cost_terms.append(x[i] * processing_cost)

    model.Minimize(sum(cost_terms))

    print(f"    Objective: {len([t for t in cost_terms if 'arc_trucks' in str(t)])} "
          f"transportation + {len([t for t in cost_terms if 'x[' in str(t)])} processing cost terms")

    # Solve with reasonable time limit
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"    MILP solver failed with status: {status}")
        return pd.DataFrame(), pd.DataFrame(), {}

    print(f"    MILP solver completed with status: {status}")

    # Extract solution
    chosen_idx = [i for i in path_keys if solver.Value(x[i]) == 1]
    selected_paths_data = []

    total_cost_optimized = solver.ObjectiveValue()
    print(f"    Total optimized cost: ${total_cost_optimized:,.0f}")

    # Build selected paths dataframe
    for i in chosen_idx:
        path_data = path_od_data[i]

        # Calculate actual costs for this path
        total_processing_cost = path_data['processing_cost_pp'] * path_data['pkgs_day']

        # Calculate transportation cost (allocate based on package share of each arc)
        total_transport_cost = 0
        for a_idx in path_arcs[i]:
            arc = arc_meta[a_idx]
            trucks_on_arc = solver.Value(arc_trucks[a_idx])
            if trucks_on_arc > 0:
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

    # Build arc summary with fill rates and dwell calculation
    dwell_threshold = float(cost_kv.get('premium_economy_dwell_threshold', 0.10))
    arc_summary_data = []

    for a_idx in range(len(arc_meta)):
        arc = arc_meta[a_idx]
        pkgs = solver.Value(arc_pkgs[a_idx])
        trucks = solver.Value(arc_trucks[a_idx])

        if pkgs > 0:
            total_cost = trucks * arc['cost_per_truck']
            cube = pkgs * weighted_cube

            # Calculate fill rates using raw capacities (executive metric standard)
            if strategy == "container":
                # Container calculations
                gaylord_row = container_params[
                    container_params["container_type"].str.lower() == "gaylord"
                    ].iloc[0]
                raw_container_cube = float(gaylord_row["usable_cube_cuft"])
                pack_util_container = float(gaylord_row["pack_utilization_container"])
                effective_container_cube = raw_container_cube * pack_util_container
                containers_per_truck = int(gaylord_row["containers_per_truck"])

                # Actual containers needed (based on effective capacity)
                actual_containers = max(1, int(np.ceil(cube / effective_container_cube)))

                # Fill rates measured against raw capacity (standard industry metric)
                container_fill_rate = cube / (actual_containers * raw_container_cube)
                truck_fill_rate = cube / (trucks * raw_trailer_cube)
            else:
                # Fluid strategy: no containers
                container_fill_rate = 0.0
                actual_containers = 0
                truck_fill_rate = cube / (trucks * raw_trailer_cube)

            # Calculate dwell based on premium economy threshold
            # Business rule: If fractional truck < threshold, round down and dwell excess
            exact_trucks_needed = cube / effective_truck_cube

            if exact_trucks_needed <= 1.0:
                # Always use at least 1 truck (never round to 0)
                packages_dwelled = 0
            else:
                fractional_part = exact_trucks_needed - int(exact_trucks_needed)

                if fractional_part > 0 and fractional_part < dwell_threshold:
                    # Round down, dwell the excess
                    optimal_trucks = int(exact_trucks_needed)
                    if trucks == optimal_trucks:
                        # We rounded down, calculate dwell
                        excess_cube = cube - (trucks * effective_truck_cube)
                        packages_dwelled = max(0, excess_cube / weighted_cube)
                    else:
                        packages_dwelled = 0
                else:
                    # Fractional part >= threshold, use the extra truck
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

    # Calculate network-level KPIs
    network_kpis = {}
    if not arc_summary.empty:
        total_cube_used = arc_summary['pkg_cube_cuft'].sum()
        total_cube_capacity = (arc_summary['trucks'] * raw_trailer_cube).sum()
        network_truck_fill = total_cube_used / total_cube_capacity if total_cube_capacity > 0 else 0

        # Volume-weighted container fill rate
        total_volume = arc_summary['pkgs_day'].sum()
        if total_volume > 0:
            network_container_fill = (
                                             arc_summary['container_fill_rate'] * arc_summary['pkgs_day']
                                     ).sum() / total_volume
        else:
            network_container_fill = 0.0

        total_dwelled = arc_summary['packages_dwelled'].sum()

        network_kpis = {
            "avg_truck_fill_rate": max(0.0, min(1.0, network_truck_fill)),
            "avg_container_fill_rate": max(0.0, min(1.0, network_container_fill)),
            "total_packages_dwelled": max(0, total_dwelled)
        }

        print(f"    Network avg truck fill rate: {network_truck_fill:.1%}")
        print(f"    Network avg container fill rate: {network_container_fill:.1%}")
    else:
        network_kpis = {
            "avg_truck_fill_rate": 0.0,
            "avg_container_fill_rate": 0.0,
            "total_packages_dwelled": 0
        }

    return selected_paths, arc_summary, network_kpis