"""
MILP Optimization Module

Unified path selection optimization with optional sort-level optimization.
"""

from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
from .containers_v3 import weighted_pkg_cube
from .geo_v3 import haversine_miles, band_lookup
from .config_v3 import CostParameters, TimingParameters, LoadStrategy


def solve_network_optimization(
        candidates: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        cost_params: Union[CostParameters, Dict],
        timing_params: Union[TimingParameters, Dict],
        global_strategy: Union[LoadStrategy, str],
        enable_sort_optimization: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Optional[pd.DataFrame]]:
    """
    Solve network optimization using arc-pooled MILP approach.
    """
    if isinstance(cost_params, dict):
        cost_params = CostParameters(**{
            'injection_sort_cost_per_pkg': float(cost_params.get('injection_sort_cost_per_pkg', 0)),
            'intermediate_sort_cost_per_pkg': float(cost_params.get('intermediate_sort_cost_per_pkg', 0)),
            'last_mile_sort_cost_per_pkg': float(cost_params.get('last_mile_sort_cost_per_pkg', 0)),
            'last_mile_delivery_cost_per_pkg': float(cost_params.get('last_mile_delivery_cost_per_pkg', 0)),
            'container_handling_cost': float(cost_params.get('container_handling_cost', 0)),
            'premium_economy_dwell_threshold': float(cost_params.get('premium_economy_dwell_threshold', 0.1)),
            'dwell_cost_per_pkg_per_day': float(cost_params.get('dwell_cost_per_pkg_per_day', 0)),
            'sla_penalty_per_touch_per_pkg': float(cost_params.get('sla_penalty_per_touch_per_pkg', 0))
        })

    if isinstance(timing_params, dict):
        timing_params = TimingParameters(**{
            'hours_per_touch': float(timing_params.get('hours_per_touch', 8)),
            'load_hours': float(timing_params.get('load_hours', 2)),
            'unload_hours': float(timing_params.get('unload_hours', 2)),
            'injection_va_hours': float(timing_params.get('injection_va_hours', 8)),
            'middle_mile_va_hours': float(timing_params.get('middle_mile_va_hours', 16)),
            'last_mile_va_hours': float(timing_params.get('last_mile_va_hours', 4)),
            'sort_points_per_destination': float(timing_params.get('sort_points_per_destination', 1))
        })

    if isinstance(global_strategy, str):
        global_strategy = LoadStrategy.CONTAINER if global_strategy.lower() == 'container' else LoadStrategy.FLUID

    cand = candidates.reset_index(drop=True).copy()

    print(f"    MILP optimization: {global_strategy.value} strategy (global)")
    if enable_sort_optimization:
        print(f"    Sort optimization: ENABLED")
    else:
        print(f"    Sort optimization: DISABLED (using market-level)")

    path_keys = list(cand.index)
    w_cube = weighted_pkg_cube(package_mix)

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

        effective_strategy = r.get("strategy_hint") or global_strategy.value

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

    model = cp_model.CpModel()

    groups = cand.groupby(["scenario_id", "origin", "dest", "day_type"]).indices
    x = {i: model.NewBoolVar(f"x_{i}") for i in path_keys}

    for group_name, idxs in groups.items():
        model.Add(sum(x[i] for i in idxs) == 1)

    print(f"    Created {len(groups)} OD selection constraints")

    sort_decision = {}
    if enable_sort_optimization:
        sort_decision = _create_sort_decision_variables(
            model, groups, cand, facilities, timing_params
        )
        print(f"    Created sort level decision variables")

    arc_pkgs = {a_idx: model.NewIntVar(0, 1000000, f"arc_pkgs_{a_idx}")
                for a_idx in range(len(arc_meta))}

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

    raw_trailer_cube = float(container_params["trailer_air_cube_cuft"].iloc[0])

    container_capacity = _get_container_strategy_capacity(package_mix, container_params, w_cube)
    fluid_capacity = _get_fluid_strategy_capacity(package_mix, container_params, w_cube)

    arc_trucks = {}

    for a_idx in range(len(arc_meta)):
        arc_trucks[a_idx] = model.NewIntVar(0, 1000, f"arc_trucks_{a_idx}")

        strategies_using_arc = [global_strategy.value]
        for i in path_keys:
            if a_idx in path_arcs[i]:
                strategies_using_arc.append(path_od_data[i]['effective_strategy'])

        strategy_counts = {}
        for s in strategies_using_arc:
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
        predominant_strategy = max(strategy_counts, key=strategy_counts.get)

        if predominant_strategy.lower() == "container":
            effective_truck_cube_scaled = int(container_capacity * 1000)
        else:
            effective_truck_cube_scaled = int(fluid_capacity * 1000)

        w_cube_scaled = int(w_cube * 1000)

        model.Add(arc_trucks[a_idx] * effective_truck_cube_scaled >=
                  arc_pkgs[a_idx] * w_cube_scaled)

        arc_has_pkgs = model.NewBoolVar(f"arc_has_pkgs_{a_idx}")
        BIG_M = 1000000
        model.Add(arc_pkgs[a_idx] <= BIG_M * arc_has_pkgs)
        model.Add(arc_pkgs[a_idx] >= 1 * arc_has_pkgs)
        model.Add(arc_trucks[a_idx] >= arc_has_pkgs)

    print(f"    Created arc capacity constraints")

    if enable_sort_optimization:
        _add_sort_capacity_constraints(
            model, sort_decision, groups, path_od_data,
            facilities, timing_params, cand
        )
        print(f"    Created sort capacity constraints")

    cost_terms = []

    for a_idx in range(len(arc_meta)):
        arc = arc_meta[a_idx]
        truck_cost = int(arc["cost_per_truck"])
        cost_terms.append(arc_trucks[a_idx] * truck_cost)

    for group_name, group_idxs in groups.items():
        scenario_id, origin, dest, day_type = group_name

        repr_idx = group_idxs[0]
        volume = path_od_data[repr_idx]['pkgs_day']

        if enable_sort_optimization:
            sort_vars = sort_decision[group_name]

            for sort_level in ['region', 'market', 'sort_group']:
                sort_var = sort_vars.get(sort_level)
                if sort_var is None:
                    continue

                for path_idx in group_idxs:
                    effective_strategy = path_od_data[path_idx]['effective_strategy']

                    processing_cost = _calculate_processing_cost(
                        path_od_data[path_idx], sort_level, effective_strategy,
                        cost_params, facilities, package_mix, container_params
                    )

                    total_processing_cost = int(processing_cost * volume)

                    cost_active = model.NewBoolVar(f"cost_active_{path_idx}_{sort_level}")
                    model.Add(cost_active <= x[path_idx])
                    model.Add(cost_active <= sort_var)
                    model.Add(cost_active >= x[path_idx] + sort_var - 1)

                    cost_terms.append(cost_active * total_processing_cost)
        else:
            for path_idx in group_idxs:
                effective_strategy = path_od_data[path_idx]['effective_strategy']

                processing_cost = _calculate_processing_cost(
                    path_od_data[path_idx], 'market', effective_strategy,
                    cost_params, facilities, package_mix, container_params
                )

                total_processing_cost = int(processing_cost * volume)
                cost_terms.append(x[path_idx] * total_processing_cost)

    model.Minimize(sum(cost_terms))

    print(f"    Objective includes {len(cost_terms)} cost terms")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 600.0
    solver.parameters.num_search_workers = 8

    print(f"    Starting MILP solver...")
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"    ❌ MILP solver failed with status: {status}")
        return pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame()

    print(f"    ✅ MILP solver completed with status: {status}")
    print(f"    Total optimized cost: ${solver.ObjectiveValue():,.0f}")

    chosen_idx = [i for i in path_keys if solver.Value(x[i]) == 1]

    sort_decisions = {}
    if enable_sort_optimization:
        for group_name in groups.keys():
            for sort_level in ['region', 'market', 'sort_group']:
                sort_var = sort_decision[group_name].get(sort_level)
                if sort_var is not None and solver.Value(sort_var) == 1:
                    sort_decisions[group_name] = sort_level
                    break

    arc_strategy_effective = _determine_arc_strategies_from_solution(
        arc_meta, chosen_idx, path_od_data, path_arcs, global_strategy
    )

    selected_paths = _build_selected_paths_dataframe(
        chosen_idx, path_od_data, path_arcs, arc_meta, arc_trucks, arc_pkgs,
        sort_decisions, cost_params, facilities, package_mix, container_params, solver
    )

    arc_summary = _build_arc_summary_dataframe(
        arc_meta, arc_pkgs, arc_trucks, arc_strategy_effective,
        w_cube, raw_trailer_cube, container_params, package_mix,
        cost_params, chosen_idx, path_od_data, path_arcs, solver
    )

    network_kpis = _calculate_network_kpis(
        selected_paths, arc_summary, raw_trailer_cube
    )

    sort_summary = pd.DataFrame()
    if enable_sort_optimization and sort_decisions:
        sort_summary = _build_sort_summary_dataframe(
            selected_paths, sort_decisions, cost_params, facilities
        )

    print(f"    Selected {len(selected_paths)} optimal paths using {len(arc_summary)} arcs")

    return selected_paths, arc_summary, network_kpis, sort_summary


def _extract_legs_from_path(
        row: pd.Series,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> List[Tuple[str, str, float, float, float]]:
    """
    Extract leg information from path for arc analysis.
    Returns list of (from, to, distance, cost_per_truck, mph).
    """
    nodes = row.get("path_nodes", None)

    if isinstance(nodes, list) and len(nodes) >= 2:
        pairs = list(zip(nodes[:-1], nodes[1:]))
        legs = []
        for u, v in pairs:
            dist, cost, mph = _calculate_arc_cost(u, v, facilities, mileage_bands)
            legs.append((u, v, dist, cost, mph))
        return legs

    o, d = row["origin"], row["dest"]
    dist, cost, mph = _calculate_arc_cost(o, d, facilities, mileage_bands)
    return [(o, d, dist, cost, mph)]


def _calculate_arc_cost(
        u: str,
        v: str,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> Tuple[float, float, float]:
    """Calculate distance, cost per truck, and mph for arc."""
    fac = facilities.set_index("facility_name")[["lat", "lon"]].astype(float)

    lat1, lon1 = fac.at[u, "lat"], fac.at[u, "lon"]
    lat2, lon2 = fac.at[v, "lat"], fac.at[v, "lon"]

    raw = haversine_miles(lat1, lon1, lat2, lon2)

    if u == v:
        return 0.0, 0.0, 0.0

    fixed, var, circuit, mph = band_lookup(raw, mileage_bands)
    dist = raw * circuit

    return dist, fixed + var * dist, mph


def _get_container_strategy_capacity(
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        w_cube: float
) -> float:
    """Calculate effective truck cube for container strategy."""
    gaylord_row = container_params[
        container_params["container_type"].str.lower() == "gaylord"
        ].iloc[0]

    usable_cube = float(gaylord_row["usable_cube_cuft"])
    pack_util = float(gaylord_row["pack_utilization_container"])
    containers_per_truck = int(gaylord_row["containers_per_truck"])

    effective_container_cube = usable_cube * pack_util
    return containers_per_truck * effective_container_cube


def _get_fluid_strategy_capacity(
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        w_cube: float
) -> float:
    """Calculate effective truck cube for fluid strategy."""
    trailer_cube = float(container_params["trailer_air_cube_cuft"].iloc[0])
    pack_util = float(container_params["pack_utilization_fluid"].iloc[0])
    return trailer_cube * pack_util


def _calculate_processing_cost(
        path_data: Dict,
        sort_level: str,
        effective_strategy: str,
        cost_params: CostParameters,
        facilities: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame
) -> float:
    """Calculate per-package processing cost based on sort level and strategy."""
    path_nodes = path_data['path_nodes']
    origin = path_data['origin']
    dest = path_data['dest']

    processing_cost = cost_params.injection_sort_cost_per_pkg

    if origin == dest:
        processing_cost += cost_params.last_mile_sort_cost_per_pkg
        processing_cost += cost_params.last_mile_delivery_cost_per_pkg
        return processing_cost

    intermediate_facilities = path_nodes[1:-1] if len(path_nodes) > 2 else []

    if intermediate_facilities:
        if sort_level == 'sort_group':
            num_containers = _estimate_containers_per_pkg(package_mix, container_params)
            processing_cost += (len(intermediate_facilities) *
                                cost_params.container_handling_cost *
                                num_containers)

        elif effective_strategy.lower() == 'container':
            num_containers = _estimate_containers_per_pkg(package_mix, container_params)
            processing_cost += (len(intermediate_facilities) *
                                cost_params.container_handling_cost *
                                num_containers)

        else:
            processing_cost += (len(intermediate_facilities) *
                                cost_params.intermediate_sort_cost_per_pkg)

    if sort_level == 'sort_group':
        processing_cost += 0.0
    else:
        processing_cost += cost_params.last_mile_sort_cost_per_pkg

    processing_cost += cost_params.last_mile_delivery_cost_per_pkg

    return processing_cost


def _estimate_containers_per_pkg(
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame
) -> float:
    """Estimate containers needed per package."""
    w_cube = weighted_pkg_cube(package_mix)

    gaylord_row = container_params[
        container_params["container_type"].str.lower() == "gaylord"
        ].iloc[0]

    raw_container_cube = float(gaylord_row["usable_cube_cuft"])
    pack_util = float(gaylord_row["pack_utilization_container"])
    effective_container_cube = raw_container_cube * pack_util

    return w_cube / effective_container_cube


def _create_sort_decision_variables(
        model: cp_model.CpModel,
        groups: Dict,
        cand: pd.DataFrame,
        facilities: pd.DataFrame,
        timing_params: TimingParameters
) -> Dict:
    """Create sort level decision variables with validity checking."""
    sort_decision = {}
    fac_lookup = facilities.set_index('facility_name')

    for group_name in groups.keys():
        scenario_id, origin, dest, day_type = group_name

        valid_levels = _get_valid_sort_levels(origin, dest, fac_lookup)

        group_sort_vars = {}
        for sort_level in ['region', 'market', 'sort_group']:
            if sort_level in valid_levels:
                group_sort_vars[sort_level] = model.NewBoolVar(
                    f"sort_{scenario_id}_{origin}_{dest}_{day_type}_{sort_level}"
                )
            else:
                group_sort_vars[sort_level] = None

        sort_decision[group_name] = group_sort_vars

        valid_vars = [var for var in group_sort_vars.values() if var is not None]
        if valid_vars:
            model.Add(sum(valid_vars) == 1)

    return sort_decision


def _get_valid_sort_levels(origin: str, dest: str, fac_lookup: pd.DataFrame) -> List[str]:
    """Determine valid sort levels for an OD pair."""
    if origin == dest:
        return ['sort_group']

    if dest in fac_lookup.index:
        dest_regional_hub = fac_lookup.at[dest, 'regional_sort_hub']

        if origin == dest_regional_hub:
            return ['market', 'sort_group']

    return ['region', 'market', 'sort_group']


def _add_sort_capacity_constraints(
        model: cp_model.CpModel,
        sort_decision: Dict,
        groups: Dict,
        path_od_data: Dict,
        facilities: pd.DataFrame,
        timing_params: TimingParameters,
        cand: pd.DataFrame
) -> None:
    """Add sort capacity constraints per facility based on unique destinations."""
    fac_lookup = facilities.set_index('facility_name')
    sort_points_per_dest = timing_params.sort_points_per_destination

    hub_facilities = facilities[
        facilities['type'].str.lower().isin(['hub', 'hybrid'])
    ]['facility_name'].tolist()

    print(f"    Adding sort capacity constraints for {len(hub_facilities)} hub/hybrid facilities")

    for facility_name in hub_facilities:
        if facility_name not in fac_lookup.index:
            continue

        max_capacity = fac_lookup.at[facility_name, 'max_sort_points_capacity']
        if pd.isna(max_capacity) or max_capacity <= 0:
            continue

        max_capacity = int(max_capacity)

        unique_destinations = set()
        destination_sort_levels = {}
        dest_to_regional_hub = {}

        for group_name, group_idxs in groups.items():
            scenario_id, origin, dest, day_type = group_name

            if origin != facility_name:
                continue

            unique_destinations.add(dest)

            if dest not in destination_sort_levels:
                destination_sort_levels[dest] = sort_decision[group_name]

            if dest in fac_lookup.index:
                regional_hub = fac_lookup.at[dest, 'regional_sort_hub']
                dest_to_regional_hub[dest] = regional_hub if not pd.isna(regional_hub) else dest
            else:
                dest_to_regional_hub[dest] = dest

        if not unique_destinations:
            continue

        regional_hub_to_dests = {}
        for dest in unique_destinations:
            hub = dest_to_regional_hub[dest]
            if hub not in regional_hub_to_dests:
                regional_hub_to_dests[hub] = []
            regional_hub_to_dests[hub].append(dest)

        unique_regional_hubs = len(regional_hub_to_dests)

        print(
            f"      {facility_name}: {len(unique_destinations)} unique destinations, {unique_regional_hubs} regional hubs, capacity={max_capacity}")

        facility_sort_points = model.NewIntVar(0, max_capacity, f"sort_points_{facility_name}")

        sort_point_terms = []

        for regional_hub, dests_in_hub in regional_hub_to_dests.items():
            hub_region_var = model.NewBoolVar(f"hub_region_{facility_name}_{regional_hub}")

            region_vars_for_hub = []
            for dest in dests_in_hub:
                sort_vars = destination_sort_levels.get(dest)
                if sort_vars and sort_vars.get('region') is not None:
                    region_vars_for_hub.append(sort_vars['region'])

            if region_vars_for_hub:
                for var in region_vars_for_hub:
                    model.Add(hub_region_var >= var)
                model.Add(hub_region_var <= sum(region_vars_for_hub))

                regional_points = int(sort_points_per_dest)
                sort_point_terms.append(hub_region_var * regional_points)

        for dest in unique_destinations:
            sort_vars = destination_sort_levels.get(dest)
            if not sort_vars:
                continue

            if sort_vars.get('market') is not None:
                market_points = int(sort_points_per_dest)
                sort_point_terms.append(sort_vars['market'] * market_points)

            if sort_vars.get('sort_group') is not None:
                if dest in fac_lookup.index:
                    sort_groups = fac_lookup.at[dest, 'last_mile_sort_groups_count']
                    if pd.isna(sort_groups) or sort_groups <= 0:
                        sort_groups = 4
                    sort_groups = int(sort_groups)
                    sort_group_points = int(sort_points_per_dest * sort_groups)
                    sort_point_terms.append(sort_vars['sort_group'] * sort_group_points)

        if sort_point_terms:
            model.Add(facility_sort_points >= sum(sort_point_terms))
            model.Add(facility_sort_points <= max_capacity)

            min_possible = unique_regional_hubs * int(sort_points_per_dest)
            market_possible = len(unique_destinations) * int(sort_points_per_dest)
            max_possible = sum([
                int(sort_points_per_dest * fac_lookup.at[dest, 'last_mile_sort_groups_count'])
                if dest in fac_lookup.index else int(sort_points_per_dest * 4)
                for dest in unique_destinations
            ])

            print(
                f"        Sort points: {min_possible} (region) | {market_possible} (market) | {max_possible} (sort_group)")

            if min_possible > max_capacity:
                print(f"        ⚠️  WARNING: {facility_name} needs {min_possible} but only has {max_capacity}")
        else:
            model.Add(facility_sort_points == 0)


def _determine_arc_strategies_from_solution(
        arc_meta: List[Dict],
        chosen_idx: List[int],
        path_od_data: Dict,
        path_arcs: Dict,
        global_strategy: LoadStrategy
) -> Dict[int, str]:
    """Determine effective strategy per arc based on selected paths only."""
    arc_strategy_effective = {}

    for a_idx in range(len(arc_meta)):
        strategies_using_arc = []
        for i in chosen_idx:
            if a_idx in path_arcs[i]:
                strategies_using_arc.append(path_od_data[i]['effective_strategy'])

        if strategies_using_arc:
            strategy_counts = {}
            for s in strategies_using_arc:
                strategy_counts[s] = strategy_counts.get(s, 0) + 1
            predominant_strategy = max(strategy_counts, key=strategy_counts.get)
        else:
            predominant_strategy = global_strategy.value

        arc_strategy_effective[a_idx] = predominant_strategy

    return arc_strategy_effective


def _build_selected_paths_dataframe(
        chosen_idx: List[int],
        path_od_data: Dict,
        path_arcs: Dict,
        arc_meta: List[Dict],
        arc_trucks: Dict,
        arc_pkgs: Dict,
        sort_decisions: Dict,
        cost_params: CostParameters,
        facilities: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        solver: cp_model.CpSolver
) -> pd.DataFrame:
    """Build selected paths DataFrame with costs and metrics."""
    selected_paths_data = []

    for i in chosen_idx:
        path_data = path_od_data[i]

        group_key = (path_data['scenario_id'], path_data['origin'],
                     path_data['dest'], path_data['day_type'])
        chosen_sort_level = sort_decisions.get(group_key, 'market')

        total_transport_cost = 0
        for a_idx in path_arcs[i]:
            arc = arc_meta[a_idx]
            trucks_on_arc = solver.Value(arc_trucks[a_idx])
            if trucks_on_arc > 0:
                arc_total_pkgs = solver.Value(arc_pkgs[a_idx])
                path_share = path_data['pkgs_day'] / max(arc_total_pkgs, 1e-9)
                allocated_cost = trucks_on_arc * arc['cost_per_truck'] * path_share
                total_transport_cost += allocated_cost

        processing_cost_per_pkg = _calculate_processing_cost(
            path_data, chosen_sort_level, path_data['effective_strategy'],
            cost_params, facilities, package_mix, container_params
        )
        total_processing_cost = processing_cost_per_pkg * path_data['pkgs_day']

        total_cost = total_transport_cost + total_processing_cost

        selected_paths_data.append({
            **path_data,
            'total_cost': total_cost,
            'linehaul_cost': total_transport_cost,
            'processing_cost': total_processing_cost,
            'cost_per_pkg': total_cost / path_data['pkgs_day'],
            'chosen_sort_level': chosen_sort_level if sort_decisions else 'market'
        })

    return pd.DataFrame(selected_paths_data)


def _build_arc_summary_dataframe(
        arc_meta: List[Dict],
        arc_pkgs: Dict,
        arc_trucks: Dict,
        arc_strategy_effective: Dict,
        w_cube: float,
        raw_trailer_cube: float,
        container_params: pd.DataFrame,
        package_mix: pd.DataFrame,
        cost_params: CostParameters,
        chosen_idx: List[int],
        path_od_data: Dict,
        path_arcs: Dict,
        solver: cp_model.CpSolver
) -> pd.DataFrame:
    """Build arc summary DataFrame with fill rates and costs."""
    arc_summary_data = []

    gaylord_row = container_params[
        container_params["container_type"].str.lower() == "gaylord"
        ].iloc[0]
    raw_container_cube = float(gaylord_row["usable_cube_cuft"])

    for a_idx in range(len(arc_meta)):
        arc = arc_meta[a_idx]
        pkgs = solver.Value(arc_pkgs[a_idx])
        trucks = solver.Value(arc_trucks[a_idx])

        if pkgs > 0:
            total_cost = trucks * arc['cost_per_truck']
            cube = pkgs * w_cube

            strategy = arc_strategy_effective[a_idx]

            if strategy.lower() == 'container':
                pack_util = float(gaylord_row["pack_utilization_container"])
                effective_container_cube = raw_container_cube * pack_util
                actual_containers = max(1, int(np.ceil(cube / effective_container_cube)))

                container_fill_rate = min(1.0, cube / (actual_containers * raw_container_cube))
                truck_fill_rate = min(1.0, cube / (trucks * raw_trailer_cube))
            else:
                container_fill_rate = 0.0
                actual_containers = 0
                truck_fill_rate = min(1.0, cube / (trucks * raw_trailer_cube))

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
                "cost_per_truck": arc["cost_per_truck"],
                "total_cost": total_cost,
                "CPP": total_cost / pkgs,
                "effective_strategy": strategy
            })

    return pd.DataFrame(arc_summary_data).sort_values(
        ["scenario_id", "day_type", "from_facility", "to_facility"]
    ).reset_index(drop=True)


def _calculate_network_kpis(
        selected_paths: pd.DataFrame,
        arc_summary: pd.DataFrame,
        raw_trailer_cube: float
) -> Dict[str, float]:
    """Calculate network-level KPIs."""
    network_kpis = {}

    if not arc_summary.empty:
        total_cube = arc_summary['pkg_cube_cuft'].sum()
        total_capacity = (arc_summary['trucks'] * raw_trailer_cube).sum()
        network_truck_fill = total_cube / total_capacity if total_capacity > 0 else 0

        total_volume = arc_summary['pkgs_day'].sum()
        if total_volume > 0:
            network_container_fill = (
                                             arc_summary['container_fill_rate'] * arc_summary['pkgs_day']
                                     ).sum() / total_volume
        else:
            network_container_fill = 0.0

        network_kpis = {
            "avg_truck_fill_rate": max(0.0, min(1.0, network_truck_fill)),
            "avg_container_fill_rate": max(0.0, min(1.0, network_container_fill)),
            "total_cost": selected_paths['total_cost'].sum() if not selected_paths.empty else 0,
            "total_packages": selected_paths['pkgs_day'].sum() if not selected_paths.empty else 0
        }
    else:
        network_kpis = {
            "avg_truck_fill_rate": 0.0,
            "avg_container_fill_rate": 0.0,
            "total_cost": 0,
            "total_packages": 0
        }

    return network_kpis


def _build_sort_summary_dataframe(
        selected_paths: pd.DataFrame,
        sort_decisions: Dict,
        cost_params: CostParameters,
        facilities: pd.DataFrame
) -> pd.DataFrame:
    """Build sort decision summary DataFrame."""
    summary_data = []

    for _, od_row in selected_paths.iterrows():
        group_key = (od_row['scenario_id'], od_row['origin'],
                     od_row['dest'], od_row['day_type'])
        chosen_sort_level = sort_decisions.get(group_key, 'market')

        summary_data.append({
            'origin': od_row['origin'],
            'dest': od_row['dest'],
            'pkgs_day': od_row['pkgs_day'],
            'chosen_sort_level': chosen_sort_level,
            'total_cost': od_row['total_cost'],
            'processing_cost': od_row['processing_cost']
        })

    return pd.DataFrame(summary_data)