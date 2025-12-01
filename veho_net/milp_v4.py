"""
MILP Network Optimization Module

Solves minimum-cost network configuration using arc-pooled Mixed Integer Linear Programming.
Uses container loading strategy as baseline with optional sort level optimization.
Fluid load opportunities are identified post-optimization via separate analysis.

Key Features:
- Arc-pooled formulation for computational efficiency
- Sort level optimization (region/market/sort_group)
- Capacity constraints at origin facilities
- Premium economy dwell logic for partial truck utilization
"""

from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional

from .containers_v4 import (
    weighted_pkg_cube,
    get_container_capacity,
    get_containers_per_truck,
    get_trailer_capacity,
    get_raw_trailer_cube,
    calculate_containers_per_package,
)
from .geo_v4 import haversine_miles, band_lookup
from .config_v4 import (
    CostParameters,
    TimingParameters,
    LoadStrategy,
    OptimizationConstants,
)
from .utils import safe_divide, get_facility_lookup, extract_path_nodes

MAX_SAFE_INT = 2 ** 31 - 1

# Valid sort levels - used for validation
VALID_SORT_LEVELS = frozenset(['region', 'market', 'sort_group'])


def safe_int_cost(value: float, context: str = "") -> int:
    """Convert cost to safe integer for MILP solver."""
    int_val = int(round(value))
    if abs(int_val) > MAX_SAFE_INT:
        raise ValueError(
            f"Cost overflow in {context}: {value:,.0f} exceeds solver bounds. "
            f"Consider reducing cost values or scaling package volumes."
        )
    return int_val


def _build_arc_structures(cand, facilities, mileage_bands, path_keys):
    """Build arc-pooling data structures."""
    arc_index_map = {}
    arc_meta = []
    path_arcs = {}
    path_od_data = {}

    for i in path_keys:
        r = cand.loc[i]
        nodes = extract_path_nodes(r)

        legs = []
        for u, v in zip(nodes[:-1], nodes[1:]):
            legs.append(_calc_arc(u, v, facilities, mileage_bands))

        path_od_data[i] = {
            'origin': r["origin"],
            'dest': r["dest"],
            'pkgs_day': float(r["pkgs_day"]),
            'scenario_id': r.get("scenario_id", "default"),
            'day_type': r.get("day_type", "peak"),
            'path_str': r.get("path_str", "->".join(nodes)),
            'path_type': r.get("path_type", "direct"),
            'path_nodes': nodes,
            'effective_strategy': r.get("strategy_hint") or "container"
        }

        arc_ids = []
        for (u, v, dist, cost, mph) in legs:
            key = (u, v)
            if key not in arc_index_map:
                arc_index_map[key] = len(arc_meta)
                arc_meta.append({
                    "from": u, "to": v,
                    "distance_miles": dist,
                    "cost_per_truck": cost,
                    "mph": mph
                })
            arc_ids.append(arc_index_map[key])

        path_arcs[i] = arc_ids

    return arc_index_map, arc_meta, path_arcs, path_od_data


def _calc_arc(u, v, facilities, mileage_bands):
    """Calculate arc metrics from mileage_bands."""
    fac = facilities.set_index("facility_name")[["lat", "lon"]].astype(float)

    lat1, lon1 = fac.at[u, "lat"], fac.at[u, "lon"]
    lat2, lon2 = fac.at[v, "lat"], fac.at[v, "lon"]

    raw = haversine_miles(lat1, lon1, lat2, lon2)

    if u == v:
        return (u, v, 0.0, 0.0, 0.0)

    fixed, var, circuit, mph = band_lookup(raw, mileage_bands)
    dist = raw * circuit

    return (u, v, dist, fixed + var * dist, mph)


def _calculate_processing_cost(path_data, sort_level, strategy, cost_params,
                               facilities, package_mix, container_params):
    """
    Calculate per-package processing cost.

    Handles injection, intermediate, and last-mile costs based on sort level.

    Sort Level Cost Logic:
    - Region: Freight unsorted to regional hub → intermediate facilities do full sort
    - Market: Freight sorted to destination → intermediate facilities crossdock only
    - Sort_group: Freight pre-sorted to route groups → intermediate facilities crossdock only

    Raises:
        ValueError: If sort_level is not one of: region, market, sort_group
    """
    # Validate sort_level upfront
    if sort_level not in VALID_SORT_LEVELS:
        raise ValueError(
            f"Invalid sort_level '{sort_level}'. "
            f"Expected one of: {sorted(VALID_SORT_LEVELS)}"
        )

    nodes = path_data['path_nodes']
    origin = path_data['origin']
    dest = path_data['dest']

    # Injection sort at origin (always required)
    cost = cost_params.injection_sort_cost_per_pkg

    # O=D special case: no linehaul, just last-mile
    if origin == dest:
        cost += cost_params.last_mile_sort_cost_per_pkg
        cost += cost_params.last_mile_delivery_cost_per_pkg
        return cost

    # Intermediate facility costs
    intermediates = nodes[1:-1] if len(nodes) > 2 else []

    if intermediates:
        containers_per_package = calculate_containers_per_package(
            package_mix, container_params
        )

        if sort_level == 'region':
            # Region sort: freight unsorted → intermediates must do full sort
            cost += len(intermediates) * cost_params.intermediate_sort_cost_per_pkg

        elif sort_level == 'market':
            # Market sort: freight sorted to destination
            if strategy.lower() == 'container':
                # Container: crossdock only (handle containers)
                cost += len(intermediates) * cost_params.container_handling_cost * containers_per_package
            else:
                # Fluid: must sort at intermediate
                cost += len(intermediates) * cost_params.intermediate_sort_cost_per_pkg

        elif sort_level == 'sort_group':
            # Sort_group: freight pre-sorted to route groups → crossdock only
            cost += len(intermediates) * cost_params.container_handling_cost * containers_per_package

    # Last-mile costs
    if sort_level != 'sort_group':
        # Region and market require last-mile sort
        cost += cost_params.last_mile_sort_cost_per_pkg

    cost += cost_params.last_mile_delivery_cost_per_pkg

    return cost


def _create_sort_decision_variables(model, groups, cand, facilities, timing_params):
    """Create sort level decision variables."""
    sort_decision = {}
    fac_lookup = get_facility_lookup(facilities)

    for group_name in groups.keys():
        scenario_id, origin, dest, day_type = group_name

        valid_levels = _get_valid_sort_levels(origin, dest, fac_lookup)

        group_vars = {}
        for level in ['region', 'market', 'sort_group']:
            if level in valid_levels:
                group_vars[level] = model.NewBoolVar(
                    f"sort_{scenario_id}_{origin}_{dest}_{day_type}_{level}"
                )
            else:
                group_vars[level] = None

        sort_decision[group_name] = group_vars

        valid_vars = [v for v in group_vars.values() if v is not None]
        if valid_vars:
            model.Add(sum(valid_vars) == 1)

    return sort_decision


def _get_valid_sort_levels(origin, dest, fac_lookup):
    """Determine valid sort levels for OD pair."""
    if origin == dest:
        return ['sort_group']

    regional_hub_map = {}
    for facility in fac_lookup.index:
        hub = fac_lookup.at[facility, 'regional_sort_hub']
        if pd.isna(hub) or hub == "":
            hub = facility
        regional_hub_map[facility] = hub

    dest_hub = regional_hub_map.get(dest, dest)

    if origin == dest_hub:
        return ['market', 'sort_group']

    return ['region', 'market', 'sort_group']


def _add_sort_capacity_constraints(model, sort_decision, groups, path_od_data,
                                   facilities, timing_params, cand, scenario_row=None):
    """Enforce sort capacity limits with proper point counting."""
    fac_lookup = get_facility_lookup(facilities)

    if hasattr(timing_params, 'sort_points_per_destination'):
        sort_points_per_dest = float(timing_params.sort_points_per_destination)
    else:
        sort_points_per_dest = float(timing_params['sort_points_per_destination'])

    hub_facilities = facilities[
        facilities['type'].str.lower().isin(['hub', 'hybrid'])
    ]['facility_name'].tolist()

    facility_day_combos = set()
    for group_name in groups.keys():
        scenario_id, origin, dest, day_type = group_name
        if origin in hub_facilities:
            facility_day_combos.add((origin, day_type))

    scenario_capacity_override = None
    if scenario_row is not None and 'max_sort_points_capacity' in scenario_row.index:
        override_val = scenario_row['max_sort_points_capacity']
        if pd.notna(override_val) and override_val > 0:
            scenario_capacity_override = int(override_val)

    for (facility, day_type) in facility_day_combos:
        if facility not in fac_lookup.index:
            continue

        if scenario_capacity_override is not None:
            max_capacity = scenario_capacity_override
        else:
            max_capacity = fac_lookup.at[facility, 'max_sort_points_capacity']
            if pd.isna(max_capacity) or max_capacity <= 0:
                continue
            max_capacity = int(max_capacity)

        facility_groups = [
            (group_name, group_idxs)
            for group_name, group_idxs in groups.items()
            if group_name[1] == facility and group_name[3] == day_type
        ]

        if not facility_groups:
            continue

        regional_hubs = {}
        dest_to_hub = {}

        for (group_name, _) in facility_groups:
            _, _, dest, _ = group_name

            if dest in fac_lookup.index:
                hub = fac_lookup.at[dest, 'regional_sort_hub']
                if pd.isna(hub) or hub == '':
                    hub = dest
                dest_to_hub[dest] = hub

                if hub not in regional_hubs:
                    regional_hubs[hub] = []
                regional_hubs[hub].append(dest)

        facility_sort_points = model.NewIntVar(
            0, max_capacity,
            f"sort_pts_{facility}_{day_type}"
        )

        point_terms = []

        for region_hub, dests_in_region in regional_hubs.items():
            region_var = model.NewBoolVar(
                f"region_{facility}_{day_type}_{region_hub}"
            )

            region_sort_vars = []
            for (group_name, _) in facility_groups:
                _, _, dest, _ = group_name
                if dest_to_hub.get(dest) == region_hub:
                    sort_vars = sort_decision.get(group_name, {})
                    if sort_vars.get('region') is not None:
                        region_sort_vars.append(sort_vars['region'])

            if region_sort_vars:
                model.AddMaxEquality(region_var, region_sort_vars)
                point_terms.append(region_var * int(sort_points_per_dest))

        for (group_name, _) in facility_groups:
            _, _, dest, _ = group_name
            sort_vars = sort_decision.get(group_name, {})

            if sort_vars.get('market') is not None:
                point_terms.append(
                    sort_vars['market'] * int(sort_points_per_dest)
                )

        for (group_name, _) in facility_groups:
            _, _, dest, _ = group_name
            sort_vars = sort_decision.get(group_name, {})

            if sort_vars.get('sort_group') is not None:
                if dest not in fac_lookup.index:
                    raise ValueError(
                        f"Destination facility '{dest}' not found in facilities master data"
                    )

                groups_count = fac_lookup.at[dest, 'last_mile_sort_groups_count']
                if pd.isna(groups_count) or groups_count <= 0:
                    raise ValueError(
                        f"Destination facility '{dest}' missing valid last_mile_sort_groups_count"
                    )
                groups_count = int(groups_count)

                point_terms.append(
                    sort_vars['sort_group'] * int(sort_points_per_dest * groups_count)
                )

        if point_terms:
            model.Add(facility_sort_points == sum(point_terms))
            model.Add(facility_sort_points <= max_capacity)


def _create_decision_variables(
        model: cp_model.CpModel,
        path_keys: List[int],
        groups: Dict,
        arc_meta: List[Dict],
        enable_sort_optimization: bool,
        cand: pd.DataFrame,
        facilities: pd.DataFrame,
        timing_params: Dict
) -> Dict:
    """
    Create all MILP decision variables.

    Returns dictionary with:
        x: Path selection {path_idx: BoolVar}
        arc_pkgs: Packages per arc {arc_idx: IntVar}
        arc_trucks: Trucks per arc {arc_idx: IntVar}
        sort_decision: Sort choices {group_key: {level: BoolVar}} (if enabled)
    """
    variables = {}

    variables['x'] = {i: model.NewBoolVar(f"x_{i}") for i in path_keys}

    variables['arc_pkgs'] = {
        a_idx: model.NewIntVar(0, 1000000, f"arc_pkgs_{a_idx}")
        for a_idx in range(len(arc_meta))
    }

    variables['arc_trucks'] = {
        a_idx: model.NewIntVar(0, 1000, f"arc_trucks_{a_idx}")
        for a_idx in range(len(arc_meta))
    }

    if enable_sort_optimization:
        variables['sort_decision'] = _create_sort_decision_variables(
            model, groups, cand, facilities, timing_params
        )
    else:
        variables['sort_decision'] = {}

    return variables


def _add_od_selection_constraints(
        model: cp_model.CpModel,
        x: Dict[int, cp_model.IntVar],
        groups: Dict
) -> None:
    """
    Add one-path-per-OD constraints.

    For each (scenario, origin, dest, day_type) group, exactly one path must be selected.
    """
    for group_name, idxs in groups.items():
        model.Add(sum(x[i] for i in idxs) == 1)


def _add_arc_capacity_constraints(
        model: cp_model.CpModel,
        variables: Dict,
        arc_meta: List[Dict],
        path_arcs: Dict,
        path_od_data: Dict,
        path_keys: List[int],
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        global_strategy: LoadStrategy
) -> None:
    """
    Add arc capacity constraints.

    For each arc:
    1. Package volume = sum of selected paths using this arc
    2. Trucks needed >= packages * cube / truck_capacity
    3. At least 1 truck if arc has packages
    """
    x = variables['x']
    arc_pkgs = variables['arc_pkgs']
    arc_trucks = variables['arc_trucks']

    w_cube = weighted_pkg_cube(package_mix)
    raw_trailer_cube = get_raw_trailer_cube(container_params)
    container_capacity = get_container_capacity(container_params) * get_containers_per_truck(container_params)
    fluid_capacity = get_trailer_capacity(container_params)

    # Package aggregation constraints
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

    # Truck capacity constraints
    for a_idx in range(len(arc_meta)):
        # Determine predominant strategy for this arc
        strategies = [global_strategy.value]
        for i in path_keys:
            if a_idx in path_arcs[i]:
                strategies.append(path_od_data[i]['effective_strategy'])

        strategy_counts = {}
        for s in strategies:
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
        predominant = max(strategy_counts, key=strategy_counts.get)

        if predominant.lower() == "container":
            cap_scaled = int(container_capacity * OptimizationConstants.CUBE_SCALE_FACTOR)
        else:
            cap_scaled = int(fluid_capacity * OptimizationConstants.CUBE_SCALE_FACTOR)

        w_cube_scaled = int(w_cube * OptimizationConstants.CUBE_SCALE_FACTOR)

        # Trucks must fit packages
        model.Add(arc_trucks[a_idx] * cap_scaled >= arc_pkgs[a_idx] * w_cube_scaled)

        # At least 1 truck if arc has packages
        arc_has_pkgs = model.NewBoolVar(f"arc_has_{a_idx}")
        model.Add(arc_pkgs[a_idx] <= OptimizationConstants.BIG_M * arc_has_pkgs)
        model.Add(arc_pkgs[a_idx] >= 1 * arc_has_pkgs)
        model.Add(arc_trucks[a_idx] >= arc_has_pkgs)


def _build_objective_function(
        model: cp_model.CpModel,
        variables: Dict,
        arc_meta: List[Dict],
        path_keys: List[int],
        path_arcs: Dict,
        path_od_data: Dict,
        groups: Dict,
        enable_sort_optimization: bool,
        cost_params: CostParameters,
        facilities: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame
) -> int:
    """
    Build objective function: minimize total cost.

    Returns: Number of cost terms added (for logging)
    """
    x = variables['x']
    arc_trucks = variables['arc_trucks']
    sort_decision = variables['sort_decision']

    cost_terms = []

    # Transport costs
    for a_idx in range(len(arc_meta)):
        truck_cost = safe_int_cost(
            arc_meta[a_idx]["cost_per_truck"],
            f"arc_{a_idx}_transport"
        )
        cost_terms.append(arc_trucks[a_idx] * truck_cost)

    # Processing costs
    processing_terms = 0

    for group_name, group_idxs in groups.items():
        repr_idx = group_idxs[0]
        volume = path_od_data[repr_idx]['pkgs_day']

        if enable_sort_optimization:
            sort_vars = sort_decision[group_name]

            for sort_level in ['region', 'market', 'sort_group']:
                sort_var = sort_vars.get(sort_level)
                if sort_var is None:
                    continue

                for path_idx in group_idxs:
                    strategy = path_od_data[path_idx]['effective_strategy']

                    proc_cost = _calculate_processing_cost(
                        path_od_data[path_idx], sort_level, strategy,
                        cost_params, facilities, package_mix, container_params
                    )

                    total_cost = safe_int_cost(
                        proc_cost * volume,
                        f"path_{path_idx}_{sort_level}_proc"
                    )

                    cost_active = model.NewBoolVar(f"active_{path_idx}_{sort_level}")
                    model.Add(cost_active <= x[path_idx])
                    model.Add(cost_active <= sort_var)
                    model.Add(cost_active >= x[path_idx] + sort_var - 1)

                    cost_terms.append(cost_active * total_cost)
                    processing_terms += 1
        else:
            # Baseline: market sort for all
            for path_idx in group_idxs:
                strategy = path_od_data[path_idx]['effective_strategy']

                proc_cost = _calculate_processing_cost(
                    path_od_data[path_idx], 'market', strategy,
                    cost_params, facilities, package_mix, container_params
                )

                total_cost = safe_int_cost(
                    proc_cost * volume,
                    f"path_{path_idx}_baseline_proc"
                )

                cost_terms.append(x[path_idx] * total_cost)
                processing_terms += 1

    model.Minimize(sum(cost_terms))

    return len(arc_meta) + processing_terms


def solve_network_optimization(
        candidates: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        cost_params: CostParameters,
        timing_params: Dict,
        global_strategy: LoadStrategy,
        enable_sort_optimization: bool,
        scenario_row: pd.Series = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Optional[pd.DataFrame]]:
    """
    Solve network optimization using arc-pooled MILP.

    Decomposed into focused functions for clarity and maintainability:
    - Setup: Build arc structures and prepare data
    - Variables: Create decision variables
    - Constraints: Add OD selection, arc capacity, and sort capacity
    - Objective: Build cost minimization objective
    - Solve: Run CP-SAT solver
    - Extract: Build result dataframes

    ALL PARAMETERS FROM INPUT FILE - NO HARDCODING.
    """
    # ========================================================================
    # SETUP
    # ========================================================================
    cand = candidates.reset_index(drop=True).copy()

    print(f"    Strategy: {global_strategy.value}")
    print(f"    Sort optimization: {'ENABLED' if enable_sort_optimization else 'DISABLED'}")

    path_keys = list(cand.index)
    w_cube = weighted_pkg_cube(package_mix)

    arc_index_map, arc_meta, path_arcs, path_od_data = _build_arc_structures(
        cand, facilities, mileage_bands, path_keys
    )

    groups = cand.groupby(["scenario_id", "origin", "dest", "day_type"]).indices

    print(f"    Network: {len(arc_meta)} arcs, {len(path_keys)} paths, {len(groups)} OD pairs")

    # ========================================================================
    # CREATE MODEL AND VARIABLES
    # ========================================================================
    model = cp_model.CpModel()

    variables = _create_decision_variables(
        model, path_keys, groups, arc_meta,
        enable_sort_optimization, cand, facilities, timing_params
    )
    x = variables['x']
    arc_pkgs = variables['arc_pkgs']
    arc_trucks = variables['arc_trucks']
    sort_decision = variables['sort_decision']

    # ========================================================================
    # ADD CONSTRAINTS
    # ========================================================================
    _add_od_selection_constraints(model, x, groups)

    _add_arc_capacity_constraints(
        model, variables, arc_meta, path_arcs, path_od_data, path_keys,
        package_mix, container_params, global_strategy
    )

    if enable_sort_optimization:
        _add_sort_capacity_constraints(
            model, sort_decision, groups, path_od_data,
            facilities, timing_params, cand, scenario_row
        )

    print(f"    Constraints: {len(groups)} OD + {len(arc_meta)} arc capacity")

    # ========================================================================
    # BUILD OBJECTIVE
    # ========================================================================
    num_cost_terms = _build_objective_function(
        model, variables, arc_meta, path_keys, path_arcs, path_od_data,
        groups, enable_sort_optimization, cost_params, facilities,
        package_mix, container_params
    )

    print(f"    Objective: Minimize {num_cost_terms} cost terms")

    # ========================================================================
    # SOLVE
    # ========================================================================
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = OptimizationConstants.MAX_SOLVER_TIME_SECONDS
    solver.parameters.num_search_workers = OptimizationConstants.NUM_SOLVER_WORKERS

    print(f"    Solving MILP...")
    status = solver.Solve(model)

    status_messages = {
        cp_model.OPTIMAL: "OPTIMAL (proven minimum cost)",
        cp_model.FEASIBLE: "FEASIBLE (good solution, optimality not proven)",
        cp_model.INFEASIBLE: "INFEASIBLE (no valid solution exists)",
        cp_model.MODEL_INVALID: "MODEL INVALID (constraint error)",
        cp_model.UNKNOWN: "UNKNOWN (timeout or solver error)",
    }
    status_msg = status_messages.get(status, f"Unexpected status code: {status}")

    print(f"    Result: {status_msg}")
    print(f"    Solve time: {solver.WallTime():.2f}s")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"    ERROR: Optimization failed")

        empty_kpis = {
            "solver_status": status_msg,
            "solve_time": solver.WallTime(),
            "num_paths": len(path_keys),
            "num_arcs": len(arc_meta),
            "avg_truck_fill_rate": 0.0,
            "avg_container_fill_rate": 0.0,
            "total_cost": 0,
            "total_packages": 0
        }

        return pd.DataFrame(), pd.DataFrame(), empty_kpis, pd.DataFrame()

    # ========================================================================
    # EXTRACT SOLUTION
    # ========================================================================
    milp_objective = solver.ObjectiveValue()

    chosen_idx = [i for i in path_keys if solver.Value(x[i]) == 1]
    print(f"    Selected {len(chosen_idx)} optimal paths")

    # Extract sort decisions
    sort_decisions = {}
    if enable_sort_optimization:
        for group_name in groups.keys():
            for level in ['region', 'market', 'sort_group']:
                var = sort_decision[group_name].get(level)
                if var is not None and solver.Value(var) == 1:
                    sort_decisions[group_name] = level
                    break

    # Determine arc strategies
    arc_strategies = _determine_arc_strategies(
        arc_meta, chosen_idx, path_od_data, path_arcs, global_strategy
    )

    # Build result dataframes
    selected_paths = _build_selected_paths(
        chosen_idx, path_od_data, path_arcs, arc_meta, arc_trucks, arc_pkgs,
        sort_decisions, cost_params, facilities, package_mix, container_params,
        solver, enable_sort_optimization
    )

    arc_summary = _build_arc_summary(
        arc_meta, arc_pkgs, arc_trucks, arc_strategies, w_cube,
        get_raw_trailer_cube(container_params),
        container_params, package_mix, chosen_idx, path_od_data, path_arcs, solver
    )

    kpis = _calculate_kpis(selected_paths, arc_summary, get_raw_trailer_cube(container_params))

    sort_summary = pd.DataFrame()
    if enable_sort_optimization and sort_decisions:
        sort_summary = _build_sort_summary(selected_paths, sort_decisions, facilities)

    # ========================================================================
    # LOG RESULTS
    # ========================================================================
    actual_total_cost = selected_paths['total_cost'].sum()
    actual_total_pkgs = selected_paths['pkgs_day'].sum()
    actual_cpp = safe_divide(actual_total_cost, actual_total_pkgs)

    cost_delta = actual_total_cost - milp_objective
    delta_pct = safe_divide(abs(cost_delta), milp_objective) * 100

    print(f"    Optimization complete:")
    print(f"       Network cost: ${actual_total_cost:,.2f} (${actual_cpp:.3f}/pkg)")
    if abs(delta_pct) > 0.01:
        print(f"       (MILP integer objective: ${milp_objective:,.0f}, Δ=${cost_delta:+,.0f} from rounding)")

    return selected_paths, arc_summary, kpis, sort_summary


# ============================================================================
# SOLUTION EXTRACTION HELPERS
# ============================================================================

def _determine_arc_strategies(arc_meta, chosen_idx, path_od_data, path_arcs, global_strategy):
    """Determine effective strategy per arc."""
    arc_strategies = {}

    for a_idx in range(len(arc_meta)):
        strategies = []
        for i in chosen_idx:
            if a_idx in path_arcs[i]:
                strategies.append(path_od_data[i]['effective_strategy'])

        if strategies:
            counts = {}
            for s in strategies:
                counts[s] = counts.get(s, 0) + 1
            predominant = max(counts, key=counts.get)
        else:
            predominant = global_strategy.value

        arc_strategies[a_idx] = predominant

    return arc_strategies


def _build_selected_paths(chosen_idx, path_od_data, path_arcs, arc_meta,
                          arc_trucks, arc_pkgs, sort_decisions, cost_params,
                          facilities, package_mix, container_params, solver,
                          enable_sort_optimization):
    """Build selected paths DataFrame."""
    data = []

    for i in chosen_idx:
        path_data = path_od_data[i].copy()

        group_key = (path_data['scenario_id'], path_data['origin'],
                     path_data['dest'], path_data['day_type'])

        if enable_sort_optimization:
            sort_level = sort_decisions.get(group_key, 'market')
        else:
            sort_level = 'market'

        transport_cost = 0
        for a_idx in path_arcs[i]:
            arc = arc_meta[a_idx]
            trucks = solver.Value(arc_trucks[a_idx])
            arc_pkgs_val = solver.Value(arc_pkgs[a_idx])

            if arc_pkgs_val > 0:
                pkg_share = path_data['pkgs_day'] / arc_pkgs_val
                arc_cost = trucks * arc['cost_per_truck']
                transport_cost += arc_cost * pkg_share

        proc_cost_per_pkg = _calculate_processing_cost(
            path_data, sort_level, path_data['effective_strategy'],
            cost_params, facilities, package_mix, container_params
        )
        proc_cost_total = proc_cost_per_pkg * path_data['pkgs_day']

        total_cost = transport_cost + proc_cost_total

        nodes = path_data['path_nodes']

        data.append({
            'scenario_id': path_data['scenario_id'],
            'origin': path_data['origin'],
            'dest': path_data['dest'],
            'pkgs_day': path_data['pkgs_day'],
            'day_type': path_data['day_type'],
            'path_str': path_data['path_str'],
            'path_type': path_data['path_type'],
            'path_nodes': nodes,
            'effective_strategy': path_data['effective_strategy'],
            'total_cost': total_cost,
            'linehaul_cost': transport_cost,
            'processing_cost': proc_cost_total,
            'cost_per_pkg': safe_divide(total_cost, path_data['pkgs_day']),
            'chosen_sort_level': sort_level
        })

    return pd.DataFrame(data)


def _build_arc_summary(arc_meta, arc_pkgs, arc_trucks, arc_strategies, w_cube,
                       raw_trailer_cube, container_params, package_mix,
                       chosen_idx, path_od_data, path_arcs, solver):
    """Build arc summary DataFrame."""
    data = []

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

            strategy = arc_strategies[a_idx]

            if strategy.lower() == 'container':
                pack_util = float(gaylord_row["pack_utilization_container"])
                effective_cube = raw_container_cube * pack_util
                actual_containers = max(1, int(np.ceil(cube / effective_cube)))

                container_fill = min(1.0, cube / (actual_containers * raw_container_cube))
                truck_fill = min(1.0, cube / (trucks * raw_trailer_cube))
            else:
                container_fill = 0.0
                actual_containers = 0
                truck_fill = min(1.0, cube / (trucks * raw_trailer_cube))

            scenario_id = "default"
            day_type = "peak"
            for i in chosen_idx:
                if a_idx in path_arcs[i]:
                    scenario_id = path_od_data[i]['scenario_id']
                    day_type = path_od_data[i]['day_type']
                    break

            data.append({
                "scenario_id": scenario_id,
                "day_type": day_type,
                "from_facility": arc["from"],
                "to_facility": arc["to"],
                "distance_miles": arc["distance_miles"],
                "pkgs_day": pkgs,
                "pkg_cube_cuft": cube,
                "trucks": trucks,
                "physical_containers": actual_containers,
                "packages_per_truck": safe_divide(pkgs, trucks),
                "cube_per_truck": safe_divide(cube, trucks),
                "container_fill_rate": container_fill,
                "truck_fill_rate": truck_fill,
                "cost_per_truck": arc["cost_per_truck"],
                "total_cost": total_cost,
                "CPP": safe_divide(total_cost, pkgs),
                "effective_strategy": strategy
            })

    return pd.DataFrame(data).sort_values(
        ["scenario_id", "day_type", "from_facility", "to_facility"]
    ).reset_index(drop=True)


def _calculate_kpis(selected_paths, arc_summary, raw_trailer_cube):
    """Calculate network KPIs."""
    kpis = {}

    if not arc_summary.empty:
        total_cube = arc_summary['pkg_cube_cuft'].sum()
        total_capacity = (arc_summary['trucks'] * raw_trailer_cube).sum()
        truck_fill = safe_divide(total_cube, total_capacity)

        total_vol = arc_summary['pkgs_day'].sum()
        if total_vol > 0:
            container_fill = (
                                     arc_summary['container_fill_rate'] * arc_summary['pkgs_day']
                             ).sum() / total_vol
        else:
            container_fill = 0.0

        kpis = {
            "avg_truck_fill_rate": max(0.0, min(1.0, truck_fill)),
            "avg_container_fill_rate": max(0.0, min(1.0, container_fill)),
            "total_cost": selected_paths['total_cost'].sum() if not selected_paths.empty else 0,
            "total_packages": selected_paths['pkgs_day'].sum() if not selected_paths.empty else 0
        }
    else:
        kpis = {
            "avg_truck_fill_rate": 0.0,
            "avg_container_fill_rate": 0.0,
            "total_cost": 0,
            "total_packages": 0
        }

    return kpis


def _build_sort_summary(selected_paths, sort_decisions, facilities):
    """Build sort summary DataFrame."""
    data = []

    fac_lookup = get_facility_lookup(facilities)

    for _, od_row in selected_paths.iterrows():
        group_key = (od_row['scenario_id'], od_row['origin'],
                     od_row['dest'], od_row['day_type'])
        sort_level = sort_decisions.get(group_key, 'market')

        origin_hub = ''
        dest_hub = ''

        if od_row['origin'] in fac_lookup.index:
            hub = fac_lookup.at[od_row['origin'], 'regional_sort_hub']
            origin_hub = hub if not pd.isna(hub) and hub != "" else od_row['origin']

        if od_row['dest'] in fac_lookup.index:
            hub = fac_lookup.at[od_row['dest'], 'regional_sort_hub']
            dest_hub = hub if not pd.isna(hub) and hub != "" else od_row['dest']

        data.append({
            'origin': od_row['origin'],
            'origin_region_hub': origin_hub,
            'dest': od_row['dest'],
            'dest_region_hub': dest_hub,
            'pkgs_day': od_row['pkgs_day'],
            'chosen_sort_level': sort_level,
            'total_cost': od_row['total_cost'],
            'processing_cost': od_row['processing_cost']
        })

    return pd.DataFrame(data)