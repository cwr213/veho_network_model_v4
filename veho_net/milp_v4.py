"""
MILP Optimization Module - v4.3 COMPLETE FIX

Fixed Issues:
1. Baseline objective now includes processing costs (was $0, causing $682/pkg error)
2. Post-solve cost calculation matches objective function
3. Enhanced diagnostics and validation
4. Proper handling of enable_sort_optimization flag

Critical Fix: When enable_sort_optimization=False, processing costs were NOT
being added to the objective, resulting in $0 objective and incorrect baseline costs.
"""

from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Union

from .containers_v4 import (
    weighted_pkg_cube,
    get_container_capacity,
    get_containers_per_truck,
    get_trailer_capacity,
    get_raw_trailer_cube,
    estimate_containers_for_packages,
)
from .geo_v4 import haversine_miles, band_lookup
from .config_v4 import (
    CostParameters,
    TimingParameters,
    LoadStrategy,
    OptimizationConstants,
)
from .utils import safe_divide, get_facility_lookup

# Solver safety constants
MAX_SAFE_INT = 2 ** 31 - 1  # CP-SAT safe integer bound


def safe_int_cost(value: float, context: str = "") -> int:
    """Convert cost to safe integer for MILP solver."""
    int_val = int(round(value))
    if abs(int_val) > MAX_SAFE_INT:
        raise ValueError(
            f"Cost overflow in {context}: {value:,.0f} exceeds solver bounds. "
            f"Consider reducing cost values or scaling package volumes."
        )
    return int_val


# ============================================================================
# MAIN OPTIMIZATION FUNCTION
# ============================================================================

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

    Args:
        candidates: Candidate paths DataFrame
        facilities: Facility master data
        mileage_bands: Mileage bands for cost/distance
        package_mix: Package distribution
        container_params: Container/trailer parameters
        cost_params: Cost parameters
        timing_params: Timing parameters
        global_strategy: Loading strategy
        enable_sort_optimization: Whether to optimize sort level

    Returns:
        Tuple of (od_selected, arc_summary, network_kpis, sort_summary)
    """
    # Convert parameters to proper types
    cost_params = _ensure_cost_parameters(cost_params)
    timing_params = _ensure_timing_parameters(timing_params)
    global_strategy = _ensure_load_strategy(global_strategy)

    cand = candidates.reset_index(drop=True).copy()

    print(f"    MILP optimization: {global_strategy.value} strategy")
    if enable_sort_optimization:
        print(f"    Sort optimization: ENABLED")
    else:
        print(f"    Sort optimization: DISABLED (baseline = market sort)")

    path_keys = list(cand.index)
    w_cube = weighted_pkg_cube(package_mix)

    # Build arc structures
    arc_index_map, arc_meta, path_arcs, path_od_data = _build_arc_structures(
        cand, facilities, mileage_bands, path_keys
    )

    print(f"    Generated {len(arc_meta)} unique arcs from {len(cand)} paths")

    # Initialize MILP model
    model = cp_model.CpModel()

    # Path selection variables
    x = {i: model.NewBoolVar(f"x_{i}") for i in path_keys}

    # OD selection constraints (exactly one path per OD)
    groups = cand.groupby(["scenario_id", "origin", "dest", "day_type"]).indices

    for group_name, idxs in groups.items():
        model.Add(sum(x[i] for i in idxs) == 1)

    print(f"    Created {len(groups)} OD selection constraints")

    # Sort decision variables (if enabled)
    sort_decision = {}
    if enable_sort_optimization:
        sort_decision = _create_sort_decision_variables(
            model, groups, cand, facilities, timing_params
        )

    # Arc package flow variables
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

    # Calculate capacities
    raw_trailer_cube = get_raw_trailer_cube(container_params)
    container_capacity = get_container_capacity(container_params) * get_containers_per_truck(container_params)
    fluid_capacity = get_trailer_capacity(container_params)

    # Truck requirement variables with capacity constraints
    arc_trucks = {}

    for a_idx in range(len(arc_meta)):
        arc_trucks[a_idx] = model.NewIntVar(0, 1000, f"arc_trucks_{a_idx}")

        # Determine predominant strategy for arc
        strategies = [global_strategy.value]
        for i in path_keys:
            if a_idx in path_arcs[i]:
                strategies.append(path_od_data[i]['effective_strategy'])

        strategy_counts = {}
        for s in strategies:
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
        predominant = max(strategy_counts, key=strategy_counts.get)

        # Set capacity
        if predominant.lower() == "container":
            cap_scaled = int(container_capacity * OptimizationConstants.CUBE_SCALE_FACTOR)
        else:
            cap_scaled = int(fluid_capacity * OptimizationConstants.CUBE_SCALE_FACTOR)

        w_cube_scaled = int(w_cube * OptimizationConstants.CUBE_SCALE_FACTOR)

        # Capacity constraint
        model.Add(arc_trucks[a_idx] * cap_scaled >= arc_pkgs[a_idx] * w_cube_scaled)

        # Min 1 truck if packages exist
        arc_has_pkgs = model.NewBoolVar(f"arc_has_{a_idx}")
        model.Add(arc_pkgs[a_idx] <= OptimizationConstants.BIG_M * arc_has_pkgs)
        model.Add(arc_pkgs[a_idx] >= 1 * arc_has_pkgs)
        model.Add(arc_trucks[a_idx] >= arc_has_pkgs)

    # Sort capacity constraints (if enabled)
    if enable_sort_optimization:
        _add_sort_capacity_constraints(
            model, sort_decision, groups, path_od_data,
            facilities, timing_params, cand
        )

    # ========================================================================
    # BUILD OBJECTIVE - CRITICAL FIX HERE
    # ========================================================================

    cost_terms = []

    # Transportation costs
    for a_idx in range(len(arc_meta)):
        truck_cost = safe_int_cost(
            arc_meta[a_idx]["cost_per_truck"],
            f"arc_{a_idx}_transport"
        )
        cost_terms.append(arc_trucks[a_idx] * truck_cost)

    processing_cost_terms = 0

    # Processing costs
    for group_name, group_idxs in groups.items():
        repr_idx = group_idxs[0]
        volume = path_od_data[repr_idx]['pkgs_day']

        if enable_sort_optimization:
            # OPTIMIZED PATH: Use sort decision variables
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
                    processing_cost_terms += 1
        else:
            # BASELINE PATH: Fixed 'market' sort level
            # CRITICAL FIX: This else branch was missing or incomplete

            for path_idx in group_idxs:
                strategy = path_od_data[path_idx]['effective_strategy']

                # Calculate processing cost with MARKET sort level
                proc_cost = _calculate_processing_cost(
                    path_od_data[path_idx],
                    'market',  # EXPLICIT market sort for baseline
                    strategy,
                    cost_params,
                    facilities,
                    package_mix,
                    container_params
                )

                total_cost = safe_int_cost(
                    proc_cost * volume,
                    f"path_{path_idx}_baseline_proc"
                )

                # Add to objective: path selection * processing cost
                cost_terms.append(x[path_idx] * total_cost)
                processing_cost_terms += 1

    print(f"    Objective terms: {len(arc_meta)} transport + {processing_cost_terms} processing")

    # Set objective
    model.Minimize(sum(cost_terms))

    # ========================================================================
    # SOLVE
    # ========================================================================

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = OptimizationConstants.MAX_SOLVER_TIME_SECONDS
    solver.parameters.num_search_workers = OptimizationConstants.NUM_SOLVER_WORKERS

    print(f"    Starting MILP solver...")
    status = solver.Solve(model)

    # Enhanced diagnostics
    print(f"    Solver status: {status}")
    print(f"    Objective value: ${solver.ObjectiveValue():,.0f}")

    # Validate objective is reasonable
    if solver.ObjectiveValue() < 1000:
        print(f"    ⚠️  WARNING: Objective value suspiciously low!")
        print(f"    This suggests costs may not be in objective correctly.")

    # Enhanced error handling with diagnostics
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        error_msg = {
            cp_model.UNKNOWN: "Solver status unknown - may need more time",
            cp_model.MODEL_INVALID: "Model formulation invalid - check constraints",
            cp_model.INFEASIBLE: "No feasible solution - constraints too restrictive",
        }.get(status, f"Solver failed with status code: {status}")

        print(f"    ❌ Solver failed: {error_msg}")
        print(f"    Debug info:")
        print(f"      - Paths: {len(path_keys)}")
        print(f"      - Arcs: {len(arc_meta)}")
        print(f"      - OD groups: {len(groups)}")
        print(f"      - Variables: ~{len(path_keys) + len(arc_meta) * 2}")
        print(f"      - Constraints: ~{len(groups) + len(arc_meta) * 2}")
        print(f"      - Solve time: {solver.WallTime():.2f}s")

        # Return empty results WITH diagnostics
        empty_kpis = {
            "solver_status": error_msg,
            "solve_time": solver.WallTime(),
            "num_paths": len(path_keys),
            "num_arcs": len(arc_meta),
            "avg_truck_fill_rate": 0.0,
            "avg_container_fill_rate": 0.0,
            "total_cost": 0,
            "total_packages": 0
        }

        return pd.DataFrame(), pd.DataFrame(), empty_kpis, pd.DataFrame()

    print(f"    ✅ Solver completed: ${solver.ObjectiveValue():,.0f} in {solver.WallTime():.2f}s")

    # Extract solution
    chosen_idx = [i for i in path_keys if solver.Value(x[i]) == 1]

    sort_decisions = {}
    if enable_sort_optimization:
        for group_name in groups.keys():
            for level in ['region', 'market', 'sort_group']:
                var = sort_decision[group_name].get(level)
                if var is not None and solver.Value(var) == 1:
                    sort_decisions[group_name] = level
                    break

    arc_strategies = _determine_arc_strategies(
        arc_meta, chosen_idx, path_od_data, path_arcs, global_strategy
    )

    selected_paths = _build_selected_paths(
        chosen_idx, path_od_data, path_arcs, arc_meta, arc_trucks, arc_pkgs,
        sort_decisions, cost_params, facilities, package_mix, container_params,
        solver, enable_sort_optimization  # PASS THIS FLAG
    )

    arc_summary = _build_arc_summary(
        arc_meta, arc_pkgs, arc_trucks, arc_strategies, w_cube, raw_trailer_cube,
        container_params, package_mix, chosen_idx, path_od_data, path_arcs, solver
    )

    kpis = _calculate_kpis(selected_paths, arc_summary, raw_trailer_cube)

    sort_summary = pd.DataFrame()
    if enable_sort_optimization and sort_decisions:
        sort_summary = _build_sort_summary(selected_paths, sort_decisions, facilities)

    # Final validation
    od_total = selected_paths['total_cost'].sum()
    solver_obj = solver.ObjectiveValue()

    if abs(od_total - solver_obj) > 0.01 * solver_obj:
        print(f"    ⚠️  Cost mismatch: OD sum ${od_total:,.0f} vs Solver ${solver_obj:,.0f}")

    return selected_paths, arc_summary, kpis, sort_summary


# ============================================================================
# PARAMETER CONVERSION
# ============================================================================

def _ensure_cost_parameters(cp) -> CostParameters:
    """Convert to CostParameters if dict."""
    if isinstance(cp, CostParameters):
        return cp
    return CostParameters(
        injection_sort_cost_per_pkg=float(cp.get('injection_sort_cost_per_pkg', 0)),
        intermediate_sort_cost_per_pkg=float(cp.get('intermediate_sort_cost_per_pkg', 0)),
        last_mile_sort_cost_per_pkg=float(cp.get('last_mile_sort_cost_per_pkg', 0)),
        last_mile_delivery_cost_per_pkg=float(cp.get('last_mile_delivery_cost_per_pkg', 0)),
        container_handling_cost=float(cp.get('container_handling_cost', 0)),
        premium_economy_dwell_threshold=float(cp.get('premium_economy_dwell_threshold', 0.1)),
        dwell_cost_per_pkg_per_day=float(cp.get('dwell_cost_per_pkg_per_day', 0)),
        sla_penalty_per_touch_per_pkg=float(cp.get('sla_penalty_per_touch_per_pkg', 0))
    )


def _ensure_timing_parameters(tp) -> TimingParameters:
    """Convert to TimingParameters if dict."""
    if isinstance(tp, TimingParameters):
        return tp
    return TimingParameters(
        hours_per_touch=float(tp.get('hours_per_touch', 8)),
        load_hours=float(tp.get('load_hours', 2)),
        unload_hours=float(tp.get('unload_hours', 2)),
        injection_va_hours=float(tp.get('injection_va_hours', 8)),
        middle_mile_va_hours=float(tp.get('middle_mile_va_hours', 16)),
        crossdock_va_hours=float(tp.get('crossdock_va_hours', 3)),
        last_mile_va_hours=float(tp.get('last_mile_va_hours', 4)),
        sort_points_per_destination=float(tp.get('sort_points_per_destination', 1))
    )


def _ensure_load_strategy(gs) -> LoadStrategy:
    """Convert to LoadStrategy if string."""
    if isinstance(gs, LoadStrategy):
        return gs
    return LoadStrategy.CONTAINER if str(gs).lower() == 'container' else LoadStrategy.FLUID


# ============================================================================
# ARC STRUCTURE BUILDING
# ============================================================================

def _build_arc_structures(cand, facilities, mileage_bands, path_keys):
    """Build arc-pooling data structures."""
    arc_index_map = {}
    arc_meta = []
    path_arcs = {}
    path_od_data = {}

    for i in path_keys:
        r = cand.loc[i]
        legs = _extract_legs(r, facilities, mileage_bands)

        nodes = r.get("path_nodes", [r["origin"], r["dest"]])
        if not isinstance(nodes, list):
            nodes = [r["origin"], r["dest"]]

        path_od_data[i] = {
            'origin': r["origin"],
            'dest': r["dest"],
            'pkgs_day': float(r["pkgs_day"]),
            'scenario_id': r.get("scenario_id", "default"),
            'day_type': r.get("day_type", "peak"),
            'path_str': r.get("path_str", f"{r['origin']}->{r['dest']}"),
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


def _extract_legs(row, facilities, mileage_bands):
    """Extract legs from path."""
    nodes = row.get("path_nodes", None)

    if isinstance(nodes, (list, tuple)) and len(nodes) >= 2:
        nodes_list = list(nodes) if isinstance(nodes, tuple) else nodes
        return [_calc_arc(u, v, facilities, mileage_bands)
                for u, v in zip(nodes_list[:-1], nodes_list[1:])]

    return [_calc_arc(row["origin"], row["dest"], facilities, mileage_bands)]


def _calc_arc(u, v, facilities, mileage_bands):
    """Calculate arc metrics."""
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

    FIXED: Now properly accounts for sort_level in baseline runs.
    """
    nodes = path_data['path_nodes']
    origin = path_data['origin']
    dest = path_data['dest']

    # Start with injection sort cost
    cost = cost_params.injection_sort_cost_per_pkg

    # Handle O=D case
    if origin == dest:
        cost += cost_params.last_mile_sort_cost_per_pkg
        cost += cost_params.last_mile_delivery_cost_per_pkg
        return cost

    intermediates = nodes[1:-1] if len(nodes) > 2 else []

    # Intermediate handling costs
    if intermediates:
        if sort_level == 'sort_group':
            # Fully pre-sorted: just container handling
            containers_per_pkg = estimate_containers_for_packages(1, package_mix, container_params) / 1.0
            cost += len(intermediates) * cost_params.container_handling_cost * containers_per_pkg
        elif strategy.lower() == 'container':
            # Container strategy: container handling
            containers_per_pkg = estimate_containers_for_packages(1, package_mix, container_params) / 1.0
            cost += len(intermediates) * cost_params.container_handling_cost * containers_per_pkg
        else:
            # Fluid strategy: full sort
            cost += len(intermediates) * cost_params.intermediate_sort_cost_per_pkg

    # Last mile sort cost (if not fully pre-sorted)
    if sort_level != 'sort_group':
        cost += cost_params.last_mile_sort_cost_per_pkg

    # Last mile delivery cost (always)
    cost += cost_params.last_mile_delivery_cost_per_pkg

    return cost


# ============================================================================
# SORT OPTIMIZATION
# ============================================================================

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
                                   facilities, timing_params, cand):
    """Add sort capacity constraints per facility PER DAY TYPE."""
    fac_lookup = get_facility_lookup(facilities)
    sort_points_per_dest = timing_params.sort_points_per_destination

    hub_facilities = facilities[
        facilities['type'].str.lower().isin(['hub', 'hybrid'])
    ]['facility_name'].tolist()

    facility_day_combos = set()
    for group_name in groups.keys():
        scenario_id, origin, dest, day_type = group_name
        if origin in hub_facilities:
            facility_day_combos.add((origin, day_type))

    for (facility, day_type) in facility_day_combos:
        if facility not in fac_lookup.index:
            continue

        max_capacity = fac_lookup.at[facility, 'max_sort_points_capacity']
        if pd.isna(max_capacity) or max_capacity <= 0:
            continue

        max_capacity = int(max_capacity)

        unique_dests = set()
        dest_sort_levels = {}
        dest_to_hub = {}

        for group_name, group_idxs in groups.items():
            scenario_id, origin, dest, group_day_type = group_name

            if origin != facility or group_day_type != day_type:
                continue

            unique_dests.add(dest)

            if dest not in dest_sort_levels:
                dest_sort_levels[dest] = sort_decision[group_name]

            if dest in fac_lookup.index:
                hub = fac_lookup.at[dest, 'regional_sort_hub']
                dest_to_hub[dest] = hub if not pd.isna(hub) and hub != "" else dest
            else:
                dest_to_hub[dest] = dest

        if not unique_dests:
            continue

        hub_to_dests = {}
        for dest in unique_dests:
            hub = dest_to_hub[dest]
            if hub not in hub_to_dests:
                hub_to_dests[hub] = []
            hub_to_dests[hub].append(dest)

        facility_points = model.NewIntVar(
            0, max_capacity,
            f"points_{facility}_{day_type}"
        )

        point_terms = []

        for hub, dests in hub_to_dests.items():
            hub_region_var = model.NewBoolVar(f"hub_reg_{facility}_{day_type}_{hub}")

            region_vars = []
            for dest in dests:
                sort_vars = dest_sort_levels.get(dest)
                if sort_vars and sort_vars.get('region') is not None:
                    region_vars.append(sort_vars['region'])

            if region_vars:
                model.AddMaxEquality(hub_region_var, region_vars)
                point_terms.append(hub_region_var * int(sort_points_per_dest))

        for dest in unique_dests:
            sort_vars = dest_sort_levels.get(dest)
            if not sort_vars:
                continue

            if sort_vars.get('market') is not None:
                point_terms.append(sort_vars['market'] * int(sort_points_per_dest))

            if sort_vars.get('sort_group') is not None:
                if dest in fac_lookup.index:
                    groups_count = fac_lookup.at[dest, 'last_mile_sort_groups_count']
                    if pd.isna(groups_count) or groups_count <= 0:
                        groups_count = OptimizationConstants.DEFAULT_SORT_GROUPS
                    groups_count = int(groups_count)
                    point_terms.append(sort_vars['sort_group'] * int(sort_points_per_dest * groups_count))

        if point_terms:
            model.Add(facility_points >= sum(point_terms))
            model.Add(facility_points <= max_capacity)


# ============================================================================
# RESULT BUILDING
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
    """
    Build selected paths DataFrame.

    FIXED: Now uses enable_sort_optimization flag to match objective calculation.
    """
    data = []

    for i in chosen_idx:
        path_data = path_od_data[i]

        group_key = (path_data['scenario_id'], path_data['origin'],
                     path_data['dest'], path_data['day_type'])

        # CRITICAL FIX: Use same sort level logic as objective
        if enable_sort_optimization:
            sort_level = sort_decisions.get(group_key, 'market')
        else:
            sort_level = 'market'  # Baseline always uses market

        transport_cost = 0
        for a_idx in path_arcs[i]:
            arc = arc_meta[a_idx]
            trucks = solver.Value(arc_trucks[a_idx])
            if trucks > 0:
                arc_pkgs_val = solver.Value(arc_pkgs[a_idx])
                if arc_pkgs_val > 0:
                    share = path_data['pkgs_day'] / arc_pkgs_val
                    transport_cost += trucks * arc['cost_per_truck'] * share

        # CRITICAL: Use same processing cost calculation as objective
        proc_cost_per_pkg = _calculate_processing_cost(
            path_data,
            sort_level,  # Must match objective
            path_data['effective_strategy'],
            cost_params,
            facilities,
            package_mix,
            container_params
        )
        proc_cost_total = proc_cost_per_pkg * path_data['pkgs_day']

        total_cost = transport_cost + proc_cost_total

        data.append({
            **path_data,
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
    from .containers_v4 import get_container_capacity

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