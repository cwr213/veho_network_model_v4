"""
Reporting Module

Generates facility-level aggregations with sort vs. crossdock breakdowns,
zone classifications, and network metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from .geo_v3 import calculate_zone_from_distance
from .containers_v3 import weighted_pkg_cube


def _determine_intermediate_operation_type(
        intermediate_facility: str,
        dest_facility: str,
        path_strategy: str,
        chosen_sort_level: str,
        facilities: pd.DataFrame
) -> str:
    """
    Determine if intermediate facility needs sort or crossdock operation.

    Business Rules:
    - Fluid strategy: Always SORT (floor loaded packages must be sorted)
    - Container + region sort + facility is regional hub: SORT (break down regional containers)
    - Container + market/sort_group: CROSSDOCK (containers stay sealed, just move between trailers)

    Args:
        intermediate_facility: Facility performing intermediate operation
        dest_facility: Final destination facility
        path_strategy: Loading strategy for this path (container or fluid)
        chosen_sort_level: Sort granularity (region, market, sort_group)
        facilities: Facility master data

    Returns:
        'sort' or 'crossdock'
    """
    if path_strategy.lower() == 'fluid':
        return 'sort'

    if path_strategy.lower() == 'container':
        if chosen_sort_level == 'region':
            fac_lookup = facilities.set_index('facility_name')

            if dest_facility in fac_lookup.index:
                dest_regional_hub = fac_lookup.at[dest_facility, 'regional_sort_hub']
                if pd.isna(dest_regional_hub) or dest_regional_hub == '':
                    dest_regional_hub = dest_facility

                if intermediate_facility == dest_regional_hub:
                    return 'sort'
                else:
                    return 'crossdock'

        elif chosen_sort_level in ['market', 'sort_group']:
            return 'crossdock'

    return 'sort'


def build_facility_rollup(
        od_selected: pd.DataFrame,
        direct_day: pd.DataFrame,
        arc_summary: pd.DataFrame,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        strategy: str,
        facilities: pd.DataFrame,
        cost_params
) -> pd.DataFrame:
    """
    Calculate facility volume and cost breakdowns by role with sort vs. crossdock distinction.

    Cost attribution philosophy:
    - Origin facilities pay for all costs of packages they originate
    - Intermediate facilities show package volumes split by operation type (sort vs. crossdock)
    - Destination facilities pay for final mile costs
    """
    volume_data = []

    all_facilities = set()
    if not od_selected.empty:
        all_facilities.update(od_selected['origin'].unique())
        all_facilities.update(od_selected['dest'].unique())
    if not direct_day.empty and 'dest' in direct_day.columns:
        all_facilities.update(direct_day['dest'].unique())

    fac_lookup = facilities.set_index('facility_name')

    for facility in all_facilities:
        try:
            direct_injection_pkgs = 0
            direct_injection_containers = 0
            direct_injection_cube = 0.0

            if not direct_day.empty:
                direct_col = 'dir_pkgs_day'
                if direct_col in direct_day.columns:
                    facility_direct = direct_day[direct_day['dest'] == facility]
                    if not facility_direct.empty:
                        direct_injection_pkgs = facility_direct[direct_col].sum()

                        if direct_injection_pkgs > 0:
                            containers_calc = _calculate_containers_for_volume(
                                direct_injection_pkgs, package_mix, container_params, strategy
                            )
                            direct_injection_containers = containers_calc['containers']
                            direct_injection_cube = containers_calc['cube']

            mm_injection_pkgs = 0
            mm_injection_containers = 0
            mm_injection_cube = 0.0
            injection_sort_cost = 0.0
            injection_linehaul_cost = 0.0
            intermediate_processing_cost_origin = 0.0
            intermediate_linehaul_cost_origin = 0.0

            if not od_selected.empty:
                outbound_ods = od_selected[od_selected['origin'] == facility]
                if not outbound_ods.empty:
                    mm_injection_pkgs = outbound_ods['pkgs_day'].sum()
                    injection_sort_cost = cost_params.injection_sort_cost_per_pkg * mm_injection_pkgs

                    for _, od_row in outbound_ods.iterrows():
                        path_nodes = od_row.get('path_nodes', [])
                        if not isinstance(path_nodes, list):
                            continue

                        intermediate_facilities = path_nodes[1:-1] if len(path_nodes) > 2 else []
                        path_strategy = od_row.get('effective_strategy', strategy)

                        if intermediate_facilities:
                            if path_strategy.lower() == 'container':
                                num_containers_per_pkg = _estimate_containers_per_pkg(package_mix, container_params)
                                intermediate_processing_cost_origin += (len(intermediate_facilities) *
                                                                        cost_params.container_handling_cost * num_containers_per_pkg *
                                                                        od_row['pkgs_day'])
                            else:
                                intermediate_processing_cost_origin += (len(intermediate_facilities) *
                                                                        cost_params.intermediate_sort_cost_per_pkg *
                                                                        od_row['pkgs_day'])

                        if not arc_summary.empty:
                            for node_idx in range(len(path_nodes) - 1):
                                from_node = path_nodes[node_idx]
                                to_node = path_nodes[node_idx + 1]

                                matching_arc = arc_summary[
                                    (arc_summary['from_facility'] == from_node) &
                                    (arc_summary['to_facility'] == to_node)
                                    ]

                                if not matching_arc.empty:
                                    arc_row = matching_arc.iloc[0]
                                    if arc_row['pkgs_day'] > 0:
                                        path_share = od_row['pkgs_day'] / arc_row['pkgs_day']
                                        allocated_cost = arc_row['total_cost'] * path_share

                                        if node_idx == 0:
                                            injection_linehaul_cost += allocated_cost
                                        else:
                                            intermediate_linehaul_cost_origin += allocated_cost

                    if mm_injection_pkgs > 0:
                        containers_calc = _calculate_containers_for_volume(
                            mm_injection_pkgs, package_mix, container_params, strategy
                        )
                        mm_injection_containers = containers_calc['containers']
                        mm_injection_cube = containers_calc['cube']

            intermediate_pkgs_sort = 0
            intermediate_pkgs_crossdock = 0
            intermediate_containers_sort = 0
            intermediate_containers_crossdock = 0
            intermediate_cube_sort = 0.0
            intermediate_cube_crossdock = 0.0
            intermediate_sort_cost = 0.0
            intermediate_crossdock_cost = 0.0

            if not od_selected.empty:
                for _, path_row in od_selected.iterrows():
                    path_nodes = path_row.get('path_nodes', [])
                    if not isinstance(path_nodes, list):
                        continue

                    final_dest = path_row['dest']
                    path_strategy = path_row.get('effective_strategy', strategy)
                    chosen_sort_level = path_row.get('chosen_sort_level', 'market')

                    for node in path_nodes[1:-1]:
                        if node == facility:
                            operation = _determine_intermediate_operation_type(
                                intermediate_facility=facility,
                                dest_facility=final_dest,
                                path_strategy=path_strategy,
                                chosen_sort_level=chosen_sort_level,
                                facilities=facilities
                            )

                            pkgs = path_row['pkgs_day']

                            if operation == 'sort':
                                intermediate_pkgs_sort += pkgs
                                intermediate_sort_cost += cost_params.intermediate_sort_cost_per_pkg * pkgs
                            else:
                                intermediate_pkgs_crossdock += pkgs
                                num_containers_per_pkg = _estimate_containers_per_pkg(package_mix, container_params)
                                intermediate_crossdock_cost += cost_params.container_handling_cost * num_containers_per_pkg * pkgs

                            containers_calc = _calculate_containers_for_volume(
                                pkgs, package_mix, container_params, path_strategy
                            )

                            if operation == 'sort':
                                intermediate_containers_sort += containers_calc['containers']
                                intermediate_cube_sort += containers_calc['cube']
                            else:
                                intermediate_containers_crossdock += containers_calc['containers']
                                intermediate_cube_crossdock += containers_calc['cube']

            intermediate_pkgs = intermediate_pkgs_sort + intermediate_pkgs_crossdock
            intermediate_containers = intermediate_containers_sort + intermediate_containers_crossdock
            intermediate_cube = intermediate_cube_sort + intermediate_cube_crossdock

            last_mile_pkgs = direct_injection_pkgs
            last_mile_containers = direct_injection_containers
            last_mile_cube = direct_injection_cube
            last_mile_sort_cost = 0.0
            last_mile_delivery_cost = 0.0
            last_mile_linehaul_cost = 0.0

            if not od_selected.empty:
                inbound_ods = od_selected[od_selected['dest'] == facility]
                if not inbound_ods.empty:
                    mm_last_mile_pkgs = inbound_ods['pkgs_day'].sum()
                    last_mile_pkgs += mm_last_mile_pkgs

                    for _, od_row in inbound_ods.iterrows():
                        chosen_sort_level = od_row.get('chosen_sort_level', 'market')
                        if chosen_sort_level != 'sort_group':
                            last_mile_sort_cost += cost_params.last_mile_sort_cost_per_pkg * od_row['pkgs_day']

                        path_nodes = od_row.get('path_nodes', [])
                        if isinstance(path_nodes, list) and len(path_nodes) >= 2:
                            last_from = path_nodes[-2]
                            last_to = path_nodes[-1]

                            if not arc_summary.empty:
                                matching_arc = arc_summary[
                                    (arc_summary['from_facility'] == last_from) &
                                    (arc_summary['to_facility'] == last_to)
                                    ]

                                if not matching_arc.empty:
                                    arc_row = matching_arc.iloc[0]
                                    if arc_row['pkgs_day'] > 0:
                                        path_share = od_row['pkgs_day'] / arc_row['pkgs_day']
                                        last_mile_linehaul_cost += arc_row['total_cost'] * path_share

                    last_mile_delivery_cost = cost_params.last_mile_delivery_cost_per_pkg * mm_last_mile_pkgs

                    if mm_last_mile_pkgs > 0:
                        containers_calc = _calculate_containers_for_volume(
                            mm_last_mile_pkgs, package_mix, container_params, strategy
                        )
                        last_mile_containers += containers_calc['containers']
                        last_mile_cube += containers_calc['cube']

            if direct_injection_pkgs > 0:
                last_mile_delivery_cost += cost_params.last_mile_delivery_cost_per_pkg * direct_injection_pkgs

            total_daily_containers = mm_injection_containers + intermediate_containers + last_mile_containers

            if not od_selected.empty:
                od_paths = od_selected[od_selected['origin'] == facility]
                for _, od_row in od_paths.iterrows():
                    if od_row['origin'] == od_row['dest']:
                        od_containers = _calculate_containers_for_volume(
                            od_row['pkgs_day'], package_mix, container_params, strategy
                        )['containers']
                        total_daily_containers -= od_containers

            outbound_trucks = 0
            inbound_trucks = 0
            if not arc_summary.empty:
                outbound_arcs = arc_summary[
                    (arc_summary['from_facility'] == facility) &
                    (arc_summary['from_facility'] != arc_summary['to_facility'])
                    ]
                if not outbound_arcs.empty:
                    outbound_trucks = int(outbound_arcs['trucks'].sum())

                inbound_arcs = arc_summary[
                    (arc_summary['to_facility'] == facility) &
                    (arc_summary['from_facility'] != arc_summary['to_facility'])
                    ]
                if not inbound_arcs.empty:
                    inbound_trucks = int(inbound_arcs['trucks'].sum())

            total_cost = (injection_sort_cost + injection_linehaul_cost +
                          intermediate_processing_cost_origin + intermediate_linehaul_cost_origin +
                          last_mile_sort_cost + last_mile_delivery_cost + last_mile_linehaul_cost)

            total_pkgs = mm_injection_pkgs + intermediate_pkgs + last_mile_pkgs
            total_cpp = total_cost / total_pkgs if total_pkgs > 0 else 0

            volume_entry = {
                'facility': facility,
                'direct_injection_pkgs_day': direct_injection_pkgs,
                'direct_injection_containers': direct_injection_containers,
                'direct_injection_cube_cuft': direct_injection_cube,
                'middle_mile_injection_pkgs_day': mm_injection_pkgs,
                'middle_mile_injection_containers': mm_injection_containers,
                'middle_mile_injection_cube_cuft': mm_injection_cube,
                'middle_mile_intermediate_pkgs_sort': intermediate_pkgs_sort,
                'middle_mile_intermediate_pkgs_crossdock': intermediate_pkgs_crossdock,
                'middle_mile_intermediate_pkgs_day': intermediate_pkgs,
                'middle_mile_intermediate_containers_sort': intermediate_containers_sort,
                'middle_mile_intermediate_containers_crossdock': intermediate_containers_crossdock,
                'middle_mile_intermediate_containers': intermediate_containers,
                'middle_mile_intermediate_cube_sort_cuft': intermediate_cube_sort,
                'middle_mile_intermediate_cube_crossdock_cuft': intermediate_cube_crossdock,
                'middle_mile_intermediate_cube_cuft': intermediate_cube,
                'last_mile_pkgs_day': last_mile_pkgs,
                'last_mile_containers': last_mile_containers,
                'last_mile_cube_cuft': last_mile_cube,
                'injection_sort_cost': injection_sort_cost,
                'injection_sort_cpp': injection_sort_cost / mm_injection_pkgs if mm_injection_pkgs > 0 else 0,
                'intermediate_sort_cost': intermediate_sort_cost,
                'intermediate_sort_cpp': intermediate_sort_cost / intermediate_pkgs_sort if intermediate_pkgs_sort > 0 else 0,
                'intermediate_crossdock_cost': intermediate_crossdock_cost,
                'intermediate_crossdock_cpp': intermediate_crossdock_cost / intermediate_pkgs_crossdock if intermediate_pkgs_crossdock > 0 else 0,
                'intermediate_processing_cost': intermediate_processing_cost_origin,
                'intermediate_processing_cpp': intermediate_processing_cost_origin / mm_injection_pkgs if mm_injection_pkgs > 0 else 0,
                'injection_linehaul_cost': injection_linehaul_cost,
                'injection_linehaul_cpp': injection_linehaul_cost / mm_injection_pkgs if mm_injection_pkgs > 0 else 0,
                'intermediate_linehaul_cost': intermediate_linehaul_cost_origin,
                'intermediate_linehaul_cpp': intermediate_linehaul_cost_origin / mm_injection_pkgs if mm_injection_pkgs > 0 else 0,
                'last_mile_sort_cost': last_mile_sort_cost,
                'last_mile_sort_cpp': last_mile_sort_cost / last_mile_pkgs if last_mile_pkgs > 0 else 0,
                'last_mile_delivery_cost': last_mile_delivery_cost,
                'last_mile_delivery_cpp': last_mile_delivery_cost / last_mile_pkgs if last_mile_pkgs > 0 else 0,
                'last_mile_linehaul_cost': last_mile_linehaul_cost,
                'last_mile_linehaul_cpp': last_mile_linehaul_cost / last_mile_pkgs if last_mile_pkgs > 0 else 0,
                'total_cost': total_cost,
                'total_cpp': total_cpp,
                'total_daily_containers': max(0, total_daily_containers),
                'outbound_trucks': outbound_trucks,
                'inbound_trucks': inbound_trucks
            }

            volume_data.append(volume_entry)

        except Exception as e:
            print(f"    Warning: Error calculating volume for facility {facility}: {e}")
            continue

    return pd.DataFrame(volume_data)


def _calculate_containers_for_volume(
        packages: float,
        package_mix: pd.DataFrame,
        container_params: pd.DataFrame,
        strategy: str
) -> Dict[str, float]:
    """Calculate containers and cube for given package volume."""
    w_cube = weighted_pkg_cube(package_mix)
    total_cube = packages * w_cube

    if strategy.lower() == "container":
        gaylord_row = container_params[
            container_params["container_type"].str.lower() == "gaylord"
            ].iloc[0]
        raw_container_cube = float(gaylord_row["usable_cube_cuft"])
        pack_util = float(gaylord_row["pack_utilization_container"])
        effective_cube = raw_container_cube * pack_util

        containers = max(1, int(np.ceil(total_cube / effective_cube)))
    else:
        containers = 0

    return {'containers': containers, 'cube': total_cube}


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


def calculate_hourly_throughput(
        facility_rollup: pd.DataFrame,
        timing_params: Dict
) -> pd.DataFrame:
    """Calculate hourly throughput requirements with separate sort and crossdock rates."""
    df = facility_rollup.copy()

    injection_va_hours = float(timing_params['injection_va_hours'])
    middle_mile_va_hours = float(timing_params['middle_mile_va_hours'])
    crossdock_va_hours = float(timing_params.get('crossdock_va_hours', 3.0))
    last_mile_va_hours = float(timing_params['last_mile_va_hours'])

    df['injection_hourly_throughput'] = df['middle_mile_injection_pkgs_day'] / injection_va_hours
    df['intermediate_sort_hourly_throughput'] = df['middle_mile_intermediate_pkgs_sort'] / middle_mile_va_hours
    df['intermediate_crossdock_hourly_throughput'] = df['middle_mile_intermediate_pkgs_crossdock'] / crossdock_va_hours
    df['last_mile_hourly_throughput'] = df['last_mile_pkgs_day'] / last_mile_va_hours

    df['peak_hourly_throughput'] = df[[
        'injection_hourly_throughput',
        'intermediate_sort_hourly_throughput',
        'intermediate_crossdock_hourly_throughput',
        'last_mile_hourly_throughput'
    ]].max(axis=1)

    throughput_cols = [
        'injection_hourly_throughput',
        'intermediate_sort_hourly_throughput',
        'intermediate_crossdock_hourly_throughput',
        'last_mile_hourly_throughput',
        'peak_hourly_throughput'
    ]

    for col in throughput_cols:
        df[col] = df[col].fillna(0).round(0).astype(int)

    return df


def add_zone_classification(
        od_df: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame
) -> pd.DataFrame:
    """Add zone classification based on O-D straight-line distance."""
    if od_df.empty:
        return od_df

    od_df = od_df.copy()
    od_df['zone'] = 'unknown'

    for idx, row in od_df.iterrows():
        origin = row['origin']
        dest = row['dest']

        if origin == dest:
            od_df.at[idx, 'zone'] = '2'
        else:
            zone = calculate_zone_from_distance(origin, dest, facilities, mileage_bands)
            od_df.at[idx, 'zone'] = zone

    return od_df


def build_path_steps(
        od_selected: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame,
        timing_params: Dict
) -> pd.DataFrame:
    """Generate path steps from selected OD paths."""
    from .geo_v3 import haversine_miles, band_lookup

    path_steps = []

    fac_lookup = facilities.set_index('facility_name')[['lat', 'lon']].astype(float)
    hours_per_touch = float(timing_params['hours_per_touch'])

    for _, od_row in od_selected.iterrows():
        path_str = str(od_row.get('path_str', f"{od_row['origin']}->{od_row['dest']}"))
        scenario_id = od_row.get('scenario_id', 'default')

        if '->' not in path_str:
            continue

        nodes = [n.strip() for n in path_str.split('->')]

        for i, (from_fac, to_fac) in enumerate(zip(nodes[:-1], nodes[1:])):
            if from_fac == to_fac:
                from_lat = fac_lookup.at[from_fac, 'lat'] if from_fac in fac_lookup.index else 0
                from_lon = fac_lookup.at[from_fac, 'lon'] if from_fac in fac_lookup.index else 0
                to_lat = from_lat
                to_lon = from_lon

                path_steps.append({
                    'scenario_id': scenario_id,
                    'origin': od_row['origin'],
                    'dest': od_row['dest'],
                    'step_order': i + 1,
                    'from_facility': from_fac,
                    'to_facility': to_fac,
                    'from_lat': from_lat,
                    'from_lon': from_lon,
                    'to_lat': to_lat,
                    'to_lon': to_lon,
                    'distance_miles': 0.0,
                    'drive_hours': 0.0,
                    'processing_hours_at_destination': hours_per_touch
                })
                continue

            if from_fac in fac_lookup.index and to_fac in fac_lookup.index:
                lat1, lon1 = fac_lookup.at[from_fac, 'lat'], fac_lookup.at[from_fac, 'lon']
                lat2, lon2 = fac_lookup.at[to_fac, 'lat'], fac_lookup.at[to_fac, 'lon']

                raw_dist = haversine_miles(lat1, lon1, lat2, lon2)
                fixed, var, circuit, mph = band_lookup(raw_dist, mileage_bands)
                actual_dist = raw_dist * circuit
                drive_hours = actual_dist / mph

                path_steps.append({
                    'scenario_id': scenario_id,
                    'origin': od_row['origin'],
                    'dest': od_row['dest'],
                    'step_order': i + 1,
                    'from_facility': from_fac,
                    'to_facility': to_fac,
                    'from_lat': lat1,
                    'from_lon': lon1,
                    'to_lat': lat2,
                    'to_lon': lon2,
                    'distance_miles': actual_dist,
                    'drive_hours': drive_hours,
                    'processing_hours_at_destination': hours_per_touch
                })
            else:
                print(f"    Warning: Missing coordinates for {from_fac} or {to_fac}")

    return pd.DataFrame(path_steps)


def build_sort_summary(
        selected_paths: pd.DataFrame,
        sort_decisions: Dict,
        facilities: pd.DataFrame
) -> pd.DataFrame:
    """Build sort decision summary with regional hub info."""
    summary_data = []

    fac_lookup = facilities.set_index('facility_name')

    for _, od_row in selected_paths.iterrows():
        group_key = (od_row['scenario_id'], od_row['origin'],
                     od_row['dest'], od_row['day_type'])
        chosen_sort_level = sort_decisions.get(group_key, 'market')

        origin_region_hub = ''
        dest_region_hub = ''

        if od_row['origin'] in fac_lookup.index:
            origin_region_hub = fac_lookup.at[od_row['origin'], 'regional_sort_hub']
            if pd.isna(origin_region_hub):
                origin_region_hub = od_row['origin']

        if od_row['dest'] in fac_lookup.index:
            dest_region_hub = fac_lookup.at[od_row['dest'], 'regional_sort_hub']
            if pd.isna(dest_region_hub):
                dest_region_hub = od_row['dest']

        summary_data.append({
            'origin': od_row['origin'],
            'origin_region_hub': origin_region_hub,
            'dest': od_row['dest'],
            'dest_region_hub': dest_region_hub,
            'pkgs_day': od_row['pkgs_day'],
            'chosen_sort_level': chosen_sort_level,
            'total_cost': od_row['total_cost'],
            'processing_cost': od_row['processing_cost']
        })

    return pd.DataFrame(summary_data)


def validate_network_aggregations(
        od_selected: pd.DataFrame,
        arc_summary: pd.DataFrame,
        facility_rollup: pd.DataFrame
) -> Dict:
    """Validate aggregate calculations with sort/crossdock consistency checks."""
    validation_results = {}

    try:
        total_od_pkgs = od_selected['pkgs_day'].sum() if not od_selected.empty else 0
        total_facility_mm_injection = facility_rollup[
            'middle_mile_injection_pkgs_day'].sum() if not facility_rollup.empty else 0

        validation_results['total_od_packages'] = total_od_pkgs
        validation_results['total_facility_mm_injection'] = total_facility_mm_injection
        validation_results['package_consistency'] = abs(total_od_pkgs - total_facility_mm_injection) < 0.01

        if not facility_rollup.empty:
            for _, row in facility_rollup.iterrows():
                sort_pkgs = row.get('middle_mile_intermediate_pkgs_sort', 0)
                crossdock_pkgs = row.get('middle_mile_intermediate_pkgs_crossdock', 0)
                total_intermediate = row.get('middle_mile_intermediate_pkgs_day', 0)

                if abs((sort_pkgs + crossdock_pkgs) - total_intermediate) > 0.01:
                    facility_name = row.get('facility', 'unknown')
                    print(f"  ⚠️  Intermediate package mismatch at {facility_name}: "
                          f"sort={sort_pkgs:.0f} + crossdock={crossdock_pkgs:.0f} != total={total_intermediate:.0f}")
                    validation_results['intermediate_consistency'] = False
                else:
                    validation_results['intermediate_consistency'] = True

        total_od_cost = od_selected['total_cost'].sum() if 'total_cost' in od_selected.columns else 0
        total_arc_cost = arc_summary[
            'total_cost'].sum() if not arc_summary.empty and 'total_cost' in arc_summary.columns else 0

        validation_results['total_od_cost'] = total_od_cost
        validation_results['total_arc_cost'] = total_arc_cost

        if not arc_summary.empty and 'truck_fill_rate' in arc_summary.columns:
            non_od_arcs = arc_summary[arc_summary['from_facility'] != arc_summary['to_facility']]

            if not non_od_arcs.empty:
                total_pkg_cube = non_od_arcs['pkg_cube_cuft'].sum() if 'pkg_cube_cuft' in non_od_arcs.columns else 0
                total_truck_cube = (non_od_arcs['trucks'] * non_od_arcs.get('cube_per_truck',
                                                                            0)).sum() if 'trucks' in non_od_arcs.columns else 1

                if total_truck_cube > 0:
                    validation_results['network_avg_truck_fill'] = total_pkg_cube / total_truck_cube
                else:
                    validation_results['network_avg_truck_fill'] = 0
            else:
                validation_results['network_avg_truck_fill'] = 0
        else:
            validation_results['network_avg_truck_fill'] = 0

    except Exception as e:
        validation_results['validation_error'] = str(e)

    return validation_results