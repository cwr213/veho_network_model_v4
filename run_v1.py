def _allocate_lane_costs_to_ods(od_selected: pd.DataFrame, arc_summary: pd.DataFrame, costs: dict,
                                strategy: str) -> pd.DataFrame:
    """
    COMPREHENSIVE FIX: Proper cost allocation including packages_dwelled from lanes to ODs.
    """
    od = od_selected.copy()

    # Calculate touch costs per OD based on containerization level
    touch_map = {"direct": 0, "1_touch": 1, "2_touch": 2, "3_touch": 3, "4_touch": 4}
    od['num_touches'] = od['path_type'].map(touch_map).fillna(0)

    crossdock_pp = float(costs.get("crossdock_touch_cost_per_pkg", 0.0))
    sort_pp = float(costs.get("sort_cost_per_pkg", 0.0))
    lm_sort_pp = float(costs.get("last_mile_sort_cost_per_pkg", 0.0))

    if strategy.lower() == "container":
        od['touch_cost'] = od['num_touches'] * crossdock_pp * od['pkgs_day']
        od['touch_cpp'] = od['num_touches'] * crossdock_pp

        if 'containerization_level' in od.columns:
            lm_sort_multiplier = od['containerization_level'].map({
                'region': 1.0,
                'market': 0.5,
                'sort_group': 0.1
            }).fillna(1.0)

            od['lm_sort_cost'] = lm_sort_multiplier * lm_sort_pp * od['pkgs_day']
            od['touch_cost'] += od['lm_sort_cost']
            od['touch_cpp'] += lm_sort_multiplier * lm_sort_pp
    else:
        od['touch_cost'] = (od['num_touches'] + 1) * sort_pp * od['pkgs_day']
        od['touch_cpp'] = (od['num_touches'] + 1) * sort_pp

    # Initialize cost and dwelled package columns
    od['linehaul_cost'] = 0.0
    od['linehaul_cpp'] = 0.0
    od['packages_dwelled'] = 0.0  # CRITICAL: Initialize this column

    if arc_summary is None or arc_summary.empty:
        print("Warning: No arc_summary provided for cost allocation")
        od['total_cost'] = od['touch_cost']
        od['cost_per_pkg'] = od['touch_cpp']
        return od

    print(f"Allocating costs from {len(arc_summary)} lanes to {len(od)} OD pairs...")

    # CRITICAL FIX: For each OD, find its path legs and allocate costs AND packages_dwelled
    allocation_debug = []

    for idx, row in od.iterrows():
        path_str = str(row.get('path_str', ''))
        if not path_str or '->' not in path_str:
            continue

        nodes = path_str.split('->')
        od_pkgs = float(row['pkgs_day'])
        od_linehaul_cost = 0.0
        od_packages_dwelled = 0.0

        # Sum costs and dwelled packages across all legs in this path
        for i in range(len(nodes) - 1):
            from_fac = nodes[i].strip()
            to_fac = nodes[i + 1].strip()

            # Find this lane in arc_summary
            lane = arc_summary[
                (arc_summary['from_facility'] == from_fac) &
                (arc_summary['to_facility'] == to_fac)
                ]

            if not lane.empty:
                lane_row = lane.iloc[0]
                lane_total_cost = float(lane_row.get('total_cost', 0))
                lane_total_pkgs = float(lane_row.get('pkgs_day', 1))
                lane_packages_dwelled = float(lane_row.get('packages_dwelled', 0))

                if lane_total_pkgs > 0:
                    od_share = od_pkgs / lane_total_pkgs
                    allocated_cost = lane_total_cost * od_share
                    allocated_dwelled = lane_packages_dwelled * od_share

                    od_linehaul_cost += allocated_cost
                    od_packages_dwelled += allocated_dwelled

                    # Debug tracking
                    allocation_debug.append({
                        'od_pair': f"{row['origin']}->{row['dest']}",
                        'lane': f"{from_fac}->{to_fac}",
                        'od_pkgs': od_pkgs,
                        'lane_pkgs': lane_total_pkgs,
                        'od_share': od_share,
                        'lane_dwelled': lane_packages_dwelled,
                        'allocated_dwelled': allocated_dwelled
                    })

        od.at[idx, 'linehaul_cost'] = od_linehaul_cost
        od.at[idx, 'linehaul_cpp'] = od_linehaul_cost / od_pkgs if od_pkgs > 0 else 0
        od.at[idx, 'packages_dwelled'] = od_packages_dwelled  # CRITICAL: Properly allocated

    # Calculate totals
    od['total_cost'] = od['linehaul_cost'] + od['touch_cost']
    od['cost_per_pkg'] = od['total_cost'] / od['pkgs_day'].replace(0, 1)

    # Debug output
    total_lane_dwelled = arc_summary['packages_dwelled'].sum()
    total_od_dwelled = od['packages_dwelled'].sum()

    print(f"COST ALLOCATION SUMMARY:")
    print(f"  Total linehaul cost: ${od['linehaul_cost'].sum():,.2f}")
    print(f"  Total touch cost: ${od['touch_cost'].sum():,.2f}")
    print(f"  CRITICAL - Dwelled packages:")
    print(f"    Lane level total: {total_lane_dwelled:,.0f}")
    print(f"    OD level total: {total_od_dwelled:,.0f}")
    print(f"    Allocation efficiency: {(total_od_dwelled / max(total_lane_dwelled, 1)) * 100:.1f}%")

    # Show specific examples if dwelled packages > 0
    if total_od_dwelled > 0:
        dwelled_ods = od[od['packages_dwelled'] > 0]
        if not dwelled_ods.empty:
            print(f"  Sample dwelled ODs:")
            for i, (_, sample_od) in enumerate(dwelled_ods.head(3).iterrows()):
                print(f"    {sample_od['origin']}->{sample_od['dest']}: {sample_od['packages_dwelled']:.1f} dwelled")
    else:
        print("  WARNING: No dwelled packages allocated to ODs - this indicates an allocation problem")
        # Show debug info for first few allocations
        if allocation_debug:
            print("  Allocation debug (first 5):")
            for debug in allocation_debug[:5]:
                print(
                    f"    {debug['od_pair']}: lane_dwelled={debug['lane_dwelled']:.1f}, allocated={debug['allocated_dwelled']:.1f}")

    return od