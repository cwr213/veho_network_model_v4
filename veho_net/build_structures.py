# veho_net/build_structures.py - WITH YOUR HUB RULES
import pandas as pd
import numpy as np
from .geo import haversine_miles


def build_od_and_direct(
        facilities: pd.DataFrame,
        zips: pd.DataFrame,
        year_demand: pd.DataFrame,
        injection_distribution: pd.DataFrame,
):
    """
    Enhanced OD generation ensuring only hub/hybrid facilities can originate middle-mile flows.
    Launch facilities only receive direct injection, never send outbound flows.
    """
    # Schema validation
    req_fac = {"facility_name", "type", "lat", "lon", "parent_hub_name", "is_injection_node"}
    if not req_fac.issubset(facilities.columns):
        missing = req_fac - set(facilities.columns)
        raise ValueError(f"facilities missing required columns: {sorted(missing)}")

    req_zips = {"zip", "facility_name_assigned", "population"}
    if not req_zips.issubset(zips.columns):
        missing = req_zips - set(zips.columns)
        raise ValueError(f"zips sheet missing required columns: {sorted(missing)}")

    req_dem = {
        "year", "annual_pkgs", "offpeak_pct_of_annual", "peak_pct_of_annual",
        "middle_mile_share_offpeak", "middle_mile_share_peak",
    }
    if not req_dem.issubset(year_demand.columns):
        missing = req_dem - set(year_demand.columns)
        raise ValueError(f"demand sheet missing required columns: {sorted(missing)}")

    req_inj = {"facility_name", "absolute_share"}
    if not req_inj.issubset(injection_distribution.columns):
        missing = req_inj - set(injection_distribution.columns)
        raise ValueError(f"injection_distribution missing required columns: {sorted(missing)}")

    # Data preparation
    fac = facilities.drop_duplicates(subset=["facility_name"]).reset_index(drop=True)
    z = zips[["zip", "facility_name_assigned", "population"]].drop_duplicates(subset=["zip"]).copy()

    # Destination population shares
    pop_by_dest = z.groupby("facility_name_assigned", as_index=False)["population"].sum()
    pop_by_dest["dest_pop_share"] = pop_by_dest["population"] / pop_by_dest["population"].sum()

    # Demand parameters
    yd = year_demand.copy()
    year_val = int(yd["year"].iloc[0])
    annual_total = float(yd["annual_pkgs"].sum())
    off_pct = float(yd["offpeak_pct_of_annual"].iloc[0])
    peak_pct = float(yd["peak_pct_of_annual"].iloc[0])
    mm_off = float(yd["middle_mile_share_offpeak"].iloc[0])
    mm_peak = float(yd["middle_mile_share_peak"].iloc[0])

    # Direct injection (all facilities can receive direct injection)
    direct = pop_by_dest.rename(columns={"facility_name_assigned": "dest"}).copy()
    direct["year"] = year_val
    direct["dir_pkgs_offpeak_day"] = annual_total * off_pct * (1.0 - mm_off) * direct["dest_pop_share"]
    direct["dir_pkgs_peak_day"] = annual_total * peak_pct * (1.0 - mm_peak) * direct["dest_pop_share"]

    # Middle-mile injection: ONLY hub/hybrid facilities can be origins
    inj = injection_distribution[["facility_name", "absolute_share"]].copy()

    # Join facilities data and filter
    inj = inj.merge(
        fac[["facility_name", "is_injection_node", "type", "parent_hub_name"]],
        on="facility_name",
        how="left",
        validate="many_to_one",
    )

    if inj["is_injection_node"].isna().any():
        missing = inj.loc[inj["is_injection_node"].isna(), "facility_name"].unique().tolist()
        raise ValueError(
            f"injection_distribution.facility_name not found in facilities: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    # Critical constraint: Only hub/hybrid facilities can originate middle-mile flows
    valid_origin_types = ['hub', 'hybrid']
    inj = inj[
        (inj["is_injection_node"].astype(int) == 1) &
        (inj["type"].isin(valid_origin_types))
        ].copy()

    if inj.empty:
        raise ValueError("No valid injection origins found. Ensure hub/hybrid facilities have is_injection_node=1")

    print(f"Valid MM injection origins: {len(inj)} hub/hybrid facilities")

    # Normalize injection shares
    inj["abs_w"] = pd.to_numeric(inj["absolute_share"], errors="coerce").fillna(0.0)
    total_w = float(inj["abs_w"].sum())
    if total_w <= 0:
        raise ValueError("injection_distribution.absolute_share must sum > 0 across enabled hub/hybrid injection nodes")

    inj["inj_share"] = inj["abs_w"] / total_w
    inj = inj[["facility_name", "inj_share"]].rename(columns={"facility_name": "origin"})

    # Build middle-mile OD matrix
    dest2 = pop_by_dest.rename(columns={"facility_name_assigned": "dest"})[["dest", "dest_pop_share"]]
    grid = inj.assign(_k=1).merge(dest2.assign(_k=1), on="_k").drop(columns="_k")

    od = grid.copy()
    od["pkgs_offpeak_day"] = annual_total * off_pct * mm_off * od["inj_share"] * od["dest_pop_share"]
    od["pkgs_peak_day"] = annual_total * peak_pct * mm_peak * od["inj_share"] * od["dest_pop_share"]

    # Remove O==D pairs (handled by direct injection)
    od = od[od["origin"] != od["dest"]].reset_index(drop=True)

    print(f"Generated {len(od)} middle-mile OD pairs")

    return od, direct, pop_by_dest


def candidate_paths(
        od: pd.DataFrame,
        facilities: pd.DataFrame,
        mileage_bands: pd.DataFrame,
        around_factor: float = 1.5,
) -> pd.DataFrame:
    """
    Path generation with your specific hub rules:

    RULE 1: Secondary hubs (parent_hub != facility_name) must route ALL OUTBOUND
            volume through their parent hub. Inbound volume is flexible.

    RULE 2: Launch facilities must have their parent hub as the second-to-last touch.

    This creates realistic routing while maintaining operational control.
    """
    enforce_thresh = facilities.attrs.get("enforce_parent_hub_over_miles", 500)

    fac = facilities.set_index("facility_name").copy()
    fac["type"] = fac["type"].astype(str).str.lower()

    # Only hub/hybrid facilities can be intermediate stops
    valid_intermediate_types = ["hub", "hybrid"]
    hubs_enabled = fac.index[fac["type"].isin(valid_intermediate_types)]

    print(f"Available intermediate facilities: {len(hubs_enabled)} hub/hybrid facilities")

    def raw_distance(o, d):
        return haversine_miles(
            float(fac.at[o, "lat"]), float(fac.at[o, "lon"]),
            float(fac.at[d, "lat"]), float(fac.at[d, "lon"])
        )

    # Build parent hub mapping and classify facilities
    parent_map = {}
    primary_hubs = set()
    secondary_hubs = set()
    launch_facilities = set()

    for facility_name in fac.index:
        facility_type = fac.at[facility_name, "type"]
        parent_hub = fac.at[facility_name, "parent_hub_name"]

        if pd.isna(parent_hub) or parent_hub == "":
            parent_hub = facility_name
        parent_map[facility_name] = parent_hub

        if facility_type in ["hub", "hybrid"]:
            if parent_hub == facility_name:
                primary_hubs.add(facility_name)
            else:
                secondary_hubs.add(facility_name)
        elif facility_type == "launch":
            launch_facilities.add(facility_name)

    print(
        f"Facility classification: {len(primary_hubs)} primary hubs, {len(secondary_hubs)} secondary hubs, {len(launch_facilities)} launch facilities")

    def is_secondary_hub(facility):
        """Check if facility is a secondary hub (must route outbound through parent)"""
        return facility in secondary_hubs

    def get_launch_parent(launch_facility):
        """Get the required parent hub for a launch facility"""
        return parent_map.get(launch_facility, launch_facility)

    def build_path_with_constraints(origin, dest, od_volume):
        """
        Build valid paths respecting your two rules:
        1. Secondary hub outbound constraint
        2. Launch facility parent as second-to-last
        """
        paths = []

        # Determine origin constraint (RULE 1)
        if is_secondary_hub(origin):
            # Secondary hub: MUST route outbound through parent
            origin_parent = parent_map[origin]
            if origin_parent == origin:
                # Shouldn't happen, but handle gracefully
                required_start = [origin]
            else:
                required_start = [origin, origin_parent]
        else:
            # Primary hub or launch: flexible outbound (but launch can't be origin anyway)
            required_start = [origin]

        # Determine destination constraint (RULE 2)
        if dest in launch_facilities:
            # Launch facility: parent hub must be second-to-last
            dest_parent = get_launch_parent(dest)
            if dest_parent == dest:
                # Launch with no parent or self-parent
                required_end = [dest]
            else:
                required_end = [dest_parent, dest]
        else:
            # Hub/hybrid destination: flexible inbound
            required_end = [dest]

        # Build paths connecting required start and end segments
        start_node = required_start[-1]  # Last node of required start
        end_node = required_end[0]  # First node of required end

        if start_node == end_node:
            # Direct connection possible
            paths.append(required_start + required_end[1:])
        else:
            # Need intermediate connection(s)
            direct_distance = raw_distance(start_node, end_node)

            # Option 1: Direct intermediate connection
            paths.append(required_start + required_end)

            # Option 2: One intermediate hub (for consolidation)
            if od_volume < 1500 and len(hubs_enabled) > 0:  # Only for smaller volumes
                candidate_intermediates = [h for h in hubs_enabled
                                           if h not in required_start and h not in required_end]

                if candidate_intermediates:
                    # Choose best intermediate hub
                    best_intermediate = min(candidate_intermediates,
                                            key=lambda h: raw_distance(start_node, h) + raw_distance(h, end_node))

                    # Check if detour is reasonable
                    intermediate_distance = (raw_distance(start_node, best_intermediate) +
                                             raw_distance(best_intermediate, end_node))

                    if intermediate_distance <= around_factor * direct_distance:
                        intermediate_path = required_start + [best_intermediate] + required_end
                        # Avoid duplicates
                        if intermediate_path not in paths:
                            paths.append(intermediate_path)

        return paths

    # Generate all candidate paths
    rows = []
    path_counts = {"direct": 0, "1_touch": 0, "2_touch": 0, "3_touch": 0, "4_touch": 0}

    for _, r in od.iterrows():
        o = r["origin"]
        d = r["dest"]
        od_volume = r.get("pkgs_peak_day", r.get("pkgs_offpeak_day", 0))

        # Generate paths with constraints
        candidate_paths_list = build_path_with_constraints(o, d, od_volume)

        for path_nodes in candidate_paths_list:
            # Determine path type based on number of facilities
            num_facilities = len(path_nodes)
            if num_facilities == 2:
                path_type = "direct"
            elif num_facilities == 3:
                path_type = "1_touch"
            elif num_facilities == 4:
                path_type = "2_touch"
            elif num_facilities == 5:
                path_type = "3_touch"
            else:
                path_type = "4_touch"

            path_counts[path_type] += 1

            rows.append({
                "origin": o,
                "dest": d,
                "path_type": path_type,
                "path_nodes": path_nodes,
                "path_str": "->".join(path_nodes),
                "od_volume": od_volume,
                "enforced_secondary_outbound": is_secondary_hub(o),
                "enforced_launch_parent": d in launch_facilities,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Final validation: ensure no launch facilities as intermediate stops
    def validate_path_nodes(path_nodes):
        # Check intermediate nodes (skip first and last)
        for i, node in enumerate(path_nodes[1:-1], 1):
            if node in fac.index and fac.at[node, "type"] == "launch":
                return False
        return True

    initial_count = len(df)
    df = df[df["path_nodes"].apply(validate_path_nodes)].reset_index(drop=True)
    if len(df) < initial_count:
        print(f"Filtered out {initial_count - len(df)} paths with launch facilities as intermediate stops")

    # Remove exact duplicates
    df["path_nodes_tuple"] = df["path_nodes"].apply(tuple)
    df = df.drop_duplicates(subset=["origin", "dest", "path_nodes_tuple"]).drop(
        columns=["path_nodes_tuple"]).reset_index(drop=True)

    print(f"Generated {len(df)} valid candidate paths:")
    for path_type, count in path_counts.items():
        if count > 0:
            print(f"  {path_type}: {count}")

    # Print rule enforcement summary
    secondary_outbound_enforced = (df["enforced_secondary_outbound"] == True).sum()
    launch_parent_enforced = (df["enforced_launch_parent"] == True).sum()

    print(f"Rule enforcement:")
    print(f"  Secondary hub outbound routing: {secondary_outbound_enforced} paths")
    print(f"  Launch facility parent hub: {launch_parent_enforced} paths")

    return df.drop(columns=["od_volume", "enforced_secondary_outbound", "enforced_launch_parent"])