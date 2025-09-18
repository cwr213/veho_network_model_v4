# veho_net/build_structures.py
import pandas as pd
import numpy as np
from .geo import haversine_miles

# Canonical column names this module expects (strict):
# facilities: ['facility_name','type','lat','lon','parent_hub_name','is_injection_node', ...]
# zips:       ['zip','facility_name_assigned','population']
# demand:     ['year','annual_pkgs','offpeak_pct_of_annual','peak_pct_of_annual',
#              'middle_mile_share_offpeak','middle_mile_share_peak']  (one row for the chosen year)
# injection_distribution: ['facility_name','absolute_share']  (weights); optionally you may include others
#   NOTE: enablement is taken from facilities.is_injection_node (1/0). We normalize shares over enabled nodes only.


def build_od_and_direct(
    facilities: pd.DataFrame,
    zips: pd.DataFrame,
    year_demand: pd.DataFrame,
    injection_distribution: pd.DataFrame,
):
    """
    Returns:
      od: O!=D pairs with daily MM volumes by day type
          ['origin','dest','inj_share','dest_pop_share','pkgs_offpeak_day','pkgs_peak_day']
      direct_fac: direct-injection (O==D) daily volumes by day type
          ['dest','year','dest_pop_share','dir_pkgs_offpeak_day','dir_pkgs_peak_day']
      dest_pop: destination population totals/shares
          ['facility_name_assigned','population','dest_pop_share']
    """
    # ----- Strict schema checks
    req_fac = {"facility_name", "type", "lat", "lon", "parent_hub_name", "is_injection_node"}
    if not req_fac.issubset(facilities.columns):
        missing = req_fac - set(facilities.columns)
        raise ValueError(f"facilities missing required columns: {sorted(missing)}")

    req_zips = {"zip", "facility_name_assigned", "population"}
    if not req_zips.issubset(zips.columns):
        missing = req_zips - set(zips.columns)
        raise ValueError(f"zips sheet missing required columns: {sorted(missing)}")

    req_dem = {
        "year",
        "annual_pkgs",
        "offpeak_pct_of_annual",
        "peak_pct_of_annual",
        "middle_mile_share_offpeak",
        "middle_mile_share_peak",
    }
    if not req_dem.issubset(year_demand.columns):
        missing = req_dem - set(year_demand.columns)
        raise ValueError(f"demand sheet missing required columns: {sorted(missing)}")

    req_inj = {"facility_name", "absolute_share"}
    if not req_inj.issubset(injection_distribution.columns):
        missing = req_inj - set(injection_distribution.columns)
        raise ValueError(f"injection_distribution missing required columns: {sorted(missing)}")

    # ----- Dedup and prep
    fac = facilities.drop_duplicates(subset=["facility_name"]).reset_index(drop=True)

    z = (
        zips[["zip", "facility_name_assigned", "population"]]
        .drop_duplicates(subset=["zip"])
        .copy()
    )

    # Destination population shares by assigned facility
    pop_by_dest = z.groupby("facility_name_assigned", as_index=False)["population"].sum()
    pop_by_dest["dest_pop_share"] = pop_by_dest["population"] / pop_by_dest["population"].sum()

    # Demand primitives
    yd = year_demand.copy()
    year_val = int(yd["year"].iloc[0])
    annual_total = float(yd["annual_pkgs"].sum())
    off_pct = float(yd["offpeak_pct_of_annual"].iloc[0])
    peak_pct = float(yd["peak_pct_of_annual"].iloc[0])
    mm_off = float(yd["middle_mile_share_offpeak"].iloc[0])
    mm_peak = float(yd["middle_mile_share_peak"].iloc[0])

    # Direct injection (O==D) by population share
    direct = pop_by_dest.rename(columns={"facility_name_assigned": "dest"}).copy()
    direct["year"] = year_val
    direct["dir_pkgs_offpeak_day"] = annual_total * off_pct * (1.0 - mm_off) * direct["dest_pop_share"]
    direct["dir_pkgs_peak_day"] = annual_total * peak_pct * (1.0 - mm_peak) * direct["dest_pop_share"]

    # Injection distribution:
    #   Use absolute_share; normalize over facilities where is_injection_node == 1.
    inj = injection_distribution[["facility_name", "absolute_share"]].copy()
    # Join to facilities to filter to enabled injection nodes (1/0)
    inj = inj.merge(
        fac[["facility_name", "is_injection_node"]],
        on="facility_name",
        how="left",
        validate="many_to_one",
    )
    if inj["is_injection_node"].isna().any():
        missing = inj.loc[inj["is_injection_node"].isna(), "facility_name"].unique().tolist()
        raise ValueError(
            f"injection_distribution.facility_name not found in facilities: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    inj = inj[inj["is_injection_node"].astype(int) == 1].copy()
    inj["abs_w"] = pd.to_numeric(inj["absolute_share"], errors="coerce").fillna(0.0)
    total_w = float(inj["abs_w"].sum())
    if total_w <= 0:
        raise ValueError("injection_distribution.absolute_share must sum > 0 across enabled injection nodes")

    inj["inj_share"] = inj["abs_w"] / total_w
    inj = inj[["facility_name", "inj_share"]].rename(columns={"facility_name": "origin"})

    # Build O!=D grid: origin inj share x destination pop share
    dest2 = pop_by_dest.rename(columns={"facility_name_assigned": "dest"})[["dest", "dest_pop_share"]]
    grid = inj.assign(_k=1).merge(dest2.assign(_k=1), on="_k").drop(columns="_k")

    od = grid.copy()
    od["pkgs_offpeak_day"] = annual_total * off_pct * mm_off * od["inj_share"] * od["dest_pop_share"]
    od["pkgs_peak_day"] = annual_total * peak_pct * mm_peak * od["inj_share"] * od["dest_pop_share"]

    # Remove O==D; those volumes are handled by direct injection table
    od = od[od["origin"] != od["dest"]].reset_index(drop=True)

    return od, direct, pop_by_dest


def candidate_paths(
    od: pd.DataFrame,
    facilities: pd.DataFrame,
    mileage_bands: pd.DataFrame,
    around_factor: float = 1.5,
) -> pd.DataFrame:
    """
    Policy-first candidates (with parent hub enforcement over a threshold):
      • Always include DIRECT (O->D).
      • If raw(O,D) <= enforce_parent_hub_over_miles (default 500):
          include ONE 1-touch: shorter of (nearest hub to O) vs (nearest hub to D).
      • If raw(O,D)  > enforce_parent_hub_over_miles:
          enforce parent hubs:
            – 1-touch via D's parent (parent_hub_name) if defined
            – 2-touch via O's parent then D's parent
      • Prune any candidate whose RAW path miles > around_factor * RAW direct miles.
    """
    enforce_thresh = facilities.attrs.get("enforce_parent_hub_over_miles", 500)

    fac = facilities.set_index("facility_name").copy()
    # normalize 'type' to lowercase
    fac["type"] = fac["type"].astype(str).str.lower()

    # hubs/hybrids are eligible as intermediate touches
    hubs_enabled = fac.index[fac["type"].isin(["hub", "hybrid"])]

    def raw(o, d):
        return haversine_miles(
            float(fac.at[o, "lat"]), float(fac.at[o, "lon"]),
            float(fac.at[d, "lat"]), float(fac.at[d, "lon"])
        )

    # Parent hub mapping (exact column name: parent_hub_name)
    if "parent_hub_name" not in fac.columns:
        raise ValueError("facilities missing 'parent_hub_name' column")

    parent_map = (
        fac.reset_index()[["facility_name", "parent_hub_name"]]
        .set_index("facility_name")["parent_hub_name"]
        .to_dict()
    )

    rows = []
    for _, r in od.iterrows():
        o = r["origin"]
        d = r["dest"]
        direct_raw = raw(o, d)

        # Always include direct
        rows.append(
            {"origin": o, "dest": d, "path_type": "direct", "path_nodes": [o, d], "path_str": f"{o}->{d}"}
        )

        if direct_raw <= enforce_thresh:
            # Local/regional: allow one 1-touch via nearest hub to O or nearest hub to D (pick shorter)
            if len(hubs_enabled) > 0:
                nearest_to_o = min(hubs_enabled, key=lambda h: raw(o, h))
                nearest_to_d = min(hubs_enabled, key=lambda h: raw(h, d))
                cand1 = [o, nearest_to_o, d]
                cand2 = [o, nearest_to_d, d]
                miles1 = raw(o, nearest_to_o) + raw(nearest_to_o, d)
                miles2 = raw(o, nearest_to_d) + raw(nearest_to_d, d)
                best = cand1 if miles1 <= miles2 else cand2
                rows.append(
                    {
                        "origin": o,
                        "dest": d,
                        "path_type": "1_touch",
                        "path_nodes": best,
                        "path_str": "->".join(best),
                    }
                )
        else:
            # Long-haul: enforce parent hubs
            o_parent = parent_map.get(o, o)
            d_parent = parent_map.get(d, d)

            # If parent hub is missing or not a known facility, default to self
            if (o_parent is None) or (o_parent not in fac.index):
                o_parent = o
            if (d_parent is None) or (d_parent not in fac.index):
                d_parent = d

            # 1-touch via D's parent if it is different from D
            if d_parent != d:
                rows.append(
                    {
                        "origin": o,
                        "dest": d,
                        "path_type": "1_touch",
                        "path_nodes": [o, d_parent, d],
                        "path_str": f"{o}->{d_parent}->{d}",
                    }
                )
            # 2-touch via O's parent then D's parent
            rows.append(
                {
                    "origin": o,
                    "dest": d,
                    "path_type": "2_touch",
                    "path_nodes": [o, o_parent, d_parent, d],
                    "path_str": f"{o}->{o_parent}->{d_parent}->{d}",
                }
            )

        # Prune by RAW detour factor at build time
        local = [rr for rr in rows if rr["origin"] == o and rr["dest"] == d]
        pruned = []
        for rr in local:
            nodes = rr["path_nodes"]
            path_raw = sum(raw(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1))
            if path_raw <= around_factor * direct_raw:
                pruned.append(rr)
        rows = [rr for rr in rows if not (rr["origin"] == o and rr["dest"] == d)] + pruned

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Deduplicate by exact node sequence
    df["path_nodes_tuple"] = df["path_nodes"].apply(tuple)
    df = (
        df.drop_duplicates(subset=["origin", "dest", "path_type", "path_nodes_tuple"])
        .drop(columns=["path_nodes_tuple"])
        .reset_index(drop=True)
    )
    return df
