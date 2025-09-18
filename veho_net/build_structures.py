import pandas as pd
import numpy as np
from .geo import haversine_miles

def _banded_miles(m_raw: float, bands: pd.DataFrame) -> float:
    """Apply circuity factor from the band that covers raw haversine miles."""
    if pd.isna(m_raw):
        return np.inf
    # find band row
    r = bands[(bands["mileage_band_min"] <= m_raw) & (m_raw <= bands["mileage_band_max"])]
    if r.empty:
        r = bands.iloc[[-1]]
    circuit = float(r.iloc[0]["circuity_factor"])
    return float(m_raw) * circuit

def _facility_population_shares(zips: pd.DataFrame) -> pd.DataFrame:
    pop = (
        zips.groupby("facility_name_assigned", as_index=False)["population"]
        .sum()
        .rename(columns={"facility_name_assigned": "dest", "population": "facility_population"})
    )
    total_pop = float(pop["facility_population"].sum())
    pop["dest_pop_share"] = 0.0 if total_pop <= 0 else pop["facility_population"] / total_pop
    return pop

def _enabled_injection_nodes(injection_distribution: pd.DataFrame, facilities: pd.DataFrame) -> pd.DataFrame:
    enabled = facilities.query("is_injection_node == 1")[["facility_name"]].copy()
    inj = injection_distribution.merge(enabled, on="facility_name", how="inner").copy()
    if inj.empty:
        inj["norm_share"] = 0.0
        return inj
    inj["norm_share"] = inj["absolute_share"] / inj["absolute_share"].sum()
    return inj

def _dest_facilities(facilities: pd.DataFrame) -> pd.DataFrame:
    return (
        facilities[facilities["type"].isin(["launch", "hybrid"])][["facility_name"]]
        .rename(columns={"facility_name": "dest"})
        .copy()
    )

def _origins_enabled(facilities: pd.DataFrame) -> pd.DataFrame:
    return facilities.query("is_injection_node == 1")[["facility_name"]].rename(columns={"facility_name": "origin"}).copy()

# --------- OD + Direct (strict shares already implemented earlier) ---------

def build_od_and_direct(
    facilities: pd.DataFrame,
    zips: pd.DataFrame,
    demand_year_df: pd.DataFrame,
    injection_distribution: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    req = ["annual_pkgs", "offpeak_pct_of_annual", "peak_pct_of_annual",
           "middle_mile_share_offpeak", "middle_mile_share_peak"]
    missing = [c for c in req if c not in demand_year_df.columns]
    if missing:
        raise ValueError(
            f"build_od_and_direct: demand_year_df missing columns {missing}. "
            f"Columns present: {list(demand_year_df.columns)}"
        )
    for col in ["offpeak_pct_of_annual", "peak_pct_of_annual",
                "middle_mile_share_offpeak", "middle_mile_share_peak"]:
        if demand_year_df[col].isna().any():
            bad_idx = list(demand_year_df[demand_year_df[col].isna()].index[:5])
            raise ValueError(f"demand_year_df has NaN in '{col}' for selected year. Example row indices: {bad_idx}")
        if not demand_year_df[col].between(0, 1, inclusive="both").all():
            bad = demand_year_df[~demand_year_df[col].between(0, 1, inclusive="both")][[col]].head(5)
            raise ValueError(f"demand_year_df '{col}' values must be within [0,1]. Offenders (first 5 rows):\n{bad}")

    dest_pop = _facility_population_shares(zips)
    origins = _origins_enabled(facilities)
    dests = _dest_facilities(facilities)
    inj = _enabled_injection_nodes(injection_distribution, facilities)[["facility_name", "norm_share"]]
    inj = inj.rename(columns={"facility_name": "origin"})

    # cross join O x D x demand rows (single year upstream)
    od_nat = origins.assign(key=1).merge(dests.assign(key=1), on="key", how="outer").drop(columns="key")
    od_nat = od_nat.assign(key=1).merge(demand_year_df.assign(key=1), on="key", how="left").drop(columns="key")

    # Attach shares
    od_nat = (
        od_nat.merge(inj, on="origin", how="left")
              .merge(dest_pop[["dest", "dest_pop_share"]], on="dest", how="left")
    ).fillna({"norm_share": 0.0, "dest_pop_share": 0.0})

    # Day-type totals
    od_nat["offpeak_day_total"] = od_nat["annual_pkgs"] * od_nat["offpeak_pct_of_annual"]
    od_nat["peak_day_total"]    = od_nat["annual_pkgs"] * od_nat["peak_pct_of_annual"]

    # Middle-mile day demand per OD (allocate by injection norm_share * destination population share)
    od_nat["mm_offpeak_day"] = (
        od_nat["offpeak_day_total"] * od_nat["middle_mile_share_offpeak"] * od_nat["norm_share"] * od_nat["dest_pop_share"]
    )
    od_nat["mm_peak_day"] = (
        od_nat["peak_day_total"] * od_nat["middle_mile_share_peak"] * od_nat["norm_share"] * od_nat["dest_pop_share"]
    )

    cols = ["origin", "dest", "year", "annual_pkgs", "offpeak_pct_of_annual", "peak_pct_of_annual"]
    od = (
        od_nat.groupby(cols, as_index=False)[["mm_offpeak_day", "mm_peak_day"]].sum()
              .rename(columns={"mm_offpeak_day": "pkgs_offpeak_day", "mm_peak_day": "pkgs_peak_day"})
    )

    # Direct distribution day demand by destination (dest_pop)
    dir_fac = (
        dests.assign(key=1).merge(demand_year_df.assign(key=1), on="key", how="left").drop(columns="key")
             .merge(dest_pop[["dest", "dest_pop_share"]], on="dest", how="left")
    )
    dir_fac["dir_pkgs_offpeak_day"] = (
        dir_fac["annual_pkgs"] * dir_fac["offpeak_pct_of_annual"] * (1.0 - dir_fac["middle_mile_share_offpeak"]) * dir_fac["dest_pop_share"]
    )
    dir_fac["dir_pkgs_peak_day"] = (
        dir_fac["annual_pkgs"] * dir_fac["peak_pct_of_annual"] * (1.0 - dir_fac["middle_mile_share_peak"]) * dir_fac["dest_pop_share"]
    )

    return od, dir_fac[["dest","year","dir_pkgs_offpeak_day","dir_pkgs_peak_day"]], dest_pop

# --------- Candidate paths (shortest direct / shortest 1-touch / shortest 2-touch) ---------

def candidate_paths(od: pd.DataFrame, facilities: pd.DataFrame, bands: pd.DataFrame,
                    k1: int = 10, k2: int = 5, k3: int = 5, around_factor: float = 2.0) -> pd.DataFrame:
    """
    For each OD, return exactly three candidates:
      - direct: [O, D]
      - shortest_1: [O, H, D] minimizing banded miles across a shortlist of hubs
      - shortest_2: [O, H1, H2, D] minimizing banded miles across pairs from that shortlist

    Shortlist pool = union of:
      top-k1 by (O->H + H->D) banded miles,
      top-k2 by O->H banded miles,
      top-k3 by H->D banded miles,
      plus parent(D) if available.
    """
    fac = facilities.set_index("facility_name")
    hubs = facilities[facilities["type"].isin(["hub", "hybrid"])]["facility_name"].tolist()
    parent = fac["parent_hub_name"].to_dict()

    def raw_m(o, d):
        return haversine_miles(float(fac.at[o, "lat"]), float(fac.at[o, "lon"]),
                               float(fac.at[d, "lat"]), float(fac.at[d, "lon"]))

    rows = []
    for _, r in od.iterrows():
        o, d = r["origin"], r["dest"]
        if o not in fac.index or d not in fac.index:
            continue

        # Direct
        m_dir_raw = raw_m(o, d)
        m_dir = _banded_miles(m_dir_raw, bands)
        rows.append({"origin": o, "dest": d, "path_type": "direct", "path_nodes": [o, d], "path_str": f"{o} → {d}"})

        # Build hub pool
        # score all hubs by components
        scores = []
        for h in hubs:
            if h in (o, d):
                continue
            m_oh = _banded_miles(raw_m(o, h), bands)
            m_hd = _banded_miles(raw_m(h, d), bands)
            scores.append((h, m_oh + m_hd, m_oh, m_hd))
        if not scores:
            continue
        df = pd.DataFrame(scores, columns=["hub", "score_od", "score_o", "score_d"]).sort_values("score_od")

        pool = set(df.head(k1)["hub"])
        pool |= set(df.sort_values("score_o").head(k2)["hub"])
        pool |= set(df.sort_values("score_d").head(k3)["hub"])
        ph = parent.get(d, None)
        if ph and ph not in (o, d):
            pool.add(ph)

        pool = [h for h in pool if h not in (o, d)]

        # Shortest 1-touch over pool
        best1 = None
        best1_m = np.inf
        for h in pool:
            m = _banded_miles(raw_m(o, h), bands) + _banded_miles(raw_m(h, d), bands)
            if m < best1_m:
                best1_m = m
                best1 = [o, h, d]

        # Shortest 2-touch over pairs from pool
        best2 = None
        best2_m = np.inf
        pool_list = list(pool)
        for i in range(len(pool_list)):
            h1 = pool_list[i]
            for j in range(i+1, len(pool_list)):
                h2 = pool_list[j]
                if len({o, d, h1, h2}) < 4:
                    continue
                m = (_banded_miles(raw_m(o, h1), bands)
                     + _banded_miles(raw_m(h1, h2), bands)
                     + _banded_miles(raw_m(h2, d), bands))
                if m < best2_m:
                    best2_m = m
                    best2 = [o, h1, h2, d]

        # Around-the-world prune
        if best1 is not None and best1_m <= around_factor * m_dir:
            rows.append({"origin": o, "dest": d, "path_type": "1_touch",
                         "path_nodes": best1, "path_str": " → ".join(best1)})
        if best2 is not None and best2_m <= around_factor * m_dir:
            rows.append({"origin": o, "dest": d, "path_type": "2_touch",
                         "path_nodes": best2, "path_str": " → ".join(best2)})

    df_out = pd.DataFrame(rows)
    return df_out.reset_index(drop=True)
