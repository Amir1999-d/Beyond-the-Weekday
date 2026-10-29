"""
phase1_corridors.py — corridors of exclusion and service-floor analysis.

Implements thesis §5.2 on the regular urban network: a usable weekday baseline
plus at least one route not excluded by the configured night/interurban rules.
Ranks holiday social burden and reports the 30-minute descriptive service-floor
statistic for the full regular-network subset and its high-vulnerability tertile.
The floor is a deliberately permissive low-frequency threshold, not a regulatory
standard and not a complete test of vertical equity.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import config

config.validate()


# -----------------------------------------------------------------------------
# 1. load corrected stop-level state
# -----------------------------------------------------------------------------
def load_stop_state():
    print("[1] Loading corrected stop-hourly table and aggregating to stops...")
    df = pd.read_csv(os.path.join(config.OUTPUT_DIR, "milan_stop_hourly.csv"))
    df = df[config.operational_hour_filter(df["hour"])].copy()

    has_sun = "hw_sun" in df.columns
    agg = {"v_index": ("v_index", "mean"),
           "hw_wd":   ("hw_wd",   "median"),
           "hw_hol":  ("hw_hol",  "median")}
    if has_sun:
        agg["hw_sun"] = ("hw_sun", "median")
    stop = df.groupby("stop_id").agg(**agg).reset_index()
    stop["stop_id"] = stop["stop_id"].astype(str).str.strip()

    # Burden uses SERVED headways; no imputation here.
    stop["burden_wd"]  = stop["v_index"] * stop["hw_wd"]
    stop["burden_hol"] = stop["v_index"] * stop["hw_hol"]
    if has_sun:
        stop["burden_sun"] = stop["v_index"] * stop["hw_sun"]
    print(f"    Stops: {len(stop)}   Sunday column present: {has_sun}")
    return stop, has_sun


# -----------------------------------------------------------------------------
# 2. attach geometry + line names
# -----------------------------------------------------------------------------
def attach_metadata(stop):
    print("[2] Attaching stop geometry, names, and serving lines...")

    # geometry + stop_name from the builder's GeoPackage
    stops_gdf = gpd.read_file(os.path.join(config.OUTPUT_DIR, "milan_stops.gpkg"))
    stops_gdf["stop_id"] = stops_gdf["stop_id"].astype(str).str.strip()
    meta_cols = ["stop_id", "geometry"]
    for c in ("stop_name", "stop_lat", "stop_lon"):
        if c in stops_gdf.columns:
            meta_cols.append(c)
    stop = stop.merge(stops_gdf[meta_cols], on="stop_id", how="left")

    # line names from the primary GTFS feed
    feed = config.GTFS_FEEDS[config.PRIMARY_FEED]

    def read_gtfs(name, usecols):
        df = pd.read_csv(os.path.join(feed, name), usecols=usecols, dtype=str)
        return df.apply(lambda x: x.astype("string").str.strip()
                        if pd.api.types.is_string_dtype(x.dtype) or pd.api.types.is_object_dtype(x.dtype)
                        else x)

    trips = read_gtfs("trips.txt", ["trip_id", "route_id"]).drop_duplicates()
    st    = read_gtfs("stop_times.txt", ["trip_id", "stop_id"]).drop_duplicates()
    routes = read_gtfs("routes.txt", ["route_id", "route_short_name"]).drop_duplicates()

    st["stop_id"] = st["stop_id"].astype(str).str.strip()
    stop_lines = (st.merge(trips, on="trip_id")
                    .merge(routes, on="route_id")
                    .groupby("stop_id")["route_short_name"]
                    .apply(lambda x: ", ".join(sorted(set(x.dropna()))))
                    .reset_index()
                    .rename(columns={"route_short_name": "lines"}))
    stop = stop.merge(stop_lines, on="stop_id", how="left")
    return gpd.GeoDataFrame(stop, geometry="geometry", crs=config.CRS_GEOGRAPHIC)


# -----------------------------------------------------------------------------
# 2b. flag the regular urban network (scope filter)
# -----------------------------------------------------------------------------
def flag_regular_urban(stop_gdf):
    """Tag each stop as belonging to the regular urban network.

    A stop is regular-urban if its median weekday headway is at most
    REGULAR_NETWORK_HW_MAX_MIN (i.e. it has a usable weekday baseline) AND its
    route set contains at least one route that is not a known non-urban line
    (night lines, interurban coaches).
    """
    s = stop_gdf.copy()

    # criterion 1: has a usable weekday baseline
    has_baseline = s["hw_wd"].notna() & (s["hw_wd"] <= config.REGULAR_NETWORK_HW_MAX_MIN)

    # criterion 2: serves at least one non-excluded route
    def is_urban_route_list(lines_str):
        if pd.isna(lines_str) or not lines_str:
            return False
        routes = [r.strip() for r in str(lines_str).split(",") if r.strip()]
        for r in routes:
            if r in config.NON_URBAN_ROUTE_EXACT:
                continue
            if any(r.startswith(p) for p in config.NON_URBAN_ROUTE_PREFIXES):
                continue
            return True   # at least one urban-eligible route found
        return False

    has_urban_route = s["lines"].apply(is_urban_route_list)
    s["regular_urban"] = has_baseline & has_urban_route

    n_total = len(s); n_reg = int(s["regular_urban"].sum())
    print(f"    Weekday-baseline network: {n_reg}/{n_total} stops "
          f"({n_reg/n_total*100:.1f}%) "
          f"[hw_wd <= {config.REGULAR_NETWORK_HW_MAX_MIN} min AND non-excluded route]")
    print(f"    Excluded: night lines (prefixes {config.NON_URBAN_ROUTE_PREFIXES}), "
          f"interurban {config.NON_URBAN_ROUTE_EXACT}, or no weekday baseline.")

    # -- structural-deficit complement: stops EXCLUDED from the weekday-baseline
    # network because their weekday headway exceeds REGULAR_NETWORK_HW_MAX_MIN.
    # These are a separate analytical category — stops with persistent weekday
    # service below the spontaneous-use threshold — and are reported here for
    # transparency so the reader can see how many vulnerable stops fall in this
    # category (which is NOT a holiday-equity finding; it is a baseline-service
    # deficit, addressed in §4 of the thesis as a separate population).
    no_baseline = s["hw_wd"].notna() & (s["hw_wd"] > config.REGULAR_NETWORK_HW_MAX_MIN)
    n_struct_deficit = int(no_baseline.sum())
    if n_struct_deficit > 0:
        v_high = s["v_index"].quantile(config.HIGH_VULN_TERTILE)
        n_struct_high_vuln = int((no_baseline & (s["v_index"] >= v_high)).sum())
        print(f"    Structural weekday-service deficit "
              f"(hw_wd > {config.REGULAR_NETWORK_HW_MAX_MIN} min): "
              f"{n_struct_deficit} stops, of which {n_struct_high_vuln} are in "
              f"the high-vulnerability tertile.")
        print("    (Reported separately — these stops have a baseline-service")
        print("     problem distinct from the holiday-equity question and are")
        print("     not part of the corridors-of-exclusion ranking.)")
    return s


# -----------------------------------------------------------------------------
# 3. top-N corridors by holiday burden
# -----------------------------------------------------------------------------
def top_corridors(stop_gdf, n=10):
    print(f"[3] Ranking top-{n} corridors of exclusion (by holiday burden)...")
    top = stop_gdf.nlargest(n, "burden_hol").copy().reset_index(drop=True)
    top.index += 1
    top["hw_ratio_hol_wd"] = (top["hw_hol"] / top["hw_wd"]).round(2)
    out_cols = ["stop_id", "stop_name", "lines",
                "stop_lat", "stop_lon",
                "v_index", "hw_wd", "hw_hol", "hw_ratio_hol_wd",
                "burden_wd", "burden_hol"]
    out_cols = [c for c in out_cols if c in top.columns]
    table = top[out_cols].copy()
    for c in ("v_index", "burden_wd", "burden_hol"):
        if c in table: table[c] = table[c].round(3)
    for c in ("hw_wd", "hw_hol"):
        if c in table: table[c] = table[c].round(1)
    return table


# -----------------------------------------------------------------------------
# 4. service-floor table
# -----------------------------------------------------------------------------
def service_floor(stop, has_sun, floor=None):
    floor = floor or config.SERVICE_FLOOR_MIN
    print(f"[4] Service-floor analysis (failing if headway > {floor} min)...")

    v67 = stop["v_index"].quantile(config.HIGH_VULN_TERTILE)
    high = stop[stop["v_index"] >= v67].copy()

    def share(d, col):
        s = d[col].dropna()
        return round((s > floor).mean() * 100, 1) if len(s) else float("nan")

    rows = [{"Day type":               "Weekday (Tue/Wed/Thu)",
             "All stops % failing":    share(stop, "hw_wd"),
             "High-vuln. % failing":   share(high, "hw_wd"),
             "High-vuln. count":       int((high["hw_wd"] > floor).sum())}]
    if has_sun:
        rows.append({"Day type":         "Sunday",
                     "All stops % failing":  share(stop, "hw_sun"),
                     "High-vuln. % failing": share(high, "hw_sun"),
                     "High-vuln. count":     int((high["hw_sun"] > floor).sum())})
    rows.append({"Day type":               "Public holiday",
                 "All stops % failing":    share(stop, "hw_hol"),
                 "High-vuln. % failing":   share(high, "hw_hol"),
                 "High-vuln. count":       int((high["hw_hol"] > floor).sum())})

    # coverage (unserved stop-hours): reported alongside, not imputed
    coverage_note = (f"Unserved stops in holiday baseline: "
                     f"{stop['hw_hol'].isna().sum()} "
                     f"({stop['hw_hol'].isna().mean()*100:.1f}%)")
    return pd.DataFrame(rows), coverage_note


# -----------------------------------------------------------------------------
# 5. figure
# -----------------------------------------------------------------------------
def make_figure_burden(top):
    """Panel A alone: top-N stops by holiday social burden (was top panel)."""
    print("[5a] Building corridors-of-exclusion burden figure...")
    fig, ax = plt.subplots(figsize=(14, 6))
    n = len(top)
    x = np.arange(n)
    w = 0.38
    labels = [
        f"#{i+1}\n{(row.get('stop_name') or '')[:22]}\n"
        f"({(row.get('lines') or '')[:18]})"
        for i, row in top.iterrows()
    ]
    ax.bar(x - w/2, top["burden_wd"],  w, label="Weekday burden",
           color="#16A34A", alpha=0.85)
    ax.bar(x + w/2, top["burden_hol"], w, label="Holiday burden",
           color="#DC2626", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Social burden (V x headway)", fontsize=10)
    ax.set_title(f"Top {n} stops by holiday social burden — corridors of exclusion",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9); ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "thesis_phase1_burden.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    return path


def make_figure_floor_and_distribution(floor_df, stop, top_ids, n_top):
    """Panels B+C together: service-floor failures and headway distribution
    (was the bottom row of the combined figure)."""
    print("[5b] Building service-floor + headway-distribution figure...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---- left: service-floor bars ----
    ax = axes[0]
    cats = floor_df["Day type"].tolist()
    xb = np.arange(len(cats))
    bars_all = ax.bar(xb - 0.2, floor_df["All stops % failing"],  0.38,
                       label="All stops", color="#0D9488", alpha=0.85)
    bars_vuln = ax.bar(xb + 0.2, floor_df["High-vuln. % failing"], 0.38,
                        label="High-vulnerability tertile", color="#DC2626", alpha=0.85)
    ax.set_xticks(xb); ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel(f"% stops failing >{config.SERVICE_FLOOR_MIN}-min standard",
                  fontsize=10)
    ax.set_title("Service-floor failures by day-type and vulnerability",
                 fontsize=11, fontweight="bold", pad=10)
    # headroom above the tallest bar so labels never crowd the title
    ymax = max(floor_df["All stops % failing"].max(),
               floor_df["High-vuln. % failing"].max())
    ax.set_ylim(0, ymax * 1.22)
    ax.legend(fontsize=9, loc="upper left")
    ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
    # label offset scaled to the chart's own range instead of a fixed value,
    # so tall bars never crowd the title
    label_offset = ymax * 0.03
    for rect in list(bars_all) + list(bars_vuln):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + label_offset,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)

    # ---- right: headway distribution, top-N vs rest ----
    ax = axes[1]
    rest = stop[~stop["stop_id"].isin(top_ids)]
    topd = stop[ stop["stop_id"].isin(top_ids)]
    ax.hist(rest["hw_hol"].dropna().clip(upper=120), bins=30, alpha=0.55,
            color="#0D9488", label="All other stops", density=True)
    ax.hist(topd["hw_hol"].dropna().clip(upper=120), bins=10, alpha=0.75,
            color="#DC2626", label=f"Top {n_top} worst stops", density=True)
    ax.axvline(config.SERVICE_FLOOR_MIN, color="orange", linestyle="--",
               linewidth=1.8, label=f"{config.SERVICE_FLOOR_MIN}-min standard")
    ax.set_xlabel("Holiday headway (min)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Holiday headway distribution: top-N vs. rest",
                 fontsize=11, fontweight="bold", pad=10)
    ax.legend(fontsize=8.5)

    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "thesis_phase1_floor_distribution.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    return path


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main(top_n=10):
    print("--- Phase 2: Corridors of Exclusion & Service Floor ---")
    stop, has_sun = load_stop_state()
    stop_gdf = attach_metadata(stop)
    stop_gdf = flag_regular_urban(stop_gdf)

    # ---- DUAL RANKING ----
    top_all = top_corridors(stop_gdf, n=top_n)
    top_reg = top_corridors(stop_gdf[stop_gdf["regular_urban"]], n=top_n)

    # ---- SERVICE FLOOR: full network AND regular urban only ----
    floor_all, cov_all = service_floor(stop_gdf, has_sun)
    floor_reg, cov_reg = service_floor(stop_gdf[stop_gdf["regular_urban"]], has_sun)

    # ---- exports ----
    p_top_all = os.path.join(config.OUTPUT_DIR, "phase1_top10_corridors_all.csv")
    p_top_reg = os.path.join(config.OUTPUT_DIR, "phase1_top10_corridors_regular.csv")
    top_all.to_csv(p_top_all, index=True)
    top_reg.to_csv(p_top_reg, index=True)
    p_floor_all = os.path.join(config.OUTPUT_DIR, "phase1_service_floor_all.csv")
    p_floor_reg = os.path.join(config.OUTPUT_DIR, "phase1_service_floor_regular.csv")
    floor_all.to_csv(p_floor_all, index=False)
    floor_reg.to_csv(p_floor_reg, index=False)
    fig_path_burden = make_figure_burden(top_reg)
    fig_path_floor_dist = make_figure_floor_and_distribution(
        floor_reg, stop_gdf[stop_gdf["regular_urban"]],
        top_reg["stop_id"].tolist(), top_n)

    # ---- console summary ----
    cols_show = [c for c in ["stop_name", "lines", "v_index",
                              "hw_wd", "hw_hol", "hw_ratio_hol_wd", "burden_hol"]
                 if c in top_reg.columns]

    print("\n=== TOP CORRIDORS — ABSOLUTE (full network) ===")
    print("(includes night and interurban lines; dominated by fringe services)")
    print(top_all[cols_show].to_string())

    print("\n=== TOP CORRIDORS — REGULAR URBAN NETWORK ===")
    print("(stops with weekday baseline; the corridors-of-exclusion proper)")
    print(top_reg[cols_show].to_string())

    print("\n=== SERVICE-FLOOR TABLE (regular urban network) ===")
    print(floor_reg.to_string(index=False))
    print(f"\n{cov_reg}")

    print("\n=== SERVICE-FLOOR TABLE (full network, for comparison) ===")
    print(floor_all.to_string(index=False))

    # headline sentences from the regular-network cut
    hi_hol_pct = floor_reg.loc[floor_reg["Day type"] == "Public holiday",
                               "High-vuln. % failing"].values[0]
    hi_wd_pct  = floor_reg.loc[floor_reg["Day type"].str.startswith("Weekday"),
                               "High-vuln. % failing"].values[0]
    print("\nSummary:")
    print("  (1) Of the high-vulnerability tertile of stops on the regular urban")
    print(f"      network, {hi_hol_pct:.1f}% fail the {config.SERVICE_FLOOR_MIN}-minute "
          f"service floor on public holidays,")
    print(f"      compared with {hi_wd_pct:.1f}% on weekdays.")
    print("  (2) When the full feed is considered, including night and interurban")
    print(f"      services, the failing share rises to "
          f"{floor_all.loc[floor_all['Day type']=='Public holiday','High-vuln. % failing'].values[0]:.1f}% "
          f"on holidays.")

    rp = config.write_run_parameters("phase1", extra={
        "top_n": top_n,
        "service_floor_min": config.SERVICE_FLOOR_MIN,
        "regular_network_hw_max_min": config.REGULAR_NETWORK_HW_MAX_MIN,
        "non_urban_route_prefixes": list(config.NON_URBAN_ROUTE_PREFIXES),
        "non_urban_route_exact": list(config.NON_URBAN_ROUTE_EXACT),
        "stops_total":          int(len(stop_gdf)),
        "stops_regular_urban":  int(stop_gdf["regular_urban"].sum()),
        "stops_structural_deficit": int(
            (stop_gdf["hw_wd"].notna() &
             (stop_gdf["hw_wd"] > config.REGULAR_NETWORK_HW_MAX_MIN)).sum()),
        "stops_structural_deficit_high_vuln": int((
            stop_gdf["hw_wd"].notna() &
            (stop_gdf["hw_wd"] > config.REGULAR_NETWORK_HW_MAX_MIN) &
            (stop_gdf["v_index"] >= stop_gdf["v_index"].quantile(
                config.HIGH_VULN_TERTILE))).sum()),
        "high_vuln_tertile_cut": float(stop_gdf["v_index"].quantile(
            config.HIGH_VULN_TERTILE)),
        "high_vuln_hol_fail_pct_regular": float(hi_hol_pct),
    })
    print("\nOutputs:")
    for p in (p_top_all, p_top_reg, p_floor_all, p_floor_reg,
              fig_path_burden, fig_path_floor_dist, rp):
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
