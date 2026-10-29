"""
================================================================================
build_stop_table.py  —  Core analytical-table builder
================================================================================
Holiday Transit Equity framework — Phase 1 (vulnerability) + Phase 2 (supply).

Reads raw inputs, builds the stop-hourly analytical table that every downstream
phase consumes. All parameters come from config.py; nothing is hard-coded here.

What this does, in order:
  1. Build the vulnerability index (PTD) from ISTAT shares + configured weights,
     plus an equal-weight sensitivity variant.
  2. Reconstruct the additive GTFS calendar for the PRIMARY feed, SELF-DERIVING
     the number of service days per day-type (no hard-coded denominators).
  3. Compute frequency-derived hourly headways for weekday / Sunday / holiday,
     with the NaN-canonical no-service rule (unserved stop-hours stay NaN).
  4. Attach stops to census sections by point-in-polygon, then snap unmatched
     stops to the nearest section with a v_index within the configured cap
     (zero-population sections are retained; see README, "Known issues").
  5. Compute burden (V x headway) per day-type.
  6. Emit:
       - milan_stop_hourly.csv      (core stop-hour table)
       - milan_stops.gpkg           (stop-level aggregate + geometry)
       - milan_zones.gpkg           (zone choropleth with v_index)
       - run_parameters_builder.json
  7. Print a validation block to compare against the established key facts.

Usage:
    python build_stop_table.py
================================================================================
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler

import config

config.validate()


# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------
def _strip(df):
    """Strip whitespace and stray quotes from textual GTFS columns."""
    def clean(c):
        if pd.api.types.is_string_dtype(c.dtype) or pd.api.types.is_object_dtype(c.dtype):
            return c.astype("string").str.replace('"', "", regex=False).str.strip()
        return c
    return df.apply(clean)


def _read_gtfs(feed_path, filename, usecols=None):
    return _strip(pd.read_csv(os.path.join(feed_path, filename),
                              usecols=usecols, dtype=str))


def _hour_from_arrival(series):
    """
    GTFS arrival_time -> integer hour, tolerant of times >= 24:00:00 AND of
    blank arrival_time fields (GTFS permits empty times on non-timepoint rows;
    a bare .astype(int) would crash the whole build on the first blank).
    Blank/unparseable rows become NaN and are dropped by the window filter.
    """
    return pd.to_numeric(series.str.split(":").str[0], errors="coerce")


# =============================================================================
# 1. VULNERABILITY INDEX (PTD)
# =============================================================================
def build_vulnerability():

    print("[1] Building vulnerability index from ISTAT shares...")

    istat = pd.read_excel(config.ISTAT_XLSX)

    # --- CRITICAL: restrict to study-area comune BEFORE normalising ---

    if config.ISTAT_COMUNE_CODE is not None:

        col = config.ISTAT_COMUNE_COLUMN

        before = len(istat)

        istat = istat[istat[col].astype(str).str.split(".").str[0]

                      == str(config.ISTAT_COMUNE_CODE)].copy()

        print(f"    Filtered ISTAT to comune {config.ISTAT_COMUNE_CODE}: "

              f"{before} -> {len(istat)} rows (normalisation now within study area)")

        if len(istat) == 0:
            raise ValueError(
                f"No ISTAT rows match {col} == {config.ISTAT_COMUNE_CODE!r}. "
                "Check ISTAT_COMUNE_CODE in config.py (codes are compared as "
                "text, so a leading zero such as '001272' will not match 1272).")

    # Submitted-analysis behaviour: retain all census sections inside Milan.
    # Undefined demographic ratios (for example, zero-denominator sections) are
    # converted to 0 below and low-population sections are flagged rather than
    # excluded. This preserves the analytical universe used for the thesis run.

    # Compute each configured share = sum(numerator cols) / denominator col

    share_cols = []

    for name, spec in config.VULNERABILITY_VARIABLES.items():

        num = istat[spec["numerator"]].apply(pd.to_numeric, errors="coerce").sum(axis=1)

        den = pd.to_numeric(istat[spec["denominator"]], errors="coerce")

        istat[name] = (num / den).replace([np.inf, -np.inf], np.nan).fillna(0)

        share_cols.append(name)

 
    # Flag sections below the population floor as statistically unstable.
    istat["_pop_total"] = pd.to_numeric(istat[config.POPULATION_COLUMN], errors="coerce").fillna(0)
    istat["low_population_flag"] = istat["_pop_total"] < config.MIN_POPULATION_FOR_VINDEX
    n_flagged = int(istat["low_population_flag"].sum())
    print(f"    Sections flagged as low-population (<{config.MIN_POPULATION_FOR_VINDEX}): "
          f"{n_flagged} of {len(istat)} ({n_flagged/len(istat)*100:.1f}%)")

    # Normalise shares to [0,1] then apply configured weights
    scaler = MinMaxScaler()
    norm = scaler.fit_transform(istat[share_cols].fillna(0))
    norm_df = pd.DataFrame(norm, columns=share_cols, index=istat.index)

    istat["v_index"] = sum(config.VULNERABILITY_WEIGHTS[c] * norm_df[c] for c in share_cols)

    if config.RUN_EQUAL_WEIGHT_SENSITIVITY:
        w_eq = 1.0 / len(share_cols)
        istat["v_index_eqw"] = sum(w_eq * norm_df[c] for c in share_cols)

    istat["SEZ_JOIN_ID"] = istat[config.ZONE_ID_COLUMN].astype(str).str.split(".").str[0]

    keep = ["SEZ_JOIN_ID", "v_index", "low_population_flag"] + (
        ["v_index_eqw"] if config.RUN_EQUAL_WEIGHT_SENSITIVITY else [])
    return istat[keep].copy()


# =============================================================================
# 2. CALENDAR RECONSTRUCTION + SELF-DERIVED DENOMINATORS
# =============================================================================
def reconstruct_calendar(feed_path):
    """Reconstruct explicit date-service pairs from a GTFS feed.

    Supports both conventional ``calendar.txt`` recurrence and additive
    ``calendar_dates.txt`` feeds.  Exception type 2 removes service on a date;
    exception type 1 adds service.  The Milan feeds used in the thesis are
    additive, so this generalisation does not alter their intended treatment.
    """
    print("[2] Reconstructing GTFS calendar (self-derived denominators)...")

    cal_path = os.path.join(feed_path, "calendar.txt")
    pairs = []
    if os.path.exists(cal_path):
        try:
            base = _read_gtfs(feed_path, "calendar.txt")
        except pd.errors.EmptyDataError:
            base = pd.DataFrame()
        if len(base):
            day_cols = ["monday", "tuesday", "wednesday", "thursday",
                        "friday", "saturday", "sunday"]
            for row in base.itertuples(index=False):
                r = row._asdict()
                dates = pd.date_range(pd.to_datetime(r["start_date"], format="%Y%m%d"),
                                      pd.to_datetime(r["end_date"], format="%Y%m%d"), freq="D")
                active = {i for i, c in enumerate(day_cols) if str(r.get(c, "0")) == "1"}
                for d in dates:
                    if d.weekday() in active:
                        pairs.append((str(r["service_id"]), d.strftime("%Y%m%d")))

    explicit = pd.DataFrame(pairs, columns=["service_id", "date"]).drop_duplicates()

    cd_path = os.path.join(feed_path, "calendar_dates.txt")
    if os.path.exists(cd_path):
        try:
            exc = _read_gtfs(feed_path, "calendar_dates.txt")
        except pd.errors.EmptyDataError:
            exc = pd.DataFrame(columns=["service_id", "date", "exception_type"])
        if len(exc):
            if "exception_type" not in exc.columns:
                exc["exception_type"] = "1"
            exc["exception_type"] = exc["exception_type"].astype(str).str.strip()
            adds = exc[exc["exception_type"] == "1"][["service_id", "date"]]
            removes = exc[exc["exception_type"] == "2"][["service_id", "date"]]
            if len(removes) and len(explicit):
                rm = set(map(tuple, removes.astype(str).to_numpy()))
                explicit = explicit[~explicit[["service_id", "date"]].apply(tuple, axis=1).isin(rm)]
            explicit = pd.concat([explicit, adds], ignore_index=True).drop_duplicates()

    if explicit.empty:
        raise ValueError(f"No operating date-service pairs reconstructed from {feed_path}")

    cal = explicit.copy()
    cal["date_dt"] = pd.to_datetime(cal["date"], format="%Y%m%d")
    cal["dayname"] = cal["date_dt"].dt.day_name()

    core_holidays = set(config.CORE_HOLIDAY_DATES)
    all_holidays = set(config.ALL_HOLIDAY_DATES)
    is_core_holiday = cal["date"].isin(core_holidays)
    is_any_holiday = cal["date"].isin(all_holidays)

    wd_mask = cal["dayname"].isin(config.WEEKDAY_DAYS) & ~is_any_holiday
    sun_mask = (cal["dayname"] == "Sunday") & ~is_any_holiday
    hol_mask = is_core_holiday

    ids = {
        "wd": cal.loc[wd_mask, "service_id"].unique(),
        "sun": cal.loc[sun_mask, "service_id"].unique(),
        "hol": cal.loc[hol_mask, "service_id"].unique(),
    }
    n_days = {
        "wd": cal.loc[wd_mask, "date"].nunique(),
        "sun": cal.loc[sun_mask, "date"].nunique(),
        "hol": cal.loc[hol_mask, "date"].nunique(),
    }
    print(f"    Self-derived day counts: weekday={n_days['wd']}, "
          f"Sunday={n_days['sun']}, holiday={n_days['hol']}")
    print(f"    Feed window: {cal['date_dt'].min().date()} -> {cal['date_dt'].max().date()}")
    return cal, ids, n_days


# =============================================================================
# 3. FREQUENCY-DERIVED HOURLY HEADWAYS (NaN-canonical no-service)
# =============================================================================
def hourly_headways(feed_path, ids, n_days, cal):
    print("[3] Computing frequency-derived hourly headways...")
    trips = _read_gtfs(feed_path, "trips.txt", usecols=["trip_id", "service_id"])
    st = _read_gtfs(feed_path, "stop_times.txt",
                    usecols=["trip_id", "arrival_time", "stop_id"])
    st["hour"] = _hour_from_arrival(st["arrival_time"])
    st = st[config.operational_hour_filter(st["hour"])].copy()
    st["hour"] = st["hour"].astype(int)

    # Same two-set holiday logic as reconstruct_calendar (see comment there):
    # exclusion uses ALL holiday dates; the holiday day-type uses core dates.
    core_set = set(config.CORE_HOLIDAY_DATES)
    all_set  = set(config.ALL_HOLIDAY_DATES)

    def day_mask(label):
        if label == "wd":
            return cal["dayname"].isin(config.WEEKDAY_DAYS) & ~cal["date"].isin(all_set)
        if label == "sun":
            return (cal["dayname"] == "Sunday") & ~cal["date"].isin(all_set)
        return cal["date"].isin(core_set)  # hol

    def one(label):
        sids = ids[label]
        days = n_days[label]
        if len(sids) == 0 or days == 0:
            return pd.DataFrame(columns=["stop_id", "hour", f"hw_{label}"])

        # dates-of-this-day-type each service_id actually runs on
        dates_per_sid = (cal.loc[day_mask(label)]
                         .groupby("service_id")["date"].nunique())

        tids = trips[trips["service_id"].isin(sids)][["trip_id", "service_id"]]
        rel = st.merge(tids, on="trip_id")
        # each stop_time operates on as many days as its service_id runs
        rel["sid_dates"] = rel["service_id"].map(dates_per_sid).fillna(0)
        # total operations over the window, summed per stop-hour
        ops = (rel.groupby(["stop_id", "hour"])["sid_dates"]
               .sum().reset_index(name="ops"))
        # trips per day in that hour = ops / days  ->  headway = 60 / (ops/days)
        ops[f"hw_{label}"] = 60.0 / (ops["ops"] / days)
        return ops[["stop_id", "hour", f"hw_{label}"]]

    hw_wd  = one("wd")
    hw_hol = one("hol")
    hw_sun = one("sun") if config.INCLUDE_SUNDAY else None

    master = hw_wd.merge(hw_hol, on=["stop_id", "hour"], how="outer")
    if config.INCLUDE_SUNDAY:
        master = master.merge(hw_sun, on=["stop_id", "hour"], how="outer")
    return master

# =============================================================================
# 4. SPATIAL JOIN  +  5. BURDEN
# =============================================================================
def attach_geography_and_burden(master, vuln):
    print("[4] Spatial join to census zones + boundary clip...")
    zones = gpd.read_file(config.ZONES_GEOJSON).to_crs(config.CRS_GEOGRAPHIC)
    zones["SEZ_JOIN_ID"] = zones[config.ZONE_ID_COLUMN].astype(str).str.split(".").str[0]
    zones = zones.merge(vuln, on="SEZ_JOIN_ID", how="inner")

    feed_path = config.GTFS_FEEDS[config.PRIMARY_FEED]
    stops = _read_gtfs(feed_path, "stops.txt", usecols=["stop_id", "stop_name", "stop_lat", "stop_lon"])
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs=config.CRS_GEOGRAPHIC,
    )

    # ── Two-stage vulnerability assignment (metric CRS for true distances) ──
    # Stage 1 (within): point-in-polygon on every census section that carries a
    #     v_index, i.e. every section of the study-area comune present in both
    #     the ISTAT table and the boundary file.
    # Stage 2 (snap):   stops not inside any such section take the v_index of
    #     the NEAREST one, if within config.ZONE_SNAP_MAX_DIST_M.
    # KNOWN ISSUE (kept for reproducibility of the submitted thesis; see README):
    #     sections present in the ISTAT table with zero resident population are
    #     NOT excluded. build_vulnerability() sets their undefined shares to 0,
    #     so they carry a v_index and stops inside them are assigned in stage 1
    #     rather than snapped to a populated neighbour.
    # Stops farther than the cap from any section with a v_index lie outside the
    # municipality's residential fabric and are EXCLUDED: this cap-based
    # exclusion supersedes the former dissolved-boundary clip, which was
    # built from populated sections only and therefore had piazzas and rail
    # yards as holes — silently dropping Duomo, Cairoli, Centrale FS and
    # ~530 further stops (diagnosed 2026-07-10 via the metro
    # demand-validation layer).
    zones_m = zones.to_crs(config.CRS_METRIC)
    pts_m = stops_gdf.to_crs(config.CRS_METRIC)

    j1 = gpd.sjoin(pts_m, zones_m[["geometry", "v_index"]],
                   how="left", predicate="within").drop(columns="index_right")
    j1 = j1[~j1.index.duplicated(keep="first")]   # section-border double hits
    inside = j1["v_index"].notna()

    snap = gpd.sjoin_nearest(
        pts_m.loc[~inside].drop(columns=[], errors="ignore"),
        zones_m[["geometry", "v_index"]],
        how="left", max_distance=config.ZONE_SNAP_MAX_DIST_M,
        distance_col="snap_dist_m").drop(columns="index_right")
    snap = snap[~snap.index.duplicated(keep="first")]
    snapped = snap["v_index"].notna()

    n_within = int(inside.sum())
    n_snap = int(snapped.sum())
    n_excl = int((~inside).sum()) - n_snap
    print(f"    Stage 1 — inside a census section with a v_index: {n_within}")
    print(f"    Stage 2 — snapped to nearest section with a v_index "
          f"(<= {config.ZONE_SNAP_MAX_DIST_M} m): {n_snap}")
    if n_snap:
        print(f"              snap distance: median "
              f"{snap.loc[snapped, 'snap_dist_m'].median():.0f} m, "
              f"max {snap.loc[snapped, 'snap_dist_m'].max():.0f} m")
    print(f"    Excluded  — > {config.ZONE_SNAP_MAX_DIST_M} m from any "
          f"section with a v_index (outside residential fabric): {n_excl}")

    joined = pd.concat([j1.loc[inside],
                        snap.loc[snapped].drop(columns="snap_dist_m")])
    joined = gpd.GeoDataFrame(joined, geometry="geometry",
                              crs=config.CRS_METRIC).to_crs(config.CRS_GEOGRAPHIC)

    n_stops_in_boundary = joined["stop_id"].nunique()
    # NOTE: this counts every stop located in the study area. The analytical
    # network is the subset with scheduled service in the operational window
    # on a baseline weekday, Sunday or core holiday (reported in the
    # validation block; 3,187 stops in the thesis run).
    print(f"    Stops located in the study area (stage 1 + stage 2): {n_stops_in_boundary}")

    # Merge v_index onto the stop-hourly table.
    # Deduplicate on stop_id ALONE: a stop lying exactly on a shared zone
    # boundary (or a duplicated stop_id row in stops.txt) can match more than
    # one zone; drop_duplicates on (stop_id, v_index) would keep both rows and
    # silently DUPLICATE every stop-hour record for that stop downstream.
    stop_v = joined[["stop_id", "v_index"]]
    n_multi = stop_v["stop_id"].duplicated().sum()
    if n_multi:
        print(f"    NOTE: {n_multi} stop(s) matched more than one census zone "
              "(boundary coincidence); keeping the first match per stop.")
    stop_v = stop_v.drop_duplicates("stop_id")
    df = master.merge(stop_v, on="stop_id", how="inner")

    print("[5] Computing burden (V x headway) per day-type...")
    for label in (["wd", "sun", "hol"] if config.INCLUDE_SUNDAY else ["wd", "hol"]):
        hwcol = f"hw_{label}"
        if hwcol in df.columns:
            df[f"burden_{label}"] = df["v_index"] * df[hwcol]  # NaN where unserved

    return df, joined, zones, n_stops_in_boundary


# =============================================================================
# 6/7. EXPORTS + VALIDATION BLOCK
# =============================================================================
def export_and_validate(df, joined, zones, n_days, n_stops):
    print("[6] Exporting outputs...")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(config.OUTPUT_DIR, "milan_stop_hourly.csv")
    df.to_csv(csv_path, index=False)

    # Stop-level aggregate (median headway, mean burden) + geometry
    agg_spec = {"v_index": ("v_index", "mean"),
                "hw_wd": ("hw_wd", "median"),
                "hw_hol": ("hw_hol", "median")}
    if config.INCLUDE_SUNDAY and "hw_sun" in df.columns:
        agg_spec["hw_sun"] = ("hw_sun", "median")
    stop_level = df.groupby("stop_id").agg(**agg_spec).reset_index()
    stops_geom = joined[["stop_id", "stop_name", "geometry"]].drop_duplicates("stop_id")
    stops_out = gpd.GeoDataFrame(stop_level.merge(stops_geom, on="stop_id", how="left"),
                                 crs=config.CRS_GEOGRAPHIC)
    stops_gpkg = os.path.join(config.OUTPUT_DIR, "milan_stops.gpkg")
    stops_out.to_file(stops_gpkg, driver="GPKG")

    zones_gpkg = os.path.join(config.OUTPUT_DIR, "milan_zones.gpkg")
    zones.to_file(zones_gpkg, driver="GPKG")

    # run-parameters record (reproducibility evidence)
    rp = config.write_run_parameters("builder", extra={
        "n_weekday_days": int(n_days["wd"]),
        "n_sunday_days": int(n_days["sun"]),
        "n_holiday_days": int(n_days["hol"]),
        # n_stops_analysed keeps its original meaning for comparability with
        # earlier run records: stops located in the study area (stage 1 +
        # stage 2). The analytical network is n_stops_analytical_network.
        "n_stops_analysed": int(n_stops),
        "n_stops_located_in_study_area": int(n_stops),
        "n_stops_analytical_network": int(len(stop_level)),
        "n_zones_with_vindex": int(len(zones)),
    })

    # --- VALIDATION BLOCK ---------------------------------------------------
    print("\n" + "=" * 70)
    print("VALIDATION BLOCK  — compare against established key facts")
    print("=" * 70)

    # served-only medians (NaN-canonical: NaN simply excluded by .median())
    med_wd  = stop_level["hw_wd"].median()
    med_hol_served = stop_level["hw_hol"].median()
    exp = config.VALIDATION_EXPECTATIONS or {}

    def _exp(key):
        """Format the frozen-run expectation, if one is pinned in config."""
        return f"   [frozen-run expectation: {exp[key]}]" if key in exp else ""

    print(f"Stops located in the study area:         {n_stops}"
          f"{_exp('n_stops_analysed')}")
    print(f"Analytical network (stops with service "
          f"{config.OPERATIONAL_HOUR_START:02d}:00-{config.OPERATIONAL_HOUR_END:02d}:00 "
          f"on a baseline weekday, Sunday or core holiday): {len(stop_level)}")
    print(f"Zones with v_index:                      {len(zones)}")
    print(f"Median weekday headway (served):         {med_wd:.1f} min")
    print(f"Median holiday headway (served):         {med_hol_served:.1f} min")
    if config.INCLUDE_SUNDAY and "hw_sun" in stop_level.columns:
        print(f"Median Sunday headway (served):          {stop_level['hw_sun'].median():.1f} min")

    # Coverage statistic (the NaN-canonical first-class finding)
    if config.REPORT_COVERAGE_SEPARATELY:
        served_hol = df["hw_hol"].notna().mean() * 100
        unserved_stops = stop_level["hw_hol"].isna().sum()
        print(f"Stop-hours WITH holiday service:         {served_hol:.1f}%")
        print(f"Stops with NO holiday service (any hr):  {unserved_stops}")

    # Burden ratio — two views, to show the no-service-rule effect explicitly
    # (a) served-only (NaN excluded)
    bw = df["burden_wd"].mean()
    bh_served = df["burden_hol"].mean()
    # (b) penalised (unserved holiday stop-hours -> NO_SERVICE_PENALTY_MIN),
    #     the same disclosed convention the optimiser uses at its boundary
    bh_pen = (df["v_index"] * df["hw_hol"].fillna(config.NO_SERVICE_PENALTY_MIN)).mean()
    # (c) penalised at STOP level: the basis of the penalised ratio reported in
    #     the thesis (§5.2) and checked by thesis_code.ipynb. Stop-level
    #     headways are medians of served hours; stops with no holiday service
    #     take the penalty. Printed only; nothing downstream reads it.
    bh_pen_stop = (stop_level["v_index"] *
                   stop_level["hw_hol"].fillna(config.NO_SERVICE_PENALTY_MIN)).mean()
    bw_stop = (stop_level["v_index"] * stop_level["hw_wd"]).mean()
    print(f"Holiday/weekday burden ratio (served, stop-hour basis; thesis §5.2):    "
          f"{bh_served / bw:.2f}x")
    print(f"Holiday/weekday burden ratio (penalised, stop-hour basis):              "
          f"{bh_pen / bw:.2f}x{_exp('burden_ratio_penalised')}")
    print(f"Holiday/weekday burden ratio (penalised, stop level; thesis §5.2):      "
          f"{bh_pen_stop / bw_stop:.2f}x")
    print("  (served = NaN excluded; penalised = unserved stop-hours, or stops with no "
          f"holiday service, set to {config.NO_SERVICE_PENALTY_MIN} min, the disclosed "
          "optimiser convention)")

    # equal-weight sensitivity spatial agreement
    if config.RUN_EQUAL_WEIGHT_SENSITIVITY and "v_index_eqw" in zones.columns:
        corr = zones["v_index"].corr(zones["v_index_eqw"])
        print(f"v_index vs equal-weight variant corr:    {corr:.3f}   "
              "[high = weight choice not driving the pattern]")

    print("=" * 70)
    print("Outputs written:")
    for p in (csv_path, stops_gpkg, zones_gpkg, rp):
        print(f"  {os.path.basename(p)}")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("--- Consolidated builder (Phases 1-2) ---")
    vuln = build_vulnerability()
    feed_path = config.GTFS_FEEDS[config.PRIMARY_FEED]
    cal, ids, n_days = reconstruct_calendar(feed_path)
    master = hourly_headways(feed_path, ids, n_days, cal)
    df, joined, zones, n_stops = attach_geography_and_burden(master, vuln)
    export_and_validate(df, joined, zones, n_days, n_stops)
    print("\nDone. Next: optimise.py (README.md gives the full run order).")


if __name__ == "__main__":
    main()
