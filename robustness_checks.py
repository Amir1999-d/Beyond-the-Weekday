"""
robustness_checks.py — consolidated robustness and defence diagnostics.

These checks do not overwrite principal thesis outputs. They read the existing
pipeline outputs and write robustness_*.csv files. Use --skip-osm to avoid the
live OpenStreetMap boundary-buffer check and --skip-atm to omit the optional
private metro passenger-cost check.
"""
import argparse, os, glob, traceback, importlib
import numpy as np
import pandas as pd
import geopandas as gpd

import config
import optimise
import build_stop_table as bst

OUT = config.OUTPUT_DIR
LINE = "=" * 78
Q_BURDEN = config.DOUBLE_EXCLUSION_BURDEN_Q
PEN = config.NO_SERVICE_PENALTY_MIN
PUBLISHED = {"double_excluded": 415, "priority": 184, "donors": 336, "recipients": 361,
             "moderate_gini": 20.8, "tertiles": (6.8, 10.2, 22.1)}
PUBLISHED_TERTILES = {"Low": 6.8, "Middle": 10.2, "High": 22.1}

# Lazy cache for GTFS tables shared by checks 2, 3 and 8.
_G = {}


def header(t):
    print(f"\n{LINE}\n{t}\n{LINE}")

def sid(s):
    return s.astype(str).str.strip()

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else float("nan")

def base_stops():
    s = optimise.load_stop_level()
    s["stop_id"] = sid(s["stop_id"])
    return s

def published_access():
    p4 = gpd.read_file(os.path.join(OUT, "phase4_essential_access.gpkg"))
    p4["stop_id"] = sid(p4["stop_id"])
    return p4[["stop_id", "essential_access"]].rename(columns={"essential_access": "access_ok"}).assign(
        access_ok=lambda d: d["access_ok"].astype(str).str.lower().isin(["true", "1", "yes"]))

def decomposition(stop, access, label):
    """Double exclusion (penalised burden, top quartile) and its headway-only counterpart."""
    d = stop.merge(access, on="stop_id", how="inner")
    fail = ~d["access_ok"].astype(bool)
    t = pd.qcut(d["v_index"], 3, labels=["Low", "Middle", "High"])
    de = (d["burden_hol"] >= d["burden_hol"].quantile(Q_BURDEN)) & fail
    q_h = d["hw_hol"].quantile(Q_BURDEN)
    de_h = (d["hw_hol"] > q_h) & fail
    tab = pd.DataFrame({
        "stops": t.value_counts().sort_index(),
        "access_failure_%": (fail.groupby(t, observed=False).mean() * 100),
        "double_exclusion_%": (de.groupby(t, observed=False).mean() * 100),
        f"headway>{q_h:.0f}_and_poor_access_%": (de_h.groupby(t, observed=False).mean() * 100),
    }).round(1)
    ratio = tab.loc["High", "double_exclusion_%"] / tab.loc["Low", "double_exclusion_%"]
    ratio_h = tab.iloc[2, 3] / tab.iloc[0, 3] if tab.iloc[0, 3] else float("nan")
    print(f"\n   [{label}] double-excluded {int(de.sum())} | high/low {ratio:.2f}x "
          f"| headway-only high/low {ratio_h:.2f}x")
    print(tab.to_string())
    d["double_exclusion"] = de
    return d, tab

class _MaxScaler:
    """Divide each share by its maximum: 0% keeps a true zero."""
    def fit_transform(self, X):
        X = np.asarray(X, dtype=float)
        m = X.max(axis=0)
        m[m == 0] = 1.0
        return X / m

def index_zero_point():
    header("B. VULNERABILITY INDEX WITH A NATURAL ZERO (share / max) VERSUS MIN-MAX")
    base = base_stops()
    acc = published_access()

    # published index, same code path: must reproduce the thesis before comparing
    orig = optimise.classify(base.copy())
    d0, t0 = decomposition(orig, acc, "published index (min-max)")
    pri0 = set(d0.loc[d0["double_exclusion"] & (d0["role"] == "recipient"), "stop_id"])
    _, st0 = optimise.run_scenario(orig.copy(), config.SCENARIOS[config.RECOMMENDED_SCENARIO])
    ok = (int(d0["double_exclusion"].sum()) == PUBLISHED["double_excluded"]
          and len(pri0) == PUBLISHED["priority"])
    print(f"   reproduction: {int(d0['double_exclusion'].sum())} double-excluded, {len(pri0)} priority, "
          f"Moderate Gini {st0['gini_improvement_%']:.2f}% -> {'OK' if ok else 'MISMATCH (read with care)'}")

    # alternative index through the builder's own functions
    saved = bst.MinMaxScaler
    bst.MinMaxScaler = _MaxScaler
    try:
        vuln_alt = bst.build_vulnerability()
    finally:
        bst.MinMaxScaler = saved
    master = pd.DataFrame({"stop_id": base["stop_id"]})
    df_alt, _, _, _ = bst.attach_geography_and_burden(master, vuln_alt)
    df_alt["stop_id"] = sid(df_alt["stop_id"])
    v_alt = df_alt.drop_duplicates("stop_id").set_index("stop_id")["v_index"]

    alt = base.copy()
    alt["v_index"] = alt["stop_id"].map(v_alt)
    missing = int(alt["v_index"].isna().sum())
    alt = alt.dropna(subset=["v_index"])
    alt["burden_hol"] = alt["v_index"] * alt["hw_hol"]
    print(f"\n   alternative index attached to {len(alt)} stops (missing {missing}); "
          f"range {alt['v_index'].min():.3f}-{alt['v_index'].max():.3f}")
    rho = base.set_index("stop_id")["v_index"].rank().corr(alt.set_index("stop_id")["v_index"].rank())
    print(f"   Spearman rank correlation with the published index: {rho:.3f}")

    alt = optimise.classify(alt)
    d1, t1 = decomposition(alt, acc, "natural-zero index")
    pri1 = set(d1.loc[d1["double_exclusion"] & (d1["role"] == "recipient"), "stop_id"])
    _, st1 = optimise.run_scenario(alt.copy(), config.SCENARIOS[config.RECOMMENDED_SCENARIO])

    rows = [
        ("donors", (orig["role"] == "donor").sum(), (alt["role"] == "donor").sum(),
         jaccard(orig.loc[orig.role == "donor", "stop_id"], alt.loc[alt.role == "donor", "stop_id"])),
        ("recipients", (orig["role"] == "recipient").sum(), (alt["role"] == "recipient").sum(),
         jaccard(orig.loc[orig.role == "recipient", "stop_id"], alt.loc[alt.role == "recipient", "stop_id"])),
        ("double-excluded", int(d0["double_exclusion"].sum()), int(d1["double_exclusion"].sum()),
         jaccard(d0.loc[d0.double_exclusion, "stop_id"], d1.loc[d1.double_exclusion, "stop_id"])),
        ("priority set", len(pri0), len(pri1), jaccard(pri0, pri1)),
        ("Moderate Gini gain %", round(st0["gini_improvement_%"], 2), round(st1["gini_improvement_%"], 2), None),
        ("Moderate burden cut %", round(st0["burden_reduction_%"], 2), round(st1["burden_reduction_%"], 2), None),
    ]
    comp = pd.DataFrame(rows, columns=["measure", "published (min-max)", "natural zero", "Jaccard"])
    print("\n" + comp.to_string(index=False))
    comp.to_csv(os.path.join(OUT, "robustness_01_index_comparison.csv"), index=False)
    pd.concat({"min-max": t0, "natural zero": t1}).to_csv(os.path.join(OUT, "robustness_01_tertiles.csv"))
    print("\n   Reading: if the Gini gain, tertile gradient and priority set move little, the zero-point "
          "objection is answered.\n   If they move, report the natural-zero figures beside the published ones.")

def _download(polygon, ox):
    parts = []
    for cat, tags in config.ESSENTIAL_SERVICE_TAGS.items():
        f = ox.features.features_from_polygon if hasattr(ox, "features") else ox.features_from_polygon
        g = f(polygon, tags=tags)
        g = g[g.geometry.notna() & g.geometry.geom_type.isin(("Point", "Polygon", "MultiPolygon"))].copy()
        g["service_type"] = cat
        parts.append(g[["geometry", "service_type"]])
        print(f"      {cat:12s}: {len(g)} features")
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry",
                            crs="EPSG:4326").to_crs(config.CRS_METRIC)

def buffered_osm(buffer_m=1000):
    header(f"A. PHASE 4 WITH A {buffer_m} m BUFFERED OSM EXTRACT")
    import osmnx as ox
    p4 = importlib.import_module("phase4_essential_services")
    place = config.OSM_PLACE_NAME                                # the same place string as Phase 4
    boundary = ox.geocode_to_gdf(place).to_crs(config.CRS_METRIC).geometry.iloc[0]
    buffered = gpd.GeoSeries([boundary.buffer(buffer_m)], crs=config.CRS_METRIC).to_crs(4326).iloc[0]
    print("   downloading essential services for Milan + buffer (one download, clipped below)...")
    svc_all = _download(buffered, ox)
    svc_mun = svc_all[svc_all.intersects(boundary)]
    print(f"   services: {len(svc_mun)} inside Milan, {len(svc_all)} with the buffer "
          f"(+{len(svc_all) - len(svc_mun)})")

    stops0, _ = p4.load_stop_state()
    edge = stops0.geometry.distance(boundary.boundary) <= buffer_m
    cx, cy = boundary.centroid.x, boundary.centroid.y
    quad = np.where(stops0.geometry.x >= cx, "E", "W")
    quad = pd.Series(np.char.add(np.where(stops0.geometry.y >= cy, "N", "S"), quad), index=stops0.index)
    print(f"   stops within {buffer_m} m of the municipal boundary: {int(edge.sum())} of {len(stops0)}")

    published = gpd.read_file(os.path.join(OUT, "phase4_essential_access.gpkg"))
    published["stop_id"] = sid(published["stop_id"])
    pub_de = set(published.loc[published["double_exclusion"].astype(str).str.lower().isin(["true", "1"]), "stop_id"])

    results, keep = {}, stops0[["stop_id"]].copy()
    for label, svc in (("municipal clip (today's OSM)", svc_mun), (f"buffered {buffer_m} m", svc_all)):
        s = p4.attach_nearest_distances(stops0.copy(), svc)
        s = p4.compute_access(s)
        s = p4.classify_double_exclusion(s)
        s["stop_id"] = sid(s["stop_id"])
        acc = s[["stop_id", "essential_access"]].rename(columns={"essential_access": "access_ok"})
        stop = s[["stop_id", "v_index", "hw_hol", "burden_hol"]].copy()
        d, tab = decomposition(stop, acc, label)
        de = s["double_exclusion"].values
        pri = int((s["double_exclusion"] & (s["role"] == "recipient")).sum())
        results[label] = {
            "access_failure_%": round(100 * (~s["essential_access"]).mean(), 1),
            "double_excluded": int(de.sum()),
            "priority_(also recipients)": pri,
            "edge_stops_failing_access_%": round(100 * (~s.loc[edge.values, "essential_access"]).mean(), 1),
            "double_excluded_at_edge": int(de[edge.values].sum()),
            "double_excluded_S_or_E_%": round(100 * pd.Series(de).groupby(quad.values).sum()
                                              .reindex(["NE", "SE", "SW"]).sum() / max(de.sum(), 1), 1),
        }
        keep[f"de::{label}"] = de
    summ = pd.DataFrame(results)
    print("\n" + summ.to_string())
    lab_m, lab_b = list(results)
    only_mun = set(keep.loc[keep[f"de::{lab_m}"], "stop_id"]) - set(keep.loc[keep[f"de::{lab_b}"], "stop_id"])
    print(f"\n   double-excluded only because services beyond the boundary were missing: {len(only_mun)}")
    print(f"   overlap with the published 415 (Jaccard): municipal {jaccard(pub_de, keep.loc[keep[f'de::{lab_m}'], 'stop_id']):.3f}, "
          f"buffered {jaccard(pub_de, keep.loc[keep[f'de::{lab_b}'], 'stop_id']):.3f}")
    summ.to_csv(os.path.join(OUT, "robustness_09_buffer_summary.csv"))
    keep.to_csv(os.path.join(OUT, "robustness_09_buffer_stops.csv"), index=False)
    print("   Reading: the buffer effect is the change from the municipal clip to the buffered run "
          "(same download, so OSM edits since the thesis run cancel out).")

def gtfs():
    if _G:
        return _G
    feed = config.GTFS_FEEDS[config.PRIMARY_FEED]
    cal, ids, n_days = bst.reconstruct_calendar(feed)
    trips = bst._read_gtfs(feed, "trips.txt", usecols=["trip_id", "service_id", "route_id"])
    routes = bst._read_gtfs(feed, "routes.txt", usecols=["route_id", "route_short_name", "route_type"])
    st = bst._read_gtfs(feed, "stop_times.txt", usecols=["trip_id", "arrival_time", "stop_id"])
    st["stop_id"] = sid(st["stop_id"])
    st["hour"] = bst._hour_from_arrival(st["arrival_time"])
    st = st[config.operational_hour_filter(st["hour"])].copy()
    st["hour"] = st["hour"].astype(int)
    core = set(config.CORE_HOLIDAY_DATES)
    per_sid = cal[cal["date"].isin(core)].groupby("service_id")["date"].nunique()
    hol = (st.merge(trips[trips["service_id"].isin(ids["hol"])], on="trip_id")
             .merge(routes, on="route_id"))
    hol["w"] = hol["service_id"].map(per_sid).fillna(0)
    _G.update(feed=feed, cal=cal, ids=ids, n_days=n_days, trips=trips, routes=routes, st=st, hol=hol)
    return _G

def is_excluded_line(r):
    r = str(r).strip()
    return r in config.NON_URBAN_ROUTE_EXACT or any(r.startswith(p) for p in config.NON_URBAN_ROUTE_PREFIXES)

def metro_passenger_cost():
    header("C. PASSENGER-WEIGHTED COST AT METRO DONOR STATIONS (MODERATE SCENARIO)")
    dv = importlib.import_module("demand_validation")
    rate = config.SCENARIOS[config.RECOMMENDED_SCENARIO]
    stop = optimise.classify(base_stops())
    metro = set(sid(pd.Series(sorted(dv.metro_stop_ids()))))
    donors = stop[(stop["role"] == "donor") & stop["stop_id"].isin(metro)].copy()
    print(f"   metro stations among the {int((stop.role == 'donor').sum())} donors: {len(donors)}")

    g = gtfs()
    h = g["hol"][g["hol"]["stop_id"].isin(donors["stop_id"])]
    dep = (h.groupby(["stop_id", "route_short_name"])["w"].sum() / g["n_days"]["hol"]).rename("dep_per_day").reset_index()
    dep["hw_dir_min"] = config.OPERATIONAL_MINUTES / (dep["dep_per_day"] / 2.0)     # both directions pooled at a station
    dep["added_wait_min"] = dep["hw_dir_min"] * (1 / (1 - rate) - 1) / 2.0          # random arrivals: wait = headway / 2
    per_station = (dep.assign(wx=dep["added_wait_min"] * dep["dep_per_day"])
                      .groupby("stop_id").agg(lines=("route_short_name", lambda x: ", ".join(sorted(set(x)))),
                                              wx=("wx", "sum"), dep=("dep_per_day", "sum")))
    per_station["added_wait_dir_min"] = per_station["wx"] / per_station["dep"]
    per_station = per_station.drop(columns=["wx", "dep"]).reset_index()

    donors = donors.merge(per_station, on="stop_id", how="left")
    donors["added_wait_pooled_min"] = donors["hw_hol"] * (1 / (1 - rate) - 1) / 2.0  # the understated version

    names = gpd.read_file(os.path.join(OUT, "milan_stops.gpkg"))[["stop_id", "stop_name"]]
    names["stop_id"] = sid(names["stop_id"])
    donors = donors.merge(names, on="stop_id", how="left")
    donors["station_key"] = donors["stop_name"].map(dv.normalise)
    dvs = pd.read_csv(os.path.join(OUT, "demand_vs_supply_station.csv"))
    hol_cols = [c for c in dvs.columns if c.startswith("hol::") and "Labour" not in c]
    dvs["entries_hol"] = dvs[hol_cols].mean(axis=1)
    donors = donors.merge(dvs[["matched_key", "entries_hol"]].dropna().drop_duplicates("matched_key"),
                          left_on="station_key", right_on="matched_key", how="left")
    matched = donors["entries_hol"].notna()
    donors["pax_min_added"] = donors["entries_hol"] * donors["added_wait_dir_min"]

    cols = ["stop_name", "lines", "hw_hol", "added_wait_pooled_min", "added_wait_dir_min", "entries_hol", "pax_min_added"]
    print("\n" + donors.loc[matched, cols].sort_values("pax_min_added", ascending=False).round(2).to_string(index=False))
    tot = donors.loc[matched, "pax_min_added"].sum()
    ent = donors.loc[matched, "entries_hol"].sum()
    print(f"\n   matched metro donor stations: {int(matched.sum())} of {len(donors)}")
    print(f"   added waiting: {tot:,.0f} passenger-minutes per holiday day "
          f"({tot / 60:,.0f} passenger-hours) over {ent:,.0f} gate entries")
    print(f"   mean added wait per entering passenger: {tot / ent:.2f} min "
          f"(pooled-headway version: {(donors.loc[matched, 'added_wait_pooled_min'] * donors.loc[matched, 'entries_hol']).sum() / ent:.2f} min)")
    print("   Caveats: gate entries miss interchange boardings (cost is understated at interchanges); this is the\n"
          "   cost side only; recipient benefit cannot be passenger-weighted without surface ridership; under the\n"
          "   operator rules (bus and tram only) metro keeps its service and this cost is zero.")
    donors.to_csv(os.path.join(OUT, "robustness_10_metro_passenger_cost.csv"), index=False)

def line_172():
    header("D. EXCLUDED LINES (172, 174, N*) IN THE REGULAR-NETWORK HEADWAYS")
    p1 = importlib.import_module("phase1_corridors")
    g = gtfs()
    hol = g["hol"].copy()
    hol["excluded_line"] = hol["route_short_name"].map(is_excluded_line)

    def stop_hw(frame):
        ops = frame.groupby(["stop_id", "hour"])["w"].sum() / g["n_days"]["hol"]
        return (60.0 / ops).groupby("stop_id").median()

    hw_all = stop_hw(hol)
    hw_urban = stop_hw(hol[~hol["excluded_line"]])
    share = (hol.loc[hol["excluded_line"]].groupby("stop_id")["w"].sum()
             .div(hol.groupby("stop_id")["w"].sum(), fill_value=0).rename("excluded_share"))

    stop, _ = p1.load_stop_state()
    s = p1.flag_regular_urban(p1.attach_metadata(stop))
    s["stop_id"] = sid(s["stop_id"])
    reg = s[s["regular_urban"]].copy()
    reg["hw_hol_all_lines"] = reg["stop_id"].map(hw_all)
    reg["hw_hol_urban_lines"] = reg["stop_id"].map(hw_urban)
    reg["excluded_share"] = reg["stop_id"].map(share).fillna(0)
    reg["burden_urban"] = reg["v_index"] * reg["hw_hol_urban_lines"]

    top_pub = reg.nlargest(10, "burden_hol")
    top_urban = reg.nlargest(10, "burden_urban")
    cols = ["stop_name", "lines", "v_index", "hw_hol", "hw_hol_urban_lines", "excluded_share", "burden_hol", "burden_urban"]
    print("\n   published top 10 (regular network), with urban-only headways:")
    print(top_pub[cols].round(3).to_string(index=False))
    kept = len(set(top_pub["stop_id"]) & set(top_urban["stop_id"]))
    print(f"\n   top 10 recomputed with urban lines only: {kept} of 10 stops unchanged")
    print(top_urban[cols].round(3).to_string(index=False))
    affected = reg[(reg["excluded_share"] > 0)]
    print(f"\n   regular-network stops with any holiday departure on an excluded line: {len(affected)} of {len(reg)}")
    reg[["stop_id", "stop_name", "lines", "hw_hol", "hw_hol_urban_lines", "excluded_share"]].to_csv(
        os.path.join(OUT, "robustness_02_excluded_lines.csv"), index=False)

def labour_day():
    header("E. LABOUR DAY: WHAT THE SPRING FEED LISTS")
    g = gtfs()
    cal, trips, routes, st = g["cal"], g["trips"], g["routes"], g["st"]
    print(f"   feed window: {cal['date_dt'].min().date()} -> {cal['date_dt'].max().date()}")
    rows = []
    for d, lab in (("20260424", "Fri 24 Apr"), ("20260425", "Sat 25 Apr (holiday)"), ("20260426", "Sun 26 Apr"),
                   ("20260430", "Thu 30 Apr"), ("20260501", "Fri 1 May (holiday)"), ("20260502", "Sat 2 May")):
        sids = set(cal.loc[cal["date"] == d, "service_id"])
        t = trips[trips["service_id"].isin(sids)].merge(routes, on="route_id")
        served = st[st["trip_id"].isin(set(t["trip_id"]))]["stop_id"].nunique()
        rows.append({"date": lab, "service_ids": len(sids), "trips": len(t), "routes": t["route_id"].nunique(),
                     "metro_trips": int((t["route_type"].astype(str) == "1").sum()),
                     "stops_served_06_22": served,
                     "example_service_ids": ", ".join(sorted(sids)[:4])})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    tab.to_csv(os.path.join(OUT, "robustness_03_labour_day.csv"), index=False)
    print("   Reading: compare 1 May with the Saturday/Sunday templates around it. The 1 May collapse is date-specific; 2 May is complete, so this is not an end-of-feed truncation.")

def knee():
    header("F. THE KNEE RULE (sweep.py) AND ITS SENSITIVITY")
    for name in ("sweep_curve.csv", "sweep_curve_cap_rule.csv"):
        p = os.path.join(OUT, name)
        if not os.path.exists(p):
            continue
        c = pd.read_csv(p).sort_values("rate")
        g, r = c["gini_improvement_%"].values, c["rate"].values
        m = np.gradient(g, r)
        early = m[:5].mean()
        out = []
        for frac in (0.20, 0.25, 0.33, 0.50):
            hit = m < frac * early
            k = int(np.argmax(hit)) if hit.any() else len(r) - 1
            out.append(f"{frac:.0%} -> {r[k]:.0%}")
        i20 = int(np.argmin(np.abs(r - 0.20)))
        print(f"   {name}: knee (marginal slope below X of early slope): " + " | ".join(out))
        print(f"      at 20%: marginal gain {m[i20] / 100:.3f} Gini-pp per rate-pp, "
              f"{m[i20] / early:.0%} of the early slope")

def _first(cols, candidates):
    return next((c for c in candidates if c in cols), None)

def plateau():
    print(f"\n{LINE}\n1. PLATEAU OF THE CEILING CURVE\n{LINE}")
    files = sorted(glob.glob(os.path.join(OUT, "sweep_curve*.csv")))
    if not files:
        print("   sweep_curve.csv not found: run sweep.py first"); return
    rows = []
    for f in files:
        s = pd.read_csv(f).sort_values("rate").reset_index(drop=True)
        g, r = s["gini_improvement_%"].values, s["rate"].values
        k = int(np.argmax(g)); gmax, rmax = g[k], r[k]
        mod = g[np.argmin(np.abs(r - config.SCENARIOS[config.RECOMMENDED_SCENARIO]))]
        print(f"\n   file: {os.path.basename(f)}")
        print(f"   peak: {gmax:.2f}% at {rmax:.0%} | Moderate: {mod:.2f}% = {mod / gmax:.1%} of the peak")
        for label, thr in (("within 0.1 pp", gmax - 0.1), ("within 99.5%", 0.995 * gmax), ("within 99%", 0.99 * gmax)):
            lo = k
            while lo > 0 and g[lo - 1] >= thr: lo -= 1
            hi = k
            while hi < len(g) - 1 and g[hi + 1] >= thr: hi += 1
            print(f"   rates {label} of the peak: {r[lo]:.0%} to {r[hi]:.0%}")
            rows.append({"file": os.path.basename(f), "band": label, "from": r[lo], "to": r[hi],
                         "peak_rate": rmax, "peak_gini": gmax, "moderate_share_of_peak": mod / gmax})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "robustness_06_ceiling_plateau.csv"), index=False)
    print("\n   Reading: the 91% (a ratio of Gini values) does not depend on where exactly the peak sits;")
    print("   '49% of the peak's donor reduction' does. Report the band instead of a single peak rate.")

def tertile_diagnostic():
    print(f"\n{LINE}\n2. WHERE THE DOUBLE-EXCLUSION GRADIENT COMES FROM\n{LINE}")
    stop = optimise.load_stop_level()               # 120-min penalty for unserved stops, as in the original run
    stop["stop_id"] = stop["stop_id"].astype(str).str.strip()
    # The full per-stop Phase 4 output is the GeoPackage; the CSV of the same name family
    # can be a short display extract. Use the first file that covers the whole network.
    cands = [os.path.join(OUT, "phase4_essential_access.gpkg"), os.path.join(OUT, "phase4_double_exclusion.csv")]
    p4, src = None, None
    for c in cands:
        if not os.path.exists(c):
            continue
        if c.endswith(".gpkg"):
            import geopandas as gpd
            df = pd.DataFrame(gpd.read_file(c).drop(columns="geometry", errors="ignore"))
        else:
            df = pd.read_csv(c, dtype={"stop_id": str})
        if "stop_id" in df.columns and len(df) >= 0.9 * len(stop):
            p4, src = df, c; break
        print(f"   skipped {os.path.basename(c)}: {len(df)} rows, not the full network of {len(stop)} stops")
    if p4 is None:
        print("   no Phase 4 file covers the whole network. Phase 4 files present:",
              sorted(os.path.basename(f) for f in glob.glob(os.path.join(OUT, "phase4_*")))); return
    print(f"   Phase 4 source: {os.path.basename(src)} ({len(p4)} stops)")
    p4["stop_id"] = p4["stop_id"].astype(str).str.strip()
    cols = list(p4.columns)

    # --- access, rebuilt from Phase 4's own columns --------------------------------------
    # The file stores distances and a count of categories within their catchment, not a
    # pass/fail flag. Access = all three categories within their catchments.
    thr = None
    for name in dir(config):
        v = getattr(config, name)
        if "CATCH" in name.upper() and isinstance(v, dict) and len(v) >= 3:
            thr, thr_src = v, f"config.{name}"; break
    if thr is None:
        thr, thr_src = {"hospital": 800, "pharmacy": 400, "supermarket": 400}, "thesis defaults"
    print(f"   catchments used: {thr} ({thr_src})")
    def within(cat):
        key = next((k for k in thr if cat in k.lower()), None)
        return p4[f"dist_{cat}"] <= thr[key] if key is not None and f"dist_{cat}" in cols else None
    parts = [within(c) for c in ("hospital", "pharmacy", "supermarket")]
    access_d = parts[0] & parts[1] & parts[2] if all(x is not None for x in parts) else None
    access_n = p4["n_services_within"] >= 3 if "n_services_within" in cols else None
    flag = _first(cols, ["essential_access", "access_ok", "meets_access", "essential_ok", "access_all"])
    access_f = p4[flag].astype(str).str.lower().isin(["true", "1", "yes"]) if flag else None
    routes = {k: v for k, v in (("flag", access_f), ("count", access_n), ("distances", access_d)) if v is not None}
    if not routes:
        print("   cannot rebuild the access test. Columns in the Phase 4 file:\n  ", cols); return
    names = list(routes)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            print(f"   access from {names[a]} vs {names[b]}: {(routes[names[a]] == routes[names[b]]).mean():.1%} agreement")
    p4["_access"] = routes[names[0]]
    print(f"   stops meeting all three catchments: {int(p4['_access'].sum())} of {len(p4)} "
          f"(thesis: 1,819 of 3,187)")

    # Phase 4's own v_index, headway and burden when present; the stop table otherwise
    base = p4 if {"v_index", "hw_hol", "burden_hol"} <= set(cols) else \
        stop[["stop_id", "v_index", "hw_hol", "burden_hol"]].merge(p4[["stop_id", "_access"]], on="stop_id")
    m = base.copy()
    print(f"   stops analysed: {len(m)} ({'Phase 4 values' if base is p4 else 'stop table merged with Phase 4'})")
    fail = ~m["_access"].astype(bool)
    cls = _first(cols, ["exclusion_class", "excl_class", "exclusion", "class", "category"])
    dbl = _first(cols, ["double_exclusion", "is_double_excluded", "double_excluded"])
    d_obs = (m[dbl].astype(str).str.lower().isin(["true", "1", "yes"]) if dbl else
             m[cls].astype(str).str.lower().str.contains("double") if cls else None)

    # --- flags ------------------------------------------------------------------------
    q_b = m["burden_hol"].quantile(0.75)
    q_h = m["hw_hol"].quantile(0.75)
    top_b = m["burden_hol"] >= q_b                  # the original definition (burden = v x headway)
    top_h = m["hw_hol"] >= q_h                      # headway only: vulnerability removed
    d_rec = top_b & fail
    print(f"   burden top-quartile cut: {q_b:.2f} (thesis: 5.89) | headway top-quartile cut: {q_h:.1f} min")
    print(f"   double-excluded stops rebuilt: {int(d_rec.sum())} (thesis: 415)")
    if d_obs is not None:
        agree = (d_rec == d_obs).mean()
        print(f"   rebuilt double exclusion agrees with the Phase 4 file for {agree:.1%} of stops "
              f"({int(d_rec.sum())} vs {int(d_obs.sum())} flagged; thesis: 415)")

    # --- tertiles, and the reproduction check ---------------------------------------------
    m["tertile"] = pd.qcut(m["v_index"], 3, labels=["Low", "Middle", "High"])
    rep = (d_rec.groupby(m["tertile"], observed=True).mean() * 100).round(1)
    ok = all(abs(rep[t] - PUBLISHED_TERTILES[t]) <= 0.15 for t in PUBLISHED_TERTILES)
    print(f"   reproduction of the published incidence: {rep.to_dict()} vs {PUBLISHED_TERTILES} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print("   The tertile or burden definition differs from Phase 4; not interpreting further.")
        print("   Check how Phase 4 cuts the tertiles (by stop or by census section) and adjust pd.qcut.")
        return

    rows = []
    for t in ["Low", "Middle", "High"]:
        s = m["tertile"] == t
        a, b, f = top_h[s].mean(), top_b[s].mean(), fail[s].mean()
        d, e = (top_b & fail)[s].mean(), (top_h & fail)[s].mean()
        rows.append({"tertile": t, "stops": int(s.sum()),
                     "headway_top_quartile_%": 100 * a, "burden_top_quartile_%": 100 * b,
                     "access_failure_%": 100 * f,
                     "double_exclusion_%": 100 * d, "expected_if_independent_%": 100 * b * f,
                     "lift_vs_independence": d / (b * f) if b * f else np.nan,
                     "headway_based_double_%": 100 * e, "headway_based_expected_%": 100 * a * f,
                     "headway_based_lift": e / (a * f) if a * f else np.nan})
    t = pd.DataFrame(rows).set_index("tertile")
    pd.set_option("display.width", 160)
    print("\n" + t.round(2).to_string())
    ratio = lambda c: t.loc["High", c] / t.loc["Low", c] if t.loc["Low", c] else np.nan
    print("\n   High / Low contrast:")
    for c, label in (("headway_top_quartile_%", "headway alone (no vulnerability weight)"),
                     ("burden_top_quartile_%", "burden (v x headway)"),
                     ("access_failure_%", "access failure alone"),
                     ("double_exclusion_%", "double exclusion, as published"),
                     ("headway_based_double_%", "double exclusion using headway instead of burden")):
        v = ratio(c)
        print(f"      {label:52s} " + (f"{v:5.2f}x" if np.isfinite(v) else "  n/a (zero in the low tertile)"))
    t.to_csv(os.path.join(OUT, "robustness_04_gradient_decomposition.csv"))
    print("\n   Reading:")
    print("   - burden contrast >> headway contrast: part of the gradient comes from the vulnerability weight")
    print("     inside burden (mechanical); the headway-based row is the gradient without it.")
    print("   - a steep access-failure contrast on its own: check it with the buffered OSM run, since the")
    print("     municipal boundary can only hide facilities near the edge, never add them.")
    print("   - lift > 1: burden and access failure coincide more often than chance within the tertile.")

def _load():
    """Phase 4 per-stop table with v_index, hw_hol, burden_hol and access flag (as in robustness_checks)."""
    import geopandas as gpd
    p4 = pd.DataFrame(gpd.read_file(os.path.join(OUT, "phase4_essential_access.gpkg")).drop(columns="geometry"))
    p4["stop_id"] = p4["stop_id"].astype(str).str.strip()
    thr = config.SERVICE_CATCHMENTS_M
    ok = np.ones(len(p4), bool)
    for cat in ("hospital", "pharmacy", "supermarket"):
        key = next(k for k in thr if cat in k.lower())
        ok &= (p4[f"dist_{cat}"] <= thr[key]).values
    p4["access_ok"] = ok
    if not {"v_index", "hw_hol", "burden_hol"} <= set(p4.columns):
        st = optimise.load_stop_level(); st["stop_id"] = st["stop_id"].astype(str).str.strip()
        p4 = st[["stop_id", "v_index", "hw_hol", "burden_hol"]].merge(p4[["stop_id", "access_ok"]], on="stop_id")
    p4["tertile"] = pd.qcut(p4["v_index"], 3, labels=["Low", "Middle", "High"])
    return p4

def tie_check():
    print(f"\n{LINE}\nA. DOES 'NEED-BLIND' SURVIVE THE 20-MINUTE TIES?\n{LINE}")
    m = _load(); fail = ~m["access_ok"]; t = m["tertile"]
    q = m["hw_hol"].quantile(.75)
    n25 = int(round(.25 * len(m)))
    print(f"   stops at exactly {q:.1f} min: {int((m['hw_hol'] == q).sum())}")

    def row(name, flag):
        a = flag.groupby(t, observed=True).mean() * 100
        d = (flag & fail).groupby(t, observed=True).mean() * 100
        return {"cut": name, "share_all_%": 100 * flag.mean(),
                **{f"{k}_%": a[k] for k in a.index}, "high/low": a["High"] / a["Low"],
                **{f"double_{k}_%": d[k] for k in d.index},
                "double_high/low": d["High"] / d["Low"] if d["Low"] else np.nan}
    rows = [row(f">= {q:.1f} (published run)", m["hw_hol"] >= q),
            row(f">  {q:.1f} (strict)", m["hw_hol"] > q),
            row("> 30 (service floor)", m["hw_hol"] > 30)]
    # exact 25% with deterministic random tie-breaking, 500 draws
    rng = np.random.default_rng(20260926)
    ratios, dratios = [], []
    for _ in range(500):
        key = m["hw_hol"].values + rng.uniform(0, 1e-6, len(m))
        flag = pd.Series(key >= np.sort(key)[-n25], index=m.index)
        r = row("", flag); ratios.append(r["high/low"]); dratios.append(r["double_high/low"])
    rows.append({"cut": "exact 25%, random ties (500 draws, median)",
                 "share_all_%": 25.0, "high/low": np.median(ratios),
                 "double_high/low": np.median(dratios)})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))
    print(f"   exact-25% draws: high/low headway ratio 5-95% = "
          f"{np.percentile(ratios, 5):.2f}-{np.percentile(ratios, 95):.2f}; "
          f"double-exclusion ratio 5-95% = {np.percentile(dratios, 5):.2f}-{np.percentile(dratios, 95):.2f}")
    served = m[m["hw_hol"] < PEN]
    med = served.groupby("tertile", observed=True)["hw_hol"].agg(["median", "mean", "count"])
    print("\n   holiday headway among served stops, by tertile:\n" + med.round(2).to_string())
    unserved = (m["hw_hol"] >= PEN).groupby(t, observed=True).sum()
    print("   unserved (120-min penalty) stops by tertile:", unserved.to_dict())
    fa = fail.groupby(t, observed=True).mean() * 100
    print("   access failure by tertile (%):", fa.round(1).to_dict())
    out.to_csv(os.path.join(OUT, "robustness_05_headway_ties.csv"), index=False)
    print("\n   Reading: if every high/low ratio stays at or below ~1 and the served-stop medians do")
    print("   not rise with vulnerability, 'need-blind' holds and the thesis text stands. If the")
    print("   strict or exact cut gives a ratio clearly above 1, report it and soften §5.4.")

def _events():
    feed = config.GTFS_FEEDS[config.PRIMARY_FEED]
    trips = bst._read_gtfs(feed, "trips.txt")
    cols = ["trip_id", "route_id"] + (["direction_id"] if "direction_id" in trips.columns else [])
    routes = bst._read_gtfs(feed, "routes.txt", usecols=["route_id", "route_type", "route_short_name"])
    st = bst._read_gtfs(feed, "stop_times.txt", usecols=["trip_id", "stop_id"])
    ev = st.merge(trips[cols], on="trip_id").merge(routes, on="route_id")
    ev["stop_id"] = ev["stop_id"].astype(str).str.strip()
    ev["route_type"] = ev["route_type"].astype(str)
    if "direction_id" not in ev.columns:
        ev["direction_id"] = np.nan
    return ev

def direction_factors():
    """Per stop: all departures / departures in one direction per route.
    Routes with no direction_id (metro in this feed) are assumed to serve both directions
    symmetrically at a station, so their one-direction count is half."""
    ev = _events()
    ev["dir"] = ev["direction_id"].astype("string").fillna("NA")
    c = ev.groupby(["stop_id", "route_id", "route_type", "dir"]).size().rename("n").reset_index()
    per_route = c.groupby(["stop_id", "route_id", "route_type"]).agg(
        n_all=("n", "sum"), n_max=("n", "max"), dirs=("dir", lambda d: set(d)))
    per_route = per_route.reset_index()
    no_dir = per_route["dirs"].map(lambda s: s == {"NA"})
    per_route["n_one"] = np.where(no_dir & (per_route["route_type"] == "1"),
                                  per_route["n_all"] / 2, per_route["n_max"])
    per_route["both_same_route"] = per_route["dirs"].map(lambda s: len(s - {"NA"}) >= 2)
    per_route["metro_station"] = no_dir & (per_route["route_type"] == "1")
    f = per_route.groupby("stop_id").agg(n_all=("n_all", "sum"), n_one=("n_one", "sum"),
                                         both_same_route=("both_same_route", "any"),
                                         metro_station=("metro_station", "any"),
                                         route_types=("route_type", lambda x: ",".join(sorted(set(x)))))
    f["factor"] = f["n_all"] / f["n_one"]
    # naive cross-route flag, kept to show how much it overcounts
    cross = ev.dropna(subset=["direction_id"]).groupby("stop_id")["direction_id"].nunique() >= 2
    f["cross_route_flag"] = cross.reindex(f.index, fill_value=False).astype(bool)
    f["mode"] = f["route_types"].map(lambda s: "metro" if "1" in s.split(",") else
                                     {"0": "tram", "3": "bus", "2": "rail"}.get(s.split(",")[0], s))
    return f

def direction_check_v2():
    print(f"\n{LINE}\nB. BOTH DIRECTIONS UNDER ONE ID — PER ROUTE\n{LINE}")
    f = direction_factors()
    stop = optimise.classify(optimise.load_stop_level())
    stop["stop_id"] = stop["stop_id"].astype(str).str.strip()
    j = stop.merge(f, left_on="stop_id", right_index=True, how="left")
    j["pooled"] = j["both_same_route"].fillna(False) | j["metro_station"].fillna(False)
    art = j["cross_route_flag"].fillna(False) & ~j["both_same_route"].fillna(False)
    print(f"   naive flag (direction_id compared across routes): {int(j['cross_route_flag'].fillna(False).sum())} stops")
    print(f"   of which cross-route artefacts (no single route in both directions): {int(art.sum())}")
    print(f"   corrected: same route in both directions: {int(j['both_same_route'].fillna(False).sum())} stops; "
          f"metro stations: {int(j['metro_station'].fillna(False).sum())}")
    tab = j.groupby("role").agg(stops=("stop_id", "size"),
                                pooled=("pooled", "sum"),
                                same_route_both=("both_same_route", lambda s: int(s.fillna(False).sum())),
                                metro=("metro_station", lambda s: int(s.fillna(False).sum())),
                                median_factor=("factor", "median"))
    tab["pooled_%"] = (100 * tab["pooled"] / tab["stops"]).round(1)
    print("\n" + tab.to_string())
    print(f"\n   network: {int(j['pooled'].sum())} of {len(j)} analytical stops "
          f"({100 * j['pooled'].mean():.1f}%) have a pooled headway")
    j[["stop_id", "role", "mode", "route_types", "n_all", "n_one", "factor", "both_same_route",
       "metro_station", "cross_route_flag"]].to_csv(os.path.join(OUT, "robustness_08_direction_pooling.csv"), index=False)
    print("   -> reported in thesis §6.3.2")
    return f

def direction_sensitivity(f=None):
    """Compare the published and corrected one-direction sensitivity variants.

    The thesis-era sensitivity rescaled headways and burdens but retained the
    original pooled ``trips_hol`` values used by the optimiser.  That legacy
    implementation is preserved only to reproduce the published 18.8% result.

    The corrected variant also recomputes ``trips_hol`` from the adjusted
    holiday headway before allocation.  It is the internally consistent
    post-thesis diagnostic and does not replace the principal 20.8% result.
    """
    print(f"\n{LINE}\nC. SENSITIVITY: HEADWAYS RESCALED TO ONE DIRECTION PER ROUTE\n{LINE}")
    if f is None:
        f = direction_factors()

    base = optimise.load_stop_level()
    base["stop_id"] = base["stop_id"].astype(str).str.strip()
    adj = base.merge(f[["factor"]], left_on="stop_id", right_index=True, how="left")
    adj["factor"] = adj["factor"].fillna(1.0)

    hw_cols = [c for c in adj.columns
               if c.startswith("hw_") and pd.api.types.is_numeric_dtype(adj[c])]
    for c in hw_cols:
        served = adj[c] < PEN
        adj.loc[served, c] = np.minimum(
            adj.loc[served, c] * adj.loc[served, "factor"], PEN
        )
        burden_col = c.replace("hw_", "burden_")
        if burden_col in adj.columns:
            adj[burden_col] = adj["v_index"] * adj[c]

    legacy = adj.drop(columns="factor").copy()
    corrected = legacy.copy()
    corrected["trips_hol"] = config.OPERATIONAL_MINUTES / corrected["hw_hol"]

    legacy.to_csv(
        os.path.join(OUT, "robustness_08_stop_table_onedir_legacy.csv"), index=False
    )
    corrected.to_csv(
        os.path.join(OUT, "robustness_08_stop_table_onedir_corrected.csv"), index=False
    )
    # Backwards-compatible filename points to the corrected table.
    corrected.to_csv(
        os.path.join(OUT, "robustness_08_stop_table_onedir.csv"), index=False
    )
    print(f"   rescaled columns: {hw_cols}; burden recomputed in both variants")
    print("   published legacy: original pooled trips_hol retained")
    print("   corrected variant: trips_hol recomputed from adjusted holiday headway")

    c0 = optimise.classify(base.copy())
    c_legacy = optimise.classify(legacy.copy())
    c_corrected = optimise.classify(corrected.copy())

    role_rows = []
    for role in ("donor", "recipient"):
        s0 = set(c0.loc[c0["role"] == role, "stop_id"])
        s1 = set(c_corrected.loc[c_corrected["role"] == role, "stop_id"])
        jac = len(s0 & s1) / len(s0 | s1) if s0 | s1 else np.nan
        role_rows.append({
            "role": role,
            "published_n": len(s0),
            "one_direction_n": len(s1),
            "jaccard": jac,
        })
        print(f"   {role:9s}: {len(s0)} -> {len(s1)} stops, Jaccard {jac:.3f}")

    rate = config.SCENARIOS[config.RECOMMENDED_SCENARIO]
    variants = (
        ("published headways", "principal", c0,
         "original service equivalents"),
        ("one direction per route", "published_legacy", c_legacy,
         "original pooled trips_hol retained"),
        ("one direction per route", "corrected_consistent", c_corrected,
         "trips_hol recomputed from adjusted hw_hol"),
    )
    result_rows = []
    for headways, implementation, classified, service_equivalents in variants:
        for dynamic in (False, True):
            _, stats = optimise.run_scenario(
                classified.copy(), rate, dynamic_burden=dynamic
            )
            result_rows.append({
                "headways": headways,
                "implementation": implementation,
                "service_equivalents": service_equivalents,
                "dynamic_burden": dynamic,
                "donors": int((classified["role"] == "donor").sum()),
                "recipients": int((classified["role"] == "recipient").sum()),
                "gini_improvement_%": float(stats["gini_improvement_%"]),
                "burden_reduction_%": float(stats["burden_reduction_%"]),
                "freed_trips": float(stats["freed_trips"]),
                "unallocated_trips": float(stats["unallocated_trips"]),
            })

    results = pd.DataFrame(result_rows)
    results.to_csv(
        os.path.join(OUT, "robustness_08_direction_sensitivity.csv"), index=False
    )
    pd.DataFrame(role_rows).to_csv(
        os.path.join(OUT, "robustness_08_role_overlap.csv"), index=False
    )

    static = results[~results["dynamic_burden"]]
    legacy_row = static[static["implementation"] == "published_legacy"].iloc[0]
    corrected_row = static[static["implementation"] == "corrected_consistent"].iloc[0]
    print(
        f"   Published thesis-era sensitivity: "
        f"Gini improvement {legacy_row['gini_improvement_%']:.2f}% | "
        f"burden reduction {legacy_row['burden_reduction_%']:.2f}%"
    )
    print(
        f"   Corrected consistent sensitivity: "
        f"Gini improvement {corrected_row['gini_improvement_%']:.2f}% | "
        f"burden reduction {corrected_row['burden_reduction_%']:.2f}%"
    )
    print("   Thesis reference: 18.8% is reproduced by the published legacy implementation.")
    print("   The corrected value is reported separately and does not alter the principal pooled-headway result.")

CHECKS = {
    1: ("vulnerability natural zero", index_zero_point),
    2: ("excluded lines / line 172", line_172),
    3: ("Labour Day feed diagnosis", labour_day),
    4: ("double-exclusion gradient decomposition", tertile_diagnostic),
    5: ("20-minute headway ties", tie_check),
    6: ("ceiling plateau", plateau),
    7: ("knee sensitivity", knee),
    8: ("direction pooling and sensitivity", None),
    9: ("OSM boundary buffer", buffered_osm),
    10: ("supplementary metro passenger cost", metro_passenger_cost),
}

def _parse_checks(spec):
    if not spec:
        return list(CHECKS)
    out=[]
    for part in spec.split(","):
        part=part.strip()
        if "-" in part:
            a,b=map(int,part.split("-",1)); out.extend(range(a,b+1))
        else:
            out.append(int(part))
    return [n for n in out if n in CHECKS]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checks", help="comma/range selection, e.g. 1,3,6-8")
    ap.add_argument("--skip-osm", action="store_true")
    ap.add_argument("--skip-atm", action="store_true")
    ap.add_argument("--list", action="store_true")
    args=ap.parse_args()
    if args.list:
        for n,(label,_) in CHECKS.items(): print(f"{n:2d}  {label}")
        return
    config.validate()
    chosen=_parse_checks(args.checks)
    status=[]
    for n in chosen:
        label,fn=CHECKS[n]
        if (n==9 and args.skip_osm) or (n==10 and args.skip_atm):
            print(f"SKIP {n}: {label}"); status.append((n,label,"skipped")); continue
        try:
            header(f"CHECK {n}: {label}")
            if n==8:
                f=direction_check_v2(); direction_sensitivity(f)
            else:
                fn()
            status.append((n,label,"ok"))
        except Exception:
            print(traceback.format_exc()[-3000:])
            status.append((n,label,"failed"))
    pd.DataFrame(status,columns=["check","label","status"]).to_csv(
        os.path.join(OUT,"robustness_check_status.csv"),index=False)
    print(f"\nWritten: robustness_*.csv and robustness_check_status.csv in {OUT}")

if __name__ == "__main__":
    main()
