"""
================================================================================
phase4_essential_services.py  —  Phase 4: Essential Services & Double Exclusion
================================================================================

METHODOLOGY (thesis §5.4, double exclusion)
--------------------------------------------------------------------------------
The double-exclusion analysis crosses the temporal-equity finding (holiday
service deficit weighted by vulnerability) with a spatial-accessibility
measure (walking proximity to essential everyday services). It identifies
stops where vulnerable populations face BOTH a substantial holiday transit
penalty AND an essential-service desert: the spatial concentration of
mobility poverty in the sense of Lucas [LUCAS]. The analysis follows three
methodological commitments.

First, essential services are kept disaggregated by category rather than
aggregated into a single "any essential service" composite. The pitfall of
the composite approach — that a stop adjacent to a hospital can appear
equally well-served as a stop adjacent to a pharmacy — is named explicitly
by Mouratidis [MOURATIDIS] as a recommendation to avoid: facilities serving
different needs should not be conflated. The framework therefore computes
distance from each stop to its nearest pharmacy, nearest supermarket, and
nearest hospital independently, applying category-specific catchment radii
(config.SERVICE_CATCHMENTS_M: 400 m for pharmacy, 400 m for supermarket and
800 m for hospital in the thesis). The
category catchments are 15-Minute-City service-access thresholds and are
distinct from SDG 11.2.1's transit-access thresholds (500 m for low-
capacity and 1,000 m for high-capacity transit), which govern the
transit-access dimension elsewhere in this thesis. The two should not be
conflated.

Second, the multi-service composite is constraint-limited rather than
averaged. A stop has essential-service access only if it lies within every
category's catchment simultaneously (Olafsson et al. [OLAFSSON]; the
constraint-limited form of the X-Minute City). Operationally the access
indicator is binary, taking the value one only when all three category
indicators are one. A complementary diagnostic, the count of categories
within catchment, is reported for descriptive purposes but does not enter
the double-exclusion classification.

Third, the double-exclusion classification combines the two dimensions
through a categorical rather than a normalised composite. A stop is flagged
as double-excluded if its holiday social burden is in the top quartile of
the analytical network AND its essential-service access indicator is zero.
This avoids the methodological hazard of normalising distances against the
maximum observed in the dataset (which is sensitive to single outlier stops
and is not transferable between cities); instead, each dimension is
classified against an absolute, literature-grounded threshold (quartile
cut for burden, catchment-fail for access) and the two flags are
intersected.

The double-exclusion stops are then cross-referenced against the recipient
set from Phase 3. Stops that are double-excluded AND identified as
optimisation recipients form the priority set for local review: the
reallocation directs additional modelled service to them and the spatial analysis
identifies limited geographic proximity to the selected essential-service categories.
This intersection is reported in §5.4 and discussed as a screening set in Chapter 6.

Fourth, stops carrying no holiday service at any hour enter the burden
ranking at the disclosed no-service penalty of the methodology chapter, the
same convention the Phase 5 reallocation applies. Computing burden on served
stops alone would leave these stops with undefined burden, excluding from the
double-exclusion classification precisely those stops whose holiday service is
most deficient. The number of stops affected is recorded in the run parameters.

A geometric refinement applies to large polygonal service features. OSM
encodes hospital complexes and supermarket buildings as polygons spanning
city blocks; their geometric centroids can lie 100 to 300 metres inside
the building footprint and overestimate geometric proximity to a facility. The framework therefore
retains polygon geometries and measures distance to the nearest geometry boundary.
This is a geometric proximity measure, not distance to a verified pedestrian entrance.

[LUCAS]      K. Lucas, Transport Policy 20, 2012.
[MOURATIDIS] K. Mouratidis, 2024 (on 15-min-city disaggregation).
[OLAFSSON]   A. S. Olafsson et al., on constraint-limited X-minute access.

--------------------------------------------------------------------------------
INPUTS  (from the corrected pipeline + OpenStreetMap via osmnx)
    milan_stop_hourly.csv             v_index and headways
    milan_stops.gpkg                  stop geometry
    milan_optimised_<rec>.csv         optimisation roles (Phase 3)
    config.FROZEN_OSM_SERVICES        frozen OSM essential-service snapshot
                                      (data/osm_services_2026-07-05.gpkg);
                                      if absent, current OSM via osmnx, which
                                      may not reproduce the thesis counts

OUTPUTS
    phase4_essential_access.gpkg      per-stop with per-category access flags
    phase4_services.gpkg              the downloaded OSM service points
    phase4_double_exclusion.csv       ranked top-N double-excluded stops
    phase4_priority_table.csv         double-excluded AND recipient (priority)
    thesis_phase4_scatter.png         figure: burden vs. service-access scatter
    thesis_phase4_map.png             figure: spatial distribution of exclusion classes
    thesis_phase4_tertile.png         figure: exclusion-class composition by vulnerability tertile
    run_parameters_phase4.json

Usage:
    python phase4_essential_services.py
================================================================================
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import config

config.validate()


# -----------------------------------------------------------------------------
# 1. load stop state + Phase 5 roles
# -----------------------------------------------------------------------------
def load_stop_state():
    print("[1] Loading stop state (corrected) and Phase 5 roles...")
    df = pd.read_csv(os.path.join(config.OUTPUT_DIR, "milan_stop_hourly.csv"))
    df = df[config.operational_hour_filter(df["hour"])].copy()
    stop = df.groupby("stop_id").agg(
        v_index=("v_index", "mean"),
        hw_wd  =("hw_wd",   "median"),
        hw_hol =("hw_hol",  "median"),
    ).reset_index()
    stop["stop_id"] = stop["stop_id"].astype(str).str.strip()
    # Burden uses the disclosed no-service convention of §3.3.3, matching
    # optimise.py. A stop with no holiday service at any hour enters the burden
    # ranking at the penalty headway rather than being dropped from it, so
    # Phases 4 and 5 classify the same stops on the same basis. On a served-only
    # basis these stops carry undefined burden, fail the quartile test, and can
    # never be classified as double-excluded — the stops with the worst possible
    # holiday service excluded from the category built to identify them.
    n_unserved = int(stop["hw_hol"].isna().sum())
    stop["hw_hol"] = stop["hw_hol"].fillna(config.NO_SERVICE_PENALTY_MIN)
    stop["burden_hol"] = stop["v_index"] * stop["hw_hol"]
    print(f"    unserved holiday stops set to "
          f"{config.NO_SERVICE_PENALTY_MIN} min penalty: {n_unserved}")

    # Phase 5 roles (recipient set is the priority cross-reference)
    opt_name = f"milan_optimised_{config.RECOMMENDED_SCENARIO.lower()}.csv"
    opt_path = os.path.join(config.OUTPUT_DIR, opt_name)
    if os.path.exists(opt_path):
        opt = pd.read_csv(opt_path, usecols=["stop_id", "role"])
        opt["stop_id"] = opt["stop_id"].astype(str).str.strip()
        stop = stop.merge(opt, on="stop_id", how="left")
        print(f"    Phase 5 roles merged: "
              f"{stop['role'].value_counts(dropna=False).to_dict()}")
    else:
        print(f"    WARNING: {opt_name} not found — Phase 3 cross-reference disabled")
        stop["role"] = None

    # geometry
    g = gpd.read_file(os.path.join(config.OUTPUT_DIR, "milan_stops.gpkg"))
    g["stop_id"] = g["stop_id"].astype(str).str.strip()
    keep = ["stop_id", "geometry"]
    if "stop_name" in g.columns: keep.append("stop_name")
    stops_gdf = g[keep].merge(stop, on="stop_id", how="inner")
    stops_gdf = gpd.GeoDataFrame(stops_gdf, crs=config.CRS_GEOGRAPHIC).to_crs(
        config.CRS_METRIC)
    print(f"    {len(stops_gdf)} stops with geometry, projected to {config.CRS_METRIC}")
    return stops_gdf, n_unserved


# -----------------------------------------------------------------------------
# 2. download essential services from OSM
# -----------------------------------------------------------------------------
def download_services():
    frozen = getattr(config, "FROZEN_OSM_SERVICES", None)
    if frozen and os.path.exists(frozen):
        print(f"[2] Loading frozen OSM service snapshot: {os.path.basename(frozen)}")
        try:
            import fiona
            layers = fiona.listlayers(frozen)
            parts = [gpd.read_file(frozen, layer=layer) for layer in layers]
            svc = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        except Exception:
            svc = gpd.read_file(frozen)
        svc = gpd.GeoDataFrame(svc, geometry="geometry", crs=getattr(svc, "crs", None) or "EPSG:4326")
        if "service_type" not in svc.columns and "category" in svc.columns:
            svc = svc.rename(columns={"category": "service_type"})
        if "service_type" not in svc.columns:
            raise ValueError("Frozen OSM layer must contain a service_type/category field")
        return svc.to_crs(config.CRS_METRIC)

    print("[2] Frozen OSM snapshot not found; downloading current OSM data.")
    print("    WARNING: current OSM may not reproduce the thesis count of 415 double-excluded stops.")
    try:
        import osmnx as ox
    except ImportError:
        raise SystemExit("osmnx is required. Install with: pip install osmnx")

    ox.settings.log_console = False
    place = config.OSM_PLACE_NAME

    services = []
    for category, tags in config.ESSENTIAL_SERVICE_TAGS.items():
        try:
            # osmnx 2.x uses features.features_from_place; older uses features_from_place
            if hasattr(ox, "features"):
                gdf = ox.features.features_from_place(place, tags=tags)
            else:
                gdf = ox.features_from_place(place, tags=tags)
            gdf = gdf[gdf.geometry.notna()].copy()
            # Retain point/polygon geometries so nearest distance is measured to the
            # feature geometry rather than a polygon centroid. This is geometric
            # proximity, not a verified entrance-distance measure.
            gdf = gdf[gdf.geometry.geom_type.isin(
                ("Point", "Polygon", "MultiPolygon"))].copy()
            gdf["service_type"] = category
            services.append(gdf[["geometry", "service_type"]])
            n_pts = int((gdf.geometry.geom_type == "Point").sum())
            n_poly = int(gdf.geometry.geom_type.isin(("Polygon","MultiPolygon")).sum())
            print(f"    {category:12s}: {len(gdf)} features  "
                  f"({n_pts} points, {n_poly} polygons; "
                  f"polygon distance = distance-to-boundary)")
        except Exception as e:
            print(f"    {category}: FAILED ({e})")
    if not services:
        raise RuntimeError("No essential services downloaded — "
                           "check internet connection.")

    svc = pd.concat(services, ignore_index=True)
    svc = gpd.GeoDataFrame(svc, geometry="geometry", crs="EPSG:4326").to_crs(
        config.CRS_METRIC)
    print(f"    Total: {len(svc)} essential service points "
          f"(projected to {config.CRS_METRIC})")
    return svc


# -----------------------------------------------------------------------------
# 3. nearest-distance per service category (sjoin_nearest, vectorised)
# -----------------------------------------------------------------------------
def attach_nearest_distances(stops, services):
    print("[3] Computing nearest distance per service category "
          "(vectorised sjoin_nearest)...")
    for category in config.ESSENTIAL_SERVICE_TAGS:
        svc = services[services["service_type"] == category].copy()
        if len(svc) == 0:
            stops[f"dist_{category}"] = np.nan
            continue
        # sjoin_nearest on metric CRS: distance is in metres
        joined = gpd.sjoin_nearest(stops[["stop_id", "geometry"]],
                                   svc[["geometry"]], how="left",
                                   distance_col=f"dist_{category}")
        # one row per stop (sjoin_nearest can produce ties — keep min)
        per_stop = joined.groupby("stop_id")[f"dist_{category}"].min().reset_index()
        stops = stops.merge(per_stop, on="stop_id", how="left")

    print("    distance summaries (median, in metres):")
    for category in config.ESSENTIAL_SERVICE_TAGS:
        col = f"dist_{category}"
        if col in stops.columns:
            print(f"      {category:12s}: median={stops[col].median():.0f} m, "
                  f"max={stops[col].max():.0f} m")
    return stops


# -----------------------------------------------------------------------------
# 4. per-category catchment flags + constraint-limited access
# -----------------------------------------------------------------------------
def compute_access(stops):
    print("[4] Computing per-category access + constraint-limited composite...")
    n_within = pd.Series(0, index=stops.index)
    for category, threshold_m in config.SERVICE_CATCHMENTS_M.items():
        col_d = f"dist_{category}"
        col_f = f"near_{category}"
        stops[col_f] = (stops[col_d] <= threshold_m).fillna(False)
        n_within += stops[col_f].astype(int)
        n_yes = int(stops[col_f].sum())
        print(f"    near_{category:12s} ({threshold_m} m): "
              f"{n_yes}/{len(stops)} stops ({n_yes/len(stops)*100:.1f}%)")
    stops["n_services_within"] = n_within
    # constraint-limited binary access: all three categories satisfied
    stops["essential_access"] = (
        n_within == len(config.SERVICE_CATCHMENTS_M)
    )
    n_full = int(stops["essential_access"].sum())
    print(f"    essential_access (ALL three categories): "
          f"{n_full}/{len(stops)} stops ({n_full/len(stops)*100:.1f}%)")
    return stops


# -----------------------------------------------------------------------------
# 5. double-exclusion classification
# -----------------------------------------------------------------------------
def classify_double_exclusion(stops):
    print("[5] Classifying double exclusion...")
    burden_q = stops["burden_hol"].quantile(config.DOUBLE_EXCLUSION_BURDEN_Q)
    high_burden = stops["burden_hol"] >= burden_q
    fails_access = ~stops["essential_access"].astype(bool)

    stops["high_burden"] = high_burden
    stops["fails_access"] = fails_access
    stops["double_exclusion"] = high_burden & fails_access

    # four-class label for QGIS styling
    def label_row(hb, fa):
        if hb and fa:  return "double_exclusion"
        if hb:         return "high_burden_only"
        if fa:         return "isolated_only"
        return "standard"
    stops["exclusion_class"] = [
        label_row(hb, fa) for hb, fa in zip(high_burden, fails_access)
    ]
    counts = stops["exclusion_class"].value_counts().to_dict()
    print(f"    burden quartile cut (Q={config.DOUBLE_EXCLUSION_BURDEN_Q}): "
          f"burden_hol >= {burden_q:.2f}")
    for k in ["double_exclusion", "high_burden_only",
              "isolated_only", "standard"]:
        v = counts.get(k, 0)
        print(f"      {k:22s} {v:5d} ({v/len(stops)*100:.1f}%)")
    return stops


# -----------------------------------------------------------------------------
# 6. priority list: double-excluded AND a Phase 5 recipient
# -----------------------------------------------------------------------------
def priority_table(stops, top_n=20):
    print("[6] Building priority table (double-excluded AND recipient)...")
    if "role" not in stops.columns or stops["role"].isna().all():
        print("    Phase 5 roles unavailable — skipping priority cross-reference")
        return pd.DataFrame(), 0
    pri = stops[(stops["double_exclusion"]) & (stops["role"] == "recipient")].copy()
    pri = pri.sort_values("burden_hol", ascending=False)
    # Flag rows whose hw_hol is the no-service penalty rather than measured
    # service, so the exported column cannot be read as an observed headway.
    pri["hw_is_penalty"] = (pri["hw_hol"].round(1)
                            == float(config.NO_SERVICE_PENALTY_MIN))
    n_total = len(pri)
    print(f"    Priority stops (double-excluded AND recipient): {n_total} "
          f"(full set; CSV exports all, console shows top {top_n})")
    cols_keep = ["stop_id", "v_index", "hw_hol", "hw_is_penalty", "burden_hol"]
    if "stop_name" in pri.columns: cols_keep.insert(1, "stop_name")
    cols_keep += [f"dist_{c}" for c in config.ESSENTIAL_SERVICE_TAGS]
    cols_keep += ["n_services_within"]
    # ranked full table -> export the lot; head() only for console display
    full = pri[cols_keep].copy()
    for c in [f"dist_{c}" for c in config.ESSENTIAL_SERVICE_TAGS]:
        if c in full: full[c] = full[c].round(0).astype("Int64")
    for c in ("v_index", "burden_hol"):
        if c in full: full[c] = full[c].round(3)
    full["hw_hol"] = full["hw_hol"].round(1)
    return full, n_total


# -----------------------------------------------------------------------------
# 7. figures (split into two separate exports)
# -----------------------------------------------------------------------------
def _palette_and_labels():
    palette = {
        "standard":          "#94A3B8",
        "high_burden_only":  "#F59E0B",
        "isolated_only":     "#818CF8",
        "double_exclusion":  "#DC2626",
    }
    labels = {
        "standard":         "Standard (low burden, near services)",
        "high_burden_only": "High holiday burden only",
        "isolated_only":    "Essential-access failure only",
        "double_exclusion": "Double exclusion",
    }
    return palette, labels


def make_figure_scatter(stops):
    """Figure A: scatter of holiday burden vs. number of service categories
    within catchment, coloured by exclusion class."""
    print("[7a] Building burden-vs-access scatter figure...")
    palette, labels = _palette_and_labels()
    fig, ax = plt.subplots(figsize=(9, 7))
    rng = np.random.default_rng(42)
    for k, c in palette.items():
        sub = stops[stops["exclusion_class"] == k]
        ax.scatter(sub["n_services_within"] +
                   rng.uniform(-0.15, 0.15, len(sub)),
                   sub["burden_hol"], c=c, s=8, alpha=0.55,
                   linewidth=0, label=labels[k])
    ax.set_xlabel("Number of essential-service categories within catchment",
                  fontsize=11)
    ax.set_ylabel("Holiday social burden (V × headway)", fontsize=11)
    ax.set_title("Holiday burden vs. essential-service access",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks([0, 1, 2, 3])
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "thesis_phase4_scatter.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    return path


def make_figure_map(stops):
    """Figure B: spatial distribution of exclusion classes."""
    print("[7b] Building exclusion-class spatial map figure...")
    palette, labels = _palette_and_labels()
    fig, ax = plt.subplots(figsize=(10, 9))
    for k, c in palette.items():
        sub = stops[stops["exclusion_class"] == k]
        if len(sub) == 0:
            continue
        size  = 18 if k == "double_exclusion" else (8 if k != "standard" else 4)
        alpha = 0.9 if k == "double_exclusion" else (0.6 if k != "standard" else 0.3)
        ax.scatter(sub.geometry.x, sub.geometry.y, c=c, s=size,
                   alpha=alpha, linewidth=0, label=labels[k])
    ax.set_aspect("equal"); ax.set_facecolor("#EFF6FF")
    ax.set_title("Spatial distribution of exclusion classes",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Easting (m)", fontsize=9)
    ax.set_ylabel("Northing (m)", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "thesis_phase4_map.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    return path


def make_figure_tertile(stops):
    """Figure C: exclusion-class composition by vulnerability tertile."""
    print("[7c] Building exclusion-class tertile-composition figure...")
    palette, labels = _palette_and_labels()
    fig, ax = plt.subplots(figsize=(8, 7))
    stops_local = stops.copy()
    stops_local["vuln_tertile"] = pd.qcut(
        stops_local["v_index"], 3,
        labels=["Low vulnerability", "Mid vulnerability", "High vulnerability"])
    pivot = stops_local.groupby(
        ["vuln_tertile", "exclusion_class"], observed=True
    ).size().unstack(fill_value=0)
    for k in palette:
        if k not in pivot.columns:
            pivot[k] = 0
    pivot = pivot[["standard", "high_burden_only", "isolated_only",
                   "double_exclusion"]]
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct.plot(kind="bar", stacked=True, ax=ax,
                   color=[palette[c] for c in pivot_pct.columns],
                   alpha=0.9, edgecolor="white", width=0.7)
    ax.set_xlabel("Vulnerability tertile", fontsize=11)
    ax.set_ylabel("% of stops (within tertile)", fontsize=11)
    ax.set_title("Exclusion-class composition by vulnerability tertile",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend([labels[c] for c in pivot_pct.columns],
              fontsize=9, loc="upper left")
    ax.tick_params(axis="x", rotation=20, labelsize=10)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "thesis_phase4_tertile.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    return path


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main(top_n=20):
    print("--- Phase 4: Essential Services & Double Exclusion ---")
    stops, n_unserved = load_stop_state()
    services = download_services()
    stops = attach_nearest_distances(stops, services)
    stops = compute_access(stops)
    stops = classify_double_exclusion(stops)
    priority, n_priority = priority_table(stops, top_n=top_n)
    fig_path_scatter = make_figure_scatter(stops)
    fig_path_map = make_figure_map(stops)
    fig_path_tertile = make_figure_tertile(stops)

    # ---- exports ----
    p_stops = os.path.join(config.OUTPUT_DIR, "phase4_essential_access.gpkg")
    cols_out = ["stop_id", "geometry", "v_index", "hw_hol", "burden_hol",
                "n_services_within", "essential_access", "high_burden",
                "fails_access", "double_exclusion", "exclusion_class", "role"]
    for c in config.ESSENTIAL_SERVICE_TAGS:
        cols_out += [f"dist_{c}", f"near_{c}"]
    if "stop_name" in stops.columns:
        cols_out.insert(2, "stop_name")
    cols_out = [c for c in cols_out if c in stops.columns]
    stops[cols_out].to_file(p_stops, driver="GPKG")

    # Services carry MIXED geometry types (Points for small amenities,
    # Polygons for large complexes — the polygons are retained deliberately
    # for boundary-distance accuracy). A single GPKG layer with mixed types
    # fails or degrades under some GDAL/pyogrio versions, so export one layer
    # per geometry family within the same GeoPackage.
    p_svc = os.path.join(config.OUTPUT_DIR, "phase4_services.gpkg")
    pts  = services[services.geometry.geom_type == "Point"]
    poly = services[services.geometry.geom_type.isin(("Polygon", "MultiPolygon"))]
    if len(pts):
        pts.to_file(p_svc, driver="GPKG", layer="services_points")
    if len(poly):
        poly.to_file(p_svc, driver="GPKG", layer="services_polygons")

    p_doubl = os.path.join(config.OUTPUT_DIR, "phase4_double_exclusion.csv")
    double_table = stops[stops["double_exclusion"]].copy()
    double_table = double_table.sort_values("burden_hol", ascending=False).head(top_n)
    double_table["hw_is_penalty"] = (double_table["hw_hol"].round(1)
                                     == float(config.NO_SERVICE_PENALTY_MIN))
    keep = [c for c in ["stop_id", "stop_name", "v_index", "hw_hol",
                         "hw_is_penalty", "burden_hol", "n_services_within"]
            + [f"dist_{c}" for c in config.ESSENTIAL_SERVICE_TAGS]
            if c in double_table.columns]
    double_table[keep].to_csv(p_doubl, index=False)

    p_pri = os.path.join(config.OUTPUT_DIR, "phase4_priority_table.csv")
    if not priority.empty:
        priority.to_csv(p_pri, index=False)   # the full ranked list, not truncated

    # ---- console summary ----
    n_double = int(stops["double_exclusion"].sum())
    n_iso    = int((stops["exclusion_class"] == "isolated_only").sum())
    n_burden = int((stops["exclusion_class"] == "high_burden_only").sum())
    n_recip_double = n_priority   # full count, not the truncated display

    # mean distance to nearest service across categories, for the comparison sentence
    dist_cols = [f"dist_{c}" for c in config.ESSENTIAL_SERVICE_TAGS]
    avg_dist_all = stops[dist_cols].mean(axis=1).mean()
    avg_dist_dbl = stops.loc[stops["double_exclusion"], dist_cols].mean(axis=1).mean()
    avg_dist_std = stops.loc[stops["exclusion_class"] == "standard", dist_cols].mean(axis=1).mean()

    print("\n=== EXCLUSION SUMMARY ===")
    print(f"  Stops analysed:          {len(stops)}")
    print(f"  Double-excluded:         {n_double} ({n_double/len(stops)*100:.1f}%)")
    print(f"  High burden only:        {n_burden}")
    print(f"  Essential-access failure only: {n_iso}")
    print("  Mean dist-to-services (all categories averaged):")
    print(f"      all stops:           {avg_dist_all:.0f} m")
    print(f"      double-excluded:     {avg_dist_dbl:.0f} m")
    print(f"      standard:            {avg_dist_std:.0f} m")
    if n_recip_double:
        print(f"  Priority stops (double-excluded AND Phase 5 recipient): "
              f"{n_recip_double}")

    print("\nSummary:")
    print(f"  Of the {len(stops)} analytical stops, {n_double} "
          f"({n_double/len(stops)*100:.1f}%) are classified as double-excluded:")
    print("  high holiday social burden combined with failure to meet the")
    print("  constraint-limited essential-access criterion (within catchment of")
    print("  pharmacy, supermarket, AND hospital simultaneously). On average")
    print(f"  these stops sit {avg_dist_dbl:.0f} m from the relevant service "
          f"categories, compared with {avg_dist_std:.0f} m for non-excluded stops")
    print(f"  (a {avg_dist_dbl/max(avg_dist_std,1):.1f}× distance ratio). "
          f"Of the double-excluded set, {n_recip_double} are also Phase 3")
    print("  recipients — the priority set for local review "
          "(§5.4).")

    rp = config.write_run_parameters("phase4", extra={
        "n_stops_analysed":     int(len(stops)),
        "n_double_exclusion":   n_double,
        "n_high_burden_only":   n_burden,
        "n_isolated_only":      n_iso,
        "n_priority_stops":     int(n_recip_double),
        "no_service_convention": "penalised",
        "no_service_penalty_min": float(config.NO_SERVICE_PENALTY_MIN),
        "n_unserved_holiday_stops": int(n_unserved),
        "service_catchments_m": config.SERVICE_CATCHMENTS_M,
        "burden_quartile_cut":  float(stops["burden_hol"].quantile(
            config.DOUBLE_EXCLUSION_BURDEN_Q)),
    })

    print("\nOutputs:")
    for p in [p_stops, p_svc, p_doubl, p_pri,
              fig_path_scatter, fig_path_map, fig_path_tertile, rp]:
        if os.path.exists(p):
            print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
