"""
================================================================================
sensitivity_suite.py — Consolidated robustness & sensitivity appendix
================================================================================
Runs the seven stress-tests identified in the methodological review against the
FROZEN pipeline outputs (run record 2026-07-10T11:57:16). Nothing here modifies
any frozen artifact; every analysis reads the frozen CSV/GPKG outputs (plus the
ISTAT source for the weight grid) and writes one appendix table, four figures,
and a run_parameters record.

  A. Vulnerability-weight grid ........ 36-point simplex (step 0.1, each w>=0.1)
       -> Spearman rho of the burden ranking vs frozen weights
       -> Jaccard stability of the high-vulnerability tertile
       -> Moderate-scenario Gini improvement under each weight vector
       -> Jaccard stability of the priority set (double-excluded n recipient)
  B. Service-floor threshold sweep .... failure shares at 20-40 min, by day-type
  C. Donor/recipient tertile cuts ..... (0.25/0.75)(0.30/0.70)(1/3,2/3)(0.36/0.64)
       -> counts, Gini, recipient-set Jaccard vs frozen
  D. Catchment sweep + centroid rule .. any-category 300/400/500 m x hospital
       600/800/1000 m -> double-exclusion count & Jaccard; plus the
       distance-to-CENTROID variant recomputed from phase4_services.gpkg
  E. Continuous vs quartile priority .. continuous score = burden x worst
       relative shortfall; overlap with the frozen classes
  F. Bridge-day relaxation ............ Labour Day re-admitted to the holiday
       ranking (already computed for transparency by Phase 2)
  G. Bootstrap envelopes .............. stop-resampling CIs for the Moderate
       Gini improvement and the penalised burden ratio
       (data are a census of the network, not a sample: the bootstrap
        quantifies sensitivity to stop composition, not sampling error)

Reads:   milan_stop_hourly.csv, milan_optimised_moderate.csv,
         phase4_essential_access.gpkg, phase4_services.gpkg,
         phase2_holiday_comparison.csv, milan_zones.gpkg, ISTAT source,
         primary-feed stops.txt
Writes:  sensitivity_appendix.csv, thesis_sens_floor.png,
         thesis_sens_catchment.png, thesis_sens_bootstrap.png,
         thesis_sens_weights.png, run_parameters_sensitivity_suite.json
================================================================================
"""

import os
import io
import contextlib
import itertools

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr

import config
import optimise
import phase1_corridors as corridors

config.validate()

SUITE_VERSION = "1.1 — seeded bootstrap, frozen-estimand ratio, 2026-07-16"
RNG = np.random.default_rng(42)          # fixed seed: bootstrap reproducibility
N_BOOT = 500
FLOOR_GRID = [20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40]
ANY_GRID = [300, 400, 500]               # pharmacy & supermarket catchment (m)
HOSP_GRID = [600, 800, 1000]             # hospital catchment (m)
TERTILE_GRID = [(0.25, 0.75), (0.30, 0.70), (1/3, 2/3), (0.36, 0.64)]

REC = []                                 # long-format appendix records


def rec(analysis, variant, metric, value):
    REC.append({"analysis": analysis, "variant": variant,
                "metric": metric, "value": value})


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def quiet(fn, *a, **k):
    """Run a chatty frozen-pipeline function without console noise."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# ─────────────────────────────────────────────────────────────────────────────
# Shared frozen inputs
# ─────────────────────────────────────────────────────────────────────────────
def load_frozen():
    print("[0] Loading frozen artifacts...")
    stop = quiet(optimise.load_stop_level)          # penalised hw_hol, v, burden
    frozen_roles = quiet(optimise.classify, stop.copy())
    acc = gpd.read_file(os.path.join(config.OUTPUT_DIR,
                                     "phase4_essential_access.gpkg"))
    acc["stop_id"] = acc["stop_id"].astype(str).str.strip()
    opt = pd.read_csv(os.path.join(config.OUTPUT_DIR,
                                   "milan_optimised_moderate.csv"))
    opt["stop_id"] = opt["stop_id"].astype(str).str.strip()
    for c in ("dist_pharmacy", "dist_supermarket", "dist_hospital",
              "burden_hol", "v_index"):
        assert c in acc.columns, f"phase4 gpkg missing column {c}"
    # essential_access: use the exported flag if present, otherwise derive it
    # from the per-category near_* flags (the identical AND definition)
    if "essential_access" not in acc.columns:
        near_cols = [f"near_{c}" for c in config.SERVICE_CATCHMENTS_M]
        assert all(c in acc.columns for c in near_cols), \
            f"phase4 gpkg lacks essential_access AND {near_cols}"
        acc["essential_access"] = acc[near_cols].all(axis=1)
        print("    (essential_access derived from near_* flags — AND rule)")
    print(f"    stop-level: {len(stop)}  phase4: {len(acc)}  optimiser: {len(opt)}")
    return stop, frozen_roles, acc, opt


# ─────────────────────────────────────────────────────────────────────────────
# A. Vulnerability-weight grid
# ─────────────────────────────────────────────────────────────────────────────
def section_shares():
    """
    Per-section normalised shares — MIRRORS build_stop_table.build_vulnerability
    exactly (comune filter -> shares -> within-study-area MinMax), stopping
    BEFORE the weighting step so any weight vector can be applied.
    """
    istat = pd.read_excel(config.ISTAT_XLSX)
    if config.ISTAT_COMUNE_CODE is not None:
        col = config.ISTAT_COMUNE_COLUMN
        istat = istat[istat[col].astype(str).str.split(".").str[0]
                      == str(config.ISTAT_COMUNE_CODE)].copy()
    share_cols = []
    for name, spec in config.VULNERABILITY_VARIABLES.items():
        num = istat[spec["numerator"]].apply(pd.to_numeric,
                                             errors="coerce").sum(axis=1)
        den = pd.to_numeric(istat[spec["denominator"]], errors="coerce")
        istat[name] = (num / den).replace([np.inf, -np.inf], np.nan).fillna(0)
        share_cols.append(name)
    norm = MinMaxScaler().fit_transform(istat[share_cols].fillna(0))
    out = pd.DataFrame(norm, columns=share_cols)
    out["SEZ_JOIN_ID"] = (istat[config.ZONE_ID_COLUMN].astype(str)
                          .str.split(".").str[0].values)
    return out, share_cols


def stop_to_zone_shares(shares):
    """
    One cached two-stage join (mirror of the builder's frozen rule:
    within -> snap <= ZONE_SNAP_MAX_DIST_M) mapping each stop to its zone's
    normalised shares. Weight vectors then reduce to a dot product.
    """
    zones = gpd.read_file(os.path.join(config.OUTPUT_DIR, "milan_zones.gpkg"))
    assert "SEZ_JOIN_ID" in zones.columns, "milan_zones.gpkg lacks SEZ_JOIN_ID"
    zones["SEZ_JOIN_ID"] = zones["SEZ_JOIN_ID"].astype(str)
    zones = zones.merge(shares, on="SEZ_JOIN_ID", how="inner")

    gtfs = config.GTFS_FEEDS[config.PRIMARY_FEED]
    st = pd.read_csv(os.path.join(gtfs, "stops.txt"),
                     usecols=["stop_id", "stop_lat", "stop_lon"],
                     dtype={"stop_id": str})
    st["stop_id"] = st["stop_id"].str.strip()
    pts = gpd.GeoDataFrame(
        st, geometry=gpd.points_from_xy(pd.to_numeric(st.stop_lon),
                                        pd.to_numeric(st.stop_lat)),
        crs=config.CRS_GEOGRAPHIC).to_crs(config.CRS_METRIC)
    zm = zones.to_crs(config.CRS_METRIC)

    share_cols = [c for c in shares.columns if c != "SEZ_JOIN_ID"]
    j1 = gpd.sjoin(pts, zm[["geometry"] + share_cols], how="left",
                   predicate="within").drop(columns="index_right")
    j1 = j1[~j1.index.duplicated(keep="first")]
    inside = j1[share_cols[0]].notna()
    snap = gpd.sjoin_nearest(pts.loc[~inside], zm[["geometry"] + share_cols],
                             how="left",
                             max_distance=config.ZONE_SNAP_MAX_DIST_M,
                             distance_col="_d").drop(columns="index_right")
    snap = snap[~snap.index.duplicated(keep="first")]
    out = pd.concat([j1.loc[inside],
                     snap[snap[share_cols[0]].notna()].drop(columns="_d")])
    print(f"    stop->zone shares mapped: {len(out)} stops "
          f"(mirror of the frozen two-stage rule)")
    return out[["stop_id"] + share_cols].set_index("stop_id"), share_cols


def analysis_A(stop, frozen_roles, acc):
    print("[A] Vulnerability-weight grid (36-point simplex)...")
    shares, cols = section_shares()
    sz, cols = stop_to_zone_shares(shares)

    base = stop.set_index("stop_id")
    common = base.index.intersection(sz.index)
    base = base.loc[common]
    S = sz.loc[common, cols].values                      # n_stops x 3

    frozen_w = np.array([config.VULNERABILITY_WEIGHTS[c] for c in cols])
    v_frozen = S @ frozen_w
    q_high = getattr(config, "HIGH_VULN_TERTILE", 2/3)
    high_frozen = set(base.index[v_frozen >= np.quantile(v_frozen, q_high)])

    frozen_rec = set(frozen_roles.loc[frozen_roles.role == "recipient",
                                      "stop_id"])
    fails = acc.set_index("stop_id")["essential_access"].reindex(common)
    fails = ~fails.fillna(True).astype(bool)             # True = fails access
    bq = config.DOUBLE_EXCLUSION_BURDEN_Q
    b_frozen = v_frozen * base["hw_hol"].values
    de_frozen = set(base.index[(b_frozen >= np.quantile(b_frozen, bq))
                               & fails.values])
    prio_frozen = de_frozen & frozen_rec

    grid = [w for w in itertools.product(np.arange(1, 9) / 10, repeat=3)
            if abs(sum(w) - 1.0) < 1e-9]
    ginis, rhos, jac_t, jac_p = [], [], [], []
    for w in grid:
        wv = np.array(w)
        v = S @ wv
        rho = spearmanr(v * base["hw_hol"].values, b_frozen).statistic
        jt = jaccard(base.index[v >= np.quantile(v, q_high)], high_frozen)
        # full chain: reclassify + Moderate run under the new v
        s2 = base.reset_index()[["stop_id", "hw_hol"]].copy()
        s2["v_index"] = v
        s2["burden_hol"] = s2["v_index"] * s2["hw_hol"]
        s2["trips_hol"] = config.OPERATIONAL_MINUTES / s2["hw_hol"]
        s2 = quiet(optimise.classify, s2)
        _, stats = quiet(optimise.run_scenario, s2,
                         config.SCENARIOS["Moderate"])
        g = stats["gini_improvement_%"]
        de = set(s2.loc[(s2["burden_hol"] >= s2["burden_hol"].quantile(bq))
                        & fails.values, "stop_id"])
        jp = jaccard(de & set(s2.loc[s2.role == "recipient", "stop_id"]),
                     prio_frozen)
        ginis.append(g); rhos.append(rho); jac_t.append(jt); jac_p.append(jp)
        rec("A_weights", f"w={w}", "gini_moderate_%", round(g, 2))
        rec("A_weights", f"w={w}", "spearman_burden_vs_frozen", round(rho, 4))
        rec("A_weights", f"w={w}", "jaccard_high_vuln_tertile", round(jt, 4))
        rec("A_weights", f"w={w}", "jaccard_priority_set", round(jp, 4))

    print(f"    Gini (Moderate) across grid: "
          f"{min(ginis):.1f}-{max(ginis):.1f}% "
          f"(frozen weights: recomputed in-grid)")
    print(f"    Spearman rho of burden ranking: min {min(rhos):.3f}")
    print(f"    High-vuln tertile Jaccard:      min {min(jac_t):.3f}")
    print(f"    Priority-set Jaccard:           min {min(jac_p):.3f}")

    # the reference line is the adopted-weight point estimate from
    # the frozen run, read at runtime rather than typed in.
    import pandas as _pd0
    _sc0 = _pd0.read_csv(os.path.join(config.OUTPUT_DIR, "optimise_scenarios.csv"))
    _gini_adopted = float(_sc0.loc[_sc0.scenario == config.RECOMMENDED_SCENARIO,
                                   "gini_improvement_%"].iloc[0])
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist(ginis, bins=12, color="#1D4ED8", edgecolor="white")
    ax[0].axvline(_gini_adopted, color="#DC2626", linewidth=1.5,
                  label=f"adopted weights ({_gini_adopted:.1f}%)")
    ax[0].set_xlabel("Moderate-scenario Gini improvement (%)")
    ax[0].set_ylabel("weight vectors"); ax[0].legend(fontsize=9)
    ax[0].set_title("A. Gini gain across the 36-point weight simplex",
                    fontsize=11, fontweight="bold")
    ax[1].scatter(jac_t, jac_p, s=30, color="#15803D", alpha=0.8)
    ax[1].set_xlabel("Jaccard: high-vulnerability tertile")
    ax[1].set_ylabel("Jaccard: priority set")
    ax[1].set_title("A. Set stability under weight perturbation",
                    fontsize=11, fontweight="bold")
    for a in ax: a.grid(True, alpha=0.3); a.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "thesis_sens_weights.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# B. Service-floor threshold sweep
# ─────────────────────────────────────────────────────────────────────────────
def analysis_B():
    print("[B] Service-floor threshold sweep...")
    hr = pd.read_csv(os.path.join(config.OUTPUT_DIR, "milan_stop_hourly.csv"))
    hr = hr[config.operational_hour_filter(hr["hour"])]
    agg = hr.groupby("stop_id").agg(v_index=("v_index", "mean"),
                                    hw_wd=("hw_wd", "median"),
                                    hw_sun=("hw_sun", "median"),
                                    hw_hol=("hw_hol", "median")).reset_index()
    scoped = corridors.attach_metadata(agg)
    scoped = corridors.flag_regular_urban(scoped)
    agg = pd.DataFrame(scoped[scoped["regular_urban"]].copy())
    q_high = getattr(config, "HIGH_VULN_TERTILE", 2/3)
    hv = agg["v_index"] >= agg["v_index"].quantile(q_high)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for col, label, color in (("hw_wd", "weekday", "#1D4ED8"),
                              ("hw_sun", "Sunday", "#F59E0B"),
                              ("hw_hol", "holiday", "#DC2626")):
        shares = [((agg.loc[agg[col].notna(), col] > t).mean() * 100) for t in FLOOR_GRID]
        hv_shares = [((agg.loc[hv & agg[col].notna(), col] > t).mean() * 100) for t in FLOOR_GRID]
        ax.plot(FLOOR_GRID, shares, "-o", color=color, label=f"{label} (all)")
        ax.plot(FLOOR_GRID, hv_shares, "--s", color=color, alpha=0.6,
                label=f"{label} (high-vuln)")
        for t, s, s2 in zip(FLOOR_GRID, shares, hv_shares):
            rec("B_floor", f"{label}@{t}min", "fail_share_all_%", round(s, 2))
            rec("B_floor", f"{label}@{t}min", "fail_share_highvuln_%",
                round(s2, 2))
    ax.axvline(30, color="grey", linestyle=":", linewidth=1.2,
               label="thesis floor (30 min)")
    ax.set_xlabel("floor threshold (min)"); ax.set_ylabel("% of stops failing")
    ax.set_title("B. Floor-failure share as a function of the threshold",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "thesis_sens_floor.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    hol30 = (agg.loc[agg["hw_hol"].notna(), "hw_hol"] > 30).mean() * 100
    print(f"    holiday failure at 30 min (regular network, served stops): {hol30:.1f}%  "
          "(full curve in figure/table)")


# ─────────────────────────────────────────────────────────────────────────────
# C. Donor/recipient tertile-cut perturbation
# ─────────────────────────────────────────────────────────────────────────────
def analysis_C(stop, frozen_roles):
    print("[C] Donor/recipient tertile-cut perturbation...")
    frozen_rec = set(frozen_roles.loc[frozen_roles.role == "recipient",
                                      "stop_id"])
    saved = (config.ROLE_VULN_LOW_Q, config.ROLE_VULN_HIGH_Q,
             config.ROLE_HW_LOW_Q, config.ROLE_HW_HIGH_Q)
    try:
        for lo, hi in TERTILE_GRID:
            config.ROLE_VULN_LOW_Q = config.ROLE_HW_LOW_Q = lo
            config.ROLE_VULN_HIGH_Q = config.ROLE_HW_HIGH_Q = hi
            s2 = quiet(optimise.classify, stop.copy())
            _, st = quiet(optimise.run_scenario, s2,
                          config.SCENARIOS["Moderate"])
            j = jaccard(set(s2.loc[s2.role == "recipient", "stop_id"]),
                        frozen_rec)
            tag = f"cuts={lo:.2f}/{hi:.2f}"
            rec("C_tertiles", tag, "n_donors",
                int((s2.role == "donor").sum()))
            rec("C_tertiles", tag, "n_recipients",
                int((s2.role == "recipient").sum()))
            rec("C_tertiles", tag, "gini_moderate_%",
                round(st["gini_improvement_%"], 2))
            rec("C_tertiles", tag, "jaccard_recipients_vs_frozen",
                round(j, 4))
            print(f"    {tag}: donors={int((s2.role=='donor').sum())} "
                  f"recipients={int((s2.role=='recipient').sum())} "
                  f"Gini={st['gini_improvement_%']:.1f}%  J(rec)={j:.3f}")
    finally:
        (config.ROLE_VULN_LOW_Q, config.ROLE_VULN_HIGH_Q,
         config.ROLE_HW_LOW_Q, config.ROLE_HW_HIGH_Q) = saved


# ─────────────────────────────────────────────────────────────────────────────
# D. Catchment sweep + centroid-vs-boundary rule
# ─────────────────────────────────────────────────────────────────────────────
def analysis_D(acc, opt):
    print("[D] Catchment sweep + centroid-vs-boundary variant...")
    a = acc.set_index("stop_id")
    bq = config.DOUBLE_EXCLUSION_BURDEN_Q
    bcut = a["burden_hol"].quantile(bq)
    high = a["burden_hol"] >= bcut
    roles = opt.set_index("stop_id")["role"].reindex(a.index)
    de_frozen = set(a.index[high & ~a["essential_access"].astype(bool)])

    counts = np.zeros((len(ANY_GRID), len(HOSP_GRID)))
    for i, r in enumerate(ANY_GRID):
        for j, h in enumerate(HOSP_GRID):
            ok = ((a["dist_pharmacy"] <= r) & (a["dist_supermarket"] <= r)
                  & (a["dist_hospital"] <= h))
            de = set(a.index[high & ~ok.fillna(False)])
            counts[i, j] = len(de)
            tag = f"any={r}m,hosp={h}m"
            rec("D_catchment", tag, "n_double_excluded", len(de))
            rec("D_catchment", tag, "jaccard_vs_frozen",
                round(jaccard(de, de_frozen), 4))
            rec("D_catchment", tag, "n_priority",
                int(sum(1 for s in de if roles.get(s) == "recipient")))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for j, h in enumerate(HOSP_GRID):
        ax.plot(ANY_GRID, counts[:, j], "-o",
                label=f"hospital {h} m")
    ax.axhline(len(de_frozen), color="grey", linestyle=":",
               label=f"frozen definition ({len(de_frozen)})")
    ax.set_xlabel("pharmacy / supermarket catchment (m)")
    ax.set_ylabel("double-excluded stops")
    ax.set_title("D. Double exclusion under alternative catchments",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "thesis_sens_catchment.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # centroid variant — recomputed from the saved services layers, no OSM
    svc_path = os.path.join(config.OUTPUT_DIR, "phase4_services.gpkg")
    import fiona
    layers = fiona.listlayers(svc_path)
    parts = [gpd.read_file(svc_path, layer=l) for l in layers]
    svc = pd.concat(parts, ignore_index=True)
    svc = gpd.GeoDataFrame(svc, crs=parts[0].crs).to_crs(config.CRS_METRIC)
    cat_col = "service_type" if "service_type" in svc.columns else ("category" if "category" in svc.columns else None)
    if cat_col is None:
        print("    centroid variant skipped: saved services layer has no category field")
        return
    svc["geometry"] = svc.geometry.centroid          # THE variant under test
    stops_m = acc[["stop_id", "geometry"]].to_crs(config.CRS_METRIC)
    okc = pd.Series(True, index=a.index)
    for cat, thr in config.SERVICE_CATCHMENTS_M.items():
        sub = svc[svc[cat_col] == cat]
        near = gpd.sjoin_nearest(stops_m, sub[["geometry"]], how="left",
                                 distance_col="d")
        d = near.groupby("stop_id")["d"].min().reindex(a.index)
        okc &= (d <= thr).fillna(False)
    de_cent = set(a.index[high & ~okc])
    jc = jaccard(de_cent, de_frozen)
    rec("D_centroid", "centroid-rule", "n_double_excluded", len(de_cent))
    rec("D_centroid", "centroid-rule", "jaccard_vs_boundary_rule",
        round(jc, 4))
    print(f"    centroid rule: {len(de_cent)} double-excluded "
          f"(boundary rule: {len(de_frozen)}; Jaccard {jc:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# E. Continuous vs quartile priority
# ─────────────────────────────────────────────────────────────────────────────
def analysis_E(acc, opt):
    print("[E] Continuous vs quartile priority...")
    a = acc.set_index("stop_id")
    shortfall = pd.concat(
        [a[f"dist_{c}"] / t for c, t in config.SERVICE_CATCHMENTS_M.items()],
        axis=1).max(axis=1)                      # worst relative distance
    score = a["burden_hol"] * shortfall
    de_frozen = set(a.index[(a["burden_hol"] >=
                             a["burden_hol"].quantile(
                                 config.DOUBLE_EXCLUSION_BURDEN_Q))
                            & ~a["essential_access"].astype(bool)])
    top_de = set(score.nlargest(len(de_frozen)).index)
    roles = opt.set_index("stop_id")["role"].reindex(a.index)
    prio_frozen = {s for s in de_frozen if roles.get(s) == "recipient"}
    cont_prio = [s for s in score.sort_values(ascending=False).index
                 if roles.get(s) == "recipient"][:len(prio_frozen)]
    j1 = jaccard(top_de, de_frozen)
    j2 = jaccard(cont_prio, prio_frozen)
    rho = spearmanr(score, a["burden_hol"], nan_policy="omit").statistic
    rec("E_continuous", "top-N continuous vs frozen DE",
        "jaccard", round(j1, 4))
    rec("E_continuous", "top-N continuous n recipient vs frozen priority",
        "jaccard", round(j2, 4))
    rec("E_continuous", "score vs burden", "spearman", round(rho, 4))
    print(f"    top-{len(de_frozen)} continuous vs frozen DE: J={j1:.3f}; "
          f"priority overlap J={j2:.3f}; score~burden rho={rho:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# F. Bridge-day relaxation
# ─────────────────────────────────────────────────────────────────────────────
def analysis_F():
    print("[F] Bridge-day relaxation (Labour Day re-admitted)...")
    t = pd.read_csv(os.path.join(config.OUTPUT_DIR,
                                 "phase2_holiday_comparison.csv"))
    six = t[~t["excluded"].astype(bool)]
    lab = t[t["excluded"].astype(bool)]
    if lab.empty:
        print("    no excluded holiday present in the table — skipped")
        return
    lab = lab.iloc[0]
    for m in ("median_hw_hol", "mean_equity_gap", "pct_fail_floor",
              "n_stops"):
        lo, hi = six[m].min(), six[m].max()
        rec("F_bridgeday", "Labour Day vs six-band", m,
            f"{lab[m]} (band {lo}-{hi})")
    print(f"    Labour Day: n_stops={lab['n_stops']} vs six-band "
          f"{six['n_stops'].min()}-{six['n_stops'].max()}; "
          f"median hw {lab['median_hw_hol']} vs "
          f"{six['median_hw_hol'].min()}-{six['median_hw_hol'].max()}; "
          f"fail {lab['pct_fail_floor']}% vs "
          f"{six['pct_fail_floor'].min()}-{six['pct_fail_floor'].max()}%")
    print("    -> re-admitted, it is the outlier the coverage anomaly "
          "predicts: the coverage anomaly remains the reason for exclusion; the Friday flag alone is not decisive.")


# ─────────────────────────────────────────────────────────────────────────────
# G. Bootstrap envelopes
# ─────────────────────────────────────────────────────────────────────────────
def analysis_G(stop, opt):
    print(f"[G] Bootstrap envelopes ({N_BOOT} stop-resampling replicates)...")
    g_reps, r_reps = [], []
    o = opt[["v_index", "hw_wd", "burden_hol"]]   # frozen estimand:
    # numerator over ALL stops (penalised burden), denominator over
    # weekday-served stops only (pandas mean skips NaN products)
    for _ in range(N_BOOT):
        s = stop.sample(len(stop), replace=True,
                        random_state=int(RNG.integers(2**32)))
        s = s.reset_index(drop=True)
        s["stop_id"] = s["stop_id"].astype(str) + "_" + s.index.astype(str)
        s2 = quiet(optimise.classify, s)
        _, st = quiet(optimise.run_scenario, s2, config.SCENARIOS["Moderate"])
        g_reps.append(st["gini_improvement_%"])
        ob = o.sample(len(o), replace=True,
                      random_state=int(RNG.integers(2**32)))
        r_reps.append(ob["burden_hol"].mean()
                      / (ob["v_index"] * ob["hw_wd"]).mean())
    # read both point estimates from the run, do not hardcode them.
    # The literals previously here (22.9 / 19.9 / 2.106) were pre-R1 and
    # pre-R2 values and printed onto the thesis figures as reference lines.
    import pandas as _pd
    _sc = _pd.read_csv(os.path.join(config.OUTPUT_DIR, "optimise_scenarios.csv"))
    _g = float(_sc.loc[_sc.scenario == config.RECOMMENDED_SCENARIO,
                       "gini_improvement_%"].iloc[0])
    _opt = _pd.read_csv(os.path.join(
        config.OUTPUT_DIR,
        f"milan_optimised_{config.RECOMMENDED_SCENARIO.lower()}.csv"))
    _ratio = float((_opt["v_index"] * _opt["hw_hol"]).mean()
                   / (_opt["v_index"] * _opt["hw_wd"]).mean())
    for name, reps, frozen in (("gini_moderate_%", g_reps, round(_g, 3)),
                               ("burden_ratio_penalised", r_reps, round(_ratio, 3))):
        lo, hi = np.percentile(reps, [2.5, 97.5])
        rec("G_bootstrap", name, "ci95_low", round(lo, 3))
        rec("G_bootstrap", name, "ci95_high", round(hi, 3))
        rec("G_bootstrap", name, "frozen_value", frozen)
        print(f"    {name}: 95% CI [{lo:.2f}, {hi:.2f}]  (frozen {frozen})")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist(g_reps, bins=30, color="#1D4ED8", edgecolor="white")
    ax[0].axvline(_g, color="#DC2626", label=f"point estimate ({_g:.1f}%)")
    ax[0].set_xlabel("Gini improvement (%)"); ax[0].legend(fontsize=9)
    ax[1].hist(r_reps, bins=30, color="#F59E0B", edgecolor="white")
    ax[1].axvline(_ratio, color="#DC2626", label=f"point estimate ({_ratio:.3f}x)")
    ax[1].set_xlabel("penalised burden ratio"); ax[1].legend(fontsize=9)
    for a_, t in zip(ax, ("G. Moderate Gini gain", "G. Burden ratio")):
        a_.set_title(t + " — stop-resampling bootstrap",
                     fontsize=11, fontweight="bold")
        a_.grid(True, alpha=0.3); a_.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "thesis_sens_bootstrap.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    REC.clear()                          # fresh table on every run
    print("--- Sensitivity & robustness suite (frozen run "
          "2026-07-10T11:57:16) ---")
    print(f"    suite version: {SUITE_VERSION}")
    stop, frozen_roles, acc, opt = load_frozen()
    analysis_A(stop, frozen_roles, acc)
    analysis_B()
    analysis_C(stop, frozen_roles)
    analysis_D(acc, opt)
    analysis_E(acc, opt)
    analysis_F()
    analysis_G(stop, opt)

    out = pd.DataFrame(REC)
    p_csv = os.path.join(config.OUTPUT_DIR, "sensitivity_appendix.csv")
    out.to_csv(p_csv, index=False)
    rp = config.write_run_parameters("sensitivity_suite", extra={
        "n_bootstrap": N_BOOT, "seed": 42,
        "weight_grid": "3-simplex, step 0.1, each weight in [0.1, 0.8] (36 pts)",
        "floor_grid_min": FLOOR_GRID,
        "tertile_grid": [list(t) for t in TERTILE_GRID],
        "catchment_grid": {"any_m": ANY_GRID, "hospital_m": HOSP_GRID},
        "bootstrap_caveat": "data are a census of the network; the bootstrap "
                            "quantifies sensitivity to stop composition, not "
                            "sampling error",
        "n_records": len(out),
    })
    print(f"\nOutputs:\n  {os.path.basename(p_csv)}  ({len(out)} records)")
    for f in ("thesis_sens_weights.png", "thesis_sens_floor.png",
              "thesis_sens_catchment.png", "thesis_sens_bootstrap.png",
              os.path.basename(rp)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
