"""
phase2_holiday_disaggregation.py — per-holiday disaggregation (thesis Phase 3).

Each civil holiday is evaluated from the GTFS archive that contains its date and
compared with the fixed weekday baseline. The analysis uses the same regular-urban
network definition as the corridor analysis. Friday dates are flagged for review, but
are not excluded automatically: Labour Day is excluded from the comparative ranking
because its GTFS coverage collapses to 118 stops; Ferragosto and Santo Stefano are
retained.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import build_stop_table as bst
import phase1_corridors as corridors

config.validate()


# -----------------------------------------------------------------------------
# 1. feed diagnostic: which holidays does each feed actually cover?
# -----------------------------------------------------------------------------
def feed_diagnostic():
    """Print and return which holiday-feed pairings actually contain the date."""
    print("[1] Feed diagnostic — does each declared feed contain its holiday(s)?")
    rows = []
    for label, info in config.HOLIDAYS.items():
        feed_key  = info["feed"]
        date_str  = info["date"]
        feed_path = config.GTFS_FEEDS.get(feed_key)
        present   = False
        n_dates   = 0
        date_min  = date_max = None
        if feed_path and os.path.isdir(feed_path):
            cal_path = os.path.join(feed_path, "calendar_dates.txt")
            if os.path.exists(cal_path):
                try:
                    cal = pd.read_csv(cal_path, dtype=str).apply(
                        lambda x: x.astype("string").str.strip()
                        if pd.api.types.is_string_dtype(x.dtype) or pd.api.types.is_object_dtype(x.dtype)
                        else x)
                    present = date_str in cal["date"].values
                    n_dates = cal["date"].nunique()
                    date_min, date_max = cal["date"].min(), cal["date"].max()
                except Exception as e:
                    print(f"    {label}: could not read calendar_dates ({e})")
        mark = "✓" if present else "✗"
        win  = f"{date_min}->{date_max}" if date_min else "feed not found"
        print(f"    {mark}  {label:30s}  date={date_str}  "
              f"feed='{feed_key}' [{win}]")
        rows.append({"holiday": label, "date": date_str, "feed": feed_key,
                     "feed_path_exists": bool(feed_path and os.path.isdir(feed_path)),
                     "present_in_feed": present,
                     "feed_date_range": win, "feed_n_dates": n_dates})
    diag = pd.DataFrame(rows)
    return diag


# -----------------------------------------------------------------------------
# 2. corrected baseline (v_index + hw_wd per stop, regular urban only)
# -----------------------------------------------------------------------------
def baseline_state():
    """Load corrected stop state and apply the regular-urban scope filter."""
    print("[2] Loading corrected weekday baseline + regular-urban scope...")
    df = pd.read_csv(os.path.join(config.OUTPUT_DIR, "milan_stop_hourly.csv"))
    df = df[config.operational_hour_filter(df["hour"])].copy()
    stop = df.groupby("stop_id").agg(
        v_index=("v_index", "mean"),
        hw_wd  =("hw_wd",   "median"),
    ).reset_index()
    stop["stop_id"] = stop["stop_id"].astype(str).str.strip()

    # Apply exactly the same regular-urban scope used in the §5.2 corridor analysis.
    before = len(stop)
    scoped = corridors.attach_metadata(stop)
    scoped = corridors.flag_regular_urban(scoped)
    scoped = scoped[scoped["regular_urban"]].copy()
    keep = ["stop_id", "v_index", "hw_wd"]
    print(f"    Regular urban network: {len(scoped)}/{before} stops")
    return pd.DataFrame(scoped[keep])


# -----------------------------------------------------------------------------
# 3. per-holiday stop-level headway from its own feed
# -----------------------------------------------------------------------------
def _read_gtfs(feed_path, name, **kwargs):
    df = pd.read_csv(os.path.join(feed_path, name), dtype=str, **kwargs)
    return df.apply(lambda c: c.astype("string").str.strip()
                    if pd.api.types.is_string_dtype(c.dtype) or pd.api.types.is_object_dtype(c.dtype)
                    else c)


def holiday_stops(label, info):
    """Compute stop-level (trips, headway) for a single civil holiday date,
    pulled from its declared feed. Returns DataFrame or None if not found.
    """
    feed_path = config.GTFS_FEEDS.get(info["feed"])
    if not feed_path or not os.path.isdir(feed_path):
        return None

    cal, _, _ = bst.reconstruct_calendar(feed_path)
    if info["date"] not in set(cal["date"].astype(str)):
        return None

    sids = cal.loc[cal["date"].astype(str) == info["date"], "service_id"].unique()
    trips = _read_gtfs(feed_path, "trips.txt",
                       usecols=["trip_id", "service_id"])
    tids = trips.loc[trips["service_id"].isin(sids), "trip_id"]
    st = _read_gtfs(feed_path, "stop_times.txt",
                    usecols=["trip_id", "arrival_time", "stop_id"])
    st = st[st["trip_id"].isin(tids)].copy()
    # NaN-safe hour parse (GTFS permits blank arrival_time on non-timepoints)
    st["hour"] = pd.to_numeric(st["arrival_time"].str.split(":").str[0],
                               errors="coerce")
    st = st[config.operational_hour_filter(st["hour"])]
    if len(st) == 0:
        return None

    counts = st.groupby("stop_id").size().reset_index(name="trips_hol")
    # canonical single-date headway: T_op / trips_on_that_date
    counts["hw_hol"] = config.OPERATIONAL_MINUTES / counts["trips_hol"]
    counts["holiday"] = label
    counts["date"]    = info["date"]
    return counts


# -----------------------------------------------------------------------------
# 4. assemble per-holiday metrics against the corrected baseline
# -----------------------------------------------------------------------------
def compute_all_holidays(baseline):
    print("[3] Computing per-holiday headways against corrected baseline...")
    frames = []
    for label, info in config.HOLIDAYS.items():
        df = holiday_stops(label, info)
        if df is None:
            print(f"    skipped: {label} (not in declared feed)")
            continue
        df["stop_id"] = df["stop_id"].astype(str).str.strip()
        df = df.merge(baseline[["stop_id", "v_index", "hw_wd"]],
                      on="stop_id", how="inner")  # keeps only regular-urban
        if len(df) == 0:
            continue
        df["burden_hol"] = df["v_index"] * df["hw_hol"]
        df["retention"]  = (df["hw_wd"] / df["hw_hol"]).clip(0, 1)
        df["equity_gap"] = df["v_index"] * (1 - df["retention"])
        df["excluded_from_ranking"] = label in config.CROSS_HOLIDAY_EXCLUSIONS
        df["review_flag"] = label in config.CROSS_HOLIDAY_REVIEW
        frames.append(df)
        excl_mark = " [EXCLUDED from cross-holiday ranking]" \
                    if label in config.CROSS_HOLIDAY_EXCLUSIONS else ""
        print(f"    computed: {label}  ({len(df)} stops){excl_mark}")
    if not frames:
        raise RuntimeError("No holidays could be computed. Check feeds.")
    return pd.concat(frames, ignore_index=True)


# -----------------------------------------------------------------------------
# 5. summary tables
# -----------------------------------------------------------------------------
def summarise(hol_df):
    print("[4] Building per-holiday summary table...")
    def agg(g):
        return pd.Series({
            "n_stops":        len(g),
            "median_hw_hol":  round(g["hw_hol"].median(), 1),
            "mean_burden":    round(g["burden_hol"].mean(), 2),
            "mean_equity_gap": round(g["equity_gap"].mean(), 3),
            "pct_fail_floor": round((g["hw_hol"] > config.SERVICE_FLOOR_MIN)
                                     .mean() * 100, 1),
            "excluded":       bool(g["excluded_from_ranking"].iloc[0]),
            "review":         bool(g["review_flag"].iloc[0]),
        })
    summary = hol_df.groupby("holiday").apply(agg, include_groups=False).reset_index()

    # split: ranking on included only; descriptive shows all
    ranked = summary[~summary["excluded"]].copy()
    ranked = ranked.sort_values("mean_equity_gap", ascending=False)
    excluded = summary[summary["excluded"]].copy()
    return summary, ranked, excluded


def summarise_high_vuln(hol_df):
    print("[5] Building high-vulnerability-tertile summary...")
    cut = hol_df["v_index"].quantile(config.HIGH_VULN_TERTILE)
    sub = hol_df[hol_df["v_index"] >= cut].copy()
    def agg(g):
        return pd.Series({
            "n_stops_high_vuln": len(g),
            "median_hw_hol":     round(g["hw_hol"].median(), 1),
            "mean_equity_gap":   round(g["equity_gap"].mean(), 3),
            "pct_fail_floor":    round((g["hw_hol"] > config.SERVICE_FLOOR_MIN)
                                        .mean() * 100, 1),
            "excluded":          bool(g["excluded_from_ranking"].iloc[0]),
        })
    return sub.groupby("holiday").apply(agg, include_groups=False).reset_index()


# -----------------------------------------------------------------------------
# 6. figure: three panels
# -----------------------------------------------------------------------------
def make_figure(summary, ranked):
    print("[6] Building three-panel figure...")
    # all included + excluded but flagged in colour
    order = list(ranked["holiday"]) + list(summary.loc[summary["excluded"], "holiday"])
    df = summary.set_index("holiday").loc[order].reset_index()
    excluded_mask = df["excluded"].values

    n = len(df); x = np.arange(n)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle(
        "Phase 2 — Per-Holiday Disaggregation\n"
        "Milan ATM, regular urban network, corrected baseline",
        fontsize=13, fontweight="bold")

    def bar_panel(ax, col, title, ylabel, fmt="{:.1f}", floor=None):
        colours = ["#94A3B8" if excl else "#1D4ED8" for excl in excluded_mask]
        bars = ax.bar(x, df[col], color=colours, edgecolor="white", linewidth=0.5)
        if floor is not None:
            ax.axhline(floor, color="orange", linestyle="--",
                       linewidth=1.4, label=f"{floor}-min floor")
            ax.legend(fontsize=8)
        ax.set_xticks(x)
        labels = [f"{lab}{' (excl.)' if excl else ''}"
                  for lab, excl in zip(df["holiday"], excluded_mask)]
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(df[col]) * 0.01,
                    fmt.format(val), ha="center", fontsize=8)

    bar_panel(axes[0], "median_hw_hol",   "Median holiday headway",
              "min", "{:.1f}", floor=config.SERVICE_FLOOR_MIN)
    bar_panel(axes[1], "mean_equity_gap", "Mean equity gap (V × (1 − retention))",
              "equity gap", "{:.3f}")
    bar_panel(axes[2], "pct_fail_floor",  "% stops failing 30-min floor",
              "%", "{:.1f}%")

    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "thesis_phase2_figure.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    return path


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    print("--- Phase 3: Holiday Disaggregation ---")
    diag = feed_diagnostic()
    baseline = baseline_state()
    hol_df   = compute_all_holidays(baseline)

    summary, ranked, excluded = summarise(hol_df)
    hv = summarise_high_vuln(hol_df)

    # ---- exports ----
    p_diag   = os.path.join(config.OUTPUT_DIR, "phase2_feed_diagnostic.csv")
    diag.to_csv(p_diag, index=False)
    p_sum    = os.path.join(config.OUTPUT_DIR, "phase2_holiday_comparison.csv")
    summary.to_csv(p_sum, index=False)
    p_hv     = os.path.join(config.OUTPUT_DIR, "phase2_holiday_vuln_comparison.csv")
    hv.to_csv(p_hv, index=False)
    fig_path = make_figure(summary, ranked)

    # ---- console summary ----
    print("\n=== PER-HOLIDAY COMPARISON (all included holidays, ranked) ===")
    print(ranked.to_string(index=False))
    if len(excluded) > 0:
        print("\n=== EXCLUDED FROM RANKING (computed for transparency) ===")
        print(excluded.to_string(index=False))
        for label in excluded["holiday"]:
            why = config.CROSS_HOLIDAY_EXCLUSIONS[label]
            print(f"  {label}: {why}")

    print("\n=== HIGH-VULNERABILITY TERTILE ===")
    print(hv.to_string(index=False))

    # structural-uniformity assessment
    if len(ranked) >= 2:
        spread_hw   = ranked["median_hw_hol"].max() - ranked["median_hw_hol"].min()
        spread_gap  = ranked["mean_equity_gap"].max() - ranked["mean_equity_gap"].min()
        spread_fail = ranked["pct_fail_floor"].max() - ranked["pct_fail_floor"].min()
        print("\n=== STRUCTURAL UNIFORMITY CHECK ===")
        print(f"  Median headway spread across included holidays: {spread_hw:.1f} min")
        print(f"  Mean equity-gap spread:                          {spread_gap:.3f}")
        print(f"  % failing floor spread:                          {spread_fail:.1f} pp")
        if spread_hw < 5 and spread_gap < 0.03 and spread_fail < 3:
            print("  -> CONCLUSION: holidays produce structurally similar service")
            print("     profiles. The equity gap is location-determined, not")
            print("     event-determined (thesis contribution #2).")
        else:
            print("  -> NOTE: holidays show meaningful variation. Inspect ranked")
            print("     table for which holidays cut deepest.")

    # summary paragraph
    print("\nSummary:")
    if len(ranked) >= 2:
        worst = ranked.iloc[0]; best = ranked.iloc[-1]
        print(f"  Across {len(ranked)} civil public holidays in the analysed window,")
        print(f"  the per-holiday median headway ranged from {best['median_hw_hol']:.1f} to")
        print(f"  {worst['median_hw_hol']:.1f} minutes and the mean equity gap from "
              f"{best['mean_equity_gap']:.3f} to {worst['mean_equity_gap']:.3f}.")
    if len(excluded) > 0:
        names = ", ".join(excluded["holiday"])
        print(f"  {names} was excluded from the cross-holiday ranking as a bridge-")
        print("  day artefact (its values are reported separately for transparency).")

    rp = config.write_run_parameters("phase2", extra={
        "holidays_computed":       sorted(summary["holiday"].tolist()),
        "holidays_excluded":       sorted(excluded["holiday"].tolist()),
        "holidays_in_ranking":     sorted(ranked["holiday"].tolist()),
        "n_baseline_stops_used":   int(len(baseline)),
        "structural_uniformity_spread": {
            "median_hw_min":    float(ranked["median_hw_hol"].max() -
                                       ranked["median_hw_hol"].min())
                                   if len(ranked) >= 2 else None,
            "equity_gap":       float(ranked["mean_equity_gap"].max() -
                                       ranked["mean_equity_gap"].min())
                                   if len(ranked) >= 2 else None,
            "pct_fail_pp":      float(ranked["pct_fail_floor"].max() -
                                       ranked["pct_fail_floor"].min())
                                   if len(ranked) >= 2 else None,
        }
    })
    print("\nOutputs:")
    for p in (p_diag, p_sum, p_hv, fig_path, rp):
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
