"""
================================================================================
hour_diff.py  —  Per-stop headway difference between two hours
================================================================================
Reads the corrected stop-hourly table and computes, for each stop, the
difference between its holiday headway at hour A and hour B:

    delta_hw = hw_hol(A) - hw_hol(B)

    delta_hw > 0  ->  hour A is WORSE (longer wait) than hour B at this stop
    delta_hw < 0  ->  hour A is BETTER (shorter wait) than hour B
    delta_hw = 0  ->  no change between the two hours

Use it to answer: "between 8am and 5pm, where does holiday service actually
change, and where is it constant?" \u2014 which the four side-by-side maps can't
show clearly because of point overlap.

Outputs:
    milan_hour_diff_AvB.gpkg    : stops with delta_hw and absolute_change
    thesis_hour_diff_AvB.png    : two-panel static figure (map + histogram)
    run_parameters_hour_diff.json

Usage:
    python hour_diff.py                 # default: hours 8 vs 17
    python hour_diff.py --a 8 --b 12    # any two hours within the operational window
================================================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import config

config.validate()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=int, default=8,
                   help="first hour (default 8)")
    p.add_argument("--b", type=int, default=17,
                   help="second hour (default 17)")
    p.add_argument("--metric", default="hw_hol",
                   choices=["hw_hol", "hw_wd", "hw_sun"],
                   help="which headway column to difference (default holiday)")
    return p.parse_args()


def main(a=8, b=17, metric="hw_hol"):
    # half-open window [START, END): the last valid hourly bin is END-1
    if not (config.OPERATIONAL_HOUR_START <= a < config.OPERATIONAL_HOUR_END):
        raise ValueError(f"hour A={a} outside operational window "
                         f"[{config.OPERATIONAL_HOUR_START}, {config.OPERATIONAL_HOUR_END})")
    if not (config.OPERATIONAL_HOUR_START <= b < config.OPERATIONAL_HOUR_END):
        raise ValueError(f"hour B={b} outside operational window "
                         f"[{config.OPERATIONAL_HOUR_START}, {config.OPERATIONAL_HOUR_END})")
    if a == b:
        raise ValueError("hours A and B must differ")

    print(f"--- Per-stop {metric} difference: hour {a} vs hour {b} ---")

    # --- 1. load stop-hourly + geometry ---
    df = pd.read_csv(os.path.join(config.OUTPUT_DIR, "milan_stop_hourly.csv"))
    df["stop_id"] = df["stop_id"].astype(str).str.strip()

    # dropna on the metric: a stop-hour row can exist (served on another
    # day-type) while THIS metric is NaN (unserved on this day-type). Without
    # the dropna, NaN deltas leak into the histogram and map.
    a_df = (df[df["hour"] == a][["stop_id", metric]]
            .dropna(subset=[metric]).rename(columns={metric: "hw_a"}))
    b_df = (df[df["hour"] == b][["stop_id", metric]]
            .dropna(subset=[metric]).rename(columns={metric: "hw_b"}))

    # inner-merge: only stops served at BOTH hours can be differenced
    pair = a_df.merge(b_df, on="stop_id", how="inner")
    n_total_a = len(a_df); n_total_b = len(b_df); n_paired = len(pair)
    print(f"    stops served at hour {a}: {n_total_a}")
    print(f"    stops served at hour {b}: {n_total_b}")
    print(f"    stops served at BOTH hours (pairable): {n_paired}")

    pair["delta_hw"] = pair["hw_a"] - pair["hw_b"]
    pair["abs_change"] = pair["delta_hw"].abs()
    # human-readable category for QGIS styling convenience
    pair["change_dir"] = pd.cut(
        pair["delta_hw"],
        bins=[-np.inf, -5, -1, 1, 5, np.inf],
        labels=[f"much better at {a}", f"slightly better at {a}",
                "essentially unchanged",
                f"slightly worse at {a}", f"much worse at {a}"]
    )

    # --- 2. attach geometry ---
    stops = gpd.read_file(os.path.join(config.OUTPUT_DIR, "milan_stops.gpkg"))
    stops["stop_id"] = stops["stop_id"].astype(str).str.strip()
    out = stops[["stop_id", "geometry"]].merge(pair, on="stop_id", how="inner")
    out = gpd.GeoDataFrame(out, crs=config.CRS_GEOGRAPHIC)

    # --- 3. export GeoPackage ---
    tag = f"{metric}_h{a}_vs_h{b}"
    gpkg = os.path.join(config.OUTPUT_DIR, f"milan_hour_diff_{tag}.gpkg")
    out.to_file(gpkg, driver="GPKG")

    # --- 4. summary stats ---
    print(f"\n    delta distribution (min):  mean={pair['delta_hw'].mean():+.2f}  "
          f"median={pair['delta_hw'].median():+.2f}  "
          f"min={pair['delta_hw'].min():+.1f}  max={pair['delta_hw'].max():+.1f}")
    cat_counts = pair["change_dir"].value_counts().reindex(
        [f"much better at {a}", f"slightly better at {a}",
         "essentially unchanged",
         f"slightly worse at {a}", f"much worse at {a}"])
    print("    by category:")
    for k, v in cat_counts.items():
        print(f"      {k:30s} {v:5d}   ({v/n_paired*100:.1f}%)")

    # --- 5. static figure: map (left) + histogram (right) ---
    proj = out.to_crs(config.CRS_METRIC)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7),
                             gridspec_kw={"width_ratios": [3, 2]})

    # map: diverging colours centred on 0
    vmax = max(abs(pair["delta_hw"].quantile(0.02)),
               abs(pair["delta_hw"].quantile(0.98)), 5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    sc = axes[0].scatter(proj.geometry.x, proj.geometry.y,
                         c=out["delta_hw"], cmap="RdBu_r", norm=norm,
                         s=6, alpha=0.8, linewidth=0)
    axes[0].set_facecolor("#F1F5F9")
    axes[0].set_aspect("equal")
    axes[0].set_title(
        f"Per-stop {metric} difference\nhour {a} minus hour {b}\n"
        f"red = worse at {a}, blue = better at {a}",
        fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Easting (m)", fontsize=9)
    axes[0].set_ylabel("Northing (m)", fontsize=9)
    axes[0].tick_params(labelsize=7)
    plt.colorbar(sc, ax=axes[0], shrink=0.75,
                 label=f"\u0394 headway (min)  [+ = worse at hour {a}]")

    # histogram
    axes[1].hist(pair["delta_hw"], bins=40, color="#94A3B8",
                 edgecolor="white", linewidth=0.5)
    axes[1].axvline(0, color="black", linewidth=1.2)
    axes[1].axvline(pair["delta_hw"].median(), color="#DC2626",
                    linestyle="--", linewidth=1.2,
                    label=f"median = {pair['delta_hw'].median():+.1f} min")
    axes[1].set_xlabel(f"\u0394 headway (min): hour {a} \u2212 hour {b}", fontsize=10)
    axes[1].set_ylabel("Number of stops", fontsize=10)
    axes[1].set_title("Distribution of per-stop change", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3); axes[1].set_axisbelow(True)

    plt.suptitle(
        f"Within-day variation of holiday service: hour {a} vs hour {b}",
        fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig_path = os.path.join(config.OUTPUT_DIR, f"thesis_hour_diff_{tag}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()

    rp = config.write_run_parameters("hour_diff", extra={
        "hour_a": a, "hour_b": b, "metric": metric,
        "n_paired": int(n_paired),
        "delta_median_min": float(pair["delta_hw"].median()),
        "delta_mean_min": float(pair["delta_hw"].mean()),
    })

    print("\nOutputs:")
    for p in (gpkg, fig_path, rp):
        print(f"  {os.path.basename(p)}")
    print(f"\nQGIS: style milan_hour_diff_{tag}.gpkg with Graduated on 'delta_hw',")
    print("      ramp RdBu inverted, MANUAL breaks centred on 0:")
    print("      e.g. classes at -10, -3, 3, 10  (anything below/above is the tail).")


if __name__ == "__main__":
    args = parse_args()
    main(a=args.a, b=args.b, metric=args.metric)
