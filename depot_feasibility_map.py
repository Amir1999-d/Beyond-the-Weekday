"""
================================================================================
depot_feasibility_map.py  —  Stop-level map of depot compatibility
================================================================================
Visualises the depot-attribution screen at stop level: which modelled changes
are structurally compatible within depot attribution, which are only partially
compatible, which imply an inter-depot imbalance, and which are non-transferable.

ATTRIBUTION RULE (disclosed — this is a visualisation of the depot screen,
not a duty-roster simulation)
--------------------------------------------------------------------------------
"Same-depot servability" is a depot-level netting concept: individual trips
carry no depot tag. Each stop's delta is therefore attributed pro-rata:
  1. split stop delta equally across serving surface lines, then equally
     across each line's depots (identical to depot_feasibility.py);
  2. per depot d, internal ratio:
        recipients: r_d = min(freed_d, gained_d) / gained_d
        donors:     r_d = min(freed_d, gained_d) / freed_d
     i.e. the fraction of that depot's gains (freed trips) coverable
     (absorbable) within the depot itself;
  3. a stop's internal_share = sum over its depots of (share of the stop's
     |delta| in d) x r_d.
Moved stops with no transferable surface attribution are classed
NON-TRANSFERABLE. This includes metro, rail and operator-excluded/unmatched
service represented outside the transferable surface pool.

Classes (thresholds disclosed below):
  same_depot        internal_share >= 0.75
  mixed             0.25 < internal_share < 0.75
  requires_transfer internal_share <= 0.25
  non_transferable  no transferable surface attribution

Requires: the July-2026 depot_feasibility.py (operator exclusions + ambito)
in the same folder — its loaders are imported, so the operator-mandated line
exclusions (161, BDQ/Q) apply automatically.

Writes:  depot_feasibility_map.gpkg   (stops layer, EPSG:4326, QGIS-ready)
         thesis_depot_feasibility_map.png
         run_parameters_depot_feasibility_map.json
================================================================================
"""

import os

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import config
import depot_feasibility as dfz

config.validate()

THRESH_SAME = 0.75      # internal_share >= : "same_depot"
THRESH_TRANSFER = 0.25  # internal_share <= : "requires_transfer"

CLASS_COLORS = {
    "same_depot":        "#15803D",   # green  — structurally compatible within depot
    "mixed":             "#F59E0B",   # amber
    "requires_transfer": "#DC2626",   # red    — needs inter-depot moves
    "non_transferable":  "#6D28D9",   # violet — metro-bound capacity
}


def main():
    print("--- Stop-level depot-feasibility map "
          f"({config.RECOMMENDED_SCENARIO} scenario) ---")
    print("[1] Rebuilding the depot attribution (shared loaders)...")
    depot = dfz.load_depot_table()
    opt_path = os.path.join(
        config.OUTPUT_DIR,
        f"milan_optimised_{config.RECOMMENDED_SCENARIO.lower()}.csv")
    opt = pd.read_csv(opt_path)
    opt["stop_id"] = opt["stop_id"].astype(str).str.strip()
    moved = opt[opt["delta_trips"].abs() > 1e-9][["stop_id", "role",
                                                  "delta_trips"]]
    pair = dfz.stop_to_lines()

    m = moved.merge(pair, on="stop_id", how="left")
    metro_only = m["line"].isna()
    m["line"] = m["line"].fillna("UNASSIGNED")
    m["n_lines"] = m.groupby("stop_id")["line"].transform("count")
    m["delta_line"] = m["delta_trips"] / m["n_lines"]
    md = m.merge(depot, on="line", how="left")
    md["depot"] = md["depot"].fillna("UNASSIGNED")
    md["n_depots"] = md["n_depots"].fillna(1)
    md["delta_depot"] = md["delta_line"] / md["n_depots"]

    print("[2] Depot-level internal ratios...")
    g = md.groupby("depot")["delta_depot"]
    bal = pd.DataFrame({
        "gained": g.apply(lambda s: s[s > 0].sum()),
        "freed":  g.apply(lambda s: -s[s < 0].sum()),
    }).fillna(0.0)
    bal["internal"] = np.minimum(bal["gained"], bal["freed"])
    r_gain = (bal["internal"] / bal["gained"].replace(0, np.nan)).fillna(0.0)
    r_free = (bal["internal"] / bal["freed"].replace(0, np.nan)).fillna(0.0)
    r_gain.loc["UNASSIGNED"] = 0.0   # metro pool: nothing is 'internal'
    r_free.loc["UNASSIGNED"] = 0.0

    print("[3] Pro-rata stop-level internal share...")
    md["ratio"] = np.where(md["delta_depot"] > 0,
                           md["depot"].map(r_gain),
                           md["depot"].map(r_free))
    md["w"] = md["delta_depot"].abs()
    stop = (md.assign(wr=md["w"] * md["ratio"])
              .groupby("stop_id")
              .agg(role=("role", "first"),
                   delta_trips=("delta_trips", "first"),
                   w=("w", "sum"), wr=("wr", "sum"),
                   depots=("depot", lambda s: "; ".join(sorted(set(s)))),
                   ambito=("ambito", lambda s: "; ".join(
                       sorted(set(x for x in s.dropna())))))
              .reset_index())
    stop["internal_share"] = (stop["wr"] / stop["w"].replace(0, np.nan)) \
        .fillna(0.0).round(4)
    stop = stop.drop(columns=["w", "wr"])

    is_metro_only = stop["stop_id"].isin(m.loc[metro_only, "stop_id"])
    stop["feas_class"] = np.select(
        [is_metro_only,
         stop["internal_share"] >= THRESH_SAME,
         stop["internal_share"] <= THRESH_TRANSFER],
        ["non_transferable", "same_depot", "requires_transfer"],
        default="mixed")

    counts = stop["feas_class"].value_counts()
    print("    classes: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    print("[4] Attaching geometry...")
    stops_g = gpd.read_file(os.path.join(config.OUTPUT_DIR, "milan_stops.gpkg"))
    stops_g["stop_id"] = stops_g["stop_id"].astype(str).str.strip()
    gdf = stops_g[["stop_id", "stop_name", "geometry"]].merge(stop, on="stop_id")
    n_lost = len(stop) - len(gdf)
    if n_lost:
        print(f"    NOTE: {n_lost} moved stops without geometry (should be 0)")
    gdf = gpd.GeoDataFrame(gdf, crs=stops_g.crs).to_crs(config.CRS_GEOGRAPHIC)

    p_gpkg = os.path.join(config.OUTPUT_DIR, "depot_feasibility_map.gpkg")
    gdf.to_file(p_gpkg, driver="GPKG")

    print("[5] Thesis figure...")
    plot = gdf.to_crs(config.CRS_METRIC)
    fig, ax = plt.subplots(figsize=(11, 11))
    # context: all analytical stops in light grey
    ctx = stops_g.to_crs(config.CRS_METRIC)
    ax.scatter(ctx.geometry.x, ctx.geometry.y, s=2, color="#CBD5E1",
               alpha=0.5, linewidth=0)
    order = ["mixed", "same_depot", "requires_transfer", "non_transferable"]
    labels = {
        "same_depot": "Compatible within depot attribution",
        "mixed": "Partially within depot",
        "requires_transfer": "Inter-depot imbalance",
        "non_transferable": "Non-transferable surface attribution",
    }
    for cls in order:
        sub = plot[plot["feas_class"] == cls]
        size = 6 + 1.2 * sub["delta_trips"].abs().clip(upper=40)
        ax.scatter(sub.geometry.x, sub.geometry.y, s=size,
                   color=CLASS_COLORS[cls], alpha=0.85, linewidth=0,
                   label=f"{labels[cls]} (n={len(sub)})")
    ax.set_title("Stop-level depot compatibility of the "
                 f"{config.RECOMMENDED_SCENARIO} reallocation\n"
                 "pro-rata attribution of depot netting to stops "
                 "(disclosed rule; size = |Δ trips/day|)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_aspect("equal")
    plt.tight_layout()
    p_fig = os.path.join(config.OUTPUT_DIR, "thesis_depot_feasibility_map.png")
    plt.savefig(p_fig, dpi=300, bbox_inches="tight")
    plt.show()

    rp = config.write_run_parameters("depot_feasibility_map", extra={
        "scenario": config.RECOMMENDED_SCENARIO,
        "attribution_rule": "stop delta -> equal split lines -> equal split "
                            "depots; internal_share = pro-rata depot internal "
                            "ratio (min(freed,gained)/side total); stops without transferable surface attribution "
                            "classed non_transferable",
        "class_thresholds": {"same_depot": f">= {THRESH_SAME}",
                             "requires_transfer": f"<= {THRESH_TRANSFER}"},
        "class_counts": {k: int(v) for k, v in counts.items()},
    })
    print("\nOutputs:")
    for p in (p_gpkg, p_fig, rp):
        print(f"  {os.path.basename(p)}")
    print("""
QGIS STYLING:
  Open depot_feasibility_map.gpkg
  Option A (categorical): Categorized on 'feas_class'
      same_depot        -> #15803D  green
      mixed             -> #F59E0B  amber
      requires_transfer -> #DC2626  red
      non_transferable  -> #6D28D9  violet
  Option B (continuous): Graduated on 'internal_share', ramp RdYlGn,
      5 classes, equal interval 0-1 (violet override for non_transferable
      via a rule-based second symbol on feas_class).
  Size: data-defined override on abs("delta_trips"):
      scale_linear(abs("delta_trips"), 0, 40, 2, 8)
  Underlay: milan_zones.gpkg at 40% opacity styled by v_index, plus
      milan_stop_vindex.gpkg in light grey for network context.""")


if __name__ == "__main__":
    main()
