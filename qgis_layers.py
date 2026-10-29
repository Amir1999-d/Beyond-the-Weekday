"""
================================================================================
qgis_layers.py  —  Visualisation & QGIS export  (reads corrected builder output)
================================================================================
Produces, all from the corrected milan_stop_hourly.csv / milan_zones.gpkg:

  A. Vulnerability layers
     - milan_zones.gpkg is already exported by the builder (zone choropleth);
       here we additionally export a STOP-LEVEL v_index point layer.
  B. Hourly headway pattern
     - milan_hourly_headway_long.gpkg : per-stop-per-hour (QGIS filter/animate)
     - milan_hourly_headway_curve.csv : network-aggregate median per hour
     - thesis_hourly_headway.png      : the aggregate hour curve (figure)
  C. Holiday deterioration index (literature-standard need x supply-shortfall)
     - equity_gap_sun = v_index * (1 - retention_sun)
     - equity_gap_hol = v_index * (1 - retention_hol)
     - holiday_deterioration = equity_gap_hol - equity_gap_sun
       (the holiday-specific penalty BEYOND the ordinary Sunday reduction)
     exported per stop for QGIS.

All inputs corrected (Milan-normalised v_index, frequency-derived headways,
self-derived denominators). All parameters from config.py.

Usage:
    python qgis_layers.py
================================================================================
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import config

config.validate()


def load_hourly():
    path = os.path.join(config.OUTPUT_DIR, "milan_stop_hourly.csv")
    df = pd.read_csv(path)
    df = df[config.operational_hour_filter(df["hour"])].copy()
    return df


def load_stop_geometry():
    """Stop geometry + v_index from the builder's milan_stops.gpkg."""
    g = gpd.read_file(os.path.join(config.OUTPUT_DIR, "milan_stops.gpkg"))
    g["stop_id"] = g["stop_id"].astype(str).str.strip()
    return g


# =============================================================================
# A. STOP-LEVEL VULNERABILITY POINT LAYER
# =============================================================================
def export_stop_vindex(stops_gdf):
    print("[A] Exporting stop-level v_index point layer...")
    cols = ["stop_id", "v_index", "geometry"]
    cols = [c for c in cols if c in stops_gdf.columns]
    out = stops_gdf[cols].copy()
    path = os.path.join(config.OUTPUT_DIR, "milan_stop_vindex.gpkg")
    out.to_file(path, driver="GPKG")
    print(f"    Saved: {os.path.basename(path)}  ({len(out)} stops)")
    return path


# =============================================================================
# B. HOURLY HEADWAY PATTERN
# =============================================================================
def export_hourly(df, stops_gdf):
    print("[B] Building hourly headway pattern...")

    hw_cols = [c for c in ["hw_wd", "hw_sun", "hw_hol"] if c in df.columns]

    # --- B1: per-stop-per-hour long table (with geometry) for QGIS ---
    long = df[["stop_id", "hour", "v_index"] + hw_cols].copy()
    long["stop_id"] = long["stop_id"].astype(str).str.strip()
    geom = stops_gdf[["stop_id", "geometry"]].drop_duplicates("stop_id")
    long_gdf = gpd.GeoDataFrame(long.merge(geom, on="stop_id", how="left"),
                                crs=config.CRS_GEOGRAPHIC)
    long_path = os.path.join(config.OUTPUT_DIR, "milan_hourly_headway_long.gpkg")
    long_gdf.to_file(long_path, driver="GPKG")
    print(f"    Saved: {os.path.basename(long_path)}  "
          f"({len(long_gdf)} stop-hour rows; filter by 'hour' in QGIS)")

    # --- B2: network-aggregate median headway per hour ---
    curve = df.groupby("hour")[hw_cols].median()
    curve_path = os.path.join(config.OUTPUT_DIR, "milan_hourly_headway_curve.csv")
    curve.to_csv(curve_path)
    print(f"    Saved: {os.path.basename(curve_path)}")

    # --- B3: the aggregate hour-curve figure ---
    label = {"hw_wd": "Weekday", "hw_sun": "Sunday", "hw_hol": "Public holiday"}
    colour = {"hw_wd": "#16A34A", "hw_sun": "#EA580C", "hw_hol": "#DC2626"}
    fig, ax = plt.subplots(figsize=(11, 6))
    for c in hw_cols:
        ax.plot(curve.index, curve[c], marker="o", linewidth=2.4,
                color=colour[c], label=label[c])
    ax.axhline(config.SERVICE_FLOOR_MIN, color="#6B7280", linestyle="--",
               linewidth=1.4, label=f"{config.SERVICE_FLOOR_MIN}-min service threshold")
    ax.set_xlabel("Hour of day", fontsize=11)
    ax.set_ylabel("Median headway (min, served stops)", fontsize=11)
    # bins are half-open [START, END): last hourly bin is END-1
    ax.set_xticks(range(config.OPERATIONAL_HOUR_START, config.OPERATIONAL_HOUR_END))
    ax.set_ylim(0, max(curve.max().max() * 1.15, config.SERVICE_FLOOR_MIN + 5))
    ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_title("Milan service level by hour of day: median headway by day-type\n"
                 "(corrected frequency-derived headways, served stops)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(config.OUTPUT_DIR, "thesis_hourly_headway.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"    Saved: {os.path.basename(fig_path)}")
    return long_path, curve_path, fig_path


# =============================================================================
# C. HOLIDAY DETERIORATION INDEX  (need x supply-shortfall; difference vs Sunday)
# =============================================================================
def export_deterioration(df, stops_gdf):
    print("[C] Computing holiday deterioration index...")

    has_sun = "hw_sun" in df.columns
    # trips per day per stop-hour = operational_minutes-equivalent inverse of headway,
    # but retention is a RATIO of trips, so we can use headway directly:
    #   retention_x = trips_x / trips_wd = hw_wd / hw_x   (since trips ∝ 1/headway)
    # This is robust and avoids re-deriving trip counts.
    agg_cols = {"v_index": ("v_index", "mean"),
                "hw_wd": ("hw_wd", "median"),
                "hw_hol": ("hw_hol", "median")}
    if has_sun:
        agg_cols["hw_sun"] = ("hw_sun", "median")
    stop = df.groupby("stop_id").agg(**agg_cols).reset_index()
    stop["stop_id"] = stop["stop_id"].astype(str).str.strip()

    # retention = weekday_headway / daytype_headway, clipped to [0,1].
    # (shorter holiday headway than weekday -> retention 1; longer -> <1)
    def retention(hw_col):
        r = stop["hw_wd"] / stop[hw_col]
        return r.replace([np.inf, -np.inf], np.nan).clip(0, 1)

    stop["retention_hol"] = retention("hw_hol")
    stop["equity_gap_hol"] = stop["v_index"] * (1 - stop["retention_hol"])

    if has_sun:
        stop["retention_sun"] = retention("hw_sun")
        stop["equity_gap_sun"] = stop["v_index"] * (1 - stop["retention_sun"])
        # the holiday-SPECIFIC penalty, beyond the ordinary Sunday reduction
        stop["holiday_deterioration"] = stop["equity_gap_hol"] - stop["equity_gap_sun"]
    else:
        stop["holiday_deterioration"] = stop["equity_gap_hol"]

    # attach geometry + export
    geom = stops_gdf[["stop_id", "geometry"]].drop_duplicates("stop_id")
    out = gpd.GeoDataFrame(stop.merge(geom, on="stop_id", how="left"),
                           crs=config.CRS_GEOGRAPHIC)
    path = os.path.join(config.OUTPUT_DIR, "milan_holiday_deterioration.gpkg")
    keep = ["stop_id", "v_index", "hw_wd", "hw_hol", "retention_hol", "equity_gap_hol"]
    if has_sun:
        keep += ["hw_sun", "retention_sun", "equity_gap_sun", "holiday_deterioration"]
    else:
        keep += ["holiday_deterioration"]
    keep += ["geometry"]
    out[keep].to_file(path, driver="GPKG")
    print(f"    Saved: {os.path.basename(path)}")

    # a few headline numbers
    print(f"    Mean holiday equity gap:            {stop['equity_gap_hol'].mean():.3f}")
    if has_sun:
        worse = (stop["holiday_deterioration"] > 0).sum()
        print(f"    Mean Sunday equity gap:             {stop['equity_gap_sun'].mean():.3f}")
        print(f"    Stops worse on holiday than Sunday: {worse} "
              f"({worse/len(stop)*100:.1f}%)")
    return path, stop


def main():
    print("--- QGIS layers & hourly/deterioration visualisation ---")
    df = load_hourly()
    stops_gdf = load_stop_geometry()

    p_vindex = export_stop_vindex(stops_gdf)
    p_long, p_curve, p_fig = export_hourly(df, stops_gdf)
    p_det, _ = export_deterioration(df, stops_gdf)

    config.write_run_parameters("qgis_layers", extra={
        "deterioration_definition": "v_index*(1-retention); holiday minus Sunday",
        "retention_definition": "hw_wd / hw_daytype, clipped [0,1]",
    })

    print("\n" + "=" * 70)
    print("QGIS LAYERS READY")
    print("=" * 70)
    print("Vulnerability:")
    print("   milan_zones.gpkg            (zone choropleth, from builder) -> style by v_index")
    print(f"   {os.path.basename(p_vindex):28s} (stop points)             -> graduated by v_index")
    print("Hourly pattern:")
    print(f"   {os.path.basename(p_long):28s} (stop-hour) -> filter/animate by 'hour', style hw_hol")
    print(f"   {os.path.basename(p_fig):28s} (thesis figure)")
    print("Holiday deterioration:")
    print("   milan_holiday_deterioration.gpkg -> diverging style on 'holiday_deterioration'")
    print("     (>0 red = worse on holiday than Sunday; <0 blue = not)")
    print("=" * 70)
    print("Outputs written (incl. run_parameters_qgis_layers.json)")


if __name__ == "__main__":
    main()
