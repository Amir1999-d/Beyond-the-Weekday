"""
================================================================================
phase3_optimisation_map.py  —  Phase 3: Before/After Optimisation Map
================================================================================

METHODOLOGY (thesis §5.5, before and after the reallocation)
--------------------------------------------------------------------------------
The optimisation-map analysis spatialises the result of the budget-neutral
reallocation defined in Chapter 3 and computed in optimise.py. Each stop on
the analytical network carries a role assigned by the optimiser:

    donor      a low-vulnerability, high-frequency stop from which trips have
               been trimmed at the scenario reduction rate;
    recipient  a high-vulnerability, low-frequency stop that receives freed
               trips, weighted by its initial holiday burden and subject to the
               stop-level recipient modelling cap (config.MIN_HEADWAY_FLOOR_MIN;
               8 minutes in the thesis);
    neutral    a stop in neither tertile combination, whose holiday service
               is unchanged by the reallocation.

Recipients are further distinguished as ACTIVE (headway after reallocation
still above the cap) or SATURATED (allocation capped because the stop reached
the cap). Saturated recipients are not recipients that received nothing: in
the thesis run they receive the largest allocations of any recipient. The
"functional reallocation ceiling" finding rests on the share of recipients
that saturate as the scenario rate rises.

For each stop the analysis computes the change in trips per day and the
change in holiday headway between current holiday service and the optimised
state under the recommended scenario (config.RECOMMENDED_SCENARIO; Moderate,
20% donor reduction, in the thesis). These
deltas are exported as a QGIS-styleable GeoPackage and rendered as a
two-panel before/after figure showing the spatial pattern of the
reallocation. The role labels are taken directly from the optimiser output
rather than re-derived from observed deltas, since the optimiser is the
authoritative source of the donor/recipient classification (Pereira et al.
[PEREIRA]; Litman [LITMAN]). The map answers the question "where, exactly,
does the budget-neutral redistribution move service?" — which is the
spatial counterpart of the curve-based ceiling analysis (§5.5.3) and the
direct input to Phase 4's double-exclusion cross-reference.

[PEREIRA] R. H. M. Pereira et al., 2017.
[LITMAN]  T. Litman, 2017.

--------------------------------------------------------------------------------
INPUTS  (from the corrected pipeline)
    milan_optimised_<recommended>.csv   per-stop role + delta_trips + delta_hw
    milan_stops.gpkg                    stop geometry from the builder

OUTPUTS
    phase3_optimisation_map.gpkg        per-stop, with role/sub-category
    phase3_summary_table.csv            counts and mean deltas by group
    thesis_phase3_before_after.png      two-panel publication figure
    run_parameters_phase3.json

Usage:
    python phase3_optimisation_map.py
================================================================================
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import config

config.validate()


# Sub-categories used for map styling.
ROLE_DISPLAY = {
    "donor":              {"colour": "#DC2626", "label": "Donor (frequency reduced)"},
    "recipient_active":   {"colour": "#1D4ED8", "label": "Recipient (frequency increased)"},
    "recipient_saturated":{"colour": "#7C3AED", "label": "Recipient (saturated at floor)"},
    "neutral":            {"colour": "#94A3B8", "label": "Neutral (unchanged)"},
}


# -----------------------------------------------------------------------------
# 1. load corrected optimiser output
# -----------------------------------------------------------------------------
def load_optimised():
    name = f"milan_optimised_{config.RECOMMENDED_SCENARIO.lower()}.csv"
    path = os.path.join(config.OUTPUT_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {name} in {config.OUTPUT_DIR}.\n"
            f"Run optimise.py first."
        )
    print(f"[1] Loading corrected optimiser output: {name}")
    df = pd.read_csv(path)
    df["stop_id"] = df["stop_id"].astype(str).str.strip()

    expected = {"stop_id", "role", "v_index", "hw_hol", "hw_hol_opt",
                "trips_hol", "trips_hol_opt", "burden_hol", "burden_hol_opt",
                "delta_trips", "delta_hw"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"Optimiser output is missing expected columns: {missing}. "
            "Re-run optimise.py — its output schema must match Phase 3's "
            "input contract."
        )

    print(f"    Stops: {len(df)}   Role counts: {df['role'].value_counts().to_dict()}")
    return df


# -----------------------------------------------------------------------------
# 2. sub-classify recipients into active vs saturated
# -----------------------------------------------------------------------------
def sub_classify(df):
    """Distinguish active recipients from those that reach the stop-level
    recipient headway cap during allocation, consistent with optimise.py and
    Table 5.5.
    """
    df = df.copy()
    floor = config.MIN_HEADWAY_FLOOR_MIN
    df["display_role"] = df["role"]
    is_rec = df["role"] == "recipient"
    at_floor = df["hw_hol_opt"] <= floor + 1e-6
    df.loc[is_rec & ~at_floor, "display_role"] = "recipient_active"
    df.loc[is_rec &  at_floor, "display_role"] = "recipient_saturated"

    counts = df["display_role"].value_counts().to_dict()
    print(f"[2] Sub-classification: "
          f"donor={counts.get('donor',0)}, "
          f"recipient_active={counts.get('recipient_active',0)}, "
          f"recipient_saturated={counts.get('recipient_saturated',0)}, "
          f"neutral={counts.get('neutral',0)}")
    return df


# -----------------------------------------------------------------------------
# 3. attach geometry
# -----------------------------------------------------------------------------
def attach_geometry(df):
    print("[3] Attaching geometry from milan_stops.gpkg...")
    stops = gpd.read_file(os.path.join(config.OUTPUT_DIR, "milan_stops.gpkg"))
    stops["stop_id"] = stops["stop_id"].astype(str).str.strip()
    keep = ["stop_id", "geometry"]
    for c in ("stop_name",):
        if c in stops.columns: keep.append(c)
    merged = stops[keep].merge(df, on="stop_id", how="inner")
    g = gpd.GeoDataFrame(merged, crs=config.CRS_GEOGRAPHIC)
    if len(g) < len(df):
        print(f"    WARNING: {len(df)-len(g)} optimiser stops without geometry "
              "(stop_id mismatch). Check builder output.")
    return g


# -----------------------------------------------------------------------------
# 4. summary table
# -----------------------------------------------------------------------------
def summarise(gdf):
    print("[4] Building summary table...")
    rows = []
    for role in ["donor", "recipient_active", "recipient_saturated", "neutral"]:
        sub = gdf[gdf["display_role"] == role]
        if len(sub) == 0:
            continue
        rows.append({
            "Group":               ROLE_DISPLAY[role]["label"],
            "Count":               len(sub),
            "Mean v_index":        round(sub["v_index"].mean(), 3),
            "Mean hw before (min)": round(sub["hw_hol"].mean(), 1),
            "Mean hw after (min)":  round(sub["hw_hol_opt"].mean(), 1),
            "Mean Δ headway (min)": round(sub["delta_hw"].mean(), 1),
            "Mean Δ trips/day":     round(sub["delta_trips"].mean(), 1),
            "Total Δ trips/day":    round(sub["delta_trips"].sum(), 1),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 5. figure
# -----------------------------------------------------------------------------
def make_figure(gdf):
    print("[5] Building two-panel before/after figure...")
    proj = gdf.to_crs(config.CRS_METRIC)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.patch.set_facecolor("#F8FAFC")
    for ax in axes:
        ax.set_facecolor("#EFF6FF"); ax.set_aspect("equal")
        ax.tick_params(labelsize=7)

    # ---- left: BEFORE (baseline holiday burden) ----
    axL = axes[0]
    axL.set_title("BEFORE: holiday social burden\n(V × headway, higher = more inequitable)",
                  fontsize=11, fontweight="bold", pad=10)
    sc = axL.scatter(proj.geometry.x, proj.geometry.y,
                     c=proj["burden_hol"].clip(upper=proj["burden_hol"].quantile(0.98)),
                     cmap="RdYlGn_r", s=6, alpha=0.75, linewidth=0)
    plt.colorbar(sc, ax=axL, shrink=0.7, label="Social burden (V × headway)")
    axL.set_xlabel("Easting (m)", fontsize=9)
    axL.set_ylabel("Northing (m)", fontsize=9)

    # ---- right: AFTER (roles) ----
    axR = axes[1]
    axR.set_title(f"AFTER: {config.RECOMMENDED_SCENARIO} reallocation "
                  f"({int(config.SCENARIOS[config.RECOMMENDED_SCENARIO]*100)}% donor reduction)\n"
                  "blue = active recipient · violet = saturated · red = donor",
                  fontsize=11, fontweight="bold", pad=10)
    # neutral first (background)
    for role in ["neutral", "donor", "recipient_saturated", "recipient_active"]:
        sub = proj[proj["display_role"] == role]
        if len(sub) == 0:
            continue
        # size scales with |delta_trips| for donor/active recipient; flat for the rest
        if role in ("donor", "recipient_active"):
            sizes = np.clip(sub["delta_trips"].abs() * 2.0, 10, 80)
            alpha = 0.85
        else:
            sizes = 4 if role == "neutral" else 25
            alpha = 0.40 if role == "neutral" else 0.85
        axR.scatter(sub.geometry.x, sub.geometry.y,
                    c=ROLE_DISPLAY[role]["colour"],
                    s=sizes, alpha=alpha, linewidth=0,
                    label=f"{ROLE_DISPLAY[role]['label']} (n={len(sub)})")
    axR.legend(fontsize=8.5, loc="lower right", framealpha=0.9,
               edgecolor="#CBD5E1")
    axR.set_xlabel("Easting (m)", fontsize=9)

    plt.suptitle(
        f"Phase 5 — Budget-neutral reallocation, before and after\n"
        f"{config.RECOMMENDED_SCENARIO} scenario "
        f"({int(config.SCENARIOS[config.RECOMMENDED_SCENARIO]*100)}% donor reduction, "
        f"{config.MIN_HEADWAY_FLOOR_MIN}-min recipient modelling cap)  ·  "
        "Milan ATM + ISTAT 2021",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    path = os.path.join(config.OUTPUT_DIR, "thesis_phase3_before_after.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    return path


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    print("--- Phase 5: Before/After Reallocation Map ---")
    df = load_optimised()
    df = sub_classify(df)
    gdf = attach_geometry(df)
    summary = summarise(gdf)
    fig_path = make_figure(gdf)

    # ---- exports ----
    gpkg_path = os.path.join(config.OUTPUT_DIR, "phase3_optimisation_map.gpkg")
    cols = ["stop_id", "geometry", "role", "display_role", "v_index",
            "hw_hol", "hw_hol_opt", "delta_hw",
            "trips_hol", "trips_hol_opt", "delta_trips",
            "burden_hol", "burden_hol_opt"]
    if "stop_name" in gdf.columns: cols.insert(2, "stop_name")
    gdf[cols].to_file(gpkg_path, driver="GPKG")

    summary_path = os.path.join(config.OUTPUT_DIR, "phase3_summary_table.csv")
    summary.to_csv(summary_path, index=False)

    # ---- console summary ----
    print("\n=== SUMMARY BY GROUP ===")
    print(summary.to_string(index=False))

    n_donor = int((gdf["display_role"] == "donor").sum())
    n_active = int((gdf["display_role"] == "recipient_active").sum())
    n_sat = int((gdf["display_role"] == "recipient_saturated").sum())
    mean_recip_imp = float(gdf.loc[gdf["display_role"] == "recipient_active",
                                    "delta_hw"].mean())
    sat_pct = n_sat / (n_active + n_sat) * 100 if (n_active + n_sat) else 0

    print("\nSummary:")
    print(f"  Under the {config.RECOMMENDED_SCENARIO} scenario "
          f"({int(config.SCENARIOS[config.RECOMMENDED_SCENARIO]*100)}% donor reduction), "
          f"{n_donor} stops surrender trips and {n_active} active recipient stops")
    # Saturated recipients are those whose
    # allocation was CAPPED at the floor, not those that received nothing —
    # in this run they receive the largest allocations of any recipient.
    print(f"  gain trips, with mean recipient headway reduced by "
          f"{abs(mean_recip_imp):.1f} minutes across the active set. A further "
          f"{n_sat} recipients")
    print(f"  ({sat_pct:.1f}% of the recipient set) reach the "
          f"{config.MIN_HEADWAY_FLOOR_MIN}-minute stop-level modelling cap during "
          f"allocation and are capped")
    print("  there, which is the mechanism underlying the functional-ceiling "
          "finding (§5.5.3).")

    rp = config.write_run_parameters("phase3", extra={
        "scenario":           config.RECOMMENDED_SCENARIO,
        "scenario_rate":      config.SCENARIOS[config.RECOMMENDED_SCENARIO],
        "n_donors":           n_donor,
        "n_recipients_active":    n_active,
        "n_recipients_saturated": n_sat,
        "mean_recipient_hw_improvement_min": round(abs(mean_recip_imp), 2),
    })

    print("\nOutputs:")
    for p in (gpkg_path, summary_path, fig_path, rp):
        print(f"  {os.path.basename(p)}")

    print("""
    QGIS STYLING:
      Open phase3_optimisation_map.gpkg
      Symbology: Categorized on 'display_role'
        donor                -> red       (#DC2626)
        recipient_active     -> blue      (#1D4ED8)
        recipient_saturated  -> violet    (#7C3AED)
        neutral              -> grey 40%  (#94A3B8)
      Size: data-defined override on |delta_trips|:  scale_linear("delta_trips",-30,30,2,8)
      Add milan_zones.gpkg underneath at 40% opacity, styled by v_index.
    """)


if __name__ == "__main__":
    main()
