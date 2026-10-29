"""
================================================================================
depot_feasibility.py  —  Post-analysis depot compatibility screen
================================================================================
Attributes the Moderate scenario's modelled stop-service changes to ATM's
line-to-depot registry as a post-analysis structural compatibility screen. It is
not a vehicle-block, duty-roster or depot-capacity feasibility assessment.

Method
--------------------------------------------------------------------------------
1. Rebuild the stop->serving-lines mapping from the primary GTFS feed
   (self-contained; surface modes only — metro and rail route_types excluded,
   trolleybus retained whatever its route_type coding).
2. Split each stop's optimised delta_trips EQUALLY across its serving surface
   lines (disclosed rule).
3. Split each line's delta EQUALLY across the depots it is assigned to —
   the register can assign a line to MULTIPLE depots, so a first-match rule
   would mis-assign it (disclosed rule).
4. Aggregate freed (donor) and gained (recipient) trips per depot and report:
       internal_coverage = sum_d min(freed_d, gained_d) / total_gained
   (share of recipient gains attributable within the SAME depot as the
   corresponding surrendered equivalents) and the inter-depot transfer requirement in trips/day. Lines absent
   from the depot sheet fall into 'UNASSIGNED', reported, never dropped.

DATA STRUCTURE (verified against the July 2026 workbook revision)
--------------------------------------------------------------------------------
Sheet 'Linee-Deposito': columns nomedepo / Divisione / DES_Modalità /
linea_carto / ambito (values carry trailing spaces — stripped), one row per
line-depot pair. The 'ambito' column is the operator's classification of each
line as 'Urbano' or 'Area Urbana'.

OPERATOR-MANDATED LINE EXCLUSIONS (ATM, July 2026)
--------------------------------------------------------------------------------
Two code families in the GTFS are excluded from the stop->line mapping on
the operator's explicit guidance: line 161 is a placeholder code for buses
temporarily replacing tram routes (roadworks and similar), and the BDQ/Q
codes are Radiobus di Quartiere night on-call services operating only
22:00-02:00 — entirely outside the analytical window [06:00, 22:00), so
they must not absorb shares of daytime trip deltas at the stops they touch.

Reads:   ATM workbook, milan_optimised_<recommended>.csv, primary GTFS
Writes:  depot_balance.csv, thesis_depot_feasibility.png,
         run_parameters_depot_feasibility.json

Usage:
    python depot_feasibility.py
================================================================================
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config

config.validate()

# [INSTANTIATION] verified 2026-07 against the workbook ATM sent. The workbook
# path, sheet and column names are defined once, in config.py.
ATM_WORKBOOK = config.DEPOT_REGISTRY
DEPOT_SHEET = config.DEPOT_REGISTRY_SHEET
COL_DEPOT, COL_LINE, COL_MODE = (config.DEPOT_REGISTRY_DEPOT_COL,
                                 config.DEPOT_REGISTRY_LINE_COL, "DES_Modalità")
# GTFS route_types treated as NON-surface (excluded): 1 metro, 2 rail.
# Everything else (0 tram, 3 bus, 11/800 trolleybus, ...) is retained so the
# filobus lines 90-93 survive whatever coding the feed uses.
NON_SURFACE_ROUTE_TYPES = (1, 2)

# Operator-mandated exclusions (see docstring): exact codes and prefixes.
OPERATOR_EXCLUDED_EXACT = {"161"}
OPERATOR_EXCLUDED_PREFIXES = ("BDQ", "Q")   # Radiobus di Quartiere variants
COL_AMBITO = config.DEPOT_REGISTRY_AMBITO_COL


def load_depot_table():
    df = pd.read_excel(ATM_WORKBOOK, sheet_name=DEPOT_SHEET)
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c].dtype) or pd.api.types.is_object_dtype(df[c].dtype):
            df[c] = df[c].astype("string").str.strip()
    df = df.rename(columns={COL_LINE: "line", COL_DEPOT: "depot",
                            COL_MODE: "mode", COL_AMBITO: "ambito"})
    df["line"] = df["line"].str.upper()
    df = df.dropna(subset=["line", "depot"]).drop_duplicates(["line", "depot"])
    multi = df["line"].value_counts()
    multi = multi[multi > 1]
    print(f"    depot table: {df['line'].nunique()} lines, "
          f"{df['depot'].nunique()} depots; "
          f"{len(multi)} lines assigned to multiple depots "
          "(delta split equally across their depots)")
    # line -> number of depots (for the equal split)
    df["n_depots"] = df.groupby("line")["depot"].transform("count")
    if "ambito" in df.columns:
        print("    ambito classification: "
              + ", ".join(f"{k}={v}" for k, v in
                          df.drop_duplicates("line")["ambito"]
                            .value_counts().items()))
        return df[["line", "depot", "mode", "ambito", "n_depots"]]
    print("    WARNING: 'ambito' column not found — Urbano/Area Urbana "
          "breakdown will be skipped. Is this the July 2026 workbook?")
    df["ambito"] = "unknown"
    return df[["line", "depot", "mode", "ambito", "n_depots"]]


def stop_to_lines():
    """stop_id -> serving surface lines, from the primary feed."""
    gtfs = config.GTFS_FEEDS[config.PRIMARY_FEED]
    routes = pd.read_csv(os.path.join(gtfs, "routes.txt"),
                         usecols=["route_id", "route_short_name", "route_type"])
    print("    route_type distribution in feed: "
          f"{routes['route_type'].value_counts().to_dict()}")
    routes = routes[~routes["route_type"].isin(NON_SURFACE_ROUTE_TYPES)].copy()
    routes["line"] = (routes["route_short_name"].astype(str)
                      .str.strip().str.upper())
    trips = pd.read_csv(os.path.join(gtfs, "trips.txt"),
                        usecols=["route_id", "trip_id"])
    st = pd.read_csv(os.path.join(gtfs, "stop_times.txt"),
                     usecols=["trip_id", "stop_id"], dtype={"stop_id": str})
    # operator-mandated exclusions: replacement code 161 and BDQ/Q
    # night on-call services (outside the analytical window)
    excl = routes["line"].isin(OPERATOR_EXCLUDED_EXACT) | \
        routes["line"].str.startswith(OPERATOR_EXCLUDED_PREFIXES)
    if excl.any():
        print(f"    operator-mandated exclusions applied: "
              f"{sorted(routes.loc[excl, 'line'].unique())}")
    routes = routes[~excl]

    tl = trips.merge(routes[["route_id", "line"]], on="route_id")
    pair = (st.merge(tl[["trip_id", "line"]], on="trip_id")
              [["stop_id", "line"]].drop_duplicates())
    pair["stop_id"] = pair["stop_id"].astype(str).str.strip()
    print(f"    stop->line pairs (surface): {len(pair)} "
          f"({pair['stop_id'].nunique()} stops, {pair['line'].nunique()} lines)")
    return pair


def main():
    print("--- Depot-level feasibility screen "
          f"({config.RECOMMENDED_SCENARIO} scenario) ---")
    print("[1] Loading depot table...")
    depot = load_depot_table()

    print("[2] Loading optimiser output...")
    opt_path = os.path.join(
        config.OUTPUT_DIR,
        f"milan_optimised_{config.RECOMMENDED_SCENARIO.lower()}.csv")
    opt = pd.read_csv(opt_path)
    opt["stop_id"] = opt["stop_id"].astype(str).str.strip()
    moved = opt[opt["delta_trips"].abs() > 1e-9][["stop_id", "role",
                                                  "delta_trips"]]
    print(f"    stops with non-zero delta: {len(moved)}")

    print("[3] Rebuilding stop->serving-lines from primary GTFS...")
    pair = stop_to_lines()

    print("[4] Equal split of delta_trips: stop -> lines -> depots...")
    m = moved.merge(pair, on="stop_id", how="left")
    n_unmatched = m.loc[m["line"].isna(), "stop_id"].nunique()
    if n_unmatched:
        print(f"    NOTE: {n_unmatched} moved stops have no surface line in "
              "the primary feed (metro-only or ID mismatch) -> UNASSIGNED.")
    m["line"] = m["line"].fillna("UNASSIGNED")
    m["n_lines"] = m.groupby("stop_id")["line"].transform("count")
    m["delta_line"] = m["delta_trips"] / m["n_lines"]

    md = m.merge(depot, on="line", how="left")
    missing_lines = md.loc[md["depot"].isna() & (md["line"] != "UNASSIGNED"),
                           "line"].unique()
    if len(missing_lines):
        print(f"    NOTE: {len(missing_lines)} lines carry moved trips but "
              f"are absent from the depot sheet -> UNASSIGNED: "
              f"{sorted(missing_lines)[:15]}")
    md["depot"] = md["depot"].fillna("UNASSIGNED")
    md["n_depots"] = md["n_depots"].fillna(1)
    md["delta_depot"] = md["delta_line"] / md["n_depots"]

    print("[5] Aggregating to depots...")
    g = md.groupby("depot")["delta_depot"]
    bal = pd.DataFrame({
        "gained": g.apply(lambda s: s[s > 0].sum()),
        "freed":  g.apply(lambda s: -s[s < 0].sum()),
    }).fillna(0.0)
    bal["net"] = bal["gained"] - bal["freed"]
    bal["internal"] = np.minimum(bal["gained"], bal["freed"])
    bal = bal.sort_values("net")

    assigned = bal.index != "UNASSIGNED"

    # Compute within-depot coverage over assigned surface gains only; UNASSIGNED
    # service must not inflate the structural compatibility measure.
    surf = bal.loc[assigned]
    total_gained_surface = surf["gained"].sum()
    internal_cov = (surf["internal"].sum() / total_gained_surface
                    if total_gained_surface else 0.0)

    # Inter-depot imbalance under the attribution screen.
    transfer_req = surf["net"].clip(lower=0).sum()

    # Share of freed equivalents not attributable to transferable surface service.
    total_freed = bal["freed"].sum()
    nontransferable_freed = (bal.loc["UNASSIGNED", "freed"]
                             if "UNASSIGNED" in bal.index else 0.0)
    nontransferable_share = (nontransferable_freed / total_freed
                             if total_freed else 0.0)

    # ── ambito breakdown: where do freed / gained trips sit in the
    #    operator's Urbano vs Area Urbana classification? This quantifies the
    #    governance dimension of the reallocation (operator consultation).
    amb = md.copy()
    amb["ambito"] = amb["ambito"].fillna("unassigned")
    g_amb = amb.groupby("ambito")["delta_depot"]
    amb_tab = pd.DataFrame({
        "gained": g_amb.apply(lambda s: s[s > 0].sum()),
        "freed":  g_amb.apply(lambda s: -s[s < 0].sum()),
    }).fillna(0.0).round(1)
    au_gain_share = (amb_tab.loc["Area Urbana", "gained"] /
                     amb_tab["gained"].sum()
                     if "Area Urbana" in amb_tab.index else 0.0)
    au_free_share = (amb_tab.loc["Area Urbana", "freed"] /
                     amb_tab["freed"].sum()
                     if "Area Urbana" in amb_tab.index else 0.0)

    out_csv = os.path.join(config.OUTPUT_DIR, "depot_balance.csv")
    bal.round(1).to_csv(out_csv)
    amb_csv = os.path.join(config.OUTPUT_DIR, "depot_ambito_balance.csv")
    amb_tab.to_csv(amb_csv)

    print("\n=== DEPOT BALANCE (trips/day, "
          f"{config.RECOMMENDED_SCENARIO} scenario) ===")
    print(bal.round(1).to_string())
    print(f"\n  Internal coverage (surface, within-depot):    "
          f"{internal_cov*100:.1f}%")
    print(f"  Inter-depot transfer requirement:             "
          f"{transfer_req:.0f} trips/day")
    print(f"  Non-transferable share of freed capacity      "
          f"(metro/rail or unmatched/excluded service):          "
          f"{nontransferable_share*100:.1f}%  "
          f"({nontransferable_freed:.0f} of {total_freed:.0f} trips/day)")
    print("\n=== AMBITO BREAKDOWN (official operator classification) ===")
    print(amb_tab.to_string())
    print(f"  Share of recipient gains on Area Urbana lines:  "
          f"{au_gain_share*100:.1f}%")
    print(f"  Share of freed trips from Area Urbana lines:    "
          f"{au_free_share*100:.1f}%")

    print("\nSummary:")
    print(f"  Mapping the {config.RECOMMENDED_SCENARIO} scenario's "
          f"reallocation onto ATM's line-to-depot distribution, "
          f"{internal_cov*100:.0f}% of recipient trip gains on the surface "
          f"network are")
    print(f"  attributable within the same depot as the corresponding surrendered "
          f"equivalents; the residual {transfer_req:.0f} equivalents/day indicates "
          f"an inter-depot imbalance requiring further operational")
    print("  assessment; this screen does not identify vehicle or driver transfers "
          "or establish timetable feasibility.")

    plot = bal[assigned]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(plot))))
    colors = ["#1D4ED8" if v > 0 else "#DC2626" for v in plot["net"]]
    ax.barh(plot.index, plot["net"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.1)
    ax.set_xlabel("Net trips/day (blue = depot gains service, "
                  "red = depot surrenders)", fontsize=10)
    ax.set_title(f"Depot-level balance of the {config.RECOMMENDED_SCENARIO} "
                 "reallocation\n"
                 f"internal coverage {internal_cov*100:.0f}% · "
                 f"inter-depot transfer {transfer_req:.0f} trips/day",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout()
    fig_path = os.path.join(config.OUTPUT_DIR, "thesis_depot_feasibility.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()

    rp = config.write_run_parameters("depot_feasibility", extra={
        "scenario": config.RECOMMENDED_SCENARIO,
        "allocation_rule": "delta split equally stop->serving surface lines, "
                           "then equally line->assigned depots (disclosed)",
        "non_surface_route_types_excluded": list(NON_SURFACE_ROUTE_TYPES),
        "internal_coverage_surface": round(float(internal_cov), 4),
        "transfer_requirement_trips_day": round(float(transfer_req), 1),
        "nontransferable_freed_share": round(float(nontransferable_share), 4),
        "nontransferable_freed_trips_day": round(float(nontransferable_freed), 1),
        "n_depots": int(assigned.sum()),
        "operator_excluded_lines": {
            "exact": sorted(OPERATOR_EXCLUDED_EXACT),
            "prefixes": list(OPERATOR_EXCLUDED_PREFIXES),
            "reason": "161 = temporary tram-replacement code; BDQ/Q = "
                      "Radiobus di Quartiere night on-call (22:00-02:00, "
                      "outside the analytical window). Operator guidance, "
                      "ATM, July 2026.",
        },
        "ambito_source": "official ATM classification, workbook rev. July 2026",
        "area_urbana_share_of_gains": round(float(au_gain_share), 4),
        "area_urbana_share_of_freed": round(float(au_free_share), 4),
        "unassigned_gained": round(float(bal.loc["UNASSIGNED", "gained"])
                                   if "UNASSIGNED" in bal.index else 0.0, 1),
        "unassigned_freed": round(float(bal.loc["UNASSIGNED", "freed"])
                                  if "UNASSIGNED" in bal.index else 0.0, 1),
    })
    print("\nOutputs:")
    for p in (out_csv, amb_csv, fig_path, rp):
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()