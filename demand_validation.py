"""
================================================================================
demand_validation.py  —  Metro demand-side consistency check (ATM entrance data)
================================================================================
Compares OBSERVED demand retention (station entrances on holidays/Sundays
relative to weekdays, from ATM's cleaned daily averages) against SCHEDULED
supply retention (hw_wd / hw_daytype from the corrected pipeline) at metro
stations.

METHODOLOGICAL FRAMING
--------------------------------------------------------------------------------
Observed entrances are demand CONDITIONAL ON THE OFFERED SUPPLY: trips
suppressed by inadequate service are invisible in gate counts. The comparison
is therefore a CONSISTENCY CHECK between scheduled service reductions and
realised passenger volumes, not a causal validation. Scope: metro network only;
the result cannot be transferred directly to surface corridors. Labour Day demand
is reported descriptively but excluded from the supply comparison because the
GTFS coverage for that date is severely truncated. Liberation Day 2026 falls on a SATURDAY: its demand retention vs the
weekday baseline conflates the holiday effect with the ordinary weekend
effect, so a vs-Saturday ratio is reported alongside.

DATA STRUCTURE (verified against Tornellati-LineeDepostio_-_Tesista.xlsx)
--------------------------------------------------------------------------------
Sheet 'Tornellati': WIDE layout, four side-by-side blocks (header row 9,
data from row 10, 1-indexed): Weekday | Saturday | Sunday (monthly daily
averages MAR/APR/MAG 2026) | Public Holiday (four dated columns: Easter
Sunday, Easter Monday, 25 April, 1 May). Rows are GATE-GROUPS, aggregated
here to physical stations by name normalisation + SUM. Two aggregation
mechanisms cooperate: the 'lato ...' suffix is stripped by normalise(), and
gate-complexes named after streets (Lambrate Bottini/Pacini, Loreto
Abruzzi/Argentina, Porta Venezia Buenos Aires/Oberdan) are collapsed via the
GATE_MERGE map before summation. The 'Totale complessivo' row is dropped.

STATION MATCHING (audited)
--------------------------------------------------------------------------------
Metro stops are identified by GTFS route_type == 1 (NOT by name tokens —
surface stops at interchanges carry the station name plus line token,
while several metro platforms carry none). Matching is strictest first:
manual alias -> exact key join -> fuzzy fallback. Known
false fuzzy pairings are blocked via FUZZY_BLOCK. Stations outside the
municipal boundary stay UNMATCHED by construction (milan_stops.gpkg is
boundary-clipped) and constitute the documented scope exclusion; gate
complexes spanning two stations (Duomo Cordusio, Duomo San Babila
mezzanines) stay UNMATCHED deliberately as unattributable. Every decision is
written to demand_station_match_review.csv — review it after each run.

Reads:   ATM workbook, milan_stop_hourly.csv, milan_stops.gpkg
Writes:  demand_vs_supply_station.csv        per-station comparison
         demand_station_match_review.csv     name-matching audit (CHECK THIS)
         thesis_demand_validation.png        scatter + gap distribution
         run_parameters_demand_validation.json

Usage:
    python demand_validation.py
================================================================================
"""

import os
import re
import unicodedata
from difflib import get_close_matches

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import config

config.validate()

# ─────────────────────────────────────────────────────────────────────────────
# [INSTANTIATION]  ATM workbook structure (verified 2026-07; edit only if ATM
# resends a restructured file). Column indices are 0-based positions in the
# 'Tornellati' sheet; HEADER_ROW is the 0-based row holding block headers.
# ─────────────────────────────────────────────────────────────────────────────
ATM_WORKBOOK = config.DEPOT_REGISTRY     # same ATM workbook, other sheet
ENTRANCE_SHEET = "Tornellati"
HEADER_ROW = 8
BLOCKS = {
    #        line  station  value columns          value labels
    "wd":  (0, 1, [2, 3, 4],      ["mar", "apr", "mag"]),
    "sat": (6, 7, [8, 9, 10],     ["mar", "apr", "mag"]),
    "sun": (12, 13, [14, 15, 16], ["mar", "apr", "mag"]),
    "hol": (18, 19, [20, 21, 22, 23],
            ["Easter Sunday (5 Apr)", "Easter Monday (6 Apr)",
             "Liberation Day (25 Apr)", "Labour Day (1 May)"]),
}
TOTAL_ROW_LABEL = "Totale complessivo"
MIN_MATCH_SCORE = 0.80               # fuzzy station->stop name acceptance

# ─────────────────────────────────────────────────────────────────────────────
# GATE_MERGE: gate-complexes that are separate turnstile rows of ONE physical
# station, named after streets rather than with the 'lato ...' pattern (which
# normalise() already handles). Applied BEFORE aggregation, so their
# entrances SUM into the station total, exactly like the 'lato' rows.
# Target keys use the supply-side (GTFS-derived) spelling.
# ─────────────────────────────────────────────────────────────────────────────
GATE_MERGE = {
    "lambrate bottini": "lambrate",
    "lambrate pacini": "lambrate",
    "loreto abruzzi": "loreto",
    "loreto argentina": "loreto",
    "porta venezia buenos aires": "p ta venezia",
    "porta venezia oberdan": "p ta venezia",
}

# ─────────────────────────────────────────────────────────────────────────────
# ALIASES: manual station aliases, turnstile station_key -> GTFS station_key,
# for pure naming differences (abbreviations, appendages) where the two rows
# are the SAME single station. Fill ONLY unambiguous mappings, using the
# candidate suggestions printed for each UNMATCHED station (and stored in the
# review CSV).
# Do NOT alias:
#   - stations outside the municipal boundary (Assago, the M2 branch beyond
#     Cascina Gobba, Pero/Rho, Sesto FS/Rondo, Linate in Segrate, ...): they
#     have no supply-side counterpart BY CONSTRUCTION — milan_stops.gpkg is
#     boundary-clipped — and their exclusion is the documented scope;
#   - gate complexes spanning TWO stations (Duomo Cordusio, Duomo San Babila
#     mezzanines): unattributable, documented as such.
# ─────────────────────────────────────────────────────────────────────────────
ALIASES = {
    "amendola fiera": "amendola",
    "missori velasca": "missori",
    "repubblica passante": "repubblica",
    "porta genova": "p ta genova f s",
    "vil s giovanni": "villa s g",
}

# Known FALSE fuzzy pairings, found in the audit: (turnstile_key, supply_key)
# pairs that difflib scores above the cutoff but that are DIFFERENT stations.
# A blocked pair falls through to UNMATCHED (with candidates listed) instead.
FUZZY_BLOCK = {
    ("isola", "piola"),
}


def normalise(name):
    """
    Canonical station key: lower-case, de-accent, strip metro-line tokens,
    gate-side descriptors ('lato ...'), and generic words. Gate-groups of the
    same physical station collapse onto one key (entrances are then SUMMED).
    """
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"\blato\b.*$", " ", s)           # 'affori fnm lato comasina' -> 'affori fnm'
    s = re.sub(r"\bm[1-5]\b", " ", s)
    s = re.sub(r"\b(fs|fn|fnm|stazione|metro)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_entrances():
    """Wide 4-block sheet -> one tidy station-level frame (gate-groups summed)."""
    raw = pd.read_excel(ATM_WORKBOOK, sheet_name=ENTRANCE_SHEET, header=None)
    frames = []
    for block, (lc, sc, vcols, labels) in BLOCKS.items():
        sub = raw.iloc[HEADER_ROW + 1:, [lc, sc] + vcols].copy()
        sub.columns = ["line", "station"] + [f"{block}::{l}" for l in labels]
        sub = sub[(sub["line"] != TOTAL_ROW_LABEL) & sub["station"].notna()]
        for c in sub.columns[2:]:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub["station_key"] = sub["station"].map(normalise)
        # collapse street-named gate-complexes onto their station key, so the
        # groupby below sums their entrances (same treatment as 'lato' rows)
        sub["station_key"] = sub["station_key"].replace(GATE_MERGE)
        numeric_cols = list(sub.columns[2:-1])
        summed = sub.groupby("station_key")[numeric_cols].sum(min_count=1)
        first = sub.groupby("station_key")[["station", "line"]].first()
        agg = summed.join(first)
        frames.append(agg)
    df = pd.concat(frames, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]     # keep first station/line cols
    n_gate = (raw.iloc[HEADER_ROW + 1:, 1].notna()
              & (raw.iloc[HEADER_ROW + 1:, 0] != TOTAL_ROW_LABEL)).sum()
    print(f"    gate-group rows: {n_gate} -> physical stations: {len(df)}")
    return df.reset_index()


def demand_retention(df):
    """Weekday/Saturday/Sunday baselines + per-holiday retention ratios."""
    for b in ("wd", "sat", "sun"):
        cols = [c for c in df.columns if c.startswith(f"{b}::")]
        df[f"ent_{b}"] = df[cols].mean(axis=1)   # mean of monthly daily averages

    df["dem_ret_sun"] = (df["ent_sun"] / df["ent_wd"]).clip(upper=1.5)
    for c in [c for c in df.columns if c.startswith("hol::")]:
        holiday = c.split("::", 1)[1]
        df[f"dem_ret::{holiday}"] = (df[c] / df["ent_wd"]).clip(upper=1.5)
    # Liberation Day fell on a Saturday: also report vs the Saturday baseline
    lib = "hol::Liberation Day (25 Apr)"
    if lib in df.columns:
        df["dem_ret_25apr_vs_sat"] = (df[lib] / df["ent_sat"]).clip(upper=2.0)
    return df


def metro_stop_ids():
    """
    stop_ids served by metro routes in the primary feed, identified by GTFS
    route_type == 1. This is the authoritative metro definition: name-based
    detection is unreliable in the ATM feed, because surface bus/tram stops
    at interchanges are named after the station WITH the line token (e.g.
    'buonarroti m1' is a bus stop), while several metro platforms carry
    plain names WITHOUT a token (SAN BABILA, ISOLA, GERUSALEMME, TRE TORRI).
    """
    gtfs = config.GTFS_FEEDS[config.PRIMARY_FEED]
    routes = pd.read_csv(os.path.join(gtfs, "routes.txt"),
                         usecols=["route_id", "route_type"])
    metro = routes.loc[routes["route_type"] == 1, "route_id"]
    trips = pd.read_csv(os.path.join(gtfs, "trips.txt"),
                        usecols=["route_id", "trip_id"])
    tids = trips.loc[trips["route_id"].isin(metro), "trip_id"]
    st = pd.read_csv(os.path.join(gtfs, "stop_times.txt"),
                     usecols=["trip_id", "stop_id"], dtype={"stop_id": str})
    ids = (st.loc[st["trip_id"].isin(set(tids)), "stop_id"]
             .astype(str).str.strip().unique())
    print(f"    metro stop_ids in feed (route_type=1 across "
          f"{len(metro)} routes): {len(ids)}")
    return set(ids)


def supply_retention():
    """Metro-station supply retention from the corrected pipeline."""
    hr = pd.read_csv(os.path.join(config.OUTPUT_DIR, "milan_stop_hourly.csv"))
    hr = hr[config.operational_hour_filter(hr["hour"])]
    stop = hr.groupby("stop_id").agg(hw_wd=("hw_wd", "median"),
                                     hw_sun=("hw_sun", "median"),
                                     hw_hol=("hw_hol", "median")).reset_index()
    stop["stop_id"] = stop["stop_id"].astype(str).str.strip()

    stops = gpd.read_file(os.path.join(config.OUTPUT_DIR, "milan_stops.gpkg"))
    stops["stop_id"] = stops["stop_id"].astype(str).str.strip()
    if "stop_name" not in stops.columns:
        raise ValueError("milan_stops.gpkg carries no stop_name — cannot match "
                         "stations by name; re-export the builder with names.")

    # metro = stops served by route_type-1 routes, NOT stops with 'M#' in the
    # name (see metro_stop_ids docstring for why name-matching is wrong here)
    ids = metro_stop_ids()
    dropped = sorted(ids - set(stops["stop_id"]))
    if dropped:
        print(f"    NOTE: {len(dropped)} metro stop_ids in the feed are "
              "absent from the analytical network (boundary/zone-match "
              f"exclusion in the builder): {dropped}")

    s = stops[stops["stop_id"].isin(ids)][["stop_id", "stop_name"]] \
            .merge(stop, on="stop_id")
    print(f"    metro platform stops in the analytical network: {len(s)}")
    s["station_key"] = s["stop_name"].map(normalise)

    for src_col, dst in (("hw_sun", "sup_ret_sun"), ("hw_hol", "sup_ret_hol")):
        r = s["hw_wd"] / s[src_col]
        s[dst] = r.replace([np.inf, -np.inf], np.nan).clip(0, 1)
    return (s.groupby("station_key")
             .agg(sup_ret_sun=("sup_ret_sun", "median"),
                  sup_ret_hol=("sup_ret_hol", "median"),
                  n_platforms=("stop_id", "count")).reset_index())


def match_stations(demand, supply):
    """
    Station-name matching, strictest first: (1) manual alias, (2) exact join
    on the normalised key, (3) fuzzy fallback above MIN_MATCH_SCORE, with
    known false pairings suppressed via FUZZY_BLOCK. Every decision is
    audited to demand_station_match_review.csv, and each UNMATCHED station
    carries its three nearest supply-side candidates so the ALIASES dict can
    be completed without re-deriving anything. Returns MATCHED stations only.
    """
    sup_keys = supply["station_key"].tolist()
    sup_set = set(sup_keys)
    rows = []
    for k in demand["station_key"]:
        alias = ALIASES.get(k)
        if alias is not None and alias in sup_set:
            rows.append({"station_key": k, "matched_key": alias,
                         "how": "alias", "candidates": ""})
        elif alias is not None:
            # alias points at a key that does not exist — surface loudly
            rows.append({"station_key": k, "matched_key": None,
                         "how": "ALIAS_TARGET_MISSING",
                         "candidates": alias})
        elif k in sup_set:
            rows.append({"station_key": k, "matched_key": k,
                         "how": "exact", "candidates": ""})
        else:
            hit = get_close_matches(k, sup_keys, n=1, cutoff=MIN_MATCH_SCORE)
            if hit and (k, hit[0]) in FUZZY_BLOCK:
                hit = []                       # known false pairing: suppress
            cands = get_close_matches(k, sup_keys, n=3, cutoff=0.5)
            rows.append({"station_key": k,
                         "matched_key": hit[0] if hit else None,
                         "how": "fuzzy" if hit else "UNMATCHED",
                         "candidates": "; ".join(cands)})
    audit = pd.DataFrame(rows)

    print("    matching: " + ", ".join(f"{k}={v}" for k, v in
          audit["how"].value_counts().items()))
    bad = audit[audit["how"] == "ALIAS_TARGET_MISSING"]
    if len(bad):
        print("    ERROR: aliases pointing at non-existent supply keys — "
              "fix ALIASES:")
        for _, r in bad.iterrows():
            print(f"      '{r['station_key']}' -> '{r['candidates']}'")
    unmatched = audit[audit["how"] == "UNMATCHED"]
    if len(unmatched):
        print(f"    {len(unmatched)} UNMATCHED — nearest supply-side "
              "candidates (empty list = outside the municipal boundary, or "
              "excluded upstream by the builder's zone match):")
        for _, r in unmatched.iterrows():
            print(f"      '{r['station_key']}' -> "
                  f"[{r['candidates'] or 'no candidate above 0.5'}]")

    (audit.merge(demand[["station_key", "station"]], on="station_key")
          .to_csv(os.path.join(config.OUTPUT_DIR,
                               "demand_station_match_review.csv"),
                  index=False))

    matched = audit[audit["matched_key"].notna()]
    return (demand.merge(matched[["station_key", "matched_key"]],
                         on="station_key")
                  .merge(supply, left_on="matched_key",
                         right_on="station_key", suffixes=("", "_sup")))


def main():
    print("--- Demand-side consistency check (ATM metro entrances) ---")
    print("[1] Loading ATM entrance sheet (wide 4-block layout)...")
    ent = load_entrances()
    print("[2] Demand retention per station...")
    dem = demand_retention(ent)
    print("[3] Supply retention per metro station (corrected pipeline)...")
    sup = supply_retention()
    print("[4] Matching station names (alias/exact-first, fuzzy fallback, "
          "audited)...")
    df = match_stations(dem, sup)

    # core-holiday demand retention = mean over the three non-Labour-Day
    # holidays (Labour Day demand kept descriptive only — supply artefact)
    core_cols = [c for c in df.columns if c.startswith("dem_ret::")
                 and "Labour" not in c]
    df["dem_ret_hol"] = df[core_cols].mean(axis=1)
    df["gap_hol"] = df["dem_ret_hol"] - df["sup_ret_hol"]   # >0: entry retention exceeds supply retention
    df["gap_sun"] = df["dem_ret_sun"] - df["sup_ret_sun"]

    out_csv = os.path.join(config.OUTPUT_DIR, "demand_vs_supply_station.csv")
    df.to_csv(out_csv, index=False)

    n = df["dem_ret_hol"].notna() & df["sup_ret_hol"].notna()
    corr = df.loc[n, ["dem_ret_hol", "sup_ret_hol"]].corr().iloc[0, 1]
    overshoot = int((df.loc[n, "gap_hol"] > 0.10).sum())
    lab = "dem_ret::Labour Day (1 May)"

    print("\n=== DEMAND vs SUPPLY (holiday, core three) ===")
    print(f"  Stations compared:                 {int(n.sum())}")
    print(f"  Median demand retention (holiday): {df.loc[n,'dem_ret_hol'].median():.2f}")
    print(f"  Median supply retention (holiday): {df.loc[n,'sup_ret_hol'].median():.2f}")
    print(f"  Median demand retention (Sunday):  {df['dem_ret_sun'].median():.2f}")
    print(f"  Pearson r (demand vs supply):      {corr:.2f}")
    print(f"  Stations with entry retention > supply retention by >10 pp: {overshoot}")
    if "dem_ret_25apr_vs_sat" in df.columns:
        print(f"  25 Apr (Saturday coincidence): median retention vs weekday "
              f"{df['dem_ret::Liberation Day (25 Apr)'].median():.2f}, "
              f"vs Saturday baseline {df['dem_ret_25apr_vs_sat'].median():.2f}")
    if lab in df.columns:
        print(f"  Labour Day demand retention (descriptive only): "
              f"median {df[lab].median():.2f}")

    # figure: scatter + gap distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ax = axes[0]
    ax.scatter(df.loc[n, "sup_ret_hol"], df.loc[n, "dem_ret_hol"],
               s=30, alpha=0.75, color="#1D4ED8", linewidth=0)
    lim = [0, 1.2]
    ax.plot(lim, lim, color="#6B7280", linestyle="--", linewidth=1.2,
            label="demand retention = supply retention")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Scheduled supply retention (hw_wd / hw_hol)", fontsize=10)
    ax.set_ylabel("Observed demand retention (entrances)", fontsize=10)
    ax.set_title("Holiday: observed demand vs scheduled supply\n"
                 "(above the line: entry retention exceeds supply retention)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.set_axisbelow(True); ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(df.loc[n, "gap_hol"].dropna(), bins=25, color="#94A3B8",
            edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlabel("Demand retention − supply retention", fontsize=10)
    ax.set_ylabel("Stations", fontsize=10)
    ax.set_title("Distribution of the demand–supply gap", fontsize=11,
                 fontweight="bold")
    ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
    plt.suptitle("Demand-side consistency check — ATM metro entrances vs "
                 "corrected GTFS supply", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(config.OUTPUT_DIR, "thesis_demand_validation.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()

    rp = config.write_run_parameters("demand_validation", extra={
        "stations_compared": int(n.sum()),
        "median_demand_retention_hol": float(df.loc[n, "dem_ret_hol"].median()),
        "median_supply_retention_hol": float(df.loc[n, "sup_ret_hol"].median()),
        "pearson_r": float(corr),
        "overshoot_stations_gt10pp": overshoot,
        "gate_group_rule": "entrances summed across gate-groups per station: "
                           "'lato' suffixes stripped by normalise(); "
                           "street-named complexes collapsed via GATE_MERGE",
        "aliases_applied": ALIASES,
        "fuzzy_blocked_pairs": sorted(list(FUZZY_BLOCK)),
        "baseline_rule": "weekday baseline = mean of MAR/APR/MAG monthly daily averages",
        "labour_day_treatment": "demand descriptive only; supply comparison "
                                "excluded (GTFS coverage anomaly)",
        "liberation_day_note": "25 Apr 2026 is a Saturday; vs-Saturday ratio "
                               "reported alongside vs-weekday",
        "metro_identification": "GTFS route_type == 1 (name-token detection "
                                "abandoned: surface interchange stops are "
                                "token-named, some platforms are not)",
        "framing": "consistency check conditional on offered supply; metro scope",
    })
    print("\nOutputs:")
    for p in (out_csv, fig_path, rp):
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
