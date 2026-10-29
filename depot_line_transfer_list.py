"""
================================================================================
depot_line_transfer_list.py  —  which lines give and which receive, per depot
                                 (step 2 of the depot extension, thesis §5.6.3)
================================================================================

Translates the STRICT depot-constrained reallocation (every change stays inside
the depot that supplies it) from stops to lines, so that the result can be
discussed with the operator depot by depot.

HOW A STOP-LEVEL CHANGE BECOMES A LINE-LEVEL ONE
--------------------------------------------------------------------------------
The optimiser changes service at stops. Each stop's change is attributed to
the lines serving it and to those lines' depots with the same disclosed rule as
depot_constrained_optimise.py (stop -> line by STOP_LINE_ATTRIBUTION, line ->
depot equal split): donors through their holiday lines, recipients through
their weekday lines.

At one stop, a modelled stop-service equivalent is one departure per day, so
the change attributed to a line at a stop is that line's change in departures
per day AT THAT STOP. A line-level figure therefore cannot be a sum along the
line: one extra trip adds a departure at every stop it serves. The list
reports instead:

  departures_cut_median   typical per-stop reduction at the line's donor stops
  departures_add_median   typical per-stop increase at its recipient stops
  departures_add_max      increase needed to meet every recipient stop's target
  share_of_line_stops     how much of the line the change actually concerns:
                          a whole-line change also alters every other stop it
                          serves, so a low share means a segment-level or
                          short-turn change, not whole trips
  mixed                   the line both gives and receives within the depot:
                          only a segment-level change can deliver it

The PAIRS sheet splits each depot's pool pro rata: service surrendered by
line i goes to line j in proportion to j's share of the depot's gains. Service
is pooled inside a depot, so this is ONE allocation consistent with the
result, not a unique pairing.

WHAT THIS IS NOT
--------------------------------------------------------------------------------
An equity target expressed per line, for the operator's schedulers to test.
It does not model vehicle blocks, running times, interlining or duties, and
it is not a timetable. Units are modelled stop-service equivalents.

OUTPUTS
--------------------------------------------------------------------------------
  depot_line_transfers_<scenario>_<attribution>.xlsx   README, Depots, Lines, Pairs, Stops
  depot_line_transfers_<scenario>_<attribution>_lines.csv / _pairs.csv
  run_parameters_depot_line_transfers.json

Usage:
    python depot_line_transfer_list.py
================================================================================
"""

import os
import numpy as np
import pandas as pd

import config
import optimise
import build_stop_table as bst
import depot_constrained_optimise as dc

config.validate()

SCENARIO = getattr(config, "DEPOT_LIST_SCENARIO", config.RECOMMENDED_SCENARIO)
# Optional override for this list only: "trips" weights each stop's change by
# the lines' actual departures there, rather than splitting it equally.
ATTRIBUTION = getattr(config, "DEPOT_LIST_ATTRIBUTION", None)
if ATTRIBUTION:
    dc.STOP_LINE_ATTRIBUTION = ATTRIBUTION
MIN_PAIR = 1.0          # pairs below this many equivalents/day are not listed
EPS = 1e-9


def _line_attribution(sl, ld):
    """Transferable (stop, line, depot) rows with the product share."""
    sl = sl.copy()
    ok = sl["route_type"].isin(dc.TRANSFERABLE_ROUTE_TYPES) & ~sl["line"].map(dc._excluded)
    m = sl[ok].merge(ld[["line", "depot", "share"]], on="line", suffixes=("_sl", "_ld"))
    m["w"] = m["share_sl"] * m["share_ld"]
    return m


def _spread(amount, A, depots, stop_idx, rows, sign):
    """Distribute a (stop, depot) amount over that stop's lines in that depot."""
    d_idx = {d: k for k, d in enumerate(depots)}
    r = rows.copy()
    r["si"] = r["stop_id"].map(stop_idx)
    r = r.dropna(subset=["si"])
    r["si"] = r["si"].astype(int)
    r["di"] = r["depot"].map(d_idx)
    a = A[r["si"].values, r["di"].values]
    amt = amount[r["si"].values, r["di"].values]
    r["delta"] = sign * np.divide(amt * r["w"].values, a, out=np.zeros(len(r)), where=a > EPS)
    keep = ["stop_id", "line", "depot", "delta"] + (
        ["events_per_day"] if "events_per_day" in r.columns else [])
    return r[r["delta"].abs() > EPS][keep]


def main():
    rate = config.SCENARIOS[SCENARIO]
    print(f"--- Line-level transfer list by depot: {SCENARIO} ({rate:.0%}), strict mode ---")

    stop = optimise.classify(optimise.load_stop_level())
    stop["stop_id"] = stop["stop_id"].astype(str).str.strip()
    ids = stop["stop_id"].tolist()
    stop_idx = {sid: k for k, sid in enumerate(ids)}

    ld = dc.line_depot_shares()
    depots = sorted(ld["depot"].unique())
    sl_hol, sl_wd = dc.stop_line_shares("hol"), dc.stop_line_shares("wd")
    A_don, _, _ = dc.build_attribution(ids, sl_hol, ld, depots)
    sl_rec = dc.recipient_stop_lines(sl_wd, sl_hol)      # same rule as the model
    A_rec, _, _ = dc.build_attribution(ids, sl_rec, ld, depots)

    s, stats, per_depot, det = dc.run_constrained(stop, A_don, A_rec, depots, rate,
                                                  mode="strict", return_detail=True)
    print(f"    reallocated within depots: {stats['reallocated']:,.0f} equivalents/day | "
          f"Gini improvement {stats['gini_improvement_%']:.1f}%")

    # ---- stop x line x depot changes ---------------------------------------
    give = _spread(det["net_give"], A_don, depots, stop_idx, _line_attribution(sl_hol, ld), -1)
    recv = _spread(det["recv"], A_rec, depots, stop_idx, _line_attribution(sl_rec, ld), +1)

    # conservation, depot by depot: what lines give equals what lines receive
    g_d = -give.groupby("depot")["delta"].sum()
    r_d = recv.groupby("depot")["delta"].sum()
    chk = pd.concat([g_d.rename("given"), r_d.rename("received")], axis=1).fillna(0.0)
    bad = chk[(chk["given"] - chk["received"]).abs() > 1e-6]
    assert bad.empty, f"depot balance violated:\n{bad}"
    assert abs(chk["received"].sum() - stats["reallocated"]) < 1e-6, "line totals != reallocation"

    # ---- feasibility: a line cannot lose more departures than it runs ------
    if "events_per_day" in give.columns:
        give = give.rename(columns={"events_per_day": "line_departures_per_day"})
        give["cut_exceeds_line_service"] = (-give["delta"]) > give["line_departures_per_day"] + 1e-6
        n_bad = int(give["cut_exceeds_line_service"].sum())
        print(f"    FEASIBILITY: {n_bad} of {len(give)} stop-line cuts exceed the line's own holiday "
              f"departures at that stop (attribution '{dc.STOP_LINE_ATTRIBUTION}')")
    else:
        give["line_departures_per_day"] = np.nan
        give["cut_exceeds_line_service"] = False
    if "events_per_day" in recv.columns:
        recv = recv.rename(columns={"events_per_day": "line_departures_per_day"})
    # recipients with NO holiday service today, taken from the stop table before
    # the penalty fill: a served stop can also reach or exceed the penalty value
    unserved = dc.unserved_holiday_stops()
    pen = s.loc[s["stop_id"].isin(unserved), "hw_hol"]
    assert (abs(pen - config.NO_SERVICE_PENALTY_MIN) < 1e-9).all(), \
        "unserved stops should carry exactly the penalty headway"
    recv["restores_holiday_service"] = recv["stop_id"].isin(unserved)
    print(f"    stops with no holiday service: {len(unserved)} "
          f"(must equal the penalty count printed by load_stop_level above)")

    # ---- line table -------------------------------------------------------------
    stops_on_line_hol = sl_hol.groupby("line")["stop_id"].nunique()
    stops_on_line_wd = sl_rec.groupby("line")["stop_id"].nunique()
    gl = (give.groupby(["depot", "line"])
              .agg(stops_cut=("stop_id", "nunique"),
                   stops_cut_infeasible=("cut_exceeds_line_service", "sum"),
                   equivalents_removed=("delta", lambda x: -x.sum()),
                   departures_cut_median=("delta", lambda x: -x.median()))
              .reset_index())
    rl = (recv.groupby(["depot", "line"])
              .agg(stops_added=("stop_id", "nunique"),
                   equivalents_added=("delta", "sum"),
                   departures_add_median=("delta", "median"),
                   departures_add_max=("delta", "max"),
                   stops_restoring_service=("restores_holiday_service", "sum"))
              .reset_index())
    lines = gl.merge(rl, on=["depot", "line"], how="outer")
    for c in ("stops_cut", "stops_added", "stops_cut_infeasible", "stops_restoring_service"):
        lines[c] = lines[c].fillna(0).astype(int)
    for c in ("equivalents_removed", "equivalents_added"):
        lines[c] = lines[c].fillna(0.0)
    lines["net_equivalents"] = lines["equivalents_added"] - lines["equivalents_removed"]
    lines["share_of_line_stops_cut"] = lines["stops_cut"] / lines["line"].map(stops_on_line_hol)
    lines["share_of_line_stops_added"] = lines["stops_added"] / lines["line"].map(stops_on_line_wd)
    lines["role"] = np.select(
        [(lines["stops_cut"] > 0) & (lines["stops_added"] > 0), lines["stops_cut"] > 0],
        ["mixed (segment-level change)", "gives"], default="receives")
    rt = sl_rec.drop_duplicates("line").set_index("line")["route_type"]
    lines["mode"] = lines["line"].map(rt).map({"0": "tram", "3": "bus"}).fillna("?")
    if "ambito" in ld.columns:
        lines["ambito"] = lines["line"].map(ld.drop_duplicates("line").set_index("line")["ambito"])
    lines = lines.sort_values(["depot", "net_equivalents"]).reset_index(drop=True)

    # ---- pro-rata pairs inside each depot ----------------------------------
    pairs, unlisted = [], {}
    for d, grp in lines.groupby("depot"):
        G = grp[grp["equivalents_removed"] > EPS].set_index("line")["equivalents_removed"]
        Rr = grp[grp["equivalents_added"] > EPS].set_index("line")["equivalents_added"]
        if G.empty or Rr.empty:
            continue
        for li, g in G.items():
            for lj, r in Rr.items():
                f = g * r / Rr.sum()
                if f < MIN_PAIR:
                    unlisted[d] = unlisted.get(d, 0.0) + f
                if f >= MIN_PAIR:
                    pairs.append({"depot": d, "from_line": li, "to_line": lj,
                                  "equivalents_per_day": f,
                                  "same_line": li == lj})
    pairs = (pd.DataFrame(pairs).sort_values(["depot", "equivalents_per_day"],
                                             ascending=[True, False])
             if pairs else pd.DataFrame(columns=["depot", "from_line", "to_line",
                                                 "equivalents_per_day", "same_line"]))

    # ---- depot summary ------------------------------------------------------
    dep = per_depot.rename(columns={"freed": "freed_equivalents",
                                    "placed": "placed_equivalents",
                                    "returned_to_donors": "returned_equivalents"})
    cnt = lines.groupby("depot")["role"].value_counts().unstack(fill_value=0)
    dep = dep.merge(cnt, left_on="depot", right_index=True, how="left").fillna(0)
    dep["unlisted_pair_flow"] = dep["depot"].map(unlisted).fillna(0.0)
    dep["placed_%"] = np.where(dep["freed_equivalents"] > EPS,
                               dep["placed_equivalents"] / dep["freed_equivalents"] * 100, np.nan)

    # ---- stop sheet with names -----------------------------------------------
    feed = config.GTFS_FEEDS[config.PRIMARY_FEED]
    names = bst._read_gtfs(feed, "stops.txt", usecols=["stop_id", "stop_name"])
    stops_long = (pd.concat([give.assign(change="cut"), recv.assign(change="add")])
                    .merge(names, on="stop_id", how="left")
                    .merge(s[["stop_id", "v_index", "hw_hol", "hw_hol_opt", "role"]],
                           on="stop_id", how="left")
                    .rename(columns={"delta": "departures_per_day"}))

    # ---- write -------------------------------------------------------------------
    out = config.OUTPUT_DIR
    tag = f"{SCENARIO.lower()}_{dc.STOP_LINE_ATTRIBUTION}"
    readme = pd.DataFrame({"note": [
        "A depot-balanced line-level service target for operational scheduling assessment. "
        "Not a timetable.",
        f"Scenario: {SCENARIO}, {rate:.0%} donor reduction, STRICT depot mode: every change stays "
        "inside the depot that supplies it; unplaced service returns to that depot's donors.",
        "Units: modelled stop-service equivalents. At a single stop, one equivalent is one departure "
        "per day. Line totals are sums over stops and are NOT vehicle trips.",
        "departures_*: per-stop departure changes on that line. departures_add_max is the increase "
        "needed to meet every recipient stop's target on the line.",
        "share_of_line_stops_*: fraction of the line's stops concerned. A whole-trip change alters "
        "every stop of the line; a low share indicates a segment-level or short-turn change.",
        "mixed: the line both gives and receives within the depot; only a segment-level change can "
        "deliver it.",
        "stops_cut_infeasible: stops where the cut assigned to the line exceeds the line's own holiday "
        "departures there. Non-zero values are an artefact of the attribution rule, not a real target: "
        "compare with the 'trips' attribution before using the list.",
        "stops_restoring_service: recipient stops with no holiday service at all today (taken from "
        "the stop table before the penalty headway is applied). Their change reads 'restore holiday "
        "service at this stop', not a count of extra departures.",
        "Pairs: each depot's pool split pro rata between giving and receiving lines. One allocation "
        "consistent with the result, not a unique pairing.",
        f"Pairs below {MIN_PAIR:g} equivalent per day are not listed, so the Pairs sheet sums to "
        f"slightly less than each depot's placed total; the unlisted flow per depot is in the Depots "
        f"sheet (unlisted_pair_flow, {sum(unlisted.values()):.1f} equivalents/day in all).",
        ("No stop-line cut exceeds the line's own holiday departures at that stop. "
         if int(give["cut_exceeds_line_service"].sum()) == 0 else
         f"{int(give['cut_exceeds_line_service'].sum())} stop-line cut(s) exceed the line's own "
         "holiday departures at that stop; inspect stops_cut_infeasible. ")
        + "This check is necessary, not sufficient: route continuity, vehicle blocks, running times, "
          "interlining and duties are not verified.",
        "Not a timetable: vehicle blocks, running times, interlining and duties are not modelled. "
        "An equity target for the operator's schedulers to test.",
        f"Attribution: stop->line '{dc.STOP_LINE_ATTRIBUTION}', line->depot equal split; donors via "
        "holiday lines, recipients via weekday lines (holiday lines where a stop has no weekday "
        f"service). Night and interurban lines excluded: {dc.EXCLUDE_NON_URBAN}. "
        "Source: thesis pipeline, §5.6.3 depot extension.",
    ]})
    xlsx = os.path.join(out, f"depot_line_transfers_{tag}.xlsx")
    with pd.ExcelWriter(xlsx) as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        dep.round(1).to_excel(xw, sheet_name="Depots", index=False)
        lines.round(3).to_excel(xw, sheet_name="Lines", index=False)
        pairs.round(2).to_excel(xw, sheet_name="Pairs", index=False)
        stops_long.round(3).to_excel(xw, sheet_name="Stops", index=False)
    lines.to_csv(os.path.join(out, f"depot_line_transfers_{tag}_lines.csv"), index=False)
    pairs.to_csv(os.path.join(out, f"depot_line_transfers_{tag}_pairs.csv"), index=False)

    rp = config.write_run_parameters(f"depot_line_transfers_{dc.STOP_LINE_ATTRIBUTION}", extra={
        "scenario": SCENARIO, "rate": rate, "mode": "strict",
        "stop_line_attribution": dc.STOP_LINE_ATTRIBUTION,
        "reallocated_equivalents": float(stats["reallocated"]),
        "gini_improvement_%": float(stats["gini_improvement_%"]),
        "n_lines_listed": int(len(lines)),
        "n_mixed_lines": int((lines["role"].str.startswith("mixed")).sum()),
        "n_pairs": int(len(pairs)),
        "unlisted_pair_flow_total": float(sum(unlisted.values())),
        "n_infeasible_stop_line_cuts": int(give["cut_exceeds_line_service"].sum()),
    })
    print(f"    lines: {(lines['role'] == 'gives').sum()} give, "
          f"{(lines['role'] == 'receives').sum()} receive, "
          f"{lines['role'].str.startswith('mixed').sum()} mixed | pairs listed: {len(pairs)}")
    print("    depot balances verified: every depot's lines give exactly what its lines receive")
    print("Written:", os.path.basename(xlsx), "| record:", os.path.basename(rp))


if __name__ == "__main__":
    main()
