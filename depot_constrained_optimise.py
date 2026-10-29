"""
================================================================================
depot_constrained_optimise.py  —  budget-neutral reallocation with depot
                                   and mode constraints (thesis §5.6.3)
================================================================================

WHAT THIS ADDS TO optimise.py
--------------------------------------------------------------------------------
optimise.run_scenario pools every donor's freed service into ONE network-wide
pot and allocates it to any recipient. The depot screen of §5.6.2 then asks,
after the fact, how much of that allocation an operator could actually source.
This script moves the depot from an ex-post screen into the optimisation
itself:

  1. MODE RESTRICTION. Only service on transferable lines (by default bus and
     tram; GTFS route_type 3 and 0) can be freed. A donor's metro, rail or
     excluded-code service is never surrendered, so a metro-only donor stop
     gives up nothing.

  2. DEPOT POOLS. Freed service is attributed to the depots that supply the
     donor's lines, giving one pool per depot instead of one network pool.

  3. DEPOT-LOCAL SUPPLY. A recipient can only draw from the depots that supply
     its own lines, in proportion to that attribution. A recipient served only
     by non-transferable lines cannot receive anything, and is reported.

  4. INTER-DEPOT TRANSFERS, three modes (DEPOT_TRANSFER_MODE):
       "strict"     no transfer between depots. Whatever a depot's own
                    recipients cannot absorb is returned to that depot's
                    donors, so budget neutrality holds exactly.
       "allowance"  up to DEPOT_TRANSFER_ALLOWANCE equivalents per day may
                    move between depots after local allocation; the rest is
                    returned to donors.
       "free"       unlimited transfer: the depot restriction on SUPPLY is
                    lifted, but the mode restriction still applies. This is
                    the mode-only benchmark.

Everything else is inherited unchanged from optimise.py: the same stop table
and penalised baseline (load_stop_level), the same donor/recipient roles
(classify), the same static burden weights, the same 8-minute stop-level RECIPIENT modelling cap
(never applied to donors or neutrals), and the same Gini and burden metrics.
Results are therefore directly comparable with the thesis figures.

ATTRIBUTION (departure-weighted rule used in thesis §5.6.3)
--------------------------------------------------------------------------------
  stop -> line   STOP_LINE_ATTRIBUTION = "trips": share proportional to each
                 line's departures at the stop (the §5.6.3 rule). "equal" is
                 retained as an optional alternative for diagnostic comparison.
                 Recipients with no weekday service in the window fall back
                 to their holiday lines.
                 Donors are attributed through their HOLIDAY lines (what they
                 surrender); recipients through their WEEKDAY lines (the lines
                 that would carry extra holiday service), so the 35 stops with
                 no holiday service can still receive.
  line -> depot  equal split across every depot the registry lists for a line.

  a[s, d] = sum over transferable lines l of share(s, l) * share(l, d)

WHAT THIS IS NOT
--------------------------------------------------------------------------------
Still a stop-level equity-target model, measured in modelled stop-service
equivalents, not vehicle trips. It does not model route continuity, vehicle
blocks, interlining, driver duties or depot vehicle counts. It answers: "how
much of the equity gain survives if service may only move where the operator's
depot and mode structure allows?" — a stop-level test (thesis §5.6.3), not a timetable.

INPUTS
--------------------------------------------------------------------------------
  milan_stop_hourly.csv      via optimise.load_stop_level()
  primary GTFS feed          stops served, lines, route types (holiday service)
  DEPOT_REGISTRY             operator line-to-depot file (.xlsx or .csv).
                             NOT redistributed; its column names are set below.

OUTPUTS
--------------------------------------------------------------------------------
  depot_constrained_scenarios.csv    named scenarios, all transfer modes,
                                     side by side with the unconstrained model
  depot_constrained_sweep.csv        dense sweep: unconstrained (residual returned),
                                     allowance and strict, all from the same run
  depot_constrained_unlinked_recipients.csv
                                     recipients with no surface depot link, by reason
  depot_constrained_depots.csv       per-depot pools at the recommended rate
  depot_constrained_line_audit.csv   GTFS lines missing from the registry
  run_parameters_depot_constrained.json

Usage:
    python depot_constrained_optimise.py
================================================================================
"""

import os
import numpy as np
import pandas as pd

import config
import optimise
import build_stop_table as bst

config.validate()

# -----------------------------------------------------------------------------
# PARAMETERS. Read from config.py when present, so they can be moved there
# (with [INSTANTIATION] tags) without editing this file.
# -----------------------------------------------------------------------------
def _cfg(name, default):
    return getattr(config, name, default)

DEPOT_REGISTRY = _cfg("DEPOT_REGISTRY",
                      os.path.join(config.BASE_PATH, "line_depot_registry.xlsx"))
REGISTRY_SHEET = _cfg("DEPOT_REGISTRY_SHEET", 0)
REGISTRY_LINE_COL = _cfg("DEPOT_REGISTRY_LINE_COL", "line")      # e.g. "Linea"
REGISTRY_DEPOT_COL = _cfg("DEPOT_REGISTRY_DEPOT_COL", "depot")   # e.g. "Deposito"
REGISTRY_AMBITO_COL = _cfg("DEPOT_REGISTRY_AMBITO_COL", "ambito")  # optional

# GTFS route_type values whose service can be moved between surface stops
TRANSFERABLE_ROUTE_TYPES = _cfg("TRANSFERABLE_ROUTE_TYPES", {"0", "3"})  # tram, bus
# operator-mandated exclusions (§4.2.4): line 161 and the BDQ / Q Radiobus codes
EXCLUDED_LINE_CODES = _cfg("DEPOT_EXCLUDED_LINE_CODES", {"161"})
EXCLUDED_LINE_PREFIXES = _cfg("DEPOT_EXCLUDED_LINE_PREFIXES", ("BDQ", "Q"))

STOP_LINE_ATTRIBUTION = _cfg("STOP_LINE_ATTRIBUTION", "trips")   # "equal" | "trips"
DEPOT_TRANSFER_MODE = _cfg("DEPOT_TRANSFER_MODE", "strict")      # strict|allowance|free
DEPOT_TRANSFER_ALLOWANCE = float(_cfg("DEPOT_TRANSFER_ALLOWANCE", 2000.0))  # equivalents/day

RATE_MIN, RATE_MAX, RATE_STEP = 0.05, 0.70, 0.01
EPS = 1e-9


def _norm_line(x):
    return str(x).strip().upper()


# Night and interurban lines are left out of the regular urban network by the
# thesis (NON_URBAN_ROUTE_PREFIXES / _EXACT). With this switch on, they can
# neither give nor receive: a night line such as NM1 gaining daytime holiday
# service is not a meaningful operational change.
EXCLUDE_NON_URBAN = bool(_cfg("DEPOT_EXCLUDE_NON_URBAN_LINES", True))
NON_URBAN_PREFIXES = tuple(_cfg("NON_URBAN_ROUTE_PREFIXES", ()))
NON_URBAN_EXACT = {str(x).strip().upper() for x in _cfg("NON_URBAN_ROUTE_EXACT", ())}


def _operator_excluded(code):
    c = _norm_line(code)
    return c in {_norm_line(e) for e in EXCLUDED_LINE_CODES} or \
        any(c.startswith(p) for p in EXCLUDED_LINE_PREFIXES)


def _non_urban(code):
    c = _norm_line(code)
    return c in NON_URBAN_EXACT or any(c.startswith(p) for p in NON_URBAN_PREFIXES)


def _excluded(code):
    return _operator_excluded(code) or (EXCLUDE_NON_URBAN and _non_urban(code))


# =============================================================================
# 1. ATTRIBUTION MATRICES
# =============================================================================
def stop_line_shares(day_type):
    """
    Lines serving each stop inside the operational window of the primary feed,
    on the given day type ("hol" = core holiday dates, "wd" = regular
    weekdays). Returns a long frame [stop_id, line, route_type, share].

    Donors are attributed through their HOLIDAY lines: that is the service
    they actually surrender. Recipients are attributed through their WEEKDAY
    lines: extra holiday service at a stop would run on the lines that serve
    it in ordinary operation. Using holiday lines for recipients would give
    the 35 stops with no holiday service no depot link at all, so the
    worst-served stops could never receive anything — the same structural exclusion avoided by the Phase 4 no-service convention.
    """
    feed = config.GTFS_FEEDS[config.PRIMARY_FEED]
    cal, ids, n_days = bst.reconstruct_calendar(feed)

    # dates of this day type each service_id runs on — the same additive-
    # calendar logic as build_stop_table.hourly_headways, so a trip whose
    # service covers three holiday dates counts three times, not once
    core_set, all_set = set(config.CORE_HOLIDAY_DATES), set(config.ALL_HOLIDAY_DATES)
    if day_type == "wd":
        mask = cal["dayname"].isin(config.WEEKDAY_DAYS) & ~cal["date"].isin(all_set)
    elif day_type == "sun":
        mask = (cal["dayname"] == "Sunday") & ~cal["date"].isin(all_set)
    else:
        mask = cal["date"].isin(core_set)
    n_dates = cal.loc[mask].groupby("service_id")["date"].nunique()

    trips = bst._read_gtfs(feed, "trips.txt", usecols=["trip_id", "route_id", "service_id"])
    trips = trips[trips["service_id"].isin(n_dates.index)]
    routes = bst._read_gtfs(feed, "routes.txt",
                            usecols=["route_id", "route_short_name", "route_type"])
    st = bst._read_gtfs(feed, "stop_times.txt",
                        usecols=["trip_id", "stop_id", "arrival_time"])
    st = st[st["trip_id"].isin(trips["trip_id"])]
    st["hour"] = bst._hour_from_arrival(st["arrival_time"])
    st = st[config.operational_hour_filter(st["hour"])]

    ev = (st.merge(trips[["trip_id", "route_id", "service_id"]], on="trip_id")
            .merge(routes, on="route_id"))
    ev["dates"] = ev["service_id"].map(n_dates)
    ev["line"] = ev["route_short_name"].map(_norm_line)
    ev["stop_id"] = ev["stop_id"].astype(str).str.strip()

    g = (ev.groupby(["stop_id", "line", "route_type"])["dates"].sum()
           .rename("events").reset_index())          # date-weighted stop events
    # the line's own departures per day at the stop on this day type
    g["events_per_day"] = g["events"] / max(n_days[day_type], 1)
    if STOP_LINE_ATTRIBUTION == "trips":
        g["share"] = g["events"] / g.groupby("stop_id")["events"].transform("sum")
    else:
        g["share"] = 1.0 / g.groupby("stop_id")["line"].transform("count")
    return g


def recipient_stop_lines(sl_wd, sl_hol):
    """
    Lines through which a recipient can receive: its weekday lines, and for a
    stop with no weekday service in the window, its holiday lines instead.
    Weekday lines include the stops with no holiday service at all; the
    fallback includes stops served on holidays but not on weekdays, which a
    weekday-only rule would silently exclude. Shares stay per stop, so each
    stop's shares still sum to one.
    """
    fallback = sl_hol[~sl_hol["stop_id"].isin(set(sl_wd["stop_id"]))]
    return pd.concat([sl_wd, fallback], ignore_index=True)


def line_depot_shares():
    """Registry -> long frame [line, depot, share, ambito]; equal split per line."""
    if not os.path.exists(DEPOT_REGISTRY):
        raise FileNotFoundError(
            f"Depot register not found: {DEPOT_REGISTRY}\n"
            "Place ATM's line-to-depot file there, or set DEPOT_REGISTRY in config.py.")
    ext = os.path.splitext(DEPOT_REGISTRY)[1].lower()
    reg = (pd.read_excel(DEPOT_REGISTRY, sheet_name=REGISTRY_SHEET, dtype=str)
           if ext in (".xlsx", ".xls") else pd.read_csv(DEPOT_REGISTRY, dtype=str))
    missing = {REGISTRY_LINE_COL, REGISTRY_DEPOT_COL} - set(reg.columns)
    if missing:
        raise KeyError(
            f"Registry lacks column(s) {missing}. Columns found: {list(reg.columns)}. "
            "Set DEPOT_REGISTRY_LINE_COL / DEPOT_REGISTRY_DEPOT_COL in config.py.")
    cols = [REGISTRY_LINE_COL, REGISTRY_DEPOT_COL] + (
        [REGISTRY_AMBITO_COL] if REGISTRY_AMBITO_COL in reg.columns else [])
    reg = reg[cols].dropna(subset=[REGISTRY_LINE_COL, REGISTRY_DEPOT_COL]).copy()
    reg["line"] = reg[REGISTRY_LINE_COL].map(_norm_line)
    reg["depot"] = reg[REGISTRY_DEPOT_COL].astype(str).str.strip()
    reg = reg.drop_duplicates(["line", "depot"])
    reg["share"] = 1.0 / reg.groupby("line")["depot"].transform("count")
    if REGISTRY_AMBITO_COL in reg.columns:
        reg["ambito"] = reg[REGISTRY_AMBITO_COL].astype(str).str.strip()
    return reg[["line", "depot", "share"] + (["ambito"] if "ambito" in reg.columns else [])]


def build_attribution(stop_ids, sl, ld, depots=None):
    """
    A[s, d]: share of stop s's holiday service that is transferable AND
    supplied by depot d. The row remainder (1 - sum_d A[s, d]) is service on
    metro, rail, excluded codes or lines absent from the registry.
    """
    sl = sl.copy()
    sl["transferable"] = sl["route_type"].isin(TRANSFERABLE_ROUTE_TYPES) & \
        ~sl["line"].map(_excluded)
    in_reg = set(ld["line"])
    sl["in_registry"] = sl["line"].isin(in_reg)

    audit = (sl[sl["transferable"] & ~sl["in_registry"]]
             .groupby("line").agg(stops=("stop_id", "nunique"),
                                  share_sum=("share", "sum")).reset_index()
             .sort_values("share_sum", ascending=False))

    use = sl[sl["transferable"] & sl["in_registry"]]
    m = use.merge(ld[["line", "depot", "share"]], on="line", suffixes=("_sl", "_ld"))
    m["a"] = m["share_sl"] * m["share_ld"]
    depots = depots if depots is not None else sorted(ld["depot"].unique())
    piv = (m.pivot_table(index="stop_id", columns="depot", values="a", aggfunc="sum")
             .reindex(index=stop_ids, columns=depots).fillna(0.0))
    return piv.values, depots, audit


def unserved_holiday_stops():
    """
    Stops with NO holiday service, identified before optimise.load_stop_level
    fills their missing headway with the penalty. Mirrors that function
    exactly (operational-window filter, then the stop-level median): a stop is
    unserved when its median holiday headway is missing, not when it happens
    to equal or exceed the penalty value, which a served stop can also reach.
    """
    df = pd.read_csv(os.path.join(config.OUTPUT_DIR, "milan_stop_hourly.csv"),
                     usecols=["stop_id", "hour", "hw_hol"])
    df = df[config.operational_hour_filter(df["hour"])]
    hw = df.groupby("stop_id")["hw_hol"].median()
    return set(hw[hw.isna()].index.astype(str).str.strip())


def run_unconstrained(stop, rate, rule):
    """optimise.run_scenario under a given overflow rule, restoring config after."""
    prev = config.OVERFLOW_RULE
    try:
        config.OVERFLOW_RULE = rule
        return optimise.run_scenario(stop, rate)
    finally:
        config.OVERFLOW_RULE = prev


LIKE_FOR_LIKE_RULE = "return_residual_to_donors"   # what the constrained modes do


# =============================================================================
# 2. CONSTRAINED ALLOCATION
# =============================================================================
def _allocate(pools, W, headroom, alloc):
    """
    Simultaneous proportional allocation from several pools into recipients
    with a SHARED headroom cap.

      pools     (n_depots,)   remaining supply per depot  (modified in place)
      W         (n_recip, n_depots)  weight of recipient j on depot d
                (burden_j * attribution_jd; zero where j has no link to d)
      headroom  (n_recip,)    max extra equivalents each recipient can absorb
      alloc     (n_recip, n_depots)  running allocation   (modified in place)

    Each round every pool offers its remainder to its eligible recipients in
    proportion to W; recipients whose combined offers exceed their remaining
    headroom are scaled down to fit and become saturated; the unused part
    stays in the pools for the next round. This is the multi-pool
    generalisation of optimise.run_scenario's redistribute-then-cap cascade,
    and reduces to it exactly when there is a single pool.
    """
    active = headroom > EPS
    for _ in range(10_000):
        colsum = (W * active[:, None]).sum(axis=0)
        live = (pools > EPS) & (colsum > EPS)
        if not live.any():
            break
        offer = np.zeros_like(W)
        offer[:, live] = pools[live] * (W[:, live] * active[:, None]) / colsum[live]
        room = headroom - alloc.sum(axis=1)
        want = offer.sum(axis=1)
        scale = np.where(want > room, np.divide(room, want, out=np.zeros_like(room),
                                                where=want > EPS), 1.0)
        take = offer * scale[:, None]
        if take.sum() <= 1e-12:
            break
        alloc += take
        pools -= take.sum(axis=0)
        active = active & ((headroom - alloc.sum(axis=1)) > EPS)
    return alloc


def run_constrained(stop, A_don, A_rec, depots, rate, mode=DEPOT_TRANSFER_MODE,
                    allowance=DEPOT_TRANSFER_ALLOWANCE, return_detail=False):
    """
    Depot- and mode-constrained counterpart of optimise.run_scenario.
    A_don: donor attribution (holiday lines); A_rec: recipient attribution
    (weekday lines). Both are (n_stops, n_depots) on the same depot columns.
    return_detail=True adds a fourth value: the per-stop, per-depot net
    surrender of donors and allocation to recipients (used by the line list).
    """
    s = stop.copy()
    floor = config.MIN_HEADWAY_FLOOR_MIN
    at_floor = config.OPERATIONAL_MINUTES / floor
    donor = (s["role"] == "donor").values
    recip = (s["role"] == "recipient").values
    trips = s["trips_hol"].values.astype(float)

    # ---- supply: only the transferable, depot-attributed share is freed ----
    give = np.zeros((len(s), len(depots)))
    give[donor] = rate * trips[donor, None] * A_don[donor]      # per depot
    freed_by_depot = give.sum(axis=0)
    freed_total = freed_by_depot.sum()
    retained_nontransfer = rate * trips[donor].sum() - freed_total

    # ---- demand: recipients draw only from their own depots ----------------
    R = np.where(recip)[0]
    headroom = np.clip(at_floor - trips[R], 0, None)
    link = A_rec[R]                                              # (n_recip, n_depots)
    no_link = link.sum(axis=1) <= EPS
    W = s["burden_hol"].values[R, None] * link
    alloc = np.zeros_like(link)
    pools = freed_by_depot.copy()

    if mode == "free":
        # supply restriction lifted: one pool, any recipient with a surface link
        one = np.array([pools.sum()], dtype=float)
        a1 = np.zeros((len(R), 1))
        W1 = s["burden_hol"].values[R, None] * (~no_link)[:, None]
        _allocate(one, W1, headroom, a1)
        # depots are deliberately unconstrained here: no within/inter split exists
        within = np.nan; inter = np.nan; returned = max(float(one[0]), 0.0)
        # attribute the single-pool allocation back to depots only for reporting
        alloc_total = a1[:, 0]
        returned_by_depot = pools * (returned / pools.sum()) if pools.sum() > EPS \
            else np.zeros_like(pools)
    else:
        _allocate(pools, W, headroom, alloc)                     # depot-local stage
        within = alloc.sum()
        inter = 0.0
        if mode == "allowance" and allowance > EPS and pools.sum() > EPS:
            # move up to `allowance` from leftover pools to any linked recipient
            budget = float(min(allowance, pools.sum()))
            drawn = pools * (budget / pools.sum())
            pool2 = np.array([budget], dtype=float)
            a2 = np.zeros((len(R), 1))
            W2 = s["burden_hol"].values[R, None] * (~no_link)[:, None]
            _allocate(pool2, W2, headroom - alloc.sum(axis=1), a2)
            used = budget - pool2[0]
            inter = used
            pools -= drawn * (used / budget)                     # charge depots pro rata
            alloc_total = alloc.sum(axis=1) + a2[:, 0]
        else:
            alloc_total = alloc.sum(axis=1)
        returned_by_depot = np.clip(pools, 0.0, None)   # drop float noise below zero
        returned = float(returned_by_depot.sum())

    # ---- return unplaced supply to the donors of each depot, pro rata -----
    surrendered = give.copy()
    back = np.zeros(len(s))
    for d in range(len(depots)):
        tot = surrendered[:, d].sum()
        if tot > EPS and returned_by_depot[d] > EPS:
            back += returned_by_depot[d] * surrendered[:, d] / tot

    s["trips_hol_opt"] = trips
    s.loc[donor, "trips_hol_opt"] = trips[donor] - give[donor].sum(axis=1) + back[donor]
    s.loc[recip, "trips_hol_opt"] = trips[recip] + alloc_total

    s["hw_hol_opt"] = config.OPERATIONAL_MINUTES / s["trips_hol_opt"]
    s.loc[recip, "hw_hol_opt"] = s.loc[recip, "hw_hol_opt"].clip(lower=floor)  # recipients only
    s["burden_hol_opt"] = s["v_index"] * s["hw_hol_opt"]

    before, after = s["trips_hol"].sum(), s["trips_hol_opt"].sum()
    if abs(before - after) > config.BUDGET_NEUTRALITY_TOLERANCE:
        raise AssertionError(f"Conservation violated: {before:.2f} -> {after:.2f}")

    gb, ga = optimise.gini(s["burden_hol"].values), optimise.gini(s["burden_hol_opt"].values)
    bb, ba = s["burden_hol"].mean(), s["burden_hol_opt"].mean()
    rmask, dmask = s["role"] == "recipient", s["role"] == "donor"
    stats = {
        "rate": rate, "mode": mode,
        "freed_surface": freed_total,
        "retained_nontransferable": retained_nontransfer,
        "allocated_within_depot": within,
        "allocated_inter_depot": inter,
        "returned_to_donors": returned,
        "reallocated": float(alloc_total.sum()),
        "recipients_without_surface_link": int(no_link.sum()),
        "n_recipients_saturated": int((headroom - alloc_total <= 1e-6).sum()),
        "gini_improvement_%": (gb - ga) / gb * 100,
        "burden_reduction_%": (bb - ba) / bb * 100,
        "recip_improvement_min": s.loc[rmask, "hw_hol"].mean() - s.loc[rmask, "hw_hol_opt"].mean(),
        "donor_hw_change_min": s.loc[dmask, "hw_hol_opt"].mean() - s.loc[dmask, "hw_hol"].mean(),
        "budget_neutral": True,          # conservation asserted above
    }
    per_depot = pd.DataFrame({
        "depot": depots, "freed": freed_by_depot,
        "returned_to_donors": returned_by_depot,
        "placed": freed_by_depot - returned_by_depot,
    })
    if not return_detail:
        return s, stats, per_depot
    # net surrender per donor stop and depot, after the pro-rata return
    net_give = give.copy()
    for d in range(len(depots)):
        tot = give[:, d].sum()
        if tot > EPS:
            net_give[:, d] -= returned_by_depot[d] * give[:, d] / tot
    recv = np.zeros((len(s), len(depots)))
    if mode == "strict":
        recv[R] = alloc                               # depot-local allocation only
    detail = {"net_give": net_give, "recv": recv, "recip_index": R}
    return s, stats, per_depot, detail


# =============================================================================
# 3. DRIVER
# =============================================================================
def _knee(r, g):
    m = np.gradient(g, r)
    early = m[:5].mean()
    hit = np.where(m < 0.25 * early)[0]
    return float(r[hit[0]]) if len(hit) else float(r[-1])


def main():
    print("--- Depot- and mode-constrained reallocation (§5.6.3) ---")
    stop = optimise.classify(optimise.load_stop_level())
    stop["stop_id"] = stop["stop_id"].astype(str).str.strip()

    print("[1] Attribution: stop -> line (holiday service), line -> depot")
    ld = line_depot_shares()
    depots = sorted(ld["depot"].unique())
    ids_ = stop["stop_id"].tolist()
    sl_hol, sl_wd = stop_line_shares("hol"), stop_line_shares("wd")
    sl_rec = recipient_stop_lines(sl_wd, sl_hol)
    A_don, _, audit = build_attribution(ids_, sl_hol, ld, depots)
    A_rec, _, audit_rec = build_attribution(ids_, sl_rec, ld, depots)
    audit = (pd.concat([audit.assign(side="donor (holiday lines)"),
                        audit_rec.assign(side="recipient (weekday, else holiday lines)")])
             .reset_index(drop=True))
    donors = (stop["role"] == "donor").values
    recips = (stop["role"] == "recipient").values
    tr = stop["trips_hol"].values
    print(f"    depots: {len(depots)} | transferable share of donor holiday service: "
          f"{(A_don[donors].sum(axis=1) * tr[donors]).sum() / tr[donors].sum():.1%}")
    print(f"    recipients with no surface depot link: "
          f"{int((A_rec[recips].sum(axis=1) <= EPS).sum())} of {int(recips.sum())}")
    fallback_stops = set(sl_rec["stop_id"]) - set(sl_wd["stop_id"])
    print(f"    recipients linked through holiday lines (no weekday service in window): "
          f"{int(stop.loc[recips, 'stop_id'].isin(fallback_stops).sum())} | "
          f"night/interurban lines excluded: {EXCLUDE_NON_URBAN}")
    n_gap = audit["line"].nunique() if len(audit) else 0
    if n_gap:
        print(f"    WARNING: {n_gap} transferable GTFS line(s) absent from the registry "
              f"(treated as non-transferable). See depot_constrained_line_audit.csv")

    # why each unlinked recipient has no surface depot link
    in_reg = set(ld["line"])
    unlinked = []
    for sid in stop.loc[recips & (A_rec.sum(axis=1) <= EPS), "stop_id"]:
        g = sl_rec[sl_rec["stop_id"] == sid]
        surf = g["route_type"].isin(TRANSFERABLE_ROUTE_TYPES)
        op_ex = g["line"].map(_operator_excluded) if len(g) else pd.Series(dtype=bool)
        nu = g["line"].map(lambda c: EXCLUDE_NON_URBAN and _non_urban(c)) if len(g) else pd.Series(dtype=bool)
        if g.empty:
            why = "no weekday or holiday service in the window"
        elif (surf & ~op_ex & ~nu & ~g["line"].isin(in_reg)).any():
            why = "surface line(s) missing from the register"
        elif (surf & nu).any():
            why = "only night or interurban lines"
        elif (surf & op_ex).any():
            why = "only operator-excluded surface codes"
        else:
            why = "metro or rail only"
        unlinked.append({"stop_id": sid, "reason": why,
                         "lines": ", ".join(sorted(g["line"].unique()))})
    unlinked = pd.DataFrame(unlinked, columns=["stop_id", "reason", "lines"])
    reasons = unlinked["reason"].value_counts().to_dict()
    for why, n in reasons.items():
        print(f"      {n:3d}  {why}")

    print("[2] Named scenarios: unconstrained vs strict / allowance / free")
    rows = []
    for name, rate in config.SCENARIOS.items():
        _, base = optimise.run_scenario(stop, rate)             # principal unconstrained rule
        _, like = run_unconstrained(stop, rate, LIKE_FOR_LIKE_RULE)
        row = {"scenario": name, "rate": rate,
               "gini_unconstrained_%": base["gini_improvement_%"],
               "unconstrained_budget_neutral": bool(base["budget_neutral"]),
               "gini_unconstrained_residual_%": like["gini_improvement_%"],
               "burden_red_unconstrained_%": base["burden_reduction_%"]}
        for mode in ("strict", "allowance", "free"):
            _, st, _ = run_constrained(stop, A_don, A_rec, depots, rate, mode=mode)
            row[f"gini_{mode}_%"] = st["gini_improvement_%"]
            row[f"reallocated_{mode}"] = st["reallocated"]
            row[f"returned_{mode}"] = st["returned_to_donors"]
            # like-for-like: both sides return unplaced service to donors
            row[f"retained_{mode}_share_of_gain"] = (
                st["gini_improvement_%"] / like["gini_improvement_%"]
                if like["gini_improvement_%"] else np.nan)
        rows.append(row)
    scen = pd.DataFrame(rows)
    print(scen[["scenario", "rate", "gini_unconstrained_%", "gini_unconstrained_residual_%",
                "gini_free_%", "gini_allowance_%", "gini_strict_%"]].round(2).to_string(index=False))
    if not scen["unconstrained_budget_neutral"].all():
        print("    note: under the principal redistribute-then-cap rule, unconstrained is not budget-neutral at "
              + ", ".join(scen.loc[~scen["unconstrained_budget_neutral"], "scenario"])
              + "; the retained shares compare against the residual-return variant")

    print("[3] Dense sweep: unconstrained (residual returned), allowance, strict")
    rates = np.round(np.arange(RATE_MIN, RATE_MAX + 1e-9, RATE_STEP), 4)
    sweep = []
    for rate in rates:
        _, u = run_unconstrained(stop, float(rate), LIKE_FOR_LIKE_RULE)
        sweep.append({**u, "rate": float(rate), "mode": "unconstrained_residual"})
        for mode in ("strict", "allowance"):
            _, st, _ = run_constrained(stop, A_don, A_rec, depots, float(rate), mode=mode)
            sweep.append(st)
    sweep = pd.DataFrame(sweep)
    summary = {}
    for mode in ("unconstrained_residual", "allowance", "strict"):
        c = sweep[sweep["mode"] == mode].reset_index(drop=True)
        pk = c.loc[c["gini_improvement_%"].idxmax()]
        summary[mode] = {"peak_rate": float(pk["rate"]),
                         "peak_gini_%": float(pk["gini_improvement_%"]),
                         "knee_rate": _knee(c["rate"].values, c["gini_improvement_%"].values)}
        print(f"    {mode:22s}: ceiling {pk['rate']:.0%} (Gini {pk['gini_improvement_%']:.1f}%), "
              f"knee ~{summary[mode]['knee_rate']:.0%}")

    rec = config.SCENARIOS[config.RECOMMENDED_SCENARIO]
    _, _, per_depot = run_constrained(stop, A_don, A_rec, depots, rec, mode=DEPOT_TRANSFER_MODE)

    out = config.OUTPUT_DIR
    scen.to_csv(os.path.join(out, "depot_constrained_scenarios.csv"), index=False)
    sweep.to_csv(os.path.join(out, "depot_constrained_sweep.csv"), index=False)
    per_depot.to_csv(os.path.join(out, "depot_constrained_depots.csv"), index=False)
    audit.to_csv(os.path.join(out, "depot_constrained_line_audit.csv"), index=False)
    unlinked.to_csv(os.path.join(out, "depot_constrained_unlinked_recipients.csv"), index=False)
    rp = config.write_run_parameters("depot_constrained", extra={
        "depot_registry": os.path.basename(DEPOT_REGISTRY),
        "stop_line_attribution": STOP_LINE_ATTRIBUTION,
        "transferable_route_types": sorted(TRANSFERABLE_ROUTE_TYPES),
        "transfer_mode_for_depot_table": DEPOT_TRANSFER_MODE,
        "inter_depot_allowance": DEPOT_TRANSFER_ALLOWANCE,
        "n_depots": len(depots),
        "n_registry_gaps": int(n_gap),
        "unlinked_recipients_by_reason": reasons,
        "exclude_non_urban_lines": EXCLUDE_NON_URBAN,
        "like_for_like_unconstrained_rule": LIKE_FOR_LIKE_RULE,
        "sweep_summary": summary,
    })
    print("Outputs written; record:", os.path.basename(rp))


if __name__ == "__main__":
    main()
