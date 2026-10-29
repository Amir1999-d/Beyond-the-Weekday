"""
================================================================================
optimise.py  —  Floored "Robin Hood" reallocation  (Phase 5)
================================================================================
Budget-neutral frequency redistribution on the holiday service level.

Trips are trimmed from low-vulnerability, high-frequency DONOR stops and
reallocated to high-vulnerability, low-frequency RECIPIENT stops, weighted by
current burden, subject to a MINIMUM HEADWAY FLOOR on recipients.

The floor + overflow rule is what produces the "functional reallocation
ceiling" (thesis contribution #3): as the donor-reduction rate rises, more
recipients saturate at the floor, so the marginal equity gain diminishes and
the Gini-improvement curve bends.

Reads:   milan_stop_hourly.csv   (from build_stop_table.py)
Writes:  optimise_scenarios.csv             (one row per scenario)
         milan_optimised_<recommended>.csv  (stop-level detail, recommended scenario)
         run_parameters_optimise.json

All parameters come from config.py.

Usage:
    python optimise.py
================================================================================
"""

import os
import numpy as np
import pandas as pd

import config

config.validate()


# -----------------------------------------------------------------------------
# inequality helpers
# -----------------------------------------------------------------------------
def gini(arr):
    """Gini coefficient of a non-negative array."""
    arr = np.sort(np.asarray(arr, dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    cum = np.cumsum(arr)
    return (2.0 * np.sum(np.arange(1, n + 1) * arr) - (n + 1) * cum[-1]) / (n * cum[-1])


def lorenz_coords(arr):
    """Cumulative-population vs cumulative-burden coordinates for a Lorenz curve."""
    arr = np.sort(np.asarray(arr, dtype=float))
    pop = np.linspace(0, 1, len(arr) + 1)
    burden = np.concatenate([[0], np.cumsum(arr) / arr.sum()])
    return pop, burden


# -----------------------------------------------------------------------------
# 1. load + aggregate to stop level
# -----------------------------------------------------------------------------
def load_stop_level():
    print("[1] Loading stop-hourly table and aggregating to stop level...")
    path = os.path.join(config.OUTPUT_DIR, "milan_stop_hourly.csv")
    df = pd.read_csv(path)
    df = df[config.operational_hour_filter(df["hour"])].copy()

    stop = df.groupby("stop_id").agg(
        v_index=("v_index", "mean"),
        hw_wd=("hw_wd", "median"),
        hw_hol=("hw_hol", "median"),
    ).reset_index()

    # No-service rule: at the optimisation boundary a single continuous value is
    # required, so unserved holiday stops take the disclosed penalty (NaN -> penalty).
    n_unserved = stop["hw_hol"].isna().sum()
    stop["hw_hol"] = stop["hw_hol"].fillna(config.NO_SERVICE_PENALTY_MIN)
    print(f"    Stops: {len(stop)}  (unserved holiday stops set to "
          f"{config.NO_SERVICE_PENALTY_MIN} min penalty: {n_unserved})")

    # Burden and trips/day (continuous proxy: trips = operational_minutes / headway)
    stop["burden_hol"] = stop["v_index"] * stop["hw_hol"]
    stop["trips_hol"] = config.OPERATIONAL_MINUTES / stop["hw_hol"]
    return stop


# -----------------------------------------------------------------------------
# 2. classify donor / recipient / neutral
# -----------------------------------------------------------------------------
def classify(stop):
    print("[2] Classifying donor / recipient / neutral...")
    v_lo = stop["v_index"].quantile(config.ROLE_VULN_LOW_Q)
    v_hi = stop["v_index"].quantile(config.ROLE_VULN_HIGH_Q)
    hw_lo = stop["hw_hol"].quantile(config.ROLE_HW_LOW_Q)
    hw_hi = stop["hw_hol"].quantile(config.ROLE_HW_HIGH_Q)

    stop["role"] = "neutral"
    # donors: low vulnerability AND already high frequency (low headway)
    stop.loc[(stop["v_index"] <= v_lo) & (stop["hw_hol"] <= hw_lo), "role"] = "donor"
    # recipients: high vulnerability AND poorly served (high headway)
    stop.loc[(stop["v_index"] >= v_hi) & (stop["hw_hol"] >= hw_hi), "role"] = "recipient"

    counts = stop["role"].value_counts().to_dict()
    print(f"    donors={counts.get('donor',0)}, "
          f"recipients={counts.get('recipient',0)}, "
          f"neutral={counts.get('neutral',0)}")
    return stop


# -----------------------------------------------------------------------------
# 3. one scenario, with recipient headway cap + redistribute_then_cap
# -----------------------------------------------------------------------------
def run_scenario(stop, rate, dynamic_burden=False):
    """
    Returns (scenario_frame, stats_dict).

    Floor logic (redistribute_then_cap):
      - Trim `rate` of each donor's trips -> freed pool.
      - Recipients ranked by burden_hol (highest first) receive freed trips, but
        each recipient's post headway cannot go below MIN_HEADWAY_FLOOR_MIN, i.e.
        its trips cannot exceed trips_at_floor = operational_minutes / floor.
      - A recipient that would exceed the floor is capped; its unused allocation
        cascades to the next-priority unsaturated recipient.
      - Once all recipients are saturated, the residual is handled by
        config.OVERFLOW_RULE: under "redistribute_then_cap" it is recorded as
        unallocated (trips leave the system; strict budget neutrality no longer
        holds, and budget_neutral reports False); under
        "return_residual_to_donors" it is returned to donors pro-rata, so total
        trips are conserved exactly at every rate.

    Burden-weighting modes:
      - dynamic_burden=False (default): weights stay at the recipient's initial
        burden_hol throughout the cascade. Simpler and deterministic.
      - dynamic_burden=True: each iteration recomputes the active recipient's
        *current* burden from its running allocation
        (current_burden = v_index * current_hw), redirecting later iterations
        toward recipients that remain poorly served — the mathematically more
        refined formulation. The two modes are compared empirically by
        sensitivity_dynamic_burden.py; their agreement or divergence is a
        computed result of that script, not an assumption made here.
    """
    s = stop.copy()
    floor = config.MIN_HEADWAY_FLOOR_MIN
    trips_at_floor = config.OPERATIONAL_MINUTES / floor

    donor = s["role"] == "donor"
    recip = s["role"] == "recipient"

    # default: everyone keeps current trips
    s["trips_hol_opt"] = s["trips_hol"]

    # donors give up `rate` of their trips
    s.loc[donor, "trips_hol_opt"] = s.loc[donor, "trips_hol"] * (1 - rate)
    freed = (s.loc[donor, "trips_hol"] - s.loc[donor, "trips_hol_opt"]).sum()

    # recipients, highest burden first
    r = s.loc[recip, ["trips_hol", "burden_hol", "v_index"]].copy()
    r = r.sort_values("burden_hol", ascending=False)
    # headroom = how many extra trips each recipient can absorb before the floor
    r["headroom"] = (trips_at_floor - r["trips_hol"]).clip(lower=0)

    remaining = freed
    allocated = pd.Series(0.0, index=r.index)
    # weight by burden share, but cap at headroom; cascade residual
    # iterative because capping one recipient frees its excess for others
    active = r.index.tolist()
    n_saturated = 0
    safety = 0
    while remaining > 1e-9 and active and safety < 10_000:
        safety += 1

        # ── weight choice: static (initial burden) vs dynamic (current burden)
        if dynamic_burden:
            # current trips after this loop's running allocation
            current_trips = r.loc[active, "trips_hol"] + allocated.loc[active]
            # current headway from current trips (cannot fall below floor)
            current_hw = (config.OPERATIONAL_MINUTES /
                          current_trips.clip(lower=trips_at_floor / 1e6))
            current_hw = current_hw.clip(lower=floor)
            weights = r.loc[active, "v_index"] * current_hw
        else:
            weights = r.loc[active, "burden_hol"]

        wsum = weights.sum()
        if wsum <= 0:
            break
        give = remaining * (weights / wsum)
        # cap each at its remaining headroom
        room = r.loc[active, "headroom"] - allocated.loc[active]
        take = np.minimum(give, room)
        allocated.loc[active] += take
        remaining -= take.sum()
        # drop newly-saturated recipients from the active set
        newly_full = [i for i in active if (r.loc[i, "headroom"] - allocated.loc[i]) <= 1e-9]
        n_saturated += len(newly_full)
        active = [i for i in active if i not in newly_full]
        if take.sum() <= 1e-12:   # nothing more can be absorbed
            break

    unallocated = max(remaining, 0.0)
    s.loc[r.index, "trips_hol_opt"] = r["trips_hol"] + allocated

    # ── OVERFLOW RULE ────────────────────────────────────────────────────────
    # "redistribute_then_cap": unallocated trips leave the system — the network
    #     loses service and the scenario is NOT strictly budget-neutral past
    #     the point where all recipients saturate. Reported, never hidden.
    # "return_residual_to_donors": unallocated trips are handed back to donors
    #     pro-rata to what each surrendered, so total trips are conserved
    #     EXACTLY at every rate and the ceiling curve isolates the pure
    #     saturation mechanism from the trip-destruction artefact.
    if config.OVERFLOW_RULE == "return_residual_to_donors" and unallocated > 1e-12:
        surrendered = s.loc[donor, "trips_hol"] - s.loc[donor, "trips_hol_opt"]
        total_surrendered = surrendered.sum()
        if total_surrendered > 0:
            s.loc[donor, "trips_hol_opt"] += unallocated * (surrendered /
                                                            total_surrendered)
            unallocated = 0.0

    # recompute headway + burden after reallocation
    s["hw_hol_opt"] = config.OPERATIONAL_MINUTES / s["trips_hol_opt"]
    # The floor is a RECIPIENT constraint: it bounds how much additional service
    # a recipient can absorb. It is not a property of the network, so it must not
    # be applied to donors or neutrals. A donor whose holiday headway is already
    # below the floor (e.g. a high-frequency metro stop) would otherwise be
    # reported AT the floor after reallocation, raising its post-reallocation
    # burden with no corresponding trip change and compressing the burden
    # distribution from below — which lowers the Gini independently of the
    # reallocation itself. For recipients the clip is redundant (headroom already
    # caps them) and is retained only as a numerical guard.
    s.loc[recip, "hw_hol_opt"] = s.loc[recip, "hw_hol_opt"].clip(lower=floor)
    s["burden_hol_opt"] = s["v_index"] * s["hw_hol_opt"]

    # ── BUDGET ACCOUNTING ────────────────────────────────────────────────────
    # Two distinct checks, kept separate on purpose:
    #   conservation_ok : internal invariant of the algorithm's bookkeeping —
    #                     trips_after must equal trips_before - unallocated.
    #                     A failure here is a CODE bug.
    #   budget_neutral  : the thesis definition:
    #                     sum of donor trips freed == sum of recipient trips
    #                     gained, i.e. no trips leave the system. This is
    #                     False by construction under redistribute_then_cap
    #                     once unallocated > tolerance — a scenario property,
    #                     not a bug, and it must be reported as such.
    reallocated = allocated.sum()
    trips_before = s["trips_hol"].sum()
    trips_after = s["trips_hol_opt"].sum()
    conservation_ok = abs((trips_before - trips_after) - unallocated) \
        < config.BUDGET_NEUTRALITY_TOLERANCE
    if not conservation_ok:
        raise AssertionError(
            f"Trip conservation violated: before={trips_before:.2f}, "
            f"after={trips_after:.2f}, unallocated={unallocated:.2f}. "
            "This indicates a bookkeeping bug in run_scenario.")
    budget_ok = unallocated < config.BUDGET_NEUTRALITY_TOLERANCE

    b_before = s["burden_hol"].mean()
    b_after = s["burden_hol_opt"].mean()
    g_before = gini(s["burden_hol"].values)
    g_after = gini(s["burden_hol_opt"].values)

    recip_hw_before = s.loc[recip, "hw_hol"].mean()
    recip_hw_after = s.loc[recip, "hw_hol_opt"].mean()
    donor_hw_before = s.loc[donor, "hw_hol"].mean()
    donor_hw_after = s.loc[donor, "hw_hol_opt"].mean()

    stats = {
        "rate": rate,
        "freed_trips": freed,
        "reallocated_trips": reallocated,
        "unallocated_trips": unallocated,
        "n_recipients_saturated": int(n_saturated),
        "burden_before": b_before,
        "burden_after": b_after,
        "burden_reduction_%": (b_before - b_after) / b_before * 100,
        "gini_before": g_before,
        "gini_after": g_after,
        "gini_improvement_%": (g_before - g_after) / g_before * 100,
        "recip_hw_before": recip_hw_before,
        "recip_hw_after": recip_hw_after,
        "recip_improvement_min": recip_hw_before - recip_hw_after,
        "donor_hw_before": donor_hw_before,
        "donor_hw_after": donor_hw_after,
        "budget_neutral": budget_ok,
    }
    return s, stats


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    print("--- Floored Robin Hood optimiser (Phase 5) ---")
    stop = load_stop_level()
    stop = classify(stop)

    rows = []
    scenario_frames = {}
    print("\n[3] Running scenarios (with "
          f"{config.MIN_HEADWAY_FLOOR_MIN}-min recipient modelling cap, "
          f"rule='{config.OVERFLOW_RULE}')...")
    for name, rate in config.SCENARIOS.items():
        s, stats = run_scenario(stop, rate)
        stats["scenario"] = name
        rows.append(stats)
        scenario_frames[name] = s
        print(f"    {name:14s} rate={rate:.0%}  "
              f"Gini -{stats['gini_improvement_%']:5.1f}%  "
              f"burden -{stats['burden_reduction_%']:5.1f}%  "
              f"recip -{stats['recip_improvement_min']:4.1f}min  "
              f"saturated={stats['n_recipients_saturated']}  "
              f"unalloc={stats['unallocated_trips']:.0f}  "
              f"{'budget-neutral' if stats['budget_neutral'] else 'NOT neutral (unallocated trips leave system)'}")

    res = pd.DataFrame(rows)[[
        "scenario", "rate", "freed_trips", "reallocated_trips", "unallocated_trips",
        "n_recipients_saturated", "burden_reduction_%", "gini_before", "gini_after",
        "gini_improvement_%", "recip_improvement_min", "budget_neutral",
    ]]

    # ---- the functional-ceiling check -------------------------------------
    print("\n" + "=" * 70)
    print("FUNCTIONAL REALLOCATION CEILING — does Gini gain bend back?")
    print("=" * 70)
    print(res[["scenario", "rate", "gini_improvement_%",
               "n_recipients_saturated", "unallocated_trips"]].to_string(index=False))
    # Turning-point check over budget-neutral scenarios only
    # A scenario whose recipients have all saturated sheds trips (unallocated > 0)
    # and is therefore not comparable on equity grounds. The ceiling claim is that
    # the curve turns over WITHIN the budget-neutral range, not that the
    # recommended rate beats the most aggressive one.
    ok = res[res["unallocated_trips"] < config.BUDGET_NEUTRALITY_TOLERANCE]
    dropped = res[res["unallocated_trips"] >= config.BUDGET_NEUTRALITY_TOLERANCE]

    best = ok.loc[ok["gini_improvement_%"].idxmax()]
    print(f"\nMax Gini improvement at: {best['scenario']} "
          f"({best['rate']:.0%}) = {best['gini_improvement_%']:.1f}%")

    turns_over = best.name != ok.index[-1]
    if turns_over:
        last = ok.iloc[-1]
        print(f"Numerical ceiling candidate: the curve peaks at {best['scenario']} "
              f"({best['rate']:.0%}) = {best['gini_improvement_%']:.1f}% and declines to "
              f"{last['gini_improvement_%']:.1f}% by {last['rate']:.0%} — "
              "diminishing returns present within the budget-neutral range.")
    else:
        print("NO INTERIOR CEILING: Gini improvement is still rising at the highest "
              "budget-neutral rate. The ceiling claim is not supported by these scenarios; "
              "consult the continuous sweep before reporting.")

    for _, r in dropped.iterrows():
        print(f"  note: {r['scenario']} ({r['rate']:.0%}) excluded from the ceiling "
              f"comparison — {r['unallocated_trips']:,.0f} trips unallocated, "
              "so the scenario is not budget-neutral.")
    print("=" * 70)

    # ---- exports ----------------------------------------------------------
    res_path = os.path.join(config.OUTPUT_DIR, "optimise_scenarios.csv")
    res.to_csv(res_path, index=False)

    rec_name = config.RECOMMENDED_SCENARIO
    rec_frame = scenario_frames[rec_name]
    # add delta columns the Phase 3 map expects
    rec_frame["delta_trips"] = rec_frame["trips_hol_opt"] - rec_frame["trips_hol"]
    rec_frame["delta_hw"] = rec_frame["hw_hol_opt"] - rec_frame["hw_hol"]
    rec_frame["delta_abs"] = rec_frame["delta_trips"].abs()
    rec_path = os.path.join(config.OUTPUT_DIR,
                            f"milan_optimised_{rec_name.lower()}.csv")
    rec_frame.to_csv(rec_path, index=False)

    rp = config.write_run_parameters("optimise", extra={
        "scenarios": {r["scenario"]: {
            "gini_improvement_%": round(r["gini_improvement_%"], 2),
            "n_recipients_saturated": r["n_recipients_saturated"],
        } for r in rows},
        "recommended_scenario": rec_name,
    })

    print("\nOutputs written:")
    for p in (res_path, rec_path, rp):
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
