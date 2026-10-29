"""
test_depot_constrained.py — checks of depot_constrained_optimise.py on
synthetic data, run against the UNMODIFIED optimise.py of the pipeline.

No project data is read: config and build_stop_table are replaced by stubs,
so these tests exercise the allocation algorithm only. They verify
properties that must hold for any input:

  T1  single depot + everything transferable  ->  identical to
      optimise.run_scenario (the constrained model nests the thesis model)
  T2  total service conserved exactly, every rate and every mode
  T3  no recipient is ever pushed past the 8-minute stop-level modelling cap
  T4  strict mode: no service crosses depots
  T5  allowance mode: exactly min(allowance, placeable) crosses depots
  T6  mode restriction: a donor with no transferable service gives nothing
  T7  a recipient with NO holiday service but a weekday link can receive
  T8  reallocated volume is ordered strict <= allowance <= free
  T9  free mode reports no within/inter-depot split (not attributable)
  T10 unserved stops come from the missing median, not from the penalty
      value: a SERVED stop with a 180-minute median is not flagged
  T11 the like-for-like unconstrained run conserves service at every rate
      and leaves config.OVERFLOW_RULE exactly as it found it
  T12 recipients use weekday lines, and fall back to holiday lines only
      where a stop has no weekday service
  T13 night and interurban lines are excluded with the switch on, and only
      operator-excluded codes with it off

Usage:  python test_depot_constrained.py   (from the pipeline folder)
"""
import sys, types
import numpy as np
import pandas as pd

# ---- stubs: no paths, no files ----------------------------------------------
cfg = types.ModuleType("config")
cfg.validate = lambda: None
cfg.OPERATIONAL_MINUTES = 960
cfg.MIN_HEADWAY_FLOOR_MIN = 8
cfg.BUDGET_NEUTRALITY_TOLERANCE = 1.0
cfg.OVERFLOW_RULE = "redistribute_then_cap"
cfg.NO_SERVICE_PENALTY_MIN = 120
cfg.ROLE_VULN_LOW_Q, cfg.ROLE_VULN_HIGH_Q = 0.33, 0.67
cfg.ROLE_HW_LOW_Q, cfg.ROLE_HW_HIGH_Q = 0.33, 0.67
cfg.SCENARIOS = {"Moderate": 0.20}
cfg.RECOMMENDED_SCENARIO = "Moderate"
cfg.BASE_PATH = cfg.OUTPUT_DIR = "."
cfg.operational_hour_filter = lambda h: (h >= 6) & (h < 22)
cfg.NON_URBAN_ROUTE_PREFIXES, cfg.NON_URBAN_ROUTE_EXACT = ("N",), ("172", "174")
sys.modules["config"] = cfg
sys.modules["build_stop_table"] = types.ModuleType("build_stop_table")

import optimise                                   # the real, frozen module
import depot_constrained_optimise as dc

rng = np.random.default_rng(7)
FLOOR_TRIPS = 960 / 8


def synthetic(n=600, unserved=12):
    v = rng.beta(2, 5, n)
    hw = np.exp(rng.normal(np.log(15), 0.55, n)).clip(3, 90)
    hw[:unserved] = np.nan                        # stops with no holiday service
    s = pd.DataFrame({"stop_id": [f"s{i}" for i in range(n)], "v_index": v, "hw_hol": hw})
    s["hw_hol"] = s["hw_hol"].fillna(cfg.NO_SERVICE_PENALTY_MIN)     # penalised basis
    s["burden_hol"] = s["v_index"] * s["hw_hol"]
    s["trips_hol"] = 960 / s["hw_hol"]
    s = optimise.classify(s)
    # make the unserved stops high-vulnerability so they are recipients
    s.loc[:unserved - 1, "v_index"] = 0.95
    s["burden_hol"] = s["v_index"] * s["hw_hol"]
    return optimise.classify(s)


def attribution(s, n_depots, seed=1, metro_share=0.0):
    r = np.random.default_rng(seed)
    A = r.dirichlet(np.ones(n_depots), len(s))
    if metro_share:
        A *= (1 - metro_share)
    return A


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise SystemExit(1)


print("Depot-constrained allocator — synthetic checks")
s = synthetic()
one = np.ones((len(s), 1))

# T1 nesting: one depot, all transferable, rates below full saturation
for rate in (0.05, 0.10, 0.20):
    a, st = optimise.run_scenario(s, rate)
    check(f"T1 setup rate {rate:.0%}: unconstrained run has no residual",
          st["unallocated_trips"] <= 1e-9)
    b, sc, _ = dc.run_constrained(s, one, one, ["D"], rate, mode="strict")
    same = np.allclose(a["trips_hol_opt"].values, b["trips_hol_opt"].values, atol=1e-6)
    check(f"T1 rate {rate:.0%}: matches optimise.run_scenario "
          f"(Gini {st['gini_improvement_%']:.4f} vs {sc['gini_improvement_%']:.4f})",
          same and abs(st["gini_improvement_%"] - sc["gini_improvement_%"]) < 1e-9)

# T2 / T3 on a realistic multi-depot setting with 25% metro service
A = attribution(s, 12, metro_share=0.25)
for mode in ("strict", "allowance", "free"):
    for rate in (0.05, 0.20, 0.41, 0.65):
        b, sc, _ = dc.run_constrained(s, A, A, [f"D{i}" for i in range(12)], rate,
                                      mode=mode, allowance=500)
        cons = abs(b["trips_hol"].sum() - b["trips_hol_opt"].sum()) < 1e-6
        rec = b["role"] == "recipient"
        cap = (b.loc[rec, "trips_hol_opt"] <= FLOOR_TRIPS + 1e-6).all()
        if not (cons and cap):
            check(f"T2/T3 {mode} {rate:.0%}", False)
check("T2 service conserved exactly in every mode and rate", True)
check("T3 no recipient pushed past the 8-minute stop-level modelling cap", True)

# T4 strict: donors only in depot 0, recipients only in depot 1 -> nothing moves
D = ["A", "B"]
A_don = np.tile([1.0, 0.0], (len(s), 1))
A_rec = np.tile([0.0, 1.0], (len(s), 1))
b, sc, pdp = dc.run_constrained(s, A_don, A_rec, D, 0.20, mode="strict")
check(f"T4 strict: nothing crosses depots (reallocated {sc['reallocated']:.3f}, "
      f"all {sc['returned_to_donors']:.1f} returned)",
      sc["reallocated"] < 1e-9 and abs(sc["gini_improvement_%"]) < 1e-9)

# T5 allowance: exactly the allowance crosses when supply and headroom exceed it
b, sc, _ = dc.run_constrained(s, A_don, A_rec, D, 0.20, mode="allowance", allowance=300)
check(f"T5 allowance: {sc['allocated_inter_depot']:.2f} crossed (allowance 300)",
      abs(sc["allocated_inter_depot"] - 300) < 1e-6 and abs(sc["reallocated"] - 300) < 1e-6)

# T6 mode restriction: donors with no transferable service surrender nothing
A_metro = np.zeros((len(s), 1))
b, sc, _ = dc.run_constrained(s, A_metro, one, ["D"], 0.20, mode="strict")
don = b["role"] == "donor"
check("T6 metro-only donors keep all their service",
      np.allclose(b.loc[don, "trips_hol_opt"], b.loc[don, "trips_hol"]) and sc["freed_surface"] == 0)

# T7 unserved recipient with a weekday link receives service
unserved_recip = (s["hw_hol"] == 120) & (s["role"] == "recipient")
b, sc, _ = dc.run_constrained(s, one, one, ["D"], 0.20, mode="strict")
gain = (b.loc[unserved_recip, "trips_hol_opt"] - b.loc[unserved_recip, "trips_hol"]).min()
check(f"T7 {int(unserved_recip.sum())} unserved recipients all receive (min gain {gain:.2f})",
      unserved_recip.sum() > 0 and gain > 0)

# T8 ordering across modes, in a SPARSE setting where the constraint binds:
# as on a real network, each stop is supplied by one or two depots only, and
# donors are concentrated in a few depots while recipients are spread out
def sparse(s, n_depots, donor_depots, seed):
    r = np.random.default_rng(seed)
    A = np.zeros((len(s), n_depots))
    for i, role in enumerate(s["role"]):
        pool = donor_depots if role == "donor" else range(n_depots)
        k = r.choice(pool, size=r.integers(1, 3), replace=False)
        A[i, k] = 1.0 / len(k)
    return A
Asp = sparse(s, 12, [0, 1, 2], seed=3)
ALLOW = 100.0     # below the ~289 left stranded by strict mode, so it binds
out = {m: dc.run_constrained(s, Asp, Asp, [f"D{i}" for i in range(12)], 0.30, mode=m,
                             allowance=ALLOW)[1]
       for m in ("strict", "allowance", "free")}
v = {m: out[m]["reallocated"] for m in out}
g = {m: out[m]["gini_improvement_%"] for m in out}
check(f"T8 constraint binds: strict {v['strict']:.1f} < allowance {v['allowance']:.1f} "
      f"< free {v['free']:.1f}", v["strict"] < v["allowance"] < v["free"] - 1)
check(f"T8 allowance adds exactly its {ALLOW:.0f} on top of strict "
      f"({v['allowance'] - v['strict']:.2f})", abs(v["allowance"] - v["strict"] - ALLOW) < 1e-6)
check(f"T8 equity follows: Gini strict {g['strict']:.2f}% < allowance {g['allowance']:.2f}% "
      f"< free {g['free']:.2f}%", g["strict"] < g["allowance"] < g["free"])

# T9 free mode: the within/inter split does not exist
_, sc, _ = dc.run_constrained(s, A, A, [f"D{i}" for i in range(12)], 0.20, mode="free")
check("T9 free mode: within/inter-depot split reported as not attributable",
      np.isnan(sc["allocated_within_depot"]) and np.isnan(sc["allocated_inter_depot"]))

# T10 unserved = missing median; written to a TEMPORARY folder, never the working one
import tempfile, os
tmp = tempfile.mkdtemp()
pd.DataFrame({"stop_id": ["A", "A", "B", "B", "C", "D"],
              "hour":    [8, 9, 8, 9, 8, 23],
              "hw_hol":  [np.nan, np.nan, 180.0, 180.0, 20.0, 15.0]}).to_csv(
    os.path.join(tmp, "milan_stop_hourly.csv"), index=False)
prev_out = cfg.OUTPUT_DIR
try:
    cfg.OUTPUT_DIR = tmp
    un = dc.unserved_holiday_stops()
finally:
    cfg.OUTPUT_DIR = prev_out
# D runs only at 23:00, outside the window: load_stop_level drops it entirely, so
# it is not in the stop table at all and must not be reported either way
check(f"T10 unserved = {sorted(un)}: A (no service) flagged; B (served, 180-min median), "
      "C (served) and D (outside the window, absent from the stop table) not", un == {"A"})

# T11 like-for-like unconstrained: conserved, and config restored
rule_before = cfg.OVERFLOW_RULE
ok = True
for rate in (0.20, 0.41, 0.65):
    a, st = dc.run_unconstrained(s, rate, dc.LIKE_FOR_LIKE_RULE)
    ok &= abs(a["trips_hol"].sum() - a["trips_hol_opt"].sum()) < 1e-6
check("T11 residual-return unconstrained run conserves service at 20/41/65%, "
      f"rule restored to '{cfg.OVERFLOW_RULE}'", ok and cfg.OVERFLOW_RULE == rule_before)

# T12 recipient line set: weekday first, holiday lines only as a fallback
wd = pd.DataFrame({"stop_id": ["X"], "line": ["10"], "route_type": ["3"], "share": [1.0]})
hol = pd.DataFrame({"stop_id": ["X", "X", "Y"], "line": ["10", "11", "20"],
                    "route_type": ["3", "3", "3"], "share": [0.5, 0.5, 1.0]})
rec = dc.recipient_stop_lines(wd, hol)
check("T12 X keeps its weekday line only, Y (holiday-only) falls back to line 20, shares sum to 1",
      sorted(map(tuple, rec[["stop_id", "line"]].values.tolist())) == [("X", "10"), ("Y", "20")]
      and np.allclose(rec.groupby("stop_id")["share"].sum(), 1.0))

# T13 exclusions: operator codes always; night and interurban lines only with the switch on
codes = ["NM1", "N15", "172", "161", "BDQ39", "90"]
on = {c: dc._excluded(c) for c in codes}
prev = dc.EXCLUDE_NON_URBAN
try:
    dc.EXCLUDE_NON_URBAN = False
    off = {c: dc._excluded(c) for c in codes}
finally:
    dc.EXCLUDE_NON_URBAN = prev
check(f"T13 switch on excludes {[c for c in codes if on[c]]}; off excludes {[c for c in codes if off[c]]}",
      on == {"NM1": True, "N15": True, "172": True, "161": True, "BDQ39": True, "90": False}
      and off == {"NM1": False, "N15": False, "172": False, "161": True, "BDQ39": True, "90": False})

print("All checks passed.")
