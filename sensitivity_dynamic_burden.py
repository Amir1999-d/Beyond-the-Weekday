"""
================================================================================
sensitivity_dynamic_burden.py  —  static vs dynamic burden-weight sensitivity
================================================================================

Runs the sweep TWICE — once with the default static-burden weighting and once
with dynamic-burden weighting (where each iteration recomputes the active
recipient's current burden from its running allocation) — and tabulates the
difference in headline metrics (Gini improvement and recipient mean
headway) at every scenario rate.

What the comparison answers
---------------------------
The static formulation weights freed-trip allocation by the recipient's
INITIAL holiday burden, kept constant through the inner cascade loop. The
dynamic formulation weights allocation by the recipient's CURRENT burden,
recomputed each iteration from the running allocation. The dynamic form is
mathematically more refined; the static form is simpler and deterministic.

The two formulations can only diverge once the inner cascade loop runs
more than one iteration, i.e. once at least one recipient saturates at the
stop-level headway cap. Below the saturation-onset rate they are identical by
construction; above it, the difference is an empirical question. The
saturation-onset rate itself is a computed output of sweep.py (recorded in
run_parameters_sweep.json), not an assumption of this script.

This script settles the static-vs-dynamic question empirically, producing
a defensible sensitivity statement for the results chapter.

Outputs
-------
    sensitivity_dynamic_burden.csv  per-rate comparison table
    sensitivity_dynamic_burden.png  curve overlay figure
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import optimise

config.validate()


def _gini(x):
    """1D Gini coefficient (matches optimise.gini)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0 or x.sum() == 0:
        return 0.0
    x = np.sort(np.abs(x))
    n = x.size
    return (2 * np.arange(1, n + 1).dot(x) - (n + 1) * x.sum()) / (n * x.sum())


def run_one(stop, rate, dynamic_burden):
    """Run one (rate, mode) combination, return headline stats."""
    s, _ = optimise.run_scenario(stop, rate, dynamic_burden=dynamic_burden)
    recip = s["role"] == "recipient"
    g_before = _gini(s["burden_hol"].values)
    g_after = _gini(s["burden_hol_opt"].values)
    return {
        "rate": rate,
        "dynamic_burden": dynamic_burden,
        "gini_improvement_pct": round((g_before - g_after) / g_before * 100, 2),
        "mean_hw_recipient_after": round(s.loc[recip, "hw_hol_opt"].mean(), 2),
        "mean_hw_recipient_before": round(s.loc[recip, "hw_hol"].mean(), 2),
        "max_to_mean_burden_after": round(
            s["burden_hol_opt"].max() / s["burden_hol_opt"].mean(), 3),
    }


def main(rates=None):
    print("--- Sensitivity: static vs dynamic burden weights ---")
    if rates is None:
        # dense around the knee + ceiling region; lighter elsewhere
        rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.27, 0.30, 0.35,
                 0.40, 0.44, 0.50, 0.55, 0.60, 0.65, 0.70]
    print(f"  Sweeping {len(rates)} rates, two modes each ({len(rates)*2} runs)...")

    print("[1] Loading stop table + classifying roles (one-time)...")
    stop = optimise.load_stop_level()
    stop = optimise.classify(stop)
    print(f"    Donors / recipients / neutrals: "
          f"{(stop['role']=='donor').sum()} / "
          f"{(stop['role']=='recipient').sum()} / "
          f"{(stop['role']=='neutral').sum()}")

    rows = []
    for rate in rates:
        for dyn in (False, True):
            stats = run_one(stop, rate, dyn)
            rows.append(stats)
        # print pairwise progress
        st = rows[-2]; dy = rows[-1]
        print(f"  rate={rate*100:4.0f}%   "
              f"static: Gini+{st['gini_improvement_pct']:5.2f}%  "
              f"hw_rec={st['mean_hw_recipient_after']:5.2f} min  ||  "
              f"dynamic: Gini+{dy['gini_improvement_pct']:5.2f}%  "
              f"hw_rec={dy['mean_hw_recipient_after']:5.2f} min")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(config.OUTPUT_DIR, "sensitivity_dynamic_burden.csv")
    df.to_csv(out_csv, index=False)

    # difference table
    pivot = df.pivot(index="rate", columns="dynamic_burden",
                     values="gini_improvement_pct")
    pivot.columns = ["static", "dynamic"]
    pivot["abs_diff_pp"] = (pivot["dynamic"] - pivot["static"]).round(3)
    pivot["rel_diff_pct"] = ((pivot["dynamic"] - pivot["static"]) /
                              pivot["static"].clip(lower=1e-6) * 100).round(2)

    # ── figure: overlay curves ──
    fig, ax = plt.subplots(figsize=(8, 5))
    static = df[df["dynamic_burden"] == False].sort_values("rate")
    dynamic = df[df["dynamic_burden"] == True].sort_values("rate")
    ax.plot(static["rate"] * 100, static["gini_improvement_pct"],
            color="#1A2B47", lw=2.4, marker="o", markersize=5,
            label="Static burden (default)")
    ax.plot(dynamic["rate"] * 100, dynamic["gini_improvement_pct"],
            color="#B85042", lw=2.0, marker="s", markersize=4,
            linestyle="--", label="Dynamic burden (sensitivity)")
    ax.set_xlabel("Donor frequency reduction (%)", fontsize=11)
    ax.set_ylabel("Gini improvement (%)", fontsize=11)
    ax.set_title("Static vs dynamic burden-weight sensitivity",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
    ax.legend(fontsize=10)
    ax.axvline(20, color="#94A3B8", linestyle=":", lw=1, alpha=0.6)
    ax.text(20.5, ax.get_ylim()[0] + 1, "Moderate (20%)", fontsize=9,
            color="#6B6B7B", rotation=90, va="bottom")

    plt.tight_layout()
    fig_path = os.path.join(config.OUTPUT_DIR, "sensitivity_dynamic_burden.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.show()

    # ── verdict ──
    moderate = pivot.loc[0.20] if 0.20 in pivot.index else None
    max_abs_diff = pivot["abs_diff_pp"].abs().max()
    max_rel_diff = pivot["rel_diff_pct"].abs().max()
    arg_max_rate = pivot["abs_diff_pp"].abs().idxmax()

    print("\n=== SENSITIVITY VERDICT ===")
    print(pivot.to_string())
    print(f"\n  Max absolute difference: {max_abs_diff:.3f} pp Gini "
          f"(at rate {arg_max_rate*100:.0f}%)")
    print(f"  Max relative difference: {max_rel_diff:.2f}%")
    if moderate is not None:
        print("\n  Moderate (20%) scenario:")
        print(f"    Static  Gini improvement: {moderate['static']:.2f}%")
        print(f"    Dynamic Gini improvement: {moderate['dynamic']:.2f}%")
        print(f"    Absolute difference: {moderate['abs_diff_pp']:.3f} pp")

    if max_abs_diff < 1.0 and (moderate is None or abs(moderate["abs_diff_pp"]) < 0.5):
        print("\n  -> Verdict: STATIC AND DYNAMIC PRODUCE SUBSTANTIALLY EQUIVALENT")
        print("     RESULTS. Default static formulation is retained for the main")
        print("     analysis; dynamic-burden formulation is documented as a")
        print("     sensitivity check confirming robustness.")
    else:
        print("\n  -> Verdict: MEANINGFUL DIFFERENCE BETWEEN FORMULATIONS at some")
        print("     scenario rates. Recommend switching the main analysis to")
        print("     dynamic-burden weighting and re-running Phase 5 + sweep.")

    print(f"\n  Saved: {os.path.basename(out_csv)}")
    print(f"  Saved: {os.path.basename(fig_path)}")


if __name__ == "__main__":
    main()
