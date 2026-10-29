"""
================================================================================
sweep.py  —  Continuous reduction-rate sweep for the functional ceiling
================================================================================
Runs the floored reallocation across a DENSE range of donor-reduction rates to
locate the functional reallocation ceiling precisely and characterise the
diminishing-returns curve.

Tracks three outcome metrics across rates:
  - Gini improvement (%)           — distributional equity
  - Burden reduction (%)           — absolute network burden
  - Recipient headway improvement  — minutes saved at recipient stops
plus the recipient-saturation count (the mechanism behind the bend).

Imports the validated functions from optimise.py; does NOT touch them.

Reads:   milan_stop_hourly.csv
Writes:  sweep_curve.csv                 (full curve, all metrics per rate)
         thesis_ceiling_curve.png        (dual-axis thesis figure)
         run_parameters_sweep.json

Usage:
    python sweep.py
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import optimise

config.validate()

# Dense rate grid (edit start/stop/step here if you want finer/coarser).
RATE_MIN = 0.05
RATE_MAX = 0.70
RATE_STEP = 0.01


def main(overflow_rule=None, output_tag=None):
    """Run the continuous sweep.

    ``overflow_rule`` defaults to ``config.OVERFLOW_RULE``. When a non-default
    rule is requested, pass ``output_tag`` (for example ``"residual_return"``)
    so the robustness outputs cannot overwrite the principal thesis files.
    """
    previous_rule = config.OVERFLOW_RULE
    rule = overflow_rule or previous_rule
    if rule != previous_rule and not output_tag:
        raise ValueError("A non-default overflow_rule requires output_tag to avoid overwriting principal outputs")
    config.OVERFLOW_RULE = rule
    suffix = f"_{output_tag}" if output_tag else ""
    print(f"--- Continuous ceiling sweep ({rule}) ---")
    stop = optimise.load_stop_level()
    stop = optimise.classify(stop)

    rates = np.round(np.arange(RATE_MIN, RATE_MAX + 1e-9, RATE_STEP), 4)
    print(f"[sweep] running {len(rates)} rates from {RATE_MIN:.0%} to {RATE_MAX:.0%}...")

    rows = []
    for rate in rates:
        _, st = optimise.run_scenario(stop, float(rate))
        rows.append({
            "rate": float(rate),
            "gini_improvement_%": st["gini_improvement_%"],
            "burden_reduction_%": st["burden_reduction_%"],
            "recip_improvement_min": st["recip_improvement_min"],
            "n_recipients_saturated": st["n_recipients_saturated"],
            "unallocated_trips": st["unallocated_trips"],
        })
    curve = pd.DataFrame(rows)

    curve["budget_neutral"] = curve["unallocated_trips"] < config.BUDGET_NEUTRALITY_TOLERANCE

    # The thesis ceiling is read from the budget-neutral portion of the principal curve.
    eligible = curve[curve["budget_neutral"]]
    peak_gini = eligible.loc[eligible["gini_improvement_%"].idxmax()]
    peak_burden = curve.loc[curve["burden_reduction_%"].idxmax()]
    peak_recip = curve.loc[curve["recip_improvement_min"].idxmax()]

    # rate at which saturation first begins (mechanism onset)
    sat_onset = curve.loc[curve["n_recipients_saturated"] > 0, "rate"]
    sat_onset = sat_onset.min() if len(sat_onset) else None

    print("\n" + "=" * 70)
    print("CONTINUOUS CEILING ANALYSIS")
    print("=" * 70)
    print(f"Gini improvement peaks at:      {peak_gini['rate']:.0%}  "
          f"= {peak_gini['gini_improvement_%']:.1f}%")
    print(f"Burden reduction peaks at:      {peak_burden['rate']:.0%}  "
          f"= {peak_burden['burden_reduction_%']:.1f}%")
    print(f"Recip. headway gain peaks at:   {peak_recip['rate']:.0%}  "
          f"= {peak_recip['recip_improvement_min']:.1f} min")
    print(f"Saturation begins at:           "
          f"{sat_onset:.0%}" if sat_onset is not None else "Saturation begins at: never")
    # do the three metrics peak together or diverge?
    if abs(peak_gini["rate"] - peak_burden["rate"]) > 0.02:
        print("NOTE: Gini and burden peaks DIVERGE — distributional vs absolute "
              "improvement optimise at different rates (a reportable nuance).")
    # confirm a genuine ceiling (peak strictly interior, then declines)
    after_peak = curve[curve["rate"] > peak_gini["rate"]]
    if len(after_peak) and after_peak["gini_improvement_%"].min() < peak_gini["gini_improvement_%"] - 0.5:
        print(f"Numerical Gini peak at {peak_gini['rate']:.0%}; the curve declines afterwards "
              "as recipient capacity becomes increasingly saturated.")
    else:
        print("No interior ceiling within the swept range (curve still rising at max rate).")
    print("=" * 70)

    # --- recommended operating point: knee of the Gini curve -----------------
    # The knee = rate beyond which marginal Gini gain per extra rate-point falls
    # below a small fraction of the early slope. Reported, not prescriptive.
    g = curve["gini_improvement_%"].values
    r = curve["rate"].values
    marginal = np.gradient(g, r)              # dGini/drate
    early_slope = marginal[:5].mean()
    knee_idx = np.argmax(marginal < 0.25 * early_slope) if (marginal < 0.25 * early_slope).any() else len(r) - 1
    knee_rate = r[knee_idx]
    print(f"Diminishing-returns knee (marginal slope < 25% of early): "
          f"~{knee_rate:.0%}  (Gini {g[knee_idx]:.1f}%)")
    rec_rate = config.SCENARIOS.get(config.RECOMMENDED_SCENARIO, None)
    if rec_rate is not None:
        rec_g = curve.iloc[(curve["rate"] - rec_rate).abs().idxmin()]["gini_improvement_%"]
        pct_of_max = rec_g / peak_gini["gini_improvement_%"] * 100
        print(f"Recommended '{config.RECOMMENDED_SCENARIO}' ({rec_rate:.0%}): "
              f"Gini {rec_g:.1f}% = {pct_of_max:.0f}% of the maximum achievable, "
              f"at {rec_rate/peak_gini['rate']*100:.0f}% of the peak's donor reduction.")
    print("=" * 70)

    # --- exports -------------------------------------------------------------
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    curve_path = os.path.join(config.OUTPUT_DIR, f"sweep_curve{suffix}.csv")
    curve.to_csv(curve_path, index=False)

    # --- figure: three metrics + saturation ---------------------------------
    fig, ax1 = plt.subplots(figsize=(11, 6.5))

    # normalise the three outcome metrics to their own max for one shared axis,
    # but keep Gini in true % on the primary axis (it's the headline metric)
    ax1.plot(curve["rate"] * 100, curve["gini_improvement_%"],
             color="#1D4ED8", linewidth=2.6, label="Gini improvement (%)", zorder=3)
    ax1.plot(curve["rate"] * 100, curve["burden_reduction_%"],
             color="#16A34A", linewidth=2.0, linestyle="--",
             label="Network burden reduction (%)", zorder=2)
    ax1.set_xlabel("Donor frequency reduction (%)", fontsize=11)
    ax1.set_ylabel("Improvement (%)", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)

    # recipient headway improvement on a secondary axis (different units: minutes)
    ax2 = ax1.twinx()
    ax2.plot(curve["rate"] * 100, curve["recip_improvement_min"],
             color="#EA580C", linewidth=2.0, linestyle=":",
             label="Recipient headway gain (min)", zorder=2)
    ax2.set_ylabel("Recipient headway improvement (min)",
                   fontsize=11, color="#EA580C")
    ax2.tick_params(axis="y", labelcolor="#EA580C")

    # mark the Gini peak (the functional ceiling)
    ax1.axvline(peak_gini["rate"] * 100, color="#DC2626", linestyle="-",
                linewidth=1.4, alpha=0.8)
    ax1.annotate(f"Functional ceiling\n~{peak_gini['rate']:.0%} "
                 f"(Gini {peak_gini['gini_improvement_%']:.1f}%)",
                 xy=(peak_gini["rate"] * 100, peak_gini["gini_improvement_%"]),
                 xytext=(peak_gini["rate"] * 100 + 4, peak_gini["gini_improvement_%"] - 6),
                 fontsize=9, color="#DC2626",
                 arrowprops=dict(arrowstyle="->", color="#DC2626"))

    # mark the recommended operating point
    if rec_rate is not None:
        ax1.axvline(rec_rate * 100, color="#6B7280", linestyle="-.",
                    linewidth=1.2, alpha=0.7)
        ax1.annotate(f"Recommended\noperating point\n({config.RECOMMENDED_SCENARIO}, {rec_rate:.0%})",
                     xy=(rec_rate * 100, rec_g),
                     xytext=(rec_rate * 100 - 17, rec_g + 3),
                     fontsize=9, color="#374151",
                     arrowprops=dict(arrowstyle="->", color="#6B7280"))

    # mark named scenarios as dots
    for name, rt in config.SCENARIOS.items():
        gval = curve.iloc[(curve["rate"] - rt).abs().idxmin()]["gini_improvement_%"]
        ax1.scatter([rt * 100], [gval], color="#1D4ED8", s=30, zorder=5)

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

    plt.title("Functional reallocation ceiling: equity gain vs donor reduction\n"
              "Milan holiday service  ·  redistribute-then-cap  ·  "
              f"{config.MIN_HEADWAY_FLOOR_MIN}-min recipient modelling cap",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(config.OUTPUT_DIR, f"thesis_ceiling_curve{suffix}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()

    rp = config.write_run_parameters(f"sweep{suffix}", extra={
        "overflow_rule": rule,
        "rate_grid": [RATE_MIN, RATE_MAX, RATE_STEP],
        "gini_peak_rate": float(peak_gini["rate"]),
        "gini_peak_value": float(peak_gini["gini_improvement_%"]),
        "burden_peak_rate": float(peak_burden["rate"]),
        "recip_peak_rate": float(peak_recip["rate"]),
        "saturation_onset_rate": (float(sat_onset) if sat_onset is not None else None),
        "knee_rate": float(knee_rate),
    })

    config.OVERFLOW_RULE = previous_rule
    print("\nOutputs written:")
    for p in (curve_path, fig_path, rp):
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
