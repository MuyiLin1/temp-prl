"""
Turn data/sweep_results.csv into the headline trade-off figure.

Left panel (the money plot): measured steady-state certificate value
lim sup V versus the SINDy model error delta, aggregated over seeds with error
bars, with the Theorem 6 prediction C eps^{1/n} + B delta / lambda_1 drawn
through the points. A twin axis shows the goal-reached rate (the metric that
actually carries signal; the raw safe-set exit flag saturates at 1.0 because of
transient overshoot under the large control authority and is therefore not
plotted by default).

Right panel: delta versus dataset size N (and noise), i.e. the data -> model
error half of the sample-complexity story.

Usage:
    python -m Lyapunov_guided_diffusion.sindy.plot_tradeoff \
        --input data/sweep_results.csv --output tradeoff.png \
        --delta_max 1.0
"""
import argparse
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) if v not in ("", None) else np.nan
                         for k, v in r.items()})
    return rows


def aggregate(rows, key_fields, value_field):
    """Group rows by key_fields, return mean/std of value_field per group."""
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in key_fields)
        groups[key].append(r[value_field])
    out = {}
    for key, vals in groups.items():
        arr = np.array(vals, dtype=float)
        out[key] = (np.nanmean(arr), np.nanstd(arr))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/sweep_results.csv")
    parser.add_argument("--output", type=str, default="tradeoff.png")
    parser.add_argument("--delta_max", type=float, default=None,
                        help="Drop configs whose mean delta_sup exceeds this "
                             "(filters the vacuous large-delta regime where the "
                             "bound is not meaningful).")
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows in {args.input}")

    # Aggregate over seeds, keyed by (num_trajectories, obs_noise_std).
    keys = [(r["num_trajectories"], r["obs_noise_std"]) for r in rows]
    uniq = sorted(set(keys))

    delta_mean = aggregate(rows, ["num_trajectories", "obs_noise_std"], "delta_sup")
    limsup_mean = aggregate(rows, ["num_trajectories", "obs_noise_std"], "limsup_V")
    goal_mean = aggregate(rows, ["num_trajectories", "obs_noise_std"], "goal_rate")
    pred_mean = aggregate(rows, ["num_trajectories", "obs_noise_std"], "predicted_limsup")
    N_mean = aggregate(rows, ["num_trajectories", "obs_noise_std"], "N")

    # Optionally drop the vacuous large-delta configs.
    if args.delta_max is not None:
        uniq = [k for k in uniq if delta_mean[k][0] <= args.delta_max]
        if not uniq:
            raise SystemExit(
                f"All configs exceed --delta_max={args.delta_max}; "
                "nothing left to plot."
            )

    d = np.array([delta_mean[k][0] for k in uniq])
    d_err = np.array([delta_mean[k][1] for k in uniq])
    v = np.array([limsup_mean[k][0] for k in uniq])
    v_err = np.array([limsup_mean[k][1] for k in uniq])
    goal = np.array([goal_mean[k][0] for k in uniq])
    pred = np.array([pred_mean[k][0] for k in uniq])
    Ns = np.array([N_mean[k][0] for k in uniq])

    order = np.argsort(d)
    d, d_err, v, v_err, goal, pred, Ns = (
        d[order], d_err[order], v[order], v_err[order], goal[order], pred[order], Ns[order]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # ---- Left: safety vs delta, with bound overlay ----
    ax1.errorbar(d, v, xerr=d_err, yerr=v_err, fmt="o", color="tab:blue",
                 capsize=3, label=r"measured $\limsup_t V$", zorder=3)
    # Predicted bound line (sorted by delta).
    ax1.plot(d, pred, "r--", lw=2, label=r"bound $C\varepsilon^{1/n}+B\delta/\lambda_1$",
             zorder=2)
    ax1.set_xlabel(r"SINDy model error $\delta=\sup\|f-\hat f\|$")
    ax1.set_ylabel(r"steady-state $\limsup_t V(x_t)$", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title("Downstream safety degrades gracefully in $\\delta$")
    ax1.grid(True, alpha=0.3)

    ax1b = ax1.twinx()
    ax1b.plot(d, goal, "s:", color="tab:green", alpha=0.8, label="goal-reached rate")
    ax1b.set_ylabel("goal-reached rate", color="tab:green")
    ax1b.tick_params(axis="y", labelcolor="tab:green")
    ax1b.set_ylim(-0.02, 1.02)

    # Combined legend.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    # ---- Right: delta vs N ----
    ax2.loglog(Ns, d, "o-", color="tab:purple")
    ax2.set_xlabel("dataset size $N$ (transitions)")
    ax2.set_ylabel(r"model error $\delta$")
    ax2.set_title("More data $\\Rightarrow$ smaller $\\delta$")
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"saved {args.output}")
    print(f"  {len(uniq)} configs, delta range [{d.min():.4f}, {d.max():.4f}], "
          f"goal-rate range [{goal.min():.3f}, {goal.max():.3f}]")


if __name__ == "__main__":
    main()
