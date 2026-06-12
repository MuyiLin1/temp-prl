"""
Data trade-off sweep: how much does learned-dynamics safety degrade as the
SINDy model error delta grows?

For each configuration (dataset size x measurement-noise x seed) this script:
  1. collects transition data from the true system,
  2. identifies f_hat with SINDy,
  3. measures the model error delta over the operating region,
  4. trains the CLBF certificate entirely against f_hat,
  5. rolls the resulting controller out on the TRUE system and records
     downstream safety (safe-set exit rate, steady-state V), and
  6. estimates the Theorem 6 constants (B, lambda_1, C eps^{1/n}).

Each run appends one row to data/sweep_results.csv. The companion script
plot_tradeoff.py turns that table into the headline figure: measured safety
degradation vs delta, with the predicted bound C eps^{1/n} + B delta / lambda_1
overlaid.

Usage (small local pilot):
    python -m Lyapunov_guided_diffusion.sindy.sweep \
        --num_trajectories 5 25 200 --obs_noise_std 0.0 \
        --seeds 0 --max_epochs 20 --accelerator cpu

Usage (full cluster sweep, GPU):
    python -m Lyapunov_guided_diffusion.sindy.sweep \
        --num_trajectories 2 5 10 25 50 200 \
        --obs_noise_std 0.0 0.05 0.1 0.2 \
        --seeds 0 1 2 --max_epochs 30 --accelerator gpu
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from neural_clbf.systems import InvertedPendulum
from neural_clbf.systems.sindy_estimated_system import SINDyEstimatedSystem

from Lyapunov_guided_diffusion.sindy.collect_data import collect_safe_transitions
from Lyapunov_guided_diffusion.sindy.identify_dynamics import identify_dynamics
from Lyapunov_guided_diffusion.sindy.evaluate import evaluate_on_true_system
from Lyapunov_guided_diffusion.sindy.theory_constants import (
    estimate_B,
    estimate_lambda1,
    estimate_Ceps,
)
from Lyapunov_guided_diffusion.training.train_inverted_pendulum_sindy import build_and_train


# Operating region used for the reported delta (matches the paper's setup).
OPERATING_DOMAIN = ((-1.0, 1.0), (-1.0, 1.0))
CLF_LAMBDA = 1.0


def measure_delta(sindy_system, true_system, n_grid: int = 60, seed: int = 0):
    """Worst-case / mean / median ||f - f_hat|| over the operating region.

    Restricted to the operating box (not the full state space) so the value
    matches the delta the certificate and sampler actually experience.
    """
    g = torch.Generator().manual_seed(seed)
    axes = [torch.linspace(lo, hi, n_grid) for (lo, hi) in OPERATING_DOMAIN]
    mesh = torch.meshgrid(*axes, indexing="ij")
    x = torch.stack([m.reshape(-1) for m in mesh], dim=-1).float()
    u_upper, u_lower = true_system.control_limits
    u = (
        torch.rand(x.shape[0], true_system.n_controls, generator=g)
        * (u_upper - u_lower)
        + u_lower
    )
    with torch.no_grad():
        xdot_true = true_system.closed_loop_dynamics(x, u)
        xdot_est = sindy_system.closed_loop_dynamics(x, u)
    err = torch.norm(xdot_true - xdot_est, dim=-1)
    return {
        "delta_sup": float(err.max().item()),
        "delta_mean": float(err.mean().item()),
        "delta_median": float(err.median().item()),
    }


def steady_state_V(V_values: np.ndarray, tail_frac: float = 0.2) -> float:
    """Mean certificate value over the final tail of the rollouts."""
    T = V_values.shape[1]
    k = max(1, int(tail_frac * T))
    return float(V_values[:, -k:].mean())


def run_one(
    num_trajectories: int,
    trajectory_length: int,
    obs_noise_std: float,
    seed: int,
    max_epochs: int,
    accelerator: str,
    num_rollouts: int,
    t_sim: float,
):
    """Execute one full pipeline run and return a results row (dict)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    simulation_dt = 0.01
    controller_period = 0.05
    nominal_params = {"m": 1.0, "L": 1.0, "b": 0.01}
    scenarios = [nominal_params]

    true_system = InvertedPendulum(
        nominal_params,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
    )

    # 1) Collect (deliberately degraded) data.
    X, U, Xdot = collect_safe_transitions(
        true_system,
        num_trajectories=num_trajectories,
        trajectory_length=trajectory_length,
        exploration_std=0.5,
        obs_noise_std=obs_noise_std,
        seed=seed,
    )
    N = X.shape[0]

    # 2) Identify f_hat.
    model = identify_dynamics(X, U, Xdot)

    # 3) Wrap and measure delta over the operating region.
    sindy_system = SINDyEstimatedSystem(
        model,
        reference_system=true_system,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
    )
    delta = measure_delta(sindy_system, true_system, seed=seed)

    # 4) Train the certificate against f_hat (no TB logging during sweeps).
    controller = build_and_train(
        sindy_system,
        scenarios,
        max_epochs=max_epochs,
        accelerator=accelerator,
        enable_logger=False,
    )
    controller.eval()

    # 5) Roll out on the TRUE system: downstream safety.
    results = evaluate_on_true_system(
        controller,
        true_system,
        num_rollouts=num_rollouts,
        t_sim=t_sim,
        controller_period=controller_period,
    )
    safety_rate = float(results["safe_flags"].mean())
    goal_rate = float(results["goal_reached"].mean())
    limsup_V = steady_state_V(results["V_values"])
    t_array = np.linspace(0, t_sim, results["V_values"].shape[1])

    # 6) Theory constants and predicted bound.
    B = estimate_B(controller, domain=OPERATING_DOMAIN)
    lambda1 = estimate_lambda1(results["V_values"], t_array)
    ceps = estimate_Ceps(
        controller, sindy_system, clf_lambda=CLF_LAMBDA, domain=OPERATING_DOMAIN
    )
    predicted_limsup = ceps["C_eps_term"] + B * delta["delta_sup"] / lambda1

    return {
        "num_trajectories": num_trajectories,
        "trajectory_length": trajectory_length,
        "N": N,
        "obs_noise_std": obs_noise_std,
        "seed": seed,
        "max_epochs": max_epochs,
        "delta_sup": delta["delta_sup"],
        "delta_mean": delta["delta_mean"],
        "delta_median": delta["delta_median"],
        "safety_rate": safety_rate,
        "exit_rate": 1.0 - safety_rate,
        "goal_rate": goal_rate,
        "limsup_V": limsup_V,
        "B": B,
        "lambda1": lambda1,
        "bad_set_fraction": ceps["bad_set_fraction"],
        "C_eps_term": ceps["C_eps_term"],
        "predicted_limsup": predicted_limsup,
    }


def main():
    parser = argparse.ArgumentParser(description="SINDy data trade-off sweep")
    parser.add_argument("--num_trajectories", type=int, nargs="+",
                        default=[2, 5, 10, 25, 50, 200],
                        help="Dataset-size knob (trajectories per config)")
    parser.add_argument("--trajectory_length", type=int, default=100)
    parser.add_argument("--obs_noise_std", type=float, nargs="+",
                        default=[0.0],
                        help="Measurement-noise knob (one run per value)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--accelerator", type=str, default="cpu",
                        help="cpu, gpu, or auto")
    parser.add_argument("--num_rollouts", type=int, default=100)
    parser.add_argument("--t_sim", type=float, default=10.0)
    parser.add_argument("--output", type=str, default="data/sweep_results.csv")
    args = parser.parse_args()

    fieldnames = [
        "num_trajectories", "trajectory_length", "N", "obs_noise_std", "seed",
        "max_epochs", "delta_sup", "delta_mean", "delta_median", "safety_rate",
        "exit_rate", "goal_rate", "limsup_V", "B", "lambda1",
        "bad_set_fraction", "C_eps_term", "predicted_limsup",
    ]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    # Append-friendly: write header only if the file is new/empty.
    write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
    f = open(args.output, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        f.flush()

    configs = [
        (nt, ns, seed)
        for nt in args.num_trajectories
        for ns in args.obs_noise_std
        for seed in args.seeds
    ]
    print(f"Running {len(configs)} configurations -> {args.output}")

    for i, (nt, ns, seed) in enumerate(configs, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(configs)}] num_traj={nt} noise={ns} seed={seed} ...")
        try:
            row = run_one(
                num_trajectories=nt,
                trajectory_length=args.trajectory_length,
                obs_noise_std=ns,
                seed=seed,
                max_epochs=args.max_epochs,
                accelerator=args.accelerator,
                num_rollouts=args.num_rollouts,
                t_sim=args.t_sim,
            )
        except Exception as exc:  # keep the sweep alive on a bad config
            print(f"    FAILED: {exc!r}")
            continue
        writer.writerow(row)
        f.flush()  # persist incrementally so a crash never loses prior runs
        dt = time.time() - t0
        print(
            f"    N={row['N']} delta={row['delta_sup']:.4f} "
            f"exit_rate={row['exit_rate']:.3f} limsupV={row['limsup_V']:.4f} "
            f"pred={row['predicted_limsup']:.4f} ({dt:.0f}s)"
        )

    f.close()
    print(f"\nSweep complete. Results in {args.output}")


if __name__ == "__main__":
    main()
