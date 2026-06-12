"""
Step A: Safe Data Collection for SINDYc Identification.

Collects transition data D = {(x_i, u_i, x_dot_i)} from the true system
using a safe LQR-based nominal policy with small exploratory noise.
This avoids unsafe exploration while covering enough of the state space
for SINDy to identify the dynamics.

Usage:
    python -m Lyapunov_guided_diffusion.sindy.collect_data \
        --system inverted_pendulum \
        --num_trajectories 200 \
        --trajectory_length 100 \
        --output data/sindy_transitions.npz
"""
import argparse
import os
from typing import Tuple

import numpy as np
import torch

from neural_clbf.systems import InvertedPendulum


def collect_safe_transitions(
    system,
    num_trajectories: int = 200,
    trajectory_length: int = 100,
    exploration_std: float = 0.5,
    obs_noise_std: float = 0.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect transition data using the system's LQR nominal controller
    plus small Gaussian exploration noise, restricted to safe states.

    Args:
        system: a ControlAffineSystem instance (with true dynamics)
        num_trajectories: number of rollouts to collect
        trajectory_length: timesteps per rollout
        exploration_std: std of additive Gaussian noise on control
        obs_noise_std: std of Gaussian measurement noise added to the
            *recorded* states and derivatives that SINDy sees. The rollout
            itself still uses the clean true dynamics so trajectories stay
            safe; only the identification dataset is corrupted. This is the
            primary knob for deliberately enlarging the model error delta.
        seed: random seed

    Returns:
        X: (N, n_dims) state matrix
        U: (N, n_controls) control matrix
        Xdot: (N, n_dims) derivative matrix (finite difference)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    dt = system.dt
    n_dims = system.n_dims
    n_controls = system.n_controls
    upper_lim, lower_lim = system.state_limits
    u_upper, u_lower = system.control_limits

    all_x = []
    all_u = []
    all_xdot = []

    for traj_idx in range(num_trajectories):
        # Sample initial condition from the safe region
        x = system.sample_safe(1)  # (1, n_dims)

        for t in range(trajectory_length):
            # Compute nominal LQR control
            u_nom = system.u_nominal(x)  # (1, n_controls)

            # Add conservative exploration noise
            noise = exploration_std * torch.randn(1, n_controls)
            u = u_nom + noise

            # Clamp control to limits
            for dim in range(n_controls):
                u[:, dim] = torch.clamp(
                    u[:, dim],
                    min=u_lower[dim].item(),
                    max=u_upper[dim].item(),
                )

            # Compute true dynamics: x_dot = f(x) + g(x) * u
            with torch.no_grad():
                xdot = system.closed_loop_dynamics(x, u)

            # Record the transition. Measurement noise corrupts only the
            # recorded (identification) data, not the rollout, so the
            # trajectory remains safe while SINDy sees a degraded dataset.
            x_rec = x.numpy().flatten()
            u_rec = u.numpy().flatten()
            xdot_rec = xdot.numpy().flatten()
            if obs_noise_std > 0.0:
                x_rec = x_rec + np.random.normal(0.0, obs_noise_std, size=x_rec.shape)
                xdot_rec = xdot_rec + np.random.normal(0.0, obs_noise_std, size=xdot_rec.shape)
            all_x.append(x_rec)
            all_u.append(u_rec)
            all_xdot.append(xdot_rec)

            # Euler step to get next state
            x_next = x + dt * xdot

            # Check safety: if we leave the safe region, reset
            if system.out_of_bounds_mask(x_next).any():
                break

            x = x_next

    X = np.array(all_x)      # (N, n_dims)
    U = np.array(all_u)      # (N, n_controls)
    Xdot = np.array(all_xdot)  # (N, n_dims)

    return X, U, Xdot


def main():
    parser = argparse.ArgumentParser(description="Collect safe transition data for SINDYc")
    parser.add_argument("--system", type=str, default="inverted_pendulum",
                        choices=["inverted_pendulum"],
                        help="Which system to collect data from")
    parser.add_argument("--num_trajectories", type=int, default=200,
                        help="Number of rollout trajectories")
    parser.add_argument("--trajectory_length", type=int, default=100,
                        help="Max timesteps per trajectory")
    parser.add_argument("--exploration_std", type=float, default=0.5,
                        help="Std of exploration noise on controls")
    parser.add_argument("--obs_noise_std", type=float, default=0.0,
                        help="Std of measurement noise on recorded states/derivatives "
                             "(knob to enlarge SINDy error delta)")
    parser.add_argument("--output", type=str, default="data/sindy_transitions.npz",
                        help="Output file path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Instantiate the true system
    if args.system == "inverted_pendulum":
        nominal_params = {"m": 1.0, "L": 1.0, "b": 0.01}
        system = InvertedPendulum(nominal_params, dt=0.01)
    else:
        raise ValueError(f"Unknown system: {args.system}")

    print(f"Collecting data from '{args.system}'...")
    print(f"  num_trajectories = {args.num_trajectories}")
    print(f"  trajectory_length = {args.trajectory_length}")
    print(f"  exploration_std = {args.exploration_std}")

    X, U, Xdot = collect_safe_transitions(
        system,
        num_trajectories=args.num_trajectories,
        trajectory_length=args.trajectory_length,
        exploration_std=args.exploration_std,
        obs_noise_std=args.obs_noise_std,
        seed=args.seed,
    )

    print(f"Collected {X.shape[0]} transitions.")
    print(f"  X shape: {X.shape}")
    print(f"  U shape: {U.shape}")
    print(f"  Xdot shape: {Xdot.shape}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(
        args.output,
        X=X,
        U=U,
        Xdot=Xdot,
        system=args.system,
        dt=system.dt,
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
