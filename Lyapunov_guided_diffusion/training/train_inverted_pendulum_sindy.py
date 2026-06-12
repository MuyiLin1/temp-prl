"""
Step D: Training script for the inverted pendulum using SINDYc-learned dynamics.

This script mirrors train_inverted_pendulum.py but swaps out the true
InvertedPendulum dynamics model for the SINDyEstimatedSystem surrogate.

The NeuralCLBFController's diffusion sampler and CLBF training then operate
entirely against the estimated dynamics f_hat, validating Theorem 6:
the system converges to a neighborhood of the goal with bound
    V(x_t) ≤ e^{-λ₁t} V(x_0) + Cε^{1/n} + Bδ/λ₁

Usage:
    # First run data collection and identification:
    python -m Lyapunov_guided_diffusion.sindy.collect_data
    python -m Lyapunov_guided_diffusion.sindy.identify_dynamics

    # Then train with learned dynamics:
    python -m Lyapunov_guided_diffusion.training.train_inverted_pendulum_sindy
"""
from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.systems import InvertedPendulum
from neural_clbf.systems.sindy_estimated_system import (
    SINDyEstimatedSystem,
    load_sindy_system,
)
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutStateSpaceExperiment,
)
from neural_clbf.training.utils import current_git_hash


torch.multiprocessing.set_sharing_strategy("file_system")

batch_size = 64
controller_period = 0.05

start_x = torch.tensor(
    [
        [0.5, 0.5],
        [-0.2, 1.0],
        [0.2, -1.0],
        [-0.2, -1.0],
    ]
)
simulation_dt = 0.01


def build_and_train(
    dynamics_model,
    scenarios,
    max_epochs: int = 51,
    accelerator: str = "cpu",
    log_dir: str = "logs/inverted_pendulum_sindy",
    log_name: str = None,
    enable_logger: bool = True,
    detect_anomaly: bool = False,
):
    """Build a NeuralCLBFController on the given dynamics model and train it.

    Factored out of main() so the sweep driver can call it in-process with an
    in-memory SINDy system, a reduced epoch budget, and a GPU accelerator on
    the cluster.

    Args:
        dynamics_model: the (SINDy-estimated or true) ControlAffineSystem.
        scenarios: scenario list passed to the controller.
        max_epochs: training epochs (use a smaller value for sweeps).
        accelerator: pytorch-lightning accelerator ("cpu", "gpu", or "auto").
        log_dir / log_name: TensorBoard logging location.
        enable_logger: if False, disables logging entirely (sweep runs).
        detect_anomaly: enable autograd anomaly detection (slow; debug only).

    Returns:
        The trained NeuralCLBFController.
    """
    initial_conditions = [
        (-np.pi / 2, np.pi / 2),  # theta
        (-1.0, 1.0),              # theta_dot
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=0,
        trajectory_length=1,
        fixed_samples=10000,
        max_points=100000,
        val_split=0.1,
        batch_size=64,
    )

    V_contour_experiment = CLFContourExperiment(
        "V_Contour",
        domain=[(-2.0, 2.0), (-2.0, 2.0)],
        n_grid=30,
        x_axis_index=InvertedPendulum.THETA,
        y_axis_index=InvertedPendulum.THETA_DOT,
        x_axis_label="$\\theta$",
        y_axis_label="$\\dot{\\theta}$",
        plot_unsafe_region=False,
    )
    rollout_experiment = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        InvertedPendulum.THETA,
        "$\\theta$",
        InvertedPendulum.THETA_DOT,
        "$\\dot{\\theta}$",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=5.0,
    )
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment])

    clbf_controller = NeuralCLBFController(
        dynamics_model,
        scenarios,
        data_module,
        experiment_suite=experiment_suite,
        clbf_hidden_layers=2,
        clbf_hidden_size=64,
        clf_lambda=1.0,
        safe_level=1.0,
        controller_period=controller_period,
        clf_relaxation_penalty=1e2,
        num_init_epochs=5,
        epochs_per_episode=100,
        barrier=False,
        disable_gurobi=True,
    )

    if enable_logger:
        if log_name is None:
            log_name = f"commit_{current_git_hash()}"
        logger = pl_loggers.TensorBoardLogger(log_dir, name=log_name)
    else:
        # The controller's experiment suite calls self.logger.experiment during
        # validation, so a logger must exist even for sweep runs. Point it at a
        # throwaway temp directory instead of disabling logging entirely.
        import tempfile
        logger = pl_loggers.TensorBoardLogger(
            tempfile.mkdtemp(prefix="sindy_sweep_"), name="run"
        )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=max_epochs,
        accelerator=accelerator,
    )

    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    trainer.fit(clbf_controller)
    return clbf_controller


def main(args):
    # Define the scenarios (same as original)
    nominal_params = {"m": 1.0, "L": 1.0, "b": 0.01}
    scenarios = [nominal_params]

    # ================================================================
    # KEY DIFFERENCE: Load SINDYc-learned dynamics instead of true model
    # ================================================================

    # First, create the reference (true) system for geometry/limits
    true_system = InvertedPendulum(
        nominal_params,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
    )

    # Load the SINDy-identified model and wrap it as a ControlAffineSystem
    sindy_model_path = args.sindy_model if hasattr(args, 'sindy_model') else "data/sindy_model.pkl"
    dynamics_model = load_sindy_system(
        model_path=sindy_model_path,
        reference_system=true_system,
        dt=simulation_dt,
        controller_dt=controller_period,
        scenarios=scenarios,
    )

    # Compute and report the dynamics error δ (Eq. 13)
    print("\n=== Dynamics Error Analysis (Theorem 6) ===")
    error_metrics = dynamics_model.compute_dynamics_error(true_system, num_samples=10000)
    print(f"  δ (sup ||f - f_hat||₂) = {error_metrics['delta_sup']:.6f}")
    print(f"  δ (mean)               = {error_metrics['delta_mean']:.6f}")
    print(f"  δ (median)             = {error_metrics['delta_median']:.6f}")
    print(f"  For Theorem 6: lim sup V(x_t) ≤ Cε^(1/n) + B·{error_metrics['delta_sup']:.4f}/λ₁")
    print()

    # Build the controller on the SINDy-estimated dynamics and train it.
    build_and_train(
        dynamics_model,
        scenarios,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        detect_anomaly=True,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sindy_model", type=str, default="data/sindy_model.pkl",
                        help="Path to the fitted SINDYc model")
    parser.add_argument("--max_epochs", type=int, default=51,
                        help="Number of training epochs")
    parser.add_argument("--accelerator", type=str, default="cpu",
                        help="pytorch-lightning accelerator: cpu, gpu, or auto")
    args = parser.parse_args()

    main(args)
