"""
Step C: SINDy Estimated System — a surrogate dynamics model.

This class wraps a SINDYc-identified model so that it conforms to the
ControlAffineSystem interface used throughout the diffusion CLBF codebase.

The key override is `closed_loop_dynamics(x, u)`: instead of computing
hardcoded physics, it evaluates the learned feature library multiplied
by the sparse coefficient matrix Ξ_hat to produce x_dot = f_hat(x, u).

This enables the entire diffusion pipeline (trajectory sampling, CLBF
training, Lie derivative computation) to operate on the estimated dynamics
without modifying the core algorithm — validating Theorem 6.
"""
import pickle
from typing import Tuple, Optional, List

import numpy as np
import torch

from neural_clbf.systems.control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList


class SINDyEstimatedSystem(ControlAffineSystem):
    """
    A dynamics model whose equations of motion come from a SINDYc fit.

    Rather than implementing _f(x) and _g(x) analytically, this class
    stores the PySINDy model and evaluates it to produce x_dot given (x, u).

    It inherits metadata (state/control dims, limits, safe/unsafe masks)
    from a reference "ground truth" system so that the rest of the pipeline
    (data sampling, experiment evaluation) remains compatible.
    """

    def __init__(
        self,
        sindy_model,
        reference_system: ControlAffineSystem,
        nominal_params: Optional[Scenario] = None,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
    ):
        """
        Initialize the SINDy-estimated system.

        Args:
            sindy_model: a fitted PySINDy model (from identify_dynamics)
            reference_system: the true system, used to inherit geometry
                (state limits, control limits, safe/unsafe masks, etc.)
            nominal_params: scenario dict (passed through but not used for dynamics)
            dt: simulation timestep
            controller_dt: controller discretization period
            scenarios: scenario list for robust training
        """
        self.sindy_model = sindy_model
        self.reference_system = reference_system

        # Store dimensions from the reference system
        self._n_dims = reference_system.n_dims
        self._n_controls = reference_system.n_controls
        self._angle_dims = reference_system.angle_dims
        self._state_limits = reference_system.state_limits
        self._control_limits = reference_system.control_limits

        # Use reference's nominal params if none provided
        if nominal_params is None:
            nominal_params = reference_system.nominal_params

        # Initialize the base class. We pass use_linearized_controller=True so the
        # LQR gain K and Lyapunov matrix P are computed from the LEARNED dynamics
        # f_hat (via the overridden compute_A_matrix below and the SINDy-based _g),
        # NOT copied from the true system. This keeps the data-driven pipeline free
        # of any true-physics quantity: the nominal/exploration controller is
        # derived entirely from the estimated model.
        super().__init__(
            nominal_params=nominal_params,
            dt=dt,
            controller_dt=controller_dt if controller_dt else dt,
            use_linearized_controller=True,
            scenarios=scenarios,
        )

        # compute_linearized_controller stores K and P as float64 tensors. The
        # rest of the pipeline operates in float32, and Apple's MPS backend does
        # not support float64; cast here so the learned controller is consistent
        # with the network dtype and runs on all backends.
        if hasattr(self, "K"):
            self.K = self.K.float()
        if hasattr(self, "P"):
            self.P = self.P.float()

    def validate_params(self, params: Scenario) -> bool:
        """Accept any params — the SINDy model doesn't use them for dynamics."""
        return True

    @property
    def n_dims(self) -> int:
        return self._n_dims

    @property
    def angle_dims(self) -> List[int]:
        return self._angle_dims

    @property
    def n_controls(self) -> int:
        return self._n_controls

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._state_limits

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._control_limits

    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
        return self.reference_system.safe_mask(x)

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        return self.reference_system.unsafe_mask(x)

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        return self.reference_system.goal_mask(x)

    @property
    def goal_point(self):
        return self.reference_system.goal_point

    @property
    def u_eq(self):
        return self.reference_system.u_eq

    def _f(self, x: torch.Tensor, params: Scenario):
        """
        Return the drift component f(x) of the control-affine dynamics.

        For the SINDy model, we cannot cleanly separate f(x) and g(x) because
        the learned model gives f_hat(x, u) directly. We approximate by evaluating
        the model at u=0 to get the drift.

        Args:
            x: (bs, n_dims) tensor of states
            params: scenario parameters (unused for SINDy dynamics)
        Returns:
            f: (bs, n_dims, 1) tensor
        """
        batch_size = x.shape[0]
        x_np = x.detach().cpu().numpy()
        u_zero = np.zeros((batch_size, self._n_controls))

        # Evaluate SINDy model at u=0 to get drift
        xdot_np = self.sindy_model.predict(x_np, u=u_zero)
        f = torch.from_numpy(xdot_np).float().to(x.device)
        return f.unsqueeze(-1)  # (bs, n_dims, 1)

    def _g(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-dependent component g(x).

        We estimate g(x) by finite differences on the SINDy model:
            g_j(x) ≈ (f_hat(x, e_j) - f_hat(x, 0)) / 1.0

        This gives the linear control gain matrix at each state.

        Args:
            x: (bs, n_dims) tensor of states
            params: scenario parameters (unused)
        Returns:
            g: (bs, n_dims, n_controls) tensor
        """
        batch_size = x.shape[0]
        x_np = x.detach().cpu().numpy()
        u_zero = np.zeros((batch_size, self._n_controls))

        # Get baseline (drift)
        xdot_base = self.sindy_model.predict(x_np, u=u_zero)

        # Compute g by perturbing each control dimension
        g_np = np.zeros((batch_size, self._n_dims, self._n_controls))
        for j in range(self._n_controls):
            u_pert = np.zeros((batch_size, self._n_controls))
            u_pert[:, j] = 1.0
            xdot_pert = self.sindy_model.predict(x_np, u=u_pert)
            g_np[:, :, j] = xdot_pert - xdot_base

        g = torch.from_numpy(g_np).float().to(x.device)
        return g

    def compute_A_matrix(self, scenario: Optional[Scenario] = None) -> np.ndarray:
        """
        Linearize the LEARNED drift f_hat about the goal using central finite
        differences, evaluated solely through the SINDy model.

        The base-class implementation differentiates closed_loop_dynamics with
        autograd, but the SINDy prediction is a detached numpy evaluation whose
        autograd gradient is zero (the straight-through term carries no gradient).
        We therefore compute the state Jacobian A = d f_hat / d x at (x_star, u=0)
        directly from the learned model, so that the LQR gain K and the Lyapunov
        matrix P derive purely from f_hat rather than from the true dynamics.

        Args:
            scenario: unused (the SINDy model is a single learned model)
        Returns:
            A: (n_dims, n_dims) numpy array
        """
        x0_np = self.goal_point.detach().cpu().numpy()
        u0_np = self.u_eq.detach().cpu().numpy()
        n = self._n_dims
        eps = 1e-4

        A = np.zeros((n, n))
        for j in range(n):
            x_plus = x0_np.copy()
            x_minus = x0_np.copy()
            x_plus[0, j] += eps
            x_minus[0, j] -= eps
            f_plus = self.sindy_model.predict(x_plus, u=u0_np)
            f_minus = self.sindy_model.predict(x_minus, u=u0_np)
            A[:, j] = ((f_plus - f_minus) / (2.0 * eps)).flatten()

        return A

    def closed_loop_dynamics(
        self, x: torch.Tensor, u: torch.Tensor, params: Optional[Scenario] = None
    ) -> torch.Tensor:
        """
        Return x_dot = f_hat(x, u) using the SINDYc model.

        This is the crucial override: instead of calling f(x) + g(x)*u with
        hardcoded physics, we evaluate the sparse-identified model directly.

        Corresponds to Equation (12) in the paper:
            f_hat(x, u) = Ξ_hat^T * Θ(x, u)^T

        Args:
            x: (bs, n_dims) tensor of states
            u: (bs, n_controls) tensor of controls
            params: scenario parameters (ignored — single learned model)
        Returns:
            xdot: (bs, n_dims) tensor of state derivatives
        """
        batch_size = x.shape[0]
        x_np = x.detach().cpu().numpy()
        u_np = u.detach().cpu().numpy()

        # Evaluate the SINDy model: Θ(x, u) @ Ξ_hat
        xdot_np = self.sindy_model.predict(x_np, u=u_np)

        xdot = torch.from_numpy(xdot_np).float().to(x.device)

        # Preserve gradient flow: use straight-through estimator
        # The SINDy prediction is used as a detached target, but we add a
        # zero-gradient term so the tensor stays on the computation graph
        if x.requires_grad:
            xdot = xdot + 0.0 * x.sum(dim=-1, keepdim=True)

        return xdot

    def compute_dynamics_error(
        self,
        true_system: ControlAffineSystem,
        num_samples: int = 10000,
    ) -> dict:
        """
        Compute the dynamics error δ = sup ||f - f_hat||_2 over sampled states.

        This is the key quantity that enters Theorem 6:
            lim sup V(x_t) ≤ Cε^(1/n) + Bδ/λ₁

        Args:
            true_system: the ground truth system to compare against
            num_samples: number of points to evaluate
        Returns:
            dict with error metrics
        """
        # Sample states and controls
        x = true_system.sample_state_space(num_samples)
        u_upper, u_lower = true_system.control_limits
        u = torch.rand(num_samples, self._n_controls) * (u_upper - u_lower) + u_lower

        with torch.no_grad():
            # True dynamics
            xdot_true = true_system.closed_loop_dynamics(x, u)
            # Estimated dynamics
            xdot_est = self.closed_loop_dynamics(x, u)

        residual = xdot_true - xdot_est
        pointwise_error = torch.norm(residual, dim=-1)

        return {
            "delta_sup": pointwise_error.max().item(),
            "delta_mean": pointwise_error.mean().item(),
            "delta_median": pointwise_error.median().item(),
            "delta_std": pointwise_error.std().item(),
        }


def load_sindy_system(
    model_path: str,
    reference_system: ControlAffineSystem,
    dt: float = 0.01,
    controller_dt: Optional[float] = None,
    scenarios: Optional[ScenarioList] = None,
) -> SINDyEstimatedSystem:
    """
    Convenience function to load a saved SINDy model and wrap it.

    Args:
        model_path: path to the pickled PySINDy model
        reference_system: the true system (for geometry/limits)
        dt: simulation timestep
        controller_dt: controller period
        scenarios: scenario list
    Returns:
        SINDyEstimatedSystem instance
    """
    with open(model_path, "rb") as f:
        sindy_model = pickle.load(f)

    return SINDyEstimatedSystem(
        sindy_model=sindy_model,
        reference_system=reference_system,
        dt=dt,
        controller_dt=controller_dt,
        scenarios=scenarios,
    )
