"""
Empirical estimation of the Theorem 6 constants.

Theorem 6 predicts the certificate contracts to the sublevel set

    lim sup_t V(x_t) <= C eps^{1/n} + B delta / lambda_1,

so to overlay the bound on measured data we need empirical estimates of the
three constants, computed directly from the trained certificate rather than
the hardcoded placeholders used in evaluate.py:

    B            = sup ||grad V||      over the operating region
    lambda_1     = exponential decay rate fitted to the V(x_t) rollouts
    C eps^{1/n}  = bad-set contribution, from the measured fraction of the
                   operating region on which the learned dissipation fails.

These are deliberately simple, transparent estimators: the goal is a
defensible line through the sweep points, not a tight constant.
"""
from typing import Optional

import numpy as np
import torch


def estimate_B(
    controller,
    domain=((-1.0, 1.0), (-1.0, 1.0)),
    n_grid: int = 60,
) -> float:
    """Estimate B = sup ||grad V||_2 over a grid of the operating region.

    Args:
        controller: trained NeuralCLBFController exposing V(x).
        domain: per-dimension (low, high) box to grid over.
        n_grid: points per dimension.

    Returns:
        Maximum gradient norm of V over the grid.
    """
    axes = [torch.linspace(lo, hi, n_grid) for (lo, hi) in domain]
    mesh = torch.meshgrid(*axes, indexing="ij")
    x = torch.stack([m.reshape(-1) for m in mesh], dim=-1).float()
    x.requires_grad_(True)

    V = controller.V(x)
    grad = torch.autograd.grad(V.sum(), x, create_graph=False)[0]
    grad_norms = torch.norm(grad, dim=-1)
    return float(grad_norms.max().item())


def estimate_lambda1(
    V_values: np.ndarray,
    t_array: np.ndarray,
    floor: float = 1e-9,
) -> float:
    """Fit an exponential decay rate lambda_1 to mean V(x_t).

    Fits log V_mean(t) ~ log V0 - lambda_1 * t by least squares over the
    transient where V is still above the numerical floor.

    Args:
        V_values: (num_rollouts, T) certificate values along rollouts.
        t_array: (T,) time stamps.
        floor: ignore samples with V below this (post-convergence noise).

    Returns:
        Estimated decay rate lambda_1 (>0); falls back to a small positive
        value if the fit is degenerate.
    """
    V_mean = np.asarray(V_values).mean(axis=0)
    mask = V_mean > floor
    if mask.sum() < 3:
        return 1.0  # not enough transient to fit; use the design lambda
    t = np.asarray(t_array)[mask]
    y = np.log(V_mean[mask])
    # slope of log V vs t is -lambda_1
    slope, _ = np.polyfit(t, y, 1)
    lambda1 = float(-slope)
    return lambda1 if lambda1 > 1e-3 else 1e-3


def estimate_Ceps(
    controller,
    dynamics_model,
    clf_lambda: float = 1.0,
    domain=((-1.0, 1.0), (-1.0, 1.0)),
    n_grid: int = 60,
    C: float = 1.0,
) -> dict:
    """Estimate the bad-set contribution C * eps^{1/n}.

    The bad set is the region where the learned dissipation condition fails
    along the trained controller, i.e.

        grad V(x)^T f_hat(x, u(x)) + lambda V(x) > 0.

    We measure its volume fraction on a grid, multiply by the box volume to
    get eps = Vol(Omega), and return C * eps^{1/n} with n = state dimension.

    Args:
        controller: trained NeuralCLBFController (provides V and the policy).
        dynamics_model: the SINDy-estimated system (provides f_hat).
        clf_lambda: the lambda used in the dissipation condition.
        domain: per-dimension (low, high) box.
        n_grid: points per dimension.
        C: proportionality constant (kept explicit; default 1).

    Returns:
        dict with bad_set_fraction, eps (volume), n, and C_eps_term.
    """
    n = len(domain)
    axes = [torch.linspace(lo, hi, n_grid) for (lo, hi) in domain]
    mesh = torch.meshgrid(*axes, indexing="ij")
    x = torch.stack([m.reshape(-1) for m in mesh], dim=-1).float()
    x.requires_grad_(True)

    V = controller.V(x)
    grad = torch.autograd.grad(V.sum(), x, create_graph=False)[0]

    with torch.no_grad():
        u = controller(x.detach())
        f_hat = dynamics_model.closed_loop_dynamics(x.detach(), u)
        Vdot = (grad.detach() * f_hat).sum(dim=-1)
        dissipation = Vdot + clf_lambda * V.detach().flatten()
        bad = (dissipation > 0).float()

    bad_fraction = float(bad.mean().item())
    box_volume = float(np.prod([hi - lo for (lo, hi) in domain]))
    eps = bad_fraction * box_volume
    C_eps_term = C * (eps ** (1.0 / n)) if eps > 0 else 0.0

    return {
        "bad_set_fraction": bad_fraction,
        "eps": eps,
        "n": n,
        "C_eps_term": C_eps_term,
    }
