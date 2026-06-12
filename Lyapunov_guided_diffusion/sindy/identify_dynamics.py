"""
Step B: Run SINDYc to learn f_hat(x, u).

Uses PySINDy with control inputs (SINDYc) to identify the dynamics
from collected transition data. Builds a feature library including
polynomials and trigonometric functions, then performs sparse regression.

Usage:
    python -m Lyapunov_guided_diffusion.sindy.identify_dynamics \
        --data data/sindy_transitions.npz \
        --output data/sindy_model.pkl
"""
import argparse
import os
import pickle

import numpy as np
import pysindy as ps


def build_feature_library(poly_degree: int = 3, include_trig: bool = True):
    """
    Build a comprehensive SINDYc feature library.

    For the inverted pendulum with state [theta, theta_dot] and control [u],
    the true dynamics involve sin(theta), so trigonometric features are essential.

    Args:
        poly_degree: max polynomial degree for state/control features
        include_trig: whether to include sin/cos features

    Returns:
        A PySINDy feature library (GeneralizedLibrary or ConcatLibrary)
    """
    libraries = []

    # Polynomial features up to specified degree (includes cross terms with control)
    poly_lib = ps.PolynomialLibrary(degree=poly_degree, include_interaction=True)
    libraries.append(poly_lib)

    # Trigonometric features (sin, cos of each state variable)
    if include_trig:
        trig_lib = ps.FourierLibrary(n_frequencies=1)
        libraries.append(trig_lib)

    # Combine into a concatenated library
    combined_library = ps.ConcatLibrary(libraries)
    return combined_library


def identify_dynamics(
    X: np.ndarray,
    U: np.ndarray,
    Xdot: np.ndarray,
    poly_degree: int = 3,
    include_trig: bool = True,
    threshold: float = 0.05,
    alpha: float = 0.5,
    max_iter: int = 50,
):
    """
    Run SINDYc identification.

    Args:
        X: (N, n_dims) state data
        U: (N, n_controls) control data
        Xdot: (N, n_dims) derivative data
        poly_degree: polynomial degree for library
        include_trig: include trig functions
        threshold: STLSQ sparsification threshold
        alpha: elastic net mixing (0=ridge, 1=lasso)
        max_iter: max iterations for STLSQ

    Returns:
        model: fitted PySINDy model
    """
    # Build the feature library
    feature_library = build_feature_library(
        poly_degree=poly_degree,
        include_trig=include_trig,
    )

    # Use Sequentially Thresholded Least Squares (STLSQ) optimizer
    # This is the core SINDy algorithm from Brunton et al. 2016
    optimizer = ps.STLSQ(
        threshold=threshold,
        alpha=alpha,
        max_iter=max_iter,
    )

    # Build and fit the SINDYc model
    # The key distinction: we pass u as the control input so PySINDy
    # augments the feature matrix with control terms
    model = ps.SINDy(
        feature_library=feature_library,
        optimizer=optimizer,
    )

    # Fit using pre-computed derivatives (x_dot provided directly)
    # feature_names goes in fit() in PySINDy 2.x
    # t=1.0 signals that x_dot is directly provided (no differentiation needed)
    model.fit(
        X, t=1.0, u=U, x_dot=Xdot,
        feature_names=["theta", "theta_dot", "u"],
    )

    return model


def evaluate_model(model, X, U, Xdot):
    """Compute reconstruction error metrics."""
    Xdot_pred = model.predict(X, u=U)
    residual = Xdot - Xdot_pred

    mse = np.mean(residual ** 2)
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(residual))
    # delta = sup ||f - f_hat||_2 (worst-case vector field error, Eq. 13 in paper)
    delta = np.max(np.linalg.norm(residual, axis=1))

    return {
        "mse": mse,
        "rmse": rmse,
        "max_error": max_error,
        "delta_sup_norm": delta,
    }


def main():
    parser = argparse.ArgumentParser(description="SINDYc dynamics identification")
    parser.add_argument("--data", type=str, default="data/sindy_transitions.npz",
                        help="Path to collected transition data")
    parser.add_argument("--output", type=str, default="data/sindy_model.pkl",
                        help="Path to save the fitted model")
    parser.add_argument("--poly_degree", type=int, default=3,
                        help="Max polynomial degree")
    parser.add_argument("--include_trig", action="store_true", default=True,
                        help="Include trigonometric features")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="STLSQ sparsification threshold")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Elastic net regularization mixing")
    parser.add_argument("--max_iter", type=int, default=50,
                        help="Max STLSQ iterations")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    data = np.load(args.data)
    X = data["X"]
    U = data["U"]
    Xdot = data["Xdot"]
    print(f"  X: {X.shape}, U: {U.shape}, Xdot: {Xdot.shape}")

    # Identify dynamics
    print("Running SINDYc identification...")
    model = identify_dynamics(
        X, U, Xdot,
        poly_degree=args.poly_degree,
        include_trig=args.include_trig,
        threshold=args.threshold,
        alpha=args.alpha,
        max_iter=args.max_iter,
    )

    # Print discovered equations
    print("\n=== Discovered Equations ===")
    model.print()

    # Evaluate
    print("\n=== Reconstruction Metrics ===")
    metrics = evaluate_model(model, X, U, Xdot)
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    print(f"\n  δ (sup vector-field error) = {metrics['delta_sup_norm']:.6f}")
    print(f"  This δ enters Theorem 6 bound: lim sup V(x_t) ≤ Cε^(1/n) + Bδ/λ₁")

    # Save model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved SINDYc model to {args.output}")

    # Also save the coefficient matrix for inspection
    coeff_path = args.output.replace(".pkl", "_coefficients.npz")
    np.savez(
        coeff_path,
        coefficients=model.coefficients(),
        feature_names=model.get_feature_names(),
    )
    print(f"Saved coefficient matrix to {coeff_path}")


if __name__ == "__main__":
    main()
