#!/usr/bin/env bash
# =============================================================================
# End-to-end pipeline: SINDYc-guided Lyapunov Diffusion Control
#
# This script runs the full pipeline:
#   A) Collect safe transition data from the true system
#   B) Run SINDYc to identify the dynamics f_hat
#   C) (Surrogate class is already defined in systems/sindy_estimated_system.py)
#   D) Train the CLBF controller using learned dynamics
#   E) Evaluate on the true system to validate Theorem 6
# =============================================================================
set -e

echo "================================================================"
echo " SINDYc-Guided Lyapunov Diffusion Control Pipeline"
echo " (Validating Theorem 6: Convergence under Learned Dynamics)"
echo "================================================================"
echo ""

# Configuration
SYSTEM="inverted_pendulum"
DATA_DIR="data"
RESULTS_DIR="results/sindy_evaluation"
NUM_TRAJECTORIES=200
TRAJECTORY_LENGTH=100
EXPLORATION_STD=0.5
POLY_DEGREE=3
STLSQ_THRESHOLD=0.05
MAX_EPOCHS=51

mkdir -p "$DATA_DIR" "$RESULTS_DIR"

# ---- Step A: Collect transition data ----
echo "=== Step A: Collecting safe transition data ==="
python -m neural_clbf.sindy.collect_data \
    --system "$SYSTEM" \
    --num_trajectories "$NUM_TRAJECTORIES" \
    --trajectory_length "$TRAJECTORY_LENGTH" \
    --exploration_std "$EXPLORATION_STD" \
    --output "$DATA_DIR/sindy_transitions.npz"
echo ""

# ---- Step B: SINDYc identification ----
echo "=== Step B: Running SINDYc identification ==="
python -m neural_clbf.sindy.identify_dynamics \
    --data "$DATA_DIR/sindy_transitions.npz" \
    --output "$DATA_DIR/sindy_model.pkl" \
    --poly_degree "$POLY_DEGREE" \
    --threshold "$STLSQ_THRESHOLD" \
    --include_trig
echo ""

# ---- Step D: Train with learned dynamics ----
echo "=== Step D: Training CLBF controller on learned dynamics ==="
python -m neural_clbf.training.train_inverted_pendulum_sindy \
    --sindy_model "$DATA_DIR/sindy_model.pkl" \
    --max_epochs "$MAX_EPOCHS" \
    --accelerator auto
echo ""

# ---- Step E: Evaluate on true dynamics ----
echo "=== Step E: Evaluating controller on true system ==="
# Find the latest checkpoint
CKPT=$(find logs/inverted_pendulum_sindy -name "*.ckpt" -type f | sort | head -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found. Training may have failed."
    exit 1
fi
echo "Using checkpoint: $CKPT"

python -m neural_clbf.sindy.evaluate \
    --checkpoint "$CKPT" \
    --sindy_model "$DATA_DIR/sindy_model.pkl" \
    --num_rollouts 100 \
    --t_sim 10.0 \
    --output_dir "$RESULTS_DIR"

echo ""
echo "================================================================"
echo " Pipeline complete. Results in: $RESULTS_DIR/"
echo "================================================================"
