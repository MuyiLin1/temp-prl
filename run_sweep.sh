#!/usr/bin/env bash
# =============================================================================
# Data trade-off sweep for the learned-dynamics safety experiment.
#
# Runs the full grid (dataset size x measurement noise x seeds), appending one
# row per run to data/sweep_results.csv, then renders tradeoff.png.
#
# Designed to run unattended on the GPU cluster:
#   1. copy the repo (or just this folder) to the cluster,
#   2. activate the environment that has torch + pysindy + pytorch_lightning,
#   3. run:   bash run_sweep.sh
#   4. copy data/sweep_results.csv and tradeoff.png back.
#
# The sweep is crash-safe: results are flushed to CSV after every run, and the
# CSV is appended to, so re-running resumes by adding rows (de-dup later).
# =============================================================================
set -e

# --- Config (override via environment, e.g. ACCELERATOR=gpu bash run_sweep.sh) ---
#
# Corrected design: hold the dataset size FIXED at a healthy value and use
# measurement noise as the delta knob. This keeps delta in the small,
# meaningful range (~0.05 -> ~0.5) where Theorem 6's bound is non-vacuous,
# instead of the tiny-dataset regime where SINDy extrapolation blows delta up
# into the thousands and the bound becomes meaningless.
PYTHON="${PYTHON:-python}"                       # use the env's python
ACCELERATOR="${ACCELERATOR:-gpu}"                # gpu on the cluster, cpu locally
MAX_EPOCHS="${MAX_EPOCHS:-30}"                   # converged floor, not 1e-14
SEEDS="${SEEDS:-0 1 2}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-50}"       # fixed healthy data size
OBS_NOISE="${OBS_NOISE:-0.0 0.02 0.05 0.1 0.15 0.2 0.3}"  # delta knob
NUM_ROLLOUTS="${NUM_ROLLOUTS:-100}"
T_SIM="${T_SIM:-10.0}"
DELTA_MAX="${DELTA_MAX:-1.0}"                     # drop vacuous large-delta configs in the plot
OUTPUT="${OUTPUT:-data/sweep_results.csv}"

echo "================================================================"
echo " SINDy data trade-off sweep"
echo "   accelerator   = $ACCELERATOR"
echo "   max_epochs    = $MAX_EPOCHS"
echo "   seeds         = $SEEDS"
echo "   num_traj grid = $NUM_TRAJECTORIES"
echo "   noise grid    = $OBS_NOISE"
echo "   output        = $OUTPUT"
echo "================================================================"

mkdir -p data

$PYTHON -m Lyapunov_guided_diffusion.sindy.sweep \
    --num_trajectories $NUM_TRAJECTORIES \
    --obs_noise_std $OBS_NOISE \
    --seeds $SEEDS \
    --max_epochs "$MAX_EPOCHS" \
    --accelerator "$ACCELERATOR" \
    --num_rollouts "$NUM_ROLLOUTS" \
    --t_sim "$T_SIM" \
    --output "$OUTPUT"

echo ""
echo "=== Rendering trade-off figure ==="
$PYTHON -m Lyapunov_guided_diffusion.sindy.plot_tradeoff \
    --input "$OUTPUT" \
    --output tradeoff.png \
    --delta_max "$DELTA_MAX"

echo ""
echo "Done. Copy back: $OUTPUT and tradeoff.png"
