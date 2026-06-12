# Diffusion-based Control Lyapunov Barrier Functions

## Abstract

This repository contains the implementation code for the NeurIPS 2025 submission "Safe and Stable Control with Lyapunov-Guided Diffusion Models." Our work introduces a novel framework for safe and stable control that leverages Lyapunov-guided diffusion models. On the other hand, the diffusion-sampled policy is used to generate trajectories that are used to update the CLBF.

## Methodology

### Probabilistic CLBF Formulation

We reformulate the traditional CLBF optimization problem as a probabilistic sampling task. The target trajectory distribution is defined as a Gibbs measure:

```
p(U) ∝ p_safe(U) · p_stable(U) · p_cost(U)
```

where:
- **p_safe**: Ensures trajectories remain within safe regions (V(x) ≤ c)
- **p_stable**: Ensures CLBF decreases along trajectories through an Almost-Lyapunov formulation
- **p_cost**: Biases sampling toward nominal or low-cost trajectories

### Almost Lyapunov Theory

We leverage Almost Lyapunov theory in our formulation. This allows for occasional violations of the Lie derivative condition as long as they occur with sufficiently small probability and in regions of minimal influence. Mathematically, we implement this as:

```
p_stable ∝ exp(-1/γ₂ · Σᵗ || [L_f V(xₜ) + λV(xₜ)]⁺ ||²)
```

where [z]⁺ = ReLU(z) and γ₂ is a small temperature parameter.

### Diffusion Sampling Algorithm

Our implementation uses Monte Carlo score ascent within diffusion sampling:


1. Reverse diffusion gradually denoises trajectories guided by CLBF
2. The score function is estimated using Monte Carlo sampling

### CLBF Learning

The CLBF itself is updated using sampled trajectories to satisfy several constraints including positivity, goal minimization, safety level-sets, and stability conditions.

## Code Structure

- `neural_clbf/controllers/diffusion_clbf.py`: Main implementation of diffusion-based CLBF controller
- `neural_clbf/systems/`: System dynamics implementations
- `evaluation/`: Training and evaluation scripts

## Running Experiments

To reproduce the results in our paper:

```bash
# Clone the repository
git clone [anonymous-repo-link]

# Install dependencies
pip install -r requirements.txt

# Run example experiment
python neural_clbf/training/xxx.py
```

## Results

Our method demonstrates significant improvements over baseline approaches:

1. **Success Rate**: Higher probability of reaching goals while maintaining safety and stability
2. **Non-convex Constraints**: Effective handling of non-convex safe regions
3. **Efficiency**: Requires fewer inference time compared to model-based based diffusion approaches

## License

Anonymous License - To be updated after review process
