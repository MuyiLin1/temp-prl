import itertools
from typing import Tuple, List, Optional
from collections import OrderedDict
import random

import numpy as np

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import ScenarioList
from neural_clbf.controllers.clf_controller import CLFController
from neural_clbf.controllers.controller_utils import normalize_with_angles
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.experiments import ExperimentSuite




class NeuralCLBFController(pl.LightningModule, CLFController):
    """
    A neural rCLBF controller. Differs from the CLFController in that it uses a
    neural network to learn the CLF, and it turns it from a CLF to a CLBF by making sure
    that a level set of the CLF separates the safe and unsafe regions.

    More specifically, the CLBF controller looks for a V that satisfies the following
    conditions for some constants c, λ > 0:

    Equilibrium:           V(x*) = 0
    Positivity:            V(x) > 0,     ∀x ∈ X \ {x*}
    Safe State:            V(x) ≤ c,     ∀x ∈ X_s
    Unsafe State:          V(x) > c,     ∀x ∈ X_u
    Uniform Dissipation:   inf L_f V(x) + λV(x) ≤ 0,   ∀x ∈ X \ {x*}
                           u∈U

    CLBFs are Lyapunov-like functions that simultaneously guarantee a system's safety and
    stability. Under these conditions, there exists a control policy that steers the system
    toward a unique equilibrium point with monotonic decrease. In general, the uniform
    dissipation implies exponential stability, and since the c-sublevel set of V is forward
    invariant and contains the safe set, we prove that the unsafe region is not reachable
    from the safe region.
    """

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        scenarios: ScenarioList,
        datamodule: EpisodicDataModule,
        experiment_suite: ExperimentSuite,
        clbf_hidden_layers: int = 2,
        clbf_hidden_size: int = 48,
        clf_lambda: float = 1.0,
        safe_level: float = 1.0,
        clf_relaxation_penalty: float = 50.0,
        controller_period: float = 0.01,
        primal_learning_rate: float = 1e-3,
        epochs_per_episode: int = 5,
        penalty_scheduling_rate: float = 0.0,
        num_init_epochs: int = 5,
        barrier: bool = True,
        add_nominal: bool = False,
        normalize_V_nominal: bool = False,
        disable_gurobi: bool = True,
    ):
        """Initialize the controller.

        args:
            dynamics_model: the control-affine dynamics of the underlying system
            scenarios: a list of parameter scenarios to train on
            experiment_suite: defines the experiments to run during training
            clbf_hidden_layers: number of hidden layers to use for the CLBF network
            clbf_hidden_size: number of neurons per hidden layer in the CLBF network
            clf_lambda: convergence rate for the CLBF
            safe_level: safety level set value for the CLBF
            controller_period: the timestep to use in simulating forward Vdot
            primal_learning_rate: the learning rate for SGD for the network weights,
                                  applied to the CLBF decrease loss
            epochs_per_episode: the number of epochs to include in each episode
            penalty_scheduling_rate: the rate at which to ramp the rollout relaxation
                                     penalty up to clf_relaxation_penalty. Set to 0 to
                                     disable penalty scheduling (use constant penalty)
            num_init_epochs: the number of epochs to pretrain the controller on the
                             linear controller
            barrier: if True, train the CLBF to act as a barrier functions. If false,
                     effectively trains only a CLF.
            add_nominal: if True, add the nominal V
            normalize_V_nominal: if True, normalize V_nominal so that its average is 1
            disable_gurobi: if True, Gurobi will not be used during evaluation. 
                Default is train with CVXPYLayers, evaluate with Gurobi; 
                setting this to true will evaluate with CVXPYLayers instead 
                (to avoid requiring a Gurobi license)
        """
        # Initialize LightningModule and CLFController separately due to
        # PyTorch Lightning >= 2.0 not accepting arbitrary kwargs in __init__
        pl.LightningModule.__init__(self)
        CLFController.__init__(
            self,
            dynamics_model=dynamics_model,
            scenarios=scenarios,
            experiment_suite=experiment_suite,
            clf_lambda=clf_lambda,
            clf_relaxation_penalty=clf_relaxation_penalty,
            controller_period=controller_period,
            disable_gurobi=disable_gurobi,
        )
        self.save_hyperparameters()

        # Save the provided model
        # self.dynamics_model = dynamics_model
        self.scenarios = scenarios
        self.n_scenarios = len(scenarios)

        # Save the datamodule
        self.datamodule = datamodule

        # Save the experiments suits
        self.experiment_suite = experiment_suite

        # Save the other parameters
        self.safe_level = safe_level
        self.unsafe_level = safe_level
        self.primal_learning_rate = primal_learning_rate
        self.epochs_per_episode = epochs_per_episode
        self.penalty_scheduling_rate = penalty_scheduling_rate
        self.num_init_epochs = num_init_epochs
        self._train_step_outputs = []
        self._val_step_outputs = []
        self.barrier = barrier
        self.add_nominal = add_nominal
        self.normalize_V_nominal = normalize_V_nominal
        self.V_nominal_mean = 1.0

        # Compute and save the center and range of the state variables
        x_max, x_min = dynamics_model.state_limits
        self.x_center = (x_max + x_min) / 2.0
        self.x_range = (x_max - x_min) / 2.0
        # Scale to get the input between (-k, k), centered at 0
        self.k = 1.0
        self.x_range = self.x_range / self.k
        # We shouldn't scale or offset any angle dimensions
        self.x_center[self.dynamics_model.angle_dims] = 0.0
        self.x_range[self.dynamics_model.angle_dims] = 1.0

        # Some of the dimensions might represent angles. We want to replace these
        # dimensions with two dimensions: sin and cos of the angle. To do this, we need
        # to figure out how many numbers are in the expanded state
        n_angles = len(self.dynamics_model.angle_dims)
        self.n_dims_extended = self.dynamics_model.n_dims + n_angles

        # Define the CLBF network, which we denote V
        self.clbf_hidden_layers = clbf_hidden_layers
        self.clbf_hidden_size = clbf_hidden_size
        # We're going to build the network up layer by layer, starting with the input
        self.V_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.V_layers["input_linear"] = nn.Linear(
            self.n_dims_extended, self.clbf_hidden_size
        )
        self.V_layers["input_activation"] = nn.Tanh()
        for i in range(self.clbf_hidden_layers):
            self.V_layers[f"layer_{i}_linear"] = nn.Linear(
                self.clbf_hidden_size, self.clbf_hidden_size
            )
            if i < self.clbf_hidden_layers - 1:
                self.V_layers[f"layer_{i}_activation"] = nn.Tanh()
        self.V_layers["output_linear"] = nn.Linear(self.clbf_hidden_size, 1)
        self.V_nn = nn.Sequential(self.V_layers)

    def prepare_data(self):
        return self.datamodule.prepare_data()

    def setup(self, stage: Optional[str] = None):
        return self.datamodule.setup(stage)

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()

    def V_with_jacobian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the CLBF value and its Jacobian

        args:
            x: bs x self.dynamics_model.n_dims the points at which to evaluate the CLBF
        returns:
            V: bs tensor of CLBF values
            JV: bs x 1 x self.dynamics_model.n_dims Jacobian of each row of V wrt x
        """
        # Apply the offset and range to normalize about zero
        x_norm = normalize_with_angles(self.dynamics_model, x)

        # Compute the CLBF layer-by-layer, computing the Jacobian alongside

        # We need to initialize the Jacobian to reflect the normalization that's already
        # been done to x
        bs = x_norm.shape[0]
        JV = torch.zeros(
            (bs, self.n_dims_extended, self.dynamics_model.n_dims)
        ).type_as(x)
        # and for each non-angle dimension, we need to scale by the normalization
        for dim in range(self.dynamics_model.n_dims):
            JV[:, dim, dim] = 1.0 / self.x_range[dim].type_as(x)

        # And adjust the Jacobian for the angle dimensions
        for offset, sin_idx in enumerate(self.dynamics_model.angle_dims):
            cos_idx = self.dynamics_model.n_dims + offset
            JV[:, sin_idx, sin_idx] = x_norm[:, cos_idx]
            JV[:, cos_idx, sin_idx] = -x_norm[:, sin_idx]

        # Now step through each layer in V
        V = x_norm
        for layer in self.V_nn:
            V = layer(V)

            if isinstance(layer, nn.Linear):
                JV = torch.matmul(layer.weight, JV)
            elif isinstance(layer, nn.Tanh):
                JV = torch.matmul(torch.diag_embed(1 - V ** 2), JV)
            elif isinstance(layer, nn.ReLU):
                JV = torch.matmul(torch.diag_embed(torch.sign(V)), JV)

        # Compute the final activation
        JV = torch.bmm(V.unsqueeze(1), JV)
        V = 0.5 * (V * V).sum(dim=1)

        if self.add_nominal:
            # Get the nominal Lyapunov function
            P = self.dynamics_model.P.type_as(x)
            x0 = self.dynamics_model.goal_point.type_as(x)
            # Reshape to use pytorch's bilinear function
            P = P.reshape(1, self.dynamics_model.n_dims, self.dynamics_model.n_dims)
            V_nominal = 0.5 * F.bilinear(x - x0, x - x0, P).squeeze()
            # Reshape again to calculate the gradient
            P = P.reshape(self.dynamics_model.n_dims, self.dynamics_model.n_dims)
            JV_nominal = F.linear(x - x0, P)
            JV_nominal = JV_nominal.reshape(x.shape[0], 1, self.dynamics_model.n_dims)

            if self.normalize_V_nominal:
                V_nominal /= self.V_nominal_mean
                JV_nominal /= self.V_nominal_mean

            V = V + V_nominal
            JV = JV + JV_nominal

        return V, JV
    
    def u_trajectory_nominal(self, x: torch.Tensor, horizon: int = 100) -> torch.Tensor:

        # u_sequence: bs x n_steps (horizon) x control_dim
        # x_sequence: bs x n_steps (horizon) x state_dim


        batch_size = x.shape[0]

        #initialize the control and state sequence
        u_sequence = torch.randn(batch_size, horizon, self.dynamics_model.n_controls).type_as(x)
        x_sequence = torch.zeros(batch_size, horizon, self.dynamics_model.n_dims).type_as(x)

        x_sequence[:, 0, :] = x

        x_current = x.clone()
        # print("x_sequence", x_sequence.shape)
        # print("x_current", x_current.shape)

        #recursive update the control and state for the next n_steps
        for i in range(horizon):
            u_current = self.dynamics_model.u_nominal(x_current)

            # u_current = u_current.to(device=x.device)

            # clip the control to be within the limits
            # u_current = torch.clamp(u_current, self.dynamics_model.control_limits[1], self.dynamics_model.control_limits[0])

            # store the current control
            u_sequence[:, i, :] = u_current

            # update the state
            x_dot = self.dynamics_model.closed_loop_dynamics(x_current, u_current)
            x_next = x_current + self.controller_period * x_dot

            # print("x_next", x_next.shape)

            # store the current state
            # x_sequence[:, i+1, :] = x_next

            x_current = x_next

            if i < horizon - 1:
                x_sequence[:, i+1, :] = x_next

        # print("x_sequence", x_sequence[1, :, :])

        return x_sequence, u_sequence



    def sample_trajectories(self, x: torch.Tensor, 
                            horizon: int = 10, 
                            num_samples: int = 50, 
                            iters: int = 5, 
                            temperature: float = 1.0,
                            use_mean: bool = True
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample trajectories using diffusion in normalized action space."""

        # Control limits
        u_max, u_min = self.dynamics_model.control_limits
        u_min = u_min.type_as(x)
        u_max = u_max.type_as(x)

        # Normalize helper functions
        def normalize_u(u):
            return 2.0 * (u - u_min) / (u_max - u_min) - 1.0

        def denormalize_u(u_norm):
            return 0.5 * (u_norm + 1.0) * (u_max - u_min) + u_min

        # Get nominal trajectory (normalized)
        _, u_nominal = self.u_trajectory_nominal(x, horizon=horizon)
        u_nominal = normalize_u(u_nominal)

        batch_size = x.shape[0]
        n_dims = self.dynamics_model.n_dims
        n_controls = self.dynamics_model.n_controls

        # Diffusion parameters
        n_diffuse = 50
        beta0 = 1e-4
        betaT = 2e-2
        betas = torch.linspace(beta0, betaT, n_diffuse).type_as(x)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        sigmas = torch.sqrt(1 - alphas_bar)

        # Start with normalized zero controls
        Y_current = torch.zeros((batch_size, horizon, n_controls)).type_as(x)
        trajectories_all = []

        # Reverse diffusion process
        for i in range(n_diffuse - 1, 0, -1):
            Yi = Y_current * torch.sqrt(alphas_bar[i])
            eps_u = torch.randn(batch_size, num_samples, horizon, n_controls).type_as(x)
            Y0s = eps_u * sigmas[i] + Y_current.unsqueeze(1).expand(-1, num_samples, -1, -1)
            Y0s = torch.clamp(Y0s, -1.0, 1.0)  # Clip in normalized space

            # Simulate trajectories
            trajectories = torch.zeros(batch_size, num_samples, horizon, n_dims).type_as(x)
            trajectories[:, :, 0, :] = x.unsqueeze(1)

            for t in range(horizon - 1):
                current_states = trajectories[:, :, t, :].reshape(-1, n_dims)
                current_controls = denormalize_u(Y0s[:, :, t, :]).reshape(-1, n_controls)

                xdot_total = 0
                for scenario in self.scenarios:
                    xdot_total += self.dynamics_model.closed_loop_dynamics(current_states, current_controls, params=scenario)
                xdot = xdot_total / len(self.scenarios)

                next_states = current_states + self.controller_period * xdot
                next_states = next_states.reshape(batch_size, num_samples, n_dims)
                trajectories[:, :, t+1, :] = next_states

            # Evaluate with CLBF
            violations = torch.zeros(batch_size, num_samples).type_as(x)
            for t in range(horizon):
                states_flat = trajectories[:, :, t, :].reshape(-1, n_dims)
                V_values = self.V(states_flat).reshape(batch_size, num_samples)

                if self.barrier:
                    safe_violation = 2 * torch.nn.functional.relu(V_values - self.safe_level)**2
                    violations += safe_violation

                if t < horizon - 1:
                    V_current = self.V(states_flat)
                    next_states_flat = trajectories[:, :, t+1, :].reshape(-1, n_dims)
                    V_next = self.V(next_states_flat).reshape(batch_size, num_samples)

                    V_dot = (V_next - V_current.reshape(batch_size, num_samples))**2 / self.controller_period
                    decrease_violation = 2 * torch.nn.functional.relu(
                        V_dot + self.clf_lambda * V_current.reshape(batch_size, num_samples)
                    )
                    violations += decrease_violation

            control_distances = torch.sum((Y0s - u_nominal.unsqueeze(1)) ** 2, dim=[2, 3])
            combined_costs = control_distances + 10.0 * violations

            logits = -combined_costs / temperature
            weights = torch.nn.functional.softmax(logits, dim=1)

            Y_updated = torch.sum(weights.unsqueeze(2).unsqueeze(3) * Y0s, dim=1)

            score = 1.0 / (1.0 - alphas_bar[i]) * (-Yi + torch.sqrt(alphas_bar[i]) * Y_updated)
            Y_im1 = 1.0 / torch.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)

            Y_current = Y_im1 / torch.sqrt(alphas_bar[i-1]) if i > 1 else Y_im1
            trajectories_all.append(Y_current.clone())

        # Final best control: denormalize
        best_controls = denormalize_u(Y_current)

        # Simulate final trajectory
        best_trajectories = torch.zeros(batch_size, horizon, n_dims).type_as(x)
        best_trajectories[:, 0, :] = x

        for t in range(horizon - 1):
            current_states = best_trajectories[:, t, :]
            current_controls = best_controls[:, t, :]

            xdot_total = 0
            for scenario in self.scenarios:
                xdot_total += self.dynamics_model.closed_loop_dynamics(current_states, current_controls, params=scenario)
            xdot = xdot_total / len(self.scenarios)

            next_states = current_states + self.controller_period * xdot
            best_trajectories[:, t+1, :] = next_states

        return best_trajectories, best_controls


        # return u_nominal, u_nominal

    # def forward(self, x):
    #     """Determine the control input for a given state using a QP

    #     args:
    #         x: bs x self.dynamics_model.n_dims tensor of state
    #     returns:
    #         u: bs x self.dynamics_model.n_controls tensor of control inputs
    #     """
    #     # s_time = time.time()
    #     # u_forward = self.u(x)
    #     # e_time = time.time()
    #     # print("QP time", e_time - s_time)

    #     # clip 
    #     # u_forward = torch.clamp(u_forward, self.dynamics_model.control_limits[1], self.dynamics_model.control_limits[0])

    #     # return self.dynamics_model.u_nominal(x)
    #     return self.u(x)

    def forward(self, x):
        """Determine the control input for a given state using diffusion-based sampling
        
        args:
            x: bs x self.dynamics_model.n_dims tensor of state
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
        """
        # Sample trajectories using diffusion process and get optimal control sequence
        s_time = time.time()
        
        # Use diffusion-based trajectory sampling with appropriate parameters
        _, best_controls = self.sample_trajectories(
            x, 
            horizon=3,             # Using a shorter horizon for computational efficiency
            num_samples=32,        # Balance between quality and computational cost
            temperature=0.5,       # Temperature for weighting samples
            use_mean=True,         # Use weighted average of trajectories
        )
        
        best_controls_adopt = best_controls.detach()
        e_time = time.time()
        print("Diffusion sampling time", e_time - s_time)
        
        # Only take the first control action from the trajectory (MPC-style)
        u_forward = best_controls_adopt[:, 0, :]
        
        # Ensure controls are within limits
        u_max, u_min = self.dynamics_model.control_limits
        u_forward = torch.clamp(u_forward, u_min.type_as(x), u_max.type_as(x))
        
        return u_forward


    # def forward(self, x):
        
    #     return self.u(x)

    def boundary_loss(
        self,
        x: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        accuracy: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CLBF due to boundary conditions

        args:
            x: the points at which to evaluate the loss,
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
            accuracy: if True, return the accuracy (from 0 to 1) as well as the losses
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        eps = 1e-2
        # Compute loss to encourage satisfaction of the following conditions...
        loss = []

        V = self.V(x)

        #   1.) CLBF should be minimized on the goal point
        V_goal_pt = self.V(self.dynamics_model.goal_point.type_as(x))
        goal_term = 1e1 * V_goal_pt.mean()
        loss.append(("CLBF goal term", goal_term))

        #   1b.) CLBF should be positive everywhere (positivity condition)
        positivity_violation = F.relu(eps - V)  # Penalize when V is less than eps (not positive)
        positivity_term = 1e2 * positivity_violation.mean()
        loss.append(("CLBF positivity term", positivity_term))
        if accuracy:
            positivity_acc = (positivity_violation <= eps).sum() / positivity_violation.nelement()
            loss.append(("CLBF positivity accuracy", positivity_acc))

        # Only train these terms if we have a barrier requirement
        if self.barrier:
            #   2.) 0 < V <= safe_level in the safe region
            V_safe = V[safe_mask]
            safe_violation = F.relu(eps + V_safe - self.safe_level)
            safe_V_term = 1e2 * safe_violation.mean()
            loss.append(("CLBF safe region term", safe_V_term))
            if accuracy:
                safe_V_acc = (safe_violation <= eps).sum() / safe_violation.nelement()
                loss.append(("CLBF safe region accuracy", safe_V_acc))

            #   3.) V >= unsafe_level in the unsafe region
            V_unsafe = V[unsafe_mask]
            unsafe_violation = F.relu(eps + self.unsafe_level - V_unsafe)
            unsafe_V_term = 1e2 * unsafe_violation.mean()
            loss.append(("CLBF unsafe region term", unsafe_V_term))
            if accuracy:
                unsafe_V_acc = (
                    unsafe_violation <= eps
                ).sum() / unsafe_violation.nelement()
                loss.append(("CLBF unsafe region accuracy", unsafe_V_acc))
        
        return loss

    def descent_loss(
            self,
            x: torch.Tensor,
            accuracy: bool = False,
            requires_grad: bool = False,
            horizon: int = 3,
            num_samples: int = 20,
            iters: int = 5,
        ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CLBF due to the descent condition along trajectories
        
        args:
            x: the points at which to evaluate the loss
            accuracy: if True, return the accuracy (from 0 to 1) as well as the losses
            requires_grad: if True, ensure gradients can flow
            horizon: number of steps for trajectory sampling
            num_samples: number of trajectory samples
            iters: number of MPPI iterations to perform
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        # Generate trajectories from the input states
        with torch.no_grad():
            # Sample trajectories using the dynamics model with improved parameters
            sampled_trajectories, sampled_controls = self.sample_trajectories(
                x, 
                horizon=horizon,             # Keep existing horizon parameter
                num_samples=num_samples,     # Keep existing sample count
                temperature=0.25,            # Lower temperature for more exploitation
                use_mean=True                # Use weighted average rather than best trajectory
            )
        
        # Compute loss to encourage satisfaction of the following conditions along trajectories
        loss = []
        batch_size = sampled_trajectories.shape[0]
        trajectory_length = sampled_trajectories.shape[1]
        
        # Flatten the trajectories to process all points at once
        trajectory_points = sampled_trajectories.reshape(-1, self.dynamics_model.n_dims)


        # print("trajectory_points", trajectory_points.shape)
        
        # Compute V values for all trajectory points
        V = self.V(trajectory_points)
        
        eps = 1.0
        clbf_descent_term_lin = torch.tensor(0.0).type_as(x)
        clbf_descent_acc_lin = torch.tensor(0.0).type_as(x)
        
        # Determine which points need the descent condition
        if self.barrier:
            condition_active = torch.sigmoid(10 * (self.safe_level + eps - V))
        else:
            condition_active = torch.ones_like(V)
        
        # Get the Lie derivatives at all trajectory points
        Lf_V, Lg_V = self.V_lie_derivatives(trajectory_points)
        
        # Reshape sampled controls to match the points (excluding the last trajectory point which has no control)
        # We need to handle the reshaping carefully
        controls_flat = sampled_controls.reshape(-1, self.dynamics_model.n_controls)
        
        for i, s in enumerate(self.scenarios):
            # Use the dynamics to compute the derivative of V
            Vdot = Lf_V[:, i, :].unsqueeze(1) + torch.bmm(
                Lg_V[:, i, :].unsqueeze(1),
                controls_flat.reshape(-1, self.dynamics_model.n_controls, 1),
            )
            Vdot = Vdot.reshape(V.shape)
            violation = F.relu(eps + Vdot + self.clf_lambda * V)
            violation = violation * condition_active
            clbf_descent_term_lin = clbf_descent_term_lin + violation.mean()
            clbf_descent_acc_lin = clbf_descent_acc_lin + (violation <= eps).sum() / (
                violation.nelement() * self.n_scenarios
            )

        loss.append(("CLBF descent term (linearized)", clbf_descent_term_lin))
        if accuracy:
            loss.append(("CLBF descent accuracy (linearized)", clbf_descent_acc_lin))

        # Now compute the decrease using simulation
        eps = 1.0
        clbf_descent_term_sim = torch.tensor(0.0).type_as(x)
        clbf_descent_acc_sim = torch.tensor(0.0).type_as(x)
        for s in self.scenarios:
            xdot = self.dynamics_model.closed_loop_dynamics(trajectory_points, controls_flat, params=s)
            trajectory_points_next = trajectory_points + self.dynamics_model.dt * xdot
            V_next = self.V(trajectory_points_next)
            violation = F.relu(
                eps + (V_next - V) / self.controller_period + self.clf_lambda * V
            )
            violation = violation * condition_active

            clbf_descent_term_sim = clbf_descent_term_sim + violation.mean()
            clbf_descent_acc_sim = clbf_descent_acc_sim + (violation <= eps).sum() / (
                violation.nelement() * self.n_scenarios
            )
        loss.append(("CLBF descent term (simulated)", clbf_descent_term_sim))
    
        if accuracy:
            loss.append(("CLBF descent accuracy (simulated)", clbf_descent_acc_sim))

        return loss
    
    def initial_loss(self, x: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
        """
        Compute the loss during the initialization epochs, which trains the net to
        match the local linear lyapunov function
        """
        loss = []

        # The initial losses should decrease exponentially to zero, based on the epoch
        epoch_count = max(self.current_epoch - self.num_init_epochs, 0)
        decrease_factor = 0.5 ** epoch_count

        #   1.) Compare the CLBF to the nominal solution
        # Get the learned CLBF
        x = x.reshape(-1, self.dynamics_model.n_dims)
        V = self.V(x)

        # Get the nominal Lyapunov function
        P = self.dynamics_model.P.type_as(x)
        x0 = self.dynamics_model.goal_point.type_as(x)
        # Reshape to use pytorch's bilinear function
        P = P.reshape(1, self.dynamics_model.n_dims, self.dynamics_model.n_dims)
        V_nominal = 0.5 * F.bilinear(x - x0, x - x0, P).squeeze()

        if self.normalize_V_nominal:
            self.V_nominal_mean = V_nominal.mean()
            # V_nominal /= self.V_nominal_mean
            V_nominal = V_nominal * (1.0 / self.V_nominal_mean)


        # Compute the error between the two
        clbf_mse_loss = (V - V_nominal) ** 2
        clbf_mse_loss = decrease_factor * clbf_mse_loss.mean()
        loss.append(("CLBF MSE", clbf_mse_loss))

        # print("clbf_mse_loss", clbf_mse_loss)

        return loss

    def training_step(self, batch, batch_idx):
        """Conduct the training step for the given batch"""
        # Extract the input and masks from the batch
        x, goal_mask, safe_mask, unsafe_mask = batch

        # Compute the losses
        component_losses = {}
        initial_loss = self.initial_loss(x)
        # todo: add mask self.dynamics_model.safe_mask(x.resjape(-1, n))
        component_losses.update(initial_loss)
        component_losses.update(
            self.boundary_loss(x, goal_mask, safe_mask, unsafe_mask)
        )
        component_losses.update(
            self.descent_loss(x, requires_grad=True)
        )
        
        # print each individual loss
        for key, loss_value in component_losses.items():
            print(key, loss_value)

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(x)
        # For the objectives, we can just sum them
        for _, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss = total_loss + loss_value

        # print("total_loss", total_loss)
        
        batch_dict = {"loss": total_loss, **component_losses}
        self._train_step_outputs.append(batch_dict)
        return batch_dict

    def on_train_epoch_end(self):
        """This function is called after every epoch is completed."""
        outputs = self._train_step_outputs
        self._train_step_outputs = []
        if not outputs:
            return

        # Gather up all of the losses for each component from all batches
        losses = {}
        for batch_output in outputs:
            for key in batch_output.keys():
                # if we've seen this key before, add this component loss to the list
                if key in losses:
                    losses[key].append(batch_output[key])
                else:
                    # otherwise, make a new list
                    losses[key] = [batch_output[key]]

        # Average all the losses
        avg_losses = {}
        for key in losses.keys():
            key_losses = torch.stack(losses[key])
            avg_losses[key] = torch.nansum(key_losses) / key_losses.shape[0]

        # Log the overall loss...
        self.log("Total loss / train", avg_losses["loss"], sync_dist=True)
        # And all component losses
        for loss_key in avg_losses.keys():
            # We already logged overall loss, so skip that here
            if loss_key == "loss":
                continue
            # Log the other losses
            self.log(loss_key + " / train", avg_losses[loss_key], sync_dist=True)

    def validation_step(self, batch, batch_idx):
        """Conduct the validation step for the given batch"""
        # Extract the input and masks from the batch
        x, goal_mask, safe_mask, unsafe_mask = batch

        # Get the various losses
        component_losses = {}
        component_losses.update(
            self.boundary_loss(x, goal_mask, safe_mask, unsafe_mask)
        )
        component_losses.update(self.descent_loss(x))

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(x)
        # For the objectives, we can just sum them
        for _, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss += loss_value

        # Also compute the accuracy associated with each loss
        component_losses.update(
            self.boundary_loss(x, goal_mask, safe_mask, unsafe_mask, accuracy=True)
        )
        component_losses.update(
            self.descent_loss(x, accuracy=True)
        )

        batch_dict = {"val_loss": total_loss, **component_losses}
        self._val_step_outputs.append(batch_dict)
        return batch_dict

    def on_validation_epoch_end(self):
        """This function is called after every validation epoch is completed."""
        outputs = self._val_step_outputs
        self._val_step_outputs = []
        if outputs:
            # Gather up all of the losses for each component from all batches
            losses = {}
            for batch_output in outputs:
                for key in batch_output.keys():
                    if key in losses:
                        losses[key].append(batch_output[key])
                    else:
                        losses[key] = [batch_output[key]]

            # Average all the losses
            avg_losses = {}
            for key in losses.keys():
                key_losses = torch.stack(losses[key])
                avg_losses[key] = torch.nansum(key_losses) / key_losses.shape[0]

            # Log the overall loss...
            self.log("Total loss / val", avg_losses["val_loss"], sync_dist=True)
            # And all component losses
            for loss_key in avg_losses.keys():
                if loss_key == "val_loss":
                    continue
                self.log(loss_key + " / val", avg_losses[loss_key], sync_dist=True)

            # Only plot every 5 epochs
            if self.current_epoch % 5 == 0:
                self.experiment_suite.run_all_and_log_plots(
                    self, self.logger, self.current_epoch
                )

        # Generate new data at the end of every episode
        if self.current_epoch > 0 and self.current_epoch % self.epochs_per_episode == 0:
            if self.penalty_scheduling_rate > 0:
                relaxation_penalty = (
                    self.clf_relaxation_penalty
                    * self.current_epoch
                    / self.penalty_scheduling_rate
                )
            else:
                relaxation_penalty = self.clf_relaxation_penalty

            def simulator_fn_wrapper(x_init: torch.Tensor, num_steps: int):
                return self.simulator_fn(
                    x_init,
                    num_steps,
                    relaxation_penalty=relaxation_penalty,
                )

            self.datamodule.add_data(simulator_fn_wrapper)

    try:
        _auto_move = pl.core.decorators.auto_move_data
    except AttributeError:
        # PyTorch Lightning >= 2.0 moved/removed this decorator
        def _auto_move(fn):
            return fn

    @_auto_move
    def simulator_fn(
        self,
        x_init: torch.Tensor,
        num_steps: int,
        relaxation_penalty: Optional[float] = None,
    ):
        # Choose parameters randomly
        random_scenario = {}
        for param_name in self.scenarios[0].keys():
            param_max = max([s[param_name] for s in self.scenarios])
            param_min = min([s[param_name] for s in self.scenarios])
            random_scenario[param_name] = random.uniform(param_min, param_max)

        return self.dynamics_model.simulate(
            x_init,
            num_steps,
            self.u,
            guard=self.dynamics_model.out_of_bounds_mask,
            controller_period=self.controller_period,
            params=random_scenario,
        )

    def configure_optimizers(self):
        clbf_params = list(self.V_nn.parameters())

        clbf_opt = torch.optim.SGD(
            clbf_params,
            lr=self.primal_learning_rate,
            weight_decay=1e-6,
        )

        self.opt_idx_dict = {0: "clbf"}

        return [clbf_opt]
