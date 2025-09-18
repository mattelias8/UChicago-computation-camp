import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict

import argparse
import os
import yaml
from pydantic import BaseModel
import json

class SimConfig(BaseModel):
    var_e: float = 0.5      # Variance of transitory component (e_it)
    var_u: float = 0.3      # Variance of permanent shock (u_it)
    var_p1: float = 0.8     # Variance of initial permanent component (p_i1)
    theta: float = 0.12     # Theta parameter
    rho: float = 1.0        # Persistence parameter (rho), usually 1.0 for unit root
    nt: int = 10            # Number of time periods (T)
    ni: int = 1000          # Number of individuals (N)
    rep: int = 3            # Number of simulation repetitions
    seed: int = 42          # Random seed for reproducibility


class PermanentTransitorySimulator:
    """
    Simulates data from permanent transitory earnings model:
    y_it = alpha_i + p_it + e_it
    p_it = p_it-1 + u_it
    """

    def __init__(self, var_e: float, var_u: float, var_p1: float, theta: float, rho: float = 1.0,):
        """
        Parameters:
        - var_e: variance of transitory component e_it
        - var_u: variance of permanent shock u_it
        - var_p1: variance of initial permanent component p_i1
        - rho: persistence parameter (fixed at 1.0 for unit root)
        """
        self.var_e = var_e
        self.var_u = var_u
        self.var_p1 = var_p1
        self.theta = theta
        self.rho = rho

    def simulate(self, N: int, T: int, seed: int = None) -> torch.Tensor:
        """
        Simulate earnings data for N individuals over T periods

        Returns:
        - y: tensor of shape (N, T) with simulated earnings
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Fixed effects alpha_i (assumed zero for simplicity)
        alpha = torch.zeros(N, 1)

        # Initial permanent component p_i1
        p_initial = torch.sqrt(torch.tensor(self.var_p1)) * torch.randn(N, 1)

        # Permanent shocks u_it
        u = torch.sqrt(torch.tensor(self.var_u)) * torch.randn(N, T)

        # Transitory shocks e_it
        #e = torch.sqrt(torch.tensor(self.var_e)) * torch.randn(N, T)
        e_initial = torch.sqrt(torch.tensor(self.var_e)) * torch.randn(N, T)

        e = torch.zeros(N, T)
        for t in range(1, T):
            e[:, t] = self.theta * e_initial[:, t-1] + e_initial[:,t]

        # Generate permanent component p_it = p_it-1 + u_it
        p = torch.zeros(N, T)
        p[:, 0] = p_initial.squeeze() + u[:, 0]
        for t in range(1, T):
            p[:, t] = self.rho * p[:, t-1] + u[:, t]

        # Generate earnings y_it = alpha_i + p_it + e_it
        y = alpha + p + e

        return y


class PermanentTransitoryEstimator(nn.Module):
    """
    PyTorch model for estimating permanent transitory model parameters
    using moment matching on variance and autocovariances of growth rates
    """

    def __init__(self):
        super().__init__()
        # Parameters to estimate (log scale for positivity)
        self.log_var_e = nn.Parameter(torch.tensor(0.0))
        self.log_var_u = nn.Parameter(torch.tensor(0.0))
        self.log_var_p1 = nn.Parameter(torch.tensor(0.0))
        self.theta = nn.Parameter(torch.tensor(0.0))

    @property
    def var_e(self):
        return torch.exp(self.log_var_e)

    @property
    def var_u(self):
        return torch.exp(self.log_var_u)

    @property
    def var_p1(self):
        return torch.exp(self.log_var_p1)
    
    def theoretical_moments(self, T: int) -> Dict[str, torch.Tensor]:
        """
        Compute theoretical variance and autocovariances of growth rates
        For growth Δy_it = y_it - y_it-1 = u_it + e_it - e_it-1
        """
        moments = {}

        # Variance of growth: Var(Δy_it) = var_u + 2*(theta^2 - theta + 1) var_e
        moments['var_growth'] = self.var_u + 2 * (self.theta**2 - self.theta + 1) * self.var_e

        # First autocovariance: Cov(Δy_it, Δy_it-1) = -1 * (theta)^2 -var_e
        moments['cov1_growth'] = -(1-self.theta)**2 * self.var_e

        # Second autocovariance: Cov(Δy_it, Δy_it-2)
        moments['cov2_growth'] = -self.theta * self.var_e

        # Higher order autocovariances are zero
        for lag in range(3, T):
            moments[f'cov{lag}_growth'] = torch.tensor(0.0)

        return moments

    def sample_moments(self, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute sample variance and autocovariances of growth rates
        """
        N, T = y.shape

        # Compute growth rates Δy_it = y_it - y_it-1
        growth = y[:, 1:] - y[:, :-1]  # Shape: (N, T-1)

        moments = {}

        # Sample variance of growth
        moments['var_growth'] = torch.var(growth, unbiased=True)

        # Sample autocovariances
        for lag in range(1, min(T-1, 10)):  # Limit lags for efficiency
            if growth.shape[1] > lag:
                cov = torch.mean((growth[:, lag:] - torch.mean(growth[:, lag:])) *
                               (growth[:, :-lag] - torch.mean(growth[:, :-lag])))
                moments[f'cov{lag}_growth'] = cov
            else:
                moments[f'cov{lag}_growth'] = torch.tensor(0.0)

        return moments

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute loss as sum of squared differences between theoretical and sample moments
        """
        N, T = y.shape

        # Get theoretical and sample moments
        theoretical = self.theoretical_moments(T)
        sample = self.sample_moments(y)

        # Compute squared differences
        loss = torch.tensor(0.0)
        for key in theoretical:
            if key in sample:
                diff = theoretical[key] - sample[key]
                loss = loss + diff ** 2

        return loss


def estimate_model(y_data: torch.Tensor, lr: float = 0.01, max_iter: int = 1000) -> Tuple[Dict[str, float], list]:
    """
    Estimate the permanent transitory model using the provided data

    Returns:
    - estimates: dictionary with estimated parameters
    - loss_history: list of loss values during optimization
    """
    model = PermanentTransitoryEstimator()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    for i in range(max_iter):
        optimizer.zero_grad()
        loss = model(y_data)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.item():.6f}")

    estimates = {
        'var_e': model.var_e.item(),
        'var_u': model.var_u.item(),
        'var_p1': model.var_p1.item(),
        'theta': model.theta.item()
    }

    return estimates, loss_history


if __name__ == "__main__":
    # Example usage

    parser = argparse.ArgumentParser(description="Permanent-Transitory Model Simulation and Estimation")
    parser.add_argument('-o', '--output', type=str, required=True, help='Output file to save results')
    parser.add_argument('-c', '--config', type=str, required=False, help='YAML config file for simulation parameters')
    parser.add_argument('-i', '--index', type=int, required=False, help='Index for random seed variation', default=0)
    args = parser.parse_args()

    # Add pydantic and yaml imports
    config = SimConfig()
    if args.config:
        with open(args.config, 'r') as f:
            yaml_data = yaml.safe_load(f)
        config = SimConfig(**yaml_data)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    simulator = PermanentTransitorySimulator(
        rho=1.0,
        var_e = config.var_e,
        var_u = config.var_u,
        var_p1 = config.var_p1,
        theta = config.theta
        ) 

    results = []
    reps = config.rep

    for i in range(reps):
        y_sim = simulator.simulate(
            N=config.ni, 
            T=config.nt, 
            seed= config.seed + i + args.index*config.rep)
        estimates, loss_hist = estimate_model(y_sim, lr=0.02, max_iter=2000)

        print("\nEstimated parameters:", estimates)
        results.append({
            'rep': i + args.index*config.rep,
            'estimates': estimates,
            'final_loss': loss_hist[-1]
        })

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

