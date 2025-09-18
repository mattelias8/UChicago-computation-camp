import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict

import argparse
import os
import yaml
from pydantic import BaseModel
import json

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
