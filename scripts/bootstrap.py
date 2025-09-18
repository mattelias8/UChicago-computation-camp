import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
import camp.panel_estimator as pe

import argparse
import os
import yaml
from pydantic import BaseModel
import json

import numpy as np

class SimConfig(BaseModel):
    n_boot: int = 100        # Number of bootstrap draws
    seed: int = 42          # Random seed for reproducibility



## Import data to bootstrap from

bpp_matrix = np.load("data/bpp_y_matrix.npy")

n, t = np.shape(bpp_matrix)

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

    results = []
    reps = config.n_boot

    for i in range(reps):
        ## Bootstrap draw

        np.random.seed(config.seed + i + args.index*config.n_boot)
        indices = np.random.choice(range(0,n), size = n, replace = True)

        # sample the rows
        sampled_rows = torch.tensor(bpp_matrix[indices])

        #construct parameter estimate
        estimates, loss_hist = pe.estimate_model(sampled_rows, lr=0.02, max_iter=2000)

        print("\nEstimated parameters:", estimates)
        results.append({
            'rep': i + args.index*reps,
            'estimates': estimates,
            'final_loss': loss_hist[-1]
        })

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
