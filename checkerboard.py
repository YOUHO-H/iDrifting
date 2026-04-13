import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time
from models.drifting import compute_drift, compute_drift_hybrid
from models.model import MLP
import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

def drifting_loss(gen: torch.Tensor, pos: torch.Tensor, compute_drift):
    """Drifting loss: MSE(gen, stopgrad(gen + V))."""
    with torch.no_grad():
        V = compute_drift(gen, pos)
        target = (gen + V).detach()
    return F.mse_loss(gen, target)


# ============================================================
# Toy Dataset Samplers (minimal; copied from toy_mean_drift.py)
# ============================================================

def sample_checkerboard(n: int, noise: float = 0.05, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    b = torch.randint(0, 2, (n,), generator=g)
    i = torch.randint(0, 2, (n,), generator=g) * 2 + b
    j = torch.randint(0, 2, (n,), generator=g) * 2 + b
    u = torch.rand(n, generator=g)
    v = torch.rand(n, generator=g)
    pts = torch.stack([i + u, j + v], dim=1) - 2.0
    pts = pts / 2.0
    if noise > 0:
        pts = pts + noise * torch.randn(pts.shape, generator=g)
    return pts

# ============================================================
# Training Loop for Toy 2D
# ============================================================
from functools import partial
def train_toy(sampler, steps=2000, data_batch_size=4096, 
              gen_batch_size=4096, lr=5e-4, temp=0.05,
              in_dim=32, hidden=256, plot_every=2000, kernel='Laplace', 
              seed=42):
    """Train drifting model. Returns model and loss history."""
    torch.manual_seed(seed)
    model = MLP(in_dim=in_dim, hidden=hidden, out_dim=2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    ema = None
    pbar = tqdm(range(1, steps + 1))
    strt_time = time.time()
    for step in pbar:
        pos = sampler(data_batch_size).to(DEVICE)
        gen = model(torch.randn(gen_batch_size, in_dim, device=DEVICE))

        if kernel == 'Laplace':
            loss = drifting_loss(gen, 
                                 pos, 
                                 compute_drift=partial(compute_drift, temp=temp))
        elif kernel == 'Hybrid':
            loss = drifting_loss(gen, 
                                 pos, 
                                 compute_drift=partial(compute_drift_hybrid, m=128, 
                                                       sigma=0.1, laplace_scale=0.05, wc=1.0))
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        ema = loss.item() if ema is None else 0.96 * ema + 0.04 * loss.item()
        pbar.set_postfix(loss=f"{ema:.2e}")

        if step % plot_every == 0 or step == 1:
            with torch.no_grad():
                vis = model(torch.randn(5000, in_dim, device=DEVICE)).cpu().numpy()
                gt = sampler(5000).numpy()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
            ax1.scatter(gt[:, 0], gt[:, 1], s=3, alpha=0.5, c='black')
            ax1.set_title('Target'); ax1.set_aspect('equal'); ax1.axis('off')
            ax2.scatter(vis[:, 0], vis[:, 1], s=3, alpha=0.5, c='tab:orange')
            ax2.set_title(f'Generated (step {step})'); ax2.set_aspect('equal'); ax2.axis('off')
            plt.tight_layout()
            # save the plot
            if os.path.exists('outputs_checkerboard') == False:
                os.makedirs('outputs_checkerboard')
            plt.savefig(f'outputs_checkerboard/checkerboard_step_{step}.png', dpi=300)
            plt.close(fig)
    end_time = time.time()
    print(f"Training completed in {(end_time - strt_time)/60:.2f} minutes.")
    return model, loss_history

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temp', type=float, default=0.05)
    parser.add_argument('--in_dim', type=int, default=32)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--plot_every', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kernel', type=str, default='Laplace', choices=['Laplace', 'Hybrid'])
    args = parser.parse_args()

    # Train on Checkerboard
    print("\nTraining on Checkerboard...")
    model_checker, loss_checker = train_toy(sample_checkerboard, 
                                            steps=args.steps, 
                                            lr=args.lr, 
                                            temp=args.temp,
                                            kernel=args.kernel,
                                            )

    plt.figure(figsize=(6, 3))
    plt.plot(loss_checker, alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Checkerboard Loss Curve')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # save the plot
    plt.savefig('outputs_checkerboard/checkerboard_loss_curve.png', dpi=300)
