"""Visualize the weight matrix evolution of a 2-region brain.

Runs the TwoRegionBrain on alternating A/B patterns and saves three PNGs:
1. weights_before.png — initial weight matrix
2. weights_after.png — weight matrix after training
3. weights_evolution.png — line traces of each synapse weight over epochs
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from brain.two_region import TwoRegionBrain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("logs"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    torch.manual_seed(args.seed)
    brain = TwoRegionBrain(
        num_sensory=4,
        num_feature=2,
        a_plus=0.05,
        a_minus=0.05,
        w_init_jitter=0.1,
    )

    pattern_a = torch.tensor([2.0, 2.0, 0.0, 0.0])
    pattern_b = torch.tensor([0.0, 0.0, 2.0, 2.0])

    weights_before = brain.synapse.weights.clone().numpy()
    history: list[np.ndarray] = []

    for _ in range(args.epochs):
        for _ in range(5):
            brain.tick(pattern_a)
        for _ in range(5):
            brain.tick(torch.zeros(4))
        for _ in range(5):
            brain.tick(pattern_b)
        for _ in range(5):
            brain.tick(torch.zeros(4))
        history.append(brain.synapse.weights.clone().numpy())

    weights_after = brain.synapse.weights.clone().numpy()

    # Plot before
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(weights_before, vmin=0, vmax=1, cmap="viridis")
    ax.set_title("Weights — before training")
    ax.set_xlabel("sensory neuron")
    ax.set_ylabel("feature neuron")
    plt.colorbar(im)
    fig.tight_layout()
    fig.savefig(args.out / "weights_before.png", dpi=120)
    plt.close(fig)

    # Plot after
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(weights_after, vmin=0, vmax=1, cmap="viridis")
    ax.set_title("Weights — after training")
    ax.set_xlabel("sensory neuron")
    ax.set_ylabel("feature neuron")
    plt.colorbar(im)
    fig.tight_layout()
    fig.savefig(args.out / "weights_after.png", dpi=120)
    plt.close(fig)

    # Plot evolution: each feature neuron's weights over time
    history_arr = np.stack(history)  # (epochs, feature, sensory)
    fig, axes = plt.subplots(brain.feature.num_neurons, 1, figsize=(6, 4), sharex=True)
    if brain.feature.num_neurons == 1:
        axes = [axes]
    for f, ax in enumerate(axes):
        for s in range(brain.sensory.num_neurons):
            ax.plot(history_arr[:, f, s], label=f"sensory {s}")
        ax.set_ylabel(f"feature {f} weights")
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("epoch")
    fig.suptitle("Weight evolution during STDP training")
    fig.tight_layout()
    fig.savefig(args.out / "weights_evolution.png", dpi=120)
    plt.close(fig)

    print(f"Wrote 3 plots to {args.out}/")
    print(f"Final weights:\n{weights_after}")


if __name__ == "__main__":
    main()
