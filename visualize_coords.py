#!/usr/bin/env python3
"""Visualize 2D hand trajectories from coords NPZ files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIR1 = "4-24-2026-xavier"
DIR2 = "4-21-2026-henry_horizontal"
DIR3 = "4-24-2026-william"


def plot_trajectory(ax, x: np.ndarray, y: np.ndarray, title: str):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        ax.set_title(title + "\n(no valid data)", fontsize=7)
        ax.axis("off")
        return

    t = np.linspace(0, 1, len(x))
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], color=plt.cm.coolwarm(t[i]), linewidth=1.5)

    ax.plot(x[0], y[0], "go", markersize=5, label="start")
    ax.plot(x[-1], y[-1], "rs", markersize=5, label="end")
    ax.set_title(title, fontsize=7)
    ax.set_xlabel("x (m)", fontsize=6)
    ax.set_ylabel("y (m)", fontsize=6)
    ax.tick_params(labelsize=6)
    ax.autoscale()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coords-root", type=Path, nargs="+",
                        default=[
                            Path("data") / DIR1 / "coords",
                            Path("data") / DIR2 / "coords",
                            Path("data") / DIR3 / "coords",
                        ])
    parser.add_argument("--output", type=Path,
                        default=Path("data/testing/trajectories.png"))
    parser.add_argument("--max-chunks", type=int, default=3,
                        help="Max chunks to show per NPZ file")
    args = parser.parse_args()

    npz_files = sorted(p for root in args.coords_root for p in root.rglob("*.npz"))
    if not npz_files:
        print("No coords NPZ files found. Run compute_2d_coords_bulk.py first.")
        return

    # Collect (title, x, y) for each chunk to plot
    panels = []
    for npz_path in npz_files:
        gesture = npz_path.parent.name
        subject = next(
            (r.parent.name for r in args.coords_root if npz_path.is_relative_to(r)),
            npz_path.parts[-4],
        )
        with np.load(npz_path) as data:
            x_all = data["x_coords"]
            y_all = data["y_coords"]
        for i in range(min(args.max_chunks, x_all.shape[0])):
            panels.append((f"{subject}\n{gesture}[{i}]", x_all[i], y_all[i]))

    ncols = 8
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    axes = np.array(axes).flatten()

    for ax, (title, x, y) in zip(axes, panels):
        plot_trajectory(ax, x, y, title)

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle("2D Hand Trajectories (blue=start → red=end)", fontsize=11)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Saved {len(panels)} trajectories → {args.output}")


if __name__ == "__main__":
    main()
