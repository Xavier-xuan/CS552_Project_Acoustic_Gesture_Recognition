#!/usr/bin/env python3
"""Quick diagnostic plots for gesture path-change data."""

from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_gesture_data(iq_root: Path):
    grouped: dict[str, dict[str, list[np.ndarray] | np.ndarray | list[str]]] = {}

    for path in sorted(iq_root.rglob("*.npz")):
        with np.load(path) as data:
            gesture = str(data["gesture"])
            subject = str(data["subject"])
            time_axis = data["time_axis"].astype(np.float32)
            carrier_freqs = data["carrier_freqs"].astype(np.float32)
            left_reg = data["left_regression_distance"].astype(np.float32)
            right_reg = data["right_regression_distance"].astype(np.float32)
            left_freq = data["left_per_freq_distance"].astype(np.float32)
            right_freq = data["right_per_freq_distance"].astype(np.float32)

        entry = grouped.setdefault(
            gesture,
            {
                "time_axis": time_axis,
                "carrier_freqs": carrier_freqs,
                "subjects": [],
                "left_reg": [],
                "right_reg": [],
                "left_freq": [],
                "right_freq": [],
            },
        )
        entry["subjects"].append(subject)
        entry["left_reg"].append(left_reg)
        entry["right_reg"].append(right_reg)
        entry["left_freq"].append(left_freq)
        entry["right_freq"].append(right_freq)

    for gesture, entry in grouped.items():
        entry["left_reg"] = np.concatenate(entry["left_reg"], axis=0)
        entry["right_reg"] = np.concatenate(entry["right_reg"], axis=0)
        entry["left_freq"] = np.concatenate(entry["left_freq"], axis=0)
        entry["right_freq"] = np.concatenate(entry["right_freq"], axis=0)

    return grouped


def pick_subset(array: np.ndarray, max_samples: int) -> np.ndarray:
    if array.shape[0] <= max_samples:
        return array
    indices = np.linspace(0, array.shape[0] - 1, max_samples, dtype=int)
    return array[indices]


def plot_overlay(grouped, output_dir: Path, max_samples: int):
    gestures = sorted(grouped)
    fig, axes = plt.subplots(len(gestures), 2, figsize=(14, 3.5 * len(gestures)), squeeze=False)

    for row, gesture in enumerate(gestures):
        entry = grouped[gesture]
        t = entry["time_axis"]
        left = pick_subset(entry["left_reg"], max_samples)
        right = pick_subset(entry["right_reg"], max_samples)

        for curve in left:
            axes[row, 0].plot(t, curve * 1000, alpha=0.45, linewidth=0.9)
        axes[row, 0].set_title(f"{gesture} | Left Regression Path Change")
        axes[row, 0].set_xlabel("Time (s)")
        axes[row, 0].set_ylabel("Path Change (mm)")
        axes[row, 0].grid(alpha=0.3)

        for curve in right:
            axes[row, 1].plot(t, curve * 1000, alpha=0.45, linewidth=0.9)
        axes[row, 1].set_title(f"{gesture} | Right Regression Path Change")
        axes[row, 1].set_xlabel("Time (s)")
        axes[row, 1].set_ylabel("Path Change (mm)")
        axes[row, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "01_overlay_regression_paths.png", dpi=180)
    plt.close(fig)


def plot_mean_std(grouped, output_dir: Path):
    gestures = sorted(grouped)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), squeeze=False)
    ax_left = axes[0, 0]
    ax_right = axes[1, 0]

    for gesture in gestures:
        entry = grouped[gesture]
        t = entry["time_axis"]
        left = entry["left_reg"] * 1000
        right = entry["right_reg"] * 1000

        left_mean = left.mean(axis=0)
        left_std = left.std(axis=0)
        right_mean = right.mean(axis=0)
        right_std = right.std(axis=0)

        ax_left.plot(t, left_mean, linewidth=1.8, label=gesture)
        ax_left.fill_between(t, left_mean - left_std, left_mean + left_std, alpha=0.15)

        ax_right.plot(t, right_mean, linewidth=1.8, label=gesture)
        ax_right.fill_between(t, right_mean - right_std, right_mean + right_std, alpha=0.15)

    ax_left.set_title("Mean ± Std Regression Path Change | Left Channel")
    ax_left.set_xlabel("Time (s)")
    ax_left.set_ylabel("Path Change (mm)")
    ax_left.grid(alpha=0.3)
    ax_left.legend(ncol=3)

    ax_right.set_title("Mean ± Std Regression Path Change | Right Channel")
    ax_right.set_xlabel("Time (s)")
    ax_right.set_ylabel("Path Change (mm)")
    ax_right.grid(alpha=0.3)
    ax_right.legend(ncol=3)

    fig.tight_layout()
    fig.savefig(output_dir / "02_mean_std_regression_paths.png", dpi=180)
    plt.close(fig)


def plot_2d_trajectories(grouped, output_dir: Path, max_samples: int):
    gestures = sorted(grouped)
    cols = 3
    rows = ceil(len(gestures) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.2 * rows), squeeze=False)
    axes = axes.flatten()
    global_xmin = np.inf
    global_xmax = -np.inf
    global_ymin = np.inf
    global_ymax = -np.inf

    for gesture in gestures:
        entry = grouped[gesture]
        left = pick_subset(entry["left_reg"], max_samples) * 1000
        right = pick_subset(entry["right_reg"], max_samples) * 1000
        global_xmin = min(global_xmin, float(np.min(left)))
        global_xmax = max(global_xmax, float(np.max(left)))
        global_ymin = min(global_ymin, float(np.min(right)))
        global_ymax = max(global_ymax, float(np.max(right)))

    x_margin = 0.05 * max(global_xmax - global_xmin, 1.0)
    y_margin = 0.05 * max(global_ymax - global_ymin, 1.0)

    for ax, gesture in zip(axes, gestures):
        entry = grouped[gesture]
        left = pick_subset(entry["left_reg"], max_samples) * 1000
        right = pick_subset(entry["right_reg"], max_samples) * 1000

        for left_curve, right_curve in zip(left, right):
            ax.plot(left_curve, right_curve, alpha=0.5, linewidth=0.9)

        ax.set_title(f"{gesture} | 2D Trajectory")
        ax.set_xlabel("Left Path Change (mm)")
        ax.set_ylabel("Right Path Change (mm)")
        ax.grid(alpha=0.3)
        ax.set_xlim(global_xmin - x_margin, global_xmax + x_margin)
        ax.set_ylim(global_ymin - y_margin, global_ymax + y_margin)
        ax.set_aspect("equal", adjustable="box")

    for ax in axes[len(gestures):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "03_2d_trajectories.png", dpi=180)
    plt.close(fig)


def plot_heatmaps(grouped, output_dir: Path):
    gestures = sorted(grouped)
    cols = 2
    rows = len(gestures)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.6 * rows), squeeze=False)

    for row, gesture in enumerate(gestures):
        entry = grouped[gesture]
        t = entry["time_axis"]
        freqs = entry["carrier_freqs"]
        left_heat = entry["left_freq"].mean(axis=0) * 1000
        right_heat = entry["right_freq"].mean(axis=0) * 1000

        im_left = axes[row, 0].imshow(
            left_heat,
            aspect="auto",
            origin="lower",
            extent=[t[0], t[-1], freqs[0], freqs[-1]],
            cmap="coolwarm",
        )
        axes[row, 0].set_title(f"{gesture} | Left 16-Frequency Mean Heatmap")
        axes[row, 0].set_xlabel("Time (s)")
        axes[row, 0].set_ylabel("Carrier Frequency (Hz)")
        fig.colorbar(im_left, ax=axes[row, 0], label="Path Change (mm)")

        im_right = axes[row, 1].imshow(
            right_heat,
            aspect="auto",
            origin="lower",
            extent=[t[0], t[-1], freqs[0], freqs[-1]],
            cmap="coolwarm",
        )
        axes[row, 1].set_title(f"{gesture} | Right 16-Frequency Mean Heatmap")
        axes[row, 1].set_xlabel("Time (s)")
        axes[row, 1].set_ylabel("Carrier Frequency (Hz)")
        fig.colorbar(im_right, ax=axes[row, 1], label="Path Change (mm)")

    fig.tight_layout()
    fig.savefig(output_dir / "04_frequency_heatmaps.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate quick diagnostic gesture plots.")
    parser.add_argument("--iq-root", type=Path, default=Path("data") / "preliminary_data" / "IQ",
                        help="Root directory containing IQ/path-change NPZ files")
    parser.add_argument("--output-dir", type=Path, default=Path("data") / "preliminary_data" / "visualizations",
                        help="Directory to store generated figures")
    parser.add_argument("--max-samples", type=int, default=8,
                        help="Maximum number of chunk trajectories to overlay per gesture")
    args = parser.parse_args()

    grouped = load_gesture_data(args.iq_root)
    if not grouped:
        print(f"No .npz files found under {args.iq_root}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_overlay(grouped, args.output_dir, args.max_samples)
    plot_mean_std(grouped, args.output_dir)
    plot_2d_trajectories(grouped, args.output_dir, args.max_samples)
    plot_heatmaps(grouped, args.output_dir)

    print(f"Saved quick diagnostic plots to: {args.output_dir}")
    for figure_name in [
        "01_overlay_regression_paths.png",
        "02_mean_std_regression_paths.png",
        "03_2d_trajectories.png",
        "04_frequency_heatmaps.png",
    ]:
        print(f"  - {args.output_dir / figure_name}")


if __name__ == "__main__":
    main()
