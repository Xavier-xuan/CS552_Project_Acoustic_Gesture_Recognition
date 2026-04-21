#!/usr/bin/env python3
"""t-SNE projection for gesture chunks based on IQ/path-change features."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def resample_1d(curve: np.ndarray, target_len: int) -> np.ndarray:
    if curve.shape[0] == target_len:
        return curve.astype(np.float64)
    x_old = np.linspace(0.0, 1.0, curve.shape[0])
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, curve)


def build_feature_vector(
    left_reg: np.ndarray,
    right_reg: np.ndarray,
    left_freq: np.ndarray,
    right_freq: np.ndarray,
    target_len: int,
    dec_rate: int = 3000,
) -> np.ndarray:
    # Static removal: subtract 2–3 s window mean as DC reference
    ref_s, ref_e = 2 * dec_rate, 3 * dec_rate
    left = left_reg.astype(np.float64) - np.mean(left_reg[ref_s:ref_e])
    right = right_reg.astype(np.float64) - np.mean(right_reg[ref_s:ref_e])

    # Use only the first 1 second (gesture window)
    left = resample_1d(left[:dec_rate], target_len)
    right = resample_1d(right[:dec_rate], target_len)

    # Joint shape normalization (preserves L/R relative magnitude)
    joint_mean = np.mean(np.concatenate([left, right]))
    joint_std = np.std(np.concatenate([left, right])) + 1e-8
    left = (left - joint_mean) / joint_std
    right = (right - joint_mean) / joint_std

    # Velocity and acceleration
    left_vel = np.gradient(left)
    right_vel = np.gradient(right)
    left_acc = np.gradient(left_vel)
    right_acc = np.gradient(right_vel)

    # Normalised 2D trajectory direction (unit tangent)
    speed = np.sqrt(left_vel ** 2 + right_vel ** 2) + 1e-8
    traj_dir_l = left_vel / speed
    traj_dir_r = right_vel / speed

    # Per-frequency spread in the first 1 s
    left_spread = resample_1d(left_freq.std(axis=0)[:dec_rate], target_len)
    right_spread = resample_1d(right_freq.std(axis=0)[:dec_rate], target_len)
    left_spread = (left_spread - left_spread.mean()) / (left_spread.std() + 1e-8)
    right_spread = (right_spread - right_spread.mean()) / (right_spread.std() + 1e-8)

    return np.concatenate([
        left, right,
        left_vel, right_vel,
        left_acc, right_acc,
        traj_dir_l, traj_dir_r,
        left_spread, right_spread,
    ]).astype(np.float32)


def load_dataset(iq_root: Path, target_len: int):
    features, labels, sources, subjects = [], [], [], []

    for path in sorted(iq_root.rglob("*.npz")):
        with np.load(path) as data:
            gesture = str(data["gesture"])
            subject = str(data["subject"])
            dec_rate = int(float(data["new_sample_rate"]))
            left_reg = data["left_mean_distance"].astype(np.float32)
            right_reg = data["right_mean_distance"].astype(np.float32)
            left_freq = data["left_per_freq_distance"].astype(np.float32)
            right_freq = data["right_per_freq_distance"].astype(np.float32)

        for idx in range(left_reg.shape[0]):
            features.append(
                build_feature_vector(
                    left_reg[idx], right_reg[idx],
                    left_freq[idx], right_freq[idx],
                    target_len,
                    dec_rate=dec_rate,
                )
            )
            labels.append(gesture)
            subjects.append(subject)
            sources.append(f"{path.stem}#{idx:02d}")

    if not features:
        raise ValueError(f"No .npz files found under {iq_root}")

    return np.stack(features), np.array(labels), np.array(subjects), sources


def run_tsne(X_reduced: np.ndarray, perplexity: float, random_state: int) -> np.ndarray:
    perplexity = min(perplexity, max(5.0, (X_reduced.shape[0] - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=2000,
        random_state=random_state,
    ).fit_transform(X_reduced), perplexity


def plot_embedding(ax, embedding, labels, subjects, sources, title, cmap_name="tab10"):
    unique_labels = sorted(set(labels.tolist()))
    unique_subjects = sorted(set(subjects.tolist()))
    cmap = plt.get_cmap(cmap_name)
    color_map = {lbl: cmap(i / max(len(unique_labels) - 1, 1))
                 for i, lbl in enumerate(unique_labels)}
    marker_map = {s: m for s, m in zip(unique_subjects, ["o", "s", "^", "D", "v", "P"])}

    for lbl in unique_labels:
        for subj in unique_subjects:
            mask = (labels == lbl) & (subjects == subj)
            if not mask.any():
                continue
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                s=60, alpha=0.85,
                color=color_map[lbl],
                marker=marker_map.get(subj, "o"),
                edgecolors="white", linewidths=0.4,
                label=f"{lbl} ({subj})" if len(unique_subjects) > 1 else lbl,
                zorder=3,
            )

    # Annotate chunk index on each point
    for i, src in enumerate(sources):
        chunk_idx = src.split("#")[-1]
        ax.annotate(
            chunk_idx,
            (embedding[i, 0], embedding[i, 1]),
            fontsize=5, alpha=0.6,
            xytext=(3, 3), textcoords="offset points",
        )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("t-SNE 1", fontsize=8)
    ax.set_ylabel("t-SNE 2", fontsize=8)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2, loc="best")


def main():
    parser = argparse.ArgumentParser(description="t-SNE projection for gesture chunks.")
    parser.add_argument("--iq-root", type=Path,
                        default=Path("data") / "preliminary_data" / "IQ")
    parser.add_argument("--output", type=Path,
                        default=Path("data") / "preliminary_data" / "visualizations" / "05_tsne_projection.png")
    parser.add_argument("--target-len", type=int, default=150,
                        help="Resample length for regression curves")
    parser.add_argument("--pca-dims", type=int, default=30,
                        help="PCA dims before t-SNE")
    parser.add_argument("--perplexity", type=float, default=None,
                        help="Single perplexity (default: show 4 values)")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    print("Loading dataset...")
    X, labels, subjects, sources = load_dataset(args.iq_root, args.target_len)
    print(f"  samples={X.shape[0]}, feature_dim={X.shape[1]}")
    print(f"  gestures: {sorted(set(labels.tolist()))}")
    print(f"  subjects: {sorted(set(subjects.tolist()))}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float64)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    pca_dims = min(args.pca_dims, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=pca_dims, svd_solver="full", random_state=args.random_state)
    X_reduced = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"  PCA {pca_dims}d explains {var_explained:.1%} variance")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.perplexity is not None:
        # Single perplexity — one plot
        emb, used_perp = run_tsne(X_reduced, args.perplexity, args.random_state)
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_embedding(ax, emb, labels, subjects, sources,
                       f"t-SNE  perplexity={used_perp:.0f}  n={X.shape[0]}")
        fig.tight_layout()
        fig.savefig(args.output, dpi=180)
        plt.close(fig)
        print(f"Saved → {args.output}")
    else:
        # 2×2 grid with four perplexity values for easy comparison
        n = X_reduced.shape[0]
        candidate_perps = [5.0, max(10.0, n / 10), max(20.0, n / 5), max(30.0, n / 3)]
        candidate_perps = sorted(set(round(p) for p in candidate_perps))[:4]

        ncols = 2
        nrows = (len(candidate_perps) + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6 * nrows))
        axes = np.array(axes).reshape(-1)

        for ax, perp in zip(axes, candidate_perps):
            print(f"  Running t-SNE perplexity={perp}...")
            emb, used_perp = run_tsne(X_reduced, perp, args.random_state)
            plot_embedding(ax, emb, labels, subjects, sources,
                           f"t-SNE  perplexity={used_perp:.0f}  n={n}")

        for ax in axes[len(candidate_perps):]:
            ax.set_visible(False)

        fig.suptitle(
            f"Gesture t-SNE  |  features={X.shape[1]}d → PCA {pca_dims}d ({var_explained:.0%} var)",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(args.output, dpi=180)
        plt.close(fig)
        print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
