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


def build_feature_vector_aggregated(
    left_agg: np.ndarray,
    right_agg: np.ndarray,
    left_freq: np.ndarray,
    right_freq: np.ndarray,
    target_len: int,
    dec_rate: int = 3000,
) -> np.ndarray:
    """Feature vector from a single aggregated (mean or regression) L/R curve."""
    ref_s, ref_e = 2 * dec_rate, 3 * dec_rate
    left = left_agg.astype(np.float64) - np.mean(left_agg[ref_s:ref_e])
    right = right_agg.astype(np.float64) - np.mean(right_agg[ref_s:ref_e])

    left = resample_1d(left[:dec_rate], target_len)
    right = resample_1d(right[:dec_rate], target_len)

    joint_mean = np.mean(np.concatenate([left, right]))
    joint_std = np.std(np.concatenate([left, right])) + 1e-8
    left = (left - joint_mean) / joint_std
    right = (right - joint_mean) / joint_std

    left_vel = np.gradient(left)
    right_vel = np.gradient(right)
    left_acc = np.gradient(left_vel)
    right_acc = np.gradient(right_vel)

    speed = np.sqrt(left_vel ** 2 + right_vel ** 2) + 1e-8
    traj_dir_l = left_vel / speed
    traj_dir_r = right_vel / speed

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


def build_feature_vector_per_freq(
    left_freq: np.ndarray,
    right_freq: np.ndarray,
    target_len: int,
    dec_rate: int = 3000,
) -> np.ndarray:
    """Feature vector from all 16 per-frequency L/R curves concatenated."""
    ref_s, ref_e = 2 * dec_rate, 3 * dec_rate
    num_freqs = left_freq.shape[0]

    parts = []
    for f in range(num_freqs):
        lf = left_freq[f].astype(np.float64) - np.mean(left_freq[f, ref_s:ref_e])
        rf = right_freq[f].astype(np.float64) - np.mean(right_freq[f, ref_s:ref_e])

        lf = resample_1d(lf[:dec_rate], target_len)
        rf = resample_1d(rf[:dec_rate], target_len)

        joint_mean = np.mean(np.concatenate([lf, rf]))
        joint_std = np.std(np.concatenate([lf, rf])) + 1e-8
        lf = (lf - joint_mean) / joint_std
        rf = (rf - joint_mean) / joint_std

        parts.extend([lf, rf])

    return np.concatenate(parts).astype(np.float32)


SIGNAL_TYPES = {
    "mean": "Mean-aggregated distance",
    "regression": "Regression-fused distance",
    "per_freq": "Raw per-frequency distances (16 freqs)",
}


def load_dataset(iq_root: Path, target_len: int, signal_type: str):
    features, labels, sources, subjects = [], [], [], []

    for path in sorted(iq_root.rglob("*.npz")):
        with np.load(path) as data:
            gesture = str(data["gesture"])
            subject = str(data["subject"])
            dec_rate = int(float(data["new_sample_rate"]))
            left_freq = data["left_per_freq_distance"].astype(np.float32)
            right_freq = data["right_per_freq_distance"].astype(np.float32)

            if signal_type == "mean":
                left_agg = data["left_mean_distance"].astype(np.float32)
                right_agg = data["right_mean_distance"].astype(np.float32)
            elif signal_type == "regression":
                left_agg = data["left_regression_distance"].astype(np.float32)
                right_agg = data["right_regression_distance"].astype(np.float32)
            else:
                left_agg = right_agg = None

        num_chunks = left_freq.shape[0]
        for idx in range(num_chunks):
            if signal_type == "per_freq":
                feat = build_feature_vector_per_freq(
                    left_freq[idx], right_freq[idx], target_len, dec_rate=dec_rate
                )
            else:
                feat = build_feature_vector_aggregated(
                    left_agg[idx], right_agg[idx],
                    left_freq[idx], right_freq[idx],
                    target_len, dec_rate=dec_rate,
                )
            features.append(feat)
            labels.append(gesture)
            subjects.append(subject)
            sources.append(f"{path.stem}#{idx:02d}")

    if not features:
        raise ValueError(f"No .npz files found under {iq_root}")

    return np.stack(features), np.array(labels), np.array(subjects), sources


def reduce_features(X: np.ndarray, pca_dims: int, random_state: int):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float64)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    pca_dims = min(pca_dims, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=pca_dims, svd_solver="full", random_state=random_state)
    X_reduced = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_.cumsum()[-1]
    return X_reduced, pca_dims, var_explained


def run_tsne(X_reduced: np.ndarray, perplexity: float, random_state: int):
    perplexity = min(perplexity, max(5.0, (X_reduced.shape[0] - 1) / 3.0))
    emb = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=2000,
        random_state=random_state,
    ).fit_transform(X_reduced)
    return emb, perplexity


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

    for i, src in enumerate(sources):
        chunk_idx = src.split("#")[-1]
        ax.annotate(
            chunk_idx,
            (embedding[i, 0], embedding[i, 1]),
            fontsize=5, alpha=0.6,
            xytext=(3, 3), textcoords="offset points",
        )

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("t-SNE 1", fontsize=8)
    ax.set_ylabel("t-SNE 2", fontsize=8)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2, loc="best")


def run_single_type(iq_root, signal_type, target_len, pca_dims, perplexities, random_state):
    print(f"\n=== {SIGNAL_TYPES[signal_type]} ===")
    X, labels, subjects, sources = load_dataset(iq_root, target_len, signal_type)
    print(f"  samples={X.shape[0]}, feature_dim={X.shape[1]}")

    X_reduced, used_dims, var_exp = reduce_features(X, pca_dims, random_state)
    print(f"  PCA {used_dims}d explains {var_exp:.1%} variance")

    results = []
    for perp in perplexities:
        print(f"  t-SNE perplexity={perp}...")
        emb, used_perp = run_tsne(X_reduced, perp, random_state)
        results.append((emb, used_perp))

    return results, labels, subjects, sources, X.shape[1], used_dims, var_exp


def main():
    parser = argparse.ArgumentParser(description="t-SNE projection for gesture chunks.")
    parser.add_argument("--iq-root", type=Path,
                        default=Path("data") / "preliminary_data" / "IQ")
    parser.add_argument("--output", type=Path,
                        default=Path("data") / "preliminary_data" / "visualizations" / "05_tsne_projection.png")
    parser.add_argument("--signal-type", choices=list(SIGNAL_TYPES) + ["compare"], default="compare",
                        help="Signal aggregation type. 'compare' runs all three side by side.")
    parser.add_argument("--target-len", type=int, default=150)
    parser.add_argument("--pca-dims", type=int, default=30)
    parser.add_argument("--perplexity", type=float, default=None,
                        help="Single perplexity (default: two values)")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Determine perplexity values to show
    if args.perplexity is not None:
        perplexities = [args.perplexity]
    else:
        # Peek at sample count to set sensible defaults
        n_samples = sum(
            np.load(p)["left_per_freq_distance"].shape[0]
            for p in sorted(args.iq_root.rglob("*.npz"))
        )
        perplexities = sorted({5.0, round(max(10.0, n_samples / 10))})

    if args.signal_type != "compare":
        # Single signal type, grid of perplexity values
        results, labels, subjects, sources, feat_dim, pca_d, var_exp = run_single_type(
            args.iq_root, args.signal_type, args.target_len,
            args.pca_dims, perplexities, args.random_state,
        )
        ncols = min(2, len(perplexities))
        nrows = (len(perplexities) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
        axes = np.array(axes).reshape(-1)

        for ax, (emb, used_perp) in zip(axes, results):
            plot_embedding(ax, emb, labels, subjects, sources,
                           f"{SIGNAL_TYPES[args.signal_type]}\nt-SNE perp={used_perp:.0f}  n={len(labels)}")
        for ax in axes[len(results):]:
            ax.set_visible(False)

        fig.suptitle(
            f"{SIGNAL_TYPES[args.signal_type]}  |  feat={feat_dim}d → PCA {pca_d}d ({var_exp:.0%} var)",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(args.output, dpi=180)
        plt.close(fig)
        print(f"\nSaved → {args.output}")

    else:
        # Compare all 3 signal types: rows = perplexity, cols = signal type
        signal_types = list(SIGNAL_TYPES.keys())
        nrows = len(perplexities)
        ncols = len(signal_types)

        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
        axes = np.array(axes).reshape(nrows, ncols)

        type_meta = {}
        for col, stype in enumerate(signal_types):
            results, labels, subjects, sources, feat_dim, pca_d, var_exp = run_single_type(
                args.iq_root, stype, args.target_len,
                args.pca_dims, perplexities, args.random_state,
            )
            type_meta[stype] = (feat_dim, pca_d, var_exp)
            for row, (emb, used_perp) in enumerate(results):
                ax = axes[row, col]
                plot_embedding(ax, emb, labels, subjects, sources,
                               f"{SIGNAL_TYPES[stype]}\nperp={used_perp:.0f}  feat={feat_dim}d→PCA{pca_d}d ({var_exp:.0%})")

        fig.suptitle(
            f"Gesture t-SNE — signal aggregation comparison  |  n={len(labels)}",
            fontsize=12,
        )
        fig.tight_layout()
        out = args.output.with_name(args.output.stem + "_compare.png")
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
