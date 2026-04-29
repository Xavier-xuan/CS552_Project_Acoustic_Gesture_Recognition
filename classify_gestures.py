#!/usr/bin/env python3
"""XGBoost gesture classification with cross-validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# Reuse feature builders from tsne_gestures
from tsne_gestures import (
    SIGNAL_TYPES,
    build_feature_vector_aggregated,
    build_feature_vector_per_freq,
)


def load_dataset(iq_root: Path | list[Path], target_len: int, signal_type: str):
    features, labels, subjects, sources = [], [], [], []

    if isinstance(iq_root, Path):
        file_list = sorted(iq_root.rglob("*.npz"))
    else:
        file_list = sorted(p for root in iq_root for p in root.rglob("*.npz"))

    for path in file_list:
        with np.load(path) as data:
            gesture = str(data["gesture"])
            subject = str(data["subject"])
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

        for idx in range(left_freq.shape[0]):
            if signal_type == "per_freq":
                feat = build_feature_vector_per_freq(
                    left_freq[idx], right_freq[idx], target_len
                )
            else:
                feat = build_feature_vector_aggregated(
                    left_agg[idx], right_agg[idx],
                    left_freq[idx], right_freq[idx],
                    target_len,
                )
            features.append(feat)
            labels.append(gesture)
            subjects.append(subject)
            sources.append(f"{path.stem}#{idx:02d}")

    return np.stack(features), np.array(labels), np.array(subjects), sources


def prepare(X: np.ndarray, pca_dims: int, random_state: int):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float64)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    pca_dims = min(pca_dims, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=pca_dims, svd_solver="full", random_state=random_state)
    X_reduced = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_.cumsum()[-1]
    return X_reduced, var_exp


def classify_and_plot(
    X: np.ndarray,
    y_enc: np.ndarray,
    label_names: list[str],
    signal_type: str,
    feat_dim: int,
    var_exp: float,
    n_folds: int,
    random_state: int,
    ax_cm,
):
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=random_state,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(clf, X, y_enc, cv=cv)

    acc = np.mean(y_pred == y_enc)
    cm = confusion_matrix(y_enc, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    print(f"\n[{SIGNAL_TYPES[signal_type]}]")
    print(f"  Overall accuracy ({n_folds}-fold CV): {acc:.1%}")
    for name, a in zip(label_names, per_class_acc):
        print(f"    {name}: {a:.1%}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title(
        f"{SIGNAL_TYPES[signal_type]}\nfeat={feat_dim}d → PCA ({var_exp:.0%})  acc={acc:.1%}",
        fontsize=9,
    )
    ax_cm.tick_params(axis="x", rotation=45)

    return acc, per_class_acc


def main():
    parser = argparse.ArgumentParser(description="XGBoost gesture classification.")
    parser.add_argument("--iq-root", type=Path, nargs="+",
                        default=[
                            Path("data") / "4-24-2026-xavier" / "IQ",
                            Path("data") / "4-21-2026-henry_horizontal" / "IQ",
                            Path("data") / "4-24-2026-william" / "IQ",
                        ])
    parser.add_argument("--output", type=Path,
                        default=Path("data") / "testing" / "06_xgboost_classification_all_subjects.png")
    parser.add_argument("--target-len", type=int, default=150)
    parser.add_argument("--pca-dims", type=int, default=30)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--signal-type", choices=list(SIGNAL_TYPES) + ["compare"], default="compare")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    signal_types = list(SIGNAL_TYPES.keys()) if args.signal_type == "compare" else [args.signal_type]

    fig, axes = plt.subplots(1, len(signal_types), figsize=(7 * len(signal_types), 6))
    if len(signal_types) == 1:
        axes = [axes]

    le = LabelEncoder()
    summary = []

    for ax, stype in zip(axes, signal_types):
        print(f"\nLoading {stype}...")
        X, labels, _, _ = load_dataset(
            args.iq_root if len(args.iq_root) > 1 else args.iq_root[0],
            args.target_len, stype,
        )
        if le.classes_.size == 0 if hasattr(le, "classes_") and le.classes_.size == 0 else not hasattr(le, "classes_") or le.classes_.size == 0:
            le.fit(labels)
        y_enc = le.transform(labels)
        label_names = list(le.classes_)

        X_red, var_exp = prepare(X, args.pca_dims, args.random_state)

        acc, per_cls = classify_and_plot(
            X_red, y_enc, label_names, stype,
            X.shape[1], var_exp, args.n_folds, args.random_state, ax,
        )
        summary.append((stype, acc))

    fig.suptitle(
        f"XGBoost {args.n_folds}-fold CV  |  n={len(labels)} samples  |  {len(label_names)} gestures",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print("\n=== Summary ===")
    for stype, acc in summary:
        print(f"  {SIGNAL_TYPES[stype]}: {acc:.1%}")
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
