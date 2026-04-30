#!/usr/bin/env python3
"""XGBoost gesture classification — no PCA, raw features directly."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from tsne_gestures import (
    SIGNAL_TYPES,
    build_feature_vector_aggregated,
    build_feature_vector_per_freq,
)


def load_dataset(iq_root: Path | list[Path], target_len: int, signal_type: str, gestures: list[str] | None = None):
    features, labels, subjects, sources = [], [], [], []

    if isinstance(iq_root, Path):
        fileList = iq_root.rglob("*.npz")
    else:
        fileList = []
        for p in iq_root:
            fileList.extend(p.rglob("*.npz"))

    for path in sorted(fileList):
        with np.load(path) as data:
            gesture = str(data["gesture"])
            if gestures is not None and gesture not in gestures:
                continue
            subject = str(data["subject"])
            left_freq = data["left_per_freq_distance"].astype(np.float32)
            right_freq = data["right_per_freq_distance"].astype(np.float32)

            if signal_type == "mean":
                left_agg = data["left_mean_distance"].astype(np.float32)
                right_agg = data["right_mean_distance"].astype(np.float32)
            elif signal_type == "regression":
                left_agg = data["left_regression_distance"].astype(np.float32)
                right_agg = data["right_regression_distance"].astype(np.float32)
            elif signal_type == "best_freq":
                left_agg = data["left_best_freq_distance"].astype(np.float32)
                right_agg = data["right_best_freq_distance"].astype(np.float32)
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


def train_and_predict(
    X: np.ndarray,
    y_enc: np.ndarray,
    X2: np.ndarray,
    y_enc2: np.ndarray,
    label_names: list[str],
    signal_type: str,
    feat_dim: int,
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

    clf.fit(X, y_enc)
    y_pred = clf.predict(X2)

    cm = confusion_matrix(y_enc2, y_pred, labels=np.arange(len(label_names)))
    row_totals = cm.sum(axis=1)
    per_class_acc = np.divide(
        cm.diagonal(),
        row_totals,
        out=np.zeros_like(row_totals, dtype=float),
        where=row_totals != 0,
    )
    acc = per_class_acc.mean()

    print(f"\n[{SIGNAL_TYPES[signal_type]}]")
    print(f"  Macro avg accuracy: {acc:.1%}")
    for name, a in zip(label_names, per_class_acc):
        print(f"    {name}: {a:.1%}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title(
        f"{SIGNAL_TYPES[signal_type]}\nfeat={feat_dim}  macro-acc={acc:.1%}",
        fontsize=9,
    )
    ax_cm.tick_params(axis="x", rotation=45)

    return acc, per_class_acc


DIR1 = "4-24-2026-william"
DIR2 = "4-24-2026-xavier"
DIR3 = "4-21-2026-henry_horizontal"

def main():
    parser = argparse.ArgumentParser(description="XGBoost gesture classification (no PCA).")
    parser.add_argument("--iq-root-training", type=Path,
                        default=Path("data") / DIR1 / "IQ")
    parser.add_argument("--iq-root-training2", type=Path,
                        default=Path("data") / DIR2 / "IQ")
    parser.add_argument("--iq-root-training3", type=Path, default=None)
    parser.add_argument("--iq-root-eval", type=Path, default=None,
                        help="Held-out subject root for leave-one-subject-out eval")
    parser.add_argument("--output", type=Path,
                        default=Path("data") / "testing" / "predict_nopca.png")
    parser.add_argument("--target-len", type=int, default=150)
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data used for testing in random split (default 0.2)")
    parser.add_argument("--signal-type", choices=list(SIGNAL_TYPES) + ["compare"], default="compare")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--gestures", nargs="+", default=None,
                        help="Only load these gesture classes (e.g. --gestures LeftSwipeHorizontal RightSwipeHorizontal)")
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
        train_roots = [r for r in [args.iq_root_training, args.iq_root_training2, args.iq_root_training3] if r is not None]
        X_train, labels_train, _, _ = load_dataset(train_roots, args.target_len, stype, args.gestures)
        feat_dim = X_train.shape[1]

        if args.iq_root_eval is not None:
            X_eval, labels_eval, _, _ = load_dataset(args.iq_root_eval, args.target_len, stype, args.gestures)
            all_labels = np.concatenate([labels_train, labels_eval])
            le.fit(all_labels)
            label_names = list(le.classes_)
            y_enc = le.transform(labels_train)
            y_enc2 = le.transform(labels_eval)
        else:
            y_enc_all = le.fit_transform(labels_train)
            label_names = list(le.classes_)
            X_train, X_eval, y_enc, y_enc2 = train_test_split(
                X_train, y_enc_all,
                test_size=args.test_size,
                stratify=y_enc_all,
                random_state=args.random_state,
            )

        acc, per_cls = train_and_predict(
            X_train, y_enc, X_eval, y_enc2, label_names, stype,
            feat_dim, args.random_state, ax,
        )
        summary.append((stype, acc))

    n_train = len(labels_train)
    train_pct = int((1 - args.test_size) * 100)
    test_pct = int(args.test_size * 100)
    title_mode = "leave-one-subject-out" if args.iq_root_eval is not None else f"stratified {train_pct}/{test_pct} split"
    fig.suptitle(
        f"XGBoost {title_mode} (no PCA)  |  n_train={n_train}  |  {len(label_names)} gestures",
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
