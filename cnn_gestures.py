#!/usr/bin/env python3
"""1D CNN gesture classifier using MLX (Apple Silicon native).

Input: left + right per-frequency path-change curves → (32, T)
       16 freq × 2 channels (L/R), resampled to target_len timesteps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder

from tsne_gestures import SIGNAL_TYPES

TARGET_LEN = 150
SUBJECTS = {
    "xavier":  Path("data/4-24-2026-xavier/IQ"),
    "henry":   Path("data/4-21-2026-henry_horizontal/IQ"),
    "william": Path("data/4-24-2026-william/IQ"),
}


# ── Data loading ──────────────────────────────────────────────────────────────

def resample(curve: np.ndarray, length: int) -> np.ndarray:
    if curve.shape[0] == length:
        return curve.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, curve.shape[0])
    x_new = np.linspace(0.0, 1.0, length)
    return np.interp(x_new, x_old, curve).astype(np.float32)


def load_subject(iq_root: Path, target_len: int = TARGET_LEN):
    """Return X (N, 32, T), labels (N,), gesture names."""
    X, labels = [], []
    for path in sorted(iq_root.rglob("*.npz")):
        with np.load(path) as d:
            gesture = str(d["gesture"])
            left  = d["left_per_freq_distance"].astype(np.float32)   # (N, 16, T)
            right = d["right_per_freq_distance"].astype(np.float32)

        for i in range(left.shape[0]):
            channels = []
            for f in range(left.shape[1]):
                lf = resample(left[i, f], target_len)
                rf = resample(right[i, f], target_len)
                joint_max = np.max(np.abs(np.concatenate([lf, rf]))) + 1e-8
                channels.extend([lf / joint_max, rf / joint_max])
            X.append(np.stack(channels))          # (32, T)
            labels.append(gesture)

    return np.stack(X), np.array(labels)


# ── Model ─────────────────────────────────────────────────────────────────────

class GestureCNN(nn.Module):
    def __init__(self, n_classes: int, in_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm(128)
        self.head  = nn.Linear(128, n_classes)

    def __call__(self, x):
        # x: (B, C, T) → MLX Conv1d expects (B, T, C)
        x = x.transpose(0, 2, 1)
        x = nn.relu(self.bn1(self.conv1(x)))
        x = x[:, ::2, :]          # stride-2 downsample
        x = nn.relu(self.bn2(self.conv2(x)))
        x = x[:, ::2, :]
        x = nn.relu(self.bn3(self.conv3(x)))
        x = x.mean(axis=1)        # global average pooling
        return self.head(x)


# ── Training helpers ──────────────────────────────────────────────────────────

def loss_fn(model, X, y):
    logits = model(X)
    return nn.losses.cross_entropy(logits, y).mean()


def accuracy(model, X, y):
    logits = model(X)
    preds  = mx.argmax(logits, axis=1)
    return (preds == y).mean().item()


def run_epoch(model, optimizer, X, y, batch_size, train: bool):
    idx = np.random.permutation(len(X)) if train else np.arange(len(X))
    losses = []
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for start in range(0, len(X), batch_size):
        batch_idx = idx[start:start + batch_size]
        Xb = mx.array(X[batch_idx])
        yb = mx.array(y[batch_idx])
        if train:
            loss, grads = loss_and_grad(model, Xb, yb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
        else:
            loss = loss_fn(model, Xb, yb)
        losses.append(loss.item())
    return float(np.mean(losses))


# ── Evaluation modes ──────────────────────────────────────────────────────────

def leave_one_out(le: LabelEncoder, data: dict[str, tuple],
                  epochs: int, batch_size: int, lr: float):
    print("\n=== Leave-One-Subject-Out ===")
    for test_subj in SUBJECTS:
        X_test, y_test = data[test_subj]
        X_train = np.concatenate([data[s][0] for s in SUBJECTS if s != test_subj])
        y_train = np.concatenate([data[s][1] for s in SUBJECTS if s != test_subj])

        model = GestureCNN(n_classes=len(le.classes_))
        mx.eval(model.parameters())
        opt = optim.Adam(learning_rate=lr)

        for ep in range(1, epochs + 1):
            run_epoch(model, opt, X_train, y_train, batch_size, train=True)

        acc = accuracy(model, mx.array(X_test), mx.array(y_test))
        print(f"  test={test_subj:8s}  acc={acc:.1%}")


def mixed_cv(le: LabelEncoder, data: dict[str, tuple],
             n_folds: int, epochs: int, batch_size: int, lr: float):
    from sklearn.model_selection import StratifiedKFold
    print("\n=== Mixed-Subject CV ===")
    X_all = np.concatenate([data[s][0] for s in SUBJECTS])
    y_all = np.concatenate([data[s][1] for s in SUBJECTS])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accs = []
    for fold, (tr, te) in enumerate(skf.split(X_all, y_all), 1):
        model = GestureCNN(n_classes=len(le.classes_))
        mx.eval(model.parameters())
        opt = optim.Adam(learning_rate=lr)

        for _ in range(epochs):
            run_epoch(model, opt, X_all[tr], y_all[tr], batch_size, train=True)

        acc = accuracy(model, mx.array(X_all[te]), mx.array(y_all[te]))
        accs.append(acc)
        print(f"  fold {fold}: {acc:.1%}")
    print(f"  mean: {np.mean(accs):.1%}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-len",  type=int,   default=TARGET_LEN)
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--n-folds",     type=int,   default=5)
    parser.add_argument("--mode", choices=["loso", "cv", "both"], default="both")
    args = parser.parse_args()

    print("Loading data...")
    le = LabelEncoder()
    data = {}
    for name, root in SUBJECTS.items():
        X, labels = load_subject(root, args.target_len)
        data[name] = (X, labels)

    all_labels = np.concatenate([data[s][1] for s in SUBJECTS])
    le.fit(all_labels)
    for name in SUBJECTS:
        X, labels = data[name]
        data[name] = (X, le.transform(labels))

    print(f"Classes: {list(le.classes_)}")
    for name in SUBJECTS:
        print(f"  {name}: {len(data[name][0])} samples")

    if args.mode in ("loso", "both"):
        leave_one_out(le, data, args.epochs, args.batch_size, args.lr)
    if args.mode in ("cv", "both"):
        mixed_cv(le, data, args.n_folds, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
