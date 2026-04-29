#!/usr/bin/env python3
"""Recognize gestures from 2D hand trajectories using Claude Vision API.

Pipeline:
  coords NPZ  →  render trajectory as image  →  Claude Vision  →  predicted label
"""

from __future__ import annotations

import argparse
import base64
import io
import re
from collections import defaultdict
from pathlib import Path

import anthropic
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KNOWN_GESTURES = [
    "AHorizontal", "BHorizontal", "CHorizontal", "DHorizontal",
    "EHorizontal", "FHorizontal", "LeftSwipeHorizontal", "RightSwipeHorizontal",
]

SYSTEM_PROMPT = """\
You are a gesture recognition system. You will be shown a 2D hand trajectory plot \
(x = lateral position in metres, y = depth position in metres, time flows from \
blue → red along the stroke). Identify which gesture was performed.

Reply with exactly one of these labels and nothing else:
AHorizontal, BHorizontal, CHorizontal, DHorizontal, EHorizontal, FHorizontal, \
LeftSwipeHorizontal, RightSwipeHorizontal"""


def render_trajectory(x: np.ndarray, y: np.ndarray) -> bytes:
    """Render a single (x, y) trajectory as a PNG and return raw bytes."""
    fig, ax = plt.subplots(figsize=(3, 3))

    # Drop NaN frames
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        x, y = np.array([0.0, 0.0]), np.array([0.0, 0.0])

    t = np.linspace(0, 1, len(x))
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], color=plt.cm.coolwarm(t[i]), linewidth=2)

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def predict_gesture(client: anthropic.Anthropic, png_bytes: bytes) -> str:
    b64 = base64.standard_b64encode(png_bytes).decode()
    msg = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=32,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [{
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            }],
        }],
    )
    raw = msg.content[0].text.strip()
    # Tolerate minor casing / punctuation differences
    for label in KNOWN_GESTURES:
        if label.lower() in raw.lower():
            return label
    return raw


def main():
    parser = argparse.ArgumentParser(description="Recognize gestures via Claude Vision.")
    parser.add_argument("--coords-root", type=Path, nargs="+",
                        default=[
                            Path("data/4-24-2026-xavier/coords"),
                            Path("data/4-21-2026-henry_horizontal/coords"),
                            Path("data/4-24-2026-william/coords"),
                        ])
    parser.add_argument("--max-chunks", type=int, default=3,
                        help="Max trajectory chunks per NPZ file to send (API cost control)")
    args = parser.parse_args()

    client = anthropic.Anthropic()

    npz_files = sorted(p for root in args.coords_root for p in root.rglob("*.npz"))
    if not npz_files:
        print("No coords NPZ files found. Run compute_2d_coords_bulk.py first.")
        return

    correct = total = 0
    per_class: dict[str, list[bool]] = defaultdict(list)

    for npz_path in npz_files:
        gesture = npz_path.parent.name  # directory name is the gesture label
        with np.load(npz_path) as data:
            x_all = data["x_coords"]  # (num_chunks, T)
            y_all = data["y_coords"]

        n = min(args.max_chunks, x_all.shape[0])
        for i in range(n):
            png = render_trajectory(x_all[i], y_all[i])
            pred = predict_gesture(client, png)
            hit = pred == gesture
            correct += hit
            total += 1
            per_class[gesture].append(hit)
            status = "✓" if hit else f"✗ (got {pred})"
            print(f"  {npz_path.name}[{i}]  true={gesture}  {status}")

    print(f"\n=== Summary ===")
    print(f"  Overall accuracy: {correct}/{total} = {correct/max(total,1):.1%}")
    for g in KNOWN_GESTURES:
        if g in per_class:
            hits = per_class[g]
            print(f"  {g}: {sum(hits)}/{len(hits)} = {sum(hits)/len(hits):.1%}")


if __name__ == "__main__":
    main()
