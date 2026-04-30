#!/usr/bin/env python3
"""Visualize IQ before and after static component removal for one chunked file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cal_distance_bulk import (
    IGNORE_SEC,
    STATIC_SEC,
    GESTURE_SEC,
    down_convert,
    lowpass_cic_filter,
    normalize_channel,
)


def remove_static_component(
    filtered: np.ndarray,
    sample_rate_hz: float,
    static_sec: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    static_len = int(round(static_sec * sample_rate_hz))
    if static_len <= 0:
        raise ValueError("static_sec must produce at least one filtered sample")

    t = np.arange(filtered.shape[0], dtype=np.float64)
    t_static = t[:static_len]
    static = filtered[:static_len]
    real_c = np.polyfit(t_static, static.real, 1)
    imag_c = np.polyfit(t_static, static.imag, 1)
    baseline = np.polyval(real_c, t) + 1j * np.polyval(imag_c, t)
    return filtered - baseline, baseline, static_len


def path_change(iq: np.ndarray, carrier_freq_hz: float) -> np.ndarray:
    phase = np.unwrap(np.angle(iq))
    wavelength = 343.0 / carrier_freq_hz
    return (phase - phase[0]) * wavelength / (2 * np.pi)


def draw_iq_panel(ax, chunk_idx: int, iq: np.ndarray, label: str, color: str) -> None:
    ax.axhline(0, color="#111827", linewidth=0.8, alpha=0.35)
    ax.axvline(0, color="#111827", linewidth=0.8, alpha=0.35)
    ax.plot(iq.real, iq.imag, color=color, linewidth=0.9, alpha=0.9)
    ax.scatter(0, 0, color="#111827", marker="+", s=70, linewidths=1.6, label="origin", zorder=4)
    ax.scatter(iq[0].real, iq[0].imag, color="#16a34a", s=18, label="start", zorder=3)
    ax.scatter(iq[-1].real, iq[-1].imag, color="#dc2626", s=18, label="end", zorder=3)
    ax.set_title(f"chunk {chunk_idx + 1}: {label}")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot static component removal before/after for right20to30 data."
    )
    parser.add_argument(
        "--chunked-file",
        type=Path,
        default=Path("data/4-27-2026-henry/chunked/right20to30")
        / "sample_right20to30-henry20260427150656.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/4-27-2026-henry/visualizations")
        / "static_component_removal_right20to30.png",
    )
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--carrier-freq", type=float, default=19800.0)
    parser.add_argument("--channel", choices=["left", "right"], default="right")
    parser.add_argument("--decimation-factor", type=int, default=16)
    parser.add_argument("--difference-delay", type=int, default=17)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--static-sec", type=float, default=STATIC_SEC)
    parser.add_argument("--ignore-sec", type=float, default=IGNORE_SEC)
    parser.add_argument("--gesture-sec", type=float, default=GESTURE_SEC)
    args = parser.parse_args()

    with np.load(args.chunked_file) as data:
        audio = data["audio"]
        sample_rate = int(data["sample_rate"])
        gesture = str(data["gesture"])
        subject = str(data["subject"])
        source = str(data["source"])

    if audio.ndim != 3 or audio.shape[2] < 2:
        raise ValueError(f"Expected stereo chunked audio shaped (chunks, samples, channels), got {audio.shape}")

    channel_index = 0 if args.channel == "left" else 1
    chunks_to_plot = min(args.max_chunks, audio.shape[0])
    new_sample_rate = sample_rate / args.decimation_factor

    fig, axes = plt.subplots(chunks_to_plot, 2, figsize=(10, 4.6 * chunks_to_plot), squeeze=False)

    for chunk_idx in range(chunks_to_plot):
        channel_audio = normalize_channel(audio[chunk_idx, :, channel_index])
        baseband = down_convert(channel_audio, sample_rate, args.carrier_freq)
        before = lowpass_cic_filter(
            baseband,
            args.decimation_factor,
            args.difference_delay,
            args.stages,
        )
        after, _, _ = remove_static_component(before, new_sample_rate, args.static_sec)

        draw_iq_panel(axes[chunk_idx, 0], chunk_idx, before, "before removal", "#8a8f98")
        draw_iq_panel(axes[chunk_idx, 1], chunk_idx, after, "after removal", "#0f766e")

    title = (
        f"I/Q before vs after static component removal | {gesture} | {subject} | "
        f"{args.channel} channel | {args.carrier_freq:.0f} Hz\n"
        f"{source}"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = args.output.with_suffix(".jpg")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved I/Q comparison to: {output}")


if __name__ == "__main__":
    main()
