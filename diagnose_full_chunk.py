#!/usr/bin/env python3
"""Plot the FULL (undivided) path-change curve for one chunk to locate where motion occurs.

Usage:
    .venv/bin/python3 diagnose_full_chunk.py --gesture right20to30 --chunk 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cal_distance_bulk import (
    STATIC_SEC, IGNORE_SEC, GESTURE_SEC,
    down_convert, lowpass_cic_filter, normalize_channel,
)

DECIMATION_FACTOR = 16
DIFFERENCE_DELAY  = 17
STAGES            = 3
START_FREQ        = 17_000.0
FREQ_GAP          = 350.0
FREQ_COUNT        = 16

CARRIER_FREQS = np.array(
    [START_FREQ + FREQ_GAP * j for j in range(FREQ_COUNT)], dtype=np.float64
)

CHUNKED_ROOT = Path("data/4-27-2026-henry/chunked")
OUTPUT_DIR   = Path("data/4-27-2026-henry/visualizations")


def full_path_change(channel_audio, carrier_freq, sample_rate, remove_static):
    """Return path-change for the ENTIRE chunk (no windowing)."""
    baseband = down_convert(channel_audio, sample_rate, carrier_freq)
    filtered = lowpass_cic_filter(baseband, DECIMATION_FACTOR, DIFFERENCE_DELAY, STAGES)
    new_sr   = sample_rate / DECIMATION_FACTOR
    static_len = int(round(STATIC_SEC * new_sr))

    if remove_static:
        t      = np.arange(len(filtered), dtype=np.float64)
        static = filtered[:static_len]
        real_c = np.polyfit(t[:static_len], static.real, 1)
        imag_c = np.polyfit(t[:static_len], static.imag, 1)
        baseline = np.polyval(real_c, t) + 1j * np.polyval(imag_c, t)
        filtered = filtered - baseline

    phase = np.unwrap(np.angle(filtered))
    wl    = 343.0 / carrier_freq
    curve = (phase - phase[0]) * wl / (2 * np.pi) * 100 / 2  # physical cm
    return curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gesture", default="right20to30")
    parser.add_argument("--chunk", type=int, default=0)
    args = parser.parse_args()

    npz_path = next(CHUNKED_ROOT.glob(f"{args.gesture}/*.npz"), None)
    if npz_path is None:
        raise FileNotFoundError(f"No NPZ for gesture {args.gesture}")

    with np.load(npz_path) as d:
        audio       = d["audio"]
        sample_rate = int(d["sample_rate"])

    chunk_audio_l = normalize_channel(audio[args.chunk, :, 0])
    chunk_audio_r = normalize_channel(audio[args.chunk, :, 1])

    new_sr    = sample_rate / DECIMATION_FACTOR
    total_len = (audio.shape[1] + DECIMATION_FACTOR - 1) // DECIMATION_FACTOR
    time_axis = np.arange(total_len) / new_sr

    # 标注窗口边界
    static_end  = STATIC_SEC
    ignore_end  = STATIC_SEC + IGNORE_SEC
    gesture_start_t = time_axis[-1] - GESTURE_SEC

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, (ch_audio, ch_name) in zip(axes, [(chunk_audio_l, "Left mic"), (chunk_audio_r, "Right mic")]):
        # 取频率均值
        curves_before = np.stack([
            full_path_change(ch_audio, f, sample_rate, False) for f in CARRIER_FREQS
        ])
        curves_after = np.stack([
            full_path_change(ch_audio, f, sample_rate, True) for f in CARRIER_FREQS
        ])

        # 画每个频率（淡色）
        for c in curves_before:
            ax.plot(time_axis, c, color="#d1d5db", linewidth=0.5, alpha=0.5)
        for c in curves_after:
            ax.plot(time_axis, c, color="#a7f3d0", linewidth=0.5, alpha=0.5)

        # 画均值（粗线）
        ax.plot(time_axis, curves_before.mean(axis=0), color="#6b7280", linewidth=1.5, label="before (mean)")
        ax.plot(time_axis, curves_after.mean(axis=0),  color="#059669", linewidth=1.5, label="after (mean)")

        # 窗口边界
        ax.axvspan(0,             static_end,      alpha=0.08, color="blue",   label=f"static ({STATIC_SEC}s)")
        ax.axvspan(static_end,    ignore_end,      alpha=0.08, color="orange", label=f"ignore ({IGNORE_SEC}s)")
        ax.axvspan(gesture_start_t, time_axis[-1], alpha=0.08, color="red",    label=f"gesture ({GESTURE_SEC}s)")

        ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_ylabel("Physical Δd (cm)")
        ax.set_title(f"{ch_name} — {args.gesture} chunk {args.chunk}")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Full chunk path-change | {args.gesture} | chunk {args.chunk}\n"
        f"Blue=static window, Orange=ignore, Red=gesture window",
        fontsize=11,
    )
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"full_chunk_{args.gesture}_chunk{args.chunk}.jpg"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
