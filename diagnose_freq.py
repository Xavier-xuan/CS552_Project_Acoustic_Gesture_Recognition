#!/usr/bin/env python3
"""Visualize path-change for ALL chunks of a gesture, averaged across frequencies.

Usage:
    .venv/bin/python3 diagnose_freq.py --gesture right20to30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cal_distance_bulk import (
    STATIC_SEC,
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

WINDOW_START_SEC = 0.9
WINDOW_END_SEC   = 2.0

CHUNKED_ROOT = Path("data/4-27-2026-henry/chunked")
OUTPUT_DIR   = Path("data/4-27-2026-henry/visualizations")


def extract_path_change(channel_audio, carrier_freq, sample_rate, remove_static):
    baseband = down_convert(channel_audio, sample_rate, carrier_freq)
    filtered = lowpass_cic_filter(baseband, DECIMATION_FACTOR, DIFFERENCE_DELAY, STAGES)
    new_sr   = sample_rate / DECIMATION_FACTOR

    win_start = int(round(WINDOW_START_SEC * new_sr))
    win_end   = int(round(WINDOW_END_SEC   * new_sr))

    if remove_static:
        static_len = int(round(STATIC_SEC * new_sr))
        t      = np.arange(len(filtered), dtype=np.float64)
        static = filtered[:static_len]
        real_c = np.polyfit(t[:static_len], static.real, 1)
        imag_c = np.polyfit(t[:static_len], static.imag, 1)
        baseline = np.polyval(real_c, t) + 1j * np.polyval(imag_c, t)
        filtered = filtered - baseline

    seg   = filtered[win_start:win_end]
    phase = np.unwrap(np.angle(seg))
    wl    = 343.0 / carrier_freq
    return (phase - phase[0]) * wl / (2 * np.pi) * 100 / 2  # physical cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gesture", default="right20to30")
    args = parser.parse_args()

    npz_path = next(CHUNKED_ROOT.glob(f"{args.gesture}/*.npz"), None)
    if npz_path is None:
        raise FileNotFoundError(f"No NPZ for gesture {args.gesture}")

    with np.load(npz_path) as d:
        audio       = d["audio"]
        sample_rate = int(d["sample_rate"])

    n_chunks  = audio.shape[0]
    new_sr    = sample_rate / DECIMATION_FACTOR
    win_start = int(round(WINDOW_START_SEC * new_sr))
    win_end   = int(round(WINDOW_END_SEC   * new_sr))
    win_len   = win_end - win_start
    time_axis = np.arange(win_len) / new_sr  # 0 → (WINDOW_END - WINDOW_START) s

    print(f"Loaded {npz_path.name}: {n_chunks} chunks, window [{WINDOW_START_SEC}s, {WINDOW_END_SEC}s] = {win_len} samples")

    ncols = 4
    nrows = (n_chunks + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3),
                             sharex=True, sharey=True)
    axes_flat = axes.flat if hasattr(axes, 'flat') else [axes]

    print(f"\n{'Chunk':>6}  {'Channel':>8}  {'peak(cm)':>10}  {'final(cm)':>10}")
    for chunk_idx in range(n_chunks):
        ax = next(axes_flat)
        for ch_idx, (ch_name, color) in enumerate([("left", "#0f766e"), ("right", "#b45309")]):
            ch_audio = normalize_channel(audio[chunk_idx, :, ch_idx])
            curves = np.stack([
                extract_path_change(ch_audio, f, sample_rate, remove_static=False)
                for f in CARRIER_FREQS
            ])  # (16, T)

            # 每个频率淡线
            for c in curves:
                ax.plot(time_axis, c, color=color, linewidth=0.4, alpha=0.25)

            mean_curve = curves.mean(axis=0)
            peak_val   = mean_curve[np.argmax(np.abs(mean_curve))]
            final_val  = mean_curve[-1]
            ax.plot(time_axis, mean_curve, color=color, linewidth=1.5,
                    label=f"{ch_name} pk={peak_val:+.1f}cm")
            print(f"  {chunk_idx:>4}   {ch_name:>8}  {peak_val:>+10.2f}  {final_val:>+10.2f}")

        ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_title(f"chunk {chunk_idx}", fontsize=9)
        ax.legend(fontsize=6, loc="upper left")
        ax.grid(True, alpha=0.2)

    for ax in list(axes_flat):
        ax.set_visible(False)

    fig.suptitle(
        f"{args.gesture} | no static removal | window [{WINDOW_START_SEC}s – {WINDOW_END_SEC}s]\n"
        f"thin=per-freq, thick=mean across 16 freqs | unit: physical cm",
        fontsize=11,
    )
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"all_chunks_{args.gesture}.jpg"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
