#!/usr/bin/env python3
"""Plot mean-across-frequencies path-change for all gestures, all chunks overlaid.

Usage:
    .venv/bin/python3 diagnose_mean.py
"""

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
CARRIER_FREQS = np.array(
    [17000.0 + 350.0 * j for j in range(16)], dtype=np.float64
)

WINDOW_START_SEC = 0.9
WINDOW_END_SEC   = 2.0

CHUNKED_ROOT = Path("data/4-27-2026-henry/chunked")
OUTPUT_DIR   = Path("data/4-27-2026-henry/visualizations")


def mean_path_change(channel_audio, sample_rate):
    new_sr    = sample_rate / DECIMATION_FACTOR
    win_start = int(round(WINDOW_START_SEC * new_sr))
    win_end   = int(round(WINDOW_END_SEC   * new_sr))

    curves = []
    for freq in CARRIER_FREQS:
        baseband = down_convert(channel_audio, sample_rate, freq)
        filtered = lowpass_cic_filter(baseband, DECIMATION_FACTOR, DIFFERENCE_DELAY, STAGES)
        seg   = filtered[win_start:win_end]
        phase = np.unwrap(np.angle(seg))
        wl    = 343.0 / freq
        curves.append((phase - phase[0]) * wl / (2 * np.pi) * 100 / 2)  # cm

    return np.stack(curves).mean(axis=0)


def main():
    gestures = sorted(p.name for p in CHUNKED_ROOT.iterdir() if p.is_dir())
    n = len(gestures)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3),
                             sharex=True)
    axes_flat = axes.flat

    for gesture in gestures:
        ax = next(axes_flat)
        npz_path = next(CHUNKED_ROOT.glob(f"{gesture}/*.npz"), None)
        if npz_path is None:
            ax.set_title(gesture); continue

        with np.load(npz_path) as d:
            audio       = d["audio"]
            sample_rate = int(d["sample_rate"])

        new_sr    = sample_rate / DECIMATION_FACTOR
        win_len   = int(round(WINDOW_END_SEC * new_sr)) - int(round(WINDOW_START_SEC * new_sr))
        time_axis = np.arange(win_len) / new_sr

        for chunk_idx in range(audio.shape[0]):
            for ch_idx, color in [(0, "#0f766e"), (1, "#b45309")]:
                ch_audio = normalize_channel(audio[chunk_idx, :, ch_idx])
                curve = mean_path_change(ch_audio, sample_rate)
                ax.plot(time_axis, curve, color=color, linewidth=0.9, alpha=0.5)

        ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_title(gesture, fontsize=9)
        ax.set_ylabel("cm", fontsize=8)
        ax.grid(True, alpha=0.2)

    for ax in list(axes_flat):
        ax.set_visible(False)

    fig.suptitle(
        f"Mean across 16 freqs | window [{WINDOW_START_SEC}–{WINDOW_END_SEC}s] | no static removal\n"
        f"Green=left mic, Orange=right mic, all chunks overlaid",
        fontsize=11,
    )
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "mean_all_gestures.jpg"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
