#!/usr/bin/env python3
"""MAE of 1D path-change estimate vs ground truth, before/after static removal.

Full-chunk path-change (0 → 3s). Mean across 16 frequencies.
Displacement estimate = mean of last 0.5s (settled state after gesture).
Ground truth from gesture name (e.g. right10to20 → 10 cm).
"""

import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from cal_distance_bulk import (
    STATIC_SEC,
    down_convert, lowpass_cic_filter, normalize_channel,
)

DECIMATION_FACTOR = 16
DIFFERENCE_DELAY  = 17
STAGES            = 3
CARRIER_FREQS = np.array([17000.0 + 350.0 * j for j in range(16)], dtype=np.float64)

CHUNKED_ROOT  = Path("data/4-27-2026-henry/chunked")
TAIL_SEC      = 0.5   # use last 0.5s as the settled displacement estimate


def best_freq_curve_cm(audio_1ch, sample_rate, remove_static, tail):
    """Per-freq path-change curves; return the one with largest |tail mean|, in physical cm."""
    new_sr = sample_rate / DECIMATION_FACTOR
    curves = []
    for freq in CARRIER_FREQS:
        bb  = down_convert(audio_1ch, sample_rate, freq)
        flt = lowpass_cic_filter(bb, DECIMATION_FACTOR, DIFFERENCE_DELAY, STAGES)
        if remove_static:
            sl = int(round(STATIC_SEC * new_sr))
            t  = np.arange(len(flt), dtype=np.float64)
            s  = flt[:sl]
            flt -= (np.polyval(np.polyfit(t[:sl], s.real, 1), t) +
                    1j * np.polyval(np.polyfit(t[:sl], s.imag, 1), t))
        ph = np.unwrap(np.angle(flt))
        curves.append((ph - ph[0]) * (343.0 / freq) / (2 * np.pi) * 100 / 2)
    curves = np.stack(curves)          # (16, T)
    tail_means = np.abs(curves[:, -tail:].mean(axis=1))   # |末尾均值| per freq
    best = np.argmax(tail_means)
    return curves[best]


def run(remove_static):
    errors = []
    for npz in sorted(p for p in CHUNKED_ROOT.rglob("*.npz") if "_chunk_" not in p.stem):
        gesture = npz.parent.name
        m = re.search(r"(\d+)to(\d+)", gesture)
        if not m:
            continue
        gt_cm = float(abs(int(m.group(2)) - int(m.group(1))))

        with np.load(npz) as d:
            audio, sr = d["audio"], int(d["sample_rate"])

        tail = int(round(TAIL_SEC * sr / DECIMATION_FACTOR))
        print(f"  {gesture} (gt={gt_cm:+.0f}cm)", flush=True)

        for i in range(audio.shape[0]):
            l_curve = best_freq_curve_cm(normalize_channel(audio[i, :, 0]), sr, remove_static, tail)
            r_curve = best_freq_curve_cm(normalize_channel(audio[i, :, 1]), sr, remove_static, tail)
            l_est = float(l_curve[-tail:].mean())
            r_est = float(r_curve[-tail:].mean())
            errors.append((gesture, gt_cm, abs(abs(l_est) - gt_cm), abs(abs(r_est) - gt_cm)))
    return errors


def report(label, errors):
    print(f"\n=== {label} ===")
    print(f"  {'Gesture':<22}  {'MAE_L(cm)':>10}  {'MAE_R(cm)':>10}")
    by_g = defaultdict(list)
    for g, _, ae_l, ae_r in errors:
        by_g[g].append((ae_l, ae_r))
    all_l, all_r = [], []
    for g in sorted(by_g):
        ae_l = [x[0] for x in by_g[g]]
        ae_r = [x[1] for x in by_g[g]]
        all_l += ae_l; all_r += ae_r
        print(f"  {g:<22}  {np.mean(ae_l):>10.2f}  {np.mean(ae_r):>10.2f}")
    print(f"  {'OVERALL':<22}  {np.mean(all_l):>10.2f}  {np.mean(all_r):>10.2f}")
    return np.mean(all_l), np.mean(all_r)


print("--- before ---"); eb = run(False)
print("--- after ---");  ea = run(True)
bl, br = report("BEFORE static removal", eb)
al, ar = report("AFTER  static removal", ea)
print(f"\n  Improvement: L {al-bl:+.2f} cm  R {ar-br:+.2f} cm")
