#!/usr/bin/env python3
"""Compute 2D hand coordinates from acoustic path-length changes.

Implements the ellipse-intersection method from LLAP (Wang et al., MobiCom 2016), §4.5 eq. 6.

Coordinate system (matches paper Fig. 9b):
    Speaker        at (0,  0)   — origin
    Left  mic (L)  at (0,  L1)
    Right mic (R)  at (0, -L2)
    Hand           at (x,  y)   — x ≥ 0 is the frontal half-plane

The left/right channels in the IQ NPZ map to mic 1 (L1) and mic 2 (L2) respectively.
"""

from pathlib import Path

import numpy as np

# ── Device geometry ──────────────────────────────────────────────────────────
# Measure on your recording device: distance in metres from the speaker to each mic.
L1 = 0.16    # speaker → left microphone
L2 = 0.03    # speaker → right microphone

# ── Initial absolute path lengths ────────────────────────────────────────────
# Total acoustic path (speaker → hand + hand → mic) at the start of each gesture.
# e.g. hand ~30 cm directly in front, mic 10 cm from speaker:
#   d = 0.30 + sqrt(0.30² + 0.10²) ≈ 0.62 m
D1_INIT = 0.60   # initial d1 for left mic  (metres)
D2_INIT = 0.60   # initial d2 for right mic (metres)

# ── Delay-profile coarse correction (§4.1) ───────────────────────────────────
# Requires left_per_freq_iq / right_per_freq_iq in the input NPZ.
# Re-run cal_distance_bulk.py with SAVE_IQ = True to generate those fields.
USE_DELAY_PROFILE = False
EMA_ALPHA = 0.10        # blend weight: higher → faster adaptation to coarse estimate
DELAY_PROFILE_MIN_PEAK_FRACTION = 0.20  # minimum (peak / sum) to accept coarse estimate

# ── I/O for single-file mode ─────────────────────────────────────────────────
INPUT_NPZ  = Path("data/preliminary_data/IQ/lowerA/sample_lowerA-Henry-20260403140615.npz")
OUTPUT_NPZ = Path("data/preliminary_data/coords/lowerA/sample_lowerA-Henry-20260403140615.npz")

# Speed of sound (m/s)
_C = 343.0


# ─────────────────────────────────────────────────────────────────────────────
# Core geometry
# ─────────────────────────────────────────────────────────────────────────────

def compute_2d_from_path_lengths(
    d1: np.ndarray, d2: np.ndarray, l1: float, l2: float
):
    """Return (x, y) arrays for hand positions given absolute path lengths d1, d2.

    d1, d2: absolute round-trip path lengths in metres, shape (...,)
    l1, l2: speaker-to-mic distances in metres (scalars)

    Points where the geometric constraints are violated become NaN.
    """
    denom = 2.0 * (d1 * l2 + d2 * l1)
    safe = np.abs(denom) > 1e-9

    radicand = (
        (d1 ** 2 - l1 ** 2)
        * (d2 ** 2 - l2 ** 2)
        * ((l1 + l2) ** 2 - (d1 - d2) ** 2)
    )
    x = np.where(safe, np.sqrt(np.maximum(radicand, 0.0)) / denom, np.nan)
    y = np.where(
        safe,
        (d1 * (d2 ** 2 - l2 ** 2) - d2 * (d1 ** 2 - l1 ** 2)) / denom,
        np.nan,
    )
    return x.astype(np.float32), y.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Delay-profile coarse path estimation (LLAP §4.1)
# ─────────────────────────────────────────────────────────────────────────────

def _delay_profile_estimate(iq_freqs: np.ndarray, delta_f: float) -> np.ndarray:
    """Coarse absolute path length from IDFT across N carrier frequencies.

    iq_freqs: complex array (N_freqs, T)
    delta_f:  frequency spacing in Hz

    Returns d_coarse (T,) in metres; NaN where the peak is too weak.
    """
    N = iq_freqs.shape[0]
    # IDFT along frequency axis: peak index n_hat → d = n_hat * c / (N * Δf)
    profile = np.abs(np.fft.ifft(iq_freqs, n=N, axis=0))   # (N, T)
    peak_idx = np.argmax(profile, axis=0).astype(np.float64)  # (T,)
    peak_val = profile.max(axis=0)
    total_energy = profile.sum(axis=0) + 1e-12
    norm_peak = peak_val / total_energy

    d_coarse = peak_idx * _C / (N * delta_f)
    d_coarse = np.where(norm_peak >= DELAY_PROFILE_MIN_PEAK_FRACTION, d_coarse, np.nan)
    return d_coarse


def _apply_ema_correction(
    d_fine: np.ndarray, d_coarse: np.ndarray, d_init: float
) -> np.ndarray:
    """Blend fine-grained phase accumulation with coarse delay-profile estimate via EMA.

    d_fine:   shape (T,) — d_init + delta_d from phase changes
    d_coarse: shape (T,) — coarse absolute estimate (may contain NaN)
    d_init:   scalar initial path length used to seed the EMA
    """
    T = len(d_fine)
    d_out = d_fine.copy()
    d_anchor = d_init  # running EMA anchor
    for t in range(T):
        if not np.isnan(d_coarse[t]):
            d_anchor = (1.0 - EMA_ALPHA) * d_anchor + EMA_ALPHA * d_coarse[t]
        # shift the fine curve so it tracks the EMA-corrected anchor
        d_out[t] = d_fine[t] + (d_anchor - d_init)
    return d_out


# ─────────────────────────────────────────────────────────────────────────────
# Per-file processing
# ─────────────────────────────────────────────────────────────────────────────

def compute_coords_for_npz(
    input_npz: Path,
    output_path: Path,
    l1: float = L1,
    l2: float = L2,
    d1_init: float = D1_INIT,
    d2_init: float = D2_INIT,
):
    """Load an IQ NPZ, compute 2D coordinates, and save an augmented NPZ."""
    with np.load(input_npz, allow_pickle=True) as data:
        delta_d1 = data["left_regression_distance"].astype(np.float64)   # (C, T)
        delta_d2 = data["right_regression_distance"].astype(np.float64)  # (C, T)
        carrier_freqs = data["carrier_freqs"].astype(np.float64)
        saved_fields = {k: data[k] for k in data.files}
        has_iq = "left_per_freq_iq" in data.files

    num_chunks, T = delta_d1.shape
    delta_f = float(np.diff(carrier_freqs).mean()) if len(carrier_freqs) > 1 else 350.0

    if USE_DELAY_PROFILE and not has_iq:
        print(
            f"  Warning: USE_DELAY_PROFILE=True but 'left_per_freq_iq' is absent in\n"
            f"  {input_npz}\n"
            f"  Re-run cal_distance_bulk.py with SAVE_IQ=True, then retry.\n"
            f"  Falling back to fixed D1_INIT / D2_INIT."
        )

    x_all = np.full((num_chunks, T), np.nan, dtype=np.float32)
    y_all = np.full((num_chunks, T), np.nan, dtype=np.float32)
    d1_abs_all = np.zeros((num_chunks, T), dtype=np.float32)
    d2_abs_all = np.zeros((num_chunks, T), dtype=np.float32)

    # Load IQ once if needed to avoid repeated file I/O in the loop
    left_iq_all = right_iq_all = None
    if USE_DELAY_PROFILE and has_iq:
        with np.load(input_npz, allow_pickle=True) as data:
            left_iq_all = data["left_per_freq_iq"]   # (C, N_freqs, T)
            right_iq_all = data["right_per_freq_iq"]

    for c in range(num_chunks):
        d1_fine = d1_init + delta_d1[c]   # (T,)
        d2_fine = d2_init + delta_d2[c]

        if USE_DELAY_PROFILE and has_iq:
            d1_coarse = _delay_profile_estimate(left_iq_all[c].T, delta_f)
            d2_coarse = _delay_profile_estimate(right_iq_all[c].T, delta_f)
            d1_abs = _apply_ema_correction(d1_fine, d1_coarse, d1_init)
            d2_abs = _apply_ema_correction(d2_fine, d2_coarse, d2_init)
        else:
            d1_abs = d1_fine
            d2_abs = d2_fine

        x, y = compute_2d_from_path_lengths(d1_abs, d2_abs, l1, l2)
        x_all[c] = x
        y_all[c] = y
        d1_abs_all[c] = d1_abs.astype(np.float32)
        d2_abs_all[c] = d2_abs.astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **saved_fields,
        x_coords=x_all,
        y_coords=y_all,
        d1_abs=d1_abs_all,
        d2_abs=d2_abs_all,
        L1=np.float32(l1),
        L2=np.float32(l2),
        d1_init=np.float32(d1_init),
        d2_init=np.float32(d2_init),
    )
    print(f"Saved 2D coords ({num_chunks} chunk(s)) → {output_path}")
    return num_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Single-file entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not INPUT_NPZ.exists():
        raise FileNotFoundError(
            f"INPUT_NPZ not found: {INPUT_NPZ}\n"
            "Edit the INPUT_NPZ constant at the top of this file."
        )
    compute_coords_for_npz(INPUT_NPZ, OUTPUT_NPZ)


if __name__ == "__main__":
    main()
