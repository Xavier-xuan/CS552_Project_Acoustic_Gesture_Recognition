#!/usr/bin/env python3
"""Bulk path-change estimation from aggregated chunked NPZ files."""

import argparse
from pathlib import Path

import numpy as np

# Set True to also save per-frequency complex dynamic IQ vectors (needed for delay-profile
# absolute path estimation in compute_2d_coords.py).  Increases output file size ~4× per chunk.
SAVE_IQ = False

STATIC_SEC = 0.8
IGNORE_SEC = 0.2
GESTURE_SEC = 1.0


def derive_gesture_dir(input_npz: Path, chunked_root: Path) -> str:
    try:
        return input_npz.parent.relative_to(chunked_root).parts[0]
    except Exception:
        return "unknown"


def is_aggregated_chunk_file(path: Path) -> bool:
    return path.suffix == ".npz" and "_chunk_" not in path.stem


def already_processed(input_npz: Path, output_dir: Path) -> bool:
    return (output_dir / input_npz.name).exists()


def lowpass_cic_filter(signal: np.ndarray, decimation_factor: int,
                       difference_delay: int = 1, stages: int = 3) -> np.ndarray:
    signal = signal.astype(np.complex128, copy=False)

    for _ in range(stages):
        signal = np.cumsum(signal)

    signal = signal[::decimation_factor]

    for _ in range(stages):
        delayed = np.concatenate((
            np.zeros(difference_delay, dtype=signal.dtype),
            signal[:-difference_delay],
        ))
        signal = signal - delayed

    signal /= (decimation_factor * difference_delay) ** stages
    return signal


def down_convert(audio: np.ndarray, sample_rate_hz: int, carrier_freq: float) -> np.ndarray:
    t = np.arange(len(audio), dtype=np.float64) / sample_rate_hz
    i_signal = audio * np.cos(2 * np.pi * carrier_freq * t)
    q_signal = audio * -np.sin(2 * np.pi * carrier_freq * t)
    return i_signal + 1j * q_signal


def normalize_channel(audio: np.ndarray) -> np.ndarray:
    audio = audio.astype(np.float32, copy=False)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def extract_path_change_curve(channel_audio: np.ndarray, carrier_freq: float,
                              sample_rate_hz: int, decimation_factor: int,
                              difference_delay: int, stages: int,
                              static_sec: float, ignore_sec: float,
                              gesture_sec: float,
                              return_iq: bool = False):
    baseband = down_convert(channel_audio, sample_rate_hz, carrier_freq)
    filtered = lowpass_cic_filter(baseband, decimation_factor, difference_delay, stages)
    new_sample_rate = sample_rate_hz / decimation_factor
    static_len = int(round(static_sec * new_sample_rate))
    ignore_len = int(round(ignore_sec * new_sample_rate))
    gesture_len = int(round(gesture_sec * new_sample_rate))
    gesture_end = filtered.shape[0]
    gesture_start = gesture_end - gesture_len

    if static_len <= 0:
        raise ValueError("static_sec must produce at least one filtered sample")
    if gesture_start < static_len + ignore_len:
        available_sec = filtered.shape[0] / new_sample_rate
        raise ValueError(
            f"Need {static_sec + ignore_sec + gesture_sec:.3f}s per chunk for "
            f"static/ignore/gesture windows, got {available_sec:.3f}s after filtering"
        )

    # Subtract the mean of the full IQ signal for static component removal.
    filtered = filtered - filtered.mean()

    gesture_filtered = filtered[gesture_start:gesture_end]
    phase = np.unwrap(np.angle(gesture_filtered))
    wavelength = 343.0 / carrier_freq
    path_change = (phase - phase[0]) * wavelength / (2 * np.pi)
    if return_iq:
        return path_change, gesture_filtered
    return path_change


def robust_multifreq_combine(path_change_curves: np.ndarray, time_axis: np.ndarray,
                             sample_rate_hz: float, window_ms: float = 10.0,
                             min_keep: int = 4, sigma: float = 3.0):
    window_len = max(4, int(round(window_ms * 1e-3 * sample_rate_hz)))
    freq_count, sample_count = path_change_curves.shape
    fused = np.zeros(sample_count, dtype=np.float64)
    kept_counts = np.zeros(sample_count, dtype=np.int32)
    rmse_before = np.zeros(sample_count, dtype=np.float64)
    rmse_after = np.zeros(sample_count, dtype=np.float64)

    for start in range(0, sample_count, window_len):
        end = min(sample_count, start + window_len)
        t_win = time_axis[start:end]
        y_win = path_change_curves[:, start:end]

        coeff_all = np.polyfit(np.tile(t_win, freq_count), y_win.reshape(-1), 1)
        pred_all = np.polyval(coeff_all, t_win)
        residuals = np.sqrt(np.mean((y_win - pred_all[None, :]) ** 2, axis=1))

        median_residual = np.median(residuals)
        mad = np.median(np.abs(residuals - median_residual)) + 1e-12
        threshold = median_residual + sigma * 1.4826 * mad
        keep = residuals <= threshold

        if keep.sum() < min_keep:
            keep = np.zeros(freq_count, dtype=bool)
            keep[np.argsort(residuals)[:min_keep]] = True

        coeff_keep = np.polyfit(np.tile(t_win, int(keep.sum())), y_win[keep].reshape(-1), 1)
        pred_keep = np.polyval(coeff_keep, t_win)

        fused[start:end] = pred_keep
        kept_counts[start:end] = keep.sum()
        rmse_before[start:end] = np.sqrt(np.mean((y_win - pred_all[None, :]) ** 2))
        rmse_after[start:end] = np.sqrt(np.mean((y_win[keep] - pred_keep[None, :]) ** 2))

    return fused, kept_counts, rmse_before, rmse_after


def calculate_chunk_distances(audio_chunks: np.ndarray, carrier_freqs: np.ndarray,
                              sample_rate_hz: int, decimation_factor: int,
                              difference_delay: int, stages: int,
                              window_ms: float, min_keep: int, sigma: float,
                              static_sec: float, ignore_sec: float,
                              gesture_sec: float,
                              save_iq: bool = False):
    num_chunks = audio_chunks.shape[0]
    new_sample_rate = sample_rate_hz / decimation_factor
    filtered_len = int(round(gesture_sec * new_sample_rate))
    filtered_total_len = (audio_chunks.shape[1] + decimation_factor - 1) // decimation_factor
    gesture_start_sec = (filtered_total_len - filtered_len) / new_sample_rate
    num_freqs = carrier_freqs.shape[0]

    left_per_freq = np.zeros((num_chunks, num_freqs, filtered_len), dtype=np.float32)
    right_per_freq = np.zeros((num_chunks, num_freqs, filtered_len), dtype=np.float32)
    left_mean = np.zeros((num_chunks, filtered_len), dtype=np.float32)
    right_mean = np.zeros((num_chunks, filtered_len), dtype=np.float32)
    left_regression = np.zeros((num_chunks, filtered_len), dtype=np.float32)
    right_regression = np.zeros((num_chunks, filtered_len), dtype=np.float32)
    left_kept_counts = np.zeros((num_chunks, filtered_len), dtype=np.int32)
    right_kept_counts = np.zeros((num_chunks, filtered_len), dtype=np.int32)
    left_rmse_before = np.zeros((num_chunks, filtered_len), dtype=np.float32)
    left_rmse_after = np.zeros((num_chunks, filtered_len), dtype=np.float32)
    right_rmse_before = np.zeros((num_chunks, filtered_len), dtype=np.float32)
    right_rmse_after = np.zeros((num_chunks, filtered_len), dtype=np.float32)

    if save_iq:
        left_iq = np.zeros((num_chunks, num_freqs, filtered_len), dtype=np.complex64)
        right_iq = np.zeros((num_chunks, num_freqs, filtered_len), dtype=np.complex64)

    time_axis = np.arange(filtered_len, dtype=np.float64) / new_sample_rate

    for chunk_idx in range(num_chunks):
        audio_stereo = audio_chunks[chunk_idx].astype(np.float32, copy=False)
        audio_left = normalize_channel(audio_stereo[:, 0])
        audio_right = normalize_channel(audio_stereo[:, 1])

        left_curves = []
        right_curves = []

        for f_idx, carrier_freq in enumerate(carrier_freqs):
            if save_iq:
                l_curve, l_filt = extract_path_change_curve(
                    audio_left, carrier_freq, sample_rate_hz,
                    decimation_factor, difference_delay, stages,
                    static_sec, ignore_sec, gesture_sec, return_iq=True,
                )
                r_curve, r_filt = extract_path_change_curve(
                    audio_right, carrier_freq, sample_rate_hz,
                    decimation_factor, difference_delay, stages,
                    static_sec, ignore_sec, gesture_sec, return_iq=True,
                )
                left_iq[chunk_idx, f_idx] = l_filt.astype(np.complex64)
                right_iq[chunk_idx, f_idx] = r_filt.astype(np.complex64)
            else:
                l_curve = extract_path_change_curve(
                    audio_left, carrier_freq, sample_rate_hz,
                    decimation_factor, difference_delay, stages,
                    static_sec, ignore_sec, gesture_sec,
                )
                r_curve = extract_path_change_curve(
                    audio_right, carrier_freq, sample_rate_hz,
                    decimation_factor, difference_delay, stages,
                    static_sec, ignore_sec, gesture_sec,
                )
            left_curves.append(l_curve)
            right_curves.append(r_curve)

        left_curves = np.stack(left_curves, axis=0)
        right_curves = np.stack(right_curves, axis=0)

        left_per_freq[chunk_idx] = left_curves.astype(np.float32)
        right_per_freq[chunk_idx] = right_curves.astype(np.float32)
        left_mean[chunk_idx] = np.mean(left_curves, axis=0).astype(np.float32)
        right_mean[chunk_idx] = np.mean(right_curves, axis=0).astype(np.float32)

        left_fused, left_kept, left_before, left_after = robust_multifreq_combine(
            left_curves, time_axis, new_sample_rate, window_ms, min_keep, sigma
        )
        right_fused, right_kept, right_before, right_after = robust_multifreq_combine(
            right_curves, time_axis, new_sample_rate, window_ms, min_keep, sigma
        )

        left_regression[chunk_idx] = left_fused.astype(np.float32)
        right_regression[chunk_idx] = right_fused.astype(np.float32)
        left_kept_counts[chunk_idx] = left_kept
        right_kept_counts[chunk_idx] = right_kept
        left_rmse_before[chunk_idx] = left_before.astype(np.float32)
        left_rmse_after[chunk_idx] = left_after.astype(np.float32)
        right_rmse_before[chunk_idx] = right_before.astype(np.float32)
        right_rmse_after[chunk_idx] = right_after.astype(np.float32)

    # Best-frequency selection: pick the carrier with the highest std over the gesture window
    left_best_freq = np.zeros((num_chunks, filtered_len), dtype=np.float32)
    right_best_freq = np.zeros((num_chunks, filtered_len), dtype=np.float32)
    for c in range(num_chunks):
        l_std = left_per_freq[c].std(axis=1)
        r_std = right_per_freq[c].std(axis=1)
        left_best_freq[c] = left_per_freq[c, np.argmax(l_std)]
        right_best_freq[c] = right_per_freq[c, np.argmax(r_std)]

    out = {
        "time_axis": time_axis.astype(np.float32),
        "new_sample_rate": np.array(new_sample_rate, dtype=np.float32),
        "static_sec": np.array(static_sec, dtype=np.float32),
        "min_ignore_sec": np.array(ignore_sec, dtype=np.float32),
        "ignored_sec": np.array(gesture_start_sec - static_sec, dtype=np.float32),
        "gesture_sec": np.array(gesture_sec, dtype=np.float32),
        "gesture_start_sec": np.array(gesture_start_sec, dtype=np.float32),
        "left_per_freq_distance": left_per_freq,
        "right_per_freq_distance": right_per_freq,
        "left_mean_distance": left_mean,
        "right_mean_distance": right_mean,
        "left_regression_distance": left_regression,
        "right_regression_distance": right_regression,
        "left_best_freq_distance": left_best_freq,
        "right_best_freq_distance": right_best_freq,
        "left_kept_counts": left_kept_counts,
        "right_kept_counts": right_kept_counts,
        "left_rmse_before": left_rmse_before,
        "left_rmse_after": left_rmse_after,
        "right_rmse_before": right_rmse_before,
        "right_rmse_after": right_rmse_after,
    }
    if save_iq:
        out["left_per_freq_iq"] = left_iq
        out["right_per_freq_iq"] = right_iq
    return out


def process_file(input_npz: Path, output_dir: Path, carrier_freqs: np.ndarray,
                 decimation_factor: int, difference_delay: int, stages: int,
                 window_ms: float, min_keep: int, sigma: float,
                 static_sec: float, ignore_sec: float, gesture_sec: float,
                 save_iq: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(input_npz) as chunk_data:
        audio = chunk_data["audio"]
        sample_rate = int(chunk_data["sample_rate"])
        channels = int(chunk_data["channels"])
        subject = str(chunk_data["subject"])
        gesture = str(chunk_data["gesture"])
        chunk_channels = chunk_data["chunk_channels"]
        chunk_indices = chunk_data["chunk_indices"]
        chunk_starts = chunk_data["chunk_starts"]
        chunk_ends = chunk_data["chunk_ends"]
        source = str(chunk_data["source"])

    if audio.ndim != 3 or channels < 2:
        raise ValueError(f"Expected aggregated stereo chunks in {input_npz}, got shape {audio.shape}")

    print(f"Loaded {input_npz}: {audio.shape[0]} chunk(s), {audio.shape[1]} samples/chunk")

    results = calculate_chunk_distances(
        audio_chunks=audio,
        carrier_freqs=carrier_freqs,
        sample_rate_hz=sample_rate,
        decimation_factor=decimation_factor,
        difference_delay=difference_delay,
        stages=stages,
        window_ms=window_ms,
        min_keep=min_keep,
        sigma=sigma,
        static_sec=static_sec,
        ignore_sec=ignore_sec,
        gesture_sec=gesture_sec,
        save_iq=save_iq,
    )

    out_path = output_dir / input_npz.name
    np.savez_compressed(
        out_path,
        subject=np.array(subject),
        gesture=np.array(gesture),
        source=np.array(source),
        sample_rate=np.array(sample_rate, dtype=np.int32),
        channels=np.array(channels, dtype=np.int32),
        carrier_freqs=carrier_freqs.astype(np.float32),
        chunk_channels=chunk_channels,
        chunk_indices=chunk_indices.astype(np.int32),
        chunk_starts=chunk_starts.astype(np.int32),
        chunk_ends=chunk_ends.astype(np.int32),
        decimation_factor=np.array(decimation_factor, dtype=np.int32),
        difference_delay=np.array(difference_delay, dtype=np.int32),
        stages=np.array(stages, dtype=np.int32),
        window_ms=np.array(window_ms, dtype=np.float32),
        min_keep=np.array(min_keep, dtype=np.int32),
        sigma=np.array(sigma, dtype=np.float32),
        measurement_note=np.array(
            "Values are relative acoustic path-length changes estimated from phase, not absolute range."
        ),
        **results,
    )

    print(f"Saved distance estimates to: {out_path}")
    return audio.shape[0]


def main():
    parser = argparse.ArgumentParser(
        description="Incrementally estimate left/right path-change curves from chunked NPZ files."
    )
    parser.add_argument("--chunked-root", type=Path, default=Path("data") / "preliminary_data" / "chunked",
                        help="Root directory containing aggregated chunked NPZ files")
    parser.add_argument("--output-root", type=Path, default=Path("data") / "preliminary_data" / "IQ",
                        help="Root directory for aggregated distance/path-change NPZ files")
    parser.add_argument("--start-freq", type=float, default=17000.0,
                        help="First carrier frequency in Hz")
    parser.add_argument("--freq-gap", type=float, default=350.0,
                        help="Carrier spacing in Hz")
    parser.add_argument("--freq-count", type=int, default=16,
                        help="Number of carrier frequencies")
    parser.add_argument("--decimation-factor", type=int, default=16,
                        help="CIC decimation factor")
    parser.add_argument("--difference-delay", type=int, default=17,
                        help="CIC differential delay")
    parser.add_argument("--stages", type=int, default=3,
                        help="Number of CIC stages")
    parser.add_argument("--window-ms", type=float, default=10.0,
                        help="Window length in milliseconds for regression fusion")
    parser.add_argument("--min-keep", type=int, default=4,
                        help="Minimum number of frequencies kept per regression window")
    parser.add_argument("--sigma", type=float, default=3.0,
                        help="MAD-based outlier threshold for regression fusion")
    parser.add_argument("--static-sec", type=float, default=STATIC_SEC,
                        help="Seconds from the start of each chunk used for static component removal")
    parser.add_argument("--ignore-sec", type=float, default=IGNORE_SEC,
                        help="Seconds ignored between the static and gesture windows")
    parser.add_argument("--gesture-sec", type=float, default=GESTURE_SEC,
                        help="Seconds used for the output gesture-recognition distance curves")
    parser.add_argument("--force", action="store_true",
                        help="Recompute files even if outputs already exist")
    args = parser.parse_args()

    carrier_freqs = np.array(
        [args.start_freq + args.freq_gap * j for j in range(args.freq_count)],
        dtype=np.float64,
    )

    chunked_files = sorted(
        path for path in args.chunked_root.rglob("*.npz") if is_aggregated_chunk_file(path)
    )

    if not chunked_files:
        print(f"No aggregated .npz files found under {args.chunked_root}")
        return

    processed = 0
    skipped = 0
    chunks_seen = 0

    for input_npz in chunked_files:
        gesture = derive_gesture_dir(input_npz, args.chunked_root)
        output_dir = args.output_root / gesture

        if not args.force and already_processed(input_npz, output_dir):
            print(f"Skipping already processed file: {input_npz}")
            skipped += 1
            continue

        print(f"\nProcessing {input_npz} -> {output_dir}")
        chunks_seen += process_file(
            input_npz=input_npz,
            output_dir=output_dir,
            carrier_freqs=carrier_freqs,
            decimation_factor=args.decimation_factor,
            difference_delay=args.difference_delay,
            stages=args.stages,
            window_ms=args.window_ms,
            min_keep=args.min_keep,
            sigma=args.sigma,
            static_sec=args.static_sec,
            ignore_sec=args.ignore_sec,
            gesture_sec=args.gesture_sec,
            save_iq=SAVE_IQ,
        )
        processed += 1

    print("\nSummary:")
    print(f"  aggregated chunked files found: {len(chunked_files)}")
    print(f"  processed: {processed}")
    print(f"  skipped: {skipped}")
    print(f"  chunks analyzed: {chunks_seen}")


if __name__ == "__main__":
    main()
