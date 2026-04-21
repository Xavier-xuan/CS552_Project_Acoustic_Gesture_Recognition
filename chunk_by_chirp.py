#!/usr/bin/env python3
"""Chunk PCM audio by detecting chirps and exporting one aggregated NPZ per input."""

import argparse
from importlib import reload

from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, fftconvolve
import utils
reload(utils)

DEFAULT_CHUNK_SEC = 5.0


def parse_subject_and_gesture(input_pcm: Path):
    stem_parts = input_pcm.stem.split("-")
    subject = stem_parts[1] if len(stem_parts) >= 2 else "unknown"

    if input_pcm.parent.name:
        gesture = input_pcm.parent.name
    elif len(stem_parts) >= 1 and "_" in stem_parts[0]:
        gesture = stem_parts[0].split("_")[-1]
    else:
        gesture = "unknown"

    return subject, gesture


def detect_chirps(mono: np.ndarray, ref_chirp: np.ndarray, sample_rate: int,
                  peak_threshold: float = 0.3, min_distance_sec: float = 1.0,
                  prominence: float = 0.5):
    """Find chirp locations via normalized cross-correlation.

    Returns a list of sample indices where each chirp starts.
    """
    # Normalize both signals
    ref = ref_chirp / (np.linalg.norm(ref_chirp) + 1e-12)
    sig = mono / (np.max(np.abs(mono)) + 1e-12)

    # Cross-correlate via FFT (equivalent to np.correlate 'valid', orders of magnitude faster)
    corr = fftconvolve(sig, ref[::-1], mode='valid') / len(ref)
    corr = np.abs(corr)

    # Normalize correlation to [0, 1]
    corr /= (np.max(corr) + 1e-12)

    # Find peaks
    min_distance_samples = int(min_distance_sec * sample_rate)
    # Use numpy to find peaks
    peak_indices, _ = find_peaks(
        corr,
        height=peak_threshold,
        distance=min_distance_samples,
        prominence=prominence,
    )
    return peak_indices.tolist()


def load_reference_chirp(chirp_path: Path, dtype: np.dtype, channels: int, sample_rate: int):
    _, ref_chirp = utils.read_pcm_mono(chirp_path, dtype=dtype, channels=channels)
    print(f"Loaded reference chirp: {ref_chirp.shape[0]} samples, "
          f"{ref_chirp.shape[0] / sample_rate:.2f}s")
    return ref_chirp


def chunk_file(input_pcm: Path, output_dir: Path, sample_rate: int, channels: int,
               dtype: np.dtype, chunk_sec: float, peak_threshold: float,
               chirp_delay: float, prominence: float, chirp_path: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    full_data, left, right = utils.read_pcm_stereo(input_pcm, dtype=dtype, channels=channels)
    print(f"Loaded {input_pcm}: {full_data.shape[0]} samples, "
          f"{full_data.shape[0] / sample_rate:.2f}s")

    ref_chirp = load_reference_chirp(chirp_path, dtype=dtype, channels=channels, sample_rate=sample_rate)
    chirp_len = ref_chirp.shape[0]
    chunk_len = int(chunk_sec * sample_rate)
    total_samples = full_data.shape[0]
    stem = input_pcm.stem
    subject, gesture = parse_subject_and_gesture(input_pcm)
    chunk_audio = []
    chunk_channels = []
    chunk_indices = []
    chirp_starts_all = []
    chirp_ends_all = []
    chunk_starts_all = []
    chunk_ends_all = []

    for ch_data, ch_label in [(left, "L"), (right, "R")]:
        chirp_starts = detect_chirps(
            ch_data,
            ref_chirp,
            sample_rate,
            peak_threshold=peak_threshold,
            min_distance_sec=1.0,
            prominence=prominence,
        )

        if not chirp_starts:
            print(f"No chirps detected for channel {ch_label}.")
            continue

        print(f"Found {len(chirp_starts)} chirp(s) on channel {ch_label}:")
        for i, cs in enumerate(chirp_starts, 1):
            chirp_end = cs + chirp_len
            seg_start = chirp_end + int(chirp_delay * sample_rate)
            seg_end = min(seg_start + chunk_len, total_samples)

            print(f"  chirp {i}: {cs / sample_rate:.3f}s, "
                  f"chunk: {seg_start / sample_rate:.3f}s – {seg_end / sample_rate:.3f}s "
                  f"({(seg_end - seg_start) / sample_rate:.2f}s)")

            if seg_end <= seg_start:
                continue

            if seg_end - seg_start < chunk_len:
                print(f"  skipping chirp {i} on channel {ch_label}: incomplete chunk at file end")
                continue

            chunk_audio.append(full_data[seg_start:seg_end].astype(dtype, copy=False))
            chunk_channels.append(ch_label)
            chunk_indices.append(i)
            chirp_starts_all.append(cs)
            chirp_ends_all.append(chirp_end)
            chunk_starts_all.append(seg_start)
            chunk_ends_all.append(seg_end)

    exported = len(chunk_audio)
    out_path = output_dir / f"{stem}.npz"

    if exported == 0:
        print(f"Exported 0 chunk(s) to: {output_dir}")
        return 0

    np.savez_compressed(
        out_path,
        audio=np.stack(chunk_audio, axis=0),
        sample_rate=np.array(sample_rate, dtype=np.int32),
        channels=np.array(channels, dtype=np.int32),
        dtype=np.array(dtype.name),
        source=np.array(str(input_pcm)),
        subject=np.array(subject),
        gesture=np.array(gesture),
        chunk_channels=np.array(chunk_channels),
        chunk_indices=np.array(chunk_indices, dtype=np.int32),
        chirp_starts=np.array(chirp_starts_all, dtype=np.int32),
        chirp_ends=np.array(chirp_ends_all, dtype=np.int32),
        chunk_starts=np.array(chunk_starts_all, dtype=np.int32),
        chunk_ends=np.array(chunk_ends_all, dtype=np.int32),
    )

    print(f"Exported {exported} chunk(s) to: {out_path}")
    return exported


def main():
    p = argparse.ArgumentParser(
        description="Detect chirps via cross-correlation and export one NPZ per input file."
    )
    p.add_argument("input_pcm", type=Path, help="Input raw PCM file")
    p.add_argument("output_dir", type=Path, help="Output folder for aggregated NPZ files")
    p.add_argument("--sample-rate", type=int, default=48000, help="Sample rate (Hz)")
    p.add_argument("--channels", type=int, default=2, help="Number of channels")
    p.add_argument("--dtype", choices=["int16", "int32", "float32"], default="int16",
                   help="PCM sample type")
    p.add_argument("--chunk-sec", type=float, default=DEFAULT_CHUNK_SEC,
                   help="Length of segment to extract after each chirp (s)")
    p.add_argument("--threshold", type=float, default=0.6,
                   help="Normalized correlation peak threshold (0-1)")
    p.add_argument("--chirp-delay", type=float, default=0.5,
                   help="Delay after chirp before extracting chunk (s)")
    p.add_argument("--prominence", type=float, default=0.5,
                   help="Minimum normalized prominence for a detected peak (0-1)")
    p.add_argument("--chirp-path", type=Path, default=Path("data") / "chirp.pcm",
                   help="Reference chirp PCM path")
    args = p.parse_args()

    dtype = np.dtype(args.dtype)
    chunk_file(
        input_pcm=args.input_pcm,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        channels=args.channels,
        dtype=dtype,
        chunk_sec=args.chunk_sec,
        peak_threshold=args.threshold,
        chirp_delay=args.chirp_delay,
        prominence=args.prominence,
        chirp_path=args.chirp_path,
    )


if __name__ == "__main__":
    main()
