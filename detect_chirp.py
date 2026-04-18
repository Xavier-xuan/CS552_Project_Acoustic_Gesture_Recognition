#!/usr/bin/env python3
"""Chunk PCM audio by detecting chirps via cross-correlation with a reference chirp.

Each detected chirp is followed by a 2-second chunk that is exported.
"""

import argparse
from importlib import reload

from pathlib import Path

import numpy as np
import utils
reload(utils)


def find_peaks_numpy(signal, height=None, distance=1):
    signal = np.asarray(signal)

    # Find local maxima
    peaks = np.where(
        (signal[1:-1] > signal[:-2]) &
        (signal[1:-1] > signal[2:])
    )[0] + 1

    # Apply height threshold
    if height is not None:
        peaks = peaks[signal[peaks] >= height]

    # Apply minimum distance
    if distance > 1 and len(peaks) > 0:
        filtered = [peaks[0]]
        for p in peaks[1:]:
            if p - filtered[-1] >= distance:
                filtered.append(p)
        peaks = np.array(filtered)

    properties = {"peak_heights": signal[peaks]}
    return peaks, properties

def detect_chirps(mono: np.ndarray, ref_chirp: np.ndarray, sample_rate: int,
                  peak_threshold: float = 0.3, min_distance_sec: float = 1.0):
    """Find chirp locations via normalized cross-correlation.

    Returns a list of sample indices where each chirp starts.
    """
    # Normalize both signals
    ref = ref_chirp / (np.linalg.norm(ref_chirp) + 1e-12)
    sig = mono / (np.max(np.abs(mono)) + 1e-12)

    # Cross-correlate (use 'valid' so output indices map directly to signal) using numpy
    corr = np.correlate(sig, ref, mode='valid') / len(ref) 
    corr = np.abs(corr)

    # Normalize correlation to [0, 1]
    corr /= (np.max(corr) + 1e-12)

    # Find peaks
    min_distance_samples = int(min_distance_sec * sample_rate)
    # Use numpy to find peaks
    peak_indices, properties = find_peaks_numpy(corr, height=peak_threshold, distance=min_distance_samples)
# (
#         corr, height=peak_threshold, distance=min_distance_samples
#     )
    return peak_indices.tolist()


def main():
    p = argparse.ArgumentParser(
        description="Detect chirps via cross-correlation and extract 2s chunks after each."
    )
    p.add_argument("input_pcm", type=Path, help="Input raw PCM file")
    p.add_argument("output_dir", type=Path, help="Output folder for chunked PCM files")
    p.add_argument("--sample-rate", type=int, default=48000, help="Sample rate (Hz)")
    p.add_argument("--channels", type=int, default=2, help="Number of channels")
    p.add_argument("--dtype", choices=["int16", "int32", "float32"], default="int16",
                   help="PCM sample type")
    p.add_argument("--chunk-sec", type=float, default=2.0,
                   help="Length of segment to extract after each chirp (s)")
    p.add_argument("--threshold", type=float, default=0.6,
                   help="Normalized correlation peak threshold (0-1)")
    p.add_argument("--chirp-delay", type=float, default=0.5,
                   help="Delay after chirp before extracting chunk (s)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(args.dtype)

    # Load audio
    full_data, left, right = utils.read_pcm_stereo(args.input_pcm, dtype=dtype, channels=args.channels)
    print(f"Loaded {args.input_pcm}: {full_data.shape[0]} samples, "
          f"{full_data.shape[0] / args.sample_rate:.2f}s")

    # Generate reference chirp and detect
    chirp_path = "data/chirp.pcm"
    sample_rate = 48000   # Hz
    channels = 2          # set to 2 for stereo PCM
    dtype = np.int16      # common PCM format

    # Load raw PCM
    _, ref_chirp = utils.read_pcm_mono(chirp_path, dtype=dtype, channels=channels)
    
    print(f"Loaded reference chirp: {ref_chirp.shape[0]} samples, "
          f"{ref_chirp.shape[0] / sample_rate:.2f}s")
    
    for ch_idx, (ch_data, ch_label) in enumerate([(left, "L"), (right, "R")]):
        chirp_starts = detect_chirps(ch_data, ref_chirp, args.sample_rate,
                                    peak_threshold=args.threshold)

        if not chirp_starts:
            print("No chirps detected.")
            return

        chirp_len = ref_chirp.shape[0]
        chunk_len = int(args.chunk_sec * args.sample_rate)
        total_samples = full_data.shape[0]

        print(f"Found {len(chirp_starts)} chirp(s):")
        stem = args.input_pcm.stem
        exported = 0

        for i, cs in enumerate(chirp_starts, 1):
            chirp_end = cs + chirp_len
            seg_start = chirp_end + int(args.chirp_delay * args.sample_rate)
            seg_end = min(seg_start + chunk_len, total_samples)

            print(f"  chirp {i}: {cs / args.sample_rate:.3f}s, "
                f"chunk: {seg_start / args.sample_rate:.3f}s – {seg_end / args.sample_rate:.3f}s "
                f"({(seg_end - seg_start) / args.sample_rate:.2f}s)")

            if seg_end <= seg_start:
                continue

            chunk = full_data[seg_start:seg_end]
            out_path = args.output_dir / f"{stem}_chunk_{ch_label}_{i:04d}.pcm"
            chunk.astype(dtype, copy=False).reshape(-1).tofile(out_path)
            exported += 1

        print(f"Exported {exported} chunk(s) to: {args.output_dir}")


if __name__ == "__main__":
    main()
