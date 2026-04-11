import argparse
from pathlib import Path
import numpy as np

#!/usr/bin/env python3



def read_pcm(path: Path, dtype: np.dtype, channels: int):
    raw = np.fromfile(path, dtype=dtype)
    if channels <= 0:
        raise ValueError("channels must be >= 1")
    if raw.size % channels != 0:
        raw = raw[: raw.size - (raw.size % channels)]
    data = raw.reshape(-1, channels)
    mono = data.mean(axis=1).astype(np.float32)
    return data, mono


def compute_band_energy(mono, sr, low_hz, high_hz, frame_size=4096, hop=512):
    """Compute per-frame energy in [low_hz, high_hz] band."""
    if len(mono) < frame_size:
        return np.zeros(0, dtype=np.float32), frame_size, hop

    window = np.hanning(frame_size).astype(np.float32)
    n_frames = 1 + (len(mono) - frame_size) // hop
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sr)
    band = (freqs >= low_hz) & (freqs <= high_hz)

    energies = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        frame = mono[s : s + frame_size] * window
        spec = np.fft.rfft(frame)
        energies[i] = np.abs(spec[band]).mean()

    return energies, frame_size, hop


def energy_to_intervals(energies, threshold, hop, frame_size, total_samples,
                        min_on_frames=3, max_gap_frames=2):
    """Threshold energy curve, smooth, and return sample-level intervals."""
    mask = energies >= threshold

    # Fill short OFF gaps between ON regions
    off_idx = np.where(~mask)[0]
    if off_idx.size > 0:
        runs, start, prev = [], off_idx[0], off_idx[0]
        for x in off_idx[1:]:
            if x == prev + 1:
                prev = x
            else:
                runs.append((start, prev)); start = x; prev = x
        runs.append((start, prev))
        for a, b in runs:
            if (b - a + 1) <= max_gap_frames:
                if (a > 0 and mask[a - 1]) and (b < mask.size - 1 and mask[b + 1]):
                    mask[a : b + 1] = True

    # Remove short ON runs
    on_idx = np.where(mask)[0]
    if on_idx.size > 0:
        runs, start, prev = [], on_idx[0], on_idx[0]
        for x in on_idx[1:]:
            if x == prev + 1:
                prev = x
            else:
                runs.append((start, prev)); start = x; prev = x
        runs.append((start, prev))
        for a, b in runs:
            if (b - a + 1) < min_on_frames:
                mask[a : b + 1] = False

    # Convert frame mask to sample intervals
    on_idx = np.where(mask)[0]
    if on_idx.size == 0:
        return []
    intervals = []
    start = on_idx[0]; prev = on_idx[0]
    for idx in on_idx[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            s = start * hop
            e = prev * hop + frame_size
            intervals.append((max(0, s), min(total_samples, e)))
            start = idx; prev = idx
    s = start * hop
    e = prev * hop + frame_size
    intervals.append((max(0, s), min(total_samples, e)))
    return intervals


def find_chirp_boundaries(mono, sr, chirp_freq=440.0, chirp_band=100.0,
                          frame_size=4096, hop=512, threshold_percentile=90.0,
                          min_chirp_sec=0.3, merge_gap_sec=0.2):
    """Find sample intervals where a chirp tone is present."""
    energies, fs, hp = compute_band_energy(
        mono, sr, chirp_freq - chirp_band, chirp_freq + chirp_band, frame_size, hop
    )
    if energies.size == 0:
        return []
    thr = np.percentile(energies, threshold_percentile)
    min_on = max(1, int(0.1 * sr / hop))   # chirp must be >= ~100ms
    max_gap = max(1, int(0.05 * sr / hop))  # bridge tiny gaps
    intervals = energy_to_intervals(energies, thr, hop, fs, len(mono), min_on, max_gap)

    # Merge chirps that are very close together
    merge_gap_samples = int(merge_gap_sec * sr)
    merged = []
    for s, e in intervals:
        if merged and s - merged[-1][1] <= merge_gap_samples:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # Filter out chirps shorter than min_chirp_sec
    min_samples = int(min_chirp_sec * sr)
    return [(s, e) for s, e in merged if (e - s) >= min_samples]


def extract_segments_after_chirps(chirp_intervals, total_samples, segment_samples):
    """For each chirp, return the interval starting at chirp-end for segment_samples."""
    segments = []
    for _, chirp_end in chirp_intervals:
        seg_start = chirp_end
        seg_end = min(seg_start + segment_samples, total_samples)
        if seg_end > seg_start:
            segments.append((seg_start, seg_end))
    return segments


def main():
    p = argparse.ArgumentParser(
        description="Chunk PCM: detect 440Hz chirp dividers, export the 17-23kHz segments that follow."
    )
    p.add_argument("input_pcm", type=Path, help="Input raw PCM file")
    p.add_argument("output_dir", type=Path, help="Output folder for chunked PCM files")
    p.add_argument("--sample-rate", type=int, required=True, help="Sample rate (Hz)")
    p.add_argument("--channels", type=int, default=1, help="Number of channels")
    p.add_argument(
        "--dtype",
        choices=["int16", "int32", "float32"],
        default="int16",
        help="PCM sample type",
    )
    p.add_argument("--chirp-freq", type=float, default=440.0,
                   help="Chirp divider frequency (Hz)")
    p.add_argument("--chirp-band", type=float, default=100.0,
                   help="+/- band around chirp frequency (Hz)")
    p.add_argument("--chunk-sec", type=float, default=2.0,
                   help="Length of segment to extract after each chirp (s)")
    p.add_argument("--threshold-percentile", type=float, default=90.0,
                   help="Percentile threshold for chirp detection")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dtype = np.dtype(args.dtype)
    full_data, mono = read_pcm(args.input_pcm, dtype=dtype, channels=args.channels)

    chirps = find_chirp_boundaries(
        mono, args.sample_rate,
        chirp_freq=args.chirp_freq,
        chirp_band=args.chirp_band,
        threshold_percentile=args.threshold_percentile,
    )
    if not chirps:
        print("No 440Hz chirp dividers found.")
        return

    print(f"Found {len(chirps)} chirp(s):")
    for i, (s, e) in enumerate(chirps, 1):
        dur = (e - s) / args.sample_rate
        print(f"  chirp {i}: {s/args.sample_rate:.3f}s – {e/args.sample_rate:.3f}s  ({dur:.3f}s)")

    segment_samples = int(args.chunk_sec * args.sample_rate)
    segments = extract_segments_after_chirps(chirps, full_data.shape[0], segment_samples)

    if not segments:
        print("No segments to extract after chirps.")
        return

    stem = args.input_pcm.stem
    for i, (s, e) in enumerate(segments, 1):
        chunk = full_data[s:e]
        out_path = args.output_dir / f"{stem}_chunk_{i:04d}.pcm"
        chunk.astype(dtype, copy=False).reshape(-1).tofile(out_path)
        dur = (e - s) / args.sample_rate
        print(f"  chunk {i}: {s/args.sample_rate:.3f}s – {e/args.sample_rate:.3f}s  ({dur:.2f}s)")

    print(f"Exported {len(segments)} chunks to: {args.output_dir}")


if __name__ == "__main__":
    main()