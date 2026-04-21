#!/usr/bin/env python3
"""Bulk incremental chunking for raw PCM gesture files into aggregated NPZ files."""

import argparse
from pathlib import Path

import numpy as np

from chunk_by_chirp import DEFAULT_CHUNK_SEC, chunk_file


def derive_gesture_dir(input_pcm: Path, raw_root: Path) -> str:
    try:
        return input_pcm.parent.relative_to(raw_root).parts[0]
    except Exception:
        return "unknown"


def already_chunked(input_pcm: Path, output_dir: Path) -> bool:
    stem = input_pcm.stem
    return (output_dir / f"{stem}.npz").exists()


def main():
    parser = argparse.ArgumentParser(
        description="Incrementally chunk all raw PCM files into one NPZ per source file."
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data") / "preliminary_data" / "raw",
                        help="Root directory containing raw gesture PCM files")
    parser.add_argument("--chunked-root", type=Path, default=Path("data") / "preliminary_data" / "chunked",
                        help="Root directory for generated chunked NPZ files")
    parser.add_argument("--chirp-path", type=Path, default=Path("data") / "chirp.pcm",
                        help="Reference chirp PCM path")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Sample rate (Hz)")
    parser.add_argument("--channels", type=int, default=2, help="Number of channels")
    parser.add_argument("--dtype", choices=["int16", "int32", "float32"], default="int16",
                        help="PCM sample type")
    parser.add_argument("--chunk-sec", type=float, default=DEFAULT_CHUNK_SEC,
                        help="Length of segment to extract after each chirp (s)")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Normalized correlation peak threshold (0-1)")
    parser.add_argument("--chirp-delay", type=float, default=0.5,
                        help="Delay after chirp before extracting chunk (s)")
    parser.add_argument("--prominence", type=float, default=0.5,
                        help="Minimum normalized prominence for a detected peak (0-1)")
    parser.add_argument("--force", action="store_true",
                        help="Re-chunk files even if output chunks already exist")
    args = parser.parse_args()

    raw_files = sorted(args.raw_root.rglob("*.pcm"))
    dtype = np.dtype(args.dtype)

    if not raw_files:
        print(f"No raw .pcm files found under {args.raw_root}")
        return

    processed = 0
    skipped = 0
    exported = 0

    for input_pcm in raw_files:
        gesture = derive_gesture_dir(input_pcm, args.raw_root)
        output_dir = args.chunked_root / gesture

        if not args.force and already_chunked(input_pcm, output_dir):
            print(f"Skipping already chunked file: {input_pcm}")
            skipped += 1
            continue

        print(f"\nProcessing {input_pcm} -> {output_dir}")
        exported += chunk_file(
            input_pcm=input_pcm,
            output_dir=output_dir,
            sample_rate=args.sample_rate,
            channels=args.channels,
            dtype=dtype,
            chunk_sec=args.chunk_sec,
            peak_threshold=args.threshold,
            chirp_delay=args.chirp_delay,
            prominence=args.prominence,
            chirp_path=args.chirp_path,
        )
        processed += 1

    print("\nSummary:")
    print(f"  raw files found: {len(raw_files)}")
    print(f"  processed: {processed}")
    print(f"  skipped: {skipped}")
    print(f"  chunks exported: {exported}")


if __name__ == "__main__":
    main()
