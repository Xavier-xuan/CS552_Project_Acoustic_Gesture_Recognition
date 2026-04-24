#!/usr/bin/env python3
"""Bulk 2D coordinate computation over all IQ NPZ files.

Mirrors the incremental-processing pattern of cal_distance_bulk.py.
Geometry and delay-profile settings are inherited from compute_2d_coords.py.
"""

from pathlib import Path

from compute_2d_coords import (
    D1_INIT,
    D2_INIT,
    L1,
    L2,
    compute_coords_for_npz,
)

# ── Directory constants ───────────────────────────────────────────────────────
IQ_ROOT     = Path("data/preliminary_data/IQ")
COORDS_ROOT = Path("data/preliminary_data/coords")

# Set True to recompute files that already exist in COORDS_ROOT.
FORCE = False


def already_processed(input_npz: Path) -> bool:
    rel = input_npz.relative_to(IQ_ROOT)
    return (COORDS_ROOT / rel).exists()


def main():
    npz_files = sorted(IQ_ROOT.rglob("*.npz"))
    if not npz_files:
        print(f"No .npz files found under {IQ_ROOT}")
        return

    processed = skipped = total_chunks = 0

    for input_npz in npz_files:
        if not FORCE and already_processed(input_npz):
            print(f"Skipping already processed: {input_npz}")
            skipped += 1
            continue

        rel = input_npz.relative_to(IQ_ROOT)
        output_path = COORDS_ROOT / rel
        print(f"\nProcessing {input_npz}")
        total_chunks += compute_coords_for_npz(
            input_npz, output_path,
            l1=L1, l2=L2, d1_init=D1_INIT, d2_init=D2_INIT,
        )
        processed += 1

    print("\nSummary:")
    print(f"  IQ files found:  {len(npz_files)}")
    print(f"  processed:       {processed}")
    print(f"  skipped:         {skipped}")
    print(f"  chunks computed: {total_chunks}")


if __name__ == "__main__":
    main()
