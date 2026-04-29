#!/usr/bin/env python3
"""Bulk 2D coordinate computation over all IQ NPZ files.

Mirrors the incremental-processing pattern of cal_distance_bulk.py.
Geometry and delay-profile settings are inherited from compute_2d_coords.py.
"""

import argparse
from pathlib import Path

from compute_2d_coords import (
    D1_INIT,
    D2_INIT,
    L1,
    L2,
    compute_coords_for_npz,
)


def main():
    parser = argparse.ArgumentParser(description="Bulk 2D coordinate computation from IQ NPZ files.")
    parser.add_argument("--iq-root", type=Path, nargs="+",
                        default=[Path("data/preliminary_data/IQ")])
    parser.add_argument("--coords-root", type=Path,
                        default=Path("data/preliminary_data/coords"))
    parser.add_argument("--force", action="store_true",
                        help="Recompute files that already exist in coords-root")
    args = parser.parse_args()

    iq_roots = args.iq_root
    npz_files = sorted(p for root in iq_roots for p in root.rglob("*.npz"))

    if not npz_files:
        print(f"No .npz files found under {iq_roots}")
        return

    processed = skipped = total_chunks = 0

    for input_npz in npz_files:
        # Preserve subdirectory structure under coords-root using the IQ root it came from
        matched_root = next(r for r in iq_roots if input_npz.is_relative_to(r))
        rel = input_npz.relative_to(matched_root.parent)
        output_path = args.coords_root / rel

        if not args.force and output_path.exists():
            print(f"Skipping already processed: {input_npz}")
            skipped += 1
            continue

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
