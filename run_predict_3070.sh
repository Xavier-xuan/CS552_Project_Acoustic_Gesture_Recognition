#!/usr/bin/env bash
set -e

DATA="data"
XAVIER="4-24-2026-xavier"
HENRY="4-21-2026-henry_horizontal"
WILLIAM="4-24-2026-william"
OUT="$DATA/testing"

echo "=== 30/70 split: all three subjects ==="
uv run predict_gestures_nopca.py \
  --iq-root-training  "$DATA/$XAVIER/IQ" \
  --iq-root-training2 "$DATA/$HENRY/IQ" \
  --iq-root-training3 "$DATA/$WILLIAM/IQ" \
  --test-size 0.7 \
  --output "$OUT/predict_3070_all_nopca.png"
