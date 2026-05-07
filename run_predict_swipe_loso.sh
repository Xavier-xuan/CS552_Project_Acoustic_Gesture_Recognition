#!/usr/bin/env bash
set -e

DATA="data"
XAVIER="4-24-2026-xavier"
HENRY="4-21-2026-henry_horizontal"
WILLIAM="4-30-2026-2-william"
OUT="$DATA/testing"
GESTURES="LeftSwipeHorizontal RightSwipeHorizontal"

uv run chunk_by_chirp_bulk.py \
  --raw-root "$DATA/$WILLIAM/raw" \
  --chunked-root "$DATA/$WILLIAM/chunked" 

# Step 2: compute IQ features
uv run cal_distance_bulk.py \
  --chunked-root "$DATA/$WILLIAM/chunked" \
  --output-root "$DATA/$WILLIAM/IQ" 


# leave-one-subject-out prediction
echo "=== Swipe only — Test: Xavier (train: Henry + William) ==="
uv run predict_gestures_nopca.py \
  --iq-root-training  "$DATA/$HENRY/IQ" \
  --iq-root-training2 "$DATA/$WILLIAM/IQ" \
  --iq-root-eval      "$DATA/$XAVIER/IQ" \
  --gestures $GESTURES \
  --output "$OUT/predict_swipe_loso-xavier_nopca.png"

echo "=== Swipe only — Test: Henry (train: Xavier + William) ==="
uv run predict_gestures_nopca.py \
  --iq-root-training  "$DATA/$XAVIER/IQ" \
  --iq-root-training2 "$DATA/$WILLIAM/IQ" \
  --iq-root-eval      "$DATA/$HENRY/IQ" \
  --gestures $GESTURES \
  --output "$OUT/predict_swipe_loso-henry_nopca.png"

echo "=== Swipe only — Test: William (train: Xavier + Henry) ==="
uv run predict_gestures_nopca.py \
  --iq-root-training  "$DATA/$XAVIER/IQ" \
  --iq-root-training2 "$DATA/$HENRY/IQ" \
  --iq-root-eval      "$DATA/$WILLIAM/IQ" \
  --gestures $GESTURES \
  --output "$OUT/predict_swipe_loso-william_nopca.png"
