#!/usr/bin/env bash
set -e

DATA="data"
XAVIER="4-24-2026-xavier"
HENRY="4-21-2026-henry_horizontal"
WILLIAM="4-24-2026-william"
OUT="$DATA/testing"

# Step 1: chunk raw pcm
uv run chunk_by_chirp_bulk.py \
  --raw-root "$DATA/$XAVIER/raw" \
  --chunked-root "$DATA/$XAVIER/chunked" \
  --force

uv run chunk_by_chirp_bulk.py \
  --raw-root "$DATA/$HENRY/raw" \
  --chunked-root "$DATA/$HENRY/chunked" \
  --force

# Step 2: compute IQ features
uv run cal_distance_bulk.py \
  --chunked-root "$DATA/$XAVIER/chunked" \
  --output-root "$DATA/$XAVIER/IQ" \
  --force

uv run cal_distance_bulk.py \
  --chunked-root "$DATA/$HENRY/chunked" \
  --output-root "$DATA/$HENRY/IQ" \
  --force

# Step 3: leave-one-subject-out prediction
echo "=== Test: Xavier (train: Henry + William) ==="
uv run predict_gestures.py \
  --iq-root-training  "$DATA/$HENRY/IQ" \
  --iq-root-training2 "$DATA/$WILLIAM/IQ" \
  --iq-root-eval      "$DATA/$XAVIER/IQ" \
  --output "$OUT/predict_eval-xavier_train-henry+william.png"

echo "=== Test: Henry (train: Xavier + William) ==="
uv run predict_gestures.py \
  --iq-root-training  "$DATA/$XAVIER/IQ" \
  --iq-root-training2 "$DATA/$WILLIAM/IQ" \
  --iq-root-eval      "$DATA/$HENRY/IQ" \
  --output "$OUT/predict_eval-henry_train-xavier+william.png"

echo "=== Test: William (train: Xavier + Henry) ==="
uv run predict_gestures.py \
  --iq-root-training  "$DATA/$XAVIER/IQ" \
  --iq-root-training2 "$DATA/$HENRY/IQ" \
  --iq-root-eval      "$DATA/$WILLIAM/IQ" \
  --output "$OUT/predict_eval-william_train-xavier+henry.png"
