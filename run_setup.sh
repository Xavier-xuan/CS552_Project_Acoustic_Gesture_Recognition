set -e

DATA="data"
XAVIER="4-24-2026-xavier"
HENRY="4-21-2026-henry_horizontal"
WILLIAM="4-24-2026-william"
WILLIAM2="4-30-2026-2-william"
TRACKING_1D="4-27-2026-henry"
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

uv run chunk_by_chirp_bulk.py \
  --raw-root "$DATA/$WILLIAM/raw" \
  --chunked-root "$DATA/$WILLIAM/chunked" \
  --force

uv run chunk_by_chirp_bulk.py \
  --raw-root "$DATA/$WILLIAM2/raw" \
  --chunked-root "$DATA/$WILLIAM2/chunked" \
  --force

uv run chunk_by_chirp_bulk.py \
  --raw-root "$DATA/$TRACKING_1D/raw" \
  --chunked-root "$DATA/$TRACKING_1D/chunked" \
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

uv run cal_distance_bulk.py \
  --chunked-root "$DATA/$WILLIAM/chunked" \
  --output-root "$DATA/$WILLIAM/IQ" \
  --force

uv run cal_distance_bulk.py \
  --chunked-root "$DATA/$WILLIAM2/chunked" \
  --output-root "$DATA/$WILLIAM2/IQ" \
  --force

uv run cal_distance_bulk.py \
  --chunked-root "$DATA/$TRACKING_1D/chunked" \
  --output-root "$DATA/$TRACKING_1D/IQ" \
  --force
