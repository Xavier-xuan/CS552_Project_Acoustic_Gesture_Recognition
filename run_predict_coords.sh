#!/usr/bin/env bash
set -e

PYTHON=".venv/bin/python"
DATA="data"
XAVIER="4-24-2026-xavier"
HENRY="4-21-2026-henry_horizontal"
WILLIAM="4-24-2026-william"

if [[ -z "$ANTHROPIC_API_KEY" ]]; then
  echo "Error: ANTHROPIC_API_KEY is not set."
  echo "Run: export ANTHROPIC_API_KEY=your_key_here"
  exit 1
fi

# Step 1: Compute 2D coords for each subject
echo "=== Computing 2D coordinates ==="

for PERSON in "$XAVIER" "$HENRY" "$WILLIAM"; do
  $PYTHON compute_2d_coords_bulk.py \
    --iq-root "$DATA/$PERSON/IQ" \
    --coords-root "$DATA/$PERSON/coords" \
    --force
done

# Step 2: Recognize via Claude Vision
echo ""
echo "=== Recognizing gestures via Claude Vision ==="

$PYTHON recognize_from_coords.py \
  --coords-root \
    "$DATA/$XAVIER/coords" \
    "$DATA/$HENRY/coords" \
    "$DATA/$WILLIAM/coords" \
  --max-chunks 3
