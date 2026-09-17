#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export KOSPI_PREDICTION_LOG="$ROOT_DIR/prediction_log.jsonl"
export KOSPI_MODEL_ADJUSTMENTS="$ROOT_DIR/model_adjustments_v1.json"

exec python -m streamlit run "$ROOT_DIR/kospi_model.py" \
  --server.address 0.0.0.0 \
  --server.port 8505 \
  --server.runOnSave true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
