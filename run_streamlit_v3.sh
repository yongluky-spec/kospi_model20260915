#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export KOSPI_PREDICTION_LOG="$ROOT_DIR/prediction_log_v3.jsonl"
export KOSPI_MODEL_ADJUSTMENTS="$ROOT_DIR/model_adjustments_v3.json"

echo "[INFO] Running v3 engine on port 8506. Existing v1/v2 JSONL logs are append-only and preserved."

exec python -m streamlit run "$ROOT_DIR/kospi_engine.py" \
  --server.address 0.0.0.0 \
  --server.port 8506 \
  --server.runOnSave true \
  --server.enableCORS false \
  --server.enableXsrfProtection false