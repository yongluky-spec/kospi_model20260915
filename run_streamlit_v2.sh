#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

exec python -m streamlit run "$ROOT_DIR/kospi_engine.py" \
  --server.address 0.0.0.0 \
  --server.port 8506 \
  --server.runOnSave true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
