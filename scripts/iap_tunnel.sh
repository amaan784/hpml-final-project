#!/usr/bin/env bash
# Open an IAP-tunneled SSH session that also forwards localhost:$PORT (Mac)
# to vm:$PORT (vLLM endpoint). Keep this terminal open while benchmarking.
#
# Usage:
#   PORT=8000 bash scripts/iap_tunnel.sh         # foreground, Ctrl-C to close
#   PROJECT=... ZONE=... VM=... bash scripts/iap_tunnel.sh
# Note: was `set -euo pipefail` but `pipefail` is unsupported on dash/busybox sh
# (which is what some Windows shells route to). `-eu` covers our needs here.
set -eu

PROJECT="${PROJECT:-high-perf-ml-487201}"
ZONE="${ZONE:-us-central1-a}"
VM="${VM:-assetopsbench}"
PORT="${PORT:-8000}"
ACCOUNT_FLAG=""
if [ -n "${GCP_ACCOUNT:-}" ]; then
  ACCOUNT_FLAG="--account=${GCP_ACCOUNT}"
fi

echo "Forwarding localhost:${PORT} -> ${VM}:${PORT} via IAP (Ctrl-C to disconnect)"
exec gcloud compute ssh "$VM" \
  --tunnel-through-iap \
  --project="$PROJECT" \
  --zone="$ZONE" \
  $ACCOUNT_FLAG \
  -- -L "${PORT}:localhost:${PORT}" -N
