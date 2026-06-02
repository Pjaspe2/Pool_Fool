#!/usr/bin/env bash
# Run from macOS Terminal (needs Camera permission for Terminal.app).
#
# Env vars:
#   WIDE=1     — capture at 640x480 + min zoom (often widest FOV)
#   REGION=... — full | half_near | half_far | center (partial table in frame)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -e ".[dev]" -q
fi

CONFIG="${CONFIG:-config/default.yaml}"
SNAP="${ROOT}/config/calibration/snapshot.jpg"
WIDE="${WIDE:-1}"
REGION="${REGION:-}"

CAP_ARGS=(--config "$CONFIG" --camera 0 --output "$SNAP")
TABLE_ARGS=(--config "$CONFIG" --image "$SNAP")
[[ "$WIDE" == "1" ]] && CAP_ARGS+=(--wide) && TABLE_ARGS+=(--wide)
[[ -n "$REGION" ]] && TABLE_ARGS+=(--region "$REGION")

echo "=== Pool Fool table calibration (Mac) ==="
echo ""
echo "Software options if the whole table does not fit:"
echo "  1) WIDE=1 (default) — widest camera mode (640x480 + zoom out)"
echo "  2) REGION=half_near  — calibrate only the visible half (camera at foot end)"
echo "  3) pool-fool-calibrate probe — list what resolutions your cam actually gives"
echo ""
echo "Mounting higher/wider lens still helps; partial cal is fine for testing."
echo ""

if ! .venv/bin/pool-fool-calibrate capture "${CAP_ARGS[@]}"; then
  echo ""
  echo "=== Camera failed — running doctor ==="
  .venv/bin/pool-fool-calibrate doctor || true
  echo ""
  echo "Workaround (no live camera): take a photo of the table, save as snapshot.jpg, then:"
  echo "  pool-fool-calibrate table --config $CONFIG --image config/calibration/snapshot.jpg"
  exit 1
fi
.venv/bin/pool-fool-calibrate table "${TABLE_ARGS[@]}"

echo ""
echo "Done. Test with:"
echo "  .venv/bin/pool-fool-app --config $CONFIG --camera 0"
