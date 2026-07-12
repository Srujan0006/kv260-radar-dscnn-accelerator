#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOLUTION="$ROOT/hardware/hls/radar_cnn_hls_accel/v1_frame_buffered/radar_cnn_hls_prj/solution1"
DEST="$ROOT/hardware/hls/radar_cnn_hls_accel/reports/synthesis"
mkdir -p "$DEST"

copy_if_present() {
  local source="$1"
  local destination="$2"
  if [[ -f "$source" ]]; then
    cp "$source" "$destination"
    echo "Collected: $destination"
  fi
}

copy_if_present \
  "$SOLUTION/syn/report/radar_cnn_accel_csynth.rpt" \
  "$DEST/radar_cnn_accel_csynth.rpt"
copy_if_present \
  "$SOLUTION/sim/report/radar_cnn_accel_csim.log" \
  "$DEST/radar_cnn_accel_csim.log"
copy_if_present \
  "$SOLUTION/impl/report/verilog/radar_cnn_accel_export.rpt" \
  "$DEST/radar_cnn_accel_export.rpt"

if [[ -d "$SOLUTION/impl/ip" ]]; then
  cp -R "$SOLUTION/impl/ip" "$DEST/exported_ip"
  echo "Collected: $DEST/exported_ip"
fi

if [[ ! -f "$DEST/radar_cnn_accel_csynth.rpt" ]]; then
  echo "No synthesis report found. Run vitis-run --mode hls --tcl run_hls.tcl first (Vitis 2025.1)." >&2
  exit 1
fi
