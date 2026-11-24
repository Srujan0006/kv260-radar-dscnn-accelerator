#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <input_quantized_xmodel> <output_dir>"
  echo "Example: $0 outputs/vai_quant/Radar2DCNN_int.xmodel outputs/compiled_kv260"
  exit 1
fi

INPUT_XMODEL="$1"
OUT_DIR="$2"
NET_NAME="${NET_NAME:-radar_2dcnn}"

if [ ! -f "$INPUT_XMODEL" ]; then
  echo "ERROR: input xmodel not found: $INPUT_XMODEL"
  exit 2
fi

mkdir -p "$OUT_DIR"

# Common KV260 arch.json locations across Vitis AI releases.
if [ -z "${ARCH_JSON:-}" ]; then
  CANDIDATES=(
    "/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"
    "/opt/vitis_ai/compiler/arch/DPUCZDX8G/kv260/arch.json"
    "/opt/vitis_ai/compiler/arch/DPUCZDX8G/K26/arch.json"
    "/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json"
  )
  for c in "${CANDIDATES[@]}"; do
    if [ -f "$c" ]; then
      ARCH_JSON="$c"
      break
    fi
  done
fi

if [ -z "${ARCH_JSON:-}" ] || [ ! -f "$ARCH_JSON" ]; then
  echo "ERROR: Could not find arch.json. Set ARCH_JSON manually."
  echo "Example: ARCH_JSON=/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json $0 $INPUT_XMODEL $OUT_DIR"
  exit 3
fi

echo "Input xmodel : $INPUT_XMODEL"
echo "Arch JSON    : $ARCH_JSON"
echo "Output dir   : $OUT_DIR"
echo "Net name     : $NET_NAME"

vai_c_xir \
  -x "$INPUT_XMODEL" \
  -a "$ARCH_JSON" \
  -o "$OUT_DIR" \
  -n "$NET_NAME"

echo "Compile complete. Check: $OUT_DIR/${NET_NAME}.xmodel"
