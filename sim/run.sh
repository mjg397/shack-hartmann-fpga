#!/usr/bin/env bash
#
# Full Zynq SHWFS pipeline simulation + HCIPy golden-model comparison.
#
#   ./run.sh            build, simulate, compare
#   ./run.sh --ref      also regenerate the HCIPy reference (slow, ~1 min)
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RTL="$HERE/../rtl"   # single source of truth: the synthesised tree
BUILD="$HERE/build"
PY="$HERE/venv/bin/python"

[ -x "$PY" ] || { echo "venv missing -- python3 -m venv venv && venv/bin/pip install hcipy numpy scipy matplotlib"; exit 1; }

mkdir -p "$BUILD/waves"

# Gen B chain only. The Gen A files (shwfs_vivado_top, shwfs_vivado_ila_top,
# TCoG_intensity_accumulator, intensity_reciprocal) were a previous generation,
# inconsistent with the current
# slope_calculation/ematrix_accumulator widths, and are deliberately excluded.
SOURCES=(
  "$RTL/shwfs_processing.v"
  "$RTL/Thresholding_CoG_intensity_accumulator.v"
  "$RTL/intensity_reciprocal_q18.v"
  "$RTL/reciprocal_u16_q18.v"
  "$RTL/slope_calculation.v"
  "$RTL/ematrix_accumulator.v"
  "$RTL/streaming_emulator.v"
  "$HERE/bram_model.v"
  "$HERE/tb_zynq_top.v"
  "$HERE/tb_full_pipeline.v"
)

echo "==> packing image"
"$PY" "$HERE/pack_image.py"


echo "==> staging .mem files"
cp "$RTL"/*.mem "$BUILD/"

echo "==> compiling"
iverilog -g2012 -o "$BUILD/sim.vvp" -s tb_full_pipeline "${SOURCES[@]}"

echo "==> simulating"
( cd "$BUILD" && vvp sim.vvp )

if [ "${1:-}" = "--ref" ] || [ ! -f "$BUILD/hcipy_reference.txt" ]; then
  echo "==> generating HCIPy reference"
  "$PY" "$HERE/gen_reference.py"
fi

echo "==> comparing"
"$PY" "$HERE/compare_hcipy.py"

echo "==> diagnostics"
"$PY" "$HERE/diagnose.py"
