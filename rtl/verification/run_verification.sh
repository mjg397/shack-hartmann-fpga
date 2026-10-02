#!/usr/bin/env bash
#
# SHWFS hardware verification, in simulation.
#
#   ./run_verification.sh              lint + directed + randomised
#   ./run_verification.sh -r 32        more randomised frames
#   ./run_verification.sh --mutate     also prove the suite catches injected bugs
#   ./run_verification.sh --lint-only  Verilator lint only
#
# Simulator: Icarus Verilog (the environment is written to also elaborate under
# Verilator). cocotb and UVM are not used -- neither is installed, and neither
# Icarus nor Verilator supports UVM.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RTL="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$RTL/.." && pwd)"
WORK="$HERE/work"
PY="$REPO/sim/venv/bin/python"

N_RANDOM=8
SEED=1
DO_MUTATE=0
LINT_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    -r|--random) N_RANDOM="$2"; shift 2;;
    -s|--seed)   SEED="$2"; shift 2;;
    --mutate)    DO_MUTATE=1; shift;;
    --lint-only) LINT_ONLY=1; shift;;
    *) echo "unknown option: $1"; exit 2;;
  esac
done

DUT_SRC=(
  "$RTL/shwfs_processing.v"
  "$RTL/Thresholding_CoG_intensity_accumulator.v"
  "$RTL/intensity_reciprocal_q18.v"
  "$RTL/reciprocal_u16_q18.v"
  "$RTL/slope_calculation.v"
  "$RTL/ematrix_accumulator.v"
)
TB_SRC=("$HERE/tb/tb_shwfs.sv" "$HERE/tb/shwfs_ref_model.sv"
        "$HERE/tb/shwfs_scoreboard.sv" "$HERE/tb/shwfs_assertions.sv")

# ---------------------------------------------------------------- lint -------
echo "==> Verilator lint"
if command -v verilator >/dev/null 2>&1; then
  # Width warnings here are int-literal style plus the three documented port
  # mismatches; they are reported, not treated as failures.
  mkdir -p "$WORK"
  LINT_LOG="$WORK/verilator_lint.log"
  set +e
  verilator --lint-only -Wall -Wno-fatal --top-module shwfs_processing \
    "${DUT_SRC[@]}" >"$LINT_LOG" 2>&1
  LINT_STATUS=$?
  set -e
  grep -E '^%(Error|Warning)' "$LINT_LOG" | sed 's/%//' | awk '{print $1}' \
    | sort | uniq -c | sort -rn | sed 's/^/    /' || true
  if [ "$LINT_STATUS" -ne 0 ]; then
    cat "$LINT_LOG"
    exit "$LINT_STATUS"
  fi
else
  echo "    verilator not installed -- skipped"
fi
[ "$LINT_ONLY" -eq 1 ] && exit 0

# ------------------------------------------------------------- stimulus ------
echo "==> generating directed frames"
rm -rf "$WORK"; mkdir -p "$WORK"
cp "$RTL"/*.mem "$WORK/"
if [ -x "$PY" ]; then
  "$PY" "$HERE/gen_directed.py" --out "$WORK" | tail -1
else
  echo "    venv missing -- directed frames skipped, randomised only"
fi
# The repository's own demonstration frame, first in the list.
[ -f "$WORK/hcipy_image_rotating.mem" ] && \
  { echo "hcipy_image_rotating.mem"; cat "$WORK/directed.f" 2>/dev/null; } \
    > "$WORK/tests.f" || cp "$WORK/directed.f" "$WORK/tests.f"

# ------------------------------------------------------------ simulate -------
echo "==> compiling testbench"
iverilog -g2012 -o "$WORK/tb.vvp" -s tb_shwfs "${TB_SRC[@]}" "${DUT_SRC[@]}" \
  2>&1 | grep -viE 'extra digits|warning: Port' || true

echo "==> running"
( cd "$WORK" && vvp tb.vvp +TESTS=tests.f +RANDOM="$N_RANDOM" +SEED="$SEED" )
RESULT=$?

# -------------------------------------------------------------- mutate -------
if [ "$DO_MUTATE" -eq 1 ]; then
  echo ""
  echo "==> mutation testing (does the suite actually catch bugs?)"
  MUT="$WORK/mut"; mkdir -p "$MUT"; cp "$RTL"/*.v "$MUT/"
  mutate_run () {
    iverilog -g2012 -o "$WORK/mut.vvp" -s tb_shwfs "${TB_SRC[@]}" \
      "$MUT/shwfs_processing.v" "$MUT/Thresholding_CoG_intensity_accumulator.v" \
      "$MUT/intensity_reciprocal_q18.v" "$MUT/reciprocal_u16_q18.v" \
      "$MUT/slope_calculation.v" "$MUT/ematrix_accumulator.v" 2>/dev/null
    v=$( cd "$WORK" && vvp mut.vvp +RANDOM=2 +SEED=3 2>&1 | grep -oE 'RESULT: (PASS|FAIL)' | tail -1 )
    printf "    %-44s %s\n" "$1" "${v:-no result}"
  }
  mutate_run "M0 unmutated (must PASS)"
  sed -i '' 's|assign out = mult_out\[23:0\];|assign out = mult_out[35:12];|' "$MUT/slope_calculation.v"
  mutate_run "M1 centroid slice reverted (must FAIL)"
  cp "$RTL/slope_calculation.v" "$MUT/"
  sed -i '' 's|RECIP_LATENCY = 2|RECIP_LATENCY = 1|' "$MUT/intensity_reciprocal_q18.v"
  mutate_run "M2 reciprocal misaligned (must FAIL)"
  cp "$RTL/intensity_reciprocal_q18.v" "$MUT/"
  sed -i '' 's|acc_next\[i\]\[42:18\]|acc_next[i][43:19]|' "$MUT/ematrix_accumulator.v"
  mutate_run "M3 zernike slice off-by-one (must FAIL)"
  cp "$RTL/ematrix_accumulator.v" "$MUT/"
  sed -i '' "s|PIXEL_THRESHOLD             = 8'd5|PIXEL_THRESHOLD             = 8'd6|" \
    "$MUT/Thresholding_CoG_intensity_accumulator.v"
  mutate_run "M4 pixel threshold changed (must FAIL)"
fi

echo ""
echo "work directory (frames, vvp, logs): $WORK"
exit $RESULT
