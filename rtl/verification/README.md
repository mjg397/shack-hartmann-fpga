# SHWFS hardware verification

Verifies the RTL against an independent functional model, in simulation, with
directed and randomised tests, assertions and a scoreboard.

```bash
./run_verification.sh                # lint + 18 directed + 8 randomised
./run_verification.sh -r 64 -s 42    # more randomised frames, chosen seed
./run_verification.sh --mutate       # also prove the suite catches injected bugs
./run_verification.sh --lint-only    # Verilator lint only
```

Exit code is non-zero on any failure, so it drops straight into CI.

## What this is, and what it is not

It is a layered, UVM-*shaped* environment written in plain SystemVerilog
modules. It is **not** UVM, and does not use cocotb — neither Icarus nor
Verilator supports UVM, and cocotb is not installed. Everything here runs on
what is actually available:

| Tool | Version | Role |
|---|---|---|
| Icarus Verilog | installed | simulation |
| Verilator | 5.050 | lint |
| cocotb | **not installed** | — |

## Architecture

```
  driver ──> shwfs_processing (DUT) ──> monitor ──┐
     │                                            ├──> scoreboard ──> verdict
     └──────> shwfs_ref_model ────────────────────┘
                                   assertions bound to DUT internals
```

| Component | File | Role |
|---|---|---|
| Driver | `tb/tb_shwfs.sv` | streams pixels straight into the DUT with camera line framing. Deliberately does **not** use `bram_model`/`streaming_emulator`, so stimulus is under test control and randomised frames can be built at run time. |
| Functional model | `tb/shwfs_ref_model.sv` | **independent** behavioural implementation of the fixed-point spec — no pipelining, no handshaking, whole-frame in one pass. Written from the specification, not from the RTL, so a bug in one is not mirrored by the other. |
| Scoreboard | `tb/shwfs_scoreboard.sv` | exact comparison of every centroid, slope and Zernike coefficient. Both sides implement the same fixed-point arithmetic, so any difference is a real discrepancy — never rounding. |
| Assertions | `tb/shwfs_assertions.sv` | protocol and range invariants that hold for *any* data: completion counters increment by exactly one, accumulators stay inside their documented bounds, no `X` on qualified outputs, centroids inside their own subaperture, `done` only after all 168 subapertures. |

Assertions are immediate checks inside clocked blocks rather than SVA
`assert property`, because Icarus does not support concurrent assertions. The
same source therefore runs under both simulators.

## Tests

**Directed** (`gen_directed.py`, physically meaningful — built from the
repository's own optical forward model):
zero slope · near-zero slope (sign boundary) · positive slope · negative slope ·
large in-tile offset · each of the 10 pure Zernike modes · dim spots just above
threshold · saturated spots · dark frame (exercises the `sI == 0` reciprocal
path) · plus the repository's own `hcipy_image_rotating.mem`.

**Randomised** (generated in SystemVerilog from a seed): random spot position
and amplitude per subaperture. Deliberately harsher than the optical model —
it reaches intensity and position corners real wavefronts never produce, which
is what the arithmetic needs stressing against. Every randomised frame is
written to `work/rand_N.mem` before use, so a failure is exactly reproducible.

Each frame yields **682 checks**: 168 subapertures × 4 (centroid x/y, slope x/y)
plus 10 coefficients.

## Does the suite actually catch bugs?

A suite that only ever passes proves nothing, so `--mutate` injects known
faults and requires each to be caught:

| Mutation | Expected | Observed signature |
|---|---|---|
| M0 unmutated | PASS | — |
| M1 centroid slice reverted to `[35:12]` | FAIL | 672 centroid mismatches |
| M2 reciprocal misaligned (`RECIP_LATENCY` 2→1) | FAIL | 662 centroid mismatches **+ 184 assertion failures** |
| M3 Zernike output slice off by one | FAIL | **exactly 10** mismatches, centroids clean |
| M4 pixel threshold 5→6 | FAIL | 192 centroid mismatches |

M3 is the useful demonstration: a back-end-only fault appears as exactly ten
Zernike mismatches with zero centroid errors, so the layering localises the bug
rather than just flagging one.

## Current status

23 tests, **15,686 checks, 0 mismatches, 0 assertion failures.**

## Where the fixed-point analysis lives now

This folder previously held five Python design studies built around a **Q27 +
Newton** reciprocal. That design was dropped because its 27-bit output overflows
`unsigned_mult`'s 18-bit `b` port — the DSP48E1 B-port width — so it no longer
maps to a single DSP. The scripts formed one dependency cluster rooted in that
model and were removed with it.

Their function is covered, and improved on, by the evaluation suite:

| Was | Now |
|---|---|
| floating-point pipeline reference | `evaluation/common/bitaccurate.run_float_reference()` |
| fixed-point pipeline model | `evaluation/common/bitaccurate.run_pipeline()` — verified **bit-exact** against the RTL |
| reciprocal precision study | `evaluation/error_ablation/ablation.py` — 300-case ablation isolating each quantisation stage |

`docs/RECIPROCAL_PRECISION_PROOF.md` is retained as a historical record of the
Q27 rationale. Its < 1e-4 px target applies to a design that is no longer
implementable here; see `docs/EVALUATION_REPORT.md` §4.3.
