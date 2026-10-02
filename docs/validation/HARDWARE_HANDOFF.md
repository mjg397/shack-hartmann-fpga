# Hardware Handoff

Status: **BLOCKED on this machine.** No Vivado installation, no `.xpr`/`.dcp`
project, and no utilization/timing report exist anywhere searched (confirmed
independently by `docs/EVALUATION_REPORT.md` §4.9, which searched
`*.rpt`, `*.xpr`, `*.dcp`, `*utilization*`, `*timing_summary*` and any file
mentioning `shwfs`/`ematrix_accumulator` across this machine and all git
branches). This document is the exact handoff so a board/tool session can
pick this up without re-deriving context.

## Prerequisite: reconcile rtl/ before synthesizing

Per `IMPLEMENTATION_AUDIT.md`, `rtl/` as committed produces ~100% error
on every Zernike mode due to 3 documented defects (Q4.23-vs-signed-25 overflow
in the reference slope ROMs, a Q6-vs-Q18 width mismatch in `unsigned_mult`, and
a 3-cycle-vs-1-cycle reciprocal latency mismatch — full detail and fixes in
`docs/EVALUATION_REPORT.md` §1.1). **Port those 3 fixes into
`rtl/` before running synthesis**, or the timing/resource numbers will
describe a design that doesn't compute the right answer.

## Commands to run once Vivado + a target part are available

```tcl
# after implementation completes, from the Vivado project or Tcl console:
report_utilization    -file evaluation/implementation/impl_utilization.rpt
report_timing_summary  -file evaluation/implementation/impl_timing_summary.rpt
```

`evaluation/implementation/parse_reports.py` already expects exactly these
two filenames in that directory and will populate
`evaluation/results/implementation_results.{csv,tex}` automatically — no
further glue code needed. Re-run `evaluation/run_all.py` afterward to fold
the results into the master tables/figures.

## What board-side capture still needs to be built

Not present in the repo yet — needed for validation-program §8:

- An ILA/SignalTap (or on-chip counter) capture harness recording: input pixel
  acceptance edge, frame markers, output-valid edge, and a per-frame cycle
  counter. `sim/` already has the testbench-side
  equivalent (`tb_full_pipeline.v`, `tb_latency_trace.v`) — the on-chip
  version should reuse the same event definitions (see stage-anchor table in
  `docs/EVALUATION_REPORT.md` §4.6) so simulation and board numbers are comparable
  anchor-for-anchor.
- A frame-replay path that streams stored `.mem`/`.hex` frames (real or
  synthetic) into the actual input interface at the actual clock, rather than
  through the Icarus testbench driver.
- A sustained multi-frame stress test on hardware (§8 of the validation
  program) — the Icarus `tb_full_pipeline.v` equivalent exists in simulation
  only.

## Definitions to fix before any board number is reported

Use the same three anchors already defined and measured in simulation
(`docs/EVALUATION_REPORT.md` §4.6), so a board measurement is directly comparable:

| Anchor | Simulated cycles | Simulated ns @ 150 MHz |
|---|---|---|
| Streaming-emulator output bus (recommended) | 9 | 60.00 |
| Accumulator `s1` input register | 8 | 53.33 |
| Accumulator completion event | 7 | 46.67 |

Any published board latency figure must name which of these (or a new,
explicitly defined) anchor it uses — per the report's own warning, figures
using different anchors are not comparable.

## Do not do without the user

Per the validation program's own constraints and this session's standing
instructions: do not purchase tools/boards, do not contact vendors, do not
publish or submit anything, and do not claim Fmax/utilization/power from
anything other than an actual generated report once one exists.
