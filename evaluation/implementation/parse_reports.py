#!/usr/bin/env python3
"""
Vivado report parser: resource utilisation and timing.

NO Vivado reports exist anywhere in this repository (searched for *.rpt,
*utilization*, *timing_summary*, *.xpr). Nothing here is fabricated: if the
reports are absent, the parser says so and writes REQUIRED_REPORTS.md telling
you exactly which files to drop in. Resource and Fmax numbers MUST come from a
real implementation run before they go in the paper.

Drop reports into evaluation/vivado_reports/ and re-run.

Post-implementation values are preferred for paper claims; post-synthesis is
parsed too but flagged as such, since synthesis estimates routinely differ from
implemented results.

Outputs
    results/implementation_results.csv
    results/implementation_results.tex
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.common import config as C                # noqa: E402

# Vivado utilization tables look like:  | Slice LUTs | 1234 | 0 | 53200 | 2.32 |
_UTIL_ROW = re.compile(
    r"^\s*\|\s*(?P<name>[A-Za-z0-9 _.\-()]+?)\s*\|"
    r"\s*(?P<used>[\d,]+)\s*\|"
    r"(?:\s*(?P<fixed>[\d,]+)\s*\|)?"
    r"(?:\s*(?P<prohib>[\d,]+)\s*\|)?"
    r"\s*(?P<avail>[\d,]+)\s*\|"
    r"\s*(?P<pct>[\d.]+)\s*\|",
    re.MULTILINE,
)

_RESOURCE_KEYS = {
    "slice luts": "LUT",
    "clb luts": "LUT",
    "lut as logic": "LUT_logic",
    "lut as memory": "LUTRAM",
    "slice registers": "FF",
    "clb registers": "FF",
    "register as flip flop": "FF",
    "block ram tile": "BRAM",
    "ramb36/fifo": "BRAM36",
    "ramb18": "BRAM18",
    "dsps": "DSP",
    "dsp48e1 only": "DSP",
    "dsp48e2 only": "DSP",
}

_PATTERNS = {
    "device": re.compile(r"(?:Device|Part)\s*:\s*(\S+)", re.IGNORECASE),
    "wns_ns": re.compile(r"WNS\s*\(ns\)[^\n]*\n[^\n]*?(-?\d+\.\d+)", re.IGNORECASE),
    "tns_ns": re.compile(r"TNS\s*\(ns\)[^\n]*\n[^\n]*?\s(-?\d+\.\d+)", re.IGNORECASE),
    "target_period_ns": re.compile(
        r"(?:Requirement|Target Period)\s*:\s*(\d+\.\d+)ns", re.IGNORECASE),
}


def find_reports(root: Path) -> dict[str, list[Path]]:
    """Locate candidate Vivado reports, split by implementation vs synthesis."""
    impl, synth, other = [], [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".rpt", ".txt", ".log"):
            continue
        low = p.name.lower()
        if "synth" in low:
            synth.append(p)
        elif "impl" in low or "route" in low or "place" in low:
            impl.append(p)
        else:
            other.append(p)
    return {"implementation": impl, "synthesis": synth, "unclassified": other}


def parse_utilization(text: str) -> dict:
    out: dict[str, object] = {}
    for m in _UTIL_ROW.finditer(text):
        name = m.group("name").strip().lower()
        key = _RESOURCE_KEYS.get(name)
        if not key or key in out:
            continue
        out[key] = int(m.group("used").replace(",", ""))
        out[f"{key}_available"] = int(m.group("avail").replace(",", ""))
        out[f"{key}_pct"] = float(m.group("pct"))
    return out


def parse_timing(text: str) -> dict:
    out: dict[str, object] = {}
    for key, rx in _PATTERNS.items():
        m = rx.search(text)
        if m:
            val = m.group(1)
            out[key] = val if key == "device" else float(val)
    if "wns_ns" in out and "target_period_ns" in out:
        tgt = out["target_period_ns"]
        achieved = tgt - out["wns_ns"]
        if achieved > 0:
            out["achieved_period_ns"] = achieved
            out["achieved_fmax_mhz"] = 1000.0 / achieved
        out["target_fmax_mhz"] = 1000.0 / tgt if tgt else float("nan")
    return out


def parse_all(root: Path) -> tuple[dict, list[Path]]:
    found = find_reports(root)
    merged: dict[str, object] = {}
    used: list[Path] = []
    # Implementation last so it overrides synthesis for paper claims.
    for stage in ("synthesis", "unclassified", "implementation"):
        for path in found[stage]:
            text = path.read_text(errors="ignore")
            data = {**parse_utilization(text), **parse_timing(text)}
            if data:
                data["source_stage"] = stage
                data["source_file"] = path.name
                merged.update(data)
                used.append(path)
    return merged, used


REQUIRED_DOC = """# Vivado reports required

No Vivado reports were found in this directory, so
`results/implementation_results.csv` and `.tex` contain no resource or timing
numbers. **Do not put resource or Fmax figures in the paper until these are
supplied** -- they cannot be derived from simulation.

Generate them from your implemented design and copy them here:

```tcl
# after opt_design / place_design / route_design
report_utilization      -file impl_utilization.rpt
report_timing_summary   -file impl_timing_summary.rpt
# optional, for comparison only
open_run synth_1
report_utilization      -file synth_utilization.rpt
```

Required (post-implementation preferred):

| File | Supplies |
|---|---|
| `impl_utilization.rpt` | LUT, LUTRAM, FF, BRAM, DSP counts and percentages |
| `impl_timing_summary.rpt` | WNS, TNS, target period, achieved Fmax |

Name them with `impl` in the filename so the parser prefers them over
synthesis estimates. Then re-run:

```bash
python evaluation/vivado_reports/parse_reports.py
```

## Constraint to state in the paper

The design targets {clock} MHz (period {period:.3f} ns). The clock constraint
used for implementation must match, or the reported WNS is not comparable to
the latency figures in this evaluation.

## Note on which RTL to implement

The evaluation measures `sim/rtl/`, which contains three
correctness fixes not present in `rtl/`. Implementing `rtl/`
would characterise a design that produces incorrect coefficients. See
EVALUATION_REPORT.md.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", type=Path, default=C.VIVADO_REPORT_DIR)
    ap.add_argument("--out", type=Path, default=C.RESULTS_DIR)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    data, used = parse_all(a.reports)

    if not data:
        print("No Vivado reports found -- no resource or timing numbers produced.")
        doc = C.REPO_ROOT / "docs" / "REQUIRED_VIVADO_REPORTS.md"
        doc.write_text(REQUIRED_DOC.format(clock=C.CLOCK_MHZ, period=C.CLOCK_NS))
        (a.out / "implementation_results.csv").write_text(
            "metric,value,source\n"
            "# NO VIVADO REPORTS PRESENT - see evaluation/vivado_reports/"
            "REQUIRED_REPORTS.md,,\n")
        (a.out / "implementation_results.tex").write_text(
            "% No Vivado reports were available.\n"
            "% Supply post-implementation reports and re-run; see\n"
            "% docs/REQUIRED_VIVADO_REPORTS.md\n")
        print(f"  wrote {doc}")
        return

    print(f"Parsed {len(used)} report(s):")
    for p in used:
        print(f"  {p}")

    rows = sorted(data.items())
    with (a.out / "implementation_results.csv").open("w") as fh:
        fh.write("metric,value\n")
        for k, v in rows:
            fh.write(f"{k},{v}\n")

    lines = [r"\begin{tabular}{lr}", r"\hline",
             r"Metric & Value \\", r"\hline"]
    for k, v in rows:
        if k.startswith("source"):
            continue
        val = f"{v:.3f}" if isinstance(v, float) else str(v)
        lines.append(rf"{k.replace('_', ' ')} & {val} \\")
    lines += [r"\hline", r"\end{tabular}"]
    (a.out / "implementation_results.tex").write_text("\n".join(lines) + "\n")

    for k, v in rows:
        print(f"  {k:24s} {v}")


if __name__ == "__main__":
    main()
