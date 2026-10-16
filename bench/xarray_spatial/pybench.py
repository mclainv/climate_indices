"""Shared plumbing for the Python harnesses (args, numba env, timing, output).

Follows ../PROTOCOL.md. Copied verbatim into each Python competitor directory so
that each directory is self-contained.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# --- numba environment: must be set before numba is imported -----------------
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
# llvmlite 0.50's LLVM aborts ("DefIdx 1 exceeds machine model writes ... LDPXi",
# TargetSchedule.cpp) when JIT-compiling most non-trivial functions for the host
# CPU "oryon-1" (Snapdragon X Elite), because that CPU's scheduling model is
# incomplete. Work around it by targeting the generic AArch64 CPU with the host's
# real ISA features. The code is still native ARM64 (NEON/FP); only the
# instruction-scheduling tuning differs.
_ORYON_FEATURES = (
    "+aes,+bf16,+crc,+dotprod,+fullfp16,+i8mm,+jsconv,+lse,+rcpc,+sha2"
)
if os.environ.get("KBDI_NO_CPU_WORKAROUND") != "1":
    os.environ.setdefault("NUMBA_CPU_NAME", "generic")
    os.environ.setdefault("NUMBA_CPU_FEATURES", _ORYON_FEATURES)


def parse_args(variants: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, type=Path)
    ap.add_argument("--variant", default="kernel", choices=variants)
    ap.add_argument("--timings", type=Path, default=Path("out/timings.jsonl"))
    ap.add_argument("--values", type=Path)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="smoke test only: 2 warm-up calls, 3+ samples, ~0.3 s (not protocol timing)",
    )
    return ap.parse_args()


def load_workload(path: Path, names: list[str]):
    meta = json.loads((path / "meta.json").read_text())
    cells, days = int(meta["cells"]), int(meta["days"])
    arrays = {}
    for n in names:
        a = np.fromfile(path / n, dtype="<f8")
        arrays[n] = a.reshape(cells, days) if a.size == cells * days else a
    return cells, days, arrays


def run_protocol(call, checksum, cells: int, quick: bool):
    """Warm up, then sample. Returns (samples_ns, first_call_ns, checksum, last)."""
    t0 = time.perf_counter_ns()
    last = call()
    first_call_ns = time.perf_counter_ns() - t0
    min_calls, min_warm_s = (2, 0.1) if quick else (5, 0.5)
    warm = time.perf_counter()
    n = 1
    while n < min_calls or time.perf_counter() - warm < min_warm_s:
        last = call()
        n += 1
    if quick:
        min_samples, min_s, max_s = 3, 0.3, 20.0
    else:
        min_samples = 200 if cells == 1 else 15
        min_s, max_s = 3.0, 60.0
    samples: list[int] = []
    csum = 0.0
    started = time.perf_counter()
    while True:
        t = time.perf_counter_ns()
        last = call()
        samples.append(time.perf_counter_ns() - t)
        csum = checksum(last)
        el = time.perf_counter() - started
        if (len(samples) >= min_samples and el >= min_s) or (
            el >= max_s and len(samples) >= 3
        ):
            break
    return samples, first_call_ns, csum, last


def write_outputs(args, impl, workload, cells, days, samples, first_call_ns,
                  csum, notes, values_cm):
    """`values_cm`: float64 [cell][day] array in mm, or None."""
    if args.values is not None:
        args.values.parent.mkdir(parents=True, exist_ok=True)
        np.ascontiguousarray(values_cm, dtype="<f8").tofile(args.values)
    line = {
        "impl": impl, "variant": args.variant, "workload": workload,
        "cells": cells, "days": days, "threads": 1, "samples_ns": samples,
        "first_call_ns": first_call_ns, "checksum": csum, "notes": notes,
    }
    args.timings.parent.mkdir(parents=True, exist_ok=True)
    with open(args.timings, "a") as f:
        f.write(json.dumps(line) + "\n")
    s = sorted(samples)
    print(
        f"{impl} {args.variant} {workload}: median {s[len(s)//2]/1e6:.4f} ms, "
        f"min {s[0]/1e6:.4f} ms, n={len(s)}, first {first_call_ns/1e6:.1f} ms, "
        f"checksum {csum}",
        file=sys.stderr,
    )
