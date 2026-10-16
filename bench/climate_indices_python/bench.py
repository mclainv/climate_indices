#!/usr/bin/env python3
"""climate_indices KBDI through its Python entry points, per ``../PROTOCOL.md``.

Variants:

- ``api``: ``climate_indices.fire.kbdi`` on time-first NumPy arrays, the
  default configuration a user gets (Rust dispatch, INFO logging to stderr).
- ``api_quiet``: the same call with ``CLIMATE_INDICES_LOG_LEVEL=WARNING``,
  so the two structlog events per call are filtered out.
- ``api_xarray``: the beta xarray path on ``(time, location)`` DataArrays with
  CF units attributes (quiet logging).
- ``native``: the extension function ``climate_indices._native.kbdi`` itself,
  called with the arrays ``fire._native`` would prepare: the PyO3 boundary plus
  the Rust kernel, without the Python validation and dispatch layer.
- ``python``: ``climate_indices.fire.kbdi`` with the Rust dispatch disabled
  (``tests/conftest.disable_native``): the pure-Python reference (quiet logging).
- ``api_dask``: ``api_xarray`` on Dask-backed DataArrays, ``location`` chunked
  into ``--threads`` blocks with one ``time`` chunk (chunking outside the timed
  call); the timed call runs ``.compute()`` on Dask's threaded scheduler with
  ``--threads`` workers, exactly as a user would. Dask's worker threads start
  with NumPy's default floating-point policy (``warn``), so the chunks may stay
  on the Python path; the notes record how many chunks reached Rust.
- ``api_dask_ignore``: ``api_dask`` with the worker threads' NumPy policy set
  to ``ignore`` (a ``ThreadPoolExecutor`` initializer passed as Dask's pool),
  the setting that lets each chunk dispatch to Rust.

Run with the native ARM64 venv from the repository root, e.g.::

    python benchmarks/kbdi_competitors/climate_indices_python/bench.py \
        --workload benchmarks/kbdi_competitors/data/station_1825 --variant api \
        --timings benchmarks/kbdi_competitors/out/timings.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument(
        "--variant",
        required=True,
        choices=["api", "api_quiet", "api_xarray", "api_dask", "api_dask_ignore", "native", "python"],
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--values", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse()
    if args.variant != "api":
        os.environ["CLIMATE_INDICES_LOG_LEVEL"] = "WARNING"
    import numpy as np
    import xarray as xr

    import climate_indices
    from climate_indices import fire
    from climate_indices.fire import _native as fire_native

    meta = json.loads((args.workload / "meta.json").read_text())
    cells, days = meta["cells"], meta["days"]

    def load(stem: str) -> np.ndarray:
        return np.fromfile(args.workload / f"{stem}.bin", dtype="<f8")

    # time-first, as the API documents; built once, outside timing
    precip = np.ascontiguousarray(load("precip_mm").reshape(cells, days).T)
    tmax = np.ascontiguousarray(load("tmax_c").reshape(cells, days).T)
    map_mm = load("map_mm")
    if cells == 1:
        precip, tmax, map_scalar = precip[:, 0].copy(), tmax[:, 0].copy(), float(map_mm[0])
    else:
        map_scalar = None
    map_arg = map_scalar if cells == 1 else map_mm

    patch = None
    threads = 1
    native_chunks: list[bool] = []
    if args.variant == "api" or args.variant == "api_quiet":
        call = lambda: fire.kbdi(precip, tmax, map_arg)  # noqa: E731
    elif args.variant == "api_xarray":
        dims = ("time",) if cells == 1 else ("time", "location")
        p_da = xr.DataArray(precip, dims=dims, attrs={"units": "mm"})
        t_da = xr.DataArray(tmax, dims=dims, attrs={"units": "degC"})
        m_da = map_scalar if cells == 1 else xr.DataArray(map_mm, dims=("location",), attrs={"units": "mm"})
        call = lambda: fire.kbdi(p_da, t_da, m_da).values  # noqa: E731
    elif args.variant in ("api_dask", "api_dask_ignore"):
        from concurrent.futures import ThreadPoolExecutor

        import dask

        from climate_indices.fire import _kbdi

        threads = max(1, min(args.threads, cells))
        size = -(-cells // threads)
        p_da = xr.DataArray(precip, dims=("time", "location"), attrs={"units": "mm"}).chunk({"time": -1, "location": size})
        t_da = xr.DataArray(tmax, dims=("time", "location"), attrs={"units": "degC"}).chunk({"time": -1, "location": size})
        m_da = xr.DataArray(map_mm, dims=("location",), attrs={"units": "mm"}).chunk({"location": size})
        # record, per chunk, whether the Rust kernel took it
        dispatch = _kbdi.kbdi_recurrence

        def spy(*a, **k):
            result = dispatch(*a, **k)
            native_chunks.append(result is not None)
            return result

        _kbdi.kbdi_recurrence = spy
        if args.variant == "api_dask":
            options = {"scheduler": "threads", "num_workers": threads}
        else:
            pool = ThreadPoolExecutor(threads, initializer=lambda: np.seterr(all="ignore"))
            options = {"scheduler": "threads", "pool": pool}
        call = lambda: dask.compute(fire.kbdi(p_da, t_da, m_da).data, **options)[0]  # noqa: E731
    elif args.variant == "native":
        native = climate_indices._native
        p2 = precip.reshape(days, cells)
        t2 = tmax.reshape(days, cells)
        m1 = np.ascontiguousarray(map_mm)
        zeros = np.zeros(cells)
        weather_valid = np.isfinite(p2) & np.isfinite(t2)
        static_valid = np.ones(cells, dtype=bool)
        gaps = np.full(cells, -1, dtype=np.int64)

        def call():
            return native.kbdi(
                p2, t2, m1, zeros, zeros, weather_valid, static_valid, None, gaps, 0, "propagate", 0, True
            )[0]
    else:
        sys.path.insert(0, str(ROOT / "tests"))
        import pytest

        import conftest

        patch = pytest.MonkeyPatch()
        conftest.disable_native(patch)
        assert fire_native._native is None
        call = lambda: fire.kbdi(precip, tmax, map_arg)  # noqa: E731

    # the guard that lets a call reach the extension (benchmarks/rust_vs_python.native_policy)
    warnings.resetwarnings()
    warnings.simplefilter("ignore", RuntimeWarning)
    np.seterr(all="ignore")

    def checksum(out) -> float:
        return float(np.sum(out))

    t0 = time.perf_counter_ns()
    out = call()
    first_call_ns = time.perf_counter_ns() - t0
    warm_start = time.perf_counter()
    warm_calls = 1
    while warm_calls < 5 or time.perf_counter() - warm_start < 0.5:
        out = call()
        warm_calls += 1

    first_native = list(native_chunks)
    min_samples = 200 if cells == 1 else 15
    samples: list[int] = []
    started = time.perf_counter()
    total = 0.0
    while True:
        t0 = time.perf_counter_ns()
        out = call()
        samples.append(time.perf_counter_ns() - t0)
        total = checksum(out)
        elapsed = time.perf_counter() - started
        if (len(samples) >= min_samples and elapsed >= 3.0) or (elapsed >= 60.0 and len(samples) >= 3):
            break
    dispatched = fire_native._native is not None
    if patch is not None:
        patch.undo()

    if args.values:
        args.values.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(out, dtype="<f8").reshape(days, cells)
        np.ascontiguousarray(arr.T).tofile(args.values)

    line = {
        "impl": "climate_indices_python",
        "variant": args.variant,
        "workload": args.workload.name,
        "cells": cells,
        "days": days,
        "threads": threads,
        "samples_ns": samples,
        "first_call_ns": first_call_ns,
        "checksum": total,
        "notes": f"climate_indices {climate_indices.__version__}; rust dispatch available: {dispatched}; "
        f"numpy {np.__version__}; python {sys.version.split()[0]}"
        + (f"; chunks on Rust during warm-up: {sum(first_native)}/{len(first_native)}" if first_native else ""),
    }
    args.timings.parent.mkdir(parents=True, exist_ok=True)
    with args.timings.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line) + "\n")
    s = sorted(samples)
    print(
        f"{args.variant} {args.workload.name}: median {s[len(s) // 2] / 1e6:.4f} ms, min {s[0] / 1e6:.4f} ms, "
        f"n={len(s)}, checksum {total}",
        file=sys.stderr if args.variant != "api" else sys.stdout,
    )


if __name__ == "__main__":
    main()
