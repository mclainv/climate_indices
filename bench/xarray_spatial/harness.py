"""xarray-spatial 0.10.19 KBDI harness. Follows ../PROTOCOL.md.

xrspatial's KBDI is a single-day update on a 2-D raster with a *scalar*
``annual_precip``; there is no time loop, so the harness drives it over every
day, feeding each day's result back as ``kbdi_prev``. Our workloads have a
per-cell mean annual precipitation, which the library cannot take, so cells are
grouped by identical MAP (``kernel``/``api``); each group is one 2-D raster of
shape ``(1, group_size)``. For station workloads that is a single 1x1 raster.

Variants (single-threaded, numpy-backed; the library computes in float32):

- ``kernel``: ``xrspatial.fire._kbdi_cpu`` (``@ngjit``) per group per day on
  float32 arrays prepared before timing; records the float32 history.
- ``api``: ``xrspatial.kbdi`` per group per day on float64 DataArrays (the
  function validates and casts to float32 internally and wraps the result).
- ``kernel_raster`` / ``api_raster``: the intended raster use. The whole grid
  is one ``(R, C)`` raster per day (20x20 for 400 cells, 50x50 for 2500) with
  a single scalar ``annual_precip`` = mean of the per-cell MAP. This does NOT
  reproduce the per-cell MAP inputs, so its values differ from ``kernel``;
  use it only to see raster throughput.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pybench  # noqa: E402  (sets numba env before numba is imported)

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

warnings.filterwarnings("ignore")


def raster_shape(cells: int) -> tuple[int, int]:
    r = int(round(cells ** 0.5))
    return (r, r) if r * r == cells else (1, cells)


def main() -> None:
    args = pybench.parse_args(["kernel", "api", "kernel_raster", "api_raster"])
    cells, days, a = pybench.load_workload(
        args.workload, ["precip_mm.bin", "tmax_c.bin", "map_mm.bin"]
    )
    precip, tmax, pa = a["precip_mm.bin"], a["tmax_c.bin"], a["map_mm.bin"]
    workload = args.workload.name

    import numba
    import xrspatial
    from xrspatial.fire import _kbdi_cpu

    use_raster = args.variant.endswith("_raster")
    use_api = args.variant.startswith("api")
    if use_raster:
        shp = raster_shape(cells)
        groups = [(np.arange(cells), shp, float(pa.mean()))]
    else:
        uniq, inv = np.unique(pa, return_inverse=True)
        groups = [
            (np.flatnonzero(inv == k), (1, int((inv == k).sum())), float(m))
            for k, m in enumerate(uniq)
        ]

    # per group: (idx, shape, MAP, per-day inputs)
    prepared = []
    for idx, shp, m in groups:
        p = precip[idx].T.reshape(days, *shp)  # (day, rows, cols)
        t = tmax[idx].T.reshape(days, *shp)
        if use_api:
            ps = [xr.DataArray(np.ascontiguousarray(p[d])) for d in range(days)]
            ts = [xr.DataArray(np.ascontiguousarray(t[d])) for d in range(days)]
            q0 = xr.DataArray(np.zeros(shp, dtype="f4"))
        else:
            ps = [np.ascontiguousarray(p[d], dtype="f4") for d in range(days)]
            ts = [np.ascontiguousarray(t[d], dtype="f4") for d in range(days)]
            q0 = np.zeros(shp, dtype="f4")
        prepared.append((idx, shp, m, ps, ts, q0))

    kbdi = xrspatial.kbdi

    def call():
        hist = []
        for idx, shp, m, ps, ts, q0 in prepared:
            h = np.empty((days, *shp), dtype="f4")
            q = q0
            if use_api:
                for d in range(days):
                    q = kbdi(q, ts[d], ps[d], m)
                    h[d] = q.data
            else:
                for d in range(days):
                    q = _kbdi_cpu(q, ts[d], ps[d], m)
                    h[d] = q
            hist.append(h)
        return hist

    def checksum(hist):
        return float(sum(h.sum(dtype="f8") for h in hist))

    samples, first_ns, csum, last = pybench.run_protocol(
        call, checksum, cells, args.quick
    )
    out = np.empty((cells, days))
    for (idx, shp, *_), h in zip(prepared, last):
        out[idx] = h.reshape(days, -1).T
    notes = (
        f"xarray-spatial {xrspatial.__version__}, numba {numba.__version__}, "
        f"NUMBA_NUM_THREADS={numba.config.NUMBA_NUM_THREADS}, "
        f"NUMBA_CPU_NAME={numba.config.CPU_NAME or 'host'} (oryon-1 LLVM workaround); "
        f"{len(prepared)} raster group(s), single-day step called {days} times per group; "
        f"float32 compute, 800-scale non-standard algorithm (NOT scientifically "
        f"comparable); output values are the library's own units (written as-is)"
    )
    pybench.write_outputs(args, "xarray_spatial", workload, cells, days, samples,
                          first_ns, csum, notes, out)


if __name__ == "__main__":
    main()
