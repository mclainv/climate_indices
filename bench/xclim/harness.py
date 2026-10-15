"""xclim 0.62.0 KBDI harness. Follows ../PROTOCOL.md.

Variants (float64; all single-threaded and numpy-backed except ``api_dask``):

- ``kernel``: ``xclim.indices.fire._ffdi._keetch_byram_drought_index`` (numba
  ``@guvectorize``, signature ``(n),(n),(),()->(n)``, default target ``cpu`` =
  serial) on C-contiguous ``(cell, time)`` numpy arrays; ``kbdi0`` is a zeros
  ``(cell,)`` array and ``pa`` the per-cell mean annual precipitation (mm).
- ``api``: ``xclim.indices.fire.keetch_byram_drought_index`` on ``(location,
  time)`` DataArrays with CF units (``mm/d``, ``degC``, ``mm/year``); includes
  its unit conversion, ``xr.full_like`` for kbdi0, ``xr.apply_ufunc`` and
  ``declare_units`` checks.
- ``indicator``: ``xclim.atmos.keetch_byram_drought_index`` (the FireWeather
  Indicator wrapper: CF/metadata checks, missing-value handling ``skip``,
  history/attrs formatting) on the same DataArrays.
- ``kernel_threads``: the same unmodified ``kernel`` gufunc, called on
  ``--threads`` contiguous row blocks of the ``(cell, time)`` arrays from a
  ``ThreadPoolExecutor`` (the gufunc releases the GIL). Counterpart of our
  ``kernel_mt``; not an xclim-provided entry point, just the parallel wrapper
  a user would write around the kernel.
- ``api_dask``: ``api`` on the same DataArrays chunked along ``location`` into
  ``--threads`` equal blocks (chunking done once, outside the timed call, as
  if the data were opened with ``chunks=``); the timed call builds the graph
  and runs ``.compute()`` on dask's threaded scheduler with ``--threads``
  workers. ``apply_ufunc(dask="parallelized")`` maps the serial gufunc over
  the chunks.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pybench  # noqa: E402  (sets numba env before numba is imported)

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

warnings.filterwarnings("ignore", message="Import")


def main() -> None:
    args = pybench.parse_args(["kernel", "kernel_threads", "api", "indicator", "api_dask"])
    cells, days, a = pybench.load_workload(
        args.workload, ["precip_mm.bin", "tmax_c.bin", "map_mm.bin"]
    )
    precip, tmax, pa = a["precip_mm.bin"], a["tmax_c.bin"], a["map_mm.bin"]
    workload = args.workload.name

    import numba
    import xclim
    from xclim.indices.fire import _ffdi

    if args.variant == "kernel":
        gu = _ffdi._keetch_byram_drought_index
        kbdi0 = np.zeros(cells)

        def call():
            # kbdi0 is passed as a scalar core arg, so it is not modified
            return gu(precip, tmax, pa, kbdi0)

        def checksum(out):
            return float(out.sum())

        note_extra = "guvectorized kernel on numpy (cell,time); target=cpu (serial)"
    elif args.variant == "kernel_threads":
        from concurrent.futures import ThreadPoolExecutor

        gu = _ffdi._keetch_byram_drought_index
        kbdi0 = np.zeros(cells)
        chunk = -(-cells // args.threads)
        blocks = [slice(i, i + chunk) for i in range(0, cells, chunk)]
        pool = ThreadPoolExecutor(args.threads)

        def run_block(b):
            return gu(precip[b], tmax[b], pa[b], kbdi0[b])

        def call():
            # per-block outputs, joined after timing (as our kernel_mt does)
            return list(pool.map(run_block, blocks))

        def checksum(out):
            return float(sum(o.sum() for o in out))

        note_extra = (
            f"guvectorized kernel on {len(blocks)} row blocks of {chunk} cells, "
            f"ThreadPoolExecutor({args.threads}); per-block outputs, joined after timing"
        )
    else:
        time_axis = xr.date_range(
            "2000-01-01", periods=days, freq="D", use_cftime=False
        )
        pr = xr.DataArray(
            precip, dims=("location", "time"), coords={"time": time_axis},
            attrs={"units": "mm/d", "standard_name": "precipitation_flux",
                   "cell_methods": "time: mean"},
        )
        tx = xr.DataArray(
            tmax, dims=("location", "time"), coords={"time": time_axis},
            attrs={"units": "degC", "standard_name": "air_temperature",
                   "cell_methods": "time: maximum"},
        )
        pan = xr.DataArray(pa, dims=("location",), attrs={"units": "mm/year"})
        if args.variant == "api":
            fn = xclim.indices.fire.keetch_byram_drought_index

            def call():
                return fn(pr, tx, pan)

            note_extra = "xclim.indices.fire.keetch_byram_drought_index on DataArrays"
        elif args.variant == "api_dask":
            import dask
            from concurrent.futures import ThreadPoolExecutor

            fn = xclim.indices.fire.keetch_byram_drought_index
            chunk = -(-cells // args.threads)
            prc = pr.chunk(location=chunk, time=-1)
            txc = tx.chunk(location=chunk, time=-1)
            panc = pan.chunk(location=chunk)
            pool = ThreadPoolExecutor(args.threads)
            dask.config.set(scheduler="threads", pool=pool)

            def call():
                return fn(prc, txc, panc).compute()

            note_extra = (
                f"xclim.indices.fire.keetch_byram_drought_index on dask DataArrays, "
                f"location chunk {chunk} ({-(-cells // chunk)} chunks), "
                f"dask {dask.__version__} threaded scheduler, {args.threads} workers"
            )
        else:
            fn = xclim.atmos.keetch_byram_drought_index

            def call():
                return fn(pr=pr, tasmax=tx, pr_annual=pan)

            note_extra = "xclim.atmos.keetch_byram_drought_index Indicator on DataArrays"

        def checksum(out):
            return float(out.values.sum())

    samples, first_ns, csum, last = pybench.run_protocol(
        call, checksum, cells, args.quick
    )
    if args.variant == "kernel_threads":
        last = np.concatenate(last)
    out = last if args.variant.startswith("kernel") else last.transpose("location", "time").values
    notes = (
        f"xclim {xclim.__version__}, numba {numba.__version__}, "
        f"NUMBA_NUM_THREADS={numba.config.NUMBA_NUM_THREADS}, "
        f"NUMBA_CPU_NAME={numba.config.CPU_NAME or 'host'} (oryon-1 LLVM workaround), "
        f"cache=True so first_call_ns may include loading the cached object code; "
        f"{note_extra}; float64; output mm"
    )
    pybench.write_outputs(args, "xclim", workload, cells, days, samples,
                          first_ns, csum, notes, out,
                          threads=args.threads if args.variant in ("kernel_threads", "api_dask") else 1)


if __name__ == "__main__":
    main()
