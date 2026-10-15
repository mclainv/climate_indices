#!/usr/bin/env python3
"""Write the shared KBDI benchmark inputs every harness in this directory reads.

Usage:
    python benchmarks/kbdi_competitors/make_inputs.py

All workloads come from the committed Fresno GHCN-Daily fixture
(``tests/fixture/kbdi_ghcn/fresno_1991_2020.csv``: 10958 complete days of
PRCP and TMAX, no gaps), so every implementation sees real weather:

- ``station_1825``: the first 1825 days (1991-01-01 onward), one station.
- ``station_10958``: the whole 30-year record, one station; the fixture's
  ``kbdi_mm`` column is its expected answer.
- ``grid_400x3650``: 400 locations x 3650 days. Location ``i`` is the
  3650-day window starting at day ``18 * i``, so neighbouring locations are
  shifted copies of one real record rather than random noise. Each location's
  mean annual precipitation is its own window's mean daily total times 365.25.
- ``grid_2500x3650``: the same construction with 2500 locations, windows
  starting at day ``(i * 7308) // 2499``, for scaling.

Each workload directory holds little-endian float64 files laid out
cell-major (``[cell][day]``, each cell's series contiguous):

- ``precip_mm.bin``, ``tmax_c.bin``: metric inputs.
- ``precip_in.bin``, ``tmax_f.bin``: the same values in inches and degrees
  Fahrenheit for implementations that take US customary units.
- ``map_mm.bin``, ``map_in.bin``: mean annual precipitation per cell.
- ``meta.json``: shape and provenance.

The files are regenerated deterministically and are not committed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIXTURE = ROOT / "tests" / "fixture" / "kbdi_ghcn" / "fresno_1991_2020.csv"
DATA = HERE / "data"
MM_PER_INCH = 25.4
DAYS_PER_YEAR = 365.25


def read_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with FIXTURE.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    precipitation = np.array([float(row["precipitation_mm"]) for row in rows])
    temperature = np.array([float(row["maximum_temperature_c"]) for row in rows])
    expected = np.array([float(row["kbdi_mm"]) for row in rows])
    return precipitation, temperature, expected


def write_workload(name: str, precipitation: np.ndarray, temperature: np.ndarray, map_mm: np.ndarray, note: str) -> None:
    cells, days = precipitation.shape
    directory = DATA / name
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        "precip_mm": precipitation,
        "tmax_c": temperature,
        "precip_in": precipitation / MM_PER_INCH,
        "tmax_f": temperature * 9.0 / 5.0 + 32.0,
        "map_mm": map_mm,
        "map_in": map_mm / MM_PER_INCH,
    }
    for stem, array in arrays.items():
        np.ascontiguousarray(array, dtype="<f8").tofile(directory / f"{stem}.bin")
    meta = {"name": name, "cells": cells, "days": days, "layout": "cell-major float64 little-endian", "note": note}
    (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"{name}: {cells} cells x {days} days")


def windows(precipitation: np.ndarray, temperature: np.ndarray, starts: list[int], days: int):
    p = np.stack([precipitation[s : s + days] for s in starts])
    t = np.stack([temperature[s : s + days] for s in starts])
    map_mm = p.sum(axis=1) / days * DAYS_PER_YEAR
    return p, t, map_mm


def main() -> None:
    precipitation, temperature, expected = read_fixture()
    total = precipitation.size
    # scripts/prepare_kbdi_fixtures.py: the record's mean daily total times 365.25 (278.900607 mm)
    fresno_map_mm = sum(precipitation.tolist()) / total * DAYS_PER_YEAR
    write_workload(
        "station_1825",
        precipitation[None, :1825],
        temperature[None, :1825],
        np.array([fresno_map_mm]),
        "Fresno USW00093193, 1991-01-01 onward; 30-year mean annual precipitation",
    )
    write_workload(
        "station_10958",
        precipitation[None, :],
        temperature[None, :],
        np.array([fresno_map_mm]),
        "Fresno USW00093193, 1991-2020; expected.bin is the fixture's kbdi_mm column",
    )
    np.ascontiguousarray(expected, dtype="<f8").tofile(DATA / "station_10958" / "expected_kbdi_mm.bin")
    days = 3650
    p, t, m = windows(precipitation, temperature, [18 * i for i in range(400)], days)
    write_workload("grid_400x3650", p, t, m, "location i = Fresno days [18i, 18i+3650); MAP = window mean * 365.25")
    last = total - days
    p, t, m = windows(precipitation, temperature, [(i * last) // 2499 for i in range(2500)], days)
    write_workload("grid_2500x3650", p, t, m, "location i = Fresno days [(i*7308)//2499, +3650); MAP = window mean * 365.25")


if __name__ == "__main__":
    main()
