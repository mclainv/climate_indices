//! Prototype KBDI loop structures, to size the optimizations proposed in
//! `../REPORT.md`. **Not production code**: nothing here is called by
//! `climate_indices`, and every variant must reproduce
//! `climate_core::fire::kbdi` bit for bit on the benchmark workloads (the
//! harness checks this before timing and aborts otherwise).
//!
//! The daily step below is a verbatim copy of `kbdi_next` in
//! `crates/climate-core/src/fire.rs` (private there). The prototypes keep the
//! production missing-day contract for the default `propagate` policy only: a
//! started cell that meets a missing day is poisoned (NaN) for the rest of the
//! run; a cell never starts before its first valid day.
//!
//! Variants (all single-threaded unless suffixed `_mt`):
//!
//! - `production`: `climate_core::fire::kbdi` on `(days, cells)`, as the
//!   `climate_indices_rust` harness, for a same-binary reference.
//! - `daymajor`: the production `(days, cells)` layout, one pass per day over
//!   plain state vectors (no `updates` buffer, no `ndarray` indexing in the
//!   hot loop).
//! - `daymajor_hoist`: `daymajor` with the climatology denominator
//!   `1 + 10.88 exp(-0.001736 R)` computed once per cell instead of once per
//!   cell-day.
//! - `cellmajor_hoist`: one cell at a time over a contiguous `[cell][day]`
//!   series with the hoisted denominator, writing a `[cell][day]` history.
//! - `daymajor_hoist_libm` / `cellmajor_hoist_libm`: the hoisted loops with
//!   the pure-Rust `libm` crate's `exp` (a musl port, inlinable) instead of
//!   the platform C runtime's; not bit-identical, so held to the
//!   `rtol = atol = 1e-10` native-parity tolerance instead.
//! - `cellmajor_pre`: `cellmajor_hoist` restructured into two passes over
//!   blocks of `PRE_BLOCK` days: first the temperature term
//!   `0.968 exp(0.0875 T + 1.5552) - 8.30` for every day (it does not depend
//!   on the index state), then the serial recurrence reading it. Same
//!   operations in the same order, so bit-identical. The output is written
//!   once (no NaN pre-fill).
//! - `cellmajor_pre_vexp`: `cellmajor_pre` with `vexp`, a branch-free
//!   polynomial `exp` that LLVM inlines and vectorizes (NEON, 2 lanes) in the
//!   first pass; held to the `rtol = atol = 1e-10` native-parity tolerance.
//! - `daymajor_hoist_mt` / `cellmajor_hoist_mt` / `cellmajor_pre_mt` /
//!   `cellmajor_pre_vexp_mt`: the same split across threads by contiguous
//!   cell ranges.
//!
//! The timed region covers one call that returns the full history; the
//! checksum over it is taken after the clock stops (as the Python harnesses
//! do).

use std::fs;
use std::hint::black_box;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use climate_core::fire::{self, KbdiCell};
use climate_core::recurrence::{MissingDayPolicy, RecurrenceInputs};
use ndarray::{Array1, Array2};

const KBDI_MAX_MM: f64 = 203.2;
const KBDI_RAIN_THRESHOLD_MM: f64 = 5.08;
const KBDI_DRYING_TEMPERATURE_CELSIUS: f64 = 10.0;

fn maximum(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else if left >= right {
        left
    } else {
        right
    }
}

fn minimum(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else if left <= right {
        left
    } else {
        right
    }
}

/// `kbdi_next` with the climatology denominator passed in. With
/// `denominator = 1.0 + 10.88 * (-0.001736 * R).exp()` it performs the same
/// operations in the same order as the production step.
#[inline(always)]
fn kbdi_step(
    previous: KbdiCell,
    precipitation_mm: f64,
    maximum_temperature_celsius: f64,
    denominator: f64,
    exp: impl Fn(f64) -> f64,
) -> KbdiCell {
    kbdi_step_core(previous, precipitation_mm, maximum_temperature_celsius, denominator, || {
        0.968 * exp(0.0875 * maximum_temperature_celsius + 1.5552) - 8.30
    })
}

/// The step with the temperature term `0.968 exp(0.0875 T + 1.5552) - 8.30`
/// supplied by `heat` (evaluated only on drought days).
#[inline(always)]
fn kbdi_step_core(
    previous: KbdiCell,
    precipitation_mm: f64,
    maximum_temperature_celsius: f64,
    denominator: f64,
    heat: impl FnOnce() -> f64,
) -> KbdiCell {
    let rained = precipitation_mm > 0.0;
    let event_total = previous.wet_spell_precipitation + precipitation_mm;
    let crossing_threshold =
        rained && previous.wet_spell_precipitation <= KBDI_RAIN_THRESHOLD_MM && event_total > KBDI_RAIN_THRESHOLD_MM;
    let continuing_wet_spell = rained && previous.wet_spell_precipitation > KBDI_RAIN_THRESHOLD_MM;
    let net_rain = if crossing_threshold {
        event_total - KBDI_RAIN_THRESHOLD_MM
    } else if continuing_wet_spell {
        precipitation_mm
    } else {
        0.0
    };
    let wet_spell_precipitation = if rained { event_total } else { 0.0 };

    let after_rain = maximum(0.0, previous.kbdi - net_rain);
    let drought_day = maximum_temperature_celsius >= KBDI_DRYING_TEMPERATURE_CELSIUS && after_rain < KBDI_MAX_MM;
    let drying = if drought_day {
        (KBDI_MAX_MM - after_rain) * heat() / denominator
            * 1e-3
    } else {
        0.0
    };
    KbdiCell {
        kbdi: minimum(KBDI_MAX_MM, after_rain + maximum(drying, 0.0)),
        wet_spell_precipitation,
    }
}

fn denominator(mean_annual_precipitation_mm: f64) -> f64 {
    1.0 + 10.88 * (-0.001736 * mean_annual_precipitation_mm).exp()
}

const START: KbdiCell = KbdiCell {
    kbdi: 0.0,
    wet_spell_precipitation: 0.0,
};

/// Day-major pass; `hoist` selects the per-cell denominator.
fn daymajor(precip: &[f64], tmax: &[f64], map_mm: &[f64], days: usize, hoist: bool) -> Vec<f64> {
    daymajor_with(precip, tmax, map_mm, days, hoist, f64::exp)
}

fn daymajor_with(
    precip: &[f64],
    tmax: &[f64],
    map_mm: &[f64],
    days: usize,
    hoist: bool,
    exp: impl Fn(f64) -> f64 + Copy,
) -> Vec<f64> {
    let cells = map_mm.len();
    let denominators: Vec<f64> = map_mm.iter().map(|&r| denominator(r)).collect();
    let mut state = vec![START; cells];
    let mut started = vec![false; cells];
    let mut out = vec![f64::NAN; days * cells];
    for day in 0..days {
        let row = day * cells;
        let (p_row, t_row) = (&precip[row..row + cells], &tmax[row..row + cells]);
        let out_row = &mut out[row..row + cells];
        for cell in 0..cells {
            let (p, t) = (p_row[cell], t_row[cell]);
            if p.is_finite() && t.is_finite() {
                let d = if hoist { denominators[cell] } else { denominator(map_mm[cell]) };
                let next = kbdi_step(state[cell], p, t, d, exp);
                state[cell] = next;
                started[cell] = true;
                out_row[cell] = next.kbdi;
            } else if started[cell] {
                state[cell].kbdi = f64::NAN;
            }
        }
    }
    out
}

/// One contiguous series at a time, history written `[cell][day]`.
fn cellmajor_hoist(precip: &[f64], tmax: &[f64], map_mm: &[f64], days: usize) -> Vec<f64> {
    cellmajor_hoist_with(precip, tmax, map_mm, days, f64::exp)
}

fn cellmajor_hoist_with(precip: &[f64], tmax: &[f64], map_mm: &[f64], days: usize, exp: impl Fn(f64) -> f64 + Copy) -> Vec<f64> {
    let cells = map_mm.len();
    let mut out = vec![f64::NAN; days * cells];
    for cell in 0..cells {
        let d = denominator(map_mm[cell]);
        let span = cell * days..(cell + 1) * days;
        let (p_series, t_series, out_series) = (&precip[span.clone()], &tmax[span.clone()], &mut out[span]);
        let mut state = START;
        let mut started = false;
        for day in 0..days {
            let (p, t) = (p_series[day], t_series[day]);
            if p.is_finite() && t.is_finite() {
                state = kbdi_step(state, p, t, d, exp);
                started = true;
                out_series[day] = state.kbdi;
            } else if started {
                state.kbdi = f64::NAN;
            }
        }
    }
    out
}

/// Days per block in the two-pass variants (temperature term buffered on the stack).
const PRE_BLOCK: usize = 512;

/// Branch-free `exp` for the drying term's argument range: Cody-Waite
/// reduction by ln 2, a degree-12 Taylor polynomial on |r| <= ln2/2 (error
/// below 1 ulp-ish, ~2e-16 relative), and 2^n built from the exponent bits.
/// Inputs are clamped to [-700, 700]; NaN maps to NaN.
#[inline(always)]
fn vexp(x: f64) -> f64 {
    const LOG2E: f64 = std::f64::consts::LOG2_E;
    const LN2_HI: f64 = 6.931_471_803_691_238_164_90e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_700_02e-10;
    let x = x.clamp(-700.0, 700.0);
    let n = (x * LOG2E).round();
    let r = (-n).mul_add(LN2_LO, (-n).mul_add(LN2_HI, x));
    let mut poly: f64 = 1.0 / 479_001_600.0;
    for c in [
        1.0 / 39_916_800.0,
        1.0 / 3_628_800.0,
        1.0 / 362_880.0,
        1.0 / 40_320.0,
        1.0 / 5_040.0,
        1.0 / 720.0,
        1.0 / 120.0,
        1.0 / 24.0,
        1.0 / 6.0,
        0.5,
        1.0,
        1.0,
    ] {
        poly = poly.mul_add(r, c);
    }
    let scale = f64::from_bits(((n as i64 + 1023) as u64) << 52);
    poly * scale
}

/// Two-pass cell-major loop; the output is written exactly once.
fn cellmajor_pre_with(precip: &[f64], tmax: &[f64], map_mm: &[f64], days: usize, exp: impl Fn(f64) -> f64 + Copy) -> Vec<f64> {
    let cells = map_mm.len();
    let mut out: Vec<f64> = Vec::with_capacity(days * cells);
    let mut heat = [0.0_f64; PRE_BLOCK];
    for cell in 0..cells {
        let d = denominator(map_mm[cell]);
        let span = cell * days..(cell + 1) * days;
        let (p_series, t_series) = (&precip[span.clone()], &tmax[span]);
        let mut state = START;
        let mut started = false;
        for start in (0..days).step_by(PRE_BLOCK) {
            let end = (start + PRE_BLOCK).min(days);
            let (p_block, t_block) = (&p_series[start..end], &t_series[start..end]);
            let heat = &mut heat[..end - start];
            for (h, &t) in heat.iter_mut().zip(t_block) {
                *h = 0.968 * exp(0.0875 * t + 1.5552) - 8.30;
            }
            for ((&p, &t), &h) in p_block.iter().zip(t_block).zip(heat.iter()) {
                if p.is_finite() && t.is_finite() {
                    state = kbdi_step_core(state, p, t, d, || h);
                    started = true;
                    out.push(state.kbdi);
                } else {
                    if started {
                        state.kbdi = f64::NAN;
                    }
                    out.push(f64::NAN);
                }
            }
        }
    }
    out
}

fn production(precip: &Array2<f64>, tmax: &Array2<f64>, map_mm: &Array1<f64>) -> Array2<f64> {
    let (days, cells) = precip.dim();
    let weather_valid = Array2::from_elem((days, cells), true);
    let static_valid = Array1::from_elem(cells, true);
    let trailing_gap_days = Array1::from_elem(cells, -1_i64);
    let initial = vec![START; cells];
    let inputs = RecurrenceInputs {
        weather_valid: weather_valid.view(),
        static_valid: static_valid.view(),
        in_season: None,
        trailing_gap_days: trailing_gap_days.view(),
        spin_up: 0,
        policy: MissingDayPolicy::Propagate,
    };
    fire::kbdi(precip.view(), tmax.view(), map_mm.view(), &initial, &inputs, true)
        .expect("kbdi")
        .values
        .expect("history")
}

fn read_f64(path: &Path) -> Vec<f64> {
    fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn meta_usize(meta: &str, key: &str) -> usize {
    let at = meta.find(&format!("\"{key}\"")).expect(key);
    let rest = &meta[at + key.len() + 2..];
    let rest = rest[rest.find(':').unwrap() + 1..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap();
    rest[..end].parse().unwrap()
}

fn transpose(values: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; values.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = values[r * cols + c];
        }
    }
    out
}

/// Split `cells` into `threads` contiguous ranges.
fn ranges(cells: usize, threads: usize) -> Vec<(usize, usize)> {
    (0..threads).map(|t| (t * cells / threads, (t + 1) * cells / threads)).collect()
}

/// Cell-major variants split across scoped threads by contiguous cell ranges.
fn cellmajor_mt(p_cm: &[f64], t_cm: &[f64], map_mm: &[f64], days: usize, parts: &[(usize, usize)], variant: &str) -> Vec<Vec<f64>> {
    std::thread::scope(|s| {
        let hs: Vec<_> = parts
            .iter()
            .map(|&(a, b)| {
                let (p, t, m) = (&p_cm[a * days..b * days], &t_cm[a * days..b * days], &map_mm[a..b]);
                s.spawn(move || match variant {
                    "cellmajor_hoist_mt" => cellmajor_hoist(p, t, m, days),
                    "cellmajor_pre_mt" => cellmajor_pre_with(p, t, m, days, f64::exp),
                    "cellmajor_pre_vexp_mt" => cellmajor_pre_with(p, t, m, days, vexp),
                    other => unreachable!("{other}"),
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

fn main() {
    let mut it = std::env::args().skip(1);
    let (mut workload, mut variant, mut timings, mut values) = (None, String::new(), PathBuf::new(), None);
    let mut threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    while let Some(flag) = it.next() {
        let v = it.next().expect("value");
        match flag.as_str() {
            "--workload" => workload = Some(PathBuf::from(v)),
            "--variant" => variant = v,
            "--timings" => timings = PathBuf::from(v),
            "--values" => values = Some(PathBuf::from(v)),
            "--threads" => threads = v.parse().unwrap(),
            other => panic!("unknown flag {other}"),
        }
    }
    let workload = workload.expect("--workload");
    let meta = fs::read_to_string(workload.join("meta.json")).unwrap();
    let (cells, days) = (meta_usize(&meta, "cells"), meta_usize(&meta, "days"));
    let p_cm = read_f64(&workload.join("precip_mm.bin"));
    let t_cm = read_f64(&workload.join("tmax_c.bin"));
    let map_mm = read_f64(&workload.join("map_mm.bin"));
    let p_dm = transpose(&p_cm, cells, days);
    let t_dm = transpose(&t_cm, cells, days);
    let p_arr = Array2::from_shape_vec((days, cells), p_dm.clone()).unwrap();
    let t_arr = Array2::from_shape_vec((days, cells), t_dm.clone()).unwrap();
    let m_arr = Array1::from(map_mm.clone());
    let threads = if variant.ends_with("_mt") { threads.min(cells) } else { 1 };
    let parts = ranges(cells, threads);

    // per-range inputs for the threaded variants, prepared once
    let dm_parts: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = parts
        .iter()
        .map(|&(a, b)| {
            let width = b - a;
            let mut p = Vec::with_capacity(days * width);
            let mut t = Vec::with_capacity(days * width);
            for day in 0..days {
                p.extend_from_slice(&p_dm[day * cells + a..day * cells + b]);
                t.extend_from_slice(&t_dm[day * cells + a..day * cells + b]);
            }
            (p, t, map_mm[a..b].to_vec())
        })
        .collect();

    // every variant returns the history cell-major so the outputs compare directly
    let call = || -> Vec<f64> {
        match variant.as_str() {
            "production" => {
                let out = production(black_box(&p_arr), black_box(&t_arr), black_box(&m_arr));
                transpose(out.as_slice().unwrap(), days, cells)
            }
            "daymajor" => transpose(&daymajor(black_box(&p_dm), &t_dm, &map_mm, days, false), days, cells),
            "daymajor_hoist" => transpose(&daymajor(black_box(&p_dm), &t_dm, &map_mm, days, true), days, cells),
            "cellmajor_hoist" => cellmajor_hoist(black_box(&p_cm), &t_cm, &map_mm, days),
            "daymajor_hoist_libm" => {
                transpose(&daymajor_with(black_box(&p_dm), &t_dm, &map_mm, days, true, libm::exp), days, cells)
            }
            "cellmajor_hoist_libm" => cellmajor_hoist_with(black_box(&p_cm), &t_cm, &map_mm, days, libm::exp),
            "cellmajor_pre" => cellmajor_pre_with(black_box(&p_cm), &t_cm, &map_mm, days, f64::exp),
            "cellmajor_pre_vexp" => cellmajor_pre_with(black_box(&p_cm), &t_cm, &map_mm, days, vexp),
            "cellmajor_pre_mt" | "cellmajor_pre_vexp_mt" => cellmajor_mt(&p_cm, &t_cm, &map_mm, days, &parts, &variant).concat(),
            "daymajor_hoist_mt" => {
                let blocks: Vec<Vec<f64>> = std::thread::scope(|s| {
                    let hs: Vec<_> = dm_parts
                        .iter()
                        .map(|(p, t, m)| s.spawn(move || daymajor(p, t, m, days, true)))
                        .collect();
                    hs.into_iter().map(|h| h.join().unwrap()).collect()
                });
                let mut out = Vec::with_capacity(cells * days);
                for ((a, b), block) in parts.iter().zip(&blocks) {
                    out.extend(transpose(block, days, b - a));
                    let _ = a;
                }
                out
            }
            "cellmajor_hoist_mt" => {
                let blocks: Vec<Vec<f64>> = std::thread::scope(|s| {
                    let hs: Vec<_> = parts
                        .iter()
                        .map(|&(a, b)| {
                            let (p, t, m) = (&p_cm[a * days..b * days], &t_cm[a * days..b * days], &map_mm[a..b]);
                            s.spawn(move || cellmajor_hoist(p, t, m, days))
                        })
                        .collect();
                    hs.into_iter().map(|h| h.join().unwrap()).collect()
                });
                blocks.concat()
            }
            other => panic!("unknown variant {other}"),
        }
    };

    // bit-for-bit check against the production kernel before any timing
    let reference = transpose(production(&p_arr, &t_arr, &m_arr).as_slice().unwrap(), days, cells);
    let candidate = call();
    let identical = reference
        .iter()
        .zip(&candidate)
        .all(|(a, b)| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()));
    let max_abs = reference
        .iter()
        .zip(&candidate)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    if variant.ends_with("_libm") || variant.contains("_vexp") {
        // a different exp: not bit-identical, held to the native-parity tolerance instead
        let within = reference
            .iter()
            .zip(&candidate)
            .all(|(a, b)| (a - b).abs() <= 1e-10 + 1e-10 * a.abs());
        assert!(within, "{variant} exceeds rtol = atol = 1e-10 (max abs {max_abs:e})");
    } else {
        assert!(identical, "{variant} does not reproduce the production kernel bit for bit");
    }

    // the timed region excludes the cell-major transposes done only for comparison:
    // time the raw loops directly where a transpose was added above; each call
    // returns its history blocks, checksummed after the clock stops
    let timed = || -> Vec<Vec<f64>> {
        match variant.as_str() {
            "production" => vec![production(black_box(&p_arr), black_box(&t_arr), black_box(&m_arr)).into_raw_vec_and_offset().0],
            "daymajor" => vec![daymajor(black_box(&p_dm), &t_dm, &map_mm, days, false)],
            "daymajor_hoist" => vec![daymajor(black_box(&p_dm), &t_dm, &map_mm, days, true)],
            "cellmajor_hoist" => vec![cellmajor_hoist(black_box(&p_cm), &t_cm, &map_mm, days)],
            "daymajor_hoist_libm" => vec![daymajor_with(black_box(&p_dm), &t_dm, &map_mm, days, true, libm::exp)],
            "cellmajor_hoist_libm" => vec![cellmajor_hoist_with(black_box(&p_cm), &t_cm, &map_mm, days, libm::exp)],
            "cellmajor_pre" => vec![cellmajor_pre_with(black_box(&p_cm), &t_cm, &map_mm, days, f64::exp)],
            "cellmajor_pre_vexp" => vec![cellmajor_pre_with(black_box(&p_cm), &t_cm, &map_mm, days, vexp)],
            "daymajor_hoist_mt" => std::thread::scope(|s| {
                let hs: Vec<_> = dm_parts
                    .iter()
                    .map(|(p, t, m)| s.spawn(move || daymajor(p, t, m, days, true)))
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            }),
            "cellmajor_hoist_mt" | "cellmajor_pre_mt" | "cellmajor_pre_vexp_mt" => {
                cellmajor_mt(black_box(&p_cm), &t_cm, &map_mm, days, &parts, &variant)
            }
            _ => unreachable!(),
        }
    };
    let checksum_of = |blocks: &[Vec<f64>]| blocks.iter().map(|b| b.iter().sum::<f64>()).sum::<f64>();

    let first = Instant::now();
    let mut last = black_box(timed());
    let first_call_ns = first.elapsed().as_nanos();
    let mut checksum;
    let warm = Instant::now();
    let mut n = 1;
    while n < 5 || warm.elapsed() < Duration::from_millis(500) {
        drop(last);
        last = black_box(timed());
        n += 1;
    }
    let min_samples = if cells == 1 { 200 } else { 15 };
    let mut samples = Vec::new();
    let started = Instant::now();
    loop {
        drop(last);
        let t0 = Instant::now();
        last = black_box(timed());
        samples.push(t0.elapsed().as_nanos());
        checksum = checksum_of(&last);
        let e = started.elapsed();
        if (samples.len() >= min_samples && e >= Duration::from_secs(3)) || (e >= Duration::from_secs(60) && samples.len() >= 3) {
            break;
        }
    }
    if let Some(path) = values {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, candidate.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    }
    let name = workload.file_name().unwrap().to_string_lossy().into_owned();
    let list = samples.iter().map(u128::to_string).collect::<Vec<_>>().join(",");
    let line = format!(
        "{{\"impl\": \"rust_prototype\", \"variant\": \"{variant}\", \"workload\": \"{name}\", \"cells\": {cells}, \"days\": {days}, \"threads\": {threads}, \"samples_ns\": [{list}], \"first_call_ns\": {first_call_ns}, \"checksum\": {checksum}, \"notes\": \"prototype; bit-identical to climate_core::fire::kbdi: {identical}; max abs diff {max_abs:e} mm\"}}\n"
    );
    fs::create_dir_all(timings.parent().unwrap()).unwrap();
    fs::OpenOptions::new().create(true).append(true).open(&timings).unwrap().write_all(line.as_bytes()).unwrap();
    let mut s = samples.clone();
    s.sort_unstable();
    eprintln!(
        "{variant} {name} t={threads}: median {:.4} ms, min {:.4} ms, n={}, bit-identical {identical}, max abs {max_abs:e}",
        s[s.len() / 2] as f64 / 1e6,
        s[0] as f64 / 1e6,
        s.len()
    );
}
