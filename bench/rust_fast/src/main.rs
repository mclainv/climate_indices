//! Second-stage KBDI prototypes (see `NOTES.md` and `kernels.rs`). Not
//! production code. Follows `../PROTOCOL.md`; every variant is checked
//! against `climate_core::fire::kbdi` before timing: bit for bit for the
//! `ref_*` variants, `rtol = atol = 1e-10` (the native-parity contract) for
//! the rest.
//!
//! Layouts: `cm_*` variants read and write `[cell][day]` (the file layout and
//! xclim's); `dm_*` and `ref_dm` read and write `[day][cell]` (the production
//! layout). Transposes for the comparison happen outside the timed region.

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// libmimalloc-sys's MSVC build references the x86-only `__popcnt64` intrinsic
// on Windows ARM64; supply it for this experiment
#[cfg(all(feature = "mimalloc", windows, target_arch = "aarch64"))]
#[unsafe(no_mangle)]
extern "C" fn __popcnt64(x: u64) -> u64 {
    u64::from(x.count_ones())
}

mod exp_table;
mod kernels;

use std::fs;
use std::hint::black_box;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use climate_core::fire::{self, KbdiCell};
use climate_core::recurrence::{MissingDayPolicy, RecurrenceInputs};
use ndarray::{Array1, Array2};

const START: KbdiCell = KbdiCell {
    kbdi: 0.0,
    wet_spell_precipitation: 0.0,
};

// --- bit-identical baselines (the first-stage `*_hoist` prototypes) -------

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

/// Production `kbdi_next` with the denominator hoisted (same operations, same order).
#[inline(always)]
fn ref_step(previous: KbdiCell, p: f64, t: f64, denominator: f64) -> KbdiCell {
    let rained = p > 0.0;
    let event_total = previous.wet_spell_precipitation + p;
    let crossing = rained && previous.wet_spell_precipitation <= 5.08 && event_total > 5.08;
    let continuing = rained && previous.wet_spell_precipitation > 5.08;
    let net_rain = if crossing {
        event_total - 5.08
    } else if continuing {
        p
    } else {
        0.0
    };
    let wet_spell_precipitation = if rained { event_total } else { 0.0 };
    let after_rain = maximum(0.0, previous.kbdi - net_rain);
    let drying = if t >= 10.0 && after_rain < 203.2 {
        (203.2 - after_rain) * (0.968 * (0.0875 * t + 1.5552).exp() - 8.30) / denominator * 1e-3
    } else {
        0.0
    };
    KbdiCell {
        kbdi: minimum(203.2, after_rain + maximum(drying, 0.0)),
        wet_spell_precipitation,
    }
}

fn denominator(r: f64) -> f64 {
    1.0 + 10.88 * (-0.001736 * r).exp()
}

fn ref_cm(p: &[f64], t: &[f64], map: &[f64], days: usize, out: &mut [f64]) {
    for (cell, &m) in map.iter().enumerate() {
        let d = denominator(m);
        let span = cell * days..(cell + 1) * days;
        let mut state = START;
        for ((&p, &t), o) in p[span.clone()].iter().zip(&t[span.clone()]).zip(&mut out[span]) {
            state = ref_step(state, p, t, d);
            *o = state.kbdi;
        }
    }
}

fn ref_dm(p: &[f64], t: &[f64], map: &[f64], days: usize, out: &mut [f64]) {
    let cells = map.len();
    let dens: Vec<f64> = map.iter().map(|&m| denominator(m)).collect();
    let mut state = vec![START; cells];
    for day in 0..days {
        let row = day * cells..(day + 1) * cells;
        for ((((&p, &t), &d), s), o) in p[row.clone()].iter().zip(&t[row.clone()]).zip(&dens).zip(&mut state).zip(&mut out[row]) {
            *s = ref_step(*s, p, t, d);
            *o = s.kbdi;
        }
    }
}

fn production(p_dm: &[f64], t_dm: &[f64], map: &[f64], days: usize) -> Vec<f64> {
    let cells = map.len();
    let p = Array2::from_shape_vec((days, cells), p_dm.to_vec()).unwrap();
    let t = Array2::from_shape_vec((days, cells), t_dm.to_vec()).unwrap();
    let m = Array1::from(map.to_vec());
    let weather_valid = Array2::from_elem((days, cells), true);
    let static_valid = Array1::from_elem(cells, true);
    let trailing_gap_days = Array1::from_elem(cells, -1_i64);
    let inputs = RecurrenceInputs {
        weather_valid: weather_valid.view(),
        static_valid: static_valid.view(),
        in_season: None,
        trailing_gap_days: trailing_gap_days.view(),
        spin_up: 0,
        policy: MissingDayPolicy::Propagate,
    };
    fire::kbdi(p.view(), t.view(), m.view(), &vec![START; cells], &inputs, true)
        .expect("kbdi")
        .values
        .expect("history")
        .into_raw_vec_and_offset()
        .0
}

// --- harness ---------------------------------------------------------------

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

type Kernel = fn(&[f64], &[f64], &[f64], usize, &mut [f64]);

fn cm_kernel(name: &str) -> Option<Kernel> {
    use kernels::*;
    Some(match name {
        "ref_cm" => ref_cm,
        "cm_scalar" => cm_scalar,
        "cm_block_vexp" => cm_block::<Vexp>,
        "cm_block_texp" => cm_block::<Texp>,
        "cm_lanes2" => cm_lanes::<2, Texp>,
        "cm_lanes4" => cm_lanes::<4, Texp>,
        "cm_lanes8" => cm_lanes::<8, Texp>,
        "cm_lanes16" => cm_lanes::<16, Texp>,
        "cm_lanes8_vexp" => cm_lanes::<8, Vexp>,
        "cm_scan4" => cm_scan::<4, Texp, false>,
        "cm_scan8" => cm_scan::<8, Texp, false>,
        "cm_scan16" => cm_scan::<16, Texp, false>,
        "cm_scan32" => cm_scan::<32, Texp, false>,
        "cm_scan8_vexp" => cm_scan::<8, Vexp, false>,
        "cm_scanf4" => cm_scan::<4, Texp, true>,
        "cm_scanf8" => cm_scan::<8, Texp, true>,
        "cm_scanf16" => cm_scan::<16, Texp, true>,
        "cm_auto" => cm_auto::<Texp>,
        "probe_fraction" => probe_fraction::<Texp>,
        "probe_fraction_vexp" => probe_fraction::<Vexp>,
        "probe_lanes8" => probe_lanes::<8>,
        "probe_spawn" => probe_fraction::<Texp>,
        _ => return None,
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
    // `_mt`: one shared output written by every worker; `_mtb`: each worker
    // owns a contiguous cell block and allocates its own output, as the
    // `climate_indices_rust` `kernel_mt` harness and xclim `kernel_threads` do
    // `_mtp`: as `_mtb` on a persistent rayon pool (built before timing),
    // the counterpart of xclim's persistent `ThreadPoolExecutor`
    let pooled = variant.ends_with("_mtp");
    let blocked = variant.ends_with("_mtb") || pooled;
    let threads = if variant.ends_with("_mt") || blocked { threads.min(cells) } else { 1 };
    let base = variant.trim_end_matches("_mtp").trim_end_matches("_mtb").trim_end_matches("_mt").to_owned();
    let rayon_pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    let day_major = base.starts_with("dm_") || base == "ref_dm" || base == "production";
    let parts = kernels::ranges(cells, threads, 8);
    // day-major blocks are sliced out once before timing, as that harness does
    let dm_parts: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = if blocked && day_major {
        parts
            .iter()
            .map(|r| {
                let mut p = Vec::with_capacity(days * r.len());
                let mut t = Vec::with_capacity(days * r.len());
                for day in 0..days {
                    p.extend_from_slice(&p_dm[day * cells + r.start..day * cells + r.end]);
                    t.extend_from_slice(&t_dm[day * cells + r.start..day * cells + r.end]);
                }
                (p, t, map_mm[r.clone()].to_vec())
            })
            .collect()
    } else {
        Vec::new()
    };
    let block_kernel = |name: &str| -> Kernel {
        match name {
            "dm_tiled_256x32" => kernels::dm_tiled::<8, 256, 32, kernels::Texp>,
            "dm_lanes8" => kernels::dm_lanes::<8, kernels::Texp>,
            "ref_dm" => ref_dm,
            other => cm_kernel(other).unwrap_or_else(|| panic!("unknown variant {other}")),
        }
    };
    let call_pooled = || -> Vec<Vec<f64>> {
        use rayon::prelude::*;
        let k = block_kernel(&base);
        rayon_pool.install(|| {
            (0..parts.len())
                .into_par_iter()
                .with_max_len(1)
                .map(|i| {
                    let (p, t, m): (&[f64], &[f64], &[f64]) = if day_major {
                        let (p, t, m) = &dm_parts[i];
                        (p, t, m)
                    } else {
                        let r = &parts[i];
                        let span = r.start * days..r.end * days;
                        (&p_cm[span.clone()], &t_cm[span], &map_mm[r.clone()])
                    };
                    let mut out = vec![0.0; p.len()];
                    k(black_box(p), t, m, days, &mut out);
                    out
                })
                .collect()
        })
    };
    let call_blocks = || -> Vec<Vec<f64>> {
        let k = block_kernel(&base);
        std::thread::scope(|s| {
            let handles: Vec<_> = if day_major {
                dm_parts
                    .iter()
                    .map(|(p, t, m)| {
                        s.spawn(move || {
                            let mut out = vec![0.0; p.len()];
                            k(black_box(p), t, m, days, &mut out);
                            out
                        })
                    })
                    .collect()
            } else {
                parts
                    .iter()
                    .map(|r| {
                        let span = r.start * days..r.end * days;
                        let (p, t, m) = (&p_cm[span.clone()], &t_cm[span], &map_mm[r.clone()]);
                        s.spawn(move || {
                            let mut out = vec![0.0; p.len()];
                            k(black_box(p), t, m, days, &mut out);
                            out
                        })
                    })
                    .collect()
            };
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    };

    // one complete call, allocating the output as the production API does;
    // `vec![0.0; n]` is a zeroed allocation (lazy pages), written once
    // diagnostic only (not protocol-fair): KBDI_REUSE=1 recycles the output
    // buffer between calls, to size the cost of committing fresh pages
    let reuse = std::env::var("KBDI_REUSE").is_ok_and(|v| v == "1");
    let pool: std::cell::RefCell<Vec<Vec<f64>>> = std::cell::RefCell::new(Vec::new());
    let recycle = |v: Vec<Vec<f64>>| {
        if reuse && v.len() == 1 {
            pool.borrow_mut().extend(v);
        }
    };
    let call_single = || -> Vec<f64> {
        let mut out = pool.borrow_mut().pop().unwrap_or_else(|| vec![0.0; cells * days]);
        let (p, t) = if day_major { (black_box(&p_dm[..]), &t_dm[..]) } else { (black_box(&p_cm[..]), &t_cm[..]) };
        match (base.as_str(), threads > 1) {
            ("production", _) => return production(p, t, &map_mm, days),
            ("ref_dm", false) => ref_dm(p, t, &map_mm, days, &mut out),
            ("dm_simd", false) => kernels::dm_simd::<kernels::Texp>(p, t, &map_mm, days, &mut out),
            ("dm_simd_vexp", false) => kernels::dm_simd::<kernels::Vexp>(p, t, &map_mm, days, &mut out),
            ("dm_lanes4", false) => kernels::dm_lanes::<4, kernels::Texp>(p, t, &map_mm, days, &mut out),
            ("dm_lanes8", false) => kernels::dm_lanes::<8, kernels::Texp>(p, t, &map_mm, days, &mut out),
            ("dm_lanes16", false) => kernels::dm_lanes::<16, kernels::Texp>(p, t, &map_mm, days, &mut out),
            ("dm_lanes8", true) => kernels::dm_lanes_mt::<8, kernels::Texp>(p, t, &map_mm, days, &mut out, threads),
            ("dm_tiled_64x32", false) => kernels::dm_tiled::<8, 64, 32, kernels::Texp>(p, t, &map_mm, days, &mut out),
            ("dm_tiled_64x128", false) => kernels::dm_tiled::<8, 64, 128, kernels::Texp>(p, t, &map_mm, days, &mut out),
            ("dm_tiled_128x64", false) => kernels::dm_tiled::<8, 128, 64, kernels::Texp>(p, t, &map_mm, days, &mut out),
            ("dm_tiled_256x32", false) => kernels::dm_tiled::<8, 256, 32, kernels::Texp>(p, t, &map_mm, days, &mut out),
            ("dm_tiled_512x16", false) => kernels::dm_tiled::<8, 512, 16, kernels::Texp>(p, t, &map_mm, days, &mut out),
            ("dm_tiled_128x64", true) => kernels::dm_tiled_mt::<8, 128, 64, kernels::Texp>(p, t, &map_mm, days, &mut out, threads),
            ("dm_tiled_256x32", true) => kernels::dm_tiled_mt::<8, 256, 32, kernels::Texp>(p, t, &map_mm, days, &mut out, threads),
            ("dm_tiled_64x128", true) => kernels::dm_tiled_mt::<8, 64, 128, kernels::Texp>(p, t, &map_mm, days, &mut out, threads),
            ("dm_simd", true) => kernels::dm_simd_mt::<kernels::Texp>(p, t, &map_mm, days, &mut out, threads),
            (name, mt) => {
                let k = cm_kernel(name).unwrap_or_else(|| panic!("unknown variant {variant}"));
                if mt {
                    kernels::cm_mt(k, p, t, &map_mm, days, &mut out, threads)
                } else {
                    k(p, t, &map_mm, days, &mut out)
                }
            }
        }
        out
    };
    let call = || -> Vec<Vec<f64>> {
        if pooled {
            call_pooled()
        } else if blocked {
            call_blocks()
        } else if base == "probe_spawn" {
            // probe: the cost of spawning and joining `threads` scoped threads
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| black_box(0));
                }
            });
            vec![vec![0.0; cells * days]]
        } else {
            vec![call_single()]
        }
    };

    // check against the production kernel before any timing
    let reference = transpose(&production(&p_dm, &t_dm, &map_mm, days), days, cells);
    let got = call();
    let candidate: Vec<f64> = if blocked && day_major {
        got.iter().zip(&parts).flat_map(|(b, r)| transpose(b, days, r.len())).collect()
    } else if day_major {
        transpose(&got[0], days, cells)
    } else {
        got.concat()
    };
    let identical = reference.iter().zip(&candidate).all(|(a, b)| a.to_bits() == b.to_bits());
    let max_abs = reference.iter().zip(&candidate).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
    let max_rel = reference
        .iter()
        .zip(&candidate)
        .filter(|(a, _)| **a != 0.0)
        .map(|(a, b)| ((a - b) / a).abs())
        .fold(0.0_f64, f64::max);
    if base.starts_with("probe_") {
        eprintln!("probe: output is not KBDI, check skipped");
    } else if variant.starts_with("ref_") || variant == "production" {
        assert!(identical, "{variant} does not reproduce the production kernel bit for bit");
    } else {
        let within = reference.iter().zip(&candidate).all(|(a, b)| (a - b).abs() <= 1e-10 + 1e-10 * a.abs());
        assert!(within, "{variant} exceeds rtol = atol = 1e-10 (max abs {max_abs:e})");
    }

    let first = Instant::now();
    let mut last = black_box(call());
    let first_call_ns = first.elapsed().as_nanos();
    let warm = Instant::now();
    let mut n = 1;
    while n < 5 || warm.elapsed() < Duration::from_millis(500) {
        recycle(last);
        last = black_box(call());
        n += 1;
    }
    let min_samples = if cells == 1 { 200 } else { 15 };
    let mut samples = Vec::new();
    let started = Instant::now();
    let mut checksum;
    loop {
        recycle(last);
        let t0 = Instant::now();
        last = black_box(call());
        samples.push(t0.elapsed().as_nanos());
        checksum = last.iter().flatten().sum::<f64>();
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
        "{{\"impl\": \"rust_fast\", \"variant\": \"{variant}\", \"workload\": \"{name}\", \"cells\": {cells}, \"days\": {days}, \"threads\": {threads}, \"reuse_output\": {reuse}, \"allocator\": \"{}\", \"samples_ns\": [{list}], \"first_call_ns\": {first_call_ns}, \"checksum\": {checksum}, \"notes\": \"prototype; bit-identical to climate_core::fire::kbdi: {identical}; max abs diff {max_abs:e} mm; max rel diff {max_rel:e}\"}}\n",
        if cfg!(feature = "mimalloc") { "mimalloc" } else { "system" }
    );
    fs::create_dir_all(timings.parent().unwrap()).unwrap();
    fs::OpenOptions::new().create(true).append(true).open(&timings).unwrap().write_all(line.as_bytes()).unwrap();
    let mut s = samples.clone();
    s.sort_unstable();
    eprintln!(
        "{variant:>16} {name:>15} t={threads:<2} median {:>9.4} ms  min {:>9.4} ms  n={:<6} bit-identical {identical:<5} max abs {max_abs:.2e} max rel {max_rel:.2e}",
        s[s.len() / 2] as f64 / 1e6,
        s[0] as f64 / 1e6,
        s.len()
    );
}
