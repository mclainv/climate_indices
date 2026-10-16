//! `climate_core::fire::kbdi` timed directly, with no Python in the loop.
//!
//! Follows `../PROTOCOL.md`. The kernel takes `(days, cells)` arrays, so the
//! cell-major inputs are transposed once before timing. Variants:
//!
//! - `kernel`: one `fire::kbdi` call over the whole block, recording history.
//! - `kernel_mt`: the block split into contiguous cell ranges, one
//!   `fire::kbdi` call per range on `std::thread::scope` threads (the kernel
//!   itself is serial; this is the harness's split, as the Python layer's
//!   thread pool does for spatial blocks).

use std::fs;
use std::hint::black_box;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use climate_core::fire::{self, KbdiCell};
use climate_core::recurrence::{MissingDayPolicy, RecurrenceInputs};
use ndarray::{Array1, Array2, s};

struct Args {
    workload: PathBuf,
    variant: String,
    timings: PathBuf,
    values: Option<PathBuf>,
    threads: usize,
}

fn parse_args() -> Args {
    let mut workload = None;
    let mut variant = "kernel".to_owned();
    let mut timings = PathBuf::from("out/timings.jsonl");
    let mut values = None;
    let mut threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--workload" => workload = Some(PathBuf::from(value)),
            "--variant" => variant = value,
            "--timings" => timings = PathBuf::from(value),
            "--values" => values = Some(PathBuf::from(value)),
            "--threads" => threads = value.parse().expect("--threads"),
            other => panic!("unknown flag {other}"),
        }
    }
    Args {
        workload: workload.expect("--workload is required"),
        variant,
        timings,
        values,
        threads,
    }
}

fn read_f64(path: &Path) -> Vec<f64> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
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

/// Cell-major `[cell][day]` to the kernel's `(days, cells)`.
fn day_major(values: &[f64], cells: usize, days: usize) -> Array2<f64> {
    Array2::from_shape_fn((days, cells), |(day, cell)| values[cell * days + day])
}

/// One fresh recurrence over `cells` columns: zero index, empty wet spell,
/// never started, every day observed, the ADR-0007 default policy.
fn run_block(precip: &Array2<f64>, tmax: &Array2<f64>, map_mm: &Array1<f64>) -> Array2<f64> {
    let (days, cells) = precip.dim();
    let weather_valid = Array2::from_elem((days, cells), true);
    let static_valid = Array1::from_elem(cells, true);
    let trailing_gap_days = Array1::from_elem(cells, -1_i64);
    let initial = vec![
        KbdiCell {
            kbdi: 0.0,
            wet_spell_precipitation: 0.0
        };
        cells
    ];
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
        .expect("recorded history")
}

fn main() {
    let args = parse_args();
    let meta = fs::read_to_string(args.workload.join("meta.json")).expect("meta.json");
    let (cells, days) = (meta_usize(&meta, "cells"), meta_usize(&meta, "days"));
    let precip = day_major(&read_f64(&args.workload.join("precip_mm.bin")), cells, days);
    let tmax = day_major(&read_f64(&args.workload.join("tmax_c.bin")), cells, days);
    let map_mm = Array1::from(read_f64(&args.workload.join("map_mm.bin")));
    let workload_name = args.workload.file_name().unwrap().to_string_lossy().into_owned();

    let threads = if args.variant == "kernel_mt" { args.threads.min(cells) } else { 1 };
    let bounds: Vec<(usize, usize)> = (0..threads)
        .map(|t| (t * cells / threads, (t + 1) * cells / threads))
        .collect();
    // column views of the inputs, sliced once before timing
    let blocks: Vec<(Array2<f64>, Array2<f64>, Array1<f64>)> = bounds
        .iter()
        .map(|&(a, b)| {
            (
                precip.slice(s![.., a..b]).to_owned(),
                tmax.slice(s![.., a..b]).to_owned(),
                map_mm.slice(s![a..b]).to_owned(),
            )
        })
        .collect();

    let call = || -> Vec<Array2<f64>> {
        if threads == 1 {
            let (p, t, m) = &blocks[0];
            vec![run_block(black_box(p), black_box(t), black_box(m))]
        } else {
            std::thread::scope(|scope| {
                let handles: Vec<_> = blocks
                    .iter()
                    .map(|(p, t, m)| scope.spawn(move || run_block(p, t, m)))
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            })
        }
    };
    let checksum = |out: &[Array2<f64>]| out.iter().map(|a| a.sum()).sum::<f64>();

    let first = Instant::now();
    let mut last = black_box(call());
    let first_call_ns = first.elapsed().as_nanos();
    let warm = Instant::now();
    let mut warm_calls = 1;
    while warm_calls < 5 || warm.elapsed() < Duration::from_millis(500) {
        last = black_box(call());
        warm_calls += 1;
    }

    let min_samples = if cells == 1 { 200 } else { 15 };
    let mut samples: Vec<u128> = Vec::new();
    let mut sum = 0.0;
    let started = Instant::now();
    loop {
        let t0 = Instant::now();
        last = black_box(call());
        samples.push(t0.elapsed().as_nanos());
        sum = black_box(checksum(&last));
        let elapsed = started.elapsed();
        let enough = samples.len() >= min_samples && elapsed >= Duration::from_secs(3);
        if enough || (elapsed >= Duration::from_secs(60) && samples.len() >= 3) {
            break;
        }
    }

    if let Some(path) = &args.values {
        let mut out = vec![0.0_f64; cells * days];
        for ((a, _), block) in bounds.iter().zip(&last) {
            for ((day, col), &v) in block.indexed_iter() {
                out[(a + col) * days + day] = v;
            }
        }
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).unwrap();
        }
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(path, bytes).unwrap();
    }

    let list = samples.iter().map(u128::to_string).collect::<Vec<_>>().join(",");
    let line = format!(
        "{{\"impl\": \"climate_indices_rust\", \"variant\": \"{}\", \"workload\": \"{}\", \"cells\": {}, \"days\": {}, \"threads\": {}, \"samples_ns\": [{}], \"first_call_ns\": {}, \"checksum\": {}, \"notes\": \"climate_core::fire::kbdi direct, recorded history, float64\"}}\n",
        args.variant, workload_name, cells, days, threads, list, first_call_ns, sum
    );
    if let Some(dir) = args.timings.parent() {
        fs::create_dir_all(dir).unwrap();
    }
    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.timings)
        .unwrap()
        .write_all(line.as_bytes())
        .unwrap();
    let mut sorted = samples.clone();
    sorted.sort_unstable();
    eprintln!(
        "{} {} {}: median {:.4} ms, min {:.4} ms, n={}, checksum {}",
        args.variant,
        workload_name,
        threads,
        sorted[sorted.len() / 2] as f64 / 1e6,
        sorted[0] as f64 / 1e6,
        samples.len(),
        sum
    );
}
