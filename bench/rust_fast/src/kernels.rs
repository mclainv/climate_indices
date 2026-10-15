//! Fast KBDI kernels (prototypes, not production code).
//!
//! Every kernel here assumes complete input (every precipitation and
//! temperature value finite), the protocol's start (KBDI 0, empty wet spell)
//! and writes every output value. A production version would scan the input
//! once for non-finite values and fall back to the general driver if it finds
//! any.
//!
//! Three ideas, kept separable so they can be measured one at a time:
//!
//! 1. **Short dependency chain.** The production step evaluates
//!    `(203.2 - Q1) * heat / den * 1e-3` on the index chain (a divide among
//!    seven dependent operations). The drying fraction
//!    `k = max(heat * (1e-3 / den), 0)` (0 below 10 C) does not depend on the
//!    index, so it is computed off the chain and the index step becomes
//!    `Q = min(203.2, Q1 + (203.2 - Q1) * k)`, with branch-free selects for
//!    the wet spell. Not bit-identical (one rounding moves), well inside the
//!    `rtol = atol = 1e-10` parity contract.
//! 2. **Portable vectorizable `exp`** (`vexp`): no `round()`, no float to
//!    integer conversion and no fused multiply-add, all of which are library
//!    calls on the x86-64 SSE2 baseline that wheels target. Plain mul/add
//!    Horner (Rust never contracts them), so the bits are the same on every
//!    platform, which the C runtime `exp` does not give today.
//! 3. **Independent series in lanes.** Grids: `L` cells advance together (in
//!    either layout). One series: the clamp-affine daily maps compose in
//!    closed form, so the series is split into `L` chunks that start after a
//!    dry day (empty wet spell), each chunk's map is composed in lanes, the
//!    chunk start values are chained, and the chunks are rerun in lanes.

pub const MAX: f64 = 203.2;
const RAIN: f64 = 5.08;
const WARM: f64 = 10.0;

// `f64::max` is one instruction on AArch64 (`fmaxnm`) but a NaN fix-up
// sequence on x86; the compare-select is one `maxsd`/`maxpd` there. Identical
// for the finite values these kernels see.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fmax(a: f64, b: f64) -> f64 {
    a.max(b)
}
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fmin(a: f64, b: f64) -> f64 {
    a.min(b)
}
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn fmax(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn fmin(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

/// `1e-3 / (1 + 10.88 exp(-0.001736 R))`, once per cell.
pub fn scale(mean_annual_precipitation_mm: f64) -> f64 {
    1e-3 / (1.0 + 10.88 * (-0.001736 * mean_annual_precipitation_mm).exp())
}

/// Horner (`false`, degree-12 Taylor) or Estrin (`true`, degree-11 economized)
/// polynomial in `vexp`. Estrin's dependency depth is about a third of
/// Horner's, which matters when the out-of-order window cannot overlap enough
/// iterations to hide Horner's latency.
const ESTRIN: bool = true;

#[inline(always)]
fn exp_horner(r: f64) -> f64 {
    let mut poly = 1.0 / 479_001_600.0;
    poly = poly * r + 1.0 / 39_916_800.0;
    poly = poly * r + 1.0 / 3_628_800.0;
    poly = poly * r + 1.0 / 362_880.0;
    poly = poly * r + 1.0 / 40_320.0;
    poly = poly * r + 1.0 / 5_040.0;
    poly = poly * r + 1.0 / 720.0;
    poly = poly * r + 1.0 / 120.0;
    poly = poly * r + 1.0 / 24.0;
    poly = poly * r + 1.0 / 6.0;
    poly = poly * r + 0.5;
    poly = poly * r + 1.0;
    poly * r + 1.0
}

/// Degree-11 Chebyshev economization of the degree-16 Taylor series on
/// |r| <= 0.3466 (exact rational arithmetic, rounded once): max relative
/// error 4.3e-16 in Estrin order, measured against a 40-digit `exp`.
#[inline(always)]
fn exp_estrin(r: f64) -> f64 {
    const C: [f64; 12] = [
        1.0,
        1.0,
        0.5000000000000019,
        0.16666666666666702,
        0.041666666666487974,
        0.008333333333309512,
        0.001388888895234707,
        0.00019841269909250833,
        2.4801485448150653e-05,
        2.7557224927468285e-06,
        2.763265216954066e-07,
        2.511487974429777e-08,
    ];
    let r2 = r * r;
    let r4 = r2 * r2;
    let r8 = r4 * r4;
    let p01 = C[0] + C[1] * r;
    let p23 = C[2] + C[3] * r;
    let p45 = C[4] + C[5] * r;
    let p67 = C[6] + C[7] * r;
    let p89 = C[8] + C[9] * r;
    let pab = C[10] + C[11] * r;
    let q0 = p01 + p23 * r2;
    let q1 = p45 + p67 * r2;
    let q2 = p89 + pab * r2;
    (q0 + q1 * r4) + q2 * r8
}

/// Branch-free `exp`: Cody-Waite reduction by ln 2 with the round-to-nearest
/// shifter trick, a polynomial on |r| <= ~0.35, and 2^n assembled in the
/// exponent bits. Input clamped to [-708, 709].
#[inline(always)]
pub fn vexp(x: f64) -> f64 {
    const SHIFT: f64 = 6_755_399_441_055_744.0; // 1.5 * 2^52
    const LOG2E: f64 = std::f64::consts::LOG2_E;
    // fdlibm's split of ln 2: n * LN2_HI is exact for |n| < 2^11
    const LN2_HI: f64 = 6.931_471_803_691_238_164_90e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_700_02e-10;
    let x = fmin(fmax(x, -708.0), 709.0);
    let t = x * LOG2E + SHIFT;
    let n = t - SHIFT;
    let r = (x - n * LN2_HI) - n * LN2_LO;
    let poly = if ESTRIN { exp_estrin(r) } else { exp_horner(r) };
    // t's mantissa holds n (|n| < 2^51), so the bit difference is n
    let bits = t.to_bits().wrapping_sub(SHIFT.to_bits()).wrapping_add(1023) << 52;
    poly * f64::from_bits(bits)
}

/// Drying fraction with the vectorizable `exp`, computed for every day and
/// selected (branch-free).
#[inline(always)]
pub fn fraction_vexp(temperature: f64, scale: f64) -> f64 {
    let heat = 0.968 * vexp(0.0875 * temperature + 1.5552) - 8.30;
    let k = fmax(heat * scale, 0.0);
    if temperature >= WARM { k } else { 0.0 }
}

/// Drying fraction with the platform `exp`, called on warm days only.
#[inline(always)]
pub fn fraction_crt(temperature: f64, scale: f64) -> f64 {
    if temperature >= WARM {
        fmax((0.968 * (0.0875 * temperature + 1.5552).exp() - 8.30) * scale, 0.0)
    } else {
        0.0
    }
}

/// Table-driven `exp(0.0875 T + 1.5552) * 0.968` (the 0.968 folded into the
/// argument as `+ ln 0.968`): `x = (m + j/256) ln 2 + r` with `|r| <= ln2/512`,
/// `2^(j/256)` from a 256-entry table (two scalar loads per NEON vector), and
/// a degree-4 polynomial (truncation `r^5/120` < 4e-17 relative). Only an upper
/// clamp: lanes below 10 C are discarded by the caller's select, so the
/// garbage a very negative argument produces never reaches the index.
#[inline(always)]
pub fn heat_texp(temperature: f64) -> f64 {
    use crate::exp_table::{EXP2_TABLE_BITS, HEAT_OFFSET};
    const SHIFT: f64 = 6_755_399_441_055_744.0; // 1.5 * 2^52
    const N_OVER_LN2: f64 = 256.0 / std::f64::consts::LN_2;
    // ln 2 / 256 split so that n * HI is exact for |n| < 2^21
    const LN2_HI: f64 = 6.931_471_803_691_238_164_90e-1 / 256.0;
    const LN2_LO: f64 = 1.908_214_929_270_587_700_02e-10 / 256.0;
    let x = fmin(0.0875 * temperature + HEAT_OFFSET, 709.0);
    let t = x * N_OVER_LN2 + SHIFT;
    let n = t - SHIFT;
    let r = (x - n * LN2_HI) - n * LN2_LO;
    let bits = t.to_bits().wrapping_sub(SHIFT.to_bits());
    let j = (bits & 255) as usize;
    let m = ((bits as i64) >> 8) << 52;
    let s = f64::from_bits(EXP2_TABLE_BITS[j].wrapping_add(m as u64));
    let r2 = r * r;
    let poly = r + r2 * ((0.5 + r * (1.0 / 6.0)) + r2 * (1.0 / 24.0));
    (s + s * poly) - 8.30
}

/// The drying fraction for one temperature and cell scale.
pub trait Frac {
    fn k(temperature: f64, scale: f64) -> f64;
}

/// Polynomial `vexp`.
pub struct Vexp;
impl Frac for Vexp {
    #[inline(always)]
    fn k(temperature: f64, scale: f64) -> f64 {
        fraction_vexp(temperature, scale)
    }
}

/// Table-driven `heat_texp`.
pub struct Texp;
impl Frac for Texp {
    #[inline(always)]
    fn k(temperature: f64, scale: f64) -> f64 {
        let k = fmax(heat_texp(temperature) * scale, 0.0);
        if temperature >= WARM { k } else { 0.0 }
    }
}

/// Wet-spell bookkeeping: the day's net rain and the new wet-spell total.
/// Same values as the production branches.
#[inline(always)]
fn rain(wet: f64, precipitation: f64) -> (f64, f64) {
    let rained = precipitation > 0.0;
    let event = wet + precipitation;
    let crossing = fmax(event - RAIN, 0.0);
    let net = if wet > RAIN { precipitation } else { crossing };
    if rained { (net, event) } else { (0.0, 0.0) }
}

/// The index update on the dependency chain.
#[inline(always)]
fn index(q: f64, net: f64, k: f64) -> f64 {
    let q1 = fmax(q - net, 0.0);
    fmin(q1 + (MAX - q1) * k, MAX)
}

#[inline(always)]
fn step(q: f64, wet: f64, precipitation: f64, k: f64) -> (f64, f64) {
    let (net, wet) = rain(wet, precipitation);
    (index(q, net, k), wet)
}

/// One cell at a time, platform `exp` on warm days, short chain.
pub fn cm_scalar(p: &[f64], t: &[f64], map: &[f64], days: usize, out: &mut [f64]) {
    for (cell, &m) in map.iter().enumerate() {
        let s = scale(m);
        let span = cell * days..(cell + 1) * days;
        let (mut q, mut wet) = (0.0, 0.0);
        for ((&p, &t), o) in p[span.clone()].iter().zip(&t[span.clone()]).zip(&mut out[span]) {
            (q, wet) = step(q, wet, p, fraction_crt(t, s));
            *o = q;
        }
    }
}

const BLOCK: usize = 512;

/// One cell at a time; per block of days the fractions are computed first
/// (vectorized along days), then the short-chain recurrence reads them.
pub fn cm_block<F: Frac>(p: &[f64], t: &[f64], map: &[f64], days: usize, out: &mut [f64]) {
    let mut kbuf = [0.0_f64; BLOCK];
    for (cell, &m) in map.iter().enumerate() {
        let s = scale(m);
        let (mut q, mut wet) = (0.0, 0.0);
        for start in (0..days).step_by(BLOCK) {
            let end = (start + BLOCK).min(days);
            let at = cell * days;
            let (pb, tb) = (&p[at + start..at + end], &t[at + start..at + end]);
            let ob = &mut out[at + start..at + end];
            let kb = &mut kbuf[..end - start];
            for (k, &t) in kb.iter_mut().zip(tb) {
                *k = F::k(t, s);
            }
            for ((&p, &k), o) in pb.iter().zip(kb.iter()).zip(ob) {
                (q, wet) = step(q, wet, p, k);
                *o = q;
            }
        }
    }
}

/// Advance `L` independent series together for `len` days. Lane `l` reads
/// day `d` at `p[l] + d * stride` and `src[l] + d * stride` (temperature or a
/// precomputed fraction, mapped to the fraction by `fraction(value, lane)`)
/// and writes `out[l] + d * stride`. `stride` is 1 for cell-major series and
/// the row width for day-major columns.
///
/// # Safety
/// Every pointer must be valid for `len` strided elements and the output
/// elements must not overlap each other or the inputs.
#[inline(always)]
unsafe fn lanes<const L: usize>(
    p: [*const f64; L],
    src: [*const f64; L],
    out: [*mut f64; L],
    len: usize,
    stride: usize,
    q: &mut [f64; L],
    wet: &mut [f64; L],
    fraction: impl Fn(f64, usize) -> f64,
) {
    let (mut qs, mut ws) = (*q, *wet);
    for d in 0..len {
        // SAFETY: the caller guarantees `len` valid elements per lane
        let pv: [f64; L] = std::array::from_fn(|l| unsafe { *p[l].add(d * stride) });
        let sv: [f64; L] = std::array::from_fn(|l| unsafe { *src[l].add(d * stride) });
        let kv: [f64; L] = std::array::from_fn(|l| fraction(sv[l], l));
        for l in 0..L {
            (qs[l], ws[l]) = step(qs[l], ws[l], pv[l], kv[l]);
        }
        for l in 0..L {
            // SAFETY: as above; lanes write disjoint ranges
            unsafe { *out[l].add(d * stride) = qs[l] };
        }
    }
    *q = qs;
    *wet = ws;
}

/// Cell-major `[cell][day]`, `L` cells in lanes; leftover cells one at a time.
pub fn cm_lanes<const L: usize, F: Frac>(p: &[f64], t: &[f64], map: &[f64], days: usize, out: &mut [f64]) {
    let cells = map.len();
    assert!(p.len() == cells * days && t.len() == cells * days && out.len() == cells * days);
    let groups = cells / L;
    let out_ptr = out.as_mut_ptr();
    for g in 0..groups {
        let base = g * L;
        let scales: [f64; L] = std::array::from_fn(|l| scale(map[base + l]));
        let pp = std::array::from_fn(|l| p[(base + l) * days..].as_ptr());
        let tp = std::array::from_fn(|l| t[(base + l) * days..].as_ptr());
        // SAFETY: base + l < cells, so each lane's `days` outputs are in bounds and disjoint
        let op = std::array::from_fn(|l| unsafe { out_ptr.add((base + l) * days) });
        let (mut q, mut wet) = ([0.0; L], [0.0; L]);
        // SAFETY: lengths asserted above
        unsafe { lanes(pp, tp, op, days, 1, &mut q, &mut wet, |t, l| F::k(t, scales[l])) };
    }
    let rest = groups * L;
    cm_block::<F>(&p[rest * days..], &t[rest * days..], &map[rest..], days, &mut out[rest * days..]);
}

/// Probe (not KBDI): the drying fraction of every cell-day, nothing else.
pub fn probe_fraction<F: Frac>(_p: &[f64], t: &[f64], map: &[f64], days: usize, out: &mut [f64]) {
    for (cell, &m) in map.iter().enumerate() {
        let s = scale(m);
        let span = cell * days..(cell + 1) * days;
        for (o, &t) in out[span.clone()].iter_mut().zip(&t[span]) {
            *o = F::k(t, s);
        }
    }
}

/// Probe (not KBDI): the `L`-lane recurrence reading `t` as if it were the
/// precomputed fraction (no `exp`), to size the recurrence alone.
pub fn probe_lanes<const L: usize>(p: &[f64], t: &[f64], map: &[f64], days: usize, out: &mut [f64]) {
    let out_ptr = out.as_mut_ptr();
    for g in 0..map.len() / L {
        let base = g * L;
        let pp = std::array::from_fn(|l| p[(base + l) * days..].as_ptr());
        let kp = std::array::from_fn(|l| t[(base + l) * days..].as_ptr());
        // SAFETY: as in `cm_lanes`
        let op = std::array::from_fn(|l| unsafe { out_ptr.add((base + l) * days) });
        let (mut q, mut wet) = ([0.0; L], [0.0; L]);
        unsafe { lanes(pp, kp, op, days, 1, &mut q, &mut wet, |k, _| k * 1e-4) };
    }
}

/// A composed run of daily maps, `x -> min(hi, max(lo, a x + b))`.
#[derive(Clone, Copy)]
struct Map {
    a: f64,
    b: f64,
    lo: f64,
    hi: f64,
}

impl Map {

    /// Append one day: `q1 = max(q - net, 0)`, `q = min(203.2, q1 + (203.2 - q1) k)`,
    /// which is `clamp((1 - k) q + 203.2 k - (1 - k) net, 203.2 k, 203.2)`.
    #[inline(always)]
    fn then(self, net: f64, k: f64) -> Map {
        let alpha = 1.0 - k;
        let lo = MAX * k;
        let beta = lo - alpha * net;
        Map {
            a: alpha * self.a,
            b: alpha * self.b + beta,
            lo: fmin(fmax(alpha * self.lo + beta, lo), MAX),
            hi: fmin(fmax(alpha * self.hi + beta, lo), MAX),
        }
    }

    fn apply(self, x: f64) -> f64 {
        fmin(self.hi, fmax(self.lo, self.a * x + self.b))
    }
}

/// One series, time split into `L` chunks run in lanes (see the module docs).
/// `kbuf` is scratch.
///
/// `FUSE`: compute the fractions inside pass 1 (in lanes, stored for pass 3)
/// instead of in a separate pass first, so `exp` overlaps the composition.
fn scan_series<const L: usize, F: Frac, const FUSE: bool>(
    p: &[f64],
    t: &[f64],
    s: f64,
    out: &mut [f64],
    kbuf: &mut Vec<f64>,
) {
    let n = p.len();
    kbuf.clear();
    if FUSE {
        kbuf.resize(n, 0.0);
    } else {
        kbuf.extend(t.iter().map(|&t| F::k(t, s)));
    }

    // chunk starts, each right after a dry day so the wet spell is empty
    let mut starts = [0usize; L];
    for j in 1..L {
        let mut i = (j * n / L).max(starts[j - 1] + 1);
        while i < n && p[i - 1] > 0.0 {
            i += 1;
        }
        starts[j] = i;
    }
    let ends: [usize; L] = std::array::from_fn(|j| if j + 1 < L { starts[j + 1] } else { n });
    if n < 4 * L || (0..L).any(|j| starts[j] >= ends[j]) {
        // too short or no dry days to split at: plain recurrence
        if FUSE {
            for (k, &t) in kbuf.iter_mut().zip(t) {
                *k = F::k(t, s);
            }
        }
        let (mut q, mut wet) = (0.0, 0.0);
        for ((&p, &k), o) in p.iter().zip(kbuf.iter()).zip(out.iter_mut()) {
            (q, wet) = step(q, wet, p, k);
            *o = q;
        }
        return;
    }
    let common = (0..L).map(|j| ends[j] - starts[j]).min().unwrap();

    // pass 1: each chunk's composed map, chunks in lanes
    // struct of arrays, so the lanes vectorize
    let (mut ma, mut mb) = ([1.0; L], [0.0; L]);
    let (mut mlo, mut mhi) = ([f64::NEG_INFINITY; L], [f64::INFINITY; L]);
    let mut wet = [0.0; L];
    let pp: [*const f64; L] = std::array::from_fn(|l| p[starts[l]..].as_ptr());
    let tp: [*const f64; L] = std::array::from_fn(|l| t[starts[l]..].as_ptr());
    let kw = kbuf.as_mut_ptr();
    let kp: [*const f64; L] = std::array::from_fn(|l| kbuf[starts[l]..].as_ptr());
    for d in 0..common {
        // SAFETY: starts[l] + common <= ends[l] <= n, for every array of length n
        let pv: [f64; L] = std::array::from_fn(|l| unsafe { *pp[l].add(d) });
        let kv: [f64; L] = if FUSE {
            let kv = std::array::from_fn(|l| F::k(unsafe { *tp[l].add(d) }, s));
            for l in 0..L {
                unsafe { *kw.add(starts[l] + d) = kv[l] };
            }
            kv
        } else {
            std::array::from_fn(|l| unsafe { *kp[l].add(d) })
        };
        for l in 0..L {
            let (net, w) = rain(wet[l], pv[l]);
            wet[l] = w;
            let m = Map { a: ma[l], b: mb[l], lo: mlo[l], hi: mhi[l] }.then(net, kv[l]);
            (ma[l], mb[l], mlo[l], mhi[l]) = (m.a, m.b, m.lo, m.hi);
        }
    }
    let mut maps: [Map; L] = std::array::from_fn(|l| Map { a: ma[l], b: mb[l], lo: mlo[l], hi: mhi[l] });
    if FUSE {
        // the tails' fractions (the last chunk's tail has no map to compose)
        for l in 0..L {
            for d in starts[l] + common..ends[l] {
                kbuf[d] = F::k(t[d], s);
            }
        }
    }
    let k = kbuf.as_slice();
    for l in 0..L - 1 {
        for d in starts[l] + common..ends[l] {
            let (net, w) = rain(wet[l], p[d]);
            wet[l] = w;
            maps[l] = maps[l].then(net, k[d]);
        }
    }

    // pass 2: chain the chunk start values
    let mut q = [0.0; L];
    for j in 1..L {
        q[j] = maps[j - 1].apply(q[j - 1]);
    }

    // pass 3: rerun every chunk from its start value, chunks in lanes
    let mut wet = [0.0; L];
    let out_ptr = out.as_mut_ptr();
    // SAFETY: chunks are disjoint ranges of `out`, each at least `common` long
    let op = std::array::from_fn(|l| unsafe { out_ptr.add(starts[l]) });
    unsafe { lanes(pp, kp, op, common, 1, &mut q, &mut wet, |k, _| k) };
    for l in 0..L {
        let (mut ql, mut wl) = (q[l], wet[l]);
        for d in starts[l] + common..ends[l] {
            (ql, wl) = step(ql, wl, p[d], k[d]);
            out[d] = ql;
        }
    }
}

/// Every cell with the time-parallel scan.
pub fn cm_scan<const L: usize, F: Frac, const FUSE: bool>(p: &[f64], t: &[f64], map: &[f64], days: usize, out: &mut [f64]) {
    let mut kbuf = Vec::with_capacity(days);
    for (cell, &m) in map.iter().enumerate() {
        let span = cell * days..(cell + 1) * days;
        scan_series::<L, F, FUSE>(&p[span.clone()], &t[span.clone()], scale(m), &mut out[span], &mut kbuf);
    }
}

/// Cell-major dispatcher: fewer cells than lanes uses the time scan.
pub fn cm_auto<F: Frac>(p: &[f64], t: &[f64], map: &[f64], days: usize, out: &mut [f64]) {
    if map.len() < 8 {
        cm_scan::<8, F, true>(p, t, map, days, out);
    } else {
        cm_lanes::<8, F>(p, t, map, days, out);
    }
}

/// Day-major `[day][cell]` (the production layout), `L` adjacent columns in
/// lanes (contiguous within a row, so the lanes load and store as vectors),
/// over columns `cols` of rows `stride` wide; a leftover narrower than `L`
/// goes through `dm_cols`.
///
/// # Safety
/// As `dm_cols`.
pub unsafe fn dm_lanes_cols<const L: usize, F: Frac>(
    p: &[f64],
    t: &[f64],
    scales: &[f64],
    days: usize,
    stride: usize,
    cols: std::ops::Range<usize>,
    out: *mut f64,
) {
    assert!(p.len() == days * stride && t.len() == days * stride && cols.end <= stride);
    let full = cols.start + (cols.len() / L) * L;
    for base in (cols.start..full).step_by(L) {
        let sc: [f64; L] = std::array::from_fn(|l| scales[base + l]);
        let pp = std::array::from_fn(|l| p[base + l..].as_ptr());
        let tp = std::array::from_fn(|l| t[base + l..].as_ptr());
        // SAFETY: base + l < stride, rows in bounds by the assert; columns owned by the caller
        let op = std::array::from_fn(|l| unsafe { out.add(base + l) });
        let (mut q, mut wet) = ([0.0; L], [0.0; L]);
        unsafe { lanes(pp, tp, op, days, stride, &mut q, &mut wet, |t, l| F::k(t, sc[l])) };
    }
    if full < cols.end {
        // SAFETY: forwarded contract
        unsafe { dm_cols::<F>(p, t, scales, days, stride, full..cols.end, out) };
    }
}

/// Day-major, tiled: blocks of `W` columns (`W / L` lane groups) advance
/// `B` days at a time, so a tile touches `B` rows once for all its groups
/// (pages and adjacent cache lines reused) instead of each 8-column group
/// walking every row on its own. Lane state is parked in `W`-wide arrays
/// between day blocks. Leftover columns go through `dm_cols`.
///
/// # Safety
/// As `dm_cols`.
pub unsafe fn dm_tiled_cols<const L: usize, const W: usize, const B: usize, F: Frac>(
    p: &[f64],
    t: &[f64],
    scales: &[f64],
    days: usize,
    stride: usize,
    cols: std::ops::Range<usize>,
    out: *mut f64,
) {
    assert!(W % L == 0 && p.len() == days * stride && t.len() == days * stride && cols.end <= stride);
    // tiles of W columns; the last one may be narrower (a multiple of L)
    let full = cols.start + (cols.len() / L) * L;
    for block in (cols.start..full).step_by(W) {
        let width = W.min(full - block);
        let mut q = [0.0; W];
        let mut wet = [0.0; W];
        for day0 in (0..days).step_by(B) {
            let len = B.min(days - day0);
            let row = day0 * stride;
            for g in (0..width).step_by(L) {
                let base = block + g;
                let sc: [f64; L] = std::array::from_fn(|l| scales[base + l]);
                let pp = std::array::from_fn(|l| p[row + base + l..].as_ptr());
                let tp = std::array::from_fn(|l| t[row + base + l..].as_ptr());
                // SAFETY: in bounds by the assert; columns owned by the caller
                let op = std::array::from_fn(|l| unsafe { out.add(row + base + l) });
                let qg: &mut [f64; L] = (&mut q[g..g + L]).try_into().unwrap();
                let wg: &mut [f64; L] = (&mut wet[g..g + L]).try_into().unwrap();
                unsafe { lanes(pp, tp, op, len, stride, qg, wg, |t, l| F::k(t, sc[l])) };
            }
        }
    }
    if full < cols.end {
        // SAFETY: forwarded contract
        unsafe { dm_cols::<F>(p, t, scales, days, stride, full..cols.end, out) };
    }
}

pub fn dm_tiled<const L: usize, const W: usize, const B: usize, F: Frac>(
    p: &[f64],
    t: &[f64],
    map: &[f64],
    days: usize,
    out: &mut [f64],
) {
    let cells = map.len();
    assert!(out.len() == cells * days);
    let scales: Vec<f64> = map.iter().map(|&m| scale(m)).collect();
    // SAFETY: `out` holds days * cells elements and is exclusively borrowed
    unsafe { dm_tiled_cols::<L, W, B, F>(p, t, &scales, days, cells, 0..cells, out.as_mut_ptr()) };
}

pub fn dm_tiled_mt<const L: usize, const W: usize, const B: usize, F: Frac>(
    p: &[f64],
    t: &[f64],
    map: &[f64],
    days: usize,
    out: &mut [f64],
    threads: usize,
) {
    let cells = map.len();
    assert!(out.len() == cells * days);
    let scales: Vec<f64> = map.iter().map(|&m| scale(m)).collect();
    let scales = scales.as_slice();
    let base = out.as_mut_ptr();
    std::thread::scope(|s| {
        for r in ranges(cells, threads, L.max(8)) {
            let ptr = SendPtr(base);
            s.spawn(move || {
                let ptr = ptr;
                // SAFETY: disjoint column ranges of an exclusively borrowed output
                unsafe { dm_tiled_cols::<L, W, B, F>(p, t, scales, days, cells, r, ptr.0) }
            });
        }
    });
}

pub fn dm_lanes<const L: usize, F: Frac>(p: &[f64], t: &[f64], map: &[f64], days: usize, out: &mut [f64]) {
    let cells = map.len();
    assert!(out.len() == cells * days);
    let scales: Vec<f64> = map.iter().map(|&m| scale(m)).collect();
    // SAFETY: `out` holds days * cells elements and is exclusively borrowed
    unsafe { dm_lanes_cols::<L, F>(p, t, &scales, days, cells, 0..cells, out.as_mut_ptr()) };
}

pub fn dm_lanes_mt<const L: usize, F: Frac>(p: &[f64], t: &[f64], map: &[f64], days: usize, out: &mut [f64], threads: usize) {
    let cells = map.len();
    assert!(out.len() == cells * days);
    let scales: Vec<f64> = map.iter().map(|&m| scale(m)).collect();
    let scales = scales.as_slice();
    let base = out.as_mut_ptr();
    std::thread::scope(|s| {
        for r in ranges(cells, threads, L.max(8)) {
            let ptr = SendPtr(base);
            s.spawn(move || {
                let ptr = ptr;
                // SAFETY: disjoint column ranges of an exclusively borrowed output
                unsafe { dm_lanes_cols::<L, F>(p, t, scales, days, cells, r, ptr.0) }
            });
        }
    });
}

/// Day-major `[day][cell]` (the production layout) on columns `cols` of rows
/// `stride` wide, every cell of a row in turn (vectorized across cells).
///
/// # Safety
/// `out` must be valid for `days * stride` elements, and no other thread may
/// write columns `cols` of it concurrently.
pub unsafe fn dm_cols<F: Frac>(
    p: &[f64],
    t: &[f64],
    scales: &[f64],
    days: usize,
    stride: usize,
    cols: std::ops::Range<usize>,
    out: *mut f64,
) {
    let width = cols.len();
    let mut q = vec![0.0; width];
    let mut wet = vec![0.0; width];
    let scales = &scales[cols.clone()];
    for day in 0..days {
        let row = day * stride + cols.start;
        let (pr, tr) = (&p[row..row + width], &t[row..row + width]);
        // SAFETY: in bounds by the caller's contract; columns owned by this call
        let or = unsafe { std::slice::from_raw_parts_mut(out.add(row), width) };
        for (((((&p, &t), &s), q), w), o) in pr.iter().zip(tr).zip(scales).zip(&mut q).zip(&mut wet).zip(or) {
            (*q, *w) = step(*q, *w, p, F::k(t, s));
            *o = *q;
        }
    }
}

pub fn dm_simd<F: Frac>(p: &[f64], t: &[f64], map: &[f64], days: usize, out: &mut [f64]) {
    let cells = map.len();
    assert!(p.len() == cells * days && t.len() == cells * days && out.len() == cells * days);
    let scales: Vec<f64> = map.iter().map(|&m| scale(m)).collect();
    // SAFETY: `out` holds days * cells elements and is exclusively borrowed
    unsafe { dm_cols::<F>(p, t, &scales, days, cells, 0..cells, out.as_mut_ptr()) };
}

/// Contiguous ranges of `cells` for `threads` workers, each a multiple of
/// `align` except the last.
pub fn ranges(cells: usize, threads: usize, align: usize) -> Vec<std::ops::Range<usize>> {
    let mut bounds: Vec<usize> = (0..=threads).map(|i| (i * cells / threads) / align * align).collect();
    bounds[threads] = cells;
    bounds.windows(2).filter(|w| w[0] < w[1]).map(|w| w[0]..w[1]).collect()
}

pub fn cm_mt(
    kernel: fn(&[f64], &[f64], &[f64], usize, &mut [f64]),
    p: &[f64],
    t: &[f64],
    map: &[f64],
    days: usize,
    out: &mut [f64],
    threads: usize,
) {
    let parts = ranges(map.len(), threads, 8);
    std::thread::scope(|s| {
        let mut rest = out;
        for r in parts {
            let (mine, tail) = rest.split_at_mut(r.len() * days);
            rest = tail;
            let span = r.start * days..r.end * days;
            let (p, t, m) = (&p[span.clone()], &t[span], &map[r]);
            s.spawn(move || kernel(p, t, m, days, mine));
        }
    });
}

struct SendPtr(*mut f64);
// SAFETY: workers write disjoint column ranges
unsafe impl Send for SendPtr {}

pub fn dm_simd_mt<F: Frac>(p: &[f64], t: &[f64], map: &[f64], days: usize, out: &mut [f64], threads: usize) {
    let cells = map.len();
    assert!(p.len() == cells * days && t.len() == cells * days && out.len() == cells * days);
    let scales: Vec<f64> = map.iter().map(|&m| scale(m)).collect();
    let scales = scales.as_slice();
    // 8-column alignment keeps workers off each other's cache lines
    let parts = ranges(cells, threads, 8);
    let base = out.as_mut_ptr();
    std::thread::scope(|s| {
        for r in parts {
            let ptr = SendPtr(base);
            s.spawn(move || {
                let ptr = ptr;
                // SAFETY: disjoint column ranges of an exclusively borrowed output
                unsafe { dm_cols::<F>(p, t, scales, days, cells, r, ptr.0) }
            });
        }
    });
}
