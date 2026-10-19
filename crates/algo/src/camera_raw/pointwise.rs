//! The pointwise stages of the Camera Raw Filter as Photoshop 25.4 / Camera Raw 16.1.1 applies
//! them, measured as a black box on synthetic images (a 33³ colour cube, grey ramps, hue sweeps)
//! sent through the filter by script on an offline Photoshop; no Adobe code or data was used.
//!
//! What the measurements showed:
//!
//! * The filter works in linear ProPhoto. Its order: white balance and Exposure, the local tone
//!   controls, Whites / Blacks, then Calibration, the Color Mixer, Saturation and Vibrance, then
//!   Contrast, the parametric and point curves, then Color Grading (pairwise probes of the sliders
//!   composed as 3-D lookup tables: the right order reproduces the pair to ΔE 0.03–1.3, the wrong
//!   one 1–20).
//! * A *baseline rendering* sits between the scene-referred controls and the rest: a tone curve
//!   that is linear in the shadows, steepens to 1.37 stops of output per stop of light in the
//!   midtones and runs into a flat shoulder at white. The filter inverts it on the way in, so its
//!   defaults are an identity, and the scene-referred controls act on the inverted (scene)
//!   values. [`SCENE_LOG2`] holds its inverse, recovered by chaining the Exposure +1 curve down
//!   from the shadows, where the response is one stop per stop.
//! * **Exposure** is not a plain gain through that curve: its highlight shoulder depends on the
//!   amount (white stays white down to −1 stop, −2 takes it to 0.77, and +1 applied twice is not
//!   +2 near white). It is kept as the measured family of curves, one per quarter stop
//!   ([`EXPOSURE`]), interpolated between stops.
//! * **Contrast** is a curve of the encoded value applied after the baseline rendering (Exposure
//!   then Contrast composes to 1 level; the other order misses by 4000). Linear in the slider to
//!   0.4 %; the measured rows are kept ([`CONTRAST`]).
//! * The **parametric** regions are smooth bumps (Shadows over 0–50 %, Highlights over 50–100 %,
//!   Darks and Lights over the whole range) that are neither linear in their sliders nor
//!   additive; the measured rows at ±50 and ±100 are kept and interpolated ([`PARAMETRIC_*`]).
//! * The **point curve** is a straight line in the sRGB encoding between its points (residual
//!   1e-5), so that is the encoding every per-channel curve here is sampled in.
//! * Per-channel curves (Exposure, Contrast, the tone curves) act on the ProPhoto channels, not
//!   the document's; and Camera Raw keeps the hue: the per-channel result's luminance and chroma
//!   magnitude with the input's chroma direction reproduces the colours to ΔE 0.6–2.1 where the
//!   plain per-channel curve gives 3–5 and a luminance-only curve 8–15.
//! * **Saturation** scales the chroma about CIE Y in scene space, `1 + s / 100` per unit: −100 is a
//!   grey of the same luminance (ΔE 0.36), −50 ΔE 1.3.
//!
//! All tables are sampled at [`STEPS`] equal steps of the sRGB-encoded value.

use std::sync::OnceLock;

use super::filter_tables::{
    CALIBRATION, CONTRAST_HERMITE, CONTRAST_PIVOT, EXPOSURE, EXPOSURE_STOPS, GRADE_GLOBAL_DIRS, GRADE_GLOBAL_LUM, GRADE_GLOBAL_MASK, GRADE_GLOBAL_S50,
    GRADE_HIGHLIGHTS_DIRS, GRADE_HIGHLIGHTS_LUM, GRADE_HIGHLIGHTS_MASK, GRADE_HIGHLIGHTS_S50, GRADE_MIDTONES_DIRS, GRADE_MIDTONES_LUM, GRADE_MIDTONES_MASK,
    GRADE_MIDTONES_S50, GRADE_SHADOWS_DIRS, GRADE_SHADOWS_LUM, GRADE_SHADOWS_MASK, GRADE_SHADOWS_S50, HSL_POSITION, PARAMETRIC_AMOUNTS, PARAMETRIC_DARKS,
    PARAMETRIC_HIGHLIGHTS, PARAMETRIC_LIGHTS, PARAMETRIC_SHADOWS, SCENE_LOG2, STEPS, VIBRANCE_GAIN, VIBRANCE_SAT_BINS, WHITE_BALANCE,
};
use super::{CameraRaw, Encoding};

type Mat3 = [[f32; 3]; 3];

/// CIE Y of linear ProPhoto (D50).
pub(super) const PRO_Y: [f32; 3] = [0.288_040_2, 0.711_874_1, 0.000_085_7];

// Linear sRGB (Bradford-adapted to D50) ↔ linear ProPhoto.
const SRGB_TO_PRO: Mat3 = [[0.529_295_7, 0.330_408_9, 0.140_295_4], [0.098_361_2, 0.873_395_4, 0.028_243_4], [0.016_882_5, 0.117_658_4, 0.865_459_1]];
const PRO_TO_SRGB: Mat3 = [[2.034_368_7, -0.727_460_6, -0.306_908_1], [-0.228_805_1, 1.231_729_7, -0.002_924_6], [-0.008_577_4, -0.153_253_8, 1.161_831_2]];
// Linear Adobe RGB (1998) (Bradford to D50) ↔ linear ProPhoto.
const ADOBE_TO_PRO: Mat3 = [[0.714_158_5, 0.145_431, 0.140_410_5], [0.090_421_6, 0.881_335_8, 0.028_242_6], [0.0, 0.0, 1.0]];
const PRO_TO_ADOBE: Mat3 = [[1.434_303_5, -0.236_686_5, -0.194_727_9], [-0.147_152_9, 1.162_335_9, -0.015_192_3], [0.0, 0.0, 1.0]];
const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn mul(m: &Mat3, c: [f32; 3]) -> [f32; 3] {
    [m[0][0] * c[0] + m[0][1] * c[1] + m[0][2] * c[2], m[1][0] * c[0] + m[1][1] * c[1] + m[1][2] * c[2], m[2][0] * c[0] + m[2][1] * c[1] + m[2][2] * c[2]]
}

/// A transfer curve as tables: decoding over the encoded range 0..=1 (65537 entries, exact at
/// 16-bit levels) and encoding over linear 0..=1 (16385 entries, within 2e-5), each interpolated
/// linearly; values outside 0..=1 go through the formula. A pass over a 24 MP image spends most
/// of its time in these conversions, and `powf` is 20–40× slower than a lookup.
pub(super) struct Transfer {
    enc: Encoding,
    decode: Vec<f32>,
    encode: Vec<f32>,
}

const DECODE_N: usize = 65536;
const ENCODE_N: usize = 16384;

impl Transfer {
    fn build(enc: Encoding) -> Transfer {
        let decode = (0..=DECODE_N).map(|i| enc.decode(i as f32 / DECODE_N as f32)).collect();
        let encode = (0..=ENCODE_N).map(|i| enc.encode(i as f32 / ENCODE_N as f32)).collect();
        Transfer { enc, decode, encode }
    }
    /// The tables of the curve `enc` (built once per curve).
    pub(super) fn of(enc: Encoding) -> &'static Transfer {
        static SRGB: OnceLock<Transfer> = OnceLock::new();
        static PRO: OnceLock<Transfer> = OnceLock::new();
        static ADOBE: OnceLock<Transfer> = OnceLock::new();
        static LINEAR: OnceLock<Transfer> = OnceLock::new();
        static OTHER: OnceLock<std::sync::Mutex<Vec<&'static Transfer>>> = OnceLock::new();
        let g = enc.gamma();
        if g == 0.0 {
            SRGB.get_or_init(|| Transfer::build(enc))
        } else if (g - 1.8).abs() < 1e-3 {
            PRO.get_or_init(|| Transfer::build(enc))
        } else if (g - 563.0 / 256.0).abs() < 1e-3 {
            ADOBE.get_or_init(|| Transfer::build(enc))
        } else if (g - 1.0).abs() < 1e-3 {
            LINEAR.get_or_init(|| Transfer::build(enc))
        } else {
            // Any other gamma: built on first use and kept (a handful at most).
            let mut cache = OTHER.get_or_init(|| std::sync::Mutex::new(Vec::new())).lock().unwrap_or_else(|e| e.into_inner());
            if let Some(t) = cache.iter().find(|t| t.enc == enc) {
                return t;
            }
            let t: &'static Transfer = Box::leak(Box::new(Transfer::build(enc)));
            cache.push(t);
            t
        }
    }
    #[inline]
    pub(super) fn decode(&self, v: f32) -> f32 {
        if !(0.0..=1.0).contains(&v) {
            return self.enc.decode(v);
        }
        let p = v * DECODE_N as f32;
        let i = (p as usize).min(DECODE_N - 1);
        let f = p - i as f32;
        self.decode[i] + (self.decode[i + 1] - self.decode[i]) * f
    }
    #[inline]
    pub(super) fn encode(&self, v: f32) -> f32 {
        if !(0.0..=1.0).contains(&v) {
            return self.enc.encode(v);
        }
        let p = v * ENCODE_N as f32;
        let i = (p as usize).min(ENCODE_N - 1);
        let f = p - i as f32;
        self.encode[i] + (self.encode[i + 1] - self.encode[i]) * f
    }
}

/// sRGB encoding of a linear value, by table.
#[inline]
pub(super) fn enc_srgb(v: f32) -> f32 {
    Transfer::of(Encoding::of(0.0)).encode(v)
}

/// sRGB decoding of an encoded value, by table.
#[inline]
pub(super) fn dec_srgb(v: f32) -> f32 {
    Transfer::of(Encoding::of(0.0)).decode(v)
}

/// The document's colour space: its transfer curve and the matrices to and from linear ProPhoto
/// (the primaries follow the curve as the engine reports it: ProPhoto's 1.8, Adobe RGB's
/// 563/256, else sRGB's).
#[derive(Clone, Copy)]
pub(super) struct Space {
    transfer: &'static Transfer,
    to_pro: &'static Mat3,
    from_pro: &'static Mat3,
}

impl Space {
    pub(super) fn of(encoding_gamma: f32) -> Self {
        let enc = Encoding::of(encoding_gamma);
        let (to_pro, from_pro) = if (encoding_gamma - 1.8).abs() < 0.01 {
            (&IDENTITY, &IDENTITY)
        } else if (encoding_gamma - 563.0 / 256.0).abs() < 0.01 {
            (&ADOBE_TO_PRO, &PRO_TO_ADOBE)
        } else {
            (&SRGB_TO_PRO, &PRO_TO_SRGB)
        };
        Space { transfer: Transfer::of(enc), to_pro, from_pro }
    }
    /// Document pixel → linear ProPhoto.
    #[inline]
    pub(super) fn to_pro(self, q: [f32; 3]) -> [f32; 3] {
        mul(self.to_pro, q.map(|v| self.transfer.decode(v)))
    }
    /// Linear ProPhoto → document pixel.
    #[inline]
    pub(super) fn to_document(self, c: [f32; 3]) -> [f32; 3] {
        mul(self.from_pro, c).map(|v| self.transfer.encode(v))
    }
    /// The document's transfer tables.
    #[inline]
    pub(super) fn transfer(self) -> &'static Transfer {
        self.transfer
    }
    /// Document pixel → linear light with sRGB primaries.
    #[inline]
    pub(super) fn to_linear_srgb(self, q: [f32; 3]) -> [f32; 3] {
        mul(&PRO_TO_SRGB, self.to_pro(q))
    }
}

/// Linear interpolation in a [`STEPS`] table over 0..=1, extended linearly outside.
#[inline]
pub(super) fn table(t: &[f32; STEPS], x: f32) -> f32 {
    let n = (STEPS - 1) as f32;
    if x <= 0.0 {
        return t[0] + (t[1] - t[0]) * x * n;
    }
    if x >= 1.0 {
        return t[STEPS - 1] + (t[STEPS - 1] - t[STEPS - 2]) * (x - 1.0) * n;
    }
    let p = x * n;
    let i = p as usize;
    let f = p - i as f32;
    t[i] + (t[i + 1] - t[i]) * f
}

/// A row of a family at `amount`, interpolated between the measured amounts (held at the ends).
fn family_row<const R: usize>(amounts: &[f32; R], rows: &[[f32; STEPS]; R], amount: f32, out: &mut [f32; STEPS]) {
    if amount <= amounts[0] {
        *out = rows[0];
        return;
    }
    if amount >= amounts[R - 1] {
        *out = rows[R - 1];
        return;
    }
    let k = amounts.windows(2).position(|w| amount <= w[1]).unwrap_or(R - 2);
    let t = (amount - amounts[k]) / (amounts[k + 1] - amounts[k]);
    for i in 0..STEPS {
        out[i] = rows[k][i] + (rows[k + 1][i] - rows[k][i]) * t;
    }
}

fn identity_row() -> [f32; STEPS] {
    std::array::from_fn(|i| i as f32 / (STEPS - 1) as f32)
}

// ---------------------------------------------------------------- scene space

/// Scene value of an output-linear value (the baseline rendering inverted).
#[inline]
pub(super) fn scene(lin: f32) -> f32 {
    if lin <= 0.0 {
        return lin;
    }
    let x = enc_srgb(lin.min(1.0));
    let rel = table(&SCENE_LOG2, x);
    lin * rel.exp2()
}

/// Output-linear value of a scene value (the baseline rendering), from a table over log₂ u
/// built once from [`scene`].
#[inline]
pub(super) fn rendered(u: f32) -> f32 {
    static INV: OnceLock<Vec<f32>> = OnceLock::new();
    const LO: f32 = -16.0;
    const HI: f32 = 8.0;
    const N: usize = 8192;
    let inv = INV.get_or_init(|| {
        // scene(lin) is monotone: bisect it for each log₂ u node.
        (0..=N)
            .map(|k| {
                let target = (LO + (HI - LO) * k as f32 / N as f32).exp2();
                let (mut a, mut b) = (0.0f32, 1.0f32);
                if scene(1.0) < target {
                    // Above the measured range the shoulder is held: lin grows as u does.
                    return 1.0 + (target - scene(1.0)) / scene(1.0).max(1e-6);
                }
                for _ in 0..40 {
                    let m = 0.5 * (a + b);
                    if scene(m) < target { a = m } else { b = m }
                }
                0.5 * (a + b)
            })
            .collect()
    });
    if u <= 0.0 {
        return u;
    }
    let p = ((u.log2() - LO) / (HI - LO) * N as f32).clamp(0.0, N as f32);
    let i = (p as usize).min(N - 1);
    let f = p - i as f32;
    inv[i] + (inv[i + 1] - inv[i]) * f
}

// ---------------------------------------------------------------- hue

/// The per-channel result `out` with the input's chroma direction: the result's luminance and
/// chroma magnitude about CIE Y, the input's hue (Camera Raw keeps hues through its curves).
#[inline]
pub(super) fn hue_kept(input: [f32; 3], out: [f32; 3]) -> [f32; 3] {
    let y_in = PRO_Y[0] * input[0] + PRO_Y[1] * input[1] + PRO_Y[2] * input[2];
    let y_out = PRO_Y[0] * out[0] + PRO_Y[1] * out[1] + PRO_Y[2] * out[2];
    let d_in = [input[0] - y_in, input[1] - y_in, input[2] - y_in];
    let d_out = [out[0] - y_out, out[1] - y_out, out[2] - y_out];
    let n_in = (d_in[0] * d_in[0] + d_in[1] * d_in[1] + d_in[2] * d_in[2]).sqrt();
    if n_in < 1e-7 {
        return out;
    }
    let n_out = (d_out[0] * d_out[0] + d_out[1] * d_out[1] + d_out[2] * d_out[2]).sqrt();
    let s = n_out / n_in;
    [y_out + s * d_in[0], y_out + s * d_in[1], y_out + s * d_in[2]]
}

/// A per-channel curve (sRGB-encoded in and out) on linear ProPhoto, hue kept. Values above 1
/// (32-bit documents) are beyond the measured range and get the gain `over` instead.
#[inline]
pub(super) fn per_channel(c: [f32; 3], curve: &[f32; STEPS], over: f32) -> [f32; 3] {
    let out = c.map(|v| if v > 1.0 { v * over } else { dec_srgb(table(curve, enc_srgb(v))) });
    hue_kept(c, out)
}

// ---------------------------------------------------------------- stages

/// The Exposure curve at `stops` (sRGB-encoded in → out).
pub(super) fn exposure_curve(stops: f32) -> [f32; STEPS] {
    let mut row = identity_row();
    family_row(&EXPOSURE_STOPS, &EXPOSURE, stops, &mut row);
    row
}

/// Contrast, the parametric regions and the point curves: `master` is the curve shared by the
/// channels (Contrast, the regions, the master point curve; applied with the hue kept) and
/// `channels` the red, green and blue point curves (applied as they are: they are meant to shift
/// hues). Each is `None` when neutral.
pub(super) struct ToneCurves {
    pub(super) master: Option<[f32; STEPS]>,
    pub(super) channels: Option<[[f32; STEPS]; 3]>,
}

pub(super) fn tone_curves(p: &CameraRaw, pivot: f32) -> Option<ToneCurves> {
    let mut curve = identity_row();
    let mut any = false;
    if p.contrast != 0.0 {
        curve = contrast_curve(p.contrast, pivot);
        any = true;
    }
    let regions = [
        (p.curve_shadows, &PARAMETRIC_SHADOWS),
        (p.curve_darks, &PARAMETRIC_DARKS),
        (p.curve_lights, &PARAMETRIC_LIGHTS),
        (p.curve_highlights, &PARAMETRIC_HIGHLIGHTS),
    ];
    if regions.iter().any(|(a, _)| *a != 0.0) {
        let mut delta = [0.0f32; STEPS];
        let mut row = [0.0f32; STEPS];
        for (amount, rows) in regions {
            if amount == 0.0 {
                continue;
            }
            // Between 0 and ±50 the change scales with the slider (its first measured row).
            let a = amount.clamp(-100.0, 100.0);
            if a.abs() < 50.0 {
                let r = if a < 0.0 { &rows[1] } else { &rows[2] };
                let k = a.abs() / 50.0;
                for i in 0..STEPS {
                    delta[i] += r[i] * k;
                }
            } else {
                family_row(&PARAMETRIC_AMOUNTS, rows, a, &mut row);
                for i in 0..STEPS {
                    delta[i] += row[i];
                }
            }
        }
        let warp = split_warp(p.curve_splits);
        let next: [f32; STEPS] = std::array::from_fn(|i| {
            let x = curve[i];
            // The bumps were measured at the default splits; other splits stretch them.
            (x + table(&delta, warp(x))).clamp(0.0, 1.0)
        });
        curve = next;
        any = true;
    }
    if !super::is_linear(&p.point_curve) {
        let lut = super::curve_lut(&p.point_curve, 1024);
        for v in curve.iter_mut() {
            *v = super::lut_eval(&lut, *v);
        }
        any = true;
    }
    let master = any.then_some(curve);
    let mut channels = [identity_row(); 3];
    let mut any_channel = false;
    for (ch, pts) in [&p.point_curve_red, &p.point_curve_green, &p.point_curve_blue].into_iter().enumerate() {
        if super::is_linear(pts) {
            continue;
        }
        let lut = super::curve_lut(pts, 1024);
        for v in channels[ch].iter_mut() {
            *v = super::lut_eval(&lut, *v);
        }
        any_channel = true;
    }
    (any || any_channel).then_some(ToneCurves { master, channels: any_channel.then_some(channels) })
}

/// The tone curves on linear ProPhoto: the master curve with the hue kept, then the channel
/// curves as they are.
#[inline]
pub(super) fn tone(c: [f32; 3], curves: &ToneCurves) -> [f32; 3] {
    let mut c = c;
    if let Some(m) = &curves.master {
        c = per_channel(c, m, 1.0);
    }
    if let Some(ch) = &curves.channels {
        c = [0, 1, 2].map(|k| dec_srgb(table(&ch[k], enc_srgb(c[k]))));
    }
    c
}

/// Maps the encoded value under splits `s` (percent) to its place under the default 25 / 50 / 75,
/// piecewise linearly, so the measured bumps stretch between the splits.
fn split_warp(s: [f32; 3]) -> impl Fn(f32) -> f32 {
    let s = s.map(|v| v.clamp(1.0, 99.0) / 100.0);
    let (a, b, c) = (s[0].min(s[1] - 0.01), s[1], s[2].max(s[1] + 0.01));
    move |x: f32| {
        if x <= a {
            x / a * 0.25
        } else if x <= b {
            0.25 + (x - a) / (b - a) * 0.25
        } else if x <= c {
            0.5 + (x - b) / (c - b) * 0.25
        } else {
            0.75 + (x - c) / (1.0 - c) * 0.25
        }
    }
}

/// Saturation in scene space: chroma about CIE Y scaled by `1 + s / 100`.
#[inline]
pub(super) fn saturate(u: [f32; 3], k: f32) -> [f32; 3] {
    let y = PRO_Y[0] * u[0] + PRO_Y[1] * u[1] + PRO_Y[2] * u[2];
    [y + k * (u[0] - y), y + k * (u[1] - y), y + k * (u[2] - y)].map(|v| v.max(0.0))
}

// ---------------------------------------------------------------- contrast

/// The Contrast pivot (sRGB-encoded) for an image whose mean log₂ linear channel value (floor
/// 1/4096) is `mean_log2`: Camera Raw's curve turns about a level that follows the image (on grey
/// ramps over backgrounds 0.2–0.9 the pivot moved 0.50–0.72; colourful images sit lower than
/// this line predicts, by up to 0.1).
pub(super) fn contrast_pivot(mean_log2: f32) -> f32 {
    (CONTRAST_PIVOT[0] + CONTRAST_PIVOT[1] * mean_log2).clamp(0.3, 0.85)
}

/// Contrast as two cubic Hermite arcs meeting at the pivot: slope `k` there and `m0`, `m1` at black
/// and white, each interpolated between the measured slider values (the curve is linear in the
/// slider to 0.4 %; the arcs fit the ramps to 50 levels of 32768 on average).
pub(super) fn contrast_curve(amount: f32, pivot: f32) -> [f32; STEPS] {
    let a = amount.clamp(-100.0, 100.0);
    let (k, m0, m1) = if a == 0.0 {
        (1.0, 1.0, 1.0)
    } else {
        let rows = &CONTRAST_HERMITE;
        // Rows are sorted by amount; 0 sits between −10 and +10 with the identity.
        let mut lo = [0.0f32, 1.0, 1.0, 1.0];
        let mut hi = lo;
        if a < 0.0 {
            hi = [0.0, 1.0, 1.0, 1.0];
            for r in rows.iter().rev() {
                if r[0] <= a {
                    lo = *r;
                    break;
                }
                hi = *r;
            }
        } else {
            lo = [0.0, 1.0, 1.0, 1.0];
            for r in rows.iter() {
                if r[0] >= a {
                    hi = *r;
                    break;
                }
                lo = *r;
            }
        }
        let t = if hi[0] != lo[0] { (a - lo[0]) / (hi[0] - lo[0]) } else { 0.0 };
        (lo[1] + (hi[1] - lo[1]) * t, lo[2] + (hi[2] - lo[2]) * t, lo[3] + (hi[3] - lo[3]) * t)
    };
    let p = pivot;
    std::array::from_fn(|i| {
        let x = i as f32 / (STEPS - 1) as f32;
        let (x0, x1, y0, y1, s0, s1) = if x <= p { (0.0, p, 0.0, p, m0, k) } else { (p, 1.0, p, 1.0, k, m1) };
        let h = x1 - x0;
        let t = (x - x0) / h;
        let (t2, t3) = (t * t, t * t * t);
        ((2.0 * t3 - 3.0 * t2 + 1.0) * y0 + (t3 - 2.0 * t2 + t) * h * s0 + (-2.0 * t3 + 3.0 * t2) * y1 + (t3 - t2) * h * s1).clamp(0.0, 1.0)
    })
}

// ---------------------------------------------------------------- white balance and calibration

fn mat_identity() -> Mat3 {
    IDENTITY
}

/// `a · b` (row-vector convention: a pixel goes through `a` first).
fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut m = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    m
}

/// `I + t (m − I)`.
fn mat_lerp_identity(m: &Mat3, t: f32) -> Mat3 {
    let mut o = IDENTITY;
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = IDENTITY[i][j] + t * (m[i][j] - IDENTITY[i][j]);
        }
    }
    o
}

fn mat_lerp(a: &Mat3, b: &Mat3, t: f32) -> Mat3 {
    let mut o = IDENTITY;
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[i][j] + t * (b[i][j] - a[i][j]);
        }
    }
    o
}

/// Row-vector product `u · m`.
#[inline]
fn row_mul(u: [f32; 3], m: &Mat3) -> [f32; 3] {
    [u[0] * m[0][0] + u[1] * m[1][0] + u[2] * m[2][0], u[0] * m[0][1] + u[1] * m[1][1] + u[2] * m[2][1], u[0] * m[0][2] + u[1] * m[1][2] + u[2] * m[2][2]]
}

/// A slider's matrix from its measured ±50 / ±100 matrices (`plus100, plus50, minus50, minus100`).
fn slider_matrix(v: f32, plus100: &Mat3, plus50: &Mat3, minus50: &Mat3, minus100: &Mat3) -> Mat3 {
    let v = v.clamp(-100.0, 100.0);
    if v == 0.0 {
        mat_identity()
    } else if v > 0.0 {
        if v <= 50.0 { mat_lerp_identity(plus50, v / 50.0) } else { mat_lerp(plus50, plus100, (v - 50.0) / 50.0) }
    } else if v >= -50.0 {
        mat_lerp_identity(minus50, -v / 50.0)
    } else {
        mat_lerp(minus50, minus100, (-v - 50.0) / 50.0)
    }
}

/// The scene-space matrix of Temperature and Tint (the measured ±50 / ±100 matrices interpolated;
/// an approximation: Camera Raw's white balance on rendered images is not a matrix, these fit the
/// colour cube to ΔE 5–8 at ±50).
pub(super) fn white_balance_matrix(temperature: f32, tint: f32) -> Option<Mat3> {
    if temperature == 0.0 && tint == 0.0 {
        return None;
    }
    let t = slider_matrix(temperature, &WHITE_BALANCE[0], &WHITE_BALANCE[1], &WHITE_BALANCE[2], &WHITE_BALANCE[3]);
    let n = slider_matrix(tint, &WHITE_BALANCE[4], &WHITE_BALANCE[5], &WHITE_BALANCE[6], &WHITE_BALANCE[7]);
    Some(mat_mul(&t, &n))
}

/// The scene-space matrix of the Calibration primaries (each slider's ±100 matrix scaled toward
/// the identity; measured ΔE 1.7–5 on the colour cube at ±100).
pub(super) fn calibration_matrix(p: &CameraRaw) -> Option<Mat3> {
    let sliders = [
        p.calibration_red_hue,
        p.calibration_red_saturation,
        p.calibration_green_hue,
        p.calibration_green_saturation,
        p.calibration_blue_hue,
        p.calibration_blue_saturation,
    ];
    if sliders.iter().all(|v| *v == 0.0) {
        return None;
    }
    let mut m = mat_identity();
    for (k, v) in sliders.iter().enumerate() {
        let v = v.clamp(-100.0, 100.0);
        if v == 0.0 {
            continue;
        }
        let (plus, minus) = (&CALIBRATION[2 * k], &CALIBRATION[2 * k + 1]);
        let s = if v > 0.0 { mat_lerp_identity(plus, v / 100.0) } else { mat_lerp_identity(minus, -v / 100.0) };
        m = mat_mul(&m, &s);
    }
    Some(m)
}

#[inline]
pub(super) fn apply_matrix(u: [f32; 3], m: &Mat3) -> [f32; 3] {
    row_mul(u, m).map(|v| v.max(0.0))
}

// ---------------------------------------------------------------- Color Mixer (HSL)

/// Hue (degrees) of linear ProPhoto `c` in the HSV sense, with its max and min.
#[inline]
fn hue_max_min(c: [f32; 3]) -> (f32, f32, f32) {
    let (mx, mn) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
    let d = mx - mn;
    if d <= 1e-9 {
        return (0.0, mx, mn);
    }
    let h = if mx == c[0] {
        60.0 * ((c[1] - c[2]) / d).rem_euclid(6.0)
    } else if mx == c[1] {
        60.0 * ((c[2] - c[0]) / d + 2.0)
    } else {
        60.0 * ((c[0] - c[1]) / d + 4.0)
    };
    (h, mx, mn)
}

/// RGB of hue `h` (degrees) with the given max and min.
#[inline]
fn rgb_of_hue(h: f32, mx: f32, mn: f32) -> [f32; 3] {
    let h = h.rem_euclid(360.0) / 60.0;
    let c = mx - mn;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r + mn, g + mn, b + mn]
}

/// Band position (0 … 8) of a hue in degrees (the measured mapping).
#[inline]
fn hsl_position(hue: f32) -> f32 {
    let h = hue.rem_euclid(360.0);
    let i = h as usize;
    let f = h - i as f32;
    HSL_POSITION[i] + (HSL_POSITION[(i + 1).min(360)] - HSL_POSITION[i]) * f
}

/// Hue (degrees) of a band position (the mapping inverted).
fn hsl_hue_of(pos: f32) -> f32 {
    let p = pos.rem_euclid(8.0);
    // HSL_POSITION is monotone from 0 to 8 over 0..=360°.
    let (mut lo, mut hi) = (0usize, 360usize);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if HSL_POSITION[mid] <= p { lo = mid } else { hi = mid }
    }
    let span = HSL_POSITION[hi] - HSL_POSITION[lo];
    lo as f32 + if span > 1e-6 { (p - HSL_POSITION[lo]) / span } else { 0.0 }
}

/// Band weights (hat functions in position) as the two neighbouring bands and their weights.
#[inline]
fn hsl_bands(pos: f32) -> [(usize, f32); 2] {
    let k = pos.floor();
    let f = pos - k;
    let k = (k as usize) % 8;
    [(k, 1.0 - f), ((k + 1) % 8, f)]
}

/// The Color Mixer on linear ProPhoto: hue shifts within the band (±0.9 of the way to the next
/// band at ±100, max and min kept), saturation scales the HSL chroma about `(max + min) / 2`, and
/// luminance a gain that grows with the colour's lightness and saturation (measured on hue sweeps).
pub(super) fn color_mixer(c: [f32; 3], p: &CameraRaw) -> [f32; 3] {
    let (h, mx, mn) = hue_max_min(c);
    if mx - mn <= 1e-7 {
        return c;
    }
    let pos = hsl_position(h);
    let bands = hsl_bands(pos);
    let (mut dh, mut ds, mut dl) = (0.0f32, 0.0f32, 0.0f32);
    for (k, w) in bands {
        dh += w * p.hsl_hue[k];
        ds += w * p.hsl_sat[k];
        dl += w * p.hsl_lum[k];
    }
    let mut c = c;
    let (mut mx, mut mn) = (mx, mn);
    if dh != 0.0 {
        let nh = hsl_hue_of(pos + dh / 100.0 * 0.9);
        c = rgb_of_hue(nh, mx, mn);
    }
    if ds != 0.0 {
        // Chroma about the HSL lightness; −100 leaves 3 % (the measured residual).
        let k = if ds < 0.0 { 1.0 + 0.97 * ds / 100.0 } else { 1.0 + ds / 100.0 };
        let l = 0.5 * (mx + mn);
        c = c.map(|v| (l + k * (v - l)).max(0.0));
        let (_, m2, n2) = hue_max_min(c);
        mx = m2;
        mn = n2;
    }
    if dl != 0.0 {
        let l = 0.5 * (mx + mn);
        let sat = if mx > 1e-7 { 1.0 - mn / mx } else { 0.0 };
        let gain = if dl > 0.0 { dl / 100.0 * (0.67 + 0.245 * l) } else { dl / 100.0 * (0.67 + 5.1 * l).min(4.0) };
        let f = (gain * sat.powf(0.7)).exp2();
        c = c.map(|v| v * f);
    }
    c
}

// ---------------------------------------------------------------- vibrance

/// Vibrance on output-linear ProPhoto: the chroma about the channel mean scaled by the measured
/// gain, which falls with the colour's saturation (and is not linear in the slider).
pub(super) fn vibrance(c: [f32; 3], amount: f32) -> [f32; 3] {
    let a = amount.clamp(-100.0, 100.0);
    if a == 0.0 {
        return c;
    }
    let (mx, mn) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
    if mx <= 1e-7 || mx - mn <= 1e-7 {
        return c;
    }
    let sat = 1.0 - mn / mx;
    let gain_at = |row: usize| -> f32 {
        let bins = &VIBRANCE_SAT_BINS;
        let n = bins.len() - 1;
        let mut i = 0;
        while i + 1 < n && sat >= bins[i + 1] {
            i += 1;
        }
        // Centres of the bins; interpolate between neighbours.
        let centre = |j: usize| 0.5 * (bins[j] + bins[j + 1]);
        let (j0, j1) = if sat < centre(i) { (i.saturating_sub(1), i) } else { (i, (i + 1).min(n - 1)) };
        let (c0, c1) = (centre(j0), centre(j1));
        let t = if c1 > c0 { ((sat - c0) / (c1 - c0)).clamp(0.0, 1.0) } else { 0.0 };
        VIBRANCE_GAIN[row][j0] + (VIBRANCE_GAIN[row][j1] - VIBRANCE_GAIN[row][j0]) * t
    };
    // Rows: −100, −50, +50, +100; 0 is the identity.
    let k = if a <= -50.0 {
        let t = (-a - 50.0) / 50.0;
        gain_at(1) + (gain_at(0) - gain_at(1)) * t
    } else if a < 0.0 {
        1.0 + (gain_at(1) - 1.0) * (-a / 50.0)
    } else if a <= 50.0 {
        1.0 + (gain_at(2) - 1.0) * (a / 50.0)
    } else {
        let t = (a - 50.0) / 50.0;
        gain_at(2) + (gain_at(3) - gain_at(2)) * t
    };
    let m = (c[0] + c[1] + c[2]) / 3.0;
    c.map(|v| (m + k * (v - m)).max(0.0))
}

// ---------------------------------------------------------------- Color Grading

struct WheelTables {
    dirs: &'static [[f32; 3]; 3],
    mask: &'static [f32; STEPS],
    s50: f32,
    lum: &'static [[f32; STEPS]; 2],
}

const WHEELS: [WheelTables; 4] = [
    WheelTables { dirs: &GRADE_SHADOWS_DIRS, mask: &GRADE_SHADOWS_MASK, s50: GRADE_SHADOWS_S50, lum: &GRADE_SHADOWS_LUM },
    WheelTables { dirs: &GRADE_MIDTONES_DIRS, mask: &GRADE_MIDTONES_MASK, s50: GRADE_MIDTONES_S50, lum: &GRADE_MIDTONES_LUM },
    WheelTables { dirs: &GRADE_HIGHLIGHTS_DIRS, mask: &GRADE_HIGHLIGHTS_MASK, s50: GRADE_HIGHLIGHTS_S50, lum: &GRADE_HIGHLIGHTS_LUM },
    WheelTables { dirs: &GRADE_GLOBAL_DIRS, mask: &GRADE_GLOBAL_MASK, s50: GRADE_GLOBAL_S50, lum: &GRADE_GLOBAL_LUM },
];

/// The cast direction of a wheel at `hue` degrees: the measured directions at 0, 120 and 240
/// interpolated around the circle.
fn wheel_direction(t: &WheelTables, hue: f32) -> [f32; 3] {
    let h = hue.rem_euclid(360.0) / 120.0;
    let k = (h as usize).min(2);
    let f = h - k as f32;
    let (a, b) = (t.dirs[k], t.dirs[(k + 1) % 3]);
    [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}

/// The prepared casts of the four wheels: `(mask, direction × amount)` and whether any Luminance
/// curve applies.
pub(super) struct Grading {
    casts: Vec<(&'static [f32; STEPS], [f32; 3])>,
    lum: Vec<[f32; STEPS]>,
}

pub(super) fn grading(p: &CameraRaw) -> Option<Grading> {
    let wheels = [&p.grade_shadows, &p.grade_midtones, &p.grade_highlights, &p.grade_global];
    let mut casts = Vec::new();
    let mut lum = Vec::new();
    for (w, t) in wheels.iter().zip(WHEELS.iter()) {
        let s = w.sat.clamp(0.0, 100.0);
        if s > 0.0 {
            // Saturation 50 makes `s50` of the +100 cast; linear on each side of it.
            let f = if s <= 50.0 { t.s50 * s / 50.0 } else { t.s50 + (1.0 - t.s50) * (s - 50.0) / 50.0 };
            let d = wheel_direction(t, w.hue).map(|v| v * f);
            casts.push((t.mask, d));
        }
        let l = w.lum.clamp(-100.0, 100.0);
        if l != 0.0 {
            let row = if l < 0.0 { &t.lum[0] } else { &t.lum[1] };
            let k = l.abs() / 100.0;
            lum.push(std::array::from_fn(|i| {
                let x = i as f32 / (STEPS - 1) as f32;
                x + (row[i] - x) * k
            }));
        }
    }
    (!casts.is_empty() || !lum.is_empty()).then_some(Grading { casts, lum })
}

/// Color Grading on linear ProPhoto: each wheel's cast (an additive change in units of the pixel's
/// luminance, masked by the encoded luminance), then the Luminance curves per channel.
#[inline]
pub(super) fn grade(c: [f32; 3], g: &Grading) -> [f32; 3] {
    let mut c = c;
    if !g.casts.is_empty() {
        let y = PRO_Y[0] * c[0] + PRO_Y[1] * c[1] + PRO_Y[2] * c[2];
        if y > 0.0 {
            let x = enc_srgb(y.min(1.0));
            for (mask, d) in &g.casts {
                let w = table(mask, x).max(0.0) * y;
                c[0] += w * d[0];
                c[1] += w * d[1];
                c[2] += w * d[2];
            }
            c = c.map(|v| v.max(0.0));
        }
    }
    for curve in &g.lum {
        c = c.map(|v| dec_srgb(table(curve, enc_srgb(v))));
    }
    c
}

// ---------------------------------------------------------------- the pointwise plan

/// Everything the pointwise stages need, prepared once per develop: the early stages (Exposure,
/// white balance) and the late ones (Calibration, the Color Mixer, Saturation, Vibrance, the tone
/// curves, Color Grading).
pub(super) struct Pointwise<'a> {
    p: &'a CameraRaw,
    exposure: Option<[f32; STEPS]>,
    exposure_gain: f32,
    wb: Option<Mat3>,
    calibration: Option<Mat3>,
    mixer: bool,
    sat_k: f32,
    tone: Option<ToneCurves>,
    grading: Option<Grading>,
}

impl<'a> Pointwise<'a> {
    pub(super) fn new(p: &'a CameraRaw, pivot: f32) -> Self {
        Pointwise {
            p,
            exposure: (p.exposure != 0.0).then(|| exposure_curve(p.exposure)),
            exposure_gain: p.exposure.exp2(),
            wb: white_balance_matrix(p.temperature, p.tint),
            calibration: calibration_matrix(p),
            mixer: !p.hsl_neutral(),
            sat_k: (1.0 + p.saturation.clamp(-100.0, 100.0) / 100.0).max(0.0),
            tone: tone_curves(p, pivot),
            grading: grading(p),
        }
    }
    pub(super) fn has_early(&self) -> bool {
        self.exposure.is_some() || self.wb.is_some()
    }
    pub(super) fn has_late(&self) -> bool {
        self.calibration.is_some() || self.mixer || self.p.saturation != 0.0 || self.p.vibrance != 0.0 || self.tone.is_some() || self.grading.is_some()
    }
    /// Exposure and the white balance on linear ProPhoto.
    #[inline]
    pub(super) fn early(&self, c: [f32; 3]) -> [f32; 3] {
        let mut c = c;
        if let Some(curve) = &self.exposure {
            c = per_channel(c, curve, self.exposure_gain);
        }
        if let Some(m) = &self.wb {
            c = apply_matrix(c.map(scene), m).map(rendered);
        }
        c
    }
    /// The colour stages after the local tone stages, on linear ProPhoto.
    #[inline]
    pub(super) fn late(&self, c: [f32; 3]) -> [f32; 3] {
        let p = self.p;
        let mut c = c;
        if self.calibration.is_some() || self.mixer || p.saturation != 0.0 {
            let mut u = c.map(scene);
            if let Some(m) = &self.calibration {
                u = apply_matrix(u, m);
            }
            if self.mixer {
                u = color_mixer(u, p);
            }
            if p.saturation != 0.0 {
                u = saturate(u, self.sat_k);
            }
            c = u.map(rendered);
        }
        if p.vibrance != 0.0 {
            c = vibrance(c, p.vibrance);
        }
        if let Some(curves) = &self.tone {
            c = tone(c, curves);
        }
        if let Some(g) = &self.grading {
            c = grade(c, g);
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_curve_is_linear_in_the_shadows_and_invertible() {
        for lin in [0.0005, 0.001, 0.002] {
            assert!((scene(lin) / lin - 1.0).abs() < 0.02, "{lin}");
        }
        for lin in [0.001, 0.01, 0.1, 0.3, 0.6, 0.9, 0.99] {
            let back = rendered(scene(lin));
            assert!((back - lin).abs() < 2e-3, "{lin} -> {back}");
        }
        // Midtones: a stop of output is about 0.73 stop of scene light (1.37 stops out per stop in).
        let r = (scene(0.21) / scene(0.105)).log2();
        assert!((r - 0.73).abs() < 0.1, "{r}");
    }

    #[test]
    fn contrast_turns_about_the_pivot_and_is_steeper_there() {
        let c = contrast_curve(100.0, 0.577);
        assert!((table(&c, 0.577) - 0.577).abs() < 1e-3);
        assert!(table(&c, 0.25) < 0.25 - 0.08 && table(&c, 0.8) > 0.8 + 0.05);
        let flat = contrast_curve(-100.0, 0.577);
        assert!(table(&flat, 0.25) > 0.25 + 0.08);
        assert_eq!(contrast_curve(0.0, 0.5)[128], 0.5);
        assert!((contrast_pivot(-2.93) - 0.588).abs() < 0.01);
    }

    #[test]
    fn color_mixer_bands_map_hues_both_ways() {
        for h in [0.0, 30.0, 95.0, 160.0, 230.0, 275.0, 330.0, 359.0] {
            let p = hsl_position(h);
            assert!((hsl_hue_of(p) - h).abs() < 1.5, "{h} -> {p} -> {}", hsl_hue_of(p));
        }
        // A red hue +100 turns about 0.9 of the way to orange and keeps max and min.
        let mut p = CameraRaw::default();
        p.hsl_hue[0] = 100.0;
        let out = color_mixer([0.5, 0.1, 0.1], &p);
        let (h, mx, mn) = hue_max_min(out);
        assert!((mx - 0.5).abs() < 1e-5 && (mn - 0.1).abs() < 1e-5);
        assert!(h > 20.0 && h < 30.0, "{h}");
        // Greys are untouched.
        assert_eq!(color_mixer([0.3, 0.3, 0.3], &p), [0.3, 0.3, 0.3]);
    }

    #[test]
    fn vibrance_raises_dull_colours_more_than_saturated_ones() {
        let dull = vibrance([0.5, 0.4, 0.4], 100.0);
        let vivid = vibrance([0.5, 0.05, 0.05], 100.0);
        let chroma = |c: [f32; 3]| c[0] - c[1];
        assert!(chroma(dull) / 0.1 > 1.8, "{dull:?}");
        assert!(chroma(vivid) / 0.45 < 1.3, "{vivid:?}");
        assert_eq!(vibrance([0.3, 0.3, 0.3], -100.0), [0.3, 0.3, 0.3]);
    }

    #[test]
    fn exposure_rows_interpolate_and_hold_at_the_ends() {
        let zero = exposure_curve(0.0);
        assert!((zero[128] - 0.5).abs() < 1e-6);
        let minus = exposure_curve(-1.0);
        // Measured: sRGB 0.5 goes to 0.323 at −1 stop.
        assert!((table(&minus, 0.5) - 0.323).abs() < 0.005, "{}", table(&minus, 0.5));
        let half = exposure_curve(-0.75);
        assert!(table(&half, 0.5) > table(&minus, 0.5) && table(&half, 0.5) < table(&exposure_curve(-0.5), 0.5));
        assert_eq!(exposure_curve(-9.0), exposure_curve(-5.0));
    }

    #[test]
    fn hue_is_kept_through_a_per_channel_curve() {
        let c = [0.5, 0.1, 0.05];
        let curve = exposure_curve(1.0);
        let out = per_channel(c, &curve, 2.0);
        let y_in = PRO_Y.iter().zip(c).map(|(w, v)| w * v).sum::<f32>();
        let y_out = PRO_Y.iter().zip(out).map(|(w, v)| w * v).sum::<f32>();
        let din = c.map(|v| v - y_in);
        let dout = out.map(|v| v - y_out);
        let cross = (din[0] * dout[1] - din[1] * dout[0]).abs() + (din[1] * dout[2] - din[2] * dout[1]).abs();
        assert!(cross < 1e-4, "{cross}");
        assert!(y_out > y_in);
    }
}
