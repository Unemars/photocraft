//! Filter › Camera Raw Filter: a float develop pipeline over RGB pixels.
//!
//! The pointwise stages, Detail, Defringe, the vignettes and grain follow black-box measurements
//! of Photoshop 25.4's Camera Raw Filter (Camera Raw 16.1.1) on synthetic images: a 33³ colour
//! cube, grey ramps on several backgrounds, hue sweeps, edges, gratings, noise and flat fields
//! sent through the filter by script. No Adobe code or data was used. The order of the stages
//! was established from pairs of sliders (see `pointwise`).
//!
//! Stages, in order (each skipped when its controls are neutral, so defaults are an identity):
//!
//! 1. **Exposure and white balance** on linear ProPhoto: Exposure as the measured family of
//!    curves (one per quarter stop, per channel with the hue kept); Temperature and Tint as
//!    scene-space matrices interpolated between the measured ±50 / ±100 ones (an approximation,
//!    ΔE 3–8 on the colour cube at ±50).
//! 2. **Dehaze**: K. He, J. Sun, X. Tang, *Single Image Haze Removal Using Dark Channel Prior*,
//!    CVPR 2009 (dark channel, atmospheric light from its brightest 0.1 %, transmission refined
//!    with the guided filter); negative amounts add haze toward the atmospheric light.
//! 3. **Local tone**: Highlights / Shadows as a local Laplacian filter fitted to Photoshop's (see
//!    `local_tone`); Clarity (large radius, midtone weighted) and Texture (small radius) add
//!    band-pass local contrast on luminance. Whites / Blacks follow as Photoshop's Camera Raw
//!    applies them: image-adaptive global curves (see `whites_blacks`).
//! 4. **Calibration**, the **Color Mixer** and **Saturation** in scene space, then **Vibrance**:
//!    the primaries as measured matrices (ΔE 1.5–5); the Mixer's eight bands as hat functions
//!    over the measured band positions in linear ProPhoto's hue, hue shifts keeping max and min,
//!    saturation about the HSL lightness, luminance as a gain (ΔE 0.7–2.5); Saturation as the
//!    chroma about CIE Y scaled by `1 + s / 100` (ΔE 0.4–3); Vibrance by a measured gain that
//!    falls with the colour's saturation (ΔE 3–9).
//! 5. **Contrast and the tone curves**: Contrast as two cubic arcs about a pivot that follows the
//!    image's mean log of its darkest channel; the parametric regions as measured bumps; the
//!    point curves as natural cubic splines in the sRGB encoding; all per channel in ProPhoto with
//!    the hue kept (ΔE 1–3).
//! 6. **Color Grading**: each wheel's cast as an additive change masked by the measured luminance
//!    profile (the casts on colours are an approximation, ΔE 11–28; on greys exact) and the
//!    Luminance sliders as the measured per-channel curves (ΔE 0.02). Blending and Balance are
//!    not measured yet.
//! 7. **Detail**: luminance noise reduction by a guided filter blended to the measured noise
//!    reduction per level; colour noise reduction by a chroma blur with the measured residual;
//!    sharpening as an unsharp mask on log₂ luminance with a halo limit, calibrated to the
//!    measured halos, ringing and noise gain; **Defringe** desaturating the purple / green hue
//!    windows next to luminance edges.
//! 8. **Effects**: the lens vignette (Optics), the post-crop vignette (amount, midpoint, roundness,
//!    feather, highlights, style) and grain, each from measured flat-field profiles and
//!    statistics.

#![allow(clippy::needless_range_loop)] // index loops mirror the maths (pixels × channels)

use serde::{Deserialize, Serialize};

/// Most points a Camera Raw point curve accepts from commands and the control channel.
pub const MAX_CURVE_POINTS: usize = 16;

/// Checks a point curve from untrusted input: empty (linear) or 2..=16 finite points in 0..=255
/// with inputs increasing by at least one level. Deserialization stays lenient on purpose:
/// earlier editors saved curves this rejects, and [`curve_lut`] already sorts and de-duplicates
/// them, so stored Smart Filters keep rendering.
pub fn validate_curve(points: &[[f32; 2]]) -> Result<(), String> {
    if points.len() == 1 {
        return Err("a nonempty curve needs at least two points".into());
    }
    if points.len() > MAX_CURVE_POINTS {
        return Err(format!("at most {MAX_CURVE_POINTS} curve points"));
    }
    if !points.iter().flatten().all(|v| v.is_finite() && (0.0..=255.0).contains(v)) {
        return Err("curve coordinates must be finite and in 0..=255".into());
    }
    if points.windows(2).any(|w| w[1][0] - w[0][0] < 1.0) {
        return Err("curve inputs must increase by at least one level".into());
    }
    Ok(())
}

use crate::photo_util::{hash01, linear_to_srgb, par_rows, srgb_to_linear};

mod filter_tables;
mod local_tone;
mod pointwise;
mod whites_blacks;

/// How the pixels encode light: the sRGB curve (gamma 0) or a pure power law (ProPhoto's 1.8,
/// Adobe RGB's 2.2). The tone controls work on light, so the document's own curve decodes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Encoding(f32);

impl Encoding {
    pub(crate) fn of(gamma: f32) -> Self {
        Encoding(if gamma.is_finite() && gamma > 0.0 { gamma.clamp(0.5, 4.0) } else { 0.0 })
    }
    pub(crate) fn gamma(self) -> f32 {
        self.0
    }
    /// A value in this encoding as linear light.
    pub(crate) fn decode(self, v: f32) -> f32 {
        if self.0 > 0.0 { v.max(0.0).powf(self.0) } else { srgb_to_linear(v) }
    }
    /// Linear light back into this encoding, the inverse of [`Encoding::decode`].
    pub(crate) fn encode(self, v: f32) -> f32 {
        if self.0 > 0.0 { v.max(0.0).powf(1.0 / self.0) } else { linear_to_srgb(v) }
    }
}

fn luma_linear(c: [f32; 3]) -> f32 {
    luma(c)
}

/// One colour grading wheel.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Wheel {
    /// Degrees 0..360.
    pub hue: f32,
    /// 0..=100.
    pub sat: f32,
    /// −100..=100.
    pub lum: f32,
}

impl Wheel {
    fn neutral(&self) -> bool {
        self.sat == 0.0 && self.lum == 0.0
    }
}

/// Camera Raw settings in Adobe Camera Raw's slider units.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct CameraRaw {
    // Basic
    pub temperature: f32,
    pub tint: f32,
    pub exposure: f32,
    pub contrast: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub whites: f32,
    pub blacks: f32,
    pub texture: f32,
    pub clarity: f32,
    pub dehaze: f32,
    pub vibrance: f32,
    pub saturation: f32,
    // Tone curve
    pub curve_highlights: f32,
    pub curve_lights: f32,
    pub curve_darks: f32,
    pub curve_shadows: f32,
    /// Region splits (percent).
    pub curve_splits: [f32; 3],
    /// Point curves as `[input, output]` in 0..=255 (empty = linear).
    pub point_curve: Vec<[f32; 2]>,
    pub point_curve_red: Vec<[f32; 2]>,
    pub point_curve_green: Vec<[f32; 2]>,
    pub point_curve_blue: Vec<[f32; 2]>,
    // Color mixer: red, orange, yellow, green, aqua, blue, purple, magenta.
    pub hsl_hue: [f32; 8],
    pub hsl_sat: [f32; 8],
    pub hsl_lum: [f32; 8],
    // Color grading
    pub grade_shadows: Wheel,
    pub grade_midtones: Wheel,
    pub grade_highlights: Wheel,
    pub grade_global: Wheel,
    pub grade_blending: f32,
    pub grade_balance: f32,
    // Optics: Defringe (amounts 0…20, hue windows 0…100) and the lens vignette (−100…100, midpoint 0…100).
    pub defringe_purple_amount: f32,
    pub defringe_purple_hue: [f32; 2],
    pub defringe_green_amount: f32,
    pub defringe_green_hue: [f32; 2],
    pub lens_vignette_amount: f32,
    pub lens_vignette_midpoint: f32,
    // Calibration: the primaries' hue and saturation (−100…100).
    pub calibration_red_hue: f32,
    pub calibration_red_saturation: f32,
    pub calibration_green_hue: f32,
    pub calibration_green_saturation: f32,
    pub calibration_blue_hue: f32,
    pub calibration_blue_saturation: f32,
    // Detail
    pub sharpen_amount: f32,
    pub sharpen_radius: f32,
    pub sharpen_detail: f32,
    pub sharpen_masking: f32,
    pub noise_luminance: f32,
    pub noise_luminance_detail: f32,
    pub noise_color: f32,
    pub noise_color_detail: f32,
    // Effects
    pub grain_amount: f32,
    pub grain_size: f32,
    pub grain_roughness: f32,
    pub vignette_amount: f32,
    pub vignette_midpoint: f32,
    pub vignette_roundness: f32,
    pub vignette_feather: f32,
    pub vignette_highlights: f32,
    /// `highlightPriority | colorPriority | paintOverlay`.
    pub vignette_style: String,
    pub seed: u32,
    /// Pixel-size multiplier for previews on downsampled proxies (radii in px scale by it).
    pub pixel_scale: f32,
    /// The document's transfer curve: 0 for sRGB's, else the gamma of a power-law space (1.8
    /// for ProPhoto). Set from the document's profile; Whites and Blacks act on light.
    pub encoding_gamma: f32,
}

impl Default for CameraRaw {
    fn default() -> Self {
        CameraRaw {
            temperature: 0.0,
            tint: 0.0,
            exposure: 0.0,
            contrast: 0.0,
            highlights: 0.0,
            shadows: 0.0,
            whites: 0.0,
            blacks: 0.0,
            texture: 0.0,
            clarity: 0.0,
            dehaze: 0.0,
            vibrance: 0.0,
            saturation: 0.0,
            curve_highlights: 0.0,
            curve_lights: 0.0,
            curve_darks: 0.0,
            curve_shadows: 0.0,
            curve_splits: [25.0, 50.0, 75.0],
            point_curve: Vec::new(),
            point_curve_red: Vec::new(),
            point_curve_green: Vec::new(),
            point_curve_blue: Vec::new(),
            hsl_hue: [0.0; 8],
            hsl_sat: [0.0; 8],
            hsl_lum: [0.0; 8],
            grade_shadows: Wheel::default(),
            grade_midtones: Wheel::default(),
            grade_highlights: Wheel::default(),
            grade_global: Wheel::default(),
            grade_blending: 50.0,
            grade_balance: 0.0,
            defringe_purple_amount: 0.0,
            defringe_purple_hue: [30.0, 70.0],
            defringe_green_amount: 0.0,
            defringe_green_hue: [40.0, 60.0],
            lens_vignette_amount: 0.0,
            lens_vignette_midpoint: 50.0,
            calibration_red_hue: 0.0,
            calibration_red_saturation: 0.0,
            calibration_green_hue: 0.0,
            calibration_green_saturation: 0.0,
            calibration_blue_hue: 0.0,
            calibration_blue_saturation: 0.0,
            sharpen_amount: 0.0,
            sharpen_radius: 1.0,
            sharpen_detail: 25.0,
            sharpen_masking: 0.0,
            noise_luminance: 0.0,
            noise_luminance_detail: 50.0,
            noise_color: 0.0,
            noise_color_detail: 50.0,
            grain_amount: 0.0,
            grain_size: 25.0,
            grain_roughness: 50.0,
            vignette_amount: 0.0,
            vignette_midpoint: 50.0,
            vignette_roundness: 0.0,
            vignette_feather: 50.0,
            vignette_highlights: 0.0,
            vignette_style: "highlightPriority".into(),
            seed: 0,
            pixel_scale: 1.0,
            encoding_gamma: 0.0,
        }
    }
}

/// Hue band centres of the Color Mixer (degrees).
pub const HSL_BANDS: [f32; 8] = [0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0];

fn luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Box mean of a single-channel buffer (radius `r`, edge clamped): rows in parallel, then
/// columns in parallel bands with running column sums (cache-friendly, no transposes).
fn box_mean(v: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 || w == 0 || h == 0 {
        return v.to_vec();
    }
    let n = (2 * r + 1) as f64;
    let mut tmp = vec![0.0f32; w * h];
    par_rows(&mut tmp, w, 1, |y, row| {
        let src = &v[y * w..(y + 1) * w];
        let at = |k: i64| src[k.clamp(0, w as i64 - 1) as usize] as f64;
        let mut acc: f64 = (-(r as i64)..=r as i64).map(at).sum();
        for (x, o) in row.iter_mut().enumerate() {
            *o = (acc / n) as f32;
            acc += at(x as i64 + r as i64 + 1) - at(x as i64 - r as i64);
        }
    });
    const BAND: usize = 256;
    let bands = crate::photo_util::par_map(w.div_ceil(BAND), |b| {
        let (x0, x1) = (b * BAND, ((b + 1) * BAND).min(w));
        let bw = x1 - x0;
        let row = |y: i64| &tmp[y.clamp(0, h as i64 - 1) as usize * w + x0..y.clamp(0, h as i64 - 1) as usize * w + x1];
        let mut acc = vec![0.0f64; bw];
        for y in -(r as i64)..=r as i64 {
            for (a, v) in acc.iter_mut().zip(row(y)) {
                *a += *v as f64;
            }
        }
        let mut out = vec![0.0f32; bw * h];
        for y in 0..h {
            for (o, a) in out[y * bw..(y + 1) * bw].iter_mut().zip(&acc) {
                *o = (*a / n) as f32;
            }
            let (add, sub) = (row(y as i64 + r as i64 + 1), row(y as i64 - r as i64));
            for ((a, p), m) in acc.iter_mut().zip(add).zip(sub) {
                *a += (*p - *m) as f64;
            }
        }
        out
    });
    let mut out = vec![0.0f32; w * h];
    for (b, band) in bands.iter().enumerate() {
        let (x0, x1) = (b * BAND, ((b + 1) * BAND).min(w));
        let bw = x1 - x0;
        for y in 0..h {
            out[y * w + x0..y * w + x1].copy_from_slice(&band[y * bw..(y + 1) * bw]);
        }
    }
    out
}

/// Approximate Gaussian (three box passes).
fn gauss(v: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma < 0.3 {
        return v.to_vec();
    }
    let mut out = v.to_vec();
    for r in crate::fxutil::gauss_box_radii(sigma) {
        out = box_mean(&out, w, h, r);
    }
    out
}

/// Fast guided filter (K. He, J. Sun, *Fast Guided Filter*, arXiv 2015): the linear
/// coefficients are computed on a `s×` subsampled copy and upsampled bilinearly, which is
/// visually equivalent for the large radii used by tone controls at a fraction of the cost.
fn guided_fast(i: &[f32], p: &[f32], w: usize, h: usize, r: usize, eps: f32) -> Vec<f32> {
    let s = (r / 8).clamp(1, 8);
    if s == 1 {
        return guided(i, p, w, h, r, eps);
    }
    let (sw, sh) = (w.div_ceil(s), h.div_ceil(s));
    let down = |v: &[f32]| -> Vec<f32> {
        let mut o = vec![0.0f32; sw * sh];
        let mut c = vec![0.0f32; sw * sh];
        for y in 0..h {
            for x in 0..w {
                let k = (y / s) * sw + x / s;
                o[k] += v[y * w + x];
                c[k] += 1.0;
            }
        }
        o.iter().zip(&c).map(|(a, b)| a / b.max(1.0)).collect()
    };
    let (si, sp) = (down(i), down(p));
    let rs = (r / s).max(1);
    let mi = box_mean(&si, sw, sh, rs);
    let mp = box_mean(&sp, sw, sh, rs);
    let ip: Vec<f32> = si.iter().zip(&sp).map(|(a, b)| a * b).collect();
    let ii: Vec<f32> = si.iter().map(|a| a * a).collect();
    let mip = box_mean(&ip, sw, sh, rs);
    let mii = box_mean(&ii, sw, sh, rs);
    let a: Vec<f32> = (0..sw * sh).map(|k| (mip[k] - mi[k] * mp[k]) / (mii[k] - mi[k] * mi[k] + eps)).collect();
    let b: Vec<f32> = (0..sw * sh).map(|k| mp[k] - a[k] * mi[k]).collect();
    let (ma, mb) = (box_mean(&a, sw, sh, rs), box_mean(&b, sw, sh, rs));
    let mut out = vec![0.0f32; w * h];
    let sf = s as f32;
    par_rows(&mut out, w, 1, |y, row| {
        let fy = (y as f32 + 0.5) / sf - 0.5;
        for (x, o) in row.iter_mut().enumerate() {
            let fx = (x as f32 + 0.5) / sf - 0.5;
            let av = crate::photo_util::bilinear(&ma, sw, sh, 1, 0, fx, fy);
            let bv = crate::photo_util::bilinear(&mb, sw, sh, 1, 0, fx, fy);
            *o = av * i[y * w + x] + bv;
        }
    });
    out
}

/// Guided filter of `p` guided by `i` (He et al. 2010).
fn guided(i: &[f32], p: &[f32], w: usize, h: usize, r: usize, eps: f32) -> Vec<f32> {
    let mi = box_mean(i, w, h, r);
    let mp = box_mean(p, w, h, r);
    let ip: Vec<f32> = i.iter().zip(p).map(|(a, b)| a * b).collect();
    let ii: Vec<f32> = i.iter().map(|a| a * a).collect();
    let mip = box_mean(&ip, w, h, r);
    let mii = box_mean(&ii, w, h, r);
    let a: Vec<f32> = (0..w * h).map(|k| (mip[k] - mi[k] * mp[k]) / (mii[k] - mi[k] * mi[k] + eps)).collect();
    let b: Vec<f32> = (0..w * h).map(|k| mp[k] - a[k] * mi[k]).collect();
    let ma = box_mean(&a, w, h, r);
    let mb = box_mean(&b, w, h, r);
    (0..w * h).map(|k| ma[k] * i[k] + mb[k]).collect()
}

/// The natural cubic spline through `pts` (x ascending, 0..=255), evaluated into a LUT over
/// 0..=1 and clamped to it: Camera Raw's point curve (measured on Photoshop 25.4: the spline
/// reproduces every tested curve within a level of 32768; a monotone cubic misses by up to 3000).
pub fn curve_lut(pts: &[[f32; 2]], n: usize) -> Vec<f32> {
    let mut p: Vec<[f32; 2]> = pts.iter().map(|q| [q[0] / 255.0, q[1] / 255.0]).collect();
    p.sort_by(|a, b| a[0].total_cmp(&b[0]));
    p.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-6);
    if p.len() < 2 {
        return (0..n).map(|k| k as f32 / (n - 1) as f32).collect();
    }
    let m = p.len();
    // Second derivatives of the natural spline (zero at the ends): a tridiagonal system.
    let h: Vec<f32> = (0..m - 1).map(|k| (p[k + 1][0] - p[k][0]).max(1e-6)).collect();
    let mut z = vec![0.0f32; m];
    if m > 2 {
        let mut a = vec![0.0f32; m];
        let mut b = vec![0.0f32; m];
        let mut c = vec![0.0f32; m];
        let mut r = vec![0.0f32; m];
        for k in 1..m - 1 {
            a[k] = h[k - 1];
            b[k] = 2.0 * (h[k - 1] + h[k]);
            c[k] = h[k];
            r[k] = 6.0 * ((p[k + 1][1] - p[k][1]) / h[k] - (p[k][1] - p[k - 1][1]) / h[k - 1]);
        }
        // Thomas algorithm on rows 1..m-1.
        for k in 2..m - 1 {
            let w = a[k] / b[k - 1];
            b[k] -= w * c[k - 1];
            r[k] -= w * r[k - 1];
        }
        z[m - 2] = r[m - 2] / b[m - 2];
        for k in (1..m - 2).rev() {
            z[k] = (r[k] - c[k] * z[k + 1]) / b[k];
        }
    }
    (0..n)
        .map(|i| {
            let x = i as f32 / (n - 1) as f32;
            if x <= p[0][0] {
                return p[0][1].clamp(0.0, 1.0);
            }
            if x >= p[m - 1][0] {
                return p[m - 1][1].clamp(0.0, 1.0);
            }
            let k = p.windows(2).position(|wv| x <= wv[1][0]).unwrap_or(m - 2);
            let hk = h[k];
            let (t0, t1) = (p[k + 1][0] - x, x - p[k][0]);
            let y = z[k] * t0 * t0 * t0 / (6.0 * hk)
                + z[k + 1] * t1 * t1 * t1 / (6.0 * hk)
                + (p[k + 1][1] / hk - z[k + 1] * hk / 6.0) * t1
                + (p[k][1] / hk - z[k] * hk / 6.0) * t0;
            y.clamp(0.0, 1.0)
        })
        .collect()
}

fn lut_eval(lut: &[f32], v: f32) -> f32 {
    if !(0.0..=1.0).contains(&v) {
        // Extend linearly outside (float documents).
        return if v < 0.0 { v + lut[0] } else { lut[lut.len() - 1] + (v - 1.0) };
    }
    let x = v * (lut.len() - 1) as f32;
    let i = (x as usize).min(lut.len() - 2);
    let t = x - i as f32;
    lut[i] + (lut[i + 1] - lut[i]) * t
}

/// The tone curve Contrast, the parametric regions and the point curves make, as a LUT of the
/// sRGB-encoded grey value (`n` samples; the measured curves, see `pointwise`).
pub fn parametric_lut(p: &CameraRaw, n: usize) -> Vec<f32> {
    // The preview has no image to take the pivot from: the ramps' 0.577.
    let curves = pointwise::tone_curves(p, 0.577);
    (0..n)
        .map(|i| {
            let x = i as f32 / n.saturating_sub(1).max(1) as f32;
            curves.as_ref().and_then(|c| c.master.as_ref()).map_or(x, |m| pointwise::table(m, x).clamp(0.0, 1.0))
        })
        .collect()
}

/// Weights of the eight HSL bands at hue `h` (hat functions between neighbouring centres).
pub fn band_weights(h: f32) -> [f32; 8] {
    let mut w = [0.0f32; 8];
    for k in 0..8 {
        let (c, prev, next) = (HSL_BANDS[k], HSL_BANDS[(k + 7) % 8], HSL_BANDS[(k + 1) % 8]);
        let diff = ((h - c + 540.0) % 360.0) - 180.0;
        let span = if diff >= 0.0 { (next - c).rem_euclid(360.0) } else { (c - prev).rem_euclid(360.0) };
        w[k] = (1.0 - diff.abs() / span.max(1e-3)).max(0.0);
    }
    w
}

impl CameraRaw {
    /// Rejects malformed point curves in new settings (see [`validate_curve`]).
    pub fn validate(&self) -> Result<(), String> {
        for (name, curve) in [
            ("pointCurve", &self.point_curve),
            ("pointCurveRed", &self.point_curve_red),
            ("pointCurveGreen", &self.point_curve_green),
            ("pointCurveBlue", &self.point_curve_blue),
        ] {
            validate_curve(curve).map_err(|e| format!("{name}: {e}"))?;
        }
        Ok(())
    }
    /// [`validate`](Self::validate) for an edit of `stored` settings: a curve left as stored is
    /// accepted, so editing another control of an older Smart Filter still works.
    pub fn validate_changes(&self, stored: &CameraRaw) -> Result<(), String> {
        for (name, curve, kept) in [
            ("pointCurve", &self.point_curve, &stored.point_curve),
            ("pointCurveRed", &self.point_curve_red, &stored.point_curve_red),
            ("pointCurveGreen", &self.point_curve_green, &stored.point_curve_green),
            ("pointCurveBlue", &self.point_curve_blue, &stored.point_curve_blue),
        ] {
            if curve != kept {
                validate_curve(curve).map_err(|e| format!("{name}: {e}"))?;
            }
        }
        Ok(())
    }
    fn wb_neutral(&self) -> bool {
        self.temperature == 0.0 && self.tint == 0.0 && self.exposure == 0.0
    }
    fn tone_neutral(&self) -> bool {
        self.local_tone_neutral() && self.whites == 0.0 && self.blacks == 0.0
    }
    /// Contrast, Highlights, Shadows, Texture and Clarity (the luminance stage).
    fn local_tone_neutral(&self) -> bool {
        [self.contrast, self.highlights, self.shadows, self.texture, self.clarity].iter().all(|v| *v == 0.0)
    }
    fn hsl_neutral(&self) -> bool {
        self.hsl_hue.iter().chain(&self.hsl_sat).chain(&self.hsl_lum).all(|v| *v == 0.0)
    }
    fn calibration_neutral(&self) -> bool {
        [
            self.calibration_red_hue,
            self.calibration_red_saturation,
            self.calibration_green_hue,
            self.calibration_green_saturation,
            self.calibration_blue_hue,
            self.calibration_blue_saturation,
        ]
        .iter()
        .all(|v| *v == 0.0)
    }
    fn grade_neutral(&self) -> bool {
        self.grade_shadows.neutral() && self.grade_midtones.neutral() && self.grade_highlights.neutral() && self.grade_global.neutral()
    }
    fn curves_neutral(&self) -> bool {
        [self.curve_highlights, self.curve_lights, self.curve_darks, self.curve_shadows].iter().all(|v| *v == 0.0)
            && [&self.point_curve, &self.point_curve_red, &self.point_curve_green, &self.point_curve_blue].iter().all(|c| is_linear(c))
    }
    /// True when every control is neutral (the filter is an identity).
    pub fn is_identity(&self) -> bool {
        self.wb_neutral()
            && self.tone_neutral()
            && self.dehaze == 0.0
            && self.vibrance == 0.0
            && self.saturation == 0.0
            && self.curves_neutral()
            && self.calibration_neutral()
            && self.hsl_neutral()
            && self.grade_neutral()
            && self.defringe_purple_amount <= 0.0
            && self.defringe_green_amount <= 0.0
            && self.lens_vignette_amount == 0.0
            && self.sharpen_amount == 0.0
            && self.noise_luminance == 0.0
            && self.noise_color == 0.0
            && self.grain_amount == 0.0
            && self.vignette_amount == 0.0
    }
}

fn is_linear(c: &[[f32; 2]]) -> bool {
    c.len() < 2 || c.iter().all(|p| (p[0] - p[1]).abs() < 1e-3)
}

/// Camera Raw's white-balance channel gains for Temperature / Tint (−100…100), relative to the
/// as-shot balance: warmer raises red and lowers blue (±35 % at the ends); a negative tint
/// raises green, a positive one lowers it toward magenta (±25 %). An observed approximation of
/// Adobe Camera Raw's sliders, not a colour-temperature model; shared by the filter (in linear
/// RGB) and the raw open (on the camera's white-balance multipliers) so both agree.
pub fn white_balance_gains(temperature: f32, tint: f32) -> [f32; 3] {
    let t = temperature / 100.0;
    let tn = tint / 100.0;
    [1.0 + 0.35 * t, 1.0 - 0.25 * tn, 1.0 - 0.35 * t]
}

/// Runs the Camera Raw pipeline on straight RGBA pixels (`w × h`, display-encoded RGB).
/// `float` keeps values above 1 (32-bit documents).
pub fn develop(px: &mut [[f32; 4]], w: usize, h: usize, p: &CameraRaw, float: bool) {
    if px.is_empty() || p.is_identity() {
        return;
    }
    let ps = p.pixel_scale.clamp(0.01, 1.0);
    let long = w.max(h) as f32;
    let space = pointwise::Space::of(p.encoding_gamma);
    // Contrast turns about a pivot that follows the image as it was before any edit (measured:
    // Exposure −1 leaves the pivot where it is).
    let pivot = if p.contrast != 0.0 { pointwise::contrast_pivot(mean_log2(px, space)) } else { 0.5 };
    let point = pointwise::Pointwise::new(p, pivot);
    // The local stages (Dehaze, Shadows / Highlights, Clarity / Texture, Whites / Blacks) sit
    // between the white balance and the colour stages; without them every pointwise stage runs
    // in one pass over the pixels.
    let local = p.dehaze != 0.0 || p.highlights != 0.0 || p.shadows != 0.0 || p.texture != 0.0 || p.clarity != 0.0 || p.whites != 0.0 || p.blacks != 0.0;
    // 1. Exposure (the measured family of curves, per channel with the hue kept) and the white
    //    balance in scene space (see `pointwise`); on their own pass only when a local stage follows.
    if local && point.has_early() {
        par_rows(px, w, 1, |_, row| {
            for q in row.iter_mut() {
                let o = space.to_document(point.early(space.to_pro([q[0], q[1], q[2]])));
                q[0] = o[0];
                q[1] = o[1];
                q[2] = o[2];
            }
        });
    }
    // 2. Dehaze.
    if p.dehaze != 0.0 {
        dehaze(px, w, h, p.dehaze / 100.0);
    }
    // 3. Tone. Highlights and Shadows: a local Laplacian filter (see `local_tone`).
    if p.highlights != 0.0 || p.shadows != 0.0 {
        local_tone::apply(px, w, h, p.shadows, p.highlights, Encoding::of(p.encoding_gamma));
    }
    if p.texture != 0.0 || p.clarity != 0.0 {
        let l: Vec<f32> = px.iter().map(|q| luma([q[0], q[1], q[2]]).max(0.0)).collect();
        let lc: Vec<f32> = l.iter().map(|v| v.min(1.0)).collect();
        let clar = (p.clarity != 0.0).then(|| guided_fast(&lc, &lc, w, h, ((long * 0.02) as usize).max(3), 0.005));
        let tex = (p.texture != 0.0).then(|| gauss(&lc, w, h, (long * 0.002).max(1.5 * ps)));
        par_rows(px, w, 1, |y, row| {
            for (x, q) in row.iter_mut().enumerate() {
                let i = y * w + x;
                let l0 = l[i];
                let mut lv = lc[i];
                if let Some(c) = &clar {
                    let mw = 1.0 - (2.0 * lv.clamp(0.0, 1.0) - 1.0).powi(2);
                    lv += p.clarity / 100.0 * 0.8 * (lc[i] - c[i]) * mw;
                }
                if let Some(t) = &tex {
                    lv += p.texture / 100.0 * 0.9 * (lc[i] - t[i]).clamp(-0.1, 0.1);
                }
                let lv = lv.max(0.0) + (l0 - lc[i]);
                let ratio = if l0 > 1e-5 { lv / l0 } else { 1.0 };
                for c in 0..3 {
                    q[c] = if l0 > 1e-5 { q[c] * ratio } else { q[c] + (lv - l0) };
                }
            }
        });
    }
    // 3b. Whites and Blacks: global curves adapted to the image's black and white points.
    if p.whites != 0.0 || p.blacks != 0.0 {
        let enc = Encoding::of(p.encoding_gamma);
        let (bp, wp) = luminance_points(px, w, h, enc);
        let (xw, xb) = whites_blacks::amounts(p.whites.clamp(-100.0, 100.0), p.blacks.clamp(-100.0, 100.0), bp, wp);
        whites_blacks::apply(px, w, xw, xb, enc);
    }
    // 4–7. Calibration, the Color Mixer, Saturation and Vibrance, Contrast and the tone curves,
    //      then Color Grading, in one pass (with Exposure and the white balance first when no
    //      local stage separated them).
    if point.has_late() || (!local && point.has_early()) {
        let early = !local;
        par_rows(px, w, 1, |_, row| {
            for q in row.iter_mut() {
                let mut c = space.to_pro([q[0], q[1], q[2]]);
                if early {
                    c = point.early(c);
                }
                let o = space.to_document(point.late(c));
                q[0] = o[0];
                q[1] = o[1];
                q[2] = o[2];
            }
        });
    }
    // 8. Detail: noise reduction, then sharpening on the log₂ luminance (calibrated to Camera Raw's
    //    halo size, ringing and noise gain), then Defringe.
    if p.noise_luminance > 0.0 || p.noise_color > 0.0 || p.sharpen_amount > 0.0 {
        detail(px, w, h, p, space, ps);
    }
    if p.defringe_purple_amount > 0.0 || p.defringe_green_amount > 0.0 {
        defringe(px, w, h, p, space, ps);
    }
    // 9. Effects: the lens vignette (Optics), the post-crop vignette and grain, each as measured on
    //    flat fields (see `vignette_gain`).
    if p.lens_vignette_amount != 0.0 {
        lens_vignette(px, w, h, p, space);
    }
    if p.vignette_amount != 0.0 {
        vignette(px, w, h, p, space);
    }
    if p.grain_amount > 0.0 {
        grain(px, w, p, space, ps);
    }
    if !float {
        for q in px.iter_mut() {
            for c in 0..3 {
                q[c] = q[c].clamp(0.0, 1.0);
            }
        }
    }
}

/// The image's black and white points as sRGB-encoded grey levels: the 3rd percentile of the
/// luminance and the 99.5th of the brightest channel, on a copy at most 256 pixels across (so a
/// preview proxy and the full image agree).
fn luminance_points(px: &[[f32; 4]], w: usize, h: usize, enc: Encoding) -> (f32, f32) {
    let f = w.max(h).div_ceil(256).max(1);
    let (sw, sh) = (w.div_ceil(f), h.div_ceil(f));
    let mut sum = vec![[0.0f64; 4]; sw * sh];
    for y in 0..h {
        for x in 0..w {
            let q = px[y * w + x];
            let s = &mut sum[(y / f) * sw + x / f];
            for c in 0..3 {
                s[c] += f64::from(enc.decode(q[c]).clamp(0.0, 1.0));
            }
            s[3] += 1.0;
        }
    }
    let cells: Vec<[f32; 3]> = sum.iter().filter(|s| s[3] > 0.0).map(|s| [0, 1, 2].map(|c| (s[c] / s[3]) as f32)).collect();
    if cells.is_empty() {
        return (0.0, 1.0);
    }
    let pct = |mut v: Vec<f32>, q: f32| {
        v.sort_by(f32::total_cmp);
        v[((v.len() - 1) as f32 * q).round() as usize]
    };
    let bp = pct(cells.iter().map(|c| luma(*c)).collect(), 0.03);
    let wp = pct(cells.iter().map(|c| c[0].max(c[1]).max(c[2])).collect(), 0.995);
    (linear_to_srgb(bp), linear_to_srgb(wp))
}

/// Mean log₂ of each pixel's darkest linear channel (sRGB primaries, floor 1/4096), on every
/// `k`-th pixel so that at most 65536 are read: the statistic Camera Raw's Contrast pivot follows.
fn mean_log2(px: &[[f32; 4]], space: pointwise::Space) -> f32 {
    let n = px.len();
    if n == 0 {
        return -3.0;
    }
    let step = n.div_ceil(65536).max(1);
    let (mut sum, mut count) = (0.0f64, 0usize);
    for q in px.iter().step_by(step) {
        let c = space.to_linear_srgb([q[0], q[1], q[2]]);
        let mn = c[0].min(c[1]).min(c[2]);
        sum += f64::from(mn.max(1.0 / 4096.0).log2());
        count += 1;
    }
    (sum / count.max(1) as f64) as f32
}

fn dehaze(px: &mut [[f32; 4]], w: usize, h: usize, amount: f32) {
    // Transmission on a ≤ 768 px proxy, upsampled.
    let k = w.max(h).div_ceil(768).max(1);
    let (sw, sh) = (w.div_ceil(k), h.div_ceil(k));
    let mut small = vec![[0.0f32; 3]; sw * sh];
    let mut cnt = vec![0f32; sw * sh];
    for y in 0..h {
        for x in 0..w {
            let i = (y / k) * sw + x / k;
            for c in 0..3 {
                small[i][c] += px[y * w + x][c].clamp(0.0, 1.0);
            }
            cnt[i] += 1.0;
        }
    }
    for (s, c) in small.iter_mut().zip(&cnt) {
        *s = s.map(|v| v / c.max(1.0));
    }
    let r = (sw.max(sh) / 60).max(2);
    let min_filter = |v: &[f32]| -> Vec<f32> {
        let mut a = vec![0.0f32; sw * sh];
        for y in 0..sh {
            for x in 0..sw {
                let (x0, x1) = (x.saturating_sub(r), (x + r).min(sw - 1));
                a[y * sw + x] = (x0..=x1).map(|xx| v[y * sw + xx]).fold(f32::MAX, f32::min);
            }
        }
        let mut b = vec![0.0f32; sw * sh];
        for y in 0..sh {
            for x in 0..sw {
                let (y0, y1) = (y.saturating_sub(r), (y + r).min(sh - 1));
                b[y * sw + x] = (y0..=y1).map(|yy| a[yy * sw + x]).fold(f32::MAX, f32::min);
            }
        }
        b
    };
    let dark = min_filter(&small.iter().map(|c| c[0].min(c[1]).min(c[2])).collect::<Vec<_>>());
    // Atmospheric light: mean colour of the brightest 0.1 % of the dark channel.
    let mut idx: Vec<usize> = (0..sw * sh).collect();
    idx.sort_by(|&a, &b| dark[b].total_cmp(&dark[a]));
    let top = (sw * sh / 1000).max(1);
    let mut air = [0.0f32; 3];
    for &i in &idx[..top] {
        for c in 0..3 {
            air[c] += small[i][c] / top as f32;
        }
    }
    let air = air.map(|v| v.max(0.2));
    let norm_dark = min_filter(&small.iter().map(|c| (c[0] / air[0]).min(c[1] / air[1]).min(c[2] / air[2])).collect::<Vec<_>>());
    let t_raw: Vec<f32> = norm_dark.iter().map(|d| 1.0 - 0.95 * d).collect();
    let guide: Vec<f32> = small.iter().map(|c| luma(*c)).collect();
    let t = guided(&guide, &t_raw, sw, sh, r * 2, 1e-3);
    let ks = k as f32;
    par_rows(px, w, 1, |y, row| {
        for (x, q) in row.iter_mut().enumerate() {
            let tv = crate::photo_util::bilinear(&t, sw, sh, 1, 0, (x as f32 + 0.5) / ks - 0.5, (y as f32 + 0.5) / ks - 0.5).clamp(0.1, 1.0);
            for c in 0..3 {
                let i = q[c];
                let j = (i - air[c]) / tv + air[c];
                q[c] = if amount >= 0.0 { i + (j - i) * amount } else { i + (air[c] - i) * (-amount) * 0.5 * (1.0 - tv * 0.5) };
            }
        }
    });
}

fn detail(px: &mut [[f32; 4]], w: usize, h: usize, p: &CameraRaw, space: pointwise::Space, ps: f32) {
    // Work in Y + chroma (BT.709 luma of the encoded pixel, B−Y, R−Y).
    let mut y: Vec<f32> = px.iter().map(|q| luma([q[0], q[1], q[2]])).collect();
    let mut cb: Vec<f32> = px.iter().zip(&y).map(|(q, l)| q[2] - l).collect();
    let mut cr: Vec<f32> = px.iter().zip(&y).map(|(q, l)| q[0] - l).collect();
    if p.noise_luminance > 0.0 {
        // Measured (noise of σ 0.02 and 0.04 in linear light on grey patches): the noise left is
        // 0.53 at Luminance 25, 0.38 at 50 and 0.13 at 100 of the original in the shadows and
        // midtones, less is taken in the highlights (0.80 at 50 on a 0.9 grey); Detail 0 leaves
        // 0.35 at every level, 100 leaves 0.8; edges keep their gradient.
        let n = (p.noise_luminance / 100.0).clamp(0.0, 1.0);
        let r = ((2.0 + 2.0 * n) * ps).round().max(1.0) as usize;
        let smooth = guided(&y, &y, w, h, r, 0.15 * 0.15);
        let removed = if n <= 0.5 { 0.53 + (n - 0.25) / 0.25 * 0.17 } else { 0.70 + (n - 0.5) / 0.5 * 0.29 };
        let removed = if n < 0.25 { 0.53 * n / 0.25 } else { removed };
        let d = (p.noise_luminance_detail / 100.0).clamp(0.0, 1.0);
        let keep = 0.7 * d.powf(3.5);
        let tr = space.transfer();
        for i in 0..w * h {
            let level = tr.decode(y[i]).clamp(0.0, 1.0);
            let f = removed * (1.0 - keep) * (1.0 - 0.75 * level.powf(2.2));
            y[i] += (smooth[i] - y[i]) * f;
        }
    }
    if p.noise_color > 0.0 {
        // Measured: Camera Raw's colour noise reduction takes chroma noise down to about 0.28 of
        // its amplitude from Color 25 on (more Color changes little); Detail keeps part of it
        // (0 → 0.22, 50 → 0.28, 100 → 0.74).
        let sigma = (2.0 * ps).max(0.3);
        let (bcb, bcr) = (gauss(&cb, w, h, sigma), gauss(&cr, w, h, sigma));
        let d = (p.noise_color_detail / 100.0).clamp(0.0, 1.0);
        let keep = 0.12 + 0.62 * d * d;
        for i in 0..w * h {
            cb[i] = bcb[i] + (cb[i] - bcb[i]) * keep;
            cr[i] = bcr[i] + (cr[i] - bcr[i]) * keep;
        }
    }
    if p.sharpen_amount > 0.0 {
        // Measured on edges, lines, a chirp and noise (Photoshop 25.4): the halo is symmetric in
        // log₂ of the linear luminance (+0.45 / −0.2 stop on a 4.3-stop edge at Amount 100,
        // Detail 25), its width follows Radius, Detail limits the halo (0 halves it) and lets
        // ringing through at 100, the fine-detail gain saturates with Amount (×1.67, ×1.93, ×2.04
        // at 50, 100, 150). An unsharp mask on log₂ luminance with a halo limit fits those figures
        // (the gain at the finest detail stays under Camera Raw's).
        let radius = p.sharpen_radius.clamp(0.5, 3.0);
        let (sigma, gain) = if radius <= 1.0 {
            (0.6 + (radius - 0.5) * 0.1, 3.0 + (radius - 0.5) * 2.8)
        } else if radius <= 2.0 {
            (0.65 + (radius - 1.0) * 0.67, 4.4 - (radius - 1.0) * 2.1)
        } else {
            (1.32 + (radius - 2.0) * 1.09, 2.3 - (radius - 2.0) * 1.15)
        };
        let sigma = (sigma * ps).max(0.3);
        let a = p.sharpen_amount.clamp(0.0, 150.0) / 100.0;
        let strength = if a <= 0.5 {
            a / 0.5 * 0.39
        } else if a <= 1.0 {
            0.39 + (a - 0.5) / 0.5 * 0.61
        } else {
            1.0 + (a - 1.0) / 0.5 * 0.27
        };
        let d = (p.sharpen_detail / 100.0).clamp(0.0, 1.0);
        let limit = (0.30 + 0.12 * a.min(1.5)) * (0.45 + 0.55 * (d / 0.5).min(1.0)) * (1.0 + 0.3 * (d - 0.5).max(0.0));
        let tr = space.transfer();
        let lg: Vec<f32> = y.iter().map(|v| tr.decode(*v).max(1e-4).log2()).collect();
        let blur = gauss(&lg, w, h, sigma);
        // Masking: an edge mask from the gradient of the smoothed log luminance; its threshold
        // rises 22× between Masking 50 and 100 (measured on the chirp).
        let mask: Option<Vec<f32>> = (p.sharpen_masking > 0.0).then(|| {
            let m = p.sharpen_masking.clamp(0.0, 100.0);
            let t = 0.07 * 22f32.powf((m - 50.0) / 50.0);
            let s = gauss(&lg, w, h, 1.0 * ps.max(0.5));
            crate::photo_util::par_map(w * h, |i| {
                let (x, yy) = (i % w, i / w);
                let gx = s[yy * w + (x + 1).min(w - 1)] - s[yy * w + x.saturating_sub(1)];
                let gy = s[(yy + 1).min(h - 1) * w + x] - s[yy.saturating_sub(1) * w + x];
                (gx.hypot(gy) / t).clamp(0.0, 1.0)
            })
        });
        for i in 0..w * h {
            let hp = (gain * strength * (lg[i] - blur[i])).clamp(-limit, limit);
            let m = mask.as_ref().map_or(1.0, |m| m[i]);
            if y[i] > 1e-4 {
                y[i] = tr.encode((lg[i] + hp * m).exp2());
            }
        }
    }
    for i in 0..w * h {
        let (l, b, r) = (y[i], cb[i], cr[i]);
        let (rr, bb) = (r + l, b + l);
        let g = (l - 0.2126 * rr - 0.0722 * bb) / 0.7152;
        px[i][0] = rr;
        px[i][1] = g;
        px[i][2] = bb;
    }
}

/// Defringe: colours inside the purple or green hue window next to a strong luminance edge lose
/// their chroma (measured: a 3 px purple fringe of chroma 48 keeps 9 at Purple Amount 10 or 20;
/// the hue windows are 0…100 over blue–magenta (purple) and yellow–cyan (green)).
fn defringe(px: &mut [[f32; 4]], w: usize, h: usize, p: &CameraRaw, space: pointwise::Space, ps: f32) {
    let lum: Vec<f32> = px.iter().map(|q| luma([q[0], q[1], q[2]]).max(0.0)).collect();
    let s = gauss(&lum, w, h, (1.0 * ps).max(0.5));
    // Edge strength: the gradient of the smoothed luminance, relative to the level.
    let edge = crate::photo_util::par_map(w * h, |i| {
        let (x, yy) = (i % w, i / w);
        let gx = s[yy * w + (x + 1).min(w - 1)] - s[yy * w + x.saturating_sub(1)];
        let gy = s[(yy + 1).min(h - 1) * w + x] - s[yy.saturating_sub(1) * w + x];
        gx.hypot(gy) / (s[i] + 0.05)
    });
    // Fringes sit within a few pixels of the edge: spread the edge strength.
    let near = gauss(&edge, w, h, (2.0 * ps).max(0.6));
    let windows = [
        (
            p.defringe_purple_amount.clamp(0.0, 20.0),
            230.0 + p.defringe_purple_hue[0].clamp(0.0, 100.0) * 0.8,
            230.0 + p.defringe_purple_hue[1].clamp(0.0, 100.0) * 0.8,
        ),
        (
            p.defringe_green_amount.clamp(0.0, 20.0),
            76.0 + p.defringe_green_hue[0].clamp(0.0, 100.0) * 0.6,
            76.0 + p.defringe_green_hue[1].clamp(0.0, 100.0) * 0.6,
        ),
    ];
    par_rows(px, w, 1, |yy, row| {
        for (x, q) in row.iter_mut().enumerate() {
            let i = yy * w + x;
            let e = (near[i] / 0.08).clamp(0.0, 1.0);
            if e <= 0.01 {
                continue;
            }
            let c = space.to_pro([q[0], q[1], q[2]]);
            let (mx, mn) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
            if mx - mn <= 1e-6 {
                continue;
            }
            let d = mx - mn;
            let hue = if mx == c[0] {
                60.0 * ((c[1] - c[2]) / d).rem_euclid(6.0)
            } else if mx == c[1] {
                60.0 * ((c[2] - c[0]) / d + 2.0)
            } else {
                60.0 * ((c[0] - c[1]) / d + 4.0)
            };
            let mut keep = 1.0f32;
            for (amount, lo, hi) in windows {
                if amount <= 0.0 || hue < lo - 10.0 || hue > hi + 10.0 {
                    continue;
                }
                let inside = smoothstep(lo - 10.0, lo, hue) * (1.0 - smoothstep(hi, hi + 10.0, hue));
                // Amount 10 already removes most of the chroma; 0.18 of it stays.
                let f = (1.0 - amount / 8.0).max(0.18);
                keep = keep.min(1.0 - (1.0 - f) * inside * e);
            }
            if keep < 1.0 {
                let y = pointwise::PRO_Y[0] * c[0] + pointwise::PRO_Y[1] * c[1] + pointwise::PRO_Y[2] * c[2];
                let o = space.to_document(c.map(|v| y + (v - y) * keep));
                q[0] = o[0];
                q[1] = o[1];
                q[2] = o[2];
            }
        }
    });
}

/// log₂ gain of a vignette profile at elliptical radius `r` (0 at the centre, 1 at the edge
/// midpoints, √2 at the corners), from the measured table.
fn vignette_gain(table: &[f32; filter_tables::VIGNETTE_STEPS], r: f32) -> f32 {
    let n = filter_tables::VIGNETTE_STEPS;
    // Each table value sits at the centre of its bin.
    let p = (r / filter_tables::VIGNETTE_R_MAX * (n - 1) as f32 - 0.5).clamp(0.0, (n - 1) as f32);
    let i = (p as usize).min(n - 2);
    let f = p - i as f32;
    table[i] + (table[i + 1] - table[i]) * f
}

/// The elliptical radius of pixel `(x, y)` with Camera Raw's roundness: 0 follows the frame, +100
/// is a circle, −100 a rounded rectangle.
fn vignette_radius(x: f32, y: f32, w: usize, h: usize, roundness: f32) -> f32 {
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let round = roundness.clamp(-100.0, 100.0) / 100.0;
    // Measured: at +100 the vignette is round with the mean of the half sizes as its radius.
    let (ax, ay) = if round > 0.0 {
        let r = 0.5 * (cx + cy);
        (cx + (r - cx) * round, cy + (r - cy) * round)
    } else {
        (cx, cy)
    };
    let (dx, dy) = (((x + 0.5 - cx) / ax).abs(), ((y + 0.5 - cy) / ay).abs());
    let pw = 2.0 + (-round).max(0.0) * 6.0;
    (dx.powf(pw) + dy.powf(pw)).powf(1.0 / pw)
}

/// The post-crop vignette as measured on a flat field: the gain profile at ±50 / ±100 (midpoint
/// 50, feather 50) interpolated in the amount, the midpoint moving the profile in or out (the
/// −1-stop radius is 0.51, 0.80 and 1.3 at midpoints 0, 50, 100), the feather sharpening or
/// softening the transition about it, Highlights letting bright pixels through (at 50 a 0.95
/// patch keeps 83 % of its light, at 100 94 %), and Paint Overlay blending toward black or white.
fn vignette(px: &mut [[f32; 4]], w: usize, h: usize, p: &CameraRaw, space: pointwise::Space) {
    use filter_tables::{
        VIGNETTE_PCV_M50, VIGNETTE_PCV_M100, VIGNETTE_PCV_M100_FEATHER0, VIGNETTE_PCV_M100_FEATHER100, VIGNETTE_PCV_M100_MID0, VIGNETTE_PCV_M100_MID100,
        VIGNETTE_PCV_P50, VIGNETTE_PCV_P100,
    };
    let amt = p.vignette_amount.clamp(-100.0, 100.0);
    let mid = p.vignette_midpoint.clamp(0.0, 100.0);
    let feather = p.vignette_feather.clamp(0.0, 100.0);
    let hl = p.vignette_highlights.clamp(0.0, 100.0) / 100.0;
    let style = p.vignette_style.as_str();
    let tr = space.transfer();
    // The midpoint and feather reshape the profile as the measured midpoint 0 / 100 and feather
    // 0 / 100 profiles at −100 do, relative to the default one.
    let reshape = |r: f32, g: f32| -> f32 {
        let base = vignette_gain(&VIGNETTE_PCV_M100, r);
        let mut g = g;
        for (value, lo_table, hi_table) in
            [(mid, &VIGNETTE_PCV_M100_MID0, &VIGNETTE_PCV_M100_MID100), (feather, &VIGNETTE_PCV_M100_FEATHER0, &VIGNETTE_PCV_M100_FEATHER100)]
        {
            if value == 50.0 {
                continue;
            }
            let other = if value < 50.0 { vignette_gain(lo_table, r) } else { vignette_gain(hi_table, r) };
            let k = (value - 50.0).abs() / 50.0;
            g = if base < -1e-3 { g * (1.0 + k * (other / base - 1.0)).clamp(0.0, 8.0) } else { g + k * other * g.signum().abs() };
        }
        g
    };
    let profile = |r: f32| -> f32 {
        let (a, b, t) = if amt <= -50.0 {
            (&VIGNETTE_PCV_M100, &VIGNETTE_PCV_M50, (amt + 100.0) / 50.0)
        } else if amt < 0.0 {
            (&VIGNETTE_PCV_M50, &VIGNETTE_PCV_M50, 0.0)
        } else if amt <= 50.0 {
            (&VIGNETTE_PCV_P50, &VIGNETTE_PCV_P50, 0.0)
        } else {
            (&VIGNETTE_PCV_P50, &VIGNETTE_PCV_P100, (amt - 50.0) / 50.0)
        };
        let g = vignette_gain(a, r) + (vignette_gain(b, r) - vignette_gain(a, r)) * t;
        // Below ±50 the gain scales with the amount.
        let g = if amt.abs() < 50.0 { g * amt.abs() / 50.0 } else { g };
        reshape(r, g)
    };
    par_rows(px, w, 1, |y, row| {
        for (x, q) in row.iter_mut().enumerate() {
            let r = vignette_radius(x as f32, y as f32, w, h, p.vignette_roundness);
            let mut g = profile(r);
            if g.abs() < 1e-4 {
                continue;
            }
            let c = space.to_pro([q[0], q[1], q[2]]);
            let yv = (pointwise::PRO_Y[0] * c[0] + pointwise::PRO_Y[1] * c[1] + pointwise::PRO_Y[2] * c[2]).max(0.0);
            if amt < 0.0 && hl > 0.0 {
                // Bright pixels keep their light: measured on 0.95 and 0.75 patches.
                let b = ((yv - 0.3) / 0.6).clamp(0.0, 1.0);
                let removed = (hl.powf(0.7) * 1.4 * b * b).min(0.94);
                g *= 1.0 - removed;
            }
            // Each style is a gain, an offset along grey or a blend toward black or white: the same
            // in the document's linear primaries as in ProPhoto's, so greys stay exactly grey.
            let lin = [q[0], q[1], q[2]].map(|v| tr.decode(v));
            let o = match style {
                "paintOverlay" => {
                    let target = if amt < 0.0 { 0.0 } else { 1.0 };
                    // Measured: the blend toward black is 0.78 where the gain is −3.06 stops.
                    let wgt = (1.0 - (0.72 * g).exp2()).clamp(0.0, 1.0).abs();
                    let wgt = if amt < 0.0 { wgt } else { (1.0 - (-0.72 * g).exp2()).clamp(0.0, 1.0) };
                    lin.map(|v| v + (target - v) * wgt)
                }
                "colorPriority" => {
                    let k = g.exp2();
                    lin.map(|v| v + yv * (k - 1.0))
                }
                _ => lin.map(|v| v * g.exp2()),
            };
            q[..3].copy_from_slice(&o.map(|v| tr.encode(v)));
        }
    });
}

/// The Optics panel's lens vignette: a smooth fall-off (measured −1.32 stops at the edge midpoints
/// and −2.73 at the corners at −100, midpoint 50), the midpoint moving it in or out.
fn lens_vignette(px: &mut [[f32; 4]], w: usize, h: usize, p: &CameraRaw, space: pointwise::Space) {
    use filter_tables::{VIGNETTE_LENS_M100, VIGNETTE_LENS_M100_MID0, VIGNETTE_LENS_M100_MID100, VIGNETTE_LENS_P100};
    let amt = p.lens_vignette_amount.clamp(-100.0, 100.0);
    let mid = p.lens_vignette_midpoint.clamp(0.0, 100.0);
    let profile = |r: f32| -> f32 {
        let base = if amt < 0.0 { vignette_gain(&VIGNETTE_LENS_M100, r) } else { vignette_gain(&VIGNETTE_LENS_P100, r) };
        let shaped = if mid < 50.0 {
            let other = vignette_gain(&VIGNETTE_LENS_M100_MID0, r);
            base + (other - vignette_gain(&VIGNETTE_LENS_M100, r)) * (50.0 - mid) / 50.0
                * if amt < 0.0 { 1.0 } else { -base / vignette_gain(&VIGNETTE_LENS_M100, r).min(-1e-3) }
        } else if mid > 50.0 {
            let other = vignette_gain(&VIGNETTE_LENS_M100_MID100, r);
            base + (other - vignette_gain(&VIGNETTE_LENS_M100, r)) * (mid - 50.0) / 50.0
                * if amt < 0.0 { 1.0 } else { -base / vignette_gain(&VIGNETTE_LENS_M100, r).min(-1e-3) }
        } else {
            base
        };
        shaped * amt.abs() / 100.0
    };
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let half_diag = cx.hypot(cy).max(1.0);
    // A gain in linear light: the same in the document's primaries as in ProPhoto's.
    let tr = space.transfer();
    par_rows(px, w, 1, |y, row| {
        for (x, q) in row.iter_mut().enumerate() {
            // Round: the pixel radius over the half diagonal, on the table's 0…1.45 scale.
            let r = (x as f32 + 0.5 - cx).hypot(y as f32 + 0.5 - cy) / half_diag * 1.45;
            let g = profile(r);
            if g.abs() < 1e-4 {
                continue;
            }
            let k = g.exp2();
            for v in &mut q[..3] {
                *v = tr.encode(tr.decode(*v) * k);
            }
        }
    });
}

/// Grain as measured on a flat field: Gaussian, the same in every channel, added to the
/// sRGB-encoded ProPhoto channels with a standard deviation of 0.00158 × Amount (Size 25,
/// Roughness 50); Size 10 → ×1.09, 50 → ×0.86, 100 → ×0.62 and coarser (positive correlation
/// between neighbours at 100), Roughness 0 → ×1.75 and finer, 100 → ×0.98 and coarser.
fn grain(px: &mut [[f32; 4]], w: usize, p: &CameraRaw, space: pointwise::Space, ps: f32) {
    let size = p.grain_size.clamp(0.0, 100.0);
    let rough = p.grain_roughness.clamp(0.0, 100.0);
    let f_size = if size <= 25.0 {
        1.09 + (size - 10.0).max(0.0) / 15.0 * (1.0 - 1.09)
    } else if size <= 50.0 {
        1.0 + (size - 25.0) / 25.0 * (0.86 - 1.0)
    } else {
        0.86 + (size - 50.0) / 50.0 * (0.62 - 0.86)
    };
    let f_rough = if rough <= 50.0 { 1.75 + rough / 50.0 * (1.0 - 1.75) } else { 1.0 + (rough - 50.0) / 50.0 * (0.98 - 1.0) };
    let sd = 0.00158 * p.grain_amount.clamp(0.0, 100.0) * f_size * f_rough;
    // Neighbour correlation (lag 1): −0.10 at the defaults, −0.14 at Size 10, −0.20 at Roughness 0,
    // +0.23 at Size 100, +0.20 at Roughness 100: a share of the four neighbours is subtracted
    // (negative) or added (positive).
    let alpha = 0.25 + 0.4 * ((50.0 - rough) / 50.0).clamp(0.0, 1.0) + 0.1 * ((25.0 - size) / 15.0).clamp(0.0, 1.0)
        - 0.85 * ((size - 50.0) / 50.0).clamp(0.0, 1.0)
        - 0.8 * ((rough - 50.0) / 50.0).clamp(0.0, 1.0);
    // The filter's own gain on white noise, so the standard deviation stays as measured.
    let norm = (1.0 + alpha * alpha / 4.0).sqrt();
    let seed = p.seed as u64;
    let pscale = 1.0 / ps;
    let gaussian = |x: i64, y: i64| -> f32 {
        // Box–Muller on two hashes.
        let u1 = hash01(x, y, seed).max(1e-6);
        let u2 = hash01(x, y, seed.wrapping_add(0x9E37_79B9));
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    };
    par_rows(px, w, 1, |y, row| {
        for (x, q) in row.iter_mut().enumerate() {
            // Sample in full-resolution coordinates so previews match the result.
            let (fx, fy) = ((x as f32 * pscale).round() as i64, (y as f32 * pscale).round() as i64);
            let n4 = gaussian(fx - 1, fy) + gaussian(fx + 1, fy) + gaussian(fx, fy - 1) + gaussian(fx, fy + 1);
            let n = (gaussian(fx, fy) - alpha * n4 / 4.0) / norm;
            let c = space.to_pro([q[0], q[1], q[2]]);
            let o = space.to_document(c.map(|v| pointwise::dec_srgb((pointwise::enc_srgb(v.max(0.0)) + n * sd).max(0.0))));
            q[0] = o[0];
            q[1] = o[1];
            q[2] = o[2];
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every vignette style and the lens vignette are a gain, an offset along grey or a blend
    /// toward black or white: a grey image stays exactly grey.
    #[test]
    fn vignettes_keep_greys_grey() {
        let (w, h) = (48usize, 32usize);
        for p in [
            CameraRaw { vignette_amount: -60.0, vignette_highlights: 40.0, ..Default::default() },
            CameraRaw { vignette_amount: 40.0, vignette_style: "colorPriority".into(), ..Default::default() },
            CameraRaw { vignette_amount: -80.0, vignette_style: "paintOverlay".into(), ..Default::default() },
            CameraRaw { lens_vignette_amount: -50.0, ..Default::default() },
        ] {
            let mut px: Vec<[f32; 4]> = (0..w * h)
                .map(|i| {
                    let v = 0.1 + 0.8 * (i % w) as f32 / w as f32;
                    [v, v, v, 1.0]
                })
                .collect();
            let before = px.clone();
            develop(&mut px, w, h, &p, true);
            assert_ne!(px, before, "{p:?} darkens or lightens the corners");
            for q in &px {
                assert!((q[0] - q[1]).abs() < 1e-5 && (q[1] - q[2]).abs() < 1e-5, "{p:?}: {q:?}");
            }
        }
    }

    #[test]
    fn new_curves_are_validated_but_saved_curves_still_load_and_render() {
        for field in ["pointCurve", "pointCurveRed", "pointCurveGreen", "pointCurveBlue"] {
            for points in [
                serde_json::json!([[128, 128]]),
                serde_json::json!([[255, 255], [0, 0]]),
                serde_json::json!([[0, 0], [0, 255]]),
                serde_json::json!([[0, -1], [255, 255]]),
                serde_json::json!([[0, 0], [256, 255]]),
                serde_json::json!([[0, 0], [1e30, 255]]),
                serde_json::json!((0..17).map(|i| [i * 15, i * 15]).collect::<Vec<_>>()),
            ] {
                let p: CameraRaw = serde_json::from_value(serde_json::json!({field: points})).unwrap();
                assert!(p.validate().is_err(), "{field}: {points}");
            }
            for points in
                [serde_json::json!([]), serde_json::json!([[0, 0], [255, 255]]), serde_json::json!((0..16).map(|i| [i * 17, i * 17]).collect::<Vec<_>>())]
            {
                let p: CameraRaw = serde_json::from_value(serde_json::json!({field: points})).unwrap();
                assert!(p.validate().is_ok(), "{field}: {points}");
            }
        }
        assert_eq!(serde_json::from_str::<CameraRaw>("{}").unwrap(), CameraRaw::default());
        // An older editor could stack two points on one input and exceed 16 points; such a
        // saved Smart Filter must still develop instead of silently disappearing.
        let mut legacy: Vec<[f32; 2]> = (0..20).map(|i| [i as f32 * 12.0, i as f32 * 12.0]).collect();
        legacy.insert(5, [60.0, 200.0]);
        let p: CameraRaw = serde_json::from_value(serde_json::json!({"pointCurve": legacy})).unwrap();
        assert!(p.validate().is_err());
        let edited = CameraRaw { exposure: 1.0, ..p.clone() };
        assert!(edited.validate_changes(&p).is_ok(), "an unchanged stored curve stays editable");
        let broken = CameraRaw { point_curve: vec![[10.0, 10.0]], ..p.clone() };
        assert!(broken.validate_changes(&p).is_err(), "a changed curve is validated");
        let mut px = img(4, 4);
        develop(&mut px, 4, 4, &p, false);
        assert!(px.iter().flatten().all(|v| v.is_finite()));
    }

    fn img(w: usize, h: usize) -> Vec<[f32; 4]> {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 / w as f32, (i / w) as f32 / h as f32);
                [0.2 + 0.6 * x, 0.3 + 0.4 * y, 0.5 - 0.3 * x * y, 1.0]
            })
            .collect()
    }

    fn mean(px: &[[f32; 4]], c: usize) -> f32 {
        px.iter().map(|q| q[c]).sum::<f32>() / px.len() as f32
    }

    #[test]
    fn defaults_are_identity() {
        let mut a = img(40, 30);
        let b = a.clone();
        develop(&mut a, 40, 30, &CameraRaw::default(), false);
        assert_eq!(a, b);
    }

    #[test]
    fn basic_panel_moves_the_right_way() {
        let (w, h) = (64, 48);
        let base = img(w, h);
        let run = |p: CameraRaw| {
            let mut a = base.clone();
            develop(&mut a, w, h, &p, false);
            a
        };
        let warm = run(CameraRaw { temperature: 50.0, ..Default::default() });
        assert!(mean(&warm, 0) > mean(&base, 0) && mean(&warm, 2) < mean(&base, 2));
        let bright = run(CameraRaw { exposure: 1.0, ..Default::default() });
        assert!(mean(&bright, 1) > mean(&base, 1) + 0.1);
        let shadows = run(CameraRaw { shadows: 100.0, ..Default::default() });
        let dark_px = |a: &[[f32; 4]]| a[0][1];
        assert!(dark_px(&shadows) > dark_px(&base));
        let hl = run(CameraRaw { highlights: -100.0, ..Default::default() });
        assert!(hl[w * h - 1][0] < base[w * h - 1][0]);
        let con = run(CameraRaw { contrast: 100.0, ..Default::default() });
        // More contrast: darker darks, brighter brights.
        assert!(con[0][1] < base[0][1] && con[w * h - 1][0] > base[w * h - 1][0]);
        let desat = run(CameraRaw { saturation: -100.0, ..Default::default() });
        assert!(desat.iter().all(|q| (q[0] - q[1]).abs() < 1e-4 && (q[1] - q[2]).abs() < 1e-4));
        let vib = run(CameraRaw { vibrance: 100.0, ..Default::default() });
        let spread = |q: [f32; 4]| q[0].max(q[1]).max(q[2]) - q[0].min(q[1]).min(q[2]);
        assert!(spread(vib[w * h / 2]) > spread(base[w * h / 2]));
        for p in [
            CameraRaw { clarity: 80.0, ..Default::default() },
            CameraRaw { texture: 80.0, ..Default::default() },
            CameraRaw { dehaze: 60.0, ..Default::default() },
            CameraRaw { dehaze: -60.0, ..Default::default() },
            CameraRaw { whites: 50.0, blacks: -50.0, ..Default::default() },
        ] {
            let out = run(p.clone());
            assert!(out.iter().all(|q| q[..3].iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v))), "{p:?}");
            assert_ne!(out, base);
        }
    }

    #[test]
    fn curves_hsl_grading_detail_effects() {
        // Monotone cubic LUT through the points.
        let lut = curve_lut(&[[0.0, 0.0], [64.0, 40.0], [192.0, 220.0], [255.0, 255.0]], 256);
        assert!((lut[64] - 40.0 / 255.0).abs() < 0.01 && (lut[192] - 220.0 / 255.0).abs() < 0.01);
        assert!(lut.windows(2).all(|p| p[1] >= p[0]));
        // Band weights sum to one everywhere.
        for hh in (0..360).step_by(7) {
            let s: f32 = band_weights(hh as f32).iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "{hh}: {s}");
        }
        let (w, h) = (48, 32);
        // A red at the band's centre (Camera Raw's red band peaks a little below hue 0).
        let reds: Vec<[f32; 4]> = vec![[0.8, 0.2, 0.3, 1.0]; w * h];
        let mut a = reds.clone();
        let mut hs = [0.0; 8];
        hs[0] = -100.0;
        develop(&mut a, w, h, &CameraRaw { hsl_sat: hs, ..Default::default() }, false);
        assert!(a[0][0] - a[0][1] < 0.12, "reds desaturated: {:?}", a[0]);
        let blues: Vec<[f32; 4]> = vec![[0.2, 0.2, 0.8, 1.0]; w * h];
        let mut b = blues.clone();
        develop(&mut b, w, h, &CameraRaw { hsl_sat: hs, ..Default::default() }, false);
        assert!((b[0][2] - 0.8).abs() < 1e-4, "blues untouched");
        // Grading: blue shadows tint dark pixels, leave bright ones nearly alone.
        let mut g: Vec<[f32; 4]> = vec![[0.1, 0.1, 0.1, 1.0], [0.95, 0.95, 0.95, 1.0]];
        develop(&mut g, 2, 1, &CameraRaw { grade_shadows: Wheel { hue: 220.0, sat: 80.0, lum: 0.0 }, ..Default::default() }, false);
        assert!(g[0][2] > g[0][0] + 0.05 && (g[1][2] - g[1][0]).abs() < 0.02, "{g:?}");
        // Point curve + parametric.
        let mut c = img(w, h);
        develop(&mut c, w, h, &CameraRaw { point_curve: vec![[0.0, 30.0], [255.0, 255.0]], curve_shadows: 50.0, ..Default::default() }, false);
        assert!(c.iter().all(|q| q[1] >= 30.0 / 255.0 - 1e-4));
        // Noise reduction reduces variance; sharpening increases it; grain is deterministic.
        let noisy: Vec<[f32; 4]> = (0..w * h)
            .map(|i| {
                let n = hash01(i as i64, 0, 3) * 0.2;
                [0.4 + n, 0.4 + n, 0.4 + n, 1.0]
            })
            .collect();
        let var = |a: &[[f32; 4]]| {
            let m = mean(a, 1);
            a.iter().map(|q| (q[1] - m).powi(2)).sum::<f32>() / a.len() as f32
        };
        let mut nr = noisy.clone();
        develop(&mut nr, w, h, &CameraRaw { noise_luminance: 80.0, ..Default::default() }, false);
        assert!(var(&nr) < var(&noisy) * 0.5);
        let mut sh = noisy.clone();
        develop(&mut sh, w, h, &CameraRaw { sharpen_amount: 100.0, ..Default::default() }, false);
        assert!(var(&sh) > var(&noisy));
        let mut g1 = img(w, h);
        let mut g2 = img(w, h);
        let gp = CameraRaw { grain_amount: 50.0, seed: 4, ..Default::default() };
        develop(&mut g1, w, h, &gp, false);
        develop(&mut g2, w, h, &gp, false);
        assert_eq!(g1, g2);
        assert_ne!(g1, img(w, h));
        // Post-crop vignette darkens corners, leaves the centre.
        let flat = vec![[0.5f32, 0.5, 0.5, 1.0]; w * h];
        let mut v = flat.clone();
        develop(&mut v, w, h, &CameraRaw { vignette_amount: -80.0, ..Default::default() }, false);
        assert!(v[0][0] < 0.35 && (v[(h / 2) * w + w / 2][0] - 0.5).abs() < 0.01, "{:?}", v[0]);
    }

    #[test]
    fn optics_and_calibration_act_as_measured() {
        // A purple fringe beside a dark bar loses its chroma; the grey around it is untouched.
        let (w, h) = (64usize, 8usize);
        let mut px: Vec<[f32; 4]> = (0..w * h)
            .map(|i| {
                if (i % w) < 20 {
                    [0.5, 0.5, 0.5, 1.0]
                } else if (i % w) < 23 {
                    [0.55, 0.3, 0.6, 1.0]
                } else {
                    [0.1, 0.1, 0.1, 1.0]
                }
            })
            .collect();
        develop(&mut px, w, h, &CameraRaw { defringe_purple_amount: 10.0, ..Default::default() }, false);
        let fringe = px[4 * w + 21];
        assert!(fringe[0] - fringe[1] < 0.1, "{fringe:?}");
        assert!((px[4 * w + 5][0] - 0.5).abs() < 1e-3 && (px[4 * w + 40][0] - 0.1).abs() < 1e-3);
        // The lens vignette darkens the corners of a flat field, not its centre; +100 lightens them.
        let (w, h) = (120usize, 80usize);
        let flat = vec![[0.5, 0.5, 0.5, 1.0]; w * h];
        for (amount, darker) in [(-100.0, true), (100.0, false)] {
            let mut px = flat.clone();
            develop(&mut px, w, h, &CameraRaw { lens_vignette_amount: amount, ..Default::default() }, false);
            assert!((px[40 * w + 60][1] - 0.5).abs() < 0.01);
            assert_eq!(px[0][1] < 0.5, darker, "{amount}: {}", px[0][1]);
        }
        // Calibration's red hue turns a red toward orange or magenta and leaves grey alone.
        for (hue, greener) in [(100.0, true), (-100.0, false)] {
            let mut px = vec![[0.8, 0.2, 0.2, 1.0], [0.5, 0.5, 0.5, 1.0]];
            develop(&mut px, 2, 1, &CameraRaw { calibration_red_hue: hue, ..Default::default() }, false);
            assert_eq!(px[0][1] > px[0][2], greener, "{hue}: {:?}", px[0]);
            assert!((px[1][0] - 0.5).abs() < 1e-3 && (px[1][2] - 0.5).abs() < 1e-3);
        }
    }

    #[test]
    fn contrast_turns_about_a_pivot_that_follows_the_image() {
        // The same grey level goes down on a bright image and up on a dark one at Contrast +100.
        let grey = [0.6f32, 0.6, 0.6, 1.0];
        let mut bright: Vec<[f32; 4]> = vec![[0.9, 0.9, 0.9, 1.0]; 256];
        bright[0] = grey;
        let mut dark: Vec<[f32; 4]> = vec![[0.1, 0.1, 0.1, 1.0]; 256];
        dark[0] = grey;
        let p = CameraRaw { contrast: 100.0, ..Default::default() };
        develop(&mut bright, 16, 16, &p, false);
        develop(&mut dark, 16, 16, &p, false);
        assert!(bright[0][1] < 0.6 && dark[0][1] > 0.6, "{} {}", bright[0][1], dark[0][1]);
    }

    #[test]
    fn float_documents_keep_overrange() {
        let mut a = vec![[2.0f32, 1.5, 0.5, 1.0]; 16];
        develop(&mut a, 4, 4, &CameraRaw { exposure: 0.5, ..Default::default() }, true);
        assert!(a[0][0] > 2.0);
    }
}
