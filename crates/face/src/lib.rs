//! The 478-point face mesh for PhotoCraft's face-aware tools (Face-Aware Liquify).
//!
//! One ONNX model does the work: MediaPipe's face landmark model (Face Mesh v2, Apache-2.0), a
//! 256 × 256 crop of one face in, 478 points and a face-presence score out. It runs on
//! [`tract_onnx`], so there is no C or C++ runtime and nothing to install beyond the model file.
//! The model is not bundled: the caller passes its bytes ([`FaceMesh::from_onnx`]) after finding
//! the file wherever the Craft apps keep their models.
//!
//! Faces are found elsewhere (PhotoCraft's own detector, `photocraft_algo::faces`, gives a box and
//! the eyes). From those [`Roi::around`] builds the square region the model expects: 1.5 × the
//! box, rotated so the eyes are level. [`FaceMesh::mesh`] crops it with bilinear sampling, runs
//! the model and maps the points back into image pixels.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::fmt;

use tract_onnx::prelude::*;

/// Landmarks in the face mesh.
pub const MESH_POINTS: usize = 478;
/// The model's input side (px).
pub const CROP: usize = 256;
/// Region side over the face box's longer side, as MediaPipe crops.
pub const ROI_SCALE: f32 = 1.5;

/// Something went wrong loading or running the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaceError(pub String);

impl fmt::Display for FaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FaceError {}

fn err(e: impl fmt::Display) -> FaceError {
    FaceError(e.to_string())
}

/// An RGB image, 8 bits per channel, interleaved, rows top to bottom.
#[derive(Clone, Copy, Debug)]
pub struct Rgb8<'a> {
    pub width: usize,
    pub height: usize,
    pub data: &'a [u8],
}

impl Rgb8<'_> {
    /// Does the buffer hold `width × height` RGB pixels?
    pub fn valid(&self) -> bool {
        self.width > 0 && self.height > 0 && self.width.checked_mul(self.height).and_then(|n| n.checked_mul(3)) == Some(self.data.len())
    }

    /// Bilinear sample of channel `c` at `(x, y)` (pixel centres at integers); 0 outside.
    fn sample(&self, x: f32, y: f32, c: usize) -> f32 {
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let at = |xi: i64, yi: i64| -> f32 {
            if xi < 0 || yi < 0 || xi >= self.width as i64 || yi >= self.height as i64 {
                return 0.0;
            }
            let i = (yi as usize * self.width + xi as usize) * 3 + c;
            self.data.get(i).map_or(0.0, |v| f32::from(*v))
        };
        let (xi, yi) = (x0 as i64, y0 as i64);
        let top = at(xi, yi) * (1.0 - fx) + at(xi + 1, yi) * fx;
        let bottom = at(xi, yi + 1) * (1.0 - fx) + at(xi + 1, yi + 1) * fx;
        top * (1.0 - fy) + bottom * fy
    }
}

/// The square region of the image the model looks at: its centre, side (image px) and rotation
/// (radians, image y down; the eyes lie level in the crop).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Roi {
    pub centre: [f32; 2],
    pub size: f32,
    pub angle: f32,
}

impl Roi {
    /// The region MediaPipe crops around a detector box `[x, y, w, h]` whose eyes are at
    /// `eyes` (the image-left one first): [`ROI_SCALE`] × the longer side, square, turned so
    /// the eyes are level. `None` for a degenerate box.
    pub fn around(rect: [f32; 4], eyes: [[f32; 2]; 2]) -> Option<Roi> {
        let size = rect[2].max(rect[3]) * ROI_SCALE;
        if !(size.is_finite() && size > 1.0) || rect.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let centre = [rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0];
        let angle = (eyes[1][1] - eyes[0][1]).atan2(eyes[1][0] - eyes[0][0]);
        Some(Roi { centre, size, angle: if angle.is_finite() { angle } else { 0.0 } })
    }

    /// Crop → image affine `[a, b, c, d, e, f]` (x = a·u + b·v + c, y = d·u + e·v + f): the crop's
    /// centre lands on the region's centre, its side spans `size` pixels, turned by `angle`.
    pub fn affine(&self) -> [f32; 6] {
        let (s, c) = self.angle.sin_cos();
        let k = self.size / CROP as f32;
        let h = CROP as f32 / 2.0;
        [c * k, -s * k, self.centre[0] - (c * h - s * h) * k, s * k, c * k, self.centre[1] - (s * h + c * h) * k]
    }
}

/// One face's mesh: the 478 points in image pixels and the model's presence score.
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    pub points: Vec<[f32; 2]>,
    /// The model's face-presence probability (0–1); MediaPipe keeps faces above about 0.5.
    pub presence: f32,
}

type Plan = std::sync::Arc<TypedRunnableModel>;

/// The loaded landmark model.
pub struct FaceMesh {
    plan: Plan,
}

impl fmt::Debug for FaceMesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FaceMesh")
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-80.0, 80.0)).exp())
}

impl FaceMesh {
    /// Loads the landmark model from ONNX bytes (a `[1, 256, 256, 3]` RGB input in `0..=1`; the
    /// outputs are the 478 × 3 landmarks in crop pixels and the presence logit, in any order).
    pub fn from_onnx(landmarks: &[u8]) -> Result<Self, FaceError> {
        let mut reader = std::io::Cursor::new(landmarks);
        let plan = tract_onnx::onnx()
            .model_for_read(&mut reader)
            .and_then(|m| m.with_input_fact(0, f32::fact([1, CROP, CROP, 3]).into()))
            .and_then(|m| m.into_optimized())
            .and_then(|m| m.into_runnable())
            .map_err(|e| FaceError(format!("face landmark model: {e}")))?;
        Ok(FaceMesh { plan })
    }

    /// The face mesh inside `roi` of `img`.
    pub fn mesh(&self, img: Rgb8, roi: Roi) -> Result<Mesh, FaceError> {
        if !img.valid() {
            return Err(FaceError("the image buffer does not match its size".into()));
        }
        let m = roi.affine();
        if m.iter().any(|v| !v.is_finite()) {
            return Err(FaceError("the face region is not finite".into()));
        }
        let mut input = vec![0.0f32; CROP * CROP * 3];
        for v in 0..CROP {
            for u in 0..CROP {
                let (uf, vf) = (u as f32, v as f32);
                let x = m[0] * uf + m[1] * vf + m[2];
                let y = m[3] * uf + m[4] * vf + m[5];
                for c in 0..3 {
                    if let Some(slot) = input.get_mut((v * CROP + u) * 3 + c) {
                        *slot = img.sample(x, y, c) / 255.0;
                    }
                }
            }
        }
        let t = Tensor::from_shape(&[1, CROP, CROP, 3], &input).map_err(err)?;
        let out = self.plan.run(tvec!(t.into())).map_err(err)?;
        let views: Vec<Vec<f32>> =
            out.iter().map(|o| o.to_plain_array_view::<f32>().map(|a| a.iter().copied().collect()).map_err(err)).collect::<Result<_, _>>()?;
        // The points are the output with 478 × 3 values; the presence logit the one with one.
        let pts = views.iter().find(|v| v.len() >= MESH_POINTS * 3).ok_or_else(|| FaceError("the face landmark model returned no points".into()))?;
        let presence = views.iter().find(|v| v.len() == 1).and_then(|v| v.first().copied()).map_or(1.0, sigmoid);
        let points = pts.as_chunks::<3>().0.iter().take(MESH_POINTS).map(|p| [m[0] * p[0] + m[1] * p[1] + m[2], m[3] * p[0] + m[4] * p[1] + m[5]]).collect();
        Ok(Mesh { points, presence })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_region_is_a_scaled_box_turned_to_level_the_eyes() {
        let roi = Roi::around([100.0, 50.0, 40.0, 60.0], [[110.0, 70.0], [130.0, 70.0]]).unwrap();
        assert_eq!(roi.centre, [120.0, 80.0]);
        assert_eq!(roi.size, 90.0);
        assert_eq!(roi.angle, 0.0);
        let tilted = Roi::around([0.0, 0.0, 40.0, 40.0], [[10.0, 10.0], [30.0, 30.0]]).unwrap();
        assert!((tilted.angle - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
        assert!(Roi::around([0.0, 0.0, 0.0, 0.0], [[0.0; 2]; 2]).is_none());
        assert!(Roi::around([0.0, 0.0, f32::NAN, 40.0], [[0.0; 2]; 2]).is_none());
    }

    #[test]
    fn the_affine_maps_the_crop_centre_to_the_region_centre() {
        let m = Roi { centre: [100.0, 50.0], size: 256.0, angle: 0.3 }.affine();
        let (u, v) = (128.0f32, 128.0f32);
        let x = m[0] * u + m[1] * v + m[2];
        let y = m[3] * u + m[4] * v + m[5];
        assert!((x - 100.0).abs() < 1e-3 && (y - 50.0).abs() < 1e-3);
        // A crop corner lands half a side away from the centre, whatever the angle.
        let (cx, cy) = (m[2] - 100.0, m[5] - 50.0);
        assert!((cx.hypot(cy) - 128.0 * std::f32::consts::SQRT_2).abs() < 1e-2);
    }

    #[test]
    fn bad_input_is_an_error_not_a_panic() {
        assert!(FaceMesh::from_onnx(b"not onnx").is_err());
        let img = Rgb8 { width: 4, height: 4, data: &[0; 5] };
        assert!(!img.valid());
        assert_eq!(Rgb8 { width: 2, height: 2, data: &[255; 12] }.sample(0.5, 0.5, 1), 255.0);
        assert_eq!(Rgb8 { width: 2, height: 2, data: &[255; 12] }.sample(-5.0, 0.0, 0), 0.0);
    }
}
