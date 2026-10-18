//! Face-Aware Liquify (#3603): Photoshop's face sliders (Eyes, Nose, Mouth, Face Shape) as local
//! warps composed into the Liquify field ([`LiquifyField::compose`]).
//!
//! A face comes as [`FaceGeometry`]: where its eyes, nose, mouth and outline are, either exact,
//! from the 478-point face mesh, or estimated in a face frame from the detector's five landmarks
//! (eyes, nose tip, mouth corners). Each slider is one or two [`LocalWarp`]s: a scale, a turn or
//! a move about a point of the face, strongest there and fading to nothing at a radius
//! proportional to the eye distance, so a slider does the same to a face of any size or tilt.
//! The warps are composed in a fixed order from absolute slider values, so a face stored with its
//! geometry and sliders replays exactly (undo, smart filters).
//!
//! The strengths at ±100 are a first calibration, one constant each (Photoshop's could not be
//! measured: its Face-Aware Liquify needs the GPU, which the measuring installation lacks).

use crate::liquify::LiquifyField;
use photocraft_geom::Rect;

/// Points in the face mesh.
pub const MESH_POINTS: usize = 478;

/// The sliders in the panel's order, as `filter.liquify` names them.
pub const SLIDERS: [&str; 16] = [
    "eyeSize",
    "eyeHeight",
    "eyeWidth",
    "eyeTilt",
    "eyeDistance",
    "noseHeight",
    "noseWidth",
    "smile",
    "upperLip",
    "lowerLip",
    "mouthWidth",
    "mouthHeight",
    "forehead",
    "chinHeight",
    "jawline",
    "faceWidth",
];

/// The eye sliders take a value per eye.
pub const PER_EYE: [&str; 4] = ["eyeSize", "eyeHeight", "eyeWidth", "eyeTilt"];

// Face mesh indices (MediaPipe Face Mesh; lower numbers lie on the image's left).
const LEFT_EYE: [usize; 16] = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
const RIGHT_EYE: [usize; 16] = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];
const NOSE_TIP: usize = 1;
const NOSE_BRIDGE: usize = 6;
const NOSE_BOTTOM: usize = 2;
const NOSE_WINGS: [usize; 2] = [129, 358];
const MOUTH_CORNERS: [usize; 2] = [61, 291];
const LIP_TOP: usize = 0;
const LIP_BOTTOM: usize = 17;
const FOREHEAD: usize = 10;
const CHIN: usize = 152;
const JAW: [usize; 2] = [172, 397];
const CHEEKS: [usize; 2] = [234, 454];

// Strengths at ±100, in eye distances (`e`) or as scale factors.
const EYE_SCALE: f64 = 0.5;
const EYE_TILT_DEG: f64 = 20.0;
const EYE_DISTANCE: f64 = 0.12;
const EYE_REACH: f64 = 2.4;
const NOSE_HEIGHT: f64 = 0.15;
const NOSE_WIDTH: f64 = 0.35;
const NOSE_REACH: f64 = 0.6;
const SMILE_UP: f64 = 0.15;
const SMILE_OUT: f64 = 0.04;
const CORNER_REACH: f64 = 0.45;
const LIP: f64 = 0.1;
const LIP_REACH: f64 = 0.45;
const MOUTH_SCALE: f64 = 0.35;
const MOUTH_REACH: [f64; 2] = [0.8, 0.6];
const FOREHEAD_RISE: f64 = 0.2;
const FOREHEAD_REACH: f64 = 1.2;
const CHIN_DROP: f64 = 0.2;
const CHIN_REACH: f64 = 0.9;
const JAW_OUT: f64 = 0.15;
const JAW_REACH: f64 = 0.8;
const FACE_WIDTH: f64 = 0.2;
const FACE_REACH: f64 = 2.3;

/// Where a face's features came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeometrySource {
    /// The detector's five landmarks: the rest of the face at typical proportions.
    Landmarks,
    /// The 478-point face mesh.
    Mesh,
}

/// An eye: its centre and half extents along the face frame's `u` (across) and `v` (down).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eye {
    pub centre: [f64; 2],
    pub half: [f64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nose {
    pub tip: [f64; 2],
    pub bridge: [f64; 2],
    pub bottom: [f64; 2],
    /// Image-left, image-right.
    pub wings: [[f64; 2]; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mouth {
    /// Image-left, image-right.
    pub corners: [[f64; 2]; 2],
    /// Top of the upper lip, bottom of the lower lip, and the centre between the corners.
    pub top: [f64; 2],
    pub bottom: [f64; 2],
    pub centre: [f64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Outline {
    pub forehead: [f64; 2],
    pub chin: [f64; 2],
    /// Image-left, image-right, at the mouth's level.
    pub jaw: [[f64; 2]; 2],
    /// Image-left, image-right, at the eyes' level.
    pub cheeks: [[f64; 2]; 2],
}

/// Where a face's features are, in document pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceGeometry {
    pub source: GeometrySource,
    /// Between the eyes.
    pub centre: [f64; 2],
    /// Unit vector from the image-left eye to the image-right eye.
    pub u: [f64; 2],
    /// Unit vector down the face, towards the mouth.
    pub v: [f64; 2],
    /// The eye distance (px): the face's scale.
    pub e: f64,
    /// Image-left eye, image-right eye.
    pub eyes: [Eye; 2],
    pub nose: Nose,
    pub mouth: Mouth,
    pub outline: Outline,
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn add(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn mul(a: [f64; 2], k: f64) -> [f64; 2] {
    [a[0] * k, a[1] * k]
}

fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn finite(p: [f64; 2]) -> bool {
    p[0].is_finite() && p[1].is_finite()
}

fn pt(p: [f32; 2]) -> [f64; 2] {
    [f64::from(p[0]), f64::from(p[1])]
}

/// The face frame: the centre between the eyes, `u` across, `v` down the face, the eye distance.
struct Frame {
    centre: [f64; 2],
    u: [f64; 2],
    v: [f64; 2],
    e: f64,
}

/// The face frame from the two eyes and a point below them: `None` when the eyes coincide or
/// anything is not finite.
fn frame(left: [f64; 2], right: [f64; 2], below: [f64; 2]) -> Option<Frame> {
    let d = sub(right, left);
    let e = d[0].hypot(d[1]);
    if !(e.is_finite() && e > 1e-3) || !finite(left) || !finite(below) {
        return None;
    }
    let u = mul(d, 1.0 / e);
    let centre = mul(add(left, right), 0.5);
    // Down the face is the normal to `u` on the mouth's side.
    let mut v = [-u[1], u[0]];
    if dot(v, sub(below, centre)) < 0.0 {
        v = mul(v, -1.0);
    }
    Some(Frame { centre, u, v, e })
}

impl FaceGeometry {
    /// Document point `du` eye distances across and `dv` down from the frame's centre.
    pub fn at(&self, du: f64, dv: f64) -> [f64; 2] {
        add(self.centre, add(mul(self.u, du * self.e), mul(self.v, dv * self.e)))
    }

    /// From the detector's five landmarks (image-left eye, image-right eye, nose tip, image-left
    /// mouth corner, image-right mouth corner): the frame from the eyes and the mouth, the eyes'
    /// extents and the outline at typical proportions.
    pub fn from_landmarks(l: &[[f32; 2]; 5]) -> Option<FaceGeometry> {
        let p: Vec<[f64; 2]> = l.iter().copied().map(pt).collect();
        let mouth_centre = mul(add(p[3], p[4]), 0.5);
        let Frame { centre, u, v, e } = frame(p[0], p[1], mouth_centre)?;
        if !finite(p[2]) {
            return None;
        }
        let at = |du: f64, dv: f64| add(centre, add(mul(u, du * e), mul(v, dv * e)));
        let eye = |c: [f64; 2]| Eye { centre: c, half: [0.22 * e, 0.12 * e] };
        let nose_tip = p[2];
        Some(FaceGeometry {
            source: GeometrySource::Landmarks,
            centre,
            u,
            v,
            e,
            eyes: [eye(p[0]), eye(p[1])],
            nose: Nose {
                tip: nose_tip,
                bridge: at(0.0, 0.1),
                bottom: add(nose_tip, mul(v, 0.12 * e)),
                wings: [add(nose_tip, add(mul(u, -0.2 * e), mul(v, 0.04 * e))), add(nose_tip, add(mul(u, 0.2 * e), mul(v, 0.04 * e)))],
            },
            mouth: Mouth {
                corners: [p[3], p[4]],
                top: add(mouth_centre, mul(v, -0.12 * e)),
                bottom: add(mouth_centre, mul(v, 0.15 * e)),
                centre: mouth_centre,
            },
            outline: Outline {
                forehead: at(0.0, -0.95),
                chin: add(mouth_centre, mul(v, 0.75 * e)),
                jaw: [add(mouth_centre, add(mul(u, -0.72 * e), mul(v, 0.25 * e))), add(mouth_centre, add(mul(u, 0.72 * e), mul(v, 0.25 * e)))],
                cheeks: [at(-1.05, 0.25), at(1.05, 0.25)],
            },
        })
    }

    /// From the 478-point face mesh (`None` with fewer points or a degenerate face).
    pub fn from_mesh(m: &[[f32; 2]]) -> Option<FaceGeometry> {
        if m.len() < MESH_POINTS {
            return None;
        }
        let g = |i: usize| m.get(i).copied().map(pt).filter(|p| finite(*p));
        let mean = |idx: &[usize]| -> Option<[f64; 2]> {
            let pts: Option<Vec<[f64; 2]>> = idx.iter().map(|&i| g(i)).collect();
            let pts = pts?;
            Some(mul(pts.iter().fold([0.0; 2], |a, p| add(a, *p)), 1.0 / pts.len() as f64))
        };
        let (le, re) = (mean(&LEFT_EYE)?, mean(&RIGHT_EYE)?);
        let mouth_centre = mul(add(g(MOUTH_CORNERS[0])?, g(MOUTH_CORNERS[1])?), 0.5);
        let Frame { centre, u, v, e } = frame(le, re, mouth_centre)?;
        // An eye's half extents: its contour projected on the frame.
        let eye = |idx: &[usize], c: [f64; 2]| -> Option<Eye> {
            let mut lo = [f64::MAX; 2];
            let mut hi = [f64::MIN; 2];
            for &i in idx {
                let d = sub(g(i)?, c);
                for (k, axis) in [u, v].into_iter().enumerate() {
                    let t = dot(d, axis);
                    lo[k] = lo[k].min(t);
                    hi[k] = hi[k].max(t);
                }
            }
            Some(Eye { centre: c, half: [((hi[0] - lo[0]) / 2.0).max(0.02 * e), ((hi[1] - lo[1]) / 2.0).max(0.01 * e)] })
        };
        Some(FaceGeometry {
            source: GeometrySource::Mesh,
            centre,
            u,
            v,
            e,
            eyes: [eye(&LEFT_EYE, le)?, eye(&RIGHT_EYE, re)?],
            nose: Nose { tip: g(NOSE_TIP)?, bridge: g(NOSE_BRIDGE)?, bottom: g(NOSE_BOTTOM)?, wings: [g(NOSE_WINGS[0])?, g(NOSE_WINGS[1])?] },
            mouth: Mouth { corners: [g(MOUTH_CORNERS[0])?, g(MOUTH_CORNERS[1])?], top: g(LIP_TOP)?, bottom: g(LIP_BOTTOM)?, centre: mouth_centre },
            outline: Outline { forehead: g(FOREHEAD)?, chin: g(CHIN)?, jaw: [g(JAW[0])?, g(JAW[1])?], cheeks: [g(CHEEKS[0])?, g(CHEEKS[1])?] },
        })
    }

    /// The document rectangle the face's warps can reach.
    pub fn reach(&self) -> Rect {
        let r = FACE_REACH * self.e + 1.0;
        let c = self.at(0.0, 0.5);
        Rect::new((c[0] - r).floor() as i32, (c[1] - r).floor() as i32, (c[0] + r).ceil() as i32 + 1, (c[1] + r).ceil() as i32 + 1)
    }
}

/// The sliders, each −100..100 with 0 as unchanged. The eye ones hold the image-left eye and the
/// image-right eye.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct FaceSliders {
    pub eye_size: [f32; 2],
    pub eye_height: [f32; 2],
    pub eye_width: [f32; 2],
    pub eye_tilt: [f32; 2],
    pub eye_distance: f32,
    pub nose_height: f32,
    pub nose_width: f32,
    pub smile: f32,
    pub upper_lip: f32,
    pub lower_lip: f32,
    pub mouth_width: f32,
    pub mouth_height: f32,
    pub forehead: f32,
    pub chin_height: f32,
    pub jawline: f32,
    pub face_width: f32,
}

impl FaceSliders {
    pub fn is_neutral(&self) -> bool {
        *self == FaceSliders::default()
    }

    /// The slider `key`'s values (both eyes for a per-eye slider, the same value twice otherwise).
    pub fn get(&self, key: &str) -> Option<[f32; 2]> {
        Some(match key {
            "eyeSize" => self.eye_size,
            "eyeHeight" => self.eye_height,
            "eyeWidth" => self.eye_width,
            "eyeTilt" => self.eye_tilt,
            "eyeDistance" => [self.eye_distance; 2],
            "noseHeight" => [self.nose_height; 2],
            "noseWidth" => [self.nose_width; 2],
            "smile" => [self.smile; 2],
            "upperLip" => [self.upper_lip; 2],
            "lowerLip" => [self.lower_lip; 2],
            "mouthWidth" => [self.mouth_width; 2],
            "mouthHeight" => [self.mouth_height; 2],
            "forehead" => [self.forehead; 2],
            "chinHeight" => [self.chin_height; 2],
            "jawline" => [self.jawline; 2],
            "faceWidth" => [self.face_width; 2],
            _ => return None,
        })
    }

    /// Sets slider `key` (clamped to −100..100); `eye` picks one eye of a per-eye slider, `None`
    /// sets both. False for an unknown key.
    pub fn set(&mut self, key: &str, value: f32, eye: Option<usize>) -> bool {
        let value = if value.is_finite() { value.clamp(-100.0, 100.0) } else { 0.0 };
        let pair = |slot: &mut [f32; 2]| match eye {
            Some(i) if i < 2 => slot[i] = value,
            _ => *slot = [value; 2],
        };
        match key {
            "eyeSize" => pair(&mut self.eye_size),
            "eyeHeight" => pair(&mut self.eye_height),
            "eyeWidth" => pair(&mut self.eye_width),
            "eyeTilt" => pair(&mut self.eye_tilt),
            "eyeDistance" => self.eye_distance = value,
            "noseHeight" => self.nose_height = value,
            "noseWidth" => self.nose_width = value,
            "smile" => self.smile = value,
            "upperLip" => self.upper_lip = value,
            "lowerLip" => self.lower_lip = value,
            "mouthWidth" => self.mouth_width = value,
            "mouthHeight" => self.mouth_height = value,
            "forehead" => self.forehead = value,
            "chinHeight" => self.chin_height = value,
            "jawline" => self.jawline = value,
            "faceWidth" => self.face_width = value,
            _ => return false,
        }
        true
    }
}

/// A smooth local affine warp: content at `x` within `radius` of `centre` moves by
/// `fall(x) · ((M − I)(x − centre) + t)`, with `fall = (1 − (|x − centre| / radius)²)²`: the
/// full move at the centre, nothing at the radius, no edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalWarp {
    pub centre: [f64; 2],
    pub radius: f64,
    pub m: [[f64; 2]; 2],
    pub t: [f64; 2],
}

impl LocalWarp {
    fn fall(&self, x: [f64; 2]) -> f64 {
        let d = sub(x, self.centre);
        let t2 = dot(d, d) / (self.radius * self.radius);
        if t2 >= 1.0 { 0.0 } else { (1.0 - t2).powi(2) }
    }

    /// How far the content at `x` moves.
    pub fn forward(&self, x: [f64; 2]) -> [f64; 2] {
        let k = self.fall(x);
        if k == 0.0 {
            return [0.0; 2];
        }
        let d = sub(x, self.centre);
        let a = [(self.m[0][0] - 1.0) * d[0] + self.m[0][1] * d[1] + self.t[0], self.m[1][0] * d[0] + (self.m[1][1] - 1.0) * d[1] + self.t[1]];
        mul(a, k)
    }

    /// The source point the output at `p` shows after the warp (the inverse of
    /// `x ↦ x + forward(x)`, by fixed-point iteration: the moves are small against the radius).
    pub fn inverse(&self, p: [f64; 2]) -> [f64; 2] {
        let mut q = p;
        for _ in 0..8 {
            q = sub(p, self.forward(q));
        }
        q
    }

    /// The document rectangle the warp reaches.
    pub fn region(&self) -> Rect {
        let r = self.radius + 1.0;
        Rect::new(
            (self.centre[0] - r).floor() as i32,
            (self.centre[1] - r).floor() as i32,
            (self.centre[0] + r).ceil() as i32 + 1,
            (self.centre[1] + r).ceil() as i32 + 1,
        )
    }

    /// A scale by `1 + su` along `u` and `1 + sv` along `v` about `centre`.
    fn scale(centre: [f64; 2], radius: f64, u: [f64; 2], v: [f64; 2], su: f64, sv: f64) -> LocalWarp {
        let (a, b) = (1.0 + su, 1.0 + sv);
        let m =
            [[a * u[0] * u[0] + b * v[0] * v[0], a * u[0] * u[1] + b * v[0] * v[1]], [a * u[1] * u[0] + b * v[1] * v[0], a * u[1] * u[1] + b * v[1] * v[1]]];
        LocalWarp { centre, radius, m, t: [0.0; 2] }
    }

    /// A turn by `angle` (radians, image y down) about `centre`.
    fn turn(centre: [f64; 2], radius: f64, angle: f64) -> LocalWarp {
        let (s, c) = angle.sin_cos();
        LocalWarp { centre, radius, m: [[c, -s], [s, c]], t: [0.0; 2] }
    }

    /// A move of the content about `centre` by `t`.
    fn shift(centre: [f64; 2], radius: f64, t: [f64; 2]) -> LocalWarp {
        LocalWarp { centre, radius, m: [[1.0, 0.0], [0.0, 1.0]], t }
    }
}

/// The warps for `sliders` on face `g`, in the order they are composed: eyes, nose, mouth, face
/// shape. Neutral sliders give none.
pub fn face_warps(g: &FaceGeometry, s: &FaceSliders) -> Vec<LocalWarp> {
    let mut out = Vec::new();
    let (u, v, e) = (g.u, g.v, g.e);
    let k = |x: f32| f64::from(x) / 100.0;
    for (i, eye) in g.eyes.iter().enumerate() {
        let r = eye.half[0].max(eye.half[1]) * EYE_REACH;
        if s.eye_size[i] != 0.0 {
            let a = EYE_SCALE * k(s.eye_size[i]);
            out.push(LocalWarp::scale(eye.centre, r, u, v, a, a));
        }
        if s.eye_height[i] != 0.0 {
            out.push(LocalWarp::scale(eye.centre, r, u, v, 0.0, EYE_SCALE * k(s.eye_height[i])));
        }
        if s.eye_width[i] != 0.0 {
            out.push(LocalWarp::scale(eye.centre, r, u, v, EYE_SCALE * k(s.eye_width[i]), 0.0));
        }
        if s.eye_tilt[i] != 0.0 {
            // Positive lifts the outer corner of both eyes: the image-left eye turns one way, the
            // image-right eye the other.
            let sign = if i == 0 { 1.0 } else { -1.0 };
            out.push(LocalWarp::turn(eye.centre, r, sign * EYE_TILT_DEG.to_radians() * k(s.eye_tilt[i])));
        }
        if s.eye_distance != 0.0 {
            let sign = if i == 0 { -1.0 } else { 1.0 };
            out.push(LocalWarp::shift(eye.centre, r, mul(u, sign * EYE_DISTANCE * e * k(s.eye_distance))));
        }
    }
    if s.nose_height != 0.0 {
        // Positive: a longer nose, its tip further down the face.
        out.push(LocalWarp::shift(g.nose.tip, NOSE_REACH * e, mul(v, NOSE_HEIGHT * e * k(s.nose_height))));
    }
    if s.nose_width != 0.0 {
        let c = mul(add(g.nose.wings[0], g.nose.wings[1]), 0.5);
        out.push(LocalWarp::scale(c, NOSE_REACH * e, u, v, NOSE_WIDTH * k(s.nose_width), 0.0));
    }
    if s.mouth_width != 0.0 {
        out.push(LocalWarp::scale(g.mouth.centre, MOUTH_REACH[0] * e, u, v, MOUTH_SCALE * k(s.mouth_width), 0.0));
    }
    if s.mouth_height != 0.0 {
        out.push(LocalWarp::scale(g.mouth.centre, MOUTH_REACH[1] * e, u, v, 0.0, MOUTH_SCALE * k(s.mouth_height)));
    }
    if s.smile != 0.0 {
        // Positive lifts the corners and draws them a little apart.
        for (i, c) in g.mouth.corners.iter().enumerate() {
            let out_sign = if i == 0 { -1.0 } else { 1.0 };
            let t = add(mul(v, -SMILE_UP * e * k(s.smile)), mul(u, out_sign * SMILE_OUT * e * k(s.smile)));
            out.push(LocalWarp::shift(*c, CORNER_REACH * e, t));
        }
    }
    if s.upper_lip != 0.0 {
        // Positive: a fuller upper lip, its top edge higher.
        out.push(LocalWarp::shift(g.mouth.top, LIP_REACH * e, mul(v, -LIP * e * k(s.upper_lip))));
    }
    if s.lower_lip != 0.0 {
        out.push(LocalWarp::shift(g.mouth.bottom, LIP_REACH * e, mul(v, LIP * e * k(s.lower_lip))));
    }
    if s.face_width != 0.0 {
        out.push(LocalWarp::scale(g.at(0.0, 0.5), FACE_REACH * e, u, v, FACE_WIDTH * k(s.face_width), 0.0));
    }
    if s.jawline != 0.0 {
        for (i, j) in g.outline.jaw.iter().enumerate() {
            let sign = if i == 0 { -1.0 } else { 1.0 };
            out.push(LocalWarp::shift(*j, JAW_REACH * e, mul(u, sign * JAW_OUT * e * k(s.jawline))));
        }
    }
    if s.chin_height != 0.0 {
        out.push(LocalWarp::shift(g.outline.chin, CHIN_REACH * e, mul(v, CHIN_DROP * e * k(s.chin_height))));
    }
    if s.forehead != 0.0 {
        // Positive: a taller forehead, the hairline higher.
        let c = add(g.outline.forehead, mul(v, -0.2 * e));
        out.push(LocalWarp::shift(c, FOREHEAD_REACH * e, mul(v, -FOREHEAD_RISE * e * k(s.forehead))));
    }
    out
}

/// What a drag along one axis at a handle does: which slider (and eye), the unit direction in
/// the document, and how far the handle's feature moves at +100.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Binding {
    pub slider: &'static str,
    pub eye: Option<usize>,
    pub axis: [f64; 2],
    pub per_hundred: f64,
}

/// A handle of the Face tool: where it sits on the face and what dragging it moves.
#[derive(Clone, Debug, PartialEq)]
pub struct Handle {
    pub at: [f64; 2],
    pub along: Vec<Binding>,
}

impl Handle {
    /// The slider values after a drag by `delta` (document px) from the values at its start.
    pub fn drag(&self, start: &FaceSliders, delta: [f64; 2], both_eyes: bool) -> FaceSliders {
        let mut s = *start;
        for b in &self.along {
            if b.per_hundred <= 0.0 {
                continue;
            }
            let eye = b.eye.filter(|_| !both_eyes);
            let from = start.get(b.slider).map_or(0.0, |v| v[b.eye.unwrap_or(0)]);
            let value = f64::from(from) + 100.0 * dot(delta, b.axis) / b.per_hundred;
            s.set(b.slider, value as f32, eye);
        }
        s
    }
}

/// The Face tool's handles on `g`: the eyes' top (height), bottom (size), outer (width) and inner
/// (distance) edges; the nose's bottom (height) and wings (width); the mouth's corners (smile up,
/// width across), top (upper lip) and bottom (lower lip); the forehead, the chin, the jaw and the
/// cheeks.
pub fn handles(g: &FaceGeometry) -> Vec<Handle> {
    let (u, v, e) = (g.u, g.v, g.e);
    let neg = |a: [f64; 2]| mul(a, -1.0);
    let bind = |slider: &'static str, eye: Option<usize>, axis: [f64; 2], per_hundred: f64| Binding { slider, eye, axis, per_hundred };
    let mut out = Vec::new();
    for (i, eye) in g.eyes.iter().enumerate() {
        let side = if i == 0 { neg(u) } else { u };
        let (hw, hh) = (eye.half[0], eye.half[1]);
        out.push(Handle { at: add(eye.centre, mul(v, -hh)), along: vec![bind("eyeHeight", Some(i), neg(v), EYE_SCALE * hh)] });
        out.push(Handle { at: add(eye.centre, mul(v, hh)), along: vec![bind("eyeSize", Some(i), v, EYE_SCALE * hh)] });
        out.push(Handle { at: add(eye.centre, mul(side, hw)), along: vec![bind("eyeWidth", Some(i), side, EYE_SCALE * hw)] });
        out.push(Handle { at: add(eye.centre, mul(side, -hw)), along: vec![bind("eyeDistance", None, side, EYE_DISTANCE * e)] });
    }
    out.push(Handle { at: g.nose.bottom, along: vec![bind("noseHeight", None, v, NOSE_HEIGHT * e)] });
    let wing_half = (dot(sub(g.nose.wings[1], g.nose.wings[0]), u) / 2.0).abs().max(0.05 * e);
    for (i, w) in g.nose.wings.iter().enumerate() {
        let side = if i == 0 { neg(u) } else { u };
        out.push(Handle { at: *w, along: vec![bind("noseWidth", None, side, NOSE_WIDTH * wing_half)] });
    }
    let mouth_half = (dot(sub(g.mouth.corners[1], g.mouth.corners[0]), u) / 2.0).abs().max(0.05 * e);
    for (i, c) in g.mouth.corners.iter().enumerate() {
        let side = if i == 0 { neg(u) } else { u };
        out.push(Handle { at: *c, along: vec![bind("smile", None, neg(v), SMILE_UP * e), bind("mouthWidth", None, side, MOUTH_SCALE * mouth_half)] });
    }
    out.push(Handle { at: g.mouth.top, along: vec![bind("upperLip", None, neg(v), LIP * e)] });
    out.push(Handle { at: g.mouth.bottom, along: vec![bind("lowerLip", None, v, LIP * e)] });
    out.push(Handle { at: g.outline.forehead, along: vec![bind("forehead", None, neg(v), FOREHEAD_RISE * e)] });
    out.push(Handle { at: g.outline.chin, along: vec![bind("chinHeight", None, v, CHIN_DROP * e)] });
    for (i, j) in g.outline.jaw.iter().enumerate() {
        let side = if i == 0 { neg(u) } else { u };
        out.push(Handle { at: *j, along: vec![bind("jawline", None, side, JAW_OUT * e)] });
    }
    let cheek_half = (dot(sub(g.outline.cheeks[1], g.outline.cheeks[0]), u) / 2.0).abs().max(0.5 * e);
    for (i, c) in g.outline.cheeks.iter().enumerate() {
        let side = if i == 0 { neg(u) } else { u };
        out.push(Handle { at: *c, along: vec![bind("faceWidth", None, side, FACE_WIDTH * cheek_half)] });
    }
    out
}

/// Composes the face's sliders into `field`. Returns the document rect that changed.
pub fn apply_face(field: &mut LiquifyField, g: &FaceGeometry, s: &FaceSliders) -> Rect {
    let mut changed = Rect::EMPTY;
    for w in face_warps(g, s) {
        let r = field.compose(w.region(), &|p| w.inverse(p));
        if !r.is_empty() {
            changed = if changed.is_empty() { r } else { changed.union(&r) };
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An upright face: eyes 100 px apart at y = 100, nose and mouth below.
    const FACE: [[f32; 2]; 5] = [[150.0, 100.0], [250.0, 100.0], [200.0, 150.0], [170.0, 190.0], [230.0, 190.0]];

    fn field() -> LiquifyField {
        LiquifyField::new(Rect::new(0, 0, 400, 400), 2.0)
    }

    #[test]
    fn the_frame_follows_the_eyes_and_the_mouth_sets_down() {
        let g = FaceGeometry::from_landmarks(&FACE).unwrap();
        assert_eq!((g.centre, g.e), ([200.0, 100.0], 100.0));
        assert!((g.u[0] - 1.0).abs() < 1e-9 && g.u[1].abs() < 1e-9);
        assert!(g.v[0].abs() < 1e-9 && (g.v[1] - 1.0).abs() < 1e-9, "down is +y for an upright face: {:?}", g.v);
        assert_eq!(g.source, GeometrySource::Landmarks);
        assert!(g.outline.chin[1] > g.mouth.centre[1] && g.outline.forehead[1] < g.centre[1]);
        // Turned 90° (eyes one above the other, the mouth to the left): the frame turns with it.
        let turned = [[300.0, 150.0], [300.0, 250.0], [250.0, 200.0], [210.0, 170.0], [210.0, 230.0]];
        let g = FaceGeometry::from_landmarks(&turned).unwrap();
        assert!((g.u[1] - 1.0).abs() < 1e-9 && (g.v[0] + 1.0).abs() < 1e-9, "u {:?} v {:?}", g.u, g.v);
        // Upside down: the mouth is above the eyes, so down the face is −y.
        let flipped = [[250.0, 300.0], [150.0, 300.0], [200.0, 250.0], [230.0, 210.0], [170.0, 210.0]];
        let g = FaceGeometry::from_landmarks(&flipped).unwrap();
        assert!((g.v[1] + 1.0).abs() < 1e-9, "{:?}", g.v);
        assert!(FaceGeometry::from_landmarks(&[[1.0, 1.0]; 5]).is_none(), "coincident eyes");
        assert!(FaceGeometry::from_landmarks(&[[f32::NAN, 1.0], [2.0, 1.0], [1.0, 2.0], [0.0, 3.0], [2.0, 3.0]]).is_none());
    }

    #[test]
    fn the_mesh_gives_the_features_exactly() {
        let mut m = vec![[200.0f32, 150.0]; MESH_POINTS];
        for (i, p) in LEFT_EYE.iter().zip([[140.0, 90.0], [160.0, 90.0], [150.0, 85.0], [150.0, 95.0]].iter().cycle()) {
            m[*i] = *p;
        }
        for (i, p) in RIGHT_EYE.iter().zip([[240.0, 90.0], [260.0, 90.0], [250.0, 85.0], [250.0, 95.0]].iter().cycle()) {
            m[*i] = *p;
        }
        m[MOUTH_CORNERS[0]] = [170.0, 190.0];
        m[MOUTH_CORNERS[1]] = [230.0, 190.0];
        m[NOSE_TIP] = [200.0, 150.0];
        m[CHIN] = [200.0, 250.0];
        m[FOREHEAD] = [200.0, 10.0];
        let g = FaceGeometry::from_mesh(&m).unwrap();
        assert_eq!(g.source, GeometrySource::Mesh);
        assert_eq!(g.e, 100.0);
        assert_eq!(g.eyes[0].centre, [150.0, 90.0]);
        assert_eq!(g.eyes[0].half, [10.0, 5.0]);
        assert_eq!(g.mouth.centre, [200.0, 190.0]);
        assert_eq!(g.outline.chin, [200.0, 250.0]);
        assert!(FaceGeometry::from_mesh(&m[..400]).is_none(), "too few points");
    }

    #[test]
    fn neutral_sliders_change_nothing() {
        let g = FaceGeometry::from_landmarks(&FACE).unwrap();
        let mut f = field();
        assert!(face_warps(&g, &FaceSliders::default()).is_empty());
        assert!(apply_face(&mut f, &g, &FaceSliders::default()).is_empty());
        assert!(f.is_identity());
    }

    #[test]
    fn eye_size_grows_the_eye_about_its_centre_and_leaves_the_rest() {
        let g = FaceGeometry::from_landmarks(&FACE).unwrap();
        let mut f = field();
        let mut s = FaceSliders { eye_size: [100.0, 0.0], ..Default::default() };
        let changed = apply_face(&mut f, &g, &s);
        assert!(changed.contains(150, 100) && !changed.contains(250, 100), "only the image-left eye: {changed:?}");
        // The output beside the eye's centre shows source nearer the centre (the eye grew).
        let d = f.sample(165.0, 100.0);
        assert!(d[0] < -2.0 && d[1].abs() < 0.5, "{d:?}");
        let d = f.sample(150.0, 108.0);
        assert!(d[1] < -1.0, "{d:?}");
        assert!(f.sample(150.0, 100.0).iter().all(|v| v.abs() < 0.01), "the centre stays: {:?}", f.sample(150.0, 100.0));
        assert_eq!(f.sample(250.0, 100.0), [0.0, 0.0], "the other eye stays");
        assert_eq!(f.sample(200.0, 300.0), [0.0, 0.0], "far away nothing moves");
        // Negative shrinks it.
        let mut f = field();
        s.eye_size = [-100.0, -100.0];
        apply_face(&mut f, &g, &s);
        assert!(f.sample(165.0, 100.0)[0] > 2.0 && f.sample(265.0, 100.0)[0] > 2.0);
    }

    #[test]
    fn smile_lifts_the_corners_and_chin_height_moves_the_chin_both_ways() {
        let g = FaceGeometry::from_landmarks(&FACE).unwrap();
        let s = FaceSliders { smile: 100.0, ..Default::default() };
        let mut f = field();
        apply_face(&mut f, &g, &s);
        // Content moved up: the output at the corner shows source from below.
        let d = f.sample(170.0, 190.0);
        assert!(d[1] > 3.0, "left corner {d:?}");
        let d = f.sample(230.0, 190.0);
        assert!(d[1] > 3.0 && d[0] < 0.0, "right corner {d:?}");
        for (v, sign) in [(100.0f32, 1.0f64), (-100.0, -1.0)] {
            let s = FaceSliders { chin_height: v, ..Default::default() };
            let mut f = field();
            apply_face(&mut f, &g, &s);
            let chin = g.outline.chin;
            let d = f.sample(chin[0], chin[1]);
            assert!(d[1] * sign < -3.0, "chin {v}: {d:?}");
        }
    }

    #[test]
    fn warps_compose_with_strokes_and_replay_the_same() {
        let g = FaceGeometry::from_landmarks(&FACE).unwrap();
        let mut s = FaceSliders { face_width: 60.0, ..Default::default() };
        s.eye_tilt = [40.0, 40.0];
        let mut a = field();
        apply_face(&mut a, &g, &s);
        let mut b = field();
        apply_face(&mut b, &g, &s);
        assert_eq!(a, b, "deterministic");
        assert!(!a.is_identity());
        let r = LocalWarp::shift([100.0, 100.0], 50.0, [10.0, 0.0]);
        let p = r.inverse([105.0, 100.0]);
        let back = add(p, r.forward(p));
        assert!((back[0] - 105.0).abs() < 1e-6 && (back[1] - 100.0).abs() < 1e-6, "the inverse inverts: {back:?}");
    }

    #[test]
    fn handles_sit_on_the_features_and_drag_their_sliders() {
        let g = FaceGeometry::from_landmarks(&FACE).unwrap();
        let h = handles(&g);
        assert_eq!(h.len(), 8 + 3 + 4 + 2 + 2 + 2);
        let find = |slider: &str, eye: Option<usize>| h.iter().find(|h| h.along.iter().any(|b| b.slider == slider && b.eye == eye)).unwrap();
        // The image-left eye's top handle: dragging it up (−y) raises Eye Height; a drag of the
        // feature's full +100 move gives +100, and the other eye follows only when linked.
        let top = find("eyeHeight", Some(0));
        assert!((top.at[1] - (100.0 - g.eyes[0].half[1])).abs() < 1e-9 && (top.at[0] - 150.0).abs() < 1e-9);
        let per = top.along[0].per_hundred;
        let s = top.drag(&FaceSliders::default(), [0.0, -per], false);
        assert_eq!(s.eye_height, [100.0, 0.0]);
        let s = top.drag(&FaceSliders::default(), [0.0, -per / 2.0], true);
        assert_eq!(s.eye_height, [50.0, 50.0]);
        let s = top.drag(&FaceSliders::default(), [0.0, per * 5.0], false);
        assert_eq!(s.eye_height, [-100.0, 0.0], "clamped");
        // A mouth corner: up is Smile, outward is Mouth Width, both from one drag.
        let corner = find("smile", None);
        assert_eq!(corner.at, g.mouth.corners[0]);
        let s = corner.drag(&FaceSliders::default(), [-corner.along[1].per_hundred, -corner.along[0].per_hundred], false);
        assert_eq!((s.smile, s.mouth_width), (100.0, 100.0));
        // The chin handle moves Chin Height down the face; on a turned face the axis turns too.
        let turned = [[300.0, 150.0], [300.0, 250.0], [250.0, 200.0], [210.0, 170.0], [210.0, 230.0]];
        let g2 = FaceGeometry::from_landmarks(&turned).unwrap();
        let chin = handles(&g2).into_iter().find(|h| h.along[0].slider == "chinHeight").unwrap();
        assert!((chin.along[0].axis[0] + 1.0).abs() < 1e-9, "down the turned face is −x: {:?}", chin.along[0].axis);
    }
}
