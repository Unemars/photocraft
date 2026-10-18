//! The 478-point face mesh for Face-Aware Liquify: MediaPipe's face landmark model, where it is
//! looked for, and the mesh of a face the detector found.
//!
//! The model is not bundled. It is the file the Craft apps' shared AI catalogue declares
//! (`face-landmarks/face_landmarks_detector.onnx`, Apache-2.0, about 5 MB), looked for in order in
//! `$PHOTOCRAFT_FACE_MODELS`, the family's shared models folder (`ArtCraft/Craft AI/models/`
//! under the user's data directory) and PhotoCraft's own `photocraft/face-models/`. Without it
//! (or in a build without the `face-mesh` feature) faces keep the detector's five landmarks and
//! Face-Aware Liquify estimates the rest; with it the features are exact.

use std::path::PathBuf;

use photocraft_algo::faces::Face;

/// The model file's name.
pub const MODEL_FILE: &str = "face_landmarks_detector.onnx";
/// Its folder in the shared catalogue.
pub const MODEL_FOLDER: &str = "face-landmarks";
/// Longest side the image is reduced to before the crop is taken (the model sees 256 px).
#[cfg(any(feature = "face-mesh", test))]
const MESH_SIDE: usize = 2048;

/// Whether the mesh can be computed in this build and on this machine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeshStatus {
    /// The model file in use.
    Available(PathBuf),
    /// No model file in any of these places.
    Missing(Vec<PathBuf>),
    /// This build has no `face-mesh` feature.
    NotBuilt,
}

impl MeshStatus {
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            MeshStatus::Available(p) => serde_json::json!({"status": "available", "path": p.display().to_string()}),
            MeshStatus::Missing(looked) => {
                serde_json::json!({"status": "missing", "file": MODEL_FILE, "lookedIn": looked.iter().map(|p| p.display().to_string()).collect::<Vec<_>>()})
            }
            MeshStatus::NotBuilt => serde_json::json!({"status": "notBuilt"}),
        }
    }
}

/// The user's data directory, as the platform names it.
fn data_dir() -> Option<PathBuf> {
    let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    if cfg!(target_os = "windows") {
        return var("APPDATA");
    }
    if cfg!(target_os = "macos") {
        return var("HOME").map(|h| h.join("Library").join("Application Support"));
    }
    var("XDG_DATA_HOME").or_else(|| var("HOME").map(|h| h.join(".local").join("share")))
}

/// Where the model file is looked for, in order.
pub fn candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(d) = std::env::var_os("PHOTOCRAFT_FACE_MODELS").filter(|d| !d.is_empty()) {
        out.push(PathBuf::from(d).join(MODEL_FILE));
    }
    if let Some(data) = data_dir() {
        out.push(data.join("ArtCraft").join("Craft AI").join("models").join(MODEL_FOLDER).join(MODEL_FILE));
        out.push(data.join("photocraft").join("face-models").join(MODEL_FILE));
    }
    out
}

/// The first candidate that exists.
pub fn model_path() -> Option<PathBuf> {
    candidates().into_iter().find(|p| p.is_file())
}

pub fn status() -> MeshStatus {
    if !cfg!(feature = "face-mesh") {
        return MeshStatus::NotBuilt;
    }
    match model_path() {
        Some(p) => MeshStatus::Available(p),
        None => MeshStatus::Missing(candidates()),
    }
}

/// `px` (`w × h` straight RGBA in `0..=1`) as 8-bit RGB reduced so the longer side is at most
/// [`MESH_SIDE`] (box filter; transparent pixels count as white, as the detector sees them).
/// Returns the image and its scale (document px per reduced px).
#[cfg(any(feature = "face-mesh", test))]
fn reduced_rgb(px: &[[f32; 4]], w: usize, h: usize) -> (Vec<u8>, usize, usize, f32) {
    let step = w.max(h).div_ceil(MESH_SIDE).max(1);
    let (sw, sh) = (w.div_ceil(step), h.div_ceil(step));
    let mut rgb = vec![0u8; sw * sh * 3];
    for y in 0..sh {
        for x in 0..sw {
            let (mut acc, mut n) = ([0.0f32; 3], 0.0f32);
            for yy in y * step..((y + 1) * step).min(h) {
                for xx in x * step..((x + 1) * step).min(w) {
                    let Some([r, g, b, a]) = px.get(yy * w + xx) else { continue };
                    let a = a.clamp(0.0, 1.0);
                    for (k, v) in [r, g, b].into_iter().enumerate() {
                        acc[k] += v.clamp(0.0, 1.0) * a + (1.0 - a);
                    }
                    n += 1.0;
                }
            }
            let i = (y * sw + x) * 3;
            if let Some(out) = rgb.get_mut(i..i + 3) {
                for (o, a) in out.iter_mut().zip(acc) {
                    *o = (a / n.max(1.0) * 255.0).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    (rgb, sw, sh, step as f32)
}

#[cfg(feature = "face-mesh")]
mod imp {
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use std::time::SystemTime;

    use photocraft_face::FaceMesh;

    /// The loaded model, keyed by its path and modification time, so a file that appears or
    /// changes while PhotoCraft runs is picked up.
    type Loaded = (PathBuf, Option<SystemTime>, Arc<FaceMesh>);
    static MODEL: Mutex<Option<Loaded>> = Mutex::new(None);

    pub(super) fn model() -> Result<Option<Arc<FaceMesh>>, String> {
        let Some(path) = super::model_path() else { return Ok(None) };
        let modified = std::fs::metadata(&path).ok().and_then(|m| m.modified().ok());
        let mut slot = MODEL.lock().map_err(|_| "face model lock".to_string())?;
        if let Some((p, m, model)) = slot.as_ref()
            && *p == path
            && *m == modified
        {
            return Ok(Some(model.clone()));
        }
        let bytes = std::fs::read(&path).map_err(|e| format!("face landmark model: cannot read {}: {e}", path.display()))?;
        let model = Arc::new(FaceMesh::from_onnx(&bytes).map_err(|e| format!("{} : {e}", path.display()))?);
        *slot = Some((path, modified, model.clone()));
        Ok(Some(model))
    }
}

/// One mesh per detected face: the 478 points, or `None` for a face the model finds nothing in.
pub type Meshes = Vec<Option<Vec<[f32; 2]>>>;

/// The meshes of `faces` (the detector's, in pixels of `px`, `w × h` straight RGBA): one per
/// face, `None` for a face the model finds nothing in. `Ok(None)` when the model is not
/// available (not built, or no file); an error when it is but fails.
pub fn meshes(px: &[[f32; 4]], w: usize, h: usize, faces: &[Face]) -> Result<Option<Meshes>, String> {
    #[cfg(not(feature = "face-mesh"))]
    {
        let _ = (px, w, h, faces);
        Ok(None)
    }
    #[cfg(feature = "face-mesh")]
    {
        let Some(model) = imp::model()? else { return Ok(None) };
        if faces.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let (rgb, sw, sh, k) = reduced_rgb(px, w, h);
        let img = photocraft_face::Rgb8 { width: sw, height: sh, data: &rgb };
        let mut out = Vec::with_capacity(faces.len());
        for f in faces {
            // Detector coordinates are pixel positions in the full image; the reduced image's
            // pixel centres sit at ((x + 0.5) / k − 0.5).
            let down = |p: [f32; 2]| [(p[0] + 0.5) / k - 0.5, (p[1] + 0.5) / k - 0.5];
            let rect = [(f.rect[0] + 0.5) / k - 0.5, (f.rect[1] + 0.5) / k - 0.5, f.rect[2] / k, f.rect[3] / k];
            let roi = photocraft_face::Roi::around(rect, [down(f.landmarks[0]), down(f.landmarks[1])]);
            let mesh = roi.and_then(|roi| model.mesh(img, roi).ok()).filter(|m| m.presence >= 0.3);
            out.push(mesh.map(|m| m.points.into_iter().map(|p| [(p[0] + 0.5) * k - 0.5, (p[1] + 0.5) * k - 0.5]).collect()));
        }
        Ok(Some(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_model_is_looked_for_in_the_shared_folder_and_photocrafts_own() {
        let c = candidates();
        assert!(c.iter().any(|p| p.ends_with(PathBuf::from("ArtCraft").join("Craft AI").join("models").join(MODEL_FOLDER).join(MODEL_FILE))), "{c:?}");
        assert!(c.iter().any(|p| p.ends_with(PathBuf::from("photocraft").join("face-models").join(MODEL_FILE))), "{c:?}");
        let s = status();
        assert!(matches!(s, MeshStatus::Available(_) | MeshStatus::Missing(_) | MeshStatus::NotBuilt));
        let j = s.to_json();
        assert!(j["status"].is_string());
    }

    #[test]
    fn the_reduced_image_keeps_the_scale_and_treats_transparency_as_white() {
        let px = vec![[0.0, 0.0, 0.0, 0.0]; 4000 * 10];
        let (rgb, sw, sh, k) = reduced_rgb(&px, 4000, 10);
        assert_eq!((sw, sh, k), (2000, 5, 2.0));
        assert!(rgb.iter().all(|v| *v == 255));
        let px = vec![[1.0, 0.5, 0.0, 1.0]; 4];
        let (rgb, sw, sh, k) = reduced_rgb(&px, 2, 2);
        assert_eq!((sw, sh, k), (2, 2, 1.0));
        assert_eq!(&rgb[..3], &[255, 128, 0]);
    }
}
