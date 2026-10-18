//! Face detection (#3603): the faces in the document, with their box and five landmarks, from
//! [`photocraft_algo::faces`], and their 478-point mesh when the landmark model is installed
//! ([`crate::face_mesh`]). Select › Color Range's Detect Faces reads faces through [`faces_in`],
//! Face-Aware Liquify through [`faces_with_mesh`]; `image.detectFaces` reports them to scripts
//! and agents.

use photocraft_algo::faces::{self, Face};
use photocraft_doc::{Document, LayerId};
use photocraft_geom::Rect;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

/// Faces scoring below this are left out unless asked for (libfacedetection's own threshold for a
/// reliable face).
pub const MIN_SCORE: f32 = 0.5;

/// Why there are no faces in the browser build (see [`photocraft_algo::faces`]).
const UNAVAILABLE: &str = "face detection isn't available in the browser version of PhotoCraft; use the desktop app";

/// The faces in `px` (`w × h` straight RGBA), as [`faces::detect`] finds them.
pub fn detect(px: &[[f32; 4]], w: usize, h: usize, min_score: f32) -> Result<Vec<Face>> {
    if !faces::AVAILABLE {
        return Err(EngineError::Other(UNAVAILABLE.into()));
    }
    faces::detect(px, w, h, min_score).ok_or_else(|| EngineError::Other("face detection failed".into()))
}

/// The pixels face detection looks at: the composite, or the active layer's own pixels when
/// `all_layers` is off, over the canvas.
fn pixels(doc: &Document, active: Option<LayerId>, all_layers: bool) -> (Rect, Vec<[f32; 4]>) {
    let area = doc.bounds();
    let (w, h) = (area.width() as usize, area.height() as usize);
    let surface = active.and_then(|id| doc.layer(id)).and_then(|l| l.surface()).filter(|_| !all_layers);
    let px = match surface {
        Some(surf) => {
            let mut px = vec![[0.0f32; 4]; w * h];
            surf.read_rgba_into(area, &mut px);
            px
        }
        None => photocraft_compose::render(doc, area).px,
    };
    (area, px)
}

/// The faces in `doc`, best first: in the composite, or in the active layer's pixels when
/// `all_layers` is off. Coordinates are document pixels.
pub fn faces_in(doc: &Document, active: Option<LayerId>, all_layers: bool, min_score: f32) -> Result<Vec<Face>> {
    let (area, px) = pixels(doc, active, all_layers);
    let found = detect(&px, area.width() as usize, area.height() as usize, min_score)?;
    let (x0, y0) = (area.x0 as f32, area.y0 as f32);
    Ok(found
        .into_iter()
        .map(|f| Face { rect: [f.rect[0] + x0, f.rect[1] + y0, f.rect[2], f.rect[3]], landmarks: f.landmarks.map(|[x, y]| [x + x0, y + y0]), ..f })
        .collect())
}

/// A detected face with its 478-point mesh, when the landmark model found one.
#[derive(Clone, Debug, PartialEq)]
pub struct FoundFace {
    pub face: Face,
    /// The mesh in document pixels (`None`: no model, or none found in this face).
    pub mesh: Option<Vec<[f32; 2]>>,
}

/// [`faces_in`] plus each face's mesh from [`crate::face_mesh`] (none when the model is not
/// installed: the faces still come back). A model that is there but fails is an error.
pub fn faces_with_mesh(doc: &Document, active: Option<LayerId>, all_layers: bool, min_score: f32) -> Result<Vec<FoundFace>> {
    let (area, px) = pixels(doc, active, all_layers);
    let (w, h) = (area.width() as usize, area.height() as usize);
    let found = detect(&px, w, h, min_score)?;
    let meshes = crate::face_mesh::meshes(&px, w, h, &found).map_err(EngineError::Other)?;
    let (x0, y0) = (area.x0 as f32, area.y0 as f32);
    Ok(found
        .into_iter()
        .enumerate()
        .map(|(i, f)| FoundFace {
            face: Face { rect: [f.rect[0] + x0, f.rect[1] + y0, f.rect[2], f.rect[3]], landmarks: f.landmarks.map(|[x, y]| [x + x0, y + y0]), ..f },
            mesh: meshes.as_ref().and_then(|m| m.get(i).cloned().flatten()).map(|m| m.into_iter().map(|[x, y]| [x + x0, y + y0]).collect()),
        })
        .collect())
}

fn face_json(f: &Face) -> Value {
    let pt = |i: usize| json!([f.landmarks[i][0], f.landmarks[i][1]]);
    json!({
        "score": f.score,
        "rect": f.rect,
        "landmarks": {"rightEye": pt(0), "leftEye": pt(1), "nose": pt(2), "mouthRight": pt(3), "mouthLeft": pt(4)},
    })
}

fn detect_faces(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "image.detectFaces";
    let bad = |msg: &str| EngineError::BadParams { cmd: CMD.into(), msg: msg.into() };
    let min_score = match p.get("minScore") {
        None => MIN_SCORE,
        Some(v) => v.as_f64().filter(|v| (0.0..=1.0).contains(v)).ok_or_else(|| bad("minScore must be a number in 0..1"))? as f32,
    };
    let all_layers = match p.get("sampleAllLayers") {
        None => true,
        Some(v) => v.as_bool().ok_or_else(|| bad("sampleAllLayers must be true or false"))?,
    };
    let mesh = match p.get("mesh") {
        None => false,
        Some(v) => v.as_bool().ok_or_else(|| bad("mesh must be true or false"))?,
    };
    let d = s.active().ok_or(EngineError::NoDocument)?;
    if !mesh {
        let found = faces_in(&d.doc, d.active_layer, all_layers, min_score)?;
        return Ok(json!({ "faces": found.iter().map(face_json).collect::<Vec<_>>() }));
    }
    let found = faces_with_mesh(&d.doc, d.active_layer, all_layers, min_score)?;
    let faces: Vec<Value> = found
        .iter()
        .map(|f| {
            let mut j = face_json(&f.face);
            if let Some(m) = &f.mesh {
                j["mesh"] = json!(m);
            }
            j
        })
        .collect();
    Ok(json!({ "faces": faces, "meshModel": crate::face_mesh::status().to_json() }))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: "image.detectFaces",
        label: "Detect Faces",
        menu: &[],
        shortcut: None,
        params: r##"{"minScore":0..1=0.5,"sampleAllLayers":bool=true,"mesh":bool=false} → {"faces":[{"score","rect":[x,y,w,h],"landmarks":{"rightEye","leftEye","nose","mouthRight","mouthLeft"}: [x,y] each,"mesh":[[x,y]×478]? (with mesh, when the landmark model is installed)},…],"meshModel":{"status":"available|missing|notBuilt",…}?} (best first; "rightEye" is the one on the image's left)"##,
        enabled: |s| {
            if !faces::AVAILABLE {
                return Err(UNAVAILABLE.into());
            }
            s.active().map(|_| ()).ok_or_else(|| "no document open".into())
        },
        run: detect_faces,
        journal: false,
    }]
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::Session;

    #[test]
    fn detect_faces_reports_no_faces_on_a_plain_image_and_validates_params() {
        let mut s = Session::new();
        assert!(s.execute("image.detectFaces", json!({})).is_err(), "no document");
        assert!(!s.is_enabled("image.detectFaces"));
        s.execute("file.new", json!({"width": 64, "height": 48, "background": "white"})).unwrap();
        assert!(s.is_enabled("image.detectFaces"));
        s.execute("select.all", json!({})).unwrap();
        assert_eq!(s.execute("image.detectFaces", json!({})).unwrap(), json!({"faces": []}));
        assert_eq!(s.execute("image.detectFaces", json!({"minScore": 0, "sampleAllLayers": false})).unwrap(), json!({"faces": []}));
        // No history step of its own: Undo undoes the Select All before it.
        s.execute("edit.undo", json!({})).unwrap();
        assert!(s.active().unwrap().doc.selection.is_none());
        for bad in [json!({"minScore": 2}), json!({"minScore": "high"}), json!({"sampleAllLayers": 1})] {
            assert!(s.execute("image.detectFaces", bad.clone()).is_err(), "{bad}");
        }
    }
}
