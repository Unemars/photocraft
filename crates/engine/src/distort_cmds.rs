//! Filter › Liquify, Edit › Puppet Warp and Edit › Perspective Warp (plus their Layer › Smart
//! Objects twins).
//!
//! All three take their whole edit as data, so each is one replayable command and one history
//! step: Liquify the brush strokes (replayed onto a fresh displacement field,
//! `photocraft_algo::liquify`), Puppet Warp the pins (`photocraft_algo::puppet`, ARAP on a mesh
//! of the opaque region), Perspective Warp the planes (`photocraft_algo::perspective`). On a
//! smart object each becomes a smart filter (as in Photoshop): the params are stored with the
//! smart object and re-applied to a fresh render of the source, so editing never accumulates
//! resampling. The params are the data in every case (strokes, pins and planes are small
//! compared with a field or a mesh, and replay is deterministic), so nothing needs a blob.

use std::cell::RefCell;

use photocraft_algo::liquify::{LiquifyField, LiquifyStroke, apply_liquify, auto_cell};
use photocraft_algo::perspective::{PerspectiveMap, Plane, Straighten, straighten};
use photocraft_algo::puppet::{PuppetDensity, PuppetMode, PuppetPin, PuppetWarp, puppet_warp, puppet_warp_with_mask};
use photocraft_algo::transform::Interp;
use photocraft_algo::warp::{warp_mesh_gray, warp_mesh_surface};
use photocraft_color::PixelFormat;
use photocraft_doc::{Document, Layer, LayerContent, LayerId, Locks, SmartFilter};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::{Value, json};

use photocraft_algo::face_liquify::{FaceGeometry, FaceSliders, MESH_POINTS, PER_EYE, SLIDERS, apply_face};

use crate::commands::CommandSpec;
use crate::face_cmds::FoundFace;
use crate::photo_cmds::Stopwatch;
use crate::{EngineError, Result, Session};

/// Most faces one Liquify takes (Face-Aware Liquify).
pub const MAX_FACES: usize = 16;

pub const LIQUIFY: &str = "filter.liquify";
pub const PUPPET: &str = "edit.puppetWarp";
pub const PUPPET_SMART: &str = "layer.smartObjects.puppetWarp";
pub const PERSPECTIVE: &str = "edit.perspectiveWarp";
pub const PERSPECTIVE_SMART: &str = "layer.smartObjects.perspectiveWarp";

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

// ---------- params ----------

/// Liquify params: the strokes and the field resolution (`meshSize`, px per field node).
pub fn liquify_params(cmd: &str, p: &Value, canvas: Rect) -> Result<(Vec<LiquifyStroke>, f64)> {
    let strokes: Vec<LiquifyStroke> = match p.get("strokes") {
        Some(v) => serde_json::from_value(v.clone()).map_err(|e| bad(cmd, format!("bad `strokes`: {e}")))?,
        None => Vec::new(),
    };
    for s in &strokes {
        s.validate().map_err(|e| bad(cmd, e))?;
    }
    let cell = match p.get("meshSize") {
        None | Some(Value::Null) => auto_cell(canvas),
        Some(v) => {
            let c = v.as_f64().ok_or_else(|| bad(cmd, "`meshSize` is px per field node (1..64)"))?;
            if !(1.0..=64.0).contains(&c) {
                return Err(bad(cmd, "`meshSize` must be 1..64"));
            }
            c
        }
    };
    Ok((strokes, cell))
}

/// One face of a Liquify (Face-Aware Liquify): its geometry, from the 478-point mesh when there
/// is one, else the detector's five landmarks; its sliders; and the face as JSON for the result
/// and the stored params (index, landmarks, mesh, the non-zero sliders), so a replay never
/// detects again.
#[derive(Clone, Debug, PartialEq)]
pub struct LiquifyFace {
    pub geometry: FaceGeometry,
    pub sliders: FaceSliders,
    pub json: Value,
}

/// A face as `filter.liquify` takes it: the sliders, and either the geometry or the index of the
/// face to detect.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceSpec {
    pub index: Option<usize>,
    pub landmarks: Option<[[f32; 2]; 5]>,
    pub mesh: Option<Vec<[f32; 2]>>,
    pub sliders: FaceSliders,
}

fn in_range(v: f64) -> bool {
    v.is_finite() && (-100.0..=100.0).contains(&v)
}

/// The sliders named in object `v`: a number each, or `[left, right]` for a per-eye one. Keys other
/// than the sliders and `face`, `landmarks`, `mesh` are refused, as are values outside −100..100.
pub fn sliders_from_json(v: &Value) -> std::result::Result<FaceSliders, String> {
    let obj = v.as_object().ok_or("a face must be an object")?;
    let mut s = FaceSliders::default();
    for (k, val) in obj {
        if matches!(k.as_str(), "face" | "landmarks" | "mesh") {
            continue;
        }
        if !SLIDERS.contains(&k.as_str()) {
            return Err(format!("unknown face slider `{k}` ({})", SLIDERS.join("|")));
        }
        match val {
            Value::Number(n) => {
                let x = n.as_f64().filter(|x| in_range(*x)).ok_or_else(|| format!("`{k}` must be −100..100"))?;
                s.set(k, x as f32, None);
            }
            Value::Array(a) if a.len() == 2 && PER_EYE.contains(&k.as_str()) => {
                for (i, e) in a.iter().enumerate() {
                    let x = e.as_f64().filter(|x| in_range(*x)).ok_or_else(|| format!("`{k}` must be −100..100 per eye"))?;
                    s.set(k, x as f32, Some(i));
                }
            }
            _ => return Err(format!("`{k}` must be a number in −100..100{}", if PER_EYE.contains(&k.as_str()) { ", or [left, right]" } else { "" })),
        }
    }
    Ok(s)
}

/// The non-zero sliders as `filter.liquify` reads them (a per-eye slider with equal values as one
/// number).
pub fn sliders_to_json(s: &FaceSliders) -> serde_json::Map<String, Value> {
    let mut m = serde_json::Map::new();
    for k in SLIDERS {
        let Some([l, r]) = s.get(k) else { continue };
        if l == 0.0 && r == 0.0 {
            continue;
        }
        m.insert(k.to_string(), if l == r || !PER_EYE.contains(&k) { json!(l) } else { json!([l, r]) });
    }
    m
}

fn face_points(cmd: &str, v: &Value, what: &str) -> Result<Vec<[f32; 2]>> {
    let a = v.as_array().ok_or_else(|| bad(cmd, format!("`{what}` must be an array of [x, y] points")))?;
    a.iter()
        .map(|p| {
            let x = p.get(0).and_then(Value::as_f64).filter(|v| v.is_finite() && v.abs() < 1e7);
            let y = p.get(1).and_then(Value::as_f64).filter(|v| v.is_finite() && v.abs() < 1e7);
            match (x, y) {
                (Some(x), Some(y)) => Ok([x as f32, y as f32]),
                _ => Err(bad(cmd, format!("`{what}` points must be pairs of finite numbers"))),
            }
        })
        .collect()
}

/// The `faces` param: at most [`MAX_FACES`] entries, each the sliders plus `face` (an index),
/// `landmarks` (five points) or `mesh` (478 points).
pub fn face_specs(cmd: &str, p: &Value) -> Result<Vec<FaceSpec>> {
    let Some(v) = p.get("faces") else { return Ok(Vec::new()) };
    let arr = v.as_array().ok_or_else(|| bad(cmd, "`faces` must be an array of faces"))?;
    if arr.len() > MAX_FACES {
        return Err(bad(cmd, format!("`faces` takes at most {MAX_FACES} faces")));
    }
    arr.iter()
        .map(|f| {
            let sliders = sliders_from_json(f).map_err(|e| bad(cmd, e))?;
            let index = match f.get("face") {
                None => None,
                Some(v) => Some(v.as_u64().filter(|n| *n < 1000).ok_or_else(|| bad(cmd, "`face` must be the index of a detected face"))? as usize),
            };
            let landmarks = match f.get("landmarks") {
                None => None,
                Some(v) => {
                    let pts = face_points(cmd, v, "landmarks")?;
                    let five: [[f32; 2]; 5] =
                        pts.try_into().map_err(|_| bad(cmd, "`landmarks` takes five points: the eyes (image-left first), the nose tip, the mouth corners"))?;
                    Some(five)
                }
            };
            let mesh = match f.get("mesh") {
                None => None,
                Some(v) => {
                    let pts = face_points(cmd, v, "mesh")?;
                    if pts.len() != MESH_POINTS {
                        return Err(bad(cmd, format!("`mesh` takes the {MESH_POINTS} face-mesh points")));
                    }
                    Some(pts)
                }
            };
            Ok(FaceSpec { index, landmarks, mesh, sliders })
        })
        .collect()
}

/// The geometry of a face given by its mesh or landmarks (`None` when it has neither).
fn given_geometry(cmd: &str, spec: &FaceSpec) -> Result<Option<(FaceGeometry, Value)>> {
    if let Some(m) = &spec.mesh {
        let g = FaceGeometry::from_mesh(m).ok_or_else(|| bad(cmd, "`mesh` is a degenerate face (coincident eyes)"))?;
        let mut j = json!({"mesh": m});
        if let Some(l) = &spec.landmarks {
            j["landmarks"] = json!(l);
        }
        return Ok(Some((g, j)));
    }
    if let Some(l) = &spec.landmarks {
        let g = FaceGeometry::from_landmarks(l).ok_or_else(|| bad(cmd, "`landmarks` is a degenerate face (coincident eyes)"))?;
        return Ok(Some((g, json!({"landmarks": l}))));
    }
    Ok(None)
}

fn with_sliders(mut j: Value, sliders: &FaceSliders) -> Value {
    if let Value::Object(m) = &mut j {
        m.extend(sliders_to_json(sliders));
    }
    j
}

/// The faces of a Liquify on layer `id` of `doc`: a face given by its geometry is taken as is; the
/// others are detected once on the layer (with their mesh when the landmark model is installed)
/// and picked by `face` (or their place in the list).
pub fn resolve_faces(doc: &Document, id: LayerId, specs: &[FaceSpec], cmd: &str) -> Result<Vec<LiquifyFace>> {
    let mut detected: Option<Vec<FoundFace>> = None;
    let mut out = Vec::with_capacity(specs.len());
    for (i, spec) in specs.iter().enumerate() {
        let (geometry, json) = match given_geometry(cmd, spec)? {
            Some(g) => g,
            None => {
                if detected.is_none() {
                    detected = Some(crate::face_cmds::faces_with_mesh(doc, Some(id), false, crate::face_cmds::MIN_SCORE)?);
                }
                let found = detected.as_deref().unwrap_or(&[]);
                let k = spec.index.unwrap_or(i);
                let f = found.get(k).ok_or_else(|| bad(cmd, format!("no face {k} (found {})", found.len())))?;
                let face = liquify_face_of(k, f).ok_or_else(|| bad(cmd, format!("face {k} is degenerate")))?;
                (face.geometry, face.json)
            }
        };
        out.push(LiquifyFace { geometry, sliders: spec.sliders, json: with_sliders(json, &spec.sliders) });
    }
    Ok(out)
}

/// A detected face as a Liquify face with neutral sliders: its geometry from the mesh when the
/// landmark model found one, else from the five landmarks.
fn liquify_face_of(k: usize, f: &FoundFace) -> Option<LiquifyFace> {
    let geometry = f.mesh.as_deref().and_then(FaceGeometry::from_mesh).or_else(|| FaceGeometry::from_landmarks(&f.face.landmarks))?;
    let mut json = json!({"face": k, "landmarks": f.face.landmarks});
    if let Some(m) = &f.mesh {
        json["mesh"] = json!(m);
    }
    Some(LiquifyFace { geometry, sliders: FaceSliders::default(), json })
}

/// Every face on layer `id` of `doc` (best first), with neutral sliders: what the Liquify dialog
/// opens with. A face the detector found but whose geometry is degenerate is left out.
pub fn detect_liquify_faces(doc: &Document, id: LayerId) -> Result<Vec<LiquifyFace>> {
    let found = crate::face_cmds::faces_with_mesh(doc, Some(id), false, crate::face_cmds::MIN_SCORE)?;
    Ok(found.iter().enumerate().filter_map(|(k, f)| liquify_face_of(k, f)).collect())
}

/// The faces a stored Liquify (a smart filter) replays: those given by their geometry. A face
/// that would need detecting is left out, so the replay never depends on the pixels.
pub fn given_faces(specs: &[FaceSpec], cmd: &str) -> Result<Vec<LiquifyFace>> {
    let mut out = Vec::new();
    for spec in specs {
        if let Some((geometry, json)) = given_geometry(cmd, spec)? {
            out.push(LiquifyFace { geometry, sliders: spec.sliders, json: with_sliders(json, &spec.sliders) });
        }
    }
    Ok(out)
}

/// The field of a Liquify: the faces' sliders first, then the strokes in order, on a fresh field.
pub fn liquify_field(canvas: Rect, cell: f64, faces: &[LiquifyFace], strokes: &[LiquifyStroke]) -> LiquifyField {
    let mut f = LiquifyField::new(canvas, cell);
    for face in faces {
        apply_face(&mut f, &face.geometry, &face.sliders);
    }
    for s in strokes {
        f.apply_stroke(s);
    }
    f
}

/// Puppet Warp params.
pub fn puppet_params(cmd: &str, p: &Value) -> Result<PuppetWarp> {
    let pins: Vec<PuppetPin> = match p.get("pins") {
        Some(v) => serde_json::from_value(v.clone())
            .map_err(|e| bad(cmd, format!("bad `pins` (each {{\"src\":[x,y],\"dst\":[x,y],\"rotate\":deg?,\"depth\":n?}}): {e}")))?,
        None => Vec::new(),
    };
    if pins.iter().any(|q| q.src.iter().chain(&q.dst).any(|v| !v.is_finite()) || q.rotate.is_some_and(|r| !r.is_finite())) {
        return Err(bad(cmd, "pin coordinates must be finite"));
    }
    let mode = match p.get("mode").and_then(Value::as_str) {
        Some(m) => PuppetMode::parse(m).ok_or_else(|| bad(cmd, format!("unknown mode \"{m}\" (rigid|normal|distort)")))?,
        None => PuppetMode::Normal,
    };
    let density = match p.get("density").and_then(Value::as_str) {
        Some(d) => PuppetDensity::parse(d).ok_or_else(|| bad(cmd, format!("unknown density \"{d}\" (fewer|normal|more)")))?,
        None => PuppetDensity::Normal,
    };
    let expansion = p.get("expansion").and_then(Value::as_f64).unwrap_or(2.0);
    if !expansion.is_finite() || expansion.abs() > 200.0 {
        return Err(bad(cmd, "`expansion` must be -200..200 px"));
    }
    Ok(PuppetWarp { pins, mode, density, expansion })
}

/// Perspective Warp params (planes, after the optional straighten).
pub fn perspective_params(cmd: &str, p: &Value) -> Result<Vec<Plane>> {
    let mut planes: Vec<Plane> = match p.get("planes") {
        Some(v) => serde_json::from_value(v.clone()).map_err(|e| bad(cmd, format!("bad `planes` (each {{\"src\":[4×[x,y]],\"dst\":[4×[x,y]]}}): {e}")))?,
        None => return Err(bad(cmd, "pass `planes`: [{\"src\":[[x,y]×4],\"dst\":[[x,y]×4]}]")),
    };
    if planes.is_empty() {
        return Err(bad(cmd, "`planes` is empty"));
    }
    if planes.iter().any(|q| q.src.iter().chain(&q.dst).flatten().any(|v| !v.is_finite())) {
        return Err(bad(cmd, "plane corners must be finite"));
    }
    if let Some(s) = p.get("straighten").and_then(Value::as_str) {
        let mode = Straighten::parse(s).ok_or_else(|| bad(cmd, format!("unknown straighten \"{s}\" (horizontal|vertical|auto)")))?;
        straighten(&mut planes, mode);
    }
    if PerspectiveMap::new(&planes).is_none() {
        return Err(bad(cmd, "a plane is degenerate (its corners must form a quad)"));
    }
    Ok(planes)
}

fn interp(p: &Value) -> Interp {
    Interp::parse(p.get("interpolation").and_then(Value::as_str).unwrap_or("bicubic"))
}

// ---------- pixel operations ----------

/// Mixes `new` over `old` by the selection's coverage (premultiplied when there is alpha).
fn mix_by_selection(old: &Surface, new: &Surface, sel: &Surface) -> Surface {
    let area = old.content_bounds().union(&new.content_bounds());
    let mut out = new.clone();
    if area.is_empty() {
        return out;
    }
    let fmt = old.format();
    let n = fmt.channels();
    let (o, w) = (old.read_region(area), new.read_region(area));
    let mut r = vec![0.0f32; o.len()];
    let aw = area.width() as usize;
    for (i, ((op, np), rp)) in o.chunks_exact(n).zip(w.chunks_exact(n)).zip(r.chunks_exact_mut(n)).enumerate() {
        let k = sel.sample_channel(area.x0 + (i % aw) as i32, area.y0 + (i / aw) as i32, 0).clamp(0.0, 1.0);
        if !fmt.alpha {
            for c in 0..n {
                rp[c] = op[c] + (np[c] - op[c]) * k;
            }
            continue;
        }
        let a = n - 1;
        let oa = op[a] + (np[a] - op[a]) * k;
        rp[a] = oa;
        for c in 0..a {
            let pm = op[c] * op[a] + (np[c] * np[a] - op[c] * op[a]) * k;
            rp[c] = if oa > 0.0 { (pm / oa).clamp(0.0, 1.0) } else { 0.0 };
        }
    }
    out.write_region(area, &r);
    out.prune();
    out
}

/// Liquifies a surface over `canvas` (the field's extent): the faces, then the strokes.
pub fn liquify_surface(surf: &Surface, faces: &[LiquifyFace], strokes: &[LiquifyStroke], cell: f64, canvas: Rect) -> Surface {
    let field = liquify_field(canvas, cell, faces, strokes);
    apply_liquify(surf, &field)
}

/// Puppet-warps a surface's content; `None` when the result would span more than
/// [`photocraft_algo::puppet::MAX_WARP_PIXELS`].
pub fn puppet_surface(surf: &Surface, w: &PuppetWarp, interp: Interp) -> Option<Surface> {
    let b = surf.content_bounds();
    if w.is_identity() || b.is_empty() {
        return Some(surf.clone());
    }
    puppet_warp(surf, b, w, interp)
}

/// Perspective-warps a surface's content.
pub fn perspective_surface(surf: &Surface, planes: &[Plane], interp: Interp) -> Surface {
    let b = surf.content_bounds();
    if PerspectiveMap::is_identity(planes) || b.is_empty() {
        return surf.clone();
    }
    let Some(m) = PerspectiveMap::new(planes) else { return surf.clone() };
    warp_mesh_surface(surf, b, &|x, y| m.map(x, y), interp)
}

/// Smart-filter hook: re-applies a stored Liquify / Puppet / Perspective Warp to a rendered
/// smart object (`None` for other ids).
pub fn apply_to_surface(id: &str, params: &Value, surf: &Surface, canvas: Rect) -> Option<Surface> {
    match id {
        LIQUIFY => {
            let (strokes, cell) = liquify_params(id, params, canvas).ok()?;
            let faces = given_faces(&face_specs(id, params).ok()?, id).ok()?;
            Some(liquify_surface(surf, &faces, &strokes, cell, canvas))
        }
        // A stored warp too far to render leaves the smart object as it is (#1019).
        PUPPET | PUPPET_SMART => Some(puppet_surface(surf, &puppet_params(id, params).ok()?, interp(params)).unwrap_or_else(|| surf.clone())),
        PERSPECTIVE | PERSPECTIVE_SMART => Some(perspective_surface(surf, &perspective_params(id, params).ok()?, interp(params))),
        _ => None,
    }
}

// ---------- commands ----------

fn target(s: &Session, p: &Value) -> Result<LayerId> {
    match p.get("layer").and_then(Value::as_u64) {
        Some(v) => Ok(LayerId(v)),
        None => s.active().and_then(|d| d.active_layer).ok_or(EngineError::Other("no active layer".into())),
    }
}

fn pixel_layer(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    let id = d.active_layer.ok_or("no active layer")?;
    match &d.doc.layer(id).ok_or("no active layer")?.content {
        LayerContent::Raster(_) => Ok(()),
        LayerContent::Smart(sm) if sm.cache.is_some() => Ok(()),
        other => Err(format!("needs a pixel layer or smart object (active layer is {} {} layer)", other.article(), other.kind_name())),
    }
}

fn smart_layer(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    match d.active_layer.and_then(|id| d.doc.layer(id)).map(|l| &l.content) {
        Some(LayerContent::Smart(_)) => Ok(()),
        _ => Err("the active layer is not a smart object".into()),
    }
}

fn check_locks(l: &Layer, group: Locks, position: bool) -> Result<()> {
    let locks = l.locks.union(group);
    if locks.all || locks.pixels || (position && locks.position) {
        return Err(EngineError::Other(format!("Could not complete your request because the layer \"{}\" is locked", l.name)));
    }
    Ok(())
}

/// Shared driver: a smart object records `cmd` as a smart filter; a pixel layer is edited in
/// place by `f(surface, selection)`.
/// Deforms a linked layer mask the same way, limited to the selection when there is one.
type MaskFn<'a> = Option<&'a dyn Fn(&Surface, Option<&Surface>) -> Result<Surface>>;
/// Deforms a layer's pixels (with the selection and canvas); an `Err` leaves the layer as it was.
type LayerFn<'a> = &'a dyn Fn(&Surface, Option<&Surface>, Rect) -> Result<Surface>;

fn run_on_layer(s: &mut Session, cmd: &str, label: &str, p: &Value, moves: bool, f: LayerFn, mask: MaskFn) -> Result<Value> {
    let id = target(s, p)?;
    let mut params = p.clone();
    if let Value::Object(m) = &mut params {
        m.remove("layer");
    }
    s.edit(label, |doc: &mut Document, _| {
        let canvas = doc.bounds();
        let selection = doc.selection.clone();
        let group = crate::transform_cmds::group_locks(doc, id);
        // A Background gains transparency when the distortion moves pixels (as in Photoshop).
        if moves {
            crate::extra_cmds::background_to_layer(doc, id);
        }
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        if let LayerContent::Smart(_) = l.content {
            // As filters do: a pixel or full lock refuses the smart filter (the position lock
            // doesn't, since the smart object itself stays put).
            check_locks(l, group, false)?;
            let sf = SmartFilter { command: cmd.to_string(), params: params.clone(), blend: photocraft_color::BlendMode::Normal, opacity: 1.0, visible: true };
            return crate::smart_cmds::add_smart_filter(doc, id, sf, selection.as_ref());
        }
        check_locks(l, group, moves)?;
        let LayerContent::Raster(surf) = &mut l.content else {
            return Err(EngineError::Other(format!("{label} needs a pixel layer (rasterize it first)")));
        };
        *surf = f(surf, selection.as_ref(), canvas)?;
        // A linked layer mask follows a geometric warp, only where the pixels moved (#3083).
        if let (Some(mf), Some(m)) = (mask, l.mask.as_mut())
            && m.linked
        {
            m.surface = mf(&m.surface, selection.as_ref())?;
        }
        Ok(())
    })?;
    Ok(json!({"layer": id.0, "changed": true}))
}

fn liquify(s: &mut Session, p: &Value) -> Result<Value> {
    let canvas = s.active().ok_or(EngineError::NoDocument)?.doc.bounds();
    let (strokes, cell) = liquify_params(LIQUIFY, p, canvas)?;
    let specs = face_specs(LIQUIFY, p)?;
    let id = target(s, p)?;
    // Face-Aware Liquify: the faces' geometry is resolved here (detected when not given), and
    // stored with the params, so the result, undo and a smart filter replay the same field.
    let faces = {
        let d = s.active().ok_or(EngineError::NoDocument)?;
        resolve_faces(&d.doc, id, &specs, LIQUIFY)?
    };
    let faces_json: Vec<Value> = faces.iter().map(|f| f.json.clone()).collect();
    if strokes.is_empty() && faces.iter().all(|f| f.sliders.is_neutral()) {
        return Ok(json!({"layer": id.0, "changed": false, "faces": faces_json}));
    }
    let t0 = Stopwatch::start();
    let field = liquify_field(canvas, cell, &faces, &strokes);
    let smart = matches!(s.active().and_then(|d| d.doc.layer(id)).map(|l| &l.content), Some(LayerContent::Smart(_)));
    if field.is_identity() && !smart {
        return Ok(json!({"layer": id.0, "changed": false, "faces": faces_json}));
    }
    let mut p = p.clone();
    if !faces.is_empty() {
        p["faces"] = json!(faces_json);
    }
    let r = run_on_layer(
        s,
        LIQUIFY,
        "Liquify",
        &p,
        false,
        &|surf, sel, _| {
            let out = apply_liquify(surf, &field);
            Ok(match sel {
                Some(sel) => mix_by_selection(surf, &out, sel),
                None => out,
            })
        },
        None,
    )?;
    let mut r = r;
    r["ms"] = json!(t0.ms());
    r["maxDisplacement"] = json!(field.max_displacement());
    if !faces_json.is_empty() {
        r["faces"] = json!(faces_json);
    }
    Ok(r)
}

/// Lifts the selected pixels (when there is a selection), deforms them with `f` and puts them
/// back over the rest; without a selection deforms the whole surface.
fn deform_selected(surf: &Surface, sel: Option<&Surface>, f: &dyn Fn(&Surface) -> Result<Surface>) -> Result<Surface> {
    match sel {
        Some(sel) => {
            let (lifted, mut rest) = crate::transform_cmds::split_selected(surf, sel)?;
            let moved = f(&lifted)?;
            crate::transform_cmds::composite_over(&mut rest, &moved);
            rest.prune();
            Ok(rest)
        }
        None => {
            let fmt = surf.format();
            let src = if fmt.alpha { surf.clone() } else { surf.convert(PixelFormat::new(fmt.mode, fmt.sample, true)) };
            f(&src)
        }
    }
}

fn puppet(s: &mut Session, p: &Value, cmd: &str) -> Result<Value> {
    let w = puppet_params(cmd, p)?;
    let id = target(s, p)?;
    let smart = matches!(s.active().and_then(|d| d.doc.layer(id)).map(|l| &l.content), Some(LayerContent::Smart(_)));
    if w.is_identity() && !smart {
        return Ok(json!({"layer": id.0, "changed": false}));
    }
    let it = interp(p);
    let too_far = || bad(cmd, "the pins stretch the content too far: the result would cover more than 2^30 pixels");
    // A linked layer mask moves with the pixels on the same mesh (#4910): the layer's warp
    // solves the mesh once and leaves the moved mask here for the mask step.
    let linked = s.active().and_then(|d| d.doc.layer(id)).and_then(|l| l.mask.as_ref()).filter(|m| m.linked).map(|m| m.surface.clone());
    let moved_mask: RefCell<Option<Surface>> = RefCell::new(None);
    let layer_fn = |surf: &Surface, sel: Option<&Surface>, _: Rect| {
        deform_selected(surf, sel, &|x| {
            let b = x.content_bounds();
            if b.is_empty() {
                return Ok(x.clone());
            }
            let (out, mask) = puppet_warp_with_mask(x, b, &w, it, linked.as_ref()).ok_or_else(too_far)?;
            *moved_mask.borrow_mut() = mask;
            Ok(out)
        })
    };
    let mask_fn = |m: &Surface, _: Option<&Surface>| Ok(moved_mask.borrow_mut().take().unwrap_or_else(|| m.clone()));
    run_on_layer(s, cmd, "Puppet Warp", p, true, &layer_fn, Some(&mask_fn))
}

fn perspective(s: &mut Session, p: &Value, cmd: &str) -> Result<Value> {
    let planes = perspective_params(cmd, p)?;
    let id = target(s, p)?;
    let smart = matches!(s.active().and_then(|d| d.doc.layer(id)).map(|l| &l.content), Some(LayerContent::Smart(_)));
    if PerspectiveMap::is_identity(&planes) && !smart {
        return Ok(json!({"layer": id.0, "changed": false, "planes": planes}));
    }
    let it = interp(p);
    let map = PerspectiveMap::new(&planes);
    let mask_fn = |m: &Surface, sel: Option<&Surface>| {
        let Some(map) = &map else { return Ok(m.clone()) };
        let f = |x, y| map.map(x, y);
        let whole = |m: &Surface| Ok(warp_mesh_gray(m, &f, Interp::Bilinear));
        match sel {
            Some(sel) => crate::transform_cmds::move_gray_selected(m, sel, &|lifted, src| Ok(warp_mesh_surface(lifted, src, &f, Interp::Bilinear)), &whole),
            None => whole(m),
        }
    };
    let mut r = run_on_layer(
        s,
        cmd,
        "Perspective Warp",
        p,
        true,
        &|surf, sel, _| deform_selected(surf, sel, &|x| Ok(perspective_surface(x, &planes, it))),
        Some(&mask_fn),
    )?;
    r["planes"] = json!(planes);
    Ok(r)
}

pub fn specs() -> Vec<CommandSpec> {
    const L: &str = r##"{"strokes":[{"tool":"forwardWarp|reconstruct|smooth|twirlCw|twirlCcw|pucker|bloat|pushLeft|freeze|thaw|lassoMask|reconstructAll","size":px=100,"density":0-100=50,"pressure":0-100=100,"rate":0-100=80,"points":[[x,y,pressure?]…],"amount":%? (reconstructAll; lassoMask: 1 freezes the polygon in points, 0 thaws it)}…],"faces":[{"face":n? (index of a face detected on the layer; default its place in the list),"landmarks":[[x,y]×5]? (eyes image-left first, nose tip, mouth corners),"mesh":[[x,y]×478]? (the face mesh, when the landmark model is installed),"eyeSize","eyeHeight","eyeWidth","eyeTilt" (-100..100 each; a number, or [left,right]),"eyeDistance","noseHeight","noseWidth","smile","upperLip","lowerLip","mouthWidth","mouthHeight","forehead","chinHeight","jawline","faceWidth" (-100..100 each)}…] (Face-Aware Liquify, at most 16 faces, applied before the strokes),"meshSize":px? (field resolution, px per node; default 2, or 4 above 4 MP),"layer":id?} → {"layer","changed","ms","maxDisplacement","faces":[the faces with the geometry used]?} — faces then strokes replay in order on a fresh field; a selection limits the effect; on a smart object it becomes a smart filter (with the faces' geometry stored, so it never detects again)"##;
    const P: &str = r##"{"pins":[{"src":[x,y],"dst":[x,y],"rotate":deg?,"depth":n?}…],"mode":"rigid|normal|distort"="normal","density":"fewer|normal|more"="normal","expansion":px=2,"interpolation":"bicubic|bilinear|nearest","layer":id?} — mesh over the opaque region, as-rigid-as-possible; on a smart object it becomes a smart filter"##;
    const Q: &str = r##"{"planes":[{"src":[[x,y]×4],"dst":[[x,y]×4]}…] (corners clockwise from top-left; corners that coincide in src are linked),"straighten":"horizontal|vertical|auto"?,"interpolation":"bicubic|bilinear|nearest","layer":id?}"##;
    vec![
        CommandSpec {
            id: LIQUIFY,
            label: "Liquify…",
            menu: &["Filter"],
            shortcut: Some("Cmd+Shift+X"),
            params: L,
            enabled: crate::filters::has_filterable_layer,
            journal: true,
            run: liquify,
        },
        CommandSpec {
            id: PUPPET,
            label: "Puppet Warp",
            menu: &["Edit"],
            shortcut: None,
            params: P,
            enabled: pixel_layer,
            journal: true,
            run: |s, p| puppet(s, p, PUPPET),
        },
        CommandSpec {
            id: PUPPET_SMART,
            label: "Puppet Warp",
            menu: &["Layer", "Smart Objects"],
            shortcut: None,
            params: P,
            enabled: smart_layer,
            journal: true,
            run: |s, p| puppet(s, p, PUPPET_SMART),
        },
        CommandSpec {
            id: PERSPECTIVE,
            label: "Perspective Warp",
            menu: &["Edit"],
            shortcut: None,
            params: Q,
            enabled: pixel_layer,
            journal: true,
            run: |s, p| perspective(s, p, PERSPECTIVE),
        },
        CommandSpec {
            id: PERSPECTIVE_SMART,
            label: "Perspective Warp",
            menu: &["Layer", "Smart Objects"],
            shortcut: None,
            params: Q,
            enabled: smart_layer,
            journal: true,
            run: |s, p| perspective(s, p, PERSPECTIVE_SMART),
        },
    ]
}

#[cfg(test)]
mod tests;
