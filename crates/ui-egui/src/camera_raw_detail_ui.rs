//! Full-resolution refinement of Camera Raw's interactive proxy. Native work is off the
//! UI thread; only one revision runs at a time, and obsolete results never replace new edits.
//! Below 1:1 the frame is developed at a level of its size (see [`level_scale`]) over the part on
//! screen, and the last sharp pixels stay on screen while a zoom or pan waits for its develop.
use egui::{Color32, ColorImage, Context, Pos2, Rect, TextureHandle, TextureOptions, pos2, vec2};
use photocraft_algo::camera_raw::CameraRaw;
use photocraft_geom::Rect as PixelRect;
use photocraft_raster::Surface;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
    mpsc::{Receiver, TryRecvError},
};

#[derive(Clone)]
pub(crate) enum Coverage {
    Selection(Surface),
    Mask(photocraft_doc::LayerMask),
    None,
}

type DetailResult = Result<(u64, Surface), String>;

/// What a level develop is for: the settings' revision (0 for the original), the level's size
/// and whether it is the original (Before, or every setting at its default).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ScreenKey {
    revision: u64,
    size: (usize, usize),
    original: bool,
}

/// Pixels (straight RGBA, row-major) shared between the dialog and its workers.
type Pixels = Arc<Vec<[f32; 4]>>;

/// A develop at a level over a part of it (level pixels), or `None` when it gave up.
type ScreenResult = Result<Option<(ScreenKey, PixelRect, Pixels)>, String>;

/// Largest level (pixels) the preview develops below 1:1, and the most the levels it keeps add up
/// to: about a 4K screen's preview area twice over. Past it (a large frame just under 100 %),
/// the preview develops the full frame and shows the visible part, as at 1:1, within that path's
/// limits.
const SCREEN_MAX_PIXELS: usize = 12_000_000;

/// How many levels [`ScreenCache`] keeps at most: Fit, the level a zoom passes on the way there,
/// and one more for a resized window.
const LEVELS_KEPT: usize = 3;

/// How long a view must stay still before a develop of what it shows starts (a zoom or pan
/// gesture asks for something new every frame); a changed setting or level starts at once.
const SETTLE_MS: f64 = 120.0;

/// The raw stage of a raw open: its five sliders, and Exposure / Temperature / Tint as a linear
/// matrix applied before them (see [`raw_stage_split`]).
type RawStage = (photocraft_io::raw::Tone, Option<[[f32; 3]; 3]>);

/// Splits `p` for a raw being opened (`raw`): the raw stage, which Open applies to the sensor data
/// (Shadows, Highlights, Clarity, Texture, Dehaze, Exposure, and Temperature / Tint when the file
/// has an as-shot white balance), and the settings left for the pipeline after it. Without `raw`
/// everything is left for the pipeline.
pub(crate) fn raw_stage_split(p: &CameraRaw, raw: Option<&photocraft_io::raw::RawPreview>) -> (CameraRaw, Option<RawStage>) {
    let Some(raw) = raw else { return (p.clone(), None) };
    let tone = photocraft_io::raw::Tone { shadows: p.shadows, highlights: p.highlights, clarity: p.clarity, texture: p.texture, dehaze: p.dehaze };
    let (pre, white_balance) = raw.stage_matrix(p.exposure, p.temperature, p.tint);
    let mut rest = p.clone();
    (rest.shadows, rest.highlights, rest.clarity, rest.texture, rest.dehaze, rest.exposure) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    if white_balance {
        (rest.temperature, rest.tint) = (0.0, 0.0);
    }
    (rest, Some((tone, pre)))
}

/// Results of [`tonal_stage`] kept for the settings they were made with: the raw stage's (for
/// its five sliders and its matrix) and the tonal stages' after it. A clone shares the
/// slider-independent caches.
#[derive(Clone, Default)]
pub(crate) struct TonalCache {
    raw: Option<(RawStage, Pixels)>,
    tonal: Option<(CameraRaw, Pixels)>,
    /// The raw stage's slider-independent part, so dragging one of its sliders is cheap.
    retone: Arc<std::sync::Mutex<Option<photocraft_io::raw::RetoneCache>>>,
    /// The same for the tonal stages (Shadows and Highlights of the filter).
    develop: Arc<std::sync::Mutex<photocraft_algo::camera_raw::DevelopCache>>,
}

/// A level below 1:1 the worker develops at: the frame reduced to the level's size, and its
/// tonal stages kept for the settings they were made with.
struct LevelCache {
    size: (usize, usize),
    source: Pixels,
    tonal: TonalCache,
}

/// The worker's caches for develops below 1:1, most recently used first: at most
/// [`LEVELS_KEPT`] levels, adding up to at most [`SCREEN_MAX_PIXELS`] (the newest is always kept).
#[derive(Default)]
struct ScreenCache {
    levels: Vec<LevelCache>,
}

/// What the last paint showed, for the clipping overlay: a crop of the full-size pixels (the key
/// [`DetailPreview::crop`] holds), or a part of a level (level pixels).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shown {
    Crop,
    Screen(ScreenKey, PixelRect),
}

/// What the level texture holds: a develop's key and part, the view scale it was resampled for
/// (thousandths of the level's) and the display's key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScreenTex {
    key: ScreenKey,
    part: PixelRect,
    ratio: u32,
    display: u64,
}

type OverlayKey = (Option<ScreenKey>, u64, bool, PixelRect, crate::camera_raw_scope_ui::ClippingMode, crate::theme::ThemeKind);

pub(crate) struct DetailPreview {
    source: Surface,
    /// How the open-time dialog's raw was developed: its raw-stage settings then act on the raw
    /// data, in linear light before the camera's tone curve (see [`develop`]).
    raw: Option<Arc<photocraft_io::raw::RawPreview>>,
    area: PixelRect,
    coverage: Coverage,
    result: Option<(u64, Surface)>,
    pending: Option<(u64, Receiver<DetailResult>)>,
    cancel: Arc<AtomicBool>,
    pub(crate) error: Option<String>,
    failed_revision: Option<u64>,
    crop: Option<(u64, bool, PixelRect, u64)>,
    texture: Option<TextureHandle>,
    overlay_texture: Option<TextureHandle>,
    overlay_key: Option<OverlayKey>,
    pub(crate) ready: bool,
    shown: Option<Shown>,
    /// Below 1:1 the preview develops the frame at a level (see [`level_scale`]) over the part on
    /// screen instead: the last develop, in level pixels.
    screen_result: Option<(ScreenKey, PixelRect, Pixels)>,
    screen_pending: Option<(ScreenKey, PixelRect, Receiver<ScreenResult>)>,
    /// The job the level worker is for; bumped to make it give up.
    screen_job: Arc<AtomicU64>,
    /// A level develop that failed: not asked for again.
    screen_failed: Option<ScreenKey>,
    /// The level texture: the developed part at the view's scale.
    screen_texture: Option<(ScreenTex, TextureHandle)>,
    screen_cache: Arc<std::sync::Mutex<ScreenCache>>,
    /// What the view wants that no develop covers yet, and since when (ms): see [`SETTLE_MS`].
    wanted: Option<(u64, f64)>,
}

impl Drop for DetailPreview {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl DetailPreview {
    pub(crate) fn new(source: Surface, area: PixelRect, coverage: Coverage) -> Self {
        Self {
            source,
            raw: None,
            area,
            coverage,
            result: None,
            pending: None,
            cancel: Arc::new(AtomicBool::new(false)),
            error: None,
            failed_revision: None,
            crop: None,
            texture: None,
            overlay_texture: None,
            overlay_key: None,
            ready: false,
            shown: None,
            screen_result: None,
            screen_pending: None,
            screen_job: Arc::default(),
            screen_failed: None,
            screen_texture: None,
            screen_cache: Arc::default(),
            wanted: None,
        }
    }

    /// The raw the open-time dialog develops: everything made for another one is dropped.
    pub(crate) fn set_raw(&mut self, raw: Option<Arc<photocraft_io::raw::RawPreview>>) {
        self.raw = raw;
        (self.result, self.crop, self.failed_revision, self.shown) = (None, None, None, None);
        (self.screen_result, self.screen_failed, self.screen_texture, self.wanted) = (None, None, None, None);
        // A worker still running gives up, and keeps the old cache to itself.
        self.screen_job.fetch_add(1, Ordering::Relaxed);
        self.screen_cache = Arc::default();
        self.screen_pending = None;
        self.pending = None;
    }

    /// Bench only: what the level worker develops for `params` at level `size`, now (through its
    /// caches): the whole level.
    #[cfg(test)]
    pub(crate) fn develop_screen_now(&self, params: &CameraRaw, size: (usize, usize), revision: u64) -> Pixels {
        let mut params = params.clone();
        params.pixel_scale = (size.0 as f32 / self.area.width().max(1) as f32).min(1.0);
        let key = ScreenKey { revision, size, original: false };
        let whole = PixelRect::new(0, 0, size.0 as i32, size.1 as i32);
        develop_screen(&self.source, self.area, self.raw.as_deref(), &self.screen_cache, key, whole, &params, false, &|| false).unwrap_or_default()
    }

    /// Bench only: what the background worker develops for `params`, now.
    #[cfg(test)]
    pub(crate) fn develop_now(&self, params: &CameraRaw) -> Surface {
        let mut params = params.clone();
        params.pixel_scale = 1.0;
        develop(&self.source, self.raw.as_deref(), self.area, &params, &self.coverage)
    }

    /// The size of the level the preview shows (`None`: the proxy or 1:1 pixels).
    pub(crate) fn screen_size(&self) -> Option<(usize, usize)> {
        match self.shown {
            Some(Shown::Screen(key, _)) if self.ready => Some(key.size),
            _ => None,
        }
    }

    pub(crate) fn pending(&self) -> bool {
        self.pending.is_some() || self.screen_pending.is_some()
    }

    fn poll_screen(&mut self) {
        let Some((key, _, rx)) = &self.screen_pending else { return };
        let key = *key;
        let result = match rx.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => Err("Camera Raw preview worker stopped".into()),
        };
        self.screen_pending = None;
        match result {
            Ok(Some(done)) => {
                self.screen_result = Some(done);
                self.error = None;
            }
            Ok(None) => {} // Gave up: the view wants another level or revision now.
            Err(e) => {
                self.error = Some(e);
                self.screen_failed = Some(key);
            }
        }
    }

    /// Whether a develop for `want` may start now: at once unless `settle`, else once the view
    /// has wanted it for [`SETTLE_MS`] (a zoom or pan in progress wants something new every
    /// frame, and each develop would take longer than the frame).
    fn settled(&mut self, ctx: &Context, want: u64, settle: bool) -> bool {
        let now = crate::gpu_canvas::now_ms();
        let since = match self.wanted {
            Some((w, since)) if w == want => since,
            _ => {
                self.wanted = Some((want, now));
                now
            }
        };
        if !settle || now - since >= SETTLE_MS {
            return true;
        }
        ctx.request_repaint_after(std::time::Duration::from_millis((SETTLE_MS - (now - since)).max(1.0) as u64 + 1));
        false
    }

    /// Develops `params` at `key`'s level over `part` (level pixels) in the background: one at a
    /// time; a request for another level or revision makes the running one give up at its next
    /// stage, and the next frame asks again. A develop that failed is not asked for again.
    fn request_screen(&mut self, ctx: &Context, key: ScreenKey, part: PixelRect, params: &CameraRaw) {
        if self.screen_failed == Some(key) {
            return;
        }
        if let Some((k, _, _)) = &self.screen_pending {
            if *k != key {
                self.screen_job.fetch_add(1, Ordering::Relaxed);
            }
            return;
        }
        let (source, area, raw, cache, cancel, jobs, ctx) =
            (self.source.clone(), self.area, self.raw.clone(), self.screen_cache.clone(), self.cancel.clone(), self.screen_job.clone(), ctx.clone());
        let mut params = params.clone();
        params.pixel_scale = (key.size.0 as f32 / area.width().max(1) as f32).min(1.0);
        let float = source.format().sample == photocraft_color::SampleType::F32;
        let job = self.screen_job.fetch_add(1, Ordering::Relaxed) + 1;
        let (tx, rx) = std::sync::mpsc::channel::<ScreenResult>();
        let work = move || {
            let given_up = || cancel.load(Ordering::Relaxed) || jobs.load(Ordering::Relaxed) != job;
            if given_up() {
                let _ = tx.send(Ok(None));
                ctx.request_repaint();
                return;
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                develop_screen(&source, area, raw.as_deref(), &cache, key, part, &params, float, &given_up)
            }))
            .map(|px| px.map(|px| (key, part, px)))
            .map_err(|_| "Camera Raw preview failed; the proxy remains available".to_string());
            if !cancel.load(Ordering::Relaxed) {
                let _ = tx.send(result);
                ctx.request_repaint();
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            match std::thread::Builder::new().name("camera-raw-screen".into()).spawn(work) {
                Ok(_) => self.screen_pending = Some((key, part, rx)),
                Err(e) => {
                    self.error = Some(format!("Cannot start Camera Raw preview: {e}"));
                    self.screen_failed = Some(key);
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            // No worker here: the proxy stays.
            drop(work);
            drop(rx);
            self.screen_failed = Some(key);
        }
    }

    /// The level (scale of the frame) a view at `scale` (screen pixels per image pixel) is
    /// developed at below 1:1, Fit being `fit`: [`level_scale`]'s, or the view's own when that one
    /// is past [`SCREEN_MAX_PIXELS`] and the view's is not. `None` at 1:1 and above, with a
    /// selection or mask, or when even the view's own is past it: the full-size path's crop then.
    fn level_for(&self, scale: f32, fit: f32) -> Option<f32> {
        if scale.is_nan() || scale >= 0.99 || !matches!(self.coverage, Coverage::None) {
            return None;
        }
        let fits = |level: f32| {
            let (w, h) = level_size(self.area, level);
            level < 0.99 && w.saturating_mul(h) <= SCREEN_MAX_PIXELS
        };
        let level = level_scale(scale, fit);
        if fits(level) { Some(level) } else { fits(scale).then_some(scale) }
    }

    /// Below 1:1: the frame developed at `level` over the part on screen, resampled to the view's
    /// `scale` (screen pixels per image pixel); at Fit both are Fit's own.
    #[allow(clippy::too_many_arguments)]
    fn paint_screen(
        &mut self,
        ui: &egui::Ui,
        image: Rect,
        revision: u64,
        before: bool,
        params: &CameraRaw,
        scale: f32,
        level: f32,
        display: &photocraft_engine::display_color::CanvasDisplay,
    ) -> bool {
        let original = before || params.is_identity();
        let size = level_size(self.area, level);
        let key = ScreenKey { revision: if original { 0 } else { revision }, size, original };
        let level_rect = PixelRect::new(0, 0, size.0 as i32, size.1 as i32);
        let max_side = ui.input(|i| i.max_texture_side);
        let Some(crop) = visible_pixels(image, ui.clip_rect(), level_rect, max_side) else { return false };
        let covers = |k: &ScreenKey, p: &PixelRect| *k == key && p.contains_rect(&crop);
        if !self.screen_result.as_ref().is_some_and(|(k, p, _)| covers(k, p)) {
            let part = grown_part(crop, level_rect, max_side);
            // A new level or revision develops at once; a view that left the developed part waits
            // for its gesture to settle.
            let settle = self.screen_result.as_ref().is_some_and(|(k, _, _)| *k == key);
            if self.settled(ui.ctx(), hash_of(&("level", key, crop)), settle) {
                self.request_screen(ui.ctx(), key, part, params);
                self.poll_screen();
            }
        }
        let Some((_, part, px)) = self.screen_result.as_ref().filter(|(k, p, _)| covers(k, p)) else {
            return self.paint_stale(ui, image, revision, original);
        };
        // The texture: the developed part at the view's scale (the GPU scales it a little while a
        // gesture goes on, and not at all at Fit).
        let ratio = (scale / level).min(1.0);
        let tex = ScreenTex { key, part: *part, ratio: (ratio * 1000.0).round() as u32, display: display.key };
        if self.screen_texture.as_ref().is_none_or(|(k, _)| *k != tex) {
            let (pw, ph) = (part.width() as usize, part.height() as usize);
            let (tw, th) = ((pw as f32 * ratio).round().max(1.0) as usize, (ph as f32 * ratio).round().max(1.0) as usize);
            let reduced;
            let pixels: &[[f32; 4]] = if (tw, th) == (pw, ph) {
                px
            } else {
                reduced = reduce_with(pw, ph, tw, th, |y, row| {
                    if let Some(src) = px.get(y * pw..(y + 1) * pw) {
                        row.copy_from_slice(src);
                    }
                });
                &reduced
            };
            let color = display_image(PixelRect::new(0, 0, tw as i32, th as i32), display, |r, out| {
                let start = (r.y0.max(0) as usize).saturating_mul(tw);
                if let Some(src) = pixels.get(start..start.saturating_add(out.len())) {
                    out.copy_from_slice(src);
                }
            });
            match &mut self.screen_texture {
                Some((k, t)) => {
                    t.set(color, TextureOptions::LINEAR);
                    *k = tex;
                }
                None => self.screen_texture = Some((tex, ui.ctx().load_texture("cr-screen-preview", color, TextureOptions::LINEAR))),
            }
        }
        if let Some((k, texture)) = &self.screen_texture {
            ui.painter().image(texture.id(), level_part_rect(image, k.key.size, k.part), Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
            self.ready = true;
            self.shown = Some(Shown::Screen(k.key, k.part));
        }
        true
    }

    /// While no develop covers the view: the textures of this revision that still exist, where
    /// they belong (scaled by the GPU), a level's under the full-size one. So a zoom or pan shows
    /// the last sharp pixels at once, not the proxy, until the develop of the new view lands.
    fn paint_stale(&self, ui: &egui::Ui, image: Rect, revision: u64, original: bool) -> bool {
        let current = |r: u64, o: bool| o == original && (original || r == revision);
        let uv = Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0));
        let mut drawn = false;
        if let Some((k, t)) = self.screen_texture.as_ref().filter(|(k, _)| current(k.key.revision, k.key.original)) {
            ui.painter().image(t.id(), level_part_rect(image, k.key.size, k.part), uv, Color32::WHITE);
            drawn = true;
        }
        if let (Some((r, o, crop, _)), Some(t)) = (self.crop, &self.texture)
            && current(r, o)
        {
            ui.painter().image(t.id(), area_part_rect(image, self.area, crop), uv, Color32::WHITE);
            drawn = true;
        }
        drawn
    }

    fn poll(&mut self, revision: u64) {
        let Some((requested_revision, rx)) = &self.pending else { return };
        let requested_revision = *requested_revision;
        let result = match rx.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => Err("Camera Raw detail preview worker stopped".into()),
        };
        self.pending = None;
        match result {
            Ok((r, surface)) if r == revision => {
                self.result = Some((r, surface));
                self.crop = None;
                self.error = None;
            }
            Ok(_) => {} // A parameter changed while the worker was developing this revision.
            Err(e) => {
                if requested_revision == revision {
                    self.error = Some(e);
                }
                self.failed_revision = Some(requested_revision);
            }
        }
    }

    fn request(&mut self, ctx: &Context, revision: u64, params: &CameraRaw) {
        if self.pending.is_some() || self.result.as_ref().is_some_and(|(r, _)| *r == revision) || self.failed_revision == Some(revision) {
            return;
        }
        self.result = None;
        self.crop = None;
        self.error = None;
        // The engine's full-image pipeline needs multiple temporary float buffers. Never start
        // an unbounded allocation just because a huge sparse document was zoomed into.
        if self.area.width() as u64 * self.area.height() as u64 > 64_000_000 {
            self.error = Some("Full-resolution Camera Raw preview is limited to 64 megapixels; the proxy remains available".into());
            self.failed_revision = Some(revision);
            return;
        }
        let source = self.source.clone();
        let raw = self.raw.clone();
        let area = self.area;
        let coverage = self.coverage.clone();
        let mut params = params.clone();
        params.pixel_scale = 1.0;
        let cancel = self.cancel.clone();
        let ctx = ctx.clone();
        let (tx, rx) = std::sync::mpsc::channel::<DetailResult>();
        let work = move || {
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| develop(&source, raw.as_deref(), area, &params, &coverage)))
                .map(|surface| (revision, surface))
                .map_err(|_| "Camera Raw detail preview failed; the proxy remains available".into());
            if !cancel.load(Ordering::Relaxed) {
                let _ = tx.send(result);
                ctx.request_repaint();
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            // A closed dialog can leave one non-interruptible engine stage finishing. Serialize
            // that work with the next dialog instead of multiplying its peak memory.
            static WORKER: std::sync::Mutex<()> = std::sync::Mutex::new(());
            match std::thread::Builder::new().name("camera-raw-preview".into()).spawn(move || {
                let _guard = WORKER.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                work();
            }) {
                Ok(_) => self.pending = Some((revision, rx)),
                Err(e) => {
                    self.error = Some(format!("Cannot start Camera Raw preview: {e}"));
                    self.failed_revision = Some(revision);
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            // No browser worker/executor is available here. Preserve the interactive proxy
            // instead of blocking its UI with the full-image engine pipeline.
            drop(work);
            drop(rx);
            self.error = Some("Full-resolution filtered preview requires a native Camera Raw worker; the web preview keeps the proxy".into());
            self.failed_revision = Some(revision);
        }
    }

    /// Draw only the visible pixel rectangle. No full-image GPU texture and no GPU-size limit
    /// on the source image; uploads are capped to the current viewport plus a one-pixel border.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn paint(
        &mut self,
        ui: &egui::Ui,
        image: Rect,
        revision: u64,
        before: bool,
        params: &CameraRaw,
        needed: bool,
        display: &photocraft_engine::display_color::CanvasDisplay,
    ) -> bool {
        self.ready = false;
        self.shown = None;
        self.poll(revision);
        self.poll_screen();
        if !needed {
            return false;
        }
        // Smaller than 1:1 on screen: develop the frame at a level of its size over the part on
        // screen, not the full frame.
        let ppp = ui.ctx().pixels_per_point();
        let clip = ui.clip_rect();
        let scale = image.width() * ppp / self.area.width().max(1) as f32;
        // Fit's scale: the whole frame in the view.
        let fit = (clip.width() / self.area.width().max(1) as f32).min(clip.height() / self.area.height().max(1) as f32) * ppp;
        if let Some(level) = self.level_for(scale, fit) {
            return self.paint_screen(ui, image, revision, before, params, scale, level, display);
        }
        let Some(crop) = visible_pixels(image, clip, self.area, ui.input(|i| i.max_texture_side)) else { return false };
        let original = before || params.is_identity();
        if !original {
            self.request(ui.ctx(), revision, params);
            self.poll(revision);
        }
        let surface = if original {
            &self.source
        } else if let Some((r, surface)) = &self.result
            && *r == revision
        {
            surface
        } else {
            return self.paint_stale(ui, image, revision, original);
        };
        let key = (if original { 0 } else { revision }, original, crop, display.key);
        if self.crop != Some(key) {
            let color = crop_image(surface, crop, display);
            let options = TextureOptions { magnification: egui::TextureFilter::Nearest, ..TextureOptions::LINEAR };
            match &mut self.texture {
                Some(t) => t.set(color, options),
                None => self.texture = Some(ui.ctx().load_texture("cr-detail-preview", color, options)),
            }
            self.crop = Some(key);
        }
        if let Some(texture) = &self.texture {
            ui.painter().image(texture.id(), area_part_rect(image, self.area, crop), Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
            self.ready = true;
            self.shown = Some(Shown::Crop);
        }
        true
    }
    pub(crate) fn sample(&self, p: [f32; 2], revision: u64, before: bool, params: &CameraRaw) -> Option<[f32; 4]> {
        if !self.ready || !p.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) {
            return None;
        }
        let source = if before || params.is_identity() {
            &self.source
        } else if let Some((r, source)) = &self.result
            && *r == revision
        {
            source
        } else {
            return None;
        };
        let x = self.area.x0 + ((p[0] * self.area.width() as f32) as i32).min((self.area.width() - 1) as i32);
        let y = self.area.y0 + ((p[1] * self.area.height() as f32) as i32).min((self.area.height() - 1) as i32);
        Some(source.rgba(x, y))
    }

    pub(crate) fn overlay(&mut self, ui: &egui::Ui, image: Rect, revision: u64, mode: crate::camera_raw_scope_ui::ClippingMode) -> bool {
        use crate::camera_raw_scope_ui::ClippingMode;
        if !self.ready {
            return false;
        }
        let t = crate::theme::Tokens::get(ui.ctx());
        let clip_color = |p: &[f32; 4]| {
            if p[3] <= 0.0 || !p.iter().all(|v| v.is_finite()) {
                return Color32::TRANSPARENT;
            }
            let (lo, hi) = photocraft_algo::histogram::clipping(*p);
            match mode {
                ClippingMode::ShadowChannels => t.histogram_color(7 ^ lo),
                ClippingMode::HighlightChannels => {
                    if hi == 0 {
                        t.histogram_background()
                    } else {
                        t.histogram_color(hi)
                    }
                }
                ClippingMode::Warnings(w) => {
                    if w & 2 != 0 && hi != 0 {
                        t.histogram_color(1)
                    } else if w & 1 != 0 && lo != 0 {
                        t.histogram_color(4)
                    } else {
                        Color32::TRANSPARENT
                    }
                }
                ClippingMode::None => Color32::TRANSPARENT,
            }
        };
        // Where the overlay goes on screen, and its pixels: over what the last paint showed (a
        // crop of the full-size pixels, or the whole frame at the screen's size).
        let screen = match self.shown {
            Some(Shown::Screen(key, part)) => {
                let Some((_, _, px)) = self.screen_result.as_ref().filter(|(k, p, _)| *k == key && *p == part) else { return false };
                let overlay_key = (Some(key), key.revision, key.original, part, mode, t.kind);
                if self.overlay_key != Some(overlay_key) {
                    let color = ColorImage::new([part.width() as usize, part.height() as usize], px.iter().map(clip_color).collect());
                    self.set_overlay(ui, color);
                    self.overlay_key = Some(overlay_key);
                }
                level_part_rect(image, key.size, part)
            }
            Some(Shown::Crop) => {
                let Some((r, original, crop, _display_key)) = self.crop else { return false };
                let overlay_key = (None, r, original, crop, mode, t.kind);
                if self.overlay_key != Some(overlay_key) {
                    let surface = if original {
                        &self.source
                    } else if let Some((r, surface)) = &self.result
                        && *r == revision
                    {
                        surface
                    } else {
                        return false;
                    };
                    let mut row = vec![[0.0; 4]; crop.width() as usize];
                    let mut pixels = Vec::with_capacity(crop.width() as usize * crop.height() as usize);
                    for y in crop.y0..crop.y1 {
                        surface.read_rgba_into(PixelRect::new(crop.x0, y, crop.x1, y + 1), &mut row);
                        pixels.extend(row.iter().map(clip_color));
                    }
                    self.set_overlay(ui, ColorImage::new([crop.width() as usize, crop.height() as usize], pixels));
                    self.overlay_key = Some(overlay_key);
                }
                area_part_rect(image, self.area, crop)
            }
            None => return false,
        };
        if let Some(texture) = &self.overlay_texture {
            ui.painter().image(texture.id(), screen, Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
        }
        true
    }

    fn set_overlay(&mut self, ui: &egui::Ui, color: ColorImage) {
        match &mut self.overlay_texture {
            Some(t) => t.set(color, TextureOptions::NEAREST),
            None => self.overlay_texture = Some(ui.ctx().load_texture("cr-detail-clipping", color, TextureOptions::NEAREST)),
        }
    }
}

/// The raw stage of a raw open (see [`raw_stage_split`]; on developed pixels through
/// [`photocraft_io::raw::retone_with`]) and the tonal stages of `p` on `source`, or `cache` when it
/// holds them for the same settings. Returns the settings left for
/// [`photocraft_algo::camera_raw::develop_colour`] and the tonal result.
pub(crate) fn tonal_stage(
    source: &[[f32; 4]],
    w: usize,
    h: usize,
    p: &CameraRaw,
    raw: Option<&photocraft_io::raw::RawPreview>,
    cache: &mut TonalCache,
) -> (CameraRaw, Pixels) {
    let (colour, stage) = raw_stage_split(p, raw);
    let key = photocraft_algo::camera_raw::tonal_part(p);
    if let Some((k, px)) = cache.tonal.as_ref()
        && *k == key
    {
        return (colour, px.clone());
    }
    // The raw stage only changes with its own settings.
    let staged = match (raw, stage) {
        (Some(raw), Some(stage)) if !stage.0.is_neutral() || stage.1.is_some() => match cache.raw.as_ref() {
            Some((s, px)) if *s == stage => Some(px.clone()),
            _ => {
                let mut px = source.to_vec();
                let (tone, pre) = stage;
                match cache.retone.lock() {
                    Ok(mut kept) => photocraft_io::raw::retone_with(&mut px, w, h, &raw.curve, tone, pre, Some(&mut kept)),
                    Err(_) => photocraft_io::raw::retone_with(&mut px, w, h, &raw.curve, tone, pre, None),
                }
                let px = Arc::new(px);
                cache.raw = Some((stage, px.clone()));
                Some(px)
            }
        },
        _ => None,
    };
    let mut px = staged.map_or_else(|| source.to_vec(), |r| r.to_vec());
    let tonal = photocraft_algo::camera_raw::tonal_part(&colour);
    match cache.develop.lock() {
        Ok(mut kept) => photocraft_algo::camera_raw::develop_tonal_with(&mut px, w, h, &tonal, &mut kept),
        Err(_) => photocraft_algo::camera_raw::develop_tonal(&mut px, w, h, &tonal),
    }
    let px = Arc::new(px);
    cache.tonal = Some((key, px.clone()));
    (colour, px)
}

/// `area` of `source` developed at level `key.size` with `params` (whose `pixel_scale` is that
/// level's) over `part` (level pixels), through `cache`: the whole level's raw and tonal stages
/// (kept for the settings they were made with), then the colour stages over `part` grown by
/// their reach. `None` when `given_up` says so (asked between stages).
#[allow(clippy::too_many_arguments)]
fn develop_screen(
    source: &Surface,
    area: PixelRect,
    raw: Option<&photocraft_io::raw::RawPreview>,
    cache: &std::sync::Mutex<ScreenCache>,
    key: ScreenKey,
    part: PixelRect,
    params: &CameraRaw,
    float: bool,
    given_up: &dyn Fn() -> bool,
) -> Option<Pixels> {
    use photocraft_algo::camera_raw::{Frame, colour_reach, develop_colour_in};
    let (w, h) = key.size;
    let level = PixelRect::new(0, 0, w as i32, h as i32);
    let part = part.intersect(&level);
    if part.is_empty() || w == 0 || h == 0 {
        return Some(Arc::new(Vec::new()));
    }
    let mut guard = cache.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let c = &mut *guard;
    let entry = match c.levels.iter().position(|l| l.size == key.size) {
        Some(i) => c.levels.remove(i),
        None => LevelCache { size: key.size, source: Arc::new(reduce(source, area, w, h)), tonal: TonalCache::default() },
    };
    c.levels.insert(0, entry);
    trim_levels(&mut c.levels, SCREEN_MAX_PIXELS);
    let entry = &mut c.levels[0];
    let reduced = entry.source.clone();
    let (colour, base) = if key.original { (params.clone(), reduced) } else { tonal_stage(&reduced, w, h, params, raw, &mut entry.tonal) };
    drop(guard);
    if given_up() {
        return None;
    }
    let (pw, ph) = (part.width() as usize, part.height() as usize);
    if key.original {
        let mut px = Vec::with_capacity(pw * ph);
        for row in base.chunks_exact(w).skip(part.y0 as usize).take(ph) {
            px.extend_from_slice(row.get(part.x0 as usize..part.x1 as usize)?);
        }
        return Some(Arc::new(px));
    }
    // The colour stages over `part` grown by their reach, placed in the level.
    let reach = i32::try_from(colour_reach(&colour)).unwrap_or(i32::MAX);
    let grown = part.inflate(reach).intersect(&level);
    let (gw, gh) = (grown.width() as usize, grown.height() as usize);
    let (gx, gy) = (grown.x0 as usize, grown.y0 as usize);
    let mut px = Vec::with_capacity(gw * gh);
    for row in base.chunks_exact(w).skip(gy).take(gh) {
        px.extend_from_slice(row.get(gx..gx + gw)?);
    }
    develop_colour_in(&mut px, gw, gh, &colour, float, Frame { x: gx, y: gy, width: w, height: h });
    let (ox, oy) = ((part.x0 - grown.x0) as usize, (part.y0 - grown.y0) as usize);
    let mut inner = Vec::with_capacity(pw * ph);
    for row in px.chunks_exact(gw).skip(oy).take(ph) {
        inner.extend_from_slice(row.get(ox..ox + pw)?);
    }
    Some(Arc::new(inner))
}

/// Keeps the most recently used `levels` (first) up to [`LEVELS_KEPT`] of them adding up to
/// `budget` pixels at most; the first always stays.
fn trim_levels(levels: &mut Vec<LevelCache>, budget: usize) {
    let mut total = 0usize;
    let mut kept = 0;
    for l in levels.iter() {
        total = total.saturating_add(l.size.0.saturating_mul(l.size.1));
        if kept > 0 && (kept == LEVELS_KEPT || total > budget) {
            break;
        }
        kept += 1;
    }
    levels.truncate(kept);
}

/// The scale (screen pixels per image pixel) a view at `scale` is developed at below 1:1: Fit's
/// own when the view is at Fit or further out, else the smallest halving of 1:1 at or above the
/// view's scale, and above ½ the eighth at or above it (⅝, ¾, ⅞: below 1:1 the full-size path
/// develops the whole frame on every edit). Fit stays exact (where settings are dragged most);
/// the steps keep a zoom gesture from developing the frame at every scale it passes, and the view
/// resamples from them.
fn level_scale(scale: f32, fit: f32) -> f32 {
    let fit = if fit.is_finite() && fit > 0.0 { fit.min(1.0) } else { 1.0 };
    if scale.is_nan() || scale <= fit * 1.001 {
        return fit;
    }
    if scale > 0.5 {
        return ((scale * 8.0).ceil() / 8.0).min(1.0);
    }
    let mut level = 0.5f32;
    while level * 0.5 >= scale {
        level *= 0.5;
    }
    level
}

/// The size of `area` at `level`.
fn level_size(area: PixelRect, level: f32) -> (usize, usize) {
    let size = |n: u32| ((n as f32 * level).round() as usize).clamp(1, n.max(1) as usize);
    (size(area.width()), size(area.height()))
}

/// The part of `level` to develop for the visible `crop`: a quarter of the view more on every
/// side, so a short pan stays inside, but never wider or taller than `max_side` (its texture).
fn grown_part(crop: PixelRect, level: PixelRect, max_side: usize) -> PixelRect {
    let room = |n: u32| (max_side.saturating_sub(n as usize) / 2).min(i32::MAX as usize) as i32;
    let margin = ((crop.width().max(crop.height()) / 4) as i32).min(room(crop.width())).min(room(crop.height()));
    crop.inflate(margin).intersect(&level)
}

/// Where `part` of a `level`-sized develop of the frame lies on screen when the frame fills `image`.
fn level_part_rect(image: Rect, level: (usize, usize), part: PixelRect) -> Rect {
    let scale = image.size() / vec2(level.0.max(1) as f32, level.1.max(1) as f32);
    Rect::from_min_size(image.min + vec2(part.x0 as f32, part.y0 as f32) * scale, vec2(part.width() as f32, part.height() as f32) * scale)
}

/// Where `part` of `area` lies on screen when the area fills `image`.
fn area_part_rect(image: Rect, area: PixelRect, part: PixelRect) -> Rect {
    let scale = image.size() / vec2(area.width().max(1) as f32, area.height().max(1) as f32);
    Rect::from_min_size(
        image.min + vec2((part.x0 - area.x0) as f32, (part.y0 - area.y0) as f32) * scale,
        vec2(part.width() as f32, part.height() as f32) * scale,
    )
}

fn hash_of<T: std::hash::Hash>(v: &T) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

/// `tw × th` pixels from a `w × h` image whose row `y` `read` fills: each output pixel the mean
/// of the source pixels it covers, by area. The output rows run in parallel off wasm.
fn reduce_with(w: usize, h: usize, tw: usize, th: usize, read: impl Fn(usize, &mut [[f32; 4]]) + Sync) -> Vec<[f32; 4]> {
    let mut out = vec![[0.0f32; 4]; tw * th];
    if w == 0 || h == 0 || tw == 0 || th == 0 {
        return out;
    }
    // Per output column (row): the source columns (rows) it covers and their share of it.
    let spans = |n: usize, m: usize| -> Vec<Vec<(usize, f32)>> {
        (0..m)
            .map(|o| {
                let (a, b) = (o as f64 * n as f64 / m as f64, (o + 1) as f64 * n as f64 / m as f64);
                (a.floor() as usize..(b.ceil() as usize).min(n)).map(|i| (i, ((b.min(i as f64 + 1.0) - a.max(i as f64)) / (b - a)) as f32)).collect()
            })
            .collect()
    };
    let (xs, ys) = (spans(w, tw), spans(h, th));
    let fill = |(orow, rows): (&mut [[f32; 4]], &Vec<(usize, f32)>)| {
        let mut row = vec![[0.0f32; 4]; w];
        for &(iy, wy) in rows {
            read(iy, &mut row);
            for (o, cols) in orow.iter_mut().zip(&xs) {
                for &(ix, wx) in cols {
                    let k = wx * wy;
                    for c in 0..4 {
                        o[c] += row[ix][c] * k;
                    }
                }
            }
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(tw).zip(ys.par_iter()).for_each(fill);
    }
    #[cfg(target_arch = "wasm32")]
    out.chunks_mut(tw).zip(ys.iter()).for_each(fill);
    out
}

/// `area` of `source` reduced to `tw × th` (each output pixel the mean of the source pixels it
/// covers, by area).
fn reduce(source: &Surface, area: PixelRect, tw: usize, th: usize) -> Vec<[f32; 4]> {
    reduce_with(area.width() as usize, area.height() as usize, tw, th, |y, row| {
        let y = area.y0 + y as i32;
        source.read_rgba_into(PixelRect::new(area.x0, y, area.x1, y + 1), row);
    })
}

/// The full-size result of `params` over `area`. For a raw being opened (`raw`), the raw stage
/// (see [`raw_stage_split`]) is applied first to the whole area, from the whole frame's maps as
/// Open computes them ([`photocraft_io::raw::retone_with`]), then the rest of the pipeline. Open
/// develops from the sensor data, the preview from the developed pixels, which are clipped at
/// white: at 1:1 the preview is close to what Open makes, except where highlights were clipped
/// (lowering Exposure or Highlights there recovers detail in Open only) and in the deepest
/// shadows (see [`photocraft_io::raw::retone`]).
fn develop(source: &Surface, raw: Option<&photocraft_io::raw::RawPreview>, area: PixelRect, params: &CameraRaw, coverage: &Coverage) -> Surface {
    let (rest, stage) = raw_stage_split(params, raw);
    let (retoned, params) = match (raw, stage) {
        (Some(raw), Some((tone, pre))) if !tone.is_neutral() || pre.is_some() => {
            let (w, h) = (area.width() as usize, area.height() as usize);
            let mut px = vec![[0.0f32; 4]; w * h];
            source.read_rgba_into(area, &mut px);
            photocraft_io::raw::retone_with(&mut px, w, h, &raw.curve, tone, pre, None);
            let fmt = source.format();
            let n = fmt.channels();
            // Written back in bands of rows, so only one band-sized buffer lives next to `px`
            // (a whole-area one is another 4 bytes per channel and pixel). Each band is read from
            // the source first to keep its stored alpha.
            let mut out = source.clone();
            let band = (65_536 / w.max(1)).max(1);
            let mut data = Vec::new();
            for (b, rows) in px.chunks(band * w.max(1)).enumerate() {
                let y0 = area.y0 + (b * band) as i32;
                let rect = PixelRect::new(area.x0, y0, area.x1, y0 + (rows.len() / w.max(1)) as i32);
                source.read_region_into(rect, &mut data);
                for (o, q) in data.chunks_exact_mut(n).zip(rows) {
                    let alpha = if fmt.alpha { o.get(n - 1).copied() } else { None };
                    photocraft_raster::from_rgba_into(&fmt, *q, o);
                    if let (Some(a), Some(last)) = (alpha, o.get_mut(n - 1)) {
                        *last = a;
                    }
                }
                out.write_region(rect, &data);
            }
            (Some(out), rest)
        }
        _ => (None, rest),
    };
    let base = retoned.as_ref().unwrap_or(source);
    let mut result = photocraft_engine::lens_cmds::camera_raw_surface(base, area, &params);
    if matches!(coverage, Coverage::None) {
        return result;
    }
    let width = area.width() as usize;
    let channels = source.channels();
    let mut original = Vec::new();
    let mut row = Vec::new();
    for y in area.y0..area.y1 {
        let rect = PixelRect::new(area.x0, y, area.x1, y + 1);
        source.read_region_into(rect, &mut original);
        result.read_region_into(rect, &mut row);
        for (x, (out, original)) in row.chunks_exact_mut(channels).zip(original.chunks_exact(channels)).take(width).enumerate() {
            let x = area.x0 + x as i32;
            let alpha = match coverage {
                Coverage::Selection(mask) => mask.sample_channel(x, y, 0),
                Coverage::Mask(mask) => mask.value(x, y),
                Coverage::None => 1.0,
            }
            .clamp(0.0, 1.0);
            for (out, original) in out.iter_mut().zip(original) {
                *out = *original + (*out - *original) * alpha;
            }
        }
        result.write_region(rect, &row);
    }
    result
}

fn visible_pixels(image: Rect, viewport: Rect, area: PixelRect, max_side: usize) -> Option<PixelRect> {
    let visible = image.intersect(viewport);
    if !image.is_finite() || image.width() <= 0.0 || image.height() <= 0.0 || !visible.is_positive() || area.is_empty() {
        return None;
    }
    let scale = vec2(area.width() as f32, area.height() as f32) / image.size();
    let lo = (visible.min - image.min) * scale;
    let hi = (visible.max - image.min) * scale;
    let crop = PixelRect::new(
        area.x0.saturating_add((lo.x.floor() as i32).saturating_sub(1)),
        area.y0.saturating_add((lo.y.floor() as i32).saturating_sub(1)),
        area.x0.saturating_add((hi.x.ceil() as i32).saturating_add(1)),
        area.y0.saturating_add((hi.y.ceil() as i32).saturating_add(1)),
    )
    .intersect(&area);
    // Proxy is sufficient below its native scale; a larger intermediate should never be
    // allocated if an unusually large viewport exceeds the device's texture limit.
    (crop.width() as usize <= max_side && crop.height() as usize <= max_side && crop.width() as u64 * crop.height() as u64 <= 16_777_216).then_some(crop)
}

/// Pixels per band [`display_image`] reads and colour-converts at once: at least `photocraft-cms`'s
/// parallel threshold (16K pixels), and small enough that a band's float scratch is 1 MiB.
const CROP_BAND_PIXELS: usize = 64 * 1024;

/// The visible pixels of `surface`, as the canvas shows them.
fn crop_image(surface: &Surface, area: PixelRect, display: &photocraft_engine::display_color::CanvasDisplay) -> ColorImage {
    display_image(area, display, |rect, out| surface.read_rgba_into(rect, out))
}

/// `area` as the canvas shows it, with `read` filling each band's float pixels. Converted in
/// bands of rows, in parallel off wasm, not one row or one whole copy at a time: a row is below
/// `photocraft-cms`'s parallel threshold, so a per-row conversion ran the display transform (any
/// document profile other than the monitor's, such as an Adobe RGB photo on sRGB) on one core of
/// the UI thread for every edit (#4682).
pub(crate) fn display_image(
    area: PixelRect,
    display: &photocraft_engine::display_color::CanvasDisplay,
    read: impl Fn(PixelRect, &mut [[f32; 4]]) + Sync,
) -> ColorImage {
    let (width, height) = (area.width() as usize, area.height() as usize);
    let mut pixels = vec![Color32::TRANSPARENT; width * height];
    if width == 0 {
        return ColorImage::new([width, height], pixels);
    }
    let band_rows = (CROP_BAND_PIXELS / width).max(1);
    let band = |index: usize, out: &mut [Color32]| {
        let y0 = area.y0.saturating_add((index * band_rows) as i32);
        let rect = PixelRect::new(area.x0, y0, area.x1, y0.saturating_add((out.len() / width) as i32));
        let mut buf = photocraft_compose::Buffer { rect, px: vec![[0.0; 4]; out.len()] };
        read(rect, &mut buf.px);
        let img = display.to_rgba8(&buf);
        for (o, &[r, g, b, a]) in out.iter_mut().zip(img.pixels.as_chunks::<4>().0) {
            *o = Color32::from_rgba_unmultiplied(r, g, b, a);
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        pixels.par_chunks_mut(band_rows * width).enumerate().for_each(|(i, out)| band(i, out));
    }
    #[cfg(target_arch = "wasm32")]
    for (i, out) in pixels.chunks_mut(band_rows * width).enumerate() {
        band(i, out);
    }
    ColorImage::new([width, height], pixels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::PixelFormat;
    #[test]
    fn refinement_uses_the_engine_at_source_resolution_and_respects_selection() {
        let area = PixelRect::new(10, -5, 42, 19);
        let mut source = Surface::new(PixelFormat::RGBA8);
        let mut values = vec![0.0; 32 * 24 * 4];
        for (i, p) in values.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            p.copy_from_slice(&[if i % 2 == 0 { 0.1 } else { 0.8 }, 0.3, 0.6, 1.0]);
        }
        source.write_region(area, &values);
        let params = CameraRaw { exposure: 0.5, texture: 15.0, grain_amount: 30.0, ..Default::default() };
        let expected = photocraft_engine::lens_cmds::camera_raw_surface(&source, area, &params);
        assert_eq!(develop(&source, None, area, &params, &Coverage::None), expected);
        let mask = Surface::with_default(PixelFormat::GRAY8, &[0.0]);
        assert_eq!(develop(&source, None, area, &params, &Coverage::Selection(mask)), source);
    }
    #[test]
    fn a_raw_open_refines_raw_stage_sliders_from_the_whole_frame() {
        // Shadows and Exposure on a raw being opened: the full-size refinement retones the whole
        // frame (as Open does, the exposure gain first) and runs the rest of the pipeline after,
        // instead of the filter's Shadows and Exposure.
        let area = PixelRect::new(0, 0, 96, 64);
        let mut source = Surface::new(PixelFormat::RGBA16);
        let mut values = vec![0.0; 96 * 64 * 4];
        for (i, p) in values.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let v = if (i % 96) < 48 { 0.12 } else { 0.7 };
            p.copy_from_slice(&[v, v * 0.9, v * 0.8, 1.0]);
        }
        source.write_region(area, &values);
        let curve = [[0.01, 0.01], [0.05, 0.12], [0.3, 0.6], [1.0, 1.0]];
        let params = CameraRaw { shadows: 80.0, exposure: 0.25, ..Default::default() };
        let raw = photocraft_io::raw::RawPreview::with_curve(curve.to_vec());
        let refined = develop(&source, Some(&raw), area, &params, &Coverage::None);
        let mut px = vec![[0.0f32; 4]; 96 * 64];
        source.read_rgba_into(area, &mut px);
        let tone = photocraft_io::raw::Tone { shadows: 80.0, ..Default::default() };
        let gain = 2f32.powf(0.25);
        let pre = [[gain, 0.0, 0.0], [0.0, gain, 0.0], [0.0, 0.0, gain]];
        photocraft_io::raw::retone_with(&mut px, 96, 64, &curve, tone, Some(pre), None);
        let mut retoned = source.clone();
        let flat: Vec<f32> = px.iter().flatten().copied().collect();
        retoned.write_region(area, &flat);
        let expected = photocraft_engine::lens_cmds::camera_raw_surface(&retoned, area, &CameraRaw::default());
        let (a, b) = (refined.rgba(10, 30), expected.rgba(10, 30));
        assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 2e-4), "{a:?} vs {b:?}");
        assert!(
            refined.rgba(10, 30)[0] > source.rgba(10, 30)[0] + 0.02,
            "the shadow side is lifted: {:?} from {:?}, retoned {:?}",
            refined.rgba(10, 30),
            source.rgba(10, 30),
            retoned.rgba(10, 30)
        );
        // Without a raw curve Shadows is the filter's.
        let filter = develop(&source, None, area, &params, &Coverage::None);
        assert_eq!(filter, photocraft_engine::lens_cmds::camera_raw_surface(&source, area, &params));
    }

    #[test]
    fn stale_refinements_and_failed_old_revisions_do_not_replace_new_settings() {
        let source = Surface::with_default(PixelFormat::RGBA8, &[0.2, 0.4, 0.6, 1.0]);
        let mut preview = DetailPreview::new(source.clone(), PixelRect::new(0, 0, 16, 16), Coverage::None);
        let (tx, rx) = std::sync::mpsc::channel();
        preview.pending = Some((1, rx));
        tx.send(Ok((1, source.clone()))).unwrap();
        preview.poll(2);
        assert!(preview.result.is_none());
        assert!(!preview.pending());
        let (tx, rx) = std::sync::mpsc::channel();
        preview.pending = Some((1, rx));
        tx.send(Err("old failure".into())).unwrap();
        preview.poll(2);
        assert_eq!(preview.failed_revision, Some(1), "a failure must not suppress the new revision");
        let (tx, rx) = std::sync::mpsc::channel();
        preview.pending = Some((2, rx));
        tx.send(Ok((2, source))).unwrap();
        preview.poll(2);
        assert_eq!(preview.result.as_ref().map(|(r, _)| *r), Some(2));
        assert!(preview.error.is_none());
    }

    #[test]
    fn full_resolution_sampling_and_crops_keep_fine_detail_at_every_supported_depth() {
        use photocraft_color::{ColorMode, PixelFormat, SampleType};
        for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
            for mode in [ColorMode::Rgb, ColorMode::Grayscale] {
                let format = PixelFormat { mode, sample: depth, alpha: true };
                let area = PixelRect::new(-12, 8, 28, 38);
                let mut source = Surface::new(format);
                let channels = format.channels();
                let mut pixels = vec![0.0; 40 * 30 * channels];
                for (i, pixel) in pixels.chunks_exact_mut(channels).enumerate() {
                    pixel.fill(if i % 2 == 0 { 0.1 } else { 0.9 });
                    pixel[channels - 1] = 1.0;
                }
                source.write_region(area, &pixels);
                let params = CameraRaw { exposure: 0.5, sharpen_amount: 20.0, grain_amount: 30.0, ..Default::default() };
                let full = develop(&source, None, area, &params, &Coverage::None);
                let expected = photocraft_engine::lens_cmds::camera_raw_surface(&source, area, &params);
                assert_eq!(full, expected);
                let mut preview = DetailPreview::new(source, area, Coverage::None);
                preview.result = Some((1, full));
                preview.ready = true;
                let dark = preview.sample([0.0, 0.0], 1, false, &params).unwrap();
                let light = preview.sample([1.0 / 40.0, 0.0], 1, false, &params).unwrap();
                assert!(light[0] - dark[0] > 0.5, "native alternating pixels survive: {mode:?}/{depth:?}");
                let mut doc =
                    photocraft_doc::Document::new("test", photocraft_geom::Size::new(40, 30), photocraft_doc::ColorMode::Rgb, photocraft_doc::SampleType::U8);
                doc.icc_profile = Some(photocraft_cms::Builtin::Srgb.profile().to_bytes());
                let display = photocraft_engine::color_cmds::ColorState::default().canvas_display(&doc).unwrap();
                assert_eq!(crop_image(&expected, PixelRect::new(-12, 8, -10, 9), &display).size, [2, 1]);
                assert!(preview.sample([f32::NAN, 0.0], 1, false, &params).is_none());
                assert!(preview.sample([0.5, 0.5], 2, false, &params).is_none());
            }
        }
    }

    #[test]
    fn refined_crop_uses_linear_document_display_encoding() {
        let area = PixelRect::new(0, 0, 1, 1);
        let mut source =
            Surface::new(photocraft_color::PixelFormat { mode: photocraft_color::ColorMode::Rgb, sample: photocraft_color::SampleType::F32, alpha: true });
        source.write_region(area, &[0.25, 0.25, 0.25, 1.0]);
        let mut doc = photocraft_doc::Document::new("test", photocraft_geom::Size::new(1, 1), photocraft_doc::ColorMode::Rgb, photocraft_doc::SampleType::F32);
        doc.icc_profile = Some(photocraft_cms::Builtin::LinearSrgb.profile().to_bytes());
        let mut color = photocraft_engine::color_cmds::ColorState::default();
        color.settings.monitor_profile = "srgb".into();
        let display = color.canvas_display(&doc).unwrap();
        let image = crop_image(&source, area, &display);
        assert!(image.pixels[0].r() > 130, "linear 0.25 should be shown as encoded sRGB");
    }

    #[test]
    fn huge_sparse_images_keep_the_proxy_without_starting_an_unbounded_worker() {
        let source = Surface::new(PixelFormat::RGBA8);
        let mut preview = DetailPreview::new(source, PixelRect::new(0, 0, 8001, 8000), Coverage::None);
        preview.request(&Context::default(), 1, &CameraRaw { exposure: 1.0, ..Default::default() });
        assert!(!preview.pending());
        assert!(preview.error.as_deref().unwrap().contains("64 megapixels"));
        assert_eq!(preview.failed_revision, Some(1));
    }

    #[test]
    fn zoom_uploads_only_visible_pixels_even_on_a_huge_offset_image() {
        let area = PixelRect::new(-200, 10, 15800, 10010);
        let image = Rect::from_min_size(pos2(-7800.0, -4800.0), vec2(16000.0, 10000.0));
        let visible = Rect::from_min_size(Pos2::ZERO, vec2(800.0, 600.0));
        let crop = visible_pixels(image, visible, area, 4096).unwrap();
        assert_eq!((crop.width(), crop.height()), (802, 602));
        assert!(visible_pixels(image, Rect::from_min_size(Pos2::ZERO, vec2(8000.0, 6000.0)), area, 4096).is_none());
        assert!(visible_pixels(Rect::NOTHING, visible, area, 4096).is_none());
    }

    #[test]
    fn banded_crop_matches_one_conversion_of_the_whole_area() {
        // Several bands, a ragged last one and an area off the image origin, through a real
        // transform (an Adobe RGB document on an sRGB monitor) and without one.
        let area = PixelRect::new(-3, 5, 1021, 5 + 2 * (CROP_BAND_PIXELS / 1024) as i32 + 7);
        let mut source = Surface::new(PixelFormat { mode: photocraft_color::ColorMode::Rgb, sample: photocraft_color::SampleType::U8, alpha: true });
        let n = (area.width() * area.height()) as usize;
        let px: Vec<f32> = (0..n).flat_map(|i| [(i % 251) as f32 / 255.0, (i % 241) as f32 / 255.0, (i % 239) as f32 / 255.0, (i % 7) as f32 / 6.0]).collect();
        source.write_region(area, &px);
        for profile in [photocraft_cms::Builtin::AdobeRgbCompat, photocraft_cms::Builtin::Srgb] {
            let mut doc =
                photocraft_doc::Document::new("test", photocraft_geom::Size::new(1024, 1024), photocraft_doc::ColorMode::Rgb, photocraft_doc::SampleType::U8);
            doc.icc_profile = Some(profile.profile().to_bytes());
            let mut color = photocraft_engine::color_cmds::ColorState::default();
            color.settings.monitor_profile = "srgb".into();
            let display = color.canvas_display(&doc).unwrap();
            assert_eq!(display.transform.is_some(), profile == photocraft_cms::Builtin::AdobeRgbCompat);
            let mut whole = photocraft_compose::Buffer { rect: area, px: vec![[0.0; 4]; n] };
            source.read_rgba_into(area, &mut whole.px);
            let expected = display.to_rgba8(&whole);
            let image = crop_image(&source, area, &display);
            assert_eq!(image.size, [area.width() as usize, area.height() as usize]);
            let expected: Vec<Color32> = expected.pixels.as_chunks::<4>().0.iter().map(|&[r, g, b, a]| Color32::from_rgba_unmultiplied(r, g, b, a)).collect();
            assert_eq!(image.pixels, expected, "{profile:?}");
        }
        assert_eq!(
            crop_image(
                &source,
                PixelRect::new(0, 0, 0, 4),
                &photocraft_engine::color_cmds::ColorState::default()
                    .canvas_display(&photocraft_doc::Document::new(
                        "t",
                        photocraft_geom::Size::new(1, 1),
                        photocraft_doc::ColorMode::Rgb,
                        photocraft_doc::SampleType::U8
                    ))
                    .unwrap()
            )
            .size,
            [0, 4]
        );
    }

    fn ramp(w: u32, h: u32) -> (Surface, PixelRect) {
        let area = PixelRect::new(0, 0, w as i32, h as i32);
        let mut source = Surface::new(PixelFormat::RGBA16);
        let values: Vec<f32> = (0..(w * h) as usize)
            .flat_map(|i| {
                let v = (i % w as usize) as f32 / w as f32;
                [v, v * 0.8 + 0.1, 1.0 - v, 1.0]
            })
            .collect();
        source.write_region(area, &values);
        (source, area)
    }

    fn sdisplay() -> Arc<photocraft_engine::display_color::CanvasDisplay> {
        let doc = photocraft_doc::Document::new("t", photocraft_geom::Size::new(1, 1), photocraft_doc::ColorMode::Rgb, photocraft_doc::SampleType::U8);
        photocraft_engine::color_cmds::ColorState::default().canvas_display(&doc).unwrap()
    }

    /// Below 1:1 the frame develops at a level of [`SCREEN_MAX_PIXELS`] at most (the view's own
    /// scale when the level's is past it), and the part sent to the GPU is never wider or taller
    /// than the device's texture side; past both, and with a selection or mask, the full-size
    /// path's crop takes over (#3817 review).
    #[test]
    fn the_screen_size_preview_stays_within_the_texture_side_and_its_pixel_budget() {
        let (source, area) = ramp(20_000, 4_000);
        let d = DetailPreview::new(source, area, Coverage::None);
        assert_eq!(d.level_for(0.1, 0.05), Some(0.125), "a halving: 2500 × 500");
        assert_eq!(d.level_for(0.35, 0.05), Some(0.35), "½ is 20 MP: the view's own scale, 9.8 MP");
        assert_eq!(d.level_for(0.5, 0.05), None, "20 MP either way");
        assert_eq!(d.level_for(0.99, 0.05), None, "1:1 and above");
        let (source, area) = ramp(8_000, 6_000);
        let d = DetailPreview::new(source, area, Coverage::None);
        assert_eq!(d.level_for(0.6, 0.1), None, "⅝ (18.8 MP) and 0.6 (17.3 MP) are past the pixel budget");
        assert_eq!(d.level_for(0.45, 0.1), Some(0.5), "4000 × 3000");
        let (source, area) = ramp(800, 600);
        let mask = Surface::with_default(PixelFormat::GRAY8, &[1.0]);
        assert_eq!(DetailPreview::new(source, area, Coverage::Selection(mask)).level_for(0.5, 0.5), None);
        // A view as large as the texture side gets no margin past it.
        let level = PixelRect::new(0, 0, 10_000, 3_000);
        let part = grown_part(PixelRect::new(1000, 500, 5096, 2000), level, 4096);
        assert!(part.width() <= 4096 && part.contains_rect(&PixelRect::new(1000, 500, 5096, 2000)), "{part:?}");
        let part = grown_part(PixelRect::new(1000, 500, 2000, 1500), level, 8192);
        assert_eq!(part, PixelRect::new(750, 250, 2250, 1750), "a quarter of the view more on every side");
    }

    /// A failed level develop is reported once and not asked for again every frame; a later
    /// success takes the error down (#3817 review).
    #[test]
    fn a_failed_screen_develop_is_not_retried_and_a_later_frame_clears_its_error() {
        let (source, area) = ramp(64, 48);
        let mut d = DetailPreview::new(source, area, Coverage::None);
        let key = ScreenKey { revision: 3, size: (32, 24), original: false };
        let part = PixelRect::new(0, 0, 32, 24);
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(Err("Camera Raw preview failed; the proxy remains available".to_string())).unwrap();
        d.screen_pending = Some((key, part, rx));
        d.poll_screen();
        assert!(d.error.is_some() && d.screen_failed == Some(key) && d.screen_pending.is_none());
        d.request_screen(&Context::default(), key, part, &CameraRaw { exposure: 0.5, ..Default::default() });
        assert!(d.screen_pending.is_none(), "the failed key is not asked for again");
        let next = ScreenKey { revision: 4, ..key };
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(Ok(Some((next, part, Arc::new(vec![[0.5; 4]; 32 * 24]))))).unwrap();
        d.screen_pending = Some((next, part, rx));
        d.poll_screen();
        assert_eq!(d.error, None);
    }

    /// Paints `image` until the preview is ready (and its overlay drawn), on a screen of `screen`
    /// points: Fit is its scale.
    fn paint_ready(d: &mut DetailPreview, ctx: &Context, screen: Rect, image: Rect, params: &CameraRaw, revision: u64) {
        let display = sdisplay();
        let mode = crate::camera_raw_scope_ui::ClippingMode::Warnings(3);
        let input = || egui::RawInput { screen_rect: Some(screen), ..Default::default() };
        for _ in 0..2000 {
            let mut painted = false;
            ctx.run_ui(input(), |ui| {
                painted = d.paint(ui, image, revision, false, params, true, &display) && d.ready && d.overlay(ui, image, revision, mode);
            })
            .textures_delta
            .clear();
            if painted {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        panic!("the detail preview never painted");
    }

    /// After 1:1, a Fit view's clipping overlay is made from the level part it shows, not from the
    /// last 1:1 crop (#3817 review: it covered the old 1:1 rectangle).
    #[test]
    fn the_clipping_overlay_follows_the_screen_size_frame() {
        let (source, area) = ramp(400, 300);
        let mut d = DetailPreview::new(source, area, Coverage::None);
        let ctx = Context::default();
        let screen = Rect::from_min_size(Pos2::ZERO, vec2(200.0, 150.0));
        let params = CameraRaw { exposure: 1.5, ..Default::default() };
        // 1:1: a crop of the full-size result.
        paint_ready(&mut d, &ctx, screen, Rect::from_min_size(Pos2::ZERO, vec2(400.0, 300.0)), &params, 1);
        assert!(matches!(d.shown, Some(Shown::Crop)) && d.overlay_key.is_some_and(|k| k.0.is_none()));
        // Fit at half size: the level, and its overlay.
        paint_ready(&mut d, &ctx, screen, Rect::from_min_size(Pos2::ZERO, vec2(200.0, 150.0)), &params, 1);
        assert_eq!(d.screen_size(), Some((200, 150)));
        let key = d.overlay_key.unwrap();
        assert_eq!(key.0.map(|k| k.size), Some((200, 150)));
        assert_eq!(key.3, PixelRect::new(0, 0, 200, 150));
    }

    /// Zooming between scales of one level develops nothing new: the kept part is resampled to
    /// the view; a new revision develops at once, and going back to a kept level reuses its stages.
    #[test]
    fn a_zoom_step_inside_a_kept_level_does_no_develop() {
        let (source, area) = ramp(800, 600);
        let mut d = DetailPreview::new(source, area, Coverage::None);
        let ctx = Context::default();
        let screen = Rect::from_min_size(Pos2::ZERO, vec2(200.0, 150.0));
        let params = CameraRaw { exposure: 0.7, saturation: 20.0, ..Default::default() };
        // Fit (¼), then 0.3: the ½ level, all of which fits in the view's part.
        paint_ready(&mut d, &ctx, screen, Rect::from_min_size(Pos2::ZERO, vec2(200.0, 150.0)), &params, 1);
        assert_eq!(d.screen_size(), Some((200, 150)));
        paint_ready(&mut d, &ctx, screen, Rect::from_min_size(pos2(-60.0, -45.0), vec2(240.0, 180.0)), &params, 1);
        assert_eq!(d.screen_size(), Some((400, 300)));
        let half = d.screen_result.as_ref().map(|(k, p, _)| (*k, *p)).unwrap();
        // 0.4 and 0.45: the same level and the same developed part, no new develop.
        for (w, h) in [(320.0, 240.0), (360.0, 270.0)] {
            ctx.run_ui(egui::RawInput { screen_rect: Some(screen), ..Default::default() }, |ui| {
                d.paint(ui, Rect::from_min_size(pos2(100.0 - w / 2.0, 75.0 - h / 2.0), vec2(w, h)), 1, false, &params, true, &sdisplay());
            })
            .textures_delta
            .clear();
            assert!(d.ready && d.screen_pending.is_none(), "no develop at {w}");
            assert_eq!(d.screen_result.as_ref().map(|(k, p, _)| (*k, *p)), Some(half));
        }
        let c = d.screen_cache.lock().unwrap();
        assert_eq!(c.levels.iter().map(|l| l.size).collect::<Vec<_>>(), [(400, 300), (200, 150)], "both levels kept, newest first");
    }

    /// A new raw for the open-time dialog drops what was developed for the old one.
    #[test]
    fn a_new_raw_drops_the_screen_size_frames() {
        let (source, area) = ramp(64, 48);
        let mut d = DetailPreview::new(source, area, Coverage::None);
        let key = ScreenKey { revision: 1, size: (32, 24), original: false };
        d.screen_result = Some((key, PixelRect::new(0, 0, 32, 24), Arc::new(vec![[0.5; 4]; 32 * 24])));
        let _ = d.develop_screen_now(&CameraRaw { exposure: 0.5, ..Default::default() }, (32, 24), 1);
        assert_eq!(d.screen_cache.lock().unwrap().levels.len(), 1);
        d.set_raw(Some(Arc::new(photocraft_io::raw::RawPreview::with_curve(vec![[0.5, 0.6]]))));
        assert!(d.screen_result.is_none());
        assert!(d.screen_cache.lock().unwrap().levels.is_empty());
    }

    #[test]
    fn a_view_develops_at_fit_or_at_a_step_of_one_to_one() {
        // At Fit and further out: Fit's own scale, exactly (settings are dragged there most).
        assert_eq!(level_scale(0.264, 0.264), 0.264);
        assert_eq!(level_scale(0.2, 0.264), 0.264);
        assert_eq!(level_scale(f32::NAN, 0.264), 0.264);
        // Closer: the smallest halving of 1:1 the view fits in, eighths above ½.
        assert_eq!(level_scale(0.3, 0.264), 0.5);
        assert_eq!(level_scale(0.5, 0.264), 0.5);
        assert_eq!(level_scale(0.51, 0.264), 0.625);
        assert_eq!(level_scale(0.7, 0.264), 0.75);
        assert_eq!(level_scale(0.9, 0.264), 1.0);
        assert_eq!(level_scale(0.1, 0.05), 0.125);
        // A frame smaller than the view fits at 1:1 and never develops larger.
        assert_eq!(level_scale(0.5, 1.7), 1.0);
        assert_eq!(level_scale(0.5, 0.0), 1.0);
    }

    #[test]
    fn reduce_averages_the_source_pixels_each_output_pixel_covers() {
        let src = [[1.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 1.0], [5.0, 0.0, 0.0, 1.0], [7.0, 0.0, 0.0, 1.0], [9.0, 0.0, 0.0, 1.0], [11.0, 0.0, 0.0, 1.0]];
        let read = |y: usize, row: &mut [[f32; 4]]| row.copy_from_slice(&src[y * 3..y * 3 + 3]);
        // 3 × 2 → 1 × 1: the mean of everything; → 2 × 1: the middle column shared by halves.
        let one = reduce_with(3, 2, 1, 1, read);
        assert!((one[0][0] - 6.0).abs() < 1e-5 && one[0][3] == 1.0 && one.len() == 1, "{one:?}");
        let two = reduce_with(3, 2, 2, 1, read);
        assert!(
            (two[0][0] - (1.0 + 7.0 + 0.5 * (3.0 + 9.0)) / 3.0).abs() < 1e-5 && (two[1][0] - (5.0 + 11.0 + 0.5 * (3.0 + 9.0)) / 3.0).abs() < 1e-5,
            "{two:?}"
        );
        assert_eq!(reduce_with(3, 2, 3, 2, read), src.to_vec());
        assert!(reduce_with(0, 2, 3, 2, read).iter().all(|p| *p == [0.0; 4]));
    }

    #[test]
    fn parts_land_where_the_frame_shows_them() {
        let image = Rect::from_min_size(pos2(10.0, 20.0), vec2(200.0, 100.0));
        let r = level_part_rect(image, (400, 200), PixelRect::new(100, 50, 300, 150));
        assert_eq!((r.min, r.max), (pos2(60.0, 45.0), pos2(160.0, 95.0)));
        let r = area_part_rect(image, PixelRect::new(-40, 10, 360, 210), PixelRect::new(60, 60, 260, 160));
        assert_eq!((r.min, r.max), (pos2(60.0, 45.0), pos2(160.0, 95.0)));
    }

    /// A develop of part of a level, through the kept stages, is what the whole level's develop
    /// makes there: on a raw being opened, with Detail and the positional effects.
    #[test]
    fn a_level_part_develops_as_the_whole_level_does() {
        let area = PixelRect::new(3, -2, 323, 238);
        let mut source = Surface::new(PixelFormat::RGBA16);
        let (w, h) = (area.width() as usize, area.height() as usize);
        let mut values = vec![0.0; w * h * 4];
        for (i, p) in values.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            let v = 0.08 + 0.8 * (0.5 + 0.5 * (x * 0.21).sin() * (y * 0.13).cos());
            p.copy_from_slice(&[v, v * 0.85 + 0.05 * (x % 7.0) / 7.0, v * 0.7, 1.0]);
        }
        source.write_region(area, &values);
        let raw = photocraft_io::raw::RawPreview::with_curve(vec![[0.0, 0.0], [0.05, 0.12], [0.3, 0.6], [1.0, 1.0]]);
        let size = (160, 120);
        let whole = PixelRect::new(0, 0, 160, 120);
        let part = PixelRect::new(40, 30, 110, 80);
        let settings = [
            CameraRaw {
                shadows: 60.0,
                clarity: 20.0,
                exposure: 0.3,
                vibrance: 25.0,
                sharpen_amount: 50.0,
                noise_color: 40.0,
                pixel_scale: 0.5,
                ..Default::default()
            },
            CameraRaw { highlights: -50.0, texture: 30.0, contrast: 15.0, vignette_amount: -40.0, grain_amount: 30.0, pixel_scale: 0.5, ..Default::default() },
            CameraRaw { saturation: -30.0, noise_luminance: 30.0, vignette_amount: 25.0, pixel_scale: 0.5, ..Default::default() },
        ];
        for raw in [Some(&raw), None] {
            let cache = std::sync::Mutex::new(ScreenCache::default());
            for (revision, p) in settings.iter().enumerate() {
                let key = ScreenKey { revision: revision as u64 + 1, size, original: false };
                let all = develop_screen(&source, area, raw, &cache, key, whole, p, false, &|| false).unwrap();
                let got = develop_screen(&source, area, raw, &cache, key, part, p, false, &|| false).unwrap();
                assert_eq!(got.len(), (part.width() * part.height()) as usize);
                let mut worst = 0.0f32;
                for y in part.y0..part.y1 {
                    for x in part.x0..part.x1 {
                        let (a, b) = (all[(y * 160 + x) as usize], got[((y - part.y0) * part.width() as i32 + x - part.x0) as usize]);
                        worst = a.iter().zip(&b).map(|(a, b)| (a - b).abs()).fold(worst, f32::max);
                    }
                }
                assert!(worst <= 1e-5, "{p:?} raw {}: {worst}", raw.is_some());
            }
            let c = cache.lock().unwrap();
            assert_eq!(c.levels.len(), 1, "one level, reduced once");
            drop(c);
            // The original is the reduced frame itself; a develop asked to give up stops.
            let original = ScreenKey { revision: 0, size, original: true };
            let reduced = develop_screen(&source, area, raw, &cache, original, part, &settings[0], false, &|| false).unwrap();
            let all = develop_screen(&source, area, raw, &cache, original, whole, &settings[0], false, &|| false).unwrap();
            assert_eq!(reduced[0], all[(part.y0 * 160 + part.x0) as usize]);
            assert!(
                develop_screen(&source, area, raw, &cache, ScreenKey { revision: 9, size, original: false }, part, &settings[0], false, &|| true).is_none()
            );
        }
    }

    /// The levels kept add up to the pixel budget at most; the newest always stays.
    #[test]
    fn the_kept_levels_stay_within_the_pixel_budget() {
        let (source, area) = ramp(64, 48);
        let cache = std::sync::Mutex::new(ScreenCache::default());
        let p = CameraRaw { exposure: 0.4, ..Default::default() };
        for (i, size) in [(16, 12), (32, 24), (8, 6), (48, 36)].into_iter().enumerate() {
            let key = ScreenKey { revision: 1, size, original: false };
            develop_screen(&source, area, None, &cache, key, PixelRect::new(0, 0, 4, 4), &p, false, &|| false).unwrap();
            let c = cache.lock().unwrap();
            assert_eq!(c.levels[0].size, size, "step {i}: the newest first");
            assert!(c.levels.len() <= LEVELS_KEPT);
        }
        assert_eq!(cache.lock().unwrap().levels.iter().map(|l| l.size).collect::<Vec<_>>(), [(48, 36), (8, 6), (32, 24)]);
        // Past the budget the older levels go, the newest (the one shown) stays even alone.
        let level = |w: usize, h: usize| LevelCache { size: (w, h), source: Arc::default(), tonal: TonalCache::default() };
        let mut levels = vec![level(30, 20), level(20, 10), level(10, 10)];
        trim_levels(&mut levels, 850);
        assert_eq!(levels.iter().map(|l| l.size).collect::<Vec<_>>(), [(30, 20), (20, 10)]);
        let mut levels = vec![level(40, 40), level(10, 10)];
        trim_levels(&mut levels, 900);
        assert_eq!(levels.iter().map(|l| l.size).collect::<Vec<_>>(), [(40, 40)]);
    }

    #[test]
    fn develops_wait_for_a_gesture_to_settle_but_not_for_new_settings() {
        let mut preview = DetailPreview::new(Surface::new(PixelFormat::RGBA8), PixelRect::new(0, 0, 8, 8), Coverage::None);
        let ctx = Context::default();
        assert!(preview.settled(&ctx, 1, false), "a new revision or level develops at once");
        assert!(!preview.settled(&ctx, 2, true), "a view that just changed waits");
        assert!(!preview.settled(&ctx, 2, true));
        std::thread::sleep(std::time::Duration::from_millis(SETTLE_MS as u64 + 20));
        assert!(preview.settled(&ctx, 2, true), "and develops once it has stayed");
        assert!(!preview.settled(&ctx, 3, true), "a further change waits again");
    }
}
