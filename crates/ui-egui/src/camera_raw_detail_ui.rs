//! Full-resolution refinement of Camera Raw's interactive proxy. Native work is off the
//! UI thread; only one revision runs at a time, and obsolete results never replace new edits.
use egui::{Color32, ColorImage, Context, Pos2, Rect, TextureHandle, TextureOptions, pos2, vec2};
use photocraft_algo::camera_raw::CameraRaw;
use photocraft_geom::Rect as PixelRect;
use photocraft_raster::Surface;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{Receiver, TryRecvError},
};

#[derive(Clone)]
pub(crate) enum Coverage {
    Selection(Surface),
    Mask(photocraft_doc::LayerMask),
    None,
}

type DetailResult = Result<(u64, Surface), String>;

/// What a screen-size develop is for: the settings' revision (0 for the original), the size and
/// whether it is the original (Before, or every setting at its default).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScreenKey {
    revision: u64,
    size: (usize, usize),
    original: bool,
}

/// Pixels (straight RGBA, row-major) shared between the dialog and its workers.
type Pixels = Arc<Vec<[f32; 4]>>;

type ScreenResult = Result<(ScreenKey, Pixels), String>;

/// Largest frame (pixels) the screen-size preview develops whole: about a 4K screen's preview
/// area twice over. Past it (a large frame just under 100 %), the preview develops the full
/// frame and shows the visible part, as at 1:1, within that path's limits.
const SCREEN_MAX_PIXELS: usize = 12_000_000;

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

/// The worker's caches for screen-size develops of one size: the frame reduced to it, and its
/// tonal stages.
#[derive(Default)]
struct ScreenCache {
    source: Option<((usize, usize), Pixels)>,
    tonal: Option<((usize, usize), TonalCache)>,
}

/// What the last paint showed, for the clipping overlay: a crop of the full-size pixels (the key
/// [`DetailPreview::crop`] holds), or the whole frame at a screen size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shown {
    Crop,
    Screen(ScreenKey),
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
    /// Below 1:1 the preview develops the whole frame at the size it has on screen instead.
    screen_result: Option<(ScreenKey, Pixels)>,
    screen_pending: Option<(ScreenKey, Receiver<ScreenResult>)>,
    /// A screen-size develop that failed: not asked for again.
    screen_failed: Option<ScreenKey>,
    screen_texture: Option<((ScreenKey, u64), TextureHandle)>,
    screen_cache: Arc<std::sync::Mutex<ScreenCache>>,
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
            screen_failed: None,
            screen_texture: None,
            screen_cache: Arc::default(),
        }
    }

    /// The raw the open-time dialog develops: everything made for another one is dropped.
    pub(crate) fn set_raw(&mut self, raw: Option<Arc<photocraft_io::raw::RawPreview>>) {
        self.raw = raw;
        (self.result, self.crop, self.failed_revision, self.shown) = (None, None, None, None);
        (self.screen_result, self.screen_failed, self.screen_texture) = (None, None, None);
        // A worker still running keeps the old cache to itself.
        self.screen_cache = Arc::default();
        self.screen_pending = None;
        self.pending = None;
    }

    /// Bench only: what the screen-size worker develops for `params` at `size`, now (through its caches).
    #[cfg(test)]
    pub(crate) fn develop_screen_now(&self, params: &CameraRaw, size: (usize, usize), revision: u64) -> Pixels {
        let mut params = params.clone();
        params.pixel_scale = (size.0 as f32 / self.area.width().max(1) as f32).min(1.0);
        let key = ScreenKey { revision, size, original: false };
        develop_screen(&self.source, self.area, self.raw.as_deref(), &self.screen_cache, key, &params, false)
    }

    /// Bench only: what the background worker develops for `params`, now.
    #[cfg(test)]
    pub(crate) fn develop_now(&self, params: &CameraRaw) -> Surface {
        let mut params = params.clone();
        params.pixel_scale = 1.0;
        develop(&self.source, self.raw.as_deref(), self.area, &params, &self.coverage)
    }

    /// The size of the screen-size develop the preview shows (`None`: the proxy or 1:1 pixels).
    pub(crate) fn screen_size(&self) -> Option<(usize, usize)> {
        match self.shown {
            Some(Shown::Screen(key)) if self.ready => Some(key.size),
            _ => None,
        }
    }

    pub(crate) fn pending(&self) -> bool {
        self.pending.is_some() || self.screen_pending.is_some()
    }

    fn poll_screen(&mut self) {
        let Some((key, rx)) = &self.screen_pending else { return };
        let key = *key;
        let result = match rx.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => Err("Camera Raw preview worker stopped".into()),
        };
        self.screen_pending = None;
        match result {
            Ok(done) => {
                self.screen_result = Some(done);
                self.error = None;
            }
            Err(e) => {
                self.error = Some(e);
                self.screen_failed = Some(key);
            }
        }
    }

    /// Develops the frame at `key.size` in the background (one at a time; a newer request waits
    /// for the running one and then replaces it).
    fn request_screen(&mut self, ctx: &Context, key: ScreenKey, params: &CameraRaw) {
        if self.screen_pending.is_some() || self.screen_failed == Some(key) {
            return;
        }
        let (source, area, raw, cache, cancel, ctx) =
            (self.source.clone(), self.area, self.raw.clone(), self.screen_cache.clone(), self.cancel.clone(), ctx.clone());
        let mut params = params.clone();
        params.pixel_scale = (key.size.0 as f32 / area.width().max(1) as f32).min(1.0);
        let float = source.format().sample == photocraft_color::SampleType::F32;
        let (tx, rx) = std::sync::mpsc::channel::<ScreenResult>();
        let work = move || {
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| develop_screen(&source, area, raw.as_deref(), &cache, key, &params, float)))
                .map(|px| (key, px))
                .map_err(|_| "Camera Raw preview failed; the proxy remains available".to_string());
            if !cancel.load(Ordering::Relaxed) {
                let _ = tx.send(result);
                ctx.request_repaint();
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            match std::thread::Builder::new().name("camera-raw-screen".into()).spawn(work) {
                Ok(_) => self.screen_pending = Some((key, rx)),
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

    /// The size the whole frame has on screen at `scale` (screen pixels per image pixel), when
    /// the screen-size preview takes it: below 1:1, within the device's texture side and
    /// [`SCREEN_MAX_PIXELS`]. `None`: the full-size path's crop instead.
    fn screen_size_for(&self, scale: f32, max_side: usize) -> Option<(usize, usize)> {
        if scale.is_nan() || scale >= 0.99 || !matches!(self.coverage, Coverage::None) {
            return None;
        }
        let size = |n: u32| ((n as f32 * scale).round() as usize).clamp(1, n.max(1) as usize);
        let (w, h) = (size(self.area.width()), size(self.area.height()));
        (w <= max_side && h <= max_side && w.saturating_mul(h) <= SCREEN_MAX_PIXELS).then_some((w, h))
    }

    /// Below 1:1: the frame developed at its size on screen, sharp where the proxy is soft and a
    /// fraction of the full frame's work.
    #[allow(clippy::too_many_arguments)]
    fn paint_screen(
        &mut self,
        ui: &egui::Ui,
        image: Rect,
        revision: u64,
        before: bool,
        params: &CameraRaw,
        size: (usize, usize),
        display: &photocraft_engine::display_color::CanvasDisplay,
    ) -> bool {
        let original = before || params.is_identity();
        let key = ScreenKey { revision: if original { 0 } else { revision }, size, original };
        if self.screen_result.as_ref().is_none_or(|(k, _)| *k != key) {
            self.request_screen(ui.ctx(), key, params);
            self.poll_screen();
        }
        let Some((done, px)) = &self.screen_result else { return false };
        if *done != key {
            return false;
        }
        let shown = (key, display.key);
        if self.screen_texture.as_ref().is_none_or(|(k, _)| *k != shown) {
            let color = display_image(PixelRect::new(0, 0, size.0 as i32, size.1 as i32), display, |r, out| {
                let start = (r.y0.max(0) as usize).saturating_mul(size.0);
                if let Some(src) = px.get(start..start.saturating_add(out.len())) {
                    out.copy_from_slice(src);
                }
            });
            match &mut self.screen_texture {
                Some((k, t)) => {
                    t.set(color, TextureOptions::LINEAR);
                    *k = shown;
                }
                None => self.screen_texture = Some((shown, ui.ctx().load_texture("cr-screen-preview", color, TextureOptions::LINEAR))),
            }
        }
        if let Some((_, texture)) = &self.screen_texture {
            ui.painter().image(texture.id(), image, Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
            self.ready = true;
            self.shown = Some(Shown::Screen(key));
        }
        true
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
        // Smaller than 1:1 on screen: develop the frame at the screen's size, not the full frame.
        let max_side = ui.input(|i| i.max_texture_side);
        let scale = image.width() * ui.ctx().pixels_per_point() / self.area.width().max(1) as f32;
        if let Some(size) = self.screen_size_for(scale, max_side) {
            return self.paint_screen(ui, image, revision, before, params, size, display);
        }
        let Some(crop) = visible_pixels(image, ui.clip_rect(), self.area, ui.input(|i| i.max_texture_side)) else { return false };
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
            return false;
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
            let scale = image.size() / vec2(self.area.width() as f32, self.area.height() as f32);
            let screen = Rect::from_min_size(
                image.min + vec2((crop.x0 - self.area.x0) as f32, (crop.y0 - self.area.y0) as f32) * scale,
                vec2(crop.width() as f32, crop.height() as f32) * scale,
            );
            ui.painter().image(texture.id(), screen, Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
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
            Some(Shown::Screen(key)) => {
                let Some((done, px)) = self.screen_result.as_ref().filter(|(k, _)| *k == key) else { return false };
                let overlay_key = (Some(*done), key.revision, key.original, PixelRect::new(0, 0, key.size.0 as i32, key.size.1 as i32), mode, t.kind);
                if self.overlay_key != Some(overlay_key) {
                    let color = ColorImage::new([key.size.0, key.size.1], px.iter().map(clip_color).collect());
                    self.set_overlay(ui, color);
                    self.overlay_key = Some(overlay_key);
                }
                image
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
                let scale = image.size() / vec2(self.area.width() as f32, self.area.height() as f32);
                Rect::from_min_size(
                    image.min + vec2((crop.x0 - self.area.x0) as f32, (crop.y0 - self.area.y0) as f32) * scale,
                    vec2(crop.width() as f32, crop.height() as f32) * scale,
                )
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

/// `area` of `source` developed at `key.size` with `params` (whose `pixel_scale` is that
/// size's), through `cache`.
fn develop_screen(
    source: &Surface,
    area: PixelRect,
    raw: Option<&photocraft_io::raw::RawPreview>,
    cache: &std::sync::Mutex<ScreenCache>,
    key: ScreenKey,
    params: &CameraRaw,
    float: bool,
) -> Pixels {
    let (w, h) = key.size;
    let cached = cache.lock().ok().and_then(|c| c.source.as_ref().filter(|(s, _)| *s == key.size).map(|(_, px)| px.clone()));
    let reduced = cached.unwrap_or_else(|| {
        let px = Arc::new(reduce(source, area, w, h));
        if let Ok(mut c) = cache.lock() {
            // One size at a time: a new one replaces the frame and its stages.
            *c = ScreenCache { source: Some((key.size, px.clone())), tonal: None };
        }
        px
    });
    if key.original {
        return reduced;
    }
    let mut tonal = cache.lock().ok().and_then(|c| c.tonal.as_ref().filter(|(s, _)| *s == key.size).map(|(_, t)| t.clone())).unwrap_or_default();
    let (colour, base) = tonal_stage(&reduced, w, h, params, raw, &mut tonal);
    if let Ok(mut c) = cache.lock() {
        c.tonal = Some((key.size, tonal));
    }
    let mut px = base.to_vec();
    photocraft_algo::camera_raw::develop_colour(&mut px, w, h, &colour, float);
    Arc::new(px)
}

/// `area` of `source` reduced to `tw × th` (each output pixel the mean of the source pixels it
/// covers, by area).
fn reduce(source: &Surface, area: PixelRect, tw: usize, th: usize) -> Vec<[f32; 4]> {
    let (w, h) = (area.width() as usize, area.height() as usize);
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
            let y = area.y0 + iy as i32;
            source.read_rgba_into(PixelRect::new(area.x0, y, area.x1, y + 1), &mut row);
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
            let mut data = source.read_region(area);
            for (out, q) in data.chunks_exact_mut(n).zip(&px) {
                let alpha = if fmt.alpha { out.get(n - 1).copied() } else { None };
                photocraft_raster::from_rgba_into(&fmt, *q, out);
                if let (Some(a), Some(o)) = (alpha, out.get_mut(n - 1)) {
                    *o = a;
                }
            }
            let mut out = source.clone();
            out.write_region(area, &data);
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

    /// Below 1:1 the whole frame is developed at the screen's size only within the device's
    /// texture side and [`SCREEN_MAX_PIXELS`]; past them (a panorama just under 100 %), and with a
    /// selection or mask, the full-size path's crop takes over (#3817 review).
    #[test]
    fn the_screen_size_preview_stays_within_the_texture_side_and_its_pixel_budget() {
        let (source, area) = ramp(20_000, 4_000);
        let d = DetailPreview::new(source, area, Coverage::None);
        assert_eq!(d.screen_size_for(0.1, 8192), Some((2000, 400)));
        assert_eq!(d.screen_size_for(0.5, 8192), None, "10000 px wide is past the texture side");
        assert_eq!(d.screen_size_for(0.35, 8192), Some((7000, 1400)));
        assert_eq!(d.screen_size_for(0.35, 4096), None);
        assert_eq!(d.screen_size_for(0.99, 1 << 16), None, "1:1 and above");
        let (source, area) = ramp(8_000, 6_000);
        let d = DetailPreview::new(source, area, Coverage::None);
        assert_eq!(d.screen_size_for(0.6, 16_384), None, "4800 × 3600 is past the pixel budget");
        assert_eq!(d.screen_size_for(0.5, 16_384), Some((4000, 3000)));
        let (source, area) = ramp(800, 600);
        let mask = Surface::with_default(PixelFormat::GRAY8, &[1.0]);
        assert_eq!(DetailPreview::new(source, area, Coverage::Selection(mask)).screen_size_for(0.5, 8192), None);
    }

    /// A failed screen-size develop is reported once and not asked for again every frame; a
    /// later success takes the error down (#3817 review).
    #[test]
    fn a_failed_screen_develop_is_not_retried_and_a_later_frame_clears_its_error() {
        let (source, area) = ramp(64, 48);
        let mut d = DetailPreview::new(source, area, Coverage::None);
        let key = ScreenKey { revision: 3, size: (32, 24), original: false };
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(Err("Camera Raw preview failed; the proxy remains available".to_string())).unwrap();
        d.screen_pending = Some((key, rx));
        d.poll_screen();
        assert!(d.error.is_some() && d.screen_failed == Some(key) && d.screen_pending.is_none());
        d.request_screen(&Context::default(), key, &CameraRaw { exposure: 0.5, ..Default::default() });
        assert!(d.screen_pending.is_none(), "the failed key is not asked for again");
        let next = ScreenKey { revision: 4, ..key };
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(Ok((next, Arc::new(vec![[0.5; 4]; 32 * 24])))).unwrap();
        d.screen_pending = Some((next, rx));
        d.poll_screen();
        assert_eq!(d.error, None);
    }

    /// After 1:1, a Fit view's clipping overlay is made from the screen-size frame it shows, not
    /// from the last 1:1 crop (#3817 review: it covered the old 1:1 rectangle).
    #[test]
    fn the_clipping_overlay_follows_the_screen_size_frame() {
        let (source, area) = ramp(400, 300);
        let mut d = DetailPreview::new(source, area, Coverage::None);
        let ctx = Context::default();
        let display = sdisplay();
        let params = CameraRaw { exposure: 1.5, ..Default::default() };
        let mode = crate::camera_raw_scope_ui::ClippingMode::Warnings(3);
        let frame = |d: &mut DetailPreview, image: Rect| {
            let mut painted = false;
            for _ in 0..2000 {
                ctx.run_ui(egui::RawInput::default(), |ui| {
                    painted = d.paint(ui, image, 1, false, &params, true, &display) && d.ready && d.overlay(ui, image, 1, mode);
                })
                .textures_delta
                .clear();
                if painted {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            panic!("the detail preview never painted");
        };
        // 1:1: a crop of the full-size result.
        frame(&mut d, Rect::from_min_size(Pos2::ZERO, vec2(400.0, 300.0)));
        assert!(matches!(d.shown, Some(Shown::Crop)) && d.overlay_key.is_some_and(|k| k.0.is_none()));
        // Fit at half size: the screen-size frame, and its overlay.
        frame(&mut d, Rect::from_min_size(Pos2::ZERO, vec2(200.0, 150.0)));
        assert_eq!(d.screen_size(), Some((200, 150)));
        let key = d.overlay_key.unwrap();
        assert_eq!(key.0.map(|k| k.size), Some((200, 150)));
        assert_eq!(key.3, PixelRect::new(0, 0, 200, 150));
    }

    /// A new raw for the open-time dialog drops what was developed for the old one.
    #[test]
    fn a_new_raw_drops_the_screen_size_frames() {
        let (source, area) = ramp(64, 48);
        let mut d = DetailPreview::new(source, area, Coverage::None);
        let key = ScreenKey { revision: 1, size: (32, 24), original: false };
        d.screen_result = Some((key, Arc::new(vec![[0.5; 4]; 32 * 24])));
        if let Ok(mut c) = d.screen_cache.lock() {
            c.source = Some(((32, 24), Arc::new(vec![[0.5; 4]; 32 * 24])));
        }
        d.set_raw(Some(Arc::new(photocraft_io::raw::RawPreview::with_curve(vec![[0.5, 0.6]]))));
        assert!(d.screen_result.is_none());
        assert!(d.screen_cache.lock().unwrap().source.is_none());
    }
}
