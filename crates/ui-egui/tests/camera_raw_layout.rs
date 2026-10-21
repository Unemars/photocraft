//! The Camera Raw dialog laid out as Photoshop's Camera Raw 16.1.1: panel eyes, value boxes,
//! knob resets, presets, the More menu's resets, the persisted layout and the Before/After
//! views, through the real shell and its control channel.
use egui::{Event, Modifiers, PointerButton, Pos2, accesskit::Role, pos2, vec2};
use egui_kittest::{Harness, kittest::Queryable};
use photocraft_ui_egui::{PhotocraftApp, camera_raw_ui, control, theme::ThemeKind};
use serde_json::{Value, json};

fn menu(app: &mut PhotocraftApp, ctx: &egui::Context, p: Value) -> Result<Value, String> {
    camera_raw_ui::menu(app, ctx, "filter.cameraRaw", &p).ok_or("missing handler")?
}
fn ui(h: &mut Harness<'_, PhotocraftApp>, request: Value) -> Result<Value, String> {
    let ctx = h.ctx.clone();
    let r = menu(h.state_mut(), &ctx, json!({ "ui": request }));
    h.run_steps(3);
    settle(h);
    r
}
/// Waits for the proxy develop the dialog's own controls start on a worker (#5964), so the
/// histogram describes the settings of the moment.
fn settle(h: &mut Harness<'_, PhotocraftApp>) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while inspect(h)["previewPending"] == true {
        assert!(std::time::Instant::now() < deadline, "the Camera Raw preview worker never finished");
        std::thread::sleep(std::time::Duration::from_millis(2));
        h.run_steps(1);
    }
}
fn inspect(h: &Harness<'_, PhotocraftApp>) -> Value {
    control::inspect(h.state(), &h.ctx)["cameraRaw"].clone()
}
fn fixture() -> Harness<'static, PhotocraftApp> {
    fixture_on(false)
}
/// The dialog on a filled layer, or on that layer made a smart object (`smart`), so OK adds a
/// smart filter.
fn fixture_on(smart: bool) -> Harness<'static, PhotocraftApp> {
    // The tests query English labels. The frame loop sets the language from the preference on
    // every frame, so pin the preference, not only the current language.
    photocraft_ui_egui::i18n::set_current(photocraft_ui_egui::i18n::Lang::EN);
    let mut h = Harness::builder().with_step_dt(1.0 / 60.0).with_size(vec2(1632.0, 1046.0)).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, ThemeKind::ProMedium);
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.session.prefs.edit(|p| p.interface.language = "en".into());
        app.set_theme(&cc.egui_ctx, ThemeKind::ProMedium);
        app.run("file.new", json!({"width":64,"height":48})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("edit.fill", json!({"color":"#708090"})).unwrap();
        if smart {
            app.run("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
        }
        camera_raw_ui::open(&mut app, &cc.egui_ctx).unwrap();
        app
    });
    h.run_steps(4);
    h
}
/// Types `text` into the value box of the slider labelled `label`; `commit` presses Enter.
fn type_value(h: &mut Harness<'_, PhotocraftApp>, label: &str, text: &str, commit: bool) {
    let slider = h.get_by_role_and_label(Role::Slider, label).rect();
    // The value box sits above the track's right end.
    click(h, pos2(slider.right() - 24.0, slider.top() - 5.0), 1);
    h.event(Event::Key { key: egui::Key::A, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::COMMAND });
    h.event(Event::Text(text.into()));
    if commit {
        h.event(Event::Key { key: egui::Key::Enter, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
    }
    h.run_steps(4);
}
fn click(h: &mut Harness<'_, PhotocraftApp>, p: Pos2, times: usize) {
    h.event(Event::PointerMoved(p));
    for _ in 0..times {
        for pressed in [true, false] {
            h.event(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
        }
        h.run_steps(1);
    }
    h.run_steps(2);
    settle(h);
}

#[test]
fn a_panel_eye_hides_its_edits_in_the_preview_only() {
    let mut h = fixture();
    let unedited = inspect(&h)["histogram"].clone();
    ui(&mut h, json!({"set":{"exposure":1.0}})).unwrap();
    let edited = inspect(&h)["histogram"].clone();
    assert_ne!(edited["green"], unedited["green"], "exposure changes the preview");
    // The Light panel's eye: the preview shows the image without Light's settings.
    let eye = h.get_by_label("Toggle visibility of Light settings").rect().center();
    click(&mut h, eye, 1);
    let hidden = inspect(&h);
    assert_eq!(hidden["hidden"], json!(["light"]));
    assert_eq!(hidden["histogram"]["green"], unedited["green"], "hidden panel left out of the preview");
    assert_eq!(hidden["params"]["exposure"], 1.0, "hiding never edits the settings");
    click(&mut h, eye, 1);
    assert_eq!(inspect(&h)["hidden"], json!([]));
    assert_eq!(inspect(&h)["histogram"]["green"], edited["green"]);
    // OK applies the settings even while a panel is hidden.
    ui(&mut h, json!({"hidden":["light"]})).unwrap();
    let history = h.state().session.active().unwrap().history.past_len();
    ui(&mut h, json!({"commit":true})).unwrap();
    assert_eq!(h.state().session.active().unwrap().history.past_len(), history + 1);
}

#[test]
fn value_boxes_take_typed_numbers_and_the_knob_resets_on_double_click() {
    let mut h = fixture();
    let slider = h.get_by_role_and_label(Role::Slider, "Contrast").rect();
    // The value box sits above the track's right end.
    let field = pos2(slider.right() - 24.0, slider.top() - 5.0);
    click(&mut h, field, 1);
    h.event(Event::Key { key: egui::Key::A, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::COMMAND });
    h.event(Event::Text("35".into()));
    h.event(Event::Key { key: egui::Key::Enter, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(4);
    assert_eq!(inspect(&h)["params"]["contrast"], 35.0, "typed value");
    // A double-click on the track resets to the default.
    let track = h.get_by_role_and_label(Role::Slider, "Contrast").rect();
    click(&mut h, track.center(), 2);
    assert_eq!(inspect(&h)["params"]["contrast"], 0.0, "double-click resets");
}

#[test]
fn presets_apply_with_an_amount_and_reset_to_open_restores_the_settings() {
    let mut h = fixture();
    ui(&mut h, json!({"layout":{"tool":"presets"}})).unwrap();
    h.get_by_label("Vivid").click();
    h.run_steps(4);
    let p = inspect(&h)["params"].clone();
    assert_eq!(p["vibrance"], 35.0);
    assert_eq!(p["contrast"], 15.0);
    // Back to Edit: More › Reset to Open.
    ui(&mut h, json!({"layout":{"tool":"edit"}})).unwrap();
    let more = h.get_by_label("More image settings").rect().center();
    click(&mut h, more, 1);
    h.get_by_label("Reset to Open").click();
    h.run_steps(4);
    let p = inspect(&h)["params"].clone();
    assert_eq!((p["vibrance"].clone(), p["contrast"].clone()), (json!(0.0), json!(0.0)));
}

#[test]
fn layout_persists_between_openings_and_rejects_unknown_ids() {
    let mut h = fixture();
    ui(&mut h, json!({"layout":{"open":["optics","detail"],"groups":["defringe"],"curve":"point"}})).unwrap();
    assert!(h.query_by_label("Defringe").is_some(), "the open group is shown");
    assert!(h.query_by_label("Exposure").is_none(), "Light was closed");
    ui(&mut h, json!({"cancel":true})).unwrap();
    let ctx = h.ctx.clone();
    camera_raw_ui::open(h.state_mut(), &ctx).unwrap();
    h.run_steps(4);
    let layout = &h.state().ui.camera_raw_layout;
    assert_eq!(layout.open, ["optics", "detail"]);
    assert_eq!(layout.curve, "point");
    for bad in [
        json!({"layout":{"tool":"crop"}}),
        json!({"layout":{"open":["basic"]}}),
        json!({"layout":{"groups":["nope"]}}),
        json!({"layout":{"unknown":1}}),
        json!({"hidden":["light","nope"]}),
        json!({"beforeAfter":"diagonal"}),
        json!({"profileBrowser":"yes"}),
    ] {
        assert!(ui(&mut h, bad.clone()).is_err(), "rejects {bad}");
    }
}

#[test]
fn before_after_views_cycle_and_copy_settings() {
    let mut h = fixture();
    ui(&mut h, json!({"set":{"exposure":1.0}})).unwrap();
    for (key, want) in [("leftRight", "leftRight"), ("topBottomSplit", "topBottomSplit"), ("off", "off")] {
        ui(&mut h, json!({"beforeAfter": key})).unwrap();
        assert_eq!(inspect(&h)["beforeAfter"], want);
    }
    // Q cycles the views.
    h.key_press(egui::Key::Q);
    h.run_steps(3);
    assert_eq!(inspect(&h)["beforeAfter"], "leftRight");
    // Copy After settings to Before, then Swap: the current settings become the Before ones.
    h.get_by_label("Copy After settings to Before (Ctrl+Alt+P)").click();
    h.run_steps(3);
    h.get_by_label("Swap Before and After settings (Ctrl+Alt+Shift+P)").click();
    h.run_steps(3);
    assert_eq!(inspect(&h)["params"]["exposure"], 1.0, "swapping equal settings keeps them");
    h.key_press(egui::Key::Q);
    h.run_steps(3);
    assert_eq!(inspect(&h)["beforeAfter"], "leftRightSplit");
}

#[test]
fn every_panel_and_tool_draws_without_editing_the_document() {
    let mut h = fixture();
    let doc = h.state().session.active().unwrap().doc.clone();
    let before = inspect(&h)["params"].clone();
    let groups = ["vignette", "grain", "sharpening", "noiseReduction", "colorNoiseReduction", "lensVignette", "defringe", "manualTransformations"];
    for panel in ["light", "color", "effects", "curve", "mixer", "grading", "detail", "optics", "geometry", "calibration"] {
        ui(&mut h, json!({"layout":{"tool":"edit","open":[panel],"groups":groups}})).unwrap();
        h.run_steps(3);
    }
    for (curve, mixer, adjust, tab, grading) in
        [("point", "pointColor", "hsl", "all", "shadows"), ("red", "mixer", "color", "hue", "global"), ("blue", "mixer", "hsl", "all", "threeWay")]
    {
        ui(&mut h, json!({"layout":{"open":["curve","mixer","grading"],"curve":curve,"mixer":mixer,"mixerAdjust":adjust,"mixerTab":tab,"grading":grading}}))
            .unwrap();
    }
    for tool in ["remove", "masking", "redEye", "presets", "edit"] {
        ui(&mut h, json!({"layout":{"tool":tool}})).unwrap();
    }
    ui(&mut h, json!({"profileBrowser":true})).unwrap();
    ui(&mut h, json!({"profileBrowser":false})).unwrap();
    assert_eq!(inspect(&h)["params"], before);
    assert!(std::sync::Arc::ptr_eq(&doc, &h.state().session.active().unwrap().doc));
}

/// A click focuses a slider; then the arrow keys and the wheel over it step its value (Shift: ten),
/// the wheel elsewhere does not, and Tab moves to the next slider.
#[test]
fn a_focused_slider_steps_with_the_keys_and_the_wheel() {
    let mut h = fixture();
    // Enough open panels for the settings to scroll.
    ui(&mut h, json!({"layout":{"open":["light","color","effects","detail","optics","geometry"]}})).unwrap();
    let key = |h: &mut Harness<'_, PhotocraftApp>, key: egui::Key, modifiers: Modifiers| {
        h.event(Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers });
        h.event(Event::Key { key, physical_key: None, pressed: false, repeat: false, modifiers });
        h.run_steps(2);
    };
    let wheel = |h: &mut Harness<'_, PhotocraftApp>, at: Pos2, notches: f32| {
        h.event(Event::PointerMoved(at));
        h.event(Event::MouseWheel { unit: egui::MouseWheelUnit::Line, delta: vec2(0.0, notches), modifiers: Modifiers::NONE, phase: egui::TouchPhase::Move });
        h.run_steps(2);
    };
    let contrast = h.get_by_role_and_label(Role::Slider, "Contrast").rect();
    click(&mut h, contrast.center(), 1);
    let start = inspect(&h)["params"]["contrast"].as_f64().unwrap();
    for _ in 0..3 {
        key(&mut h, egui::Key::ArrowUp, Modifiers::NONE);
    }
    assert_eq!(inspect(&h)["params"]["contrast"].as_f64().unwrap(), start + 3.0);
    key(&mut h, egui::Key::ArrowRight, Modifiers::SHIFT);
    assert_eq!(inspect(&h)["params"]["contrast"].as_f64().unwrap(), start + 13.0);
    key(&mut h, egui::Key::ArrowLeft, Modifiers::NONE);
    assert_eq!(inspect(&h)["params"]["contrast"].as_f64().unwrap(), start + 12.0);
    let row = contrast.center() - vec2(0.0, 12.0);
    wheel(&mut h, row, 2.0);
    assert_eq!(inspect(&h)["params"]["contrast"].as_f64().unwrap(), start + 14.0, "the wheel over the row");
    wheel(&mut h, row, -4.0);
    assert_eq!(inspect(&h)["params"]["contrast"].as_f64().unwrap(), start + 10.0);
    // egui spreads a notch over several frames: none of it scrolls the panels under the slider.
    h.run_steps(30);
    assert_eq!(h.get_by_role_and_label(Role::Slider, "Contrast").rect(), contrast, "the panels did not scroll");
    // Elsewhere over the panels the wheel scrolls them and leaves the focused slider alone.
    wheel(&mut h, row + vec2(0.0, 120.0), -3.0);
    h.run_steps(30);
    assert_eq!(inspect(&h)["params"]["contrast"].as_f64().unwrap(), start + 10.0, "not the wheel elsewhere");
    assert!(h.get_by_role_and_label(Role::Slider, "Contrast").rect().top() < contrast.top() - 10.0, "the panels scrolled");
    key(&mut h, egui::Key::ArrowUp, Modifiers::NONE);
    assert_eq!(inspect(&h)["params"]["contrast"].as_f64().unwrap(), start + 11.0, "still focused");
    // The pointer leaving the panels takes the focus away: the keys and the wheel are free again.
    let preview = pos2(300.0, 400.0);
    wheel(&mut h, preview, 3.0);
    key(&mut h, egui::Key::ArrowUp, Modifiers::NONE);
    assert_eq!(inspect(&h)["params"]["contrast"].as_f64().unwrap(), start + 11.0, "unfocused");
    // Tab: the next slider (Highlights) takes the keys.
    // A click on the label focuses the slider again without moving it.
    let contrast = h.get_by_role_and_label(Role::Slider, "Contrast").rect();
    click(&mut h, contrast.center() - vec2(0.0, 13.0), 1);
    assert_eq!(inspect(&h)["params"]["contrast"].as_f64().unwrap(), start + 11.0);
    key(&mut h, egui::Key::Tab, Modifiers::NONE);
    key(&mut h, egui::Key::ArrowDown, Modifiers::NONE);
    let p = inspect(&h)["params"].clone();
    assert_eq!(p["contrast"].as_f64().unwrap(), start + 11.0);
    assert_eq!(p["highlights"], -1.0, "{p}");
    // Exposure steps by 0.05.
    let exposure = h.get_by_role_and_label(Role::Slider, "Exposure").rect();
    click(&mut h, exposure.center(), 1);
    let e0 = inspect(&h)["params"]["exposure"].as_f64().unwrap();
    key(&mut h, egui::Key::ArrowUp, Modifiers::NONE);
    assert!((inspect(&h)["params"]["exposure"].as_f64().unwrap() - (e0 + 0.05)).abs() < 1e-6);
}

/// The arrow keys and the wheel on a focused slider are undoable: a run of them is one step
/// (#3817 review: they changed the settings without an undo step).
#[test]
fn a_run_of_slider_nudges_is_one_undo_step() {
    let mut h = fixture();
    let key = |h: &mut Harness<'_, PhotocraftApp>, key: egui::Key, modifiers: Modifiers| {
        h.event(Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers });
        h.event(Event::Key { key, physical_key: None, pressed: false, repeat: false, modifiers });
        h.run_steps(2);
    };
    let rect = h.get_by_role_and_label(Role::Slider, "Contrast").rect();
    click(&mut h, rect.center(), 1);
    let start = inspect(&h)["params"]["contrast"].as_f64().unwrap();
    for _ in 0..3 {
        key(&mut h, egui::Key::ArrowUp, Modifiers::NONE);
    }
    assert_eq!(inspect(&h)["params"]["contrast"].as_f64().unwrap(), start + 3.0);
    key(&mut h, egui::Key::Z, Modifiers::COMMAND);
    assert_eq!(inspect(&h)["params"]["contrast"].as_f64().unwrap(), start, "one Ctrl+Z undoes the run");
    key(&mut h, egui::Key::Z, Modifiers::COMMAND | Modifiers::SHIFT);
    assert_eq!(inspect(&h)["params"]["contrast"].as_f64().unwrap(), start + 3.0, "and redo brings it back");
    // A later turn of the wheel over the row is a step of its own.
    h.run_steps(90);
    let row = rect.center() - vec2(0.0, 12.0);
    h.event(Event::PointerMoved(row));
    h.event(Event::MouseWheel { unit: egui::MouseWheelUnit::Line, delta: vec2(0.0, 2.0), modifiers: Modifiers::NONE, phase: egui::TouchPhase::Move });
    h.run_steps(2);
    assert_eq!(inspect(&h)["params"]["contrast"].as_f64().unwrap(), start + 5.0);
    key(&mut h, egui::Key::Z, Modifiers::COMMAND);
    assert_eq!(inspect(&h)["params"]["contrast"].as_f64().unwrap(), start + 3.0);
}

/// Esc first closes what is open over the dialog (the More menu, the zoom menu, Preferences);
/// only Esc on the bare dialog cancels it.
#[test]
fn esc_closes_an_open_menu_before_it_cancels_the_dialog() {
    let mut h = fixture();
    let more = h.get_by_label("More image settings").rect().center();
    click(&mut h, more, 1);
    assert!(h.query_by_label("Reset to Open").is_some(), "the More menu is open");
    h.key_press(egui::Key::Escape);
    h.run_steps(3);
    assert!(h.query_by_label("Reset to Open").is_none(), "Esc closed the menu");
    assert!(!inspect(&h).is_null(), "and left the dialog open");
    // The zoom menu, right of the 100% tab.
    let tab = h.get_by_label("100%").rect();
    click(&mut h, pos2(tab.right() + 11.0, tab.center().y), 1);
    assert!(h.query_by_label("1600%").is_some(), "the zoom menu is open");
    h.key_press(egui::Key::Escape);
    h.run_steps(3);
    assert!(h.query_by_label("1600%").is_none());
    assert!(!inspect(&h).is_null(), "Esc in the zoom menu keeps the dialog");
    h.key_press_modifiers(Modifiers::COMMAND, egui::Key::K);
    h.run_steps(3);
    h.key_press(egui::Key::Escape);
    h.run_steps(3);
    assert!(!inspect(&h).is_null(), "Esc in Preferences keeps the dialog");
    h.key_press(egui::Key::Escape);
    h.run_steps(3);
    assert!(inspect(&h).is_null(), "Esc on the bare dialog cancels");
}

/// The Before/After bar's shortcuts carry extra modifiers on P, so they are matched before
/// Shift+P (Presets) and P (the Before toggle).
#[test]
fn before_after_shortcuts_come_before_presets_and_the_before_toggle() {
    let mut h = fixture();
    ui(&mut h, json!({"set":{"exposure":1.0},"beforeAfter":"leftRight"})).unwrap();
    // Ctrl+Alt+P: After to Before; Alt+Shift+P: Before (now the same) to After.
    h.key_press_modifiers(Modifiers::COMMAND | Modifiers::ALT, egui::Key::P);
    h.run_steps(3);
    ui(&mut h, json!({"set":{"exposure":0.0}})).unwrap();
    h.key_press_modifiers(Modifiers::ALT | Modifiers::SHIFT, egui::Key::P);
    h.run_steps(3);
    let view = inspect(&h);
    assert_eq!(view["params"]["exposure"], 1.0, "Alt+Shift+P copied the Before settings to After");
    assert_eq!(h.state().ui.camera_raw_layout.tool, "edit", "Alt+Shift+P is not Shift+P (Presets)");
    assert_eq!(view["before"], false, "nor P (the Before toggle)");
    // Ctrl+Alt+Shift+P swaps.
    ui(&mut h, json!({"set":{"exposure":-1.0}})).unwrap();
    h.key_press_modifiers(Modifiers::COMMAND | Modifiers::ALT | Modifiers::SHIFT, egui::Key::P);
    h.run_steps(3);
    assert_eq!(inspect(&h)["params"]["exposure"], 1.0, "the swap brought the Before settings");
    h.key_press_modifiers(Modifiers::SHIFT, egui::Key::P);
    h.run_steps(3);
    assert_eq!(h.state().ui.camera_raw_layout.tool, "presets");
}

/// The Color Mixer's Color rows are shared by the eight colours: text typed into a box and not
/// committed never lands in the colour picked next, and an open box doesn't move the rows below.
#[test]
fn a_typed_value_stays_with_its_colour_and_the_rows_keep_their_place() {
    let mut h = fixture();
    ui(&mut h, json!({"layout":{"open":["mixer"],"mixer":"mixer","mixerAdjust":"color"}})).unwrap();
    let saturation = h.get_by_role_and_label(Role::Slider, "Saturation").rect();
    type_value(&mut h, "Hue", "40", false);
    assert_eq!(h.get_by_role_and_label(Role::Slider, "Saturation").rect(), saturation, "the open box leaves the next row in place");
    h.get_by_label("Oranges").click();
    h.run_steps(4);
    assert_eq!(h.state().ui.camera_raw_layout.mixer_color, 1);
    let p = inspect(&h)["params"].clone();
    assert_eq!((p["hslHue"][0].clone(), p["hslHue"][1].clone()), (json!(0.0), json!(0.0)), "nothing typed was committed");
    type_value(&mut h, "Hue", "25", true);
    assert_eq!(inspect(&h)["params"]["hslHue"][1], 25.0);
}

/// Temperature and Tint typed in the dialog keep the smart filter exportable: it is written to
/// the PSD with those settings, not left out of it.
#[test]
fn white_balance_edits_from_the_dialog_export_to_psd() {
    let mut h = fixture_on(true);
    type_value(&mut h, "Temperature", "-26", true);
    type_value(&mut h, "Tint", "18", true);
    ui(&mut h, json!({"commit":true})).unwrap();
    let doc = h.state().session.active().unwrap().doc.clone();
    let out = photocraft_io::export(&doc, "psd", &Default::default()).unwrap();
    assert!(out.warnings.is_empty(), "{:?}", out.warnings);
    let back = photocraft_io::import("white-balance.psd", &out.bytes).unwrap().document;
    let filters = back
        .walk()
        .into_iter()
        .find_map(|(_, _, l)| match &l.content {
            photocraft_doc::LayerContent::Smart(sm) => Some(sm.smart_filters.clone()),
            _ => None,
        })
        .expect("a smart object");
    assert_eq!(filters.len(), 1);
    assert_eq!(filters[0].command, "filter.cameraRaw");
    let params = photocraft_engine::lens_cmds::raw_params("filter.cameraRaw", &filters[0].params).unwrap();
    assert_eq!((params.temperature, params.tint), (-26.0, 18.0));
}

/// #5691 reported that a double-click on the Highlights knob resets it only sometimes. One cause:
/// egui counts a click within 0.6 s of two earlier ones as a triple click, so a double-click
/// soon after another click on the slider did nothing. Repeated quick resets all work.
#[test]
fn a_double_click_on_the_highlights_knob_resets_it_every_time() {
    let mut h = fixture();
    for _ in 0..3 {
        ui(&mut h, json!({"set":{"highlights":-60.0}})).unwrap();
        let hit = h.get_by_role_and_label(Role::Slider, "Highlights").rect();
        // The track runs 6 pt inside the hit area on each side; -60 of -100..100 is a fifth along.
        let knob = pos2(hit.left() + 6.0 + 0.2 * (hit.width() - 12.0), hit.center().y);
        click(&mut h, knob, 2);
        assert_eq!(inspect(&h)["params"]["highlights"], 0.0);
    }
}
