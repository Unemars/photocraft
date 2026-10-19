//! Runs the Camera Raw Filter pipeline on a raw float image for the black-box comparisons:
//! `camera_raw_apply <in.f32> <width> <height> <out.f32> '<settings json>'` — RGB f32 triples,
//! sRGB-encoded (encodingGamma 0) unless the settings say otherwise.
use photocraft_algo::camera_raw::{CameraRaw, develop};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (w, h): (usize, usize) = (a[2].parse().unwrap(), a[3].parse().unwrap());
    let bytes = std::fs::read(&a[1]).unwrap();
    let vals: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    assert_eq!(vals.len(), w * h * 3);
    let mut px: Vec<[f32; 4]> = vals.as_chunks::<3>().0.iter().map(|c| [c[0], c[1], c[2], 1.0]).collect();
    let p: CameraRaw = serde_json::from_str(&a[5]).unwrap();
    let t = std::time::Instant::now();
    develop(&mut px, w, h, &p, false);
    eprintln!("develop {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
    let out: Vec<u8> = px.iter().flat_map(|q| q[..3].iter().flat_map(|v| v.to_le_bytes())).collect();
    std::fs::write(&a[4], out).unwrap();
}
