//! `cargo run -p photocraft-face --example mesh -- <model.onnx> <raw rgb file> <width> <height> <x> <y> <w> <h> <ex0> <ey0> <ex1> <ey1>`:
//! prints the face mesh inside the given face box (with its eyes) as JSON, for checking against
//! a reference and for drawing the points over the image.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let num = |i: usize| -> Result<f32, Box<dyn std::error::Error>> { Ok(a.get(i).ok_or("missing argument")?.parse()?) };
    let model = std::fs::read(a.get(1).ok_or("model path")?)?;
    let data = std::fs::read(a.get(2).ok_or("rgb file")?)?;
    let w: usize = a.get(3).ok_or("width")?.parse()?;
    let h: usize = a.get(4).ok_or("height")?.parse()?;
    let rect = [num(5)?, num(6)?, num(7)?, num(8)?];
    let eyes = [[num(9)?, num(10)?], [num(11)?, num(12)?]];
    let t = std::time::Instant::now();
    let mesh = photocraft_face::FaceMesh::from_onnx(&model)?;
    let t1 = std::time::Instant::now();
    let roi = photocraft_face::Roi::around(rect, eyes).ok_or("bad box")?;
    let m = mesh.mesh(photocraft_face::Rgb8 { width: w, height: h, data: &data }, roi)?;
    eprintln!("load {:?} mesh {:?} presence {:.3}", t1 - t, t1.elapsed(), m.presence);
    println!("{{\"presence\":{},\"points\":{:?}}}", m.presence, m.points);
    Ok(())
}
