use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn normalize(self) -> Self {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len > 1e-12 {
            Self::new(self.x / len, self.y / len, self.z / len)
        } else {
            Self::new(0.0, 0.0, 1.0)
        }
    }
}

pub struct Heightmap {
    pub width: usize,
    pub height: usize,
    /// Row-major Z heights in millimeters (length = width * height)
    pub z_data: Vec<f32>,
}

impl Heightmap {
    pub fn new(width: usize, height: usize, z_data: Vec<f32>) -> Self {
        assert_eq!(
            width * height,
            z_data.len(),
            "Grid dimensions must match data length"
        );
        assert!(width >= 2 && height >= 2, "Heightmap must be at least 2x2");
        Self {
            width,
            height,
            z_data,
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.z_data[y * self.width + x]
    }
}

/// Writes a single binary STL triangle (50 bytes) with right-hand rule normal.
fn write_triangle<W: Write>(writer: &mut W, a: Vec3, b: Vec3, c: Vec3) -> io::Result<()> {
    let normal = b.sub(a).cross(c.sub(a)).normalize();

    for v in [normal, a, b, c] {
        writer.write_all(&v.x.to_le_bytes())?;
        writer.write_all(&v.y.to_le_bytes())?;
        writer.write_all(&v.z.to_le_bytes())?;
    }
    // 2-byte attribute count (always 0 in standard binary STL)
    writer.write_all(&0u16.to_le_bytes())?;
    Ok(())
}

/// Exports a Heightmap to a watertight binary STL file.
///
/// * `step_mm`: Physical X/Y spacing between grid points in millimeters (e.g., 0.15).
/// * `base_thickness_mm`: Minimum solid material thickness beneath Z=0 of the relief.
pub fn export_heightmap_to_stl<P: AsRef<Path>>(
    map: &Heightmap,
    path: P,
    step_mm: f32,
    base_thickness_mm: f32,
) -> io::Result<()> {
    let w = map.width;
    let h = map.height;

    let top_tris = 2 * (w - 1) * (h - 1);
    let wall_tris = 4 * (w - 1) + 4 * (h - 1);
    let bottom_tris = 2;
    let total_tris = (top_tris + wall_tris + bottom_tris) as u32;

    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(64 * 1024, file);

    // 1. Write 80-byte binary STL header
    let mut header = [0u8; 80];
    let label = b"Rust CNC Relief Binary STL";
    header[..label.len()].copy_from_slice(label);
    writer.write_all(&header)?;

    // 2. Write total triangle count (u32 little-endian)
    writer.write_all(&total_tris.to_le_bytes())?;

    // Helper to convert grid (x, y) to 3D top-surface coordinate
    let top_pt = |x: usize, y: usize| -> Vec3 {
        Vec3::new(
            x as f32 * step_mm,
            y as f32 * step_mm,
            map.get(x, y).max(0.0) + base_thickness_mm,
        )
    };

       // 3. Top Relief Surface (consistent diagonal splitting)
    for y in 0..(h - 1) {
        for x in 0..(w - 1) {
            let v00 = top_pt(x, y);
            let v10 = top_pt(x + 1, y);
            let v01 = top_pt(x, y + 1);
            let v11 = top_pt(x + 1, y + 1);

            write_triangle(&mut writer, v00, v10, v11)?;
            write_triangle(&mut writer, v00, v11, v01)?;
        }
    }


    // 4. Perimeter Walls (connecting top border down to z = 0)
    // Front wall (y = 0, normal pointing -Y)
    for x in 0..(w - 1) {
        let t0 = top_pt(x, 0);
        let t1 = top_pt(x + 1, 0);
        let b0 = Vec3::new(t0.x, 0.0, 0.0);
        let b1 = Vec3::new(t1.x, 0.0, 0.0);
        write_triangle(&mut writer, b0, b1, t1)?;
        write_triangle(&mut writer, b0, t1, t0)?;
    }

    // Back wall (y = h - 1, normal pointing +Y)
    let max_y = (h - 1) as f32 * step_mm;
    for x in 0..(w - 1) {
        let t0 = top_pt(x, h - 1);
        let t1 = top_pt(x + 1, h - 1);
        let b0 = Vec3::new(t0.x, max_y, 0.0);
        let b1 = Vec3::new(t1.x, max_y, 0.0);
        write_triangle(&mut writer, b1, b0, t0)?;
        write_triangle(&mut writer, b1, t0, t1)?;
    }

    // Left wall (x = 0, normal pointing -X)
    for y in 0..(h - 1) {
        let t0 = top_pt(0, y);
        let t1 = top_pt(0, y + 1);
        let b0 = Vec3::new(0.0, t0.y, 0.0);
        let b1 = Vec3::new(0.0, t1.y, 0.0);
        write_triangle(&mut writer, b1, b0, t0)?;
        write_triangle(&mut writer, b1, t0, t1)?;
    }

    // Right wall (x = w - 1, normal pointing +X)
    let max_x = (w - 1) as f32 * step_mm;
    for y in 0..(h - 1) {
        let t0 = top_pt(w - 1, y);
        let t1 = top_pt(w - 1, y + 1);
        let b0 = Vec3::new(max_x, t0.y, 0.0);
        let b1 = Vec3::new(max_x, t1.y, 0.0);
        write_triangle(&mut writer, b0, b1, t1)?;
        write_triangle(&mut writer, b0, t1, t0)?;
    }

    // 5. Flat Bottom Face (z = 0, normal pointing -Z)
    let c00 = Vec3::new(0.0, 0.0, 0.0);
    let c10 = Vec3::new(max_x, 0.0, 0.0);
    let c01 = Vec3::new(0.0, max_y, 0.0);
    let c11 = Vec3::new(max_x, max_y, 0.0);
    write_triangle(&mut writer, c00, c01, c11)?;
    write_triangle(&mut writer, c00, c11, c10)?;

    writer.flush()?;
    Ok(())
}

