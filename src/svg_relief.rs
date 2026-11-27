use resvg::tiny_skia::{Pixmap, Transform};
use resvg::usvg::{Options, Tree};
use std::fs;
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub enum ReliefProfile {
    LinearBevel,
    CircularDome,
    SmoothPillow,
}

fn edt_1d(f: &[f32], n: usize) -> Vec<f32> {
    let mut d = vec![0.0f32; n];
    let mut v = vec![0usize; n];
    let mut z = vec![0.0f32; n + 1];

    let mut k = 0usize;
    *v.first_mut().unwrap() = 0;
    *z.first_mut().unwrap() = f32::NEG_INFINITY;
    *z.get_mut(1).unwrap() = f32::INFINITY;


    for q in 1..n {
        let mut s = ((f[q] + (q * q) as f32) - (f[v[k]] + (v[k] * v[k]) as f32))
            / (2.0 * (q as f32 - v[k] as f32));

        while s <= z[k] {
            k -= 1;
            s = ((f[q] + (q * q) as f32) - (f[v[k]] + (v[k] * v[k]) as f32))
                / (2.0 * (q as f32 - v[k] as f32));
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f32::INFINITY;
    }

    k = 0;
    for q in 0..n {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let dx = q as f32 - v[k] as f32;
        d[q] = dx * dx + f[v[k]];
    }

    d
}

pub fn compute_subpixel_edt(alpha_grid: &[f32], width: usize, height: usize) -> Vec<f32> {
    const INF: f32 = 1e12;
    let mut grid = vec![0.0f32; width * height];

    for i in 0..grid.len() {
        let a = alpha_grid[i].clamp(0.0, 1.0);
        grid[i] = if a <= 0.0 {
            0.0
        } else if a >= 1.0 {
            INF
        } else {
            // Exact subpixel distance from anti-aliased coverage
            a * a
        };
    }

    let mut col = vec![0.0f32; height];
    for x in 0..width {
        for y in 0..height {
            col[y] = grid[y * width + x];
        }
        let transformed = edt_1d(&col, height);
        for y in 0..height {
            grid[y * width + x] = transformed[y];
        }
    }

    for y in 0..height {
        let row_start = y * width;
        let row = &grid[row_start..row_start + width];
        let transformed = edt_1d(row, width);
        for x in 0..width {
            grid[row_start + x] = (transformed[x].sqrt() - 0.5).max(0.0);
        }
    }

    grid
}

/// Fast separable 5-tap Gaussian blur (`[1, 4, 6, 4, 1] / 16`) to eliminate
/// diagonal pixel-grid aliasing on supersampled heightfields.
fn smooth_grid_5tap(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut temp = vec![0.0f32; w * h];
    let mut dst = vec![0.0f32; w * h];
    const K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

    // Horizontal pass
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            let mut sum = 0.0;
            for i in 0..5 {
                let sx = (x as isize + i as isize - 2).clamp(0, (w - 1) as isize) as usize;
                sum += src[row + sx] * K[i];
            }
            temp[row + x] = sum;
        }
    }

    // Vertical pass
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for i in 0..5 {
                let sy = (y as isize + i as isize - 2).clamp(0, (h - 1) as isize) as usize;
                sum += temp[sy * w + x] * K[i];
            }
            dst[y * w + x] = sum;
        }
    }

    dst
}
pub fn svg_to_heightmap<P: AsRef<Path>>(
    svg_path: P,
    target_width_mm: f32,
    step_mm: f32,
    bevel_width_mm: f32,
    max_height_mm: f32,
    profile: ReliefProfile,
    invert: bool,
) -> Result<(usize, usize, Vec<f32>), Box<dyn std::error::Error>> {
    let svg_data = fs::read(svg_path)?;
    let tree = Tree::from_data(&svg_data, &Options::default())?;

    let svg_size = tree.size();
    let aspect_ratio = svg_size.height() / svg_size.width();
    let target_height_mm = target_width_mm * aspect_ratio;

    let out_w = (target_width_mm / step_mm).round().max(2.0) as usize;
    let out_h = (target_height_mm / step_mm).round().max(2.0) as usize;

    // 4x Supersampling + Gaussian anti-aliasing makes diagonal edges (like stars) razor-smooth
    const SS: usize = 4;
    let ss_w = out_w * SS;
    let ss_h = out_h * SS;
    let ss_step_mm = step_mm / SS as f32;

    let mut pixmap = Pixmap::new(ss_w as u32, ss_h as u32)
        .ok_or("Failed to allocate supersampled raster buffer")?;

    let scale_x = ss_w as f32 / svg_size.width();
    let scale_y = ss_h as f32 / svg_size.height();
    let transform = Transform::from_scale(scale_x, scale_y);

    resvg::render(&tree, transform, &mut pixmap.as_mut());

    let pixels = pixmap.pixels();
    let mut alpha_grid = vec![0.0f32; ss_w * ss_h];
    for y in 0..ss_h {
        let src_y = ss_h - 1 - y;
        for x in 0..ss_w {
            let px = pixels[src_y * ss_w + x];
            let a = px.alpha() as f32 / 255.0;
            alpha_grid[y * ss_w + x] = if invert { 1.0 - a } else { a };
        }
    }

    let dist_px = compute_subpixel_edt(&alpha_grid, ss_w, ss_h);
    let mut ss_z = Vec::with_capacity(ss_w * ss_h);

    for d_px in dist_px {
        let d_mm = d_px * ss_step_mm;
        if d_mm <= 0.0 {
            ss_z.push(0.0);
            continue;
        }

        let t = (d_mm / bevel_width_mm.max(1e-5)).clamp(0.0, 1.0);
        let z = match profile {
            ReliefProfile::LinearBevel => max_height_mm * t,
            ReliefProfile::CircularDome => {
                // Soften infinite 90-degree vertical tangent at t=0 so base edge stays ripple-free
                let t_soft = t.powf(1.12);
                max_height_mm * (1.0 - (1.0 - t_soft).powi(2)).sqrt()
            }
            ReliefProfile::SmoothPillow => max_height_mm * (3.0 * t * t - 2.0 * t * t * t),
        };
        ss_z.push(z);
    }

    // Apply 5-tap Gaussian anti-aliasing filter at 4x resolution before downsampling
    let ss_z_smooth = smooth_grid_5tap(&ss_z, ss_w, ss_h);

    // Downsample 4x4 blocks back to target CNC grid resolution
    let mut z_data = vec![0.0f32; out_w * out_h];
    let inv_area = 1.0 / (SS * SS) as f32;
    for y in 0..out_h {
        for x in 0..out_w {
            let mut sum = 0.0;
            for dy in 0..SS {
                let row = (y * SS + dy) * ss_w + (x * SS);
                for dx in 0..SS {
                    sum += ss_z_smooth[row + dx];
                }
            }
            z_data[y * out_w + x] = sum * inv_area;
        }
    }
    // Smooth final grid to eliminate square-grid triangle aliasing along diagonal slopes
    let z_data = smooth_grid_5tap(&z_data, out_w, out_h);
    let z_data = smooth_grid_5tap(&z_data, out_w, out_h);

    Ok((out_w, out_h, z_data))

    
}

