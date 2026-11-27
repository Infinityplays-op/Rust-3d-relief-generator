use image::{imageops::FilterType, GenericImageView};
use ort::{session::Session, value::Tensor};
use std::path::Path;

/// Bilateral filter (edge-preserving smoothing) to eliminate 8-bit JPEG banding
/// and AI sensor noise without blurring sharp relief edges.
pub fn bilateral_smooth(
    data: &[f32],
    width: usize,
    height: usize,
    radius: isize,
    sigma_space: f32,
    sigma_range: f32,
) -> Vec<f32> {
    let mut output = vec![0.0f32; width * height];
    let inv_two_sigma_space_sq = -0.5 / (sigma_space * sigma_space);
    let inv_two_sigma_range_sq = -0.5 / (sigma_range * sigma_range);

    for y in 0..height {
        for x in 0..width {
            let center_z = data[y * width + x];
            let mut sum_weights = 0.0f32;
            let mut sum_z = 0.0f32;

            for dy in -radius..=radius {
                let ny = y as isize + dy;
                if ny < 0 || ny >= height as isize {
                    continue;
                }
                for dx in -radius..=radius {
                    let nx = x as isize + dx;
                    if nx < 0 || nx >= width as isize {
                        continue;
                    }

                    let neighbor_z = data[ny as usize * width + nx as usize];
                    let dist_sq = (dx * dx + dy * dy) as f32;
                    let dz = neighbor_z - center_z;

                    let w_space = (dist_sq * inv_two_sigma_space_sq).exp();
                    let w_range = ((dz * dz) * inv_two_sigma_range_sq).exp();
                    let weight = w_space * w_range;

                    sum_z += neighbor_z * weight;
                    sum_weights += weight;
                }
            }

            output[y * width + x] = if sum_weights > 1e-8 {
                sum_z / sum_weights
            } else {
                center_z
            };
        }
    }

    output
}
/// Fast separable Gaussian filter to eliminate diagonal pixel-staircase ripples
/// on high-contrast PNGs, logos, and depth maps.
pub fn gaussian_smooth(data: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 0.1 || width < 3 || height < 3 {
        return data.to_vec();
    }

    let radius = (sigma * 2.5).ceil() as isize;
    let inv_two_sigma_sq = -0.5 / (sigma * sigma);

    let mut kernel = Vec::with_capacity((2 * radius + 1) as usize);
    let mut k_sum = 0.0f32;
    for i in -radius..=radius {
        let w = ((i * i) as f32 * inv_two_sigma_sq).exp();
        kernel.push(w);
        k_sum += w;
    }
    for w in &mut kernel {
        *w /= k_sum;
    }

    let mut temp = vec![0.0f32; width * height];
    let mut dst = vec![0.0f32; width * height];

    // Horizontal pass
    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            let mut sum = 0.0;
            for (k_idx, &w) in kernel.iter().enumerate() {
                let dx = k_idx as isize - radius;
                let sx = (x as isize + dx).clamp(0, (width - 1) as isize) as usize;
                sum += data[row + sx] * w;
            }
            temp[row + x] = sum;
        }
    }

    // Vertical pass
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (k_idx, &w) in kernel.iter().enumerate() {
                let dy = k_idx as isize - radius;
                let sy = (y as isize + dy).clamp(0, (height - 1) as isize) as usize;
                sum += temp[sy * width + x] * w;
            }
            dst[y * width + x] = sum;
        }
    }

    dst
}

/// Normalizes raw depth values into `[0.0, max_height_mm]` and flips the Y-axis
/// for CNC bottom-left (0,0) coordinates.
fn normalize_and_flip(
    raw: &[f32],
    width: usize,
    height: usize,
    max_height_mm: f32,
    invert: bool,
) -> Vec<f32> {
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    for &v in raw {
        if v.is_finite() {
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
    }

    let range = (max_val - min_val).max(1e-6);
    let mut z_data = vec![0.0f32; width * height];

    for y in 0..height {
        let src_y = height - 1 - y;
        for x in 0..width {
            let mut norm = ((raw[src_y * width + x] - min_val) / range).clamp(0.0, 1.0);
            if invert {
                norm = 1.0 - norm;
            }
            z_data[y * width + x] = norm * max_height_mm;
        }
    }

    z_data
}

/// Path A: Loads an existing grayscale heightmap (JPEG/PNG/TIFF, 8-bit or 16-bit)
/// and scales it to the requested CNC dimensions.
pub fn grayscale_image_to_heightmap<P: AsRef<Path>>(
    image_path: P,
    target_width_mm: f32,
    step_mm: f32,
    max_height_mm: f32,
    invert: bool,
) -> Result<(usize, usize, Vec<f32>), Box<dyn std::error::Error>> {
    let img = image::open(image_path)?;
    let (orig_w, orig_h) = img.dimensions();
    let aspect = orig_h as f32 / orig_w as f32;

    let width = (target_width_mm / step_mm).round().max(2.0) as usize;
    let height = ((target_width_mm * aspect) / step_mm).round().max(2.0) as usize;

    let resized = img.resize_exact(width as u32, height as u32, FilterType::Lanczos3);
    let luma16 = resized.to_luma16();

    let raw: Vec<f32> = luma16.as_raw().iter().map(|&v| v as f32).collect();
    let mut z_data = normalize_and_flip(&raw, width, height, max_height_mm, invert);

    // Smooth out 8-bit JPEG compression artifacts
    z_data = bilateral_smooth(&z_data, width, height, 2, 2.0, max_height_mm * 0.08);
    // Eliminate diagonal pixel-grid ripples on sharp PNG/logo edges
    z_data = gaussian_smooth(&z_data, width, height, 1.35);

    Ok((width, height, z_data))


}

/// Path B: Runs a local Depth Anything V2 ONNX model via `ort` on a standard 2D photo
/// to infer true 3D surface depth, then resizes and smooths it for CNC carving.
pub fn photo_to_ai_heightmap<P1: AsRef<Path>, P2: AsRef<Path>>(
    image_path: P1,
    model_path: P2,
    target_width_mm: f32,
    step_mm: f32,
    max_height_mm: f32,
    invert: bool,
) -> Result<(usize, usize, Vec<f32>), Box<dyn std::error::Error>> {
    let img = image::open(image_path)?;
    let (orig_w, orig_h) = img.dimensions();
    let aspect = orig_h as f32 / orig_w as f32;

    const MODEL_SIZE: usize = 518;
    let model_img = img
        .resize_exact(MODEL_SIZE as u32, MODEL_SIZE as u32, FilterType::CatmullRom)
        .to_rgb8();

    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];

    // Build NCHW flat buffer: [1, 3, 518, 518]
    let plane_size = MODEL_SIZE * MODEL_SIZE;
    let mut input_vec = vec![0.0f32; 3 * plane_size];

    for y in 0..MODEL_SIZE {
        for x in 0..MODEL_SIZE {
            let px = model_img.get_pixel(x as u32, y as u32);
            let idx = y * MODEL_SIZE + x;
            for c in 0..3 {
                let val = px[c] as f32 / 255.0;
                input_vec[c * plane_size + idx] = (val - mean[c]) / std[c];
            }
        }
    }

    // Initialize ONNX Runtime environment (required in ort 2.0.0-rc.15+)
    let env = ort::init().build()?;
    let mut session = Session::builder(&env)?.commit_from_file(model_path)?;


    let input_val = Tensor::from_array(([1, 3, MODEL_SIZE, MODEL_SIZE], input_vec))?;
    let outputs = session.run(ort::inputs![input_val])?;

    let (_shape, depth_slice) = outputs[0].try_extract_tensor::<f32>()?;

    let out_w = (target_width_mm / step_mm).round().max(2.0) as usize;
    let out_h = ((target_width_mm * aspect) / step_mm).round().max(2.0) as usize;
    let resized_depth = bilinear_resize(
        depth_slice,
        MODEL_SIZE,
        MODEL_SIZE,
        out_w,
        out_h,
    );

        let mut z_data = normalize_and_flip(&resized_depth, out_w, out_h, max_height_mm, invert);
    z_data = bilateral_smooth(&z_data, out_w, out_h, 3, 2.5, max_height_mm * 0.06);

    Ok((out_w, out_h, z_data))
}


/// Simple bilinear resampler for floating-point depth grids.
fn bilinear_resize(src: &[f32], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; dst_w * dst_h];
    let scale_x = (src_w - 1) as f32 / (dst_w - 1).max(1) as f32;
    let scale_y = (src_h - 1) as f32 / (dst_h - 1).max(1) as f32;

    for y in 0..dst_h {
        let fy = y as f32 * scale_y;
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = fy - y0 as f32;

        for x in 0..dst_w {
            let fx = x as f32 * scale_x;
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = fx - x0 as f32;

            let v00 = src[y0 * src_w + x0];
            let v10 = src[y0 * src_w + x1];
            let v01 = src[y1 * src_w + x0];
            let v11 = src[y1 * src_w + x1];

            let top = v00 * (1.0 - wx) + v10 * wx;
            let bot = v01 * (1.0 - wx) + v11 * wx;
            dst[y * dst_w + x] = top * (1.0 - wy) + bot * wy;
        }
    }
    dst
}
/// Gradient-domain Bas-Relief Dynamic Range Compression.
/// Flattens deep macro-perspective while preserving fine micro-details on shallow CNC reliefs.
///
/// * `compression`: `0.0` = off (linear depth), `0.5` = strong bas-relief detail boost, `0.85` = maximum coin-like flattening.
/// * `max_height_mm`: Rescales the integrated result cleanly back to `[0.0, max_height_mm]`.
pub fn compress_bas_relief(
    z_data: &[f32],
    width: usize,
    height: usize,
    compression: f32,
    max_height_mm: f32,
) -> Vec<f32> {
    let strength = compression.clamp(0.0, 0.9);
    if strength <= 0.01 || width < 3 || height < 3 {
        return z_data.to_vec();
    }

    // Map compression strength [0.0..0.9] to gradient attenuation exponent gamma [1.0..0.25]
    let gamma = 1.0 - strength * 0.85;

    // 1. Compute forward gradients gx, gy and measure mean gradient magnitude
    let mut gx = vec![0.0f32; width * height];
    let mut gy = vec![0.0f32; width * height];
    let mut mag_sum = 0.0f32;
    let mut count = 0usize;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let zc = z_data[idx];
            let dx = if x + 1 < width { z_data[idx + 1] - zc } else { 0.0 };
            let dy = if y + 1 < height { z_data[idx + width] - zc } else { 0.0 };
            gx[idx] = dx;
            gy[idx] = dy;

            let mag = (dx * dx + dy * dy).sqrt();
            if mag > 1e-6 {
                mag_sum += mag;
                count += 1;
            }
        }
    }

    let alpha = if count > 0 {
        (mag_sum / count as f32).max(1e-5)
    } else {
        return z_data.to_vec();
    };

    // 2. Non-linearly attenuate large macro-gradients while boosting micro-details
    for i in 0..(width * height) {
        let dx = gx[i];
        let dy = gy[i];
        let mag = (dx * dx + dy * dy).sqrt();
        if mag > 1e-7 {
            let scale = (mag / alpha + 1e-4).powf(gamma - 1.0);
            gx[i] = dx * scale;
            gy[i] = dy * scale;
        }
    }

    // 3. Compute divergence of compressed gradient field: div(g')
    let mut div = vec![0.0f32; width * height];
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let idx = y * width + x;
            let dgx = gx[idx] - gx[idx - 1];
            let dgy = gy[idx] - gy[idx - width];
            div[idx] = dgx + dgy;
        }
    }

    // 4. Solve 2D Poisson equation (Laplacian(z) = div) via Red-Black Gauss-Seidel SOR
    let mut out = z_data.to_vec();
    let omega = 1.88f32; // Over-relaxation factor for fast convergence
    let iterations = 220;

    for _ in 0..iterations {
        for pass in 0..2 {
            for y in 1..(height - 1) {
                let start_x = 1 + ((y + pass) & 1);
                let mut x = start_x;
                while x < width - 1 {
                    let idx = y * width + x;
                    let neighbors =
                        out[idx - 1] + out[idx + 1] + out[idx - width] + out[idx + width];
                    let target = 0.25 * (neighbors - div[idx]);
                    out[idx] += omega * (target - out[idx]);
                    x += 2;
                }
            }
        }
    }

    // 5. Normalize back to [0.0, max_height_mm]
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    for &v in &out {
        min_z = min_z.min(v);
        max_z = max_z.max(v);
    }
    let span = (max_z - min_z).max(1e-6);
    for v in &mut out {
        *v = ((*v - min_z) / span) * max_height_mm;
    }

    out
}

