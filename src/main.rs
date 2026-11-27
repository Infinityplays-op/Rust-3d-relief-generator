mod image_relief;
mod stl_mesher;
mod svg_relief;

use image_relief::{compress_bas_relief, gaussian_smooth, grayscale_image_to_heightmap, photo_to_ai_heightmap,};


use rfd::FileDialog;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::thread;
use stl_mesher::{export_heightmap_to_stl, Heightmap};
use svg_relief::{svg_to_heightmap, ReliefProfile};
use three_d::*;

#[derive(Clone, Debug, PartialEq)]
struct ReliefParams {
    input_path: Option<PathBuf>,
    ai_model_path: Option<PathBuf>,
    width_mm: f32,
    step_mm: f32,
    max_height_mm: f32,
    base_mm: f32,
    bevel_mm: f32,
    profile_idx: usize,
    invert: bool,
    bas_relief_compression: f32,
    smoothing: f32,
}

impl Default for ReliefParams {
    fn default() -> Self {
        Self {
            input_path: None,
            ai_model_path: None,
            width_mm: 150.0,
            step_mm: 0.25,
            max_height_mm: 6.0,
            base_mm: 3.0,
            bevel_mm: 4.0,
            profile_idx: 0,
            invert: false,
            bas_relief_compression: 0.0,
            smoothing: 1.35,
        }
    }
}

enum WorkerResult {
    Success(Heightmap),
    Error(String),
}

fn heightmap_to_cpu_mesh(map: &Heightmap, step_mm: f32, base_mm: f32) -> CpuMesh {
    let w = map.width;
    let h = map.height;
    let cx = ((w - 1) as f32 * step_mm) * 0.5;
    let cy = ((h - 1) as f32 * step_mm) * 0.5;

    let mut positions = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            positions.push(vec3(
                x as f32 * step_mm - cx,
                y as f32 * step_mm - cy,
                map.get(x, y).max(0.0) + base_mm,
            ));
        }
    }

    let mut indices = Vec::with_capacity((w - 1) * (h - 1) * 6);
    for y in 0..(h - 1) {
        for x in 0..(w - 1) {
            let i00 = (y * w + x) as u32;
            let i10 = (y * w + (x + 1)) as u32;
            let i01 = ((y + 1) * w + x) as u32;
            let i11 = ((y + 1) * w + (x + 1)) as u32;

            indices.extend_from_slice(&[i00, i10, i11, i00, i11, i01]);
        }
    }

    let mut mesh = CpuMesh {
        positions: Positions::F32(positions),
        indices: Indices::U32(indices),
        ..Default::default()
    };
    mesh.compute_normals();
    mesh
}

fn spawn_generation_job(params: ReliefParams) -> Receiver<WorkerResult> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let Some(input) = params.input_path else {
            return;
        };
        let ext = input
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();

        let res = match ext.as_str() {
            "svg" => {
                let profile = match params.profile_idx {
                    1 => ReliefProfile::LinearBevel,
                    2 => ReliefProfile::SmoothPillow,
                    _ => ReliefProfile::CircularDome,
                };
                svg_to_heightmap(
                    &input,
                    params.width_mm,
                    params.step_mm,
                    params.bevel_mm,
                    params.max_height_mm,
                    profile,
                    params.invert,
                )
            }
            "jpg" | "jpeg" | "png" | "webp" | "tif" | "tiff" | "bmp" => {
                if let Some(model) = &params.ai_model_path {
                    photo_to_ai_heightmap(
                        &input,
                        model,
                        params.width_mm,
                        params.step_mm,
                        params.max_height_mm,
                        params.invert,
                    )
                } else {
                    grayscale_image_to_heightmap(
                        &input,
                        params.width_mm,
                        params.step_mm,
                        params.max_height_mm,
                        params.invert,
                    )
                }
            }
            _ => Err(format!("Unsupported format: .{}", ext).into()),
        };

             match res {
         Ok((w, h, mut z)) => {
             if params.bas_relief_compression > 0.01 {
                 z = gaussian_smooth(&z, w, h, params.smoothing);
                 z = compress_bas_relief(
                     &z,
                     w,
                     h,
                     params.bas_relief_compression,
                     params.max_height_mm,
                 );
             }
             let _ = tx.send(WorkerResult::Success(Heightmap::new(w, h, z)));
         }
         Err(e) => {
             let _ = tx.send(WorkerResult::Error(e.to_string()));
         }
     }

    });
    rx
}

fn main() {
    let window = Window::new(WindowSettings {
        title: "Rust CNC Relief Studio".to_string(),
        max_size: Some((1440, 900)),
        ..Default::default()
    })
    .expect("Failed to create window");

    let context = window.gl();

    let mut camera = Camera::new_perspective(
        window.viewport(),
        vec3(0.0, -140.0, 140.0),
        vec3(0.0, 0.0, 0.0),
        vec3(0.0, 0.0, 1.0),
        degrees(45.0),
        0.1,
        2000.0,
    );
    let mut control = OrbitControl::new(camera.target(), 10.0, 1000.0);

    let key_light = DirectionalLight::new(&context, 2.5, Srgba::WHITE, vec3(-0.6, 0.8, -1.0));
    let fill_light = DirectionalLight::new(
        &context,
        0.9,
        Srgba::new(200, 220, 255, 255),
        vec3(0.8, -0.5, -0.6),
    );
    let ambient = AmbientLight::new(&context, 0.25, Srgba::WHITE);

    let material = PhysicalMaterial::new_opaque(
        &context,
        &CpuMaterial {
            albedo: Srgba::new(212, 175, 130, 255),
            roughness: 0.45,
            metallic: 0.15,
            ..Default::default()
        },
    );

    let mut gui = GUI::new(&context);
    let mut params = ReliefParams::default();
    let mut current_map: Option<Heightmap> = None;
    let mut relief_gm: Option<Gm<Mesh, PhysicalMaterial>> = None;
    let mut worker_rx: Option<Receiver<WorkerResult>> = None;
    let mut status_msg = String::from("Import an SVG or image file to begin.");
    let mut is_generating = false;
    window.render_loop(move |mut frame_input| {
        if let Some(rx) = &worker_rx {
            if let Ok(result) = rx.try_recv() {
                is_generating = false;
                worker_rx = None;
                match result {
                     WorkerResult::Success(map) => {
                        let cpu_mesh = heightmap_to_cpu_mesh(&map, params.step_mm, params.base_mm);
                        let gpu_mesh = Mesh::new(&context, &cpu_mesh);
                        relief_gm = Some(Gm::new(gpu_mesh, material.clone()));

                        // Auto-center & pull camera back so the full width fits inside the 3D viewport
                        let phys_w = (map.width - 1) as f32 * params.step_mm;
                        let phys_h = (map.height - 1) as f32 * params.step_mm;
                        let vp_w = (frame_input.viewport.width as f32 - 300.0 * frame_input.device_pixel_ratio).max(100.0);
                        let vp_h = (frame_input.viewport.height as f32).max(100.0);
                        let aspect = (vp_w / vp_h).max(0.2);


                        let half_fov_y = 22.5f32.to_radians();
                        let tan_y = half_fov_y.tan();
                        let tan_x = tan_y * aspect;

                   // Required distance so neither horizontal nor vertical edges clip
                        let dist_y = (phys_h * 0.5) / tan_y;
                        let dist_x = (phys_w * 0.5) / tan_x;
                        let dist = dist_x.max(dist_y).max(40.0) * 1.25;

                        camera.set_view(
                        vec3(0.0, -dist * 0.65, dist * 1.05),
                        vec3(0.0, 0.0, params.base_mm * 0.5),
                        vec3(0.0, 0.0, 1.0),
                       );

                        status_msg = format!(
                            "Ready: {}x{} grid ({} triangles)",
                            map.width,
                            map.height,
                            2 * (map.width - 1) * (map.height - 1)
                        );
                        current_map = Some(map);
                    }

                    WorkerResult::Error(err) => {
                        status_msg = format!("Error: {}", err);
                    }
                }
            }
        }

        let mut trigger_rebuild = false;
        let mut panel_width = 300.0;

        gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,

            |egui_ctx| {
                let panel_resp = egui::SidePanel::left("controls_panel")
                    .default_width(300.0)
                    .show(egui_ctx, |ui| {
                        ui.heading("CNC Relief Studio");
                        ui.separator();

                        if ui.button("Import SVG / Image...").clicked() {
                            if let Some(path) = FileDialog::new()
                                .add_filter(
                                    "Artwork & Images",
                                    &["svg", "png", "jpg", "jpeg", "webp", "tif", "tiff"],
                                )
                                .pick_file()
                            {
                                params.input_path = Some(path);
                                trigger_rebuild = true;
                            }
                        }

                        if let Some(p) = &params.input_path {
                            ui.small(format!(
                                "File: {}",
                                p.file_name().unwrap_or_default().to_string_lossy()
                            ));
                        }

                        ui.add_space(6.0);

                        ui.horizontal(|ui| {
                            if ui.button("AI Depth ONNX...").clicked() {
                                if let Some(path) = FileDialog::new()
                                    .add_filter("ONNX Model", &["onnx"])
                                    .pick_file()
                                {
                                    params.ai_model_path = Some(path);
                                    if params.input_path.is_some() {
                                        trigger_rebuild = true;
                                    }
                                }
                            }
                            if params.ai_model_path.is_some() && ui.button("X").clicked() {
                                params.ai_model_path = None;
                                if params.input_path.is_some() {
                                    trigger_rebuild = true;
                                }
                            }
                        });

                        if let Some(m) = &params.ai_model_path {
                            ui.small(format!(
                                "Model: {}",
                                m.file_name().unwrap_or_default().to_string_lossy()
                            ));
                        }

                        ui.separator();
                        ui.label("Dimensions & CNC Stock");

                        if ui
                            .add(egui::Slider::new(&mut params.width_mm, 20.0..=500.0).text("Width (mm)"))
                            .changed()
                        {
                            trigger_rebuild = true;
                        }

                        if ui
                            .add(egui::Slider::new(&mut params.step_mm, 0.08..=0.6).text("Step (mm/px)"))
                            .changed()
                        {
                            trigger_rebuild = true;
                        }

                        if ui
                            .add(egui::Slider::new(&mut params.max_height_mm, 0.5..=30.0).text("Relief Z (mm)"))
                            .changed()
                        {
                            trigger_rebuild = true;
                        }

                        if ui
                            .add(egui::Slider::new(&mut params.base_mm, 0.5..=20.0).text("Base (mm)"))
                            .changed()
                        {
                            trigger_rebuild = true;
                        }

                        if ui.checkbox(&mut params.invert, "Invert Z (Engrave Inward)").changed() {
                            trigger_rebuild = true;
                        }
                        if ui
                            .add(egui::Slider::new(&mut params.bas_relief_compression, 0.0..=0.85).text("Bas-Relief Boost"))
                            .changed()
                        {
                            trigger_rebuild = true;
                        }
                        if ui
                           .add(egui::Slider::new(&mut params.smoothing, 0.0..=4.0).text("Edge Smooth"))
                           .changed() 
                        {
                            trigger_rebuild = true;
                        }


                        let is_svg = params
                            .input_path
                            .as_ref()
                            .and_then(|p| p.extension())
                            .map(|e| e.eq_ignore_ascii_case("svg"))
                            .unwrap_or(false);

                        if is_svg {
                            ui.separator();
                            ui.label("Vector Inflation (SVG)");
                            if ui
                                .add(egui::Slider::new(&mut params.bevel_mm, 0.5..=20.0).text("Bevel Width (mm)"))
                                .changed()
                            {
                                trigger_rebuild = true;
                            }

                            egui::ComboBox::from_label("Profile")
                                .selected_text(match params.profile_idx {
                                    1 => "Linear Bevel",
                                    2 => "Smooth Pillow",
                                    _ => "Circular Dome",
                                })
                                .show_ui(ui, |ui| {
                                    if ui.selectable_value(&mut params.profile_idx, 0, "Circular Dome").changed() {
                                        trigger_rebuild = true;
                                    }
                                    if ui.selectable_value(&mut params.profile_idx, 1, "Linear Bevel").changed() {
                                        trigger_rebuild = true;
                                    }
                                    if ui.selectable_value(&mut params.profile_idx, 2, "Smooth Pillow").changed() {
                                        trigger_rebuild = true;
                                    }
                                });
                        }

                        ui.separator();

                        ui.add_enabled_ui(current_map.is_some() && !is_generating, |ui| {
                            if ui.button("Export Watertight STL...").clicked() {
                                if let Some(map) = &current_map {
                                    if let Some(save_path) = FileDialog::new()
                                        .set_file_name("cnc_relief.stl")
                                        .add_filter("Binary STL", &["stl"])
                                        .save_file()
                                    {
                                        match export_heightmap_to_stl(
                                            map,
                                            &save_path,
                                            params.step_mm,
                                            params.base_mm,
                                        ) {
                                            Ok(()) => {
                                                status_msg = format!("Saved STL to {}", save_path.display());
                                            }
                                            Err(e) => {
                                                status_msg = format!("Export failed: {}", e);
                                            }
                                        }
                                    }
                                }
                            }
                        });

                        ui.add_space(10.0);
                        if is_generating {
                            ui.spinner();
                            ui.label("Computing relief mesh...");
                        } else {
                            ui.small(&status_msg);
                        }

                        ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                            ui.small("Left-drag: Orbit | Right-drag: Pan | Scroll: Zoom");
                        });
                    });

                panel_width = panel_resp.response.rect.width() * frame_input.device_pixel_ratio;
            },
        );

        if trigger_rebuild && params.input_path.is_some() && !is_generating {
            is_generating = true;
            worker_rx = Some(spawn_generation_job(params.clone()));
        }

        let mut viewport = frame_input.viewport;
        let offset_x = panel_width as i32;
        if viewport.width > offset_x as u32 {
            viewport.x = offset_x;
            viewport.width -= offset_x as u32;
        }
        camera.set_viewport(viewport);

               let pointer_in_3d = frame_input.events.iter().all(|e| match e {
            Event::MouseMotion { position, .. }
            | Event::MousePress { position, .. }
            | Event::MouseWheel { position, .. } => position.x >= panel_width,
            _ => true,
        });
        if pointer_in_3d {
            control.handle_events(&mut camera, &mut frame_input.events);
        }

        let screen = frame_input.screen();
        screen.clear(ClearState::color_and_depth(0.14, 0.15, 0.18, 1.0, 1.0));

        if let Some(gm) = &relief_gm {
            screen.render(&camera, gm, &[&key_light, &fill_light, &ambient]);
        }

        screen.write(|| gui.render()).unwrap();

        FrameOutput::default()
    });
}
 

