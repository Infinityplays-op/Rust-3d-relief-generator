# Rust CNC Relief Studio (`relief-gen`)

A desktop 3D relief generator built in Rust for CNC machining, laser engraving, and 3D printing. Convert vector artwork (`.svg`), grayscale heightmaps, or standard 2D photographs (using local ONNX depth estimation) into watertight binary `.stl` meshes with real-time 3D previewing.

---

## Features

- **Three Input Pipelines:**
  - **Vector Inflation (`.svg`):** Uses `resvg` with 4× supersampling, subpixel Euclidean Distance Transform (EDT), and anti-aliasing to inflate vector shapes into smooth 3D reliefs with selectable bevel profiles (**Circular Dome**, **Linear Bevel**, and **Smooth Pillow**).
  - **Grayscale Heightmaps (`.png`, `.jpg`, `.webp`, `.tif`, `.bmp`):** Maps 8-bit and 16-bit luminance directly to physical Z-heights, automatically applying edge-preserving bilateral filtering and Gaussian smoothing to remove compression banding and pixel-staircase artifacts.
  - **AI Photo-to-Relief (`.onnx`):** Optionally load a local **Depth Anything V2** ONNX model via `ort` (ONNX Runtime) to infer 3D surface depth from standard 2D photographs.
- **Gradient-Domain Bas-Relief Compression:** Solves a 2D Poisson equation via Red-Black Gauss-Seidel Successive Over-Relaxation (SOR) to compress deep macro-perspective while preserving fine micro-details for shallow CNC stock.
- **Real-Time Interactive 3D Viewport:** Powered by `three-d` and `egui`, featuring physically based shading, multi-point studio lighting, automatic camera framing, and responsive background mesh generation.
- **Watertight Binary STL Export:** Generates manifold binary `.stl` files complete with consistent right-hand rule normals, perimeter sidewalls, and a flat bottom base ready for CAM software (Vectric VCarve, Fusion 360, Carveco, Easel, or 3D slicers).

---

## Installation & Building

### Prerequisites
- **Rust Toolchain** (1.75+ recommended) — [Install Rust](https://rust-lang.org)
- **System Graphics & Windowing Libraries** (required by `three-d` / `winit` / `rfd`):
  - **Windows & macOS:** Works out of the box.
  - **Linux (Ubuntu/Debian):**
    ```bash
    sudo apt-get install -y libgtk-3-dev libxkbcommon-dev libwayland-dev libgl1-mesa-dev
    ```

### Build & Run
Because heightmap smoothing, distance transforms, and Poisson solvers are computationally intensive, always build with `--release` (configured with `opt-level = 3` and `lto = true` in `Cargo.toml`):

```bash
git clone https://github.com
cd relief-gen
cargo run --release
```

*(Note: On first build, `ort` will automatically download the required ONNX Runtime binaries.)*

---

## Usage Guide

1. **Import Artwork:** Click **Import SVG / Image...** in the left control panel to load an `.svg` or raster image.
2. *(Optional)* **Load AI Depth Model:** To turn a regular photograph into a 3D relief, click **AI Depth ONNX...** and select a **Depth Anything V2** `.onnx` model file (expects a `1 × 3 × 518 × 518` NCHW float32 input tensor). Click **X** next to the button to unload the model and revert to standard grayscale luminance mode.
3. **Configure CNC Stock & Relief Parameters:**
   - **Width (mm):** Target physical X-axis width of the carving (height scales automatically to preserve aspect ratio).
   - **Step (mm/px):** Grid resolution per pixel (`0.08 mm` – `0.60 mm`). Lower values yield finer detail and higher triangle counts.
   - **Relief Z (mm):** Maximum carving depth/height above the base.
   - **Base (mm):** Solid material thickness beneath the lowest point of the relief (`Z = 0`).
   - **Invert Z (Engrave Inward):** Flips highs and lows for recessed/intaglio carving or lithophanes.
   - **Bas-Relief Boost:** Applies gradient-domain dynamic range compression (`0.0` = linear, `0.85` = coin-like flattening with boosted surface detail).
   - **Edge Smooth:** Adjusts Gaussian smoothing sigma prior to bas-relief compression.
   - **Vector Inflation (SVG only):** Adjust **Bevel Width (mm)** and switch between **Circular Dome**, **Linear Bevel**, and **Smooth Pillow** profiles.
4. **Export STL:** Once satisfied with the 3D preview, click **Export Watertight STL...** to save your binary `.stl` file.

### Viewport Controls
- **Left-Click + Drag:** Orbit camera
- **Right-Click + Drag:** Pan camera
- **Scroll Wheel:** Zoom in / out

---
# Rust-3d-relief-generator
A simple svg / png to stl generator for making 3d reliefs for a cnc

## Project Architecture

| File | Responsibility |
| :--- | :--- |
| `src/main.rs` | Application entry point, `three-d` OpenGL rendering loop, `egui` control panel, camera auto-framing, and background worker thread management. |
| `src/svg_relief.rs` | `resvg` 4× supersampled SVG rasterizer, subpixel Euclidean Distance Transform (EDT), bevel profile math, and 5-tap separable Gaussian anti-aliasing. |
| `src/image_relief.rs` | Grayscale luminance loader, `ort` ONNX Runtime inference pipeline, edge-preserving bilateral filter, Gaussian smoothing, and Red-Black SOR Poisson bas-relief solver. |
| `src/stl_mesher.rs` | `Heightmap` data structure and buffered binary `.stl` writer generating the top surface, four perimeter walls, and flat bottom face. |

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
