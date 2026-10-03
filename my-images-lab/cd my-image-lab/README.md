# Image Processing Lab

Exploration of image data, color channels, resolution, brightness, contrast, thresholding, blur, and Sobel edge detection using Python, OpenCV, NumPy, and Matplotlib.

## Setup Instructions

### Requirements
- Python 3.7+
- opencv-python
- numpy
- matplotlib

### Installation

1. Clone or navigate to the repository root.
2. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Directory Structure

Ensure the following structure before running:
```
.
├── images/
│   └── original.jpg          # Your photograph (placed here)
├── outputs/                  # Created automatically on first run
├── weeks01-03_image_lab.py
├── requirements.txt
└── README.md
```

## Run Command

From the repository root, execute:
```bash
python3 weeks01-03_image_lab.py
```

The script will:
1. Load `images/original.jpg`
2. Inspect image properties (width, height, channels, color order, data size)
3. Create channel separations (R, G, B), grayscale, and downsampled views
4. Apply brightness, contrast, and threshold adjustments
5. Generate blur and Sobel edge detection results
6. Save three labeled montages to `outputs/`:
   - `pixel_views.png` — Channels and resolution exploration
   - `adjustments.png` — Brightness, contrast, threshold comparison
   - `blur_and_edges.png` — Blur and edge detection results

## Image Data Report

### Original Image Properties
- **Width:** 1080 pixels
- **Height:** 1440 pixels
- **Channels:** 3 (BGR)
- **Shape:** [1440, 1080, 3]
- **Pixel Count:** 1,555,200 pixels
- **Estimated Data Size:** 4,665,600 bytes (~4.4 MB at 8 bits per channel)
- **Color Order:** BGR (OpenCV default)

### Observations

#### Task 2 — Color & Resolution
The original sunset photograph contains natural gradients and edges:
- **Red channel:** Prominent in sunset glow and sky reflections on water
- **Green channel:** Balanced across sky, water, and landscape; shows mid-tone detail
- **Blue channel:** Strong in upper sky; darker in water and foreground
- **Downsampling impact:** Resizing to 50% (540×720) preserves overall composition but loses fine texture detail in water ripples and rock edges

#### Task 3 — Adjustments
- **Brightness (+40):** Lifts all pixel values by 40, brightening dark foreground rocks and adding visibility to subtle water texture
- **Contrast (×1.5):** Amplifies difference between light (sun, sky) and dark (rocks, shadows), increasing visual separation
- **Threshold (127):** Creates stark binary separation; pixels ≥127 become white (sky, sun), others black (water, rocks, shadows); eliminates all mid-tone information

#### Task 4 — Blur & Edges
- **Mean blur (5×5 kernel):** Smooths high-frequency noise and fine texture; reduces sharpness of rock contours
- **Sobel edges (original):** Detects sharp transitions in the original image; strong responses at rock-water boundaries and sun-sky edge
- **Sobel edges (blurred):** Fewer detected edges due to smoothing; fine water ripples vanish; main structural edges (horizon, rocks) remain visible
- **Detail reduced by blur:** Fine water surface texture, small rock shadows, and micro-scale lighting variations are eliminated

## Script Functions

### `inspect_image(image_path: str) -> dict`
Loads image and returns: width, height, channels, shape, pixel_count, estimated_bytes, color_order.

### `create_pixel_views(image_path: str, output_dir: str) -> dict`
Creates labeled montage of original, R, G, B channels, grayscale, and downsampled image.

### `create_adjustments(image_path: str, output_dir: str, brightness_delta: int = 40, contrast_factor: float = 1.5, threshold: int = 127) -> dict`
Creates labeled montage of brightness, contrast, and threshold adjustments on grayscale.

### `create_blur_and_edges(image_path: str, output_dir: str, kernel_size: int = 5) -> dict`
Creates labeled montage of grayscale, mean blur, and Sobel edges (original and blurred).

### `run_lab(image_path: str, output_dir: str) -> dict`
Executes all four tasks and returns combined results.

### `main() -> None`
Entry point using required repository paths: `images/original.jpg` and `outputs/`.

## External Sources & Citations

- **OpenCV Documentation**: [https://docs.opencv.org/](https://docs.opencv.org/) — Used for `cv2.imread()`, `cv2.cvtColor()`, `cv2.split()`, `cv2.resize()`, `cv2.blur()`, `cv2.Sobel()`, and `cv2.threshold()`
- **NumPy Documentation**: [https://numpy.org/doc/](https://numpy.org/doc/) — Used for array operations, clipping, and mathematical operations
- **Matplotlib Documentation**: [https://matplotlib.org/](https://matplotlib.org/) — Used for image visualization and montage creation with GridSpec

## Author & Date

Created: October 2026
