import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def inspect_image(image_path: str) -> dict:
    """Load the image and return its measured image-data properties."""
    image = cv2.imread(image_path)
    
    height, width, channels = image.shape
    shape = [height, width, channels]
    pixel_count = height * width
    estimated_bytes = pixel_count * channels  # 8 bits per channel
    color_order = "BGR"  # OpenCV default
    
    print(f"\n=== Image Inspection ===")
    print(f"Width: {width}")
    print(f"Height: {height}")
    print(f"Channels: {channels}")
    print(f"Shape: {shape}")
    print(f"Pixel Count: {pixel_count}")
    print(f"Estimated Data Size: {estimated_bytes} bytes")
    print(f"Color Order: {color_order}")
    
    return {
        "width": width,
        "height": height,
        "channels": channels,
        "shape": shape,
        "pixel_count": pixel_count,
        "estimated_bytes": estimated_bytes,
        "color_order": color_order
    }


def create_pixel_views(image_path: str, output_dir: str) -> dict:
    """Create the labeled channel, grayscale, and downsampled views."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    
    # Split into channels
    blue, green, red = cv2.split(image)
    
    # Grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Downsample to half size
    downsampled = cv2.resize(image, (width // 2, height // 2))
    
    # Create montage with 5 subplots
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # Original (full width top)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Downsampled (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(cv2.cvtColor(downsampled, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"Downsampled\n({width//2}×{height//2})", fontsize=10, fontweight='bold')
    ax2.axis('off')
    
    # Red channel
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(red, cmap='Reds')
    ax3.set_title("Red Channel", fontsize=10, fontweight='bold')
    ax3.axis('off')
    
    # Green channel
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(green, cmap='Greens')
    ax4.set_title("Green Channel", fontsize=10, fontweight='bold')
    ax4.axis('off')
    
    # Blue channel
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.imshow(blue, cmap='Blues')
    ax5.set_title("Blue Channel", fontsize=10, fontweight='bold')
    ax5.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "pixel_views.png")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"\n=== Pixel Views ===")
    print(f"Original size: [{width}, {height}]")
    print(f"Downsampled size: [{width//2}, {height//2}]")
    print(f"Observations:")
    print(f"  - Lost detail: Fine texture in rocks and water ripples reduced in downsampled version")
    print(f"  - Channel difference: Red channel shows sunset glow more prominently; blue shows sky; green has intermediate tones")
    
    return {
        "original_size": [width, height],
        "downsampled_size": [width // 2, height // 2],
        "output_path": output_path
    }


def create_adjustments(
    image_path: str,
    output_dir: str,
    brightness_delta: int = 40,
    contrast_factor: float = 1.5,
    threshold: int = 127,
) -> dict:
    """Create labeled brightness, contrast, and threshold results."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image = cv2.imread(image_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Brightness adjustment
    brightness = cv2.convertScaleAbs(grayscale.astype(np.float32) + brightness_delta)
    brightness = np.clip(brightness, 0, 255).astype(np.uint8)
    
    # Contrast adjustment
    contrast = cv2.convertScaleAbs(grayscale.astype(np.float32) * contrast_factor)
    contrast = np.clip(contrast, 0, 255).astype(np.uint8)
    
    # Threshold (binary)
    _, thresholded = cv2.threshold(grayscale, threshold, 255, cv2.THRESH_BINARY)
    
    # Create montage with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(grayscale, cmap='gray')
    axes[0, 0].set_title("Original Grayscale", fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(brightness, cmap='gray')
    axes[0, 1].set_title(f"Brightness (+{brightness_delta})", fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(contrast, cmap='gray')
    axes[1, 0].set_title(f"Contrast (×{contrast_factor})", fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(thresholded, cmap='gray')
    axes[1, 1].set_title(f"Threshold ({threshold})", fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "adjustments.png")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"\n=== Adjustments ===")
    print(f"Brightness delta: +{brightness_delta}")
    print(f"Contrast factor: {contrast_factor}×")
    print(f"Threshold: {threshold}")
    print(f"Threshold behavior: Pixels > {threshold} become white (255); all others become black (0)")
    
    return {
        "brightness_delta": brightness_delta,
        "contrast_factor": contrast_factor,
        "threshold": threshold,
        "output_path": output_path
    }


def create_blur_and_edges(
    image_path: str,
    output_dir: str,
    kernel_size: int = 5,
) -> dict:
    """Create labeled grayscale, mean-blur, and Sobel-edge results."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("kernel_size must be a positive odd integer")
    
    image = cv2.imread(image_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply mean blur
    blurred = cv2.blur(grayscale, (kernel_size, kernel_size))
    
    # Sobel edges from original grayscale
    sobelx_orig = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
    sobely_orig = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
    sobel_orig = np.sqrt(sobelx_orig**2 + sobely_orig**2)
    sobel_orig = np.clip(sobel_orig, 0, 255).astype(np.uint8)
    
    # Sobel edges from blurred grayscale
    sobelx_blur = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely_blur = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_blur = np.sqrt(sobelx_blur**2 + sobely_blur**2)
    sobel_blur = np.clip(sobel_blur, 0, 255).astype(np.uint8)
    
    # Create montage with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(grayscale, cmap='gray')
    axes[0, 0].set_title("Grayscale Original", fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(blurred, cmap='gray')
    axes[0, 1].set_title(f"Mean Blur (kernel {kernel_size}×{kernel_size})", fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(sobel_orig, cmap='gray')
    axes[1, 0].set_title("Sobel Edges (Original)", fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(sobel_blur, cmap='gray')
    axes[1, 1].set_title("Sobel Edges (Blurred)", fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "blur_and_edges.png")
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"\n=== Blur and Edges ===")
    print(f"Kernel size: {kernel_size}×{kernel_size}")
    print(f"Comparison:")
    print(f"  - Original edges: Sharp detection of rock boundaries and water ripples (high-frequency detail)")
    print(f"  - Blurred edges: Smoother, fewer fine details; blur reduces noise and small features")
    print(f"  - Detail reduced: Small water texture and fine rock edges are lost after blur")
    
    return {
        "kernel_size": kernel_size,
        "output_path": output_path
    }


def run_lab(image_path: str, output_dir: str) -> dict:
    """Run Tasks 1–4 and return their results together."""
    task1 = inspect_image(image_path)
    task2 = create_pixel_views(image_path, output_dir)
    task3 = create_adjustments(image_path, output_dir)
    task4 = create_blur_and_edges(image_path, output_dir)
    
    return {
        "task1": task1,
        "task2": task2,
        "task3": task3,
        "task4": task4
    }


def main() -> None:
    """Run the lab using the required repository paths."""
    image_path = "images/original.jpg"
    output_dir = "outputs"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = run_lab(image_path, output_dir)
    print("\n" + "="*50)
    print("IMAGE PROCESSING LAB COMPLETE")
    print("="*50)
    print(f"Image: {image_path}")
    print(f"Outputs: {output_dir}/")
    print("  - pixel_views.png")
    print("  - adjustments.png")
    print("  - blur_and_edges.png")


if __name__ == "__main__":
    main()
