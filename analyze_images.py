#!/usr/bin/env python3
"""Analyze specific images to understand real vs fake characteristics."""

import cv2
import numpy as np
from glob import glob
import os

def analyze_image(img_path):
    """Detailed analysis of an image."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    results = {'shape': img.shape}

    # 1. Basic stats
    results['mean_brightness'] = np.mean(gray)
    results['std_brightness'] = np.std(gray)

    # 2. FFT analysis - look at specific frequencies
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    center_h, center_w = h // 2, w // 2
    max_radius = min(h, w) // 2

    # Create distance map
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)

    # Energy in bands
    low_mask = distance < (max_radius * 0.1)
    mid_mask = (distance >= max_radius * 0.1) & (distance < max_radius * 0.4)
    high_mask = (distance >= max_radius * 0.4) & (distance < max_radius * 0.9)

    low_energy = np.mean(magnitude[low_mask])
    mid_energy = np.mean(magnitude[mid_mask])
    high_energy = np.mean(magnitude[high_mask])
    total = low_energy + mid_energy + high_energy

    results['fft_low_ratio'] = low_energy / total
    results['fft_mid_ratio'] = mid_energy / total
    results['fft_high_ratio'] = high_energy / total

    # 3. Noise analysis
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray - blurred
    results['noise_std'] = np.std(noise)
    results['noise_mean'] = np.mean(np.abs(noise))

    # Noise uniformity across regions
    region_stds = []
    block_size = h // 4
    for i in range(4):
        for j in range(4):
            block = noise[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            region_stds.append(np.std(block))
    results['noise_uniformity'] = np.std(region_stds) / (np.mean(region_stds) + 1e-10)

    # 4. Laplacian variance (sharpness)
    gray_uint8 = gray.astype(np.uint8)
    laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
    results['laplacian_var'] = laplacian.var()

    # 5. Edge density
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    results['edge_density'] = np.mean(edges > 0)

    # 6. Local variance statistics
    kernel_size = 15
    local_mean = cv2.blur(gray, (kernel_size, kernel_size))
    local_sqr_mean = cv2.blur(gray ** 2, (kernel_size, kernel_size))
    local_var = local_sqr_mean - local_mean ** 2

    results['local_var_mean'] = np.mean(local_var)
    results['local_var_std'] = np.std(local_var)
    results['smooth_ratio'] = np.mean(local_var < 50)

    # 7. DCT analysis on 8x8 blocks
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0].astype(np.float32)
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    y_cropped = y_channel[:h8, :w8]

    dct_stats = []
    for i in range(0, h8, 8):
        for j in range(0, w8, 8):
            block = y_cropped[i:i+8, j:j+8]
            dct = cv2.dct(block)
            # High frequency energy (bottom-right of DCT block)
            hf_energy = np.mean(np.abs(dct[4:, 4:]))
            dct_stats.append(hf_energy)

    results['dct_hf_mean'] = np.mean(dct_stats)
    results['dct_hf_std'] = np.std(dct_stats)

    # 8. Color saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    results['sat_mean'] = np.mean(saturation)
    results['sat_std'] = np.std(saturation)

    return results

def main():
    data_dir = "data/ai_generated_v2"
    images = glob(os.path.join(data_dir, "*.png"))

    real_stats = {}
    fake_stats = {}

    for img_path in sorted(images):
        filename = os.path.basename(img_path)
        is_fake = "images_fake_" in filename

        results = analyze_image(img_path)
        if results is None:
            continue

        target = fake_stats if is_fake else real_stats
        for k, v in results.items():
            if k == 'shape':
                continue
            if k not in target:
                target[k] = []
            target[k].append(v)

    print("\n" + "="*70)
    print("DETAILED FEATURE COMPARISON: REAL vs FAKE")
    print("="*70)

    # Sort by absolute difference
    features = []
    for k in real_stats.keys():
        real_mean = np.mean(real_stats[k])
        fake_mean = np.mean(fake_stats[k])
        diff = fake_mean - real_mean
        sep = abs(diff) / (np.std(real_stats[k]) + np.std(fake_stats[k]) + 1e-10)
        features.append((k, real_mean, fake_mean, diff, sep))

    features.sort(key=lambda x: -abs(x[4]))  # Sort by separation

    for k, real_mean, fake_mean, diff, sep in features:
        print(f"\n{k}:")
        print(f"  Real: {real_mean:.4f} ± {np.std(real_stats[k]):.4f}")
        print(f"  Fake: {fake_mean:.4f} ± {np.std(fake_stats[k]):.4f}")
        print(f"  Diff: {diff:+.4f}  |  Separation: {sep:.3f}")

if __name__ == "__main__":
    main()
