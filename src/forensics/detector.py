"""
Module 1: Forensic Signal Detector
Pixel-level analysis for detecting AI manipulation
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict
import tempfile
import os


class ForensicDetector:
    """Detects low-level technical anomalies in images."""

    def __init__(self):
        self.ela_quality = 90  # JPEG quality for ELA

    def analyze(self, image_path: str) -> Dict:
        """Run all forensic analyses on an image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        results = {
            "fft_score": self._fft_analysis(img),
            "ela_score": self._ela_analysis(image_path),
            "noise_score": self._noise_analysis(img),
            "texture_score": self._texture_consistency(img),
            "compression_score": self._compression_analysis(image_path),
            "edge_score": self._edge_coherence(img),
            "sharpness_score": self._sharpness_analysis(img),
        }

        # Aggregate forensic score (0 = real, 1 = fake)
        # Weights tuned based on discrimination analysis:
        # - FFT/ELA/compression: poor discrimination, low weight
        # - Noise: best discriminator for blur/smooth manipulations
        # - Sharpness: good for oversharpened images
        # - Texture: moderate discrimination
        weights = {
            "fft_score": 0.02,       # Poor discriminator (always ~0.87)
            "ela_score": 0.05,       # Poor discriminator (always ~0.6)
            "noise_score": 0.40,     # Best discriminator - increased
            "texture_score": 0.15,
            "compression_score": 0.03,
            "edge_score": 0.10,
            "sharpness_score": 0.25, # Good for oversharpened
        }

        results["aggregate_score"] = sum(
            results[k] * weights[k] for k in weights
        )

        return results

    def _fft_analysis(self, img: np.ndarray) -> float:
        """
        FFT analysis to detect GAN/diffusion artifacts.
        Look for periodic patterns and anomalous frequency distributions.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shift) + 1)

        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2

        # Check for periodic artifacts (common in GANs)
        # Look for unusual peaks in the frequency domain

        # Normalize magnitude
        mag_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-10)

        # Exclude DC component area
        dc_radius = min(h, w) // 20
        y, x = np.ogrid[:h, :w]
        dc_mask = ((y - center_h) ** 2 + (x - center_w) ** 2) > dc_radius ** 2

        # Find peaks (potential periodic artifacts)
        mag_masked = mag_norm * dc_mask
        threshold = np.percentile(mag_masked[dc_mask], 99)
        peak_count = np.sum(mag_masked > threshold)

        # Analyze radial symmetry (AI often has more symmetric patterns)
        angles = np.linspace(0, 2 * np.pi, 36)
        radii = np.linspace(dc_radius, min(h, w) // 2, 20)
        radial_profile = []

        for r in radii:
            ring_values = []
            for angle in angles:
                y_coord = int(center_h + r * np.sin(angle))
                x_coord = int(center_w + r * np.cos(angle))
                if 0 <= y_coord < h and 0 <= x_coord < w:
                    ring_values.append(mag_norm[y_coord, x_coord])
            if ring_values:
                radial_profile.append(np.std(ring_values))

        # Higher symmetry (lower variance in radial profile) is suspicious
        if radial_profile:
            symmetry_score = 1.0 - np.clip(np.mean(radial_profile) * 5, 0, 1)
        else:
            symmetry_score = 0.5

        # Combine metrics
        peak_score = np.clip(peak_count / 500, 0, 1)  # More peaks = more suspicious

        score = 0.5 * peak_score + 0.5 * symmetry_score

        return float(np.clip(score, 0, 1))

    def _ela_analysis(self, image_path: str) -> float:
        """
        Error Level Analysis - detects areas with different compression levels.
        Spliced/inpainted regions often have different error levels.
        """
        # Load original
        original = Image.open(image_path).convert('RGB')

        # Resave at known quality
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
            original.save(tmp_path, 'JPEG', quality=self.ela_quality)
            resaved = Image.open(tmp_path)

        # Calculate difference
        orig_arr = np.array(original, dtype=np.float32)
        resaved_arr = np.array(resaved, dtype=np.float32)

        ela = np.abs(orig_arr - resaved_arr)

        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

        # Analyze ELA by regions
        h, w = ela.shape[:2]
        block_size = 64
        region_scores = []

        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                region = ela[i:i + block_size, j:j + block_size]
                region_scores.append(np.mean(region))

        if len(region_scores) < 4:
            return 0.5

        # High variance between regions suggests manipulation
        ela_variance = np.std(region_scores) / (np.mean(region_scores) + 1e-10)

        # Also check for unusually high ELA values
        high_ela_ratio = np.mean(ela > 20)

        # Combine metrics
        variance_score = np.clip(ela_variance / 0.5, 0, 1)
        high_ela_score = np.clip(high_ela_ratio * 10, 0, 1)

        score = 0.6 * variance_score + 0.4 * high_ela_score

        return float(np.clip(score, 0, 1))

    def _noise_analysis(self, img: np.ndarray) -> float:
        """
        Analyze noise patterns - AI images often have unnatural noise.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Extract noise using high-pass filter
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blurred

        # Analyze noise statistics
        noise_std = np.std(noise)

        # Check for noise uniformity across image regions
        h, w = noise.shape
        regions = [
            noise[:h // 2, :w // 2],
            noise[:h // 2, w // 2:],
            noise[h // 2:, :w // 2],
            noise[h // 2:, w // 2:]
        ]

        region_stds = [np.std(r) for r in regions]
        std_variance = np.std(region_stds)
        std_mean = np.mean(region_stds)

        # Very uniform noise across regions is suspicious (AI images)
        # Coefficient of variation of region stds
        cv = std_variance / (std_mean + 1e-10)
        uniformity_score = 1 - np.clip(cv * 3, 0, 1)

        # Check noise magnitude - too low suggests heavy processing
        noise_magnitude_score = 0
        if noise_std < 2.5:
            noise_magnitude_score = 0.8  # Very smooth = suspicious
        elif noise_std < 5:
            noise_magnitude_score = 0.4
        elif noise_std > 20:
            noise_magnitude_score = 0.3  # Very noisy might be fake too

        # Check for noise coherence using autocorrelation
        sample = noise[:min(256, h), :min(256, w)]
        autocorr = np.abs(np.fft.ifft2(np.abs(np.fft.fft2(sample)) ** 2))
        autocorr_score = np.clip(autocorr[1, 1] / (autocorr[0, 0] + 1e-10) * 5, 0, 1)

        score = 0.4 * uniformity_score + 0.3 * noise_magnitude_score + 0.3 * autocorr_score

        return float(np.clip(score, 0, 1))

    def _texture_consistency(self, img: np.ndarray) -> float:
        """
        Check for unnatural smoothness in textures.
        AI often produces overly smooth surfaces.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate local variance using sliding window
        kernel_size = 15
        local_mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        local_sqr_mean = cv2.blur((gray.astype(np.float32)) ** 2, (kernel_size, kernel_size))
        local_var = local_sqr_mean - local_mean ** 2

        # Find smooth regions (low variance)
        smooth_threshold = 50  # Lowered threshold
        smooth_ratio = np.mean(local_var < smooth_threshold)

        # Calculate gradient magnitude for edge analysis
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

        # Low gradient magnitude overall suggests artificial smoothing
        gradient_mean = np.mean(gradient_mag)
        gradient_score = 1 - np.clip(gradient_mean / 30, 0, 1)

        # Combine smooth ratio and gradient analysis
        smooth_score = np.clip((smooth_ratio - 0.2) / 0.5, 0, 1)

        score = 0.5 * smooth_score + 0.5 * gradient_score

        return float(np.clip(score, 0, 1))

    def _compression_analysis(self, image_path: str) -> float:
        """
        Detect compression inconsistencies from splicing.
        """
        img = cv2.imread(image_path)

        # Convert to YCrCb and analyze DCT blocks
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0].astype(np.float32)

        # Analyze 8x8 block boundaries (JPEG artifacts)
        h, w = y_channel.shape
        h8, w8 = (h // 8) * 8, (w // 8) * 8
        if h8 < 16 or w8 < 16:
            return 0.5

        y_cropped = y_channel[:h8, :w8]

        # Calculate block boundary differences
        boundary_diffs = []
        inside_diffs = []

        for i in range(0, h8 - 8, 8):
            for j in range(0, w8 - 8, 8):
                # Horizontal boundary difference
                boundary_diffs.append(abs(float(y_cropped[i + 7, j + 4]) - float(y_cropped[i + 8, j + 4])))
                inside_diffs.append(abs(float(y_cropped[i + 3, j + 4]) - float(y_cropped[i + 4, j + 4])))

        if not boundary_diffs or not inside_diffs:
            return 0.5

        # Compare boundary vs inside differences
        boundary_mean = np.mean(boundary_diffs)
        inside_mean = np.mean(inside_diffs)

        # Ratio of boundary to inside differences
        if inside_mean > 0:
            ratio = boundary_mean / inside_mean
            # Values far from 1.0 suggest compression inconsistencies
            inconsistency_score = np.clip(abs(ratio - 1.0) * 2, 0, 1)
        else:
            inconsistency_score = 0.5

        # Check variance of block differences
        diff_variance = np.std(boundary_diffs) / (np.mean(boundary_diffs) + 1e-10)
        variance_score = np.clip(diff_variance, 0, 1)

        score = 0.5 * inconsistency_score + 0.5 * variance_score

        return float(np.clip(score, 0, 1))

    def _edge_coherence(self, img: np.ndarray) -> float:
        """
        Check edge coherence - AI images often have inconsistent edges.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect edges using Canny
        edges = cv2.Canny(gray, 50, 150)

        # Calculate edge density
        edge_density = np.mean(edges > 0)

        # Very low or very high edge density is suspicious
        if edge_density < 0.02:
            density_score = 0.7  # Too few edges - over-smoothed
        elif edge_density > 0.25:
            density_score = 0.6  # Too many edges - over-sharpened
        else:
            density_score = 0.3  # Normal range

        # Check edge continuity using Hough lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

        if lines is not None and len(lines) > 0:
            # Calculate line statistics
            line_lengths = [np.sqrt((l[0][2] - l[0][0]) ** 2 + (l[0][3] - l[0][1]) ** 2) for l in lines]
            avg_length = np.mean(line_lengths)

            # Very uniform line lengths might indicate artificial generation
            length_variance = np.std(line_lengths) / (avg_length + 1e-10)
            continuity_score = 1 - np.clip(length_variance, 0, 1)
        else:
            continuity_score = 0.5

        score = 0.5 * density_score + 0.5 * continuity_score

        return float(np.clip(score, 0, 1))

    def _sharpness_analysis(self, img: np.ndarray) -> float:
        """
        Detect oversharpening and overblurring artifacts.
        Uses Laplacian variance and morphological gradient.

        Based on empirical analysis:
        - Real photos: lap_var=400-1500, grad_mean=13-25
        - Blur/smooth: lap_var=9-14, grad_mean=7-11
        - Oversharp: lap_var=2500-12000+, grad_mean=30-75
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Laplacian variance - measures sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = laplacian.var()

        # Score based on Laplacian variance
        if lap_var > 3500:
            sharpness_score = 0.95  # Very oversharpened
        elif lap_var > 2200:
            sharpness_score = 0.80  # Oversharpened
        elif lap_var > 1600:
            sharpness_score = 0.45  # Upper normal range
        elif lap_var < 30:
            sharpness_score = 0.75  # Very blurry (heavily processed)
        elif lap_var < 100:
            sharpness_score = 0.55  # Blurry
        else:
            sharpness_score = 0.20  # Normal range (300-1600)

        # Morphological gradient - detects halos from oversharpening
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        grad_mean = np.mean(gradient)

        # Gradient-based score
        if grad_mean > 35:
            halo_score = 0.90  # Strong oversharpening halos
        elif grad_mean > 27:
            halo_score = 0.70  # Moderate oversharpening
        elif grad_mean < 12:
            halo_score = 0.60  # Too smooth (blur artifacts)
        else:
            halo_score = 0.25  # Normal range

        score = 0.55 * sharpness_score + 0.45 * halo_score

        return float(np.clip(score, 0, 1))
