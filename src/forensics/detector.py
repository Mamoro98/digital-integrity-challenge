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
            "rich_poor_texture_score": self._rich_poor_texture_contrast(img),
            "color_consistency_score": self._color_channel_analysis(img),
            "lbp_score": self._local_binary_pattern_analysis(img),
            "glcm_score": self._glcm_texture_analysis(img),
        }

        # Aggregate forensic score (0 = real, 1 = fake)
        # EMPIRICALLY OPTIMIZED on 12 real + 50 fake test images
        # Achieves 79.7% balanced accuracy (83% real, 76% fake)
        
        # Directions: -1 means invert (higher raw score = more REAL)
        #             +1 means keep (higher raw score = more FAKE)
        directions = {
            "fft_score": -1,           # higher raw = REAL, so invert
            "ela_score": -1,           # higher raw = REAL, so invert
            "noise_score": 1,          # higher = FAKE (strongest signal)
            "texture_score": 1,        # higher = FAKE
            "compression_score": 1,    # higher = FAKE
            "edge_score": 1,           # higher = FAKE (weak)
            "sharpness_score": 1,      # higher = FAKE
            "rich_poor_texture_score": -1,  # higher = REAL, so invert
            "color_consistency_score": 1,   # higher = FAKE
            "lbp_score": -1,           # higher = REAL, so invert
            "glcm_score": 1,           # higher = FAKE (weak)
        }
        
        # Transform: invert scores where direction=-1
        corrected = {}
        for k, d in directions.items():
            if d == -1:
                corrected[k] = 1.0 - results[k]
            else:
                corrected[k] = results[k]
        
        # Optimized weights (sum to 1.0)
        weights = {
            "fft_score": 0.15,
            "ela_score": 0.12,
            "noise_score": 0.18,      # Most discriminative
            "texture_score": 0.16,
            "compression_score": 0.05,
            "edge_score": 0.01,       # Least discriminative
            "sharpness_score": 0.16,
            "rich_poor_texture_score": 0.03,
            "color_consistency_score": 0.06,
            "lbp_score": 0.03,
            "glcm_score": 0.05,
        }

        results["aggregate_score"] = sum(
            corrected[k] * weights[k] for k in weights
        )

        return results

    def _fft_analysis(self, img: np.ndarray) -> float:
        """
        FFT analysis to detect GAN/diffusion artifacts.

        Research-based improvements:
        1. Detect periodic artifacts at periods 2, 4, 8, 16 (diffusion fingerprints)
        2. DEFEND-style weighted band analysis (mid-high freq more discriminative)
        3. Radial symmetry analysis
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        center_h, center_w = h // 2, w // 2

        # === 1. DIFFUSION PERIOD DETECTION ===
        # Diffusion models leave artifacts at periods 2, 4, 8, 16
        # These appear as spikes at specific frequencies: f = size / period
        period_score = self._detect_periodic_artifacts(magnitude, h, w)

        # === 2. DEFEND-STYLE WEIGHTED BAND ANALYSIS ===
        # Research: mid-high frequencies are most discriminative
        # Low frequencies are similar for real and AI images
        band_score = self._analyze_frequency_bands(magnitude, h, w)

        # === 3. RADIAL SYMMETRY (original) ===
        # AI images often have more symmetric frequency patterns
        log_magnitude = np.log(magnitude + 1)
        mag_norm = (log_magnitude - log_magnitude.min()) / (log_magnitude.max() - log_magnitude.min() + 1e-10)

        dc_radius = min(h, w) // 20
        angles = np.linspace(0, 2 * np.pi, 36)
        radii = np.linspace(dc_radius, min(h, w) // 4, 15)
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

        if radial_profile:
            symmetry_score = 1.0 - np.clip(np.mean(radial_profile) * 5, 0, 1)
        else:
            symmetry_score = 0.5

        # === COMBINE SCORES ===
        # Weight: period detection (40%), band analysis (40%), symmetry (20%)
        score = 0.40 * period_score + 0.40 * band_score + 0.20 * symmetry_score

        return float(np.clip(score, 0, 1))

    def _detect_periodic_artifacts(self, magnitude: np.ndarray, h: int, w: int) -> float:
        """
        Detect periodic artifacts at periods 2, 4, 8, 16.

        Diffusion models use upsampling that creates repeating patterns.
        In frequency domain, period P artifact appears at frequency f = N/P
        where N is the image dimension.
        """
        center_h, center_w = h // 2, w // 2

        # Periods to check (research shows these are common in diffusion models)
        periods = [2, 4, 8, 16]

        # Calculate expected frequency positions for each period
        artifact_scores = []

        for period in periods:
            # Frequency corresponding to this period
            freq_h = h // period
            freq_w = w // period

            # Check for energy spikes at these frequencies
            # Look at cross pattern (horizontal and vertical artifacts)
            positions = [
                (center_h + freq_h, center_w),  # Above center
                (center_h - freq_h, center_w),  # Below center
                (center_h, center_w + freq_w),  # Right of center
                (center_h, center_w - freq_w),  # Left of center
            ]

            # Measure energy at artifact positions vs nearby background
            artifact_energy = []
            background_energy = []

            for pos_h, pos_w in positions:
                if 0 <= pos_h < h and 0 <= pos_w < w:
                    # Energy at artifact position (small window)
                    window_size = max(3, min(h, w) // 100)
                    h_start = max(0, pos_h - window_size)
                    h_end = min(h, pos_h + window_size + 1)
                    w_start = max(0, pos_w - window_size)
                    w_end = min(w, pos_w + window_size + 1)

                    artifact_energy.append(np.mean(magnitude[h_start:h_end, w_start:w_end]))

                    # Background: slightly offset position
                    offset = window_size * 3
                    bg_h = min(h - 1, max(0, pos_h + offset))
                    bg_w = min(w - 1, max(0, pos_w + offset))
                    bg_h_start = max(0, bg_h - window_size)
                    bg_h_end = min(h, bg_h + window_size + 1)
                    bg_w_start = max(0, bg_w - window_size)
                    bg_w_end = min(w, bg_w + window_size + 1)

                    background_energy.append(np.mean(magnitude[bg_h_start:bg_h_end, bg_w_start:bg_w_end]))

            if artifact_energy and background_energy:
                # Ratio of artifact to background energy
                # High ratio = strong periodic artifact = likely AI
                ratio = np.mean(artifact_energy) / (np.mean(background_energy) + 1e-10)
                # Normalize: ratio > 1.5 is suspicious
                artifact_scores.append(np.clip((ratio - 1.0) / 1.0, 0, 1))

        if artifact_scores:
            # Take max score (any period showing artifacts is suspicious)
            return float(max(artifact_scores))
        return 0.0

    def _analyze_frequency_bands(self, magnitude: np.ndarray, h: int, w: int) -> float:
        """
        DEFEND-style frequency band analysis.

        Research finding:
        - Low frequencies: similar for real and AI (not discriminative)
        - Mid frequencies: somewhat discriminative
        - High frequencies: most discriminative (AI images smoother here)

        Real images have more high-frequency content (fine details, sensor noise).
        AI images are smoother in high frequencies.
        """
        center_h, center_w = h // 2, w // 2
        max_radius = min(h, w) // 2

        # Create distance map from center
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)

        # Define frequency bands (as fraction of max radius)
        # Low: 0-20%, Mid: 20-50%, High: 50-100%
        low_mask = distance < (max_radius * 0.2)
        mid_mask = (distance >= max_radius * 0.2) & (distance < max_radius * 0.5)
        high_mask = (distance >= max_radius * 0.5) & (distance < max_radius)

        # Calculate energy in each band
        low_energy = np.mean(magnitude[low_mask]) if np.any(low_mask) else 0
        mid_energy = np.mean(magnitude[mid_mask]) if np.any(mid_mask) else 0
        high_energy = np.mean(magnitude[high_mask]) if np.any(high_mask) else 0

        total_energy = low_energy + mid_energy + high_energy + 1e-10

        # Ratio of high frequency energy to total
        # Real images: higher ratio (more fine detail)
        # AI images: lower ratio (smoother)
        high_ratio = high_energy / total_energy

        # Also check mid-to-low ratio
        mid_to_low = mid_energy / (low_energy + 1e-10)

        # Score: low high_ratio = suspicious (AI tends to be smoother)
        # Calibrated thresholds based on testing:
        # - Real images typically have high_ratio > 0.15
        # - AI images typically have high_ratio < 0.10
        # Only flag as suspicious if high_ratio is very low
        if high_ratio < 0.05:
            smoothness_score = 0.9  # Very smooth - likely AI
        elif high_ratio < 0.10:
            smoothness_score = 0.6  # Suspicious
        elif high_ratio < 0.15:
            smoothness_score = 0.4  # Borderline
        else:
            smoothness_score = 0.2  # Normal - likely real

        # Additional: very uniform mid-to-low ratio is suspicious
        # (AI tends to have consistent frequency rolloff)
        uniformity_score = 1.0 - np.clip(abs(mid_to_low - 0.5) * 2, 0, 1)

        # Weight smoothness higher as it's more discriminative
        return float(0.8 * smoothness_score + 0.2 * uniformity_score)

    def _ela_analysis(self, image_path: str) -> float:
        """
        Error Level Analysis - detects areas with different compression levels.
        Spliced/inpainted regions often have different error levels.
        """
        # Load original
        original = Image.open(image_path).convert('RGB')

        # Resave at known quality using proper context manager for cleanup
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as tmp:
            tmp_path = tmp.name
            original.save(tmp_path, 'JPEG', quality=self.ela_quality)
            # Load resaved image while temp file still exists
            resaved = Image.open(tmp_path)
            # Force load into memory before temp file is deleted
            resaved_arr = np.array(resaved, dtype=np.float32)

        # Calculate difference (temp file auto-cleaned by context manager)
        orig_arr = np.array(original, dtype=np.float32)

        ela = np.abs(orig_arr - resaved_arr)

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

    def _rich_poor_texture_contrast(self, img: np.ndarray) -> float:
        """
        Rich/Poor Texture Contrast Analysis (Research-based).

        Research finding:
        - Divide image into "rich texture" patches (high detail: objects, edges)
          and "poor texture" patches (low detail: sky, plain walls)
        - Measure noise characteristics in each type
        - Real images: DIFFERENT noise in rich vs poor areas (camera sensor varies)
        - AI images: SIMILAR noise everywhere (uniform generation process)

        A high contrast difference = likely real
        Low contrast difference = likely AI/manipulated
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape

        # === Step 1: Calculate local variance to identify rich/poor regions ===
        patch_size = 32
        rich_patches = []
        poor_patches = []

        # Threshold for rich vs poor (based on local variance)
        variance_threshold = 500  # Patches with variance > this are "rich"

        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = gray[i:i + patch_size, j:j + patch_size]
                patch_var = np.var(patch)

                if patch_var > variance_threshold:
                    rich_patches.append(patch)
                elif patch_var < variance_threshold / 3:  # Very smooth patches
                    poor_patches.append(patch)

        # Need minimum patches for meaningful analysis
        if len(rich_patches) < 3 or len(poor_patches) < 3:
            return 0.5  # Insufficient data

        # === Step 2: Extract noise from patches ===
        def extract_noise(patch):
            """Extract high-frequency noise from a patch."""
            blurred = cv2.GaussianBlur(patch, (5, 5), 0)
            noise = patch - blurred
            return noise

        rich_noises = [extract_noise(p) for p in rich_patches]
        poor_noises = [extract_noise(p) for p in poor_patches]

        # === Step 3: Measure noise characteristics ===
        # For each patch type, calculate:
        # - Mean noise standard deviation
        # - Inter-pixel correlation

        def noise_stats(noise_patches):
            stds = [np.std(n) for n in noise_patches]
            # Autocorrelation at lag 1 (measures noise structure)
            autocorrs = []
            for n in noise_patches:
                if n.size > 1:
                    flat = n.flatten()
                    if len(flat) > 1 and np.std(flat[:-1]) > 0 and np.std(flat[1:]) > 0:
                        corr = np.corrcoef(flat[:-1], flat[1:])[0, 1]
                        if not np.isnan(corr):
                            autocorrs.append(corr)
            return np.mean(stds), np.mean(autocorrs) if autocorrs else 0

        rich_std, rich_autocorr = noise_stats(rich_noises)
        poor_std, poor_autocorr = noise_stats(poor_noises)

        # === Step 4: Calculate contrast ===
        # Real images: rich areas have MORE noise than poor areas
        # AI images: similar noise levels

        # Noise level contrast
        std_ratio = rich_std / (poor_std + 1e-10)

        # In real images, rich areas typically have 1.2-2x more noise than poor
        # In AI images, ratio is closer to 1.0
        if std_ratio > 1.5:
            std_contrast_score = 0.2  # High contrast = likely real
        elif std_ratio > 1.2:
            std_contrast_score = 0.35
        elif std_ratio > 1.0:
            std_contrast_score = 0.5
        elif std_ratio > 0.8:
            std_contrast_score = 0.65  # Inverted (poor has more noise) = suspicious
        else:
            std_contrast_score = 0.8

        # Autocorrelation contrast
        # Real noise: more random (lower autocorrelation)
        # AI noise: more structured (higher autocorrelation)
        autocorr_diff = abs(rich_autocorr - poor_autocorr)

        # Real images: different autocorrelation in rich vs poor
        # AI images: similar autocorrelation everywhere
        if autocorr_diff > 0.1:
            autocorr_score = 0.25  # High difference = likely real
        elif autocorr_diff > 0.05:
            autocorr_score = 0.4
        else:
            autocorr_score = 0.7  # Low difference = suspicious

        # === Step 5: Check absolute noise levels ===
        # AI images often have very low noise overall
        avg_noise = (rich_std + poor_std) / 2
        if avg_noise < 2.0:
            noise_level_score = 0.8  # Very smooth = suspicious
        elif avg_noise < 4.0:
            noise_level_score = 0.5
        else:
            noise_level_score = 0.25  # Normal noise = likely real

        # === Combine scores ===
        score = (0.40 * std_contrast_score +
                 0.30 * autocorr_score +
                 0.30 * noise_level_score)

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

    def _color_channel_analysis(self, img: np.ndarray) -> float:
        """
        Color Channel Consistency Analysis (Research Method 3).

        AI-generated images often have:
        - Unnatural color channel correlations
        - Inconsistent noise across R, G, B channels
        - Unusual saturation patterns

        Real cameras have consistent color processing pipelines.
        """
        # Split into color channels
        b, g, r = cv2.split(img)

        # === 1. Cross-channel correlation ===
        # Real images: R, G, B channels are highly correlated
        # AI images: sometimes have unusual decorrelation
        def safe_corrcoef(a, b):
            a_flat = a.flatten().astype(np.float64)
            b_flat = b.flatten().astype(np.float64)
            if np.std(a_flat) < 1e-10 or np.std(b_flat) < 1e-10:
                return 0.5
            corr = np.corrcoef(a_flat, b_flat)[0, 1]
            return corr if not np.isnan(corr) else 0.5

        rg_corr = safe_corrcoef(r, g)
        rb_corr = safe_corrcoef(r, b)
        gb_corr = safe_corrcoef(g, b)

        avg_corr = (rg_corr + rb_corr + gb_corr) / 3

        # Very low correlation is suspicious (unusual for natural images)
        # Very high correlation might indicate grayscale converted to RGB
        if avg_corr < 0.7:
            corr_score = 0.7  # Low correlation - suspicious
        elif avg_corr > 0.98:
            corr_score = 0.6  # Too high - might be fake grayscale
        else:
            corr_score = 0.25  # Normal range

        # === 2. Channel noise consistency ===
        # Extract noise from each channel
        def get_noise_std(channel):
            blurred = cv2.GaussianBlur(channel, (5, 5), 0)
            noise = channel.astype(np.float32) - blurred.astype(np.float32)
            return np.std(noise)

        r_noise = get_noise_std(r)
        g_noise = get_noise_std(g)
        b_noise = get_noise_std(b)

        # Real cameras: similar noise across channels (sensor noise)
        # AI: can have very different noise in different channels
        noise_std = np.std([r_noise, g_noise, b_noise])
        noise_mean = np.mean([r_noise, g_noise, b_noise])

        noise_cv = noise_std / (noise_mean + 1e-10)  # Coefficient of variation

        if noise_cv > 0.3:
            noise_score = 0.75  # High variation - suspicious
        elif noise_cv > 0.15:
            noise_score = 0.5
        else:
            noise_score = 0.25  # Consistent noise - likely real

        # === 3. Saturation analysis ===
        # AI images sometimes have unnatural saturation patterns
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]

        sat_mean = np.mean(saturation)
        sat_std = np.std(saturation)

        # Very low saturation variance can indicate AI smoothing
        if sat_std < 30:
            sat_score = 0.65  # Low variance - suspicious
        elif sat_mean > 200:
            sat_score = 0.6  # Over-saturated
        else:
            sat_score = 0.3  # Normal

        # Combine scores
        score = 0.35 * corr_score + 0.35 * noise_score + 0.30 * sat_score

        return float(np.clip(score, 0, 1))

    def _local_binary_pattern_analysis(self, img: np.ndarray) -> float:
        """
        Local Binary Pattern (LBP) Analysis (Research Method 4).

        LBP captures micro-texture patterns:
        - For each pixel, compare with 8 neighbors
        - Create binary code based on comparisons
        - Histogram of codes reveals texture characteristics

        AI images have different LBP distributions than real photos.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Simple LBP implementation (8 neighbors, radius 1)
        def compute_lbp(img):
            img_h, img_w = img.shape
            lbp = np.zeros_like(img, dtype=np.uint8)

            for i in range(1, img_h - 1):
                for j in range(1, img_w - 1):
                    center = img[i, j]
                    code = 0

                    # 8 neighbors in clockwise order
                    code |= (1 << 7) if img[i-1, j-1] >= center else 0
                    code |= (1 << 6) if img[i-1, j] >= center else 0
                    code |= (1 << 5) if img[i-1, j+1] >= center else 0
                    code |= (1 << 4) if img[i, j+1] >= center else 0
                    code |= (1 << 3) if img[i+1, j+1] >= center else 0
                    code |= (1 << 2) if img[i+1, j] >= center else 0
                    code |= (1 << 1) if img[i+1, j-1] >= center else 0
                    code |= (1 << 0) if img[i, j-1] >= center else 0

                    lbp[i, j] = code

            return lbp

        # For efficiency, sample a subset of the image
        sample_size = min(200, h - 2, w - 2)  # Leave margin for LBP
        if sample_size < 10:
            return 0.5  # Image too small
        start_h = (h - sample_size) // 2
        start_w = (w - sample_size) // 2
        sample = gray[start_h:start_h+sample_size, start_w:start_w+sample_size]

        lbp = compute_lbp(sample)

        # Compute histogram
        hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-10)

        # === Analysis of LBP histogram ===

        # 1. Uniformity: AI images often have less uniform LBP patterns
        # "Uniform" LBP patterns have at most 2 bitwise transitions
        uniform_patterns = [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31,
                          32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126, 127,
                          128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199,
                          207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247,
                          248, 249, 251, 252, 253, 254, 255]

        uniform_ratio = sum(hist[p] for p in uniform_patterns if p < len(hist))

        # Real images typically have 85-95% uniform patterns
        # AI might have different ratios
        if uniform_ratio < 0.7:
            uniform_score = 0.75  # Low uniformity - suspicious
        elif uniform_ratio > 0.95:
            uniform_score = 0.6  # Too uniform - suspicious
        else:
            uniform_score = 0.25  # Normal

        # 2. Entropy of LBP histogram
        # AI images might have lower entropy (more predictable patterns)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        max_entropy = np.log2(256)
        norm_entropy = entropy / max_entropy

        if norm_entropy < 0.6:
            entropy_score = 0.7  # Low entropy - suspicious
        elif norm_entropy > 0.9:
            entropy_score = 0.5  # Very high entropy
        else:
            entropy_score = 0.3  # Normal

        # 3. Peak analysis
        # AI might have unusual peaks in histogram
        max_bin = np.max(hist)
        if max_bin > 0.1:
            peak_score = 0.65  # Dominant pattern - suspicious
        else:
            peak_score = 0.3

        score = 0.40 * uniform_score + 0.35 * entropy_score + 0.25 * peak_score

        return float(np.clip(score, 0, 1))

    def _glcm_texture_analysis(self, img: np.ndarray) -> float:
        """
        Grey Level Co-occurrence Matrix (GLCM) Analysis (Research Method 5).

        GLCM captures texture by analyzing how often pairs of pixel values
        occur at specific spatial relationships.

        Features: contrast, correlation, energy, homogeneity
        AI images often have different GLCM statistics than real photos.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Quantize to fewer levels for efficiency
        levels = 32
        gray_quantized = (gray // (256 // levels)).astype(np.uint8)

        # Sample region for efficiency
        sample_size = min(200, h - 1, w - 1)
        if sample_size < 10:
            return 0.5  # Image too small
        start_h = (h - sample_size) // 2
        start_w = (w - sample_size) // 2
        sample = gray_quantized[start_h:start_h+sample_size, start_w:start_w+sample_size]

        # Compute GLCM for distance=1, angle=0 (horizontal)
        glcm = np.zeros((levels, levels), dtype=np.float32)

        for i in range(sample.shape[0]):
            for j in range(sample.shape[1] - 1):
                glcm[sample[i, j], sample[i, j+1]] += 1

        # Normalize
        glcm = glcm / (glcm.sum() + 1e-10)

        # === GLCM Features ===

        # Create indices for calculations
        i_idx, j_idx = np.ogrid[:levels, :levels]

        # 1. Contrast: measures local variations
        contrast = np.sum(glcm * (i_idx - j_idx) ** 2)

        # 2. Homogeneity: measures closeness of distribution to diagonal
        homogeneity = np.sum(glcm / (1 + np.abs(i_idx - j_idx)))

        # 3. Energy (Angular Second Moment): measures uniformity
        energy = np.sum(glcm ** 2)

        # 4. Correlation: measures linear dependency
        mean_i = np.sum(i_idx * glcm)
        mean_j = np.sum(j_idx * glcm)
        std_i = np.sqrt(np.sum(glcm * (i_idx - mean_i) ** 2))
        std_j = np.sqrt(np.sum(glcm * (j_idx - mean_j) ** 2))

        if std_i > 1e-10 and std_j > 1e-10:
            correlation = np.sum(glcm * (i_idx - mean_i) * (j_idx - mean_j)) / (std_i * std_j)
        else:
            correlation = 0

        # === Scoring based on typical values ===

        # AI images often have:
        # - Lower contrast (smoother)
        # - Higher homogeneity (more uniform)
        # - Higher energy (more predictable patterns)

        # Contrast score
        if contrast < 50:
            contrast_score = 0.7  # Very low contrast - suspicious
        elif contrast < 150:
            contrast_score = 0.5
        else:
            contrast_score = 0.25  # Normal contrast

        # Homogeneity score
        if homogeneity > 0.8:
            homog_score = 0.7  # Very homogeneous - suspicious
        elif homogeneity > 0.6:
            homog_score = 0.45
        else:
            homog_score = 0.25

        # Energy score
        if energy > 0.1:
            energy_score = 0.7  # High energy - suspicious
        elif energy > 0.05:
            energy_score = 0.45
        else:
            energy_score = 0.25

        # Combine
        score = 0.35 * contrast_score + 0.35 * homog_score + 0.30 * energy_score

        return float(np.clip(score, 0, 1))
