#!/usr/bin/env python3
"""
Improved Forensic Detector - optimized for Flux-generated images.
Based on empirical analysis of ai_generated_v2 dataset.

Key findings from analysis:
- DCT high-frequency energy: Real > Fake (most discriminative)
- Local variance: Real > Fake (more texture detail)
- Saturation: Real > Fake
- Brightness: Real < Fake

Strategy: Focus on the most discriminative features, combine with proper weighting.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict
import tempfile
import os


class ImprovedForensicDetector:
    """Optimized detector for AI-generated real estate images."""

    def __init__(self):
        pass

    def analyze(self, image_path: str) -> Dict:
        """Run all forensic analyses on an image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        results = {}

        # === CORE FEATURES (most discriminative) ===

        # 1. DCT High-Frequency Analysis (BEST discriminator)
        results["dct_hf_score"] = self._dct_high_freq_analysis(img)

        # 2. Local Variance Analysis (second best)
        results["local_variance_score"] = self._local_variance_analysis(img)

        # 3. Saturation Analysis
        results["saturation_score"] = self._saturation_analysis(img)

        # 4. Brightness Analysis
        results["brightness_score"] = self._brightness_analysis(img)

        # === SUPPORTING FEATURES ===

        # 5. Texture complexity
        results["texture_complexity_score"] = self._texture_complexity(img)

        # 6. Noise pattern analysis
        results["noise_pattern_score"] = self._noise_pattern_analysis(img)

        # 7. Gradient distribution
        results["gradient_score"] = self._gradient_distribution(img)

        # 8. Color channel consistency
        results["color_channel_score"] = self._color_channel_analysis(img)

        # === AGGREGATION ===
        # All scores are now: 0 = likely real, 1 = likely fake

        # Weights based on discriminative power from analysis
        weights = {
            "dct_hf_score": 0.25,           # Best discriminator
            "local_variance_score": 0.20,    # Second best
            "saturation_score": 0.15,        # Good discriminator
            "brightness_score": 0.10,        # Moderate
            "texture_complexity_score": 0.12,
            "noise_pattern_score": 0.08,
            "gradient_score": 0.05,
            "color_channel_score": 0.05,
        }

        results["aggregate_score"] = sum(
            results[k] * weights[k] for k in weights
        )

        return results

    def _dct_high_freq_analysis(self, img: np.ndarray) -> float:
        """
        DCT high-frequency energy analysis.

        Real images have MORE high-frequency DCT content.
        Fake images are smoother, less HF energy.

        Lower HF energy = more likely fake.
        """
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0].astype(np.float32)
        h, w = y_channel.shape

        h8, w8 = (h // 8) * 8, (w // 8) * 8
        if h8 < 16 or w8 < 16:
            return 0.5

        y_cropped = y_channel[:h8, :w8]

        hf_energies = []
        total_energies = []

        for i in range(0, h8, 8):
            for j in range(0, w8, 8):
                block = y_cropped[i:i+8, j:j+8]
                dct = cv2.dct(block)

                # High frequency: bottom-right quadrant of 8x8 DCT
                hf_energy = np.mean(np.abs(dct[4:, 4:]))
                # Total energy for normalization
                total_energy = np.mean(np.abs(dct))

                hf_energies.append(hf_energy)
                total_energies.append(total_energy)

        mean_hf = np.mean(hf_energies)

        # From analysis: Real ~1.86, Fake ~0.89
        # Score: lower HF = higher fake score
        if mean_hf < 0.5:
            score = 0.9  # Very low HF, likely fake
        elif mean_hf < 1.0:
            score = 0.7
        elif mean_hf < 1.5:
            score = 0.5
        elif mean_hf < 2.0:
            score = 0.3
        else:
            score = 0.15  # High HF, likely real

        return float(np.clip(score, 0, 1))

    def _local_variance_analysis(self, img: np.ndarray) -> float:
        """
        Local variance analysis.

        Real images have MORE local variance (more texture detail).
        Fake images tend to be smoother.

        Lower variance = more likely fake.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        kernel_size = 15
        local_mean = cv2.blur(gray, (kernel_size, kernel_size))
        local_sqr_mean = cv2.blur(gray ** 2, (kernel_size, kernel_size))
        local_var = local_sqr_mean - local_mean ** 2

        mean_local_var = np.mean(local_var)
        std_local_var = np.std(local_var)

        # From analysis: Real ~514, Fake ~412
        # Score: lower variance = higher fake score
        if mean_local_var < 300:
            score = 0.8  # Very smooth
        elif mean_local_var < 400:
            score = 0.65
        elif mean_local_var < 500:
            score = 0.45  # Borderline
        elif mean_local_var < 600:
            score = 0.3
        else:
            score = 0.15  # High variance, likely real

        # Also consider variance of variance (texture complexity)
        if std_local_var < 700:
            score = min(score + 0.1, 1.0)  # Less varied = more suspicious

        return float(np.clip(score, 0, 1))

    def _saturation_analysis(self, img: np.ndarray) -> float:
        """
        Saturation analysis.

        Real images tend to be MORE saturated.
        Fake images often have lower/inconsistent saturation.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]

        sat_mean = np.mean(saturation)
        sat_std = np.std(saturation)

        # From analysis: Real ~95, Fake ~76
        # Lower saturation = more likely fake
        if sat_mean < 60:
            score = 0.75
        elif sat_mean < 80:
            score = 0.55
        elif sat_mean < 100:
            score = 0.35
        else:
            score = 0.2

        return float(np.clip(score, 0, 1))

    def _brightness_analysis(self, img: np.ndarray) -> float:
        """
        Brightness analysis.

        Fake images tend to be BRIGHTER.
        Real: ~112, Fake: ~128
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        # Higher brightness = more likely fake
        if mean_brightness > 140:
            score = 0.7
        elif mean_brightness > 125:
            score = 0.55
        elif mean_brightness > 110:
            score = 0.4
        else:
            score = 0.25

        return float(np.clip(score, 0, 1))

    def _texture_complexity(self, img: np.ndarray) -> float:
        """
        Texture complexity using gradient analysis.

        Real images: more varied gradients
        Fake images: smoother gradients
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Sobel gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

        # Gradient statistics
        grad_mean = np.mean(gradient_mag)
        grad_std = np.std(gradient_mag)

        # Coefficient of variation of gradients
        grad_cv = grad_std / (grad_mean + 1e-10)

        # Low gradient CV = uniform gradients = suspicious
        if grad_cv < 1.5:
            score = 0.7
        elif grad_cv < 2.0:
            score = 0.5
        else:
            score = 0.3

        return float(np.clip(score, 0, 1))

    def _noise_pattern_analysis(self, img: np.ndarray) -> float:
        """
        Noise pattern analysis.

        Real images: stochastic sensor noise
        Fake images: structured/uniform noise
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Extract noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blurred

        noise_std = np.std(noise)

        # Analyze noise uniformity across regions
        h, w = noise.shape
        block_h, block_w = h // 4, w // 4

        region_stds = []
        for i in range(4):
            for j in range(4):
                if block_h > 0 and block_w > 0:
                    block = noise[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                    if block.size > 0:
                        region_stds.append(np.std(block))

        if len(region_stds) < 4:
            return 0.5

        # Coefficient of variation of regional noise stds
        cv = np.std(region_stds) / (np.mean(region_stds) + 1e-10)

        # Very uniform noise = suspicious (AI generates uniform noise)
        if cv < 0.2:
            score = 0.7  # Too uniform
        elif cv < 0.3:
            score = 0.5
        elif cv < 0.5:
            score = 0.35
        else:
            score = 0.2  # Natural variation

        # Also check absolute noise level
        if noise_std < 4:
            score = max(score, 0.6)  # Very low noise suspicious

        return float(np.clip(score, 0, 1))

    def _gradient_distribution(self, img: np.ndarray) -> float:
        """
        Gradient distribution analysis.

        Checks for unusual gradient patterns.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Edges
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        edge_density = np.mean(edges > 0)

        # From analysis: Real ~0.107, Fake ~0.096
        # Lower edge density = slightly more suspicious
        if edge_density < 0.05:
            score = 0.65
        elif edge_density < 0.08:
            score = 0.5
        elif edge_density < 0.12:
            score = 0.4
        else:
            score = 0.3

        return float(np.clip(score, 0, 1))

    def _color_channel_analysis(self, img: np.ndarray) -> float:
        """
        Color channel consistency analysis.
        """
        b, g, r = cv2.split(img)

        def get_noise_std(channel):
            blurred = cv2.GaussianBlur(channel, (5, 5), 0)
            noise = channel.astype(np.float32) - blurred.astype(np.float32)
            return np.std(noise)

        r_noise = get_noise_std(r)
        g_noise = get_noise_std(g)
        b_noise = get_noise_std(b)

        # Coefficient of variation of noise across channels
        noise_cv = np.std([r_noise, g_noise, b_noise]) / (np.mean([r_noise, g_noise, b_noise]) + 1e-10)

        if noise_cv > 0.3:
            score = 0.65  # High variation suspicious
        elif noise_cv > 0.15:
            score = 0.45
        else:
            score = 0.3

        return float(np.clip(score, 0, 1))


# Test if run directly
if __name__ == "__main__":
    import sys
    from glob import glob
    import os

    detector = ImprovedForensicDetector()
    data_dir = "data/ai_generated_v2"

    images = glob(os.path.join(data_dir, "*.png"))

    real_scores = []
    fake_scores = []

    for img_path in sorted(images):
        filename = os.path.basename(img_path)
        is_fake = "images_fake_" in filename

        try:
            results = detector.analyze(img_path)
            score = results["aggregate_score"]

            if is_fake:
                fake_scores.append(score)
            else:
                real_scores.append(score)

        except Exception as e:
            print(f"Error: {filename}: {e}")

    print("\n" + "="*60)
    print("IMPROVED DETECTOR RESULTS")
    print("="*60)
    print(f"\nReal (n={len(real_scores)}): {np.mean(real_scores):.3f} ± {np.std(real_scores):.3f}")
    print(f"Fake (n={len(fake_scores)}): {np.mean(fake_scores):.3f} ± {np.std(fake_scores):.3f}")

    # Find best threshold
    best_acc = 0
    best_thresh = 0.5

    for thresh in np.arange(0.2, 0.8, 0.01):
        real_correct = sum(1 for s in real_scores if s < thresh)
        fake_correct = sum(1 for s in fake_scores if s >= thresh)
        acc = (real_correct + fake_correct) / (len(real_scores) + len(fake_scores))

        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    print(f"\nBest threshold: {best_thresh:.2f}")
    print(f"Best accuracy:  {best_acc*100:.1f}%")
