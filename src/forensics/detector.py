"""
Module 1: Forensic Signal Detector
Pixel-level analysis for detecting AI manipulation
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple
import scipy.fftpack as fft


class ForensicDetector:
    """Detects low-level technical anomalies in images."""
    
    def __init__(self):
        self.ela_quality = 90  # JPEG quality for ELA
        
    def analyze(self, image_path: str) -> Dict:
        """Run all forensic analyses on an image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = {
            "fft_score": self._fft_analysis(img),
            "ela_score": self._ela_analysis(image_path),
            "noise_score": self._noise_analysis(img),
            "texture_score": self._texture_consistency(img),
            "compression_score": self._compression_analysis(image_path),
        }
        
        # Aggregate forensic score (0 = real, 1 = fake)
        weights = {
            "fft_score": 0.25,
            "ela_score": 0.25,
            "noise_score": 0.20,
            "texture_score": 0.15,
            "compression_score": 0.15,
        }
        
        results["aggregate_score"] = sum(
            results[k] * weights[k] for k in weights
        )
        
        return results
    
    def _fft_analysis(self, img: np.ndarray) -> float:
        """
        FFT analysis to detect GAN/diffusion artifacts.
        AI-generated images often have distinct frequency patterns.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shift) + 1)
        
        # Analyze high-frequency components
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # High frequency region (outer ring)
        mask_size = min(h, w) // 4
        high_freq_region = magnitude.copy()
        high_freq_region[center_h-mask_size:center_h+mask_size, 
                        center_w-mask_size:center_w+mask_size] = 0
        
        # AI images often have unusual high-frequency patterns
        high_freq_energy = np.mean(high_freq_region)
        total_energy = np.mean(magnitude)
        
        ratio = high_freq_energy / (total_energy + 1e-10)
        
        # Normalize to 0-1 score (higher = more likely manipulated)
        # These thresholds may need tuning
        score = np.clip((ratio - 0.3) / 0.4, 0, 1)
        
        return float(score)
    
    def _ela_analysis(self, image_path: str) -> float:
        """
        Error Level Analysis - detects areas with different compression levels.
        Spliced/inpainted regions often have different error levels.
        """
        import tempfile
        
        # Load original
        original = Image.open(image_path).convert('RGB')
        
        # Resave at known quality
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            original.save(tmp.name, 'JPEG', quality=self.ela_quality)
            resaved = Image.open(tmp.name)
        
        # Calculate difference
        orig_arr = np.array(original, dtype=np.float32)
        resaved_arr = np.array(resaved, dtype=np.float32)
        
        ela = np.abs(orig_arr - resaved_arr)
        
        # Analyze ELA variance - manipulated regions have inconsistent ELA
        ela_std = np.std(ela)
        ela_mean = np.mean(ela)
        
        # High variance in ELA suggests manipulation
        score = np.clip(ela_std / 30, 0, 1)  # Normalize
        
        return float(score)
    
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
        
        # AI images often have too-uniform or too-smooth noise
        # Real camera noise has specific patterns
        
        # Check for noise uniformity across image regions
        h, w = noise.shape
        regions = [
            noise[:h//2, :w//2],
            noise[:h//2, w//2:],
            noise[h//2:, :w//2],
            noise[h//2:, w//2:]
        ]
        
        region_stds = [np.std(r) for r in regions]
        std_variance = np.std(region_stds)
        
        # Very uniform noise across regions is suspicious
        uniformity_score = 1 - np.clip(std_variance / 5, 0, 1)
        
        # Very low or very high noise is suspicious
        noise_level_score = 0
        if noise_std < 3 or noise_std > 25:
            noise_level_score = 0.5
        
        return float((uniformity_score + noise_level_score) / 2)
    
    def _texture_consistency(self, img: np.ndarray) -> float:
        """
        Check for unnatural smoothness in textures.
        AI often produces overly smooth surfaces.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate local variance using sliding window
        kernel_size = 15
        local_mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        local_sqr_mean = cv2.blur((gray.astype(np.float32))**2, (kernel_size, kernel_size))
        local_var = local_sqr_mean - local_mean**2
        
        # Find smooth regions (low variance)
        smooth_threshold = 100
        smooth_ratio = np.mean(local_var < smooth_threshold)
        
        # Too many smooth regions is suspicious for real estate images
        # (walls, floors should have some texture)
        score = np.clip((smooth_ratio - 0.3) / 0.4, 0, 1)
        
        return float(score)
    
    def _compression_analysis(self, image_path: str) -> float:
        """
        Detect compression inconsistencies from splicing.
        """
        img = cv2.imread(image_path)
        
        # Convert to YCrCb and analyze DCT blocks
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:,:,0].astype(np.float32)
        
        # Analyze 8x8 block boundaries (JPEG artifacts)
        h, w = y_channel.shape
        h8, w8 = (h // 8) * 8, (w // 8) * 8
        y_cropped = y_channel[:h8, :w8]
        
        # Calculate block boundary differences
        block_diffs = []
        for i in range(0, h8-8, 8):
            for j in range(0, w8-8, 8):
                # Difference at block boundary vs inside block
                boundary_diff = abs(float(y_cropped[i+7, j+4]) - float(y_cropped[i+8, j+4]))
                inside_diff = abs(float(y_cropped[i+3, j+4]) - float(y_cropped[i+4, j+4]))
                if inside_diff > 0:
                    block_diffs.append(boundary_diff / inside_diff)
        
        if block_diffs:
            # Inconsistent block artifacts suggest manipulation
            diff_variance = np.std(block_diffs)
            score = np.clip(diff_variance / 2, 0, 1)
        else:
            score = 0.5
        
        return float(score)
