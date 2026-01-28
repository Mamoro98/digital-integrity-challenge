#!/usr/bin/env python3
"""
Optimized Forensic Detector - based on research and empirical analysis.

Key insight from RESEARCH.md:
- Diffusion models show artifacts at periods 2, 4, 8
- AI images are smoother, lack high-frequency details
- DCT HF energy is the best single discriminator

From feature analysis:
- DCT HF mean: Real=1.86±1.70, Fake=0.89±1.01 (separation=0.357)
- Local variance: Real=514±332, Fake=412±222 (separation=0.185)
- Saturation: Real=95±42, Fake=76±45 (separation=0.222)

Strategy: Use z-score normalization and sigmoid scoring for continuous output.
"""

import cv2
import numpy as np
from scipy import ndimage
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class OptimizedForensicDetector:
    """Optimized detector using research-backed features."""
    
    # Empirical distributions from ai_generated_v2 dataset
    STATS = {
        'dct_hf': {'real_mean': 1.86, 'real_std': 1.70, 'fake_mean': 0.89, 'fake_std': 1.01},
        'local_var': {'real_mean': 514, 'real_std': 332, 'fake_mean': 412, 'fake_std': 222},
        'saturation': {'real_mean': 95, 'real_std': 42, 'fake_mean': 76, 'fake_std': 45},
        'brightness': {'real_mean': 112, 'real_std': 19, 'fake_mean': 128, 'fake_std': 38},
    }
    
    def __init__(self):
        pass
    
    def analyze(self, image_path: str) -> Dict:
        """Analyze image and return fake probability."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        results = {}
        
        # Extract raw features
        dct_hf = self._extract_dct_hf(img)
        local_var = self._extract_local_variance(img)
        saturation = self._extract_saturation(img)
        brightness = self._extract_brightness(img)
        
        results['dct_hf_raw'] = dct_hf
        results['local_var_raw'] = local_var
        results['saturation_raw'] = saturation
        results['brightness_raw'] = brightness
        
        # Convert to fake probability using likelihood ratio
        # P(fake|feature) ∝ P(feature|fake) / P(feature|real)
        
        dct_score = self._feature_to_score(dct_hf, 'dct_hf', invert=True)  # Lower = more fake
        var_score = self._feature_to_score(local_var, 'local_var', invert=True)  # Lower = more fake
        sat_score = self._feature_to_score(saturation, 'saturation', invert=True)  # Lower = more fake
        bright_score = self._feature_to_score(brightness, 'brightness', invert=False)  # Higher = more fake
        
        results['dct_hf_score'] = dct_score
        results['local_var_score'] = var_score
        results['saturation_score'] = sat_score
        results['brightness_score'] = bright_score
        
        # Weighted combination - based on separation scores
        # DCT HF has best separation (0.357), then saturation (0.222), then local_var (0.185)
        weights = {
            'dct': 0.45,      # Best discriminator
            'sat': 0.25,      # Second best
            'var': 0.20,      # Third
            'bright': 0.10,   # Weakest
        }
        
        aggregate = (
            weights['dct'] * dct_score +
            weights['sat'] * sat_score +
            weights['var'] * var_score +
            weights['bright'] * bright_score
        )
        
        results['aggregate_score'] = float(np.clip(aggregate, 0, 1))
        
        return results
    
    def _extract_dct_hf(self, img: np.ndarray) -> float:
        """Extract DCT high-frequency energy."""
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0].astype(np.float32)
        h, w = y.shape
        
        h8, w8 = (h // 8) * 8, (w // 8) * 8
        if h8 < 16 or w8 < 16:
            return 1.0  # Default to neutral
        
        y = y[:h8, :w8]
        hf_energies = []
        
        for i in range(0, h8, 8):
            for j in range(0, w8, 8):
                block = y[i:i+8, j:j+8]
                dct = cv2.dct(block)
                # High frequency: bottom-right 4x4 of 8x8 DCT
                hf_energy = np.mean(np.abs(dct[4:, 4:]))
                hf_energies.append(hf_energy)
        
        return float(np.mean(hf_energies))
    
    def _extract_local_variance(self, img: np.ndarray) -> float:
        """Extract mean local variance (texture complexity)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        kernel_size = 15
        local_mean = cv2.blur(gray, (kernel_size, kernel_size))
        local_sqr_mean = cv2.blur(gray ** 2, (kernel_size, kernel_size))
        local_var = local_sqr_mean - local_mean ** 2
        
        return float(np.mean(local_var))
    
    def _extract_saturation(self, img: np.ndarray) -> float:
        """Extract mean saturation."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 1]))
    
    def _extract_brightness(self, img: np.ndarray) -> float:
        """Extract mean brightness."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
    
    def _feature_to_score(self, value: float, feature: str, invert: bool) -> float:
        """
        Convert raw feature to fake probability using likelihood ratio.
        
        Uses Gaussian assumption:
        score = P(value|fake) / (P(value|fake) + P(value|real))
        
        If invert=True, lower values indicate fake (so we flip the logic).
        """
        stats = self.STATS[feature]
        
        # Compute likelihoods (Gaussian PDF, but we only need ratio)
        def gaussian_log_likelihood(x, mean, std):
            if std < 1e-6:
                std = 1e-6
            return -0.5 * ((x - mean) / std) ** 2
        
        ll_fake = gaussian_log_likelihood(value, stats['fake_mean'], stats['fake_std'])
        ll_real = gaussian_log_likelihood(value, stats['real_mean'], stats['real_std'])
        
        # Softmax to get probability
        # P(fake) = exp(ll_fake) / (exp(ll_fake) + exp(ll_real))
        # = 1 / (1 + exp(ll_real - ll_fake))
        diff = ll_real - ll_fake
        
        # Clip to avoid overflow
        diff = np.clip(diff, -20, 20)
        
        score = 1.0 / (1.0 + np.exp(diff))
        
        return float(score)


def evaluate_detector():
    """Evaluate on the dataset."""
    from glob import glob
    import os
    
    detector = OptimizedForensicDetector()
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
    print("OPTIMIZED DETECTOR RESULTS (Likelihood Ratio)")
    print("="*60)
    print(f"\nReal (n={len(real_scores)}): {np.mean(real_scores):.3f} ± {np.std(real_scores):.3f}")
    print(f"Fake (n={len(fake_scores)}): {np.mean(fake_scores):.3f} ± {np.std(fake_scores):.3f}")
    print(f"Separation: {np.mean(fake_scores) - np.mean(real_scores):.3f}")
    
    # Find best threshold
    best_acc = 0
    best_thresh = 0.5
    best_f1 = 0
    
    all_scores = real_scores + fake_scores
    all_labels = [0] * len(real_scores) + [1] * len(fake_scores)
    
    for thresh in np.arange(0.2, 0.8, 0.01):
        tp = sum(1 for s, l in zip(all_scores, all_labels) if s >= thresh and l == 1)
        tn = sum(1 for s, l in zip(all_scores, all_labels) if s < thresh and l == 0)
        fp = sum(1 for s, l in zip(all_scores, all_labels) if s >= thresh and l == 0)
        fn = sum(1 for s, l in zip(all_scores, all_labels) if s < thresh and l == 1)
        
        acc = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
        if f1 > best_f1:
            best_f1 = f1
    
    print(f"\nBest threshold: {best_thresh:.2f}")
    print(f"Best accuracy:  {best_acc*100:.1f}%")
    print(f"Best F1:        {best_f1:.3f}")
    
    # Per-feature analysis
    print("\n" + "="*60)
    print("PER-FEATURE PERFORMANCE")
    print("="*60)
    
    for feature in ['dct_hf', 'local_var', 'saturation', 'brightness']:
        real_feat = []
        fake_feat = []
        
        for img_path in sorted(images):
            filename = os.path.basename(img_path)
            is_fake = "images_fake_" in filename
            
            try:
                results = detector.analyze(img_path)
                score = results[f"{feature}_score"]
                
                if is_fake:
                    fake_feat.append(score)
                else:
                    real_feat.append(score)
            except:
                pass
        
        # Find best accuracy for this feature alone
        all_feat = real_feat + fake_feat
        best_feat_acc = 0
        for thresh in np.arange(0.2, 0.8, 0.01):
            correct = sum(1 for s in real_feat if s < thresh) + sum(1 for s in fake_feat if s >= thresh)
            acc = correct / len(all_feat)
            if acc > best_feat_acc:
                best_feat_acc = acc
        
        print(f"{feature:12s}: Real={np.mean(real_feat):.3f}, Fake={np.mean(fake_feat):.3f}, "
              f"Sep={np.mean(fake_feat)-np.mean(real_feat):.3f}, Acc={best_feat_acc*100:.1f}%")


if __name__ == "__main__":
    evaluate_detector()
