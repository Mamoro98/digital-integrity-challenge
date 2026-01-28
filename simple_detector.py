#!/usr/bin/env python3
"""
Simple optimized detector - DCT HF focus.
Real=1.86, Fake=0.89 for DCT HF mean.
"""
import cv2
import numpy as np
from glob import glob
import os

def extract_dct_hf(img):
    """Extract DCT high-frequency energy."""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)
    h, w = y.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    if h8 < 16 or w8 < 16:
        return 1.0
    y = y[:h8, :w8]
    hf_energies = []
    for i in range(0, h8, 8):
        for j in range(0, w8, 8):
            block = y[i:i+8, j:j+8]
            dct = cv2.dct(block)
            hf_energy = np.mean(np.abs(dct[4:, 4:]))
            hf_energies.append(hf_energy)
    return float(np.mean(hf_energies))

def extract_local_var(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_mean = cv2.blur(gray, (15, 15))
    local_sqr = cv2.blur(gray ** 2, (15, 15))
    local_var = local_sqr - local_mean ** 2
    return float(np.mean(local_var))

def extract_saturation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1]))

# Stats from analysis
STATS = {
    'dct': (1.86, 1.70, 0.89, 1.01),  # real_mean, real_std, fake_mean, fake_std
    'var': (514, 332, 412, 222),
    'sat': (95, 42, 76, 45),
}

def likelihood_score(val, stat):
    """P(fake|val) using Gaussian likelihood ratio."""
    rm, rs, fm, fs = stat
    ll_real = -0.5 * ((val - rm) / max(rs, 1)) ** 2
    ll_fake = -0.5 * ((val - fm) / max(fs, 1)) ** 2
    diff = np.clip(ll_real - ll_fake, -20, 20)
    return 1.0 / (1.0 + np.exp(diff))

# Evaluate
data_dir = "data/ai_generated_v2"
images = glob(os.path.join(data_dir, "*.png"))

real_scores, fake_scores = [], []

for img_path in images:
    filename = os.path.basename(img_path)
    is_fake = "images_fake_" in filename
    
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    dct_hf = extract_dct_hf(img)
    local_var = extract_local_var(img)
    sat = extract_saturation(img)
    
    # Weighted scores (DCT is best)
    score = (
        0.50 * likelihood_score(dct_hf, STATS['dct']) +
        0.30 * likelihood_score(sat, STATS['sat']) +
        0.20 * likelihood_score(local_var, STATS['var'])
    )
    
    if is_fake:
        fake_scores.append(score)
    else:
        real_scores.append(score)

print("="*50)
print("SIMPLE DETECTOR RESULTS")
print("="*50)
print(f"Real (n={len(real_scores)}): {np.mean(real_scores):.3f} ± {np.std(real_scores):.3f}")
print(f"Fake (n={len(fake_scores)}): {np.mean(fake_scores):.3f} ± {np.std(fake_scores):.3f}")
print(f"Separation: {np.mean(fake_scores) - np.mean(real_scores):.3f}")

# Best threshold
best_acc, best_thresh = 0, 0.5
for thresh in np.arange(0.3, 0.7, 0.01):
    correct = sum(1 for s in real_scores if s < thresh) + sum(1 for s in fake_scores if s >= thresh)
    acc = correct / (len(real_scores) + len(fake_scores))
    if acc > best_acc:
        best_acc, best_thresh = acc, thresh

print(f"\nBest threshold: {best_thresh:.2f}")
print(f"Best accuracy:  {best_acc*100:.1f}%")
