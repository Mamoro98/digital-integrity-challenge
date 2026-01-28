#!/usr/bin/env python3
"""Evaluate forensics detector on real vs Flux-generated images."""

import sys
sys.path.insert(0, '.')

from pathlib import Path
import numpy as np
from src.forensics.detector import ForensicDetector

REAL_DIR = Path("data/real")
FAKE_DIR = Path("data/ai_generated_v2")

def evaluate():
    detector = ForensicDetector()

    # Real estate photos (definitely real)
    real_estate_files = sorted(REAL_DIR.glob("*.jpg"))
    
    # From ai_generated_v2: files with "_fake_" are AI, files with "_real_" are real
    all_v2_files = sorted(FAKE_DIR.glob("*.png"))
    fake_files = [f for f in all_v2_files if "_fake_" in f.name]
    real_v2_files = [f for f in all_v2_files if "_real_" in f.name]
    
    # Combine all real files
    all_real_files = list(real_estate_files) + list(real_v2_files)
    
    print(f"Testing {len(all_real_files)} real ({len(real_estate_files)} real_estate + {len(real_v2_files)} v2_real)")
    print(f"Testing {len(fake_files)} fake (AI-generated)\n")

    real_scores = []
    fake_scores = []
    real_details = []
    fake_details = []

    print("=== REAL IMAGES ===")
    for f in all_real_files:
        try:
            result = detector.analyze(str(f))
            score = result['aggregate_score']
            real_scores.append(score)
            real_details.append((f.name, result))
            verdict = "CORRECT" if score < 0.5 else "WRONG"
            print(f"{f.name}: {score:.3f} - {verdict}")
        except Exception as e:
            print(f"{f.name}: ERROR - {e}")

    print("\n=== FAKE (AI-GENERATED) IMAGES ===")
    for f in fake_files:
        try:
            result = detector.analyze(str(f))
            score = result['aggregate_score']
            fake_scores.append(score)
            fake_details.append((f.name, result))
            verdict = "CORRECT" if score >= 0.5 else "WRONG"
            print(f"{f.name}: {score:.3f} - {verdict}")
        except Exception as e:
            print(f"{f.name}: ERROR - {e}")

    # Calculate accuracy
    real_correct = sum(1 for s in real_scores if s < 0.5)
    fake_correct = sum(1 for s in fake_scores if s >= 0.5)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Real images: {real_correct}/{len(real_scores)} correct ({100*real_correct/len(real_scores):.1f}%)")
    print(f"Fake images: {fake_correct}/{len(fake_scores)} correct ({100*fake_correct/len(fake_scores):.1f}%)")
    total = len(real_scores) + len(fake_scores)
    print(f"Overall: {real_correct + fake_correct}/{total} ({100*(real_correct + fake_correct)/total:.1f}%)")

    print(f"\nReal scores: mean={np.mean(real_scores):.3f}, std={np.std(real_scores):.3f}")
    print(f"Fake scores: mean={np.mean(fake_scores):.3f}, std={np.std(fake_scores):.3f}")
    print(f"Separation: {np.mean(fake_scores) - np.mean(real_scores):.3f}")

    # Analyze which signals discriminate best
    print("\n" + "="*60)
    print("SIGNAL DISCRIMINATION ANALYSIS (d' = Cohen's d)")
    print("="*60)

    signals = ['fft_score', 'ela_score', 'noise_score', 'texture_score',
               'compression_score', 'edge_score', 'sharpness_score',
               'rich_poor_texture_score', 'color_consistency_score',
               'lbp_score', 'glcm_score']

    disc_power = []
    for sig in signals:
        real_vals = [d[1][sig] for d in real_details]
        fake_vals = [d[1][sig] for d in fake_details]

        real_mean = np.mean(real_vals)
        fake_mean = np.mean(fake_vals)
        separation = fake_mean - real_mean

        # Calculate discrimination power (Cohen's d)
        real_std = np.std(real_vals)
        fake_std = np.std(fake_vals)
        pooled_std = np.sqrt((real_std**2 + fake_std**2) / 2)
        d_prime = separation / (pooled_std + 1e-10)
        disc_power.append((sig, d_prime, separation, real_mean, fake_mean))

        print(f"{sig:25s}: real={real_mean:.3f}, fake={fake_mean:.3f}, sep={separation:+.3f}, d'={d_prime:+.2f}")

    # Sort by absolute discrimination power
    disc_power.sort(key=lambda x: abs(x[1]), reverse=True)
    print("\n=== TOP DISCRIMINATORS (by |d'|) ===")
    for sig, dp, sep, rm, fm in disc_power[:5]:
        direction = "HIGHER for fake" if sep > 0 else "LOWER for fake"
        print(f"{sig:25s}: d'={dp:+.2f} ({direction})")

if __name__ == "__main__":
    evaluate()
