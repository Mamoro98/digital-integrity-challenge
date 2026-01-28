#!/usr/bin/env python3
"""Evaluate forensic detector on test dataset."""

import os
import sys
import numpy as np
from glob import glob

sys.path.insert(0, '/home/omer_aims_ac_za/digital-integrity-challenge')
from src.forensics.detector import ForensicDetector

def evaluate():
    detector = ForensicDetector()
    data_dir = "data/ai_generated_v2"

    images = glob(os.path.join(data_dir, "*.png"))

    real_scores = []
    fake_scores = []
    all_results = []

    for img_path in sorted(images):
        filename = os.path.basename(img_path)
        # Check for images_fake_ vs images_real_ pattern
        is_fake = "images_fake_" in filename

        try:
            results = detector.analyze(img_path)
            score = results["aggregate_score"]

            all_results.append({
                'filename': filename,
                'is_fake': is_fake,
                'score': score,
                'results': results
            })

            if is_fake:
                fake_scores.append(score)
            else:
                real_scores.append(score)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\n" + "="*60)
    print("SCORE DISTRIBUTION")
    print("="*60)
    print(f"\nReal images (n={len(real_scores)}):")
    print(f"  Mean: {np.mean(real_scores):.3f}")
    print(f"  Std:  {np.std(real_scores):.3f}")
    print(f"  Min:  {np.min(real_scores):.3f}")
    print(f"  Max:  {np.max(real_scores):.3f}")

    print(f"\nFake images (n={len(fake_scores)}):")
    print(f"  Mean: {np.mean(fake_scores):.3f}")
    print(f"  Std:  {np.std(fake_scores):.3f}")
    print(f"  Min:  {np.min(fake_scores):.3f}")
    print(f"  Max:  {np.max(fake_scores):.3f}")

    # Find optimal threshold
    print("\n" + "="*60)
    print("THRESHOLD ANALYSIS")
    print("="*60)

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

    # Per-feature analysis
    print("\n" + "="*60)
    print("PER-FEATURE ANALYSIS (mean fake - mean real)")
    print("="*60)

    feature_names = ['fft_score', 'ela_score', 'noise_score', 'texture_score',
                     'compression_score', 'edge_score', 'sharpness_score',
                     'rich_poor_texture_score', 'color_consistency_score',
                     'lbp_score', 'glcm_score']

    for feat in feature_names:
        real_feat = [r['results'][feat] for r in all_results if not r['is_fake']]
        fake_feat = [r['results'][feat] for r in all_results if r['is_fake']]

        diff = np.mean(fake_feat) - np.mean(real_feat)

        # Calculate feature's individual accuracy
        best_feat_acc = 0
        best_feat_dir = 1
        for thresh in np.arange(0.1, 0.9, 0.02):
            for direction in [1, -1]:
                if direction == 1:
                    real_c = sum(1 for s in real_feat if s < thresh)
                    fake_c = sum(1 for s in fake_feat if s >= thresh)
                else:
                    real_c = sum(1 for s in real_feat if s >= thresh)
                    fake_c = sum(1 for s in fake_feat if s < thresh)
                acc = (real_c + fake_c) / (len(real_feat) + len(fake_feat))
                if acc > best_feat_acc:
                    best_feat_acc = acc
                    best_feat_dir = direction

        dir_str = "(+)" if best_feat_dir == 1 else "(-)"
        print(f"  {feat:28s}: diff={diff:+.3f}  acc={best_feat_acc*100:.1f}% {dir_str}")
        print(f"      Real: {np.mean(real_feat):.3f}±{np.std(real_feat):.3f}  |  Fake: {np.mean(fake_feat):.3f}±{np.std(fake_feat):.3f}")

    # Show misclassified examples
    print("\n" + "="*60)
    print("MISCLASSIFIED EXAMPLES (at threshold 0.5)")
    print("="*60)

    print("\nFalse positives (real classified as fake):")
    for r in sorted(all_results, key=lambda x: -x['score']):
        if not r['is_fake'] and r['score'] >= 0.5:
            print(f"  {r['filename']}: {r['score']:.3f}")

    print("\nFalse negatives (fake classified as real):")
    for r in sorted(all_results, key=lambda x: x['score']):
        if r['is_fake'] and r['score'] < 0.5:
            print(f"  {r['filename']}: {r['score']:.3f}")

if __name__ == "__main__":
    evaluate()
