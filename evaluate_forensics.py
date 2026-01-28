#!/usr/bin/env python3
"""Evaluate forensic detector on test datasets."""

import sys
import os
import glob
import json
import numpy as np

sys.path.insert(0, '.')
from src.forensics.detector import ForensicDetector

def evaluate_dataset(detector, image_paths, label, threshold=0.5):
    """Evaluate detector on a set of images with known label."""
    results = []
    for path in image_paths:
        try:
            result = detector.analyze(path)
            result['path'] = os.path.basename(path)
            result['true_label'] = label
            result['predicted'] = 'fake' if result['aggregate_score'] >= threshold else 'real'
            result['correct'] = (label == 'fake' and result['predicted'] == 'fake') or \
                               (label == 'real' and result['predicted'] == 'real')
            results.append(result)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    return results

def print_analysis(all_results, threshold=0.5):
    """Print detailed analysis of results."""
    fake_results = [r for r in all_results if r['true_label'] == 'fake']
    real_results = [r for r in all_results if r['true_label'] == 'real']

    # Calculate accuracy
    fake_correct = sum(1 for r in fake_results if r['correct'])
    real_correct = sum(1 for r in real_results if r['correct'])

    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS (threshold={threshold})")
    print(f"{'='*60}")
    print(f"FAKE images: {fake_correct}/{len(fake_results)} correct ({100*fake_correct/max(1,len(fake_results)):.1f}%)")
    print(f"REAL images: {real_correct}/{len(real_results)} correct ({100*real_correct/max(1,len(real_results)):.1f}%)")
    print(f"Total accuracy: {(fake_correct+real_correct)}/{len(all_results)} ({100*(fake_correct+real_correct)/max(1,len(all_results)):.1f}%)")

    # Per-feature analysis
    features = [k for k in all_results[0].keys() if k.endswith('_score') and k != 'aggregate_score']

    print(f"\n{'='*60}")
    print("FEATURE DISCRIMINATION ANALYSIS")
    print("(Higher fake_mean - real_mean = better discriminator)")
    print(f"{'='*60}")

    discriminators = []
    for feat in features:
        fake_scores = [r[feat] for r in fake_results]
        real_scores = [r[feat] for r in real_results]
        fake_mean = np.mean(fake_scores)
        real_mean = np.mean(real_scores)
        discrimination = fake_mean - real_mean  # Positive = good (fake scores higher)
        discriminators.append((feat, discrimination, fake_mean, real_mean, np.std(fake_scores), np.std(real_scores)))

    # Sort by discrimination power
    discriminators.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Feature':<30} {'Discrim':>8} {'Fake μ':>8} {'Real μ':>8} {'Fake σ':>8} {'Real σ':>8}")
    print("-" * 78)
    for feat, disc, fake_m, real_m, fake_s, real_s in discriminators:
        print(f"{feat:<30} {disc:>+8.3f} {fake_m:>8.3f} {real_m:>8.3f} {fake_s:>8.3f} {real_s:>8.3f}")

    # Aggregate score distribution
    print(f"\n{'='*60}")
    print("AGGREGATE SCORE DISTRIBUTION")
    print(f"{'='*60}")
    fake_agg = [r['aggregate_score'] for r in fake_results]
    real_agg = [r['aggregate_score'] for r in real_results]
    print(f"FAKE: mean={np.mean(fake_agg):.3f}, std={np.std(fake_agg):.3f}, min={np.min(fake_agg):.3f}, max={np.max(fake_agg):.3f}")
    print(f"REAL: mean={np.mean(real_agg):.3f}, std={np.std(real_agg):.3f}, min={np.min(real_agg):.3f}, max={np.max(real_agg):.3f}")

    # Show misclassified examples
    print(f"\n{'='*60}")
    print("MISCLASSIFIED EXAMPLES")
    print(f"{'='*60}")

    missed_fakes = [r for r in fake_results if not r['correct']]
    false_positives = [r for r in real_results if not r['correct']]

    print(f"\nMissed FAKE images (predicted as real): {len(missed_fakes)}")
    for r in missed_fakes[:10]:
        print(f"  {r['path']}: agg={r['aggregate_score']:.3f}")

    print(f"\nFalse positives (real predicted as fake): {len(false_positives)}")
    for r in false_positives[:10]:
        print(f"  {r['path']}: agg={r['aggregate_score']:.3f}")

    return discriminators

def main():
    detector = ForensicDetector()
    all_results = []

    # Collect image paths
    data_dir = '/home/omer_aims_ac_za/digital-integrity-challenge/data'

    # AI generated images (fake)
    fake_paths = []
    fake_paths.extend(glob.glob(f'{data_dir}/ai_generated_v2/*.png'))
    fake_paths.extend(glob.glob(f'{data_dir}/ai_generated/*.png'))
    fake_paths.extend(glob.glob(f'{data_dir}/ai_generated/*.jpg'))
    fake_paths.extend(glob.glob(f'{data_dir}/manipulated/*.jpg'))
    fake_paths.extend(glob.glob(f'{data_dir}/test_subset/manip/*.jpg'))

    # Real images
    real_paths = []
    real_paths.extend(glob.glob(f'{data_dir}/real/*.jpg'))
    real_paths.extend(glob.glob(f'{data_dir}/test_subset/real/*.jpg'))

    print(f"Found {len(fake_paths)} fake images and {len(real_paths)} real images")

    # Run evaluation
    print("\nProcessing fake images...")
    fake_results = evaluate_dataset(detector, fake_paths, 'fake')
    print(f"Processed {len(fake_results)} fake images")

    print("\nProcessing real images...")
    real_results = evaluate_dataset(detector, real_paths, 'real')
    print(f"Processed {len(real_results)} real images")

    all_results = fake_results + real_results

    # Test different thresholds
    for threshold in [0.35, 0.40, 0.45, 0.50]:
        # Recalculate predictions with new threshold
        for r in all_results:
            r['predicted'] = 'fake' if r['aggregate_score'] >= threshold else 'real'
            r['correct'] = (r['true_label'] == 'fake' and r['predicted'] == 'fake') or \
                          (r['true_label'] == 'real' and r['predicted'] == 'real')

        print_analysis(all_results, threshold)

    # Save detailed results
    with open('/tmp/forensic_eval_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to /tmp/forensic_eval_results.json")

if __name__ == "__main__":
    main()
