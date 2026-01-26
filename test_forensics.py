#!/usr/bin/env python3
"""Quick test of forensic module."""

import sys
sys.path.insert(0, '.')

from src.forensics.detector import ForensicDetector

def test_with_image(image_path):
    print(f"Testing with: {image_path}")
    detector = ForensicDetector()
    results = detector.analyze(image_path)
    
    print("\nForensic Analysis Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_with_image(sys.argv[1])
    else:
        print("Usage: python test_forensics.py <image_path>")
        print("\nTo test, download a sample image first")
