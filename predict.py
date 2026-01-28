#!/usr/bin/env python3
"""
Digital Integrity Challenge - Track B: Real Estate
Detecting AI-generated/manipulated property images

Usage:
    python predict.py --input_dir /test_images --output_file predictions.json
    python predict.py --image /path/to/image.jpg --output_file predictions.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from src.forensics.detector import ForensicDetector
from src.vlm.reasoner import VLMReasoner
from src.fusion.combiner import FusionModule


def process_image(image_path: str, forensic: ForensicDetector, vlm: VLMReasoner, fusion: FusionModule) -> Dict:
    """Process a single image and return prediction."""

    # Module 1: Forensic analysis
    forensic_results = forensic.analyze(image_path)

    # Module 2: VLM reasoning
    vlm_results = vlm.analyze(image_path)

    # Fusion: Combine results
    final_result = fusion.combine(forensic_results, vlm_results)

    return {
        "image_name": os.path.basename(image_path),
        "authenticity_score": final_result["score"],
        "manipulation_type": final_result["manipulation_type"],
        "vlm_reasoning": final_result["reasoning"],
        "details": {
            "forensic_score": final_result["forensic_score"],
            "vlm_score": final_result["vlm_score"],
            "forensic_breakdown": {
                "fft": forensic_results.get("fft_score", 0),
                "ela": forensic_results.get("ela_score", 0),
                "noise": forensic_results.get("noise_score", 0),
                "texture": forensic_results.get("texture_score", 0),
                "compression": forensic_results.get("compression_score", 0),
                "edge": forensic_results.get("edge_score", 0),
                "sharpness": forensic_results.get("sharpness_score", 0),
                "rich_poor_texture": forensic_results.get("rich_poor_texture_score", 0),
                "color_consistency": forensic_results.get("color_consistency_score", 0),
                "lbp": forensic_results.get("lbp_score", 0),
                "glcm": forensic_results.get("glcm_score", 0),
            }
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Detect AI-generated/manipulated real estate images")
    parser.add_argument("--input_dir", type=str, help="Directory containing images to analyze")
    parser.add_argument("--image", type=str, help="Single image to analyze")
    parser.add_argument("--output_file", type=str, default="predictions.json", help="Output JSON file")
    parser.add_argument("--vlm_backend", type=str, default="auto", help="VLM backend: auto, qwen2vl, blip2, mock")
    args = parser.parse_args()

    if not args.input_dir and not args.image:
        parser.error("Either --input_dir or --image must be provided")

    # Initialize modules
    print("Loading models...")
    forensic = ForensicDetector()
    vlm = VLMReasoner(backend=args.vlm_backend)
    fusion = FusionModule()

    # Collect images to process
    images = []
    if args.image:
        images = [Path(args.image)]
    else:
        input_path = Path(args.input_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff', '.bmp'}
        # Recursively find all images
        images = [f for f in input_path.rglob('*') if f.suffix.lower() in image_extensions]

    print(f"Found {len(images)} images to process")

    # Process each image
    predictions = []
    for idx, img_path in enumerate(images):
        print(f"[{idx + 1}/{len(images)}] Processing: {img_path.name}")
        try:
            result = process_image(str(img_path), forensic, vlm, fusion)
            predictions.append(result)

            # Print summary
            score = result["authenticity_score"]
            manip_type = result["manipulation_type"]
            verdict = "LIKELY REAL" if score < 0.4 else ("UNCERTAIN" if score < 0.6 else "LIKELY MANIPULATED")
            print(f"    Score: {score:.3f} ({verdict}) - Type: {manip_type}")

        except Exception as e:
            print(f"    Error processing {img_path.name}: {e}")
            predictions.append({
                "image_name": img_path.name,
                "authenticity_score": 0.5,
                "manipulation_type": "error",
                "vlm_reasoning": f"Error during analysis: {str(e)}",
                "details": {}
            })

    # Save predictions
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"\nPredictions saved to {output_path}")

    # Print summary statistics
    if predictions:
        scores = [p["authenticity_score"] for p in predictions if "authenticity_score" in p]
        if scores:
            print(f"\n=== Summary ===")
            print(f"Total images: {len(predictions)}")
            print(f"Average score: {sum(scores) / len(scores):.3f}")
            print(f"Likely real (score < 0.4): {sum(1 for s in scores if s < 0.4)}")
            print(f"Uncertain (0.4-0.6): {sum(1 for s in scores if 0.4 <= s < 0.6)}")
            print(f"Likely manipulated (score >= 0.6): {sum(1 for s in scores if s >= 0.6)}")


if __name__ == "__main__":
    main()
