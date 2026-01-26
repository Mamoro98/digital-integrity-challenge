#!/usr/bin/env python3
"""
Digital Integrity Challenge - Track B: Real Estate
Detecting AI-generated/manipulated property images

Usage:
    python predict.py --input_dir /test_images --output_file predictions.json
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
        "vlm_reasoning": final_result["reasoning"]
    }


def main():
    parser = argparse.ArgumentParser(description="Detect AI-generated/manipulated real estate images")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images to analyze")
    parser.add_argument("--output_file", type=str, default="predictions.json", help="Output JSON file")
    args = parser.parse_args()
    
    # Initialize modules
    print("Loading models...")
    forensic = ForensicDetector()
    vlm = VLMReasoner()
    fusion = FusionModule()
    
    # Get all images
    input_path = Path(args.input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    images = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(images)} images to process")
    
    # Process each image
    predictions = []
    for img_path in images:
        print(f"Processing: {img_path.name}")
        try:
            result = process_image(str(img_path), forensic, vlm, fusion)
            predictions.append(result)
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            predictions.append({
                "image_name": img_path.name,
                "authenticity_score": 0.5,
                "manipulation_type": "unknown",
                "vlm_reasoning": f"Error during analysis: {str(e)}"
            })
    
    # Save predictions
    with open(args.output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Predictions saved to {args.output_file}")


if __name__ == "__main__":
    main()
