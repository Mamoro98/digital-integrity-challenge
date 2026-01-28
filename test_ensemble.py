#!/usr/bin/env python3
"""Test ensemble of CLIP + Forensics."""

import sys
sys.path.insert(0, '.')

from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from src.forensics.detector import ForensicDetector

REAL_DIR = Path("data/real")
FAKE_DIR = Path("data/ai_generated_v2")

def load_images(directory, pattern="*"):
    images = []
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    for ext in extensions:
        for f in directory.glob(f"{pattern}{ext}"):
            try:
                img = Image.open(f).convert("RGB")
                images.append((f.name, f, img))
            except:
                pass
    return images

def main():
    print("Loading models...")
    
    # CLIP
    model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
    processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    labels = [
        "a real photograph",
        "an AI-generated image",
        "a computer-generated image",
        "a synthetic image created by artificial intelligence"
    ]
    
    # Forensics
    forensic = ForensicDetector()
    
    # Load images
    real_estate = load_images(REAL_DIR)
    fake_v2 = [(n, p, i) for n, p, i in load_images(FAKE_DIR, "*_fake_*")]
    real_v2 = [(n, p, i) for n, p, i in load_images(FAKE_DIR, "*_real_*")]
    
    all_real = real_estate + real_v2
    all_fake = fake_v2
    
    print(f"Testing {len(all_real)} real, {len(all_fake)} fake images")
    
    results = []
    
    for label, images, is_fake in [("REAL", all_real, False), ("FAKE", all_fake, True)]:
        print(f"\n=== {label} ===")
        for name, path, img in images:
            # CLIP score
            inputs = processor(text=labels, images=img, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
            
            real_prob = probs[0]
            ai_prob = max(probs[1], probs[2], probs[3])
            clip_score = ai_prob / (real_prob + ai_prob + 1e-10)
            
            # Forensic score
            forensic_results = forensic.analyze(str(path))
            forensic_score = forensic_results['aggregate_score']
            
            # Ensemble - try different weights
            for w_clip in [0.7, 0.8, 0.9]:
                ensemble = w_clip * clip_score + (1 - w_clip) * forensic_score
                results.append({
                    'name': name,
                    'is_fake': is_fake,
                    'clip': clip_score,
                    'forensic': forensic_score,
                    f'ensemble_{w_clip}': ensemble,
                })
            
            print(f"{name}: CLIP={clip_score:.3f}, Forensic={forensic_score:.3f}")
    
    # Calculate accuracies
    print("\n" + "="*60)
    print("ACCURACY SUMMARY")
    print("="*60)
    
    for method in ['clip', 'forensic', 'ensemble_0.7', 'ensemble_0.8', 'ensemble_0.9']:
        # Group by unique images (results has duplicates due to ensemble weights)
        seen = set()
        real_correct = 0
        real_total = 0
        fake_correct = 0
        fake_total = 0
        
        for r in results:
            if r['name'] in seen:
                continue
            seen.add(r['name'])
            
            score = r.get(method, r.get('clip') if 'ensemble' in method else 0)
            if 'ensemble' in method:
                w = float(method.split('_')[1])
                score = w * r['clip'] + (1-w) * r['forensic']
            
            if r['is_fake']:
                fake_total += 1
                if score >= 0.5:
                    fake_correct += 1
            else:
                real_total += 1
                if score < 0.5:
                    real_correct += 1
        
        total = real_total + fake_total
        overall = (real_correct + fake_correct) / total * 100 if total > 0 else 0
        print(f"{method:20s}: Real {real_correct}/{real_total}, Fake {fake_correct}/{fake_total}, Overall {overall:.1f}%")

if __name__ == "__main__":
    main()
