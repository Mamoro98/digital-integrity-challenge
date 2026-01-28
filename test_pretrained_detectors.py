#!/usr/bin/env python3
"""
Test pre-trained AI image detectors on Flux-generated images.
No fine-tuning - just evaluation of existing models.
"""

import os
import json
import time
from pathlib import Path
from PIL import Image
import torch
from transformers import pipeline, AutoModelForImageClassification, AutoImageProcessor
import numpy as np
from tqdm import tqdm

# Paths
REAL_DIR = Path("data/real")
FAKE_DIR = Path("data/ai_generated_v2")

# Models to test
MODELS = [
    # Current baseline
    "umm-maybe/AI-image-detector",
    # SDXL-specific detector (Swin Transformer)
    "Organika/sdxl-detector",
    # Fine-tuned on 2024 generators including Flux
    "Smogy/SMOGY-Ai-images-detector",
]

def load_images(directory, limit=None):
    """Load images from directory."""
    images = []
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    files = sorted([f for f in directory.iterdir() if f.suffix.lower() in extensions])
    if limit:
        files = files[:limit]
    for f in files:
        try:
            img = Image.open(f).convert("RGB")
            images.append((f.name, img))
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return images

def test_detector(model_name, real_images, fake_images):
    """Test a single detector model."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print('='*60)

    try:
        # Load model
        start = time.time()
        classifier = pipeline(
            "image-classification",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        load_time = time.time() - start
        print(f"Model loaded in {load_time:.1f}s")

        # Get label mapping - different models use different labels
        results = {"real": [], "fake": [], "model": model_name}

        # Test real images
        print(f"\nTesting {len(real_images)} real images...")
        correct_real = 0
        for name, img in tqdm(real_images):
            try:
                pred = classifier(img)
                # Find the "real" or "human" score
                score = 0.0
                for p in pred:
                    label = p['label'].lower()
                    if 'artificial' in label or 'ai' in label or 'fake' in label:
                        score = p['score']
                        break
                    elif 'human' in label or 'real' in label:
                        score = 1.0 - p['score']
                        break

                is_correct = score < 0.5  # Real images should have low AI score
                correct_real += is_correct
                results["real"].append({
                    "name": name,
                    "ai_score": score,
                    "correct": is_correct,
                    "raw": pred
                })
            except Exception as e:
                print(f"Error on {name}: {e}")
                results["real"].append({"name": name, "error": str(e)})

        # Test fake images
        print(f"Testing {len(fake_images)} fake (AI-generated) images...")
        correct_fake = 0
        for name, img in tqdm(fake_images):
            try:
                pred = classifier(img)
                # Find the "AI" or "artificial" score
                score = 0.0
                for p in pred:
                    label = p['label'].lower()
                    if 'artificial' in label or 'ai' in label or 'fake' in label:
                        score = p['score']
                        break
                    elif 'human' in label or 'real' in label:
                        score = 1.0 - p['score']
                        break

                is_correct = score >= 0.5  # Fake images should have high AI score
                correct_fake += is_correct
                results["fake"].append({
                    "name": name,
                    "ai_score": score,
                    "correct": is_correct,
                    "raw": pred
                })
            except Exception as e:
                print(f"Error on {name}: {e}")
                results["fake"].append({"name": name, "error": str(e)})

        # Calculate metrics
        total_real = len([r for r in results["real"] if "error" not in r])
        total_fake = len([r for r in results["fake"] if "error" not in r])

        real_acc = correct_real / total_real * 100 if total_real > 0 else 0
        fake_acc = correct_fake / total_fake * 100 if total_fake > 0 else 0
        overall_acc = (correct_real + correct_fake) / (total_real + total_fake) * 100 if (total_real + total_fake) > 0 else 0

        print(f"\n📊 Results for {model_name}:")
        print(f"  Real images: {correct_real}/{total_real} ({real_acc:.1f}%)")
        print(f"  Fake images: {correct_fake}/{total_fake} ({fake_acc:.1f}%)")
        print(f"  Overall: {overall_acc:.1f}%")

        results["metrics"] = {
            "real_accuracy": real_acc,
            "fake_accuracy": fake_acc,
            "overall_accuracy": overall_acc,
            "correct_real": correct_real,
            "correct_fake": correct_fake,
            "total_real": total_real,
            "total_fake": total_fake
        }

        # Clean up
        del classifier
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return results

    except Exception as e:
        print(f"❌ Failed to load/run model: {e}")
        import traceback
        traceback.print_exc()
        return {"model": model_name, "error": str(e)}

def test_clip_zero_shot():
    """Test CLIP ViT-L with zero-shot classification."""
    from transformers import CLIPProcessor, CLIPModel

    print(f"\n{'='*60}")
    print("Testing: CLIP ViT-L Zero-Shot (laion/CLIP-ViT-L-14-laion2B-s32B-b82K)")
    print('='*60)

    try:
        model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
        processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        # Zero-shot labels
        labels = [
            "a real photograph",
            "an AI-generated image",
            "a computer-generated image",
            "a synthetic image created by artificial intelligence"
        ]

        real_images = load_images(REAL_DIR)
        fake_images = load_images(FAKE_DIR)

        results = {"real": [], "fake": [], "model": "CLIP-ViT-L Zero-Shot"}
        correct_real = 0
        correct_fake = 0

        print(f"\nTesting {len(real_images)} real images...")
        for name, img in tqdm(real_images):
            inputs = processor(text=labels, images=img, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits_per_image
                probs = logits.softmax(dim=1).cpu().numpy()[0]

            # Real photo is label 0, AI labels are 1,2,3
            real_prob = probs[0]
            ai_prob = max(probs[1], probs[2], probs[3])
            is_correct = real_prob > ai_prob
            correct_real += is_correct
            results["real"].append({"name": name, "real_prob": float(real_prob), "ai_prob": float(ai_prob), "correct": is_correct})

        print(f"Testing {len(fake_images)} fake images...")
        for name, img in tqdm(fake_images):
            inputs = processor(text=labels, images=img, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits_per_image
                probs = logits.softmax(dim=1).cpu().numpy()[0]

            real_prob = probs[0]
            ai_prob = max(probs[1], probs[2], probs[3])
            is_correct = ai_prob > real_prob
            correct_fake += is_correct
            results["fake"].append({"name": name, "real_prob": float(real_prob), "ai_prob": float(ai_prob), "correct": is_correct})

        total_real = len(real_images)
        total_fake = len(fake_images)
        real_acc = correct_real / total_real * 100 if total_real > 0 else 0
        fake_acc = correct_fake / total_fake * 100 if total_fake > 0 else 0
        overall_acc = (correct_real + correct_fake) / (total_real + total_fake) * 100

        print(f"\n📊 Results for CLIP ViT-L Zero-Shot:")
        print(f"  Real images: {correct_real}/{total_real} ({real_acc:.1f}%)")
        print(f"  Fake images: {correct_fake}/{total_fake} ({fake_acc:.1f}%)")
        print(f"  Overall: {overall_acc:.1f}%")

        results["metrics"] = {
            "real_accuracy": real_acc,
            "fake_accuracy": fake_acc,
            "overall_accuracy": overall_acc
        }

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return results

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return {"model": "CLIP-ViT-L Zero-Shot", "error": str(e)}

def main():
    print("🔍 Pre-trained AI Image Detector Evaluation")
    print(f"Real images: {REAL_DIR}")
    print(f"Fake images: {FAKE_DIR}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Load all images once
    real_images = load_images(REAL_DIR)
    fake_images = load_images(FAKE_DIR)
    print(f"\nLoaded {len(real_images)} real, {len(fake_images)} fake images")

    all_results = []

    # Test each model
    for model_name in MODELS:
        result = test_detector(model_name, real_images, fake_images)
        all_results.append(result)

    # Test CLIP zero-shot
    clip_result = test_clip_zero_shot()
    all_results.append(clip_result)

    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY - All Models")
    print("="*60)
    print(f"{'Model':<45} {'Real%':>8} {'Fake%':>8} {'Overall':>8}")
    print("-"*70)

    for r in all_results:
        if "error" in r:
            print(f"{r['model']:<45} {'ERROR':>8}")
        else:
            m = r.get("metrics", {})
            print(f"{r['model']:<45} {m.get('real_accuracy', 0):>7.1f}% {m.get('fake_accuracy', 0):>7.1f}% {m.get('overall_accuracy', 0):>7.1f}%")

    # Save results
    with open("detector_comparison.json", "w") as f:
        # Convert non-serializable items
        def serialize(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)
        json.dump(all_results, f, indent=2, default=serialize)

    print("\nResults saved to detector_comparison.json")

    # Find best model
    best = max([r for r in all_results if "error" not in r],
               key=lambda x: x.get("metrics", {}).get("overall_accuracy", 0))
    print(f"\n🏆 Best model: {best['model']} ({best.get('metrics', {}).get('overall_accuracy', 0):.1f}% accuracy)")

if __name__ == "__main__":
    main()
