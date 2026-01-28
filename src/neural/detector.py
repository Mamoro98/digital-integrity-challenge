"""
Neural Network-based AI Image Detector
Uses pre-trained models from HuggingFace for detecting AI-generated images.

Based on research recommendations:
- DINOv2/CLIP for feature extraction
- Pre-trained deepfake detectors
- Ensemble approach for robustness
"""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, Optional, Tuple
import numpy as np
import os

# Lazy imports to avoid loading everything at startup
_clip_model = None
_clip_processor = None
_ai_detector = None
_ai_detector_processor = None


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NeuralDetector:
    """
    Neural network-based detector using pre-trained models.
    
    Uses:
    1. CLIP for zero-shot AI image detection
    2. Pre-trained AI image detector from HuggingFace
    3. Ensemble of both for robust predictions
    """
    
    def __init__(self, use_clip: bool = True, use_ai_detector: bool = True):
        """
        Initialize the neural detector.
        
        Args:
            use_clip: Whether to use CLIP for zero-shot detection
            use_ai_detector: Whether to use pre-trained AI detector
        """
        self.device = get_device()
        self.use_clip = use_clip
        self.use_ai_detector = use_ai_detector
        
        # Models loaded lazily on first use
        self._clip_loaded = False
        self._detector_loaded = False
        
    def _load_clip(self):
        """Load CLIP model for zero-shot classification."""
        if self._clip_loaded:
            return
            
        global _clip_model, _clip_processor
        
        if _clip_model is None:
            from transformers import CLIPProcessor, CLIPModel
            print("Loading CLIP model...")
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model = _clip_model.to(self.device)
            _clip_model.eval()
            print("CLIP model loaded.")
            
        self._clip_loaded = True
        
    def _load_ai_detector(self):
        """Load pre-trained AI image detector."""
        if self._detector_loaded:
            return
            
        global _ai_detector, _ai_detector_processor
        
        if _ai_detector is None:
            from transformers import AutoModelForImageClassification, AutoImageProcessor
            print("Loading AI image detector...")
            
            # Try different models in order of preference
            models_to_try = [
                "umm-maybe/AI-image-detector",  # General AI detector
                "Organika/sdxl-detector",       # SDXL specific
            ]
            
            for model_name in models_to_try:
                try:
                    _ai_detector = AutoModelForImageClassification.from_pretrained(model_name)
                    _ai_detector_processor = AutoImageProcessor.from_pretrained(model_name)
                    _ai_detector = _ai_detector.to(self.device)
                    _ai_detector.eval()
                    print(f"Loaded AI detector: {model_name}")
                    break
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")
                    continue
            
            if _ai_detector is None:
                print("Warning: No AI detector model available. Using CLIP only.")
                self.use_ai_detector = False
                
        self._detector_loaded = True
        
    def analyze_with_clip(self, image: Image.Image) -> Dict:
        """
        Use CLIP for zero-shot AI image detection.
        
        Research shows CLIP can detect AI images by comparing embeddings
        to text descriptions like "AI generated image" vs "real photograph".
        """
        self._load_clip()
        
        # Text prompts for classification
        # Based on research: be specific about what we're looking for
        text_prompts = [
            "a real photograph taken by a camera",
            "an AI generated image, synthetic, artificial, computer generated",
        ]
        
        inputs = _clip_processor(
            text=text_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = _clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = F.softmax(logits_per_image, dim=1)
            
        # prob[0] = real, prob[1] = AI
        probs = probs.cpu().numpy()[0]
        
        return {
            "clip_real_prob": float(probs[0]),
            "clip_fake_prob": float(probs[1]),
            "clip_score": float(probs[1]),  # Higher = more likely AI
        }
        
    def analyze_with_detector(self, image: Image.Image) -> Dict:
        """
        Use pre-trained AI image detector.
        """
        self._load_ai_detector()
        
        if _ai_detector is None:
            return {"detector_score": 0.5, "detector_available": False}
            
        inputs = _ai_detector_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = _ai_detector(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            
        probs = probs.cpu().numpy()[0]
        
        # Model typically has labels like ['artificial', 'human'] or similar
        # Check the label order
        labels = _ai_detector.config.id2label
        
        # Find which index corresponds to AI/fake
        fake_idx = None
        for idx, label in labels.items():
            if any(kw in label.lower() for kw in ['artificial', 'ai', 'fake', 'synthetic', 'generated']):
                fake_idx = idx
                break
                
        if fake_idx is None:
            # Assume index 0 is AI (common convention)
            fake_idx = 0
            
        return {
            "detector_score": float(probs[fake_idx]),
            "detector_probs": {labels[i]: float(probs[i]) for i in range(len(probs))},
            "detector_available": True,
        }
        
    def analyze(self, image_path: str) -> Dict:
        """
        Analyze an image for AI generation.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict with detection results and aggregate score
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        results = {}
        scores = []
        weights = []
        
        # CLIP analysis
        if self.use_clip:
            try:
                clip_results = self.analyze_with_clip(image)
                results.update(clip_results)
                scores.append(clip_results["clip_score"])
                weights.append(0.4)  # CLIP weight
            except Exception as e:
                results["clip_error"] = str(e)
                
        # Pre-trained detector analysis
        if self.use_ai_detector:
            try:
                detector_results = self.analyze_with_detector(image)
                results.update(detector_results)
                if detector_results.get("detector_available", False):
                    scores.append(detector_results["detector_score"])
                    weights.append(0.6)  # Pre-trained detector weight (higher trust)
            except Exception as e:
                results["detector_error"] = str(e)
                
        # Compute aggregate score
        if scores:
            # Weighted average
            total_weight = sum(weights)
            aggregate = sum(s * w for s, w in zip(scores, weights)) / total_weight
            results["neural_aggregate_score"] = float(aggregate)
        else:
            results["neural_aggregate_score"] = 0.5  # Neutral if no models worked
            
        return results


class DINOv2Detector:
    """
    DINOv2-based detector for AI image detection.
    
    Research shows DINOv2 features are highly discriminative for AI vs real images.
    This uses DINOv2 as a feature extractor with a simple classifier head.
    
    Note: This requires training on labeled data, so we use it in feature extraction
    mode and combine with other signals.
    """
    
    def __init__(self):
        self.device = get_device()
        self.model = None
        self.processor = None
        
    def _load_model(self):
        if self.model is not None:
            return
            
        from transformers import AutoImageProcessor, AutoModel
        print("Loading DINOv2 model...")
        
        # Use smaller variant for CPU
        model_name = "facebook/dinov2-small"
        
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("DINOv2 model loaded.")
        
    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract DINOv2 features from an image."""
        self._load_model()
        
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token as image representation
            features = outputs.last_hidden_state[:, 0, :]
            
        return features.cpu().numpy()[0]
        
    def analyze(self, image_path: str) -> Dict:
        """
        Analyze image using DINOv2 features.
        
        Since we don't have a trained classifier, we use statistical properties
        of the features that research shows differ between AI and real images.
        """
        features = self.extract_features(image_path)
        
        # Research insight: AI images tend to have more uniform feature distributions
        # Real images have more varied, scene-specific features
        
        feature_std = np.std(features)
        feature_kurtosis = self._kurtosis(features)
        feature_entropy = self._entropy(features)
        
        # Normalize to 0-1 scores
        # Based on empirical observation: AI images have lower std, lower kurtosis
        # These thresholds would need calibration on actual data
        
        std_score = 1 - np.clip(feature_std / 1.0, 0, 1)  # Lower std = more suspicious
        kurtosis_score = 1 - np.clip((feature_kurtosis + 2) / 6, 0, 1)  # Lower kurtosis = suspicious
        
        # Weighted combination
        dino_score = 0.6 * std_score + 0.4 * kurtosis_score
        
        return {
            "dino_feature_std": float(feature_std),
            "dino_feature_kurtosis": float(feature_kurtosis),
            "dino_feature_entropy": float(feature_entropy),
            "dino_score": float(np.clip(dino_score, 0, 1)),
        }
        
    def _kurtosis(self, x):
        """Compute kurtosis of array."""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0
        return np.sum(((x - mean) / std) ** 4) / n - 3
        
    def _entropy(self, x):
        """Compute entropy of feature distribution."""
        # Discretize features into bins
        hist, _ = np.histogram(x, bins=50, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist + 1e-10))


def test_neural_detector():
    """Test the neural detector on sample images."""
    import glob
    
    detector = NeuralDetector()
    
    # Find test images
    fake_images = glob.glob("/home/omer_aims_ac_za/digital-integrity-challenge/data/ai_generated_v2/*.png")[:5]
    real_images = glob.glob("/home/omer_aims_ac_za/digital-integrity-challenge/data/real/*.jpg")[:5]
    
    print("\n=== Testing on FAKE images ===")
    fake_scores = []
    for img_path in fake_images:
        results = detector.analyze(img_path)
        score = results.get("neural_aggregate_score", 0.5)
        fake_scores.append(score)
        print(f"{os.path.basename(img_path)}: {score:.3f}")
    
    print("\n=== Testing on REAL images ===")
    real_scores = []
    for img_path in real_images:
        results = detector.analyze(img_path)
        score = results.get("neural_aggregate_score", 0.5)
        real_scores.append(score)
        print(f"{os.path.basename(img_path)}: {score:.3f}")
    
    print(f"\n=== Summary ===")
    print(f"FAKE avg: {np.mean(fake_scores):.3f}")
    print(f"REAL avg: {np.mean(real_scores):.3f}")
    print(f"Separation: {np.mean(fake_scores) - np.mean(real_scores):.3f}")
    
    # Good detector should have FAKE > REAL scores
    accuracy = (sum(1 for s in fake_scores if s >= 0.5) + sum(1 for s in real_scores if s < 0.5)) / (len(fake_scores) + len(real_scores))
    print(f"Accuracy (threshold=0.5): {accuracy*100:.1f}%")


if __name__ == "__main__":
    test_neural_detector()
