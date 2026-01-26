"""
Module 2: VLM Logic Reasoner
Semantic-level analysis using Vision-Language Models
"""

import base64
import os
from typing import Dict, Optional
from pathlib import Path


class VLMReasoner:
    """Uses VLM to detect semantic anomalies and provide explanations."""
    
    def __init__(self, backend: str = "auto"):
        self.backend = self._detect_backend(backend)
        self.model = None
        self.processor = None
        self._init_backend()
    
    def _detect_backend(self, backend: str) -> str:
        if backend != "auto":
            return backend
        # Prefer PaliGemma on TPU
        try:
            import jax
            if len(jax.devices()) > 0:
                return "paligemma"
        except:
            pass
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        return "blip2"
    
    def _init_backend(self):
        print(f"Initializing VLM backend: {self.backend}")
        
        if self.backend == "paligemma":
            self._init_paligemma()
        elif self.backend == "blip2":
            self._init_blip2()
        elif self.backend == "anthropic":
            self._init_anthropic()
        elif self.backend == "openai":
            self._init_openai()
        elif self.backend == "mock":
            print("Warning: Using mock VLM backend")
    
    def _init_paligemma(self):
        from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
        import jax
        
        # Use PaliGemma 2 - best balance of speed and quality
        model_id = "google/paligemma2-3b-pt-224"
        print(f"Loading {model_id} on TPU ({len(jax.devices())} devices)...")
        
        self.processor = PaliGemmaProcessor.from_pretrained(model_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto"
        )
        self.model.eval()
        print("PaliGemma loaded successfully on TPU!")
    
    def _init_blip2(self):
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import torch
        
        model_id = "Salesforce/blip2-opt-2.7b"
        print(f"Loading {model_id}...")
        
        self.processor = Blip2Processor.from_pretrained(model_id)
        
        # Load without device_map for CPU
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32
        )
        self.model.eval()
        print("BLIP-2 loaded successfully!")
    
    def _init_anthropic(self):
        import anthropic
        self.client = anthropic.Anthropic()
    
    def _init_openai(self):
        from openai import OpenAI
        self.client = OpenAI()
    
    def analyze(self, image_path: str) -> Dict:
        if self.backend == "paligemma":
            return self._analyze_paligemma(image_path)
        elif self.backend == "blip2":
            return self._analyze_blip2(image_path)
        elif self.backend == "anthropic":
            return self._analyze_anthropic(image_path)
        elif self.backend == "openai":
            return self._analyze_openai(image_path)
        else:
            return self._analyze_mock(image_path)
    
    def _analyze_paligemma(self, image_path: str) -> Dict:
        from PIL import Image
        import torch
        
        image = Image.open(image_path).convert("RGB")
        
        # PaliGemma prompt for manipulation detection
        prompt = "Is this real estate image authentic or AI-generated/manipulated? Look for: unnatural lighting, impossible geometry, texture inconsistencies, shadow errors. Answer: REAL or FAKE, then explain why."
        
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False
            )
        
        # Decode only the generated tokens
        response = self.processor.decode(
            generated_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return self._parse_paligemma_response(response)
    
    def _parse_paligemma_response(self, response: str) -> Dict:
        response_lower = response.lower()
        
        # Determine detection result
        if "fake" in response_lower or "generated" in response_lower or "manipulated" in response_lower:
            detection = "yes"
            manip_type = "manipulation_detected"
        elif "real" in response_lower or "authentic" in response_lower or "genuine" in response_lower:
            detection = "no"
            manip_type = "authentic"
        else:
            detection = "uncertain"
            manip_type = "unknown"
        
        # Determine confidence based on certainty words
        if any(w in response_lower for w in ["clearly", "definitely", "obviously", "certainly"]):
            confidence = "high"
        elif any(w in response_lower for w in ["likely", "probably", "appears", "seems"]):
            confidence = "medium"
        else:
            confidence = "medium"  # default
        
        return {
            "manipulation_detected": detection,
            "confidence": confidence,
            "manipulation_type": manip_type,
            "reasoning": response[:300]
        }
    
    def _analyze_blip2(self, image_path: str) -> Dict:
        from PIL import Image
        import torch
        
        image = Image.open(image_path).convert("RGB")
        
        questions = [
            "Is this image real or AI generated?",
            "Are there any visual artifacts in this image?",
            "Do the shadows look natural?",
            "Are surfaces unnaturally smooth?",
        ]
        
        answers = []
        for q in questions:
            inputs = self.processor(image, text=q, return_tensors="pt")
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=30)
            
            answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            answers.append((q, answer))
        
        return self._parse_blip2_answers(answers)
    
    def _parse_blip2_answers(self, qa_pairs: list) -> Dict:
        combined = " ".join([a for q, a in qa_pairs]).lower()
        
        fake_signals = ["generated", "fake", "artificial", "synthetic", "manipulated", "artifacts", "unnatural"]
        real_signals = ["real", "natural", "authentic", "consistent", "genuine", "photograph"]
        
        fake_count = sum(1 for s in fake_signals if s in combined)
        real_count = sum(1 for s in real_signals if s in combined)
        
        if fake_count > real_count:
            detection = "yes"
            confidence = "high" if fake_count > 2 else "medium"
        elif real_count > fake_count:
            detection = "no"
            confidence = "high" if real_count > 2 else "medium"
        else:
            detection = "uncertain"
            confidence = "low"
        
        # Check first answer specifically
        first_answer = qa_pairs[0][1].lower() if qa_pairs else ""
        if "generated" in first_answer or "ai" in first_answer:
            detection = "yes"
        elif "real" in first_answer or "photograph" in first_answer:
            detection = "no"
        
        reasoning = " ".join([a for q, a in qa_pairs[1:3]])[:200]
        
        manip_type = "manipulation_detected" if detection == "yes" else "authentic"
        
        return {
            "manipulation_detected": detection,
            "confidence": confidence,
            "manipulation_type": manip_type,
            "reasoning": reasoning if reasoning else "Analysis based on visual inspection.",
            "raw_answers": qa_pairs
        }
    
    def _analyze_anthropic(self, image_path: str) -> Dict:
        import base64
        
        prompt = """Analyze this real estate image for AI manipulation. Check shadows, reflections, geometry, textures.
Respond exactly as:
MANIPULATION_DETECTED: YES/NO/UNCERTAIN
CONFIDENCE: HIGH/MEDIUM/LOW  
MANIPULATION_TYPE: authentic/virtual_staging/inpainting/full_synthesis
REASONING: Two sentences explaining why."""
        
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
        
        ext = Path(image_path).suffix.lower()
        media_type = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png'}.get(ext, 'image/jpeg')
        
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                {"type": "text", "text": prompt}
            ]}]
        )
        
        return self._parse_response(message.content[0].text)
    
    def _analyze_openai(self, image_path: str) -> Dict:
        # Similar to anthropic
        return self._analyze_mock(image_path)
    
    def _analyze_mock(self, image_path: str) -> Dict:
        return {
            "manipulation_detected": "uncertain",
            "confidence": "low",
            "manipulation_type": "unknown",
            "reasoning": "Mock backend."
        }
    
    def _parse_response(self, response: str) -> Dict:
        result = {"manipulation_detected": "uncertain", "confidence": "low", "manipulation_type": "unknown", "reasoning": response}
        
        for line in response.upper().split('\n'):
            if 'MANIPULATION_DETECTED:' in line:
                result["manipulation_detected"] = "yes" if 'YES' in line else ("no" if 'NO' in line else "uncertain")
            elif 'CONFIDENCE:' in line:
                result["confidence"] = "high" if 'HIGH' in line else ("medium" if 'MEDIUM' in line else "low")
            elif 'MANIPULATION_TYPE:' in line:
                result["manipulation_type"] = line.split(':', 1)[1].strip().lower()
        
        for line in response.split('\n'):
            if line.upper().startswith('REASONING:'):
                result["reasoning"] = line.split(':', 1)[1].strip()
                break
        
        return result
