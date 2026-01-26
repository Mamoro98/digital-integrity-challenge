"""
Module 2: VLM Logic Reasoner
Semantic-level analysis using Vision-Language Models
"""

import os
from typing import Dict
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

        # Check for API keys first
        if os.environ.get("GEMINI_API_KEY"):
            return "gemini"
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"

        # Default to Qwen2-VL (works without auth, good quality)
        return "qwen2vl"

    def _init_backend(self):
        print(f"Initializing VLM backend: {self.backend}")

        if self.backend == "qwen2vl":
            self._init_qwen2vl()
        elif self.backend == "blip2":
            self._init_blip2()
        elif self.backend == "anthropic":
            self._init_anthropic()
        elif self.backend == "openai":
            self._init_openai()
        elif self.backend == "gemini":
            self._init_gemini()
        elif self.backend == "mock":
            print("Warning: Using mock VLM backend")

    def _init_qwen2vl(self):
        """Initialize Qwen2-VL-2B-Instruct model."""
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        import torch

        model_id = "Qwen/Qwen2-VL-2B-Instruct"
        print(f"Loading {model_id}...")

        self.processor = Qwen2VLProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
        self.model.eval()
        print("Qwen2-VL loaded successfully!")

    def _init_blip2(self):
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import torch

        model_id = "Salesforce/blip2-opt-2.7b"
        print(f"Loading {model_id}...")

        self.processor = Blip2Processor.from_pretrained(model_id)
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

    def _init_gemini(self):
        """Initialize Gemini API."""
        import google.generativeai as genai
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-pro')
        print("Gemini 1.5 Pro initialized!")

    def _analyze_gemini(self, image_path: str) -> Dict:
        """Analyze image using Gemini 1.5 Pro."""
        from PIL import Image
        
        image = Image.open(image_path)
        
        prompt = """Analyze this real estate image for AI manipulation. Check shadows, reflections, geometry, textures.

Respond EXACTLY in this format (no extra text):
MANIPULATION_DETECTED: YES or NO or UNCERTAIN
CONFIDENCE: HIGH or MEDIUM or LOW
MANIPULATION_TYPE: authentic or virtual_staging or inpainting or full_synthesis
REASONING: Two sentences explaining why."""

        response = self.gemini_model.generate_content([prompt, image])
        return self._parse_structured_response(response.text)

    def analyze(self, image_path: str) -> Dict:
        if self.backend == "qwen2vl":
            return self._analyze_qwen2vl(image_path)
        elif self.backend == "blip2":
            return self._analyze_blip2(image_path)
        elif self.backend == "gemini":
            return self._analyze_gemini(image_path)
        elif self.backend == "anthropic":
            return self._analyze_anthropic(image_path)
        elif self.backend == "openai":
            return self._analyze_openai(image_path)
        else:
            return self._analyze_mock(image_path)

    def _analyze_qwen2vl(self, image_path: str) -> Dict:
        """Analyze image using Qwen2-VL."""
        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")

        prompt = """Analyze this real estate image for signs of AI generation or manipulation.

Check for:
1. Lighting inconsistencies or impossible shadows
2. Texture problems (too smooth, repeating patterns, blurry areas)
3. Geometric errors (warped lines, impossible angles)
4. Object anomalies (floating furniture, merged objects)
5. Reflection inconsistencies

Respond in this exact format:
VERDICT: REAL or FAKE
CONFIDENCE: HIGH, MEDIUM, or LOW
TYPE: authentic, virtual_staging, inpainting, full_synthesis, or unknown
REASON: One sentence explaining your assessment."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)

        response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # Extract assistant response
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()

        return self._parse_qwen2vl_response(response)

    def _parse_qwen2vl_response(self, response: str) -> Dict:
        """Parse Qwen2-VL structured response."""
        result = {
            "manipulation_detected": "uncertain",
            "confidence": "medium",
            "manipulation_type": "unknown",
            "reasoning": response[:300]
        }

        response_upper = response.upper()
        response_lower = response.lower()

        # Parse VERDICT
        if "VERDICT:" in response_upper:
            verdict_line = [l for l in response.split('\n') if 'VERDICT:' in l.upper()]
            if verdict_line:
                verdict = verdict_line[0].upper()
                if "FAKE" in verdict:
                    result["manipulation_detected"] = "yes"
                elif "REAL" in verdict:
                    result["manipulation_detected"] = "no"
        else:
            # Fallback: look for keywords
            if any(w in response_lower for w in ["fake", "generated", "manipulated", "synthetic", "artificial"]):
                result["manipulation_detected"] = "yes"
            elif any(w in response_lower for w in ["real", "authentic", "genuine", "photograph"]):
                result["manipulation_detected"] = "no"

        # Parse CONFIDENCE
        if "CONFIDENCE:" in response_upper:
            conf_line = [l for l in response.split('\n') if 'CONFIDENCE:' in l.upper()]
            if conf_line:
                conf = conf_line[0].upper()
                if "HIGH" in conf:
                    result["confidence"] = "high"
                elif "LOW" in conf:
                    result["confidence"] = "low"
                else:
                    result["confidence"] = "medium"

        # Parse TYPE
        if "TYPE:" in response_upper:
            type_line = [l for l in response.split('\n') if 'TYPE:' in l.upper()]
            if type_line:
                type_text = type_line[0].split(':', 1)[-1].strip().lower()
                if type_text in ["authentic", "virtual_staging", "inpainting", "full_synthesis"]:
                    result["manipulation_type"] = type_text
                elif "staging" in type_text:
                    result["manipulation_type"] = "virtual_staging"
                elif "inpaint" in type_text:
                    result["manipulation_type"] = "inpainting"
                elif "synth" in type_text or "generat" in type_text:
                    result["manipulation_type"] = "full_synthesis"

        # Parse REASON
        if "REASON:" in response_upper:
            reason_line = [l for l in response.split('\n') if 'REASON:' in l.upper()]
            if reason_line:
                result["reasoning"] = reason_line[0].split(':', 1)[-1].strip()[:300]

        # Set manipulation_type based on detection if still unknown
        if result["manipulation_type"] == "unknown":
            if result["manipulation_detected"] == "yes":
                result["manipulation_type"] = "manipulation_detected"
            elif result["manipulation_detected"] == "no":
                result["manipulation_type"] = "authentic"

        return result

    def _analyze_blip2(self, image_path: str) -> Dict:
        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")

        questions = [
            "Question: Is this real estate image a real photograph or AI generated? Answer:",
            "Question: Are there any visual artifacts or inconsistencies in this image? Answer:",
            "Question: Do the shadows and lighting look natural in this image? Answer:",
            "Question: Are the surfaces and textures realistic or unnaturally smooth? Answer:",
        ]

        answers = []
        for q in questions:
            inputs = self.processor(image, text=q, return_tensors="pt")

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=50)

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
        if "generated" in first_answer or "ai" in first_answer or "fake" in first_answer:
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

        return self._parse_structured_response(message.content[0].text)

    def _analyze_openai(self, image_path: str) -> Dict:
        return self._analyze_mock(image_path)

    def _analyze_mock(self, image_path: str) -> Dict:
        return {
            "manipulation_detected": "uncertain",
            "confidence": "low",
            "manipulation_type": "unknown",
            "reasoning": "Mock backend - no model available."
        }

    def _parse_structured_response(self, response: str) -> Dict:
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


class GeminiVLMReasoner(VLMReasoner):
    """Gemini-specific VLM reasoner."""
    
    def __init__(self):
        self.backend = "gemini"
        self._init_gemini()
    
    def _init_gemini(self):
        """Initialize Gemini API."""
        import google.generativeai as genai
        import os
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        print("Gemini 1.5 Pro initialized!")
    
    def analyze(self, image_path: str) -> Dict:
        """Analyze image using Gemini."""
        import google.generativeai as genai
        from PIL import Image
        
        image = Image.open(image_path)
        
        prompt = """Analyze this real estate image for AI manipulation. Check shadows, reflections, geometry, textures.

Respond EXACTLY in this format (no extra text):
MANIPULATION_DETECTED: YES or NO or UNCERTAIN
CONFIDENCE: HIGH or MEDIUM or LOW
MANIPULATION_TYPE: authentic or virtual_staging or inpainting or full_synthesis
REASONING: Two sentences explaining why."""

        response = self.model.generate_content([prompt, image])
        return self._parse_structured_response(response.text)
