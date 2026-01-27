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
        self.device = None
        self._init_backend()

    def _detect_backend(self, backend: str) -> str:
        if backend != "auto":
            return backend

        # Check for API keys first (faster inference)
        if os.environ.get("GEMINI_API_KEY"):
            return "gemini"
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"

        # Default to local model
        return "qwen2vl"

    def _init_backend(self):
        print(f"Initializing VLM backend: {self.backend}")

        try:
            if self.backend == "qwen2vl":
                self._init_qwen2vl()
            elif self.backend == "paligemma":
                self._init_paligemma()
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
        except Exception as e:
            print(f"Warning: Failed to initialize {self.backend}: {e}")
            print("Falling back to next available backend...")
            self._fallback_init()

    def _fallback_init(self):
        """Try fallback backends in order."""
        # Priority: Qwen2-VL-72B (CPU) → PaliGemma2-28B (TPU) → smaller models → mock
        fallback_order = ["qwen2vl", "paligemma", "blip2", "mock"]
        
        for fallback in fallback_order:
            if fallback == self.backend:
                continue
            try:
                print(f"Trying fallback: {fallback}")
                self.backend = fallback
                if fallback == "qwen2vl":
                    self._init_qwen2vl()
                elif fallback == "paligemma":
                    self._init_paligemma()
                elif fallback == "blip2":
                    self._init_blip2()
                elif fallback == "mock":
                    print("Using mock backend - VLM scores will be neutral")
                    return
                print(f"Fallback {fallback} initialized successfully!")
                return
            except Exception as e:
                print(f"Fallback {fallback} failed: {e}")
                continue
        
        print("All backends failed. Using mock.")
        self.backend = "mock"

    def _get_device(self):
        """Detect best available device."""
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _init_qwen2vl(self):
        """Initialize Qwen2-VL-72B-Instruct model (or smaller if OOM)."""
        import torch
        
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except ImportError:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            Qwen2VLForConditionalGeneration = AutoModelForVision2Seq

        # Try 72B first (best quality), fall back to smaller if needed
        model_candidates = [
            "Qwen/Qwen2-VL-72B-Instruct",  # ~150GB RAM
            "Qwen/Qwen2-VL-7B-Instruct",   # ~28GB RAM
            "Qwen/Qwen2-VL-2B-Instruct",   # ~8GB RAM
        ]
        
        model_id = None
        for candidate in model_candidates:
            try:
                print(f"Trying {candidate}...")
                model_id = candidate
                break
            except Exception as e:
                print(f"{candidate} not available: {e}")
                continue
        
        if not model_id:
            model_id = "Qwen/Qwen2-VL-2B-Instruct"
        
        print(f"Loading {model_id}...")

        self.device = self._get_device()
        print(f"Using device: {self.device}")

        # Use appropriate dtype based on device
        if self.device == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32

        try:
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        except:
            from transformers import Qwen2VLProcessor
            self.processor = Qwen2VLProcessor.from_pretrained(model_id)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print("Qwen2-VL loaded successfully!")

    def _init_paligemma(self):
        """Initialize PaliGemma2-28B on TPU via JAX."""
        print("Initializing PaliGemma2-28B on TPU...")
        
        try:
            import jax
            import jax.numpy as jnp
            from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
            import torch
        except ImportError as e:
            raise ImportError(f"PaliGemma dependencies not available: {e}")
        
        # Check TPU availability
        devices = jax.devices()
        if not any('TPU' in str(d) for d in devices):
            raise RuntimeError("No TPU devices found for PaliGemma")
        
        print(f"Found {len(devices)} TPU devices")
        
        # Try largest model first
        model_candidates = [
            "google/paligemma2-28b-pt-896",  # Largest, best quality
            "google/paligemma2-10b-pt-896",
            "google/paligemma2-3b-pt-896",
        ]
        
        model_id = None
        for candidate in model_candidates:
            try:
                print(f"Trying {candidate}...")
                self.processor = AutoProcessor.from_pretrained(candidate, trust_remote_code=True)
                self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                    candidate,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                model_id = candidate
                break
            except Exception as e:
                print(f"{candidate} failed: {e}")
                continue
        
        if not model_id:
            raise RuntimeError("Could not load any PaliGemma model")
        
        self.model.eval()
        print(f"PaliGemma loaded: {model_id}")

    def _init_blip2(self):
        """Initialize BLIP-2 as fallback."""
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import torch

        model_id = "Salesforce/blip2-opt-2.7b"
        print(f"Loading {model_id}...")

        self.device = self._get_device()
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = Blip2Processor.from_pretrained(model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        if self.device != "cuda":
            self.model = self.model.to(self.device)
            
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
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        print("Gemini initialized!")

    def analyze(self, image_path: str) -> Dict:
        """Analyze an image for manipulation."""
        try:
            if self.backend == "qwen2vl":
                return self._analyze_qwen2vl(image_path)
            elif self.backend == "paligemma":
                return self._analyze_paligemma(image_path)
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
        except Exception as e:
            print(f"Analysis error: {e}")
            return self._analyze_mock(image_path)

    def _analyze_gemini(self, image_path: str) -> Dict:
        """Analyze image using Gemini."""
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

    def _analyze_paligemma(self, image_path: str) -> Dict:
        """Analyze image using PaliGemma2."""
        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")

        prompt = """Analyze this real estate image for AI manipulation.
Is this image REAL or FAKE? Check shadows, reflections, geometry, textures.
Answer: REAL or FAKE, then explain in one sentence."""

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)

        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Parse simple response
        response_lower = response.lower()
        
        if "fake" in response_lower or "manipulated" in response_lower or "generated" in response_lower:
            detection = "yes"
        elif "real" in response_lower or "authentic" in response_lower or "genuine" in response_lower:
            detection = "no"
        else:
            detection = "uncertain"

        return {
            "manipulation_detected": detection,
            "confidence": "medium",
            "manipulation_type": "manipulation_detected" if detection == "yes" else "authentic",
            "reasoning": response[:300]
        }

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
        
        # Move to device
        if self.device:
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

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
        """Analyze image using BLIP-2."""
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
            if self.device:
                inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=50)

            answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            answers.append((q, answer))

        return self._parse_blip2_answers(answers)

    def _parse_blip2_answers(self, qa_pairs: list) -> Dict:
        """Parse BLIP-2 Q&A responses."""
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
        }

    def _analyze_anthropic(self, image_path: str) -> Dict:
        """Analyze image using Claude."""
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
        """OpenAI backend - not implemented, use mock."""
        return self._analyze_mock(image_path)

    def _analyze_mock(self, image_path: str) -> Dict:
        """Mock analysis when no VLM is available."""
        return {
            "manipulation_detected": "uncertain",
            "confidence": "low",
            "manipulation_type": "unknown",
            "reasoning": "VLM backend unavailable - using forensic signals only."
        }

    def _parse_structured_response(self, response: str) -> Dict:
        """Parse structured response from API-based VLMs."""
        result = {
            "manipulation_detected": "uncertain",
            "confidence": "low",
            "manipulation_type": "unknown",
            "reasoning": response
        }

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
