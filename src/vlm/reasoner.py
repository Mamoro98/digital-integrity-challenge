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
        """
        Initialize VLM reasoner.
        
        Args:
            backend: "qwen", "llava", "openai", "anthropic", or "auto"
        """
        self.backend = self._detect_backend(backend)
        self.model = None
        self.processor = None
        
        self._init_backend()
    
    def _detect_backend(self, backend: str) -> str:
        """Auto-detect available backend."""
        if backend != "auto":
            return backend
        
        # Check for API keys first (faster inference)
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        
        # Fall back to local models
        try:
            import torch
            if torch.cuda.is_available():
                return "qwen"  # Qwen-VL works well on GPU
        except ImportError:
            pass
        
        return "mock"  # Fallback for testing
    
    def _init_backend(self):
        """Initialize the chosen backend."""
        print(f"Initializing VLM backend: {self.backend}")
        
        if self.backend == "qwen":
            self._init_qwen()
        elif self.backend == "anthropic":
            self._init_anthropic()
        elif self.backend == "openai":
            self._init_openai()
        elif self.backend == "mock":
            print("Warning: Using mock VLM backend")
    
    def _init_qwen(self):
        """Initialize Qwen2-VL model."""
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        import torch
        
        model_name = "Qwen/Qwen2-VL-2B-Instruct"  # Smaller version for speed
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    def _init_anthropic(self):
        """Initialize Anthropic API client."""
        import anthropic
        self.client = anthropic.Anthropic()
    
    def _init_openai(self):
        """Initialize OpenAI API client."""
        from openai import OpenAI
        self.client = OpenAI()
    
    def analyze(self, image_path: str) -> Dict:
        """Analyze image for semantic anomalies."""
        
        prompt = self._get_analysis_prompt()
        
        if self.backend == "qwen":
            response = self._analyze_qwen(image_path, prompt)
        elif self.backend == "anthropic":
            response = self._analyze_anthropic(image_path, prompt)
        elif self.backend == "openai":
            response = self._analyze_openai(image_path, prompt)
        else:
            response = self._analyze_mock(image_path)
        
        return self._parse_response(response)
    
    def _get_analysis_prompt(self) -> str:
        """Get the analysis prompt for real estate images."""
        return """Analyze this real estate image for signs of AI manipulation or digital editing.

Check for these specific issues:
1. SHADOW CONSISTENCY: Do shadows match the apparent light source direction?
2. REFLECTION ACCURACY: Do reflections in windows/mirrors match the actual room?
3. GEOMETRIC INTEGRITY: Are there impossible angles, warped lines, or furniture merging into walls?
4. TEXTURE ANOMALIES: Are surfaces unnaturally smooth or perfectly uniform?
5. LIGHTING COHERENCE: Is the lighting consistent across all objects in the scene?
6. OBJECT BOUNDARIES: Are edges between objects too sharp or unnaturally blended?
7. PERSPECTIVE: Does the perspective make geometric sense?

Respond in this exact format:
MANIPULATION_DETECTED: [YES/NO/UNCERTAIN]
CONFIDENCE: [HIGH/MEDIUM/LOW]
MANIPULATION_TYPE: [one of: authentic, virtual_staging, inpainting, full_synthesis, filter, object_removal, object_addition]
REASONING: [2 sentences explaining the key red flags or why it appears authentic]
"""
    
    def _analyze_qwen(self, image_path: str, prompt: str) -> str:
        """Analyze using Qwen2-VL."""
        from PIL import Image
        
        image = Image.open(image_path)
        
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
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return output
    
    def _analyze_anthropic(self, image_path: str, prompt: str) -> str:
        """Analyze using Claude Vision."""
        import base64
        
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
        
        # Detect media type
        ext = Path(image_path).suffix.lower()
        media_types = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}
        media_type = media_types.get(ext, 'image/jpeg')
        
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        )
        
        return message.content[0].text
    
    def _analyze_openai(self, image_path: str, prompt: str) -> str:
        """Analyze using GPT-4 Vision."""
        import base64
        
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
        
        ext = Path(image_path).suffix.lower()
        media_types = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}
        media_type = media_types.get(ext, 'image/jpeg')
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{image_data}"}
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        )
        
        return response.choices[0].message.content
    
    def _analyze_mock(self, image_path: str) -> str:
        """Mock analysis for testing."""
        return """MANIPULATION_DETECTED: UNCERTAIN
CONFIDENCE: LOW
MANIPULATION_TYPE: unknown
REASONING: Mock backend - no actual analysis performed. Please configure a real VLM backend."""
    
    def _parse_response(self, response: str) -> Dict:
        """Parse VLM response into structured format."""
        result = {
            "manipulation_detected": "uncertain",
            "confidence": "low",
            "manipulation_type": "unknown",
            "reasoning": response,
            "raw_response": response
        }
        
        lines = response.upper().split('\n')
        
        for line in lines:
            if 'MANIPULATION_DETECTED:' in line:
                if 'YES' in line:
                    result["manipulation_detected"] = "yes"
                elif 'NO' in line:
                    result["manipulation_detected"] = "no"
                    
            elif 'CONFIDENCE:' in line:
                if 'HIGH' in line:
                    result["confidence"] = "high"
                elif 'MEDIUM' in line:
                    result["confidence"] = "medium"
                    
            elif 'MANIPULATION_TYPE:' in line:
                type_part = line.split(':', 1)[1].strip().lower()
                result["manipulation_type"] = type_part
        
        # Extract reasoning (case-sensitive search in original)
        for line in response.split('\n'):
            if line.upper().startswith('REASONING:'):
                result["reasoning"] = line.split(':', 1)[1].strip()
                break
        
        return result
