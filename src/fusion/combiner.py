"""
Fusion Module: Combines forensic and VLM results
"""

from typing import Dict


class FusionModule:
    """Combines pixel-level forensics with semantic VLM analysis."""
    
    def __init__(self):
        # Weights for combining scores
        self.forensic_weight = 0.4
        self.vlm_weight = 0.6
        
    def combine(self, forensic_results: Dict, vlm_results: Dict) -> Dict:
        """
        Combine forensic and VLM results into final prediction.
        
        Args:
            forensic_results: Output from ForensicDetector
            vlm_results: Output from VLMReasoner
            
        Returns:
            Final prediction dict with score, type, and reasoning
        """
        
        # Get forensic score (already 0-1)
        forensic_score = forensic_results.get("aggregate_score", 0.5)
        
        # Convert VLM result to score
        vlm_score = self._vlm_to_score(vlm_results)
        
        # Weighted combination
        final_score = (
            self.forensic_weight * forensic_score +
            self.vlm_weight * vlm_score
        )
        
        # Determine manipulation type
        manipulation_type = self._determine_type(forensic_results, vlm_results, final_score)
        
        # Generate combined reasoning
        reasoning = self._generate_reasoning(forensic_results, vlm_results)
        
        return {
            "score": round(final_score, 3),
            "manipulation_type": manipulation_type,
            "reasoning": reasoning,
            "forensic_score": round(forensic_score, 3),
            "vlm_score": round(vlm_score, 3)
        }
    
    def _vlm_to_score(self, vlm_results: Dict) -> float:
        """Convert VLM categorical output to numeric score."""
        
        base_score = 0.5  # Uncertain default
        
        detection = vlm_results.get("manipulation_detected", "uncertain")
        confidence = vlm_results.get("confidence", "low")
        
        # Base score from detection
        if detection == "yes":
            base_score = 0.8
        elif detection == "no":
            base_score = 0.2
        
        # Adjust by confidence
        confidence_multiplier = {"high": 1.0, "medium": 0.7, "low": 0.4}
        multiplier = confidence_multiplier.get(confidence, 0.5)
        
        # Move score toward extremes based on confidence
        if detection == "yes":
            score = 0.5 + (base_score - 0.5) * multiplier
        elif detection == "no":
            score = 0.5 - (0.5 - base_score) * multiplier
        else:
            score = 0.5
        
        return score
    
    def _determine_type(self, forensic: Dict, vlm: Dict, final_score: float) -> str:
        """Determine the most likely manipulation type."""
        
        # If score is low, it's likely authentic
        if final_score < 0.35:
            return "authentic"
        
        # Use VLM type if confident
        vlm_type = vlm.get("manipulation_type", "unknown")
        if vlm_type and vlm_type not in ["unknown", "authentic"]:
            return vlm_type
        
        # Infer from forensic signals
        ela_score = forensic.get("ela_score", 0)
        texture_score = forensic.get("texture_score", 0)
        fft_score = forensic.get("fft_score", 0)
        
        # High ELA variance suggests inpainting/splicing
        if ela_score > 0.6:
            return "inpainting"
        
        # High FFT score suggests full AI generation
        if fft_score > 0.7:
            return "full_synthesis"
        
        # High texture smoothness suggests heavy filtering or AI
        if texture_score > 0.6:
            return "virtual_staging"
        
        return "manipulation_detected"
    
    def _generate_reasoning(self, forensic: Dict, vlm: Dict) -> str:
        """Generate human-readable reasoning."""
        
        reasons = []
        
        # VLM reasoning
        vlm_reasoning = vlm.get("reasoning", "")
        if vlm_reasoning and "mock" not in vlm_reasoning.lower():
            reasons.append(vlm_reasoning)
        
        # Forensic insights
        forensic_insights = []
        
        if forensic.get("ela_score", 0) > 0.5:
            forensic_insights.append("compression inconsistencies detected")
        
        if forensic.get("fft_score", 0) > 0.5:
            forensic_insights.append("unusual frequency patterns found")
            
        if forensic.get("texture_score", 0) > 0.5:
            forensic_insights.append("unnaturally smooth textures observed")
            
        if forensic.get("noise_score", 0) > 0.5:
            forensic_insights.append("inconsistent noise patterns")
        
        if forensic_insights:
            reasons.append("Forensic analysis: " + ", ".join(forensic_insights) + ".")
        
        if not reasons:
            if forensic.get("aggregate_score", 0.5) < 0.35:
                return "Image appears authentic with consistent lighting, shadows, and textures."
            else:
                return "Some anomalies detected but analysis is inconclusive."
        
        # Limit to 2 sentences
        combined = " ".join(reasons)
        sentences = combined.split('. ')
        return '. '.join(sentences[:2]).strip()
        if not combined.endswith('.'):
            combined += '.'
        return combined
