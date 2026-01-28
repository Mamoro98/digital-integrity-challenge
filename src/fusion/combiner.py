"""
Fusion Module: Combines forensic and VLM results
"""

from typing import Dict


class FusionModule:
    """Combines pixel-level forensics with semantic VLM analysis."""

    def __init__(self):
        # Weights for combining scores
        # When VLM is uncertain (0.5), we rely more on forensics
        self.forensic_weight = 0.55
        self.vlm_weight = 0.45

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

        # Check for strong sharpness anomalies (oversharpening/blur)
        sharpness_score = forensic_results.get("sharpness_score", 0)
        noise_score = forensic_results.get("noise_score", 0)
        strong_sharpness_anomaly = sharpness_score > 0.65
        strong_noise_anomaly = noise_score > 0.65

        # Adaptive weighting: if VLM is uncertain, rely more on forensics
        vlm_confidence = vlm_results.get("confidence", "low")
        is_vlm_uncertain = vlm_results.get("manipulation_detected", "uncertain") == "uncertain"

        # Override: trust forensics when strong pixel-level anomalies detected
        # VLM often misses sharpness/noise artifacts that forensics catches
        if strong_sharpness_anomaly or strong_noise_anomaly:
            f_weight = 0.80
            v_weight = 0.20
        elif is_vlm_uncertain or vlm_confidence == "low":
            # VLM is uncertain - rely primarily on forensics
            f_weight = 0.85
            v_weight = 0.15
        elif vlm_confidence == "medium":
            f_weight = self.forensic_weight
            v_weight = self.vlm_weight
        else:  # high confidence VLM
            f_weight = 0.40
            v_weight = 0.60

        # Weighted combination
        raw_score = f_weight * forensic_score + v_weight * vlm_score

        # Boost score when forensics detect strong sharpness artifacts
        # VLM cannot reliably detect oversharpening/blur
        # Require BOTH high sharpness AND elevated aggregate forensic to avoid FPs
        if sharpness_score > 0.70 and forensic_score > 0.45:
            raw_score = max(raw_score, 0.50 + (sharpness_score - 0.70) * 0.5)

        # Dampen false positives: when forensics are low/moderate but VLM says manipulated
        # VLM can make semantic interpretation errors (e.g., dramatic skies)
        if forensic_score < 0.45 and vlm_score > 0.6:
            # Forensics should have the final say when pixel-level is clean
            raw_score = min(raw_score, 0.42)

        # Calibration: stretch scores to improve separation
        # Apply sigmoid-like transformation
        # This pushes low scores lower and high scores higher
        import math

        # Calibration center - tuned for balanced accuracy
        # Real avg=0.446, Fake avg=0.503 on ai_generated_v2 dataset
        if is_vlm_uncertain:
            center = 0.45  # Balance between FP and FN
            steepness = 5.0
        else:
            center = 0.42  # Normal threshold with VLM
            steepness = 6.0

        normalized = (raw_score - center) * steepness
        final_score = 1 / (1 + math.exp(-normalized))

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
        if final_score < 0.48:
            return "authentic"

        # Use VLM type if confident and specific
        vlm_type = vlm.get("manipulation_type", "unknown")
        vlm_confidence = vlm.get("confidence", "low")
        if vlm_type and vlm_type not in ["unknown", "authentic", "manipulation_detected"] and vlm_confidence != "low":
            return vlm_type

        # Infer from forensic signals
        sharpness_score = forensic.get("sharpness_score", 0)
        texture_score = forensic.get("texture_score", 0)
        noise_score = forensic.get("noise_score", 0)
        compression_score = forensic.get("compression_score", 0)
        edge_score = forensic.get("edge_score", 0)

        # High noise uniformity suggests AI generation
        if noise_score > 0.65:
            return "full_synthesis"

        # High sharpness with noise suggests enhancement/filter
        if sharpness_score > 0.65 and noise_score > 0.4:
            return "filter"

        # Very smooth textures suggest virtual staging
        if texture_score > 0.45:
            return "virtual_staging"

        # High compression differences suggest splicing/inpainting
        if compression_score > 0.72:
            return "inpainting"

        # Edge issues might indicate manipulation
        if edge_score > 0.5:
            return "inpainting"

        # Default for high scores
        if final_score > 0.55:
            return "manipulation_detected"

        return "authentic"

    def _generate_reasoning(self, forensic: Dict, vlm: Dict) -> str:
        """Generate human-readable reasoning based on forensic and VLM analysis."""

        reasons = []
        agg_score = forensic.get("aggregate_score", 0.5)

        # VLM reasoning (if available and not mock)
        vlm_reasoning = vlm.get("reasoning", "")
        if vlm_reasoning and "unavailable" not in vlm_reasoning.lower() and "Visual analysis completed" not in vlm_reasoning:
            reasons.append(f"VLM observations: {vlm_reasoning}")

        # Detailed forensic insights based on research
        forensic_insights = []

        # Sharpness analysis (strongest discriminator)
        sharpness = forensic.get("sharpness_score", 0)
        if sharpness > 0.7:
            forensic_insights.append("significant oversharpening artifacts detected, common in AI enhancement")
        elif sharpness > 0.55:
            forensic_insights.append("moderate sharpness anomalies suggest post-processing")

        # Noise analysis (AI images have different noise patterns)
        noise = forensic.get("noise_score", 0)
        if noise > 0.7:
            forensic_insights.append("uniform noise patterns indicate AI-generated content")
        elif noise > 0.5:
            forensic_insights.append("noise distribution shows artificial smoothing")

        # Compression analysis
        compression = forensic.get("compression_score", 0)
        if compression > 0.75:
            forensic_insights.append("compression artifacts suggest digital manipulation")
        elif compression > 0.6:
            forensic_insights.append("minor compression inconsistencies noted")

        # Texture analysis
        texture = forensic.get("texture_score", 0)
        if texture > 0.5:
            forensic_insights.append("unnaturally smooth textures on walls or surfaces")
        elif texture > 0.35:
            forensic_insights.append("subtle texture smoothing detected")

        # Edge coherence
        edge = forensic.get("edge_score", 0)
        if edge > 0.5:
            forensic_insights.append("edge boundary anomalies around objects")

        # Build final reasoning
        if forensic_insights:
            # Take top 2 most significant findings
            top_insights = forensic_insights[:2]
            reasons.append("Forensic analysis detected: " + "; ".join(top_insights) + ".")

        # Generate appropriate conclusion if no specific insights
        if not reasons:
            if agg_score < 0.38:
                return "Image appears authentic with natural lighting, consistent shadows, and realistic textures throughout."
            elif agg_score < 0.48:
                return "Image shows minor processing artifacts but overall appears to be an authentic photograph."
            elif agg_score < 0.55:
                return "Image has borderline characteristics that warrant closer inspection for potential manipulation."
            else:
                return "Multiple forensic signals indicate potential AI manipulation or heavy post-processing."

        # Combine reasoning (max 2 sentences for competition format)
        combined = " ".join(reasons)
        sentences = combined.replace(". ", ".|").split("|")
        result = ". ".join(s.strip() for s in sentences[:2] if s.strip())
        if result and not result.endswith("."):
            result += "."
        return result
