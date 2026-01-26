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
        center = 0.42  # Threshold between real and fake
        steepness = 6.0  # How sharply to separate
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
        if final_score < 0.35:
            return "authentic"

        # Use VLM type if confident
        vlm_type = vlm.get("manipulation_type", "unknown")
        if vlm_type and vlm_type not in ["unknown", "authentic", "manipulation_detected"]:
            return vlm_type

        # Infer from forensic signals
        ela_score = forensic.get("ela_score", 0)
        texture_score = forensic.get("texture_score", 0)
        noise_score = forensic.get("noise_score", 0)
        edge_score = forensic.get("edge_score", 0)

        # High noise uniformity suggests AI generation or heavy processing
        if noise_score > 0.7:
            return "full_synthesis"

        # High ELA variance suggests inpainting/splicing
        if ela_score > 0.6:
            return "inpainting"

        # High texture smoothness suggests virtual staging or AI
        if texture_score > 0.5:
            return "virtual_staging"

        # Edge issues might indicate manipulation
        if edge_score > 0.5:
            return "manipulation_detected"

        # Default if score is high but no specific type
        if final_score > 0.5:
            return "manipulation_detected"

        return "authentic"

    def _generate_reasoning(self, forensic: Dict, vlm: Dict) -> str:
        """Generate human-readable reasoning."""

        reasons = []

        # VLM reasoning
        vlm_reasoning = vlm.get("reasoning", "")
        if vlm_reasoning and "mock" not in vlm_reasoning.lower():
            reasons.append(vlm_reasoning)

        # Forensic insights (thresholds tuned for meaningful signals)
        forensic_insights = []

        if forensic.get("noise_score", 0) > 0.6:
            forensic_insights.append("uniform noise patterns suggest artificial processing")

        if forensic.get("sharpness_score", 0) > 0.6:
            forensic_insights.append("oversharpening or blur artifacts detected")

        if forensic.get("texture_score", 0) > 0.45:
            forensic_insights.append("unnaturally smooth textures observed")

        if forensic.get("edge_score", 0) > 0.45:
            forensic_insights.append("edge coherence anomalies found")

        if forensic_insights:
            reasons.append("Forensic analysis: " + ", ".join(forensic_insights) + ".")

        if not reasons:
            agg_score = forensic.get("aggregate_score", 0.5)
            if agg_score < 0.4:
                return "Image appears authentic with consistent lighting, shadows, and textures."
            elif agg_score < 0.55:
                return "Image shows minor anomalies but appears largely authentic."
            else:
                return "Some anomalies detected that may indicate manipulation."

        # Combine and limit length
        combined = " ".join(reasons)
        sentences = combined.replace(". ", ".|").split("|")
        result = ". ".join(s.strip() for s in sentences[:2] if s.strip())
        if result and not result.endswith("."):
            result += "."
        return result
