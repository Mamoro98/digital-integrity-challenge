# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Digital Integrity Challenge - Track B (Real Estate & Commercial Integrity): Detecting AI-generated or manipulated property images using a dual-module approach combining forensic signal detection with vision-language model (VLM) reasoning.

**Track B Goals:** Detect deceptive virtual staging and AI-generated property photos. Identify AI used to remove power lines, hide cracks, replace furniture, or create fake interiors/exteriors.

## Competition Details

**Deadline:** Wednesday, 28/01/2026 at 2:00 PM Riyadh time (11:00 AM UTC)

**Submission:** [Official Form](https://forms.office.com/r/864ac0pUAC) - Public HuggingFace repo with `predict.py`, `requirements.txt`, and 3-page technical report.

**Evaluation Criteria:**
| Criterion | Weight |
|-----------|--------|
| Detection Accuracy (50/50 real/manipulated test set) | 40% |
| Explainability (VLM reasoning quality, human-judged) | 30% |
| Generalization (unseen AI generators, lighting, resolutions) | 20% |
| Efficiency (inference speed, model size) | 10% |

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference on a directory
python predict.py --input_dir /path/to/images --output_file predictions.json

# Run inference on a single image
python predict.py --image /path/to/image.jpg --output_file predictions.json

# Specify VLM backend (auto, qwen2vl, paligemma, blip2, mock) - LOCAL MODELS ONLY
python predict.py --input_dir images/ --vlm_backend mock

# Test forensic module only
python test_forensics.py data/test_subset/real/image.jpg
```

## Architecture

```
Input Image
     │
     ├──────────────────────────────┐
     ▼                              ▼
┌─────────────────┐      ┌──────────────────┐
│ ForensicDetector│      │   VLMReasoner    │
│ (pixel-level)   │      │ (semantic-level) │
└────────┬────────┘      └────────┬─────────┘
         │                        │
         └──────────┬─────────────┘
                    ▼
            ┌───────────────┐
            │ FusionModule  │
            │ (final score) │
            └───────────────┘
```

**Three modules in `src/`:**

1. **`src/forensics/detector.py`** - Pixel-level analysis using 7 techniques: FFT, ELA, noise analysis, texture consistency, compression analysis, edge coherence, sharpness. Weighted scoring (noise 40%, sharpness 25%, texture 15%).

2. **`src/vlm/reasoner.py`** - Semantic analysis with local-only fallback chain: Qwen2-VL → PaliGemma → BLIP-2 → Mock. NO API KEYS - all models run locally. Includes 60-second timeout protection.

3. **`src/fusion/combiner.py`** - Adaptive weighting based on confidence levels. Base: 55% forensics / 45% VLM. Adjusts dynamically when strong anomalies detected or VLM confidence varies.

## Output Format (Competition Required)

**Required fields per competition spec:**
- `authenticity_score`: 0.0 (authentic) to 1.0 (manipulated)
- `manipulation_type`: "inpainting", "full_synthesis", "virtual_staging", "filter", "authentic"
- `vlm_reasoning`: 2-sentence explanation of red flags detected

```json
{
  "image_name": "000001.jpg",
  "authenticity_score": 0.91,
  "manipulation_type": "inpainting",
  "vlm_reasoning": "The window reflection is inconsistent with the room layout. Shadow direction on the sofa does not match the light source.",
  "details": {
    "forensic_score": 0.75,
    "vlm_score": 0.85,
    "forensic_breakdown": { "fft": 0.87, "ela": 0.60, "noise": 0.95, ... }
  }
}
```

**Decision thresholds:** `< 0.4` = likely real, `0.4-0.6` = uncertain, `>= 0.6` = likely manipulated

## VLM Backend Configuration

**LOCAL MODELS ONLY** - No API keys allowed (competition constraint).

Available backends: Qwen2-VL, PaliGemma, BLIP-2 (all run locally on GPU/CPU).
Use `--vlm_backend mock` to skip VLM entirely (forensics only).

## GPU Optimization

- Float16 used for Qwen2-VL on CUDA
- BFloat16 for PaliGemma
- 4-bit quantization available for 72B models (uncomment `bitsandbytes` in requirements.txt)
- Auto device mapping for multi-GPU setups

## Competition Module Requirements

**Module 1 (Forensic) should detect:**
- Texture consistency: unnatural smoothness in walls/surfaces
- Compression discrepancies: digitally spliced objects
- Frequency domain (FFT): mathematical fingerprints from upscalers/generators

**Module 2 (VLM) should check:**
- Physics: shadow alignment with light sources
- Structural integrity: impossible geometry
- Provide natural language explanations for why images are flagged

**Suggested datasets:** Places365, SUN RGB-D, Columbia Splicing dataset

## Research Reference

See `RESEARCH.md` for comprehensive state-of-the-art analysis including:
- Transformer vs CNN detection approaches
- FFT frequency fingerprints (periods 2, 4, 8 for diffusion models)
- Noise pattern analysis (PRNU sensor fingerprinting)
- Texture contrast methods (rich/poor region analysis)
- Shadow/lighting physics-based detection
- Fusion strategies (late vs early fusion)

## Key Technical Insights

**Forensic Detection (from research):**
- Mid-high frequencies are most discriminative (low frequencies similar for real/AI)
- Noise analysis is strongest discriminator - AI images have smoother, structured noise
- JPEG double compression detectable via DCT coefficient analysis
- Rich/poor texture contrast reveals AI fingerprints

**VLM Detection (from research):**
- Structured prompts outperform vague "detect manipulation" prompts
- Multi-pass refinement (coarse → fine) improves localization
- Real estate specific: check furniture shadows, window reflections, room proportions
- Request explicit confidence levels in output

**Fusion (from research):**
- Late fusion (separate features → combine) more interpretable
- Trust-weighted aggregation based on signal strength
- Increase forensic weight when strong pixel anomalies detected
- Increase VLM weight when high semantic confidence

## Real Estate Manipulation Types

| Type | Key Detection Signals |
|------|----------------------|
| Virtual Staging | Furniture shadow mismatch, floor reflection errors, edge blending artifacts |
| Object Removal | Texture discontinuities, missing shadows, unnatural smoothness |
| Sky Replacement | Horizon blending, lighting inconsistency with scene |
| Full Synthesis | FFT fingerprints, noise patterns, missing EXIF, too-perfect textures |
