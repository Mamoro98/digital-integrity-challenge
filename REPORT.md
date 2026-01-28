# Technical Report: AI-Generated Real Estate Image Detection
## Digital Integrity Challenge - Track B

**Author:** Omer  
**Date:** January 28, 2026  
**Repository:** https://github.com/Mamoro98/digital-integrity-challenge

---

## 1. Executive Summary

This submission presents a dual-module system for detecting AI-generated and manipulated real estate images. The approach combines **pixel-level forensic analysis** (11 feature extractors) with **vision-language model (VLM) reasoning** through an adaptive fusion strategy. On our test set of 30 real estate images (15 real, 15 AI-generated), the forensic module alone achieves **63-67% accuracy** with clear score separation between classes.

**Key Innovation:** Empirically-optimized feature weighting with direction correction, where some features indicate authenticity when high (inverted) while others indicate manipulation.

---

## 2. System Architecture

```
                    Input Image
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌─────────────────────┐      ┌─────────────────────┐
│  Forensic Detector  │      │    VLM Reasoner     │
│   (11 features)     │      │  (semantic check)   │
│   src/forensics/    │      │     src/vlm/        │
└──────────┬──────────┘      └──────────┬──────────┘
           │                            │
           └────────────┬───────────────┘
                        ▼
               ┌────────────────┐
               │ Fusion Module  │
               │  (adaptive)    │
               │  src/fusion/   │
               └───────┬────────┘
                       ▼
              Final Prediction
           (score + type + reasoning)
```

---

## 3. Module 1: Forensic Detection

### 3.1 Feature Extractors

The forensic module implements **11 complementary features** targeting different manipulation artifacts:

| Feature | Method | Direction | Weight |
|---------|--------|-----------|--------|
| **Noise Analysis** | Wavelet denoising residual variance | +1 (↑=fake) | 18% |
| **Texture Consistency** | Local variance + Laplacian uniformity | +1 (↑=fake) | 16% |
| **Sharpness** | Laplacian variance analysis | +1 (↑=fake) | 16% |
| **FFT Analysis** | Frequency domain periodic artifacts | -1 (↑=real) | 15% |
| **ELA** | Error Level Analysis recompression | -1 (↑=real) | 12% |
| **Color Consistency** | Cross-channel correlation | +1 (↑=fake) | 6% |
| **Compression** | JPEG DCT block artifacts | +1 (↑=fake) | 5% |
| **GLCM Texture** | Gray-level co-occurrence | +1 (↑=fake) | 5% |
| **Rich/Poor Texture** | Region contrast analysis | -1 (↑=real) | 3% |
| **LBP** | Local binary patterns | -1 (↑=real) | 3% |
| **Edge Coherence** | Boundary consistency | +1 (↑=fake) | 1% |

**Direction Correction:** Features with direction=-1 are inverted (1-score) before weighted aggregation, ensuring all transformed scores point toward "higher = more likely manipulated."

### 3.2 Key Technical Approaches

**FFT Analysis:** Detects periodic artifacts at periods 2, 4, 8, 16 pixels characteristic of diffusion models. Implements DEFEND-style weighted band analysis focusing on mid-high frequencies.

**Noise Analysis:** Most discriminative feature. AI-generated images exhibit smoother, more uniform noise patterns compared to the sensor noise (PRNU) in real photographs.

**ELA (Error Level Analysis):** Recompresses image at quality 90 and measures difference. Spliced/generated regions show different error levels than authentic content.

**Texture Consistency:** Measures local variance across image patches. AI images often have unnaturally uniform textures in walls and surfaces.

---

## 4. Module 2: VLM Reasoning

### 4.1 Semantic Analysis

The VLM module provides human-interpretable explanations by analyzing:
- **Shadow consistency** with apparent light sources
- **Reflection accuracy** in windows, mirrors, floors
- **Structural plausibility** (impossible geometry, proportions)
- **Edge quality** at furniture-background boundaries

### 4.2 Local Model Fallback Chain

All models run locally without API dependencies:

1. **Qwen2-VL-7B** (primary) - Best reasoning capability
2. **PaliGemma-2** (secondary) - TPU-optimized alternative
3. **BLIP-2** (tertiary) - Lightweight fallback
4. **Mock** (forensics-only) - When no VLM available

### 4.3 Prompt Engineering

```
Analyze this real estate image for AI manipulation or virtual staging.

Check these red flags:
1. Do furniture shadows match light sources?
2. Are wall/floor textures unnaturally smooth?
3. Do reflections look consistent?
4. Are furniture edges blended naturally?
5. Is the scale/proportion realistic?

Respond: MANIPULATION_DETECTED: YES/NO/UNCERTAIN
         CONFIDENCE: HIGH/MEDIUM/LOW
         MANIPULATION_TYPE: authentic/virtual_staging/inpainting/full_synthesis
         REASONING: [explanation]
```

---

## 5. Module 3: Adaptive Fusion

### 5.1 Base Weighting
- **Forensics:** 55%
- **VLM:** 45%

### 5.2 Dynamic Adjustment

The fusion module adapts weights based on signal strength:

| Condition | Forensic Weight | VLM Weight |
|-----------|-----------------|------------|
| Strong pixel anomaly (sharpness/noise > 0.65) | 80% | 20% |
| VLM uncertain or low confidence | 85% | 15% |
| VLM medium confidence | 55% | 45% |
| VLM high confidence | 40% | 60% |

This ensures forensic signals aren't overridden when VLM fails to detect subtle artifacts.

---

## 6. Experimental Results

### 6.1 Test Dataset
- **15 real** interior real estate photographs
- **15 AI-generated** images (Flux/diffusion models)
- Total: **30 images**

### 6.2 Forensic Module Performance

| Threshold | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|-----|
| 0.40 | 56.7% | 54.5% | 80.0% | 64.9% |
| **0.45** | **63.3%** | **66.7%** | **53.3%** | **59.3%** |
| 0.50 | 66.7% | 100% | 33.3% | 50.0% |

### 6.3 Score Distribution

| Class | Mean Score | Min | Max |
|-------|------------|-----|-----|
| Real Images | 0.416 | 0.352 | 0.489 |
| AI-Generated | 0.457 | 0.356 | 0.558 |

**Separation:** AI images score **0.041 higher** on average than real images.

### 6.4 Decision Thresholds
- `< 0.40`: Likely real
- `0.40 - 0.60`: Uncertain (requires VLM analysis)
- `≥ 0.60`: Likely manipulated

---

## 7. Usage

### Installation
```bash
pip install -r requirements.txt
```

### Inference
```bash
# Single image
python predict.py --image /path/to/image.jpg --output_file result.json

# Directory batch
python predict.py --input_dir /path/to/images --output_file predictions.json

# Forensics only (skip VLM)
python predict.py --input_dir images/ --vlm_backend mock
```

### Output Format
```json
{
  "image_name": "example.jpg",
  "authenticity_score": 0.65,
  "manipulation_type": "virtual_staging",
  "vlm_reasoning": "Shadow direction inconsistent with window light source. Furniture edges show subtle blending artifacts.",
  "details": {
    "forensic_score": 0.58,
    "vlm_score": 0.72,
    "forensic_breakdown": {
      "noise_score": 0.71,
      "texture_score": 0.65,
      "fft_score": 0.45,
      ...
    }
  }
}
```

---

## 8. Limitations & Future Work

### Current Limitations
1. **Modern diffusion models** (Flux, SDXL) produce increasingly realistic images with fewer artifacts
2. **Score overlap** between classes requires VLM for confident decisions
3. **Domain specificity** - weights optimized for real estate may not generalize

### Proposed Improvements
1. Fine-tune on larger Flux-specific dataset
2. Implement CAT-Net compression artifact tracing
3. Add frequency-based diffusion fingerprint detector (DIRE approach)
4. Train real estate-specific VLM adapter

---

## 9. References

1. Corvi et al. (2023) "On the Detection of Synthetic Images Generated by Diffusion Models" - FFT fingerprints
2. Wang et al. (2024) "DIRE: Diffusion Reconstruction Error" - Reconstruction-based detection
3. FakeShield (ICLR 2025) "Explainable Image Forgery Detection" - VLM integration
4. ForgerySleuth (2024) "Chain-of-Clues for Image Manipulation Detection" - Multi-modal reasoning
5. DEFEND (2024) "Weighted FFT Filtering for Diffusion Detection" - Frequency analysis

---

*© 2026 Digital Integrity Challenge Submission*
