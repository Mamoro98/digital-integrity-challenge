# Digital Integrity Challenge - Sprint Plan

**Deadline:** Jan 28, 2026 2:00 PM Riyadh (11:00 AM UTC)
**Time remaining:** ~44 hours

## Deliverables
- [ ] predict.py - inference script
- [ ] requirements.txt
- [ ] predictions.json format output
- [ ] 3-page technical report

## Architecture

```
┌─────────────────────────────────────────────┐
│              Input Image                     │
└─────────────────┬───────────────────────────┘
                  │
    ┌─────────────┴─────────────┐
    ▼                           ▼
┌─────────────────┐   ┌─────────────────────┐
│ Module 1:       │   │ Module 2:           │
│ Forensic        │   │ VLM Reasoner        │
│ Detector        │   │                     │
│                 │   │ - Physics check     │
│ - FFT analysis  │   │ - Shadow consistency│
│ - Noise patterns│   │ - Geometry check    │
│ - ELA           │   │ - Generate explain  │
│ - CLIP anomaly  │   │                     │
└────────┬────────┘   └──────────┬──────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │ Fusion & Scoring      │
         │ authenticity_score    │
         │ manipulation_type     │
         │ vlm_reasoning         │
         └───────────────────────┘
```

## Sprint Schedule

### Day 1 (Today - Jan 26)
- [ ] Set up repo & environment
- [ ] Implement Module 1: Forensic detector (FFT, ELA, noise)
- [ ] Get basic VLM working (Qwen-VL or LLaVA)
- [ ] Test on sample images

### Day 2 (Jan 27)
- [ ] Refine fusion strategy
- [ ] Collect/generate test images
- [ ] Tune thresholds
- [ ] Write predict.py clean interface

### Day 3 (Jan 28 morning)
- [ ] Final testing
- [ ] Write technical report
- [ ] Submit!

## Tech Stack Options

**Forensic Module:**
- OpenCV for image processing
- NumPy/SciPy for FFT
- Pre-trained forensics model (if available)

**VLM Options:**
- Qwen2-VL (good, runs on decent GPU)
- LLaVA (proven)
- InternVL
- Or API: GPT-4V / Claude Vision (costs $)

## Quick Wins
1. Use existing CLIP for anomaly detection
2. Simple FFT high-frequency analysis
3. Error Level Analysis is easy to implement
4. VLM prompt engineering for reasoning
