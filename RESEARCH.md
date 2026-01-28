# Deep Research: AI-Generated Image Detection for Digital Integrity Challenge

## Executive Summary

This document compiles state-of-the-art research on detecting AI-generated and manipulated images, specifically for Track B (Real Estate) of the Digital Integrity Challenge.

---

## 1. State-of-the-Art Detection Methods

### 1.1 Transformer-Based Approaches (Best Performance)

**DINOv2 (Self-Distilled Transformers)**
- Uses self-distillation without labels for feature extraction
- Achieves 95%+ accuracy on AI-generated image detection
- Strong for detecting both GAN and diffusion model outputs

**Vision Transformers (ViT)**
- Outperforms CNNs for AI image detection
- Better at capturing global inconsistencies
- Recommended models: ViT-L/14, Swin Transformer

**CLIP-Based Detection**
- Zero-shot capability without specific training
- AA-CLIP (CVPR 2025): Creates anomaly-aware text anchors
- WinCLIP: Window-based feature extraction for localization

### 1.2 CNN-Based Approaches

**ResNet50 with Patch Selection Module (PSM)**
- Leverages global and local features
- Attention-based fusion mechanisms
- Good baseline performance

**EfficientNetV2 / ConvNeXt**
- Best for production with limited compute
- Strong accuracy under resource constraints

### 1.3 Multimodal LLM Approaches (For Explainability)

**Key Systems:**
- **AIGI-Holmes**: Multi-expert jury + instruction tuning + collaborative decoding
- **ThinkFake**: Chain-of-thought reasoning with GRPO optimization
- **FakeShield**: Large-scale image-text dataset for forgery detection
- **FKA-Owl**: Forgery-specific knowledge augmentation

---

## 2. Forensic Signal Detection (Module 1)

### 2.1 FFT/Frequency Domain Analysis

**Key Findings:**
- Diffusion models show distinct artifacts at periods 2, 4, 8 (some at 16)
- Low-frequency: Real and AI images are similar
- Mid-high frequency: Increasingly discriminative
- AI images are smoother, lack high-fidelity details

**Implementation: DEFEND**
- Weighted filter on Fourier spectrum
- Suppresses less discriminative bands
- Enhances informative high-frequency bands

**FreqCross Architecture:**
- Three branches: RGB spatial, FFT magnitude, radial energy distribution
- Multi-modal frequency-spatial fusion

### 2.2 Error Level Analysis (ELA)

**How It Works:**
- Re-save image at known quality (e.g., 95%)
- Compute difference between original and re-saved
- Manipulated regions show different compression levels

**Effectiveness:**
- Good for: JPEG compression detection, splicing, retouching
- Limitations: High false positive rate, not robust to all manipulation types
- Best combined with CNN (ELA-CNN integration)

### 2.3 Noise Pattern Analysis

**PRNU (Photo-Response Non-Uniformity)**
- Every sensor has unique noise fingerprint
- AI images lack genuine sensor fingerprint
- Compare PRNU against reference camera fingerprint

**Key Insight:**
- Real images: Stochastic sensor noise
- AI images: Smoother, more structured noise patterns

**Noiseprint (CNN-Based)**
- Extracts camera model fingerprint
- Scene content suppressed, model artifacts enhanced
- Absence of camera fingerprint = not camera output

### 2.4 Texture Analysis

**Rich/Poor Texture Contrast Method:**
- Divide image into rich texture (objects, borders) and poor texture (background) patches
- Measure inter-pixel correlation contrast between regions
- AI images show different contrast patterns

**Key Features:**
- GLCM (Grey Level Co-occurrence Matrix): Statistical texture patterns
- LBP (Local Binary Pattern): Local micro-textures, structural discontinuities

**AI Artifacts:**
- GANs: Local texture inconsistencies, checkerboard artifacts
- Diffusion: Globally coherent but subtly unnatural structures
- Over-smooth skin/walls, perfect lighting

### 2.5 JPEG Compression Analysis

**CAT-Net (Compression Artifact Tracing Network)**
- Detects double JPEG compression
- Transfer learning from DJPEG detection to forgery localization
- Excellent for splicing detection

**Double Compression Detection:**
- Manipulated regions have different compression history
- Analyze DCT coefficients and quantization tables
- Effective even on small patches (64x64)

---

## 3. VLM Reasoning (Module 2) - Deep Research

### 3.1 State-of-the-Art VLM Detection Systems (2024-2025)

#### FakeShield (ICLR 2025)
The first MLLM-based framework for explainable image forgery detection and localization.

**Key Components:**
- **DTE-FDM (Domain Tag-guided Explainable Forgery Detection Module):** Classifies images into three domains (PhotoShop, DeepFake, AIGC-Editing) and generates detection results with explanations grounded in pixel-level artifacts and semantic errors
- **MFLM (Multi-modal Forgery Localization Module):** Uses LLM-generated descriptions to guide SAM segmentation for precise localization
- **MMTD-Set Dataset:** GPT-4o enhanced forgery detection dataset with image-mask-description triplets

**Performance:**
- PhotoShop: 95% ACC / 95% F1
- DeepFake: 98% ACC / 99% F1
- AIGC-Editing: 93% ACC / 93% F1
- Explanation quality: 0.8758 CSS (Cosine Semantic Similarity)

**Source:** https://arxiv.org/abs/2410.02761

#### ForgerySleuth
Empowers MLLMs with Chain-of-Clues reasoning for comprehensive forgery analysis.

**Key Innovations:**
- **Chain-of-Clues (CoC):** Three-level analysis - high-level (semantic contradictions), mid-level (lighting/boundary), low-level (noise/texture)
- **Trace Encoder:** Dedicated CNN capturing manipulation features independent of vision backbone
- **ForgeryAnalysis Dataset:** 2,370 expert-reviewed + 50k auto-generated samples

**Performance:**
- Localization F1: 0.710 on IMD20 (surpassing prior methods by +0.121)
- Analysis quality: 9.45/10 vs GPT-4o's 6.96/10

**Source:** https://arxiv.org/html/2411.19466

#### TruthLens (Visual Grounding for DeepFake Reasoning)
Unites global semantic context with region-specific forensic cues.

**Architecture:**
- **Global Context:** PaliGemma2's SigLIP encoder for holistic scene features
- **Localized Context:** DINOv2 for patch-level (14×14) forensic features
- **Cross-Modal Fusion:** Interleaved-Modal approach (I-MoF) superior to concatenation

**Key Capability:** Region-level visually grounded explanations for both face-manipulated and fully synthetic content.

**Source:** https://arxiv.org/html/2503.15867

### 3.2 GPT-4V/GPT-4o Zero-Shot Detection Performance

#### Benchmark Results (2024-2025)

| Model | Real Images | Deepfakes | Notes |
|-------|-------------|-----------|-------|
| GPT-4o | 83% | 72% | Best overall LLM |
| Claude | 77% | 60% | Good explainability |
| Gemini | 90% | 27% | Highly inconsistent |
| Grok | 57% | 9% | Unreliable |

**Key Findings from Multi-Modal LLM Benchmarking:**
- GPT-4o achieves AUROC 0.77-0.80 on StyleGAN2/diffusion images (zero-shot)
- Reasoning models (O1, Gemini-2-Thinking) perform WORSE than base versions
- Post-processed fakes actually improve detection (introduces semantic anomalies)
- Best LLMs still underperform specialized detectors on in-distribution data
- BUT LLMs generalize better to out-of-distribution/unseen manipulations

**Critical Limitation:** "VLMs are prone to misclassification, particularly when faced with hyper-realistic or stylistically ambiguous content"

**Sources:**
- https://arxiv.org/html/2503.20084
- https://arxiv.org/html/2506.10474v1

### 3.3 Why VLMs Struggle (and When They Excel)

#### Fundamental Weaknesses

1. **Surface-Level Analysis:** Models rely on image quality/aesthetics rather than forensic analysis
2. **Hallucination Patterns:** ~50% of predictions show high variance
3. **Context Confusion:** Vintage styling, professional photography trigger false positives
4. **Semantic vs Pixel:** VLMs miss low-level pixel artifacts that forensics catches

#### When VLMs Excel

1. **Semantic Inconsistencies:** Impossible object arrangements, physics violations
2. **High-Level Context:** Scene plausibility, object relationships
3. **Explainability:** Natural language reasoning humans can verify
4. **Out-of-Distribution:** Better generalization to unseen manipulation types

### 3.4 Physics-Based Semantic Checks

**Shadow Consistency:**
- Geometric techniques detect physically inconsistent shadows
- Constrain projected location of light source
- Linear programming to test feasibility
- AI images often have unrealistic shadow directions

**Lighting Analysis:**
- Analyze shading consistency across objects
- Detect inconsistent light source directions
- Specular highlights should match shadow directions

**Reflection Consistency:**
- Check reflections match actual scene geometry
- Window reflections particularly revealing in real estate

**Structural Integrity:**
- Impossible architectural geometry
- Inconsistent perspective lines
- Furniture/object proportions

**For Real Estate Specifically:**
- Power lines removal artifacts
- Hidden cracks/damage
- Virtual furniture vs actual
- Room proportions and scale

### 3.5 Prompt Engineering Best Practices

#### Structured Output Format
VLMs perform better with explicit output formatting:
```json
{
  "manipulation_detected": "yes|no|uncertain",
  "confidence": "high|medium|low",
  "manipulation_type": "virtual_staging|inpainting|full_synthesis|authentic",
  "reasoning": "Two sentences explaining findings"
}
```

#### Chain-of-Thought (CoT) for Visual Analysis
Research shows CoT significantly improves VLM accuracy:
1. **See Module:** Detect all objects, generate global description
2. **Think Module:** Attend to key visual concepts, generate language descriptions
3. **Confirm Module:** Generate supporting rationale, verify consistency

**Example CoT Prompt:**
```
First, describe what you see in this real estate image in detail.
Then, analyze each element for:
- Shadow direction consistency
- Lighting uniformity
- Texture naturalness
- Geometric plausibility
Finally, provide your verdict with reasoning.
```

#### Domain-Specific Prompts (from FakeShield)
- **PhotoShop detection:** Focus on edge artifacts, resolution inconsistencies, lighting problems
- **AIGC-Editing:** Identify text generation failures, texture distortions
- **DeepFake:** Detect localized blurring, unnatural facial features

#### Few-Shot Prompting
For specialized tasks like real estate:
- Provide reference image at "known authentic" state
- Model can compare and contrast features
- Dramatically improves accuracy on domain-specific tasks

### 3.6 Recommended VLM Prompt for Real Estate

**Zero-Shot Structured Prompt:**
```
Analyze this real estate image for AI manipulation. Check shadows, reflections, geometry, textures.

Respond EXACTLY in this format (no extra text):
MANIPULATION_DETECTED: YES or NO or UNCERTAIN
CONFIDENCE: HIGH or MEDIUM or LOW
MANIPULATION_TYPE: authentic or virtual_staging or inpainting or full_synthesis
REASONING: Two sentences explaining why.
```

**Chain-of-Clues Prompt (Advanced):**
```
You are an expert image forensics analyst. Analyze this real estate image using the Chain-of-Clues method:

STEP 1 - HIGH-LEVEL (Semantic):
- Does the scene make physical sense?
- Are object arrangements plausible?
- Is there content that contradicts common sense?

STEP 2 - MID-LEVEL (Visual Defects):
- Are shadows consistent with light sources?
- Do reflections match the scene?
- Are there boundary/blending artifacts?
- Is lighting uniform across objects?

STEP 3 - LOW-LEVEL (Pixel Statistics):
- Are textures unnaturally smooth?
- Are there compression inconsistencies?
- Do furniture edges blend naturally?

Based on your analysis, provide:
VERDICT: [authentic/manipulated/uncertain]
CONFIDENCE: [high/medium/low]
TYPE: [virtual_staging/inpainting/full_synthesis/authentic]
REASONING: [2 sentences with specific evidence]
```

### 3.7 Hybrid VLM-Forensic Architecture (Recommended)

Based on research, pure VLM detection is insufficient. The optimal approach:

1. **Forensic Module First:** Extract pixel-level signals (FFT, noise, compression)
2. **VLM for Semantics:** Check physics, geometry, context
3. **Fusion with Confidence Weighting:**
   - Strong forensic anomalies → trust forensics (80/20)
   - High VLM confidence → trust VLM (40/60)
   - Both uncertain → balance (55/45)

**Key Insight from Research:**
"Crucial 'robust features' of deepfakes are in their higher semantics" - VLMs excel here
"Low-level visual forgery artifacts" - Forensics excel here

### 3.8 VLM Fine-Tuning Approaches

#### LoRA for VLM Fine-Tuning
- Modify only small number of parameters in specific layers
- 30x faster training, 60% reduced memory with frameworks like Unsloth
- Dataset format: JSON with image-conversation pairs

#### Key Training Datasets
1. **MMTD-Set:** FakeShield's multi-modal tamper description dataset
2. **ForgeryAnalysis:** 52k samples with expert annotations
3. **DF2023:** 1M images across 4 forgery categories
4. **TrainFors:** Standardized benchmark for splicing/copy-move/removal

#### Fine-Tuning Strategy
1. **Stage 1 (Alignment):** Train adapter on image-caption data, frozen LLM
2. **Stage 2 (Task-Specific):** Joint optimization on forensic datasets

---

## 4. Fusion Strategies (Module 3)

### 4.1 Multi-Modal Fusion Approaches

**Late Fusion:**
- Independent features from each forensic filter
- Fuse at decision level
- More interpretable

**Early Fusion:**
- Mix modal outputs early
- Produce combined features
- Better for complex interactions

### 4.2 Recommended Forensic Filters to Combine

1. **NoisePrint++** - Trainable noise extraction
2. **SRM (Spatial Rich Model)** - Static edge features
3. **Bayar Convolution** - Trainable high-pass filter
4. **RGB Features** - Spatial visual features

### 4.3 Weighting Strategies

**Trust-Weighted Aggregation:**
- Weight each tool based on historical reliability
- Dynamic adjustment per image type

**Confidence-Based Weighting:**
- High VLM confidence → increase VLM weight
- Strong forensic anomalies → increase forensic weight
- Uncertain → balance both

**Recommended Base Weights (from research):**
- Forensics: 55-60%
- VLM: 40-45%
- Adjust based on signal strength

---

## 5. Real Estate Specific Detection

### 5.1 Common Manipulation Types

1. **Virtual Staging** - AI-generated furniture in empty rooms
2. **Object Removal** - Power lines, cracks, damage, clutter
3. **Sky Replacement** - Improving outdoor shots
4. **Full Synthesis** - Entirely AI-generated property images
5. **Enhancement** - Over-beautification of existing images

### 5.2 Detection Focus Areas

**For Virtual Staging:**
- Furniture shadow consistency
- Reflection accuracy on floors/windows
- Texture blending at furniture edges
- Lighting match with room

**For Object Removal (Inpainting):**
- Texture discontinuities at removal boundaries
- Missing shadows where objects should be
- Unnatural smoothness in removed areas
- Compression artifact inconsistencies

**For Full Synthesis:**
- Overall noise pattern analysis
- FFT frequency fingerprints
- Lack of EXIF metadata
- Too-perfect lighting/textures

---

## 6. Best Detection Tools & Models

### 6.1 Commercial Tools (Reference)

| Tool | Strength | Accuracy |
|------|----------|----------|
| Winston AI | Best overall, face-swaps | High |
| Hive Moderation | Model identification | 98% |
| Illuminarty | Region localization | Good |
| AI or Not | NFT/general detection | Good |

### 6.2 Open Source Models

**For Forensics:**
- CAT-Net (JPEG compression)
- Noiseprint++ (noise analysis)
- DIRE (diffusion reconstruction error)

**For VLM:**
- Qwen2.5-VL-7B (good balance)
- LLaVA (proven performance)
- InternVL (strong reasoning)

---

## 7. Key References

### Frequency Analysis
- [FreqCross: Multi-Modal Frequency-Spatial Fusion](https://arxiv.org/html/2507.02995)
- [UGAD: Universal Generative AI Detector](https://arxiv.org/html/2409.07913v1)
- [Enhancing Frequency Forgery Clues](https://arxiv.org/html/2511.00429v1)

### Texture Analysis
- [Rich and Poor Texture Contrast](https://arxiv.org/html/2311.12397v2)
- [Multi-modal Texture Fusion Network](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1663292/full)

### Compression Analysis
- [CAT-Net: Compression Artifact Tracing](https://github.com/mjkwon2021/CAT-Net)
- [Learning JPEG Compression Artifacts](https://link.springer.com/article/10.1007/s11263-022-01617-5)

### VLM Detection (MLLM-Based)
- [FakeShield: Explainable Image Forgery Detection (ICLR 2025)](https://arxiv.org/abs/2410.02761)
- [ForgerySleuth: MLLM for Image Manipulation Detection](https://arxiv.org/html/2411.19466)
- [TruthLens: Visual Grounding for DeepFake Reasoning](https://arxiv.org/html/2503.15867)
- [Can Multi-modal LLMs Work as Deepfake Detectors?](https://arxiv.org/html/2503.20084)
- [LLMs Are Not Yet Ready for Deepfake Image Detection](https://arxiv.org/html/2506.10474v1)
- [Can ChatGPT Detect DeepFakes? (CVPR 2024 Workshop)](https://openaccess.thecvf.com/content/CVPR2024W/WMF/papers/Jia_Can_ChatGPT_Detect_DeepFakes_A_Study_of_Using_Multimodal_Large_CVPRW_2024_paper.pdf)
- [EDVD-LLaMA: Explainable Deepfake Video Detection](https://arxiv.org/html/2510.16442)

### VLM Prompt Engineering
- [NVIDIA VLM Prompt Engineering Guide](https://developer.nvidia.com/blog/vision-language-model-prompt-engineering-guide-for-image-and-video-understanding/)
- [Systematic Survey of Prompt Engineering on VLMs](https://arxiv.org/pdf/2307.12980)
- [Visual Chain-of-Thought Prompting (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/27888/27801)
- [SynArtifact: VLM for Artifact Classification](https://arxiv.org/html/2402.18068)

### VLM Fine-Tuning
- [Fine-Tuning VLMs using LoRA](https://gautam75.medium.com/fine-tuning-vision-language-models-using-lora-b640c9af8b3c)
- [VLM Fine-tuning for Object Detection (HuggingFace)](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_object_detection_grounding)
- [Unsloth Vision Fine-tuning Documentation](https://docs.unsloth.ai/basics/vision-fine-tuning)

### Fusion Methods
- [MMFusion: Multi-Modal Fusion for IMLD](https://arxiv.org/html/2312.01790v2)
- [From Evidence to Verdict: Agent-Based Framework](https://arxiv.org/html/2511.00181v1)

### Shadow/Physics Analysis
- [Exposing Photo Manipulation with Inconsistent Shadows](https://farid.berkeley.edu/downloads/publications/tog13/tog13.pdf)
- [Photo Forensics from Lighting](https://contentauthenticity.org/blog/photo-forensics-from-lighting-shadows-and-reflections)

### Deepfake Benchmarks
- [Deepfake-Eval-2024: In-the-Wild Benchmark](https://arxiv.org/abs/2503.02857)
- [DeepfakeBench: Comprehensive Benchmark](https://github.com/SCLBD/DeepfakeBench)

### Image Forgery Datasets
- [DF2023: Digital Forensics 2023 Dataset](https://arxiv.org/html/2503.22417v1)
- [TrainFors: Large Benchmark Training Dataset](https://arxiv.org/abs/2308.05264)
- [Image Forgery Datasets List (GitHub)](https://github.com/greatzh/Image-Forgery-Datasets-List)

---

## 8. Implementation Recommendations

### Priority 1: Improve Forensic Module
1. Add DEFEND-style weighted FFT filtering
2. Implement rich/poor texture contrast
3. Add CAT-Net style compression analysis
4. Include Noiseprint-based sensor fingerprint check

### Priority 2: Improve VLM Module
1. Use structured prompts with specific real estate criteria
2. Implement multi-pass refinement (coarse → fine)
3. Add shadow/reflection consistency checks to prompts
4. Request confidence calibration

### Priority 3: Improve Fusion
1. Implement late fusion with separate feature extraction
2. Add confidence-based dynamic weighting
3. Include trust scores per detection method
4. Calibrate thresholds on validation set

---

## 9. Gemini Deep Research API

For future research tasks, use the Gemini Deep Research API:

```python
from google import genai

client = genai.Client(api_key="YOUR_API_KEY")

interaction = client.interactions.create(
    input="Research [topic]",
    agent='deep-research-pro-preview-12-2025',
    background=True
)

# Poll for results
while True:
    interaction = client.interactions.get(interaction.id)
    if interaction.status == "completed":
        print(interaction.outputs[-1].text)
        break
    time.sleep(10)
```

**Cost:** ~$2-5 per research task
**Time:** 5-20 minutes typical
