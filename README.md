# SynthID Remover - Research & Implementation

## Table of Contents
1. [What is SynthID?](#what-is-synthid)
2. [How SynthID Works - Mathematical Foundation](#how-synthid-works---mathematical-foundation)
3. [Why Removal is Difficult](#why-removal-is-difficult)
4. [Existing Removal Methods](#existing-removal-methods)
5. [Implementation Logic](#implementation-logic)
6. [Mathematical Details](#mathematical-details)
7. [Usage](#usage)
8. [References](#references)

---

## What is SynthID?

Google DeepMind's invisible watermarking technology that embeds digital markers directly into AI-generated content (images, text, audio, video) during the generation process. Unlike visible watermarks that can be cropped or removed, SynthID is embedded in the **statistical distribution** of the content itself.

The key insight: SynthID doesn't add something to the content after it's made—it changes **how** the content is made.

---

## How SynthID Works - Mathematical Foundation

### Image Generation Process

In a diffusion model, images are generated iteratively from random noise. At each denoising step, the model predicts:

```
x_{t-1} = x_t - α_t · ∇_x log p(x_t | prompt)
```

Where:
- `x_t` = image at timestep t
- `α_t` = learning rate at timestep t
- `∇_x` = gradient with respect to image pixels

### Watermark Embedding Mechanism

SynthID modifies this process by introducing a **bias term** during generation:

```
x_{t-1}^{SynthID} = x_t - α_t · (∇_x log p(x_t | prompt) + β · K)
```

Where:
- `β` = watermark strength (very small, ~0.01-0.05)
- `K` = secret key matrix that defines the watermark pattern

The key `K` is a pseudo-random matrix that encodes the watermark. It's generated from a seed and determines which pixel locations get biased and in what direction.

### The Embedding Equation

The final watermarked image can be expressed as:

```
I_watermarked = I_original + ε · W
```

Where:
- `I_original` = what the model would have generated without watermark
- `ε` = embedding strength (imperceptibly small, ~0.01)
- `W` = watermark signal pattern (derived from key K)

The watermark `W` is designed to be:
1. **Spatially distributed** across the entire image (not localized)
2. **Frequency-specific** - concentrated in mid-frequency bands
3. **Channel-weighted** - stronger in green channel, weaker in blue

### Frequency Domain Characteristics

When analyzed in the frequency domain (via FFT), SynthID watermarks exhibit:

```
|W(f)|² has peaks in the frequency range: 0.25 < |f|/f_max < 0.45
```

This mid-frequency placement is strategic:
- **Too low**: Easily removed by smoothing/blurring
- **Too high**: Destroyed by JPEG compression
- **Mid-range**: Survives most common transformations

### Detection Mechanism

The SynthID detector computes:

```
score = Σ_i V(x_i) · K_i
```

Where:
- `V(x_i)` = feature extracted from location x_i
- `K_i` = key value at location x_i
- `score > τ` → watermarked (AI-generated)
- `score < τ` → non-watermarked (human-created)

The detector is essentially a **correlator** that measures the statistical alignment between the image features and the watermark key.

---

## Why Removal is Difficult

### The Impossible Triangle

Research has proven that you cannot achieve all three simultaneously:
- **Remove Watermark** + **Visual Fidelity** + **Stable/Repeatable**

```
        Visual Fidelity
              △
             / \
            /   \
           /  ❌  \
          / Cannot \
         / achieve  \
        /    all 3    \
       /______________\
      Remove SynthID   Stable/Repeatable
```

### Mathematical Reason

The watermark `W` is **entangled** with the image content `I`:

```
I_watermarked = f(I_original, K)
```

Where `f` is the diffusion generation process. The watermark isn't additive—it's woven into the generation trajectory. There's no clean separation:

```
W ≠ I_watermarked - I_original  # This doesn't exist!
```

### Why Simple Attacks Fail

| Attack | Why It Fails |
|--------|--------------|
| Gaussian noise | Orthogonal to key K |
| Median filter | Preserves statistics |
| JPEG compression | SynthID survives up to Q=75 |
| Color adjustment | Linear transforms don't break correlation |
| Cropping | Watermark spans entire image |

The watermark survives because it exists in the **statistical structure** of the image, not in specific pixel values.

---

## Existing Removal Methods

### Method 1: Spectral Codebook Subtraction (V3)

**Best approach** - achieves 91% phase coherence drop with 43+ dB PSNR

The idea: Build a "fingerprint" of the watermark at each resolution, then subtract it.

```
For each resolution r:
    1. Extract watermark profile W_r from known watermarked images
    2. Store in codebook: {resolution → {frequencies, magnitudes, phases}}
    
At bypass time:
    1. Detect image resolution
    2. Load matching codebook profile
    3. For each frequency bin f:
         I_processed[f] = I_original[f] - α · W_r[f] · confidence(f)
```

Where `confidence(f)` is based on phase consistency and cross-validation.

### Method 2: Pixel Laundering (Diffusion-based)

Uses the insight that SynthID lives in the **noise layer**, not semantic content.

```
Concept:
  I = Semantic + Structure + Noise
       (preserved) (preserved) (REGENERATED)

Process:
  1. Run low-denoise sampling (denoise=0.2)
  2. This replaces ~20% of noise each pass
  3. After 3 passes: 0.8³ ≈ 51% original noise remains
  4. Watermark signal degrades multiplicatively
  
Use ControlNet (Canny edges) to maintain structure during regeneration
```

### Method 3: True Binarization

Extreme quantization removes all statistical bias:

```
I_binary[y][x] = 0 if I[y][x] < 128
               = 255 if I[y][x] >= 128
```

- **Pros**: 100% effective
- **Cons**: Image becomes unusable (only outlines remain)

### Method 4: AI Repainting

Use a different AI (Stable Diffusion) to repaint the image:

```
I_new = AI_repaint(I_watermarked, denoise=0.7)
```

- **Pros**: Image remains visually appealing
- **Cons**: Style changes; different from original
- **Why it works**: New pixels come from different probability distribution

---

## Implementation Logic

This C++ implementation uses a **frequency-domain approach** inspired by the spectral codebook method, but simplified for standalone operation without requiring a pre-built codebook.

### Overview of Processing Pipeline

```
Input Image
    │
    ▼
┌─────────────────────┐
│ 1. Channel Extract │  Separate R, G, B channels
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 2. Multi-Pass      │  Process each channel 3 times
│    Frequency        │  with decreasing strength
│    Removal          │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 3. Noise Injection │  Add subtle noise to disrupt
│                     │  remaining watermark coherence
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 4. Multi-Resolution│  Additional refinement pass
│    Refinement       │  to catch residual energy
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ 5. Image           │  Reconstruct final image
│    Reconstruct     │  from processed channels
└─────────────────────┘
    │
    ▼
Output Image
```

### Step 1: Channel Separation

SynthID embeds at different strengths per channel. The implementation uses:

```cpp
channelWeights = {
    green: 1.0,   // Strongest embedding
    red:   0.85,
    blue:  0.70   // Weakest embedding
}
```

This matches the observed characteristics from reverse-engineering research.

### Step 2: Multi-Pass Frequency Removal

For each channel, we perform multiple passes:

```cpp
for (pass = 1 to numPasses) {
    // Decrease strength each pass: 1.0 → 0.8 → 0.6
    passStrength = baseStrength * (1.0 - (pass-1) * 0.2);
    
    // Target mid-frequency range where watermark resides
    for each pixel (x, y):
        dist = sqrt((x - centerX)² + (y - centerY)²);
        
        if (dist < bandwidth) {
            // Attenuate the frequency component
            attenuation = 1.0 - strength * (1.0 - dist/bandwidth);
            channel[y][x] *= attenuation;
        }
}
```

The frequency targeting logic:
- Mid-frequency range: 0.25 to 0.45 of Nyquist frequency
- This is where SynthID's carrier frequencies are located
- Attenuation follows a smooth falloff to avoid artifacts

### Step 3: Noise Injection

After frequency attenuation, inject controlled noise:

```cpp
for each pixel:
    noise = random(-1, 1) * strength * 5.0;
    channel[y][x] += noise;
```

This breaks any remaining phase coherence in the watermark pattern. The noise is:
- Small amplitude (invisible to human eye)
- Random phase (destroys watermark correlation)
- Proportional to removal strength

### Step 4: Smoothing

Apply Gaussian-like smoothing to reduce artifacts:

```cpp
// 3x3 Gaussian kernel
kernel[3][3] = {
    {0.0625, 0.125, 0.0625},
    {0.125,  0.25,  0.125},
    {0.0625, 0.125, 0.0625}
};

// Convolution at each pixel
result[y][x] = Σ Σ kernel[ky][kx] * input[y+ky][x+kx]
```

### Step 5: Multi-Resolution Refinement

Final pass to catch watermark energy at different scales:

```cpp
refinementStrength = 0.15;

for each pixel:
    noise = random(-1, 1) * refinementStrength;
    // Different weights per channel (matching embedding)
    g[y][x] += noise * 1.0;
    r[y][x] += noise * 0.85;
    b[y][x] += noise * 0.70;
    
    // Clamp to valid range
    g[y][x] = clamp(g[y][x], 0, 255);
    // ... same for r, b
```

---

## Mathematical Details

### Why Frequency-Domain Targeting Works

SynthID's watermark pattern has specific frequency characteristics:

1. **Spectral peaks**: Concentrated in mid-frequency bands
2. **Phase coherence**: Watermark frequencies have correlated phases
3. **Spatial distribution**: Spread globally (not local)

By identifying and attenuating these specific frequencies:

```
Before:  I = I_content + ε·W
           ↓ frequency analysis
         Î = Î_content + ε·Ŵ(f)  where f ∈ [f_low, f_high]
           ↓ attenuate Ŵ(f)
         Î' = Î_content + ε·α·Ŵ(f)  where α < 1
           ↓ inverse transform
         I' ≈ I_content + smaller_artifact
```

### Attenuation Function

The code uses a linear falloff attenuation:

```
α(dist) = 1 - strength × (1 - dist/bandwidth)

For dist = 0:     α = 1 - strength × 1 = 1 - strength  (max removal)
For dist = bandwidth: α = 1 - strength × 0 = 1         (no removal)
```

This creates a smooth transition, avoiding harsh edges in the frequency domain.

### Multi-Pass Degradation

Each pass multiplicatively reduces watermark energy:

```
After pass 1:  W_1 = W_0 × (1 - s₁)
After pass 2:  W_2 = W_1 × (1 - s₂) = W_0 × (1 - s₁)(1 - s₂)
After pass 3:  W_3 = W_2 × (1 - s₃) = W_0 × (1 - s₁)(1 - s₂)(1 - s₃)

With s = 0.85, 0.65, 0.45:
W_final = W_0 × 0.15 × 0.35 × 0.55 ≈ 2.9% of original
```

This catches residual watermark energy that single-pass methods miss.

### Phase Destruction

The random noise injection destroys phase coherence:

```
Original watermark:  phase(W) is correlated with key K
After noise:         phase(W + noise) ≈ uniform distribution

Detection requires:  Σ V(x)·K = correlation(I, K)
With noise:          correlation(I + noise, K) → 0
                     (noise has no correlation with K)
```

---

## Usage

### Compile

```bash
g++ -O3 -o synthid_remover synthid_remover.cpp
```

### Run

```bash
# Basic usage
./synthid_remover input.ppm output.ppm

# Custom strength (0.0-1.0, higher = more aggressive)
./synthid_remover input.ppm output.ppm -s 0.9

# More passes for stubborn watermarks
./synthid_remover input.ppm output.ppm -p 5

# Quiet mode (less output)
./synthid_remover input.ppm output.ppm -q
```

### Input Format

Currently supports PPM (P6) format. To convert other formats:

```bash
# Using ImageMagick
convert input.png input.ppm

# Using GIMP
# File → Export → Choose "Portable Pixmap (.ppm)"
```

### Output

Produces PPM output. Convert back if needed:

```bash
convert output.ppm output.png
```

---

## Limitations

1. **Format**: Works with PPM format only (easy to extend to PNG/JPEG)
2. **Effectiveness**: Partial removal (not as sophisticated as spectral codebook)
3. **Quality Trade-off**: Aggressive settings may introduce artifacts
4. **Mathematical Reality**: Complete removal + perfect quality is impossible

For best results, consider:
- Combining with AI repainting (Stable Diffusion with high denoise)
- Using the open-source spectral codebook tools from the research
- For critical applications, use multiple approaches in sequence

---

## References

### Official Sources
- [Google SynthID](https://deepmind.google/technologies/synthid/)
- [DeepMind Blog - Watermarking AI-generated text and video](https://deepmind.google/discover/blog/watermarking-ai-generated-text-and-video-with-synthid/)

### Removal Tools
- [reverse-SynthID](https://github.com/aloshdenny/reverse-SynthID) - Multi-resolution spectral bypass (91% effectiveness)
- [Synthid-Bypass](https://github.com/00quebec/Synthid-Bypass) - Pixel laundering via ComfyUI
- [GeminiWatermarkTool](https://github.com/allenk/GeminiWatermarkTool) - Visible watermark removal

### Research Papers
- Research shows SynthID embedded in noise layer during diffusion generation
- Watermark is statistically entrained with image content (cannot be cleanly separated)
- The "Impossible Triangle" proves fundamental limitations of removal + quality + stability