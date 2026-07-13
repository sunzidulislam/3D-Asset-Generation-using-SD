# Few-Shot Text-to-3D Generation via Score Distillation Sampling (SDS)

> **Status:** Active Research Project (Work in Progress)

An educational, from-scratch implementation of **Score Distillation Sampling (SDS)** for text-to-3D generation using a **frozen pretrained 2D diffusion model** as supervision.

This project aims to reproduce the core concepts introduced in **DreamFusion** while remaining lightweight enough to run on a **single NVIDIA T4 GPU (16GB)**. Beyond reproduction, the repository serves as an ongoing research platform for experimenting with improved optimization strategies, view consistency, and efficient 3D representations.

---

## Overview

Traditional 3D generative models rely on large-scale 3D datasets, which are expensive to collect and limited in diversity. In contrast, this project leverages the rich visual knowledge embedded within pretrained text-to-image diffusion models to optimize a neural 3D representation directly from text prompts.

Only the NeRF is optimized during training. Stable Diffusion remains completely frozen and acts as a guidance network through Score Distillation Sampling.

---

# Pipeline

<p align="center">
    <img src="sds_pipeline_diagram.svg" width="900">
</p>

<p align="center">
<b>Figure.</b> End-to-end Score Distillation Sampling (SDS) optimization pipeline. A text prompt is encoded once, while a differentiable NeRF is optimized using supervision from a frozen Stable Diffusion model.
</p>

---

# Features

## Need to Implement

- ✅ Tiny NeRF implemented from scratch
- ✅ Differentiable volume rendering
- ✅ Score Distillation Sampling (SDS)
- ✅ Stable Diffusion 1.5 guidance
- ✅ Classifier-Free Guidance (CFG)
- ✅ Random camera sampling
- ✅ Mixed-precision inference
- ✅ Gradient clipping
- ✅ Gradient normalization
- ✅ Kaggle T4 compatible implementation
- 🚧 View-dependent prompting
- 🚧 Training stabilization
- 🚧 Improved camera sampling
- 🚧 Better timestep scheduling

## Planned

- 📌 Mesh extraction
- 📌 Normal smoothness loss
- 📌 Orientation loss
- 📌 Multi-view diffusion guidance
- 📌 3D Gaussian Splatting backend
- 📌 Comprehensive ablation studies

---

# Method Overview

The optimization follows the standard Score Distillation Sampling framework.

```
                Text Prompt
                     │
                     ▼
        Frozen CLIP Text Encoder
                     │
                     ▼
           Text Embedding (CFG)
                     │
                     ▼
          Random Camera Sampling
                     │
                     ▼
             Tiny NeRF (Trainable)
                     │
                     ▼
      Differentiable Volume Rendering
                     │
                     ▼
             Rendered RGB Image
                     │
                     ▼
          Frozen VAE Encoder
                     │
                     ▼
          Latent Representation
                     │
                     ▼
        Add Random Noise (timestep t)
                     │
                     ▼
       Frozen UNet Noise Predictor
                     │
                     ▼
      Score Distillation Sampling
                     │
                     ▼
         SDS Gradient Computation
                     │
                     ▼
        Update Tiny NeRF Parameters
```

At every optimization step:

1. A random camera pose is sampled.
2. The current NeRF is rendered from that viewpoint.
3. The rendered image is encoded into the Stable Diffusion latent space.
4. Noise is added at a randomly selected timestep.
5. The frozen UNet predicts the noise residual.
6. SDS converts the prediction error into a training gradient.
7. Only the NeRF parameters are updated.

No part of Stable Diffusion is fine-tuned during training.

---

# Implementation Details

## Trainable Component

| Component | Status |
|-----------|--------|
| Tiny NeRF (MLP + Positional Encoding) | ✅ Trainable |

---

## Frozen Components

| Component | Source |
|-----------|--------|
| CLIP Text Encoder | Stable Diffusion v1.5 |
| VAE Encoder | Stable Diffusion v1.5 |
| UNet | Stable Diffusion v1.5 |
| DDPM Scheduler | Stable Diffusion v1.5 |

---

## Rendering

- Differentiable volume rendering
- Random camera pose sampling
- 64 samples per ray
- White background compositing
- Low-resolution NeRF rendering (64×64)
- High-resolution diffusion guidance (512×512)

---

## Optimization

- Score Distillation Sampling (SDS)
- Classifier-Free Guidance (CFG)
- Adam optimizer
- Gradient clipping
- Gradient normalization
- Mixed precision inference
- Random timestep sampling

---

# Training Configuration

| Setting | Value |
|----------|-------|
| Base Model | Stable Diffusion v1.5 |
| GPU | NVIDIA T4 (16GB) |
| Framework | PyTorch |
| Render Resolution | 64×64 |
| Guidance Resolution | 512×512 |
| Samples per Ray | 64 |
| Optimizer | Adam |
| Learning Rate | 2e-3 |
| Guidance Scale | 25 |
| Precision | FP16 (Diffusion), FP32 (NeRF) |
| Batch Size | 1 |
| Platform | Kaggle |

---

# Current Progress

The repository is under active development.

### Completed

- [x] Tiny NeRF implementation
- [x] Differentiable renderer
- [x] Stable Diffusion integration
- [x] Score Distillation Sampling
- [x] Classifier-Free Guidance
- [x] Random camera sampling
- [x] Training pipeline
- [x] Gradient stabilization

### Currently Working On

- [ ] View-dependent prompting
- [ ] Improved optimization stability
- [ ] Better view consistency
- [ ] Janus artifact reduction
- [ ] Camera pose refinement

---

# Roadmap

## Phase 1 — Baseline SDS

- [x] Tiny NeRF
- [x] Volume rendering
- [x] Stable Diffusion guidance
- [x] SDS optimization
- [x] CFG implementation

---

## Phase 2 — Stability Improvements

- [x] Gradient clipping
- [x] Gradient normalization
- [ ] Adaptive timestep sampling
- [ ] Better camera sampling
- [ ] Background regularization

---

## Phase 3 — Geometry Improvements

- [ ] View-dependent prompting
- [ ] Orientation loss
- [ ] Normal smoothness loss
- [ ] Empty-space regularization
- [ ] Density regularization

---

## Phase 4 — Mesh Generation

- [ ] Marching Cubes
- [ ] OBJ export
- [ ] PLY export
- [ ] Mesh refinement

---

## Phase 5 — Research Extensions

- [ ] Multi-view diffusion priors
- [ ] MVDream integration
- [ ] Zero-1-to-3 comparison
- [ ] 3D Gaussian Splatting backend
- [ ] Quantitative evaluation
- [ ] Ablation studies
- [ ] Performance benchmarking

---

# Results

🚧 **Results will be added as the project progresses.**

Planned results include:

- Training visualizations
- Multi-view renderings
- Camera trajectory videos
- Extracted meshes
- Quantitative comparisons
- Ablation study results
- Runtime analysis
- Memory consumption benchmarks

---

# References

1. **Poole, B., Jain, A., Barron, J. T., & Mildenhall, B.**
   *DreamFusion: Text-to-3D using 2D Diffusion.*
   arXiv, 2022.

2. **Mildenhall, B., et al.**
   *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.*
   ECCV, 2020.

3. **Rombach, R., et al.**
   *High-Resolution Image Synthesis with Latent Diffusion Models.*
   CVPR, 2022.

4. **Shi, Y., et al.**
   *MVDream: Multi-view Diffusion for 3D Generation.*
   arXiv, 2023.

5. **Lin, C., et al.**
   *Magic3D: High-Resolution Text-to-3D Content Creation.*
   CVPR, 2023.

6. **Tang, J., et al.**
   *DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation.*
   ICLR, 2024.

---

# Acknowledgements

This project is inspired by the DreamFusion family of methods and aims to provide a lightweight, educational, and extensible implementation for studying Score Distillation Sampling under limited computational resources.

---

## Project Status

> 🚧 **Active Research Repository**
>
> The implementation is continuously evolving. Features, APIs, and training strategies may change as new experiments and improvements are integrated. Contributions, discussions, and suggestions are welcome.
