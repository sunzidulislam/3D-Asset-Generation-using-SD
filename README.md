# Few-Shot Text-to-3D Generation via Score Distillation Sampling (SDS)

A from-scratch, educational reimplementation of the core idea behind
DreamFusion (Poole et al., 2022): generating a 3D object from a text
prompt alone, using a **frozen, pretrained 2D diffusion model** as the
only source of supervision — no 3D training data of any kind.

---

## 1. Motivation

Most 3D generative models are trained on 3D datasets (ShapeNet, Objaverse,
etc.), which are small, narrow in category coverage, and expensive to
collect compared to 2D image datasets. 2D diffusion models, on the other
hand, have been trained on billions of images and have a very strong
implicit understanding of what objects look like from *any* viewpoint.

**Core question this project explores:** can we "borrow" that 2D prior to
generate 3D content, without ever training on 3D data? This is the
premise of Score Distillation Sampling (SDS), introduced in DreamFusion.

This project reimplements that idea in a compact form that runs on a
single Kaggle T4 GPU, and is scoped so it can be extended with a small
novel experiment (see §6, Future Work) — turning it from a reproduction
into a research project.

---

## 2. Background

### 2.1 Neural Radiance Fields (NeRF)
A NeRF represents a 3D scene as a continuous function

```
F(x, y, z) → (σ, RGB)
```

mapping a 3D point to a volume density `σ` and a color. An image is
rendered by casting a ray per pixel, sampling points along it, and
alpha-compositing their densities/colors (volume rendering). Because this
process is fully differentiable, gradients can flow from a 2D loss on the
rendered image back into the NeRF's parameters.

### 2.2 Diffusion Models
A diffusion model is trained to reverse a fixed noising process: given a
noisy image `x_t` at noise level `t`, it predicts the noise `ε` that was
added, conditioned on a text prompt. At inference, iteratively denoising
from pure noise produces a novel image matching the prompt. Crucially,
the model's noise-prediction network `ε_φ(x_t, t, y)` implicitly encodes
"what this prompt should look like at every noise level" — this is the
signal SDS exploits.

### 2.3 Score Distillation Sampling (SDS)
Instead of running a full reverse diffusion process, SDS uses the
diffusion model purely as a **critic**. For a rendered image `x = g(θ)`
(a function of the NeRF parameters `θ`), SDS:

1. Adds noise at a random timestep `t`: `x_t = √(ᾱ_t) x + √(1-ᾱ_t) ε`
2. Asks the frozen UNet to predict the noise: `ε_φ(x_t, t, y)`
3. Treats the discrepancy `(ε_φ(x_t,t,y) − ε)` as a gradient signal

The SDS gradient with respect to the NeRF parameters is:

```
∇_θ L_SDS = E_{t,ε} [ w(t) · (ε_φ(x_t, t, y) − ε) · ∂x/∂θ ]
```

where `w(t)` is a noise-level-dependent weight. Note the UNet is **never
backpropagated through** — only used for inference — which is what makes
this tractable: we're using a huge pretrained model as a fixed critic,
not fine-tuning it.

Intuitively: at every training step, we render the NeRF from a random
camera, and nudge its parameters so that the rendered image "looks more
plausible" to the diffusion model at every noise level, for every
viewpoint. Repeated over thousands of random views, this forces the NeRF
to converge to a 3D-consistent object matching the prompt — because a
single set of NeRF parameters must satisfy the diffusion critic from
*all* angles simultaneously.

---

## 3. Methodology (this implementation)

### 3.1 Pipeline overview

```
 text prompt
     │
     ▼
 CLIP text encoder (frozen) ──► text embedding (cond + uncond, for CFG)
                                        │
random camera pose ──► ray casting ──► NeRF (TRAINABLE) ──► rendered image (64×64)
                                                                    │
                                                     bilinear upsample to 512×512
                                                                    │
                                                         VAE encoder (frozen) ──► latent (64×64×4)
                                                                    │
                                                    add noise at random timestep t
                                                                    │
                                                   UNet noise predictor (frozen) ──► ε_φ
                                                                    │
                                              SDS gradient = w(t)·(ε_φ − ε)
                                                                    │
                                       backprop through VAE-encode + render into NeRF weights
                                                                    │
                                                            Adam optimizer step
```

### 3.2 Components

| Component | Role | Trainable? | Source |
|---|---|---|---|
| Tiny NeRF (positional-encoding MLP) | 3D representation being generated | ✅ Yes | Implemented from scratch |
| CLIP text encoder | Encodes prompt/negative prompt | ❌ Frozen | `stable-diffusion-v1-5` (Hugging Face) |
| VAE (encoder only) | Maps rendered RGB → latent space | ❌ Frozen | `stable-diffusion-v1-5` (Hugging Face) |
| UNet | Noise predictor / critic | ❌ Frozen | `stable-diffusion-v1-5` (Hugging Face) |
| DDPM Scheduler | Defines noise schedule `ᾱ_t` | — | `stable-diffusion-v1-5` (Hugging Face) |

Only the NeRF (~a few hundred thousand parameters) is ever updated. Everything
from Stable Diffusion stays frozen throughout — it is used exclusively for
inference (one UNet forward pass per training step, no backward pass through it).

### 3.3 Rendering details
- Camera is sampled uniformly on a partial sphere around the origin
  (azimuth ∈ [0, 2π), elevation ∈ [−10°, 60°]) so the model sees the object
  from many angles.
- Rays are cast per pixel at a low resolution (64×64) for compute
  efficiency; volume rendering uses 64 stratified samples per ray between
  a near/far bound.
- A white background is composited in for rays with low accumulated
  density, which discourages the NeRF from filling the whole volume with
  low-opacity haze.

### 3.4 Classifier-free guidance
Both a conditional (prompt) and unconditional (negative prompt) noise
prediction are computed and combined:

```
ε_φ = ε_uncond + s · (ε_cond − ε_uncond)
```

with a large guidance scale (`s ≈ 40`), which is standard for SDS — much
higher than typical 2D image generation (`s ≈ 7.5`), because SDS needs a
stronger signal to escape blurry/degenerate solutions early in training.

### 3.5 Timestep sampling
Timesteps are restricted to `t ∈ [0.02·T, 0.98·T]`, avoiding the extremes
of the noise schedule where the gradient signal is either near-zero
(very low noise) or dominated by pure noise (very high noise) — a known
stabilization trick from the DreamFusion line of work.

---

## 4. Experimental setup

| Setting | Value |
|---|---|
| Base 2D model | Stable Diffusion 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`) |
| GPU | 1× NVIDIA T4 (16GB), Kaggle |
| NeRF render resolution | 64×64 |
| Guidance (VAE/UNet) resolution | 512×512 |
| Samples per ray | 64 |
| Guidance scale | 40 |
| Optimizer | Adam, lr = 1e-2 |
| Iterations | 1500 |
| Precision | fp16 for SD components, fp32 for NeRF |

---

## 5. How to run

1. Open a Kaggle Notebook, enable **GPU → T4 x1**.
2. Paste `sds_text_to_3d_t4.py` cell-by-cell (markers are `# %% [Cell N]`).
3. Run Cell 1's pip install, then **restart the kernel** (required — see
   comment in the script; Kaggle's preinstalled `huggingface_hub` conflicts
   with older `diffusers` releases).
4. Run Cells 2–8 in order. Set `PROMPT` in Cell 2 to your object of choice.
5. Outputs are written to `/kaggle/working/sds_output/`:
   - `preview_step{N}.png` — single-view snapshots every 250 steps
   - `turntable.gif` — 360° render of the final result

Expect ~30–50 minutes for 1500 iterations on a T4 at this resolution.

---

## 6. Limitations & future work (ideas for extending this into a research contribution)

This implementation deliberately omits several components used in full
DreamFusion-family pipelines, each of which is a legitimate direction to
extend this project:

- **No view-dependent prompting.** Real pipelines append text like "front
  view" / "back view" / "side view" to the prompt based on the sampled
  camera azimuth. Without this, the model has no signal distinguishing
  "front" from "back," which is the primary cause of the well-known
  **Janus problem** (multi-face artifacts — e.g., a generated animal
  growing a face on the back of its head). **This is the single highest-value
  addition** for turning this from a reproduction into a research project:
  implement it, then run a controlled A/B comparison (with vs. without)
  on a fixed set of prompts and measure Janus-artifact frequency.
- **No normal-smoothness / orientation losses**, which real pipelines add
  to discourage flat, degenerate, "billboard" geometry.
- **No mesh extraction.** The output stays an implicit NeRF; a natural
  extension is marching cubes to export a `.obj`/`.ply` mesh.
- **Low resolution**, constrained by T4 memory — a natural ablation is
  studying the resolution/quality tradeoff, or replacing the NeRF with
  3D Gaussian Splatting for faster convergence.
- **Single 2D prior (SD 1.5).** Swapping in a multi-view-aware diffusion
  model (e.g., MVDream, Zero-1-to-3) is a known, published mitigation for
  3D inconsistency and would make for a strong comparative experiment.

## 7. References

- Poole, Jain, Barron, Mildenhall. *DreamFusion: Text-to-3D using 2D
  Diffusion*. 2022.
- Mildenhall et al. *NeRF: Representing Scenes as Neural Radiance
  Fields for View Synthesis*. 2020.
- Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion
  Models (Stable Diffusion)*. 2022.
- Shi et al. *MVDream: Multi-view Diffusion for 3D Generation*. 2023.
