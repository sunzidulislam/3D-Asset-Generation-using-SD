# =========================================================================
# Text-to-3D via Score Distillation Sampling (SDS) — Kaggle T4 version
# -------------------------------------------------------------------------
# Idea: optimize a small NeRF from SCRATCH (no 3D data) using a frozen
# Stable Diffusion model as a "critic". At every step we:
#   1. Pick a random camera around the object
#   2. Render an image from the NeRF
#   3. Noise it, ask SD "was this noise level correct for this prompt?"
#   4. Use the prediction error as a gradient signal into the NeRF
#
# This is a simplified, educational reimplementation of the core idea
# behind DreamFusion / Stable-DreamFusion. It is NOT feature complete
# (no view-dependent prompting, no orientation/normal losses, no mesh
# extraction) — but it runs end-to-end on a single T4 (16GB) and gives
# you a real result + a real codebase to extend for your CV project.
#
# Run this in a Kaggle notebook with GPU (T4 x1 or x2) enabled.
# =========================================================================

# %% [Cell 1] Install deps (Kaggle usually has torch pre-installed)
# IMPORTANT: run this cell, then RESTART THE KERNEL (Run > Restart session),
# then run Cell 2 onward. Kaggle's pre-installed huggingface_hub is newer
# than old diffusers versions expect (removed `cached_download`), so we
# pin everything together to a known-compatible set.
# !pip install -q -U diffusers==0.31.0 transformers==4.46.0 accelerate==1.0.1 huggingface_hub==0.26.2 imageio==2.34.0

# %% [Cell 2] Imports & config
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import numpy as np
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "This script needs a GPU — enable T4 in Kaggle settings."

# ---- Config (tuned to fit T4's 16GB) ----
PROMPT          = "a corgi wearing a wizard hat, high quality, detailed fur"
NEGATIVE_PROMPT = "blurry, low quality, flat, cartoon"
NERF_RES        = 64          # NeRF render resolution (kept small — it's the expensive part)
GUIDANCE_RES    = 512         # resolution fed into SD's VAE (SD expects ~512)
N_SAMPLES       = 64          # points sampled per ray
GUIDANCE_SCALE  = 25.0        # classifier-free guidance weight. Was 40 — too aggressive, caused collapse.
N_ITERS         = 1500        # training steps; raise if you have time budget
LR              = 2e-3        # was 1e-2 — too high, caused the NeRF to blow up / saturate to one flat color
GRAD_CLIP       = 1.0         # max gradient norm; prevents divergence from large early SDS gradients
CAM_RADIUS      = 2.6
SAVE_EVERY      = 100         # save previews more often early on so you can catch a collapse quickly
OUT_DIR         = "/kaggle/working/sds_output"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"  # HF Hub mirror of SD1.5

# %% [Cell 3] Load frozen Stable Diffusion components (fp16 to save VRAM)
print("Loading Stable Diffusion components from Hugging Face...")
tokenizer    = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
vae          = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float16).to(device)
unet         = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=torch.float16).to(device)
scheduler    = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

for p in text_encoder.parameters(): p.requires_grad_(False)
for p in vae.parameters():          p.requires_grad_(False)
for p in unet.parameters():         p.requires_grad_(False)
text_encoder.eval(); vae.eval(); unet.eval()
vae.enable_slicing()  # memory saver

# min/max diffusion timesteps used for SDS (avoid extremes — standard trick)
T_MIN, T_MAX = int(0.02 * 1000), int(0.98 * 1000)


@torch.no_grad()
def get_text_embeddings(prompt, negative_prompt):
    """Returns concatenated [uncond, cond] embeddings for classifier-free guidance."""
    def encode(text):
        tok = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").to(device)
        return text_encoder(tok.input_ids)[0]
    uncond = encode(negative_prompt)
    cond   = encode(prompt)
    return torch.cat([uncond, cond], dim=0).half()


# %% [Cell 4] Tiny NeRF (positional-encoding MLP) — this is what we're TRAINING
class PositionalEncoding(nn.Module):
    def __init__(self, n_freqs=10):
        super().__init__()
        self.freqs = 2.0 ** torch.arange(n_freqs)

    def forward(self, x):
        out = [x]
        for f in self.freqs.to(x.device):
            out += [torch.sin(x * f * math.pi), torch.cos(x * f * math.pi)]
        return torch.cat(out, dim=-1)


class TinyNeRF(nn.Module):
    def __init__(self, n_freqs=10, hidden=128):
        super().__init__()
        self.pe = PositionalEncoding(n_freqs)
        in_dim = 3 + 3 * 2 * n_freqs
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )
        self.sigma_head = nn.Linear(hidden, 1)
        self.rgb_head   = nn.Linear(hidden, 3)

    def forward(self, xyz):
        h = self.backbone(self.pe(xyz))
        sigma = F.softplus(self.sigma_head(h) - 1.0)      # density >= 0, biased toward empty
        rgb   = torch.sigmoid(self.rgb_head(h))            # color in [0,1]
        return sigma, rgb


# %% [Cell 5] Camera sampling + volume rendering
def sample_camera(radius=CAM_RADIUS):
    """Random camera on a sphere looking at the origin."""
    azimuth   = np.random.uniform(0, 2 * math.pi)
    elevation = np.random.uniform(math.radians(-10), math.radians(60))
    x = radius * math.cos(elevation) * math.cos(azimuth)
    y = radius * math.sin(elevation)
    z = radius * math.cos(elevation) * math.sin(azimuth)
    cam_pos = torch.tensor([x, y, z], dtype=torch.float32)

    forward = -cam_pos / cam_pos.norm()
    up_hint = torch.tensor([0.0, 1.0, 0.0])
    right = torch.cross(forward, up_hint); right = right / (right.norm() + 1e-8)
    up = torch.cross(right, forward)
    return cam_pos, right, up, forward


def get_rays(res, cam_pos, right, up, forward, fov=50.0):
    device_ = cam_pos.device
    i, j = torch.meshgrid(
        torch.linspace(-1, 1, res, device=device_),
        torch.linspace(-1, 1, res, device=device_),
        indexing="xy",
    )
    scale = math.tan(math.radians(fov) / 2)
    dirs = (i[..., None] * right * scale
            + (-j[..., None]) * up * scale
            + forward)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    origins = cam_pos.expand_as(dirs)
    return origins.reshape(-1, 3), dirs.reshape(-1, 3)


def render_rays(nerf, origins, dirs, near=1.0, far=4.0, n_samples=N_SAMPLES):
    t = torch.linspace(near, far, n_samples, device=origins.device)
    t = t + torch.rand_like(t) * (far - near) / n_samples  # stratified jitter
    pts = origins[:, None, :] + dirs[:, None, :] * t[None, :, None]  # [N, S, 3]

    sigma, rgb = nerf(pts.reshape(-1, 3))
    sigma = sigma.reshape(-1, n_samples)
    rgb = rgb.reshape(-1, n_samples, 3)

    delta = t[1:] - t[:-1]
    delta = torch.cat([delta, torch.tensor([1e10], device=t.device)])
    alpha = 1.0 - torch.exp(-sigma * delta[None, :])
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * trans

    pixel_rgb = (weights[..., None] * rgb).sum(dim=1)
    acc = weights.sum(dim=1, keepdim=True)
    pixel_rgb = pixel_rgb + (1.0 - acc)  # white background
    return pixel_rgb


def render_image(nerf, res=NERF_RES):
    cam_pos, right, up, forward = sample_camera()
    cam_pos, right, up, forward = [x.to(device) for x in (cam_pos, right, up, forward)]
    origins, dirs = get_rays(res, cam_pos, right, up, forward)
    pixels = render_rays(nerf, origins, dirs)
    img = pixels.reshape(res, res, 3).permute(2, 0, 1)  # [3, H, W]
    return img


# %% [Cell 6] SDS loss
def sds_loss(nerf, text_embeds):
    img = render_image(nerf, NERF_RES)                 # [3, res, res], requires_grad
    img = img.clamp(0, 1)                               # guard against numerical drift outside [0,1]
    img_up = F.interpolate(img[None], size=(GUIDANCE_RES, GUIDANCE_RES),
                            mode="bilinear", align_corners=False)
    img_up = img_up.half() * 2 - 1                      # VAE expects [-1, 1]

    latents = vae.encode(img_up).latent_dist.mean * vae.config.scaling_factor  # [1,4,64,64]

    t = torch.randint(T_MIN, T_MAX, (1,), device=device).long()
    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, t)

    latent_in = torch.cat([noisy_latents] * 2)
    with torch.no_grad():
        noise_pred = unet(latent_in, t, encoder_hidden_states=text_embeds).sample
    noise_uncond, noise_cond = noise_pred.chunk(2)
    noise_pred = noise_uncond + GUIDANCE_SCALE * (noise_cond - noise_uncond)

    w = (1 - scheduler.alphas_cumprod.to(device)[t])     # SDS weighting term
    grad = w[:, None, None, None] * (noise_pred.float() - noise.float())

    # --- Gradient magnitude normalization (key stability fix) ---
    # Raw SDS gradient magnitude swings a lot step-to-step depending on the
    # sampled timestep t and how "wrong" the current render looks. Without
    # normalizing, a few outlier steps with huge magnitude dominate the
    # optimization direction even after norm-clipping, and can drag the
    # network into a saturated/degenerate flat-color state. Rescaling to a
    # consistent per-element magnitude keeps every step comparably sized.
    grad_mag = grad.abs().mean()
    if torch.isfinite(grad_mag) and grad_mag > 1e-8:
        grad = grad / grad_mag
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    # SDS trick: don't backprop through UNet; use grad as a fixed target
    target = (latents - grad).detach()
    loss = 0.5 * F.mse_loss(latents.float(), target, reduction="sum") / latents.shape[0]
    return loss, img.detach()


# %% [Cell 7] Training loop
def train():
    nerf = TinyNeRF().to(device)
    optimizer = torch.optim.Adam(nerf.parameters(), lr=LR)
    text_embeds = get_text_embeddings(PROMPT, NEGATIVE_PROMPT)

    for step in range(1, N_ITERS + 1):
        optimizer.zero_grad()
        loss, preview = sds_loss(nerf, text_embeds)

        if not torch.isfinite(loss):
            print(f"step {step:5d}: non-finite loss ({loss.item()}) — skipping this step")
            continue

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(nerf.parameters(), GRAD_CLIP)

        if not torch.isfinite(grad_norm):
            print(f"step {step:5d}: non-finite grad_norm — skipping optimizer step")
            optimizer.zero_grad()
            continue

        optimizer.step()

        if step % 50 == 0:
            print(f"step {step:5d}/{N_ITERS} | loss {loss.item():.2f} | grad_norm {grad_norm.item():.2f}")

        # early collapse check: if the rendered preview has near-zero
        # variance, the NeRF has saturated to a flat color. Cheap sanity
        # check so you don't wait 1500 steps to find out.
        if step % 100 == 0:
            std = preview.std().item()
            if std < 1e-3:
                print(f"  ⚠ WARNING: preview has near-zero variance (std={std:.5f}) "
                      f"— likely collapsed to a flat color. Consider lowering LR / "
                      f"GUIDANCE_SCALE further, or restarting.")

        if step % SAVE_EVERY == 0 or step == N_ITERS:
            save_path = os.path.join(OUT_DIR, f"preview_step{step}.png")
            imageio.imwrite(save_path, (preview.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            print(f"  saved {save_path}")

    return nerf


def render_turntable(nerf, n_frames=24, res=96, out_path=None):
    out_path = out_path or os.path.join(OUT_DIR, "turntable.gif")
    frames = []
    with torch.no_grad():
        for k in range(n_frames):
            azimuth = 2 * math.pi * k / n_frames
            cam_pos = torch.tensor([
                CAM_RADIUS * math.cos(azimuth),
                CAM_RADIUS * 0.25,
                CAM_RADIUS * math.sin(azimuth),
            ], dtype=torch.float32, device=device)
            forward = -cam_pos / cam_pos.norm()
            up_hint = torch.tensor([0.0, 1.0, 0.0], device=device)
            right = torch.cross(forward, up_hint); right = right / (right.norm() + 1e-8)
            up = torch.cross(right, forward)
            origins, dirs = get_rays(res, cam_pos, right, up, forward)
            pixels = render_rays(nerf, origins, dirs)
            img = pixels.reshape(res, res, 3).clamp(0, 1).cpu().numpy()
            frames.append((img * 255).astype(np.uint8))
    imageio.mimsave(out_path, frames, fps=12)
    print(f"Saved turntable to {out_path}")


# %% [Cell 8] Run it
if __name__ == "__main__":
    nerf = train()
    render_turntable(nerf)
