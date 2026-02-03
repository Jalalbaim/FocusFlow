from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

# -----------------------------
# 1) Forward noising for FlowMatch (your function)
# -----------------------------
def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Forward process in flow-matching
    sample_t = sigma * noise + (1 - sigma) * sample
    """
    scheduler._init_step_index(timestep)
    sigma = scheduler.sigmas[scheduler.step_index]
    return sigma * noise + (1.0 - sigma) * sample


# -----------------------------
# 2) SD3 velocity (your function, slightly hardened)
# -----------------------------
def calc_v_sd3(
    pipe,
    src_tar_latent_model_input,
    src_tar_prompt_embeds,
    src_tar_pooled_prompt_embeds,
    src_guidance_scale,
    tar_guidance_scale,
    t,
):
    # broadcast timestep
    timestep = t.expand(src_tar_latent_model_input.shape[0])

    with torch.no_grad():
        noise_pred_src_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        if pipe.do_classifier_free_guidance:
            src_u, src_c, tar_u, tar_c = noise_pred_src_tar.chunk(4)
            v_src = src_u + src_guidance_scale * (src_c - src_u)
            v_tar = tar_u + tar_guidance_scale * (tar_c - tar_u)
        else:
            # If CFG disabled, the code calling this should pass two latents (src, tar)
            # This branch is here for completeness.
            v_src, v_tar = noise_pred_src_tar.chunk(2)

    return v_src, v_tar


# -----------------------------
# 3) Utilities: prompt encoding and mask shaping
# -----------------------------
def _encode_prompts_sd3(
    pipe,
    device,
    src_prompt: str,
    tar_prompt: str,
    negative_prompt: str,
    src_guidance_scale: float,
    tar_guidance_scale: float,
):
    # Source embeds
    pipe._guidance_scale = src_guidance_scale
    (src_prompt_embeds, src_negative_prompt_embeds,
     src_pooled_prompt_embeds, src_negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    # Target embeds
    pipe._guidance_scale = tar_guidance_scale
    (tar_prompt_embeds, tar_negative_prompt_embeds,
     tar_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        device=device,
    )

    # Concatenate for calc_v_sd3 chunk(4): [src_uncond, src_text, tar_uncond, tar_text]
    src_tar_prompt_embeds = torch.cat(
        [src_negative_prompt_embeds, src_prompt_embeds,
         tar_negative_prompt_embeds, tar_prompt_embeds],
        dim=0
    )
    src_tar_pooled_prompt_embeds = torch.cat(
        [src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds,
         tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds],
        dim=0
    )

    return src_tar_prompt_embeds, src_tar_pooled_prompt_embeds


def _prep_mask_for_latents(mask_hw: torch.Tensor, x_src: torch.Tensor) -> torch.Tensor:
    """
    Convert mask (H,W) or (1,1,H,W) into (B,C,H,W) with x_src's shape.
    """
    device = x_src.device
    dtype = x_src.dtype
    B, C, H, W = x_src.shape

    M = mask_hw
    if isinstance(M, torch.Tensor) is False:
        M = torch.tensor(M)

    M = M.to(device=device, dtype=torch.float32)

    if M.ndim == 2:
        M = M[None, None, ...]   # (1,1,H,W)
    elif M.ndim == 3:
        M = M[None, ...]         # (1,*,H,W)

    if M.shape[-2:] != (H, W):
        M = F.interpolate(M, size=(H, W), mode="bilinear", align_corners=False)

    M = M.clamp(0.0, 1.0)

    if M.shape[0] == 1 and B > 1:
        M = M.expand(B, -1, -1, -1)
    if M.shape[1] == 1 and C > 1:
        M = M.expand(-1, C, -1, -1)

    return M.to(dtype=dtype)


def _clip_extremes_percentile(x: torch.Tensor, q_low: float = 0.01, q_high: float = 0.99) -> torch.Tensor:
    """
    DiffEdit says "remove extreme values" but does not specify the exact rule.
    This is an explicit, robust choice: percentile clipping.
    """
    flat = x.flatten()
    lo = torch.quantile(flat, q_low)
    hi = torch.quantile(flat, q_high)
    return x.clamp(lo, hi)


def _soften_mask(mask01_hw: torch.Tensor, blur_ks: int = 5, dilate_ks: int = 0) -> torch.Tensor:
    """
    Smooth edges to avoid seams when blending velocities.
    - blur via avg_pool
    - optional dilation via max_pool (expands region slightly)
    """
    M = mask01_hw[None, None, ...]  # (1,1,H,W)

    if dilate_ks and dilate_ks > 1:
        pad = dilate_ks // 2
        M = F.max_pool2d(M, kernel_size=dilate_ks, stride=1, padding=pad)

    if blur_ks and blur_ks > 1:
        pad = blur_ks // 2
        M = F.avg_pool2d(M, kernel_size=blur_ks, stride=1, padding=pad)

    return M.squeeze(0).squeeze(0).clamp(0.0, 1.0)


# -----------------------------
# 4) Mask generation (SD3-native, DiffEdit-style)
# -----------------------------
@torch.no_grad()
def create_diffedit_mask_sd3(
    pipe,
    scheduler,
    x_src: torch.Tensor,          # latent image, shape (B,C,H,W)
    src_prompt: str,
    tar_prompt: str,
    negative_prompt: str,
    retrieve_timesteps_fn,        # pass retrieve_timesteps from diffusers
    T_steps: int = 50,
    strength: float = 0.5,        # DiffEdit default: 50% noise
    n: int = 10,                  # DiffEdit default n=10
    guidance_mask: float = 5.0,   # DiffEdit uses >=3, default 5 on ImageNet 
    q_low: float = 0.01,
    q_high: float = 0.99,
    threshold: float = 0.5,       # DiffEdit default threshold 0.5 
    blur_ks: int = 5,
    dilate_ks: int = 0,
    seed_base: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      mask_soft_hw : float32 (H,W) in [0,1]
      mask_bin_hw  : uint8 (H,W) in {0,1}
      t_mask       : the timestep used for mask inference
    """

    device = x_src.device

    # timesteps
    timesteps, _ = retrieve_timesteps_fn(scheduler, T_steps, device, timesteps=None)

    # choose mask timestep like img2img: timesteps[-int(steps*strength)]
    init_timestep = int(T_steps * strength)
    init_timestep = max(1, min(init_timestep, len(timesteps)))
    t_mask = timesteps[-init_timestep]  # same convention as your DDIM img2img code

    # encode prompts once (use same guidance for both prompts for mask inference)
    src_tar_prompt_embeds, src_tar_pooled_prompt_embeds = _encode_prompts_sd3(
        pipe=pipe,
        device=device,
        src_prompt=src_prompt,
        tar_prompt=tar_prompt,
        negative_prompt=negative_prompt,
        src_guidance_scale=guidance_mask,
        tar_guidance_scale=guidance_mask,
    )

    B, C, H, W = x_src.shape
    acc = torch.zeros((H, W), device=device, dtype=torch.float32)

    for k in range(n):
        gen = torch.Generator(device=device).manual_seed(seed_base + 100 * k)
        fwd_noise = torch.randn_like(x_src, generator=gen)

        # noised latent at t_mask (same noised latent used for both prompts)
        x_t = scale_noise(scheduler, x_src, t_mask, noise=fwd_noise)

        # Use the same x_t for all 4 slots so that src/tar predictions are compared at the same state
        latent_in = torch.cat([x_t, x_t, x_t, x_t], dim=0) if pipe.do_classifier_free_guidance else x_t

        v_src, v_tar = calc_v_sd3(
            pipe,
            latent_in,
            src_tar_prompt_embeds,
            src_tar_pooled_prompt_embeds,
            guidance_mask,
            guidance_mask,
            t_mask,
        )

        # spatial diff map: mean absolute difference over channels
        diff_hw = (v_tar - v_src).abs().mean(dim=1).float()  # (B,H,W)
        acc += diff_hw.mean(dim=0)  # average over batch

    mask = acc / float(n)  # (H,W)

    # "remove extreme values" (explicit percentile clipping; exact rule not specified in paper)
    mask = _clip_extremes_percentile(mask, q_low=q_low, q_high=q_high)

    # rescale to [0,1] (DiffEdit)
    mmin, mmax = mask.min(), mask.max()
    mask01 = (mask - mmin) / (mmax - mmin + 1e-8)

    # soften boundaries (recommended for velocity blending)
    mask01 = _soften_mask(mask01, blur_ks=blur_ks, dilate_ks=dilate_ks)

    # binarize at 0.5 by default (DiffEdit) 
    mask_bin = (mask01 >= threshold).to(torch.uint8)

    return mask01.detach().cpu(), mask_bin.detach().cpu(), t_mask


# -----------------------------
# 5) FlowEditSD3 with masked velocity blending
# -----------------------------
@torch.no_grad()
def FlowEditSD3_masked(
    pipe,
    scheduler,
    retrieve_timesteps_fn,
    x_src: torch.Tensor,
    src_prompt: str,
    tar_prompt: str,
    negative_prompt: str,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 0,
    n_max: int = 15,
    # mask options:
    mask_soft_hw: Optional[torch.Tensor] = None,  # (H,W) in [0,1] or None
    auto_mask: bool = True,
    mask_strength: float = 0.5,
    mask_n: int = 10,
    mask_guidance: float = 5.0,
    mask_blur_ks: int = 5,
    mask_dilate_ks: int = 0,
    mask_threshold: float = 0.5,
    mask_q_low: float = 0.01,
    mask_q_high: float = 0.99,
    mask_seed_base: int = 0,
):
    device = x_src.device

    timesteps, T_steps = retrieve_timesteps_fn(scheduler, T_steps, device, timesteps=None)
    pipe._num_timesteps = len(timesteps)

    # Build prompt embeds for the main editing process (potentially different guidance scales)
    src_tar_prompt_embeds, src_tar_pooled_prompt_embeds = _encode_prompts_sd3(
        pipe=pipe,
        device=device,
        src_prompt=src_prompt,
        tar_prompt=tar_prompt,
        negative_prompt=negative_prompt,
        src_guidance_scale=src_guidance_scale,
        tar_guidance_scale=tar_guidance_scale,
    )

    # Create / prepare mask
    if mask_soft_hw is None and auto_mask:
        mask_soft_hw, mask_bin_hw, _ = create_diffedit_mask_sd3(
            pipe=pipe,
            scheduler=scheduler,
            x_src=x_src,
            src_prompt=src_prompt,
            tar_prompt=tar_prompt,
            negative_prompt=negative_prompt,
            retrieve_timesteps_fn=retrieve_timesteps_fn,
            T_steps=T_steps,
            strength=mask_strength,
            n=mask_n,
            guidance_mask=mask_guidance,
            q_low=mask_q_low,
            q_high=mask_q_high,
            threshold=mask_threshold,
            blur_ks=mask_blur_ks,
            dilate_ks=mask_dilate_ks,
            seed_base=mask_seed_base,
        )
    elif mask_soft_hw is None:
        mask_bin_hw = None
    else:
        # user provided a mask
        if not torch.is_tensor(mask_soft_hw):
            mask_soft_hw = torch.tensor(mask_soft_hw)
        mask_bin_hw = (mask_soft_hw >= mask_threshold).to(torch.uint8)

    M = None
    if mask_soft_hw is not None:
        M = _prep_mask_for_latents(mask_soft_hw.to(device=device), x_src)  # (B,C,H,W)

    # initialize ODE path
    zt_edit = x_src.clone()

    for i, t in tqdm(list(enumerate(timesteps)), desc="FlowEditSD3_masked"):
        if T_steps - i > n_max:
            continue

        t_i = t / 1000
        if i + 1 < len(timesteps):
            t_im1 = timesteps[i + 1] / 1000
        else:
            t_im1 = torch.zeros_like(t_i).to(t_i.device)

        if T_steps - i > n_min:
            # ODE phase: update with masked delta velocity
            V_delta_avg = torch.zeros_like(x_src)

            for k in range(n_avg):
                fwd_noise = torch.randn_like(x_src)

                # FlowEdit uses linear interpolant in the rectified-flow setting
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src

                latent_in = torch.cat([zt_src, zt_src, zt_tar, zt_tar], dim=0) if pipe.do_classifier_free_guidance else (zt_src, zt_tar)

                Vt_src, Vt_tar = calc_v_sd3(
                    pipe,
                    latent_in,
                    src_tar_prompt_embeds,
                    src_tar_pooled_prompt_embeds,
                    src_guidance_scale,
                    tar_guidance_scale,
                    t,
                )

                delta = (Vt_tar - Vt_src)
                if M is not None:
                    delta = M * delta  # <--- masked delta update

                V_delta_avg += delta / float(n_avg)

            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg.to(torch.float32)
            zt_edit = zt_edit.to(dtype=x_src.dtype)

        else:
            # last n_min steps: sampling-like phase with masked blending of velocities
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            # evaluate both velocities at the SAME xt_tar state
            latent_in = torch.cat([xt_tar, xt_tar, xt_tar, xt_tar], dim=0) if pipe.do_classifier_free_guidance else xt_tar

            Vt_src, Vt_tar = calc_v_sd3(
                pipe,
                latent_in,
                src_tar_prompt_embeds,
                src_tar_pooled_prompt_embeds,
                src_guidance_scale,
                tar_guidance_scale,
                t,
            )

            if M is None:
                V_final = Vt_tar
            else:
                V_final = M * Vt_tar + (1 - M) * Vt_src  # <--- your equation

            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * V_final.to(torch.float32)
            xt_tar = prev_sample.to(dtype=x_src.dtype)

    out = zt_edit if n_min == 0 else xt_tar
    return out, mask_soft_hw, mask_bin_hw


# -----------------------------
# 0) Helpers: encode/decode image <-> SD3 latents
# -----------------------------
def preprocess_pil(image: Image.Image, size=1024):
    # SD3 is commonly used at 1024; if your setup is 512, change size accordingly.
    image = image.convert("RGB").resize((size, size))
    x = torch.from_numpy(np.array(image)).float() / 255.0  # [H,W,3] in [0,1]
    x = x.permute(2, 0, 1).unsqueeze(0)                   # [1,3,H,W]
    x = x * 2.0 - 1.0                                     # [-1,1]
    return x


@torch.no_grad()
def encode_image_to_latents(pipe, image: Image.Image, size=1024):
    x = preprocess_pil(image, size=size)
    
    # Avec cpu_offload, on doit s'assurer que VAE est sur CUDA
    pipe.vae.to("cuda")
    x = x.to("cuda", dtype=pipe.vae.dtype)
    
    latents_dist = pipe.vae.encode(x).latent_dist
    latents = latents_dist.sample()

    # Use scaling_factor from VAE config if available (don't hardcode 0.18215 for SD3)
    sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
    latents = latents * sf
    return latents


@torch.no_grad()
def decode_latents_to_pil(pipe, latents: torch.Tensor):
    # S'assurer que VAE est sur CUDA
    pipe.vae.to("cuda")
    latents = latents.to("cuda")
    
    sf = getattr(pipe.vae.config, "scaling_factor", 1.0)
    z = latents / sf
    img = pipe.vae.decode(z).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
    img = (img * 255).round().astype(np.uint8)[0]
    return Image.fromarray(img)


def upsample_mask_to_image(mask_hw: np.ndarray, out_size: int):
    # mask_hw is (H,W) in latent-res; upsample to (out_size,out_size) for visualization
    m = torch.from_numpy(mask_hw).float()[None, None]  # (1,1,H,W)
    m = F.interpolate(m, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return m[0,0].clamp(0,1).numpy()


# -----------------------------
# 1) Load SD3 pipe + scheduler
# -----------------------------
MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"

dtype = torch.float16  # or torch.bfloat16 depending on your GPU + model
device = "cuda"

# Optimisations mémoire pour éviter OOM
pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    variant="fp16",  # utiliser les poids fp16
)

# Activer les optimisations mémoire disponibles pour SD3
pipe.enable_model_cpu_offload()  # décharge les modèles sur CPU quand pas utilisés

# Activer VAE tiling si disponible (pour traiter les images par tuiles)
if hasattr(pipe, 'enable_vae_tiling'):
    pipe.enable_vae_tiling()

# NOTE: On ne fait pas .to(device) car cpu_offload gère ça automatiquement

# IMPORTANT: your FlowEdit code expects a flow-matching scheduler w/ sigmas
scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = scheduler

# Nettoyer la mémoire CUDA avant de commencer
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"CUDA disponible. Mémoire GPU libre: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


# -----------------------------
# 3) Run one example
# -----------------------------
# Input image
img_path = "./FlowEdit/example_images/bear.png"
init_img = Image.open(img_path)

# Encode to latents
# NOTE: Réduit à 512 pour économiser la mémoire GPU (changez à 1024 si vous avez assez de VRAM)
IMG_SIZE = 512  # Changé de 1024 à 512 pour économiser la mémoire
x_src = encode_image_to_latents(pipe, init_img, size=IMG_SIZE)

src_prompt = "A brown bear walking through a stream of water."
tar_prompt = "A brown bear sitting on the snow and looking face forward"
negative_prompt = ""

# Run masked FlowEdit (auto_mask=True => it will generate the mask + apply it)
edited_latents, mask_soft, mask_bin = FlowEditSD3_masked(
    pipe=pipe,
    scheduler=scheduler,
    retrieve_timesteps_fn=retrieve_timesteps,
    x_src=x_src,
    src_prompt=src_prompt,
    tar_prompt=tar_prompt,
    negative_prompt=negative_prompt,

    T_steps=50,
    n_avg=1,
    src_guidance_scale=3.5,
    tar_guidance_scale=13.5,
    n_min=0,
    n_max=15,

    auto_mask=True,
    mask_strength=0.5,      # DiffEdit default idea: 50% noise
    mask_n=10,              # DiffEdit default idea: average over 10 noises
    mask_guidance=5.0,      # stable
    mask_blur_ks=5,         # reduces seams
    mask_dilate_ks=0,
    mask_threshold=0.5,
)

# Decode to image
out_img = decode_latents_to_pil(pipe, edited_latents)
out_img.save("flowedit_masked_out.png")
print("Saved:", "flowedit_masked_out.png")


# -----------------------------
# 4) Visualize mask overlay (optional)
# -----------------------------
if mask_soft is not None:
    # mask_soft is (H,W) in latent resolution, likely 64x64 or similar.
    m = upsample_mask_to_image(mask_soft.numpy(), out_size=IMG_SIZE)

    init_vis = init_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    init_arr = np.array(init_vis).astype(np.float32) / 255.0

    overlay = init_arr.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + 0.6 * m, 0, 1)  # red overlay (simple)
    plt.figure(figsize=(14,5))
    plt.subplot(1,3,1)
    plt.title("Input")
    plt.imshow(init_arr)
    plt.axis("off")
    plt.subplot(1,3,2)
    plt.title("Mask (soft)")
    plt.imshow(m, cmap="gray")
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()
