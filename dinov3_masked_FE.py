import gc
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from rembg import remove

# -----------------------------
# 1) Open Source Mask Generation (rembg)
# -----------------------------

@torch.no_grad()
def create_background_mask(image: Image.Image):
    """
    Generate a foreground mask using rembg (U2-Net based).
    Returns a tensor mask where 1.0 is foreground and 0.0 is background.
    """
    print("Generating mask using rembg...")
    # rembg.remove returns an RGBA image where alpha is the mask
    rgba = remove(image)
    mask_np = np.array(rgba)[:, :, 3] / 255.0
    mask = torch.from_numpy(mask_np).float()
    
    # RAM Cleanup
    gc.collect()
    
    return mask

# -----------------------------
# 2) Modified FlowEdit Logic for Background Editing
# -----------------------------

def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    scheduler._init_step_index(timestep)
    sigma = scheduler.sigmas[scheduler.step_index]
    return sigma * noise + (1.0 - sigma) * sample

def calc_v_sd3(
    pipe,
    src_tar_latent_model_input,
    src_tar_prompt_embeds,
    src_tar_pooled_prompt_embeds,
    src_guidance_scale,
    tar_guidance_scale,
    t,
):
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
            v_src, v_tar = noise_pred_src_tar.chunk(2)
    return v_src, v_tar

def _encode_prompts_sd3(
    pipe,
    device,
    src_prompt: str,
    tar_prompt: str,
    negative_prompt: str,
    src_guidance_scale: float,
    tar_guidance_scale: float,
):
    pipe._guidance_scale = src_guidance_scale
    (src_prompt_embeds, src_negative_prompt_embeds,
     src_pooled_prompt_embeds, src_negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=src_prompt,
        prompt_3=None,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt,
        negative_prompt_3=None,
        device=device,
    )

    pipe._guidance_scale = tar_guidance_scale
    (tar_prompt_embeds, tar_negative_prompt_embeds,
     tar_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=tar_prompt,
        prompt_3=None,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt,
        negative_prompt_3=None,
        device=device,
    )

    src_tar_prompt_embeds = torch.cat(
        [src_negative_prompt_embeds, src_prompt_embeds,
         tar_negative_prompt_embeds, tar_prompt_embeds], dim=0
    )
    src_tar_pooled_prompt_embeds = torch.cat(
        [src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds,
         tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0
    )
    return src_tar_prompt_embeds, src_tar_pooled_prompt_embeds

def _prep_mask_for_latents(mask_hw: torch.Tensor, x_src: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x_src.shape
    M = mask_hw.to(device=x_src.device, dtype=x_src.dtype)
    if M.ndim == 2:
        M = M[None, None, ...]
    if M.shape[-2:] != (H, W):
        M = F.interpolate(M, size=(H, W), mode="bilinear", align_corners=False)
    return M.expand(B, C, -1, -1)

@torch.no_grad()
def FlowEditSD3_DINO_Background(
    pipe,
    scheduler,
    retrieve_timesteps_fn,
    x_src: torch.Tensor,
    src_prompt: str,
    tar_prompt: str,
    negative_prompt: str,
    mask_fg: torch.Tensor, # Foreground mask (1 in FG, 0 in BG)
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 3.5,
    tar_guidance_scale: float = 13.5,
    n_min: int = 10,
    n_max: int = 45,
):
    device = x_src.device
    timesteps, T_steps = retrieve_timesteps_fn(scheduler, T_steps, device, timesteps=None)
    
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

    # We want to modify the background.
    # M_bg = 1 - mask_fg
    # V_final = M_fg * Vt_source + M_bg * Vt_target
    M_fg = _prep_mask_for_latents(mask_fg, x_src)
    M_bg = 1.0 - M_fg

    zt_edit = x_src.clone()

    for i, t in tqdm(list(enumerate(timesteps)), desc="FlowEdit_DINO_BG"):
        if T_steps - i > n_max:
            continue

        t_i = t / 1000
        t_im1 = timesteps[i + 1] / 1000 if i + 1 < len(timesteps) else torch.zeros_like(t_i)

        if T_steps - i > n_min:
            V_delta_avg = torch.zeros_like(x_src)
            for _ in range(n_avg):
                fwd_noise = torch.randn_like(x_src)
                zt_src = (1 - t_i) * x_src + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src
                latent_in = torch.cat([zt_src, zt_src, zt_tar, zt_tar], dim=0)

                Vt_src, Vt_tar = calc_v_sd3(
                    pipe, latent_in, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                    src_guidance_scale, tar_guidance_scale, t
                )
                
                # For background editing, we apply the delta ONLY to the background
                delta = M_bg * (Vt_tar - Vt_src)
                V_delta_avg += delta / float(n_avg)

            zt_edit = (zt_edit.to(torch.float32) + (t_im1 - t_i) * V_delta_avg.to(torch.float32)).to(dtype=x_src.dtype)
        else:
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src

            latent_in = torch.cat([xt_tar, xt_tar, xt_tar, xt_tar], dim=0)
            Vt_src, Vt_tar = calc_v_sd3(
                pipe, latent_in, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds,
                src_guidance_scale, tar_guidance_scale, t
            )
            
            # Blend velocities: Keep foreground as source, background as target
            V_final = M_fg * Vt_src + M_bg * Vt_tar
            xt_tar = (xt_tar.to(torch.float32) + (t_im1 - t_i) * V_final.to(torch.float32)).to(dtype=x_src.dtype)

    return (zt_edit if n_min == 0 else xt_tar)

# -----------------------------
# 3) Main Script
# -----------------------------

def preprocess_pil(image, size=512):
    image = image.resize((size, size), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 127.5 - 1
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device}, dtype: {dtype}")

    # First, get the mask and clear memory
    img_path = "FlowEdit/Data/Images/bear.png" 
    init_img = Image.open(img_path).convert("RGB")
    
    print("Generating Mask...")
    mask_fg = create_background_mask(init_img)
    plt.imsave("foreground_mask.png", mask_fg.numpy(), cmap="gray")
    print("Mask saved to foreground_mask.png")

    print("Loading SD3 Pipeline...")
    # Loading SD3 with RAM optimizations
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
        low_cpu_mem_usage=True,
        # SD3 has very large text encoders. If RAM is tight, we skip the heaviest (T5).
        text_encoder_3=None, 
        tokenizer_3=None,
    )
    
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cpu")
        # Enable RAM-saving optimizations
        pipe.enable_attention_slicing()
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

    scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = scheduler

    x_src = preprocess_pil(init_img, size=512).to(device, dtype=dtype)
    # Note: need to encode to latents
    def encode_image(pipe, x):
        pipe.vae.to(device)
        dist = pipe.vae.encode(x).latent_dist
        return dist.sample() * getattr(pipe.vae.config, "scaling_factor", 1.0)

    x_src_latents = encode_image(pipe, x_src)

    src_prompt = "A large brown bear walking through a stream of water."
    tar_prompt = "A large brown bear walking in a snowy forest during sunset."
    
    print("Running Masked FlowEdit for background modification...")
    edited_latents = FlowEditSD3_DINO_Background(
        pipe=pipe,
        scheduler=scheduler,
        retrieve_timesteps_fn=retrieve_timesteps,
        x_src=x_src_latents,
        src_prompt=src_prompt,
        tar_prompt=tar_prompt,
        negative_prompt="",
        mask_fg=mask_fg,
        T_steps=10,  # Reduced for CPU
        n_min=5,
        n_max=8,
    )

    def decode_image(pipe, latents):
        pipe.vae.to(device)
        z = latents / getattr(pipe.vae.config, "scaling_factor", 1.0)
        img = pipe.vae.decode(z).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        img = img.detach().cpu().permute(0, 2, 3, 1).numpy()
        return Image.fromarray((img * 255).round().astype(np.uint8)[0])

    out_img = decode_image(pipe, edited_latents)
    out_img.save("output_bg_modified.png")
    print("Result saved to output_bg_modified.png")
