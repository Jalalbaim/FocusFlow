import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms as tfms

import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import shutil
import os
from base64 import b64encode


## Import the CLIP artifacts 
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

def load_artifacts():
    '''
    A function to load all diffusion artifacts
    '''
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16).to("cuda")
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16).to("cuda")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to("cuda")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)    
    return vae, unet, tokenizer, text_encoder, scheduler

vae, unet, tokenizer, text_encoder, scheduler = load_artifacts()

def load_image(p):
    '''
    Function to load images from a defined path
    '''
    return Image.open(p).convert('RGB').resize((512,512))

def pil_to_latents(image):
    '''
    Function to convert image to latents
    '''
    init_image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
    init_image = init_image.to(device="cuda", dtype=torch.float16) 
    init_latent_dist = vae.encode(init_image).latent_dist.sample() * 0.18215
    return init_latent_dist

def latents_to_pil(latents):
    '''
    Function to convert latents to images
    '''
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def text_enc(prompts, maxlen=None):
    '''
    A function to take a texual promt and convert it into embeddings
    '''
    if maxlen is None: maxlen = tokenizer.model_max_length
    inp = tokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt") 
    return text_encoder(inp.input_ids.to("cuda"))[0].half()

@torch.no_grad()
def predict_eps_cfg(unet, scheduler, latents_t, t, cond_emb, uncond_emb, guidance_scale: float):
    """
    Return the *guided* noise estimate epsilon_hat at (latents_t, t) using CFG.
    Assumes the UNet output corresponds to the scheduler's prediction type used for sampling.
    (Most SD-style setups are epsilon-prediction; if not, you must convert accordingly.)
    """
    # 2x batch for CFG
    latent_in = torch.cat([latents_t, latents_t], dim=0)
    latent_in = scheduler.scale_model_input(latent_in, t)

    emb = torch.cat([uncond_emb, cond_emb], dim=0)
    eps_uncond, eps_cond = unet(latent_in, t, encoder_hidden_states=emb).sample.chunk(2, dim=0)

    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    return eps  # shape: (1, C, H, W)


@torch.no_grad()
def create_diffedit_mask(
    init_img_pil,
    ref_prompt: str,
    query_prompt: str,
    text_enc,
    pil_to_latents,
    unet,
    scheduler,
    steps: int = 50,
    noise_strength: float = 0.5,     # "50% noise" per paper
    n: int = 10,                     # paper default
    guidance_scale: float = 5.0,     # paper default in ImageNet ablation
    clip_q: float = 0.99,            # percentile clipping for "remove extreme values" (not specified in paper)
    threshold: float = 0.5,          # paper default after rescale to [0,1]
    device: str = "cuda",
):
    """
    Returns:
      mask_soft: float tensor in [0,1], shape (H, W) in latent resolution
      mask_bin : uint8 tensor {0,1}, shape (H, W)
    """

    # --- Encode prompts ---
    cond_ref = text_enc([ref_prompt]).to(device)
    cond_q   = text_enc([query_prompt]).to(device)
    uncond   = text_enc([""], cond_ref.shape[1]).to(device)  # matches your usage

    # --- Scheduler timesteps ---
    scheduler.set_timesteps(steps, device=device)

    # Map noise_strength to an img2img-style starting timestep
    init_timestep = int(steps * noise_strength)
    init_timestep = max(1, min(init_timestep, steps))
    t_start = steps - init_timestep
    t = scheduler.timesteps[t_start]
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=device)
    if t.ndim == 0:
        t = t[None]  # shape (1,)

    # --- Latents of input image ---
    latents0 = pil_to_latents(init_img_pil).to(device)  # shape (1, C, H, W)

    # --- Accumulate spatial difference over n noise draws ---
    acc = torch.zeros((latents0.shape[-2], latents0.shape[-1]), device=device)

    for i in range(n):
        # different noise each time (paper: average over n input noises)
        gen = torch.Generator(device=device).manual_seed(100 * i)
        noise = torch.randn(latents0.shape, generator=gen, device=device, dtype=latents0.dtype)

        latents_t = scheduler.add_noise(latents0, noise, t)

        eps_ref = predict_eps_cfg(unet, scheduler, latents_t, t, cond_ref, uncond, guidance_scale)
        eps_q   = predict_eps_cfg(unet, scheduler, latents_t, t, cond_q,   uncond, guidance_scale)

        # spatial diff map (paper doesn't specify norm; abs-mean is a standard choice)
        diff_map = (eps_ref - eps_q).abs().mean(dim=1).squeeze(0)  # (H, W)

        acc += diff_map

    mask = acc / float(n)

    # --- "remove extreme values" (paper mentions it but doesn't specify how) ---
    # Percentile clipping (explicit + stable).
    hi = torch.quantile(mask.flatten(), clip_q)
    mask = torch.clamp(mask, max=hi)

    # --- Rescale to [0,1] then binarize at 0.5 (paper default) ---
    mmin, mmax = mask.min(), mask.max()
    mask_soft = (mask - mmin) / (mmax - mmin + 1e-8)
    mask_bin = (mask_soft >= threshold).to(torch.uint8)

    return mask_soft.detach().cpu(), mask_bin.detach().cpu()


if __name__ == "__main__":

    path_img = "C:\\Users\\JALAL\\OneDrive\\Documents\\FocusFlow\\FlowEdit\\example_images\\bear.png"
    init_img = load_image(path_img)

    rp = "A  brown bear "

    qp = "A  panda  "

    mask_soft, mask_bin = create_diffedit_mask(
            init_img_pil=init_img,
            ref_prompt=rp,
            query_prompt=qp,
            text_enc=text_enc,
            pil_to_latents=pil_to_latents,
            unet=unet,
            scheduler=scheduler
         )


    plt.imshow(init_img)
    plt.imshow(
        Image.fromarray((mask_bin.numpy() * 255).astype('uint8')).resize((512,512)),
        cmap='cividis', 
        alpha=0.9*(np.array(Image.fromarray((mask_bin.numpy() * 255).astype('uint8')).resize((512,512))) > 0)  
    )
    plt.axis('off')
    plt.show()