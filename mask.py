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


def prompt_2_img_i2i_fast(prompts, init_img, g=7.5, seed=100, strength =0.5, steps=50, dim=512):
    """
    Diffusion process to convert prompt to image
    """
    # Converting textual prompts to embedding
    text = text_enc(prompts) 
    
    # Adding an unconditional prompt , helps in the generation process
    uncond =  text_enc([""], text.shape[1])
    emb = torch.cat([uncond, text])
    
    # Setting the seed
    if seed: torch.manual_seed(seed)
    
    # Setting number of steps in scheduler
    scheduler.set_timesteps(steps)
    
    # Convert the seed image to latent
    init_latents = pil_to_latents(init_img)
    
    # Figuring initial time step based on strength
    init_timestep = int(steps * strength) 
    timesteps = scheduler.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps], device="cuda")
    
    # Adding noise to the latents 
    noise = torch.randn(init_latents.shape, generator=None, device="cuda", dtype=init_latents.dtype)
    init_latents = scheduler.add_noise(init_latents, noise, timesteps)
    latents = init_latents
    
    # We need to scale the i/p latents to match the variance
    inp = scheduler.scale_model_input(torch.cat([latents] * 2), timesteps)
    # Predicting noise residual using U-Net
    with torch.no_grad(): u,t = unet(inp, timesteps, encoder_hidden_states=emb).sample.chunk(2)
         
    # Performing Guidance
    pred = u + g*(t-u)

    # Zero shot prediction
    latents = scheduler.step(pred, timesteps, latents).pred_original_sample
    
    return latents.detach().cpu()


def create_mask(init_img, rp, qp, n=20, s=0.5):
    diff = {}
    
    for idx in range(n):
        orig_noise = prompt_2_img_i2i_fast(prompts=rp, init_img=init_img, strength=s, seed = 100*idx)[0]
        query_noise = prompt_2_img_i2i_fast(prompts=qp, init_img=init_img, strength=s, seed = 100*idx)[0]
        diff[idx] = (np.array(orig_noise)-np.array(query_noise))
    
    mask = np.zeros_like(diff[0])
    
    for idx in range(n):
        mask += np.abs(diff[idx])  
        
    mask = mask.mean(0)
    mask = (mask - mask.mean()) / np.std(mask)
    
    return (mask > 0).astype("uint8")

if __name__ == "__main__":

    path_img = "C:\\Users\\JALAL\\OneDrive\\Documents\\FocusFlow\\FlowEdit\\example_images\\bear.png"
    init_img = load_image(path_img)

    rp = "A brown bear walking through a stream of water."

    qp = "A brown bear standing on water"

    print("creating mask...")
    mask_bin = create_mask(init_img=init_img, rp=[rp], qp=[qp], n=10)

    plt.imshow(init_img)
    plt.imshow(
        Image.fromarray((mask_bin * 255).astype('uint8')).resize((512,512)),
        cmap='cividis', 
        alpha=0.9*(np.array(Image.fromarray((mask_bin * 255).astype('uint8')).resize((512,512))) > 0)  
    )
    plt.axis('off')
    plt.show()