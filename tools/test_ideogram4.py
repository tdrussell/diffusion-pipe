import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import torch
import torchvision
import diffusers
from tqdm import tqdm

import comfy.utils
import comfy.sd
from comfy import model_management

torch.set_grad_enabled(False)

MODEL_PATH = '/data2/imagegen_models/comfyui-models/ideogram4_bf16.safetensors'
TEXT_ENCODER_PATH = '/data2/imagegen_models/comfyui-models/qwen3vl_8b_fp8_scaled.safetensors'
VAE_PATH = '/home/anon/ComfyUI/models/vae/flux2-vae.safetensors'
VAE_CHANNELS = 128
VAE_SPATIAL_COMPRESSION = 16

prompt = 'a golden retriever running through a grassy field'
negative_prompt = ''
w = 512
h = 512
sample_steps = 20
sample_cfg = 1
sample_shift = 3

# load vae
sd = comfy.utils.load_torch_file(VAE_PATH)
vae = comfy.sd.VAE(sd=sd)
vae.throw_exception_if_invalid()

# load clip
clip = comfy.sd.load_clip(ckpt_paths=[TEXT_ENCODER_PATH], clip_type=comfy.sd.CLIPType.IDEOGRAM4, disable_dynamic=True)

def get_text_embeddings(prompt):
    tokens = clip.tokenize(prompt)
    o = clip.encode_from_tokens_scheduled(tokens)
    text_embeds = o[0][0]
    extra = o[0][1]
    attention_mask = extra['attention_mask']
    return text_embeds.to('cuda'), attention_mask.to('cuda')

# compute conds
conds = get_text_embeddings(prompt)
if sample_cfg > 1:
    unconds = get_text_embeddings(negative_prompt)
model_management.unload_all_models()

# load diffusion model
dtype = torch.bfloat16
model_options = {}
model_options['dtype'] = dtype
model_patcher = comfy.sd.load_diffusion_model(MODEL_PATH, model_options=model_options, disable_dynamic=True)
model_patcher.set_model_compute_dtype(dtype)
with torch.no_grad():
    model_patcher.patch_model()
diffusion_model = model_patcher.model.diffusion_model

def call_model(inputs):
    x, t, context, attention_mask = inputs
    return diffusion_model(x, t, context=context, attention_mask=None)

# scheduler
scheduler = diffusers.FlowMatchEulerDiscreteScheduler(shift=sample_shift)
sigmas = torch.linspace(1.0, 1 / sample_steps, sample_steps)
scheduler.set_timesteps(sigmas=sigmas, device='cuda')

# sample
x = torch.randn((1, VAE_CHANNELS, h//VAE_SPATIAL_COMPRESSION, w//VAE_SPATIAL_COMPRESSION), device='cuda')
timesteps = scheduler.timesteps
for i, step in enumerate(tqdm(timesteps, desc='Sampling')):
    t = step / 1000
    t = t.float().view(1)
    print(t)
    inputs = (x, t, *conds)
    v = call_model(inputs).float()
    if sample_cfg > 1:
        inputs_uncond = (x, t, *unconds)
        v_uncond = call_model(inputs_uncond).float()
        v = v_uncond + sample_cfg*(v - v_uncond)
    x = scheduler.step(v, step, x, return_dict=False)[0]

# decode and save
img = vae.decode(x)
img = img.squeeze(0).movedim(-1, 0)
print(img.shape, img.min().item(), img.max().item())
torchvision.utils.save_image(img, 'example.png')
