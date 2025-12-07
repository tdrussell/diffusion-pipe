import os
import sys
from typing import List, Tuple
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from models.base import ComfyPipeline, PreprocessMediaFile, make_contiguous
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift
from utils.offloading import ModelOffloader
import comfy.ldm.common_dit

class Kandinsky5Pipeline(ComfyPipeline):
    name = 'kandinsky5' # Note: t2i and t2v/i2v model are both "video" models
    framerate = 24
    checkpointable_layers = ['TransformerEncoderBlock', 'TransformerDecoderBlock']
    adapter_target_modules = ['TransformerEncoderBlock', 'TransformerDecoderBlock']
    keep_in_high_precision = ['time_embeddings', 'text_embeddings', 'pooled_text_embeddings', 'visual_embeddings', 'out_layer']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.config = args[0]
        self.model_config = self.config['model']
        
        ckpt_path = self.model_config['diffusion_model']
        
        if "i2v" in ckpt_path:
            self.model_type = "i2v"
        elif "t2v" in ckpt_path:
            self.model_type = "t2v"
        elif "t2i" in ckpt_path:
            self.model_type = "t2i"
        else:
            raise NotImplementedError("Unsupported model type! Supported types: t2v, t2i, i2v")        
        
        self.patch_size=(1, 2, 2)

    def to_layers(self):
        diffusion_model = self.diffusion_model
        layers = [InitialLayer(diffusion_model)]
        for i, block in enumerate(diffusion_model.text_transformer_blocks):
            layers.append(TextTransformerLayer(block, i, self.offloader))
        layers.append(TextVisualTransitionalLayer(diffusion_model))
        for i, block in enumerate(diffusion_model.visual_transformer_blocks):
            layers.append(VisualTransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(diffusion_model))
        return layers

    def get_call_vae_fn(self, vae):
        is_i2v = self.model_type in ('i2v',)
        def fn(images):
            images = images.to('cuda')
            if len(images.shape) == 4:
                # [b, c, h, w] -> [b, h, w, c]
                images = torch.permute(images, (0, 2, 3, 1))
            else:
                # [b, c, t, h, w] -> [b, t, h, w, c]
                images = torch.permute(images, (0, 2, 3, 4, 1))
            
            # Pixels are in range [-1, 1], Comfy code expects [0, 1]
            images = (images + 1) / 2
            
            if len(images.shape) == 4:
                latents = vae.encode(images[:, :, :, :3])
                # Image is a "video" model
                images = images.unsqueeze(1)
            else:
                latents = vae.encode(images[:, :, :, :, :3])

            ret = {'latents': latents}
            
            if is_i2v:
                width = images[3]
                height = images[2]
                start_image= images[:, 0, ...]    
                start_image = comfy.utils.common_upscale(start_image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
                encoded = vae.encode(start_image[:, :, :, :3])

                mask = torch.ones((1, 1, latents.shape[2], latents.shape[-2], latents.shape[-1]), device=start_image.device, dtype=start_image.dtype)
                mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0
                
                ret["time_dim_replace"] = encoded
                ret["concat_mask"] = mask
            return ret
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        context = inputs['text_embeds'][0]
        y = inputs['text_embeds'][1][1] # Clip l, Pooled

        attention_mask = inputs['attention_mask']
        mask = inputs.get("mask", inputs.get("concat_mask", None))
        
        latents = comfy.ldm.common_dit.pad_to_patch_size(latents, (self.patch_size, self.patch_size))

        time_dim_replace = inputs.get('time_dim_replace', None)

        if time_dim_replace is not None:
            time_dim_replace = comfy.ldm.common_dit.pad_to_patch_size(time_dim_replace, self.patch_size)
            latents[:, :time_dim_replace.shape[1], :time_dim_replace.shape[2]] = time_dim_replace
        
        attention_mask = attention_mask.to(torch.bool)

        bs, c, t, h, w = latents.shape

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_t, img_h, img_w)
            mask = F.interpolate(mask, size=(t, h, w), mode='nearest-exact')  # resize to latent spatial dimension

        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')

        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=latents.device))
        else:
            t = dist.sample((bs,)).to(latents.device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('flux_shift', False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = latents - noise

        return (noisy_latents, t, context, y, attention_mask), (target, mask)

    def enable_block_swap(self, blocks_to_swap):
        diffusion_model = self.diffusion_model
        blocks_text = diffusion_model.text_transformer_blocks
        blocks_visual = diffusion_model.visual_transformer_blocks
        num_blocks = len(blocks_text) + len(blocks_visual)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerEncoderBlock/TransformerDecoderBlock', list(blocks_text) + list(blocks_visual), num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        ) # "block_type" is not used, can be anything
        diffusion_model.text_transformer_blocks = None
        diffusion_model.visual_transformer_blocks = None
        diffusion_model.to('cuda')
        diffusion_model.text_transformer_blocks = blocks_text
        diffusion_model.visual_transformer_blocks = blocks_visual
        self.prepare_block_swap_training()
        print(f'Block swap enabled. Swapping {blocks_to_swap} blocks out of {num_blocks} blocks.')

    def prepare_block_swap_training(self):
        self.offloader.enable_block_swap()
        self.offloader.set_forward_only(False)
        self.offloader.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap=False):
        if disable_block_swap:
            self.offloader.disable_block_swap()
        self.offloader.set_forward_only(True)
        self.offloader.prepare_block_devices_before_forward()

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(
            self.config,
            support_video=True,
            framerate=self.framerate,
        )

class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = [model]
        
    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        x, timesteps, context, y, attention_mask = inputs
        b, c, t_len, h, w = x.shape

        t = 1.0 - timesteps
        
        freqs = self.rope_encode_3d(t_len, h, w, device=x.device, dtype=x.dtype)
        freqs_text = self.rope_encode_1d(context.shape[1], device=x.device, dtype=x.dtype)
        
        context = self.text_embeddings(context)
        t = self.time_embeddings(t * 1000, x.dtype) + self.pooled_text_embeddings(y)

        return make_contiguous(x, t, context, y, attention_mask, freqs, freqs_text)

class TextTransformerLayer(nn.Module):
    def __init__(self, layer, block_idx, offloader):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, t, context, y, attention_mask, freqs, freqs_text = inputs

        self.offloader.wait_for_block(self.block_idx)
        context = self.layer(context, t, freqs_text)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(x, t, context, y, attention_mask, freqs, freqs_text)

class TextVisualTransitionalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, t, context, y, attention_mask, freqs, freqs_text = inputs
        
        x = self.visual_embeddings(x)
        visual_shape = torch.IntTensor(x.shape[:-1])
        x = x.flatten(1, -2)
        
        return make_contiguous(x, visual_shape, t, context, y, attention_mask, freqs, freqs_text)

class VisualTransformerLayer(nn.Module):
    def __init__(self, layer, block_idx, offloader):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, visual_shape, t, context, y, attention_mask, freqs, freqs_text = inputs

        self.offloader.wait_for_block(self.block_idx)
        x = self.layer(x, context, t, freqs=freqs)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(x, visual_shape, t, context, y, attention_mask, freqs, freqs_text)

class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, visual_shape, t, context, y, attention_mask, freqs, freqs_text = inputs
        x = x.reshape(*visual_shape.list(), -1)
        return self.out_layer(x, t)
