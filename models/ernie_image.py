import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import torch
from torch import nn
import torch.nn.functional as F

from models.base import ComfyPipeline, make_contiguous
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift
from utils.offloading import ModelOffloader
import comfy.ldm.common_dit


class ErnieImagePipeline(ComfyPipeline):
    name = 'ernie_image'
    checkpointable_layers = ['InitialLayer', 'TransformerLayer']
    adapter_target_modules = ['ErnieImageSharedAdaLNBlock']
    keep_in_high_precision = ['x_embedder', 'text_proj', 'time_proj', 'time_embedding', 'adaLN_modulation', 'final_norm', 'final_linear']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)

    def to_layers(self):
        diffusion_model = self.diffusion_model
        layers = [InitialLayer(diffusion_model)]
        for i, block in enumerate(diffusion_model.layers):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(diffusion_model))
        return layers

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        latents = self.model_patcher.model.process_latent_in(latents)
        text_embeds = inputs['text_embeds_0']
        attention_mask = inputs['attention_mask_0']
        mask = inputs['mask']

        bs, c, h, w = latents.shape
        device = latents.device

        # text embeds are variable length
        max_seq_len = max([e.size(0) for e in text_embeds])
        text_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in text_embeds]
        )
        attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attention_mask]
        )
        attention_mask = attention_mask.bool()
        assert text_embeds.shape[:2] == attention_mask.shape[:2]

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension

        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')

        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=device))
        else:
            t = dist.sample((bs,)).to(device)

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
        target = noise - latents
        t = t*1000

        return (noisy_latents, t, text_embeds, attention_mask), (target, mask)

    def enable_block_swap(self, blocks_to_swap):
        diffusion_model = self.diffusion_model
        blocks = diffusion_model.layers
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        diffusion_model.layers = None
        diffusion_model.to('cuda')
        diffusion_model.layers = blocks
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


class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.x_embedder = model.x_embedder
        self.text_proj = model.text_proj
        self.pos_embed = model.pos_embed
        self.time_proj = model.time_proj
        self.time_embedding = model.time_embedding
        self.adaLN_modulation = model.adaLN_modulation
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, timesteps, context, context_mask = inputs

        device, dtype = x.device, x.dtype
        B, C, H, W = x.shape
        p, Hp, Wp = self.patch_size, H // self.patch_size, W // self.patch_size
        N_img = Hp * Wp

        img_bsh = self.x_embedder(x)

        text_bth = context
        if self.text_proj is not None and text_bth.numel() > 0:
            text_bth = self.text_proj(text_bth)
        Tmax = text_bth.shape[1]

        hidden_states = torch.cat([img_bsh, text_bth], dim=1)

        text_ids = torch.zeros((B, Tmax, 3), device=device, dtype=torch.float32)
        text_ids[:, :, 0] = torch.linspace(0, Tmax - 1, steps=Tmax, device=x.device, dtype=torch.float32)
        index = float(Tmax)

        h_len, w_len = float(Hp), float(Wp)
        h_offset, w_offset = 0.0, 0.0

        image_ids = torch.zeros((Hp, Wp, 3), device=device, dtype=torch.float32)
        image_ids[:, :, 0] = image_ids[:, :, 1] + index
        image_ids[:, :, 1] = image_ids[:, :, 1] + torch.linspace(h_offset, h_len - 1 + h_offset, steps=Hp, device=device, dtype=torch.float32).unsqueeze(1)
        image_ids[:, :, 2] = image_ids[:, :, 2] + torch.linspace(w_offset, w_len - 1 + w_offset, steps=Wp, device=device, dtype=torch.float32).unsqueeze(0)

        image_ids = image_ids.view(1, N_img, 3).expand(B, -1, -1)

        rotary_pos_emb = self.pos_embed(torch.cat([image_ids, text_ids], dim=1)).to(x.dtype)
        del image_ids, text_ids

        sample = self.time_proj(timesteps).to(dtype)
        c = self.time_embedding(sample)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            t.unsqueeze(1).contiguous() for t in self.adaLN_modulation(c).chunk(6, dim=-1)
        ]

        temb = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]

        attention_mask = torch.cat([torch.ones((B, N_img), device=device, dtype=torch.bool), context_mask], dim=1)[
            :, None, None, :
        ]

        sizes = torch.tensor([H, W, p, Hp, Wp, N_img], device=device)

        outputs = make_contiguous(hidden_states, attention_mask, rotary_pos_emb, c, sizes, *temb)
        # Needed to make reentrant activation checkpointing work, and also compile + non-reentrant (default) checkpointing.
        for item in outputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, layer, block_idx, offloader):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, attention_mask, rotary_pos_emb, c, sizes, *temb = inputs

        self.offloader.wait_for_block(self.block_idx)
        hidden_states = self.layer(hidden_states, rotary_pos_emb, temb, attention_mask=attention_mask)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(hidden_states, attention_mask, rotary_pos_emb, c, sizes, *temb)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_norm =model.final_norm
        self.final_linear = model.final_linear
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, attention_mask, rotary_pos_emb, c, sizes, *temb = inputs
        B = hidden_states.shape[0]
        H, W, p, Hp, Wp, N_img = sizes
        hidden_states = self.final_norm(hidden_states, c).type_as(hidden_states)
        patches = self.final_linear(hidden_states)[:, :N_img, :]
        output = (
            patches.view(B, Hp, Wp, p, p, self.out_channels)
            .permute(0, 5, 1, 3, 2, 4)
            .contiguous()
            .view(B, self.out_channels, H, W)
        )
        return output
