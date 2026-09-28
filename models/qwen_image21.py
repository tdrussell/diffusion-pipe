import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import torch
from torch import nn
import torch.nn.functional as F

from models.base import ComfyPipeline, make_contiguous
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift
from utils.offloading import ModelOffloader
import comfy.latent_formats
from comfy.ldm.qwen_image21.model import block_causal_attention


def _split_rows(p):
    # shared modulation rows: (t = 0 row for text and references, sampled-t rows for the target)
    return p[-1:].unsqueeze(1), p[:-1].unsqueeze(1)


def pack_segments(segments):
    endpoints = torch.tensor([row[:2] for row in segments])
    masks = [row[2] for row in segments]
    masks = [m if m is not None else torch.tensor([], dtype=bool) for m in masks]
    return endpoints, masks


def unpack_segments(endpoints, masks):
    endpoints = endpoints.tolist()
    segments = []
    for ((start, end), mask) in zip(endpoints, masks):
        if mask.numel() == 0:
            mask = None
        segments.append((start, end, mask))
    return segments


class QwenImage21Pipeline(ComfyPipeline):
    name = 'qwen_image21'
    spatial_compression = 16
    channels = 64
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['QwenImage21TransformerBlock']
    keep_in_high_precision = ['img_in', 'txt_in', 'time_text_embed', 'modulation', 'norm_out', 'proj_out']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_format = comfy.latent_formats.QwenImage21()
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)

    def to_layers(self):
        diffusion_model = self.diffusion_model
        transformer_options = {
            'total_blocks': len(diffusion_model.transformer_blocks),
            'block_type': 'single',
        }
        layers = [InitialLayer(diffusion_model)]
        for i, block in enumerate(diffusion_model.transformer_blocks):
            layers.append(TransformerLayer(block, i, transformer_options, self.offloader))
        layers.append(FinalLayer(diffusion_model))
        return layers

    def get_conds(self, inputs):
        text_embeds = inputs['text_embeds_0']
        attention_mask = inputs['attention_mask_0']
        # text embeds are variable length
        max_seq_len = max([e.size(0) for e in text_embeds])
        text_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in text_embeds]
        )
        attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attention_mask]
        )
        assert text_embeds.shape[:2] == attention_mask.shape[:2]
        attention_mask = attention_mask.to(torch.bool)
        return text_embeds, attention_mask

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']

        bs, c, h, w = latents.shape
        device = latents.device

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

        return (noisy_latents, t, *self.get_conds(inputs)), (target, mask)

    def enable_block_swap(self, blocks_to_swap):
        diffusion_model = self.diffusion_model
        blocks = diffusion_model.transformer_blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        diffusion_model.transformer_blocks = None
        diffusion_model.to('cuda')
        diffusion_model.transformer_blocks = blocks
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
        self.img_in = model.img_in
        self.txt_in = model.txt_in
        self.time_text_embed = model.time_text_embed
        self.modulation = model.modulation
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    def build_sequence(self, x, context, context_mask, ref_latents, image_slots):
        # text with each reference image spliced in at its slot, target image last
        assert len(ref_latents) == 0 and len(image_slots) == 0

        bs = x.shape[0]
        txt = self.txt_in(context)
        txt_len = txt.shape[1]
        slots = (image_slots + [txt_len] * len(ref_latents))[:len(ref_latents)]
        bounds = [0] + slots + [txt_len]

        parts, ids, segments = [], [], []
        pos, length = 0, 0
        for (start, end), img in zip(zip(bounds[:-1], bounds[1:]), ref_latents + [x]):
            n = end - start
            if n > 0:
                parts.append(txt[:, start:end])
                ids.append(torch.arange(pos, pos + n, device=x.device, dtype=torch.float32).unsqueeze(1).expand(n, 3))
                # Different from source. Exactly 1 text segment, so we can do: 1D attention mask -> expand to n x n -> tril().
                # Thus, padding tokens are masked + attention is causal.
                mask = context_mask[:, None, None, :].expand(bs, 1, n, n).tril()
                segments.append((length, length + n, mask))
                pos += n
                length += n
            h, w = img.shape[-2:]
            parts.append(self.img_in(img.flatten(2).transpose(1, 2)))
            # half a token where a reference grid has the other parity, so it centres on the target
            hh = torch.arange(h, device=x.device, dtype=torch.float32) - (h - h // 2) + 0.5 * (h % 2 - x.shape[-2] % 2)
            ww = torch.arange(w, device=x.device, dtype=torch.float32) - (w - w // 2) + 0.5 * (w % 2 - x.shape[-1] % 2)
            ids.append(torch.stack([torch.full((h, w), pos, device=x.device, dtype=torch.float32), hh[:, None].expand(h, w), ww[None, :].expand(h, w)], dim=-1).flatten(0, 1))
            # Different from source. Image tokens full attend to themselves, but must only attend to non-padding tokens in text.
            mask = torch.ones((bs, 1, 1, txt_len + h*w), dtype=torch.bool, device=x.device)
            mask[:, 0, 0, :txt_len] = context_mask
            segments.append((length, length + h * w, mask))
            pos += max(h, w)
            length += h * w

        # (1, N, 1, ...): the layout the fused rms_rope wants for (B, N, H, D) queries
        pe = self.pe_embedder(torch.cat(ids, dim=0).unsqueeze(0)).transpose(1, 2).contiguous()
        return torch.cat(parts, dim=1), pe, segments

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        x, timesteps, context, context_mask = inputs

        B, C, H, W = x.shape
        dtype = x.dtype
        ref_latents = []
        image_slots = []

        hidden_states, pe, segments = self.build_sequence(x, context, context_mask, ref_latents, image_slots)
        sizes = torch.tensor([hidden_states.shape[1], H, W])

        # pipeline rounds t*1000 and t to the compute dtype; text and reference tokens modulate from t = 0
        t = ((timesteps * 1000).to(dtype) / 1000).to(dtype)
        temb = self.time_text_embed(torch.cat([t, t.new_zeros(1)]), dtype)
        scale1, gate1, scale2, gate2 = self.modulation(temb).chunk(4, dim=-1)

        mod_scale1_a, mod_scale1_b = _split_rows(scale1)
        mod_gate1_a, mod_gate1_b = _split_rows(gate1.tanh())
        mod_scale2_a, mod_scale2_b = _split_rows(scale2)
        mod_gate2_a, mod_gate2_b = _split_rows(gate2.tanh())
        mod_zeros = torch.zeros_like(scale1[:1, None])

        endpoints, masks = pack_segments(segments)

        return make_contiguous(hidden_states, temb, pe, mod_scale1_a, mod_scale1_b, mod_gate1_a, mod_gate1_b, mod_scale2_a, mod_scale2_b, mod_gate2_a, mod_gate2_b, mod_zeros, sizes, endpoints, *masks)


class TransformerLayer(nn.Module):
    def __init__(self, layer, block_idx, transformer_options, offloader):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        transformer_options['block_index'] = block_idx
        self.transformer_options = transformer_options
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, temb, pe, mod_scale1_a, mod_scale1_b, mod_gate1_a, mod_gate1_b, mod_scale2_a, mod_scale2_b, mod_gate2_a, mod_gate2_b, mod_zeros, sizes, endpoints, *masks = inputs
        mod = ((mod_scale1_a, mod_scale1_b), (mod_gate1_a, mod_gate1_b), (mod_scale2_a, mod_scale2_b), (mod_gate2_a, mod_gate2_b), mod_zeros)
        seq_len, H, W = sizes
        prefix_len = seq_len - H*W
        segments = unpack_segments(endpoints, masks)

        attn_fn = block_causal_attention(segments, self.transformer_options, None, self.block_idx, prefix_len)

        self.offloader.wait_for_block(self.block_idx)
        hidden_states = self.layer(hidden_states, mod, pe, attn_fn, prefix_len, self.transformer_options)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(hidden_states, temb, pe, mod_scale1_a, mod_scale1_b, mod_gate1_a, mod_gate1_b, mod_scale2_a, mod_scale2_b, mod_gate2_a, mod_gate2_b, mod_zeros, sizes, endpoints, *masks)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.norm_out = model.norm_out
        self.proj_out = model.proj_out
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    @torch.compiler.disable
    def forward(self, inputs):
        hidden_states, temb, pe, mod_scale1_a, mod_scale1_b, mod_gate1_a, mod_gate1_b, mod_scale2_a, mod_scale2_b, mod_gate2_a, mod_gate2_b, mod_zeros, sizes, endpoints, *masks = inputs
        seq_len, H, W = sizes
        prefix_len = seq_len - H*W
        B = hidden_states.shape[0]
        hidden_states = self.norm_out(hidden_states[:, prefix_len:], temb[:-1])
        hidden_states = self.proj_out(hidden_states)
        return hidden_states.transpose(1, 2).reshape(B, self.out_channels, H, W)
