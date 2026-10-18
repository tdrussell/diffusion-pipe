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
from comfy.ldm.flux.layers import timestep_embedding
import comfy.ldm.moss_image.model
from comfy.ldm.moss_image.model import modulated_rms_norm, rope_split_half, situ, gated_residual
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention, ComfyAttention
import comfy.model_prefetch


def attention(q, k, v, q_norm, k_norm, pe, heads, comfy_attention, transformer_options, mask=None):
    B, N, _ = q.shape
    dtype = v.dtype
    q = q.view(B, N, heads, -1)
    k = k.view(B, N, heads, -1)
    if comfy.model_management.in_training:
        q, k = rope_split_half(q_norm(q), pe).to(dtype), rope_split_half(k_norm(k), pe).to(dtype)
    else:
        q_scale, _, q_stream = comfy.ops.cast_bias_weight(q_norm, q, offloadable=True)
        k_scale, _, k_stream = comfy.ops.cast_bias_weight(k_norm, k, offloadable=True)
        q, k = comfy.quant_ops.ck.rms_rope_split_half(q, k, pe, q_scale, k_scale, q_norm.eps)
        comfy.ops.uncast_bias_weight(q_norm, q_scale, None, q_stream)
        comfy.ops.uncast_bias_weight(k_norm, k_scale, None, k_stream)
    q, k, v = AttentionTensorContainer(q.flatten(2)), AttentionTensorContainer(k.flatten(2)), AttentionTensorContainer(v)
    return optimized_attention(q, k, v, heads, mask=mask, transformer_options=transformer_options, preferred_attention=comfy_attention)

# Support mask. Cast Q, K (float32) to V's dtype.
comfy.ldm.moss_image.model.attention = attention


class TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.num_heads = num_heads
        self.norm1 = operations.RMSNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = operations.RMSNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.q_norm = operations.RMSNorm(dim // num_heads, eps=1e-6, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(dim // num_heads, eps=1e-6, dtype=dtype, device=device)
        self.qkv = operations.Linear(dim, 3 * dim, dtype=dtype, device=device)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.ffn_gate = operations.Linear(dim, ffn_dim, dtype=dtype, device=device)
        self.ffn_up = operations.Linear(dim, ffn_dim, dtype=dtype, device=device)
        self.ffn_down = operations.Linear(ffn_dim, dim, dtype=dtype, device=device)

    def forward(self, x, pe, mask, transformer_options={}):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        x = x + self.proj(attention(q, k, v, self.q_norm, self.k_norm, pe, self.num_heads, self.comfy_attention, transformer_options, mask=mask))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))

comfy.ldm.moss_image.model.TextBlock = TextBlock


class TextEmbedder(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, num_heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Linear(text_dim, dim, dtype=dtype, device=device)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, num_heads, dtype=dtype, device=device, operations=operations) for _ in range(2)])

    def forward(self, x, pe, mask, transformer_options={}):
        x = self.proj(x)
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), x.device, transformer_options)
        for block in self.blocks:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, block, x.dtype)
            x = block(x, pe, mask, transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, x.device, None)
        return x

comfy.ldm.moss_image.model.TextEmbedder = TextEmbedder


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, fused_qkv=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.num_heads = num_heads
        self.fused_qkv = fused_qkv
        self.norm1 = operations.RMSNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = operations.RMSNorm(dim, eps=1e-6, dtype=dtype, device=device)
        self.q_norm = operations.RMSNorm(dim // num_heads, eps=1e-6, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(dim // num_heads, eps=1e-6, dtype=dtype, device=device)
        # repacked files fuse q/k/v: one GEMM and one activation quantization instead of three
        names = ("qkv", "o") if fused_qkv else ("q", "k", "v", "o")
        self.proj = nn.ModuleDict({name: operations.Linear(dim, 3 * dim if name == "qkv" else dim, bias=False, dtype=dtype, device=device) for name in names})
        self.proj["ffn_up_gate"] = operations.Linear(dim, 2 * ffn_dim, bias=False, dtype=dtype, device=device)
        self.proj["ffn_down"] = operations.Linear(ffn_dim, dim, bias=False, dtype=dtype, device=device)

    def forward(self, x, mod, pe, mask, transformer_options={}):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod
        y = modulated_rms_norm(x, self.norm1, scale_a, shift_a)
        if self.fused_qkv:
            q, k, v = self.proj["qkv"](y).chunk(3, dim=-1)
        else:
            q, k, v = self.proj["q"](y), self.proj["k"](y), self.proj["v"](y)
        out = attention(q, k, v, self.q_norm, self.k_norm, pe, self.num_heads, self.comfy_attention, transformer_options, mask=mask)
        x = gated_residual(x, self.proj["o"](out), gate_a)
        y = modulated_rms_norm(x, self.norm2, scale_f, shift_f)
        hidden, gate = self.proj["ffn_up_gate"](y).chunk(2, dim=-1)
        # situ without its 4 * 25 factor, applied on the gate instead (ffn_down has no bias): two fewer passes
        out = self.proj["ffn_down"](torch.tanh(gate / 4.0) * torch.sigmoid(gate) * torch.tanh(hidden / 25.0))
        return gated_residual(x, out, gate_f * 100.0)

comfy.ldm.moss_image.model.Block = Block


class MossImagePipeline(ComfyPipeline):
    name = 'moss_image'
    spatial_compression = 16
    channels = 128
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['Block', 'TextBlock']
    keep_in_high_precision = ['shared_block_modulation', 'text_embedder', 'time_embedding', 'video_embedder', 'final_linear_video', 'final_norm_video', 'rope.bands', 'video_head_modulation']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_format = comfy.latent_formats.Flux2()
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)

    def to_layers(self):
        diffusion_model = self.diffusion_model
        transformer_options = {
            'total_blocks': len(diffusion_model.layers),
            'block_type': 'single',
        }
        layers = [InitialLayer(diffusion_model)]
        for i, block in enumerate(diffusion_model.layers):
            layers.append(TransformerLayer(block, i, transformer_options, self.offloader))
        layers.append(FinalLayer(diffusion_model))
        return layers

    def patch_quantized_modules(self, model):
        num_patched_classes = super().patch_quantized_modules(model)
        if num_patched_classes > 0:
            raise NotImplementedError(f'Training LoRA on quantized weights will not work for MOSS-Image (training will not converge; training loss goes up sharply)')
        return 0

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
        self.time_embeddings = model.time_embedding
        self.shared_block_modulation = model.shared_block_modulation
        self.video_head_modulation = model.video_head_modulation
        self.text_embedder = model.text_embedder
        self.video_embedder = model.video_embedder
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        x, timestep, context, context_mask = inputs

        B, _, H, W = x.shape
        dtype, device = x.dtype, x.device
        pe = self.rope(H, W, context.shape[1], device)

        vec = self.time_embedding(timestep_embedding(timestep, 256).to(dtype))
        mod = self.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
        head_shift, head_scale = self.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1)

        text_mask = context_mask[:, None, None, :]
        txt = self.text_embedder(context, pe[:, H * W:], text_mask)
        # allocated outside the memory compiler region, the blocks update it in place
        x = torch.cat((self.video_embedder(x.flatten(2).transpose(1, 2)), txt), dim=1)
        del txt
        sizes = torch.tensor([H, W], dtype=torch.int32, device=device)

        img_len = H*W
        sequence_mask = torch.cat([
            torch.ones((B, 1, 1, img_len), dtype=bool, device=device),
            text_mask,
        ], dim=-1)
        assert sequence_mask.shape[-1] == x.shape[1]

        return make_contiguous(x, pe, sequence_mask, head_shift, head_scale, sizes, *mod)


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
        x, pe, sequence_mask, head_shift, head_scale, sizes, *mod = inputs

        self.offloader.wait_for_block(self.block_idx)
        x = self.layer(x, mod, pe, sequence_mask, self.transformer_options)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(x, pe, sequence_mask, head_shift, head_scale, sizes, *mod)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_linear_video = model.final_linear_video
        self.final_norm_video = model.final_norm_video
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    @torch.compiler.disable
    def forward(self, inputs):
        x, pe, sequence_mask, head_shift, head_scale, sizes, *mod = inputs
        B = x.shape[0]
        H, W = sizes.tolist()
        out = self.final_linear_video(modulated_rms_norm(x[:, :H * W], self.final_norm_video, head_scale, head_shift))
        return out.transpose(1, 2).reshape(B, self.latent_channels, H, W)

