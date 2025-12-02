import os
import sys
from typing import List, Tuple
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from models.base import ComfyPipeline, make_contiguous
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift
from utils.offloading import ModelOffloader
import comfy.ldm.common_dit


class ZImagePipeline(ComfyPipeline):
    name = 'z_image'
    checkpointable_layers = ['TransformerLayer']
    # This will also train the noise_refiner and context_refiner layers, which aren't part of the main stack of transformer
    # layers, since they also use this class.
    adapter_target_modules = ['JointTransformerBlock']
    keep_in_high_precision = ['x_pad_token', 'cap_pad_token', 'x_embedder', 't_embedder', 'cap_embedder', 'final_layer']

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
        text_embeds = inputs['text_embeds']
        attention_mask = inputs['attention_mask']
        mask = inputs['mask']

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

        bs, c, h, w = latents.shape

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
        self.t_embedder = model.t_embedder
        self.cap_embedder = model.cap_embedder
        self.rope_embedder = model.rope_embedder
        self.context_refiner = model.context_refiner
        self.x_embedder = model.x_embedder
        self.noise_refiner = model.noise_refiner
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        x, timesteps, context, attention_mask = inputs

        t = 1.0 - timesteps
        cap_feats = context
        cap_mask = attention_mask
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))

        t = self.t_embedder(t * self.time_scale, dtype=x.dtype)  # (N, D)
        adaln_input = t

        cap_feats = self.cap_embedder(cap_feats)  # (N, L, D)  # todo check if able to batchify w.o. redundant compute

        x, mask, img_size, cap_size, freqs_cis = self.patchify_and_embed(x, cap_feats, cap_mask, t)
        img_size = torch.tensor(img_size).to(x.device)
        cap_size = torch.tensor(cap_size).to(x.device)
        freqs_cis = freqs_cis.to(x.device)
        return make_contiguous(x, mask, freqs_cis, adaln_input, img_size, cap_size)

    def patchify_and_embed(
        self, x: torch.Tensor, cap_feats: torch.Tensor, cap_mask: torch.Tensor, t: torch.Tensor, transformer_options={}
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]], List[int], torch.Tensor]:
        bsz = len(x)
        pH = pW = self.patch_size
        device = x.device

        if self.pad_tokens_multiple is not None:
            cap_feats_list = []
            for cap_feats_single, cap_mask_single in zip(cap_feats, cap_mask):
                cap_feats_single = cap_feats_single[cap_mask_single]
                pad_extra = (-cap_feats_single.shape[0]) % self.pad_tokens_multiple
                cap_feats_single = torch.cat((cap_feats_single, self.cap_pad_token.to(device=device, dtype=cap_feats.dtype, copy=True).repeat(pad_extra, 1)), dim=0)
                cap_feats_list.append(cap_feats_single)

            cap_item_seqlens = [len(_) for _ in cap_feats_list]
            assert all(_ % self.pad_tokens_multiple == 0 for _ in cap_item_seqlens)
            cap_max_item_seqlen = max(cap_item_seqlens)
            cap_feats = pad_sequence(cap_feats_list, batch_first=True, padding_value=0.0)
            cap_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
            for i, seq_len in enumerate(cap_item_seqlens):
                cap_mask[i, :seq_len] = 1

        cap_mask = cap_mask.view(bsz, 1, 1, -1)  # for PyTorch SDPA

        cap_pos_ids = torch.zeros(bsz, cap_feats.shape[1], 3, dtype=torch.float32, device=device)
        cap_pos_ids[:, :, 0] = torch.arange(cap_feats.shape[1], dtype=torch.float32, device=device) + 1.0

        B, C, H, W = x.shape
        x = self.x_embedder(x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2))

        H_tokens, W_tokens = H // pH, W // pW
        x_pos_ids = torch.zeros((bsz, x.shape[1], 3), dtype=torch.float32, device=device)
        x_pos_ids[:, :, 0] = cap_feats.shape[1] + 1
        x_pos_ids[:, :, 1] = torch.arange(H_tokens, dtype=torch.float32, device=device).view(-1, 1).repeat(1, W_tokens).flatten()
        x_pos_ids[:, :, 2] = torch.arange(W_tokens, dtype=torch.float32, device=device).view(1, -1).repeat(H_tokens, 1).flatten()

        if self.pad_tokens_multiple is not None:
            pad_extra = (-x.shape[1]) % self.pad_tokens_multiple
            x = torch.cat((x, self.x_pad_token.to(device=x.device, dtype=x.dtype, copy=True).unsqueeze(0).repeat(x.shape[0], pad_extra, 1)), dim=1)
            x_pos_ids = torch.nn.functional.pad(x_pos_ids, (0, 0, 0, pad_extra))

        freqs_cis = self.rope_embedder(torch.cat((cap_pos_ids, x_pos_ids), dim=1)).movedim(1, 2)

        # refine context
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_mask, freqs_cis[:, :cap_pos_ids.shape[1]], transformer_options=transformer_options)

        padded_img_mask = None
        for layer in self.noise_refiner:
            x = layer(x, padded_img_mask, freqs_cis[:, cap_pos_ids.shape[1]:], t, transformer_options=transformer_options)

        padded_full_embed = torch.cat((cap_feats, x), dim=1)
        mask = torch.cat((cap_mask, cap_mask.new_ones((bsz, 1, 1, x.shape[1]))), dim=-1)
        img_sizes = [(H, W)] * bsz
        l_effective_cap_len = [cap_feats.shape[1]] * bsz
        return padded_full_embed, mask, img_sizes, l_effective_cap_len, freqs_cis


class TransformerLayer(nn.Module):
    def __init__(self, layer, block_idx, offloader):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, mask, freqs_cis, adaln_input, img_size, cap_size = inputs

        self.offloader.wait_for_block(self.block_idx)
        x = self.layer(x, mask, freqs_cis, adaln_input)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(x, mask, freqs_cis, adaln_input, img_size, cap_size)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, mask, freqs_cis, adaln_input, img_size, cap_size = inputs
        img_size = [(row[0].item(), row[1].item()) for row in img_size]
        cap_size = [row.item() for row in cap_size]
        x = self.final_layer(x, adaln_input)
        h, w = img_size[0]  # same for all items in batch
        x = self.unpatchify(x, img_size, cap_size, return_tensor=True)[:,:,:h,:w]
        return x
