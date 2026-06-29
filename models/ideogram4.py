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
from comfy.text_encoders.llama import precompute_freqs_cis


SEQUENCE_PADDING_INDICATOR = -1
OUTPUT_IMAGE_INDICATOR = 2
LLM_TOKEN_INDICATOR = 3
IMAGE_POSITION_OFFSET = 65536


class Ideogram4Pipeline(ComfyPipeline):
    name = 'ideogram4'
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['Ideogram4TransformerBlock']
    keep_in_high_precision = ['input_proj', 'llm_cond_norm', 'llm_cond_proj', 't_embedding', 'adaln_proj', 'embed_image_indicator', 'final_layer', 'mlp_in']
    spatial_compression = 16
    channels = 128

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_format = comfy.latent_formats.Flux2()
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)

    def to_layers(self):
        diffusion_model = self.diffusion_model
        layers = [InitialLayer(diffusion_model)]
        for i, block in enumerate(diffusion_model.layers):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(diffusion_model))
        return layers

    # def to_layers(self):
    #     return [Wrapper(self.diffusion_model)]

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

        conds = self.get_conds(inputs)

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

        return (noisy_latents, t, *conds), (target, mask)

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
        self.input_proj = model.input_proj
        self.t_embedding = model.t_embedding
        self.adaln_proj = model.adaln_proj
        self.llm_cond_norm = model.llm_cond_norm
        self.llm_cond_proj = model.llm_cond_proj
        self.embed_image_indicator = model.embed_image_indicator
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    # Must NOT use autocast here or model output is so degraded it can't gen a coherent image.
    @torch.compiler.disable
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        x_chunk, timesteps, context_chunk, attn_mask_chunk = inputs
        t_chunk = 1.0 - timesteps
        bs, c, gh, gw = x_chunk.shape

        # This is only the conditional pathway
        B = x_chunk.shape[0]
        device = x_chunk.device
        img_tokens = self._img_to_tokens(x_chunk)
        L_img = img_tokens.shape[1]
        L_text = context_chunk.shape[1]
        L = L_text + L_img
        latent_dim = img_tokens.shape[-1]

        x_full = torch.zeros(B, L, latent_dim, dtype=img_tokens.dtype, device=device)
        x_full[:, L_text:] = img_tokens

        text_pos = torch.arange(L_text, device=device).view(-1, 1).expand(L_text, 3)
        img_pos = self._image_position_ids(gh, gw, device)
        position_ids = torch.cat([text_pos, img_pos], dim=0).unsqueeze(0).expand(B, L, 3)

        indicator = torch.empty(B, L, dtype=torch.long, device=device)
        indicator[:, :L_text] = LLM_TOKEN_INDICATOR
        indicator[:, L_text:] = OUTPUT_IMAGE_INDICATOR

        segment_ids = torch.ones(B, L, dtype=torch.long, device=device)
        pad = (attn_mask_chunk == 0)
        segment_ids[:, :L_text][pad] = SEQUENCE_PADDING_INDICATOR
        indicator[:, :L_text][pad] = 0
        # Block-diagonal mask from segment ids: (B, 1, L, L), True = attend.
        attn_mask = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)

        # backbone
        llm_features = context_chunk
        x = x_full
        t = t_chunk

        indicator = indicator.to(torch.long)
        output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(x.dtype).unsqueeze(-1)

        x = x * output_image_mask
        h = self.input_proj(x) * output_image_mask

        t_cond = self.t_embedding(t, dtype=x.dtype)
        if t.dim() == 1:
            t_cond = t_cond.unsqueeze(1)
        adaln_input = F.silu(self.adaln_proj(t_cond))

        # h is zero on the text rows (content lives only on image rows), add writes the text features in place
        if llm_features is not None:
            L_text = llm_features.shape[1]
            text_mask = (indicator[:, :L_text] == LLM_TOKEN_INDICATOR).to(x.dtype).unsqueeze(-1)
            llm = self.llm_cond_norm(llm_features * text_mask)
            llm = self.llm_cond_proj(llm) * text_mask
            h[:, :L_text] = h[:, :L_text] + llm

        h = h + self.embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long))

        # Qwen3-VL interleaved MRoPE; position_ids (B, L, 3) -> (3, L) (same across batch).
        freqs_cis = precompute_freqs_cis(
            self.head_dim, position_ids[0].transpose(0, 1), self.rope_theta,
            rope_dims=self.mrope_section, interleaved_mrope=True, device=position_ids.device,
        )

        if attn_mask.dtype == torch.bool:
            attn_mask = torch.zeros_like(attn_mask, dtype=h.dtype).masked_fill_(~attn_mask, -torch.finfo(h.dtype).max)

        sizes = torch.tensor([L_text, gh, gw], device=h.device)

        return make_contiguous(h, attn_mask, adaln_input, sizes, *freqs_cis)


class TransformerLayer(nn.Module):
    def __init__(self, layer, block_idx, offloader):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        h, attn_mask, adaln_input, sizes, *freqs_cis = inputs

        self.offloader.wait_for_block(self.block_idx)
        h = self.layer(h, attn_mask, freqs_cis, adaln_input)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(h, attn_mask, adaln_input, sizes, *freqs_cis)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    def forward(self, inputs):
        h, attn_mask, adaln_input, sizes, *freqs_cis = inputs
        out = self.final_layer(h, adaln_input)
        L_text, gh, gw = sizes
        out = -self._tokens_to_img(out[:, L_text:], gh, gw)
        return out


class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        x_chunk, timesteps, context, attn_mask = inputs
        out = self.model(x_chunk, timesteps, context=context, attention_mask=attn_mask)
        return out


# class NativeWrapper(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.ae_channels = 32
#         self.patch_size = 2

#     def forward(self, inputs):
#         x, timesteps, context, attn_mask = inputs
#         bs, c, gh, gw = x.shape
#         L_text = context.shape[1]

#         img_features, llm_features, position_ids, segment_ids, indicator = self._build_inputs(x, context, attn_mask)
#         t = 1 - timesteps

#         # for item in (llm_features, img_features, t, position_ids, segment_ids, indicator):
#         #     print(item.shape, item.dtype)
#         # quit()
#         out = -self.model(llm_features, img_features, t, position_ids, segment_ids, indicator)
#         return self._tokens_to_img(out[:, L_text:], gh, gw)

#     def _img_to_tokens(self, x):
#         B, C, gh, gw = x.shape
#         x = x.view(B, self.ae_channels, self.patch_size, self.patch_size, gh, gw)
#         x = x.permute(0, 4, 5, 2, 3, 1)  # (B, gh, gw, pi, pj, c)
#         return x.reshape(B, gh * gw, C)

#     def _tokens_to_img(self, tokens, gh, gw):
#         B = tokens.shape[0]
#         C = tokens.shape[-1]
#         x = tokens.reshape(B, gh, gw, self.patch_size, self.patch_size, self.ae_channels)
#         x = x.permute(0, 5, 3, 4, 1, 2)  # (B, c, pi, pj, gh, gw)
#         return x.reshape(B, C, gh, gw)

#     def _build_inputs(self, x, context, attn_mask):
#         batch_size, c, grid_h, grid_w = x.shape
#         device = x.device
#         num_image_tokens = grid_h * grid_w

#         max_text_tokens = context.shape[1]
#         total_seq_len = max_text_tokens + num_image_tokens

#         # Image position ids (t=0, h, w) offset to keep them disjoint from text positions.
#         h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
#         w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
#         t_idx = torch.zeros_like(h_idx)
#         image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

#         position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long, device=device)

#         segment_ids = torch.full(
#             (batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long, device=device
#         )
#         indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long, device=device)
#         llm_features = torch.zeros(batch_size, total_seq_len, context.shape[-1], dtype=context.dtype, device=device)
#         img_features = torch.zeros(batch_size, total_seq_len, c, dtype=x.dtype, device=device)
#         x = self._img_to_tokens(x)

#         for b, (img_feat, txt_feat) in enumerate(zip(x, context)):
#             num_text = attn_mask[b].sum().item() if attn_mask is not None else max_text_tokens
#             pad_len = max_text_tokens - num_text
#             total_unpadded = num_text + num_image_tokens

#             # Layout: [pad_len zeros] [text tokens] [image tokens]
#             offset = pad_len
#             # Image token slots stay at 0.

#             text_pos = torch.arange(num_text)
#             text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
#             position_ids[b, offset : offset + num_text] = text_pos_3d
#             position_ids[b, offset + num_text :] = image_pos

#             indicator[b, offset : offset + num_text] = LLM_TOKEN_INDICATOR
#             indicator[b, offset + num_text :] = OUTPUT_IMAGE_INDICATOR

#             # Segment id 1 for the (text+image) sample, padding stays at 0.
#             segment_ids[b, offset : offset + total_unpadded] = 1

#             llm_features[b, offset : offset + num_text] = txt_feat[:num_text]
#             img_features[b, offset + num_text :] = img_feat

#         return img_features, llm_features, position_ids, segment_ids, indicator