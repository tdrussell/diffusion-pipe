import os
import sys
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import torch
from torch import nn
import torch.nn.functional as F

from models.base import ComfyPipeline, make_contiguous, PreprocessMediaFile
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift
from utils.offloading import ModelOffloader
import comfy.ldm.common_dit
from comfy.ldm.flux.layers import timestep_embedding


class HunyuanVideo15Pipeline(ComfyPipeline):
    name = 'hunyuan_video_1.5'
    framerate = 24
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['DoubleStreamBlock']
    keep_in_high_precision = ['img_in', 'time_in', 'txt_in', 'byt5_in', 'final_layer', 'vision_in', 'cond_type_embedding']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(self.config, support_video=True, framerate=self.framerate)

    def get_call_text_encoder_fn(self, text_encoder):
        te_idx = None
        for i, te in enumerate(self.text_encoders):
            if text_encoder == te:
                te_idx = i
                break
        if te_idx is None:
            raise RuntimeError('Unknown text encoder')

        def fn(captions: list[str], is_video: list[bool]):
            bs = len(captions)
            tokenizer = getattr(text_encoder.tokenizer, text_encoder.tokenizer.clip)
            byt5_tokenizer = getattr(text_encoder.tokenizer, 'byt5')

            max_length = defaultdict(int)
            byt5_lengths = []
            for text in captions:
                tokens = text_encoder.tokenize(text)
                # tokens looks like {'qwen3_4b': [[(0, 1.0), (1, 1.0), (2, 1.0)]]}
                for k, v in tokens.items():
                    max_length[k] = max(max_length[k], len(v[0]))
                if 'byt5' in tokens:
                    byt5_lengths.append(len(tokens['byt5'][0]))
                else:
                    byt5_lengths.append(0)

            # Pad to max length in the batch. We need to do this ourselves or the ComfyUI backend code will fail (it concats tensors assumed to be the same length).
            tokenizer.min_length = max_length[text_encoder.tokenizer.clip]
            byt5_tokenizer.min_length = max_length['byt5']
            tokens_dict = defaultdict(list)
            for text in captions:
                tokens = text_encoder.tokenize(text)
                for k, v in tokens.items():
                    tokens_dict[k].extend(v)

            o = text_encoder.encode_from_tokens_scheduled(tokens_dict)

            text_embeds = o[0][0]
            extra = o[0][1]
            if 'attention_mask' in extra:
                attention_mask = extra['attention_mask'].to(torch.bool)
            else:
                # ComfyUI code removes attention_mask if it's all 1s
                attention_mask = torch.ones((bs, text_embeds.shape[1]), dtype=torch.bool)

            if 'conditioning_byt5small' in extra:
                raw_byt5_embeds = extra['conditioning_byt5small']
                byt5_embeds = torch.zeros((bs, raw_byt5_embeds.shape[1], 1472), dtype=raw_byt5_embeds.dtype)
            else:
                byt5_embeds = torch.zeros((bs, 0, 1472), dtype=text_embeds.dtype)
                assert sum(byt5_lengths) == 0

            byt5_attention_mask = torch.zeros((bs, max(byt5_lengths)), dtype=torch.bool)
            idx = 0
            for i, length in enumerate(byt5_lengths):
                if length > 0:
                    byt5_embeds[i, :length, :] = raw_byt5_embeds[idx, :length, :]
                    byt5_attention_mask[i, :length] = True
                    idx += 1

            return {
                'text_embeds': text_embeds,
                'attention_mask': attention_mask,
                'byt5_embeds': byt5_embeds,
                'byt5_attention_mask': byt5_attention_mask,
            }

        return fn

    def to_layers(self):
        diffusion_model = self.diffusion_model
        layers = [InitialLayer(diffusion_model)]
        for i, block in enumerate(diffusion_model.double_blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(diffusion_model))
        return layers

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        latents = self.model_patcher.model.process_latent_in(latents)
        text_embeds = inputs['text_embeds']
        attention_mask = inputs['attention_mask']
        byt5_embeds = inputs['byt5_embeds']
        byt5_attention_mask = inputs['byt5_attention_mask']
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

        max_seq_len = max([e.size(0) for e in byt5_embeds])
        byt5_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in byt5_embeds]
        )
        byt5_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in byt5_attention_mask]
        )
        assert byt5_embeds.shape[:2] == byt5_attention_mask.shape[:2]

        bs, c, f, h, w = latents.shape
        dtype = latents.dtype
        device = latents.device

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)  # make mask same number of dims as target

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
        t_expanded = t.view(-1, 1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents

        cond_latents_concat = torch.zeros(bs, c, f, h, w, dtype=dtype, device=device)
        mask_concat = torch.zeros(bs, 1, f, h, w, dtype=dtype, device=device)
        latent_model_input = torch.cat([noisy_latents, cond_latents_concat, mask_concat], dim=1)

        return (latent_model_input, t*1000, text_embeds, attention_mask, byt5_embeds, byt5_attention_mask), (target, mask)

    def enable_block_swap(self, blocks_to_swap):
        diffusion_model = self.diffusion_model
        blocks = diffusion_model.double_blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        diffusion_model.double_blocks = None
        diffusion_model.to('cuda')
        diffusion_model.double_blocks = blocks
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
        if model.params.guidance_embed:
            raise NotImplementedError('Training guidance distilled model is not supported.')
        assert model.vector_in is None
        self.img_in = model.img_in
        self.time_in = model.time_in
        self.txt_in = model.txt_in
        self.cond_type_embedding = model.cond_type_embedding
        self.byt5_in = model.byt5_in
        self.vision_in = model.vision_in  # not used?
        self.pe_embedder = model.pe_embedder
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        x, timesteps, context, attention_mask, context_byt5, attention_mask_byt5 = inputs

        bs = x.shape[0]
        if len(self.patch_size) == 3:
            img_ids = self.img_ids(x)
            txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        else:
            img_ids = self.img_ids_2d(x)
            txt_ids = torch.zeros((bs, context.shape[1], 2), device=x.device, dtype=x.dtype)

        img = x
        txt = context
        txt_mask = attention_mask
        txt_byt5 = context_byt5
        txt_byt5_mask = attention_mask_byt5

        initial_shape = torch.tensor(img.shape, device=img.device)
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256, time_factor=1.0).to(img.dtype))

        assert txt_mask.dtype == torch.bool
        assert txt_byt5_mask.dtype == torch.bool

        txt = self.txt_in(txt, timesteps, txt_mask)

        if self.cond_type_embedding is not None:
            self.cond_type_embedding.to(txt.device)
            cond_emb = self.cond_type_embedding(torch.zeros_like(txt[:, :, 0], device=txt.device, dtype=torch.long))
            txt = txt + cond_emb.to(txt.dtype)

        if self.byt5_in is not None and txt_byt5 is not None:
            txt_byt5 = self.byt5_in(txt_byt5)
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(torch.ones_like(txt_byt5[:, :, 0], device=txt_byt5.device, dtype=torch.long))
                txt_byt5 = txt_byt5 + cond_emb.to(txt_byt5.dtype)
                txt = torch.cat((txt_byt5, txt), dim=1) # byt5 first for HunyuanVideo1.5
                txt_mask = torch.cat((txt_byt5_mask, txt_mask), dim=1)
            else:
                txt = torch.cat((txt, txt_byt5), dim=1)
                txt_mask = torch.cat((txt_mask, txt_byt5_mask), dim=1)
            txt_byt5_ids = torch.zeros((txt_ids.shape[0], txt_byt5.shape[1], txt_ids.shape[-1]), device=txt_ids.device, dtype=txt_ids.dtype)
            txt_ids = torch.cat((txt_ids, txt_byt5_ids), dim=1)

        ids = torch.cat((img_ids, txt_ids), dim=1)
        pe = self.pe_embedder(ids)
        pe.requires_grad_(True)

        img_len = img.shape[1]

        attn_mask_len = img_len + txt.shape[1]
        attn_mask = torch.ones((bs, 1, 1, attn_mask_len), dtype=torch.bool, device=img.device)
        attn_mask[:, 0, 0, img_len:] = txt_mask

        return make_contiguous(img, txt, vec, pe, attn_mask, initial_shape)


class TransformerLayer(nn.Module):
    def __init__(self, layer, block_idx, offloader):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        img, txt, vec, pe, attn_mask, initial_shape = inputs

        self.offloader.wait_for_block(self.block_idx)
        img, txt = self.layer(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims_img=None, modulation_dims_txt=None)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(img, txt, vec, pe, attn_mask, initial_shape)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    @torch.compiler.disable
    def forward(self, inputs):
        img, txt, vec, pe, attn_mask, initial_shape = inputs
        initial_shape = initial_shape.tolist()
        img = self.final_layer(img, vec, modulation_dims=None)  # (N, T, patch_size ** 2 * out_channels)

        shape = initial_shape[-len(self.patch_size):]
        for i in range(len(shape)):
            shape[i] = shape[i] // self.patch_size[i]
        img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
        if img.ndim == 8:
            img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
            img = img.reshape(initial_shape[0], self.out_channels, initial_shape[2], initial_shape[3], initial_shape[4])
        else:
            img = img.permute(0, 3, 1, 4, 2, 5)
            img = img.reshape(initial_shape[0], self.out_channels, initial_shape[2], initial_shape[3])
        return img
