import math
import os.path
from functools import partial

import diffusers
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from deepspeed.utils.logging import logger
from safetensors import safe_open
from safetensors.torch import save_file

from models.base import BasePipeline, make_contiguous
from utils.common import AUTOCAST_DTYPE, is_main_process
from utils.offloading import ModelOffloader

NUM_DOUBLE_BLOCKS = 19
NUM_SINGLE_BLOCKS = 38

BFL_TO_DIFFUSERS_MAP = {
    "time_in.in_layer.weight": ["time_text_embed.timestep_embedder.linear_1.weight"],
    "time_in.in_layer.bias": ["time_text_embed.timestep_embedder.linear_1.bias"],
    "time_in.out_layer.weight": ["time_text_embed.timestep_embedder.linear_2.weight"],
    "time_in.out_layer.bias": ["time_text_embed.timestep_embedder.linear_2.bias"],
    "vector_in.in_layer.weight": ["time_text_embed.text_embedder.linear_1.weight"],
    "vector_in.in_layer.bias": ["time_text_embed.text_embedder.linear_1.bias"],
    "vector_in.out_layer.weight": ["time_text_embed.text_embedder.linear_2.weight"],
    "vector_in.out_layer.bias": ["time_text_embed.text_embedder.linear_2.bias"],
    "guidance_in.in_layer.weight": ["time_text_embed.guidance_embedder.linear_1.weight"],
    "guidance_in.in_layer.bias": ["time_text_embed.guidance_embedder.linear_1.bias"],
    "guidance_in.out_layer.weight": ["time_text_embed.guidance_embedder.linear_2.weight"],
    "guidance_in.out_layer.bias": ["time_text_embed.guidance_embedder.linear_2.bias"],
    "txt_in.weight": ["context_embedder.weight"],
    "txt_in.bias": ["context_embedder.bias"],
    "img_in.weight": ["x_embedder.weight"],
    "img_in.bias": ["x_embedder.bias"],
    "double_blocks.().img_mod.lin.weight": ["norm1.linear.weight"],
    "double_blocks.().img_mod.lin.bias": ["norm1.linear.bias"],
    "double_blocks.().txt_mod.lin.weight": ["norm1_context.linear.weight"],
    "double_blocks.().txt_mod.lin.bias": ["norm1_context.linear.bias"],
    "double_blocks.().img_attn.qkv.weight": ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"],
    "double_blocks.().img_attn.qkv.bias": ["attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias"],
    "double_blocks.().txt_attn.qkv.weight": ["attn.add_q_proj.weight", "attn.add_k_proj.weight", "attn.add_v_proj.weight"],
    "double_blocks.().txt_attn.qkv.bias": ["attn.add_q_proj.bias", "attn.add_k_proj.bias", "attn.add_v_proj.bias"],
    "double_blocks.().img_attn.norm.query_norm.scale": ["attn.norm_q.weight"],
    "double_blocks.().img_attn.norm.key_norm.scale": ["attn.norm_k.weight"],
    "double_blocks.().txt_attn.norm.query_norm.scale": ["attn.norm_added_q.weight"],
    "double_blocks.().txt_attn.norm.key_norm.scale": ["attn.norm_added_k.weight"],
    "double_blocks.().img_mlp.0.weight": ["ff.net.0.proj.weight"],
    "double_blocks.().img_mlp.0.bias": ["ff.net.0.proj.bias"],
    "double_blocks.().img_mlp.2.weight": ["ff.net.2.weight"],
    "double_blocks.().img_mlp.2.bias": ["ff.net.2.bias"],
    "double_blocks.().txt_mlp.0.weight": ["ff_context.net.0.proj.weight"],
    "double_blocks.().txt_mlp.0.bias": ["ff_context.net.0.proj.bias"],
    "double_blocks.().txt_mlp.2.weight": ["ff_context.net.2.weight"],
    "double_blocks.().txt_mlp.2.bias": ["ff_context.net.2.bias"],
    "double_blocks.().img_attn.proj.weight": ["attn.to_out.0.weight"],
    "double_blocks.().img_attn.proj.bias": ["attn.to_out.0.bias"],
    "double_blocks.().txt_attn.proj.weight": ["attn.to_add_out.weight"],
    "double_blocks.().txt_attn.proj.bias": ["attn.to_add_out.bias"],
    "single_blocks.().modulation.lin.weight": ["norm.linear.weight"],
    "single_blocks.().modulation.lin.bias": ["norm.linear.bias"],
    "single_blocks.().linear1.weight": ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight", "proj_mlp.weight"],
    "single_blocks.().linear1.bias": ["attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias", "proj_mlp.bias"],
    "single_blocks.().linear2.weight": ["proj_out.weight"],
    "single_blocks.().norm.query_norm.scale": ["attn.norm_q.weight"],
    "single_blocks.().norm.key_norm.scale": ["attn.norm_k.weight"],
    "single_blocks.().linear2.weight": ["proj_out.weight"],
    "single_blocks.().linear2.bias": ["proj_out.bias"],
    "final_layer.linear.weight": ["proj_out.weight"],
    "final_layer.linear.bias": ["proj_out.bias"],
    "final_layer.adaLN_modulation.1.weight": ["norm_out.linear.weight"],
    "final_layer.adaLN_modulation.1.bias": ["norm_out.linear.bias"],
}


KEEP_IN_HIGH_PRECISION = ['norm', 'bias', 'time_text_embed', 'context_embedder', 'x_embedder']


def make_diffusers_to_bfl_map(num_double_blocks: int = NUM_DOUBLE_BLOCKS, num_single_blocks: int = NUM_SINGLE_BLOCKS) -> dict[str, tuple[int, str]]:
    # make reverse map from diffusers map
    diffusers_to_bfl_map = {}  # key: diffusers_key, value: (index, bfl_key)
    for b in range(num_double_blocks):
        for key, weights in BFL_TO_DIFFUSERS_MAP.items():
            if key.startswith("double_blocks."):
                block_prefix = f"transformer_blocks.{b}."
                for i, weight in enumerate(weights):
                    diffusers_to_bfl_map[f"{block_prefix}{weight}"] = (i, key.replace("()", f"{b}"))
    for b in range(num_single_blocks):
        for key, weights in BFL_TO_DIFFUSERS_MAP.items():
            if key.startswith("single_blocks."):
                block_prefix = f"single_transformer_blocks.{b}."
                for i, weight in enumerate(weights):
                    diffusers_to_bfl_map[f"{block_prefix}{weight}"] = (i, key.replace("()", f"{b}"))
    for key, weights in BFL_TO_DIFFUSERS_MAP.items():
        if not (key.startswith("double_blocks.") or key.startswith("single_blocks.")):
            for i, weight in enumerate(weights):
                diffusers_to_bfl_map[weight] = (i, key)
    return diffusers_to_bfl_map


def is_dev(safetensors_path):
    with safe_open(safetensors_path, framework='pt', device='cpu') as f:
        for key in f.keys():
            if key.startswith('guidance_in'):
                return True
    return False


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def guidance_embed_bypass_forward(self, timestep, guidance, pooled_projection):
    timesteps_proj = self.time_proj(timestep)
    timesteps_emb = self.timestep_embedder(
        timesteps_proj.to(dtype=pooled_projection.dtype))  # (N, D)
    pooled_projections = self.text_embedder(pooled_projection)
    conditioning = timesteps_emb + pooled_projections
    return conditioning


def bypass_flux_guidance(transformer):
    if hasattr(transformer.time_text_embed, '_bfg_orig_forward'):
        return
    # dont bypass if it doesnt have the guidance embedding
    if not hasattr(transformer.time_text_embed, 'guidance_embedder'):
        return
    transformer.time_text_embed._bfg_orig_forward = transformer.time_text_embed.forward
    transformer.time_text_embed.forward = partial(
        guidance_embed_bypass_forward, transformer.time_text_embed
    )


class FluxPipeline(BasePipeline):
    # Unique name, used to make the cache_dir path.
    name = 'flux'

    # layers that will participate in activation checkpointing
    checkpointable_layers = [
        'TransformerWrapper',
        'SingleTransformerWrapper',
    ]

    adapter_target_modules = ['FluxTransformerBlock', 'FluxSingleTransformerBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.offloader_double = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.offloader_single = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)

        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        if transformer_path := self.model_config.get('transformer_path', None):
            if is_main_process():
                print(f'Overriding transformer using {transformer_path}')
            transformer_config = 'configs/flux_dev_config.json' if is_dev(transformer_path) else 'configs/flux_schnell_config.json'
            transformer = diffusers.FluxTransformer2DModel.from_single_file(
                transformer_path,
                torch_dtype=dtype,
                config=transformer_config,
                local_files_only=True,
            )
        else:
            transformer = diffusers.FluxTransformer2DModel.from_pretrained(
                os.path.join(self.model_config['diffusers_path'], 'transformer'),
                torch_dtype=dtype,
            )
        if self.model_config.get('bypass_guidance_embedding', False):
            if is_main_process():
                print('Bypassing Flux guidance')
            bypass_flux_guidance(transformer)

        for name, p in transformer.named_parameters():
            if not (any(x in name for x in KEEP_IN_HIGH_PRECISION) or name.startswith('proj_out')):
                p.data = p.data.to(transformer_dtype)

        self.diffusers_pipeline = diffusers.FluxPipeline.from_pretrained(self.model_config['diffusers_path'], torch_dtype=dtype, transformer=transformer)

        self.transformer.train()
        # We'll need the original parameter name for saving, and the name changes once we wrap modules for pipeline parallelism,
        # so store it in an attribute here. Same thing below if we're training a lora and creating lora weights.
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder, self.text_encoder_2]

    def save_adapter(self, save_dir, peft_state_dict):
        adapter_type = self.config['adapter']['type']
        if adapter_type == 'lora':
            self.save_lora_weights(save_dir, transformer_lora_layers=peft_state_dict)
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')

    def save_model(self, save_dir, diffusers_sd):
        diffusers_to_bfl_map = make_diffusers_to_bfl_map()

        # iterate over three safetensors files to reduce memory usage
        flux_sd = {}
        for diffusers_key, tensor in diffusers_sd.items():
            if diffusers_key in diffusers_to_bfl_map:
                index, bfl_key = diffusers_to_bfl_map[diffusers_key]
                if bfl_key not in flux_sd:
                    flux_sd[bfl_key] = []
                flux_sd[bfl_key].append((index, tensor))
            else:
                logger.error(f"Error: Key not found in diffusers_to_bfl_map: {diffusers_key}")
                raise KeyError(f"Key not found in diffusers_to_bfl_map: {diffusers_key}")

        # concat tensors if multiple tensors are mapped to a single key, sort by index
        for key, values in flux_sd.items():
            if len(values) == 1:
                flux_sd[key] = values[0][1]
            else:
                flux_sd[key] = torch.cat([value[1] for value in sorted(values, key=lambda x: x[0])])

        # special case for final_layer.adaLN_modulation.1.weight and final_layer.adaLN_modulation.1.bias
        def swap_scale_shift(weight):
            shift, scale = weight.chunk(2, dim=0)
            new_weight = torch.cat([scale, shift], dim=0)
            return new_weight

        if "final_layer.adaLN_modulation.1.weight" in flux_sd:
            flux_sd["final_layer.adaLN_modulation.1.weight"] = swap_scale_shift(flux_sd["final_layer.adaLN_modulation.1.weight"])
        if "final_layer.adaLN_modulation.1.bias" in flux_sd:
            flux_sd["final_layer.adaLN_modulation.1.bias"] = swap_scale_shift(flux_sd["final_layer.adaLN_modulation.1.bias"])

        save_file(flux_sd, save_dir / 'model.safetensors', metadata={"format": "pt"})

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            latents = vae.encode(tensor.to(vae.device, vae.dtype)).latent_dist.sample()
            if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                latents = latents - vae.config.shift_factor
            latents = latents * vae.config.scaling_factor
            return {'latents': latents}
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        if text_encoder == self.text_encoder:
            def fn(caption, is_video):
                # args are lists
                assert not any(is_video)
                return {'clip_embed': self._get_clip_prompt_embeds(prompt=caption, device=text_encoder.device)}
            return fn
        elif text_encoder == self.text_encoder_2:
            def fn(caption, is_video):
                # args are lists
                assert not any(is_video)
                return {'t5_embed': self._get_t5_prompt_embeds(prompt=caption, device=text_encoder.device)}
            return fn
        else:
            raise RuntimeError(f'Text encoder {text_encoder.__class__} does not have a function to call it')

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        clip_embed = inputs['clip_embed']
        t5_embed = inputs['t5_embed']
        mask = inputs['mask']

        # The following code taken and slightly modified from x-flux (https://github.com/XLabs-AI/x-flux/tree/main)
        bs, c, h, w = latents.shape
        latents = rearrange(latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        img_ids = self._prepare_latent_image_ids(bs, h // 2, w // 2, latents.device, latents.dtype)
        if img_ids.ndim == 2:
            # This method must return tensors with batch dimension, since we proceed to split along batch dimension for pipelining.
            img_ids = img_ids.unsqueeze(0).repeat((bs, 1, 1))
        txt_ids = torch.zeros(bs, t5_embed.shape[1], 3).to(latents.device, latents.dtype)

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

        x_1 = latents
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1)
        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
        target = x_0 - x_1
        guidance_vec = torch.full((x_t.shape[0],), float(self.model_config['guidance']), device=x_t.device, dtype=torch.float32)

        features = (x_t, t5_embed, clip_embed, t, img_ids, txt_ids, guidance_vec), (target, mask)

        # We pass the target through the layers of the model in the features tuple, so that it matches the noisy input when we get to the
        # last pipeline parallel stage.
        return features

    def to_layers(self):
        transformer = self.transformer
        layers = [EmbeddingWrapper(transformer.x_embedder, transformer.time_text_embed, transformer.context_embedder, transformer.pos_embed)]
        for i, block in enumerate(transformer.transformer_blocks):
            layers.append(TransformerWrapper(block, i, self.offloader_double))
        layers.append(concatenate_hidden_states)
        for i, block in enumerate(transformer.single_transformer_blocks):
            layers.append(SingleTransformerWrapper(block, i, self.offloader_single))
        layers.append(OutputWrapper(transformer.norm_out, transformer.proj_out))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        transformer = self.transformer
        double_blocks = transformer.transformer_blocks
        single_blocks = transformer.single_transformer_blocks
        num_double_blocks = len(double_blocks)
        num_single_blocks = len(single_blocks)
        double_blocks_to_swap = blocks_to_swap // 2
        # This swaps more than blocks_to_swap total blocks. A bit odd, but the model does have twice as many
        # single blocks as double. I'm just replicating the behavior of Musubi Tuner.
        single_blocks_to_swap = (blocks_to_swap - double_blocks_to_swap) * 2 + 1

        assert double_blocks_to_swap <= num_double_blocks - 2 and single_blocks_to_swap <= num_single_blocks - 2, (
            f'Cannot swap more than {num_double_blocks - 2} double blocks and {num_single_blocks - 2} single blocks. '
            f'Requested {double_blocks_to_swap} double blocks and {single_blocks_to_swap} single blocks.'
        )

        self.offloader_double = ModelOffloader(
            'DoubleBlock', double_blocks, num_double_blocks, double_blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        self.offloader_single = ModelOffloader(
            'SingleBlock', single_blocks, num_single_blocks, single_blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        transformer.transformer_blocks = None
        transformer.single_transformer_blocks = None
        transformer.to('cuda')
        transformer.transformer_blocks = double_blocks
        transformer.single_transformer_blocks = single_blocks
        self.prepare_block_swap_training()
        print(
            f'Block swap enabled. Swapping {blocks_to_swap} blocks, double blocks: {double_blocks_to_swap}, single blocks: {single_blocks_to_swap}.'
        )

    def prepare_block_swap_training(self):
        self.offloader_double.enable_block_swap()
        self.offloader_double.set_forward_only(False)
        self.offloader_double.prepare_block_devices_before_forward()
        self.offloader_single.enable_block_swap()
        self.offloader_single.set_forward_only(False)
        self.offloader_single.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap=False):
        if disable_block_swap:
            self.offloader_double.disable_block_swap()
            self.offloader_single.disable_block_swap()
        self.offloader_double.set_forward_only(True)
        self.offloader_double.prepare_block_devices_before_forward()
        self.offloader_single.set_forward_only(True)
        self.offloader_single.prepare_block_devices_before_forward()


class EmbeddingWrapper(nn.Module):
    def __init__(self, x_embedder, time_text_embed, context_embedder, pos_embed):
        super().__init__()
        self.x_embedder = x_embedder
        self.time_text_embed = time_text_embed
        self.context_embedder = context_embedder
        self.pos_embed = pos_embed

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        # Don't know why I have to do this. I had to do it in qlora-pipe also.
        # Without it, you get RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids, guidance = inputs
        hidden_states = self.x_embedder(hidden_states)
        timestep = timestep.to(hidden_states.dtype) * 1000
        guidance = guidance.to(hidden_states.dtype) * 1000
        # handle dev vs schnell
        if self.time_text_embed.__class__.__name__ == 'CombinedTimestepGuidanceTextProjEmbeddings':
            temb = self.time_text_embed(timestep, guidance, pooled_projections)
        else:
            temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        ids = torch.cat((txt_ids, img_ids), dim=0)
        freqs_cos, freqs_sin = self.pos_embed(ids)
        return make_contiguous(hidden_states, encoder_hidden_states, temb, freqs_cos, freqs_sin)


class TransformerWrapper(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, temb, freqs_cos, freqs_sin = inputs

        self.offloader.wait_for_block(self.block_idx)
        encoder_hidden_states, hidden_states = self.block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=(freqs_cos, freqs_sin),
        )
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(hidden_states, encoder_hidden_states, temb, freqs_cos, freqs_sin)


def concatenate_hidden_states(inputs):
    hidden_states, encoder_hidden_states, temb, freqs_cos, freqs_sin = inputs
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    return hidden_states, encoder_hidden_states, temb, freqs_cos, freqs_sin


class SingleTransformerWrapper(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, temb, freqs_cos, freqs_sin = inputs

        self.offloader.wait_for_block(self.block_idx)
        hidden_states = self.block(
            hidden_states=hidden_states,
            temb=temb,
            image_rotary_emb=(freqs_cos, freqs_sin),
        )
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(hidden_states, encoder_hidden_states, temb, freqs_cos, freqs_sin)


class OutputWrapper(nn.Module):
    def __init__(self, norm_out, proj_out):
        super().__init__()
        self.norm_out = norm_out
        self.proj_out = proj_out

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, temb, freqs_cos, freqs_sin = inputs
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)
