import math

import diffusers
import torch
from torch import nn
import torch.nn.functional as F

from models.base import BasePipeline, make_contiguous
from utils.common import AUTOCAST_DTYPE


KEEP_IN_HIGH_PRECISION = ['pos_embed', 'time_text_embed', 'context_embedder', 'norm_out', 'proj_out']


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


class SD3Pipeline(BasePipeline):
    name = 'sd3'
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['JointTransformerBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        dtype = self.model_config['dtype']

        if diffusers_path := self.model_config.get('diffusers_path', None):
            self.diffusers_pipeline = diffusers.StableDiffusion3Pipeline.from_pretrained(diffusers_path, torch_dtype=dtype, transformer=None)
        else:
            raise NotImplementedError()

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)
        if diffusers_path := self.model_config.get('diffusers_path', None):
            transformer = diffusers.SD3Transformer2DModel.from_pretrained(diffusers_path, torch_dtype=dtype, subfolder='transformer')
        else:
            raise NotImplementedError()

        for name, p in transformer.named_parameters():
            if not (any(x in name for x in KEEP_IN_HIGH_PRECISION) or p.ndim == 1):
                p.data = p.data.to(transformer_dtype)

        self.diffusers_pipeline.transformer = transformer

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder, self.text_encoder_2, self.text_encoder_3]

    def save_adapter(self, save_dir, peft_state_dict):
        self.save_lora_weights(save_dir, transformer_lora_layers=peft_state_dict)

    def save_model(self, save_dir, diffusers_sd):
        raise NotImplementedError()

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            tensor = tensor.to(vae.device, vae.dtype)
            tensor = tensor*2 - 1
            latents = vae.encode(tensor).latent_dist.sample()
            if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                latents = latents - vae.config.shift_factor
            latents = latents * vae.config.scaling_factor
            return {'latents': latents}
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        if text_encoder == self.text_encoder:
            def fn(caption, is_video):
                assert not any(is_video)
                prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                    prompt=caption,
                    device=text_encoder.device,
                    clip_model_index=0,
                )
                return {'prompt_embed': prompt_embed, 'pooled_prompt_embed': pooled_prompt_embed}
            return fn
        elif text_encoder == self.text_encoder_2:
            def fn(caption, is_video):
                assert not any(is_video)
                prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                    prompt=caption,
                    device=text_encoder.device,
                    clip_model_index=1,
                )
                return {'prompt_2_embed': prompt_2_embed, 'pooled_prompt_2_embed': pooled_prompt_2_embed}
            return fn
        elif text_encoder == self.text_encoder_3:
            def fn(caption, is_video):
                assert not any(is_video)
                return {'t5_prompt_embed': self._get_t5_prompt_embeds(prompt=caption, device=text_encoder.device)}
            return fn
        else:
            raise RuntimeError(f'Text encoder {text_encoder.__class__} does not have a function to call it')

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        prompt_embed = inputs['prompt_embed']
        pooled_prompt_embed = inputs['pooled_prompt_embed']
        prompt_2_embed = inputs['prompt_2_embed']
        pooled_prompt_2_embed = inputs['pooled_prompt_2_embed']
        t5_prompt_embed = inputs['t5_prompt_embed']
        mask = inputs['mask']

        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

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
        target = noise - latents

        return (noisy_latents, t * 1000, prompt_embeds, pooled_prompt_embeds), (target, mask)

    def to_layers(self):
        transformer = self.transformer
        layers = [InitialLayer(transformer)]
        for block in transformer.transformer_blocks:
            layers.append(TransformerLayer(block))
        layers.append(FinalLayer(transformer))
        return layers


class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.pos_embed = model.pos_embed
        self.time_text_embed = model.time_text_embed
        self.context_embedder = model.context_embedder
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        hidden_states, timestep, encoder_hidden_states, pooled_projections = inputs

        height, width = hidden_states.shape[-2:]
        latent_size = torch.tensor([height, width]).to(hidden_states.device)

        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        return make_contiguous(hidden_states, temb, latent_size, encoder_hidden_states)


class TransformerLayer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, temb, latent_size, *extra = inputs
        encoder_hidden_states = extra[0] if len(extra) > 0 else None
        encoder_hidden_states, hidden_states = self.block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
        )
        result = make_contiguous(hidden_states, temb, latent_size)
        if encoder_hidden_states is not None:
            result += (encoder_hidden_states,)
        return result


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.norm_out = model.norm_out
        self.proj_out = model.proj_out
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, temb, latent_size, *_ = inputs
        height = latent_size[0].item()
        width = latent_size[1].item()

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )
        return output
