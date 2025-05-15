from pathlib import Path
import os.path
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/LTX_Video'))

import random
import safetensors
import torch
from torch import nn
import torch.nn.functional as F

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from utils.common import AUTOCAST_DTYPE
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline as OriginalLTXVideoPipeline
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.models.autoencoders.vae_encode import vae_encode


KEEP_IN_HIGH_PRECISION = ['norm', 'bias', 'scale_shift_table', 'patchify_proj', 'proj_out', 'adaln_single', 'caption_projection']


class LTXVideoPipeline(BasePipeline):
    name = 'ltx-video'
    framerate = 25
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['BasicTransformerBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        dtype = self.model_config['dtype']

        diffusers_path = self.model_config['diffusers_path']
        single_file_path = Path(self.model_config['single_file_path'])

        # The VAE could be different for each model version, so we have to make sure to use different cache directories.
        self.name = single_file_path.stem

        vae = CausalVideoAutoencoder.from_pretrained(single_file_path)
        self.diffusers_pipeline = OriginalLTXVideoPipeline.from_pretrained(
            diffusers_path,
            transformer=None,
            vae=vae,
            patchifier=SymmetricPatchifier(patch_size=1),
            prompt_enhancer_image_caption_model=None,
            prompt_enhancer_image_caption_processor=None,
            prompt_enhancer_llm_model=None,
            prompt_enhancer_llm_tokenizer=None,
            torch_dtype=dtype,
        )

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def load_diffusion_model(self):
        single_file_path = self.model_config['single_file_path']
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        transformer = Transformer3DModel.from_pretrained(single_file_path, torch_dtype=dtype)
        for name, p in transformer.named_parameters():
            if not (any(x in name for x in KEEP_IN_HIGH_PRECISION)):
                p.data = p.data.to(transformer_dtype)

        transformer.train()
        for name, p in transformer.named_parameters():
            p.original_name = name
        self.diffusers_pipeline.transformer = transformer

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder]

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI format
        peft_state_dict = {'diffusion_model.'+k: v for k, v in peft_state_dict.items()}
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def save_model(self, save_dir, diffusers_sd):
        raise NotImplementedError()

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(
            self.config,
            support_video=True,
            framerate=self.framerate,
            round_height=32,
            round_width=32,
            round_frames=8,
        )

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            latents = vae_encode(
                tensor.to(dtype=vae.dtype, device=vae.device),
                vae,
                vae_per_channel_normalize=True,
            )
            return {'latents': latents}
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(caption, is_video):
            # args are lists
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self.encode_prompt(caption, do_classifier_free_guidance=False, device=text_encoder.device)
            return {'prompt_embeds': prompt_embeds, 'prompt_attention_mask': prompt_attention_mask}
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        prompt_embeds = inputs['prompt_embeds']
        prompt_attention_mask = inputs['prompt_attention_mask']
        mask = inputs['mask']

        bs, channels, num_frames, height, width = latents.shape

        temporal_downscale = self.vae.temporal_downscale_factor
        spatial_downscale = self.vae.spatial_downscale_factor
        latents, pixel_coords, conditioning_mask, num_cond_latents = (
            self.prepare_conditioning(
                conditioning_items=[],
                init_latents=latents,
                num_frames=(num_frames-1)*temporal_downscale + 1,
                height=height*spatial_downscale,
                width=width*spatial_downscale,
                vae_per_channel_normalize=True,
            )
        )

        if mask is not None:
            # untested
            mask = mask.unsqueeze(1).unsqueeze(1).expand((-1, channels, num_frames, -1, -1))  # make mask (bs, c, f, img_h, img_w)
            mask = F.interpolate(mask, size=(height, width), mode='nearest-exact')  # resize to latent spatial dimension
            mask, _ = self.patchifier.patchify(
                latents=mask
            )

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

        x_1 = latents
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1)

        # Copied and modified from https://github.com/Lightricks/LTX-Video-Trainer/blob/main/src/ltxv_trainer/trainer.py
        if mask is None:
            mask = torch.ones_like(x_1)
        first_frame_conditioning_p = self.model_config.get('first_frame_conditioning_p', 0)
        # If first frame conditioning is enabled, the first latent (first video frame) is left (almost) unchanged.
        if first_frame_conditioning_p and random.random() < first_frame_conditioning_p:
            t_expanded = t_expanded.repeat(1, x_1.shape[1], 1)
            first_frame_end_idx = height * width

            # if we only have one frame (e.g. when training on still images),
            # skip this step otherwise we have no target to train on.
            if first_frame_end_idx < x_1.shape[1]:
                t_expanded[:, :first_frame_end_idx] = 1e-5  # Small sigma close to 0 for the first frame.
                mask[:, :first_frame_end_idx] = 0.0  # Mask out the loss for the first frame.

        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
        target = x_0 - x_1

        fractional_coords = pixel_coords.to(torch.float32)
        fractional_coords[:, 0] = fractional_coords[:, 0] * (1.0 / self.framerate)

        return (x_t, prompt_embeds, prompt_attention_mask, t, fractional_coords), (target, mask)

    def to_layers(self):
        transformer = self.transformer
        layers = [InitialLayer(transformer)]
        for block in transformer.transformer_blocks:
            layers.append(TransformerLayer(block))
        layers.append(OutputLayer(transformer))
        return layers

    def get_loss_fn(self):
        def loss_fn(output, label):
            target, mask = label
            with torch.autocast('cuda', enabled=False):
                output = output.to(torch.float32)
                target = target.to(output.device, torch.float32)
                loss = F.mse_loss(output, target, reduction='none')
                # empty tensor means no masking
                if mask.numel() > 0:
                    mask = mask.to(output.device, torch.float32)
                    # Copied and modified from https://github.com/Lightricks/LTX-Video-Trainer/blob/main/src/ltxv_trainer/trainer.py
                    loss = loss.mul(mask).div(mask.mean())   # divide by mean to keep the loss scale unchanged.
                loss = loss.mean()
            return loss
        return loss_fn



class InitialLayer(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = [transformer]
        self.patchify_proj = transformer.patchify_proj
        self.timestep_scale_multiplier = transformer.timestep_scale_multiplier
        self.adaln_single = transformer.adaln_single
        self.caption_projection = transformer.caption_projection

    def __getattr__(self, name):
        return getattr(self.transformer[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        (hidden_states, encoder_hidden_states, encoder_attention_mask, timestep, indices_grid) = inputs

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        hidden_states = self.patchify_proj(hidden_states)

        if self.timestep_scale_multiplier:
            timestep = self.timestep_scale_multiplier * timestep

        freqs_cos, freqs_sin = self.precompute_freqs_cis(indices_grid)

        batch_size = hidden_states.shape[0]
        timestep, embedded_timestep = self.adaln_single(
            timestep.flatten(),
            {"resolution": None, "aspect_ratio": None},
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        # Second dimension is 1 or number of tokens (if timestep_per_token)
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(
            batch_size, -1, embedded_timestep.shape[-1]
        )

        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.shape[-1]
            )

        outputs = make_contiguous(hidden_states, encoder_hidden_states, timestep, embedded_timestep, freqs_cos, freqs_sin, encoder_attention_mask)
        for tensor in outputs:
            if torch.is_floating_point(tensor):
                tensor.requires_grad_(True)
        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, timestep, embedded_timestep, freqs_cos, freqs_sin, encoder_attention_mask = inputs
        hidden_states = self.block(
            hidden_states,
            freqs_cis=(freqs_cos, freqs_sin),
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
        )
        return make_contiguous(hidden_states, encoder_hidden_states, timestep, embedded_timestep, freqs_cos, freqs_sin, encoder_attention_mask)


class OutputLayer(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = [transformer]
        self.scale_shift_table = transformer.scale_shift_table
        self.norm_out = transformer.norm_out
        self.proj_out = transformer.proj_out

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, timestep, embedded_timestep, freqs_cos, freqs_sin, encoder_attention_mask = inputs

        scale_shift_values = (
            self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        return self.proj_out(hidden_states)
