import safetensors
import torch
from torch import nn
import torch.nn.functional as F
from diffusers import LTXPipeline

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from utils.common import AUTOCAST_DTYPE

FRAMERATE = 25


class LTXVideoPipeline(BasePipeline):
    name = 'ltx-video'
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['LTXTransformerBlock', 'LTXVideoTransformerBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']

        dtype = self.model_config['dtype']
        self.diffusers_pipeline = LTXPipeline.from_pretrained(self.model_config['diffusers_path'], torch_dtype=dtype)

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
        return (self.text_encoder,)

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # Convention is to have "transformer." prefix
        peft_state_dict = {'transformer.'+k: v for k, v in peft_state_dict.items()}
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def save_model(self, save_dir, diffusers_sd):
        raise NotImplementedError()

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(
            self.config,
            support_video=True,
            framerate=FRAMERATE,
            round_height=32,
            round_width=32,
            round_frames=8,
        )

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            latents = vae.encode(tensor.to(vae.device, vae.dtype)).latent_dist.sample()
            latents = self._normalize_latents(latents, self.vae.latents_mean, self.vae.latents_std)
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

        bs, channels, num_frames, height, width = latents.shape
        latents = self._pack_latents(
            latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
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
        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
        target = x_0 - x_1

        # Timesteps passed to model need to be in range [0, 1000]
        t = t * 1000

        num_frames = torch.full((bs,), num_frames)
        height = torch.full((bs,), height)
        width = torch.full((bs,), width)
        latent_frame_rate = FRAMERATE / self.vae_temporal_compression_ratio
        rope_interpolation_scale_time = torch.full((bs,), 1 / latent_frame_rate)
        rope_interpolation_scale_space = torch.full((bs,), self.vae_spatial_compression_ratio)

        return x_t, prompt_embeds, prompt_attention_mask, t, num_frames, height, width, rope_interpolation_scale_time, rope_interpolation_scale_space, target

    def to_layers(self):
        transformer = self.transformer
        layers = [InitialLayer(transformer)]
        for block in transformer.transformer_blocks:
            layers.append(TransformerLayer(block))
        layers.append(OutputLayer(transformer))
        return layers


class InitialLayer(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        # Prevent registering the whole Transformer.
        self.transformer = [transformer]
        # Explicitly register these modules.
        self.rope = self.transformer[0].rope
        self.proj_in = self.transformer[0].proj_in
        self.time_embed = self.transformer[0].time_embed
        self.caption_projection = self.transformer[0].caption_projection

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        hidden_states, encoder_hidden_states, encoder_attention_mask, timestep, num_frames, height, width, rope_interpolation_scale_time, rope_interpolation_scale_space, target = inputs

        rope_interpolation_scale = (
            rope_interpolation_scale_time[0].item(),
            rope_interpolation_scale_space[0].item(),
            rope_interpolation_scale_space[0].item(),
        )
        num_frames = num_frames[0].item()
        height = height[0].item()
        width = width[0].item()
        freqs_cos, freqs_sin = self.rope(hidden_states, num_frames, height, width, rope_interpolation_scale)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        batch_size = hidden_states.size(0)
        hidden_states = self.proj_in(hidden_states)

        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

        return make_contiguous(hidden_states, encoder_hidden_states, temb, embedded_timestep, freqs_cos, freqs_sin, encoder_attention_mask, target)


class TransformerLayer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, temb, embedded_timestep, freqs_cos, freqs_sin, encoder_attention_mask, target = inputs
        hidden_states = self.block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=(freqs_cos, freqs_sin),
            encoder_attention_mask=encoder_attention_mask,
        )
        return make_contiguous(hidden_states, encoder_hidden_states, temb, embedded_timestep, freqs_cos, freqs_sin, encoder_attention_mask, target)


class OutputLayer(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        # Prevent registering the whole Transformer.
        self.transformer = [transformer]
        # Explicitly register these modules.
        self.scale_shift_table = self.transformer[0].scale_shift_table
        self.norm_out = self.transformer[0].norm_out
        self.proj_out = self.transformer[0].proj_out

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, temb, embedded_timestep, freqs_cos, freqs_sin, encoder_attention_mask, target = inputs

        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)

        output = output.to(torch.float32)
        target = target.to(torch.float32)
        loss = F.mse_loss(output, target)
        return loss
