import os
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from safetensors import torch as safetorch
from transformers import AutoTokenizer, UMT5EncoderModel

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from utils.common import AUTOCAST_DTYPE, is_main_process
from utils.offloading import ModelOffloader


submodule_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/LongCat-Video')
if submodule_path not in sys.path:
    sys.path.insert(0, submodule_path)

from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel  # noqa: E402
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan  # noqa: E402
from longcat_video.modules.scheduling_flow_match_euler_discrete import (  # noqa: E402
    FlowMatchEulerDiscreteScheduler,
)


KEEP_IN_HIGH_PRECISION = [
    'norm',
    'bias',
    'scale_shift_table',
    'patchify_proj',
    'proj_out',
    'adaln_single',
    'caption_projection',
    'adaLN_modulation',
    'q_norm',
    'k_norm',
]


def retrieve_latents(encoder_output: torch.Tensor, sample_mode: str = "sample") -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample()
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class LongcatVideoPipeline(BasePipeline):
    # Use a neutral name so dataset batching follows the same path as Wan/Chroma
    # (avoids legacy special-casing tied to 'longcat-video').
    name = 'longcat'
    framerate = 15
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['LongCatSingleStreamBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']

        dtype = self.model_config['dtype']
        ckpt_path = Path(self.model_config['ckpt_path'])

        self.tokenizer_path = Path(self.model_config.get('tokenizer_path', ckpt_path / 'tokenizer'))
        self.text_encoder_path = Path(self.model_config.get('text_encoder_path', ckpt_path / 'text_encoder'))
        self.vae_path = Path(self.model_config.get('vae_path', ckpt_path / 'vae'))
        self.scheduler_path = Path(self.model_config.get('scheduler_path', ckpt_path / 'scheduler'))
        self.transformer_path = Path(self.model_config.get('transformer_path', ckpt_path / 'dit'))

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.text_encoder = UMT5EncoderModel.from_pretrained(self.text_encoder_path, torch_dtype=dtype)
        self.text_encoder.requires_grad_(False)

        self.vae = AutoencoderKLWan.from_pretrained(self.vae_path, torch_dtype=dtype)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.scheduler_path, torch_dtype=torch.float32)

        raw_mean = torch.tensor(self.vae.config.latents_mean, dtype=dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        raw_std = torch.tensor(self.vae.config.latents_std, dtype=dtype).view(1, self.vae.config.z_dim, 1, 1, 1)
        raw_std = torch.clamp(raw_std, min=1e-6)
        self.latents_mean = raw_mean
        self.latents_std = 1.0 / raw_std

        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)

    def model_specific_dataset_config_validation(self, dataset_config):
        # Follow the same behavior as other models (Wan, Chroma):
        # do not modify dataset config here.
        pass

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        transformer = LongCatVideoTransformer3DModel.from_pretrained(
            self.transformer_path,
            torch_dtype=dtype,
        )

        # Experimental FP8 workaround (inspired by docs/longcat_video_alt.py):
        # - Keep sensitive modules in base dtype (norm/bias and embedding/final layers)
        # - Cast the rest to the requested transformer_dtype (e.g., float8_e4m3fn)
        # This avoids global fallback when float8 is requested.
        if transformer_dtype == torch.float8_e4m3fn:
            KEEP_BASE_DTYPE_FP8 = [
                'norm', 'bias',
                't_embedder', 'y_embedder', 'x_embedder', 'final_layer',
            ]
            if is_main_process():
                print('LongcatVideo: enabling experimental FP8 workaround — keeping embedder/final/norm/bias in base dtype, casting others to FP8.')
            for name, parameter in transformer.named_parameters():
                if any(token in name for token in KEEP_BASE_DTYPE_FP8):
                    continue
                try:
                    parameter.data = parameter.data.to(transformer_dtype)
                except Exception as e:
                    # If any op refuses FP8 weights, keep base dtype for that param
                    if is_main_process():
                        print(f'  [FP8] Skipped casting {name} to FP8 due to: {e}')
        else:
            for name, parameter in transformer.named_parameters():
                if not any(token in name for token in KEEP_IN_HIGH_PRECISION):
                    parameter.data = parameter.data.to(transformer_dtype)
        transformer.train()

        for name, parameter in transformer.named_parameters():
            parameter.original_name = name

        self.transformer = transformer
        if getattr(self.transformer, 'cp_split_hw', None) is None:
            self.transformer.cp_split_hw = (1, 1)
        for block in getattr(self.transformer, 'blocks', []):
            attn = getattr(block, 'attn', None)
            if attn is not None:
                if getattr(attn, 'cp_split_hw', None) is None:
                    attn.cp_split_hw = (1, 1)
                if hasattr(attn, 'rope_3d') and getattr(attn.rope_3d, 'cp_split_hw', None) is None:
                    attn.rope_3d.cp_split_hw = (1, 1)

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder]

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        alpha_value = float(getattr(self.peft_config, 'lora_alpha', self.peft_config.r))

        pairs = {}
        for key, tensor in peft_state_dict.items():
            full_key = 'diffusion_model.' + key
            tensor_cpu = tensor.detach().to('cpu')
            if full_key.endswith('.lora_A.weight'):
                base_key = full_key[:-len('.lora_A.weight')]
                pairs.setdefault(base_key, {})['down'] = tensor_cpu
            elif full_key.endswith('.lora_B.weight'):
                base_key = full_key[:-len('.lora_B.weight')]
                pairs.setdefault(base_key, {})['up'] = tensor_cpu

        converted_state = {}

        def write_entry(prefix, down_tensor, up_tensor):
            converted_state[prefix + '.lora_down.weight'] = down_tensor.clone()
            converted_state[prefix + '.lora_up.weight'] = up_tensor.clone()
            converted_state[prefix + '.alpha'] = torch.tensor(alpha_value, dtype=down_tensor.dtype)

        for base_key, tensors in pairs.items():
            if 'down' not in tensors or 'up' not in tensors:
                continue
            down = tensors['down']
            up = tensors['up']

            if base_key.endswith('.attn.qkv'):
                prefix = base_key[:-len('.attn.qkv')] + '.self_attn'
                hidden = up.shape[0] // 3
                up_q, up_k, up_v = up.split(hidden, dim=0)
                write_entry(prefix + '.q', down, up_q)
                write_entry(prefix + '.k', down, up_k)
                write_entry(prefix + '.v', down, up_v)
            elif base_key.endswith('.attn.proj'):
                prefix = base_key[:-len('.attn.proj')] + '.self_attn.o'
                write_entry(prefix, down, up)
            elif base_key.endswith('.cross_attn.q_linear'):
                prefix = base_key[:-len('.cross_attn.q_linear')] + '.cross_attn.q'
                write_entry(prefix, down, up)
            elif base_key.endswith('.cross_attn.kv_linear'):
                prefix = base_key[:-len('.cross_attn.kv_linear')] + '.cross_attn'
                hidden = up.shape[0] // 2
                up_k, up_v = up.split(hidden, dim=0)
                write_entry(prefix + '.k', down, up_k)
                write_entry(prefix + '.v', down, up_v)
            elif '.ffn.' in base_key:
                write_entry(base_key, down, up)
            # skip other modules (e.g., adaLN_modulation) that ComfyUI does not consume

        safetorch.save_file(converted_state, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def get_preprocess_media_file_fn(self):
        # Mirror docs/longcat_video_alt.py and other pipelines: no extra rounding overrides.
        return PreprocessMediaFile(
            self.config,
            support_video=True,
            framerate=self.framerate,
        )

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_mean = self.latents_mean.to(latents.device, latents.dtype)
        latents_std = torch.clamp(self.latents_std.to(latents.device, latents.dtype), min=1e-6)
        return (latents - latents_mean) * latents_std

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_mean = self.latents_mean.to(latents.device, latents.dtype)
        latents_std = torch.clamp(self.latents_std.to(latents.device, latents.dtype), min=1e-6)
        return latents / latents_std + latents_mean

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            ref_param = next(vae.parameters())
            tensor = tensor.to(device=ref_param.device, dtype=ref_param.dtype)
            latents = retrieve_latents(vae.encode(tensor))
            latents = self.normalize_latents(latents)
            return {'latents': latents}
        return fn

    def encode_prompt(
        self,
        prompt,
        device,
        dtype,
        max_sequence_length: int = 512,
        num_videos_per_prompt: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt = [prompt] if isinstance(prompt, str) else list(prompt)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        hidden_states = self.text_encoder(input_ids, attention_mask).last_hidden_state.to(device=device, dtype=dtype)

        batch_size, seq_len, hidden = hidden_states.shape
        hidden_states = hidden_states.repeat_interleave(num_videos_per_prompt, dim=0)
        hidden_states = hidden_states.view(batch_size * num_videos_per_prompt, seq_len, hidden)
        hidden_states = hidden_states.unsqueeze(1)

        attention_mask = attention_mask.repeat_interleave(num_videos_per_prompt, dim=0)
        return hidden_states, attention_mask

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(captions, is_video):
            device = next(text_encoder.parameters()).device
            dtype = text_encoder.dtype
            prompt_embeds, prompt_attention_mask = self.encode_prompt(
                captions,
                device=device,
                dtype=dtype,
            )
            return {
                'prompt_embeds': prompt_embeds,
                'prompt_attention_mask': prompt_attention_mask,
            }
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        prompt_embeds = inputs['prompt_embeds']
        prompt_attention_mask = inputs['prompt_attention_mask']
        mask = inputs['mask']

        bs, channels, num_frames, height, width = latents.shape

        transformer_dtype = next(self.transformer.parameters()).dtype
        prompt_embeds = prompt_embeds.to(device=latents.device, dtype=transformer_dtype)
        if prompt_embeds.ndim == 3:
            prompt_embeds = prompt_embeds.unsqueeze(1)
        prompt_attention_mask = prompt_attention_mask.to(device=latents.device)
        if prompt_attention_mask.ndim == 1:
            prompt_attention_mask = prompt_attention_mask.unsqueeze(0)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            mask = mask.expand(-1, channels, num_frames, -1, -1)
            mask = F.interpolate(mask, size=(height, width), mode='nearest-exact')
            mask = mask.to(device=latents.device, dtype=latents.dtype)

        distribution = self.model_config.get('timestep_sample_method', 'logit_normal')
        if distribution == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif distribution == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError(f'Unknown timestep_sample_method {distribution}')

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=latents.device))
        else:
            t = dist.sample((bs,)).to(latents.device)

        if distribution == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = torch.sigmoid(t * sigmoid_scale)

        x_1 = latents
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0

        # Flow target direction: some FM/rectified-flow stacks expect x1-x0 and others x0-x1.
        # Default to x1-x0 (moves sample toward data as dt becomes negative in Euler FM schedulers).
        flow_target = self.model_config.get('flow_target', 'x1_minus_x0')
        if flow_target == 'x0_minus_x1':
            target = x_0 - x_1
        elif flow_target == 'x1_minus_x0':
            target = x_1 - x_0
        else:
            raise NotImplementedError(f'Unknown flow_target {flow_target}')

        # The transformer InitialLayer expects per-frame timesteps (shape [B, N_t]).
        # Expand scalar t to per-frame indices and scale to [0, 1000] like inference.
        timestep_input = (t * 1000.0).to(latents.device).unsqueeze(1).expand(bs, num_frames)

        if torch.isnan(x_t).any() or torch.isnan(prompt_embeds).any() or torch.isnan(prompt_attention_mask.float()).any():
            raise ValueError('NaN detected in prepared inputs')

        features = (
            x_t,
            prompt_embeds,
            prompt_attention_mask,
            timestep_input,
        )
        label = (target, mask)
        return features, label

    def to_layers(self):
        transformer = self.transformer
        # When block swap is disabled and using a single pipeline stage, Deepspeed won't
        # automatically move the top-level transformer modules referenced by Initial/Final wrappers
        # (x_embedder / t_embedder / y_embedder / final_layer), since they aren't registered as
        # submodules of those wrappers. Ensure they live on CUDA to avoid CPU/CUDA dtype mismatches.
        try:
            if self.config.get('pipeline_stages', 1) == 1:
                for mname in ('x_embedder', 't_embedder', 'y_embedder', 'final_layer'):
                    m = getattr(transformer, mname, None)
                    if m is not None:
                        m.to('cuda')
        except Exception:
            pass
        layers = [InitialLayer(transformer)]
        for idx, block in enumerate(transformer.blocks):
            layers.append(TransformerLayer(block, idx, self.offloader))
        layers.append(FinalLayer(transformer))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        blocks = self.transformer.blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap}.'
        self.offloader = ModelOffloader(
            'TransformerBlock',
            blocks,
            num_blocks,
            blocks_to_swap,
            True,
            torch.device('cuda'),
            self.config.get('reentrant_activation_checkpointing', False),
        )
        self.transformer.blocks = None
        self.transformer.to('cuda')
        self.transformer.blocks = blocks
        self.prepare_block_swap_training()
        print(f'Block swap enabled for LongcatVideo. Swapping {blocks_to_swap} of {num_blocks} blocks.')

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
    def __init__(self, transformer):
        super().__init__()
        self.transformer = [transformer]

    def __getattr__(self, name):
        return getattr(self.transformer[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, encoder_attention_mask, timestep = inputs

        for item in inputs:
            if torch.is_tensor(item) and torch.is_floating_point(item):
                item.requires_grad_(True)

        B, _, T, H, W = hidden_states.shape
        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]
        latent_shape = torch.tensor([N_t, N_h, N_w], device=hidden_states.device, dtype=torch.int32)

        dtype = self.x_embedder.proj.weight.dtype
        hidden_states = hidden_states.to(dtype)
        timestep = timestep.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)

        hidden_states = self.x_embedder(hidden_states)

        t_embed = self.t_embedder(timestep.float().flatten(), dtype=torch.float32).reshape(B, N_t, -1)
        encoder_hidden_states = self.y_embedder(encoder_hidden_states)

        if encoder_attention_mask is not None:
            attention_mask = encoder_attention_mask.to(hidden_states.device)
            encoder_hidden_states = encoder_hidden_states * attention_mask[:, None, :, None]
            encoder_attention_mask = (attention_mask * 0 + 1).to(attention_mask.dtype)
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(1)
            encoder_attention_mask = encoder_attention_mask.squeeze(1).squeeze(1)
            encoder_hidden_states = (
                encoder_hidden_states.squeeze(1)
                .masked_select(encoder_attention_mask.unsqueeze(-1) != 0)
                .view(1, -1, hidden_states.shape[-1])
            )
            y_seqlens = torch.tensor(
                encoder_attention_mask.sum(dim=1).tolist(),
                device=hidden_states.device,
                dtype=torch.int64,
            )
        else:
            encoder_hidden_states = encoder_hidden_states.squeeze(1).view(1, -1, hidden_states.shape[-1])
            y_seqlens = torch.full((B,), encoder_hidden_states.shape[1], device=hidden_states.device, dtype=torch.int64)

        outputs = make_contiguous(hidden_states, encoder_hidden_states, t_embed, y_seqlens, latent_shape)
        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, t_embed, y_seqlens, latent_shape = inputs

        self.offloader.wait_for_block(self.block_idx)

        latent_shape_tuple = tuple(latent_shape.tolist())
        seqlens_list = y_seqlens.tolist()

        hidden_states = self.block(
            hidden_states,
            encoder_hidden_states,
            t_embed,
            seqlens_list,
            latent_shape_tuple,
            num_cond_latents=None,
        )

        self.offloader.submit_move_blocks_forward(self.block_idx)
        if torch.isnan(hidden_states).any():
            raise ValueError(f'NaN detected in transformer block output at block {self.block_idx}')
        return make_contiguous(hidden_states, encoder_hidden_states, t_embed, y_seqlens, latent_shape)


class FinalLayer(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = [transformer]

    def __getattr__(self, name):
        return getattr(self.transformer[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, t_embed, y_seqlens, latent_shape = inputs
        latent_shape_tuple = tuple(latent_shape.tolist())
        t_embed = t_embed.to(torch.float32)
        hidden_states = self.transformer[0].final_layer(hidden_states, t_embed, latent_shape_tuple)
        hidden_states = self.transformer[0].unpatchify(hidden_states, *latent_shape_tuple)
        if torch.isnan(hidden_states).any():
            raise ValueError('NaN detected in transformer output')
        return hidden_states
