import os
import sys
import types
from typing import Tuple
import re
import math
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import torch
from torch import nn
import torch.nn.functional as F
import safetensors

from models.base import ComfyPipeline, make_contiguous, ModelWrapper, PreprocessMediaFile
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift, is_main_process, one_at_a_time
from utils.offloading import ModelOffloader
import comfy.ldm.common_dit
import comfy.latent_formats
import comfy.model_management
import comfy.text_encoders.lt
import comfy.ldm.common_dit
from comfy.ldm.lightricks.av_model import CompressedTimestep, BasicAVTransformerBlock
from comfy.ldm.lightricks.model import LTXRopeType


# Patch this to handle attention_mask, and also return attention_mask.
def encode_token_weights(self, token_weight_pairs):
    token_weight_pairs = token_weight_pairs["gemma3_12b"]

    out, pooled, extra = self.gemma3_12b.encode_token_weights(token_weight_pairs)
    out_device = out.device
    if comfy.model_management.should_use_bf16(self.execution_device):
        out = out.to(device=self.execution_device, dtype=torch.bfloat16)

    attention_mask = extra["attention_mask"]
    bool_mask = attention_mask.bool().unsqueeze(1).unsqueeze(-1)
    out = torch.where(bool_mask, out, torch.zeros_like(out))

    assert self.text_projection_type == "dual_linear"
    out = self.text_embedding_projection(out)
    extra = {"unprocessed_ltxav_embeds": True, "attention_mask": attention_mask}

    return out.to(device=out_device, dtype=torch.float), pooled, extra

comfy.text_encoders.lt.LTXAVTEModel.encode_token_weights = encode_token_weights


# patch to remove in-place operations (doesn't work with training)
def forward(
    self, x: Tuple[torch.Tensor, torch.Tensor], v_context=None, a_context=None, attention_mask=None, v_timestep=None, a_timestep=None,
    v_pe=None, a_pe=None, v_cross_pe=None, a_cross_pe=None, v_cross_scale_shift_timestep=None, a_cross_scale_shift_timestep=None,
    v_cross_gate_timestep=None, a_cross_gate_timestep=None, transformer_options=None, self_attention_mask=None,
    v_prompt_timestep=None, a_prompt_timestep=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    run_vx = transformer_options.get("run_vx", True)
    run_ax = transformer_options.get("run_ax", True)

    vx, ax = x
    run_ax = run_ax and ax.numel() > 0
    run_a2v = run_vx and transformer_options.get("a2v_cross_attn", True) and ax.numel() > 0
    run_v2a = run_ax and transformer_options.get("v2a_cross_attn", True)

    # video
    if run_vx:
        # video self-attention
        vshift_msa, vscale_msa = (self.get_ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(0, 2)))
        norm_vx = comfy.ldm.common_dit.rms_norm(vx) * (1 + vscale_msa) + vshift_msa
        del vshift_msa, vscale_msa
        attn1_out = self.attn1(norm_vx, pe=v_pe, mask=self_attention_mask, transformer_options=transformer_options)
        del norm_vx
        # video cross-attention
        vgate_msa = self.get_ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(2, 3))[0]
        vx = vx + attn1_out * vgate_msa
        del vgate_msa, attn1_out
        vx = vx + self._apply_text_cross_attention(
            vx, v_context, self.attn2, self.scale_shift_table,
            getattr(self, 'prompt_scale_shift_table', None),
            v_timestep, v_prompt_timestep, attention_mask, transformer_options
        )

    # audio
    if run_ax:
        # audio self-attention
        ashift_msa, ascale_msa = (self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(0, 2)))
        norm_ax = comfy.ldm.common_dit.rms_norm(ax) * (1 + ascale_msa) + ashift_msa
        del ashift_msa, ascale_msa
        attn1_out = self.audio_attn1(norm_ax, pe=a_pe, transformer_options=transformer_options)
        del norm_ax
        # audio cross-attention
        agate_msa = self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(2, 3))[0]
        ax = ax + attn1_out * agate_msa
        del agate_msa, attn1_out
        ax = ax + self._apply_text_cross_attention(
            ax, a_context, self.audio_attn2, self.audio_scale_shift_table,
            getattr(self, 'audio_prompt_scale_shift_table', None),
            a_timestep, a_prompt_timestep, attention_mask, transformer_options
        )


    # video - audio cross attention.
    if run_a2v or run_v2a:
        vx_norm3 = comfy.ldm.common_dit.rms_norm(vx)
        ax_norm3 = comfy.ldm.common_dit.rms_norm(ax)

        # audio to video cross attention
        if run_a2v:
            scale_ca_audio_hidden_states_a2v, shift_ca_audio_hidden_states_a2v = self.get_ada_values(
                self.scale_shift_table_a2v_ca_audio[:4, :], ax.shape[0], a_cross_scale_shift_timestep)[:2]
            scale_ca_video_hidden_states_a2v_v, shift_ca_video_hidden_states_a2v_v = self.get_ada_values(
                self.scale_shift_table_a2v_ca_video[:4, :], vx.shape[0], v_cross_scale_shift_timestep)[:2]

            vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_a2v_v) + shift_ca_video_hidden_states_a2v_v
            ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_a2v) + shift_ca_audio_hidden_states_a2v
            del scale_ca_video_hidden_states_a2v_v, shift_ca_video_hidden_states_a2v_v, scale_ca_audio_hidden_states_a2v, shift_ca_audio_hidden_states_a2v

            a2v_out = self.audio_to_video_attn(vx_scaled, context=ax_scaled, pe=v_cross_pe, k_pe=a_cross_pe, transformer_options=transformer_options)
            del vx_scaled, ax_scaled

            gate_out_a2v = self.get_ada_values(self.scale_shift_table_a2v_ca_video[4:, :], vx.shape[0], v_cross_gate_timestep)[0]
            vx = vx + a2v_out * gate_out_a2v
            del gate_out_a2v, a2v_out

        # video to audio cross attention
        if run_v2a:
            scale_ca_audio_hidden_states_v2a, shift_ca_audio_hidden_states_v2a = self.get_ada_values(
                self.scale_shift_table_a2v_ca_audio[:4, :], ax.shape[0], a_cross_scale_shift_timestep)[2:4]
            scale_ca_video_hidden_states_v2a, shift_ca_video_hidden_states_v2a = self.get_ada_values(
                self.scale_shift_table_a2v_ca_video[:4, :], vx.shape[0], v_cross_scale_shift_timestep)[2:4]

            ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_v2a) + shift_ca_audio_hidden_states_v2a
            vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_v2a) + shift_ca_video_hidden_states_v2a
            del scale_ca_video_hidden_states_v2a, shift_ca_video_hidden_states_v2a, scale_ca_audio_hidden_states_v2a, shift_ca_audio_hidden_states_v2a

            v2a_out = self.video_to_audio_attn(ax_scaled, context=vx_scaled, pe=a_cross_pe, k_pe=v_cross_pe, transformer_options=transformer_options)
            del ax_scaled, vx_scaled

            gate_out_v2a = self.get_ada_values(self.scale_shift_table_a2v_ca_audio[4:, :], ax.shape[0], a_cross_gate_timestep)[0]
            ax = ax + v2a_out * gate_out_v2a
            del gate_out_v2a, v2a_out

        del vx_norm3, ax_norm3

    # video feedforward
    if run_vx:
        vshift_mlp, vscale_mlp = self.get_ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(3, 5))
        vx_scaled = comfy.ldm.common_dit.rms_norm(vx) * (1 + vscale_mlp) + vshift_mlp
        del vshift_mlp, vscale_mlp

        ff_out = self.ff(vx_scaled)
        del vx_scaled

        vgate_mlp = self.get_ada_values(self.scale_shift_table, vx.shape[0], v_timestep, slice(5, 6))[0]
        vx = vx + ff_out * vgate_mlp
        del vgate_mlp, ff_out

    # audio feedforward
    if run_ax:
        ashift_mlp, ascale_mlp = self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(3, 5))
        ax_scaled = comfy.ldm.common_dit.rms_norm(ax) * (1 + ascale_mlp) + ashift_mlp
        del ashift_mlp, ascale_mlp

        ff_out = self.audio_ff(ax_scaled)
        del ax_scaled

        agate_mlp = self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], a_timestep, slice(5, 6))[0]
        ax = ax + ff_out * agate_mlp
        del agate_mlp, ff_out

    return vx, ax

BasicAVTransformerBlock.forward = forward

class LTX2Pipeline(ComfyPipeline):
    name = 'ltx2'
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['BasicAVTransformerBlock']
    # TODO: first 2 and last 2 layers like in fp8 checkpoint?
    keep_in_high_precision = [
        'audio_embeddings_connector',
        'video_embeddings_connector',
        'adaln_single',
        'audio_adaln_single',
        'audio_patchify_proj',
        'audio_proj_out',
        'audio_prompt_adaln_single',
        'audio_scale_shift_table',
        'av_ca_a2v_gate_adaln_single',
        'av_ca_audio_scale_shift_adaln_single',
        'av_ca_v2a_gate_adaln_single',
        'av_ca_video_scale_shift_adaln_single',
        'patchify_proj',
        'proj_out',
        'prompt_adaln_single',
        'scale_shift_table',
    ]

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.latent_format = comfy.latent_formats.LTXAV()
        self.framerate = 24  # TODO: configurable

        # VAE
        def load_fn():
            vae_sd = {}
            with safetensors.safe_open(self.model_config['diffusion_model'], framework='pt', device='cpu') as f:
                metadata = f.metadata()
                for key in f.keys():
                    if m := re.fullmatch(r'vae\.(.+)', key):
                        vae_sd[m.group(1)] = f.get_tensor(key)
            assert len(vae_sd) > 0

            vae = comfy.sd.VAE(sd=vae_sd, metadata=metadata)
            vae.throw_exception_if_invalid()

            def vae_encode_crop_pixels(self, pixels):
                if not self.crop_input:
                    return pixels

                downscale_ratio = self.spacial_compression_encode()

                dims = pixels.shape[-3:-1]
                for d in range(len(dims)):
                    x = (dims[d] // downscale_ratio) * downscale_ratio
                    x_offset = (dims[d] % downscale_ratio) // 2
                    if x != dims[d]:
                        pixels = pixels.narrow(d + 1, x_offset, x)
                return pixels

            # patch this to handle 5D video tensor (original code expects 4D even for video)
            vae.vae_encode_crop_pixels = types.MethodType(vae_encode_crop_pixels, self.vae)
            return vae
        self.vae = ModelWrapper(load_fn)

        # Text encoder
        def load_fn():
            return comfy.sd.load_clip(ckpt_paths=[self.model_config['text_encoder'], self.model_config['diffusion_model']], clip_type=comfy.sd.CLIPType.LTXV)
        self.text_encoders = [ModelWrapper(load_fn)]

        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)

    def _load_diffusion_model(self):
        dtype = self.model_config['dtype']

        out = comfy.sd.load_checkpoint_guess_config(self.model_config['diffusion_model'], output_vae=False, output_clip=False)
        model_patcher = out[0]

        for adapter_path in self.model_config.get('merge_adapters', []):
            print(f'Merging adapter {adapter_path}')
            sd = comfy.utils.load_torch_file(adapter_path, safe_load=True)
            model_patcher, _ = comfy.sd.load_lora_for_models(model_patcher, None, sd, 1.0, 0.0)
            del sd

        model_patcher.set_model_compute_dtype(dtype)
        with torch.no_grad():
            model_patcher.patch_model()
        self.diffusion_model = model_patcher.model.diffusion_model
        del model_patcher

        diffusion_model_dtype = self.model_config.get('diffusion_model_dtype', dtype)
        for name, p in self.diffusion_model.named_parameters():
            if any(keyword in name for keyword in self.keep_in_high_precision) or p.ndim == 1:
                continue
            p.data = p.data.to(diffusion_model_dtype)

        self.diffusion_model.train()
        for name, p in self.diffusion_model.named_parameters():
            p.original_name = name
            p.requires_grad_(True)

    def load_diffusion_model(self):
        # Model is so big, it's easy to OOM while loading with multiple GPUs.
        with one_at_a_time():
            rank = int(os.environ['LOCAL_RANK'])
            print(f'Loading model on rank {rank}')
            self._load_diffusion_model()

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(self.config, support_video=True, framerate=self.framerate, round_height=32, round_width=32, round_frames=8)

    def to_layers(self):
        diffusion_model = self.diffusion_model
        split_mode = diffusion_model.split_positional_embedding == LTXRopeType.SPLIT
        layers = [InitialLayer(diffusion_model, self.framerate)]
        for i, block in enumerate(diffusion_model.transformer_blocks):
            layers.append(TransformerLayer(block, i, self.offloader, split_mode))
        layers.append(FinalLayer(diffusion_model))
        return layers

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        text_embeds = inputs['text_embeds_0']
        attention_mask = inputs['attention_mask_0']
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
        attention_mask = attention_mask.bool()

        bs, c, f, h, w = latents.shape
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

        noisy_latents_audio = torch.zeros([bs, 0], device=device)

        return (noisy_latents, noisy_latents_audio, t, text_embeds, attention_mask), (target, mask)

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

    def get_loss_fn(self):
        def loss_fn(output, label):
            target, mask = label
            with torch.autocast('cuda', enabled=False):
                output = output.to(torch.float32)
                target = target.to(output.device, torch.float32)
                if 'huber_delta' in self.config:
                    loss = F.huber_loss(output, target, reduction='none', delta=self.config['huber_delta'])
                elif 'smooth_l1_beta' in self.config:
                    loss = F.smooth_l1_loss(output, target, reduction='none', beta=self.config['smooth_l1_beta'])
                else:
                    loss = F.mse_loss(output, target, reduction='none')
                # empty tensor means no masking
                if mask.numel() > 0:
                    mask = mask.to(output.device, torch.float32)
                    loss *= mask
                loss = loss.mean()
            return loss
        return loss_fn


class InitialLayer(nn.Module):
    def __init__(self, model, framerate):
        super().__init__()
        self.patchify_proj = model.patchify_proj
        self.adaln_single = model.adaln_single
        self.prompt_adaln_single = model.prompt_adaln_single
        self.caption_projection = model.caption_projection
        self.audio_patchify_proj = model.audio_patchify_proj
        self.audio_adaln_single = model.audio_adaln_single
        self.audio_prompt_adaln_single = model.audio_prompt_adaln_single
        self.av_ca_video_scale_shift_adaln_single = model.av_ca_video_scale_shift_adaln_single
        self.av_ca_a2v_gate_adaln_single = model.av_ca_a2v_gate_adaln_single
        self.av_ca_audio_scale_shift_adaln_single = model.av_ca_audio_scale_shift_adaln_single
        self.av_ca_v2a_gate_adaln_single = model.av_ca_v2a_gate_adaln_single
        self.audio_caption_projection = model.audio_caption_projection
        self.audio_embeddings_connector = model.audio_embeddings_connector
        self.video_embeddings_connector = model.video_embeddings_connector
        self.framerate = framerate
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    # Re-implement this ourselves to handle attention_mask properly
    def connector_forward(self, connector, hidden_states, attention_mask):
        if connector.num_learnable_registers:
            lengths = attention_mask.sum(dim=-1)
            max_length = lengths.max().item()
            num_registers_duplications = math.ceil(
                max(1024, max_length) / connector.num_learnable_registers
            )
            learnable_registers = torch.tile(
                connector.learnable_registers.to(hidden_states), (num_registers_duplications, 1)
            )

            hidden_states_list = []
            for x, length in zip(hidden_states, lengths):
                length = length.item()
                x = x[:length]
                padding = learnable_registers[length:]
                x = torch.cat([x, padding], dim=0)
                hidden_states_list.append(x)
            hidden_states = torch.stack(hidden_states_list, dim=0)

            if attention_mask is not None:
                # float or bool attention mask required for ComfyUI attention
                attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)

        indices_grid = torch.arange(
            hidden_states.shape[1], dtype=torch.float32, device=hidden_states.device
        )
        indices_grid = indices_grid[None, None, :].repeat(hidden_states.shape[0], 1, 1)
        freqs_cis = connector.precompute_freqs_cis(indices_grid, out_dtype=hidden_states.dtype)

        reshaped_attn_mask = attention_mask[:, None, None, :]
        for block in connector.transformer_1d_blocks:
            hidden_states = block(
                hidden_states, attention_mask=reshaped_attn_mask, pe=freqs_cis
            )

        hidden_states = comfy.ldm.common_dit.rms_norm(hidden_states)

        return hidden_states, attention_mask

    def preprocess_text_embeds(self, context, attention_mask):
        if context.shape[-1] == self.cross_attention_dim + self.audio_cross_attention_dim:
            context_vid = context[:, :, :self.cross_attention_dim]
            context_audio = context[:, :, self.cross_attention_dim:]
        else:
            context_vid = context
            context_audio = context
        if self.caption_proj_before_connector:
            context_vid = self.caption_projection(context_vid)
            context_audio = self.audio_caption_projection(context_audio)
        out_vid, new_attention_mask = self.connector_forward(self.video_embeddings_connector, context_vid, attention_mask)
        out_audio = self.connector_forward(self.audio_embeddings_connector, context_audio, attention_mask)[0]
        return torch.concat((out_vid, out_audio), dim=-1), new_attention_mask

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    @torch.compiler.disable()
    def forward(self, inputs):
        vx, ax, timestep, context, attention_mask = inputs
        assert attention_mask.dtype == torch.bool and attention_mask.ndim == 2

        context, attention_mask = self.preprocess_text_embeds(context, attention_mask)
        # convert attention_mask bool -> int so _prepare_attention_mask() works
        attention_mask = attention_mask.int()

        x = [vx]
        if ax.numel() > 0:
            x.append(ax)
        transformer_options = {}

        input_dtype = x[0].dtype
        batch_size = x[0].shape[0]

        merged_args = {**transformer_options, 'a_timestep': timestep}
        # TODO: denoise_mask?
        x, pixel_coords, additional_args = self._process_input(x, keyframe_idxs=None, denoise_mask=None)
        merged_args.update(additional_args)

        timestep, (v_embedded_timestep, a_embedded_timestep), prompt_timestep = self._prepare_timestep(timestep, batch_size, input_dtype, **merged_args)
        merged_args["prompt_timestep"] = prompt_timestep
        context, attention_mask = self._prepare_context(context, batch_size, x, attention_mask)

        attention_mask = self._prepare_attention_mask(attention_mask, input_dtype)
        pe = self._prepare_positional_embeddings(pixel_coords, self.framerate, input_dtype)

        # Always None
        # self_attention_mask = self._build_guide_self_attention_mask(
        #     x, transformer_options, merged_args
        # )

        vx = x[0]
        ax = x[1]
        v_context = context[0]
        a_context = context[1]
        v_timestep = timestep[0]
        a_timestep = timestep[1]
        v_pe, av_cross_video_freq_cis = pe[0]
        a_pe, av_cross_audio_freq_cis = pe[1]
        v_pe_cos, v_pe_sin = v_pe[:2]
        av_cross_video_freq_cos, av_cross_video_freq_sin = av_cross_video_freq_cis[:2]
        a_pe_cos, a_pe_sin = a_pe[:2]
        av_cross_audio_freq_cos, av_cross_audio_freq_sin = av_cross_audio_freq_cis[:2]

        (
            av_ca_audio_scale_shift_timestep,
            av_ca_video_scale_shift_timestep,
            av_ca_a2v_gate_noise_timestep,
            av_ca_v2a_gate_noise_timestep,
        ) = timestep[2]
        v_prompt_timestep = timestep[3]
        a_prompt_timestep = timestep[4]

        # Can't pass objects between PP layers, expand back to tensors. The compression doesn't get enabled anyway (why does CompressedTimestep even exist then?)
        tmp = [v_timestep, a_timestep, av_ca_audio_scale_shift_timestep, av_ca_video_scale_shift_timestep, av_ca_a2v_gate_noise_timestep, av_ca_v2a_gate_noise_timestep, v_prompt_timestep, a_prompt_timestep, v_embedded_timestep, a_embedded_timestep]
        for i, t in enumerate(tmp):
            if isinstance(t, CompressedTimestep):
                tmp[i] = t.expand()
        v_timestep, a_timestep, av_ca_audio_scale_shift_timestep, av_ca_video_scale_shift_timestep, av_ca_a2v_gate_noise_timestep, av_ca_v2a_gate_noise_timestep, v_prompt_timestep, a_prompt_timestep, v_embedded_timestep, a_embedded_timestep = tmp

        device = vx.device
        orig_shape = torch.tensor(additional_args['orig_shape'], device=device)

        outputs = make_contiguous(
            vx, ax, v_context, a_context, attention_mask, v_timestep, a_timestep, v_pe_cos, v_pe_sin, a_pe_cos, a_pe_sin, av_cross_video_freq_cos, av_cross_video_freq_sin, av_cross_audio_freq_cos, av_cross_audio_freq_sin,
            av_ca_video_scale_shift_timestep, av_ca_audio_scale_shift_timestep, av_ca_a2v_gate_noise_timestep, av_ca_v2a_gate_noise_timestep, v_prompt_timestep, a_prompt_timestep, v_embedded_timestep, a_embedded_timestep, orig_shape
        )
        for item in outputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)

        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, layer, block_idx, offloader, split_mode):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        self.offloader = offloader
        self.split_mode = split_mode

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        (
            vx, ax, v_context, a_context, attention_mask, v_timestep, a_timestep, v_pe_cos, v_pe_sin, a_pe_cos, a_pe_sin, av_cross_video_freq_cos, av_cross_video_freq_sin, av_cross_audio_freq_cos, av_cross_audio_freq_sin,
            av_ca_video_scale_shift_timestep, av_ca_audio_scale_shift_timestep, av_ca_a2v_gate_noise_timestep, av_ca_v2a_gate_noise_timestep, v_prompt_timestep, a_prompt_timestep, v_embedded_timestep, a_embedded_timestep, orig_shape
        ) = inputs

        self.offloader.wait_for_block(self.block_idx)
        vx, ax = self.layer(
            (vx, ax),
            v_context=v_context,
            a_context=a_context,
            attention_mask=attention_mask,
            v_timestep=v_timestep,
            a_timestep=a_timestep,
            v_pe=(v_pe_cos, v_pe_sin, self.split_mode),
            a_pe=(a_pe_cos, a_pe_sin, self.split_mode),
            v_cross_pe=(av_cross_video_freq_cos, av_cross_video_freq_sin, self.split_mode),
            a_cross_pe=(av_cross_audio_freq_cos, av_cross_audio_freq_sin, self.split_mode),
            v_cross_scale_shift_timestep=av_ca_video_scale_shift_timestep,
            a_cross_scale_shift_timestep=av_ca_audio_scale_shift_timestep,
            v_cross_gate_timestep=av_ca_a2v_gate_noise_timestep,
            a_cross_gate_timestep=av_ca_v2a_gate_noise_timestep,
            self_attention_mask=None,
            v_prompt_timestep=v_prompt_timestep,
            a_prompt_timestep=a_prompt_timestep,
            transformer_options={},
        )
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return (
            vx, ax, v_context, a_context, attention_mask, v_timestep, a_timestep, v_pe_cos, v_pe_sin, a_pe_cos, a_pe_sin, av_cross_video_freq_cos, av_cross_video_freq_sin, av_cross_audio_freq_cos, av_cross_audio_freq_sin,
            av_ca_video_scale_shift_timestep, av_ca_audio_scale_shift_timestep, av_ca_a2v_gate_noise_timestep, av_ca_v2a_gate_noise_timestep, v_prompt_timestep, a_prompt_timestep, v_embedded_timestep, a_embedded_timestep, orig_shape
        )


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        #self.scale_shift_table = model.scale_shift_table
        self.norm_out = model.norm_out
        self.proj_out = model.proj_out
        #self.audio_scale_shift_table = model.audio_scale_shift_table
        self.audio_norm_out = model.audio_norm_out
        self.audio_proj_out = model.audio_proj_out
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    @torch.compiler.disable()
    def forward(self, inputs):
        (
            vx, ax, v_context, a_context, attention_mask, v_timestep, a_timestep, v_pe_cos, v_pe_sin, a_pe_cos, a_pe_sin, av_cross_video_freq_cos, av_cross_video_freq_sin, av_cross_audio_freq_cos, av_cross_audio_freq_sin,
            av_ca_video_scale_shift_timestep, av_ca_audio_scale_shift_timestep, av_ca_a2v_gate_noise_timestep, av_ca_v2a_gate_noise_timestep, v_prompt_timestep, a_prompt_timestep, v_embedded_timestep, a_embedded_timestep, orig_shape
        ) = inputs
        # TODO: this will return a list with 2 elements when audio is enabled (currently single tensor)
        return self._process_output(
            [vx, ax],
            [v_embedded_timestep, a_embedded_timestep],
            keyframe_idxs=None,
            orig_shape=orig_shape.tolist(),
        )
