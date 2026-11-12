import os
import sys
import types
import random
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import torch
from torch import nn
import torch.nn.functional as F
import comfy_kitchen as ck

from models.base import ComfyPipeline, make_contiguous, PreprocessMediaFile, ModelWrapper
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift, one_at_a_time, round_down_to_multiple
from utils.offloading import ModelOffloader
import comfy.latent_formats
import comfy.model_management
import comfy.ldm.minimax.model
from comfy.ldm.modules.attention import ComfyAttention, AttentionTensorContainer, optimized_attention
from comfy.ldm.minimax.model import (
    PackedLayout, time_shift_sigma, VISUAL_COND_TIMESTEP, AUDIO_COND_TIMESTEP, patchify_video, pack_audio,
    rope_rotation_table, unpatchify_video, unpack_audio, MLP, AdalnProj,
)
import comfy.text_encoders.minimax

FRAMERATE = 24  # fixed for this model


class MiniMaxH3ClipModel(comfy.sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options={}):
        super().__init__(device=device, layer="last", layer_idx=None, textmodel_json_config={},
                         dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False,
                         model_class=comfy.text_encoders.minimax.MiniMaxQwen3VL, enable_attention_masks=True,
                         return_attention_masks=True, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs):
        out = super().encode_token_weights(token_weight_pairs)
        tags = getattr(self.transformer, "last_token_tags", None)
        if tags is not None:
            extra = out[2] if len(out) > 2 and isinstance(out[2], dict) else {}
            extra["minimax_token_tags"] = tags
            out = (out[0], out[1], extra)
        return out

# enable attention masks
comfy.text_encoders.minimax.MiniMaxH3ClipModel = MiniMaxH3ClipModel


def _mod_scale_shift(h, shift, scale, segments):
    dtype = h.dtype
    pieces = []
    # segments: [(start, stop, mod_row)] covering h contiguously.
    for a, b, row in segments:
        piece = h[:, a:b] * (1.0 + scale[:, row, None].to(dtype)) + shift[:, row, None].to(dtype)
        pieces.append(piece)
    return torch.cat(pieces, dim=1)

def _mod_gate(x, gate, other, segments):
    dtype = x.dtype
    pieces = []
    # other is the fresh attn/mlp output: accumulate the gated residual into the stream in place, one fused kernel per segment
    for a, b, row in segments:
        piece = x[:, a:b] + other[:, a:b] * gate[:, row, None].to(dtype)
        pieces.append(piece)
    return torch.cat(pieces, dim=1)

# patch these to remove in-place operations which break backward pass
comfy.ldm.minimax.model._mod_scale_shift = _mod_scale_shift
comfy.ldm.minimax.model._mod_gate = _mod_gate


class Attention(nn.Module):
    def __init__(self, hidden, heads, head_dim, eps, gate_compress=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = operations.Linear(hidden, inner * 3, bias=False, dtype=dtype, device=device)
        self.q_norm = operations.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.out_proj = operations.Linear(inner, hidden, bias=False, dtype=dtype, device=device)
        self.to_gate_compress = None
        if gate_compress:
            # VSA gate, unused by the dense forward; consumed by sparse attention patches
            self.to_gate_compress = operations.Linear(hidden, inner, bias=False, dtype=dtype, device=device)

    def forward(self, x, rope_freqs=None, attention_mask=None, transformer_options={}):
        b, s = x.shape[:2]
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        v = v.view(b, s, self.heads, self.head_dim)
        if rope_freqs is not None:
            # fused per-head RMSNorm + partial split-half rope, in  on the qkv buffer
            q = q.view(b, s, self.heads, self.head_dim)
            k = k.view(b, s, self.heads, self.head_dim)
            qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
            rot = rope_freqs.shape[-3] * 2
            # this is seemingly the only way to force eager
            q, k = ck.backends.eager.rope.rms_rope_split_half(q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        else:
            q = self.q_norm(q.view(b, s, self.heads, self.head_dim))
            k = self.k_norm(k.view(b, s, self.heads, self.head_dim))
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        q = AttentionTensorContainer(q.transpose(1, 2))
        k = AttentionTensorContainer(k.transpose(1, 2))
        v = AttentionTensorContainer(v.transpose(1, 2))
        out = optimized_attention(q, k, v, self.heads, mask=attention_mask, skip_reshape=True, transformer_options=transformer_options)
        return self.out_proj(out)

# Patch to force eager rms_rope_split_half so backward works. Even using offical methods to set 'eager' in comfy kitchen doesn't work.
comfy.ldm.minimax.model.Attention = Attention


class AdalnProj(nn.Module):
    def __init__(self, t_dim, hidden, expand, modalities, apply_silu=True,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.apply_silu = apply_silu
        self.linear = operations.Linear(t_dim, expand * hidden * modalities, bias=True, dtype=dtype, device=device)

    def forward(self, t_emb):
        # [B, M, t_dim] -> expand tensors of [B, M*modalities, hidden]
        x = self.linear(nn.functional.silu(t_emb) if self.apply_silu else t_emb)
        x = x.view(x.shape[0], x.shape[1] * self.modalities, self.expand * self.hidden)
        return x.chunk(self.expand, dim=-1)

# batch dimension
comfy.ldm.minimax.model.AdalnProj = AdalnProj


class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, t_dim, eps, qk_eps,
                 apply_silu=True, adaln_dtype=None, gate_compress=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.norm2 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, gate_compress=gate_compress,
                              dtype=dtype, device=device, operations=operations)
        self.mlp = MLP(hidden, ffn, dtype=dtype, device=device, operations=operations)
        self.adaln_proj = AdalnProj(t_dim, hidden, 6, 3, apply_silu=apply_silu,
                                    dtype=adaln_dtype if adaln_dtype is not None else dtype,
                                    device=device, operations=operations)

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None, attention_mask=None):
        attention = self.attn if attention is None else attention
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        x = _mod_gate(x, gate_msa, attention(h, rope_freqs=rope_freqs, transformer_options=transformer_options, attention_mask=attention_mask), mod_segments)
        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)

# add attention_mask
comfy.ldm.minimax.model.DiTBlock = DiTBlock


def _mod_row(vecs, row, dtype):
    # row is a mod-row index, or a per-token LongTensor of mod-row indices
    return vecs[:, row, None].to(dtype)


def pdd_grid_span(grid, sigma_v: float, sigma_v_next=None):
    """Head interval [start, stop) for PDD heads distilled on an explicit
    decreasing sigma grid (n + 1 points, MiniMax-H3-Image). A sampling step
    spans the grid points nearest sigma -> sigma_next (ComfyUI's rule); a
    single training sigma uses the one interval containing it."""
    n = len(grid) - 1
    if sigma_v_next is None:
        start = 0
        for k in range(n):
            # tolerance: a float32 sigma sitting on a grid point owns that head
            if grid[k] >= sigma_v - 1e-5:
                start = k
        return start, start + 1
    start, stop = (
        min(range(n + 1), key=lambda k: abs(grid[k] - float(s)))
        for s in (sigma_v, sigma_v_next)
    )
    start = min(start, n - 1)
    return start, max(stop, start + 1)


def build_pdd_grid_weights(sigma_v, grid, sigma_v_next=None) -> torch.Tensor:
    """(B, n) head weights on an explicit sigma grid: head 0 (the full head)
    at 1, the offset heads a step spans at their normalized dt share."""
    n = len(grid) - 1
    rows = []
    for i in range(len(sigma_v)):
        nxt = None if sigma_v_next is None else float(sigma_v_next[i])
        start, stop = pdd_grid_span(grid, float(sigma_v[i]), nxt)
        dt = [grid[k] - grid[k + 1] for k in range(start, stop)]
        w = torch.zeros(n, dtype=torch.float64)
        w[0] = 1.0
        first = max(start, 1)
        w[first:stop] = torch.tensor(dt[first - start :], dtype=torch.float64) / sum(dt)
        rows.append(w)
    return torch.stack(rows).float()


# ref: https://github.com/ostris/ai-toolkit/blob/main/extensions_built_in/diffusion_models/minimax_h3/src/transformer.py
def _blend_pdd_heads(out: torch.Tensor, weights: torch.Tensor, n: int):
    """(B, N, n * d) stacked head outputs, (B, n) weights -> (B, N, d)."""
    b, rows, _ = out.shape
    return torch.einsum(
        "brkd,bk->brd", out.view(b, rows, n, -1), weights.to(out.device, out.dtype)
    )


class FinalLayer(nn.Module):
    def __init__(self, hidden, t_dim, video_dim, audio_dim, eps, apply_silu=True, adaln_dtype=None, pdd_steps=1,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.video_dim = video_dim
        self.norm = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.adaln_proj = AdalnProj(t_dim, hidden, 2, 1, apply_silu=apply_silu,
                                    dtype=adaln_dtype if adaln_dtype is not None else dtype,
                                    device=device, operations=operations)
        # output heads are the checkpoint's fp32 island; norm/adaln are stored at model dtype.
        # a PDD checkpoint ships pdd_steps stacked heads per stream (a PDD LoRA grows plain heads to the same layout)
        self.video_out = operations.Linear(hidden, video_dim * pdd_steps, bias=True, dtype=torch.float32, device=device)
        self.audio_out = None
        if audio_dim is not None:
            self.audio_out = operations.Linear(hidden, audio_dim * pdd_steps, bias=True, dtype=torch.float32, device=device)

    @property
    def pdd_steps(self):
        return self.video_out.weight.shape[0] // self.video_dim

    def _modulate(self, x, shift, scale, seg):
        # seg: (start, stop, row) of a target stream, where row is a mod-row index or a per-token blend (see _mod_row)
        a, b, row = seg
        return (self.norm(x[:, a:b]) * (1.0 + _mod_row(scale, row, scale.dtype)) + _mod_row(shift, row, shift.dtype)).to(torch.float32)

    def forward(self, x, t_emb, video_seg, audio_seg, sigma, grids):
        # grids: one decreasing sigma grid (pdd_steps + 1 points) per output stream, video first; a
        # model without an audio stream passes one grid
        shift, scale = self.adaln_proj(t_emb)
        n = self.pdd_steps
        out = []
        for head, seg, grid in zip((self.video_out, self.audio_out), (video_seg, audio_seg), grids):
            h = self._modulate(x, shift, scale, seg)
            h = head(h)
            if n > 1:
                pdd_weights = build_pdd_grid_weights(sigma, grid)
                h = _blend_pdd_heads(h, pdd_weights, n)
            out.append(h)
        return out

# batch-enabled
comfy.ldm.minimax.model.FinalLayer = FinalLayer


def unpack_audio(rows, ch=2):
    b, s, C = rows.shape
    t = s // ch
    return rows.reshape(b, ch, t, C).permute(0, 3, 1, 2)

# batch-enabled
comfy.ldm.minimax.model.unpack_audio = unpack_audio


class PreprocessMediaFileMinimax(PreprocessMediaFile):
    def __init__(self, config):
        super().__init__(config, support_video=True, support_audio=True, framerate=FRAMERATE, audio_sample_rate=32000, round_height=32, round_width=32)

    # No offsets. VAE simply encodes each chunk of 17 frames into 5 latent frames, and then
    # slices off the last 3 latent frames from the full latent. 1 frame is special case:
    # 1 video frame -> 1 latent frame.
    def align_frames(self, frames):
        return max(round_down_to_multiple(frames, 17), 1)


class MinimaxH3Pipeline(ComfyPipeline):
    name = 'minimax_h3'
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['DiTBlock', 'RefinerBlock']
    keep_in_high_precision = ['time_embedder', 'audio_patch_proj', 'condition_proj', 'final_layer', 'rope.inv_freq', 'token_refiner', 'video_patch_proj', 'adaln_t_table']
    spatial_compression = 16
    channels = 24
    is_video_vae = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_video_vae = True
        self.latent_format = comfy.latent_formats.MiniMaxH3Video()
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.framerate = FRAMERATE

        self.cfg = self.model_config.get('cfg', 1.0)
        if self.cfg > 1:
            # Because uncond branch is no_grad() but tensors passed between pipeline parallel layers must require grad.
            assert self.config['pipeline_stages'] == 1, 'CFG training requires pipeline_stages=1'

        # combined video and (optional) audio VAE in one object
        def load_fn():
            sd = comfy.utils.load_torch_file(self.model_config['vae'])
            vae = comfy.sd.VAE(sd=sd)
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
            vae.vae_encode_crop_pixels = types.MethodType(vae_encode_crop_pixels, vae)
            vae_list = [vae]

            # audio VAE
            if audio_vae_path := self.model_config.get('audio_vae'):
                sd = comfy.utils.load_torch_file(audio_vae_path)
                vae = comfy.sd.VAE(sd=sd)
                vae.throw_exception_if_invalid()
                vae_list.append(vae)

            return vae_list

        self.vae = ModelWrapper(load_fn)

    def load_diffusion_model(self):
        # Model is so big, it's easy to OOM while loading with multiple GPUs.
        with one_at_a_time():
            rank = int(os.environ['LOCAL_RANK'])
            print(f'Loading model on rank {rank}')
            super().load_diffusion_model()

    # Override to exclude adaln, since full and pruned model have different sizes. Makes LoRA compatible with both.
    def get_target_modules(self, target_model):
        target_modules = set()
        for name, module in target_model.named_modules():
            if module.__class__.__name__ not in self.adapter_target_modules:
                continue
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if 'adaln' in full_submodule_name:
                    continue
                if isinstance(submodule, nn.Linear):
                    target_modules.add(full_submodule_name)
        return list(target_modules)

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFileMinimax(self.config)

    def vae_encode(self, img: torch.Tensor, audio: list):
        vaes = self.vae._model
        video_vae = vaes[0]
        audio_vae = vaes[1] if len(vaes) == 2 else None
        # move channel dim to end
        # works for both images (b c h w) and video (b c f h w)
        img = img.movedim(1, -1)
        latents = video_vae.encode(img)
        if self.latent_format is not None:
            # some older models do this in prepare_inputs() so it can be None
            latents = self.latent_format.process_in(latents)

        if audio_vae is not None:
            audio_latents = []
            for a in audio:
                if a is not None:
                    assert a.shape[0] == 1
                    a_latents = audio_vae.encode(a.movedim(1, -1))
                else:
                    a_latents = None
                audio_latents.append(a_latents)
        else:
            audio_latents = [None]*len(audio)

        return latents, audio_latents

    def get_call_vae_fn(self, vae):
        def fn(images: torch.Tensor, audio: list):
            if images.shape[2] > 1:
                # check videos for missing audio and warn
                missing_audio = sum(1 if a is None else 0 for a in audio)
                if missing_audio > 0:
                    print(f'WARNING: this batch had {missing_audio} videos without audio. The videos could have no audio track, or it could be a bug in the code.')
            images = images.to('cuda', self.dtype)
            latents, audio_latents = self.vae_encode(images, audio)
            return {'latents': latents, 'audio_latents': audio_latents}
        return fn

    def to_layers(self):
        diffusion_model = self.diffusion_model
        layers = [InitialLayer(diffusion_model)]
        for i, block in enumerate(diffusion_model.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(diffusion_model, self.cfg))
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

        bs, c, f, h, w = latents.shape
        device = latents.device

        audio_latents_list = inputs['audio_latents'] if 'audio_latents' in inputs else [None]*bs

        # prepare single global batched audio tensor, some may be None
        # videos are bucketed, so non-None audio should all be same length
        audio_shape = None
        audio_latents = None
        valid_audio = []
        for i, a in enumerate(audio_latents_list):
            if a is not None:
                if audio_shape is None:
                    audio_shape = a.shape
                    audio_latents = torch.empty((bs, *audio_shape[1:]), dtype=a.dtype, device=device)
                assert a.shape == audio_shape
                audio_latents[i, ...] = a
                valid_audio.append(True)
            else:
                valid_audio.append(False)
        valid_audio = torch.tensor(valid_audio, device=device)

        if audio_latents is None:
            audio_latents = torch.empty((bs, 32, 2, 0), dtype=latents.dtype, device=device)

        conds = self.get_conds(inputs)

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)

        if hasattr(self.diffusion_model, 'pdd_sigmas'):  # Image model
            valid_timesteps = self.diffusion_model.pdd_sigmas[:-1]  # ends with 0, slice it off
            # Ostris training adapter was trained for the inference timesteps only.
            t = torch.tensor(random.choices(valid_timesteps, k=bs))
        else:
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

            shift = self.model_config.get('shift', None)
            if f == 1:
                shift = self.model_config.get('image_shift', shift)
            if shift:
                t = (t * shift) / (1 + (shift - 1) * t)
            elif self.model_config.get('flux_shift', False):
                mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
                t = time_shift(mu, 1.0, t)

        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents

        # audio
        audio_noise = torch.randn_like(audio_latents)
        # fixed t -> audio_t mapping, matches the shift used in model code to derive the audio t from video t
        audio_t = time_shift_sigma(t, self.diffusion_model.sigma_shift_video, self.diffusion_model.sigma_shift_audio)
        audio_t_expanded = audio_t.view(-1, 1, 1, 1)
        noisy_audio_latents = (1 - audio_t_expanded) * audio_latents + audio_t_expanded * audio_noise
        audio_target = audio_noise - audio_latents

        if self.cfg > 1:
            tmp = {}
            for k in ('text_embeds_0', 'attention_mask_0'):
                v = self.uncond_dict[k]
                if k == 'attention_mask_0':
                    v = v.to(torch.bool)
                tmp[k] = v.unsqueeze(0).repeat(bs, *([1]*v.ndim))
            unconds = self.get_conds(tmp)
        else:
            unconds = tuple()

        return (noisy_latents, noisy_audio_latents, valid_audio, t, *conds, *unconds), (target, audio_target, mask)

    def get_loss_fn(self):
        @torch.autocast('cuda', enabled=False)
        def single_loss(output, target, mask=None):
            output = output.to(torch.float32)
            target = target.to(output.device, torch.float32)
            if 'huber_delta' in self.config:
                loss = F.huber_loss(output, target, reduction='none', delta=self.config['huber_delta'])
            elif 'smooth_l1_beta' in self.config:
                loss = F.smooth_l1_loss(output, target, reduction='none', beta=self.config['smooth_l1_beta'])
            else:
                loss = F.mse_loss(output, target, reduction='none')
            # empty tensor means no masking
            if mask is not None and mask.numel() > 0:
                mask = mask.to(output.device, torch.float32)
                loss *= mask
            return loss

        def loss_fn(outputs, label):
            output, audio_output = outputs
            target, audio_target, mask = label
            video_loss = single_loss(output, target, mask=mask)
            # audio_target may be padded to match another example's audio length from the same
            # gradient-accumulation group (see MinimaxH3Pipeline.prepare_inputs); audio_output's
            # length is authoritative since the model derives it from this example's own valid_audio flag.
            audio_target = audio_target[..., :audio_output.shape[-1]]
            audio_loss = single_loss(audio_output, audio_target)
            # make each token count the same for loss, regardless of modality
            # TODO: what is the best thing to do here? configurable audio loss scale?
            video_tokens = video_loss.numel()
            audio_tokens = audio_loss.numel()
            total_tokens = video_tokens + audio_tokens
            video_loss = video_loss.mean() * video_tokens / total_tokens
            audio_loss = audio_loss.mean() * audio_tokens / total_tokens
            loss = video_loss
            if audio_tokens > 0:  # avoid NaN for no audio
                loss = loss + audio_loss
            return loss
        return loss_fn

    def enable_block_swap(self, blocks_to_swap):
        diffusion_model = self.diffusion_model
        blocks = diffusion_model.blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        diffusion_model.blocks = None
        diffusion_model.to('cuda')
        diffusion_model.blocks = blocks
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
        if model.use_adaln_curves:
            self.adaln_t_table = model.adaln_t_table
        else:
            self.time_embedder = model.time_embedder
        self.audio_patch_proj = getattr(model, 'audio_patch_proj', None)
        self.video_patch_proj = model.video_patch_proj
        self.condition_proj = model.condition_proj
        self.rope = model.rope
        self.token_refiner = model.token_refiner
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    def make_packed_sequence(self, video_x, audio_x, valid_audio, t, context, context_mask):
        assert video_x.shape[0] == 1  # needs batch dimension with batch size 1

        if not valid_audio.item():
            audio_x = torch.empty([1, 32, 2, 0], device=video_x.device)

        transformer_options = {}
        payload = {}
        device = video_x.device
        dtype = context.dtype  # compute dtype

        latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        audio_t = audio_x.shape[-1]
        text_len = context.shape[1]
        # extra_conds prebuilds the layout once per sampling run
        layout = payload.get("layout")
        if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
            layout = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                                  keyframes=payload.get("keyframes"),
                                  refs=payload.get("refs"))

        # model_base passes model_sampling.timestep(sigma) = sigma * 1000
        shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))
        shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))
        sigma_v = (t.flatten()[0]).float().clamp(min=1e-6)  # I removed the 1000 from original comfy code since our t is [0, 1]
        t_v = float(1.0 - sigma_v.item())
        t_a = float(1.0 - time_shift_sigma(sigma_v.item(), shift_v, shift_a))

        # distinct timesteps are known analytically: text/pad follow video, cond rows pin near 1
        vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
        aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
        has_vis_cond = any(k in ("cond", "ref_img") for _, _, k in layout.segments)
        has_aud_cond = any(k == "ref_audio" for _, _, k in layout.segments)
        # Changed from original: not unique t values anymore. t_v, t_a could be same, in which case
        # deduping could cause diff number of unique t among items in a batch, breaking a stack().
        seg_t = {"text": t_v, "video": t_v, "audio": t_a}
        if has_vis_cond:
            seg_t.update({"cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug)})
        if has_aud_cond:
            seg_t.update({"ref_audio": max(t_a, aud_aug)})
        all_t = list(seg_t.values())
        t_row = {kind: i for i, kind in enumerate(seg_t.keys())}
        seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "ref_audio": 2}

        text_tags = payload.get("text_token_tags")
        mod_segments = []
        for a, b, kind in layout.segments:
            row_base = t_row[kind] * 3
            if kind == "text" and text_tags is not None:
                # the presentation text span mixes tags (vision pads carry the video modality) split into tag runs
                tags = text_tags.view(-1).tolist()
                run_start = 0
                for i in range(1, b - a + 1):
                    if i == b - a or tags[i] != tags[run_start]:
                        mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                        run_start = i
            else:
                mod_segments.append((a, b, row_base + seg_tag[kind]))

        # embed
        img_update = layout.img_update.to(device)
        audio_update = layout.audio_update.to(device)
        video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)
        audio_rows = pack_audio(audio_x.to(torch.float32))
        cond_video_rows = self._cond_video_rows(payload, device)
        cond_audio_rows = self._cond_audio_rows(payload, device)

        all_video_rows = video_rows
        if cond_video_rows is not None:
            all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
            all_video_rows[~img_update] = cond_video_rows
            all_video_rows[img_update] = video_rows
        all_audio_rows = audio_rows
        if cond_audio_rows is not None:
            all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
            all_audio_rows[~audio_update] = cond_audio_rows
            all_audio_rows[audio_update] = audio_rows

        video_embed = self.video_patch_proj(all_video_rows).to(dtype)
        if self.audio_patch_proj is not None:
            audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
        else:
            audio_embed = torch.empty([0, video_embed.shape[-1]], dtype=dtype, device=video_embed.device)

        text_states = context
        if text_states.shape[-1] != self.hidden_size:
            text_states = self.token_refiner(self.condition_proj(text_states),
                                             transformer_options=transformer_options)
        text_states = text_states[0]  # this happens after since token_refiner uses the new batch-enabled Attention

        # segments are contiguous: assemble by slices, embed rows follow segment order
        h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
        attention_mask = torch.ones((layout.seq_len,), dtype=context_mask.dtype, device=device)
        voff = aoff = 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == "text":
                h[a:b] = text_states
                attention_mask[a:b] = context_mask
            elif kind in ("cond", "ref_img", "video"):
                h[a:b] = video_embed[voff:voff + n]
                voff += n
            else:  # ref_audio / audio
                h[a:b] = audio_embed[aoff:aoff + n]
                aoff += n

        t_vals = torch.tensor(all_t, dtype=torch.float32, device=device)
        if self.use_adaln_curves:
            # adaln projections consume interpolated coordinates of the time-embedding curve
            table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
            pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)     # t in [0,1] -> fractional grid index, out-of-range t clamps to the curve ends
            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)   # lower grid row, max-clamp keeps t=1.0 on the last interval instead of reading past the table
            t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))  # blend the two rows by the fractional part
        else:
            t_emb = self.time_embedder(t_vals).to(dtype)

        # rotation table computed once per forward, consumed by the kitchen split-half rope
        rope_freqs = rope_rotation_table(self.rope_freqs(layout.position_ids, device), dtype)

        video_seg = next((a, b, t_row["video"]) for a, b, k in layout.segments if k == "video")
        audio_seg = next(
            ((a, b, t_row["audio"]) for a, b, k in layout.segments if k == "audio"),
            (0, 0, t_row["audio"]),
        )
        # pack these as ints so we can pass as tensor between pipeline parallel layers
        mod_segments = torch.tensor(mod_segments, dtype=torch.int32, device=h.device)
        extra_ints = torch.tensor([*video_seg, *audio_seg, latent_t, lat_h, lat_w], dtype=torch.int32, device=h.device)

        return h, attention_mask, t_emb, mod_segments, rope_freqs, extra_ints, sigma_v

    def make_layer_inputs(self, video_x, audio_x, valid_audio, t, context, context_mask):
        first_mod_seqments = None
        h_list, attention_mask_list, t_emb_list, sigma_v_list = [], [], [], []
        for items in zip(video_x, audio_x, valid_audio, t, context, context_mask):
            h, attention_mask, t_emb, mod_segments, rope_freqs, extra_ints, sigma_v = self.make_packed_sequence(*(x.unsqueeze(0) for x in items))
            if first_mod_seqments is None:
                first_mod_seqments = mod_segments
            else:
                # A critical assertion: the batch-enabled AdaLN code only works if all mod_segments are the same (which they will be for this training code).
                assert (mod_segments == first_mod_seqments).all()
            h_list.append(h)
            attention_mask_list.append(attention_mask)
            t_emb_list.append(t_emb)
            sigma_v_list.append(sigma_v)

        h = torch.stack(h_list, dim=0)
        attention_mask = torch.stack(attention_mask_list, dim=0)[:, None, None, :]  # 4D bool for SDPA
        t_emb = torch.stack(t_emb_list)
        sigma_v = torch.stack(sigma_v_list)
        outputs = make_contiguous(h, attention_mask, t_emb, mod_segments, rope_freqs, extra_ints, sigma_v)
        for item in outputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        return outputs

    # TODO: will need to handle text_token_tags (and probably more) for reference images. it's all 1s for pure text prompt
    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    @torch.compiler.disable
    def forward(self, inputs):
        video_x, audio_x, valid_audio, t, *variable = inputs
        if len(variable) == 2:
            context, context_mask = variable
            has_uncond_branch = False
        else:
            context, context_mask, context_uncond, context_mask_uncond = variable
            has_uncond_branch = True

        outputs = self.make_layer_inputs(video_x, audio_x, valid_audio, t, context, context_mask)
        if has_uncond_branch:
            # TODO: what happens if we have gradient on the entire uncond branch?
            with torch.no_grad():
                h_uncond, attention_mask_uncond, _, mod_segments_uncond, rope_freqs_uncond, extra_ints_uncond = self.make_layer_inputs(video_x, audio_x, valid_audio, t, context_uncond, context_mask_uncond)
            uncond_tensors = (h_uncond, attention_mask_uncond, mod_segments_uncond, rope_freqs_uncond, extra_ints_uncond)
            for x in uncond_tensors:
                x.no_backward = True
            outputs = (*outputs, *uncond_tensors)
        return outputs

class TransformerLayer(nn.Module):
    def __init__(self, layer, block_idx, offloader):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        h, attention_mask, t_emb, mod_segments, rope_freqs, extra_ints, sigma_v, *uncond_tensors = inputs

        self.offloader.wait_for_block(self.block_idx)

        h = self.layer(h, t_emb, mod_segments.tolist(), rope_freqs, attention_mask=attention_mask)
        if len(uncond_tensors) > 0:
            h_uncond, attention_mask_uncond, mod_segments_uncond, rope_freqs_uncond, extra_ints_uncond, sigma_v_uncond = uncond_tensors
            # TODO: in the unsloth checkpointer, it only sees no_backward for all 5 tensors on the very first layer. For all other layers,
            # it's only set on h_uncond. This works, and it avoids saving the one large tensor, but why is it like this?
            if h_uncond is None:
                # Backward pass recomputation, and we didn't save these tensors for backward. But we need same number of return values.
                uncond_tensors = (None,)*len(uncond_tensors)
            else:
                with torch.no_grad():
                    h_uncond = self.layer(h_uncond, t_emb, mod_segments_uncond.tolist(), rope_freqs_uncond, attention_mask=attention_mask_uncond)
                uncond_tensors = (h_uncond, attention_mask_uncond, mod_segments_uncond, rope_freqs_uncond, extra_ints_uncond, sigma_v_uncond)
                for x in uncond_tensors:
                    x.no_backward = True

        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(h, attention_mask, t_emb, mod_segments, rope_freqs, extra_ints, sigma_v, *uncond_tensors)


class FinalLayer(nn.Module):
    def __init__(self, model, cfg):
        super().__init__()
        self.final_layer = model.final_layer
        self.model = [model]
        self.cfg = cfg
        self.pdd_sigmas = getattr(model, 'pdd_sigmas', None)

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    @torch.compiler.disable
    def forward(self, inputs):
        h, attention_mask, t_emb, mod_segments, rope_freqs, extra_ints, sigma_v, *uncond_tensors = inputs
        video_seg = extra_ints[:3]
        audio_seg = extra_ints[3:6]
        latent_t, lat_h, lat_w = extra_ints[6:]

        if self.pdd_sigmas is not None:
            # Image model
            grids = (self.pdd_sigmas,)
            v, = self.final_layer(h, t_emb, video_seg, audio_seg, sigma_v.detach(), grids)
            a = torch.empty([v.shape[0], 0, 32], device=v.device)
        else:
            # Video model
            grids = (None, None)  # final_layer doesn't use these, and it should fail if it did for some reason.
            v, a = self.final_layer(h, t_emb, video_seg, audio_seg, sigma_v.detach(), grids)
        video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
        audio_out = unpack_audio(a)

        if len(uncond_tensors) > 0:
            assert self.cfg > 1
            assert self.pdd_sigmas is None  # should be training the image model with an adapter, never with CFG-augmented output
            h_uncond, attention_mask_uncond, mod_segments_uncond, rope_freqs_uncond, extra_ints_uncond, sigma_v_uncond = uncond_tensors
            video_seg = extra_ints_uncond[:3]
            audio_seg = extra_ints_uncond[3:6]
            latent_t, lat_h, lat_w = extra_ints_uncond[6:]
            with torch.no_grad():
                v_uncond, a_uncond = self.final_layer(h_uncond, t_emb, video_seg, audio_seg)
            video_out_uncond = unpatchify_video(v_uncond, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
            audio_out_uncond = unpack_audio(a_uncond)
            # "raw" velocity from guidance-distilled output and own uncond (just rearrange CFG equation)
            video_out = (video_out + (self.cfg-1)*video_out_uncond) / self.cfg
            audio_out = (audio_out + (self.cfg-1)*audio_out_uncond) / self.cfg

        # Make float tensors have a valid grad, avoiding a "assert buffer.grad is not None" error with pipeline parallelism.
        # TODO: this is so ugly, is there a better way?
        zero_but_has_grad = sum(x.flatten()[0]*0 for x in (t_emb, rope_freqs, sigma_v))

        # We don't scale the audio's velocity like in inference, since that is an inference-only
        # hack in order to use a single sampling schedule.
        return -video_out + zero_but_has_grad, -audio_out


# class Wrapper(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, inputs):
#         video_x, timestep, context, context_mask = inputs
#         audio_x = torch.empty([1, 32, 2, 0], device=video_x.device)
#         out = self.model((video_x, audio_x), timestep*1000, context=context)
#         return out[0]