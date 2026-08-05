import math
import sys
import os.path
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/OmniGen2'))

import torch
from torch import nn
import torch.nn.functional as F
import safetensors
from einops import rearrange
from deepspeed.utils.logging import logger

from models.base import BasePipeline, make_contiguous
from utils.common import AUTOCAST_DTYPE

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline as OriginalOmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2RotaryPosEmbed


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    t = math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
    return t


class OmniGen2Pipeline(BasePipeline):
    name = 'omnigen2'
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['OmniGen2TransformerBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        dtype = self.model_config['dtype']

        self.diffusers_pipeline = OriginalOmniGen2Pipeline.from_pretrained(self.model_config['diffusers_path'], torch_dtype=dtype, transformer=None)

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        if diffusers_path := self.model_config.get('diffusers_path', None):
            transformer = OmniGen2Transformer2DModel.from_pretrained(diffusers_path, torch_dtype=dtype, subfolder='transformer')
        else:
            raise NotImplementedError()

        self.diffusers_pipeline.transformer = transformer
        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.mllm]

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI format.
        peft_state_dict = {'diffusion_model.'+k: v for k, v in peft_state_dict.items()}
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

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

    def _get_qwen2_prompt_embeds(
        self,
        prompt,
        device=None,
        max_sequence_length=256,
    ):
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_inputs = self.processor.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        untruncated_ids = self.processor.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids.to(device)

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.processor.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because Gemma can only handle sequences up to"
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_attention_mask = text_inputs.attention_mask.to(device)
        prompt_embeds = self.mllm(
            text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-1]

        if self.mllm is not None:
            dtype = self.mllm.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, prompt_attention_mask

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(caption, is_video):
            # args are lists
            assert not any(is_video)
            prompt_embeds, prompt_attention_mask = self._get_qwen2_prompt_embeds(
                caption,
                device=text_encoder.device,
            )
            return {'prompt_embeds': prompt_embeds, 'prompt_attention_mask': prompt_attention_mask}
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        prompt_embeds = inputs['prompt_embeds']
        prompt_attention_mask = inputs['prompt_attention_mask']
        mask = inputs['mask']

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

        return (noisy_latents, 1-t, prompt_embeds, prompt_attention_mask), (target, mask)

    def to_layers(self):
        transformer = self.transformer
        layers = [InitialLayer(transformer)]
        for block in transformer.layers:
            layers.append(TransformerLayer(block))
        layers.append(FinalLayer(transformer))
        return layers


class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.time_caption_embed = model.time_caption_embed
        self.rope_embedder = model.rope_embedder
        self.context_refiner = model.context_refiner
        self.x_embedder = model.x_embedder
        self.ref_image_patch_embedder = model.ref_image_patch_embedder
        self.noise_refiner = model.noise_refiner
        self.ref_image_refiner = model.ref_image_refiner
        self.image_index_embedding = model.image_index_embedding
        self.model = [model]

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        hidden_states, timestep, text_hidden_states, text_attention_mask = inputs

        # 1. Condition, positional & patch embedding
        batch_size = len(hidden_states)

        assert hidden_states.ndim == 4
        hidden_states = [_hidden_states for _hidden_states in hidden_states]

        device = hidden_states[0].device

        temb, text_hidden_states = self.time_caption_embed(timestep, text_hidden_states, hidden_states[0].dtype)

        (
            hidden_states,
            ref_image_hidden_states,
            img_mask,
            ref_img_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
        ) = self.model[0].flat_and_pad_to_seq(hidden_states, ref_image_hidden_states=None)

        freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
            self.model[0].config.axes_dim_rope,
            self.model[0].config.axes_lens,
            theta=10000,
        )

        (
            context_rotary_emb,
            ref_img_rotary_emb,
            noise_rotary_emb,
            rotary_emb,
            encoder_seq_lengths,
            seq_lengths,
        ) = self.rope_embedder(
            freqs_cis,
            text_attention_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
            device,
        )

        # 2. Context refinement
        for layer in self.context_refiner:
            text_hidden_states = layer(text_hidden_states, text_attention_mask, context_rotary_emb)

        combined_img_hidden_states = self.model[0].img_patch_embed_and_refine(
            hidden_states,
            ref_image_hidden_states,
            img_mask,
            ref_img_mask,
            noise_rotary_emb,
            ref_img_rotary_emb,
            l_effective_ref_img_len,
            l_effective_img_len,
            temb,
        )

        # 3. Joint Transformer blocks
        max_seq_len = max(seq_lengths)

        attention_mask = hidden_states.new_zeros(batch_size, max_seq_len, dtype=torch.bool)
        joint_hidden_states = hidden_states.new_zeros(batch_size, max_seq_len, self.model[0].config.hidden_size)
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            attention_mask[i, :seq_len] = True
            joint_hidden_states[i, :encoder_seq_len] = text_hidden_states[i, :encoder_seq_len]
            joint_hidden_states[i, encoder_seq_len:seq_len] = combined_img_hidden_states[i, :seq_len - encoder_seq_len]

        hidden_states = joint_hidden_states

        img_sizes = torch.tensor(img_sizes, device=device)
        l_effective_img_len = torch.tensor(l_effective_img_len, device=device)
        seq_lengths = torch.tensor(seq_lengths, device=device)

        return make_contiguous(hidden_states, attention_mask, rotary_emb, temb, img_sizes, l_effective_img_len, seq_lengths)


class TransformerLayer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, attention_mask, rotary_emb, temb, img_sizes, l_effective_img_len, seq_lengths = inputs
        hidden_states = self.block(hidden_states, attention_mask, rotary_emb, temb)
        return make_contiguous(hidden_states, attention_mask, rotary_emb, temb, img_sizes, l_effective_img_len, seq_lengths)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.norm_out = model.norm_out
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, attention_mask, rotary_emb, temb, img_sizes, l_effective_img_len, seq_lengths = inputs

        # 4. Output norm & projection
        hidden_states = self.norm_out(hidden_states, temb)

        p = self.config.patch_size
        output = []
        for i, (img_size, img_len, seq_len) in enumerate(zip(img_sizes, l_effective_img_len, seq_lengths)):
            height, width = img_size
            output.append(rearrange(hidden_states[i][seq_len - img_len:seq_len], '(h w) (p1 p2 c) -> c (h p1) (w p2)', h=height // p, w=width // p, p1=p, p2=p))
        output = torch.stack(output, dim=0)
        return output