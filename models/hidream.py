import math
import os.path
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/HiDream'))

import torch
from torch import nn
import torch.nn.functional as F
from deepspeed.utils.logging import logger
from safetensors.torch import save_file
import transformers
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from einops import repeat
import diffusers

from models.base import BasePipeline, make_contiguous
from utils.common import AUTOCAST_DTYPE, empty_cuda_cache
from utils.offloading import ModelOffloader
from hi_diffusers import HiDreamImagePipeline, HiDreamImageTransformer2DModel
from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from hi_diffusers.models.moe import MoEGate


KEEP_IN_HIGH_PRECISION = ['norm', 'bias', 't_embedder', 'p_embedder', 'x_embedder', 'final_layer', 'gate']


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


class HiDreamPipeline(BasePipeline):
    name = 'hidream'

    checkpointable_layers = [
        'TransformerWrapper',
        'SingleTransformerWrapper',
    ]

    adapter_target_modules = ['HiDreamImageTransformerBlock', 'HiDreamImageSingleTransformerBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.offloader_double = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.offloader_single = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)

        dtype = self.model_config['dtype']
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        llama3_path = self.model_config['llama3_path']
        tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
            llama3_path,
            use_fast=False
        )
        self.diffusers_pipeline = HiDreamImagePipeline.from_pretrained(
            self.model_config['diffusers_path'],
            scheduler=scheduler,
            tokenizer_4=tokenizer_4,
            text_encoder_4=None,
            torch_dtype=dtype
        )

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        llama3_path = self.model_config['llama3_path']
        if self.model_config.get('llama3_4bit', False):
            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=dtype,
            )
        else:
            quantization_config = None
        text_encoder_4 = LlamaForCausalLM.from_pretrained(
            llama3_path,
            output_hidden_states=True,
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )
        for p in text_encoder_4.parameters():
            p.requires_grad_(False)
            p.data = p.data.to('cpu')
        empty_cuda_cache()
        self.diffusers_pipeline.text_encoder_4 = text_encoder_4

        if transformer_dtype == 'nf4':
            quantization_config = diffusers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=dtype,
                llm_int8_skip_modules=KEEP_IN_HIGH_PRECISION,
            )
        else:
            quantization_config = None
        self.diffusers_pipeline.transformer = HiDreamImageTransformer2DModel.from_pretrained(
            self.model_config['diffusers_path'],
            subfolder='transformer',
            torch_dtype=dtype,
            quantization_config=quantization_config,
        )
        if transformer_dtype != 'nf4':
            for name, p in self.transformer.named_parameters():
                if not (any(x in name for x in KEEP_IN_HIGH_PRECISION)):
                    p.data = p.data.to(transformer_dtype)

        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name

        # Critically important! Official code saves MoE aux losses in global state if alpha > 0. Without special handling of
        # this, it causes massive memory leak during backward pass and immediately OOMs you.
        for module in self.transformer.modules():
            if isinstance(module, MoEGate):
                module.alpha = 0

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder, self.text_encoder_2, self.text_encoder_3]

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI format.
        peft_state_dict = {'diffusion_model.'+k: v for k, v in peft_state_dict.items()}
        save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

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
                pooled_prompt_embeds_1 = self._get_clip_prompt_embeds(
                    self.tokenizer,
                    self.text_encoder,
                    prompt=caption,
                    device=text_encoder.device,
                )
                return {'pooled_prompt_embeds_1': pooled_prompt_embeds_1}
            return fn
        elif text_encoder == self.text_encoder_2:
            def fn(caption, is_video):
                assert not any(is_video)
                pooled_prompt_embeds_2 = self._get_clip_prompt_embeds(
                    self.tokenizer_2,
                    self.text_encoder_2,
                    prompt=caption,
                    device=text_encoder.device,
                )
                return {'pooled_prompt_embeds_2': pooled_prompt_embeds_2}
            return fn
        elif text_encoder == self.text_encoder_3:
            def fn(caption, is_video):
                assert not any(is_video)
                t5_prompt_embeds = self._get_t5_prompt_embeds(
                    prompt=caption,
                    device=text_encoder.device,
                )
                return {'t5_prompt_embeds': t5_prompt_embeds}
            return fn
        else:
            raise RuntimeError(f'Text encoder {text_encoder.__class__} does not have a function to call it')

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        pooled_prompt_embeds_1 = inputs['pooled_prompt_embeds_1']
        pooled_prompt_embeds_2 = inputs['pooled_prompt_embeds_2']
        t5_prompt_embeds = inputs['t5_prompt_embeds']
        mask = inputs['mask']
        caption = inputs['caption']
        pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1)

        max_llama3_sequence_length = self.model_config.get('max_llama3_sequence_length', 128)
        max_llama3_sequence_length = min(max_llama3_sequence_length, self.tokenizer_4.model_max_length)
        text_inputs = self.tokenizer_4(
            caption,
            padding="max_length",
            max_length=max_llama3_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        llama_input_ids = text_inputs.input_ids
        llama_attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer_4(caption, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= llama_input_ids.shape[-1] and not torch.equal(llama_input_ids, untruncated_ids):
            removed_text = self.tokenizer_4.batch_decode(untruncated_ids[:, max_llama3_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_llama3_sequence_length} tokens: {removed_text}"
            )

        bs, c, h, w = latents.shape
        latents, _, img_sizes = self.transformer.patchify(latents, self.transformer.max_seq)

        if mask is not None:
            mask = mask.unsqueeze(1).expand((-1, c, -1, -1))  # make mask (bs, c, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask, _, _ = self.transformer.patchify(mask, self.transformer.max_seq)

        pH, pW = img_sizes[0]
        img_ids = torch.zeros(pH, pW, 3, device=latents.device)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH, device=latents.device)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW, device=latents.device)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

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
        # Target multiplied by -1 compared to Flux.
        target = x_1 - x_0

        timesteps = t * 1000
        return (x_t, img_ids, timesteps, pooled_prompt_embeds, t5_prompt_embeds, llama_input_ids, llama_attention_mask), (target, mask)

    def to_layers(self):
        transformer = self.transformer
        layers = [LlamaLayer(self.text_encoder_4), InitialLayer(transformer)]
        global_block_idx = 0
        for i, block in enumerate(transformer.double_stream_blocks):
            layers.append(TransformerWrapper(block, i, global_block_idx, self.offloader_double))
            global_block_idx += 1
        layers.append(concatenate_hidden_states)
        for i, block in enumerate(transformer.single_stream_blocks):
            layers.append(SingleTransformerWrapper(block, i, global_block_idx, self.offloader_single))
            global_block_idx += 1
        layers.append(OutputLayer(transformer))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        transformer = self.transformer
        double_blocks = transformer.double_stream_blocks
        single_blocks = transformer.single_stream_blocks
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
        transformer.double_stream_blocks = None
        transformer.single_stream_blocks = None
        transformer.to('cuda')
        self.text_encoder_4.to('cuda')
        transformer.double_stream_blocks = double_blocks
        transformer.single_stream_blocks = single_blocks
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


class LlamaLayer(nn.Module):
    def __init__(self, llama_model):
        super().__init__()
        self.llama_model = llama_model

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, img_ids, timesteps, pooled_embeds, t5_prompt_embeds, llama_input_ids, llama_attention_mask = inputs
        with torch.no_grad():
            outputs = self.llama_model(
                llama_input_ids,
                attention_mask=llama_attention_mask,
                output_hidden_states=True,
            )

        llama3_prompt_embeds = outputs.hidden_states[1:]
        llama3_prompt_embeds = torch.stack(llama3_prompt_embeds, dim=0).detach()
        result = make_contiguous(hidden_states, img_ids, timesteps, pooled_embeds, t5_prompt_embeds, llama3_prompt_embeds)
        for item in result:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        return result


class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.t_embedder = model.t_embedder
        self.p_embedder = model.p_embedder
        self.x_embedder = model.x_embedder
        self.caption_projection = model.caption_projection
        self.pe_embedder = model.pe_embedder
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, img_ids, timesteps, pooled_embeds, t5_prompt_embeds, llama3_prompt_embeds = inputs

        batch_size = hidden_states.shape[0]
        hidden_states_type = hidden_states.dtype

        timesteps = self.expand_timesteps(timesteps, batch_size, hidden_states.device)
        timesteps = self.t_embedder(timesteps, hidden_states_type)
        p_embedder = self.p_embedder(pooled_embeds)
        adaln_input = timesteps + p_embedder

        hidden_states = self.x_embedder(hidden_states)

        T5_encoder_hidden_states = t5_prompt_embeds
        encoder_hidden_states = llama3_prompt_embeds
        encoder_hidden_states = [encoder_hidden_states[k] for k in self.llama_layers]

        assert self.caption_projection is not None
        if self.caption_projection is not None:
            new_encoder_hidden_states = []
            for i, enc_hidden_state in enumerate(encoder_hidden_states):
                enc_hidden_state = self.caption_projection[i](enc_hidden_state)
                enc_hidden_state = enc_hidden_state.view(batch_size, -1, hidden_states.shape[-1])
                new_encoder_hidden_states.append(enc_hidden_state)
            encoder_hidden_states = new_encoder_hidden_states
            T5_encoder_hidden_states = self.caption_projection[-1](T5_encoder_hidden_states)
            T5_encoder_hidden_states = T5_encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
            encoder_hidden_states.append(T5_encoder_hidden_states)

        txt_ids = torch.zeros(
            batch_size,
            encoder_hidden_states[-1].shape[1] + encoder_hidden_states[-2].shape[1] + encoder_hidden_states[0].shape[1],
            3,
            device=img_ids.device, dtype=img_ids.dtype
        )
        ids = torch.cat((img_ids, txt_ids), dim=1)
        rope = self.pe_embedder(ids)

        initial_encoder_hidden_states = torch.cat([encoder_hidden_states[-1], encoder_hidden_states[-2]], dim=1)
        llama_encoder_hidden_states = torch.stack(encoder_hidden_states[:-1], dim=0)

        # With nf4 quantization, tensors can end up float32, which breaks flash attention later, so we cast it here.
        hidden_states = hidden_states.to(AUTOCAST_DTYPE)
        initial_encoder_hidden_states = initial_encoder_hidden_states.to(AUTOCAST_DTYPE)
        llama_encoder_hidden_states = llama_encoder_hidden_states.to(AUTOCAST_DTYPE)
        adaln_input = adaln_input.to(AUTOCAST_DTYPE)

        return make_contiguous(hidden_states, initial_encoder_hidden_states, llama_encoder_hidden_states, adaln_input, rope)


class TransformerWrapper(nn.Module):
    def __init__(self, block, block_idx, global_block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.global_block_idx = global_block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, initial_encoder_hidden_states, llama_encoder_hidden_states, adaln_input, rope = inputs

        self.offloader.wait_for_block(self.block_idx)
        initial_encoder_hidden_states_seq_len = initial_encoder_hidden_states.shape[1]
        cur_llama31_encoder_hidden_states = llama_encoder_hidden_states[self.global_block_idx]
        cur_encoder_hidden_states = torch.cat([initial_encoder_hidden_states, cur_llama31_encoder_hidden_states], dim=1)
        hidden_states, initial_encoder_hidden_states = self.block(
            image_tokens=hidden_states,
            text_tokens=cur_encoder_hidden_states,
            adaln_input=adaln_input,
            rope=rope,
        )
        initial_encoder_hidden_states = initial_encoder_hidden_states[:, :initial_encoder_hidden_states_seq_len]
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(hidden_states, initial_encoder_hidden_states, llama_encoder_hidden_states, adaln_input, rope)


def concatenate_hidden_states(inputs):
    hidden_states, initial_encoder_hidden_states, llama_encoder_hidden_states, adaln_input, rope = inputs
    image_tokens_seq_len = torch.tensor(hidden_states.shape[1], device=hidden_states.device)
    hidden_states = torch.cat([hidden_states, initial_encoder_hidden_states], dim=1)
    return hidden_states, llama_encoder_hidden_states, adaln_input, rope, image_tokens_seq_len


class SingleTransformerWrapper(nn.Module):
    def __init__(self, block, block_idx, global_block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.global_block_idx = global_block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, llama_encoder_hidden_states, adaln_input, rope, image_tokens_seq_len = inputs

        self.offloader.wait_for_block(self.block_idx)
        hidden_states_seq_len = hidden_states.shape[1]
        cur_llama31_encoder_hidden_states = llama_encoder_hidden_states[self.global_block_idx]
        hidden_states = torch.cat([hidden_states, cur_llama31_encoder_hidden_states], dim=1)
        hidden_states = self.block(
            image_tokens=hidden_states,
            text_tokens=None,
            adaln_input=adaln_input,
            rope=rope,
        )
        hidden_states = hidden_states[:, :hidden_states_seq_len]
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(hidden_states, llama_encoder_hidden_states, adaln_input, rope, image_tokens_seq_len)


class OutputLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, llama_encoder_hidden_states, adaln_input, rope, image_tokens_seq_len = inputs
        hidden_states = hidden_states[:, :image_tokens_seq_len.item(), ...]
        return self.final_layer(hidden_states, adaln_input)