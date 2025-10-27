import json

import torch
from torch import nn
import torch.nn.functional as F
import transformers
import diffusers
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from models.base import BasePipeline, make_contiguous
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift, iterate_safetensors, load_state_dict
from utils.offloading import ModelOffloader


KEEP_IN_HIGH_PRECISION = ['register_tokens', 'pos_embed', 'context_embedder', 'time_step_embed', 'time_step_proj', 'norm_out', 'proj_out']


class AuraFlowPipeline(BasePipeline):
    name = 'auraflow'
    checkpointable_layers = ['DoubleTransformerLayer', 'SingleTransformerLayer']
    adapter_target_modules = ['AuraFlowJointTransformerBlock', 'AuraFlowSingleTransformerBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.max_sequence_length = self.model_config['max_sequence_length']
        dtype = self.model_config['dtype']

        with open('configs/auraflow/vae_config.json') as f:
            vae_config = json.load(f)
        with init_empty_weights():
            vae = diffusers.AutoencoderKL.from_config(vae_config)
        for key, tensor in iterate_safetensors(self.model_config['vae_path']):
            set_module_tensor_to_device(vae, key, device='cpu', dtype=dtype, value=tensor)

        tokenizer = transformers.AutoTokenizer.from_pretrained('configs/auraflow/tokenizer', local_files_only=True)

        text_encoder_config = transformers.UMT5Config.from_pretrained('configs/auraflow/text_encoder', local_files_only=True)
        with init_empty_weights():
            text_encoder = transformers.UMT5EncoderModel(text_encoder_config)
        for key, tensor in iterate_safetensors(self.model_config['text_encoder_path']):
            set_module_tensor_to_device(text_encoder, key, device='cpu', dtype=dtype, value=tensor)
        text_encoder.requires_grad_(False)

        self.diffusers_pipeline = diffusers.AuraFlowPipeline(
            scheduler=None,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=None,
        )

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        with open('configs/auraflow/transformer_config.json') as f:
            json_config = json.load(f)
        with init_empty_weights():
            transformer = diffusers.AuraFlowTransformer2DModel.from_config(json_config)
        state_dict = {
            k.replace('model.', ''): v
            for k, v in load_state_dict(self.model_config['transformer_path']).items()
        }
        state_dict = diffusers.loaders.single_file_utils.convert_auraflow_transformer_checkpoint_to_diffusers(state_dict)
        for key, tensor in state_dict.items():
            dtype_to_use = dtype if any(keyword in key for keyword in KEEP_IN_HIGH_PRECISION) or tensor.ndim == 1 else transformer_dtype
            set_module_tensor_to_device(transformer, key, device='cpu', dtype=dtype_to_use, value=tensor)

        transformer.train()
        self.diffusers_pipeline.transformer = transformer

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder]

    def save_adapter(self, save_dir, peft_state_dict):
        self.save_lora_weights(save_dir, transformer_lora_layers=peft_state_dict)

    def get_call_vae_fn(self, vae):
        def fn(image):
            p = next(vae.parameters())
            image = image.to(p.device, p.dtype)
            latents = vae.encode(image.to(vae.device, vae.dtype)).latent_dist.mode()
            if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                latents = latents - vae.config.shift_factor
            latents = latents * vae.config.scaling_factor
            return {'latents': latents}
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(captions, is_video):
            # args are lists
            prompt_embeds, prompt_attention_mask, _, _ = self.encode_prompt(captions, max_sequence_length=self.max_sequence_length, device=text_encoder.device)
            return {'prompt_embeds': prompt_embeds}
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']
        prompt_embeds = inputs['prompt_embeds']

        bs, channels, h, w = latents.shape

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
        noisy_latents = (1 - t_expanded)*latents + t_expanded*noise
        target = noise - latents

        return (noisy_latents, prompt_embeds, t), (target, mask)

    def to_layers(self):
        transformer = self.transformer
        layers = [InitialLayer(transformer)]
        for i, block in enumerate(transformer.joint_transformer_blocks):
            layers.append(DoubleTransformerLayer(block))
        layers.append(concat_hidden_states)
        for i, block in enumerate(transformer.single_transformer_blocks):
            layers.append(SingleTransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(transformer))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        transformer = self.transformer
        blocks = transformer.single_transformer_blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        transformer.single_transformer_blocks = None
        transformer.to('cuda')
        transformer.single_transformer_blocks = blocks
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
        self.pos_embed = model.pos_embed
        self.time_step_embed = model.time_step_embed
        self.time_step_proj = model.time_step_proj
        self.context_embedder = model.context_embedder
        self.register_tokens = model.register_tokens


    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)

        hidden_states, encoder_hidden_states, timestep = inputs

        height, width = hidden_states.shape[-2:]

        # Apply patch embedding, timestep embedding, and project the caption embeddings.
        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_step_embed(timestep).to(dtype=next(self.parameters()).dtype)
        temb = self.time_step_proj(temb)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        encoder_hidden_states = torch.cat(
            [self.register_tokens.repeat(encoder_hidden_states.size(0), 1, 1), encoder_hidden_states], dim=1
        )

        height_width = torch.tensor([height, width], device=hidden_states.device)

        return make_contiguous(hidden_states, encoder_hidden_states, temb, height_width)


class DoubleTransformerLayer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, temb, height_width = inputs

        encoder_hidden_states, hidden_states = self.block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
        )

        return make_contiguous(hidden_states, encoder_hidden_states, temb, height_width)


def concat_hidden_states(inputs):
    hidden_states, encoder_hidden_states, temb, height_width = inputs
    encoder_seq_len = encoder_hidden_states.size(1)
    combined_hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    return make_contiguous(combined_hidden_states, temb, height_width, torch.tensor(encoder_seq_len, device=hidden_states.device))


class SingleTransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        combined_hidden_states, temb, height_width, encoder_seq_len = inputs

        self.offloader.wait_for_block(self.block_idx)
        combined_hidden_states = self.block(hidden_states=combined_hidden_states, temb=temb)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(combined_hidden_states, temb, height_width, encoder_seq_len)


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
        combined_hidden_states, temb, height_width, encoder_seq_len = inputs
        hidden_states = combined_hidden_states[:, encoder_seq_len.item():]
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        height = height_width[0].item()
        width = height_width[1].item()
        patch_size = self.config.patch_size
        out_channels = self.config.out_channels
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], out_channels, height * patch_size, width * patch_size)
        )
        return output
