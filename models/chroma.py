import math
from dataclasses import dataclass
import sys
import os.path
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/flow'))

import diffusers
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import save_file
from accelerate import init_empty_weights

from models.base import BasePipeline, make_contiguous
from utils.common import AUTOCAST_DTYPE, load_state_dict
from utils.offloading import ModelOffloader
from src.models.chroma.model import Chroma, chroma_params, modify_mask_to_attend_padding
from src.models.chroma.module.layers import timestep_embedding, distribute_modulations, ModulationOut


KEEP_IN_HIGH_PRECISION = ['norm', 'bias', 'img_in', 'txt_in', 'distilled_guidance_layer', 'final_layer']


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


@dataclass
class ModulationOutSpec:
    shift: slice
    scale: slice
    gate: slice


# Adapted from the function of the same name in the original training code, but only computes the slices,
# doesn't actually slice the tensor yet. I did this because pipeline parallelism makes it nearly impossible
# to pass a dictionary between GPUs. So we have to pass the pre-sliced tensor then extract the slice on
# the layer right before it's used.
def distribute_modulations():
    block_dict = {}

    # HARD CODED VALUES! lookup table for the generated vectors
    # TODO: move this into chroma config!
    # Add 38 single mod blocks
    for i in range(38):
        key = f"single_blocks.{i}.modulation.lin"
        block_dict[key] = None

    # Add 19 image double blocks
    for i in range(19):
        key = f"double_blocks.{i}.img_mod.lin"
        block_dict[key] = None

    # Add 19 text double blocks
    for i in range(19):
        key = f"double_blocks.{i}.txt_mod.lin"
        block_dict[key] = None

    # Add the final layer
    block_dict["final_layer.adaLN_modulation.1"] = None
    # 6.2b version
    block_dict["lite_double_blocks.4.img_mod.lin"] = None
    block_dict["lite_double_blocks.4.txt_mod.lin"] = None

    idx = 0  # Index to keep track of the vector slices

    for key in block_dict.keys():
        if "single_blocks" in key:
            # Single block: 1 ModulationOut
            block_dict[key] = ModulationOutSpec(
                shift=slice(idx, idx+1),
                scale=slice(idx+1, idx+2),
                gate=slice(idx+2, idx+3),
            )
            idx += 3  # Advance by 3 vectors

        elif "img_mod" in key:
            # Double block: List of 2 ModulationOut
            double_block = []
            for _ in range(2):  # Create 2 ModulationOut objects
                double_block.append(
                    ModulationOutSpec(
                        shift=slice(idx, idx+1),
                        scale=slice(idx+1, idx+2),
                        gate=slice(idx+2, idx+3),
                    )
                )
                idx += 3  # Advance by 3 vectors per ModulationOut
            block_dict[key] = double_block

        elif "txt_mod" in key:
            # Double block: List of 2 ModulationOut
            double_block = []
            for _ in range(2):  # Create 2 ModulationOut objects
                double_block.append(
                    ModulationOutSpec(
                        shift=slice(idx, idx+1),
                        scale=slice(idx+1, idx+2),
                        gate=slice(idx+2, idx+3),
                    )
                )
                idx += 3  # Advance by 3 vectors per ModulationOut
            block_dict[key] = double_block

        elif "final_layer" in key:
            # Final layer: 1 ModulationOut
            block_dict[key] = [
                slice(idx, idx+1),
                slice(idx+1, idx+2),
            ]
            idx += 2  # Advance by 3 vectors

    return block_dict

modulation_distribute_dict = distribute_modulations()


class ChromaPipeline(BasePipeline):
    name = 'chroma'

    checkpointable_layers = [
        'TransformerWrapper',
        'SingleTransformerWrapper',
    ]

    adapter_target_modules = ['DoubleStreamBlock', 'SingleStreamBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.offloader_double = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.offloader_single = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)

        dtype = self.model_config['dtype']
        self.diffusers_pipeline = diffusers.FluxPipeline.from_pretrained(self.model_config['diffusers_path'], torch_dtype=dtype, transformer=None)

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)
        with init_empty_weights():
            transformer = Chroma(chroma_params)
        transformer.load_state_dict(load_state_dict(self.model_config['transformer_path']), assign=True)

        for name, p in transformer.named_parameters():
            if not any(x in name for x in KEEP_IN_HIGH_PRECISION):
                p.data = p.data.to(transformer_dtype)

        self.diffusers_pipeline.transformer = transformer
        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder_2]

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI format.
        peft_state_dict = {'diffusion_model.'+k: v for k, v in peft_state_dict.items()}
        save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def save_model(self, save_dir, diffusers_sd):
        save_file(diffusers_sd, save_dir / 'model.safetensors', metadata={"format": "pt"})

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            latents = vae.encode(tensor.to(vae.device, vae.dtype)).latent_dist.sample()
            if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                latents = latents - vae.config.shift_factor
            latents = latents * vae.config.scaling_factor
            return {'latents': latents}
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(caption, is_video):
            # args are lists
            assert not any(is_video)
            max_sequence_length = 512
            text_inputs = self.tokenizer_2(
                caption,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer_2(caption, padding="longest", return_tensors="pt").input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
                print(
                    "The following part of your input was truncated because `max_sequence_length` is set to "
                    f" {max_sequence_length} tokens: {removed_text}"
                )
            prompt_embeds = self.text_encoder_2(text_input_ids.to(text_encoder.device), output_hidden_states=False)[0]
            return {'t5_embed': prompt_embeds, 't5_attention_mask': text_inputs.attention_mask}
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        t5_embed = inputs['t5_embed']
        t5_attention_mask = inputs['t5_attention_mask']
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
        # guidance needs to be 0 on this model
        guidance_vec = torch.zeros((x_t.shape[0],), device=x_t.device, dtype=torch.float32)

        return (x_t, t5_embed, t5_attention_mask, t, img_ids, txt_ids, guidance_vec), (target, mask)

    def to_layers(self):
        transformer = self.transformer
        layers = [InitialLayer(transformer)]
        for i, block in enumerate(transformer.double_blocks):
            layers.append(TransformerWrapper(block, i, self.offloader_double))
        layers.append(concatenate_hidden_states)
        for i, block in enumerate(transformer.single_blocks):
            layers.append(SingleTransformerWrapper(block, i, self.offloader_single))
        layers.append(FinalLayer(transformer))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        transformer = self.transformer
        double_blocks = transformer.double_blocks
        single_blocks = transformer.single_blocks
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
        transformer.double_blocks = None
        transformer.single_blocks = None
        transformer.to('cuda')
        transformer.double_blocks = double_blocks
        transformer.single_blocks = single_blocks
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


class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.img_in = model.img_in
        self.txt_in = model.txt_in
        self.distilled_guidance_layer = model.distilled_guidance_layer
        self.pe_embedder = model.pe_embedder
        self.mod_index = model.mod_index
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        img, txt, txt_mask, timesteps, img_ids, txt_ids, guidance = inputs
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        img = self.img_in(img)
        txt = self.txt_in(txt)

        # See comments in original Chroma training code. This is supposed to be in a no_grad block.
        with torch.no_grad():
            distill_timestep = timestep_embedding(timesteps, 16)
            distil_guidance = timestep_embedding(guidance, 16)
            # get all modulation index
            modulation_index = timestep_embedding(self.mod_index.to(distill_timestep.device), 32)
            # we need to broadcast the modulation index here so each batch has all of the index
            modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1)
            # and we need to broadcast timestep and guidance along too
            timestep_guidance = (
                torch.cat([distill_timestep, distil_guidance], dim=1)
                .unsqueeze(1)
                .repeat(1, self.mod_index_length, 1)
            )
            # then and only then we could concatenate it together
            input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1)
            mod_vectors = self.distilled_guidance_layer(input_vec.requires_grad_(True))

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        max_len = txt.shape[1]

        with torch.no_grad():
            txt_mask_w_padding = modify_mask_to_attend_padding(
                txt_mask, max_len, 1
            )
            txt_img_mask = torch.cat(
                [
                    txt_mask_w_padding,
                    torch.ones([img.shape[0], img.shape[1]], device=txt_mask.device),
                ],
                dim=1,
            )
            txt_img_mask = txt_img_mask.float().T @ txt_img_mask.float()
            txt_img_mask = (
                txt_img_mask[None, None, ...]
                .repeat(txt.shape[0], self.num_heads, 1, 1)
                .int()
                .bool()
            )

        return make_contiguous(img, txt, pe, mod_vectors, txt_img_mask)


class TransformerWrapper(nn.Module):
    def __init__(self, block, idx, offloader):
        super().__init__()
        self.block = block
        self.idx = idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        img, txt, pe, mod_vectors, txt_img_mask = inputs

        self.offloader.wait_for_block(self.idx)

        img_mod_spec = modulation_distribute_dict[f"double_blocks.{self.idx}.img_mod.lin"]
        txt_mod_spec = modulation_distribute_dict[f"double_blocks.{self.idx}.txt_mod.lin"]
        img_mod = [
            ModulationOut(
                shift=mod_vectors[:, spec.shift, :],
                scale=mod_vectors[:, spec.scale, :],
                gate=mod_vectors[:, spec.gate, :],
            )
            for spec in img_mod_spec
        ]
        txt_mod = [
            ModulationOut(
                shift=mod_vectors[:, spec.shift, :],
                scale=mod_vectors[:, spec.scale, :],
                gate=mod_vectors[:, spec.gate, :],
            )
            for spec in txt_mod_spec
        ]
        double_mod = [img_mod, txt_mod]
        img, txt = self.block(
            img=img, txt=txt, pe=pe, distill_vec=double_mod, mask=txt_img_mask
        )

        self.offloader.submit_move_blocks_forward(self.idx)

        return make_contiguous(img, txt, pe, mod_vectors, txt_img_mask)


def concatenate_hidden_states(inputs):
    img, txt, pe, mod_vectors, txt_img_mask = inputs
    img = torch.cat((txt, img), 1)
    return img, txt, pe, mod_vectors, txt_img_mask


class SingleTransformerWrapper(nn.Module):
    def __init__(self, block, idx, offloader):
        super().__init__()
        self.block = block
        self.idx = idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        img, txt, pe, mod_vectors, txt_img_mask = inputs

        self.offloader.wait_for_block(self.idx)

        single_mod_spec = modulation_distribute_dict[f"single_blocks.{self.idx}.modulation.lin"]
        single_mod = ModulationOut(
            shift=mod_vectors[:, single_mod_spec.shift, :],
            scale=mod_vectors[:, single_mod_spec.scale, :],
            gate=mod_vectors[:, single_mod_spec.gate, :],
        )
        img = self.block(img, pe=pe, distill_vec=single_mod, mask=txt_img_mask)

        self.offloader.submit_move_blocks_forward(self.idx)

        return make_contiguous(img, txt, pe, mod_vectors, txt_img_mask)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        img, txt, pe, mod_vectors, txt_img_mask = inputs
        img = img[:, txt.shape[1] :, ...]
        final_mod_spec = modulation_distribute_dict["final_layer.adaLN_modulation.1"]
        final_mod = [mod_vectors[:, s, :] for s in final_mod_spec]
        img = self.final_layer(
            img, distill_vec=final_mod
        )  # (N, T, patch_size ** 2 * out_channels)
        return img
