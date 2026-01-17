import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import safetensors

from models.base import ComfyPipeline, make_contiguous
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift, one_at_a_time
from utils.offloading import ModelOffloader
import comfy.ldm.common_dit
from comfy.ldm.flux.layers import timestep_embedding, ModulationOut


class Flux2Pipeline(ComfyPipeline):
    name = 'flux2'
    checkpointable_layers = ['DoubleTransformerLayer', 'SingleTransformerLayer']
    adapter_target_modules = ['DoubleStreamBlock', 'SingleStreamBlock']
    keep_in_high_precision = ['img_in', 'time_in', 'guidance_in', 'txt_in', 'final_layer', 'double_stream_modulation_img', 'double_stream_modulation_txt', 'single_stream_modulation']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model_dim = None
        with safetensors.safe_open(self.model_config['diffusion_model'], framework='pt', device='cpu') as f:
            for key in f.keys():
                if 'img_in.weight' in key:
                    model_dim = f.get_tensor(key).shape[0]
                    break
        if model_dim is None:
            raise RuntimeError('Could not autodetect Flux2 model version from weights file.')

        self.is_4b = False
        self.is_9b = False
        self.is_32b = False
        if model_dim == 6144:
            self.is_32b = True
        elif model_dim == 3072:
            # The Kleins have different text encoders, so change model name so it gets a different cache dir.
            self.is_4b = True
            self.name = 'flux2_klein_4b'
        elif model_dim == 4096:
            self.is_9b = True
            self.name = 'flux2_klein_9b'
        else:
            raise RuntimeError(f'Unknown Flux2 model version with model_dim={model_dim}')

        self.offloader_double = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.offloader_single = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)

        if self.is_32b:
            te_wrapper = self.text_encoders[0]
            old_load_fn = te_wrapper._load_fn

            def new_load_fn():
                te = old_load_fn()
                # Get padding to work correctly. We need this because when training, batch_size can be >1.
                tokenizer = getattr(te.tokenizer, te.tokenizer.clip)
                tokenizer.pad_left = False  # attention_mask will be all 0s unless we do this
                clip = getattr(te.cond_stage_model, te.cond_stage_model.clip)
                clip.special_tokens['pad'] = 11  # to correctly put 0s in attention_mask
                return te

            te_wrapper._load_fn = new_load_fn

    def load_diffusion_model(self):
        if self.is_32b:
            # Model is so big, it's easy to OOM while loading with multiple GPUs.
            with one_at_a_time():
                rank = int(os.environ['LOCAL_RANK'])
                print(f'Loading Flux2 on rank {rank}')
                super().load_diffusion_model()
        else:
            super().load_diffusion_model()

    def get_call_vae_fn(self, vae):
        """
        Returns a function to encode images with the VAE, supporting optional control images.
        Similar to Qwen Edit implementation for reference/edit image pairs.
        """
        # Get parent class's VAE encoding function
        parent_vae_fn = super().get_call_vae_fn(vae)

        def fn(*args):
            if len(args) == 1:
                # Standard encoding - use parent implementation
                return parent_vae_fn(args[0])
            elif len(args) == 2:
                # Control image provided (for reference conditioning)
                tensor, control_tensor = args
                # Encode both using parent's method
                result = parent_vae_fn(tensor)
                control_result = parent_vae_fn(control_tensor)
                result['control_latents'] = control_result['latents']
                return result
            else:
                raise RuntimeError(f'Unexpected number of args: {len(args)}')
        return fn

    def to_layers(self):
        diffusion_model = self.diffusion_model
        layers = [InitialLayer(diffusion_model)]
        for i, block in enumerate(diffusion_model.double_blocks):
            layers.append(DoubleTransformerLayer(block, i, self.offloader_double))
        layers.append(ConcatenateTxtImg(diffusion_model))
        for i, block in enumerate(diffusion_model.single_blocks):
            layers.append(SingleTransformerLayer(block, i, self.offloader_single))
        layers.append(FinalLayer(diffusion_model))
        return layers

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        latents = self.model_patcher.model.process_latent_in(latents)
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

        bs, c, h, w = latents.shape
        device = latents.device

        attention_mask = attention_mask.to(torch.bool).view(bs, 1, 1, -1)  # for PyTorch SDPA

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
        t_expanded = t.view(-1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents
        guidance = torch.ones((bs,), device=device, dtype=torch.float32)

        # Handle control latents for reference conditioning (e.g., edit mode)
        extra_inputs = ()
        if 'control_latents' in inputs:
            control_latents = inputs['control_latents'].float()
            control_latents = self.model_patcher.model.process_latent_in(control_latents)
            assert control_latents.shape == latents.shape, (
                f"Control latents shape {control_latents.shape} doesn't match latents shape {latents.shape}"
            )
            extra_inputs = (control_latents,)

        return (noisy_latents, t, text_embeds, attention_mask, guidance) + extra_inputs, (target, mask)

    def enable_block_swap(self, blocks_to_swap):
        transformer = self.diffusion_model
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
        self.time_in = model.time_in
        self.guidance_in = model.guidance_in
        self.vector_in = model.vector_in
        self.txt_norm = model.txt_norm
        self.txt_in = model.txt_in
        if model.params.global_modulation:
            self.double_stream_modulation_img = model.double_stream_modulation_img
            self.double_stream_modulation_txt = model.double_stream_modulation_txt
        self.pe_embedder = model.pe_embedder
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)

        # Unpack inputs, handling optional control latents
        x, timesteps, txt, txt_mask, guidance, *extra = inputs
        has_control = len(extra) > 0
        y = None

        bs, c, h_orig, w_orig = x.shape
        device = x.device

        img, img_ids = self.process_img(x)
        img_len_orig = img.shape[1]  # Original image sequence length

        # Process control latents if present
        if has_control:
            control_latents = extra[0]
            control_img, control_img_ids = self.process_img(control_latents)
            control_img_len = control_img.shape[1]
            assert control_img_len == img_len_orig, (
                f"Control image sequence length {control_img_len} doesn't match noisy image {img_len_orig}"
            )

        img_len = torch.tensor(img.shape[1], dtype=torch.int64, device=device)

        txt_ids = torch.zeros((bs, txt.shape[1], len(self.params.axes_dim)), device=device, dtype=torch.float32)

        if len(self.params.txt_ids_dims) > 0:
            for i in self.params.txt_ids_dims:
                txt_ids[:, :, i] = torch.linspace(0, txt.shape[1] - 1, steps=txt.shape[1], device=device, dtype=torch.float32)

        img = self.img_in(img)

        # Process control image with same img_in projection if present
        if has_control:
            control_img = self.img_in(control_img)
            # Concatenate noisy image and control image along sequence dimension
            img = torch.cat([img, control_img], dim=1)
            # Update img_ids to include control image positions
            img_ids = torch.cat([img_ids, control_img_ids], dim=1)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        if self.vector_in is not None:
            if y is None:
                y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=device, dtype=img.dtype)
            vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])

        if self.txt_norm is not None:
            txt = self.txt_norm(txt)
        txt = self.txt_in(txt)

        vec_orig = vec
        if self.params.global_modulation:
            vec_img = self.double_stream_modulation_img(vec_orig)
            vec_txt = self.double_stream_modulation_txt(vec_orig)
            vec = []
            for mod_out in (*vec_img, *vec_txt):
                vec.append(mod_out.shift)
                vec.append(mod_out.scale)
                vec.append(mod_out.gate)
        else:
            vec = (vec,)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        pe.requires_grad_(True)  # work around pipeline parallelism issue
        txt_len = torch.tensor(txt.shape[1], dtype=torch.int64, device=device)
        hw = torch.tensor([h_orig, w_orig], dtype=torch.int64, device=device)

        attn_mask = torch.cat((txt_mask, torch.ones((bs, 1, 1, img.shape[1]), dtype=torch.bool, device=device)), dim=-1)

        # Pass original image length to subsequent layers so FinalLayer can extract correct portion
        extra_outputs = (torch.tensor(img_len_orig, dtype=torch.int64, device=device),) if has_control else ()

        return make_contiguous(img, txt, pe, attn_mask, txt_len, img_len, hw, vec_orig, *vec) + extra_outputs


class DoubleTransformerLayer(nn.Module):
    def __init__(self, layer, block_idx, offloader):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        # Handle optional extra outputs (e.g., img_len_orig for control)
        if torch.is_tensor(inputs[-1]) and inputs[-1].numel() == 1 and inputs[-1].dtype == torch.int64:
            # Last element is img_len_orig
            *main_inputs, extra_output = inputs
            img, txt, pe, attn_mask, txt_len, img_len, hw, vec_orig, *vec = main_inputs
            has_extra = True
        else:
            img, txt, pe, attn_mask, txt_len, img_len, hw, vec_orig, *vec = inputs
            has_extra = False

        if len(vec) == 1:
            tmp_vec = vec[0]
        else:
            mod_outs = []
            for i in range(0, len(vec), 3):
                mod_outs.append(ModulationOut(*vec[i:i+3]))
            assert len(mod_outs) == 4
            tmp_vec = ((mod_outs[0], mod_outs[1]), (mod_outs[2], mod_outs[3]))

        self.offloader.wait_for_block(self.block_idx)
        img, txt = self.layer(img=img, txt=txt, vec=tmp_vec, pe=pe, attn_mask=attn_mask)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        result = make_contiguous(img, txt, pe, attn_mask, txt_len, img_len, hw, vec_orig, *vec)
        if has_extra:
            result = result + (extra_output,)
        return result


class ConcatenateTxtImg(nn.Module):
    def __init__(self, model):
        super().__init__()
        if model.params.global_modulation:
            self.single_stream_modulation = model.single_stream_modulation
        else:
            self.single_stream_modulation = None

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        # Handle optional extra outputs (e.g., img_len_orig for control)
        if torch.is_tensor(inputs[-1]) and inputs[-1].numel() == 1 and inputs[-1].dtype == torch.int64:
            # Last element is img_len_orig
            *main_inputs, extra_output = inputs
            img, txt, pe, attn_mask, txt_len, img_len, hw, vec_orig, *vec = main_inputs
            has_extra = True
        else:
            img, txt, pe, attn_mask, txt_len, img_len, hw, vec_orig, *vec = inputs
            has_extra = False

        img = torch.cat((txt, img), 1)
        if self.single_stream_modulation is not None:
            mod_out, _ = self.single_stream_modulation(vec_orig)
            vec = [mod_out.shift, mod_out.scale, mod_out.gate]

        result = make_contiguous(img, pe, attn_mask, txt_len, img_len, hw, vec_orig, *vec)
        if has_extra:
            result = result + (extra_output,)
        return result


class SingleTransformerLayer(nn.Module):
    def __init__(self, layer, block_idx, offloader):
        super().__init__()
        self.layer = layer
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        # Handle optional extra outputs (e.g., img_len_orig for control)
        if torch.is_tensor(inputs[-1]) and inputs[-1].numel() == 1 and inputs[-1].dtype == torch.int64:
            # Last element is img_len_orig
            *main_inputs, extra_output = inputs
            img, pe, attn_mask, txt_len, img_len, hw, vec_orig, *vec = main_inputs
            has_extra = True
        else:
            img, pe, attn_mask, txt_len, img_len, hw, vec_orig, *vec = inputs
            has_extra = False

        if len(vec) == 1:
            tmp_vec = vec[0]
        else:
            assert len(vec) == 3
            tmp_vec = ModulationOut(*vec)

        self.offloader.wait_for_block(self.block_idx)
        img = self.layer(img, vec=tmp_vec, pe=pe, attn_mask=attn_mask)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        result = make_contiguous(img, pe, attn_mask, txt_len, img_len, hw, vec_orig, *vec)
        if has_extra:
            result = result + (extra_output,)
        return result


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        # Handle optional extra outputs (e.g., img_len_orig for control)
        if torch.is_tensor(inputs[-1]) and inputs[-1].numel() == 1 and inputs[-1].dtype == torch.int64:
            # Last element is img_len_orig - control image present
            *main_inputs, img_len_orig = inputs
            img, pe, attn_mask, txt_len, img_len, hw, vec_orig, *vec = main_inputs
            has_control = True
        else:
            img, pe, attn_mask, txt_len, img_len, hw, vec_orig, *vec = inputs
            has_control = False

        # Extract image portion (after text)
        img = img[:, txt_len.item():, ...]

        if has_control:
            # Only process the noisy image portion, exclude control image
            # Control image was concatenated after noisy image in InitialLayer
            img = img[:, :img_len_orig.item(), ...]

        out = self.final_layer(img, vec_orig)
        out = out[:, :img_len.item() if not has_control else img_len_orig.item()]
        h_orig = hw[0].item()
        w_orig = hw[1].item()
        patch_size = self.patch_size
        h_len = ((h_orig + (patch_size // 2)) // patch_size)
        w_len = ((w_orig + (patch_size // 2)) // patch_size)
        return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=patch_size, pw=patch_size)[:,:,:h_orig,:w_orig]
