from pathlib import Path
import argparse
import json
import os.path
import random

import safetensors
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from loguru import logger

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from utils.common import AUTOCAST_DTYPE, load_safetensors
from utils.offloading import ModelOffloader
from hyvideo.config import add_network_args, add_extra_models_args, add_denoise_schedule_args, add_inference_args, sanity_check_args
from hyvideo.modules import load_model
from hyvideo.vae import load_vae
from hyvideo.constants import PRECISION_TO_TYPE, PROMPT_TEMPLATE
from hyvideo.text_encoder import TextEncoder
from hyvideo.modules.attenion import get_cu_seqlens
from hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler
from hyvideo.diffusion.pipelines import HunyuanVideoPipeline as OriginalHunyuanVideoPipeline
from hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from .hy_framepack_transformer_blocks import HunyuanVideoTransformer3DModelPacked, pad_for_3d_conv, center_down_sample_3d, hf_clip_vision_encode, get_cu_seqlens

# Framepack
from transformers import SiglipImageProcessor, SiglipVisionModel
import einops

# In diffusion-pipe, we already converted the dtype to an object. But Hunyuan scripts want the string version in a lot of places.
TYPE_TO_PRECISION = {v: k for k, v in PRECISION_TO_TYPE.items()}

class VaeAndClip(nn.Module):
    def __init__(self, vae, clip):
        super().__init__()
        self.vae = vae
        self.clip = clip

def load_state_dict(args, pretrained_model_path):
    load_key = args.load_key
    dit_weight = Path(args.dit_weight)

    if dit_weight is None:
        model_dir = pretrained_model_path / f"t2v_{args.model_resolution}"
        files = list(model_dir.glob("*.pt"))
        if len(files) == 0:
            raise ValueError(f"No model weights found in {model_dir}")
        if str(files[0]).startswith("pytorch_model_"):
            model_path = dit_weight / f"pytorch_model_{load_key}.pt"
            bare_model = True
        elif any(str(f).endswith("_model_states.pt") for f in files):
            files = [f for f in files if str(f).endswith("_model_states.pt")]
            model_path = files[0]
            if len(files) > 1:
                logger.warning(
                    f"Multiple model weights found in {dit_weight}, using {model_path}"
                )
            bare_model = False
        else:
            raise ValueError(
                f"Invalid model path: {dit_weight} with unrecognized weight format: "
                f"{list(map(str, files))}. When given a directory as --dit-weight, only "
                f"`pytorch_model_*.pt`(provided by HunyuanDiT official) and "
                f"`*_model_states.pt`(saved by deepspeed) can be parsed. If you want to load a "
                f"specific weight file, please provide the full path to the file."
            )
    else:
        if dit_weight.is_dir():
            files = list(dit_weight.glob("*.pt"))
            if len(files) == 0:
                raise ValueError(f"No model weights found in {dit_weight}")
            if str(files[0]).startswith("pytorch_model_"):
                model_path = dit_weight / f"pytorch_model_{load_key}.pt"
                bare_model = True
            elif any(str(f).endswith("_model_states.pt") for f in files):
                files = [f for f in files if str(f).endswith("_model_states.pt")]
                model_path = files[0]
                if len(files) > 1:
                    logger.warning(
                        f"Multiple model weights found in {dit_weight}, using {model_path}"
                    )
                bare_model = False
            else:
                raise ValueError(
                    f"Invalid model path: {dit_weight} with unrecognized weight format: "
                    f"{list(map(str, files))}. When given a directory as --dit-weight, only "
                    f"`pytorch_model_*.pt`(provided by HunyuanDiT official) and "
                    f"`*_model_states.pt`(saved by deepspeed) can be parsed. If you want to load a "
                    f"specific weight file, please provide the full path to the file."
                )
        elif dit_weight.is_file():
            model_path = dit_weight
            bare_model = "unknown"
        else:
            raise ValueError(f"Invalid model path: {dit_weight}")

    if not model_path.exists():
        raise ValueError(f"model_path not exists: {model_path}")
    logger.info(f"Loading torch model {model_path}...")
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=True, mmap=True)

    if bare_model == "unknown" and ("ema" in state_dict or "module" in state_dict):
        bare_model = False
    if bare_model is False:
        if load_key in state_dict:
            state_dict = state_dict[load_key]
        else:
            raise KeyError(
                f"Missing key: `{load_key}` in the checkpoint: {model_path}. The keys in the checkpoint "
                f"are: {list(state_dict.keys())}."
            )

    return state_dict


def vae_encode(tensor, vae):
    # tensor values already in range [-1, 1] here
    latents = vae.encode(tensor).latent_dist.sample()
    return latents * vae.config.scaling_factor


# Warning: starting the conversion to full framepack
class HunyuanVideoPipeline(BasePipeline):
    name = 'framepack-hv'
    framerate = 30
    checkpointable_layers = ['DoubleBlock', 'SingleBlock']
    adapter_target_modules = ['HunyuanVideoTransformerBlock', 'HunyuanVideoSingleTransformerBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.offloader_double = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.offloader_single = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)

        dtype = self.model_config['dtype']

        parser = argparse.ArgumentParser()
        parser = add_network_args(parser)
        parser = add_extra_models_args(parser)
        parser = add_denoise_schedule_args(parser)
        parser = add_inference_args(parser)
        args = parser.parse_args([])
        if 'ckpt_path' in self.model_config:
            args.model_base = self.model_config['ckpt_path']
            args.dit_weight = os.path.join(args.model_base, 'hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt')
        self.args = sanity_check_args(args)

        if self.args.prompt_template_video is not None:
            crop_start = PROMPT_TEMPLATE[self.args.prompt_template_video].get(
                "crop_start", 0
            )
            self.max_text_length_video = self.args.text_len + crop_start
        if self.args.prompt_template is not None:
            crop_start = PROMPT_TEMPLATE[self.args.prompt_template].get("crop_start", 0)
            self.max_text_length_image = self.args.text_len + crop_start

        if vae_path := self.model_config.get('vae_path', None):
            with open('configs/hy_vae_config.json') as f:
                vae_config = json.load(f)
            vae_sd = load_safetensors(vae_path)
            vae = AutoencoderKLCausal3D.from_config(vae_config)
            vae.load_state_dict(vae_sd)
            del vae_sd
            vae.requires_grad_(False)
            vae.eval()
            vae.to(dtype=dtype)
        else:
            vae, _, _, _ = load_vae(
                self.args.vae,
                TYPE_TO_PRECISION[dtype],
                vae_path=os.path.join(self.args.model_base, 'hunyuan-video-t2v-720p/vae'),
                logger=logger,
                device='cpu',
            )
        # Enabled by default in inference scripts, so we should probably train with it.
        vae.enable_tiling()

        # Text encoder
        prompt_template = (
            PROMPT_TEMPLATE[self.args.prompt_template]
            if self.args.prompt_template is not None
            else None
        )

        prompt_template_video = (
            PROMPT_TEMPLATE[self.args.prompt_template_video]
            if self.args.prompt_template_video is not None
            else None
        )

        llm_path = self.model_config.get('llm_path', os.path.join(self.args.model_base, 'text_encoder'))
        text_encoder = TextEncoder(
            text_encoder_type=self.args.text_encoder,
            max_length=self.max_text_length_video,
            text_encoder_path=llm_path,
            text_encoder_precision=TYPE_TO_PRECISION[dtype],
            tokenizer_type=self.args.tokenizer,
            prompt_template=prompt_template,
            prompt_template_video=prompt_template_video,
            hidden_state_skip_layer=self.args.hidden_state_skip_layer,
            apply_final_norm=self.args.apply_final_norm,
            reproduce=self.args.reproduce,
            logger=logger,
            device='cpu',
        )

        clip_path = self.model_config.get('clip_path', os.path.join(self.args.model_base, 'text_encoder_2'))
        text_encoder_2 = TextEncoder(
            text_encoder_type=self.args.text_encoder_2,
            max_length=self.args.text_len_2,
            text_encoder_path=clip_path,
            text_encoder_precision=TYPE_TO_PRECISION[dtype],
            tokenizer_type=self.args.tokenizer_2,
            reproduce=self.args.reproduce,
            logger=logger,
            device='cpu',
        )

        scheduler = FlowMatchDiscreteScheduler(
            shift=7.0,
            reverse=True,
            solver="euler",
        )

        self.diffusers_pipeline = OriginalHunyuanVideoPipeline(
            transformer=None,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            scheduler=scheduler,
            args=args,
        )

        self.latent_window_size = self.model_config.get('latent_window_size', 9)
        self.t2v = self.model_config.get('t2v', False)

        print("Loading framepack specific modules")
        siglip_path = self.model_config.get('siglip_path', "lllyasviel/flux_redux_bfl")
        self.siglip_feature_extractor = SiglipImageProcessor.from_pretrained(siglip_path, subfolder='feature_extractor')
        image_encoder = SiglipVisionModel.from_pretrained(siglip_path, subfolder='image_encoder', torch_dtype=dtype).eval()
        self.vae_and_clip = VaeAndClip(vae, image_encoder).eval()

    # delay loading transformer to save RAM
    def load_diffusion_model(self):
        transformer_dtype = self.model_config.get('transformer_dtype', self.model_config['dtype'])
        with init_empty_weights():
            # default config from kijai
            cfg = {"_class_name": "HunyuanVideoTransformer3DModelPacked", "_diffusers_version": "0.33.0.dev0", "_name_or_path": "hunyuanvideo-community/HunyuanVideo",
                    "attention_head_dim": 128, "guidance_embeds": True, "has_clean_x_embedder": True, "has_image_proj": True, "image_proj_dim": 1152, "in_channels": 16,
                    "mlp_ratio": 4.0, "num_attention_heads": 24, "num_layers": 20, "num_refiner_layers": 2, "num_single_layers": 40, "out_channels": 16,
                    "patch_size": 2, "patch_size_t": 1, "pooled_projection_dim": 768, "qk_norm": "rms_norm",
                    "rope_axes_dim": [16, 56, 56], "rope_theta": 256.0, "text_embed_dim": 4096 }
            transformer = HunyuanVideoTransformer3DModelPacked(**cfg)

        transformer_path = self.model_config['transformer_path']
        state_dict = load_safetensors(transformer_path)

        params_to_keep = {"norm", "bias", "time_text_embed", "context_embedder", "x_embedder", "clean_x_embedder", "image_projection"}
        base_dtype = self.model_config['dtype']

        for name, param in transformer.named_parameters():
            dtype_to_use = base_dtype if (any(keyword in name for keyword in params_to_keep) or name.startswith('proj_out')) else transformer_dtype
            set_module_tensor_to_device(transformer, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])

        self.diffusers_pipeline.transformer = transformer
        self.transformer.train()
        # We'll need the original parameter name for saving, and the name changes once we wrap modules for pipeline parallelism,
        # so store it in an attribute here. Same thing below if we're training a lora and creating lora weights.
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def __getattr__(self, name):
        if name == "vae":
            return self.vae_and_clip
        else:
            return getattr(self.diffusers_pipeline, name)

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder, self.text_encoder_2]

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # Diffusers format (maybe, works with Kijai ComfyUI wrapper).
        peft_state_dict = {'transformer.'+k: v for k, v in peft_state_dict.items()}
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def save_model(self, save_dir, diffusers_sd):
        raise NotImplementedError()

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(
            self.config,
            support_video=True,
            framerate=self.framerate,
            round_height=8,
            round_width=8,
            round_frames=4,
        )

    # this function should be called dynamically and not preloaded
    # to have greater variety on small datasets
    def random_timecrop_framepack(self, latents):
        bs, c, f, h, w = latents.shape
        required_frame_length = self.latent_window_size if self.t2v else self.latent_window_size+1
        assert f == 1 or f >= required_frame_length

        if self.t2v or f == 1:
            first_frame = torch.zeros([bs, c, 0, h, w], device=latents.device)
        else:
            first_frame = latents[:, :, 0:1, :, :]
            latents = latents[:, :, 1:, :, :]
        first_frame_size = first_frame.size(2)

        latent_window_size = 1 if f == 1 else self.latent_window_size

        if f == 1:
            # This is technically wrong. The model seems to have been trained always with clean_latents, even if they are 0s. Doing it like this,
            # the validation loss starts a bit higher, but then quickly drops to the same value. And it's 2 times faster and uses a bit less
            # memory. So just always do it like this for images.
            return {'latents': latents}

        f = latents.size(2)
        total_latent_sections = f // latent_window_size
        latent_paddings = [0, 1] + [2] * (total_latent_sections - 3) + [3]
        latents = latents[:, :, :total_latent_sections*latent_window_size, :, :]
        latents = F.pad(latents, (0, 0, 0, 0, 0, 19))

        fragment_list = []
        clean_latents_list = []
        clean_latents_2x_list = []
        clean_latents_4x_list = []
        latent_indices_list = []
        clean_latent_indices_list = []
        clean_latent_2x_indices_list = []
        clean_latent_4x_indices_list = []

        for single_latents, single_first_frame in zip(latents, first_frame):
            # add batch dimension back
            single_latents = single_latents.unsqueeze(0)
            single_first_frame = single_first_frame.unsqueeze(0)

            chosen_segment = random.randrange(0, total_latent_sections)
            offset = chosen_segment*latent_window_size
            latent_slice = single_latents[:, :, offset:offset+latent_window_size+19, :, :]
            fragment, clean_latents_post, clean_latents_2x, clean_latents_4x = latent_slice.split([latent_window_size, 1, 2, 16], dim=2)

            latent_padding_size = latent_paddings[chosen_segment] * latent_window_size
            indices = torch.arange(0, sum([first_frame_size, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([first_frame_size, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
            clean_latents = torch.cat([single_first_frame, clean_latents_post], dim=2)

            fragment_list.append(fragment)
            clean_latents_list.append(clean_latents)
            clean_latents_2x_list.append(clean_latents_2x)
            clean_latents_4x_list.append(clean_latents_4x)
            latent_indices_list.append(latent_indices)
            clean_latent_indices_list.append(clean_latent_indices)
            clean_latent_2x_indices_list.append(clean_latent_2x_indices)
            clean_latent_4x_indices_list.append(clean_latent_4x_indices)

        return {
            'latents': torch.cat(fragment_list),
            'clean_latents': torch.cat(clean_latents_list),
            'clean_latents_2x': torch.cat(clean_latents_2x_list),
            'clean_latents_4x': torch.cat(clean_latents_4x_list),
            'latent_indices': torch.cat(latent_indices_list),
            'clean_latent_indices': torch.cat(clean_latent_indices_list),
            'clean_latent_2x_indices': torch.cat(clean_latent_2x_indices_list),
            'clean_latent_4x_indices': torch.cat(clean_latent_4x_indices_list),
        }


    def get_call_vae_fn(self, vae_and_clip):
        def fn(tensor):
            ret = {}

            vae = vae_and_clip.vae
            image_encoder = vae_and_clip.clip

            first_frame = tensor[:, :, 0, ...].clone().squeeze(0).permute(1, 2, 0)
            first_frame_0_1 = ((first_frame+1)/2).clamp(0, 1)
            image_encoder_output = hf_clip_vision_encode((first_frame_0_1.numpy()*255.).astype(np.uint8), self.siglip_feature_extractor, image_encoder)
            ret['image_encoder_output'] = image_encoder_output.last_hidden_state.to(image_encoder.device, image_encoder.dtype)
            ret['latents'] = vae_encode(tensor.to(vae.device, vae.dtype), vae)
            return ret
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        if text_encoder == self.text_encoder:
            text_encoder_idx = 1
        elif text_encoder == self.text_encoder_2:
            text_encoder_idx = 2
        else:
            raise RuntimeError()
        def fn(caption, is_video):
            # args are lists
            prompt_embeds, prompt_attention_masks = [], []
            # need to use a loop because is_video might be different for each one
            for caption, is_video in zip(caption, is_video):
                if is_video:
                    # This is tricky. The text encoder will crop off the prompt correctly based on the data_type, but the offical code only sets the max
                    # length (which needs to be set accordingly to the prompt) once. So we have to do it here each time.
                    if text_encoder_idx == 1:
                        text_encoder.max_length = self.max_text_length_video
                    data_type = 'video'
                else:
                    if text_encoder_idx == 1:
                        text_encoder.max_length = self.max_text_length_image
                    data_type = 'image'
                (
                    prompt_embed,
                    negative_prompt_embed,
                    prompt_mask,
                    negative_prompt_mask,
                ) = self.encode_prompt(
                    caption,
                    device=next(text_encoder.parameters()).device,
                    num_videos_per_prompt=1,
                    do_classifier_free_guidance=False,
                    text_encoder=text_encoder,
                    data_type=data_type,
                )
                prompt_embeds.append(prompt_embed)
                prompt_attention_masks.append(prompt_mask)
            prompt_embeds = torch.cat(prompt_embeds)
            prompt_attention_masks = torch.cat(prompt_attention_masks)
            if text_encoder_idx == 1:
                return {'prompt_embeds_1': prompt_embeds, 'prompt_attention_mask_1': prompt_attention_masks}
            elif text_encoder_idx == 2:
                return {'prompt_embeds_2': prompt_embeds}
            else:
                raise RuntimeError()
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        prompt_embeds_1 = inputs['prompt_embeds_1']
        prompt_attention_mask_1 = inputs['prompt_attention_mask_1']
        prompt_embeds_2 = inputs['prompt_embeds_2']
        mask = inputs['mask']
        image_embeddings = inputs['image_encoder_output']

        rtf = self.random_timecrop_framepack(latents)
        latents = rtf['latents']

        bs, channels, num_frames, h, w = latents.shape

        if 'clean_latents' not in rtf:
            # dummy clean latents for microbatch splitting
            clean_latents = torch.zeros((bs,), device=latents.device, dtype=latents.dtype)
            clean_latents_2x = torch.zeros((bs,), device=latents.device, dtype=latents.dtype)
            clean_latents_4x = torch.zeros((bs,), device=latents.device, dtype=latents.dtype)

            latent_indices = torch.zeros((bs,), device=latents.device, dtype=latents.dtype)
            clean_latent_indices = torch.zeros((bs,), device=latents.device, dtype=latents.dtype)
            clean_latent_2x_indices = torch.zeros((bs,), device=latents.device, dtype=latents.dtype)
            clean_latent_4x_indices = torch.zeros((bs,), device=latents.device, dtype=latents.dtype)
        else:
            clean_latents = rtf['clean_latents']
            clean_latents_2x = rtf['clean_latents_2x']
            clean_latents_4x = rtf['clean_latents_4x']

            latent_indices = rtf['latent_indices']
            clean_latent_indices = rtf['clean_latent_indices']
            clean_latent_2x_indices = rtf['clean_latent_2x_indices']
            clean_latent_4x_indices = rtf['clean_latent_4x_indices']

        if self.t2v or num_frames == 1:
            image_embeddings = torch.zeros((bs,), device=image_embeddings.device)

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)  # make mask same number of dims as target

        guidance_expand = torch.tensor(
            [self.model_config.get('guidance', 1.0)] * bs,
            dtype=torch.float32,
        ) * 1000

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

        x_1 = latents
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
        target = x_0 - x_1

        # timestep input to model needs to be in range [0, 1000]
        t = t * 1000

        return (
            x_t,
            t,
            prompt_embeds_1,
            image_embeddings,
            prompt_attention_mask_1,
            prompt_embeds_2,
            guidance_expand,
            clean_latents,
            clean_latents_2x,
            clean_latents_4x,
            latent_indices,
            clean_latent_indices,
            clean_latent_2x_indices,
            clean_latent_4x_indices
        ), (target, mask)

    def to_layers(self):
        transformer = self.transformer
        layers = [InitialLayer(transformer)]
        for i, block in enumerate(transformer.transformer_blocks):
            layers.append(DoubleBlock(block, i, self.offloader_double))
        for i, block in enumerate(transformer.single_transformer_blocks):
            layers.append(SingleBlock(block, i, self.offloader_single))
        layers.append(OutputLayer(transformer))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        transformer = self.transformer
        double_blocks = transformer.transformer_blocks
        single_blocks = transformer.single_transformer_blocks
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
        transformer.transformer_blocks = None
        transformer.single_transformer_blocks = None
        transformer.to('cuda')
        transformer.transformer_blocks = double_blocks
        transformer.single_transformer_blocks = single_blocks
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
    def __init__(self, transformer):
        super().__init__()
        # Prevent registering the whole Transformer.
        self.transformer = [transformer]

        # Explicitly register these modules.
        self.x_embedder = self.transformer[0].x_embedder
        self.rope = self.transformer[0].rope
        self.clean_x_embedder = self.transformer[0].clean_x_embedder

        self.time_text_embed = self.transformer[0].time_text_embed
        self.context_embedder = self.transformer[0].context_embedder

        self.image_projection = self.transformer[0].image_projection
        self.config = self.transformer[0].config


    def process_input_hidden_states(
        self,
        latents, latent_indices=None,
        clean_latents=None, clean_latent_indices=None,
        clean_latents_2x=None, clean_latent_2x_indices=None,
        clean_latents_4x=None, clean_latent_4x_indices=None
    ):
        hidden_states = self.x_embedder.proj(latents)
        B, C, T, H, W = hidden_states.shape

        if latent_indices is None:
            latent_indices = torch.arange(0, T).unsqueeze(0).expand(B, -1)

        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        rope_freqs = self.rope(frame_indices=latent_indices, height=H, width=W, device=hidden_states.device)
        rope_freqs = rope_freqs.flatten(2).transpose(1, 2)

        if clean_latents is not None and clean_latent_indices is not None:
            clean_latents = clean_latents.to(hidden_states)
            clean_latents = self.clean_x_embedder.proj(clean_latents)
            clean_latents = clean_latents.flatten(2).transpose(1, 2)

            clean_latent_rope_freqs = self.rope(frame_indices=clean_latent_indices, height=H, width=W, device=clean_latents.device)
            clean_latent_rope_freqs = clean_latent_rope_freqs.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([clean_latents, hidden_states], dim=1)
            rope_freqs = torch.cat([clean_latent_rope_freqs, rope_freqs], dim=1)

        if clean_latents_2x is not None and clean_latent_2x_indices is not None:
            clean_latents_2x = clean_latents_2x.to(hidden_states)
            clean_latents_2x = pad_for_3d_conv(clean_latents_2x, (2, 4, 4))
            clean_latents_2x = self.clean_x_embedder.proj_2x(clean_latents_2x)
            clean_latents_2x = clean_latents_2x.flatten(2).transpose(1, 2)

            clean_latent_2x_rope_freqs = self.rope(frame_indices=clean_latent_2x_indices, height=H, width=W, device=clean_latents_2x.device)
            clean_latent_2x_rope_freqs = pad_for_3d_conv(clean_latent_2x_rope_freqs, (2, 2, 2))
            clean_latent_2x_rope_freqs = center_down_sample_3d(clean_latent_2x_rope_freqs, (2, 2, 2))
            clean_latent_2x_rope_freqs = clean_latent_2x_rope_freqs.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([clean_latents_2x, hidden_states], dim=1)
            rope_freqs = torch.cat([clean_latent_2x_rope_freqs, rope_freqs], dim=1)

        if clean_latents_4x is not None and clean_latent_4x_indices is not None:
            clean_latents_4x = clean_latents_4x.to(hidden_states)
            clean_latents_4x = pad_for_3d_conv(clean_latents_4x, (4, 8, 8))
            clean_latents_4x = self.clean_x_embedder.proj_4x(clean_latents_4x)
            clean_latents_4x = clean_latents_4x.flatten(2).transpose(1, 2)

            clean_latent_4x_rope_freqs = self.rope(frame_indices=clean_latent_4x_indices, height=H, width=W, device=clean_latents_4x.device)
            clean_latent_4x_rope_freqs = pad_for_3d_conv(clean_latent_4x_rope_freqs, (4, 4, 4))
            clean_latent_4x_rope_freqs = center_down_sample_3d(clean_latent_4x_rope_freqs, (4, 4, 4))
            clean_latent_4x_rope_freqs = clean_latent_4x_rope_freqs.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([clean_latents_4x, hidden_states], dim=1)
            rope_freqs = torch.cat([clean_latent_4x_rope_freqs, rope_freqs], dim=1)

        return hidden_states, rope_freqs

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)

        hidden_states, timestep, encoder_hidden_states, image_embeddings, encoder_attention_mask, pooled_projections, guidance, clean_latents, clean_latents_2x, clean_latents_4x, latent_indices, clean_latent_indices, clean_latent_2x_indices, clean_latent_4x_indices = inputs

        if len(clean_latents.shape) < 2:
            clean_latents = clean_latents_2x = clean_latents_4x = clean_latent_indices = latent_indices = clean_latent_2x_indices = clean_latent_4x_indices = None

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = self.config['patch_size'], self.config['patch_size_t']
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        original_context_length = post_patch_num_frames * post_patch_height * post_patch_width

        unpatchify_args = torch.tensor([post_patch_num_frames, post_patch_height, post_patch_width], device=hidden_states.device)

        hidden_states, rope_freqs = self.process_input_hidden_states(hidden_states, latent_indices, clean_latents, clean_latent_indices, clean_latents_2x, clean_latent_2x_indices, clean_latents_4x, clean_latent_4x_indices)

        temb = self.time_text_embed(timestep, guidance, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        assert self.image_projection is not None
        # Dim 1 means we should not use image embedding (t2v or image training).
        if image_embeddings.ndim > 1:
            extra_encoder_hidden_states = self.image_projection(image_embeddings)
            extra_attention_mask = torch.ones((batch_size, extra_encoder_hidden_states.shape[1]), dtype=encoder_attention_mask.dtype, device=encoder_attention_mask.device)

            # must cat before (not after) encoder_hidden_states, due to attn masking
            encoder_hidden_states = torch.cat([extra_encoder_hidden_states, encoder_hidden_states], dim=1)
            encoder_attention_mask = torch.cat([extra_attention_mask, encoder_attention_mask], dim=1)

        with torch.no_grad():
            # Mask getting from off Framepack, with all its flaws
            if batch_size == 1:
                # When batch size is 1, we do not need any masks or var-len funcs since cropping is mathematically same to what we want
                # If they are not same, then their impls are wrong. Ours are always the correct one.
                text_len = encoder_attention_mask.sum().item()
                encoder_hidden_states = encoder_hidden_states[:, :text_len]
                # attention_mask = None, None, None, None
                cu_seqlens_q = cu_seqlens_kv = max_seqlen_q = max_seqlen_kv = torch.tensor([0], device=hidden_states.device) # Dummy value
            else:
                img_seq_len = hidden_states.shape[1]
                txt_seq_len = encoder_hidden_states.shape[1]

                cu_seqlens_q = get_cu_seqlens(encoder_attention_mask, img_seq_len)
                cu_seqlens_kv = cu_seqlens_q
                max_seqlen_q = torch.tensor([img_seq_len + txt_seq_len], device=hidden_states.device)
                max_seqlen_kv = torch.tensor([max_seqlen_q], device=hidden_states.device)

                # attention_mask = cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv

        return make_contiguous(hidden_states, encoder_hidden_states, temb, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, rope_freqs, unpatchify_args)


class DoubleBlock(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, temb, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, rope_freqs, unpatchify_args = inputs
        attention_mask = (cu_seqlens_q, cu_seqlens_kv, max_seqlen_q.item(), max_seqlen_kv.item()) if hidden_states.shape[0] != 1 else (None, None, None, None)

        self.offloader.wait_for_block(self.block_idx)
        hidden_states, encoder_hidden_states = self.block(hidden_states, encoder_hidden_states, temb, attention_mask, rope_freqs)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(hidden_states, encoder_hidden_states, temb, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, rope_freqs, unpatchify_args)


class SingleBlock(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, temb, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, rope_freqs, unpatchify_args = inputs
        attention_mask = (cu_seqlens_q, cu_seqlens_kv, max_seqlen_q.item(), max_seqlen_kv.item()) if hidden_states.shape[0] != 1 else (None, None, None, None)

        self.offloader.wait_for_block(self.block_idx)
        hidden_states, encoder_hidden_states = self.block(hidden_states, encoder_hidden_states, temb, attention_mask, rope_freqs)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(hidden_states, encoder_hidden_states, temb, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, rope_freqs, unpatchify_args)

class OutputLayer(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        # Prevent registering the whole Transformer.
        self.transformer = [transformer]
        # Explicitly register these modules.
        self.norm_out = self.transformer[0].norm_out
        self.proj_out = self.transformer[0].proj_out
        self.pt = transformer.config['patch_size_t']
        self.p = transformer.config['patch_size']

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, temb, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, rope_freqs, unpatchify_args = inputs
        tt, th, tw = (arg.item() for arg in unpatchify_args)
        original_context_length = tt*th*tw

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = hidden_states[:, -original_context_length:, :]

        # ---------------------------- Final layer ------------------------------
        hidden_states = self.proj_out(hidden_states)

        hidden_states = einops.rearrange(hidden_states, 'b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)',
                                         t=tt, h=th, w=tw,
                                         pt=self.pt, ph=self.p, pw=self.p)

        return hidden_states
