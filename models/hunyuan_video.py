from pathlib import Path
import sys
import argparse
import json
import os.path
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/HunyuanVideo'))

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
from hyvideo.modules.posemb_layers import get_nd_rotary_pos_embed
from hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler
from hyvideo.diffusion.pipelines import HunyuanVideoPipeline as OriginalHunyuanVideoPipeline
from hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D

# Framepack
from transformers import SiglipImageProcessor, SiglipVisionModel
import einops
import gc

# In diffusion-pipe, we already converted the dtype to an object. But Hunyuan scripts want the string version in a lot of places.
TYPE_TO_PRECISION = {v: k for k, v in PRECISION_TO_TYPE.items()}

def hf_clip_vision_encode(image, feature_extractor, image_encoder):
    assert isinstance(image, np.ndarray)
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.dtype == np.uint8

    preprocessed = feature_extractor.preprocess(images=image, return_tensors="pt").to(device=image_encoder.device, dtype=image_encoder.dtype)
    image_encoder_output = image_encoder(**preprocessed)

    return image_encoder_output

def get_rotary_pos_embed(transformer, video_length, height, width):
    target_ndim = 3
    ndim = 5 - 2
    rope_theta = 256
    patch_size = transformer.patch_size
    rope_dim_list = transformer.rope_dim_list
    hidden_size = transformer.hidden_size
    heads_num = transformer.heads_num
    head_dim = hidden_size // heads_num

    # 884
    latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]

    if isinstance(patch_size, int):
        assert all(s % patch_size == 0 for s in latents_size), (
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({patch_size}), "
            f"but got {latents_size}."
        )
        rope_sizes = [s // patch_size for s in latents_size]
    elif isinstance(patch_size, list):
        assert all(
            s % patch_size[idx] == 0
            for idx, s in enumerate(latents_size)
        ), (
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({patch_size}), "
            f"but got {latents_size}."
        )
        rope_sizes = [
            s // patch_size[idx] for idx, s in enumerate(latents_size)
        ]

    if len(rope_sizes) != target_ndim:
        rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis

    if rope_dim_list is None:
        rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
    assert (
        sum(rope_dim_list) == head_dim
    ), "sum(rope_dim_list) should equal to head_dim of attention layer"
    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
        rope_dim_list,
        rope_sizes,
        theta=rope_theta,
        use_real=True,
        theta_rescale_factor=1,
    )
    return freqs_cos, freqs_sin

class HunyuanVideoRotaryPosEmbed(nn.Module):
    def __init__(self, rope_dim, theta):
        super().__init__()
        self.DT, self.DY, self.DX = rope_dim
        self.theta = theta

    @torch.no_grad()
    def get_frequency(self, dim, pos):
        T, H, W = pos.shape
        freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device)[: (dim // 2)] / dim))
        freqs = torch.outer(freqs, pos.reshape(-1)).unflatten(-1, (T, H, W)).repeat_interleave(2, dim=0)
        return freqs.cos(), freqs.sin()

    @torch.no_grad()
    def forward_inner(self, frame_indices, height, width, device):
        GT, GY, GX = torch.meshgrid(
            frame_indices.to(device=device, dtype=torch.float32),
            torch.arange(0, height, device=device, dtype=torch.float32),
            torch.arange(0, width, device=device, dtype=torch.float32),
            indexing="ij"
        )

        FCT, FST = self.get_frequency(self.DT, GT)
        FCY, FSY = self.get_frequency(self.DY, GY)
        FCX, FSX = self.get_frequency(self.DX, GX)

        result_cos = torch.cat([FCT, FCY, FCX], dim=0)
        result_sin = torch.cat([FST, FSY, FSX], dim=0)

        return result_cos.to(device), result_sin.to(device)

    @torch.no_grad()
    def forward(self, frame_indices, height, width, device):
        frame_indices = frame_indices.unbind(0)
        results = [self.forward_inner(f, height, width, device) for f in frame_indices]
        results = torch.stack(results, dim=0)
        return results

class VaeAndClip(nn.Module):
    def __init__(self, vae, clip):
        super().__init__()
        self.vae = vae
        self.clip = clip

def pad_for_3d_conv(x, kernel_size):
    b, c, t, h, w = x.shape
    pt, ph, pw = kernel_size
    pad_t = (pt - (t % pt)) % pt
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode='replicate')


def center_down_sample_3d(x, kernel_size):
    # pt, ph, pw = kernel_size
    # cp = (pt * ph * pw) // 2
    # xp = einops.rearrange(x, 'b c (t pt) (h ph) (w pw) -> (pt ph pw) b c t h w', pt=pt, ph=ph, pw=pw)
    # xc = xp[cp]
    # return xc
    return torch.nn.functional.avg_pool3d(x, kernel_size, stride=kernel_size)

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


def _convert_state_dict_keys(model_state_dict, loaded_state_dict):
    if next(iter(loaded_state_dict.keys())).startswith('model.model.'):
        # ComfyUI state_dict format.
        # Construct the key mapping the same way ComfyUI does, then invert it at the very end.
        sd = {}
        for k in list(model_state_dict.keys()):
            key_out = k
            key_out = key_out.replace("txt_in.t_embedder.mlp.0.", "txt_in.t_embedder.in_layer.").replace("txt_in.t_embedder.mlp.2.", "txt_in.t_embedder.out_layer.")
            key_out = key_out.replace("txt_in.c_embedder.linear_1.", "txt_in.c_embedder.in_layer.").replace("txt_in.c_embedder.linear_2.", "txt_in.c_embedder.out_layer.")
            key_out = key_out.replace("_mod.linear.", "_mod.lin.").replace("_attn_qkv.", "_attn.qkv.")
            key_out = key_out.replace("mlp.fc1.", "mlp.0.").replace("mlp.fc2.", "mlp.2.")
            key_out = key_out.replace("_attn_q_norm.weight", "_attn.norm.query_norm.scale").replace("_attn_k_norm.weight", "_attn.norm.key_norm.scale")
            key_out = key_out.replace(".q_norm.weight", ".norm.query_norm.scale").replace(".k_norm.weight", ".norm.key_norm.scale")
            key_out = key_out.replace("_attn_proj.", "_attn.proj.")
            key_out = key_out.replace(".modulation.linear.", ".modulation.lin.")
            key_out = key_out.replace("_in.mlp.2.", "_in.out_layer.").replace("_in.mlp.0.", "_in.in_layer.")
            key_out = 'model.model.' + key_out
            sd[k] = loaded_state_dict[key_out]
        return sd
    else:
        return loaded_state_dict


def vae_encode(tensor, vae):
    # tensor values already in range [-1, 1] here
    latents = vae.encode(tensor).latent_dist.sample()
    return latents * vae.config.scaling_factor

class ClipVisionProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Linear(in_channels, out_channels * 3)
        self.down = nn.Linear(out_channels * 3, out_channels)

    def forward(self, x):
        projected_x = self.down(nn.functional.silu(self.up(x)))
        return projected_x

class HunyuanVideoPatchEmbedForCleanLatents(nn.Module):
    def __init__(self, inner_dim):
        super().__init__()
        self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))

    @torch.no_grad()
    def initialize_weight_from_another_conv3d(self, another_layer):
        weight = another_layer.weight.detach().clone()
        bias = another_layer.bias.detach().clone()

        sd = {
            'proj.weight': weight.clone(),
            'proj.bias': bias.clone(),
            'proj_2x.weight': einops.repeat(weight, 'b c t h w -> b c (t tk) (h hk) (w wk)', tk=2, hk=2, wk=2) / 8.0,
            'proj_2x.bias': bias.clone(),
            'proj_4x.weight': einops.repeat(weight, 'b c t h w -> b c (t tk) (h hk) (w wk)', tk=4, hk=4, wk=4) / 64.0,
            'proj_4x.bias': bias.clone(),
        }

        sd = {k: v.clone() for k, v in sd.items()}

        self.load_state_dict(sd)
        return

class HunyuanVideoPipeline(BasePipeline):
    name = 'hunyuan-video'
    framerate = 30
    checkpointable_layers = ['DoubleBlock', 'SingleBlock']
    adapter_target_modules = ['MMDoubleStreamBlock', 'MMSingleStreamBlock']

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

        self.framepack = config['model'].get('framepack', False)
        self.framepack_window_size = config['model'].get('framepack_window_size', 30)

        if self.framepack:
            print("Loading framepack specific modules")
            siglip_path = config['model'].get('siglip_path', "lllyasviel/flux_redux_bfl")
            self.siglip_feature_extractor = SiglipImageProcessor.from_pretrained(siglip_path, subfolder='feature_extractor')
            image_encoder = SiglipVisionModel.from_pretrained(siglip_path, subfolder='image_encoder', torch_dtype=dtype).eval()

            self.rope = HunyuanVideoRotaryPosEmbed((16, 56, 56), 256.0)
            self.vae_and_clip = VaeAndClip(vae, image_encoder).eval()

    # delay loading transformer to save RAM
    def load_diffusion_model(self):
        transformer_dtype = self.model_config.get('transformer_dtype', self.model_config['dtype'])
        # Device needs to be cuda here or we get an error. We initialize the model with empty weights so it doesn't matter, and
        # then directly load the weights onto CPU right after.
        factor_kwargs = {"device": 'cuda', "dtype": transformer_dtype}
        in_channels = self.args.latent_channels
        out_channels = self.args.latent_channels
        with init_empty_weights():
            transformer = load_model(
                self.args,
                in_channels=in_channels,
                out_channels=out_channels,
                factor_kwargs=factor_kwargs,
            )
            if self.framepack:
                inner_dim = 3072 # constant
                transformer.image_projection = ClipVisionProjection(3, inner_dim)
                transformer.clean_x_embedder = HunyuanVideoPatchEmbedForCleanLatents(inner_dim)
        if transformer_path := self.model_config.get('transformer_path', None):
            state_dict = load_safetensors(transformer_path)
            state_dict = _convert_state_dict_keys(transformer.state_dict(), state_dict)
        else:
            state_dict = load_state_dict(self.args, self.args.model_base)
        params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
        base_dtype = self.model_config['dtype']

        print("transformer.named_parameters()", str(list(transformer.named_parameters())))
        print("state_dict.keys()", str(list(state_dict.keys())))

        for name, param in transformer.named_parameters():
            dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else transformer_dtype
            set_module_tensor_to_device(transformer, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])

        self.diffusers_pipeline.transformer = transformer
        self.transformer.train()
        # We'll need the original parameter name for saving, and the name changes once we wrap modules for pipeline parallelism,
        # so store it in an attribute here. Same thing below if we're training a lora and creating lora weights.
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def __getattr__(self, name):
        if name == "vae" and self.framepack:
            return self.vae_and_clip
        else:
            return getattr(self.diffusers_pipeline, name)

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder, self.text_encoder_2]

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # Diffusers LoRA convention.
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
        total_frames = latents.shape[2]
        ret = {}

        assert (self.framepack_window_size + 3) % 4 == 0
        num_frames_framepack = (self.framepack_window_size + 3) // 4

        # fp_compression_hardcode = 1 + 2 + 16 = 19
        assert total_frames > num_frames_framepack + 19 # with the compressible part
        shift = np.random.randint(1, total_frames - num_frames_framepack)

        # train on stage 1 (full last section ctx)
        if shift > total_frames - num_frames_framepack - 19:
            fragment = latents[:, :, -num_frames_framepack:, ...]
            ret['latents'] = fragment
        else:
            # train on the stages with compression
            fragment = latents[:, :, shift:shift+num_frames_framepack, ...]
            first_frame = latents[:, :, 0:1, :, :]
            the_rest = latents[:, :, shift+num_frames_framepack:, ...]
            # The line above is from the sampler. Does it mean we are adding only 1+19 latent frames to compressed context instead of the whole video, or I'm missing something? --kabachuha
            clean_latents_post, clean_latents_2x, clean_latents_4x = the_rest[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([first_frame, clean_latents_post], dim=2)

            ret['latents'] = fragment
            ret['clean_latents'] = clean_latents
            ret['clean_latents_2x'] = clean_latents_2x
            ret['clean_latents_4x'] = clean_latents_4x

            # getting the frame indices
            indices = torch.arange(0, sum([1, shift, num_frames_framepack, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, shift, num_frames_framepack, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            ret['latent_indices'] = latent_indices
            ret['clean_latent_indices'] = clean_latent_indices
            ret['clean_latent_2x_indices'] = clean_latent_2x_indices
            ret['clean_latent_4x_indices'] = clean_latent_4x_indices

        return ret

    def get_call_vae_fn(self, vae_and_clip):
        def fn(tensor):
            ret = {}

            if self.framepack:
                vae = vae_and_clip.vae
                image_encoder = vae_and_clip.clip

                image_encoder_output = hf_clip_vision_encode((tensor[:, :, 0, ...].clone().squeeze(0).permute(1, 2, 0).numpy()*255.).astype(np.uint8), self.siglip_feature_extractor, image_encoder)
                ret['image_encoder_output'] = image_encoder_output.last_hidden_state.to(image_encoder.device, image_encoder.dtype)
                ret['latents'] = vae_encode(tensor.to(vae.device, vae.dtype), vae)
            else:
                vae = vae_and_clip
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

        if self.transformer.image_projection is not None:
            raise 'Found!'

            image_encoder_last_hidden_state = inputs['image_encoder_output']

            extra_encoder_hidden_states = self.transformer.image_projection(image_encoder_last_hidden_state)
            extra_attention_mask = torch.ones((latents.shape[0], extra_encoder_hidden_states.shape[1]), dtype=prompt_attention_mask_1.dtype, device=prompt_attention_mask_1.device)

            # must cat before (not after) encoder_hidden_states, due to attn masking
            prompt_embeds_1 = torch.cat([extra_encoder_hidden_states, prompt_embeds_1], dim=1)
            prompt_attention_mask_1 = torch.cat([extra_attention_mask, prompt_attention_mask_1], dim=1)
        else:
            raise 'Not found!'

        bs, channels, num_frames, h, w = latents.shape

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

        video_length = (num_frames - 1) * 4 + 1
        video_height = h * 8
        video_width = w * 8
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            self.transformer, video_length, video_height, video_width
        )
        freqs_cos = freqs_cos.expand(bs, -1, -1)
        freqs_sin = freqs_sin.expand(bs, -1, -1)

        if self.framepack and 'clean_latents' in inputs:

            rtf = self.random_timecrop_framepack(latents)

            clean_latents = rtf['clean_latents']
            clean_latents_2x = rtf['clean_latents_2x']
            clean_latents_4x = rtf['clean_latents_4x']

            latent_indices = rtf['latent_indices']
            clean_latent_indices = rtf['clean_latent_indices']
            clean_latent_2x_indices = rtf['clean_latent_2x_indices']
            clean_latent_4x_indices = rtf['clean_latent_4x_indices']
        else:
            # dummy clean latents for microbatch splitting
            clean_latents = torch.zeros((x_t.shape[0],), device=x_t.device, dtype=x_t.dtype)
            clean_latents_2x = torch.zeros((x_t.shape[0],), device=x_t.device, dtype=x_t.dtype)
            clean_latents_4x = torch.zeros((x_t.shape[0],), device=x_t.device, dtype=x_t.dtype)

            latent_indices = torch.zeros((x_t.shape[0],), device=x_t.device, dtype=x_t.dtype)
            clean_latent_indices = torch.zeros((x_t.shape[0],), device=x_t.device, dtype=x_t.dtype)
            clean_latent_2x_indices = torch.zeros((x_t.shape[0],), device=x_t.device, dtype=x_t.dtype)
            clean_latent_4x_indices = torch.zeros((x_t.shape[0],), device=x_t.device, dtype=x_t.dtype)

        return (
            x_t,
            t,
            prompt_embeds_1,
            prompt_attention_mask_1,
            prompt_embeds_2,
            freqs_cos,
            freqs_sin,
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
        for i, block in enumerate(transformer.double_blocks):
            layers.append(DoubleBlock(block, i, self.offloader_double))
        layers.append(concatenate_hidden_states)
        for i, block in enumerate(transformer.single_blocks):
            layers.append(SingleBlock(block, i, self.offloader_single))
        layers.append(OutputLayer(transformer))
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
    def __init__(self, transformer):
        super().__init__()
        # Prevent registering the whole Transformer.
        self.transformer = [transformer]
        # Explicitly register these modules.
        self.time_in = self.transformer[0].time_in
        self.vector_in = self.transformer[0].vector_in
        self.guidance_embed = self.transformer[0].guidance_embed
        self.guidance_in = self.transformer[0].guidance_in
        self.img_in = self.transformer[0].img_in
        self.text_projection = self.transformer[0].text_projection
        self.txt_in = self.transformer[0].txt_in

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)

        x, t, text_states, text_mask, text_states_2, freqs_cos, freqs_sin, guidance, clean_latents, clean_latents_2x, clean_latents_4x, latent_indices, clean_latent_indices, clean_latent_2x_indices, clean_latent_4x_indices = inputs

        B, _, T, H, W = x.shape
        tt, th, tw = (
            T // self.transformer[0].patch_size[0],
            H // self.transformer[0].patch_size[1],
            W // self.transformer[0].patch_size[2],
        )
        unpatchify_args = torch.tensor([tt, th, tw], device=x.device)

        img = x
        txt = text_states

        # Prepare modulation vectors.
        vec = self.time_in(t)

        # text modulation
        vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        hidden_states = self.img_in(img)

        # Framepack handling of latents
        if self.framepack:

            if latent_indices is None:
                latent_indices = torch.arange(0, T).unsqueeze(0).expand(B, -1)

            # NOTE: we have cos and sin separately because we're working with the original HY code
            freqs_cos, freqs_sin = self.rope(frame_indices=latent_indices, height=H, width=W, device=hidden_states.device)
            freqs_cos = freqs_cos.flatten(2).transpose(1, 2)
            freqs_sin = freqs_sin.flatten(2).transpose(1, 2)

            if clean_latents is not None and clean_latent_indices is not None and len(clean_latents.shape) > 1:
                clean_latents = clean_latents.to(hidden_states)
                clean_latents = self.clean_x_embedder.proj(clean_latents)
                clean_latents = clean_latents.flatten(2).transpose(1, 2)

                clean_latent_rope_freqs_cos, clean_latent_rope_freqs_sin = self.rope(frame_indices=clean_latent_indices, height=H, width=W, device=clean_latents.device)
                clean_latent_rope_freqs_cos = clean_latent_rope_freqs_cos.flatten(2).transpose(1, 2)
                clean_latent_rope_freqs_sin = clean_latent_rope_freqs_sin.flatten(2).transpose(1, 2)

                hidden_states = torch.cat([clean_latents, hidden_states], dim=1)
                freqs_cos = torch.cat([clean_latent_rope_freqs_cos, freqs_cos], dim=1)
                freqs_sin = torch.cat([clean_latent_rope_freqs_sin, freqs_sin], dim=1)

            if clean_latents_2x is not None and clean_latent_2x_indices is not None and len(clean_latent_2x_indices.shape) > 1:
                clean_latents_2x = clean_latents_2x.to(hidden_states)
                clean_latents_2x = pad_for_3d_conv(clean_latents_2x, (2, 4, 4))
                clean_latents_2x = self.clean_x_embedder.proj_2x(clean_latents_2x)
                clean_latents_2x = clean_latents_2x.flatten(2).transpose(1, 2)

                clean_latent_2x_rope_freqs_cos, clean_latent_2x_rope_freqs_sin = self.rope(frame_indices=clean_latent_2x_indices, height=H, width=W, device=clean_latents_2x.device)
                clean_latent_2x_rope_freqs_cos = pad_for_3d_conv(clean_latent_2x_rope_freqs_cos, (2, 2, 2))
                clean_latent_2x_rope_freqs_cos = center_down_sample_3d(clean_latent_2x_rope_freqs_cos, (2, 2, 2))
                clean_latent_2x_rope_freqs_cos = clean_latent_2x_rope_freqs_cos.flatten(2).transpose(1, 2)
                clean_latent_2x_rope_freqs_sin = pad_for_3d_conv(clean_latent_2x_rope_freqs_sin, (2, 2, 2))
                clean_latent_2x_rope_freqs_sin = center_down_sample_3d(clean_latent_2x_rope_freqs_sin, (2, 2, 2))
                clean_latent_2x_rope_freqs_sin = clean_latent_2x_rope_freqs_sin.flatten(2).transpose(1, 2)

                hidden_states = torch.cat([clean_latents_2x, hidden_states], dim=1)
                freqs_cos = torch.cat([clean_latent_2x_rope_freqs_cos, freqs_cos], dim=1)
                freqs_sin = torch.cat([clean_latent_2x_rope_freqs_sin, freqs_sin], dim=1)

            if clean_latents_4x is not None and clean_latent_4x_indices is not None and len(clean_latent_4x_indices.shape) > 1:
                clean_latents_4x = clean_latents_4x.to(hidden_states)
                clean_latents_4x = pad_for_3d_conv(clean_latents_4x, (4, 8, 8))
                clean_latents_4x = self.clean_x_embedder.proj_4x(clean_latents_4x)
                clean_latents_4x = clean_latents_4x.flatten(2).transpose(1, 2)

                clean_latent_4x_rope_freqs_cos, clean_latent_4x_rope_freqs_sin = self.rope(frame_indices=clean_latent_4x_indices, height=H, width=W, device=clean_latents_4x.device)
                clean_latent_4x_rope_freqs_cos = pad_for_3d_conv(clean_latent_4x_rope_freqs_cos, (4, 4, 4))
                clean_latent_4x_rope_freqs_cos = center_down_sample_3d(clean_latent_4x_rope_freqs_cos, (4, 4, 4))
                clean_latent_4x_rope_freqs_cos = clean_latent_4x_rope_freqs_cos.flatten(2).transpose(1, 2)
                clean_latent_4x_rope_freqs_sin = pad_for_3d_conv(clean_latent_4x_rope_freqs_sin, (4, 4, 4))
                clean_latent_4x_rope_freqs_sin = center_down_sample_3d(clean_latent_4x_rope_freqs_sin, (4, 4, 4))
                clean_latent_4x_rope_freqs_sin = clean_latent_4x_rope_freqs_sin.flatten(2).transpose(1, 2)

                hidden_states = torch.cat([clean_latents_4x, hidden_states], dim=1)
                freqs_cos = torch.cat([clean_latent_4x_rope_freqs_cos, freqs_cos], dim=1)
                freqs_sin = torch.cat([clean_latent_4x_rope_freqs_sin, freqs_sin], dim=1)

        # diffusion-pipe makes all input tensors have a batch dimension, but Hunyuan code wants these to not have batch dim
        assert freqs_cos.ndim == 3
        freqs_cos, freqs_sin = freqs_cos[0], freqs_sin[0]

        img = hidden_states

        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.transformer[0].use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Compute cu_squlens and max_seqlen for flash attention
        cu_seqlens = get_cu_seqlens(text_mask, img_seq_len)

        # Everything passed between layers needs to be a CUDA tensor for Deepspeed pipeline parallelism.
        txt_seq_len = torch.tensor(txt_seq_len, device=img.device)
        img_seq_len = torch.tensor(img_seq_len, device=img.device)
        max_seqlen = img_seq_len + txt_seq_len

        return make_contiguous(img, txt, vec, cu_seqlens, max_seqlen, freqs_cos, freqs_sin, txt_seq_len, img_seq_len, unpatchify_args)


class DoubleBlock(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        img, txt, vec, cu_seqlens, max_seqlen, freqs_cos, freqs_sin, txt_seq_len, img_seq_len, unpatchify_args = inputs

        self.offloader.wait_for_block(self.block_idx)
        img, txt = self.block(img, txt, vec, cu_seqlens, cu_seqlens, max_seqlen.item(), max_seqlen.item(), (freqs_cos, freqs_sin))
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(img, txt, vec, cu_seqlens, max_seqlen, freqs_cos, freqs_sin, txt_seq_len, img_seq_len, unpatchify_args)


def concatenate_hidden_states(inputs):
    img, txt, vec, cu_seqlens, max_seqlen, freqs_cos, freqs_sin, txt_seq_len, img_seq_len, unpatchify_args = inputs
    x = torch.cat((img, txt), 1)
    return x, vec, cu_seqlens, max_seqlen, freqs_cos, freqs_sin, txt_seq_len, img_seq_len, unpatchify_args


class SingleBlock(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, vec, cu_seqlens, max_seqlen, freqs_cos, freqs_sin, txt_seq_len, img_seq_len, unpatchify_args = inputs

        self.offloader.wait_for_block(self.block_idx)
        x = self.block(x, vec, txt_seq_len.item(), cu_seqlens, cu_seqlens, max_seqlen.item(), max_seqlen.item(), (freqs_cos, freqs_sin))
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(x, vec, cu_seqlens, max_seqlen, freqs_cos, freqs_sin, txt_seq_len, img_seq_len, unpatchify_args)

class OutputLayer(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        # Prevent registering the whole Transformer.
        self.transformer = [transformer]
        # Explicitly register these modules.
        self.final_layer = self.transformer[0].final_layer

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, vec, cu_seqlens, max_seqlen, freqs_cos, freqs_sin, txt_seq_len, img_seq_len, unpatchify_args = inputs
        img = x[:, :img_seq_len.item(), ...]

        tt, th, tw = (arg.item() for arg in unpatchify_args)
        # trimming if framepack
        img = img[:, -tt*th*tw:, :]

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        return self.transformer[0].unpatchify(img, tt, th, tw)
