from pathlib import Path
import re
import tarfile
import os
import sys
from collections import defaultdict
import types
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/ComfyUI'))

import peft
import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
import safetensors.torch
import torchvision
from PIL import Image, ImageOps
from torchvision import transforms
import accelerate
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm

from utils.common import is_main_process, VIDEO_EXTENSIONS, round_to_nearest_multiple, round_down_to_multiple, AUTOCAST_DTYPE, empty_cuda_cache
import comfy.utils
import comfy.sd
import comfy.sd1_clip
from comfy.sd1_clip import SD1Tokenizer
from comfy import model_management
from comfy_api.latest import InputImpl

# Avoids using comfy_kitchen RoPE implementations that don't have backward defined
model_management.in_training = True
# Increase this from the default. I OOM on text embedding caching on Minimax without this.
model_management.EXTRA_RESERVED_VRAM = 2000 * 1024 * 1024


def make_contiguous(*values):
    return tuple(x.contiguous() if torch.is_tensor(x) else x for x in values)


def convert_crop_and_resize(pil_img, width_and_height):
    if pil_img.mode not in ['RGB', 'RGBA'] and 'transparency' in pil_img.info:
        pil_img = pil_img.convert('RGBA')

    # add white background for transparent images
    if pil_img.mode == 'RGBA':
        canvas = Image.new('RGBA', pil_img.size, (255, 255, 255))
        canvas.alpha_composite(pil_img)
        pil_img = canvas.convert('RGB')
    else:
        pil_img = pil_img.convert('RGB')

    return ImageOps.fit(pil_img, width_and_height)


class PreprocessMediaFile:
    def __init__(self, config, support_video=False, support_audio=False, framerate=None, audio_sample_rate=None, round_height=16, round_width=16, round_frames=4):
        self.config = config
        self.video_clip_mode = config.get('video_clip_mode', 'single_beginning')
        print(f'using video_clip_mode={self.video_clip_mode}')
        self.pil_to_tensor = transforms.Compose([transforms.ToTensor()])
        self.support_video = support_video
        self.support_audio = support_audio
        if self.support_audio:
            self.support_video = True
        self.framerate = framerate
        print(f'using framerate={self.framerate}')
        self.audio_sample_rate = audio_sample_rate
        print(f'using audio_sample_rate={self.audio_sample_rate}')
        self.round_height = round_height
        self.round_width = round_width
        self.round_frames = round_frames
        if self.support_video:
            assert self.framerate
        if self.support_audio:
            assert self.audio_sample_rate
        self.tarfile_map = {}

    def __del__(self):
        for tar_f in self.tarfile_map.values():
            tar_f.close()

    def align_frames(self, frames):
        return round_down_to_multiple(frames - 1, self.round_frames) + 1

    # Resamples video tensor to convert to self.framerate
    def convert_framerate(self, video: torch.Tensor, source_fps: float):
        num_frames = video.shape[0]
        max_frame_index = num_frames - 1
        new_num_frames = int(num_frames * self.framerate / source_fps)
        frames_to_sample = torch.linspace(0, max_frame_index, new_num_frames).to(torch.int32)
        return video[frames_to_sample]

    def convert_audio_sample_rate(self, audio: torch.Tensor, source_sample_rate: int):
        if source_sample_rate != self.audio_sample_rate:
            audio = torchaudio.functional.resample(audio, source_sample_rate, self.audio_sample_rate)
        return audio

    def extract_clips(self, video, audio, target_frames, video_clip_mode):
        # video is (channels, num_frames, height, width)
        frames = video.shape[1]
        if frames < target_frames:
            # Dataset code has long since required a 1-to-1 mapping (assert checked). So just fail here if this ever happens (I think maybe it can't happen anymore).
            raise ValueError(f'video with shape {video.shape} has less frames ({frames}) than the target_frames {target_frames}')

        if audio is not None:
            target_audio_samples = int(target_frames / self.framerate * self.audio_sample_rate)
            # TODO: can we handle this better? Probably caused by rounding, variable frame rate, framerate conversion, audio track that is a tiny bit too short in the underlying video, etc.
            if audio.shape[-1] < target_audio_samples:
                print(f'WARNING: audio length {audio.shape[-1]} is shorter than the required {target_audio_samples}. This can rarely happen. Padding with silence.')
                audio = F.pad(audio, (0, target_audio_samples - audio.shape[-1]))

        if video_clip_mode == 'single_beginning':
            if audio is not None:
                audio = audio[:, :, :target_audio_samples]
            return [(video[:, :target_frames, ...], audio)]
        elif video_clip_mode == 'single_middle':
            if audio is not None:
                audio_start = int((audio.shape[-1] - target_audio_samples) / 2)
                audio = audio[:, :, audio_start:audio_start+target_audio_samples]
            start = int((frames - target_frames) / 2)
            assert frames-start >= target_frames
            return [(video[:, start:start+target_frames, ...], audio)]
        # elif video_clip_mode == 'multiple_overlapping':
        #     # Extract multiple clips so we use the whole video for training.
        #     # The clips might overlap a little bit. We never cut anything off the end of the video.
        #     num_clips = ((frames - 1) // target_frames) + 1
        #     start_indices = torch.linspace(0, frames-target_frames, num_clips).int()
        #     return [video[:, i:i+target_frames, ...] for i in start_indices]
        else:
            raise NotImplementedError(f'video_clip_mode={video_clip_mode} is not recognized')

    def __call__(self, spec, mask_filepath, size_bucket=None):
        extension = Path(spec[1]).suffix
        is_video = (extension in VIDEO_EXTENSIONS)

        if spec[0] is None:
            tar_f = None
            filepath_or_file = str(spec[1])
        else:
            tar_filename = spec[0]
            if tar_filename not in self.tarfile_map:
                self.tarfile_map[tar_filename] = tarfile.TarFile(tar_filename)
            tar_f = self.tarfile_map[tar_filename]
            filepath_or_file = tar_f.extractfile(str(spec[1]))

        audio = None
        if is_video:
            assert self.support_video
            comfy_video = InputImpl.VideoFromFile(filepath_or_file)
            components = comfy_video.get_components()
            video = components.images  # [f, h, w, c]
            video = video.movedim(-1, 1)  # need [f, c, h, w]
            video = self.convert_framerate(video, float(components.frame_rate))
            num_frames, _, height, width = video.shape
            if self.support_audio and components.audio is not None:
                audio = components.audio['waveform']
                sample_rate = components.audio['sample_rate']
                audio = self.convert_audio_sample_rate(audio, sample_rate)
        else:
            num_frames = 1
            pil_img = Image.open(filepath_or_file)
            height, width = pil_img.height, pil_img.width
            video = [pil_img]

        if size_bucket is not None:
            size_bucket_width, size_bucket_height, size_bucket_frames = size_bucket
        else:
            size_bucket_width, size_bucket_height, size_bucket_frames = width, height, num_frames

        height_rounded = round_to_nearest_multiple(size_bucket_height, self.round_height)
        width_rounded = round_to_nearest_multiple(size_bucket_width, self.round_width)
        frames_rounded = self.align_frames(size_bucket_frames)
        resize_wh = (width_rounded, height_rounded)

        if mask_filepath:
            mask_img = Image.open(mask_filepath).convert('RGB')
            img_hw = (height, width)
            mask_hw = (mask_img.height, mask_img.width)
            if mask_hw != img_hw:
                raise ValueError(
                    f'Mask shape {mask_hw} was not the same as image shape {img_hw}.\n'
                    f'Image path: {spec[1]}\n'
                    f'Mask path: {mask_filepath}'
                )
            mask_img = ImageOps.fit(mask_img, resize_wh)
            mask = torchvision.transforms.functional.to_tensor(mask_img)[0].to(torch.float16)  # use first channel
        else:
            mask = None

        resized_video = torch.empty((num_frames, 3, height_rounded, width_rounded))
        for i, frame in enumerate(video):
            if not isinstance(frame, Image.Image):
                frame = torchvision.transforms.functional.to_pil_image(frame)
            cropped_image = convert_crop_and_resize(frame, resize_wh)
            tmp = self.pil_to_tensor(cropped_image)
            resized_video[i, ...] = tmp

        if hasattr(filepath_or_file, 'close'):
            filepath_or_file.close()

        if not self.support_video:
            return [(resized_video.squeeze(0), None, mask)]

        # (num_frames, channels, height, width) -> (channels, num_frames, height, width)
        resized_video = torch.permute(resized_video, (1, 0, 2, 3))
        if not is_video:
            return [(resized_video, None, mask)]
        else:
            items = self.extract_clips(resized_video, audio, frames_rounded, self.video_clip_mode)
            assert len(items) <= 1  # only support 0 or 1 extracted clip; dataset limitation
            return [(item[0], item[1], mask) for item in items]


# shared functionality between BasePipeline and ComfyPipeline
class CommonPipeline:
    framerate = None
    pixels_round_to_multiple = 16
    keep_in_high_precision = []
    spatial_compression = 8
    channels = 16
    is_video_vae = False

    def __init__(self, *args, **kwargs):
        # sampling only
        self.sample_steps = 20
        self.sample_shift = 3
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=self.sample_shift)
        sigmas = torch.linspace(1.0, 1 / self.sample_steps, self.sample_steps)
        self.scheduler.set_timesteps(sigmas=sigmas, device='cuda')

    @torch.no_grad()
    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def prepare_sample_test(self, prompt, negative_prompt='', cfg=1):
        inputs = {}
        inputs_uncond = {}
        for te in self.get_text_encoders():
            if isinstance(te, nn.Module):
                te = te.to('cuda')
            else:
                te.load_model_if_needed()
            call_text_encoder_fn = self.get_call_text_encoder_fn(te)
            inputs.update(call_text_encoder_fn([prompt], is_video=[False]))
            if cfg > 1:
                inputs_uncond.update(call_text_encoder_fn([negative_prompt], is_video=[False]))
            if isinstance(te, nn.Module):
                te = te.to('cpu')
            else:
                model_management.unload_all_models()
        self.conds = tuple(tensor.cuda() for tensor in self.get_conds(inputs))
        if cfg > 1:
            self.unconds = tuple(tensor.cuda() for tensor in self.get_conds(inputs_uncond))
        self.sample_cfg = cfg

    def get_call_vae_fn(self, vae):
        def fn(images):
            images = images.to('cuda', self.dtype)
            latents = self.vae_encode(images)
            return {'latents': latents}
        return fn

    def get_target_modules(self, target_model):
        target_modules = set()
        for name, module in target_model.named_modules():
            if module.__class__.__name__ not in self.adapter_target_modules:
                continue
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, nn.Linear):
                    target_modules.add(full_submodule_name)
        return list(target_modules)

    def configure_adapter(self, target_model, adapter_config):
        target_modules = self.get_target_modules(target_model)

        adapter_type = adapter_config['type']
        if adapter_type == 'lora':
            peft_config = peft.LoraConfig(
                r=adapter_config['rank'],
                lora_alpha=adapter_config['alpha'],
                lora_dropout=adapter_config['dropout'],
                bias='none',
                target_modules=target_modules,
                exclude_modules=adapter_config.get('exclude_modules', None),
            )
        elif adapter_type == 'lokr':
            peft_config = peft.LoKrConfig(
                r=adapter_config['rank'],
                decompose_factor=adapter_config['decompose_factor'],
                alpha=adapter_config['alpha'],
                rank_dropout=adapter_config['rank_dropout'],
                target_modules=target_modules,
                exclude_modules=adapter_config.get('exclude_modules', None),
            )
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
        self.peft_config = peft_config
        self.lora_model = peft.get_peft_model(target_model, peft_config)
        if is_main_process():
            self.lora_model.print_trainable_parameters()
        for name, p in target_model.named_parameters():
            p.original_name = name
            if p.requires_grad:
                p.data = p.data.to(adapter_config['dtype'])

        # Better init for LoKr, avoids very small starting gradients that takes a long time to recover from.
        # TODO: decide if we want to do this
        # for name, p in self.lora_model.named_parameters():
        #     if 'lokr_w1.' in name:
        #         nn.init.ones_(p)
        #     elif 'lokr_w2.' in name:
        #         nn.init.zeros_(p)

    @torch.no_grad()
    def sample(self, w=512, h=512):
        x = torch.randn((1, self.channels, h//self.spatial_compression, w//self.spatial_compression), device='cuda')
        if self.is_video_vae:
            x = x.unsqueeze(2)
        timesteps = self.scheduler.timesteps
        for i, step in enumerate(tqdm(timesteps, desc='Sampling')):
            t = step / 1000
            t = t.float().view(1)
            inputs = (x, t, *self.conds)
            v = self.pipeline_model(inputs).float()
            if self.sample_cfg > 1:
                inputs_uncond = (x, t, *self.unconds)
                v_uncond = self.pipeline_model(inputs_uncond).float()
                v = v_uncond + self.sample_cfg*(v - v_uncond)
            x = self.scheduler.step(v, step, x, return_dict=False)[0]
        vae = self.get_vae()
        if isinstance(vae, nn.Module):
            vae = vae.to('cuda')
        else:
            vae.load_model_if_needed()
        img = self.vae_decode(x)
        if isinstance(vae, nn.Module):
            vae = vae.to('cpu')
        else:
            model_management.unload_all_models()
        if img.ndim == 5:
            # in case of video VAE
            img = img.squeeze(1)
        return img

    def free_vae_and_te(self):
        pass


class BasePipeline(CommonPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def load_diffusion_model(self):
        pass

    def get_vae(self):
        raise NotImplementedError()

    def get_text_encoders(self):
        raise NotImplementedError()

    def configure_adapter(self, adapter_config):
        super().configure_adapter(self.transformer, adapter_config)

    def save_adapter(self, save_dir, peft_state_dict):
        raise NotImplementedError()

    def load_adapter_weights(self, adapter_path):
        if is_main_process():
            print(f'Loading adapter weights from path {adapter_path}')
        safetensors_files = list(Path(adapter_path).glob('*.safetensors'))
        if len(safetensors_files) == 0:
            raise RuntimeError(f'No safetensors file found in {adapter_path}')
        if len(safetensors_files) > 1:
            raise RuntimeError(f'Multiple safetensors files found in {adapter_path}')
        adapter_state_dict = safetensors.torch.load_file(safetensors_files[0])
        modified_state_dict = {}
        model_parameters = set(name for name, p in self.transformer.named_parameters())
        for k, v in adapter_state_dict.items():
            # Replace Diffusers or ComfyUI prefix
            k = re.sub(r'^(transformer|diffusion_model)\.', '', k)
            # Replace weight at end for LoRA format
            k = re.sub(r'\.weight$', '.default.weight', k)
            if k not in model_parameters:
                raise RuntimeError(f'modified_state_dict key {k} is not in the model parameters')
            modified_state_dict[k] = v
        self.transformer.load_state_dict(modified_state_dict, strict=False)

    def load_and_fuse_adapter(self, path):
        peft_config = peft.LoraConfig.from_pretrained(path)
        lora_model = peft.get_peft_model(self.transformer, peft_config)
        self.load_adapter_weights(path)
        lora_model.merge_and_unload()

    def save_model(self, save_dir, diffusers_sd):
        raise NotImplementedError()

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(self.config, support_video=False)

    def get_call_text_encoder_fn(self, text_encoder):
        raise NotImplementedError()

    def prepare_inputs(self, inputs, timestep_quantile=None):
        raise NotImplementedError()

    def to_layers(self):
        raise NotImplementedError()

    def model_specific_dataset_config_validation(self, dataset_config):
        pass

    # Get param groups that will be passed into the optimizer. Models can override this, e.g. SDXL
    # supports separate learning rates for unet and text encoders.
    def get_param_groups(self, parameters):
        return [{'params': parameters}]

    # Default loss_fn. MSE between output and target, with mask support.
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

    def enable_block_swap(self, blocks_to_swap):
        raise NotImplementedError('Block swapping is not implemented for this model')

    def prepare_block_swap_training(self):
        pass

    def prepare_block_swap_inference(self, disable_block_swap=False):
        pass


def encode_token_weights(self, token_weight_pairs):
    to_encode = list()
    max_token_len = 0
    has_weights = False
    for x in token_weight_pairs:
        tokens = list(map(lambda a: a[0], x))
        max_token_len = max(len(tokens), max_token_len)
        has_weights = has_weights or not all(map(lambda a: a[1] == 1.0, x))
        to_encode.append(tokens)

    sections = len(to_encode)
    if has_weights or sections == 0:
        if hasattr(self, "gen_empty_tokens"):
            to_encode.append(self.gen_empty_tokens(self.special_tokens, max_token_len))
        else:
            to_encode.append(comfy.sd1_clip.gen_empty_tokens(self.special_tokens, max_token_len))

    o = self.encode(to_encode)
    out, pooled = o[:2]

    # if pooled is not None:
    #     first_pooled = pooled[0:1].to(model_management.intermediate_device())
    # else:
    #     first_pooled = pooled
    assert pooled is None
    first_pooled = None

    output = []
    for k in range(0, sections):
        z = out[k:k+1]
        if has_weights:
            z_empty = out[-1]
            for i in range(len(z)):
                for j in range(len(z[i])):
                    weight = token_weight_pairs[k][j][1]
                    if weight != 1.0:
                        z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
        output.append(z)

    if (len(output) == 0):
        r = (out[-1:].to(model_management.intermediate_device()), first_pooled)
    else:
        r = (torch.cat(output, dim=0).to(model_management.intermediate_device()), first_pooled)

    if len(o) > 2:
        extra = {}
        for k in o[2]:
            v = o[2][k]
            extra[k] = v

        r = r + (extra,)
    return r

# Handle batch of different prompts.
comfy.sd1_clip.ClipTokenWeightEncoder.encode_token_weights = encode_token_weights


class ModelWrapper:
    '''Simple wrapper that delays loading the model, since the model isn't needed if the training data is already cached.'''
    def __init__(self, load_fn):
        self._load_fn = load_fn
        self._model = None

    def __getattr__(self, name):
        if self._model is None:
            raise RuntimeError("Model wasn't loaded, this shouldn't ever happen.")
        return getattr(self._model, name)

    def load_model_if_needed(self):
        if self._model is None:
            self._model = self._load_fn()


def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs):
    kwargs['disable_weights'] = True
    out = {}
    out[self.clip_name] = getattr(self, self.clip).tokenize_with_weights(text, return_word_ids, **kwargs)
    return out
# patch to always disable weights, even if the child class doesn't
SD1Tokenizer.tokenize_with_weights = tokenize_with_weights


def maybe_pad(tokens, min_length, tokenizer):
    amount = min_length - len(tokens)
    if amount <= 0:
        return
    if tokenizer.pad_left:
        for _ in range(amount):
            tokens.insert(0, (tokenizer.pad_token, 1.0))
    else:
        tokens.extend([(tokenizer.pad_token, 1.0)] * amount)


class ComfyPipeline(CommonPipeline):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = self.config['model']
        self.dtype = self.model_config['dtype']
        self.latent_format = None

        # VAE
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
            vae.vae_encode_crop_pixels = types.MethodType(vae_encode_crop_pixels, self.vae)
            return vae

        self.vae = ModelWrapper(load_fn)

        # Text encoders
        self.text_encoders = []
        for te_config in self.model_config['text_encoders']:
            if 'paths' in te_config:
                paths = te_config['paths']
            elif 'path' in te_config:
                paths = te_config['path']
            else:
                raise ValueError('need text encoder path in config')
            if isinstance(paths, str):
                paths = [paths]
            clip_type = getattr(comfy.sd.CLIPType, te_config['type'].upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)

            def load_fn():
                return comfy.sd.load_clip(ckpt_paths=paths, clip_type=clip_type, disable_dynamic=True)

            self.text_encoders.append(ModelWrapper(load_fn))

    def dequantize(self, model, diffusion_model_dtype):
        operations = comfy.ops.disable_weight_init
        for mod_name, module in model.named_children():
            is_quantized = False
            for p_name, p in module.named_parameters(recurse=False):
                if p.__class__.__name__ == 'QuantizedTensor':
                    module.register_parameter(p_name, nn.Parameter(p.dequantize()))
                    p = getattr(module, p_name)
                    is_quantized = True

                name = f'{mod_name}.{p_name}'
                if any(keyword in name for keyword in self.keep_in_high_precision) or p.ndim == 1:
                    continue
                p.data = p.data.to(diffusion_model_dtype)

            if is_quantized:
                bias = module.bias is not None
                with accelerate.init_empty_weights():
                    new_linear = operations.Linear(module.in_features, module.out_features, bias=bias)
                    new_linear.comfy_cast_weights = True  # model_patcher.set_model_compute_dtype() would normally set this
                new_linear.weight = module.weight
                if bias:
                    new_linear.bias = module.bias
                model._modules[mod_name] = new_linear

            self.dequantize(module, diffusion_model_dtype)

    def patch_quantized_modules(self, model):
        # For every class that acts like a Linear but isn't a nn.Linear (e.g. quant linear), force it to be a nn.Linear
        # subclass so that we can target it with LoRA.
        linear_classes = set()
        for module in model.modules():
            if module.__class__.__name__ == 'Linear' and not isinstance(module, nn.Linear):
                linear_classes.add(module.__class__)
        for klass in linear_classes:
            bases = klass.__bases__
            assert bases[0] == nn.Module
            klass.__bases__ = (nn.Linear, *bases[1:])
            # Get rid of the special quant state dict saving/loading so Deepspeed checkpointing works. We only can train adapters on top
            # of quantized weights, so the base model weights never need to be saved / loaded in checkpoints anyway.
            del klass.state_dict
            del klass._load_from_state_dict
        return len(linear_classes) > 0

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        model_options = {}
        model_options['dtype'] = dtype
        model_patcher = comfy.sd.load_diffusion_model(self.model_config['diffusion_model'], model_options=model_options, disable_dynamic=True)

        merging_adapter = False
        for adapter_path in self.model_config.get('merge_adapters', []):
            merging_adapter = True
            if is_main_process():
                print(f'Merging adapter {adapter_path}')
            sd = comfy.utils.load_torch_file(adapter_path, safe_load=True)
            model_patcher, _ = comfy.sd.load_lora_for_models(model_patcher, None, sd, 1.0, 0.0)
            del sd

        model_patcher.set_model_compute_dtype(dtype)
        with torch.no_grad():
            # Use a few GB of VRAM in case we are merging an adapter (compute intensive).
            if merging_adapter:
                model_patcher.patch_model(device_to='cuda', lowvram_model_memory=4*(1024**3))
            # But ONLY if we have an adapter to merge, else block swapping fails with a CUDA error (???)
            else:
                model_patcher.patch_model()
        self.diffusion_model = model_patcher.model.diffusion_model
        # If we merged an adapter, this object stores an entire extra copy of the weights, so delete it now.
        del model_patcher

        if diffusion_model_dtype := self.model_config.get('diffusion_model_dtype', None):
            self.dequantize(self.diffusion_model, diffusion_model_dtype)

        self.diffusion_model.train()
        for name, p in self.diffusion_model.named_parameters():
            p.original_name = name
            p.requires_grad_(True)

        patched_something = self.patch_quantized_modules(self.diffusion_model)
        if patched_something:
            assert 'adapter' in self.config, 'You are trying to full finetune a quantized model which will not work'

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return self.text_encoders

    def free_vae_and_te(self):
        del self.vae
        del self.text_encoders
        empty_cuda_cache()

    def vae_encode(self, img):
        # move channel dim to end
        # works for both images (b c h w) and video (b c f h w)
        img = img.movedim(1, -1)
        latents = self.vae.encode(img)
        latents = self.latent_format.process_in(latents)
        return latents

    def vae_decode(self, latents):
        if self.latent_format is not None:
            latents = self.latent_format.process_out(latents)
        return self.vae.decode(latents)

    def configure_adapter(self, adapter_config):
        super().configure_adapter(self.diffusion_model, adapter_config)

    def save_adapter(self, save_dir, sd):
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI format.
        sd = {'diffusion_model.'+k: v for k, v in sd.items()}
        safetensors.torch.save_file(sd, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def load_adapter_weights(self, adapter_path):
        if is_main_process():
            print(f'Loading adapter weights from path {adapter_path}')
        safetensors_files = list(Path(adapter_path).glob('*.safetensors'))
        if len(safetensors_files) == 0:
            raise RuntimeError(f'No safetensors file found in {adapter_path}')
        if len(safetensors_files) > 1:
            raise RuntimeError(f'Multiple safetensors files found in {adapter_path}')
        adapter_state_dict = safetensors.torch.load_file(safetensors_files[0])
        modified_state_dict = {}
        model_parameters = set(name for name, p in self.diffusion_model.named_parameters())
        for k, v in adapter_state_dict.items():
            # Replace Diffusers or ComfyUI prefix
            k = re.sub(r'^(transformer|diffusion_model)\.', '', k)
            # Replace weight at end for LoRA format
            k = re.sub(r'\.weight$', '.default.weight', k)
            if k not in model_parameters:
                raise RuntimeError(f'modified_state_dict key {k} is not in the model parameters')
            modified_state_dict[k] = v
        self.diffusion_model.load_state_dict(modified_state_dict, strict=False)

    def load_and_fuse_adapter(self, path):
        raise NotImplementedError()

    def save_model(self, save_dir, sd):
        safetensors.torch.save_file(sd, save_dir / 'model.safetensors', metadata={'format': 'pt'})

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(self.config, support_video=False)

    def get_call_text_encoder_fn(self, text_encoder):
        te_idx = None
        for i, te in enumerate(self.text_encoders):
            if text_encoder == te:
                te_idx = i
                break
        if te_idx is None:
            raise RuntimeError('Unknown text encoder')

        @torch.inference_mode()
        def fn(captions: list[str], is_video: list[bool]):
            tokenizer = getattr(text_encoder.tokenizer, text_encoder.tokenizer.clip)
            original_min_length = tokenizer.min_length

            max_length = 0
            token_lengths = [0]*len(captions)
            for i, text in enumerate(captions):
                tokens = text_encoder.tokenize(text)
                # tokens looks like {'qwen3_4b': [[(0, 1.0), (1, 1.0), (2, 1.0)]]}
                for v in tokens.values():
                    L = len(v[0])
                    max_length = max(max_length, L)
                    token_lengths[i] = L

            # Pad to max length in the batch. We need to do this ourselves or the ComfyUI backend code will fail (it concats tensors assumed to be the same length).
            tokenizer.min_length = max_length
            tokens_dict = defaultdict(list)
            for text in captions:
                tokens = text_encoder.tokenize(text)
                for k, v in tokens.items():
                    token_list = v[0]
                    maybe_pad(token_list, max_length, tokenizer)  # some tokenizers don't listen to min_length
                    tokens_dict[k].append(token_list)

            o = text_encoder.encode_from_tokens_scheduled(tokens_dict)

            text_embeds = o[0][0].to(self.dtype)
            extra = o[0][1]
            if 'attention_mask' in extra:
                attention_mask = extra['attention_mask']
            else:
                # Krea2 (maybe others) removes attention_mask if it is all 1s (e.g. batch size 1)
                attention_mask = torch.ones(text_embeds.shape[:2], dtype=torch.int64, device=text_embeds.device)

            tokenizer.min_length = original_min_length
            return {
                f'text_embeds_{te_idx}': text_embeds,
                f'attention_mask_{te_idx}': attention_mask,
            }

        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        raise NotImplementedError()

    def to_layers(self):
        raise NotImplementedError()

    def model_specific_dataset_config_validation(self, dataset_config):
        pass

    # Get param groups that will be passed into the optimizer. Models can override this, e.g. SDXL
    # supports separate learning rates for unet and text encoders.
    def get_param_groups(self, parameters):
        return [{'params': parameters}]

    # Default loss_fn. MSE between output and target, with mask support.
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

    def enable_block_swap(self, blocks_to_swap):
        raise NotImplementedError('Block swapping is not implemented for this model')

    def prepare_block_swap_training(self):
        pass

    def prepare_block_swap_inference(self, disable_block_swap=False):
        pass