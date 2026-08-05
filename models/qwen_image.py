import json
from pathlib import Path
from typing import Union, Tuple, Optional
import math

import torch
from torch import nn
import torch.nn.functional as F
import diffusers
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
import safetensors
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
import transformers
from PIL import Image, ImageOps

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift, iterate_safetensors
from utils.offloading import ModelOffloader


KEEP_IN_HIGH_PRECISION = ['time_text_embed', 'img_in', 'txt_in', 'norm_out', 'proj_out']


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


# I copied this because it doesn't handle encoder_hidden_states_mask, which causes high loss values when there is a lot
# of padding. When (or if) they fix it upstream, I don't want the changes to break my workaround, which is to just set
# attention_mask.
class QwenDoubleStreamAttnProcessor2_0:
    """
    Attention processor for Qwen double-stream architecture, matching DoubleStreamLayerMegatron logic. This processor
    implements joint attention computation where text and image streams are processed together.
    """

    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Compute joint attention
        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class QwenImagePipeline(BasePipeline):
    name = 'qwen_image'
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['QwenImageTransformerBlock']

    prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    prompt_template_encode_start_idx = 34
    prompt_template_encode_edit = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
    prompt_template_encode_start_idx_edit = 64

    # Size of image fed to the VLM for Qwen-Image-Edit
    vlm_image_size = 1024

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        dtype = self.model_config['dtype']

        self.preprocess_media_file_fn = PreprocessMediaFile(self.config, support_video=True, framerate=1)

        tokenizer = transformers.Qwen2Tokenizer.from_pretrained('configs/qwen_image/tokenizer', local_files_only=True)
        processor = transformers.Qwen2VLProcessor.from_pretrained('configs/qwen_image/processor', local_files_only=True)

        if 'text_encoder_path' in self.model_config:
            text_encoder_path = self.model_config['text_encoder_path']
        else:
            text_encoder_path = Path(self.model_config['diffusers_path']) / 'text_encoder'
        text_encoder_config = transformers.Qwen2_5_VLConfig.from_pretrained('configs/qwen_image/text_encoder', local_files_only=True)
        with init_empty_weights():
            text_encoder = transformers.Qwen2_5_VLForConditionalGeneration(text_encoder_config)
        for key, tensor in iterate_safetensors(text_encoder_path):
            # The keys in the state_dict don't match the structure in the model. Annoying. Need to convert.
            key = key.replace('model.', 'language_model.')
            key = 'model.' + key
            if 'lm_head' in key:
                key = 'lm_head.weight'
            set_module_tensor_to_device(text_encoder, key, device='cpu', dtype=dtype, value=tensor)

        # TODO: make this work with ComfyUI VAE weights, which have completely different key names.
        if 'vae_path' in self.model_config:
            vae_path = self.model_config['vae_path']
        else:
            vae_path = Path(self.model_config['diffusers_path']) / 'vae'
        with open('configs/qwen_image/vae/config.json') as f:
            vae_config = json.load(f)
        with init_empty_weights():
            vae = diffusers.AutoencoderKLQwenImage.from_config(vae_config)
        for key, tensor in iterate_safetensors(vae_path):
            set_module_tensor_to_device(vae, key, device='cpu', dtype=dtype, value=tensor)

        self.diffusers_pipeline = diffusers.QwenImagePipeline(
            scheduler=None,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=None,
        )

        self.diffusers_pipeline.processor = processor

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(dtype)
        )
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(dtype)
        self.vae.register_buffer('latents_mean_tensor', latents_mean)
        self.vae.register_buffer('latents_std_tensor', latents_std)

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        if 'transformer_path' in self.model_config:
            transformer_path = self.model_config['transformer_path']
        else:
            transformer_path = Path(self.model_config['diffusers_path']) / 'transformer'
        with open('configs/qwen_image/transformer/config.json') as f:
            json_config = json.load(f)

        with init_empty_weights():
            transformer = diffusers.QwenImageTransformer2DModel.from_config(json_config)

        for key, tensor in iterate_safetensors(transformer_path):
            dtype_to_use = dtype if any(keyword in key for keyword in KEEP_IN_HIGH_PRECISION) or tensor.ndim == 1 else transformer_dtype
            set_module_tensor_to_device(transformer, key, device='cpu', dtype=dtype_to_use, value=tensor)

        attn_processor = QwenDoubleStreamAttnProcessor2_0()
        for block in transformer.transformer_blocks:
            block.attn.set_processor(attn_processor)

        self.diffusers_pipeline.transformer = transformer

        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder]

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI format.
        peft_state_dict = {'diffusion_model.'+k: v for k, v in peft_state_dict.items()}
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def save_model(self, save_dir, state_dict):
        safetensors.torch.save_file(state_dict, save_dir / 'model.safetensors', metadata={'format': 'pt'})

    def get_preprocess_media_file_fn(self):
        return self.preprocess_media_file_fn

    def get_call_vae_fn(self, vae):
        def fn(*args):
            image = args[0]
            image = image.to(vae.device, vae.dtype)
            image = image*2 - 1
            latents = vae.encode(image).latent_dist.mode()
            latents = (latents - vae.latents_mean_tensor) / vae.latents_std_tensor
            result = {'latents': latents}
            if len(args) == 2:
                control_image = args[1]
                control_image = control_image.to(vae.device, vae.dtype)
                control_image = control_image*2 - 1
                control_latents = vae.encode(control_image).latent_dist.mode()
                control_latents = (control_latents - vae.latents_mean_tensor) / vae.latents_std_tensor
                result['control_latents'] = control_latents
            return result
        return fn

    def load_image_for_vlm(self, path):
        pil_img = Image.open(path)
        height, width = pil_img.height, pil_img.width

        if pil_img.mode not in ['RGB', 'RGBA'] and 'transparency' in pil_img.info:
            pil_img = pil_img.convert('RGBA')

        # add white background for transparent images
        if pil_img.mode == 'RGBA':
            canvas = Image.new('RGBA', pil_img.size, (255, 255, 255))
            canvas.alpha_composite(pil_img)
            pil_img = canvas.convert('RGB')
        else:
            pil_img = pil_img.convert('RGB')

        scale_factor = self.vlm_image_size / math.sqrt(height*width)
        return ImageOps.scale(pil_img, scale_factor)

    def _get_qwen_prompt_embeds(
        self,
        prompt,
        control_files,
        device=None,
        dtype=None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if control_files is None:
            template = self.prompt_template_encode
            drop_idx = self.prompt_template_encode_start_idx
            txt = [template.format(e) for e in prompt]
            txt_tokens = self.tokenizer(
                txt, max_length=self.tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            attention_mask = txt_tokens.attention_mask
            outputs = self.text_encoder(
                input_ids=txt_tokens.input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        else:
            template = self.prompt_template_encode_edit
            drop_idx = self.prompt_template_encode_start_idx_edit
            txt = [template.format(e) for e in prompt]
            images = [
                self.load_image_for_vlm(file)
                for file in control_files
            ]
            model_inputs = self.processor(
                text=txt,
                images=images,
                padding=True,
                return_tensors="pt",
            ).to(device)
            attention_mask = model_inputs.attention_mask
            outputs = self.text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]

        return split_hidden_states

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(caption, is_video, control_file: list[str] | None):
            # args are lists
            assert not any(is_video)
            prompt_embeds = self._get_qwen_prompt_embeds(caption, control_file, device=text_encoder.device)
            return {'prompt_embeds': prompt_embeds}
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        prompt_embeds = inputs['prompt_embeds']
        mask = inputs['mask']
        device = latents.device

        # prompt embeds are variable length
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.bool, device=device) for e in prompt_embeds]
        max_seq_len = max([e.size(0) for e in prompt_embeds])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in prompt_embeds]
        )
        prompt_embeds_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        max_text_len = prompt_embeds_mask.sum(dim=1).max().item()
        prompt_embeds = prompt_embeds[:, :max_text_len, :]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_text_len]

        bs, channels, num_frames, h, w = latents.shape

        num_channels_latents = self.transformer.config.in_channels // 4
        assert num_channels_latents == channels
        latents = self._pack_latents(latents, bs, num_channels_latents, h, w)

        if mask is not None:
            mask = mask.unsqueeze(1).expand((-1, num_channels_latents, -1, -1))  # make mask (bs, c, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)  # add frame dimension
            mask = self._pack_latents(mask, bs, num_channels_latents, h, w)

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

        x_1 = latents
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1)
        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
        target = x_0 - x_1

        img_shapes = [(1, h // 2, w // 2)]

        if 'control_latents' in inputs:
            control_latents = inputs['control_latents'].float()
            control_latents = self._pack_latents(control_latents, bs, num_channels_latents, h, w)
            assert control_latents.shape == latents.shape, (control_latents.shape, latents.shape)
            img_seq_len = torch.tensor(x_t.shape[1], device=x_t.device).repeat((bs,))
            extra = (img_seq_len,)
            x_t = torch.cat([x_t, control_latents], dim=1)
            img_shapes.append((1, h // 2, w // 2))
        else:
            extra = tuple()

        img_shapes = torch.tensor([img_shapes], dtype=torch.int32, device=device).repeat((bs, 1, 1))
        txt_seq_lens = torch.tensor([max_text_len], dtype=torch.int32, device=device).repeat((bs,))
        img_attention_mask = torch.ones((bs, x_t.shape[1]), dtype=torch.bool, device=device)
        attention_mask = torch.cat([prompt_embeds_mask, img_attention_mask], dim=1)
        # Make broadcastable with attention weights, which are [bs, num_heads, query_len, key_value_len]
        attention_mask = attention_mask.view(bs, 1, 1, -1)
        assert attention_mask.dtype == torch.bool

        return (
            (x_t, prompt_embeds, attention_mask, t, img_shapes, txt_seq_lens) + extra,
            (target, mask),
        )

    def to_layers(self):
        transformer = self.transformer
        layers = [InitialLayer(transformer)]
        for i, block in enumerate(transformer.transformer_blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(transformer))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        transformer = self.transformer
        blocks = transformer.transformer_blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        transformer.transformer_blocks = None
        transformer.to('cuda')
        transformer.transformer_blocks = blocks
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
        self.img_in = model.img_in
        self.txt_norm = model.txt_norm
        self.txt_in = model.txt_in
        self.time_text_embed = model.time_text_embed
        self.pos_embed = model.pos_embed

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)

        hidden_states, encoder_hidden_states, attention_mask, timestep, img_shapes, txt_seq_lens, *extra = inputs

        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        temb = self.time_text_embed(timestep, hidden_states)

        img_shapes = img_shapes.tolist()
        txt_seq_lens = txt_seq_lens.tolist()
        vid_freqs, txt_freqs = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        return make_contiguous(hidden_states, encoder_hidden_states, attention_mask, temb, vid_freqs, txt_freqs) + tuple(extra)

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2, device=index.device).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs


class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, encoder_hidden_states, attention_mask, temb, vid_freqs, txt_freqs, *extra = inputs

        self.offloader.wait_for_block(self.block_idx)
        encoder_hidden_states, hidden_states = self.block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=None,
            temb=temb,
            image_rotary_emb=(vid_freqs, txt_freqs),
            joint_attention_kwargs={'attention_mask': attention_mask},
        )
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(hidden_states, encoder_hidden_states, attention_mask, temb, vid_freqs, txt_freqs) + tuple(extra)


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
        hidden_states, encoder_hidden_states, attention_mask, temb, vid_freqs, txt_freqs, *extra = inputs
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        if len(extra) > 0:
            assert len(extra) == 1
            img_seq_len = extra[0][0].item()
            output = output[:, :img_seq_len, ...]
        return output
