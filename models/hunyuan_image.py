import re
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/HunyuanImage-2.1'))

import torch
from torch import nn
import torch.nn.functional as F
import safetensors
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
import transformers

from models.base import BasePipeline, make_contiguous
from utils.common import AUTOCAST_DTYPE, get_lin_function, time_shift, iterate_safetensors, is_main_process
from utils.offloading import ModelOffloader
from models.hunyuan_image_modeling import MMDoubleStreamBlock, MMSingleStreamBlock
from hyimage.models.vae import HunyuanVAE2D
from hyimage.models.text_encoder import PROMPT_TEMPLATE
import hyimage.models.hunyuan.modules.hunyuanimage_dit
from hyimage.models.hunyuan.modules.hunyuanimage_dit import HYImageDiffusionTransformer
from hyimage.models.hunyuan.modules.flash_attn_no_pad import get_cu_seqlens


KEEP_IN_HIGH_PRECISION = ['byt5_in', 'txt_in', 'img_in', 'time_in', 'final_layer']

# Patch to use fused QKV
hyimage.models.hunyuan.modules.hunyuanimage_dit.MMDoubleStreamBlock = MMDoubleStreamBlock
hyimage.models.hunyuan.modules.hunyuanimage_dit.MMSingleStreamBlock = MMSingleStreamBlock


ORIGINAL_TO_COMFYUI_LORA_MAPPING = {
    '_attn_qkv': '_attn.qkv',
    '_attn_proj': '_attn.proj',
    '_mlp.fc1': '_mlp.0',
    '_mlp.fc2': '_mlp.2',
    '_mod.linear': '_mod.lin',
    'modulation.linear': 'modulation.lin',
}

COMFYUI_TO_ORIGINAL_LORA_MAPPING = {v: k for k, v in ORIGINAL_TO_COMFYUI_LORA_MAPPING.items()}


class HunyuanImagePipeline(BasePipeline):
    name = 'hunyuan_image'
    pixels_round_to_multiple = 32
    checkpointable_layers = ['DoubleBlock', 'SingleBlock']
    adapter_target_modules = ['MMDoubleStreamBlock', 'MMSingleStreamBlock']

    prompt_template = PROMPT_TEMPLATE['dit-llm-encode-v2']['template']
    crop_start = PROMPT_TEMPLATE['dit-llm-encode-v2']['crop_start']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.offloader_double = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        self.offloader_single = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        dtype = self.model_config['dtype']

        config = HunyuanVAE2D.load_config('configs/hunyuan_image/vae_config.json')
        with init_empty_weights():
            vae = HunyuanVAE2D.from_config(config)
        for key, tensor in iterate_safetensors(self.model_config['vae_path']):
            set_module_tensor_to_device(vae, key, device='cpu', dtype=dtype, value=tensor)
        self.vae = vae

        self.tokenizer = transformers.Qwen2Tokenizer.from_pretrained('configs/qwen_image/tokenizer', local_files_only=True)

        text_encoder_config = transformers.Qwen2_5_VLConfig.from_pretrained('configs/qwen_image/text_encoder', local_files_only=True)
        with init_empty_weights():
            text_encoder = transformers.Qwen2_5_VLForConditionalGeneration(text_encoder_config)
        for key, tensor in iterate_safetensors(self.model_config['text_encoder_path']):
            key = key.replace('model.', 'language_model.')
            key = 'model.' + key
            if 'lm_head' in key:
                key = 'lm_head.weight'
            set_module_tensor_to_device(text_encoder, key, device='cpu', dtype=dtype, value=tensor)
        self.text_encoder = text_encoder

        self.byt5_tokenizer = transformers.ByT5Tokenizer.from_pretrained('configs/hunyuan_image/byt5-small', local_files_only=True)

        byt5_config = transformers.T5Config.from_pretrained('configs/hunyuan_image/byt5-small', local_files_only=True)
        with init_empty_weights():
            byt5_full_model = transformers.T5ForConditionalGeneration(byt5_config)
        for key, tensor in iterate_safetensors(self.model_config['byt5_path']):
            set_module_tensor_to_device(byt5_full_model, key, device='cpu', dtype=dtype, value=tensor)
        self.byt5_text_encoder = byt5_full_model.get_encoder()

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        with init_empty_weights():
            transformer = HYImageDiffusionTransformer(
                in_channels=64,
                out_channels=64,
                mm_double_blocks_depth=20,
                mm_single_blocks_depth=40,
                rope_dim_list=[64, 64],
                hidden_size=3584,
                heads_num=28,
                mlp_width_ratio=4,
                patch_size=[1, 1],
                text_states_dim=3584,
                glyph_byT5_v2=True,
                guidance_embed=False,
            )

        for key, tensor in iterate_safetensors(self.model_config['transformer_path']):
            dtype_to_use = dtype if any(keyword in key for keyword in KEEP_IN_HIGH_PRECISION) or tensor.ndim == 1 else transformer_dtype
            set_module_tensor_to_device(transformer, key, device='cpu', dtype=dtype_to_use, value=tensor)

        transformer.train()
        for name, p in transformer.named_parameters():
            p.original_name = name
        self.transformer = transformer

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder, self.byt5_text_encoder]

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        sd = {}
        # ComfyUI format.
        for k, v in peft_state_dict.items():
            for target, replace in ORIGINAL_TO_COMFYUI_LORA_MAPPING.items():
                k = k.replace(target, replace)
            sd[k] = v
        sd = {'diffusion_model.'+k: v for k, v in sd.items()}
        safetensors.torch.save_file(sd, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def save_model(self, save_dir, state_dict):
        safetensors.torch.save_file(state_dict, save_dir / 'model.safetensors', metadata={'format': 'pt'})

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

            for target, replace in COMFYUI_TO_ORIGINAL_LORA_MAPPING.items():
                k = k.replace(target, replace)

            # Replace Diffusers or ComfyUI prefix
            k = re.sub(r'^(transformer|diffusion_model)\.', '', k)
            # Replace weight at end for LoRA format
            k = re.sub(r'\.weight$', '.default.weight', k)
            if k not in model_parameters:
                raise RuntimeError(f'modified_state_dict key {k} is not in the model parameters')
            modified_state_dict[k] = v
        self.transformer.load_state_dict(modified_state_dict, strict=False)

    def get_call_vae_fn(self, vae):
        def fn(image):
            latents = vae.encode(image.to(vae.device, vae.dtype)).latent_dist.mode()
            assert latents.ndim == 4
            latents = latents * vae.config.scaling_factor
            result = {'latents': latents}
            return result
        return fn

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def _qwenvl_embeddings(self, text, device):
        text = [self.prompt_template.format(c) for c in text]
        batch_encoding = self.tokenizer(
            text,
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=True,
            truncation=True,
            max_length=1000,
            padding=True,
            return_tensors='pt',
        )
        input_ids = batch_encoding.input_ids.to(device)
        attention_mask = batch_encoding.attention_mask.to(device)
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden_state = outputs.hidden_states[-3]
        last_hidden_state = last_hidden_state[:, self.crop_start:]
        attention_mask = attention_mask[:, self.crop_start:]
        return self._extract_masked_hidden(last_hidden_state, attention_mask)

    def _byt5_embeddings(self, text_list, device):
        formatted_text_list = []
        indices_to_zero = []
        for i, prompt in enumerate(text_list):
            text_prompt_texts = []
            pattern_quote_double = r'\"(.*?)\"'
            pattern_quote_chinese_single = r'‘(.*?)’'
            pattern_quote_chinese_double = r'“(.*?)”'

            matches_quote_double = re.findall(pattern_quote_double, prompt)
            matches_quote_chinese_single = re.findall(pattern_quote_chinese_single, prompt)
            matches_quote_chinese_double = re.findall(pattern_quote_chinese_double, prompt)

            text_prompt_texts.extend(matches_quote_double)
            text_prompt_texts.extend(matches_quote_chinese_single)
            text_prompt_texts.extend(matches_quote_chinese_double)

            if not text_prompt_texts:
                indices_to_zero.append(i)

            formatted_text_list.append(''.join(f'Text "{text}". ' for text in text_prompt_texts))

        batch_encoding = self.byt5_tokenizer(
            formatted_text_list,
            padding=True,
            max_length=128,
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt',
        )
        input_ids = batch_encoding.input_ids.to(device)
        attention_mask = batch_encoding.attention_mask.to(device)
        byt5_emb = self.byt5_text_encoder(
            input_ids,
            # TODO: official code casts this to float. Is that really correct?
            attention_mask=attention_mask.float(),
        )[0]
        byt5_emb[indices_to_zero] = 0.0
        return self._extract_masked_hidden(byt5_emb, attention_mask)

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(caption, is_video):
            # args are lists
            assert not any(is_video)
            device = text_encoder.device
            if text_encoder == self.text_encoder:
                return {'text_emb': self._qwenvl_embeddings(caption, device)}
            elif text_encoder == self.byt5_text_encoder:
                return {'byt5_emb': self._byt5_embeddings(caption, device)}
            else:
                raise RuntimeError('Unknown text encoder')
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        text_emb = inputs['text_emb']
        byt5_emb = inputs['byt5_emb']
        mask = inputs['mask']
        device = latents.device

        def get_emb_and_mask(prompt_embeds):
            # prompt embeds are variable length
            attn_mask_list = [torch.ones(e.size(0), dtype=torch.bool, device=device) for e in prompt_embeds]
            max_seq_len = max([e.size(0) for e in prompt_embeds])
            prompt_embeds = torch.stack(
                [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in prompt_embeds]
            )
            prompt_embeds_mask = torch.stack(
                [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
            )
            return prompt_embeds, prompt_embeds_mask

        text_emb, text_emb_mask = get_emb_and_mask(text_emb)
        byt5_emb, byt5_emb_mask = get_emb_and_mask(byt5_emb)

        bs, channels, h, w = latents.shape

        if mask is not None:
            mask = mask.unsqueeze(1).expand((-1, channels, -1, -1))  # make mask (bs, c, img_h, img_w)
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

        x_1 = latents
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1, 1)
        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
        target = x_0 - x_1

        t = t * 1000

        return (
            (x_t, t, text_emb, text_emb_mask, byt5_emb, byt5_emb_mask),
            (target, mask),
        )

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
        self.img_in = transformer.img_in
        self.time_in = transformer.time_in
        self.txt_in = transformer.txt_in
        self.byt5_in = transformer.byt5_in

    def __getattr__(self, name):
        return getattr(self.transformer[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, timestep, text_states, encoder_attention_mask, byt5_text_states, byt5_text_mask = inputs
        guidance = torch.tensor([6016.0], device=hidden_states.device, dtype=torch.bfloat16)
        img = x = hidden_states
        text_mask = encoder_attention_mask
        t = timestep
        txt = text_states
        input_shape = x.shape

        # Calculate spatial dimensions and get rotary embeddings
        assert len(input_shape) == 4
        _, _, oh, ow = x.shape
        th, tw = (
            oh // self.patch_size[0],
            ow // self.patch_size[1],
        )
        freqs_cos, freqs_sin = self.get_rotary_pos_embed((th, tw))
        freqs_cos = freqs_cos.to(img.device)
        freqs_sin = freqs_sin.to(img.device)

        img = self.img_in(img)

        # Prepare modulation vectors
        vec = self.time_in(t)

        # Embed image and text
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

        if self.glyph_byT5_v2:
            byt5_txt = self.byt5_in(byt5_text_states)
            txt, text_mask = self.reorder_txt_token(byt5_txt, txt, byt5_text_mask, text_mask)

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Calculate cu_seqlens and max_s for flash attention
        cu_seqlens, max_s = get_cu_seqlens(text_mask, img_seq_len)
        max_s = torch.tensor(max_s, device=img.device)
        txt_seq_len = torch.tensor(txt_seq_len, device=img.device)
        img_seq_len = torch.tensor(img_seq_len, device=img.device)
        th = torch.tensor(th, device=img.device)
        tw = torch.tensor(tw, device=img.device)

        outputs = make_contiguous(img, txt, vec, freqs_cos, freqs_sin, text_mask, cu_seqlens, max_s, img_seq_len, txt_seq_len, th, tw)

        for item in outputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)

        return outputs


class DoubleBlock(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        img, txt, vec, freqs_cos, freqs_sin, text_mask, cu_seqlens, max_s, img_seq_len, txt_seq_len, th, tw = inputs

        self.offloader.wait_for_block(self.block_idx)
        img, txt = self.block(img, txt, vec, (freqs_cos, freqs_sin), text_mask, cu_seqlens, max_s.item())
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(img, txt, vec, freqs_cos, freqs_sin, text_mask, cu_seqlens, max_s, img_seq_len, txt_seq_len, th, tw)


def concatenate_hidden_states(inputs):
    img, txt, vec, freqs_cos, freqs_sin, text_mask, cu_seqlens, max_s, img_seq_len, txt_seq_len, th, tw = inputs
    x = torch.cat((img, txt), 1)
    return x, vec, freqs_cos, freqs_sin, text_mask, cu_seqlens, max_s, img_seq_len, txt_seq_len, th, tw


class SingleBlock(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, vec, freqs_cos, freqs_sin, text_mask, cu_seqlens, max_s, img_seq_len, txt_seq_len, th, tw = inputs

        self.offloader.wait_for_block(self.block_idx)
        x = self.block(x, vec, txt_seq_len, (freqs_cos, freqs_sin), text_mask, cu_seqlens, max_s.item())
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(x, vec, freqs_cos, freqs_sin, text_mask, cu_seqlens, max_s, img_seq_len, txt_seq_len, th, tw)

class OutputLayer(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.final_layer = transformer.final_layer
        self.transformer = [transformer]

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, vec, freqs_cos, freqs_sin, text_mask, cu_seqlens, max_s, img_seq_len, txt_seq_len, th, tw = inputs
        img = x[:, :img_seq_len.item(), ...]
        img = self.final_layer(img, vec)
        img = self.transformer[0].unpatchify_2d(img, th.item(), tw.item())
        return img
