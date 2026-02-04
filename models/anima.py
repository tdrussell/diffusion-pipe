# Anima Pipeline for diffusion-pipe
# Based on Cosmos-Predict2 but with dual text encoders (Qwen3-0.6B + T5)
# Uses Qwen Image VAE (same architecture/normalization as Wan VAE)

import math

import torch
from torch import nn
import torch.nn.functional as F
import safetensors
import transformers
from transformers import T5TokenizerFast, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from models.anima_modeling import Anima
from models.cosmos_predict2 import get_dit_config, time_shift, get_lin_function, WanVAE, vae_encode
from utils.common import load_state_dict, AUTOCAST_DTYPE
from utils.offloading import ModelOffloader


KEEP_IN_HIGH_PRECISION = ['x_embedder', 't_embedder', 't_embedding_norm', 'final_layer', 'llm_adapter']


def _tokenize_t5(tokenizer, prompts):
    """Tokenize prompts using T5 tokenizer."""
    return tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )


def _tokenize_qwen(tokenizer, prompts):
    """Tokenize prompts using Qwen tokenizer."""
    return tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )


def _compute_qwen_embeddings(qwen_model, input_ids, attention_mask):
    """Compute Qwen3 hidden states for use as cross-attention context."""
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    input_ids = input_ids.to(qwen_model.device, dtype=torch.long)
    attention_mask = attention_mask.to(qwen_model.device, dtype=torch.long)

    with torch.no_grad():
        outputs = qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    # Use the last hidden state
    hidden_states = outputs.hidden_states[-1]

    # Zero out padding positions
    lengths = attention_mask.sum(dim=1).cpu()
    for batch_id in range(hidden_states.shape[0]):
        length = lengths[batch_id]
        if length == 1:  # Empty prompt case
            length = 0
        hidden_states[batch_id][length:] = 0

    return hidden_states


class AnimaPipeline(BasePipeline):
    name = 'anima'
    framerate = 16
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['Block', 'LLMAdapterTransformerBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        # Determine if we're doing LoRA or full-finetune based on presence of adapter config
        self.is_adapter = 'adapter' in config
        # For full-finetune, skip_lora should be False (swap all weights including what would be LoRA)
        self.skip_lora = self.is_adapter
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False, skip_lora=self.skip_lora)
        dtype = self.model_config['dtype']
        self.cache_text_embeddings = self.model_config.get('cache_text_embeddings', True)

        # Full-finetune specific options
        self.freeze_llm_adapter = self.model_config.get('freeze_llm_adapter', False)
        self.llm_adapter_lr = self.model_config.get('llm_adapter_lr', None)

        # Load transformer directly to CUDA (faster init, single GPU only)
        self.load_to_cuda = self.model_config.get('load_to_cuda', False)

        # VAE - Qwen Image VAE (16 channel, same architecture/normalization as Wan VAE)
        self.vae = WanVAE(
            vae_pth=self.model_config['vae_path'],
            dtype=dtype,
        )
        self.vae.mean = self.vae.mean.to('cuda')
        self.vae.std = self.vae.std.to('cuda')
        self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]

        # T5 Tokenizer - for getting token IDs (used by LLMAdapter)
        self.t5_tokenizer = T5TokenizerFast(
            vocab_file='configs/t5_old/spiece.model',
            tokenizer_file='configs/t5_old/tokenizer.json',
        )

        # Qwen3 Tokenizer and Model - for getting embeddings
        qwen_path = self.model_config['qwen_path']
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            qwen_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token

        # Load Qwen3-0.6B model for text encoding
        qwen_config = AutoConfig.from_pretrained(qwen_path, trust_remote_code=True, local_files_only=True)

        if self.model_config.get('qwen_nf4', False):
            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=dtype,
            )
        else:
            quantization_config = None

        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_path,
            config=qwen_config,
            torch_dtype=dtype,
            local_files_only=True,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

        if quantization_config is None and self.model_config.get('qwen_fp8', False):
            for name, p in self.qwen_model.named_parameters():
                if p.ndim == 2:
                    p.data = p.data.to(torch.float8_e4m3fn)

        self.qwen_model.requires_grad_(False)

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        # Determine target device for loading
        target_device = 'cuda' if self.load_to_cuda else 'cpu'
        if self.load_to_cuda:
            print(f"Loading transformer directly to CUDA (faster initialization)")

        state_dict = load_state_dict(self.model_config['transformer_path'])

        # Remove 'net.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('net.'):
                k = k[len('net.'):]
            # Handle ComfyUI format with 'diffusion_model.' prefix
            if k.startswith('diffusion_model.'):
                k = k[len('diffusion_model.'):]
            new_state_dict[k] = v
        state_dict = new_state_dict

        # Get config for base model (without llm_adapter weights)
        base_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('llm_adapter.')}
        dit_config = get_dit_config(base_state_dict)

        with init_empty_weights():
            transformer = Anima(**dit_config)

        for name, p in transformer.named_parameters():
            # Keep LLMAdapter and certain layers in higher precision
            dtype_to_use = dtype if (any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) or p.ndim == 1) else transformer_dtype
            if name in state_dict:
                set_module_tensor_to_device(transformer, name, device=target_device, dtype=dtype_to_use, value=state_dict[name])
            else:
                # Initialize missing weights (shouldn't happen with proper checkpoint)
                print(f"Warning: Missing weight {name}, initializing randomly")
                set_module_tensor_to_device(transformer, name, device=target_device, dtype=dtype_to_use, value=torch.randn_like(p))

        self.transformer = transformer
        self.transformer.train()

        # Handle LLMAdapter freezing for full-finetune
        if self.freeze_llm_adapter:
            for name, p in self.transformer.llm_adapter.named_parameters():
                p.requires_grad_(False)
            print("LLMAdapter is frozen for training")

        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def get_param_groups(self, parameters):
        """
        Support separate learning rate for LLMAdapter vs DiT.
        Override from BasePipeline.
        """
        if self.llm_adapter_lr is None:
            # Default behavior: single param group (same LR for all)
            return [{'params': parameters}]

        # Separate param groups for different LRs
        llm_adapter_params = []
        dit_params = []

        for p in parameters:
            if hasattr(p, 'original_name') and 'llm_adapter' in p.original_name:
                llm_adapter_params.append(p)
            else:
                dit_params.append(p)

        param_groups = []
        if dit_params:
            param_groups.append({'params': dit_params})  # Uses default LR from config
        if llm_adapter_params:
            param_groups.append({'params': llm_adapter_params, 'lr': self.llm_adapter_lr})
            print(f"Using separate LR for LLMAdapter: {self.llm_adapter_lr}")

        return param_groups

    def get_vae(self):
        return self.vae.model

    def get_text_encoders(self):
        if self.cache_text_embeddings:
            return [self.qwen_model]
        else:
            return []

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI format
        peft_state_dict = {'diffusion_model.'+k: v for k, v in peft_state_dict.items()}
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def save_model(self, save_dir, state_dict):
        state_dict = {'net.'+k: v for k, v in state_dict.items()}
        safetensors.torch.save_file(state_dict, save_dir / 'model.safetensors', metadata={'format': 'pt'})

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(
            self.config,
            support_video=True,
            framerate=self.framerate,
        )

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            p = next(vae.parameters())
            tensor = tensor.to(p.device, p.dtype)
            latents = vae_encode(tensor, self.vae)
            return {'latents': latents}
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        """
        Returns a function that computes:
        - Qwen3 embeddings (for LLMAdapter cross-attention context)
        - T5 token IDs (for LLMAdapter embedding input)
        - T5 attention mask 
        """
        def fn(captions, is_video):
            # Get Qwen3 embeddings
            qwen_encoding = _tokenize_qwen(self.qwen_tokenizer, captions)
            qwen_embeds = _compute_qwen_embeddings(
                self.qwen_model,
                qwen_encoding.input_ids,
                qwen_encoding.attention_mask
            )

            # Get T5 token IDs and attention mask
            t5_encoding = _tokenize_t5(self.t5_tokenizer, captions)

            return {
                'qwen_embeds': qwen_embeds,
                't5_input_ids': t5_encoding.input_ids,
                't5_attention_mask': t5_encoding.attention_mask,
            }
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']

        if self.cache_text_embeddings:
            qwen_inputs = (
                inputs['qwen_embeds'],
                inputs['t5_input_ids'],
                inputs['t5_attention_mask'], 
            )
        else:
            # Compute on-the-fly
            captions = inputs['caption']
            qwen_encoding = _tokenize_qwen(self.qwen_tokenizer, captions)
            qwen_inputs = (qwen_encoding.input_ids, qwen_encoding.attention_mask)
            
            t5_encoding = _tokenize_t5(self.t5_tokenizer, captions)
            t5_input_ids = t5_encoding.input_ids
            t5_attention_mask = t5_encoding.attention_mask 
            
            qwen_inputs = (*qwen_inputs, t5_input_ids, t5_attention_mask)

        bs, channels, num_frames, h, w = latents.shape

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)  # make mask same number of dims as target

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
        t_expanded = t.view(-1, 1, 1, 1, 1)
        noisy_latents = (1 - t_expanded)*latents + t_expanded*noise
        target = noise - latents

        return (noisy_latents, t.view(-1, 1), *qwen_inputs), (target, mask)

    def to_layers(self):
        transformer = self.transformer
        qwen_model = None if self.cache_text_embeddings else self.qwen_model
        layers = [InitialLayer(transformer, qwen_model, self.qwen_tokenizer, self.t5_tokenizer)]
        for i, block in enumerate(transformer.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(transformer))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        transformer = self.transformer
        blocks = transformer.blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'),
            self.config['reentrant_activation_checkpointing'], skip_lora=self.skip_lora
        )
        transformer.blocks = None
        transformer.to('cuda')
        transformer.blocks = blocks
        self.prepare_block_swap_training()
        print(f'Block swap enabled. Swapping {blocks_to_swap} blocks out of {num_blocks} blocks.')
        if not self.skip_lora:
            print('Full-finetune mode: all weights will be swapped (skip_lora=False)')

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
    def __init__(self, model, qwen_model, qwen_tokenizer, t5_tokenizer):
        super().__init__()
        self.x_embedder = model.x_embedder
        self.pos_embedder = model.pos_embedder
        if model.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder = model.extra_pos_embedder
        self.t_embedder = model.t_embedder
        self.t_embedding_norm = model.t_embedding_norm
        self.llm_adapter = model.llm_adapter
        self.qwen_model = qwen_model
        self.qwen_tokenizer = qwen_tokenizer
        self.t5_tokenizer = t5_tokenizer
        self.model = [model]

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_C_T_H_W, timesteps_B_T, *text_inputs = inputs
        batch_size = x_B_C_T_H_W.shape[0]
        target_device = x_B_C_T_H_W.device
        target_dtype = x_B_C_T_H_W.dtype

        if self.qwen_model is None:
            # Cached mode: (qwen_embeds, t5_input_ids, t5_attention_mask)
            assert len(text_inputs) == 3, f"Expected cached inputs (qwen_embeds, t5_input_ids, t5_attention_mask), got {len(text_inputs)} items."
            qwen_embeds, t5_input_ids, t5_attention_mask = text_inputs

            # Move to target device
            if qwen_embeds.device != target_device:
                qwen_embeds = qwen_embeds.to(target_device)
            if t5_input_ids.device != target_device:
                t5_input_ids = t5_input_ids.to(target_device)
            if t5_attention_mask.device != target_device:
                t5_attention_mask = t5_attention_mask.to(target_device)
                
            if t5_input_ids.dtype != torch.long:
                t5_input_ids = t5_input_ids.long()

            # Process through LLM adapter
            crossattn_emb = self.llm_adapter(qwen_embeds, t5_input_ids)
        else:
            # Non-cached mode: (qwen_input_ids, qwen_attention_mask, t5_input_ids, t5_attention_mask)
            assert len(text_inputs) == 4, f"Expected non-cached inputs (qwen_input_ids, qwen_attention_mask, t5_input_ids, t5_attention_mask), got {len(text_inputs)} items."
            qwen_input_ids, qwen_attention_mask, t5_input_ids, t5_attention_mask = text_inputs
            
            with torch.no_grad():
                qwen_embeds = _compute_qwen_embeddings(
                    self.qwen_model,
                    qwen_input_ids,
                    qwen_attention_mask,
                )

            # Move to target device
            if qwen_embeds.device != target_device:
                qwen_embeds = qwen_embeds.to(target_device)
            if t5_input_ids.device != target_device:
                t5_input_ids = t5_input_ids.to(target_device)
            if t5_attention_mask.device != target_device:
                t5_attention_mask = t5_attention_mask.to(target_device)
                
            if t5_input_ids.dtype != torch.long:
                t5_input_ids = t5_input_ids.long()

            # Process through LLM adapter to get final cross-attention embeddings
            crossattn_emb = self.llm_adapter(qwen_embeds, t5_input_ids)


        # This prevents attention leakage to garbage padding tokens
        # and prevents model collapse with empty/short captions
        if t5_attention_mask is not None:
            # Create expanded mask: (B, seq_len, 1)
            mask_expanded = t5_attention_mask.unsqueeze(-1).to(
                device=crossattn_emb.device,
                dtype=crossattn_emb.dtype
            )
            # Zero out padding positions: keep only valid tokens
            crossattn_emb = crossattn_emb * mask_expanded

        # Pad to 512 tokens if needed (padding is already zeros after masking)
        if crossattn_emb.shape[1] < 512:
            crossattn_emb = F.pad(crossattn_emb, (0, 0, 0, 512 - crossattn_emb.shape[1]))

        padding_mask = torch.zeros(x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[3], x_B_C_T_H_W.shape[4], dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.model[0].prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=None,
            padding_mask=padding_mask,
        )
        assert extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is None
        assert rope_emb_L_1_1_D is not None

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # Note: timesteps_B_T is NOT included - it's only used here in InitialLayer
        # Including it breaks pipeline parallelism (no gradient flows through unused tensors)
        outputs = make_contiguous(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D)
        for item in outputs:  
            item.requires_grad_(True)
        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D = inputs

        self.offloader.wait_for_block(self.block_idx)
        x_B_T_H_W_D = self.block(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D=rope_emb_L_1_1_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    def get_per_sigma_loss_weights(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + self.pipe.sigma_data**2) / (sigma * self.pipe.sigma_data) ** 2

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D = inputs
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        net_output_B_C_T_H_W = self.unpatchify(x_B_T_H_W_O)
        return net_output_B_C_T_H_W