import re
from pathlib import Path

import diffusers
import torch
from torch import nn
import torch.nn.functional as F
from deepspeed.utils.logging import logger
import peft
import safetensors
from safetensors.torch import save_file

from models.base import BasePipeline, make_contiguous
from utils.common import AUTOCAST_DTYPE, is_main_process



# Copied from https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_sdxl.py

# =================#
# UNet Conversion #
# =================#

unet_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
    # the following are for sdxl
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
]

unet_conversion_map_resnet = [
    # (stable-diffusion, HF Diffusers)
    ("in_layers.0", "norm1"),
    ("in_layers.2", "conv1"),
    ("out_layers.0", "norm2"),
    ("out_layers.3", "conv2"),
    ("emb_layers.1", "time_emb_proj"),
    ("skip_connection", "conv_shortcut"),
]

unet_conversion_map_layer = []
# hardcoded number of downblocks and resnets/attentions...
# would need smarter logic for other networks.
for i in range(3):
    # loop over downblocks/upblocks

    for j in range(2):
        # loop over resnets/attentions for downblocks
        hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
        sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
        unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

        if i > 0:
            hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
            sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
            unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

    for j in range(4):
        # loop over resnets/attentions for upblocks
        hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
        sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
        unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

        if i < 2:
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3 * i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

    if i < 3:
        # no downsample in down_blocks.3
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
        sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
        unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

        # no upsample in up_blocks.3
        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"output_blocks.{3*i + 2}.{1 if i == 0 else 2}."
        unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))
unet_conversion_map_layer.append(("output_blocks.2.2.conv.", "output_blocks.2.1.conv."))

hf_mid_atn_prefix = "mid_block.attentions.0."
sd_mid_atn_prefix = "middle_block.1."
unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))
for j in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{j}."
    sd_mid_res_prefix = f"middle_block.{2*j}."
    unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))


def convert_unet_state_dict(unet_state_dict):
    # buyer beware: this is a *brittle* function,
    # and correct output requires that all of these pieces interact in
    # the exact order in which I have arranged them.
    mapping = {k: k for k in unet_state_dict.keys()}
    for sd_name, hf_name in unet_conversion_map:
        mapping[hf_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in unet_conversion_map_resnet:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for sd_part, hf_part in unet_conversion_map_layer:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    new_state_dict = {sd_name: unet_state_dict[hf_name] for hf_name, sd_name in mapping.items()}
    return new_state_dict


# ================#
# VAE Conversion #
# ================#

vae_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("nin_shortcut", "conv_shortcut"),
    ("norm_out", "conv_norm_out"),
    ("mid.attn_1.", "mid_block.attentions.0."),
]

for i in range(4):
    # down_blocks have two resnets
    for j in range(2):
        hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
        sd_down_prefix = f"encoder.down.{i}.block.{j}."
        vae_conversion_map.append((sd_down_prefix, hf_down_prefix))

    if i < 3:
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
        sd_downsample_prefix = f"down.{i}.downsample."
        vae_conversion_map.append((sd_downsample_prefix, hf_downsample_prefix))

        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"up.{3-i}.upsample."
        vae_conversion_map.append((sd_upsample_prefix, hf_upsample_prefix))

    # up_blocks have three resnets
    # also, up blocks in hf are numbered in reverse from sd
    for j in range(3):
        hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
        sd_up_prefix = f"decoder.up.{3-i}.block.{j}."
        vae_conversion_map.append((sd_up_prefix, hf_up_prefix))

# this part accounts for mid blocks in both the encoder and the decoder
for i in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{i}."
    sd_mid_res_prefix = f"mid.block_{i+1}."
    vae_conversion_map.append((sd_mid_res_prefix, hf_mid_res_prefix))


vae_conversion_map_attn = [
    # (stable-diffusion, HF Diffusers)
    ("norm.", "group_norm."),
    # the following are for SDXL
    ("q.", "to_q."),
    ("k.", "to_k."),
    ("v.", "to_v."),
    ("proj_out.", "to_out.0."),
]


def reshape_weight_for_sd(w):
    # convert HF linear weights to SD conv2d weights
    if not w.ndim == 1:
        return w.reshape(*w.shape, 1, 1)
    else:
        return w


def convert_vae_state_dict(vae_state_dict):
    mapping = {k: k for k in vae_state_dict.keys()}
    for k, v in mapping.items():
        for sd_part, hf_part in vae_conversion_map:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    for k, v in mapping.items():
        if "attentions" in k:
            for sd_part, hf_part in vae_conversion_map_attn:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
    weights_to_convert = ["q", "k", "v", "proj_out"]
    for k, v in new_state_dict.items():
        for weight_name in weights_to_convert:
            if f"mid.attn_1.{weight_name}.weight" in k:
                print(f"Reshaping {k} for SD format")
                new_state_dict[k] = reshape_weight_for_sd(v)
    return new_state_dict


# =========================#
# Text Encoder Conversion #
# =========================#


textenc_conversion_lst = [
    # (stable-diffusion, HF Diffusers)
    ("transformer.resblocks.", "text_model.encoder.layers."),
    ("ln_1", "layer_norm1"),
    ("ln_2", "layer_norm2"),
    (".c_fc.", ".fc1."),
    (".c_proj.", ".fc2."),
    (".attn", ".self_attn"),
    ("ln_final.", "text_model.final_layer_norm."),
    ("token_embedding.weight", "text_model.embeddings.token_embedding.weight"),
    ("positional_embedding", "text_model.embeddings.position_embedding.weight"),
]
protected = {re.escape(x[1]): x[0] for x in textenc_conversion_lst}
textenc_pattern = re.compile("|".join(protected.keys()))

# Ordering is from https://github.com/pytorch/pytorch/blob/master/test/cpp/api/modules.cpp
code2idx = {"q": 0, "k": 1, "v": 2}


def convert_openclip_text_enc_state_dict(text_enc_dict):
    new_state_dict = {}
    capture_qkv_weight = {}
    capture_qkv_bias = {}
    for k, v in text_enc_dict.items():
        if (
            k.endswith(".self_attn.q_proj.weight")
            or k.endswith(".self_attn.k_proj.weight")
            or k.endswith(".self_attn.v_proj.weight")
        ):
            k_pre = k[: -len(".q_proj.weight")]
            k_code = k[-len("q_proj.weight")]
            if k_pre not in capture_qkv_weight:
                capture_qkv_weight[k_pre] = [None, None, None]
            capture_qkv_weight[k_pre][code2idx[k_code]] = v
            continue

        if (
            k.endswith(".self_attn.q_proj.bias")
            or k.endswith(".self_attn.k_proj.bias")
            or k.endswith(".self_attn.v_proj.bias")
        ):
            k_pre = k[: -len(".q_proj.bias")]
            k_code = k[-len("q_proj.bias")]
            if k_pre not in capture_qkv_bias:
                capture_qkv_bias[k_pre] = [None, None, None]
            capture_qkv_bias[k_pre][code2idx[k_code]] = v
            continue

        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k)
        new_state_dict[relabelled_key] = v

    for k_pre, tensors in capture_qkv_weight.items():
        if None in tensors:
            raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
        new_state_dict[relabelled_key + ".in_proj_weight"] = torch.cat(tensors)

    for k_pre, tensors in capture_qkv_bias.items():
        if None in tensors:
            raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
        relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
        new_state_dict[relabelled_key + ".in_proj_bias"] = torch.cat(tensors)

    return new_state_dict


def convert_openai_text_enc_state_dict(text_enc_dict):
    return text_enc_dict


# Copied from https://github.com/kohya-ss/sd-scripts/blob/main/library/custom_train_functions.py

def prepare_scheduler_for_custom_training(noise_scheduler):
    if hasattr(noise_scheduler, "all_snr"):
        return

    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2

    noise_scheduler.all_snr = all_snr


def fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler):
    # fix beta: zero terminal SNR
    logger.info(f"fix noise scheduler betas: https://arxiv.org/abs/2305.08891")

    def enforce_zero_terminal_snr(betas):
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1 - alphas
        return betas

    betas = noise_scheduler.betas
    betas = enforce_zero_terminal_snr(betas)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # logger.info(f"original: {noise_scheduler.betas}")
    # logger.info(f"fixed: {betas}")

    noise_scheduler.betas = betas
    noise_scheduler.alphas = alphas
    noise_scheduler.alphas_cumprod = alphas_cumprod


def apply_snr_weight(loss, timesteps, noise_scheduler, gamma, v_prediction=False):
    snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
    if v_prediction:
        # TODO: in sd-scripts, this commit: https://github.com/kohya-ss/sd-scripts/commit/6b3148fd3fb64e41aa29fc1759ebfab3a4504d45
        # made it so with v-pred, scale_v_prediction_loss_like_noise_prediction is built-in here. Is this the right thing to do?
        # I.e., does min_snr_gamma only make sense in the context of scaling v-pred loss to be like noise prediction?
        snr_weight = torch.div(min_snr_gamma, snr + 1).float().to(loss.device)
    else:
        snr_weight = torch.div(min_snr_gamma, snr).float().to(loss.device)
    loss = loss * snr_weight
    return loss


def apply_debiased_estimation(loss, timesteps, noise_scheduler, v_prediction=False):
    snr_t = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
    snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)  # if timestep is 0, snr_t is inf, so limit it to 1000
    if v_prediction:
        weight = 1 / (snr_t + 1)
    else:
        weight = 1 / torch.sqrt(snr_t)
    loss = loss * weight.to(loss.device)
    return loss


class SDXLPipeline(BasePipeline):
    # Unique name, used to make the cache_dir path.
    name = 'sdxl'

    # layers that will participate in activation checkpointing
    checkpointable_layers = [
        'InitialLayer',
        'DownBlockInnerLayer',
        'MidBlockInnerLayer',
        'UpBlockInnerLayer',
        'FinalLayer',
    ]

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.v_pred = self.model_config.get('v_pred', False)
        self.min_snr_gamma = self.model_config.get('min_snr_gamma', None)
        self.debiased_estimation_loss = self.model_config.get('debiased_estimation_loss', None)

        if self.v_pred:
            logger.info('Using v-prediction loss')
        if self.min_snr_gamma is not None:
            logger.info(f'Using min_snr_gamma={self.min_snr_gamma}')
        if self.debiased_estimation_loss:
            logger.info('Using debiased_estimation_loss')

        self.diffusers_pipeline = diffusers.StableDiffusionXLPipeline.from_single_file(
            self.model_config['checkpoint_path'],
            torch_dtype=self.model_config['dtype'],
            add_watermarker=False,
        )
        # The two tokenizers should be the same. After all, the two sets of text embeddings are concatenated token-by-token.
        # But in rare cases, the token count can differ (I've only seen off by 1), and this causes a failure. So work around it
        # by forcing the tokenizers to be the same object.
        self.diffusers_pipeline.tokenizer_2 = self.diffusers_pipeline.tokenizer
        self.diffusers_pipeline.scheduler = diffusers.DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
        )

        # TODO: sd-scripts has this come first. But that's technically wrong I think. You would want to change the scheduler
        # parameters to enforce ZTSNR before calculating the SNRs. Leaving it like this for now to match sd-scripts.
        prepare_scheduler_for_custom_training(self.scheduler)
        if self.v_pred:
            fix_noise_scheduler_betas_for_zero_terminal_snr(self.scheduler)

        # Probably good to always do this for SDXL.
        self.diffusers_pipeline.upcast_vae()
        self.unet.train()
        self.text_encoder.train()
        self.text_encoder_2.train()
        self._set_param_original_name()

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def _set_param_original_name(self):
        for state_dict_key_prefix, module in (
            ('unet.', self.unet),
            ('text_encoder.', self.text_encoder),
            ('text_encoder_2.', self.text_encoder_2),
        ):
            for name, p in module.named_parameters():
                p.original_name = state_dict_key_prefix + name

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        # TODO: support training with cached text embeddings.
        return []

    def configure_adapter(self, adapter_config):
        if 'init_from_existing' in adapter_config:
            # For this model, load_adapter_weights() both creates the LoRA and loads its weights.
            return

        # Target all linear layers in the main blocks.
        self._add_adapter(adapter_config, self.unet, [self.unet.down_blocks, self.unet.mid_block, self.unet.up_blocks], state_dict_key_prefix='unet.')
        # Target all linear layers in the text encoder.
        self._add_adapter(adapter_config, self.text_encoder, [self.text_encoder], state_dict_key_prefix='text_encoder.')
        self._add_adapter(adapter_config, self.text_encoder_2, [self.text_encoder_2], state_dict_key_prefix='text_encoder_2.')

    def _add_adapter(self, adapter_config, top_level_module, target_modules, state_dict_key_prefix=''):
        adapter_type = adapter_config['type']
        target_linear_modules = []
        for target_module in target_modules:
            for name, submodule in target_module.named_modules():
                if isinstance(submodule, nn.Linear):
                    target_linear_modules.append(name)

        if adapter_type == 'lora':
            peft_config = peft.LoraConfig(
                r=adapter_config['rank'],
                lora_alpha=adapter_config['alpha'],
                lora_dropout=adapter_config['dropout'],
                bias='none',
                target_modules=target_linear_modules
            )
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
        top_level_module.add_adapter(peft_config)
        for name, p in top_level_module.named_parameters():
            p.original_name = state_dict_key_prefix + name
            if p.requires_grad:
                p.data = p.data.to(adapter_config['dtype'])

    def save_adapter(self, save_dir, peft_state_dict):
        adapter_type = self.config['adapter']['type']
        if adapter_type == 'lora':
            # TODO: should we do any additional checks here? This helpful function appears to completely convert
            # the PEFT format state_dict to kohya format. Every key in the lora is correctly loaded by Forge.
            # But all these different formats are a mess and I hardly understand it. This seems to work though.
            kohya_sd = diffusers.utils.state_dict_utils.convert_state_dict_to_kohya(peft_state_dict)
            safetensors.torch.save_file(kohya_sd, save_dir / 'lora.safetensors', metadata={'format': 'pt'})
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')

    def load_adapter_weights(self, adapter_path):
        print(f'Loading adapter weights from path {adapter_path}')
        safetensors_files = list(Path(adapter_path).glob('*.safetensors'))
        if len(safetensors_files) == 0:
            raise RuntimeError(f'No safetensors file found in {adapter_path}')
        if len(safetensors_files) > 1:
            raise RuntimeError(f'Multiple safetensors files found in {adapter_path}')
        adapter_state_dict = safetensors.torch.load_file(safetensors_files[0])
        self.diffusers_pipeline.load_lora_weights(adapter_state_dict, adapter_name='default')
        self._set_param_original_name()

    def save_model(self, save_dir, diffusers_sd):
        unet_state_dict, text_enc_dict, text_enc_2_dict = {}, {}, {}
        for name, p in diffusers_sd.items():
            if name.startswith('unet.'):
                unet_state_dict[name[len('unet.'):]] = p
            elif name.startswith('text_encoder.'):
                text_enc_dict[name[len('text_encoder.'):]] = p
            elif name.startswith('text_encoder_2.'):
                text_enc_2_dict[name[len('text_encoder_2.'):]] = p
            else:
                raise RuntimeError(f'Unexpected parameter: {name}')

        vae_state_dict = self.vae.state_dict()

        # Convert the UNet model
        unet_state_dict = convert_unet_state_dict(unet_state_dict)
        unet_state_dict = {"model.diffusion_model." + k: v for k, v in unet_state_dict.items()}

        # Convert the VAE model
        vae_state_dict = convert_vae_state_dict(vae_state_dict)
        vae_state_dict = {"first_stage_model." + k: v for k, v in vae_state_dict.items()}

        # Convert text encoder 1
        text_enc_dict = convert_openai_text_enc_state_dict(text_enc_dict)
        text_enc_dict = {"conditioner.embedders.0.transformer." + k: v for k, v in text_enc_dict.items()}

        # Convert text encoder 2
        text_enc_2_dict = convert_openclip_text_enc_state_dict(text_enc_2_dict)
        text_enc_2_dict = {"conditioner.embedders.1.model." + k: v for k, v in text_enc_2_dict.items()}
        # We call the `.T.contiguous()` to match what's done in
        # https://github.com/huggingface/diffusers/blob/84905ca7287876b925b6bf8e9bb92fec21c78764/src/diffusers/loaders/single_file_utils.py#L1085
        text_enc_2_dict["conditioner.embedders.1.model.text_projection"] = text_enc_2_dict.pop(
            "conditioner.embedders.1.model.text_projection.weight"
        ).T.contiguous()

        # Put together new checkpoint
        state_dict = {**unet_state_dict, **vae_state_dict, **text_enc_dict, **text_enc_2_dict}

        save_file(state_dict, save_dir / 'model.safetensors', metadata={"format": "pt"})

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            tensor = tensor.to(vae.device, vae.dtype)
            tensor = tensor*2 - 1
            latents = vae.encode(tensor).latent_dist.sample()
            if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                latents = latents - vae.config.shift_factor
            latents = latents * vae.config.scaling_factor
            return {'latents': latents}
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        caption = inputs['caption']
        mask = inputs['mask']
        input_ids = self._get_input_ids(caption, self.tokenizer)
        input_ids_2 = self._get_input_ids(caption, self.tokenizer_2)

        bs, channels, h, w = latents.shape
        device = latents.device

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension

        noise = torch.randn_like(latents, device=device)
        min_timestep = 0
        max_timestep = self.scheduler.config.num_train_timesteps
        if timestep_quantile is not None:
            t = int(timestep_quantile*max_timestep)
            timesteps = torch.full((bs,), t, device=device)
        else:
            timesteps = torch.randint(min_timestep, max_timestep, (bs,), device=device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        if self.v_pred:
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        pixel_height = latents.shape[-2] * self.vae_scale_factor
        pixel_width = latents.shape[-1] * self.vae_scale_factor
        # TODO: set original size based on actual source image? Not sure what other trainers do here.
        original_size = target_size = (pixel_height, pixel_width)
        add_time_ids = self._get_add_time_ids(
            original_size,
            (0, 0),
            target_size,
            dtype=torch.float32,
            text_encoder_projection_dim=self.text_encoder_2.config.projection_dim,
        ).expand(bs, -1)

        return (noisy_latents, timesteps, input_ids, input_ids_2, add_time_ids), (target, mask)

    def _get_input_ids(self, prompt, tokenizer):
        input_ids = tokenizer(
            prompt,
            padding="longest",
            truncation=False,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(torch.int64)
        return input_ids

    def to_layers(self):
        layers = [InitialLayer(self.diffusers_pipeline)]
        unet = self.diffusers_pipeline.unet
        for block in unet.down_blocks:
            layers.extend(UnetDownBlockLayer(block).to_layers())
        if unet.mid_block is not None:
            layers.extend(UnetMidBlockLayer(unet.mid_block).to_layers())
        for i, block in enumerate(unet.up_blocks):
            is_final_block = i == len(unet.up_blocks) - 1
            layers.extend(UnetUpBlockLayer(block, is_final_block).to_layers())
        layers.append(FinalLayer(unet, self))
        return layers

    def get_param_groups(self, parameters):
        unet_params, text_encoder_params, text_encoder_2_params = [], [], []
        for p in parameters:
            if p.original_name.startswith('unet.'):
                unet_params.append(p)
            elif p.original_name.startswith('text_encoder.'):
                text_encoder_params.append(p)
            elif p.original_name.startswith('text_encoder_2.'):
                text_encoder_2_params.append(p)
            else:
                raise RuntimeError(f'Unexpected parameter: {p.original_name}')
        base_lr = self.config['optimizer'].get('lr', None)
        unet_lr = self.model_config.get('unet_lr', base_lr)
        text_encoder_lr = self.model_config.get('text_encoder_1_lr', base_lr)
        text_encoder_2_lr = self.model_config.get('text_encoder_2_lr', base_lr)
        if is_main_process():
            print(f'Using unet_lr={unet_lr}, text_encoder_1_lr={text_encoder_lr}, text_encoder_2_lr={text_encoder_2_lr}')
        unet_param_group = {'params': unet_params}
        if unet_lr is not None:
            unet_param_group['lr'] = unet_lr
        text_encoder_param_group = {'params': text_encoder_params}
        if text_encoder_lr is not None:
            text_encoder_param_group['lr'] = text_encoder_lr
        text_encoder_2_param_group = {'params': text_encoder_2_params}
        if text_encoder_2_lr is not None:
            text_encoder_2_param_group['lr'] = text_encoder_2_lr
        return [unet_param_group, text_encoder_param_group, text_encoder_2_param_group]

    def get_loss_fn(self):
        def loss_fn(output, label):
            output, timesteps = output
            target, mask = label
            with torch.autocast('cuda', enabled=False):
                output = output.to(torch.float32)
                target = target.to(output.device, torch.float32)
                loss = F.mse_loss(output, target, reduction='none')
                # empty tensor means no masking
                if mask.numel() > 0:
                    mask = mask.to(output.device, torch.float32)
                    loss *= mask
                loss = loss.mean([1, 2, 3])
                if self.min_snr_gamma is not None:
                    loss = apply_snr_weight(loss, timesteps, self.scheduler, self.min_snr_gamma, self.v_pred)
                if self.debiased_estimation_loss is not None:
                    loss = apply_debiased_estimation(loss, timesteps, self.scheduler, self.v_pred)
                loss = loss.mean()
            return loss
        return loss_fn


class InitialLayer(nn.Module):
    def __init__(self, diffusers_pipeline):
        super().__init__()
        self.clip_skip = None
        self.diffusers_pipeline = diffusers_pipeline
        self.text_encoder = self.diffusers_pipeline.text_encoder
        self.text_encoder_2 = self.diffusers_pipeline.text_encoder_2
        self.tokenizer = self.diffusers_pipeline.tokenizer
        self.tokenizer_2 = self.diffusers_pipeline.tokenizer_2
        # Unet modules we need to register on the nn.Module
        self.time_proj = self.unet.time_proj
        self.time_embedding = self.unet.time_embedding
        self.add_embedding = self.unet.add_embedding
        self.time_embed_act = self.unet.time_embed_act
        self.encoder_hid_proj = self.unet.encoder_hid_proj
        self.conv_in = self.unet.conv_in

    @property
    def unet(self):
        return self.diffusers_pipeline.unet

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for tensor in inputs:
            if torch.is_floating_point(tensor):
                tensor.requires_grad_(True)
        sample, timestep, input_ids, input_ids_2, add_time_ids = inputs

        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.unet.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break
        forward_upsample_size = torch.tensor(forward_upsample_size).to(sample.device)

        # prompt_embeds_list = []
        # for text_input_ids, text_encoder in [(input_ids, self.text_encoder), (input_ids_2, self.text_encoder_2)]:
        #     prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)
        #     # We are only ALWAYS interested in the pooled output of the final text encoder
        #     if prompt_embeds[0].ndim == 2:
        #         pooled_prompt_embeds = prompt_embeds[0]

        #     if self.clip_skip is None:
        #         prompt_embeds = prompt_embeds.hidden_states[-2]
        #     else:
        #         # "2" because SDXL always indexes from the penultimate layer.
        #         prompt_embeds = prompt_embeds.hidden_states[-(self.clip_skip + 2)]
        #     prompt_embeds_list.append(prompt_embeds)
        # encoder_hidden_states = torch.concat(prompt_embeds_list, dim=-1)
        encoder_hidden_states, pooled_prompt_embeds = self.get_text_conditioning(input_ids, input_ids_2)

        add_time_ids = add_time_ids.to(pooled_prompt_embeds.dtype)
        add_text_embeds = pooled_prompt_embeds
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        # TODO: need timestep cond? Might be optional / inference only.

        t_emb = self.unet.get_time_embed(sample=sample, timestep=timestep)
        emb = self.unet.time_embedding(t_emb, None)

        aug_emb = self.unet.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.unet.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        sample = self.conv_in(sample)
        down_block_res_samples = (sample,)

        return make_contiguous(sample, timestep, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size)

    def get_text_conditioning(self, input_ids, input_ids_2):
        prompt_embeds = self.get_prompt_embeds(input_ids, self.tokenizer, self.text_encoder)
        prompt_embeds_2, pooled_prompt_embeds = self.get_prompt_embeds(input_ids_2, self.tokenizer_2, self.text_encoder_2, return_pooled_prompt_embeds=True)
        encoder_hidden_states = torch.concat([prompt_embeds, prompt_embeds_2], dim=-1)
        return encoder_hidden_states, pooled_prompt_embeds

    # Supports arbitrary length input_ids by splitting into chunks of at most model_max_length.
    def get_prompt_embeds(self, input_ids, tokenizer, text_encoder, return_pooled_prompt_embeds=False):
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id
        bs = input_ids.shape[0]
        device = input_ids.device
        chunks = torch.split(input_ids, tokenizer.model_max_length-2, dim=-1)
        processed_chunks = []
        # Add BOS and pad token, then replace the first pad token in each example with EOS.
        for chunk in chunks:
            chunk = torch.cat(
                [
                    torch.full((bs, 1), bos, device=device),
                    chunk,
                    torch.full((bs, 1), pad, device=device)
                ],
                dim=-1
            )
            first_pad_idx = torch.argmax((chunk == pad).to(torch.int32), dim=-1)
            chunk[torch.arange(chunk.shape[0]), first_pad_idx] = eos
            processed_chunks.append(chunk)

        embed_chunks = []
        for i, input_ids in enumerate(processed_chunks):
            prompt_embeds = text_encoder(input_ids, output_hidden_states=True)
            if i == 0 and return_pooled_prompt_embeds:
                pooled_prompt_embeds = prompt_embeds[0]

            if self.clip_skip is None:
                prompt_embeds = prompt_embeds.hidden_states[-2]
            else:
                # "2" because SDXL always indexes from the penultimate layer.
                prompt_embeds = prompt_embeds.hidden_states[-(self.clip_skip + 2)]

            embed_chunks.append(prompt_embeds)

        prompt_embeds = torch.cat(embed_chunks, dim=1)
        if return_pooled_prompt_embeds:
            return prompt_embeds, pooled_prompt_embeds
        return prompt_embeds


class DownBlockInnerLayer(nn.Module):
    def __init__(self, resnet, attn, append_residual_hidden_states=True):
        super().__init__()
        self.resnet = resnet
        self.attn = attn
        self.append_residual_hidden_states = append_residual_hidden_states

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, timesteps, emb, encoder_hidden_states, *res_hidden_states, forward_upsample_size = inputs
        hidden_states = self.resnet(hidden_states, emb)
        if self.attn is not None:
            hidden_states = self.attn(hidden_states, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        res_hidden_states += (hidden_states,)
        return make_contiguous(hidden_states, timesteps, emb, encoder_hidden_states, *res_hidden_states, forward_upsample_size)


class MidBlockInnerLayer(nn.Module):
    def __init__(self, resnet, attn):
        super().__init__()
        self.resnet = resnet
        self.attn = attn

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, timesteps, emb, encoder_hidden_states, *res_hidden_states, forward_upsample_size = inputs
        hidden_states = self.resnet(hidden_states, emb)
        if self.attn is not None:
            hidden_states = self.attn(hidden_states, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        return make_contiguous(hidden_states, timesteps, emb, encoder_hidden_states, *res_hidden_states, forward_upsample_size)


class UpBlockInnerLayer(nn.Module):
    def __init__(self, resnet, attn):
        super().__init__()
        self.resnet = resnet
        self.attn = attn

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, timesteps, emb, encoder_hidden_states, *res_hidden_states, forward_upsample_size = inputs
        res_tmp = res_hidden_states[-1]
        res_hidden_states = res_hidden_states[:-1]
        hidden_states = torch.cat([hidden_states, res_tmp], dim=1)
        hidden_states = self.resnet(hidden_states, emb)
        if self.attn is not None:
            hidden_states = self.attn(hidden_states, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
        return make_contiguous(hidden_states, timesteps, emb, encoder_hidden_states, *res_hidden_states, forward_upsample_size)


class DownsamplerLayer(nn.Module):
    def __init__(self, downsamplers):
        super().__init__()
        self.downsamplers = downsamplers

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, timesteps, emb, encoder_hidden_states, *res_hidden_states, forward_upsample_size = inputs
        for downsampler in self.downsamplers:
            hidden_states = downsampler(hidden_states)
        res_hidden_states += (hidden_states,)
        return make_contiguous(hidden_states, timesteps, emb, encoder_hidden_states, *res_hidden_states, forward_upsample_size)


class UpsamplerLayer(nn.Module):
    def __init__(self, upsamplers, is_final_block):
        super().__init__()
        self.upsamplers = upsamplers
        self.is_final_block = is_final_block

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        hidden_states, timesteps, emb, encoder_hidden_states, *res_hidden_states, forward_upsample_size = inputs
        if not self.is_final_block and forward_upsample_size:
            upsample_size = res_hidden_states[-1].shape[2:]
        else:
            upsample_size = None
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states, upsample_size)
        return make_contiguous(hidden_states, timesteps, emb, encoder_hidden_states, *res_hidden_states, forward_upsample_size)


class UnetDownBlockLayer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        sample, timesteps, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size = inputs

        if hasattr(self.block, "has_cross_attention") and self.block.has_cross_attention:
            sample, res_samples = self.block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
            )
        else:
            sample, res_samples = self.block(hidden_states=sample, temb=emb)

        down_block_res_samples += res_samples
        return make_contiguous(sample, timesteps, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size)

    def to_layers(self):
        layers = []
        resnets = self.block.resnets
        attentions = getattr(self.block, 'attentions', [None] * len(resnets))
        for resnet, attention in zip(resnets, attentions):
            layers.append(DownBlockInnerLayer(resnet, attention))
        if self.block.downsamplers is not None:
            layers.append(DownsamplerLayer(self.block.downsamplers))
        return layers


class UnetMidBlockLayer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        sample, timesteps, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size = inputs

        if hasattr(self.block, "has_cross_attention") and self.block.has_cross_attention:
            sample = self.block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
            )
        else:
            sample = self.mid_block(sample, emb)

        return make_contiguous(sample, timesteps, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size)

    def to_layers(self):
        layers = []
        resnets = self.block.resnets
        attentions = self.block.attentions
        layers.append(MidBlockInnerLayer(resnets[0], None))
        for attn, resnet in zip(attentions, resnets[1:]):
            layers.append(MidBlockInnerLayer(resnet, attn))
        return layers


class UnetUpBlockLayer(nn.Module):
    def __init__(self, block, is_final_block):
        super().__init__()
        self.block = block
        self.is_final_block = is_final_block

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        sample, timesteps, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size = inputs

        res_samples = down_block_res_samples[-len(self.block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(self.block.resnets)]

        # if we have not reached the final block and need to forward the
        # upsample size, we do it here
        if not self.is_final_block and forward_upsample_size:
            upsample_size = down_block_res_samples[-1].shape[2:]
        else:
            upsample_size = None

        if hasattr(self.block, "has_cross_attention") and self.block.has_cross_attention:
            sample = self.block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                upsample_size=upsample_size,
            )
        else:
            sample = self.block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                upsample_size=upsample_size,
            )

        return make_contiguous(sample, timesteps, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size)

    def to_layers(self):
        layers = []
        resnets = self.block.resnets
        attentions = getattr(self.block, 'attentions', [None] * len(resnets))
        for resnet, attention in zip(resnets, attentions):
            layers.append(UpBlockInnerLayer(resnet, attention))
        if self.block.upsamplers is not None:
            layers.append(UpsamplerLayer(self.block.upsamplers, self.is_final_block))
        return layers


class FinalLayer(nn.Module):
    def __init__(self, unet, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.conv_norm_out = unet.conv_norm_out
        self.conv_act = unet.conv_act
        self.conv_out = unet.conv_out

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        sample, timesteps, emb, encoder_hidden_states, *down_block_res_samples, forward_upsample_size = inputs

        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        return self.conv_out(sample), timesteps
