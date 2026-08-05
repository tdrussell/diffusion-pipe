import sys
from pprint import pprint
import os.path
sys.path.insert(0, os.path.abspath('submodules/Cosmos'))

import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5TokenizerFast, T5EncoderModel
import accelerate
from einops import rearrange
import safetensors

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from utils.common import load_state_dict, AUTOCAST_DTYPE, is_main_process
from cosmos1.models.diffusion.inference.inference_utils import load_model_by_config
from cosmos1.models.autoregressive.tokenizer.modules import EncoderFactorized, DecoderFactorized, CausalConv3d


FRAMERATE = 24
SIGMA_DATA = 0.5

SUPPORTED_SIZE_BUCKETS = [
    [960, 960, 1],
    [960, 704, 1],
    [704, 960, 1],
    [1280, 704, 1],
    [704, 1280, 1],
    [960, 960, 121],
    [960, 704, 121],
    [704, 960, 121],
    [1280, 704, 121],
    [704, 1280, 121],
]


def get_per_sigma_loss_weights(sigma: torch.Tensor):
    """
    Args:
        sigma (tensor): noise level

    Returns:
        loss weights per sigma noise level
    """
    return (sigma**2 + SIGMA_DATA**2) / (sigma * SIGMA_DATA) ** 2


class CausalContinuousVideoTokenizer(nn.Module):
    def __init__(self, z_channels: int, z_factor: int, embedding_dim: int, **kwargs) -> None:
        super().__init__()
        self.name = kwargs.get("name", "CausalContinuousVideoTokenizer")
        self.embedding_dim = embedding_dim
        self.spatial_compression = kwargs['spatial_compression']
        self.temporal_compression = kwargs['temporal_compression']
        self.sigma_data = SIGMA_DATA
        self.encoder = EncoderFactorized(z_channels=z_factor * z_channels, **kwargs)
        self.decoder = DecoderFactorized(z_channels=z_channels, **kwargs)

        self.quant_conv = CausalConv3d(z_factor * z_channels, embedding_dim, kernel_size=1, padding=0)
        self.post_quant_conv = CausalConv3d(embedding_dim, z_channels, kernel_size=1, padding=0)

        latent_temporal_chunk = 16
        self.latent_mean = nn.Parameter(torch.zeros([self.embedding_dim * latent_temporal_chunk], dtype=torch.float32))
        self.latent_std = nn.Parameter(torch.ones([self.embedding_dim * latent_temporal_chunk], dtype=torch.float32))

    def encode(self, x):
        h = self.encoder(x)
        z = self.quant_conv(h)
        latent_ch = z.shape[1]
        latent_t = z.shape[2]
        dtype = z.dtype
        mean = self.latent_mean.view(latent_ch, -1)[:, : latent_t].reshape([1, latent_ch, -1, 1, 1]).to(dtype=dtype, device=z.device)
        std = self.latent_std.view(latent_ch, -1)[:, : latent_t].reshape([1, latent_ch, -1, 1, 1]).to(dtype=dtype, device=z.device)
        return ((z - mean) / std) * self.sigma_data

    def decode(self, z):
        in_dtype = z.dtype
        latent_ch = z.shape[1]
        latent_t = z.shape[2]
        mean = self.latent_mean.view(latent_ch, -1)[:, : latent_t].reshape([1, latent_ch, -1, 1, 1]).to(dtype=in_dtype, device=z.device)
        std = self.latent_std.view(latent_ch, -1)[:, : latent_t].reshape([1, latent_ch, -1, 1, 1]).to(dtype=in_dtype, device=z.device)
        z = z / self.sigma_data
        z = z * std + mean
        z = self.post_quant_conv(z)
        return self.decoder(z)


def load_custom_video_vae(path):
    with accelerate.init_empty_weights():
        vae = CausalContinuousVideoTokenizer(
            attn_resolutions=[32],
            channels=128,
            channels_mult=[2, 4, 4],
            dropout=0.0,
            in_channels=3,
            num_res_blocks=2,
            out_channels=3,
            resolution=1024,
            patch_size=4,
            patch_method="haar",
            z_channels=16,
            z_factor=1,
            num_groups=1,
            legacy_mode=False,
            spatial_compression=8,
            temporal_compression=8,
            embedding_dim=16,
        )
    missing_keys, unexpected_keys = vae.load_state_dict(load_state_dict(path), assign=True, strict=False)
    assert len(missing_keys) == 0
    vae.eval()
    return vae


def vae_encode(tensor, vae):
    # tensor values already in range [-1, 1] here
    p = next(vae.encoder.parameters())
    # TODO: the official code would call vae.encode_image() when it detects frames=1.
    # Should we use the image encoder (separate model)?
    tensor = tensor.to(p.device, p.dtype)
    tensor = tensor*2 - 1
    return vae.encode(tensor)


def dataset_config_validation(config):
    if 'min_ar' in config or 'max_ar' in config or 'num_ar_buckets' in config or 'resolutions' in config:
        return False
    size_buckets = config.get('size_buckets', [])
    if len(size_buckets) == 0:
        return False
    for size_bucket in size_buckets:
        if size_bucket not in SUPPORTED_SIZE_BUCKETS:
            return False
    return True


class CosmosPipeline(BasePipeline):
    name = 'cosmos'
    framerate = FRAMERATE
    checkpointable_layers = ['InitialLayer', 'TransformerLayer', 'FinalLayer']
    adapter_target_modules = ['GeneralDITTransformerBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']

        # TODO: different model variants
        self.model = load_model_by_config(
            config_job_name='Cosmos_1_0_Diffusion_Text2World_7B',
            config_file='submodules/Cosmos/cosmos1/models/diffusion/config/config.py',
        )

        self.vae = load_custom_video_vae(self.model_config['vae_path'])

        self.tokenizer = T5TokenizerFast(
            vocab_file='configs/t5_old/spiece.model',
            tokenizer_file='configs/t5_old/tokenizer.json',
        )
        t5_state_dict = load_state_dict(self.model_config['text_encoder_path'])
        self.text_encoder = T5EncoderModel.from_pretrained(
            None,
            config='configs/t5_old/config.json',
            state_dict=t5_state_dict,
            torch_dtype='auto',
            local_files_only=True,
        )

    def load_diffusion_model(self):
        with accelerate.init_empty_weights():
            self.model.model = self.model.build_model()
        net_state_dict = load_state_dict(self.model_config['transformer_path'])
        incompatible = self.model.model.load_state_dict(net_state_dict, strict=False, assign=True)
        missing_keys = [k for k in incompatible.missing_keys if "_extra_state" not in k]
        assert len(missing_keys) == 0
        self.transformer = self.model.net

    def model_specific_dataset_config_validation(self, dataset_config):
        passes_validation = True
        passes_validation &= dataset_config_validation(dataset_config)
        for directory_config in dataset_config['directory']:
            passes_validation &= dataset_config_validation(directory_config)
        if not passes_validation:
            if is_main_process():
                print('WARNING: Cosmos supports a limited set of resolutions. Anything else will likely not work correctly.'
                      ' See the cosmos_dataset.toml example. If you still want to proceed with the current configuration,'
                      ' run the script with the --i_know_what_i_am_doing flag.')
            quit()

    def get_vae(self):
        return self.vae

    def get_text_encoders(self):
        return [self.text_encoder]

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI format.
        peft_state_dict = {'diffusion_model.'+k: v for k, v in peft_state_dict.items()}
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(
            self.config,
            support_video=True,
            framerate=self.framerate,
            round_frames=8,
        )

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            return {'latents': vae_encode(tensor, vae)}
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(captions, is_video):
            # args are lists
            batch_encoding = self.tokenizer.batch_encode_plus(
                captions,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512,
                return_length=True,
                return_offsets_mapping=False,
            )

            input_ids = batch_encoding.input_ids
            attn_mask = batch_encoding.attention_mask

            device = text_encoder.device
            outputs = text_encoder(input_ids=input_ids.to(device), attention_mask=attn_mask.to(device))

            encoded_text = outputs.last_hidden_state
            lengths = attn_mask.sum(dim=1).cpu()

            for batch_id in range(encoded_text.shape[0]):
                encoded_text[batch_id][lengths[batch_id] :] = 0

            return {'prompt_embeds': encoded_text, 'prompt_attention_mask': attn_mask}
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        prompt_embeds = inputs['prompt_embeds']
        mask = inputs['mask']

        bs, channels, num_frames, h, w = latents.shape
        device = latents.device

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)  # make mask same number of dims as target

        noise = torch.randn_like(latents)

        dist = torch.distributions.normal.Normal(0, 1)
        if timestep_quantile is not None:
            log_sigma = dist.icdf(torch.full((bs,), timestep_quantile, device=device))
        else:
            log_sigma = dist.sample((bs,)).to(device)
        sigma = torch.exp(log_sigma)

        x_t = latents + sigma.view(-1, 1, 1, 1, 1) * noise

        c_skip, c_out, c_in, c_noise = self.model.scaling(sigma=sigma)
        x = x_t * c_in.view(-1, 1, 1, 1, 1)
        timesteps = c_noise
        target = latents

        return (x, x_t, timesteps, prompt_embeds, sigma), (target, mask)

    def to_layers(self):
        layers = [InitialLayer(self.transformer, self.vae.spatial_compression)]
        for name, block in self.transformer.blocks.items():
            layers.append(TransformerLayer(block))
        layers.append(FinalLayer(self))
        return layers

    def get_loss_fn(self):
        def loss_fn(output, label):
            output, weights_per_sigma = output
            target, mask = label
            with torch.autocast('cuda', enabled=False):
                output = output.to(torch.float32)
                target = target.to(output.device, torch.float32)
                loss = F.mse_loss(output, target, reduction='none')
                # empty tensor means no masking
                if mask.numel() > 0:
                    mask = mask.to(output.device, torch.float32)
                    loss *= mask
                loss = loss * weights_per_sigma
                loss = loss.mean()
            return loss
        return loss_fn


class InitialLayer(nn.Module):
    def __init__(self, transformer, spatial_compression_factor):
        super().__init__()
        self.transformer = [transformer]
        self.spatial_compression_factor = spatial_compression_factor
        self.x_embedder = transformer.x_embedder
        self.extra_pos_embedder = transformer.extra_pos_embedder
        self.pos_embedder = transformer.pos_embedder
        self.t_embedder = transformer.t_embedder

    def __getattr__(self, name):
        return getattr(self.transformer[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)

        x, x_t, timesteps, crossattn_emb, sigma = inputs
        original_shape = x.shape
        dtype = x.dtype
        device = x.device

        # Official code sets up these inputs in prepare_data_batch.
        fps = torch.tensor([FRAMERATE] * 1, dtype=dtype, device=device)
        height = x.shape[-2] * self.spatial_compression_factor
        width = x.shape[-1] * self.spatial_compression_factor
        image_size = torch.tensor([[height, width, height, width]] * 1, dtype=dtype, device=device)
        padding_mask = torch.zeros((1, 1, height, width), dtype=dtype, device=device)

        # x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
        #     x,
        #     fps=fps,
        #     padding_mask=padding_mask,
        #     latent_condition=None,
        #     latent_condition_sigma=None,
        # )
        inputs = self.forward_before_blocks(
            x=x,
            timesteps=timesteps,
            crossattn_emb=crossattn_emb,
            # Model should have use_cross_attn_mask=False. Will assert fail otherwise.
            crossattn_mask=None,
            fps=fps,
            image_size=image_size,
            padding_mask=padding_mask,
        )
        x, affline_emb_B_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_3D, original_shape = (
            inputs["x"],
            inputs["affline_emb_B_D"],
            inputs["crossattn_emb"],
            inputs["rope_emb_L_1_1_D"],
            inputs["adaln_lora_B_3D"],
            inputs["original_shape"],
        )
        extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = inputs["extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D"]
        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert (
                x.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape
            ), f"{x.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape} {original_shape}"

        original_shape = torch.tensor(original_shape)
        return make_contiguous(
            x,
            x_t,
            affline_emb_B_D,
            crossattn_emb,
            rope_emb_L_1_1_D,
            adaln_lora_B_3D,
            extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            original_shape,
            sigma,
        )


class TransformerLayer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, x_t, affline_emb_B_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_3D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, original_shape, sigma = inputs
        x = self.block(
            x,
            affline_emb_B_D,
            crossattn_emb,
            None,
            rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            adaln_lora_B_3D=adaln_lora_B_3D,
            extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        )
        return make_contiguous(
            x,
            x_t,
            affline_emb_B_D,
            crossattn_emb,
            rope_emb_L_1_1_D,
            adaln_lora_B_3D,
            extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            original_shape,
            sigma,
        )


class FinalLayer(nn.Module):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.final_layer = self.pipeline.transformer.final_layer

    def __getattr__(self, name):
        return getattr(self.pipeline.transformer, name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, x_t, affline_emb_B_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_3D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D, original_shape, sigma = inputs
        original_shape = original_shape.tolist()

        x_B_T_H_W_D = rearrange(x, "T H W B D -> B T H W D")
        output = self.decoder_head(
            x_B_T_H_W_D=x_B_T_H_W_D,
            emb_B_D=affline_emb_B_D,
            crossattn_emb=None,
            origin_shape=original_shape,
            crossattn_mask=None,
            adaln_lora_B_3D=adaln_lora_B_3D,
        )

        c_skip, c_out, c_in, c_noise = self.pipeline.model.scaling(sigma=sigma)
        c_skip = c_skip.view(-1, 1, 1, 1, 1)
        c_out = c_out.view(-1, 1, 1, 1, 1)
        sigma = sigma.view(-1, 1, 1, 1, 1)
        x0_pred = c_skip*x_t + c_out*output
        weights_per_sigma = get_per_sigma_loss_weights(sigma)
        return x0_pred, weights_per_sigma
