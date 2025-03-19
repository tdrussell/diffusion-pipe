import sys
import json
import math
import re
import os.path
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/Wan2_1'))

import torch
from torch import nn
import torch.nn.functional as F
import safetensors
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from utils.common import AUTOCAST_DTYPE
from utils.offloading import ModelOffloader
import wan
from wan.modules.t5 import T5Encoder, T5Decoder, T5Model
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.vae import WanVAE
from wan.modules.model import (
    WanModel, sinusoidal_embedding_1d, WanLayerNorm, WanSelfAttention, WAN_CROSSATTENTION_CLASSES
)
from wan.modules.clip import CLIPModel
from wan import configs as wan_configs
from safetensors.torch import load_file

KEEP_IN_HIGH_PRECISION = ['norm', 'bias', 'patch_embedding', 'text_embedding', 'time_embedding', 'time_projection', 'head', 'modulation']


class WanModelFromSafetensors(WanModel):
    @classmethod
    def from_pretrained(
        cls,
        weights_file,
        config_file,
        torch_dtype=torch.bfloat16,
        transformer_dtype=torch.bfloat16,
    ):
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)

        with init_empty_weights():
            model = cls(**config)

        state_dict = load_file(weights_file, device='cpu')
        state_dict = {
            re.sub(r'^model\.diffusion_model\.', '', k): v for k, v in state_dict.items()
        }

        for name, param in model.named_parameters():
            dtype_to_use = torch_dtype if any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) else transformer_dtype
            set_module_tensor_to_device(model, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])

        return model

def vae_encode(tensor, vae):
    return vae.model.encode(tensor, vae.scale)

def umt5_keys_mapping_comfy(state_dict):
    import re
    # define key mappings rule
    def execute_mapping(original_key):
        # Token embedding mapping
        if original_key == "shared.weight":
            return "token_embedding.weight"

        # Final layer norm mapping
        if original_key == "encoder.final_layer_norm.weight":
            return "norm.weight"

        # Block layer mappings
        block_match = re.match(r"encoder\.block\.(\d+)\.layer\.(\d+)\.(.+)", original_key)
        if block_match:
            block_num = block_match.group(1)
            layer_type = int(block_match.group(2))
            rest = block_match.group(3)

            # self-attn layer（layer.0）
            if layer_type == 0:
                if "SelfAttention" in rest:
                    attn_part = rest.split(".")[1]
                    if attn_part in ["q", "k", "v", "o"]:
                        return f"blocks.{block_num}.attn.{attn_part}.weight"
                    elif attn_part == "relative_attention_bias":
                        return f"blocks.{block_num}.pos_embedding.embedding.weight"
                elif rest == "layer_norm.weight":
                    return f"blocks.{block_num}.norm1.weight"

            # FFN Layer（layer.1）
            elif layer_type == 1:
                if "DenseReluDense" in rest:
                    parts = rest.split(".")
                    if parts[1] == "wi_0":
                        return f"blocks.{block_num}.ffn.gate.0.weight"
                    elif parts[1] == "wi_1":
                        return f"blocks.{block_num}.ffn.fc1.weight"
                    elif parts[1] == "wo":
                        return f"blocks.{block_num}.ffn.fc2.weight"
                elif rest == "layer_norm.weight":
                    return f"blocks.{block_num}.norm2.weight"

        return None

    new_state_dict = {}
    unmapped_keys = []

    for key, value in state_dict.items():
        new_key = execute_mapping(key)
        if new_key:
            new_state_dict[new_key] = value
        else:
            unmapped_keys.append(key)

    print(f"Unmapped keys (usually safe to ignore): {unmapped_keys}")
    del state_dict
    return new_state_dict


def umt5_keys_mapping_kijai(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("attention.", "attn.")
        new_key = new_key.replace("final_norm.weight", "norm.weight")
        new_state_dict[new_key] = value
    del state_dict
    return new_state_dict

def umt5_keys_mapping(state_dict):
    if 'blocks.0.attn.k.weight' in state_dict:
        print("loading kijai warpper umt5 safetensors model...")
        return umt5_keys_mapping_kijai(state_dict)
    else:
        print("loading comfyui repacked umt5 safetensors model...")
        return umt5_keys_mapping_comfy(state_dict)

# We can load T5 a lot faster by copying some code so we can construct the model
# inside an init_empty_weights() context.

def _t5(name,
        encoder_only=False,
        decoder_only=False,
        return_tokenizer=False,
        tokenizer_kwargs={},
        dtype=torch.float32,
        device='cpu',
        **kwargs):
    # sanity check
    assert not (encoder_only and decoder_only)

    # params
    if encoder_only:
        model_cls = T5Encoder
        kwargs['vocab'] = kwargs.pop('vocab_size')
        kwargs['num_layers'] = kwargs.pop('encoder_layers')
        _ = kwargs.pop('decoder_layers')
    elif decoder_only:
        model_cls = T5Decoder
        kwargs['vocab'] = kwargs.pop('vocab_size')
        kwargs['num_layers'] = kwargs.pop('decoder_layers')
        _ = kwargs.pop('encoder_layers')
    else:
        model_cls = T5Model

    # init model
    with torch.device(device):
        model = model_cls(**kwargs)

    # init tokenizer
    if return_tokenizer:
        tokenizer = HuggingfaceTokenizer(f'google/{name}', **tokenizer_kwargs)
        return model, tokenizer
    else:
        return model


def umt5_xxl(**kwargs):
    cfg = dict(
        vocab_size=256384,
        dim=4096,
        dim_attn=4096,
        dim_ffn=10240,
        num_heads=64,
        encoder_layers=24,
        decoder_layers=24,
        num_buckets=32,
        shared_pos=False,
        dropout=0.1)
    cfg.update(**kwargs)
    return _t5('umt5-xxl', **cfg)


class T5EncoderModel:

    def __init__(
        self,
        text_len,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        checkpoint_path=None,
        tokenizer_path=None,
        shard_fn=None,
    ):
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        with init_empty_weights():
            model = umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=dtype,
                device=device).eval().requires_grad_(False)

        if checkpoint_path.endswith('.safetensors'):
            state_dict = load_file(checkpoint_path, device='cpu')
            state_dict = umt5_keys_mapping(state_dict)
        else:
            state_dict = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(state_dict, assign=True)
        self.model = model
        if shard_fn is not None:
            self.model = shard_fn(self.model, sync_module_states=False)
        else:
            self.model.to(self.device)
        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=text_len, clean='whitespace')

    def __call__(self, texts, device):
        ids, mask = self.tokenizer(
            texts, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]


# Wrapper to hold both VAE and CLIP, so we can move both to/from GPU together.
class VaeAndClip(nn.Module):
    def __init__(self, vae, clip):
        super().__init__()
        self.vae = vae
        self.clip = clip


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        e = (self.modulation + e).chunk(6, dim=1)

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        with torch.autocast('cuda', dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


# Patch these to remove some forced casting to float32, saving memory.
wan.modules.model.WanAttentionBlock = WanAttentionBlock
wan.modules.model.Head = Head


class WanPipeline(BasePipeline):
    name = 'wan'
    framerate = 16
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = ['WanAttentionBlock']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        ckpt_dir = self.model_config['ckpt_path']
        dtype = self.model_config['dtype']

        self.original_model_config_path = os.path.join(ckpt_dir, 'config.json')
        with open(self.original_model_config_path) as f:
            json_config = json.load(f)
        self.i2v = (json_config['model_type'] == 'i2v')
        model_dim = json_config['dim']
        if not self.i2v and model_dim == 1536:
            wan_config = wan_configs.t2v_1_3B
        elif self.i2v and model_dim == 5120:
            wan_config = wan_configs.i2v_14B
        elif not self.i2v and model_dim == 5120:
            wan_config = wan_configs.t2v_14B
        else:
            raise RuntimeError(f'Could not autodetect model variant. model_dim={model_dim}, i2v={self.i2v}')

        # This is the outermost class, which isn't a nn.Module
        t5_model_path = self.model_config['llm_path'] if self.model_config.get('llm_path', None) else os.path.join(ckpt_dir, wan_config.t5_checkpoint)
        self.text_encoder = T5EncoderModel(
            text_len=wan_config.text_len,
            dtype=dtype,
            device='cpu',
            checkpoint_path=t5_model_path,
            tokenizer_path=os.path.join(ckpt_dir, wan_config.t5_tokenizer),
            shard_fn=None,
        )

        # Same here, this isn't a nn.Module.
        # TODO: by default the VAE is float32, and therefore so are the latents. Do we want to change that?
        self.vae = WanVAE(
            vae_pth=os.path.join(ckpt_dir, wan_config.vae_checkpoint),
            device='cpu',
        )
        # These need to be on the device the VAE will be moved to during caching.
        self.vae.mean = self.vae.mean.to('cuda')
        self.vae.std = self.vae.std.to('cuda')
        self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]

        if self.i2v:
            self.clip = CLIPModel(
                dtype=dtype,
                device='cpu',
                checkpoint_path=os.path.join(ckpt_dir, wan_config.clip_checkpoint),
                tokenizer_path=os.path.join(ckpt_dir, wan_config.clip_tokenizer)
            )

    # delay loading transformer to save RAM
    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        if transformer_path := self.model_config.get('transformer_path', None):
            self.transformer = WanModelFromSafetensors.from_pretrained(
                transformer_path,
                self.original_model_config_path,
                torch_dtype=dtype,
                transformer_dtype=transformer_dtype,
            )
        else:
            self.transformer = WanModel.from_pretrained(self.model_config['ckpt_path'], torch_dtype=dtype)
            for name, p in self.transformer.named_parameters():
                if not (any(x in name for x in KEEP_IN_HIGH_PRECISION)):
                    p.data = p.data.to(transformer_dtype)

        self.transformer.train()
        # We'll need the original parameter name for saving, and the name changes once we wrap modules for pipeline parallelism,
        # so store it in an attribute here. Same thing below if we're training a lora and creating lora weights.
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def __getattr__(self, name):
        return getattr(self.diffusers_pipeline, name)

    def get_vae(self):
        vae = self.vae.model
        clip = self.clip.model if self.i2v else None
        return VaeAndClip(vae, clip)

    def get_text_encoders(self):
        # Return the inner nn.Module
        return [self.text_encoder.model]

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI format.
        peft_state_dict = {'diffusion_model.'+k: v for k, v in peft_state_dict.items()}
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

    def get_call_vae_fn(self, vae_and_clip):
        def fn(tensor):
            vae = vae_and_clip.vae
            p = next(vae.parameters())
            tensor = tensor.to(p.device, p.dtype)
            latents = vae_encode(tensor, self.vae)
            ret = {'latents': latents}
            clip = vae_and_clip.clip
            if clip is not None:
                assert tensor.ndim == 5, f'i2v must train on videos, got tensor with shape {tensor.shape}'
                first_frame = tensor[:, :, 0:1, ...].clone()
                tensor[:, :, 1:, ...] = 0
                # Image conditioning. Same shame as latents, first frame is unchanged, rest is 0.
                # NOTE: encoding 0s with the VAE doesn't give you 0s in the latents, I tested this. So we need to
                # encode the whole thing here, we can't just extract the first frame from the latents later and make
                # the rest 0. But what happens if you do that? Probably things get fried, but might be worth testing.
                y = vae_encode(tensor, self.vae)
                clip_context = self.clip.visual(first_frame.to(p.device, p.dtype))
                ret['y'] = y
                ret['clip_context'] = clip_context
            return ret
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(caption, is_video):
            # Args are lists
            p = next(text_encoder.parameters())
            ids, mask = self.text_encoder.tokenizer(caption, return_mask=True, add_special_tokens=True)
            ids = ids.to(p.device)
            mask = mask.to(p.device)
            seq_lens = mask.gt(0).sum(dim=1).long()
            with torch.autocast(device_type=p.device.type, dtype=p.dtype):
                text_embeddings = text_encoder(ids, mask)
                return {'text_embeddings': text_embeddings, 'seq_lens': seq_lens}
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        # TODO: why does text_embeddings become float32 here? It's bfloat16 coming out of the text encoder.
        text_embeddings = inputs['text_embeddings']
        seq_lens = inputs['seq_lens']
        mask = inputs['mask']
        y = inputs['y'] if self.i2v else None
        clip_context = inputs['clip_context'] if self.i2v else None

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

        x_1 = latents
        x_0 = torch.randn_like(x_1)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        x_t = (1 - t_expanded) * x_1 + t_expanded * x_0
        target = x_0 - x_1

        # timestep input to model needs to be in range [0, 1000]
        t = t * 1000

        return (
            (x_t, y, t, text_embeddings, seq_lens, clip_context),
            (target, mask),
        )

    def to_layers(self):
        transformer = self.transformer
        layers = [InitialLayer(transformer)]
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
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        transformer.blocks = None
        transformer.to('cuda')
        transformer.blocks = blocks
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
        self.patch_embedding = model.patch_embedding
        self.time_embedding = model.time_embedding
        self.text_embedding = model.text_embedding
        self.time_projection = model.time_projection
        self.i2v = (model.model_type == 'i2v')
        if self.i2v:
            self.img_emb = model.img_emb
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)

        x, y, t, context, text_seq_lens, clip_fea = inputs
        bs, channels, f, h, w = x.shape
        if clip_fea.numel() == 0:
            clip_fea = None
        context = [emb[:length] for emb, length in zip(context, text_seq_lens)]

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if self.i2v:
            mask = torch.zeros((bs, 4, f, h, w), device=x.device, dtype=x.dtype)
            mask[:, :, 0, ...] = 1
            y = torch.cat([mask, y], dim=1)
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        seq_len = seq_lens.max()
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(x.device, torch.float32))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if self.i2v:
            assert clip_fea is not None
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # pipeline parallelism needs everything on the GPU
        seq_lens = seq_lens.to(x.device)
        grid_sizes = grid_sizes.to(x.device)

        return make_contiguous(x, e, e0, seq_lens, grid_sizes, self.freqs, context)


class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, e, e0, seq_lens, grid_sizes, freqs, context = inputs

        self.offloader.wait_for_block(self.block_idx)
        x = self.block(x, e0, seq_lens, grid_sizes, freqs, context, None)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(x, e, e0, seq_lens, grid_sizes, freqs, context)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.head = model.head
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x, e, e0, seq_lens, grid_sizes, freqs, context = inputs
        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x, dim=0)
