from typing import Optional
import sys
import os.path
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/HunyuanVideo'))

import torch
from torch import nn
import peft
from peft.tuners._buffer_dict import BufferDict
from transformers import CLIPTextModel, AutoModel

import hyvideo.text_encoder
from hyvideo.constants import PRECISION_TO_TYPE, TEXT_ENCODER_PATH


def _move_adapter_to_device_of_base_layer(self, adapter_name: str, device: Optional[torch.device] = None) -> None:
    """
    Move the adapter of the given name to the device of the base layer.
    """
    if device is None:
        # check weight and qweight (for GPTQ)
        for weight_name in ("weight", "qweight"):
            weight = getattr(self.get_base_layer(), weight_name, None)
            if weight is not None:
                device = weight.device
                dtype = weight.dtype
                break
        else:
            # no break encountered: could not determine the device
            return

    meta = torch.device("meta")

    # loop through all potential adapter layers and move them to the device of the base layer; be careful to only
    # move this specific adapter to the device, as the other adapters could be on different devices
    # see #1639
    for adapter_layer_name in self.adapter_layer_names + self.other_param_names:
        adapter_layer = getattr(self, adapter_layer_name, None)
        if not isinstance(adapter_layer, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
            continue
        if adapter_name not in adapter_layer:
            continue
        if any(p.device == meta for p in adapter_layer.parameters()):
            continue

        if ((weight.dtype.is_floating_point or weight.dtype.is_complex)
            # This is the part I added.
            and not (weight.dtype == torch.float8_e4m3fn or weight.dtype == torch.float8_e5m2)):
            adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device, dtype=dtype)
        else:
            adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device)


def load_text_encoder(
    text_encoder_type,
    text_encoder_precision=None,
    text_encoder_path=None,
    logger=None,
    device=None,
):
    if text_encoder_path is None:
        text_encoder_path = TEXT_ENCODER_PATH[text_encoder_type]
    if logger is not None:
        logger.info(
            f"Loading text encoder model ({text_encoder_type}) from: {text_encoder_path}"
        )

    torch_dtype = 'auto'
    if text_encoder_precision is not None:
        torch_dtype = PRECISION_TO_TYPE[text_encoder_precision]

    if text_encoder_type == "clipL":
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, torch_dtype=torch_dtype)
        text_encoder.final_layer_norm = text_encoder.text_model.final_layer_norm
    elif text_encoder_type == "llm":
        text_encoder = AutoModel.from_pretrained(
            text_encoder_path, low_cpu_mem_usage=True, torch_dtype=torch_dtype
        )
        text_encoder.final_layer_norm = text_encoder.norm
    else:
        raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")
    # from_pretrained will ensure that the model is in eval mode.

    text_encoder.requires_grad_(False)

    if logger is not None:
        logger.info(f"Text encoder to dtype: {text_encoder.dtype}")

    if device is not None:
        text_encoder = text_encoder.to(device)

    return text_encoder, text_encoder_path


def apply_patches():
    # Prevent PEFT from downcasting LoRA weights to fp8 only for this script to upcast them again.
    # TODO: probably should send a PR to PEFT. Default behavior looks like a mistake to me.
    peft.tuners.tuners_utils.BaseTunerLayer._move_adapter_to_device_of_base_layer = _move_adapter_to_device_of_base_layer

    # Use torch_dtype to avoid needlessly loading the text encoder in float32, only to cast it right after.
    hyvideo.text_encoder.load_text_encoder = load_text_encoder
