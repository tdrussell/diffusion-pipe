"""CUDA regression tests for mixed-format block swapping.

Run: python -m unittest discover -s tests -p test_offloading.py -v
The quantized fixture models separate logical and physical tensor storage,
without requiring ComfyUI or comfy-kitchen.
"""
import unittest

import torch
from torch import nn

from utils.offloading import swap_weight_devices_cuda


class WeightAlias(nn.Module):
    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer

    @property
    def weight(self):
        return self.base_layer.weight


class Block(nn.Module):
    def __init__(self, device, dtype, quantized=False, storage_shape=(8, 8), alias=False):
        super().__init__()
        self.proj = nn.Module()
        self.proj.weight = nn.Parameter(
            torch.zeros(8, 8, device=device, dtype=dtype), requires_grad=False
        )
        if quantized:
            self.proj.weight._qdata = torch.arange(
                storage_shape[0] * storage_shape[1], device=device, dtype=torch.int8
            ).reshape(storage_shape)
        else:
            self.proj.weight.data.copy_(torch.randn(8, 8, device=device, dtype=dtype))
        if alias:
            self.proj = WeightAlias(self.proj)
        # LoRA weights must stay on CUDA even while the base weight is offloaded.
        self.lora_A = nn.Linear(8, 2, bias=False, device="cuda")


def storage(block):
    weight = block.proj.weight
    return weight._qdata if hasattr(weight, "_qdata") else weight.data


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for block swapping")
class SwapWeightDevicesTest(unittest.TestCase):
    def check_swap(self, outgoing_spec, incoming_spec, reusable=False, outgoing_cpu=False):
        outgoing = Block("cpu" if outgoing_cpu else "cuda", **outgoing_spec)
        incoming = Block("cpu", **incoming_spec)
        expected_outgoing = storage(outgoing).cpu().clone()
        expected_incoming = storage(incoming).clone()
        original_dtype = (storage(outgoing).dtype, storage(incoming).dtype)
        lora_values = (outgoing.lora_A.weight.clone(), incoming.lora_A.weight.clone())

        for _ in range(3):
            cuda_ptr = storage(outgoing).data_ptr()
            swap_weight_devices_cuda(torch.device("cuda"), outgoing, incoming)
            self.assertEqual(storage(outgoing).device.type, "cpu")
            self.assertEqual(storage(incoming).device.type, "cuda")
            self.assertTrue(torch.equal(storage(outgoing), expected_outgoing))
            self.assertTrue(torch.equal(storage(incoming).cpu(), expected_incoming))
            self.assertEqual((storage(outgoing).dtype, storage(incoming).dtype), original_dtype)
            if reusable:
                self.assertEqual(storage(incoming).data_ptr(), cuda_ptr)
            swap_weight_devices_cuda(torch.device("cuda"), incoming, outgoing)
            self.assertTrue(torch.equal(storage(outgoing).cpu(), expected_outgoing))
            self.assertTrue(torch.equal(storage(incoming), expected_incoming))
            for block, expected in zip((outgoing, incoming), lora_values):
                self.assertEqual(block.lora_A.weight.device.type, "cuda")
                self.assertTrue(torch.equal(block.lora_A.weight, expected))

    def test_bf16_to_int8(self):
        self.check_swap(dict(dtype=torch.bfloat16), dict(dtype=torch.bfloat16, quantized=True))

    def test_int8_to_bf16(self):
        self.check_swap(dict(dtype=torch.bfloat16, quantized=True), dict(dtype=torch.bfloat16))

    def test_different_plain_dtypes(self):
        self.check_swap(dict(dtype=torch.bfloat16), dict(dtype=torch.float32))

    def test_different_quantized_storage_shapes(self):
        self.check_swap(
            dict(dtype=torch.bfloat16, quantized=True),
            dict(dtype=torch.bfloat16, quantized=True, storage_shape=(8, 4)),
        )

    def test_aliased_bf16_to_int8(self):
        self.check_swap(
            dict(dtype=torch.bfloat16, alias=True),
            dict(dtype=torch.bfloat16, quantized=True, alias=True),
        )

    def test_aliased_int8_to_bf16(self):
        self.check_swap(
            dict(dtype=torch.bfloat16, quantized=True, alias=True),
            dict(dtype=torch.bfloat16, alias=True),
        )

    def test_already_offloaded_outgoing_weight(self):
        self.check_swap(
            dict(dtype=torch.bfloat16), dict(dtype=torch.bfloat16, quantized=True),
            outgoing_cpu=True,
        )

    def test_same_bf16_reuses_buffer(self):
        self.check_swap(dict(dtype=torch.bfloat16), dict(dtype=torch.bfloat16), reusable=True)

    def test_same_int8_reuses_buffer(self):
        spec = dict(dtype=torch.bfloat16, quantized=True)
        self.check_swap(spec, spec, reusable=True)


if __name__ == "__main__":
    unittest.main()
