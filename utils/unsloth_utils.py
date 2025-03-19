# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# I (tdrussell) made a few modifications.

import torch
from deepspeed.runtime.activation_checkpointing.checkpointing import detach_variable


class Unsloth_Offloaded_Gradient_Checkpointer(torch.autograd.Function):
    """
    Code licensed under LGPL
    Saves VRAM by smartly offloading to RAM.
    Tiny hit to performance, since we mask the movement via non blocking calls.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, forward_function, hidden_states, *args):
        saved_hidden_states = hidden_states.to('cpu', non_blocking=True)
        with torch.no_grad():
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(saved_hidden_states)
        ctx.forward_function = forward_function
        ctx.args = args
        return output

    pass

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, *grads):
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.to('cuda', non_blocking=True).detach()
        hidden_states.requires_grad_(True)
        args = detach_variable(ctx.args)
        inputs = (hidden_states,) + args
        with torch.enable_grad():
            outputs = ctx.forward_function(*inputs)

        output_tensors = []
        grad_tensors = []
        for out, grad in zip(outputs, grads):
            if out.requires_grad:
                output_tensors.append(out)
                grad_tensors.append(grad)
        torch.autograd.backward(output_tensors, grad_tensors)
        return (None,) + tuple(input.grad for input in inputs)

    pass


pass


@torch._disable_dynamo
def unsloth_checkpoint(function, *args):
    return Unsloth_Offloaded_Gradient_Checkpointer.apply(function, *args)
