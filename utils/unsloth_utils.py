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

# I (tdrussell) made some modifications.

import torch

# Only offload Tensors with at least this many elements.
OFFLOAD_THRESHOLD = 5_000_000  # 10 MB for half-precision

class Unsloth_Offloaded_Gradient_Checkpointer(torch.autograd.Function):
    """
    Code licensed under LGPL
    Saves VRAM by smartly offloading to RAM.
    Tiny hit to performance, since we mask the movement via non blocking calls.
    """

    # Skips saving for backward any tensors with no_backward set to True.
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, forward_function, *args):
        saved_args = []
        saved_indices = []
        for i, x in enumerate(args):
            if getattr(x, 'no_backward', False):
                continue
            saved_args.append(x.to('cpu', non_blocking=True) if x.numel() >= OFFLOAD_THRESHOLD else x)
            saved_indices.append(i)

        with torch.no_grad():
            output = forward_function(*args)
        ctx.save_for_backward(*saved_args)
        ctx.forward_function = forward_function
        ctx.saved_indices = saved_indices
        ctx.num_args = len(args)
        return output

    # For tensors not saved for backward, they are passed to function as None. The function should still return the
    # same number of values in this case, but some of them can be None.
    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, *grads):
        args = [None]*ctx.num_args
        saved_indices = ctx.saved_indices
        for i, x in enumerate(ctx.saved_tensors):
            x = x.to('cuda', non_blocking=True).detach()
            if torch.is_floating_point(x):
                x.requires_grad_(True)
            args[saved_indices[i]] = x

        with torch.enable_grad():
            outputs = ctx.forward_function(*args)

        output_tensors = []
        grad_tensors = []
        for out, grad in zip(outputs, grads):
            if out is not None and out.requires_grad:
                output_tensors.append(out)
                grad_tensors.append(grad)
        torch.autograd.backward(output_tensors, grad_tensors)
        return (None,) + tuple(None if arg is None else arg.grad for arg in args)


@torch._disable_dynamo
def unsloth_checkpoint(function, *args):
    return Unsloth_Offloaded_Gradient_Checkpointer.apply(function, *args)
