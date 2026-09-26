import torch
import bitsandbytes
import bitsandbytes.functional as F


class AdamW8bitKahan(bitsandbytes.optim.AdamW8bit):
    def __init__(self, *args, stabilize=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.stabilize = stabilize

    @torch.no_grad()
    def init_state(self, group, p, gindex, pindex):
        super().init_state(group, p, gindex, pindex)
        self.state[p]['shift'] = self.get_state_buffer(p, dtype=p.dtype)

    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        # avoid update error from non-contiguous memory layout
        p.data = p.data.contiguous()
        p.grad = p.grad.contiguous()

        state = self.state[p]
        grad = p.grad

        config = self.get_config(gindex, pindex, group)

        state["step"] += 1
        step = state["step"]
        shift = state['shift']

        # StableAdamW
        if self.stabilize:
            exp_avg_sq = state['state2']
            eps_sq = torch.tensor(config['eps']**2, dtype=exp_avg_sq.dtype, device=exp_avg_sq.device)
            rms = grad.pow(2).div_(exp_avg_sq.maximum(eps_sq)).mean().sqrt()
            lr = config['lr'] / max(1, rms.item())
        else:
            lr = config['lr']

        if state["state1"].dtype == torch.float:
            F.optimizer_update_32bit(
                self.optimizer_name,
                grad,
                shift,
                state["state1"],
                config["betas"][0],
                config["eps"],
                step,
                lr,
                state["state2"],
                config["betas"][1],
                config["betas"][2] if len(config["betas"]) >= 3 else 0.0,
                config.get("alpha", 0.0),
                config["weight_decay"],
                1.0,
                state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
                skip_zeros=config["skip_zeros"],
            )

        elif state["state1"].dtype == torch.uint8:
            F.optimizer_update_8bit_blockwise(
                self.optimizer_name,
                grad,
                shift,
                state["state1"],
                state["state2"],
                config["betas"][0],
                config["betas"][1],
                config["betas"][2] if len(config["betas"]) >= 3 else 0.0,
                config.get("alpha", 0.0),
                config["eps"],
                step,
                lr,
                state["qmap1"],
                state["qmap2"],
                state["absmax1"],
                state["absmax2"],
                config["weight_decay"],
                gnorm_scale=1.0,
                skip_zeros=config["skip_zeros"],
            )

        buffer = p.clone()
        p.add_(shift)
        shift.add_(buffer.sub_(p))
