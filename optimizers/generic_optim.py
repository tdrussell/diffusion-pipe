# Implementation taken from https://github.com/timmytonga/sn-sm
# Modified to automatically do Kahan summation for bfloat16 parameters.
# Made resuming from checkpoint work, but ONLY for the svd case.
# Muon method taken from: https://github.com/KellerJordan/Muon

from typing import Callable, Iterable, Tuple
import math
from .projectors.svd_projector import SVDProjector
from .projectors.uniform_projector import UniformProjector  # get random subset
from .projectors.topk_norm_projector import TopKNormProjector  # topk indices

import torch
from torch.optim import Optimizer

from transformers.utils.versions import require_version


NS_STEPS = 5


def has_inf_or_nan(x):
    s = x.sum()
    return s.isinf() or s.isnan()


def get_and_update_subset_norm_denom(group, state, grad, beta2):
    # First, compute subset norm if applicable
    if "subset_size" in group:
        if group.get('correct_dim', False):
            reduce_fn = torch.mean
        else:
            reduce_fn = torch.sum
        if group["subset_size"] == "heuristics":  # heuristics
            if "reduce_dim" not in state:
                state["reduce_dim"] = 0 if grad.shape[0] >= grad.shape[1] else 1
            second_moment_update = reduce_fn(grad ** 2, dim=(1 - state["reduce_dim"]), keepdim=True)
        else:  # it is an int
            assert group["subset_size"] != 0, f"Subset size should not be 0."
            if "subset_shape" not in state:
                numel = grad.numel()
                if group["subset_size"] > 0:
                    reduce_size = closest_smaller_divisor_of_n_to_k(numel, group["subset_size"])
                else:  # default is sqrt
                    div = abs(int(group["subset_size"]))
                    reduce_size = closest_smaller_divisor_of_n_to_k(numel, int(math.sqrt(numel) / div))
                state["subset_shape"] = (numel // reduce_size, reduce_size)
            reshaped_grad = grad.view(state["subset_shape"])
            second_moment_update = reduce_fn(reshaped_grad ** 2, dim=1, keepdim=True)
    else:  # standard EMA
        second_moment_update = grad ** 2

    # Initialization
    if "exp_avg_sq" not in state:
        state["exp_avg_sq"] = torch.zeros_like(second_moment_update)
    exp_avg_sq = state["exp_avg_sq"]

    # Second moment term update
    if beta2 < 1:  # EMA
        exp_avg_sq.mul_(beta2).add_(second_moment_update, alpha=1.0 - beta2)
    else:  # AdaGrad
        exp_avg_sq.add_(second_moment_update)
    return exp_avg_sq.sqrt().add_(group["eps"])


def get_and_update_subspace_momentum(group, state, p):
    grad = p.grad
    beta1, beta2 = group["betas"]

    # Projection for compressing momentum term
    if "rank" in group:
        proj_grad = get_projected_grad(group, state, p)
    else:  # if not SM or module is not set then it's just standard momentum
        proj_grad = grad

    # Init
    if "exp_avg" not in state:
        state["exp_avg"] = torch.zeros_like(proj_grad)
    # Momentum term
    exp_avg = state["exp_avg"]

    # reset exp_avg state when we update as default
    if ("rank" in group and state["step"] > 1 and state["step"] % group["update_proj_gap"] == 0):
        if "overlap_state" not in group:
            state["exp_avg"] = torch.zeros_like(proj_grad)
        # else we overlap the momentum update where we don't need to do anything

    # Subspace momentum and orthogonal SGD
    if "rank" in group:
        exp_avg.mul_(beta1).add_(proj_grad, alpha=(1.0 - beta1))
        orth_comp = grad - state["projector"].project_back(proj_grad)
        numerator = state["projector"].project_back(exp_avg) + orth_comp
    else:  # just normal full momentum
        exp_avg.mul_(beta1).add_(proj_grad, alpha=(1.0 - beta1))
        numerator = exp_avg

    return numerator


def get_projected_grad(group, state, p):
    if "projector" not in state:
        state["projector"] = get_projector(group, p)
    proj_grad = state["projector"].project(p.grad, state["step"])
    return proj_grad


def get_projector(group, p):
    if group["proj_type"] == "topk":
        return TopKNormProjector(
            group["rank"], update_proj_gap=group["update_proj_gap"],
            scale=1.0, proj_type=group["proj_type"],
            param_shape=p.shape
        )
    elif group["proj_type"] == "uniform":
        return UniformProjector(
            group["rank"], update_proj_gap=group["update_proj_gap"],
            scale=1.0, param_shape=p.shape  # change scale later but don't want to
        )
    elif group["proj_type"] == "svd" or group["proj_type"] == "srht":
        if "approx_svd" not in group:
            group["approx_svd"] = False
            group['asvd_rank_scale'] = 1
        return SVDProjector(
            group["rank"], update_proj_gap=group["update_proj_gap"],
            scale=1.0, proj_type=group["proj_type"],
            approx_svd=group["approx_svd"], asvd_rank_scale=group['asvd_rank_scale'],
            param_shape=p.shape
        )
    else:
        raise ValueError(f"Invalid proj_type {group['proj_type']}")


def closest_smaller_divisor_of_n_to_k(n: int, k: int) -> int:
    """
    Helper function for subset-norm subset-size computation.
    Get the closest smaller divisor of n to k.
    """
    assert k <= n
    if n % k == 0:
        return k
    if n <= 1 or k <= 1:
        raise ValueError
    # Start from sqrt_N and work downwards
    for i in range(int(k), 0, -1):
        if n % i == 0:
            print(f"Choosing subset-size: {k} is not a divisor of total numel {n}. "
                  f"Picking {i} that is the closest smaller divisor.")
            return int(i)


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


zeropower_via_newtonschulz5_compile = None


class GenericOptim(Optimizer):
    """
    Parameters:
        params (`Iterable[Any]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
            *IMPORTANT*: If using subset-norm or subspace-momentum, must create param_groups to specify which parameters
              that we are applying SN and SM to. Please see README for more details.
              For SM, to enable, we need to specify projection type and rank.
                proj_type in ["svd", "uniform", "topk"]
                rank is an integer that is less than the dimension of each param.
              For SN, we need to specify the subset sizes.
                subset_size should be either
                 - a positive integer that is less than the number of parameters: to group params with that size
                 - a negative integer to use adaptive subset size of sqrt(d)/k params grouping
                 - "heuristics" to use the heuristics described in the paper.
        lr (`float`, *optional*, defaults to 0.001):
            The base learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Momentum and second moment averaging parameters (b1, b2).
            Set b2=1 to use AdaGrad style accumulation for the second moment term.
            Note that these parameters will only take affect if momentum_type and second_moment_type are not none.
        eps (`float`, *optional*, defaults to 1e-06):
            Second moment's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        momentum_type (`Literal["ema", "none", "sm"]`, defaults to "ema"):
            Specify the type of momentum to use. Set beta1 to 0 to NOT use momentum. This saves memory.
            "ema" is standard Adam's EMA momentum.
            "sm" means we use subspace momentum.
            "none" means we do not use momentum. Can also set beta1 = 0
              *IMPORTANT*: need to specify which parameters to use SM proj_type and rank by setting params_group.
              to enable, we need to specify projection type and rank.
                - proj_type in ["std", "uniform", "topk"]
                - rank is an integer that is less than the dimension of each param.
        second_moment_type (`Literal["ema", "none", "sn"]`, defaults to "ema"):
            Specify which type of second moment to use.
            "ema" is standard Adam/RMSprop's EMA momentum. Note we can set beta2=1 to use AdaGrad.
            "none" means we don't use adaptive step size.
            "sn" means we use subset norm.
              *IMPORTANT*: need to specify which parameters to use SN and for what subset size to use.
              subset_size should be either
                 - a positive integer that is less than the number of parameters: to group params with that size
                 - a negative integer to use adaptive subset size of sqrt(d)/k params grouping
                 - "heuristics" to use the heuristics described in the paper.
    """

    def __init__(
            self,
            params: Iterable,
            lr: float = 1e-3,
            # set beta2 = 1 to use AdaGrad-style accumulation
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            weight_decay_type: str = "default", # default_scheduled, cautious, cautious_scheduled
            correct_bias: bool = True,
            momentum_type: str = "ema",
            second_moment_type: str = "ema",
            correct_dim=False,
            cpu_offload=False,
            muon=False,
            adamuon=False,
            compile=False,
            automagic=False,
            min_lr=1e-7,
            max_lr=1e-3,
            lr_bump=1e-6, # amount to bump the lr when adjusting
            lr_decrease_factor=1.0, # how much more to decrease the LR vs increase
            skip_invalid_grads=False,
    ):
        self.momentum_type = momentum_type
        assert self.momentum_type in ["ema", "sm", "none"]
        self.second_moment_type = second_moment_type
        assert self.second_moment_type in ["ema", "none", "sn", "factored"]
        self.skip_invalid_grads = skip_invalid_grads
        
        self.weight_decay_type = weight_decay_type
        assert self.weight_decay_type in ["default", "default_scheduled", "cautious", "cautious_scheduled"]
        self.start_lr = lr
        
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] <= 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0]")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        if muon and adamuon:
            raise ValueError('Only one of muon, adamuon can be True')

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias, 'correct_dim': correct_dim,
                    'cpu_offload': cpu_offload, 'muon': muon, 'adamuon': adamuon, 'compile': compile, 'automagic': automagic, 'min_lr': min_lr,
                    'max_lr': max_lr, 'lr_bump': lr_bump, 'lr_decrease_factor': lr_decrease_factor}
        super().__init__(params, defaults)
        self.check_params()
        # Print out all configurations
        print(f"GenericOptim Configuration: lr={lr}, betas={betas}, eps={eps}, weight_decay={weight_decay}, "
              f"correct_bias={correct_bias}, momentum_type={momentum_type}, second_moment_type={second_moment_type}, correct_dim={correct_dim}, "
              f"cpu_offload={cpu_offload}, muon={muon}, adamuon={adamuon}, compile={compile}, automagic={automagic}, min_lr={min_lr}, "
              f"max_lr={max_lr}, lr_bump={lr_bump}, lr_decrease_factor={lr_decrease_factor}")

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        synchronize = False
        skipped_parameter_names = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Currently does not support sparse gradients.")

                if self.skip_invalid_grads and has_inf_or_nan(p.grad):
                    skipped_parameter_names.append(getattr(p, 'original_name', None))
                    continue

                # Setup
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                state["step"] += 1
                cpu_offload = group['cpu_offload'] if p.ndim >= 2 else False
                state_device = 'cpu' if cpu_offload else p.device

                # learning rate
                if group.get('automagic', False):
                    automagic_lr = self.update_automagic_lr(group, state, p.grad, state_device)
                    step_size = 1.0
                else:
                    automagic_lr = None
                    step_size = group['lr']

                # get momentum
                numerator = self.get_numerator(group, state, p, state_device)
                can_use_muon = numerator.ndim > 1
                muon = group['muon'] and can_use_muon
                adamuon = group['adamuon'] and can_use_muon

                if muon or adamuon:
                    rows, cols = numerator.shape[-2:]
                    if numerator.ndim == 4: # for the case of conv filters
                        numerator = numerator.view(len(numerator), -1)
                    if group['compile']:
                        global zeropower_via_newtonschulz5_compile
                        if zeropower_via_newtonschulz5_compile is None:
                            zeropower_via_newtonschulz5_compile = torch.compile(zeropower_via_newtonschulz5)
                        orthogonalize = zeropower_via_newtonschulz5_compile
                    else:
                        orthogonalize = zeropower_via_newtonschulz5
                    numerator = orthogonalize(numerator, steps=NS_STEPS)
                    step_size *= 0.2

                if muon:
                    step_size *= math.sqrt(max(rows, cols))
                    denominator = None
                elif adamuon:
                    denominator = self.get_denominator(group, state, numerator, state_device)
                    numerator.div_(denominator)
                    if group["correct_bias"]:
                        beta1, beta2 = group["betas"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        numerator.mul_(math.sqrt(bias_correction2))
                    step_size /= math.sqrt(torch.mean(numerator**2).item()) + group['eps']
                    denominator = None
                else:
                    denominator = self.get_denominator(group, state, p.grad, state_device)
                    # Bias correction
                    beta1, beta2 = group["betas"]
                    if group["correct_bias"]:
                        bias_correction1 = 1.0 - beta1 ** state["step"]
                        bias_correction2 = 1.0 - beta2 ** state["step"]
                        step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                if automagic_lr is not None:
                    numerator.mul_(automagic_lr)

                update = torch.zeros_like(p)

                # step
                if denominator is None:  # no adaptive step size
                    update.add_(numerator, alpha=-step_size)
                elif self.second_moment_type in ('ema', 'factored'):  # standard adam
                    update.addcdiv_(numerator, denominator, value=-step_size)
                elif self.second_moment_type == "sn":  # subset norm requires broadcast division
                    if "subset_size" in group and group["subset_size"] != "heuristics":
                        norm_grad = (numerator.view(state["subset_shape"]) / denominator).reshape(p.shape)
                        update.add_(norm_grad, alpha=-step_size)
                    else:  # broadcast division is default for heuristics and non-subset-norm modules
                        update.addcdiv_(numerator, denominator, value=-step_size)
                else:
                    raise ValueError(f"Should not be here. Denominator is not None but second_moment_type "
                                     f"is {self.second_moment_type}")

                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    
                    # Allows WD to cooldown as we reach the end of training process, as it bound to scheduled lr
                    if "scheduled" in self.weight_decay_type:
                        effective_wd = group["lr"] * group["lr"] * group["weight_decay"] / self.start_lr
                    else:
                        effective_wd = group["lr"] * group["weight_decay"]
                    
                    if "cautious" in self.weight_decay_type:
                        # "Cautious" weight decay (https://arxiv.org/abs/2510.12402)
                        p.copy_(torch.where(update * p >= 0, p * (1.0 - effective_wd), p))
                    else:
                        # Vanilla AdamW's decay mechanism
                        update.add_(p, alpha=-effective_wd)

                synchronize |= cpu_offload

                if p.dtype == torch.bfloat16:
                    # Kahan summation for bfloat16
                    if 'shift' not in state:
                        state['shift'] = torch.zeros_like(p)
                    shift = state['shift'].to(p.device, non_blocking=True)
                    shift.add_(update)
                    # Use grad as temp buffer
                    p.grad.copy_(p.detach())
                    p.add_(shift)
                    shift.add_(p.grad.sub_(p))
                    # TODO: non_blocking=True here causes CUDA error on first step after checkpoint save.
                    state['shift'] = shift.to(state_device)
                else:
                    p.add_(update)

        if synchronize:
            # Because we did non_blocking transfer in GPU -> CPU direction
            torch.cuda.synchronize()

        if len(skipped_parameter_names) > 0:
            print(f'WARNING: {len(skipped_parameter_names)} parameter updates were skipped due to Inf or NaN.')

        return loss

    def get_numerator(self, group, state, p, state_device):
        grad = p.grad
        beta1, beta2 = group["betas"]
        if beta1 == 0 or self.momentum_type == "none":
            return grad

        if self.momentum_type == "sm":
            return get_and_update_subspace_momentum(group, state, p)
        elif self.momentum_type == "ema":  # standard adam's ema
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(grad)
            # Momentum term
            exp_avg = state["exp_avg"].to(p.device, non_blocking=True)
            exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
            state['exp_avg'] = exp_avg.to(state_device)
            return exp_avg
        else:
            raise ValueError(f"Unrecognized momentum_type = {self.momentum_type}.")

    def get_denominator(self, group, state, grad, state_device):
        beta1, beta2 = group["betas"]
        second_moment_type = self.second_moment_type
        if second_moment_type == 'factored' and grad.ndim == 1:
            second_moment_type = 'ema'
        if beta2 == 0 or second_moment_type == "none":
            return None  # this means only use base lr
        elif second_moment_type == "ema":  # Adam style
            if "exp_avg_sq" not in state:  # initialization
                state["exp_avg_sq"] = torch.zeros_like(grad)
            exp_avg_sq = state["exp_avg_sq"].to(grad.device, non_blocking=True)
            if beta2 < 1:  # EMA
                exp_avg_sq.mul_(beta2).add_(grad**2, alpha=1.0 - beta2)
            else:  # == 1 means AdaGrad
                exp_avg_sq.add_(grad**2)
            state['exp_avg_sq'] = exp_avg_sq.to(state_device)
            return exp_avg_sq.sqrt().add_(group["eps"])
        elif second_moment_type == "sn":
            return get_and_update_subset_norm_denom(group, state, grad, beta2)
        elif second_moment_type == 'factored':
            if 'exp_avg_sq_row' not in state:
                state['exp_avg_sq_row'] = torch.zeros(grad.shape[:-1], dtype=grad.dtype, device=grad.device)
                state['exp_avg_sq_col'] = torch.zeros(grad.shape[:-2] + grad.shape[-1:], dtype=grad.dtype, device=grad.device)
            exp_avg_sq_row = state['exp_avg_sq_row']
            exp_avg_sq_col = state['exp_avg_sq_col']
            grad_sq = grad**2
            exp_avg_sq_row.mul_(beta2).add_(grad_sq.mean(dim=-1), alpha=(1.0 - beta2))
            exp_avg_sq_col.mul_(beta2).add_(grad_sq.mean(dim=-2), alpha=(1.0 - beta2))
            r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).unsqueeze(-1)
            c_factor = exp_avg_sq_col.unsqueeze(-2)
            return torch.mul(r_factor, c_factor).sqrt_().add_(group['eps'])
        else:
            raise ValueError(f"Unrecognized second moment (adaptive step size) type {second_moment_type}.")

    def update_automagic_lr(self, group, state, update, state_device):
        if 'automagic_lr' not in state:
            state['automagic_lr'] = torch.full_like(update, group['lr_bump'])
            state['avg_lr'] = torch.mean(state['automagic_lr'])
        automagic_lr = state['automagic_lr'].to(update.device, non_blocking=True)
        if 'exp_avg' in state:
            # Use momentum for last_polarity.
            exp_avg = state['exp_avg'].to(update.device, non_blocking=True)  # may be offloaded
            state['exp_avg'] = exp_avg  # slight optimization for future code if we moved it to GPU
            last_polarity = exp_avg > 0
        else:
            # We use the sign bit as the last polarity because lr must be positive.
            last_polarity = automagic_lr > 0
        automagic_lr = automagic_lr.abs()
        current_polarity = update > 0
        lr_bump = group['lr_bump']
        new_lr = torch.where(
            last_polarity == current_polarity,
            automagic_lr + lr_bump,  # Increase lr
            automagic_lr - group['lr_decrease_factor']*lr_bump  # Decrease lr
        )
        new_lr = torch.clamp(
            new_lr,
            min=group['min_lr'],
            max=group['max_lr'],
        )
        state['avg_lr'] = torch.mean(new_lr)
        # Store polarity in sign bit.
        state['automagic_lr'] = torch.where(current_polarity, new_lr, -new_lr).to(state_device)
        return new_lr

    @staticmethod
    def _get_lr(param_group, param_state):
        if 'avg_lr' in param_state:
            lr = param_state["avg_lr"]
        else:
            lr = torch.tensor(0.0)
        return lr

    @torch.no_grad()
    def check_params(self):
        """
        Check if parameters set are all okay and raise error if there is any strange combination.
        """
        have_seen_subset_size = False
        have_seen_rank = False
        # check if all the param groups are configured correctly
        for group in self.param_groups:
            if "subset_size" in group:
                print(f"GenericOptim: SubsetSize is set to {group['subset_size']}")
                have_seen_subset_size = True
                if isinstance(group["subset_size"], int):
                    assert group["subset_size"] != 0, f"Subset size must be a non-zero integer"
                else:
                    assert group["subset_size"] == "heuristics", "Subset size must be a non-zero int or 'heuristics.'"
            if "rank" in group:
                have_seen_rank = True
                assert "update_proj_gap" in group, "rank set but update_proj_gap is not set!"
                print(f"GenericOptim: ProjType={group['proj_type']}, Rank={group['rank']}, "
                      f"Gap={group['update_proj_gap']}")
            if "update_proj_gap" in group:
                assert "rank" in group, "update_proj_gap set but rank is not set!"
        if self.second_moment_type == "sn" and not have_seen_subset_size:
            raise ValueError("second_moment_type is set to use subset-norm (sn) but have not seen any subset_size "
                             "enable in param_groups. If you meant to use EMA, please set second_moment_type='ema'")
        if have_seen_subset_size and self.second_moment_type != "sn":
            raise ValueError(f"second_moment_type is set to '{self.second_moment_type}' but "
                             "encountered subset_size in param_groups."
                             "Do you mean to use subset-norm? If so, set second_moment_type to 'sn'. "
                             "Otherwise, if you want to use ema, remove subset_size from param_groups")
        if self.momentum_type == "sm" and not have_seen_rank:
            raise ValueError("Set second_moment_type to use subspace-momentum (sm) but have not seen any rank set "
                             " in any param_groups. If you meant to use EMA, please set momentum_type='ema'")

    def load_state_dict(self, sd):
        super().load_state_dict(sd)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # State is moved to the device of the param, so for offloaded state, move to CPU.
                # TODO: this kind of works but is suboptimal. It is still initially loading all state
                # on GPU. This uses more VRAM than necessary, but it isn't too bad because it happens
                # before any training steps.
                cpu_offload = group['cpu_offload'] if p.ndim >= 2 else False
                for k, v in state.items():
                    if torch.is_tensor(v) and cpu_offload:
                        state[k] = v.to('cpu')
                # The projector is a class which contains tensors, so state needs to be explicitly moved to the correct device.
                if 'projector' in state:
                    state['projector'].to(p.device)
