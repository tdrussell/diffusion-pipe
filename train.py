import argparse
import os
from datetime import datetime, timezone
import shutil
import glob
import time
import random
import json
import inspect
import signal
import sys

import toml
import deepspeed
from deepspeed import comm as dist
from deepspeed.runtime.pipe import module as ds_pipe_module
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import multiprocess as mp
import numpy as np

from utils import dataset as dataset_util
from utils import common
from utils.common import is_main_process, get_rank, DTYPE_MAP
import utils.saver
from utils.isolate_rng import isolate_rng
from utils.patches import apply_patches

TIMESTEP_QUANTILES_FOR_EVAL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--resume_from_checkpoint', nargs='?', const=True, default=None,
                    help='resume training from checkpoint. If no value is provided, resume from the most recent checkpoint. If a folder name is provided, resume from that specific folder.')
parser.add_argument('--regenerate_cache', action='store_true', default=None, help='Force regenerate cache. Useful if none of the files have changed but their contents have, e.g. modified captions.')
parser.add_argument('--cache_only', action='store_true', default=None, help='Cache model inputs then exit.')
parser.add_argument('--i_know_what_i_am_doing', action='store_true', default=None, help="Skip certain checks and overrides. You may end up using settings that won't work.")
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


# Monkeypatch this so it counts all layer parameters, not just trainable parameters.
# This helps it divide the layers between GPUs more evenly when training a LoRA.
def _count_all_layer_params(self):
    param_counts = [0] * len(self._layer_specs)
    for idx, layer in enumerate(self._layer_specs):
        if isinstance(layer, ds_pipe_module.LayerSpec):
            l = layer.build()
            param_counts[idx] = sum(p.numel() for p in l.parameters())
        elif isinstance(layer, nn.Module):
            param_counts[idx] = sum(p.numel() for p in layer.parameters())
    return param_counts
ds_pipe_module.PipelineModule._count_layer_params = _count_all_layer_params


def set_config_defaults(config):
    # Force the user to set this. If we made it a default of 1, it might use a lot of disk space.
    assert 'save_every_n_epochs' in config

    config.setdefault('pipeline_stages', 1)
    config.setdefault('activation_checkpointing', False)
    config.setdefault('warmup_steps', 0)
    if 'save_dtype' in config:
        config['save_dtype'] = DTYPE_MAP[config['save_dtype']]

    model_config = config['model']
    model_dtype_str = model_config['dtype']
    model_config['dtype'] = DTYPE_MAP[model_dtype_str]
    if 'transformer_dtype' in model_config:
        model_config['transformer_dtype'] = DTYPE_MAP[model_config['transformer_dtype']]
    model_config.setdefault('guidance', 1.0)

    if 'adapter' in config:
        adapter_config = config['adapter']
        adapter_type = adapter_config['type']
        if adapter_config['type'] == 'lora':
            if 'alpha' in adapter_config:
                raise NotImplementedError(
                    'This script forces alpha=rank to make the saved LoRA format simpler and more predictable with downstream inference programs. Please remove alpha from the config.'
                )
            adapter_config['alpha'] = adapter_config['rank']
            adapter_config.setdefault('dropout', 0.0)
            adapter_config.setdefault('dtype', model_dtype_str)
            adapter_config['dtype'] = DTYPE_MAP[adapter_config['dtype']]
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')

    config.setdefault('logging_steps', 1)
    config.setdefault('eval_datasets', [])
    config.setdefault('eval_gradient_accumulation_steps', 1)
    config.setdefault('eval_every_n_steps', None)
    config.setdefault('eval_every_n_epochs', None)
    config.setdefault('eval_before_first_step', True)


def get_most_recent_run_dir(output_dir):
    return list(sorted(glob.glob(os.path.join(output_dir, '*'))))[-1]


def print_model_info(model):
    if not is_main_process():
        return
    print(model)
    for name, module in model.named_modules():
        print(f'{type(module)}: {name}')
        for pname, p in module.named_parameters(recurse=False):
            print(pname)
            print(p.dtype)
            print(p.device)
            print(p.requires_grad)
            print()


def evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=None):
    eval_dataloader.set_eval_quantile(quantile)
    orig_micro_batches = model_engine.micro_batches
    model_engine.micro_batches = eval_gradient_accumulation_steps
    iterator = iter(eval_dataloader)
    total_loss = 0
    count = 0
    while True:
        model_engine.reset_activation_shape()
        loss = model_engine.eval_batch(iterator).item()
        eval_dataloader.sync_epoch()
        if pbar:
            pbar.update(1)
        total_loss += loss
        count += 1
        if eval_dataloader.epoch == 2:
            break

    eval_dataloader.reset()
    model_engine.micro_batches = orig_micro_batches
    return total_loss / count


def _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps):
    pbar_total = 0
    for eval_dataloader in eval_dataloaders.values():
        pbar_total += len(eval_dataloader) * len(TIMESTEP_QUANTILES_FOR_EVAL) // eval_gradient_accumulation_steps
    if is_main_process():
        print('Running eval')
        pbar = tqdm(total=pbar_total)
    else:
        pbar = None

    start = time.time()
    for name, eval_dataloader in eval_dataloaders.items():
        losses = []
        for quantile in TIMESTEP_QUANTILES_FOR_EVAL:
            loss = evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=pbar)
            losses.append(loss)
            if is_main_process():
                tb_writer.add_scalar(f'{name}/loss_quantile_{quantile:.2f}', loss, step)
        avg_loss = sum(losses) / len(losses)
        if is_main_process():
            tb_writer.add_scalar(f'{name}/loss', avg_loss, step)

    duration = time.time() - start
    if is_main_process():
        tb_writer.add_scalar('eval/eval_time_sec', duration, step)
        pbar.close()


def evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps):
    if len(eval_dataloaders) == 0:
        return
    with torch.no_grad(), isolate_rng():
        seed = get_rank()
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps)


def setup_checkpoint_signal():
    """Set up signal handler for checkpoint saving"""
    if not is_main_process():
        return
        
    def trigger_save(signum, frame):
        print("\nDetected save signal (SIGUSR1), saving checkpoint...")
        # Set a flag due to signal handler limitations
        setup_checkpoint_signal.should_save = True
    
    signal.signal(signal.SIGUSR1, trigger_save)
    setup_checkpoint_signal.should_save = False


def setup_interrupt_handler(saver):
    """Set up Ctrl+C interrupt handler"""
    if not is_main_process():
        return
        
    def signal_handler(signum, frame):
        if setup_interrupt_handler.triggered:  # Check if already triggered
            return
        setup_interrupt_handler.triggered = True  # Set flag
            
        print("\nDetected interrupt signal, saving checkpoint...")
        try:
            saver.save_checkpoint(setup_interrupt_handler.current_step)
            print("Checkpoint saved successfully!")
        except Exception as e:
            print(f"Error occurred while saving checkpoint: {e}")
        finally:
            print("Safely exiting training...")
            # Notify other processes to exit
            if dist.is_initialized():
                dist.barrier()
            sys.exit(0)
    
    # Set up handlers for SIGINT (Ctrl+C) and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    setup_interrupt_handler.current_step = 0
    setup_interrupt_handler.triggered = False  # Add flag to prevent duplicate triggers


if __name__ == '__main__':
    apply_patches()

    # needed for broadcasting Queue in dataset.py
    mp.current_process().authkey = b'afsaskgfdjh4'

    with open(args.config) as f:
        # Inline TOML tables are not pickleable, which messes up the multiprocessing dataset stuff. This is a workaround.
        config = json.loads(json.dumps(toml.load(f)))

    set_config_defaults(config)
    common.AUTOCAST_DTYPE = config['model']['dtype']

    resume_from_checkpoint = (
        args.resume_from_checkpoint if args.resume_from_checkpoint is not None
        else config.get('resume_from_checkpoint', False)
    )
    regenerate_cache = (
        args.regenerate_cache if args.regenerate_cache is not None
        else config.get('regenerate_cache', False)
    )

    deepspeed.init_distributed()
    # needed for broadcasting Queue in dataset.py (because we haven't called deepspeed.initialize() yet?)
    torch.cuda.set_device(dist.get_rank())

    model_type = config['model']['type']

    if model_type == 'flux':
        from models import flux
        model = flux.FluxPipeline(config)
    elif model_type == 'ltx-video':
        from models import ltx_video
        model = ltx_video.LTXVideoPipeline(config)
    elif model_type == 'hunyuan-video':
        from models import hunyuan_video
        model = hunyuan_video.HunyuanVideoPipeline(config)
    elif model_type == 'sdxl':
        from models import sdxl
        model = sdxl.SDXLPipeline(config)
    elif model_type == 'cosmos':
        from models import cosmos
        model = cosmos.CosmosPipeline(config)
    elif model_type == 'lumina_2':
        from models import lumina_2
        model = lumina_2.Lumina2Pipeline(config)
    elif model_type == 'wan':
        from models import wan
        model = wan.WanPipeline(config)
    elif model_type == 'chroma':
        from models import chroma
        model = chroma.ChromaPipeline(config)
    else:
        raise NotImplementedError(f'Model type {model_type} is not implemented')

    # import sys, PIL
    # test_image = sys.argv[1]
    # with torch.no_grad():
    #     vae = model.get_vae().to('cuda')
    #     latents = dataset.encode_pil_to_latents(PIL.Image.open(test_image), vae)
    #     pil_image = dataset.decode_latents_to_pil(latents, vae)
    #     pil_image.save('test.jpg')
    # quit()

    with open(config['dataset']) as f:
        dataset_config = toml.load(f)
    gradient_release = config['optimizer'].get('gradient_release', False)
    ds_config = {
        'train_micro_batch_size_per_gpu': config.get('micro_batch_size_per_gpu', 1),
        'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 1),
        # Can't do gradient clipping with gradient release, since there are no grads at the end of the step anymore.
        'gradient_clipping': 0. if gradient_release else config.get('gradient_clipping', 1.0),
        'steps_per_print': config.get('steps_per_print', 1),
    }
    caching_batch_size = config.get('caching_batch_size', 1)
    dataset_manager = dataset_util.DatasetManager(model, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)

    train_data = dataset_util.Dataset(dataset_config, model, skip_dataset_validation=args.i_know_what_i_am_doing)
    dataset_manager.register(train_data)

    eval_data_map = {}
    for i, eval_dataset in enumerate(config['eval_datasets']):
        if type(eval_dataset) == str:
            name = f'eval{i}'
            config_path = eval_dataset
        else:
            name = eval_dataset['name']
            config_path = eval_dataset['config']
        with open(config_path) as f:
            eval_dataset_config = toml.load(f)
        eval_data_map[name] = dataset_util.Dataset(eval_dataset_config, model, skip_dataset_validation=args.i_know_what_i_am_doing)
        dataset_manager.register(eval_data_map[name])

    # For testing

    # import imageio
    # from pathlib import Path
    # import torch.nn.functional as F
    # dataset_manager.cache(unload_models=False)
    # output_dir = Path('/home/anon/tmp')
    # train_data.post_init(
    #     0,
    #     1,
    #     1,
    #     1,
    # )
    # vae = model.vae
    # vae.model.to('cuda')
    # count = 1
    # for item in train_data:
    #     latents = item['latents'].to('cuda')
    #     h, w = latents.shape[-2:]
    #     mask = item['mask'].to('cuda')
    #     caption = item['caption'][0]
    #     mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
    #     mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
    #     mask = mask.unsqueeze(2)  # make mask same number of dims as target
    #     latents = latents * mask.to(latents.device)
    #     video = vae.model.decode(latents, vae.scale).float().clamp_(-1, 1).squeeze(0)
    #     video = torch.permute(video, (1, 2, 3, 0))
    #     video = (video + 1) / 2
    #     video = (video * 255).type(torch.uint8).cpu()
    #     imageio.v3.imwrite(output_dir / f'{count}.mp4', video, fps=16)
    #     with open(output_dir / f'{count}.txt', 'w') as f:
    #         f.write(caption)
    #     if count >= 10:
    #         break
    #     count += 1
    # quit()

    dataset_manager.cache()
    if args.cache_only:
        quit()

    model.load_diffusion_model()

    if adapter_config := config.get('adapter', None):
        init_from_existing = adapter_config.get('init_from_existing', None)
        # SDXL is special. LoRAs are saved in Kohya sd-scripts format, which is very difficult to load the state_dict into
        # an adapter we already configured. So, for SDXL, load_adapter_weights will use a Diffusers method to create and
        # load the adapter all at once from the sd-scripts format safetensors file.
        if not (init_from_existing and model_type == 'sdxl'):
            model.configure_adapter(adapter_config)
        is_adapter = True
        if init_from_existing:
            model.load_adapter_weights(init_from_existing)
    else:
        is_adapter = False

    # if this is a new run, create a new dir for it
    if not resume_from_checkpoint and is_main_process():
        run_dir = os.path.join(config['output_dir'], datetime.now(timezone.utc).strftime('%Y%m%d_%H-%M-%S'))
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy(args.config, run_dir)
    # wait for all processes then get the most recent dir (may have just been created)
    dist.barrier()
    if resume_from_checkpoint is True:  # No specific folder provided, use most recent
        run_dir = get_most_recent_run_dir(config['output_dir'])
    elif isinstance(resume_from_checkpoint, str):  # Specific folder provided
        run_dir = os.path.join(config['output_dir'], resume_from_checkpoint)
        if not os.path.exists(run_dir):
            raise ValueError(f"Checkpoint directory {run_dir} does not exist")
    else:  # Not resuming, use most recent (newly created) dir
        run_dir = get_most_recent_run_dir(config['output_dir'])

    # Block swapping
    if blocks_to_swap := config.get('blocks_to_swap', 0):
        assert config['pipeline_stages'] == 1, 'Block swapping only works with pipeline_stages=1'
        assert 'adapter' in config, 'Block swapping only works when training LoRA'
        # Don't automatically move to GPU, we'll do that ourselves.
        def to(self, *args, **kwargs):
            pass
        deepspeed.pipe.PipelineModule.to = to
        model.enable_block_swap(blocks_to_swap)

    layers = model.to_layers()
    additional_pipeline_module_kwargs = {}
    if config['activation_checkpointing']:
        # TODO: block swapping doesn't work with Deepspeed non-reentrant checkpoint, but PyTorch native one is fine. Some
        # weights end up on CPU where they shouldn't. Why? Are we giving anything up by not using the Deepspeed implementation?
        #checkpoint_func = deepspeed.checkpointing.non_reentrant_checkpoint
        from functools import partial
        checkpoint_func = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
        additional_pipeline_module_kwargs.update({
            'activation_checkpoint_interval': 1,
            'checkpointable_layers': model.checkpointable_layers,
            'activation_checkpoint_func': checkpoint_func,
        })
    pipeline_model = deepspeed.pipe.PipelineModule(
        layers=layers,
        num_stages=config['pipeline_stages'],
        partition_method=config.get('partition_method', 'parameters'),
        loss_fn=model.get_loss_fn(),
        **additional_pipeline_module_kwargs
    )
    parameters_to_train = [p for p in pipeline_model.parameters() if p.requires_grad]

    def get_optimizer(model_parameters):
        optim_config = config['optimizer']
        optim_type = optim_config['type'].lower()

        args = []
        kwargs = {k: v for k, v in optim_config.items() if k not in ['type', 'gradient_release']}

        if optim_type == 'adamw':
            # TODO: fix this. I'm getting "fatal error: cuda_runtime.h: No such file or directory"
            # when Deepspeed tries to build the fused Adam extension.
            # klass = deepspeed.ops.adam.FusedAdam
            klass = torch.optim.AdamW
        elif optim_type == 'adamw8bit':
            import bitsandbytes
            klass = bitsandbytes.optim.AdamW8bit
        elif optim_type == 'adamw_optimi':
            import optimi
            klass = optimi.AdamW
        elif optim_type == 'stableadamw':
            import optimi
            klass = optimi.StableAdamW
        elif optim_type == 'sgd':
            klass = torch.optim.SGD
        elif optim_type == 'adamw8bitkahan':
            from optimizers import adamw_8bit
            klass = adamw_8bit.AdamW8bitKahan
        elif optim_type == 'offload':
            from torchao.prototype.low_bit_optim import CPUOffloadOptimizer
            klass = CPUOffloadOptimizer
            args.append(torch.optim.AdamW)
            kwargs['fused'] = True
        else:
            raise NotImplementedError(optim_type)

        if optim_config.get('gradient_release', False):
            # Prevent deepspeed from logging every single param group lr
            def _report_progress(self, step):
                lr = self.get_lr()
                mom = self.get_mom()
                deepspeed.utils.logging.log_dist(f"step={step}, skipped={self.skipped_steps}, lr={lr[0]}, mom={mom[0]}", ranks=[0])
            deepspeed.runtime.engine.DeepSpeedEngine._report_progress = _report_progress

            # Deepspeed executes all the code to reduce grads across data parallel ranks even if the DP world size is 1.
            # As part of this, any grads that are None are set to zeros. We're doing gradient release to save memory,
            # so we have to avoid this.
            def _exec_reduce_grads(self):
                assert self.mpu.get_data_parallel_world_size() == 1, 'When using gradient release, data parallel world size must be 1. Make sure pipeline_stages = num_gpus.'
                return
            deepspeed.runtime.pipe.engine.PipelineEngine._INSTRUCTION_MAP[deepspeed.runtime.pipe.schedule.ReduceGrads] = _exec_reduce_grads

            # When pipelining multiple forward and backward passes, normally updating the parameter in-place causes an error when calling
            # backward() on future micro-batches. But we can modify .data directly so the autograd engine doesn't detect in-place modifications.
            # TODO: this is unbelievably hacky and not mathematically sound, I'm just seeing if it works at all.
            def add_(self, *args, **kwargs):
                self.data.add_(*args, **kwargs)
            for p in model_parameters:
                p.add_ = add_.__get__(p)

            if 'foreach' in inspect.signature(klass).parameters:
                kwargs['foreach'] = False

            # We're doing an optimizer step for each micro-batch. Scale momentum and EMA betas so that the contribution
            # decays at the same rate it would if we were doing one step per batch like normal.
            # Reference: https://alexeytochin.github.io/posts/batch_size_vs_momentum/batch_size_vs_momentum.html
            gas = ds_config['gradient_accumulation_steps']
            if 'betas' in kwargs:
                for i in range(len(kwargs['betas'])):
                    kwargs['betas'][i] = kwargs['betas'][i] ** (1/gas)
            if 'momentum' in kwargs:
                kwargs['momentum'] = kwargs['momentum'] ** (1/gas)

            optimizer_dict = {}
            for pg in model.get_param_groups(model_parameters):
                param_kwargs = kwargs.copy()
                if isinstance(pg, dict):
                    # param group
                    for p in pg['params']:
                        param_kwargs['lr'] = pg['lr']
                        optimizer_dict[p] = klass([p], **param_kwargs)
                else:
                    # param
                    optimizer_dict[pg] = klass([pg], **param_kwargs)

            def optimizer_hook(p):
                optimizer_dict[p].step()
                optimizer_dict[p].zero_grad()

            for p in model_parameters:
                p.register_post_accumulate_grad_hook(optimizer_hook)

            from optimizers import gradient_release
            return gradient_release.GradientReleaseOptimizerWrapper(list(optimizer_dict.values()))
        else:
            model_parameters = model.get_param_groups(model_parameters)
            return klass(model_parameters, *args, **kwargs)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=pipeline_model,
        model_parameters=parameters_to_train,
        optimizer=get_optimizer,
        config=ds_config,
    )
    if model_engine.is_pipe_parallel:
        grid = model_engine.grid
        model_engine.first_last_stage_group = dist.new_group(ranks=[grid.pp_group[0], grid.pp_group[-1]])
    model.model_engine = model_engine

    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    if config['warmup_steps'] > 0:
        warmup_steps = config['warmup_steps']
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/warmup_steps, total_iters=warmup_steps)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, lr_scheduler], milestones=[warmup_steps])
    model_engine.lr_scheduler = lr_scheduler

    train_data.post_init(
        model_engine.grid.get_data_parallel_rank(),
        model_engine.grid.get_data_parallel_world_size(),
        model_engine.train_micro_batch_size_per_gpu(),
        model_engine.gradient_accumulation_steps(),
    )
    for eval_data in eval_data_map.values():
        eval_data.post_init(
            model_engine.grid.get_data_parallel_rank(),
            model_engine.grid.get_data_parallel_world_size(),
            config.get('eval_micro_batch_size_per_gpu', model_engine.train_micro_batch_size_per_gpu()),
            config['eval_gradient_accumulation_steps'],
        )

    # Might be useful because we set things in fp16 / bf16 without explicitly enabling Deepspeed fp16 mode.
    # Unsure if really needed.
    communication_data_type = config['lora']['dtype'] if 'lora' in config else config['model']['dtype']
    model_engine.communication_data_type = communication_data_type

    train_dataloader = dataset_util.PipelineDataLoader(train_data, model_engine, model_engine.gradient_accumulation_steps(), model)

    step = 1
    # make sure to do this before calling model_engine.set_dataloader(), as that method creates an iterator
    # which starts creating dataloader internal state
    if resume_from_checkpoint:
        load_path, client_state = model_engine.load_checkpoint(
            run_dir,
            load_module_strict=False,
            load_lr_scheduler_states='force_constant_lr' not in config,
        )
        dist.barrier()  # just so the print below doesn't get swamped
        assert load_path is not None
        train_dataloader.load_state_dict(client_state['custom_loader'])
        step = client_state['step'] + 1
        del client_state
        if is_main_process():
            print(f'Resuming training from checkpoint. Resuming at epoch: {train_dataloader.epoch}, step: {step}')

    if 'force_constant_lr' in config:
        model_engine.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        for pg in optimizer.param_groups:
            pg['lr'] = config['force_constant_lr']

    model_engine.set_dataloader(train_dataloader)
    steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()
    model_engine.total_steps = steps_per_epoch * config['epochs']

    eval_dataloaders = {
        # Set num_dataloader_workers=0 so dataset iteration is completely deterministic.
        # We want the exact same noise for each image, each time, for a stable validation loss.
        name: dataset_util.PipelineDataLoader(eval_data, model_engine, config['eval_gradient_accumulation_steps'], model, num_dataloader_workers=0)
        for name, eval_data in eval_data_map.items()
    }

    epoch = train_dataloader.epoch
    tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None
    saver = utils.saver.Saver(args, config, is_adapter, run_dir, model, train_dataloader, model_engine, pipeline_model)

    setup_checkpoint_signal()

    setup_interrupt_handler(saver)
    
    if config['eval_before_first_step'] and not resume_from_checkpoint:
        evaluate(model_engine, eval_dataloaders, tb_writer, 0, config['eval_gradient_accumulation_steps'])

    # TODO: this is state we need to save and resume when resuming from checkpoint. It only affects logging.
    epoch_loss = 0
    num_steps = 0
    while True:
        setup_interrupt_handler.current_step = step  # Update current step
        model_engine.reset_activation_shape()
        loss = model_engine.train_batch().item()
        epoch_loss += loss
        num_steps += 1
        train_dataloader.sync_epoch()

        # Check if checkpoint save is needed
        if is_main_process() and setup_checkpoint_signal.should_save:
            saver.save_checkpoint(step)
            setup_checkpoint_signal.should_save = False
            print("Checkpoint saved successfully!")

        new_epoch, checkpointed, saved = saver.process_epoch(epoch, step)
        finished_epoch = True if new_epoch != epoch else False

        if is_main_process() and step % config['logging_steps'] == 0:
            tb_writer.add_scalar(f'train/loss', loss, step)

        if (config['eval_every_n_steps'] and step % config['eval_every_n_steps'] == 0) or (finished_epoch and config['eval_every_n_epochs'] and epoch % config['eval_every_n_epochs'] == 0):
            evaluate(model_engine, eval_dataloaders, tb_writer, step, config['eval_gradient_accumulation_steps'])

        if finished_epoch:
            if is_main_process():
                tb_writer.add_scalar(f'train/epoch_loss', epoch_loss/num_steps, epoch)
            epoch_loss = 0
            num_steps = 0
            epoch = new_epoch
            if epoch is None:
                break

        saver.process_step(step)
        step += 1

    # Save final training state checkpoint and model, unless we just saved them.
    if not checkpointed:
        saver.save_checkpoint(step)
    if not saved:
        saver.save_model(f'epoch{epoch}')

    if is_main_process():
        print('TRAINING COMPLETE!')
