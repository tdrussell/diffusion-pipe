from pathlib import Path
import os
import shutil
import time
import sys

import torch
from deepspeed import comm as dist
from deepspeed.utils.logging import logger
from huggingface_hub import HfApi

from utils.common import is_main_process


def convert_state_dict_dtype(state_dict, dtype):
    for key, v in state_dict.items():
        state_dict[key] = v.to(device='cpu', dtype=dtype)


last_checkpoint_time = None
def need_to_checkpoint(config, epoch=None):
    global last_checkpoint_time

    if epoch is not None:
        if 'checkpoint_every_n_epochs' in config and epoch % config['checkpoint_every_n_epochs'] == 0:
            last_checkpoint_time = time.time()
            return True
        else:
            return False

    if 'checkpoint_every_n_minutes' not in config:
        return False

    checkpoint = False
    # rank 0 tracks if we need to checkpoint, broadcasts to everyone else
    if is_main_process():
        current_time = time.time()
        if last_checkpoint_time is None:
            last_checkpoint_time = current_time
        elif (current_time - last_checkpoint_time) / 60 > config['checkpoint_every_n_minutes']:
            checkpoint = True
            last_checkpoint_time = current_time
    result = [checkpoint]
    torch.distributed.broadcast_object_list(result, src=0)
    return result[0]


class Saver:
    def __init__(self, args, config, is_adapter, save_root, model, train_dataloader, model_engine, pipeline_model):
        self.args = args
        self.config = config
        self.is_adapter = is_adapter
        self.save_root = Path(save_root)
        self.model = model
        self.train_dataloader = train_dataloader
        self.model_engine = model_engine
        self.pipeline_model = pipeline_model

    def save_adapter(self, name, epoch=None):
        dp_id = self.model_engine.grid.get_data_parallel_rank()
        stage_id = self.model_engine.grid.get_pipe_parallel_rank()
        save_dir = self.save_root / name
        tmp_dir = save_dir / 'tmp'
        if dp_id == 0 and stage_id == 0:
            os.makedirs(tmp_dir, exist_ok=False)
        dist.barrier()
        if dp_id == 0:
            partial_state_dict = {}
            for name, p in self.pipeline_model.named_parameters():
                if p.requires_grad:
                    if not hasattr(p, 'original_name'):
                        logger.warning(f'WARNING: parameter {name} requires_grad but does not have original_name. Not saving it.')
                        continue
                    # TODO: maybe this needs to change if we ever have non-lora adapters?
                    partial_state_dict[p.original_name.replace('.default', '').replace('.modules_to_save', '')] = p.detach()
                    if 'save_dtype' in self.config:
                        convert_state_dict_dtype(partial_state_dict, self.config['save_dtype'])
            torch.save(partial_state_dict, tmp_dir / f'state_dict_{stage_id}.bin')
        dist.barrier()
        if dp_id == 0 and stage_id == 0:
            state_dict = {}
            for path in tmp_dir.glob('*.bin'):
                state_dict.update(torch.load(path, weights_only=True, map_location='cpu'))
            self.model.save_adapter(save_dir, state_dict)
            shutil.copy(self.args.config, save_dir)
            shutil.rmtree(tmp_dir)
            
            # Upload to Hugging Face if configured and epoch condition is met
            if 'huggingface_repo' in self.config and self.config['huggingface_repo']:
                upload_hf_from_epoch = self.config.get('upload_hf_from_epoch', 0)
                should_upload = epoch is None or epoch >= upload_hf_from_epoch
                
                if should_upload:
                    try:
                        if is_main_process():
                            print(f"Uploading adapter '{name}' to Hugging Face repo: {self.config['huggingface_repo']}")
                        
                        api = HfApi()
                        api.create_repo(
                            repo_id=self.config['huggingface_repo'],
                            private=True,
                            exist_ok=True,
                            repo_type="model"
                        )
                        api.upload_folder(
                            folder_path=str(save_dir),
                            repo_id=self.config['huggingface_repo'],
                            path_in_repo=f"epoch{epoch}", # to be consistent
                            repo_type="model"
                        )
                        
                        if is_main_process():
                            print(f"Successfully uploaded adapter '{name}' to Hugging Face")
                            
                    except Exception as e:
                        if is_main_process():
                            logger.error(f"Failed to upload adapter to Hugging Face: {str(e)}")
                            print(f"Warning: Failed to upload to Hugging Face: {str(e)}")
                else:
                    if is_main_process():
                        print(f"Skipping Hugging Face upload for epoch {epoch} (upload_hf_from_epoch={upload_hf_from_epoch})")

    def save_full_model(self, name, epoch=None, max_shard_size='5GB'):
        dp_id = self.model_engine.grid.get_data_parallel_rank()
        stage_id = self.model_engine.grid.get_pipe_parallel_rank()
        save_dir = self.save_root / name
        tmp_dir = save_dir / 'tmp'
        if dp_id == 0 and stage_id == 0:
            os.makedirs(tmp_dir, exist_ok=False)
        dist.barrier()
        if dp_id == 0:
            # With BF16_Optimizer, we get pickle errors unless we do p.detach(). I have no idea why.
            partial_state_dict = {p.original_name: p.detach() for p in self.pipeline_model.parameters() if hasattr(p, 'original_name')}
            if 'save_dtype' in self.config:
                convert_state_dict_dtype(partial_state_dict, self.config['save_dtype'])
            torch.save(partial_state_dict, tmp_dir / f'state_dict_{stage_id}.bin')
        dist.barrier()
        if dp_id == 0 and stage_id == 0:
            state_dict = {}
            for path in tmp_dir.glob('*.bin'):
                state_dict.update(torch.load(path, map_location='cpu', weights_only=True))
            self.model.save_model(save_dir, state_dict)
            shutil.copy(self.args.config, save_dir)
            shutil.rmtree(tmp_dir)

    def save_model(self, name, epoch=None):
        if is_main_process():
            print(f'Saving model to directory {name}')
        if self.is_adapter:
            self.save_adapter(name, epoch)
        else:
            self.save_full_model(name, epoch)

    def save_checkpoint(self, step, examples):
        self.model_engine.save_checkpoint(
            self.save_root,
            client_state={
                'step': step,
                'examples': examples,
                'custom_loader': self.train_dataloader.state_dict(),
            },
            save_latest=True,
            exclude_frozen_parameters=True
        )

    def process_epoch(self, epoch, step, examples):
        checkpointed, saved = False, False
        if self.train_dataloader.epoch != epoch:
            if need_to_checkpoint(self.config, epoch):
                self.save_checkpoint(step, examples)
                checkpointed = True
            if 'save_every_n_epochs' in self.config and epoch % self.config['save_every_n_epochs'] == 0:
                self.save_model(f'epoch{epoch}', epoch)
                saved = True
            epoch = self.train_dataloader.epoch
            if epoch > self.config['epochs']:
                return None, checkpointed, saved
            if is_main_process():
                print(f'Started new epoch: {epoch}')
        return epoch, checkpointed, saved

    def process_step(self, step, examples):
        checkpointed, saved = False, False
        # Look at some simple "signal files" the user can write to save and optionally quit manually
        should_manually_save = False
        should_manually_quit = False
        save_signal_file = self.save_root / 'save'
        save_quit_signal_file = self.save_root / 'save_quit'
        if save_signal_file.exists() and save_signal_file.is_file():
            should_manually_save = True
            dist.barrier()
            if is_main_process():
                os.remove(save_signal_file)
        elif save_quit_signal_file.exists() and save_quit_signal_file.is_file():
            should_manually_save = True
            should_manually_quit = True
            dist.barrier()
            if is_main_process():
                os.remove(save_quit_signal_file)

        if 'save_every_n_steps' in self.config and step % self.config['save_every_n_steps'] == 0:
            self.save_model(f'step{step}')
            saved = True

        if need_to_checkpoint(self.config) or should_manually_save:
            self.save_checkpoint(step, examples)
            checkpointed = True

        if should_manually_quit:
            print('Manually quitting')
            sys.exit()

        return checkpointed, saved
