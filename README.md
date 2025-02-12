# diffusion-pipe
A pipeline parallel training script for diffusion models.

Currently supports SDXL, Flux, LTX-Video, HunyuanVideo, Cosmos, Lumina Image 2.0.

**Work in progress and highly experimental.** It is unstable and not well tested. Things might not work right.

## Features
- Pipeline parallelism, for training models larger than can fit on a single GPU
- Full fine tune support for:
    - Flux, Lumina Image 2.0
- LoRA support for:
    - SDXL, Flux, LTX-Video, HunyuanVideo, Cosmos, Lumina Image 2.0
- Useful metrics logged to Tensorboard
- Compute metrics on a held-out eval set, for measuring generalization
- Training state checkpointing and resuming from checkpoint
- Efficient multi-process, multi-GPU pre-caching of latents and text embeddings
- Easily add support for new models by implementing a single subclass

## Recent changes
- 2025-02-10
  - Fixed a bug in video training causing width and height to be flipped when bucketing by aspect ratio. This would cause videos to be over-cropped. Image-only training is unaffected. If you have been training on videos, please pull the latest code, and regenerate the cache using the --regenerate_cache flag, or delete the cache dir inside the dataset directories.
- 2025-02-09
  - Add support for Lumina Image 2.0. Both LoRA and full fine tuning are supported.
- 2025-02-08
  - Support fp8 transformer for Flux LoRAs. You can now train LoRAs with a single 24GB GPU.
  - Add tentative support for Cosmos. Cosmos doesn't fine tune well compared to HunyuanVideo, and will likely not be actively supported going forward.
- 2025-01-20
  - Properly support training Flex.1-alpha.
  - Make sure to set ```bypass_guidance_embedding=true``` in the model config. You can look at the example config file.
- 2025-01-17
  - For HunyuanVideo VAE when loaded via the ```vae_path``` option, fixed incorrect tiling sample size. The training loss is now moderately lower overall. Quality of trained LoRAs should be improved, but the improvement is likely minor.
  - You should update any cached latents made before this change. Delete the cache directory inside the dataset directories, or run the training script with the ```--regenerate_cache``` command line option.
- 2025-01-13
  - Basic SDXL support. LoRA only. Many options present in other training scripts are not implemented. If you want more features added, PRs are welcome.

## Windows support
It will be difficult or impossible to make training work on native Windows. This is because Deepspeed only has [partial Windows support](https://github.com/microsoft/DeepSpeed/blob/master/blogs/windows/08-2024/README.md). Deepspeed is a hard requirement because the entire training script is built around Deepspeed pipeline parallelism. However, it will work on Windows Subsystem for Linux, specifically WSL 2. If you must use Windows I recommend trying WSL 2.

## Installing
Clone the repository:
```
git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe
```

If you alread cloned it and forgot to do --recurse-submodules:
```
git submodule init
git submodule update
```

Install Miniconda: https://docs.anaconda.com/miniconda/

Create the environment:
```
conda create -n diffusion-pipe python=3.12
conda activate diffusion-pipe
```

Install nvcc: https://anaconda.org/nvidia/cuda-nvcc. Probably try to make it match the CUDA version that was installed on your system with PyTorch.

Install the dependencies:
```
pip install -r requirements.txt
```

### Cosmos requirements
NVIDIA Cosmos additionally requires TransformerEngine. This dependency isn't in the requirements file. Installing this was a bit tricky for me. On Ubuntu 24.04, I had to install GCC version 12 (13 is the default in the package manager), and make sure GCC 12 and CUDNN were set during installation like this:
```
CC=/usr/bin/gcc-12 CUDNN_PATH=/home/anon/miniconda3/envs/diffusion-pipe/lib/python3.12/site-packages/nvidia/cudnn pip install transformer_engine[pytorch]
```

## Dataset preparation
A dataset consists of one or more directories containing image or video files, and corresponding captions. You can mix images and videos in the same directory, but it's probably a good idea to separate them in case you need to specify certain settings on a per-directory basis. Caption files should be .txt files with the same base name as the corresponding media file, e.g. image1.png should have caption file image1.txt in the same directory. If a media file doesn't have a matching caption file, a warning is printed, but training will proceed with an empty caption.

For images, any image format that can be loaded by Pillow should work. For videos, any format that can be loaded by ImageIO should work. Note that this means **WebP videos are not supported**, because ImageIO can't load multi-frame WebPs.

## Supported models
See the [supported models doc](./docs/supported_models.md) for more information on how to configure each model, the options it supports, and the format of the saved LoRAs.

## Training
**Start by reading through the config files in the examples directory.** Almost everything is commented, explaining what each setting does.

Once you've familiarized yourself with the config file format, go ahead and make a copy and edit to your liking. At minimum, change all the paths to conform to your setup, including the paths in the dataset config file.

Launch training like this:
```
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config examples/hunyuan_video.toml
```
RTX 4000 series needs those 2 environment variables set. Other GPUs may not need them. You can try without them, Deepspeed will complain if it's wrong.

If you enabled checkpointing, you can resume training from the latest checkpoint by simply re-running the exact same command but with the `--resume_from_checkpoint` flag. You can also specify a specific checkpoint folder name after the flag to resume from that particular checkpoint (e.g. `--resume_from_checkpoint "20250212_07-06-40"`). This option is particularly useful if you have run multiple training sessions with different datasets and want to resume from a specific training folder.

## Output files
A new directory will be created in ```output_dir``` for each training run. This contains the checkpoints, saved models, and Tensorboard metrics. Saved models/LoRAs will be in directories named like epoch1, epoch2, etc. Deepspeed checkpoints are in directories named like global_step1234. These checkpoints contain all training state, including weights, optimizer, and dataloader state, but can't be used directly for inference. The saved model directory will have the safetensors weights, PEFT adapter config JSON, as well as the diffusion-pipe config file for easier tracking of training run settings.

## Parallelism
This code uses hybrid data- and pipeline-parallelism. Set the ```--num_gpus``` flag appropriately for your setup. Set ```pipeline_stages``` in the config file to control the degree of pipeline parallelism. Then the data parallelism degree will automatically be set to use all GPUs (number of GPUs must be divisible by pipeline_stages). For example, with 4 GPUs and pipeline_stages=2, you will run two instances of the model, each divided across two GPUs.

## Pre-caching
Latents and text embeddings are cached to disk before training happens. This way, the VAE and text encoders don't need to be kept loaded during training. The Huggingface Datasets library is used for all the caching. Cache files are reused between training runs if they exist. All cache files are written into a directory named "cache" inside each dataset directory.

This caching also means that training LoRAs for text encoders is not currently supported.

Two flags are relevant for caching. ```--cache_only``` does the caching flow, then exits without training anything. ```--regenerate_cache``` forces cache regeneration. If you edit the dataset in-place (like changing a caption), you need to force regenerate the cache (or delete the cache dir) for the changes to be picked up.

## Extra
You can check out my [qlora-pipe](https://github.com/tdrussell/qlora-pipe) project, which is basically the same thing as this but for LLMs.
