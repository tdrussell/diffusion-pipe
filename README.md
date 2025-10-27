# diffusion-pipe
A pipeline parallel training script for diffusion models.

Models supported: SDXL, Flux, LTX-Video, HunyuanVideo (t2v), Cosmos, Lumina Image 2.0, Wan2.1 (t2v and i2v), Chroma, HiDream, Stable Diffusion 3, Cosmos-Predict2, OmniGen2, Flux Kontext, Wan2.2, Qwen-Image, Qwen-Image-Edit, HunyuanImage-2.1, AuraFlow.

## Features
- Pipeline parallelism, for training models larger than can fit on a single GPU
- Useful metrics logged to Tensorboard
- Compute metrics on a held-out eval set, for measuring generalization
- Training state checkpointing and resuming from checkpoint
- Efficient multi-process, multi-GPU pre-caching of latents and text embeddings
- Seemlessly supports both image and video models in a unified way
- Easily add new models by implementing a single subclass

## Recent changes
- 2025-10-27
  - Add AuraFlow / Pony-V7 support.
  - Remove multiple_overlapping setting for video clip mode. This didn't even work right (it would always use a single clip from the video). And future changes I have planned won't work with it either.
  - Use float16 for cached files. Previously it was always float32 on disk. This shouldn't impact quality and will cut disk usage of cached files in half.
- 2025-09-13
  - Support HunyuanImage-2.1. This adds a new submodule; make sure to run ```git submodule update``` after pull.
- 2025-08-23
  - Support Qwen-Image-Edit. Make sure to update dependencies. You will need the latest Diffusers.
- 2025-08-07
  - Fix Flux training error caused by a breaking change in Diffusers. Make sure to update requirements.
- 2025-08-06
  - Support Qwen-Image.
  - Slight speed improvement to Automagic optimizer.
- 2025-07-29
  - Support Wan2.2.
    - The 5B is tested and fully validated on t2i training.
    - All other models and modes (A14B, i2v, timestep ranges) are tested to confirm they run and that the loss looks reasonable, but proper learning hasn't been validated yet.
- 2025-07-14
  - Merge dev branch into main. Lots of changes that aren't relevant for most users. Recommended to use ```--regenerate_cache``` (or delete the cache folders) after update.
      - If something breaks, please raise an issue and use the last known good commit in the meanwhile: ```git checkout 6940992455bb3bb2b88cd6e6c9463e7469929a70```
  - Loading speed and throughput improvements for dataset caching. Will only make a big difference for very large datasets.
  - Various dataset features and improvements to support large-scale training. Still testing, not documented yet.
  - Add ```--trust_cache``` flag that will blindly load cached metadata files if they exist, without checking if any files changed. Can make dataset loading faster for large datasets, but you must be sure nothing in the dataset has changed since last caching. You probably don't have a large enough dataset for this to be useful.
  - Add torch compile option that can speed up models. Not tested with all models.
  - Add support for edit datasets and Flux Kontext. See supported models doc for details.
- 2025-06-27
  - OmniGen2 LoRA training is supported, but only via standard t2i training.
  - Refactored Cosmos-Predict2 implementation to align with other rectified flow models. The only effective change is that the loss weighting is slightly different.

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

Install PyTorch first. It is not listed in the requirements file, because certain GPUs sometimes need different versions of PyTorch or CUDA, and you might have to find a combination that works for your hardware. As of this writing (October 26, 2025), PyTorch 2.9.0 with CUDA 12.8 works on my 4090, and is compatible with the current latest flash-attn 2.8.3:
```
pip install torch torchvision
```

Install nvcc: https://anaconda.org/nvidia/cuda-nvcc. Probably try to make it match the CUDA version of PyTorch.

Install the rest of the dependencies:
```
pip install -r requirements.txt
```

### Cosmos requirements
NVIDIA Cosmos (the original Cosmos video model, not Cosmos-Predict2) additionally requires TransformerEngine.

This dependency isn't in the requirements file. You probably need to set some environment variables for it to install correctly. The following command worked for me:
```
C_INCLUDE_PATH=/home/anon/miniconda3/envs/diffusion-pipe/lib/python3.12/site-packages/nvidia/cudnn/include:$C_INCLUDE_PATH CPLUS_INCLUDE_PATH=/home/anon/miniconda3/envs/diffusion-pipe/lib/python3.12/site-packages/nvidia/cudnn/include:$CPLUS_INCLUDE_PATH pip install --no-build-isolation transformer_engine[pytorch]
```
Edit the paths above for your conda environment.

## Dataset preparation
A dataset consists of one or more directories containing image or video files, and corresponding captions. You can mix images and videos in the same directory, but it's probably a good idea to separate them in case you need to specify certain settings on a per-directory basis. Caption files should be .txt files with the same base name as the corresponding media file, e.g. image1.png should have caption file image1.txt in the same directory. If a media file doesn't have a matching caption file, a warning is printed, but training will proceed with an empty caption.

For images, any image format that can be loaded by Pillow should work. For videos, any format that can be loaded by ImageIO should work. Note that this means **WebP videos are not supported**, because ImageIO can't load multi-frame WebPs.

## Supported models
See the [supported models doc](./docs/supported_models.md) for more information on how to configure each model, the options it supports, and the format of the saved LoRAs.

## Training
**Start by reading through the config files in the examples directory.** Almost everything is commented, explaining what each setting does. [This config file](./examples/main_example.toml) is the main example with all of the comments. [This dataset config file](./examples/dataset.toml) has the documentation for the dataset options.

Once you've familiarized yourself with the config file format, go ahead and make a copy and edit to your liking. At minimum, change all the paths to conform to your setup, including the paths in the dataset config file.

Launch training like this:
```
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config examples/hunyuan_video.toml
```
RTX 4000 series needs those 2 environment variables set. Other GPUs may not need them. You can try without them, Deepspeed will complain if it's wrong.

If you enabled checkpointing, you can resume training from the latest checkpoint by simply re-running the exact same command but with the `--resume_from_checkpoint` flag. You can also specify a specific checkpoint folder name after the flag to resume from that particular checkpoint (e.g. `--resume_from_checkpoint "20250212_07-06-40"`). This option is particularly useful if you have run multiple training sessions with different datasets and want to resume from a specific training folder.

Please note that resuming from checkpoint uses the **config file on the command line**, not the config file saved into the output directory. You are responsible for making sure that the config file you pass in matches what was previously used.

## Output files
A new directory will be created in ```output_dir``` for each training run. This contains the checkpoints, saved models, and Tensorboard metrics. Saved models/LoRAs will be in directories named like epoch1, epoch2, etc. Deepspeed checkpoints are in directories named like global_step1234. These checkpoints contain all training state, including weights, optimizer, and dataloader state, but can't be used directly for inference. The saved model directory will have the safetensors weights, PEFT adapter config JSON, as well as the diffusion-pipe config file for easier tracking of training run settings.

## Reducing VRAM requirements
The [wan_14b_min_vram.toml](./examples/wan_14b_min_vram.toml) example file has all of these settings enabled.
- Use AdamW8BitKahan optimizer:
  ```
  [optimizer]
  type = 'AdamW8bitKahan'
  lr = 5e-5
  betas = [0.9, 0.99]
  weight_decay = 0.01
  stabilize = false
  ```
- Use block swapping if the model supports it: ```blocks_to_swap = 32```
- Try the expandable_segments feature in the CUDA memory allocator:
  - ```PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config /home/you/path/to/config.toml```
  - I've seen this help a lot when training on video with multiple aspect ratio buckets.
  - On my system, sometimes this causes random CUDA failures. If training gets through a few steps though, it will train indefinitely without failures. Very weird.
- Use unsloth activation checkpointing: ```activation_checkpointing = 'unsloth'```

## Parallelism
This code uses hybrid data- and pipeline-parallelism. Set the ```--num_gpus``` flag appropriately for your setup. Set ```pipeline_stages``` in the config file to control the degree of pipeline parallelism. Then the data parallelism degree will automatically be set to use all GPUs (number of GPUs must be divisible by pipeline_stages). For example, with 4 GPUs and pipeline_stages=2, you will run two instances of the model, each divided across two GPUs.

## Pre-caching
Latents and text embeddings are cached to disk before training happens. This way, the VAE and text encoders don't need to be kept loaded during training. The Huggingface Datasets library is used for all the caching. Cache files are reused between training runs if they exist. All cache files are written into a directory named "cache" inside each dataset directory.

This caching also means that training LoRAs for text encoders is not currently supported.

Three flags are relevant for caching. ```--cache_only``` does the caching flow, then exits without training anything. ```--regenerate_cache``` forces cache regeneration. ```--trust_cache``` will blindly load the cached metadata files, without checking if any data files have changed via the fingerprint. This can speed up loading for very large datasets (100,000+ images), but you must make sure nothing in the dataset has changed.

## Extra
You can check out my [qlora-pipe](https://github.com/tdrussell/qlora-pipe) project, which is basically the same thing as this but for LLMs.
