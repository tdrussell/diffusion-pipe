# A collection of notes on Minimax H3 implementation and training

## Distillation
**Any standard training gradually undistills the model**. You may need to use CFG for inference. How much CFG you will need depends on the size of the dataset and the amount of training.
### Solutions
- CFG-augmented training (see the example TOML config). This modifies the model's output with its own unconditional prediction when fitting the target, in order to cause the model output to have built-in CFG. This preserves the guidance distillation that the model comes with. You can think of this as "baking in" as CFG value using the model's own uncond prediction during training time, even though you are still fitting the standard flow-matching target. This technique works very well and is recommended.
- Training adapter. Ostris has a [training adapter](https://huggingface.co/ostris/minimax_h3_training_adapter/tree/main) for the model, but I haven't tested this myself.

## General notes
The AdaLN weights are not trained with LoRA. This makes the LoRA compatible with both the full and pruned checkpoints, regardless of which one you trained with.

The dataset caching phase should use ComfyUI dynamic VRAM, meaning the text encoder can be larger than available VRAM. E.g. the int8 convrot TE is 26GB, but can compute text embeddings on a 24GB GPU.

If dataset caching occurs (meaning it loads the VAE / TE), the text encoder memory somehow isn't completely freed from RAM afterwards. The TE is large, and this might OOM you. Just relaunch the training script, since the dataset is now cached. Or do it in 2 phases from the beginning:
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True deepspeed --num_gpus=1 train.py --config your_config.toml --trust_cache --cache_only
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True deepspeed --num_gpus=1 train.py --config your_config.toml --trust_cache
```
`--trust_cache` loads the cache faster if it exists, but you won't pick up changes to the underlying data files. If you don't change your dataset files, this is always safe to pass.

You can train LoRAs directly on top of quantized weights, like int8 convrot, and you probably should since the model is large.

Block swapping will be needed for 24GB VRAM. `blocks_to_swap=48` is the maximum allowed for this model. `activation_checkpointing = 'unsloth'` also saves a lot of VRAM for minimal overhead.

Audio is automatically trained if your videos have it.

Training on images works fine. I saw some reports that it won't work; no, it does. The VAE is "asymmetric": the encoder can encode 1 image frame to a valid single latent frame, but the decoder can't decode that single latent frame very well. So the latent space for images is fine, and the model can learn from it. But it does gradually degrade the video/motion understanding in the model if you exclusively train on images (same as any other video model). Joint image/video training will work better.

I don't know what timestep distribution and shift value is optimal. `timestep_sample_method='uniform'` and `shift=12` matches the default inference schedule. Lowering shift to something like 8, or even lower, could help learn details at the expense of large-scale structure and motion.

Only T2I and T2V training is supported, and this might not ever change. Reference training is very complex. First/last frame to video (FL2V) is less complex than reference, but for small-scale lora training, you can just train pure T2V to learn your concept / character / style. The model already has general-purpose first/last frame conditioning. It will retain that ability even with your T2V lora. Plus your lora is now more compatible and can be used in T2V mode as well. TLDR: why bother with FL2V training at small scale.