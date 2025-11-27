# TensorBoard Metric Guide & Expected Behavior

This guide helps interpret the training charts logged during the session in TensorBoard.

## 1. Loss Metrics (Learning Progress)

* **`train/loss`**: The loss value calculated on the current training batch.
    * **Expected:** Rapid decrease at the start (warmup/early phase), followed by a slower, steady decay or plateau.
    * **Warning:**
        * Flat line from start: Model is not learning (LR too low or bug).
        * Spikes/NaNs: Instability (LR too high or bad data).
        * Approaching zero too fast: Potential overfitting to training data.

* **`train/epoch_loss`**: The average loss over the entire epoch. Smoother than step loss.

* **`validation/loss`** (e.g., `{name}/loss`): Loss on the validation dataset.
    * **Expected:** Should follow a similar downward trend to `train/loss`, usually slightly higher.
    * **Warning (CRITICAL):** If `train/loss` keeps dropping but `validation/loss` starts **rising**, the model is **overfitting**. Stop training or lower LR.

* **`eval/{name}/loss_quantile_X.X`**: Validation loss for specific noise levels (timesteps).
    * **Expected:** Helps identify if the model struggles with specific noise levels (e.g., huge noise vs. fine details). All quantiles should ideally decrease.

## 2. Stability Metrics

* **`train/total_param_norm`** (if enabled): The magnitude of model weights. Should stay stable, not explode.

## 3. Memory Usage (System & GPU)

* **`memory/vram_allocated_gb`**: Actual VRAM memory occupied by tensors (weights, gradients, activations).
    * **Expected:** Fluctuates slightly with batch sizes but generally stable.

* **`memory/vram_reserved_gb`**: VRAM reserved by PyTorch caching allocator (always >= allocated).
    * **Expected:** Represents the "working set" of memory PyTorch holds from the OS.

* **`memory/vram_peak_gb`**: **High-water mark**. The maximum VRAM used since the start.
    * **Warning:** If this is close to your GPU limit (e.g., 24GB for RTX 3090), you are at risk of OOM (Out Of Memory) crashes. Reduce batch size or enable checkpointing.

* **`memory/ram_main_process_gb`**: System RAM usage (RSS) of the main Python process.
    * **Note:** Does not include memory used by DataLoader worker processes.

## 4. Performance

* **`eval/eval_time_sec`**: Time taken to complete one evaluation round.