import gradio as gr
import toml
import os
import subprocess
import threading
import queue

# 默认配置路径
BASE_DIR = "/root/diffusion-pipe"
TRAIN_SCRIPT = "/root/diffusion-pipe/train.py"
DATASET_CONFIG = "/root/diffusion-pipe/examples/dataset.toml"
I2V_TRAIN_CONFIG = "/root/diffusion-pipe/examples/wan_i2v.toml"
T2V_TRAIN_CONFIG = "/root/diffusion-pipe/examples/wan_t2v.toml"

# i2v 默认训练配置
i2v_default_train_config = {
    "output_dir": "/data/diffusion_pipe_training_runs/wan2.1_lora_i2v",
    "dataset": "/root/diffusion-pipe/examples/dataset.toml",
    "epochs": 1000,
    "micro_batch_size_per_gpu": 1,
    "pipeline_stages": 1,
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "warmup_steps": 100,
    "blocks_to_swap": 0,
    "eval_every_n_epochs": 1,
    "eval_before_first_step": True,
    "eval_micro_batch_size_per_gpu": 1,
    "eval_gradient_accumulation_steps": 1,
    "save_every_n_epochs": 2,
    "checkpoint_every_n_minutes": 120,
    "activation_checkpointing": True,
    "save_dtype": "bfloat16",
    "caching_batch_size": 1,
    "video_clip_mode": "single_beginning",
    "partition_method": "parameters",
    "partition_split": [],
    "disable_block_swap_for_eval": False,
    "eval_datasets": [],
    "model": {
        "type": "wan",
        "ckpt_path": "/data2/imagegen_models/Wan2.1-I2V-1.3B",
        "dtype": "bfloat16",
        "transformer_dtype": "float8",
        "timestep_sample_method": "logit_normal",
        "llm_path": ""
    },
    "adapter": {
        "type": "lora",
        "rank": 32,
        "dtype": "bfloat16",
        "init_from_existing": ""
    },
    "optimizer": {
        "type": "adamw_optimi",
        "lr": 2e-5,
        "betas": [0.9, 0.99],
        "weight_decay": 0.01,
        "eps": 1e-8,
        "gradient_release": False
    }
}

# t2v 默认训练配置
t2v_default_train_config = {
    "output_dir": "/data/diffusion_pipe_training_runs/wan2.1_lora_t2v",
    "dataset": "/root/diffusion-pipe/examples/dataset.toml",
    "epochs": 1000,
    "micro_batch_size_per_gpu": 1,
    "pipeline_stages": 1,
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "warmup_steps": 100,
    "blocks_to_swap": 0,
    "eval_every_n_epochs": 1,
    "eval_before_first_step": True,
    "eval_micro_batch_size_per_gpu": 1,
    "eval_gradient_accumulation_steps": 1,
    "save_every_n_epochs": 2,
    "checkpoint_every_n_minutes": 120,
    "activation_checkpointing": True,
    "save_dtype": "bfloat16",
    "caching_batch_size": 1,
    "video_clip_mode": "single_beginning",
    "partition_method": "parameters",
    "partition_split": [],
    "disable_block_swap_for_eval": False,
    "eval_datasets": [],
    "model": {
        "type": "wan",
        "ckpt_path": "/data2/imagegen_models/Wan2.1-T2V-1.3B",
        "dtype": "bfloat16",
        "transformer_dtype": "float8",
        "timestep_sample_method": "logit_normal",
        "llm_path": ""
    },
    "adapter": {
        "type": "lora",
        "rank": 32,
        "dtype": "bfloat16",
        "init_from_existing": ""
    },
    "optimizer": {
        "type": "adamw_optimi",
        "lr": 2e-5,
        "betas": [0.9, 0.99],
        "weight_decay": 0.01,
        "eps": 1e-8,
        "gradient_release": False
    }
}

# 默认数据集配置
default_dataset_config = {
    "resolutions": [512],
    "enable_ar_bucket": True,
    "min_ar": 0.5,
    "max_ar": 2.0,
    "num_ar_buckets": 7,
    "frame_buckets": [1, 33],
    "directory": [{"path": "/home/anon/data/images/grayscale", "num_repeats": 10}]
}

default_train_config = i2v_default_train_config  # 添加全局变量初始值

# 加载配置函数
def load_config(config_path, default_config):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return toml.load(f)
    return default_config

# 初始化配置
dataset_config = load_config(DATASET_CONFIG, default_dataset_config)
i2v_train_config = load_config(I2V_TRAIN_CONFIG, i2v_default_train_config)
t2v_train_config = load_config(T2V_TRAIN_CONFIG, t2v_default_train_config)
train_config = i2v_train_config  # 默认加载 i2v 配置

# 保存配置函数
def save_configs(train_config_dict, dataset_config_dict, model_type):
    os.makedirs(os.path.dirname(DATASET_CONFIG), exist_ok=True)
    train_config_path = I2V_TRAIN_CONFIG if model_type == "i2v" else T2V_TRAIN_CONFIG
    os.makedirs(os.path.dirname(train_config_path), exist_ok=True)
    with open(train_config_path, "w") as f:
        toml.dump(train_config_dict, f)
    with open(DATASET_CONFIG, "w") as f:
        toml.dump(dataset_config_dict, f)
    return f"训练配置已保存到 {train_config_path}\n数据集配置已保存到 {DATASET_CONFIG}"

# 输入验证函数
def validate_partition_split(value, pipeline_stages, partition_method):
    if partition_method != "manual" or not value:
        return []
    try:
        split = [int(x) for x in value.split(",")]
        if len(split) != pipeline_stages - 1:
            raise ValueError(f"分割点数量 ({len(split)}) 必须等于 pipeline_stages - 1 ({pipeline_stages - 1})")
        return split
    except ValueError as e:
        raise gr.Error(f"无效的分割点: {str(e)}")

def validate_eval_datasets(value):
    try:
        return toml.loads(value)["eval_datasets"] if value else []
    except Exception as e:
        raise gr.Error(f"评估数据集格式错误: {str(e)}")

# 更新训练配置函数（修复为 44 个参数）
def update_train_config(output_dir, dataset, epochs, micro_batch_size_per_gpu, pipeline_stages, gradient_accumulation_steps,
                       gradient_clipping, warmup_steps, blocks_to_swap, eval_every_n_epochs, eval_before_first_step,
                       eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps, save_every_n_epochs,
                       checkpoint_every_n_minutes, activation_checkpointing, save_dtype, caching_batch_size,
                       video_clip_mode, ckpt_path, dtype, transformer_dtype, timestep_sample_method, rank, adapter_dtype,
                       init_from_existing, optimizer_type, lr, betas_0, betas_1, weight_decay, eps,
                       partition_method, partition_split, llm_path, eval_datasets, disable_block_swap_for_eval,
                       gradient_release):
    output_dir = output_dir.replace("\\", "/")
    dataset = dataset.replace("\\", "/")
    ckpt_path = ckpt_path.replace("\\", "/")
    init_from_existing = init_from_existing.replace("\\", "/") if init_from_existing else ""
    llm_path = llm_path.replace("\\", "/") if llm_path else ""
    
    partition_split = validate_partition_split(partition_split, pipeline_stages, partition_method)
    eval_datasets = validate_eval_datasets(eval_datasets)
    
    config = {
        "output_dir": output_dir,
        "dataset": dataset,
        "epochs": epochs,
        "micro_batch_size_per_gpu": micro_batch_size_per_gpu,
        "pipeline_stages": pipeline_stages,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": gradient_clipping,
        "warmup_steps": warmup_steps,
        "blocks_to_swap": blocks_to_swap,
        "eval_every_n_epochs": eval_every_n_epochs,
        "eval_before_first_step": eval_before_first_step,
        "eval_micro_batch_size_per_gpu": eval_micro_batch_size_per_gpu,
        "eval_gradient_accumulation_steps": eval_gradient_accumulation_steps,
        "save_every_n_epochs": save_every_n_epochs,
        "checkpoint_every_n_minutes": checkpoint_every_n_minutes,
        "activation_checkpointing": activation_checkpointing,
        "save_dtype": save_dtype,
        "caching_batch_size": caching_batch_size,
        "video_clip_mode": video_clip_mode,
        "partition_method": partition_method,
        "partition_split": partition_split if partition_split else None,
        "disable_block_swap_for_eval": disable_block_swap_for_eval,
        "eval_datasets": eval_datasets,
        "model": {
            "type": "wan",
            "ckpt_path": ckpt_path,
            "dtype": dtype,
            "transformer_dtype": transformer_dtype if transformer_dtype else None,
            "timestep_sample_method": timestep_sample_method,
            "llm_path": llm_path if llm_path else None
        },
        "adapter": {
            "type": "lora",
            "rank": rank,
            "dtype": adapter_dtype,
            "init_from_existing": init_from_existing if init_from_existing else None
        },
        "optimizer": {
            "type": optimizer_type,
            "lr": float(lr),
            "betas": [betas_0, betas_1],
            "weight_decay": weight_decay,
            "eps": eps,
            "gradient_release": gradient_release
        }
    }
    if not config["model"]["transformer_dtype"]:
        del config["model"]["transformer_dtype"]
    if not config["adapter"]["init_from_existing"]:
        del config["adapter"]["init_from_existing"]
    if not config["partition_split"]:
        del config["partition_split"]
    if not config["model"]["llm_path"]:
        del config["model"]["llm_path"]
    return config

# 更新数据集配置函数
def update_dataset_config(resolutions, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets, directory_path, num_repeats):
    directory_path = directory_path.replace("\\", "/")
    config = {
        "resolutions": [int(resolutions)],
        "enable_ar_bucket": enable_ar_bucket,
        "min_ar": min_ar,
        "max_ar": max_ar,
        "num_ar_buckets": num_ar_buckets,
        "frame_buckets": [int(f) for f in frame_buckets.split(",")],
        "directory": [{"path": directory_path, "num_repeats": num_repeats}]
    }
    return config

# 实时读取子进程输出
def read_output(pipe, output_queue):
    for line in iter(pipe.readline, ''):
        output_queue.put(line)
    pipe.close()

# 启动训练函数
def start_training(model_type, num_gpus):
    if not os.path.exists(TRAIN_SCRIPT):
        yield f"错误：训练脚本 {TRAIN_SCRIPT} 不存在！"
        return
    env = os.environ.copy()
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_IB_DISABLE"] = "1"
    train_config_path = I2V_TRAIN_CONFIG if model_type == "i2v" else T2V_TRAIN_CONFIG
    cmd = ["deepspeed", f"--num_gpus={int(num_gpus)}", TRAIN_SCRIPT, "--deepspeed", f"--config={train_config_path}"]
    
    try:
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
        output_queue = queue.Queue()
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, output_queue))
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, output_queue))
        stdout_thread.start()
        stderr_thread.start()
        
        full_output = "训练已启动！\n"
        max_lines = 100
        output_lines = [full_output]
        yield full_output
        
        while True:
            try:
                line = output_queue.get(timeout=1)
                output_lines.append(line)
                if len(output_lines) > max_lines:
                    output_lines.pop(0)
                full_output = "".join(output_lines)
                yield full_output
            except queue.Empty:
                if process.poll() is not None:
                    break
        
        stdout_thread.join()
        stderr_thread.join()
        return_code = process.returncode
        if return_code == 0:
            output_lines.append("\n训练成功完成！")
        else:
            output_lines.append(f"\n训练异常退出，返回码：{return_code}")
        if len(output_lines) > max_lines:
            output_lines.pop(0)
        full_output = "".join(output_lines)
        yield full_output
    except Exception as e:
        yield f"训练启动失败：{str(e)}"

# 更新配置函数
def update_configs(model_type):
    train_config = load_config(I2V_TRAIN_CONFIG, i2v_default_train_config) if model_type == "i2v" else load_config(T2V_TRAIN_CONFIG, t2v_default_train_config)
    dataset_config = load_config(DATASET_CONFIG, default_dataset_config)
    return (
        train_config["output_dir"],
        train_config["dataset"],
        train_config["epochs"],
        train_config["micro_batch_size_per_gpu"],
        train_config["pipeline_stages"],
        train_config["gradient_accumulation_steps"],
        train_config["gradient_clipping"],
        train_config["warmup_steps"],
        train_config["blocks_to_swap"],
        train_config["eval_every_n_epochs"],
        train_config["eval_before_first_step"],
        train_config["eval_micro_batch_size_per_gpu"],
        train_config["eval_gradient_accumulation_steps"],
        train_config["save_every_n_epochs"],
        train_config["checkpoint_every_n_minutes"],
        train_config["activation_checkpointing"],
        train_config["save_dtype"],
        train_config["caching_batch_size"],
        train_config["video_clip_mode"],
        train_config["model"]["ckpt_path"],
        train_config["model"]["dtype"],
        train_config["model"].get("transformer_dtype", ""),
        train_config["model"]["timestep_sample_method"],
        train_config["adapter"]["rank"],
        train_config["adapter"]["dtype"],
        train_config["adapter"].get("init_from_existing", ""),
        train_config["optimizer"]["type"],
        str(train_config["optimizer"]["lr"]),
        train_config["optimizer"]["betas"][0],
        train_config["optimizer"]["betas"][1],
        train_config["optimizer"]["weight_decay"],
        train_config["optimizer"]["eps"],
        train_config.get("partition_method", "parameters"),
        ",".join(map(str, train_config.get("partition_split", []))),
        train_config["model"].get("llm_path", ""),
        toml.dumps({"eval_datasets": train_config.get("eval_datasets", [])}),
        train_config.get("disable_block_swap_for_eval", False),
        train_config["optimizer"].get("gradient_release", False),
        str(dataset_config["resolutions"][0]),
        dataset_config["enable_ar_bucket"],
        dataset_config["min_ar"],
        dataset_config["max_ar"],
        dataset_config["num_ar_buckets"],
        ",".join(map(str, dataset_config["frame_buckets"])),
        dataset_config["directory"][0]["path"],
        dataset_config["directory"][0]["num_repeats"]
    )

# 动态限制函数
def restrict_blocks_to_swap(pipeline_stages):
    if pipeline_stages > 1:
        return gr.update(value=0, interactive=False, info="块交换数量 / Blocks to Swap (禁用，因pipeline_stages > 1)")
    return gr.update(interactive=True, info="将模型块移到RAM降低VRAM使用，仅在pipeline_stages=1时有效 / Move model blocks to RAM to reduce VRAM usage, only valid when pipeline_stages=1")

def restrict_gradient_clipping(gradient_release):
    if gradient_release:
        return gr.update(value=0, interactive=False, info="梯度裁剪值 / Gradient Clipping (禁用，因gradient_release启用)")
    return gr.update(interactive=True, info="限制梯度最大值，gradient_release启用时无效 / Limit maximum gradient value, invalid when gradient_release is enabled")

# 创建 Gradio 界面
with gr.Blocks(title="Wan2.1 LoRA 训练配置与启动器") as demo:
    gr.Markdown("# Wan2.1 LoRA 训练配置与启动器(B站 HooTooH)")
    gr.Markdown("输入参数，保存配置文件，然后启动训练。")

    # 训练配置部分
    gr.Markdown("## 训练配置")
    gr.Markdown("### 全局设置")
    model_type = gr.Dropdown(choices=["i2v", "t2v"], value="i2v", label="选择模型类型 / Select Model Type", info="选择要训练的模型类型：i2v 或 t2v / Select the model type to train: i2v or t2v")
    
    with gr.Row():
        num_gpus = gr.Number(value=1, label="GPU数量 / Number of GPUs", info="指定训练使用的GPU数量 / Specify the number of GPUs for training", minimum=1, step=1)
        output_dir = gr.Textbox(value=train_config["output_dir"], label="输出目录 / Output Directory", info="训练结果保存路径 / Path to save training results")
        dataset = gr.Textbox(value=train_config["dataset"], label="数据集配置文件路径 / Dataset Config File Path", info="指向 dataset.toml 的绝对路径 / Absolute path to dataset.toml")
    with gr.Row():
        epochs = gr.Number(value=train_config["epochs"], label="训练周期数 / Number of Epochs", info="总训练轮数，通常设为较大值 / Total training rounds, usually set to a large value")
        micro_batch_size_per_gpu = gr.Number(value=train_config["micro_batch_size_per_gpu"], label="每 GPU 微批次大小 / Micro Batch Size per GPU", info="单 GPU 每次前向/反向的批次大小 / Batch size per GPU for each forward/backward pass")
        pipeline_stages = gr.Number(value=train_config["pipeline_stages"], label="管道并行度 / Pipeline Stages", info="模型分担的GPU数量，应与GPU数量匹配 / Number of GPUs to split the model across, should match GPU count")
    with gr.Row():
        partition_method = gr.Dropdown(choices=["parameters", "manual"], value=train_config.get("partition_method", "parameters"), label="分区方法 / Partition Method (默认: parameters)", info="可选，参数自动分配或手动指定层分割 / Optional, auto parameter allocation or manual layer split")
        partition_split = gr.Textbox(value=",".join(map(str, train_config.get("partition_split", []))), label="分割点 / Partition Split (可选)", info="可选，仅manual模式有效，逗号分隔（如10,20） / Optional, valid only in manual mode, comma-separated (e.g., 10,20)", visible=False)
    with gr.Row():
        gradient_accumulation_steps = gr.Number(value=train_config["gradient_accumulation_steps"], label="梯度累积步数 / Gradient Accumulation Steps", info="累积多少步更新一次权重 / How many steps to accumulate before updating weights")
        gradient_clipping = gr.Number(value=train_config["gradient_clipping"], label="梯度裁剪值 / Gradient Clipping", info="限制梯度最大值，gradient_release启用时无效 / Limit maximum gradient value, invalid when gradient_release is enabled")
        warmup_steps = gr.Number(value=train_config["warmup_steps"], label="预热步数 / Warmup Steps", info="学习率逐渐增加的步数 / Steps for gradually increasing learning rate")
    with gr.Row():
        blocks_to_swap = gr.Number(value=train_config["blocks_to_swap"], label="块交换数量 / Blocks to Swap", info="将模型块移到RAM降低VRAM使用，仅在pipeline_stages=1时有效 / Move model blocks to RAM to reduce VRAM usage, only valid when pipeline_stages=1")
        eval_every_n_epochs = gr.Number(value=train_config["eval_every_n_epochs"], label="每多少轮评估 / Evaluate Every N Epochs", info="评估频率（周期数） / Evaluation frequency (in epochs)")
        eval_before_first_step = gr.Checkbox(value=train_config["eval_before_first_step"], label="训练前评估 / Evaluate Before First Step", info="是否在第一步前运行评估 / Whether to run evaluation before the first step")
    with gr.Row():
        eval_micro_batch_size_per_gpu = gr.Number(value=train_config["eval_micro_batch_size_per_gpu"], label="评估每 GPU 微批次大小 / Eval Micro Batch Size per GPU", info="评估时的微批次大小 / Micro batch size during evaluation")
        eval_gradient_accumulation_steps = gr.Number(value=train_config["eval_gradient_accumulation_steps"], label="评估梯度累积步数 / Eval Gradient Accumulation Steps", info="评估时的梯度累积步数 / Gradient accumulation steps during evaluation")
        save_every_n_epochs = gr.Number(value=train_config["save_every_n_epochs"], label="每多少轮保存 / Save Every N Epochs", info="模型保存频率（周期数） / Model save frequency (in epochs)")
    with gr.Row():
        checkpoint_every_n_minutes = gr.Number(value=train_config["checkpoint_every_n_minutes"], label="检查点保存间隔（分钟） / Checkpoint Save Interval (Minutes)", info="每隔多少分钟保存检查点，0 表示禁用 / Save checkpoint every few minutes, 0 to disable")
        activation_checkpointing = gr.Checkbox(value=train_config["activation_checkpointing"], label="激活检查点 / Activation Checkpointing", info="节省 VRAM 的技术，通常启用 / Technique to save VRAM, usually enabled")
        save_dtype = gr.Dropdown(choices=["bfloat16", "float16", "float32"], value=train_config["save_dtype"], label="保存数据类型 / Save Data Type", info="保存模型权重的数据类型 / Data type for saving model weights")
    with gr.Row():
        caching_batch_size = gr.Number(value=train_config["caching_batch_size"], label="缓存批次大小 / Caching Batch Size", info="预缓存时的批次大小，影响内存使用 / Batch size during pre-caching, affects memory usage")
        video_clip_mode = gr.Dropdown(choices=["single_beginning", "single_middle", "multiple_overlapping"], value=train_config["video_clip_mode"], label="视频剪辑模式 / Video Clip Mode", info="视频帧提取方式 / Method for extracting video frames")
    with gr.Row():
        eval_datasets = gr.Textbox(value=toml.dumps({"eval_datasets": train_config.get("eval_datasets", [])}), label="评估数据集 / Eval Datasets (可选)", info="可选，多个评估数据集配置（TOML格式） / Optional, multiple eval dataset configs (TOML format)", lines=3)
        disable_block_swap_for_eval = gr.Checkbox(value=train_config.get("disable_block_swap_for_eval", False), label="评估时禁用块交换 / Disable Block Swap for Eval", info="评估时禁用块交换以加速 / Disable block swapping during eval to speed up")

    gr.Markdown("### 模型配置")
    with gr.Row():
        ckpt_path = gr.Textbox(value=train_config["model"]["ckpt_path"], label="检查点路径 / Checkpoint Path", info="Wan2.1 模型的预训练权重路径 / Path to pre-trained weights of Wan2.1 model")
        llm_path = gr.Textbox(value=train_config["model"].get("llm_path", ""), label="UMT5路径 / UMT5 Path (可选)", info="可选，自定义UMT5权重路径 / Optional, custom UMT5 weights path")
    with gr.Row():
        dtype = gr.Dropdown(choices=["bfloat16", "float16", "float32"], value=train_config["model"]["dtype"], label="基础数据类型 / Base Data Type", info="模型计算的数据类型 / Data type for model computation")
        transformer_dtype = gr.Dropdown(choices=["", "float8", "bfloat16"], value=train_config["model"].get("transformer_dtype", ""), label="变换器数据类型 / Transformer Data Type", info="变换器部分的特殊数据类型，可选 / Special data type for transformer part, optional")
        timestep_sample_method = gr.Dropdown(choices=["logit_normal", "uniform"], value=train_config["model"]["timestep_sample_method"], label="时间步采样方法 / Timestep Sampling Method", info="训练时时间步的采样策略 / Sampling strategy for timesteps during training")

    gr.Markdown("### 适配器配置")
    with gr.Row():
        rank = gr.Number(value=train_config["adapter"]["rank"], label="LoRA 秩 / LoRA Rank", info="LoRA 的秩大小，影响模型容量 / LoRA rank size, affects model capacity")
        adapter_dtype = gr.Dropdown(choices=["bfloat16", "float16", "float32"], value=train_config["adapter"]["dtype"], label="LoRA 数据类型 / LoRA Data Type", info="LoRA 权重的数据类型 / Data type for LoRA weights")
    with gr.Row():
        init_from_existing = gr.Textbox(value=train_config["adapter"].get("init_from_existing", ""), label="从现有 LoRA 初始化 / Initialize from Existing LoRA", info="可选，现有 LoRA 权重路径 / Optional, path to existing LoRA weights")

    gr.Markdown("### 优化器配置")
    with gr.Row():
        optimizer_type = gr.Dropdown(choices=["adamw_optimi", "AdamW8bitKahan", "Prodigy"], value=train_config["optimizer"]["type"], label="优化器类型 / Optimizer Type", info="训练使用的优化器 / Optimizer used for training")
        lr = gr.Textbox(value=str(train_config["optimizer"]["lr"]), label="学习率 / Learning Rate", info="优化器的学习率，如 2e-5 / Learning rate of the optimizer, e.g., 2e-5")
        gradient_release = gr.Checkbox(value=train_config["optimizer"].get("gradient_release", False), label="梯度释放 / Gradient Release", info="实验性VRAM节省选项，启用时禁用梯度裁剪 / Experimental VRAM saving option, disables gradient clipping when enabled")
    with gr.Row():
        betas_0 = gr.Number(value=train_config["optimizer"]["betas"][0], label="Beta 1", info="Adam 优化器的第一个 beta 参数 / First beta parameter of Adam optimizer")
        betas_1 = gr.Number(value=train_config["optimizer"]["betas"][1], label="Beta 2", info="Adam 优化器的第二个 beta 参数 / Second beta parameter of Adam optimizer")
        weight_decay = gr.Number(value=train_config["optimizer"]["weight_decay"], label="权重衰减 / Weight Decay", info="正则化参数 / Regularization parameter")
        eps = gr.Number(value=train_config["optimizer"]["eps"], label="Epsilon", info="Adam 的数值稳定性参数 / Numerical stability parameter for Adam")

    # 数据集配置部分
    gr.Markdown("## 数据集配置")
    with gr.Row():
        resolutions = gr.Textbox(value=str(dataset_config["resolutions"][0]), label="分辨率 / Resolution", info="训练图像的分辨率（单个整数，如 512） / Resolution of training images (single integer, e.g., 512)")
        enable_ar_bucket = gr.Checkbox(value=dataset_config["enable_ar_bucket"], label="启用宽高比桶 / Enable Aspect Ratio Bucket", info="是否按宽高比分组图像 / Whether to group images by aspect ratio")
    with gr.Row():
        min_ar = gr.Number(value=dataset_config["min_ar"], label="最小宽高比 / Minimum Aspect Ratio", info="宽高比范围的最小值 / Minimum value of aspect ratio range")
        max_ar = gr.Number(value=dataset_config["max_ar"], label="最大宽高比 / Maximum Aspect Ratio", info="宽高比范围的最大值 / Maximum value of aspect ratio range")
        num_ar_buckets = gr.Number(value=dataset_config["num_ar_buckets"], label="宽高比桶数 / Number of Aspect Ratio Buckets", info="宽高比分组数量 / Number of aspect ratio groups")
    with gr.Row():
        frame_buckets = gr.Textbox(value=",".join(map(str, dataset_config["frame_buckets"])), label="帧数桶 / Frame Buckets", info="视频帧数分组，逗号分隔（如 1,33） / Video frame grouping, comma-separated (e.g., 1,33)")
        directory_path = gr.Textbox(value=dataset_config["directory"][0]["path"], label="数据目录路径 / Data Directory Path", info="图像/视频文件所在目录 / Directory containing image/video files")
        num_repeats = gr.Number(value=dataset_config["directory"][0]["num_repeats"], label="重复次数 / Number of Repeats", info="每个样本重复次数，影响 epoch 大小 / Number of repeats per sample, affects epoch size")

    # 输出与操作区域
    gr.Markdown("## 操作与输出")
    with gr.Row():
        train_config_output = gr.Textbox(label="训练配置文件预览 (TOML) / Training Config Preview (TOML)", lines=10)
        dataset_config_output = gr.Textbox(label="数据集配置文件预览 (TOML) / Dataset Config Preview (TOML)", lines=10)
    save_message = gr.Textbox(label="保存状态 / Save Status")
    train_output = gr.Textbox(label="训练输出 / Training Output", lines=10)

    with gr.Row():
        preview_btn = gr.Button("预览配置 / Preview Config")
        save_btn = gr.Button("保存配置文件")
        train_btn = gr.Button("开始训练")

    # 事件绑定
    num_gpus.change(fn=lambda x: x, inputs=num_gpus, outputs=pipeline_stages)
    pipeline_stages.change(fn=restrict_blocks_to_swap, inputs=pipeline_stages, outputs=blocks_to_swap)
    partition_method.change(fn=lambda x: gr.update(visible=x == "manual"), inputs=partition_method, outputs=partition_split)
    gradient_release.change(fn=restrict_gradient_clipping, inputs=gradient_release, outputs=gradient_clipping)

    # 更新配置时切换默认值
    def update_configs_with_default(model_type):
        global default_train_config
        default_train_config = i2v_default_train_config if model_type == "i2v" else t2v_default_train_config
        train_config = load_config(I2V_TRAIN_CONFIG, i2v_default_train_config) if model_type == "i2v" else load_config(T2V_TRAIN_CONFIG, t2v_default_train_config)
        dataset_config = load_config(DATASET_CONFIG, default_dataset_config)
        return (
            train_config["output_dir"],
            train_config["dataset"],
            train_config["epochs"],
            train_config["micro_batch_size_per_gpu"],
            train_config["pipeline_stages"],
            train_config["gradient_accumulation_steps"],
            train_config["gradient_clipping"],
            train_config["warmup_steps"],
            train_config["blocks_to_swap"],
            train_config["eval_every_n_epochs"],
            train_config["eval_before_first_step"],
            train_config["eval_micro_batch_size_per_gpu"],
            train_config["eval_gradient_accumulation_steps"],
            train_config["save_every_n_epochs"],
            train_config["checkpoint_every_n_minutes"],
            train_config["activation_checkpointing"],
            train_config["save_dtype"],
            train_config["caching_batch_size"],
            train_config["video_clip_mode"],
            train_config["model"]["ckpt_path"],
            train_config["model"]["dtype"],
            train_config["model"].get("transformer_dtype", ""),
            train_config["model"]["timestep_sample_method"],
            train_config["adapter"]["rank"],
            train_config["adapter"]["dtype"],
            train_config["adapter"].get("init_from_existing", ""),
            train_config["optimizer"]["type"],
            str(train_config["optimizer"]["lr"]),
            train_config["optimizer"]["betas"][0],
            train_config["optimizer"]["betas"][1],
            train_config["optimizer"]["weight_decay"],
            train_config["optimizer"]["eps"],
            train_config.get("partition_method", "parameters"),
            ",".join(map(str, train_config.get("partition_split", []))),
            train_config["model"].get("llm_path", ""),
            toml.dumps({"eval_datasets": train_config.get("eval_datasets", [])}),
            train_config.get("disable_block_swap_for_eval", False),
            train_config["optimizer"].get("gradient_release", False),
            str(dataset_config["resolutions"][0]),
            dataset_config["enable_ar_bucket"],
            dataset_config["min_ar"],
            dataset_config["max_ar"],
            dataset_config["num_ar_buckets"],
            ",".join(map(str, dataset_config["frame_buckets"])),
            dataset_config["directory"][0]["path"],
            dataset_config["directory"][0]["num_repeats"]
        )

    # 生成带标记的 TOML 输出
    def generate_marked_config(current_config, default_config):
        toml_str = ""
        for key, value in current_config.items():
            default_value = default_config.get(key)
            if isinstance(value, dict):
                toml_str += f"{key} = [\n"
                for sub_key, sub_value in value.items():
                    sub_default = default_config.get(key, {}).get(sub_key)
                    marker = " ☆" if sub_value != sub_default else ""
                    toml_str += f"    {sub_key} = {repr(sub_value)},{marker}\n"
                toml_str += "]\n"
            else:
                marker = " ☆" if value != default_value else ""
                toml_str += f"{key} = {repr(value)}{marker}\n"
        return toml_str

    model_type.change(
        fn=update_configs_with_default,
        inputs=model_type,
        outputs=[
            output_dir, dataset, epochs, micro_batch_size_per_gpu, pipeline_stages, gradient_accumulation_steps,
            gradient_clipping, warmup_steps, blocks_to_swap, eval_every_n_epochs, eval_before_first_step,
            eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps, save_every_n_epochs,
            checkpoint_every_n_minutes, activation_checkpointing, save_dtype, caching_batch_size,
            video_clip_mode, ckpt_path, dtype, transformer_dtype, timestep_sample_method, rank, adapter_dtype,
            init_from_existing, optimizer_type, lr, betas_0, betas_1, weight_decay, eps,
            partition_method, partition_split, llm_path, eval_datasets, disable_block_swap_for_eval, gradient_release,
            resolutions, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets, directory_path, num_repeats
        ]
    )
    
    preview_btn.click(
        fn=lambda *args: (
            generate_marked_config(update_train_config(*args[:38]), default_train_config),
            generate_marked_config(update_dataset_config(*args[38:46]), default_dataset_config)
        ),
        inputs=[
            output_dir, dataset, epochs, micro_batch_size_per_gpu, pipeline_stages, gradient_accumulation_steps,
            gradient_clipping, warmup_steps, blocks_to_swap, eval_every_n_epochs, eval_before_first_step,
            eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps, save_every_n_epochs,
            checkpoint_every_n_minutes, activation_checkpointing, save_dtype, caching_batch_size,
            video_clip_mode, ckpt_path, dtype, transformer_dtype, timestep_sample_method, rank, adapter_dtype,
            init_from_existing, optimizer_type, lr, betas_0, betas_1, weight_decay, eps,
            partition_method, partition_split, llm_path, eval_datasets, disable_block_swap_for_eval, gradient_release,
            resolutions, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets, directory_path, num_repeats
        ],
        outputs=[train_config_output, dataset_config_output]
    )
    
    save_btn.click(
        fn=lambda *args: (
            train_config := update_train_config(*args[:38]),
            dataset_config := update_dataset_config(*args[38:46]),
            save_configs(train_config, dataset_config, args[46]),
            generate_marked_config(train_config, default_train_config),
            generate_marked_config(dataset_config, default_dataset_config)
        ),
        inputs=[
            output_dir, dataset, epochs, micro_batch_size_per_gpu, pipeline_stages, gradient_accumulation_steps,
            gradient_clipping, warmup_steps, blocks_to_swap, eval_every_n_epochs, eval_before_first_step,
            eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps, save_every_n_epochs,
            checkpoint_every_n_minutes, activation_checkpointing, save_dtype, caching_batch_size,
            video_clip_mode, ckpt_path, dtype, transformer_dtype, timestep_sample_method, rank, adapter_dtype,
            init_from_existing, optimizer_type, lr, betas_0, betas_1, weight_decay, eps,
            partition_method, partition_split, llm_path, eval_datasets, disable_block_swap_for_eval, gradient_release,
            resolutions, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets, directory_path, num_repeats,
            model_type
        ],
        outputs=[save_message, train_config_output, dataset_config_output]
    )
    
    train_btn.click(fn=start_training, inputs=[model_type, num_gpus], outputs=train_output)

# 启动 Gradio 界面
demo.launch()