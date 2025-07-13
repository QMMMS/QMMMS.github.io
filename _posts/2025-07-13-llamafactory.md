---
title: LLaMA-Factory 实战记录
date: 2025-07-13 08:00:00 +0800

categories: [经验与总结]
tags: [经验, LLM]
media_subpath: "/assets/img/posts/2025-07-13-llamafactory"
---

本文旨在完整记录一次利用开源框架LLaMA-Factory对Qwen2.5-VL-7B-Instruct模型进行参数高效微调（Parameter-Efficient Fine-tuning, PEFT）的全过程。内容涵盖环境配置、任务定义、数据准备、训练策略、过程监控、推理验证与结果分析。

## 环境准备

使用了如下软硬件及模型资源：

- **硬件环境**: 本次实践基于8张NVIDIA A800（80GB显存）GPU服务器。
- **微调框架**: LLaMA-Factory，一个集成了多种微调方法的用户友好型开源框架。具体部署方式请遵循其官方GitHub仓库指南：https://github.com/hiyouga/LLaMA-Factory
- **基础模型**: Qwen2.5-VL-7B-Instruct，由阿里巴巴通义千问团队开源的70亿参数视觉语言模型。模型权重（约16GB）可通过ModelScope进行下载：https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct

## 任务定义

本次微调的核心任务是一个复杂的、基于规则的视觉二分类问题。模型需要根据输入的图片和一系列详尽的文本要求，判断图片中的场景陈列是否合规。如下这个例子非真实数据：

**输入**：

```
<图片>判断以上图片是否满足要求：

1. 只存在正方体、圆柱、立体圆环、五星徽章、台灯状立体，不能出现其他物体
2. 正方体与圆柱位于前方，横向排列，面积大于1m*1m
3. 五星徽章需放在立体圆环里面
4. 左上角水印，展示每个物体数量，需要和图片中物体对应
5. 左下角水印，展示台灯状立体在图片中的位置，需要和图片中物体对应
6. 右上角水印，为正方体与圆柱的数量加和，需要和图片中物体对应
7. 右下角水印，为五星徽章和立体圆环的数量乘积，需要和图片中物体对应
```

**输出**：

```
满足要求/不满足要求
```

此任务不仅考验模型的视觉识别能力，更对其遵循复杂指令、进行逻辑推理和空间关系判断的能力提出了较高要求。

## 数据准备

为了适配模型的训练，我们将图文数据对整理为`ShareGPT`格式，并存为`my_dataset.json`文件。该格式以多轮对话的形式组织数据，结构清晰。

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "你是一个视觉分类任务专家。请根据以下要求判断图片是否满足要求。"
      },
      {
        "role": "user",
        "content": "<image>判断以上图片是否满足要求：\n1. 只存在正方体、圆柱、立体圆环、五星徽章、台灯状立体，不能出现其他物体\n2. 正方体与圆柱位于前方，横向排列，面积大于1m*1m\n3. 五星徽章需放在立体圆环里面\n4. 左上角水印，展示每个物体数量，需要和图片中物体对应\n5. 左下角水印，展示台灯状立体在图片中的位置，需要和图片中物体对应\n6. 右上角水印，为正方体与圆柱的数量加和，需要和图片中物体对应\n7. 右下角水印，为五星徽章和立体圆环的数量乘积，需要和图片中物体对应\n请只输出如下格式：\"满足要求\", 或 \"不满足要求\"。不需要额外解释。请严格按照格式输出，否则判为无效答案。"
      },
      {
        "role": "assistant",
        "content": "满足要求"
      }
    ],
    "images": [
      "/data/img_01.jpg"
    ]
  },
]
```

- `messages`: 包含系统（system）、用户（user）、助手（assistant）三方角色的对话历史。
- `images`: 包含与对话相关的图片路径列表。特殊占位符`<image>`在`user`内容中指代图片。

同时，需要在`data/dataset_info.json`中注册此数据集，以便框架能够正确解析：

```json
"my_dataset": {
    "file_name": "my_dataset.json",
    "formatting": "sharegpt",
    "columns": {
        "messages": "messages",
        "images": "images"
    },
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system"
    }
}
```

##  训练配置

配置如下

```yaml
### model
model_name_or_path: /data/modelscope/Qwen2___5-VL-7B-Instruct
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### dataset
dataset: my_dataset
template: qwen2_vl
cutoff_len: 131072
max_samples: 3000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: /output/my_task
logging_steps: 10
save_steps: 100
plot_loss: true
overwrite_output_dir: false
save_only_model: false
report_to: tensorboard  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 30
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 50
```

核心目标：在一个强大且预训练好的视觉语言模型（**Qwen2.5-VL-7B-Instruct**）的基础上，使用 **LoRA** 的高效微调技术，来训练它适应我们的任务。整个过程属于**监督微调 (SFT)** 阶段。**LoRA**技术冻结模型主体，只训练新增的、极少量的“适配器”参数，从而大大提升训练效率。

一些配置的解释

- **`trust_remote_code: true`**: 一个安全设置，对于像Qwen这样包含自定义代码的模型，必须设为 `true` 才能正确加载其独特的模型结构。
- **`stage: sft`**: 指定训练阶段为**监督微调 (Supervised Fine-Tuning)**，即模型从“输入-输出”样本对中学习。
- **`lora_rank: 8`**: LoRA方法的关键超参数，定义了“适配器”的规模。`8` 是一个在效果和资源消耗上都很平衡的常用值。
- **`lora_target: all`**: Llama Factory 的一个便捷设置，它会自动找出模型中所有适合应用LoRA的层（如注意力层）并进行适配。
- **`template: qwen2_vl`**: 指定了将数据格式化成 Qwen2-VL 模型能理解的特定提示（Prompt）格式。
- **`preprocessing_num_workers` / `dataloader_num_workers`**: 分别是数据预处理和加载时使用的并行进程数，用于加速数据准备。
- **`per_device_train_batch_size: 1`**: 每个GPU设备一次处理1个样本。
- **`gradient_accumulation_steps: 8`**: 梯度累积8步之后再更新一次模型参数。这会形成 `1 * 8 = 8` 的**有效批大小 (Effective Batch Size)**，可以在不增加显存消耗的情况下，达到使用更大批次训练的稳定效果。
- **`lr_scheduler_type: cosine`**: 学习率调度策略。`cosine` 指学习率会按照余弦曲线平滑下降，有助于模型在训练后期更好地收敛。
- **`warmup_ratio: 0.1`**: 预热比例。在总训练步数的的前10%里，学习率会从0线性增长到设定的 `1.0e-4`，这有助于训练初期的稳定。
- **`bf16: true`**: 使用 BF16 混合精度进行训练，可以在支持的硬件上大幅提升训练速度并节省显存。
- **`eval_steps: 50`**: 每训练50步，就在验证集上进行一次评估。这可以帮助你密切监控模型是否出现过拟合。记录损失数据

## 训练

在LLaMA-Factory项目根目录下，通过以下命令启动训练。为保证长时间任务的稳定，推荐配合`screen`或`tmux`使用：

```bash
llamafactory-cli train train_config.yaml
```

通过观察日志，我们看到训练过程的关键信息。日志的起始部分揭示了本次训练任务的计算环境与核心并行策略

```
INFO 07-10 22:38:45 [__init__.py:239] Automatically detected platform cuda.
[INFO|2025-07-10 22:38:47] llamafactory.cli:143 >> Initializing 8 distributed tasks at: 127.0.0.1:44329
[INFO|2025-07-10 22:38:54] llamafactory.hparams.parser:383 >> Process rank: 0, world size: 8, device: cuda:0, distributed training: True, compute dtype: torch.bfloat16
```

- **分布式数据并行 (Distributed Data Parallel - DDP)**: 日志中的 `world size: 8` 和 `distributed training: True` 明确指出，本次训练采用了8个GPU进行分布式数据并行。在此策略下，模型被完整复制到每个GPU上，而训练数据集被切分并分配给各个进程。各进程独立完成前向和后向传播后，通过All-Reduce操作同步梯度，从而实现训练加速。
- **混合精度训练 (Mixed-Precision Training)**: `compute dtype: torch.bfloat16` 表明训练采用了BF16（BFloat16）混合精度。该技术使用16位浮点数进行大部分计算，显著降低了显存占用并提升了在现代GPU（如NVIDIA Ampere及更新架构）上的计算吞吐量，同时保持了接近FP32的训练稳定性。

日志接着展示了对视觉语言模型（VLM）特有的数据处理流程。

```
[INFO|tokenization_utils_base.py:2058] 2025-07-10 22:38:54,229 >> loading file vocab.json ......
[INFO|image_processing_base.py:379] 2025-07-10 22:38:54,519 >> loading configuration file /data/modelscope/Qwen2___5-VL-7B-Instruct/preprocessor_config.json
[INFO|image_processing_base.py:434] 2025-07-10 22:38:54,521 >> Image processor Qwen2VLImageProcessor {...}
[INFO|processing_utils.py:876] 2025-07-10 22:38:55,269 >> Processor Qwen2_5_VLProcessor:....
- tokenizer: Qwen2TokenizerFast(name_or_path='/data/modelscope/Qwen2___5-VL-7B-Instruct', vocab_size=151643, model_max_length=131072, is_fast=True, padding_side='right', truncation_side='right'.....
[INFO|2025-07-10 22:38:55] llamafactory.data.loader:143 >> Loading dataset my_dataset.json...
Generating train split: 2950 examples [00:00, 43902.74 examples/s]
Converting format of dataset (num_proc=16): 100%|██████████| 2950/2950 [00:00<00:00, 15694.34 examples/s]

training example:
<|im_start|>system
你是一个视觉分类任务专家。请根据以下要求判断图片是否满足要求。<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>判断以上图片是否满足要求：
1. 只存在正方体、圆柱、立体圆环、五星徽章、台灯状立体，不能出现其他物体
2. 正方体与圆柱位于前方，横向排列，面积大于1m*1m
3. 五星徽章需放在立体圆环里面
4. 左上角水印，展示每个物体数量，需要和图片中物体对应
5. 左下角水印，展示台灯状立体在图片中的位置，需要和图片中物
6. 右上角水印，为正方体与圆柱的数量加和，需要和图片中物体对应
7. 右下角水印，为五星徽章和立体圆环的数量乘积，需要和图片中物体对应
请只输出如下格式："满足要求", 或 "不满足要求"。不需要额外解释。请严格按照格式输出，否则判为无效答案。<|im_end|>
<|im_start|>assistant
满足要求<|im_end|>

[INFO|modeling_utils.py:1151] 2025-07-10 22:39:28,462 >> loading weights file /data/modelscope/Qwen2___5-VL-7B-Instruct/model.safetensors.index.json
Loading checkpoint shards: 100%|██████████| 5/5 [00:04<00:00,  1.16it/s]
```

- **视觉语言模型 (VLM) 数据流**: 日志中同时加载了`tokenizer`（文本分词器）和`Image processor`（图像处理器），证实了这是一个多模态任务。`training example`清晰地展示了图文混合输入的格式，其中文本部分被结构化为多轮对话，而图像则通过`<|vision_start|><|image_pad|><|vision_end|>`等特殊Token（Special Tokens）嵌入到文本序列中，实现了图文信息的统一表示。
- **模型分片加载**: `Loading checkpoint shards: 100%|...| 5/5` 表明基础模型（8.3B参数）的权重被存储为5个分片文件（shards），这是存储和加载大规模模型的标准做法。

本日志的核心亮点在于展示了LoRA技术的应用，这是PEFT中最具代表性的方法之一。

```
[INFO|2025-07-10 22:39:32] llamafactory.model.adapter:143 >> Upcasting trainable params to float32.
[INFO|2025-07-10 22:39:32] llamafactory.model.adapter:143 >> Fine-tuning method: LoRA
[INFO|2025-07-10 22:39:32] llamafactory.model.model_utils.misc:143 >> Found linear modules: q_proj,k_proj,gate_proj,down_proj,v_proj,o_proj,up_proj
[INFO|2025-07-10 22:39:32] llamafactory.model.model_utils.visual:143 >> Set vision model not trainable: ['visual.patch_embed', 'visual.blocks'].
[INFO|2025-07-10 22:39:32] llamafactory.model.model_utils.visual:143 >> Set multi model projector not trainable: visual.merger.
[INFO|2025-07-10 22:39:33] llamafactory.model.loader:143 >> trainable params: 20,185,088 || all params: 8,312,351,744 || trainable%: 0.2428
```

- **低秩适配 (Low-Rank Adaptation - LoRA)**: 该方法的核心思想是冻结预训练模型的绝大部分参数，仅在模型的特定层（此处为Transformer中的`q_proj`, `k_proj`, `v_proj`等线性层）旁注入小规模、可训练的低秩矩阵。
- **训练效率**: 日志给出了决定性的数据：在总计超过83亿的参数中，仅有约2000万个参数被更新，**可训练参数占比仅为0.24%**。这极大地降低了微调所需的计算资源和时间。
- **模块冻结策略**: 日志还显示，视觉编码器（`visual.blocks`等）被设置为不可训练。这是一种常见的VLM微调策略，旨在保留强大的、经预训练获得的视觉特征提取能力，仅对语言模块和图文融合模块进行适配。

训练执行阶段的日志揭示了为克服显存限制而采用的优化技术。

```
[INFO|trainer.py:2409] 2025-07-10 22:39:34,131 >> ***** Running training *****
[INFO|trainer.py:2410] 2025-07-10 22:39:34,131 >>   Num examples = 2,655
[INFO|trainer.py:2411] 2025-07-10 22:39:34,131 >>   Num Epochs = 30
[INFO|trainer.py:2412] 2025-07-10 22:39:34,132 >>   Instantaneous batch size per device = 1
[INFO|trainer.py:2415] 2025-07-10 22:39:34,132 >>   Total train batch size (w. parallel, distributed & accumulation) = 64
[INFO|trainer.py:2416] 2025-07-10 22:39:34,132 >>   Gradient Accumulation steps = 8
[INFO|trainer.py:2417] 2025-07-10 22:39:34,132 >>   Total optimization steps = 1,230
[INFO|trainer.py:2418] 2025-07-10 22:39:34,135 >>   Number of trainable parameters = 20,185,088
100%|██████████| 1230/1230 [2:08:35<00:00,  6.27s/it]
```

- **梯度累积 (Gradient Accumulation)**: 尽管单张GPU的批大小（`Instantaneous batch size per device`）仅为1，但通过设置梯度累积步数为8，实现了`8 (GPUs) * 1 (batch/GPU) * 8 (accumulation steps) = 64`的有效批大小（Effective Batch Size）。该技术在多个mini-batch上计算并累加梯度，最后统一执行一次参数更新，从而在不增加显存消耗的前提下，模拟大批量训练的稳定化效果。

训练结束时，日志输出了关键的性能指标。

```
***** train metrics *****
  epoch                    =      29.2892
  total_flos               = 3173165210GF
  train_loss               =       0.0185
  train_runtime            =   2:08:35.51
  train_samples_per_second =       10.323
  train_steps_per_second   =        0.159
  
***** eval metrics *****
  epoch                   =    29.2892
  eval_loss               =     0.0914
  eval_runtime            = 0:00:18.17
  eval_samples_per_second =     16.228
  eval_steps_per_second   =      2.035  
```

- **模型收敛性**: 训练集最终损失（`train_loss`）为0.0185，这是一个极低的值，表明模型在训练数据上已充分收敛。
- **泛化能力**: 验证集损失（`eval_loss`）为0.0914。较高于`train_loss`，不过尚处于正常现象

训练后，默认在`saves`文件夹下，会保存训练的`checkpoint`，以及训练结果等文件。注意我们保存的只是 adapter 权重，并不包括大模型原本的权重，后续推理时需要同时加载原本大模型和我们训练的 adapter。

在`tensorboard`中，我们可以随时追踪训练参数，如下

![](llm_train_trensorboard.png)

右边三张展示了训练过程中的核心指标：

- **`train/loss` (训练损失)**: 该曲线表现出理想的收敛行为。损失值从一个较高的初始点（>0.1）开始，在前200步内迅速下降，随后持续平缓减小，最终在约400步后收敛至接近于零的水平。这表明模型具有足够强的拟合能力，能够完美地学习并记忆训练数据集。
- **`train/grad_norm` (训练梯度范数)**: 梯度范数曲线与训练损失的变化高度相关。在训练初期，损失函数曲面陡峭，梯度范数出现数次剧烈峰值。随着训练损失收敛至平坦区域，梯度范数也迅速衰减并稳定在接近零的水平。这进一步证实了模型在训练集上已达到收敛，优化器不再需要进行大幅度的参数更新。
- **`train/learning_rate` (学习率)**: 图中展示了标准的**带预热（Warmup）的余弦退火（Cosine Annealing）**学习率调度策略。学习率在初始阶段线性增长至峰值（1.0e-4），随后平滑衰减。该调度策略本身是当前主流且有效的，旨在帮助模型在训练初期稳定，在后期精细收敛。

评估模型泛化能力的关键在于验证集上的表现：

- **`eval/loss` (验证损失)**: 验证损失在训练开始后，随训练损失一同下降，并在**约200-300步时达到其最小值（约0.06）**。这一点是模型泛化能力的“最佳时刻”（Sweet Spot）。然而，在此之后，验证损失**不再下降，反而开始持续、显著地上升**，最终在训练结束时达到了比初始值更高的水平（>0.09）。出现了**过拟合（Overfitting）**现象。在验证准确率和其他指标的时候，可以考虑验证最后的checkpoint以及验证300step的checkpoint

## 推理和验证

我们可以使用 `vllm` 把我们训练的 adapter 和大模型权重加载在一块进行推理。主要分为两种方式

- 在 LLaMA-Factory 中提供的 web-ui 中手动问答
- 利用脚本自动批次获取大量结果

在有明确要求的业务场景数据集中，为了评估严谨，我们用脚本来获取结果。命令是：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -u /data/LLaMA-Factory/scripts/vllm_infer.py \
    --model_name_or_path /data/modelscope/Qwen2___5-VL-7B-Instruct \
    --adapter_name_or_path /output/my_task \
    --template qwen2_vl \
    --dataset my_test \
    --pipeline_parallel_size 1 \
    --vllm_config '{"tensor_parallel_size": 4}' \
    --save_name pred.json
```

命令比较简单，使用 LLaMA-Factory 提供的示例脚本 `vllm_infer.py`，利用4个GPU（`tensor_parallel_size: 4`）对测试集`my_test`进行推理。输出的文件 `pred.json` 中，每一行都是 json 格式的完整回答过程。后续可通过编写简单的Python脚本，解析输出文件并计算准确率（Precision）、召回率（Recall）等业务指标，从而对模型性能进行量化评估。
