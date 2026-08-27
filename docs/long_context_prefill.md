# 单设备长上下文 prefill profiling

## 运行

在已安装项目 ML 依赖、准备好分片权重的目标设备上，从项目根目录调用：

```python
import torch
from utils.node_profiler import NodeProfiler

profiler = NodeProfiler(
    "shards/Llama-3___2-3B-Instruct_float16",
    device="cuda:0",
    dtype=torch.float16,
)
csv_path = profiler.profile_long_context_prefill(
    layer_num=6,
    model_type="llama3.2ins",
    use_thinking_question=False,
    sample_index=0,
    initial_context_length=32,
    context_length_unit="auto",
    repeat_num=3,
)
print(csv_path)
```

`layer_num=6` 精确加载 `block_0.pth` 至 `block_5.pth`。用 `layer_num=profiler.layer_num` 测完整模型时，另测包含 LM head 的首 token 时延。静态可装载层数只是选 N 的参考，长上下文的 KV 与中间张量仍需要余量。

默认从 `datasets/LongBench-Pro/longbench_pro.json` 按零基 `sample_index` 流式读取一条记录，只在内存中保留当前读取的记录/缓冲区，不一次加载整个数据集数组。可以通过 `dataset_path` 指定其他同结构的 JSON 数组。

## 输入规则与终止条件

选择 `context` 和完整的 `question_nonthinking`，`use_thinking_question=True` 时选择 `question_thinking`。数据中的 `answer` 不参与输入。拼接内容为：

```text
<context 前缀>\n\n\n\n<完整 question>
```

将其作为一条 user 消息，调用 `tokenizer.apply_chat_template(..., tokenize=True, add_generation_prompt=True, return_tensors="pt", truncation=False)`。`model_type` 当前只接受 `llama3.2ins`，其他值抛出 `NotImplementedError`。完整 question 及 chat template 的特殊 token 全部计入实际输入 token 数。

`context_length_unit="auto"` 根据样本 `language` 判断：Chinese/zh/zh-cn 使用 Unicode 字符数，其他使用空白分隔词数。也可显式指定 `words` 或 `characters`。按词截断使用原文字符边界，保留换行和原始空白；中文的 `context_word_count` 仍是空白分隔计数，不代表中文分词结果。

长度从 `min(initial_context_length, 完整长度)` 开始，每次翻倍，最后一次精确覆盖完整原文。开始扫描前，仅在这个初始长度预热一次；完整模型的这次预热包含 LM head。每个长度、每种输出模式执行 `repeat_num` 次正式测量，每次使用新 KV。

出现以下任一情况停止：完整 context 所有正式测量成功、捕获到 CUDA OOM/主机 MemoryError，或模板后的输入超过配置 `max_position_embeddings`。超过模型窗口记录为 `context_limit` 并在前向前停止；它与 OOM 是不同原因。其他异常保留诊断元数据后向调用者抛出。

## 时延与内存口径

连续计时覆盖输入格式化、tokenize、本机输入张量搬运、embedding、RoPE、N 层 forward。模型加载、读取样本、生成截断前缀、写文件和内存清理位于计时外。计时中不插入逐层测量、日志、磁盘写入或显存采集。

- `include_lm_head=False`：终点为所装载分片的最终 hidden states。完整模型的 shard 同时包含 final norm；部分层的 shard 不含 final norm。
- `include_lm_head=True`：只在 N 等于模型总层数时测量；终点为首个 greedy token ID 已生成并读回。LM head 只投影最后一个输入位置，不构造全部位置的词表 logits；此口径应与既有构造全位置 logits 的 worker 区分。文本 decode 和打印位于此测量范围之外。
- 完整模型的两种模式共用相同常驻权重，包括 LM head，确保权重驻留一致；`include_lm_head` 表示本轮是否执行 head。若完整权重装载失败，同名 JSON 的 `stage=load_components` 可用于定位。
- `prefill_latency_ms` 为整个连续区间的毫秒数；`prefill_latency_per_layer_ms = prefill_latency_ms / loaded_layer_count`，包含输入/输出开销的层均摊值。
- CUDA 计时前同步并重置峰值统计；完成输出、同步、停止计时后，先读取峰值，再释放张量和 KV。统计包括常驻权重和该轮峰值，CPU/不可用计数器留空。PyTorch allocated/reserved 并不等于设备总内存或 Jetson 系统 RAM 峰值。

当前前向沿用 `LlamaShardPart` 的掩码行为。解释结果前请阅读 [attention mask 优先事项](TODO.md)。

## 文件与字段

每次调用在 `results/profiling/long_context_prefill/<时间戳>-<唯一后缀>/` 创建新目录，`output_dir` 可以指定根目录。CSV 文件命名为：

```text
long_context_prefill_latency-<设备型号>-<N>layers.csv
```

设备名优先尝试 Jetson device-tree，再尝试 PyTorch CUDA 型号，最后回退到平台/设备类型；不调用 `nvidia-smi`。识别失败不会因为缺少 PC 命令而中断。可用 `device_label="Jetson Orin Nano"` 等显式覆盖，文件名中的不安全字符自动替换。

| 字段 | 含义 |
| --- | --- |
| `attempt_id`, `sample_id` | 本轮序号与数据集记录标识 |
| `phase`, `repeat_index` | `warmup` / `measure`；预热序号为 0，正式重复从 1 开始 |
| `model_type`, `question_type` | 模板适配器及选择的问题字段 |
| `context_length_unit`, `context_length` | 本轮原文截断单位及实际长度 |
| `context_word_count`, `context_char_count` | 前缀的空白分隔词数与 Unicode 字符数 |
| `input_token_count` | context、完整 question 与 chat template 组合后的实际 token 数 |
| `loaded_layer_count`, `include_lm_head` | 精确 Transformer 层数及本轮是否执行 LM head |
| `prefill_latency_ms`, `prefill_latency_per_layer_ms` | 总时延与按层均摊时延；失败行留空 |
| `is_full_context` | 本轮是否覆盖完整原文 |
| `status` | `success`、`oom` 或 `context_limit` |
| `peak_cuda_allocated_bytes`, `peak_cuda_reserved_bytes` | 计时结束后的 CUDA 峰值统计 |
| `first_token_id` | 带 LM head 成功测量的首 token ID |
| `failure_stage`, `error` | 可捕获失败的执行阶段和错误信息 |

同名 JSON 保存样本选择、设备、模型路径/配置、PyTorch 版本、attention backend、计时定义、运行状态和 `pending_attempt`。应将 CSV 与 JSON 一起归档。

## 强制结束后的分析与独立绘图

CSV 在每次完成后 `flush` + `fsync`。每轮高内存操作前，先原子更新 JSON 进度；模型加载前也先保存 CSV 表头和元数据。操作系统 kill 无法由 Python 捕获，先前已完成记录仍可用于分析。

若进程已经退出而 JSON 仍为 `running`，应将 `pending_attempt` 与 CSV 的 `attempt_id` 对照：某条 CSV 已落盘而 JSON 还没更新的短窗口中，同一 attempt 可能已经完成。没有完成记录的 pending 尝试只能标为“未完成、原因待确认”；需结合系统日志确认是否 OOM。被强制结束的轮次可能来不及记录实际 token 数。返回的 CSV/目录可用于重新分析，不提供自动续跑。

从项目根目录运行独立绘图脚本：

```bash
python test/long_context_prefill_result_fig.py "results/profiling/long_context_prefill/<运行目录>/<结果文件>.csv"
```

省略 CSV 参数时使用脚本中的 `DEFAULT_CSV_PATH`；可以在脚本顶部配置默认数据文件。可用 `--output "results/figures/prefill.png"` 指定输出位置，默认图片与 CSV 同名且后缀为 `.png`。相对 CSV 和输出路径都以项目根目录为基准，绝对路径直接使用。因此在 `test/` 目录也可执行 `python long_context_prefill_result_fig.py`，使用相同的默认输入。

绘图函数 `plot_long_context_prefill(csv_path, output_path=None)` 定义在这个脚本中，外部 Python 调用者也可导入该函数。脚本只使用标准库和 matplotlib，直接读取 CSV 与可选的同名 JSON。横轴是实际模型输入 token 数，纵轴是层均摊时延；按 hidden-state/LM-head 两种终点分别画正式重复的中位数及 min/max 区间，跳过预热和无效行，标记完整 context 与可用的停止位置。至少需要一条成功的正式测量；没有时给出明确错误。
