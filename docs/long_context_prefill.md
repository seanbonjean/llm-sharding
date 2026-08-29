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

### 固定输入的逐层时延

需要观察不同 Transformer 层的实际时延时，使用相同的 profiler：

```python
csv_path = profiler.profile_long_context_prefill_per_layer(
    layer_num=6,
    context_word_count=2048,
    model_type="llama3.2ins",
    use_thinking_question=False,
    sample_index=0,
    repeat_num=3,
)
print(csv_path)
```

`context_word_count` 指定从 `context` 开头保留的空白分隔词数；超过样本总词数时使用完整 context，并在 JSON 同时记录请求词数和实际词数。完整 question 与 chat template 仍全部保留，实际模型输入长度记录在 `input_token_count`。

该实验精确加载并按顺序执行 `[0, layer_num)`：先在计时外完成模板、tokenize、本机张量搬运、embedding 和 RoPE，再对每个 Transformer layer 分别执行“同步—开始计时—layer forward—同步—停止计时”。同一轮中的各层使用上一层输出和同一份新 KV Cache；下一轮重新创建 KV。固定输入完整预热一次，随后执行 `repeat_num` 次正式测量。

原方法会在 warmup 和每轮正式遍历后调用 `torch.cuda.empty_cache()`，适合保留偏冷分配口径。若要测量复用 CUDA allocator 显存块后的稳态逐层时延，可改用：

```python
csv_path = profiler.profile_long_context_prefill_per_layer_reuse_allocator_cache(
    layer_num=6,
    context_word_count=2048,
    model_type="llama3.2ins",
    use_thinking_question=False,
    sample_index=0,
    repeat_num=3,
)
```

该方法仍为 warmup 和每轮正式测量分别创建新的 `DynamicCache`，不会复用上一轮 KV 内容；它只保留已释放的 CUDA allocator 显存块供下一轮使用，并在整个实验结束后统一执行 `torch.cuda.empty_cache()`。两种方法使用相同的 CSV 结构和绘图脚本；同名 JSON 中的 `profile_type` 与 `clear_cuda_allocator_cache_between_attempts` 用于区分测试口径。

逐层模式不加载 final norm 和 LM head，即使 `layer_num` 等于模型总层数也是如此。每层边界的同步用于隔离 GPU 工作，因此 `layer_latency_ms` 的总和不能直接视为原始端到端函数的 `prefill_latency_ms`；后者不中断地计入输入准备和可选输出模块。

默认结果目录和文件名为：

```text
results/profiling/long_context_prefill_per_layer/<运行目录>/
long_context_prefill_per_layer_latency-<设备型号>-<N>layers-<实际词数>words.csv
```

CSV 中一个完整 attempt 为每个 `layer_index` 保存一行。`layer_index` 为零基索引，`layer_latency_ms` 为该层同步完成的毫秒时延；`peak_cuda_allocated_bytes` 和 `peak_cuda_reserved_bytes` 是整轮结束后读取的 trial 级峰值，会重复写入该轮的各层行。warmup 行保留但不参与绘图。失败行不包含单层时延；若捕获到 OOM，会记录发生时的层索引和阶段。JSON 的持久化、设备识别和强制结束语义与端到端长上下文实验一致。

可以先在脚本顶部通过 `DEFAULT_RESULT_DIRECTORY` 指定实验结果文件夹，然后直接运行：

```bash
python analysis/long_context_prefill_per_layer_fig.py
```

也可以将其他实验结果文件夹或完整逐层 CSV 作为位置参数传入。实验文件夹中应当只有一份匹配 `long_context_prefill_per_layer_latency-*.csv` 的结果文件。图中横轴为零基层索引，纵轴为单层 prefill 时延；折线是正式重复的中位数，阴影是 min/max。默认 PNG 与 CSV 同名同目录，可用 `--output` 指定其他位置。

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

端到端与逐层长上下文前向目前均使用 `attention_mask=None`。解释结果前请阅读 [attention mask 优先事项](TODO.md)。

## 文件与字段

每次调用在 `results/profiling/long_context_prefill/<时间戳>-<唯一后缀>/` 创建新目录，`output_dir` 可以指定根目录。CSV 文件命名为：

```text
long_context_prefill_latency-<设备型号>-<N>layers.csv
```

自动设备标签优先使用识别出的 Orin 系列和容量：Nano 使用 `Orin_Nano_8G` / `Orin_Nano_4G`；已确认 Nano 系列但容量未知时使用 `Orin_Nano`。已识别的 NX 使用 `Orin_NX_8G` / `Orin_NX_16G`，容量未知时使用 `Orin_NX`。其余情况依次尝试 Jetson device-tree、PyTorch CUDA 型号和平台/设备类型，不调用 `nvidia-smi`；可选系统接口缺失不会中断测量。显式 `device_label="edge-A"` 优先覆盖自动标签，文件名中的不安全字符自动替换。

例如 8G Nano 加载六层时，文件名为 `long_context_prefill_latency-Orin_Nano_8G-6layers.csv`，元数据文件同名并使用 `.json` 后缀；单设备绘图的默认输出同名并使用 `.png` 后缀。文件标签和 JSON 复用同一次硬件识别结果。该命名规则用于新生成结果，已有实验文件保持原路径；绘图同时支持两种命名。

Orin Nano 的内存版本和识别依据保存到同名 JSON，识别发生在模型加载及计时之前：

1. 优先只读查询系统公开的模块 EEPROM（支持的格式版本为 2.0），再查询设备树中启动时填入的模块 ID。P3767-0003 / P3767-0005 对应 Nano 8G，P3767-0004 对应 Nano 4G；P3767-0000 / P3767-0001 属于 NX，不会因使用 Nano 载板而按 Nano 处理。映射依据 [NVIDIA EEPROM 文档](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/HR/JetsonEepromLayout.html)。
2. 模块 ID 不可用时，读取设备树 `model` 中明确的 `Orin Nano 4GB` / `8GB` 等型号文本。兼容 `/proc/device-tree` 和 `/sys/firmware/devicetree/base`；根 `compatible` 只帮助识别平台，不以共用 DTB 的型号确定内存版本。
3. 系统信息确认是 Orin Nano、但仍没有版本时，读取 `/proc/meminfo` 的 `MemTotal`，必要时回退到 `os.sysconf` 的物理页数和页大小。依据总内存而非空闲内存、CUDA 已分配内存或进程内存限制推断：3.0–4.5 GiB 视为标称 4G，6.0–8.5 GiB 视为标称 8G，因此 3.x / 7.x GiB 均可识别。该区间是容量推断规则，不代表标称容量的严格等值。
4. 不在这些区间、系统信息不足、字段不可读或接口不存在时保留未知状态。设备识别不运行外部命令，不提升权限，不因可选系统接口缺失中断测量。

Jetson 运行的 JSON 附加字段：

| 字段 | 含义 |
| --- | --- |
| `jetson_model` | 识别出的简短系列，例如 `Jetson Orin Nano`；无法确定时为 `null` |
| `jetson_module_id` | 物理模块 ID，例如 `P3767-0005`；不可用时为 `null` |
| `jetson_memory_variant` | Nano 的 `4G` / `8G`；已知 NX 模块可为 `8G` / `16G`；未知为 `null` |
| `jetson_memory_variant_source` | `eeprom:<路径>`、`device_tree_ids:<路径>`、`device_tree_model:<路径>`、`system_memory_total_inferred` 或 `unknown` |
| `system_memory_total_bytes` | Orin Nano 操作系统可见的总物理内存字节数，与标称容量及当前空闲量不同；不可用时为 `null` |

PC 等其他平台不追加 Jetson 字段。显式 `device_label` 只覆盖文件标签，不改变实际硬件识别；EEPROM 仅提取版本和产品编号，不保存设备序列号或 MAC 地址。

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

### Windows 进度文件写入

每次更新先写入同目录的 `.json.tmp`，执行 `flush` / `fsync` 并关闭文件，再用 `os.replace()` 替换正式 JSON。若替换遇到 Windows 的拒绝访问、共享冲突或锁冲突（错误码 5、32、33），本次更新最多尝试六次；五次重试前依次等待 0.05、0.1、0.2、0.4、0.8 秒，累计退避 1.55 秒。写盘和等待均在 prefill 计时区间外，调用参数不变。

持续失败时抛出 `PermissionError`，提示检查文件占用、只读属性及目录权限，同时保留已提交 JSON（若已有）和待提交的 `.json.tmp`。临时文件保存尝试提交的快照，可用于诊断；正式 JSON 仍代表最后一次成功提交的状态。其他文件系统错误以及临时文件写入/同步错误立即抛出。

运行中发生异常时，会另行尝试保存一次错误状态快照，该次更新也遵循上述重试上限；若仍失败，向 stderr 报告二次写入错误并继续抛出最初异常。此前已落盘的 CSV 记录仍可分析。文件替换问题持续出现时，应检查相关文件占用、只读属性和目录权限，或通过 `output_dir` 选择具备写入和替换权限的结果目录后重新运行。

## 强制结束后的分析与独立绘图

CSV 在每次完成后 `flush` + `fsync`。每轮高内存操作前，先原子更新 JSON 进度；模型加载前也先保存 CSV 表头和元数据。操作系统 kill 无法由 Python 捕获，先前已完成记录仍可用于分析。

若进程已经退出而 JSON 仍为 `running`，应将 `pending_attempt` 与 CSV 的 `attempt_id` 对照：某条 CSV 已落盘而 JSON 还没更新的短窗口中，同一 attempt 可能已经完成。没有完成记录的 pending 尝试只能标为“未完成、原因待确认”；需结合系统日志确认是否 OOM。被强制结束的轮次可能来不及记录实际 token 数。返回的 CSV/目录可用于重新分析，不提供自动续跑。

从项目根目录运行独立绘图脚本：

```bash
python analysis/long_context_prefill_result_fig.py "results/profiling/long_context_prefill/<运行目录>/<结果文件>.csv"
```

省略 CSV 参数时使用脚本中的 `DEFAULT_CSV_PATH`；可以在脚本顶部配置默认数据文件。可用 `--output "results/figures/prefill.png"` 指定输出位置，默认图片与 CSV 同名且后缀为 `.png`。相对 CSV 和输出路径都以项目根目录为基准，绝对路径直接使用。因此在 `analysis/` 目录也可执行 `python long_context_prefill_result_fig.py`，使用相同的默认输入。

绘图函数 `plot_long_context_prefill(csv_path, output_path=None)` 定义在这个脚本中，外部 Python 调用者也可导入该函数。脚本只使用标准库和 matplotlib，直接读取 CSV 与可选的同名 JSON。横轴是实际模型输入 token 数，纵轴是层均摊时延；按 hidden-state/LM-head 两种终点分别画正式重复的中位数及 min/max 区间，跳过预热和无效行，标记完整 context 与可用的停止位置。至少需要一条成功的正式测量；没有时给出明确错误。

设备图例直接在绘图函数内生成。单设备图保留终点区别，例如 `Jetson Orin Nano 8G | hidden states`、`Jetson Orin Nano 8G | first token (LM head)`；PC 使用 `Quadro RTX 5000` 等简短名称。分析代码仅读取 CSV/JSON，不查询绘图主机的硬件。

## 双设备共同范围对比

在脚本顶部配置 `DEFAULT_CSV_PATH_A` 和 `DEFAULT_CSV_PATH_B`，然后从项目根目录直接运行：

```bash
python analysis/long_context_prefill_comparison_fig.py
```

两个常量当前分别指向 Jetson 的 `20260827-231615-zmq7xa6z` 和 Quadro 的 `20260828-003453-_krn076g` 结果。也可通过命令行覆盖：

```bash
python analysis/long_context_prefill_comparison_fig.py "<设备 A CSV>" "<设备 B CSV>" --label-a "Jetson Orin Nano" --label-b "Quadro RTX 5000" --output "results/figures/prefill_comparison.png"
```

脚本和 Python 函数 `compare_long_context_prefill(csv_path_a=None, csv_path_b=None, output_path=None, include_lm_head=False, label_a=None, label_b=None)` 均只依赖标准库和 matplotlib，无需 torch 或权重。从项目根目录也可用 `from analysis.long_context_prefill_comparison_fig import compare_long_context_prefill` 调用；直接 `compare_long_context_prefill()` 使用两个默认常量。

两个 CSV 参数均可省略；仅传一个时覆盖 A，B 使用默认常量。相对输入/输出路径以项目根目录为基准，不受启动目录影响。绘图函数从同名 JSON 生成简短设备名，例如 `Jetson Orin Nano 8G`、`Quadro RTX 5000`；缺少 JSON 时从 CSV 文件名提取设备名。`--label-a` / `--label-b` 可指定昵称，Orin 的已知版本会追加到昵称后。

两种图均优先读取记录中的 `jetson_memory_variant`；历史设备名若明确含有 `4G` / `8GB` 等容量，也可作为回退。旧 JSON 没有版本证据时显示 `Jetson Orin Nano`，图例中省略容量。旧结果需要在确认目标设备后补充版本信息；对比图也可用 `--label-a "Jetson Orin Nano 8G"` 显式指定已确认的版本。重新运行 profiling 会自动保存新增硬件字段。

默认输出目录由 `DEFAULT_OUTPUT_DIR` 指定，为 `results/profiling/long_context_prefill/comparisons/`。文件名按 A、B 顺序使用各自 CSV 父目录中 `YYYYMMDD-HHMMSS-` 后的完整 hash，保留开头的下划线。例如上述默认输入生成：

```text
results/profiling/long_context_prefill/comparisons/long_context_prefill_comparison-zmq7xa6z-_krn076g.png
```

若 CSV 父目录采用其他命名方式，则用该 CSV 绝对路径的 SHA-256 前八位作为回退标识。`--output` 可直接指定完整图片路径，覆盖默认目录和命名。同一组输入的默认文件名固定，重复运行会更新图片；需要保留 hidden-state 与 LM-head 两张图时，为它们分别指定输出路径。

### 对齐与统计

1. 固定一种输出终点：默认 `include_lm_head=False`；`--include-lm-head` 选择两组中的 `True` 测量。两组都必须具有所选终点的成功正式测量。
2. 仅使用 `phase=measure` 且 `status=success` 的有效时延。warmup、OOM、context limit、无效记录及 JSON 中的 pending 尝试不参与统计或扩大范围。
3. 根据 `context_length` 取双方成功点的交集，保留共同上限。若 A 成功测到 2048 词、B 成功测到 8192 词，双方共有 32、64、128、256、512、1024、2048，则仅绘制这七个点。A 在 4096 词处被 kill，并不提供 4096 词的测量值。
4. 起始长度不同或中间缺点时，只保留双方实际共有的长度，不插值；交集可以只有一个点。无交集时明确报错。
5. 相同 context 长度须具有相同的实际 token 数、原文词数/字符数及完整原文标志。横轴为实际 token 数（包含完整 question 与 chat template），按 base 2 对数显示。
6. 两条线分别为各设备每个长度上所有成功正式重复的 `prefill_latency_per_layer_ms` 中位数，阴影为最小/最大值；直接使用 CSV 的层均摊值，不再次除以 N。即使终止前只保存部分重复，也使用已完成值；控制台会报告每组绘制点的重复数范围。阴影不是置信区间。

标题注明加载层数、输出终点和共同 context 范围；控制台同时报告共同点数和各组被排除的成功点数。测量仍是单设备输入到输出的连续时延除以 N，包含输入处理及对应输出开销。

### 可比性检查

CSV 的 `sample_id`、`model_type`、`question_type`、`context_length_unit`、`loaded_layer_count` 必须组内和组间一致；完整性检查发现同一长度的输入属性混杂时会报错。同名 JSON 可用时，还核对其与 CSV 的上述条件，并比较两组的 dtype、模型总层数、窗口、计时定义、attention backend/mask、final norm、LM head 驻留和 warmup 次数。

权重路径仅比较末级目录名，兼容 Linux 与 Windows 的路径分隔符和不同机器的目录前缀。两台机器若为相同权重使用不同目录名，需要在归档结果时确认并统一模型标识；相同目录名本身不等于权重内容校验。实验者仍需保证数据集、权重、tokenizer 和测量代码版本一致。

JSON 缺失、损坏或缺少检查字段时会提示，仅按现有信息校验；CSV 仍可绘制。PyTorch 版本不同时也会提示但继续绘图，以支持 Jetson 与 PC 的不同软件发行版。比较结果反映设备及各自软件环境的整体表现，不能仅凭图形把差异全部归因于 GPU 硬件。
