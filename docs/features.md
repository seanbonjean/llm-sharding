# 功能与代码索引

按文件列出主要实现及类方法。部署入口与语义约定见 [overview.md](overview.md)，优先排查项见 [TODO.md](TODO.md)。

## utils/node_profiler.py

节点性能实验。长上下文测试直接运行本机模型组件；既有计算能力测试还支持辅助节点。新方法的详细参数与计时定义见 [long_context_prefill.md](long_context_prefill.md)。

### NodeProfiler

持有分片路径、设备、精度和模型配置；提供时延、容量、KV Cache 及推理验证方法。

#### __init__

读取 LlamaConfig，保存模型总层数及设备设置。

#### profile_long_context_prefill

精确加载 layer_num 层，对 LongBench-Pro 原文前缀翻倍测量。默认普通问题和 llama3.2ins chat template；起始长度仅一次预热。单设备连续计时到 hidden states，完整模型另测 LM head 到首 token。逐行落盘 CSV，并在高内存工作前更新进度 JSON；模型加载前复用一次 Jetson 识别结果生成文件标签，并将内存版本、来源及总内存写入元数据。异常处理中尝试记录错误状态，二次写入失败时向 stderr 输出诊断并继续抛出最初异常。

#### profile_long_context_prefill_per_layer

精确加载 `[0, layer_num)`，从 LongBench-Pro 截取形参指定的 context 词数。完整预热一次后重复执行逐层前向，每轮使用新的共享 KV；每个 attempt 为每个零基层索引保存独立时延行。预热及每轮正式遍历后清理 CUDA allocator 缓存，保留偏冷分配的测试口径。CSV 和进度 JSON 使用独立运行目录持续落盘，保留设备、模型、输入和 OOM/强制结束诊断信息。

#### profile_long_context_prefill_per_layer_reuse_allocator_cache

复用相同的固定输入、逐层计时、独立 KV Cache 和持久化流程。一次完整预热后，在正式重复之间保留 CUDA allocator 已释放的显存块，避免将重复显存申请集中计入最前面的层；整个实验结束后统一清理显存。JSON 通过 `profile_type` 和 `clear_cuda_allocator_cache_between_attempts=false` 标识该口径。

#### _profile_long_context_prefill_per_layer

逐层实验的共享实现，通过 `clear_cuda_allocator_cache_between_attempts` 精确选择轮次间清理策略，确保两种公开方法除 allocator 缓存策略和对应元数据外使用相同流程。

#### _long_context_read_sample

逐块解析 JSON 数组，读取指定零基样本，保留当前记录缓冲区而非整个数据集。

#### _long_context_prefixes

按空白分隔词或 Unicode 字符产生倍增的原文前缀，保留空白，末次覆盖完整原文。

#### _long_context_write_metadata

将运行状态写入临时 JSON，flush/fsync 并关闭后替换同名元数据文件。替换遇到 Windows 错误码 5/32/33 时最多尝试六次，依次等待 0.05、0.1、0.2、0.4、0.8 秒，累计退避 1.55 秒；等待期间保留已提交 JSON，仅复用已落盘临时快照。持续失败时保留两份文件，抛出带路径及权限/占用检查提示的 PermissionError。其他错误立即传播；全部写盘和退避均在 prefill 计时区间外。

#### _long_context_device_label

显式标签优先；自动标签复用可选的 Jetson 硬件快照，未提供时自行识别。Nano 使用 `Orin_Nano_4G` / `Orin_Nano_8G`，已确认系列但容量未知时为 `Orin_Nano`；已识别 NX 使用 `Orin_NX_8G` / `Orin_NX_16G`，容量未知时为 `Orin_NX`。其他情况依次尝试 Jetson device-tree、CUDA 型号和平台信息，统一生成安全文件名片段。

#### _long_context_cuda_stat

读取可选 CUDA 内存计数器；非 CUDA 或不可用统计返回空值。

#### _long_context_system_memory_bytes

读取 `/proc/meminfo` 的 MemTotal 并将 KiB 换算为字节；必要时回退到 sysconf 的物理页数和页大小。返回操作系统可见总物理内存，不使用空闲内存；接口不存在或不可读时返回空值。

#### _long_context_jetson_module_id

只读访问可用 EEPROM 的格式版本和产品编号，识别 Orin NX/Nano 的 P3767 模块 SKU；EEPROM 不可用时查询设备树中启动时填入的模块 ID。不使用共用 DTB 的根 compatible 确定内存版本。返回模块 ID 及来源，读取失败时安全回退。

#### _long_context_jetson_info

从设备树识别 Orin 平台，优先模块 SKU，其次明确的 Nano 型号文本；仍缺少版本时，Nano 的 3.0–4.5 GiB 总内存推断为 4G、6.0–8.5 GiB 推断为 8G。返回简短系列、模块 ID、版本、识别来源和总内存；信息不足时保留未知，非 Orin 平台返回空字典。

#### _long_context_load_components

直接加载 tokenizer、embedding、RoPE 和 [0,N) 层；端到端测试的完整模型加载 final norm/head，逐层测试通过 `include_output_modules=False` 排除这些输出模块，组件设为 eval。

#### _long_context_run_trial

创建独立 KV，连续测量模板/tokenize、本机张量搬运及前向；停止计时后、释放张量前读显存峰值。完整模型 head 仅计算末位置 logits。

#### _long_context_run_per_layer_trial

先执行模板/tokenize、本机拷贝、embedding 和 RoPE，并调用分片的 prepare_attention_mask 一次性准备因果掩码和 cache_position，再按顺序直接调用各 Transformer layer。各层共用本轮掩码、位置和新 KV Cache；调用前后分别同步设备和读取计时器，返回每层时延列表，并在全部层计时结束后读取整轮峰值显存。掩码构造位于逐层计时外，掩码应用位于计时内；final norm 与 LM head 不参与。结果 JSON 的 attention_mask 记录为 causal。

#### profile_max_layer_num

递增装载层数，记录设备可装载的最大 Transformer 层数；此容量不是长输入的可运行保证。

#### _fit_latency_models

对给定时延数据拟合一次/二次曲线，并保存实测散点及拟合图。

#### _report_prefill_decode_similarity

比较 prefill/decode 的平均每 token 时间、线性斜率及二次拟合的边际时间。

#### _synchronize_device

在 CUDA 计时边界同步指定设备；CPU 不执行 CUDA 同步。

#### _build_assisted_command

构造辅助 profiling 的握手命令字典。

#### _is_assisted_command

验证辅助命令类型及可选的具体命令名。

#### _resolve_profile_loaded_layer_num

将既有 max_layer_num 约定转换为实际装载层数，通常减一，-1 表示全模型。

#### _resolve_assisted_target_loaded_layer_num

确定辅助模式目标侧的实际层数并检查部分模型约束。

#### _build_profile_input_ids

重复固定文本片段，tokenize 后切出严格指定 token 长度的输入。

#### _iter_tensors

递归遍历容器中的 tensor，供内存统计使用。

#### _dynamic_cache_payload_size_bytes

统计 DynamicCache 中 K/V tensor 的逻辑字节数。

#### _bytes_to_mib

把字节数转换为 MiB。

#### _cuda_memory_allocated_bytes

读取 CUDA 当前已分配显存；非 CUDA 返回空值。

#### _memory_delta_bytes

计算可用当前值与基线的字节差值。

#### _plot_kv_cache_sizes

绘制 KV tensor payload 与 CUDA 内存增量随 token 长度变化的图。

#### _report_prefill_profile_results

汇总 prefill 测量、按既有口径归一化并调用拟合/绘图。

#### _report_decode_profile_results

汇总累计 decode 时延、拟合并生成图表。

#### _assisted_target_prefill_round

目标侧接收嵌入输入、运行本地分片并可选计时，完成后返回 ACK。

#### _assisted_target_decode_round

目标侧逐轮运行本地分片并把 hidden states 交给辅助侧继续生成。

#### _assistor_assist_prefill_round

辅助侧执行输入 embedding、提交目标侧并等待 prefill ACK。

#### _assistor_assist_decode_round

辅助侧运行目标缺少的层和输出头，将后续 token embedding 返回目标侧。

#### _profile_compute_capability_legacy

保留的单设备 prefill/decode 计算能力测试实现。

#### profile_kv_cache_size

测量输入/输出 token 长度与 KV payload、CUDA 分配增量的关系，并保存统计图。

#### profile_compute_capability

严格 token 长度的计算能力入口，选择本地或辅助模式，统计 prefill/decode 并按既有模型层数口径换算。

#### _profile_compute_capability_assisted_target

组织辅助 profiling 目标侧的层装载、预热、重复测试和结果汇总。

#### assist_profile_compute_capability

运行辅助侧缺失层，配合目标侧完成分阶段性能测试。

#### profile_cold_start_latency

测量层权重加载相关的冷启动时延。

#### go_through_every_shards

用 NodeWorker 贯通所有模型分片，验证自回归生成。

#### go_through_every_shards_only_by_profiler

直接使用 profiler 管理模型组件，贯通完整层链生成。

## utils/shard_loader.py

Llama 分片的实际层装载与前向执行。

### LlamaShardPart

torch.nn.Module，代表连续 Transformer 层区间及可选 final norm。

#### __init__

按 [start,end) 建立相对索引的 LlamaDecoderLayer，加载每个 block 权重及可选 final_norm.pth。

#### prepare_attention_mask

调用 Transformers 4.53.2 的 create_causal_mask，按当前 backend、输入形状和分片缓存长度构造因果约束，同时返回本轮绝对 cache_position。二维 padding mask 必须覆盖全部历史和当前 token；None 表示输入无 padding，仍启用因果注意力；已准备的四维因果掩码按官方接口传递。eager 使用显式掩码，SDPA 在适用条件下通过 is_causal 实现约束。此方法在第一个 decoder layer 更新 KV 之前调用。

#### forward

调用 prepare_attention_mask 一次，然后向各层传入相同的因果掩码、cache_position 及传入的 KV/RoPE，最后执行可选 norm 并返回 hidden states。各分片使用自己的缓存和相对层索引；多 token decode 按历史 KV 长度偏移因果边界。

注意力计算封装在 Transformers 中，调用链为 `LlamaShardPart.forward → LlamaDecoderLayer → LlamaAttention → attention backend`。eager 内部计算 QK 分数、掩码、softmax 和对 V 的加权；本方法只取 `outputs[0]` 的 hidden states，内部注意力矩阵不会作为本方法的返回值暴露给调用方。prefill 的分数形状为 `[B,头数,L,L]`，带 KV 的单 token decode 则为 `[B,头数,1,当前 KV 总长度]`。

分片与完整 LlamaModel 共用官方掩码构造逻辑，接口依据为 [create_causal_mask](https://github.com/huggingface/transformers/blob/v4.53.2/src/transformers/masking_utils.py#L644-L724)。eager 应用掩码时仍计算完整注意力矩阵；时延与峰值显存需使用修复后的实现重新标定。

## utils/forwarding_utils.py

分层推理共用的位置编码辅助函数。

### build_position_ids

根据 HF Cache 或 K/V 元组中的历史长度生成 [batch,seq_len] 的连续 position IDs，支持空缓存。

## utils/model_sharder.py

将完整模型保存成按层加载的权重文件。

### ModelSharder

模型切割器；保存分支包括 Llama/GPT 结构，现有分层前向主要针对 Llama。

#### __init__

载入完整模型、设置模型结构类型和精度，并确定带 dtype 后缀的分片目录。

#### save_shards

复制配置/tokenizer 文件，保存 embedding、各 Transformer block、final norm 与 LM head，完成后释放模型。

## utils/node_worker.py

顺序单请求模型链。该文件的全局请求状态与 pipeline 的逐 request_id 状态是两套实现。

### Communicator

顺序链的 ZMQ 数据通信层。

#### __init__

建立本节点接收 socket 与下游发送 socket。

#### change_src_addr

更改接收 socket 绑定地址。

#### change_dst_addr

更改下游连接地址。

#### transfer_data

序列化 tensor/dict 并发送给下游，可按参数保留临时数据文件。

#### receive_data

阻塞或非阻塞接收并反序列化数据。

### NodeWorker

管理单请求的模型分片、embedding、KV 与生成 token。

#### __init__

初始化配置、设备、通信及单请求状态，按输入能力加载 embedding/tokenizer。

#### _load_embedding

加载 tokenizer 与 embedding 权重。

#### load_shards

装载连续层、RoPE、KV 和末分片需要的 norm/head。

#### receive_user_request

接受原始文本或直接 input_ids，tokenize/embedding 后封装首 stage 输入信息。

#### pass_through_shard

在首分片构造 RoPE，运行本地层；中间分片返回 state，末分片返回 greedy token。

#### receive_next_token

累积并输出生成 token，检查 EOS/长度，构造下一步 embedding 或结束当前请求。

#### is_input_token_info

识别含 batch_size/seq_len 的首 stage 输入消息。

#### is_next_state_info

识别携带 cos/sin 的分片间 hidden-state 消息。

#### clear_KV_cache

释放当前请求 KV 和生成状态，保留模型权重并准备新的缓存。

#### _get_clear_KV_cache_origin

组合节点地址和分片范围，标识清理命令来源。

#### build_clear_KV_cache_command

构造携带来源信息的链式清理命令。

#### is_clear_KV_cache_command

判断数据是否为 KV 清理命令。

#### is_clear_KV_cache_command_origin

检查清理消息是否已经绕链回到发起节点。

### NodeController

顺序链控制器：监听配置，驱动单请求 worker 及状态消息循环。

#### __init__

绑定配置端口，等待配置并建立 worker、首 stage 请求通道。

#### _receive_config

阻塞或非阻塞读取 JSON 配置。

#### _change_first_node_addr

更新输入提交的首节点地址。

#### check_new_config

读取新配置，根据输入能力与层范围更新或重建 worker。

#### _forward_request

序列化嵌入后的输入并发送到首节点。

#### receive_request

将文本交给本节点 worker 做输入处理，再提交模型链。

#### run_worker_loop

处理输入/state/token/清理命令，推进单请求推理并检查配置；演示请求开关默认关闭。

## utils/config_sender.py

顺序链控制面配置发送器。

### ConfigSender

为指定节点配置分片边界、输入能力和数据路由。

#### __init__

创建连接节点配置端口的发送资源。

#### build_config

组装层区间、embedding 能力、上下游与首节点地址。

#### send_config

向指定 IP 的配置端口发送 JSON 配置。

## utils/pipeline_node_worker.py

多请求 pipeline：请求协议、逐请求 KV、首 stage 准入、owner 回路和实验 telemetry。详细协议见 pipeline_inference_v1.md。

### PipelineProtocol

集中定义带 request_id、owner 地址和阶段字段的消息。

#### build_user_request

构造客户端向 owner 提交的原始用户请求。

#### copy_telemetry_fields

沿请求链复制可选观测/实验字段。

#### base_message

组装各类 pipeline 消息共用的标识、阶段与 token 步数。

#### build_input

构造 owner 发给首 stage 的 prefill/decode embedding 输入。

#### build_state

封装 stage 间传递的 hidden states 与 RoPE。

#### build_token

封装末 stage 直接发回 owner 的 token。

#### build_done

通知首 stage 请求完成并可释放 active slot。

#### build_clear

构造沿链传播的单请求清理消息及回环终止标识。

#### is_type

检查对象是否匹配指定消息类型。

#### is_pipeline_message

检查对象是否属于 pipeline 内部协议。

### PipelineCommunicator

使用内存序列化的 ZMQ 通信，维护按目标地址复用的发送 socket。

#### __init__

绑定接收地址并记录默认下游地址。

#### _serialize

把消息以 torch 格式序列化到字节流。

#### _deserialize

从字节流还原消息。

#### _get_send_socket

获取或创建指定目标的发送连接。

#### change_src_addr

重新绑定接收地址。

#### change_dst_addr

更新默认下游，保留可复用连接。

#### send_to

向显式地址点对点发送消息，支持 owner 直连。

#### try_send_to

尽力非阻塞发送控制报告，避免中控不可达时阻塞推理。

#### transfer_data

将消息发给默认下游。

#### receive_data

阻塞或非阻塞接收消息。

### PipelineTegrastatsMonitor

在显式实验请求下启动 Jetson tegrastats，保存并汇总系统采样。

#### __init__

初始化采样进程、日志文件和运行标识。

#### is_running

判断采样进程是否仍在运行。

#### _mb_to_bytes

换算 tegrastats 的内存单位。

#### _parse_percent

从单行采样中提取百分比字段。

#### parse_line

解析 RAM、SWAP、GPU/内存控制器利用率等单条样本。

#### parse_samples

读取采样日志并解析有效行。

#### summarize_samples

汇总采样数量、内存与利用率统计。

#### start

按运行标识与间隔启动 tegrastats 并打开日志。

#### _close_file

关闭采样输出文件。

#### stop

停止采样进程，读取样本并返回汇总。

### PipelineSession

dataclass：保存一个 request_id 的 KV、生成 token、owner 信息、计数与观测基线；没有自定义方法。

### PipelineNodeWorker

按请求隔离模型执行状态；组件权重由多个 session 共用。

#### __init__

保存节点身份、路由、模型配置及 sessions 容器。

#### is_first_stage

判断本节点是否含模型第零层。

#### is_last_stage

判断本节点是否含模型最后一层。

#### _load_embedding

为 owner 输入能力加载 tokenizer/embedding。

#### load_shards

装载层区间、RoPE 与可选末层输出组件；KV 在请求到达时创建。

#### _new_request_id

生成带节点标识的请求 ID。

#### _get_or_create_session

按 request_id 获取或创建独立 session。

#### _iter_tensors

递归遍历缓存容器中的 tensor。

#### _dynamic_cache_payload_stats

计算 K/V payload 字节、元素数和可观察 token 长度。

#### _cuda_memory_allocated_bytes

读取当前进程 CUDA 分配量。

#### _cuda_mem_get_info_bytes

读取设备级 CUDA 可用/总内存，供显式遥测使用。

#### _maybe_capture_cuda_memory_baseline

在开启观测的请求首次本地 forward 前保存内存基线。

#### build_kv_cache_report

构造单请求 KV 及可选 CUDA 内存的只读报告。

#### maybe_emit_kv_cache_report

按请求的观测字段决定是否回传 KV 快照。

#### _should_trace_forward_measurement

判断本条消息是否要求细粒度 forward 测量。

#### _capture_forward_measurement_start

采集 KV/显存起点并建立同步后的 forward 计时起点。

#### _emit_forward_measurement_report

汇报本条消息在本 stage 的 forward 耗时及 KV/显存增量。

#### receive_user_request

owner 处理文本或直接 token 输入、创建 session，返回首 stage 输入消息。

#### pass_through_shard

使用请求独立 KV 运行本地层；首 stage 生成 RoPE，末 stage 生成 token。

#### receive_next_token

owner 处理生成 token，返回下一轮 decode 输入或 done 消息。

#### clear_request_state

释放指定 request_id 的状态，保留其他活动请求。

### PipelineNodeController

协调首 stage 队列、逐请求消息路由、完成清理、重配置和遥测。

#### __init__

绑定配置端口，初始化调度队列与配置状态，建立 worker。

#### is_first_stage

判断 controller 是否承担首 stage 调度。

#### _normalize_config

规范化 pipeline 配置和默认字段。

#### _emit_node_ready_report

层装载完成后向可选中控发送当前 config_id 的就绪报告。

#### _create_worker

根据当前配置创建 PipelineNodeWorker。

#### _receive_config

读取配置端口上的 JSON 配置。

#### check_new_config

非阻塞检查配置；有活动工作时暂存并进入 drain 状态。

#### _has_pipeline_work_in_progress

判断活动分片工作是否阻止配置切换，区分尚未准入的输入。

#### _has_shard_sessions

检查是否存在已进入本地分片计算的 session。

#### _preserve_owner_only_sessions_for_reconfig

保留尚未进入分片的 owner 输入状态以跨越重配置。

#### _whether_apply_deferred_config

旧工作清空后应用待处理配置。

#### _can_apply_config_without_reloading

判断是否仅改变调度/控制字段，可复用当前模型权重。

#### _apply_config_without_reloading

应用调度字段变化并回报 ready，保留已加载模型资源。

#### _apply_config

在安全切换点更新配置、模型与相关状态。

#### _release_pending_prefill_after_reconfig

按新配置准入待处理 prefill，或转发到新的首 stage。

#### receive_request

调用 owner 输入处理并提交首 stage，返回请求 ID。

#### run_worker_loop

循环处理网络消息、本地队列和配置变化。

#### _submit_input_to_first_stage

将输入加入本地首 stage 队列，或发给远程首 stage。

#### _handle_message

按消息类型分派用户请求、pipeline 数据和观测控制命令。

#### _send_tegrastats_report

回传 tegrastats 控制结果。

#### _handle_tegrastats_start

处理显式系统采样启动命令。

#### _handle_tegrastats_stop

停止采样并返回样本及汇总。

#### _handle_user_request_batch

在同一轮循环内提交整批请求，之后再推进首 stage 队列。

#### _handle_user_request

接收客户端请求、检查输入能力和参数，返回可选 ACK/错误报告。

#### _handle_kv_cache_query

响应指定请求的只读 KV 查询。

#### _handle_first_stage_input

处理 prefill 准入、pending 队列及活动请求 decode 输入。

#### _emit_pipeline_admission_report

报告 prefill 获准时刻与首 stage 本地排队时长。

#### _process_first_stage_input_once

从首 stage 输入队列推进一个 work item。

#### _route_processed_message

将分片输出 state 送往下游，或 token 直接送回 owner。

#### _handle_pipeline_token

把 token 交给 owner worker，并路由 decode/done 结果。

#### _emit_pipeline_token_report

回传 owner 收到 token 的时间和序号，供 TTFT 等统计。

#### _handle_pipeline_done

处理请求完成，回收调度状态并触发清理。

#### _emit_pipeline_done_report

向调试客户端回报请求完成信息。

#### _start_pipeline_clear

首 stage 发起该请求的沿链清理。

#### _handle_pipeline_clear

清理本地请求并转发，回到 origin 后停止。

#### _admit_pending_prefill

按空闲 active slot 从 pending 队列 FIFO 放行。

## test/pipeline_test.py

pipeline 中控/调试客户端和分布式实验入口；同时发送配置与用户请求，接收 telemetry、生成 CSV/JSON 和图表。这里的 per-node forward、pipeline TTFT 等口径应与单设备输入到输出时延区分。

### PipelineNodeSpec

dataclass：保存节点 IP、端口、分片区间及输入能力。

#### node_addr

返回节点可连接的数据地址。

#### config_addr

返回节点可连接的配置地址。

### PipelineDebugClient

维护拓扑、请求标识、连接和 telemetry 事件队列。

#### __init__

初始化节点配置与通信/事件状态。

#### _json_socket

复用或创建用于 JSON 配置的发送连接。

#### _data_socket

复用或创建用于模型协议消息的发送连接。

#### _serialize

将数据消息序列化为 torch 字节流。

#### _deserialize

反序列化 telemetry 数据。

#### _new_client_request_id

生成客户端请求标识。

#### _new_config_id

生成配置广播标识，用于匹配 ready 报告。

#### _build_user_request_payload

组装 prompt/input_ids、生成上限及观测标志。

#### build_config

按拓扑生成一个节点的配置，并附加可选中控回报地址。

#### send_config

向全部节点发送配置并返回 config_id。

#### configure_telemetry

绑定 telemetry 接收端，设置节点可达的回调地址。

#### receive_event

按超时接收一条事件。

#### wait_for_event

等待匹配条件的事件，保留其他事件。

#### drain_events

排空当前待收事件。

#### wait_for_nodes_ready

等待当前配置的所有节点完成层加载。

#### submit_request

向选定 owner 发送一个用户请求。

#### submit_burst_requests

将多个请求作为一批发送，便于 owner 同轮入队。

#### wait_for_ack

等待客户端请求的接收确认。

#### wait_for_done

等待客户端请求的完成事件。

#### query_kv_cache

向相关节点查询指定请求的 KV 状态。

#### _collect_worker_tegrastats_reports

收集节点系统采样控制的回报。

#### start_worker_tegrastats

请求所有 worker 启动本机 Jetson 采样。

#### stop_worker_tegrastats

请求所有 worker 停止采样并收集结果。

#### show_topology

显示当前节点顺序、层区间和 owner 能力。

### ask_yes_no

读取带默认值的布尔选项。

### ask_int_list

读取整数列表。

### ask_int

读取并校验整数参数。

### ask_node

提示选择 owner 节点。

### build_even_split_nodes

构造连续均分层的实验拓扑，包含特定节点内存能力的顺序安排。

### configure_even_split_topology

切换客户端拓扑并广播配置。

### ask_prompt

读取请求文本。

### ask_telemetry

交互设置中控可达地址及 telemetry。

### bytes_to_mib

将内存字节值转为 MiB。

### make_result_dir

创建实验结果目录。

### write_csv

按字段写出表格结果。

### write_json

保存运行配置或汇总属性。

### percentile

计算样本分位数。

### mean_or_none

计算非空样本均值。

### linear_fit

对数值样本做线性拟合。

### plot_scatter_with_fit

绘制散点与拟合结果。

### safe_filename_part

生成文件名安全片段。

### node_plot_filename

按节点标识和分片范围构造图名。

### plot_per_node_scatter_with_fit

逐节点绘制 KV 与 CUDA baseline delta 的散点/拟合。

### plot_pair_forward_bars

绘制双请求 forward 时延与内存比较柱图。

### _float_or_none

将可用值转换为浮点数。

### analyze_forward_critical_path

按请求先后依赖和单节点串行约束构造 DAG，估算 forward 计算关键路径。

### plot_latency_breakdown_pie

绘制时延组成汇总。

### plot_request_completion_elapsed_barh

绘制每请求完成时间，优先使用 owner 本地单调时钟的 batch elapsed。

### _menu7_phase_step_order

将 prefill 和各 decode step 映射为绘图顺序。

### plot_menu7_forward_elapsed_lines

按请求绘制各 stage 与 stage 求和的 forward 耗时序列。

### plot_sweep_metric_lines

绘制并发扫描指标曲线。

### plot_stage_time_utilization

绘制各 stage 的时间利用情况。

### plot_memory_peak_by_node

绘制逐节点内存峰值。

### plot_memory_peak_methods_by_node

对照不同统计方式的逐节点峰值。

### sample_rows_by_x_interval

按横轴间隔抽取绘图点，保留原始数据用于拟合。

### load_tokenizer

加载实验输入所用 tokenizer。

### build_input_ids_for_lengths

为指定 token 长度构造直接输入。

### build_pipeline_latency_warmup_input_ids

构造固定长度、与正式输入前缀区分的预热 token 序列。

### build_distinct_input_ids_for_same_length

为相同 token 长度生成不同请求输入。

### aggregate_reports

汇总各节点 KV 与可用内存统计。

### collect_complete_aggregate

等待所需节点报告齐全后生成聚合结果。

### submit_one

交互提交单个请求。

### submit_many

交互提交多个请求。

### change_max_active

调整 active 请求上限并发送配置。

### collect_decode_aggregates

收集按 decode step 对齐的节点报告。

### run_prefill_kv_experiment

测试输入 token 长度与 prefill 后 KV 大小的关系。

### run_decode_kv_experiment

测试输出 token 增长与 KV 大小的关系。

### run_sequential_same_prompt_experiment

顺序执行相同输入，比较请求间 KV 统计。

### run_overlap_same_prompt_experiment

重叠执行相同输入，跟踪请求级及整体 KV 时间序列。

### run_kv_cache_experiments

运行 KV Cache 实验菜单并保存结果。

### collect_shard_forward_reports

收集逐节点分片前向报告。

### collect_forward_reports_and_done

同时收集 forward 与请求完成报告。

### wait_for_warmup_batch

等待预热批次结束。

### collect_concurrency_sweep_events

收集并发扫描过程中的准入、token、forward 和完成事件。

### add_run_memory_baselines

为本轮 forward 行补充内存基线。

### summarize_memory_peaks

汇总各节点不同内存口径的峰值。

### build_tegrastats_rows

将节点系统采样展开为可保存的表格。

### build_concurrency_sweep_result_rows

计算并发扫描汇总与明细指标。

### run_simultaneous_pair_forward_experiment

同批提交两个相同输入，测 prefill forward 与 KV/显存增量；max_new_tokens=1。

### run_two_round_pipeline_latency_scenario

执行指定拓扑/请求数的 prefill 加 decode 时延场景。

### run_two_round_distinct_same_length_custom_scenario

自定义同长度不同输入的两轮场景。

### run_two_round_same_prompt_custom_scenario

自定义相同输入的两轮场景。

### run_two_round_pipeline_latency_experiment

组织菜单 7 的多拓扑两轮时延实验。

### run_concurrency_sweep_motivation_experiment

扫描 active 请求上限，收集吞吐、时延、排队、stage 利用与内存等数据。

### run_scenario

运行预设 pipeline 调试场景。

### menu_loop

显示并分派交互菜单选项。

### main

配置客户端和可选初始广播，等待就绪后进入菜单。

## test/pipeline_send_config.py

pipeline 配置发送示例，负责控制面配置。

### ConfigSenderPipeline

生成含 pipeline 路由与调度字段的配置。

#### __init__

初始化配置发送端口与连接资源。

#### build_config

组装分片范围、节点身份、owner/首节点路由和并发参数。

#### send_config

向指定节点配置端口发送 JSON。

## start_node.py

顺序版节点入口。解析配置端口，创建 NodeController，运行单请求 worker 循环。模型路径/设备/精度在脚本中配置。

## pipeline_start_node.py

pipeline 节点入口。创建 PipelineNodeController 并运行循环，等待客户端发送配置和请求。

### parse_port

解析 --port 或位置参数，默认使用 40700。

## send_config.py

顺序链示例配置发送脚本；配置多个节点及分片范围，发送后维持连接。示例中的层边界和 IP 需按实际部署调整。

## profiling.py

创建 NodeProfiler 的实验入口。当前启用 profile_long_context_prefill，按脚本参数测量 LongBench-Pro 样本并打印 CSV 路径；其他性能方法以可选择的调用示例保留。

## assist_profiling.py

辅助设备入口。装载目标侧缺失的模型层，运行 assist_profile_compute_capability，与目标侧的地址和层数设置配套。

## inference.py

使用 Hugging Face 完整模型进行自回归生成的基线脚本，输出结果和总体耗时；其计时口径不同于分片 prefill。

## run_this.sh

按配置数量启动顺序版 start_node.py，为多个 controller 分配递增的配置端口。

## cmds/install-jetson_stats.sh

Jetson 监控工具安装辅助脚本；运行前检查设备环境和脚本内容。

## analysis/long_context_prefill_result_fig.py

单设备长上下文结果的独立绘图入口，使用标准库和 matplotlib。`DEFAULT_CSV_PATH` 保存默认结果文件；输入及输出的相对路径均以项目根目录解析，支持从根目录或 `analysis/` 目录启动。

### plot_long_context_prefill

读取 CSV 和可选同名 JSON，按实际输入 token 数、hidden-state/LM-head 终点分组，绘制正式重复的中位数及 min/max。可通过 `excluded_input_token_counts` 排除指定实际 token 数的全部正式重复。函数内直接生成简短设备名，有记录的 Orin 内存版本追加到名称后，并保留输出终点。标记完整原文、确认的停止原因与未完成尝试，支持部分运行结果，返回图片路径。

### main

解析可选 CSV 位置参数和 `--output`，省略 CSV 时使用脚本默认路径，调用绘图函数并打印图片保存位置。模块导入与 `--help` 只加载标准库，matplotlib 在实际绘图时加载。

## analysis/long_context_prefill_per_layer_fig.py

固定长文本输入的逐 Transformer 层时延绘图入口，仅依赖标准库和 matplotlib。`DEFAULT_RESULT_DIRECTORY` 保存默认实验运行目录；省略命令行输入时自动定位该目录中唯一的逐层 CSV，也可显式传入其他实验目录或完整 CSV。读取逐层 CSV 与可选同名 JSON，校验层数、实际 token 数和 context 词数一致，支持已完成正式重复及可用的失败/未完成标记。

### _resolve_per_layer_csv_path

统一解析项目根目录下的相对路径。输入为 CSV 时直接使用；输入为实验目录时匹配其中唯一的 `long_context_prefill_per_layer_latency-*.csv`，缺失或存在多份匹配文件时给出明确错误。

### _short_device_name

依据 JSON 的设备标签、Jetson 系列和内存版本生成简短图例；JSON 不可用时从逐层 CSV 文件名回退提取。

### plot_long_context_prefill_per_layer

按零基层索引聚合 `phase=measure`、`status=success` 的 `layer_latency_ms`，绘制各层中位数折线及 min/max 阴影。横轴显示全部已加载层，标题包含固定 context 词数与模板后的实际 token 数；默认将图片保存为 CSV 同目录同主干的 PNG。

### main

解析可选的实验目录/逐层 CSV 和 `--output`；省略输入时使用 `DEFAULT_RESULT_DIRECTORY`，调用绘图函数并打印图片保存位置。帮助信息不加载 ML 或绘图库依赖。

## analysis/long_context_prefill_without_last_length_fig.py

单设备长上下文实验的通用末长度去点入口。输入任意同结构 CSV，自动排除最大的成功正式 `context_length` 及其全部实际 token 点；复用通用单设备绘图函数以保持中位数、min/max 阴影、坐标、图例和停止标记一致。

### _last_successful_measurement_token_counts

读取成功正式测量，忽略预热和失败轮次，返回最大的成功 `context_length` 及其对应的全部实际输入 token 数。少于两个可用实验长度时给出明确错误。

### plot_long_context_prefill_without_last_length

接收 CSV 和可选输出路径，自动检测并排除最后一个成功实验长度。默认在输入 CSV 同目录保存 `<CSV 主干>-without-last-length.png`，也可显式覆盖，最后返回实际图片路径。

### main

解析必需的 CSV 位置参数和可选 `--output`，调用末长度去点函数并打印排除信息及图片保存位置。

## analysis/long_context_prefill_comparison_fig.py

双设备长上下文 prefill 结果的独立对比入口，仅需标准库和 matplotlib。`DEFAULT_CSV_PATH_A` / `DEFAULT_CSV_PATH_B` 保存默认两组 CSV，可由命令行覆盖；按共同原文前缀和同一输出终点绘制两条层均摊时延曲线。相对路径统一基于项目根目录，`DEFAULT_OUTPUT_DIR` 为默认图片目录。

### PrefillPoint

保存一个前缀的实际 token 数、词数、字符数、完整原文标志和成功正式重复的层均摊时延列表。使用具名元组，无自定义方法。

### PrefillRun

保存输入 CSV 路径、组内一致的实验条件、按 context 长度索引的 `PrefillPoint` 与可选 JSON 元数据。使用具名元组，无自定义方法。

### _read_run

解析指定输出终点的成功正式 CSV 记录，跳过预热、失败及无效测量并聚合重复；混合实验条件或同一长度输入属性不一致时报错。读取同名 JSON，检查样本等属性与 CSV 的一致性；元数据不可用时提示并保留 CSV 分析能力。

### _common_context_lengths

检查样本、模板、问题、截断单位、层数及可用 JSON 运行配置，兼容跨操作系统的权重目录前缀。返回成功原文长度的精确交集，检查交集上的 token 数及前缀属性；无共同点或条件冲突时报错。PyTorch 版本差异、元数据不完整时提示。

### _run_directory_hash

从 CSV 父目录的 `YYYYMMDD-HHMMSS-hash` 名称提取完整 hash，保留下划线；其他目录结构使用 CSV 绝对路径的 SHA-256 前八位生成稳定回退标识，用于图片命名。

### compare_long_context_prefill

省略 CSV 参数时使用对应默认常量。默认选择 hidden-state 终点，可用 `include_lm_head=True` 选择首 token。按共同点绘制两个设备系列，横轴实际 token 数，纵轴 CSV 层均摊时延的中位数，阴影 min/max。函数内直接生成简短设备名和已记录的 Orin 内存版本，支持显式昵称/输出路径；默认保存到 `results/profiling/long_context_prefill/comparisons/`，文件名为 `long_context_prefill_comparison-<A hash>-<B hash>.png`。报告共同范围、排除点数与重复数范围，关闭图形并返回保存路径。

### main

接收两个可选 CSV 位置参数及 `--output`、`--include-lm-head`、`--label-a`、`--label-b`；省略某个 CSV 时使用其默认常量。调用对比并打印输出路径。输入或兼容性错误以命令行错误反馈；帮助信息不加载绘图库或 ML 依赖。

## analysis/batch_forward_latency.py

独立扫描 B/L 的 Llama prefill Transformer 前向时延；支持项目分片与本地完整模型目录。计时定义、运行命令和结果文件见 [batch_forward_latency.md](batch_forward_latency.md)。

### StageComponents

保存配置、分词器、embedding、RoPE、精确 Transformer 前缀及设备/精度/后端设置。输入模块用于计时外准备，只有所选 Transformer 层进入前向计时。

### resolve_path

把相对路径解析到项目根目录，使从项目外启动时也能定位模型、数据和结果。

### write_json

写入同目录临时文件并在关闭后替换目标，保存运行元数据和分析规则。

### write_csv

按显式列顺序写入二维结果表，包括汇总和实测等时延 batch 对。

### iter_dataset_records

从本地 JSON 数组逐条解码记录，缓冲区按块增长，不一次读取完整 LongBench-Pro。

### read_token_bank

从指定索引开始选取不同的足够长 context，使用模型分词器截取严格 token 前缀；记录样本 ID 和 token 哈希。过短记录跳过，不重复文本补长度。

### load_stage

从分片或本地 Llama 基础模型中保留 `[0,N)` 层，关闭训练与梯度；设置 eager/SDPA、精度、TF32 和设备，并记录环境。

### prepare_batch

构造无 padding 的连续 `[B,L]` 输入，在计时外完成设备搬运、embedding、位置编号、RoPE 和 eager 因果掩码。

eager 使用可广播的 `[1,1,L,L]` 加性因果掩码，SDPA 使用无 padding、等长、空 KV 条件下的 `is_causal` 路径。该独立测量不调用 LlamaShardPart.forward；掩码构造在计时外，各层应用掩码的开销仍在计时内。

### measure_forward

每次创建新 DynamicCache，连续执行所选层，并在 CUDA 前后同步。停止计时后核对 KV 长度并读取峰值显存，随后释放输出与本轮 KV。

峰值 allocated 包含常驻权重、KV 及前向临时张量。eager 在内部生成 `[B,注意力头数,L,L]` 的分数矩阵，并以 FP32 执行 softmax；即使 `output_attentions=False`，这些计算仍会发生。分数矩阵的元素数随 L 的平方增长，KV 则按实际 KV 头数、层数、B 和缓存长度线性增长，分析 OOM 时需分别核算。

当前 eager 的 `softmax(..., dtype=torch.float32).to(query.dtype)` 会先得到 FP32 权重，再生成 FP16 权重。右侧计算完成前，变量仍引用旧 FP16 分数；转换时旧分数、FP32 softmax 结果和新 FP16 权重存在存活重叠，同一变量名的重新赋值不会原地改写这些张量。

### run_measurements

按种子打乱 B/L 形状顺序，每种形状预热并正式重复；逐行刷新原始 CSV。记录 OOM 和上下文超限，OOM 后清理并继续其他形状。

### percentile

对重复时延做线性插值得到描述性分位数，用于曲线 p10/p90 波动范围。

### summarize_trials

排除预热与非法记录，按重复索引聚合各形状；仅重复次数完整且没有失败的形状标记为成功，保留 OOM、部分完成及未测状态。

### plot_latency_curves

绘制固定 L 扫 B、固定 B 扫 L 的中位数曲线与 p10/p90 区间，并输出带失败状态的 B/L 时延热力图；使用 Agg 和 300 DPI。

### analyze_results

从已有运行聚合汇总，拟合并验证均衡规则，导出稳定实测匹配对和三张图。模型与表格先于绘图落盘，支持中断后只分析。

### run_experiment

创建北京时间命名的独立运行目录，保存参数和进度；依次加载、准备数据、测量，异常或中断时保留已有 CSV 并尝试分析完整点。

### parse_arguments

解析模型来源、层数、设备、精度、B/L 网格、重复次数、容差及只分析选项，规范网格并在加载 ML 包前验证范围。

### main

分派实际测量或已有运行的离线分析；导入模块不会启动实验。

## analysis/batch_latency_balance.py

以成功形状的实测中位数为依据，提供非负函数拟合和实测优先的等时延推荐；导入与实测查表不依赖 ML 库。

### relative_latency_gap

计算两时延差除以均值的对称相对差，用于判定近似相等。

### _features

构造常数、B、BL 和 $BL^2$（长度平方）项，分别表示固定、逐请求、逐 token 和 attention 工作量因素。

### _fit_coefficients

缩放特征列后，枚举至多四项特征的活跃集合求非负最小二乘；拒绝秩不足设计，不依赖 SciPy。

### fit_latency_rule

比较两参数 token 总量模型与四参数混合模型。整种 B/L 形状留出验证、重复波动和最少覆盖要求通过后才采纳函数；优先较少参数，否则使用实测查表。

### predict_latency

只对已采纳函数、成功测量覆盖范围内的正 B/L 预测时延，拒绝外推。

### find_equivalent_batches

寻找 B 与 L 反向变化、中位数差在容差内且 CV 不超过半容差的实测 batch 对。

### recommend_equivalent_batch

以一个实测参考点确定目标时延，优先选择指定新 L 下的稳定实测匹配；必要时用反函数检查相邻整数 B，标注未测预测。已有实测不匹配的形状不会被预测覆盖。

## test/unittest_batch_forward_latency.py

无需模型权重/GPU的 B/L 分析回归测试；函数拟合测试需要 NumPy。

### make_point

构造带 B/L、时延中位数和 CV 的成功形状测试点。

### BalanceRuleTests

检查拟合恢复、模型采纳门槛以及实测和预测推荐的区别。

#### test_full_cost_formula_is_recovered

用合成非负函数验证整种形状留出预测、混合模型恢复及外推拒绝。

#### test_simple_formula_is_preferred_when_valid

BL 模型足以解释数据时优先使用较少参数的函数。

#### test_sparse_noisy_and_irregular_data_use_lookup

样本覆盖不足、波动过大或验证误差过大时回退实测查表。

#### test_measured_pairs_require_tradeoff_and_stability

只导出 B/L 反向变化且时延稳定、近似相等的实测对。

#### test_recommendation_prefers_stable_measured_match

更接近的点波动过大时，仍可选择容差内的稳定实测点。

#### test_model_estimates_do_not_override_measured_mismatches

拒绝用预测结果覆盖同一形状已测得的不匹配结论。

#### test_integer_prediction_is_labeled_as_an_estimate

未测整数 B 的反函数推荐标记为模型估计与待补测。

### DataPreparationTests

检查流式样本读取、严格输入长度、聚合口径及命令行设置。

#### test_streaming_and_exact_token_bank_selection

跳过短文本，选取不同记录的严格长度前缀。

#### test_warmups_failures_and_partial_shapes_are_excluded

预热不进入统计，OOM 和部分测量形状不标记为完整成功。

#### test_cli_deduplicates_grid_and_rejects_invalid_counts

网格去重排序，并拒绝非法重复次数、请求数和容差。

## test/unittest_node_profiler.py

NodeProfiler 和长上下文绘图的标准库回归测试入口，与其他测试统一位于 `test/` 目录，并使用不与标准库冲突的文件名。通过模拟 ML/通信依赖执行真实 profiler 调度和计时代码，绘图使用记录调用的 matplotlib 替身；语法检查不导入 ML 库，实际 CUDA/模型数值正确性由目标设备验证。直接运行时临时排除测试目录以加载标准库 unittest，随后恢复搜索路径。

### load_profile_module

使用模拟依赖加载真实 node_profiler.py。

### load_plot_module

按 `filename` 从 `analysis/` 加载独立绘图模块，仅依赖其标准库导入；默认加载单设备脚本，也支持对比脚本。

### plot_per_layer_with_mocked_backend

使用记录调用的 matplotlib 替身执行逐层绘图函数，验证数据聚合、图形参数、输出位置和资源关闭，无需安装或启动真实 ML 模型。

### make_profiler

建立带可控模型层数、组件加载与 trial 返回值的测试实例。

### write_dataset

创建小型 context/question JSON 测试数据。

### read_rows

读回测试生成的 CSV。

### plot_with_mocked_backend

使用记录调用的 matplotlib 替身测试独立绘图模块的 CSV 处理、分组与输出路径。

### crash_probe

在独立测试子进程中直接退出，验证绕过异常处理时的持久化结果。

### FakeOutOfMemoryError

模拟 CUDA 内存不足异常，没有自定义方法。

### FakeInferenceMode

模拟 torch.inference_mode 上下文及装饰器。

#### __enter__

进入测试推理上下文。

#### __exit__

退出上下文并保留异常传播。

### LongContextTests

测试实例持有独立临时目录、数据集和模拟 profiler；各测试不会加载实际模型或创建节点通信。

#### setUp

为每个测试建立临时目录、样本与依赖替身，并注册清理。

#### run_profile

将测试默认参数与覆盖参数合并后调用真实测量入口。

#### test_streamed_selection_and_unicode

验证跨小缓冲区读取 Unicode 样本及后续无效记录处理。

#### test_stream_errors

验证空数组、格式错误、越界和非对象记录。

#### test_prefixes_preserve_whitespace_and_full_tail

验证词数翻倍、原始空白和完整原文尾部。

#### test_character_prefixes

验证 Unicode 字符截断与倍增。

#### test_single_warmup_and_exact_layer_count

验证精确层数、单次预热、原始重复记录、均摊时延及单设备路径。

#### test_full_model_both_endpoints

验证全模型预热一次和两类正式输出终点。

#### test_component_loading_exact_layers_and_head

验证实际组件加载器的 block 范围与 final norm/head 条件。

#### test_thinking_selection_and_chinese_auto_unit

验证思考问题选择和中文自动截断单位。

#### test_validation_before_loading

验证参数错误在模型装载前抛出。

#### test_missing_question_rejected

验证缺失问题字段的错误处理。

#### test_oom_retains_prior_results_and_empty_failure_latency

验证 OOM 前结果保留和失败行的空时延。

#### test_context_limit_is_distinct

验证模型长度限制的独立停止原因。

#### test_warmup_oom_stops_before_measurements

验证预热内存不足立即停止并保存失败行。

#### test_unexpected_error_is_not_classified_as_oom

验证其他异常继续抛出并保留进度。

#### test_pending_snapshot_precedes_trial

验证开始模型计算前已有 CSV 表头和进度快照。

#### test_metadata_atomic_write_and_replace

验证 JSON 首次写入和后续替换成功，正常路径无需退避。

#### test_metadata_retries_windows_replace_errors

分别模拟 Windows 错误码 5/32/33，验证首次写入与已有 JSON 更新的重试、退避间隔，以及临时快照只写入和同步一次。

#### test_metadata_retry_exhaustion_preserves_snapshots

验证持续拒绝访问最多替换六次，保留旧 JSON 和临时快照，并提供诊断路径及原始异常链。

#### test_metadata_nonretryable_errors_propagate_immediately

验证非目标 Windows 错误、普通权限错误、缺失文件和存储错误立即传播。

#### test_metadata_temporary_write_failures_do_not_retry_replace

验证临时文件打开或 fsync 失败时保留旧 JSON，且不进入替换重试。

#### test_metadata_real_windows_file_lock

在 Windows 上以真实只读句柄阻止 JSON 替换，验证释放句柄后重试成功；其他平台跳过此项。

#### test_metadata_retry_finishes_before_trial

验证 pending 进度的替换重试完成后才进入 trial，CSV 总时延和层均摊时延继续使用 trial 返回值。

#### test_metadata_failure_retains_completed_csv

验证结果落盘后的持续替换失败保留成功 CSV、此前 JSON 和错误状态临时快照，并终止后续 trial。

#### test_error_metadata_failure_preserves_original_exception

验证写入错误状态失败时向 stderr 输出诊断，继续抛出最初的 trial 异常。

#### test_process_exit_retains_durable_progress

验证子进程直接退出后保留已完成 CSV 和 pending 尝试。

#### test_unique_run_directories

验证重复调用分别创建独立运行目录。

#### test_device_detection_fallbacks

验证 Jetson、CUDA 查询失败及显式标签的处理。

#### test_orin_device_label_variants

验证 Nano/NX 简短型号与容量标签、未知或不适用容量、显式覆盖，以及命名复用已有快照时不再次读取系统。

#### test_device_label_optional_snapshot_and_pc_fallback

验证独立自动命名会查询一次硬件，已提供的空快照保持复用，PC 使用 CUDA 型号回退。

#### test_orin_profile_filenames_share_metadata_snapshot

验证 7.x / 3.x GiB 容量推断与未知容量下的 CSV/JSON 文件名和字段一致，单设备图片继承文件名，绘图在有无 JSON 时均可解析简短名称。

#### jetson_info

控制设备树型号、物理模块 ID 和系统总内存，构造可测试的 Jetson 识别环境。

#### test_jetson_variant_system_evidence_priority

验证 Nano 4G/8G 与 NX SKU 区分，以及物理模块 ID、系统型号文本和内存推断的优先级。

#### test_jetson_variant_uses_nominal_memory_ranges

验证 7.x / 3.x GiB 对应标称 8G / 4G，异常总量保留未知，非 Jetson 设备不套用容量推断。

#### test_system_memory_reader_and_fallbacks

验证 MemTotal 的 KiB 换算、忽略空闲量、sysconf 物理页回退和不可用系统接口。

#### test_jetson_eeprom_module_id_and_permissions

验证 EEPROM 版本及产品编号解析，区分模块与载板，跳过未知格式和权限错误。

#### test_jetson_device_tree_ids_and_model_fallback

验证启动时模块 ID、备用设备树路径及 MemTotal 读取；共用 DTB 的根 compatible 不作为容量版本证据。

#### test_jetson_metadata_saved_before_model_loading

验证 Jetson 版本、来源和总内存在模型加载前已保存到 JSON，单次识别同时用于元数据和命名，显式文件标签保持覆盖且不影响硬件记录。

#### test_optional_memory_counter_failure

验证不可用显存计数器返回空值。

#### test_trial_timer_boundaries_and_last_position_head

验证连续计时、停止后读显存、末位置 head 及模板调用。

#### test_each_trial_uses_new_cache_and_hidden_endpoint

验证不同 trial 的 KV 独立及 hidden-state 模式行为。

#### test_real_trial_rejects_oversized_input_before_forward

验证实际 trial 在 tokenize 后、搬运/前向前拦截超长输入。

#### test_plot_reads_csv_with_standalone_module

在禁用 ML 库导入时验证独立绘图按终点分组并读取正式重复。

#### test_plot_incomplete_tail_and_pending_attempt

验证不完整 CSV 尾行与未完成尝试标记。

#### test_plot_optional_metadata_validation

验证异常元数据与已完成但进度未更新的尝试。

#### test_plot_with_no_measurements

验证缺少正式测量时给出明确错误。

#### test_plot_short_orin_names_and_endpoints

验证单设备图使用简短 Orin 4G/8G 名称，同时分别标注 hidden-state 和 LM-head 终点。

#### test_short_device_labels_legacy_and_overrides

验证单设备绘图直接处理旧 JSON、已知容量、字段异常及 PC 名称；版本未知时保留简短名称。

#### test_plot_relative_paths_use_project_root

验证相对 CSV 和图片输出路径均基于项目根目录解析。

#### test_plot_cli_default_and_explicit_paths

验证默认 CSV、显式输入和输出命令行参数的处理。

#### test_plot_help_runs_without_ml_dependencies

从项目外目录启动脚本，在禁用 ML 库和 matplotlib 导入的情况下验证命令行帮助。

#### test_python_sources_compile_without_importing_dependencies

对项目根目录、utils、test 和 analysis 的 Python 源码做 AST/compile 静态检查。

#### test_standard_library_imports_from_application_directories

在新进程中分别从项目根目录、test 和 analysis 目录导入 unittest/mock，验证解析到标准库包，覆盖应用启动时尚无 unittest 缓存的情况。

### PrefillComparisonTests

使用独立 CSV/JSON 测试数据检查双设备绘图的共同范围、兼容性及命令行行为，执行绘图时禁用 ML 导入。

#### setUp

创建临时目录、加载独立对比模块并注册清理。

#### write_run

生成可指定原文长度、时延比例、输出终点、CSV 条件及 JSON 属性的三次重复测量。

#### write_rows

按记录中的字段名写入 CSV，用于构造输入属性或失败状态等测试情况。

#### plot

替换 matplotlib 后端并禁止 ML 导入，调用真实对比函数，检查图形关闭并返回图形调用记录。

#### test_shared_upper_bound_and_statistics

验证 32–2048 与 32–8192 的共同范围包含 2048，两个系列使用相同横轴，且中位数和 min/max 直接来自层均摊值。

#### test_exact_shared_points_and_single_point

验证离散长度只取精确交集，并支持只有一个共同点的输入。

#### test_failed_warmup_and_pending_attempts_are_excluded

验证 warmup、OOM、context limit、不完整 CSV 尾行及 pending JSON 不扩大成功测量范围。

#### test_endpoint_selection_and_labels

验证 hidden-state/LM-head 终点选择、两条设备图例及同组输入的固定默认输出文件名。

#### test_comparison_short_device_names_and_orin_variant

验证双设备图使用 `Jetson Orin Nano 8G`、`Quadro RTX 5000` 等简短名称，显式昵称仍附带已记录的 Orin 容量，并验证 `Orin_Nano_4G` / `Orin_Nano_8G` 文件名在有无 JSON 时的兼容性。

#### test_incompatible_csv_settings

逐项验证样本、模型模板、问题类型、截断单位和层数不一致时拒绝比较。

#### test_inconsistent_tokens_prefixes_and_run_rows

验证两组 token 数/前缀属性不一致、单组重复输入不一致和混合层数时的错误。

#### test_metadata_compatibility_and_host_paths

验证模型、精度、计时、attention 等条件检查，兼容 Windows/Linux 路径前缀，并对 PyTorch 版本差异提示。

#### test_optional_metadata_and_duplicate_labels

验证缺失/无效 JSON 时的提示与 CSV 回退，以及相同设备标签的区分。

#### test_empty_overlap_missing_endpoint_and_bad_columns

验证无共同点、缺少所选终点、非法布尔参数和缺列输入的明确错误。

#### test_invalid_numeric_rows_are_skipped

验证空值、负时延及非有限时延不会混入有效点统计。

#### test_relative_paths_and_cli

验证相对输入/输出路径以项目根目录解析，并正确转发命令行选项。

#### test_default_inputs_and_run_hash_output

验证默认 CSV 常量、统一的 comparisons 输出目录，以及按输入顺序使用完整运行目录 hash 命名，保留 hash 开头的下划线。

#### test_cli_default_and_partial_csv_overrides

验证不传 CSV、仅传第一组和同时指定两组时的命令行默认值与覆盖行为。

#### test_run_hash_fallback_for_other_directories

验证普通目录下的 CSV 使用稳定的八位路径 hash，且不同路径具有不同标识。

#### test_help_without_ml_or_plotting_dependencies

从项目外启动对比脚本，在禁用 ML 与绘图库导入时验证命令行帮助。

## test/unittest_shard_loader.py

使用 CPU float32 随机权重小型 Llama，比较官方完整模型与从磁盘加载的两段分片。执行不需要下载权重。

### ShardCausalAttentionTests

验证 eager/SDPA 的因果注意力、分片边界、KV 历史和 padding 语义。

#### setUpClass

创建三层完整模型，将各层和 final norm 保存到临时目录，再加载两个起始层不同的分片。

#### tearDownClass

释放临时权重并恢复 CPU 线程数。

#### set_backend

为完整模型与各分片选择同一个 attention 后端。

#### forward_shards

从 embedding 开始按分片顺序前向，共用本轮 RoPE，各分片使用自己的相对层索引和独立 KV；返回 logits 与分片边界 hidden states。

#### assert_matches_reference

比较有效查询位置的 logits、边界 hidden states，以及各层缓存的 K/V 和长度。

#### test_prefill_matches_official_model

验证有缓存和无缓存的批量 prefill，与官方完整模型对齐。

#### test_future_tokens_do_not_change_prefix

改变输入后缀，验证所有更早位置的 logits 保持一致。

#### test_cached_single_and_multi_token_decode

依次输入 prefill、单 token 和多 token decode，比较逐步官方前向及一次性完整序列结果。

#### test_padding_is_respected_in_prefill_and_decode

验证左右 padding、改变 PAD 内容不影响有效输出，以及包含 padding 历史的多 token decode。

#### test_interleaved_requests_keep_separate_histories

交错处理两条历史长度不同的请求，验证共享模型层时缓存仍按请求独立。

#### test_chunk_mask_uses_past_length_and_padding

逐格检查带历史 KV 的多 token 查询：有效历史可见、PAD 不可见、当前块的未来位置不可见。

#### test_decode_rejects_mask_missing_history

验证漏掉历史部分的二维 padding mask 在修改缓存前报错。
