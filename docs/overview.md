# 项目概览

项目把模型的 Transformer 层保存为独立权重，并在边缘设备上按连续层区间部署。主要运行方式为顺序单请求模型链、多请求 pipeline，以及单设备 profiling。文件/类/方法索引见 [features.md](features.md)，分布式协议与实验细节见 [pipeline_inference_v1.md](pipeline_inference_v1.md)。

## 入口与职责

| 场景 | 节点启动 | 配置发送 | 用户请求或实验入口 |
| --- | --- | --- | --- |
| 顺序单请求模型链 | `start_node.py`；`run_this.sh` 可启动多个 controller | `send_config.py` / `ConfigSender` | `NodeController.receive_request()`；`run_worker_loop` 的演示请求开关默认关闭 |
| 多请求 pipeline | `pipeline_start_node.py` | `test/pipeline_test.py` 或 `test/pipeline_send_config.py` | `test/pipeline_test.py` 向 owner 投递 `user_request` / 请求批次 |
| 单设备长上下文 prefill | 本地 Python 中创建 `NodeProfiler` | 本地函数形参 | `profile_long_context_prefill()`；独立调用 `plot_long_context_prefill()` |
| 节点性能实验 | `profiling.py` | 脚本中的参数 | 默认执行 `profile_long_context_prefill()`；可选择 KV Cache、计算能力或冷启动实验 |
| 辅助节点性能实验 | 目标侧 `profiling.py`，辅助侧 `assist_profiling.py` | 地址及层数参数 | `assisted=True` 与 `assist_profile_compute_capability()` 配合 |
| 完整模型基线 | `inference.py` | 脚本中的模型路径等参数 | Hugging Face 完整模型 `generate()` |

`send_config.py` 和 `test/pipeline_send_config.py` 的职责是发送配置。顺序版目前没有启用的独立原始文本请求发送脚本；pipeline 调试客户端才同时承担请求投递。

## 两套分层推理流程

### 顺序单请求模型链

`NodeController` 接收配置，创建 `NodeWorker`，加载指定连续层。具备 embedding 的节点可以将文本转换为 hidden states 并送往模型链首层；hidden states 依次经过各分片，末层输出 token 后回到首节点继续 decode。一次请求结束时，清理命令沿模型链传递，释放本次 KV 与生成状态。

该路径使用单份请求状态。配置端口默认 40700，数据端口默认 40800；同设备多个 controller 使用不同端口。分片边界、下游地址、首节点地址和是否具备输入能力由配置指定。

### 多请求 pipeline

`PipelineNodeController` 与 `PipelineNodeWorker` 为每个 `request_id` 保存独立 `PipelineSession` 和 KV。owner 负责输入 embedding 和输出 token；首 stage 按 `max_active_requests` 接纳请求，其余 prefill 排队。不同请求可同时位于不同 stage，当前每个 work item 属于一个请求，不是 batch decode。

末 stage 将 token 直接发回对应 owner；owner 将后续 decode 输入提交首 stage。请求完成后释放 active slot，清理消息沿链回收各 stage 的 session。重配置会考虑活动请求、旧链清理和节点 ready 报告。telemetry 默认使用 40900，用于 pipeline 实验报告。

## 单设备长上下文测量

`NodeProfiler.profile_long_context_prefill()` 直接加载本机 tokenizer、embedding、RoPE 与 `[0, layer_num)` 分片。它独立于节点控制器和模型链通信；测量域为一台设备，本机 CPU→GPU 张量搬运计入输入处理时延。

连续计时区间是：context/question 拼接与 chat template、tokenize、本机输入张量搬运、embedding、RoPE、全部已加载层，直到最后 hidden states 完成；完整模型另测包含 final norm、末位置 LM head 和首 token ID 的终点。CUDA 在计时前同步，在输出完成后同步并停止计时；峰值显存在停止计时后、释放输出/KV 前读取。读取数据集、加载权重、截断原文、写盘、清理内存均在计时区间外。

长上下文逐步翻倍，起始长度只预热一次。每次正式重复使用全新的 KV，CSV 保存原始重复测量；绘图单独读取已完成记录。参数、字段、OOM 后恢复分析和调用示例见 [long_context_prefill.md](long_context_prefill.md)。

## 项目中的约定和共识

- 分片范围为左闭右开 `[start, end)`；每个分片内部 cache 的层索引使用相对索引。`LlamaShardPart` 是 Llama 层加载/前向实现。
- `profile_long_context_prefill(layer_num=N)` 精确加载 N 个 Transformer 层，范围为 1 到配置中的模型总层数。embedding、final norm、LM head 不计入 N。
- 既有 `profile_compute_capability(max_layer_num=N)` 通常实际加载 N−1 层以预留 KV 空间；其 `-1` 表示全模型。新方法的 `layer_num` 直接指定实际层数，不使用这个哨兵值。
- 新方法的 `prefill_latency_per_layer_ms` 定义为本次连续输入到输出时延除以 N，包含输入处理和相应输出头的摊销开销；不是逐层插入计时器获得的纯 Transformer 算子时延，也不外推到模型总层数。
- 既有 `profile_compute_capability` 使用严格 token 数的合成输入；长上下文方法保存真实模板后的 token 数，原文长度翻倍不保证 token 数严格翻倍。
- 长上下文模板目前仅支持 `model_type="llama3.2ins"`。`use_thinking_question=False` 选择普通问题；切换该布尔值只决定数据字段，不改变模型结构或启用某种推理模式。
- 分层推理目前沿用 `attention_mask=None`，回答质量与掩码行为需要优先核查，见 [TODO.md](TODO.md)。长上下文结果须结合其记录的 backend/掩码状态解释。
- 测量表格存 CSV，运行属性和进度存 JSON。CUDA 内存字段为字节，是当前进程的 PyTorch allocator 统计；Jetson 的统一内存与系统进程占用需要另行观察。
- 模型路径、IP、端口和层边界需要按部署环境配置。脚本中的示例配置不保证适配所有模型；特别是顺序版 `send_config.py` 含 32 层边界，应按实际模型配置调整。
- 当前依赖清单是 Windows 环境参考；Jetson 应使用匹配 JetPack/CUDA 的 PyTorch。无 ML 环境的机器可用 `python -B test/unittest_node_profiler.py` 验证控制逻辑和语法，真实前向与性能需在目标服务器验证。
- NodeProfiler 回归测试与其他测试统一放在 `test/` 目录，文件名为 `unittest_node_profiler.py`。应用入口的搜索路径应保持标准库模块可正常解析；测试使用新进程分别检查项目根目录和 `test/` 目录中的 `unittest.mock` 导入。
