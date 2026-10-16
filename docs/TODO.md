# TODO

## P0：验证分层推理的因果注意力与回答质量

分层前向通过 `LlamaShardPart.prepare_attention_mask` 复用 Transformers 4.53.2 的官方因果掩码构造逻辑。输入 None 表示无 padding；二维 padding mask 覆盖历史和当前 token。每轮在各层写入 KV 前准备一次因果掩码和 cache_position，供顺序推理、pipeline 和 NodeProfiler 共用。

- [x] 接入 prefill、单 token decode、多 token decode 的因果约束及历史 KV 偏移。
- [x] 逐层 profiler 在计时前调用同一掩码接口，各层实际应用掩码的开销计入时延；端到端 profiling 同时计入掩码构造。两种结果 JSON 均记录 `attention_mask="causal"`。
- [x] 在 PyTorch 2.7.1 / Transformers 4.53.2、CPU float32 环境通过 7 项随机权重小型 Llama 测试，覆盖 eager/SDPA、后文不影响前文、非零起始层分片、分片边界 hidden states/logits/KV 对齐、左右 padding、单/多 token decode、请求缓存隔离及错误 padding 长度。
- [x] 通过现有 NodeProfiler 的 78 项回归测试，包括因果掩码标记和逐层计时边界。

为了正确测试batching的前向推理时延，B/L 前向时延测试 `analysis\batch_forward_latency.py`中单独设置了正确的因果掩码 attention mask，绕开了 `shard_loader.forward`

目标设备验证：

- [ ] 固定真实模型权重、tokenizer、chat template、dtype、依赖版本及 backend，在 PC GPU 和 Jetson 上对比官方完整模型与分片模型的各层 hidden states、final norm、logits、首 token 和连续 decode。
- [ ] 验证实际跨设备模型链与 pipeline 多请求运行；当前 CPU 测试覆盖分片 API 和独立缓存，节点通信及有 padding 的批量输入协议需分别验证。
- [ ] 检查真实回答质量。因果注意力的数值对齐不能替代 tokenizer、chat template、RoPE 和完整生成流程的检查。
- [ ] 使用修复后的实现重新测量时延和峰值显存。历史 eager+None 记录对应不同的注意力行为，比较时固定同一实现和 backend。

## 输入模板接入两套主程序

- [ ] 决定是否将模型专用 chat template 统一接入主程序的文本 input 处，建立共享的 prompt 适配接口，避免同一请求重复套模板。
- [ ] 顺序单请求流程：节点入口 `start_node.py`，配置入口 `send_config.py`；文本入口为 `NodeController.receive_request` → `NodeWorker.receive_user_request`。当前配置发送脚本只发配置，循环内演示请求默认关闭；需要补充正式文本请求客户端/接入方式时单独设计。
- [ ] 多请求 pipeline 流程：节点入口 `pipeline_start_node.py`；`test/pipeline_test.py` 同时提供配置发送、请求投递和实验控制，`test/pipeline_send_config.py` 仅发送配置。文本在 owner 的 `PipelineNodeWorker.receive_user_request` 处 tokenize；应在这个边界统一模板规则。
- [ ] 直接 `input_ids` 的严格 token 数实验保持其显式 token 输入语义，明确与自然文本/chat-template 输入的接口边界。
- [ ] 扩展 `model_type` 适配表。当前长上下文测试仅支持 `llama3.2ins`，默认使用 `question_nonthinking`；其他模型类型需要明确模板、特殊 token、生成提示及思考问题的使用规则。

## 远程实测

- [ ] 在目标 PC GPU、Jetson 上验证完整组件加载、prefill、两种输出终点及真实峰值显存；记录实际依赖版本和设备模式。
- [ ] 将长上下文 CSV 与同名 JSON 一起保存。进程被系统强制结束时，通过系统日志核实原因；`pending_attempt` 只证明某轮已开始，不能单独证明 OOM。
- [ ] 检查权重装载余量。精确 N 层加 embedding、KV、临时张量，及完整模型的 norm/head，均会占用内存；静态最大层数不保证长输入可运行。
