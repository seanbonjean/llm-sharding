# TODO

## P0：核查 `attention_mask=None` 与分层推理回答质量

背景：构建分层推理框架时，显式设置 `attention_mask` 后程序无法正常运行，当时未找到解决方法，因此保留了 `attention_mask=None` 的实现。长期观察到分层推理的回答效果不佳，有理由怀疑掩码处理与此有关；目前这是待验证的原因，尚未确认因果关系。

涉及位置：`utils/shard_loader.py:LlamaShardPart.forward`，以及顺序推理、pipeline、NodeProfiler 调用该分片前向的路径。本次长上下文 profiling 使用现有分片前向行为，结果应同时保留 attention 实现和掩码状态。正式解释模型质量、比较有效 causal prefill 性能之前，需要完成以下验证。

建议的排查和修改方案：

1. 在远程环境固定模型权重、tokenizer、chat template、dtype、PyTorch/Transformers 版本及 attention backend。建立官方完整模型与同设备逐层模型的对照，先用短输入、batch size 1、无 padding 验证。
2. 对比每层 hidden states、final norm 输出、最后位置 logits、首 token；随后加入带 KV cache 的逐 token decode，定位最早出现偏差的位置，同时核对 RoPE、position IDs、cache 长度和分片相对层索引。
3. 检查该版本官方模型 forward 如何从二维 padding mask、cache position 和历史长度构造因果掩码。直接调用 `LlamaDecoderLayer` 时，需要核实当前 eager/SDPA 等 backend 是否具备相同的因果约束；不能把 tokenizer 的二维 mask 原样传入并假定其等价，也不能仅凭 `None` 断定所有 backend 都会缺失因果性。
4. 实现与固定版本兼容的统一 mask 构造接口，覆盖 prefill 的可见前缀及 decode 的历史 KV；结合 backend 明确二维/四维 mask、形状、dtype、device 和遮蔽值。顺序模型链与 pipeline 各 stage 根据同一请求的有效长度、padding 和 cache 状态保持一致。
5. 增加单设备逐层对齐、跨分片对齐、不同长度/padding、prefill→decode、pipeline 多请求隔离测试。通过后重新测量时延和峰值显存；显式长序列掩码可能增加内存占用，需结合 backend 验证，避免仅为构造掩码就占满内存。

当前工作范围是记录诊断计划；掩码实现待上述对照验证后单独修改。

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
