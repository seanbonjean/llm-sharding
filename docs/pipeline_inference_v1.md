# Pipeline Inference V1

本文档说明 `utils/pipeline_node_worker.py` 的通信协议、调度策略和 Jetson 测试步骤。
旧版 `utils/node_worker.py` 、 `utils/node_profiler.py` 不依赖 pipeline 文件，原 profiling 路径保持不变。

## 目标

Pipeline V1 支持多个用户请求同时处在同一条模型链的不同 stage 中。每个请求拥有独立的：

* `request_id`
* `DynamicCache`
* `generated_ids`
* token 结束条件
* owner 回路

V1 不做 batch decode。每条 work item 只属于一个请求，谁先生成结束就先释放 active slot。

## 消息流

1. owner 节点接收用户文本，执行 tokenizer 和 embedding。
2. owner 构造 `pipeline_input`，直接发送到 `first_node_addr`。
3. first node 负责 admission scheduler：
   - active 未满：请求进入 `first_stage_input_queue` 并开始 prefill。
   - active 已满：新 prefill 进入 `pending_prefill_queue` 。
   - active 请求的 decode 返回后直接进入 `first_stage_input_queue` 。
4. hidden state 沿模型链通过 `pipeline_state` 逐 stage 转发。
5. last node 生成 `pipeline_token`，直接发回 `owner_addr`。
6. owner 执行 `receive_next_token`：
   - 未结束：embedding 新 token，构造下一轮 decode `pipeline_input` 发回 first node。
   - 已结束：输出最终文本，并发送 `pipeline_done` 给 first node。
7. first node 收到 `pipeline_done` 后释放 active slot，并发送 `pipeline_clear` 沿模型链清理该 `request_id` 的 KV/session。

## Config 字段

Pipeline config 兼容旧字段，并新增：

* `node_addr`: 当前节点可被其他节点 connect 的地址，例如 `tcp://172.16.0.2:40800`。不能写 `tcp://*:40800`。
* `first_node_addr`: 模型链首分片地址。
* `pipeline_depth`: 模型链 stage 数。4 台设备各一个 stage 时填 `4`。
* `max_active_requests`: 最大 active 请求数；建议默认等于 `pipeline_depth`。
* `node_id`: request_id 和日志前缀，不参与路由。
* `controller_addr`（可选）：中控可被节点连接的 PULL 地址。节点完成 layer 装载后，
  会向此地址发送 `pipeline_node_ready`。
* `config_id`（可选）：某次 config 广播的唯一标识。ready report 会原样回显它，供中控
  区分当前拓扑与迟到的旧配置报告。

末节点的 `dst_addr` 仍建议指回 first node，因为 `pipeline_clear` 需要沿模型链绕一圈后停止。

## 节点就绪确认

节点在初次收到 config 并完成 `load_shards()` 后，会在输出
`[CONTROLLER] Pipeline node is ready.` 的同时，向可选的 `controller_addr` 发送一次
`pipeline_node_ready`。重配置完成后也会发送同类型报告，`event` 为
`reconfigured_ready`。报告包含 `node_id`、`node_addr`、分片范围、
`can_receive_user_request`、`pipeline_depth`、`max_active_requests`、`config_id` 和时间戳。

这是独立于推理数据流的 best-effort 非阻塞控制面消息：未配置 `controller_addr` 时不会发送；
中控不可达或发送队列满时只会丢弃该报告，不会阻塞 pipeline 推理。中控应等待当前
`config_id` 的所有节点 ready report 后，再投递新请求。`pipeline_test.py` 在 telemetry
PULL 已启动时会将其地址作为 `controller_addr`；initial config 与 menu 7 会自动执行这一等待。
启动 `pipeline_test.py` 时若选择发送 initial config，脚本会先启动该接收端，并以
`Initial config-ready timeout seconds`（默认 60 秒）等待全部节点 ready；超时不会进入菜单。
若 initial config 选择 `N`，脚本不会等待，直接显示菜单。

## 用户请求入口

外部用户请求通过 ZMQ 发送到 owner 节点的 `node_addr`，也就是默认 `40800` 数据端口。
消息是 torch 序列化后的 dict：

```python
{
    "type": "user_request",
    "prompt": "...",
    "max_new_tokens": 128,
}
```

节点收到后会在本地调用 `receive_request()`，完成 tokenizer/embedding，再生成正常的
`pipeline_input` 发往 first node。复用 40800 可以减少额外端口；如果以后需要隔离
外部流量和模型链内部流量，再拆出单独 client request port。

## 重配置行为

如果节点收到新 config 时仍有旧请求在 pipeline 中：

- 新 config 会先暂存在 `deferred_config`。
- 首节点会进入 `reconfig_pending`，新的 prefill 请求只进入 `pending_prefill_queue`，不会进入 `first_stage_input_queue`。
- 已经 active 的请求继续 decode，直到完成。
- 首节点会等待 `pipeline_clear` 沿旧模型链回到 origin 后再应用新 config。
- 新 config 应用后，`pending_prefill_queue` 中的请求再按 FIFO 放行。
- 如果某节点已经接收了用户请求但请求还没完成，且新 config 要关闭该节点的
  `can_receive_user_request`，该节点仍可先切换 config，但会临时保留
  tokenizer/embedding 只服务这些已有 owner 请求；controller 会拒绝新的用户请求。
  这样 pending 请求不会因为 owner 能力关闭而阻塞整次重配置。

注意：ZMQ 只能提供传输层排队，不能保证某条 hidden state 一定由正确版本的分片处理。严格实验时，中控设备最好等所有节点都完成重配置并 ready 后，再提交新的用户请求。

## Jetson 测试步骤

1. 在每台设备启动 pipeline 节点：

   

```bash
   python pipeline_start_node.py --port 40700
   ```

2. 修改 `test/pipeline_test.py` 或 `test/pipeline_send_config.py` 中的 IP、切片边界和 `pipeline_depth`，然后在中控设备执行：

   

```bash
   python test/pipeline_test.py
   ```

3. 单请求测试：
   - 设置 `max_active_requests=1` 。
   - 提交一个请求。
   - 预期：完整输出；每个节点能看到同一个 `request_id` 的处理日志。

4. owner 非首节点测试：
   - 让中间节点 `can_receive_user_request=True` 并从该节点提交请求。
   - 预期：last node 的 `pipeline_token` 直接返回该 owner；owner 继续把 decode 输入发回 first node。

5. 多请求测试：
   - 设置 `pipeline_depth=4` 、 `max_active_requests=4` 。
   - 连续提交 4 个请求。
   - 预期：4 个 `request_id` 输出互不串扰，结束顺序可以不同。

6. 排队测试：
   - 在上一步基础上连续提交 6 个请求。
   - 预期：前 4 个 active，后 2 个 pending；任一 active 请求结束后，pending 请求按 FIFO admitted。

7. 提前结束测试：
   - 设置较小 `max_new_tokens` ，例如 `8` 。
   - 预期：单个请求结束后只清理自己的 session，其他 active 请求继续生成。

8. profiling 保护测试：
   - 运行原 profiling 入口。
   - 预期：仍 import `utils.node_worker.NodeWorker` ，不会进入 pipeline 版本。

## KV Cache 实验

`test/pipeline_test.py` 的菜单第 5 项为 `run KV cache growth experiments`，用于在 pipeline/distributed 环境下收集 KV Cache 增长数据。实验脚本会在中控设备绑定一个 telemetry PULL 端口，并把可连接地址随测试请求发送给 owner 节点；worker 只在请求带 `telemetry_addr` 时回传测量结果，普通推理不受影响。

运行时需要输入 `Telemetry callback host/IP visible to Jetson nodes`，这里应填写 Jetson 节点能连到的中控设备 IP，端口默认 `40900`。结果会保存到：

```text
results/pipeline_kv_cache/<timestamp>/
```

当前包含 4 类实验：

1. prefill 后 KV Cache 大小随 input token length 增长。
2. decode 过程中 KV Cache 大小随 output token length 增长。
3. 两个相同 input prompt 顺序执行时，比较各自 prefill 后 KV Cache 大小是否一致。
4. 两个相同 input prompt 间隔进入并重叠执行时，记录每个 request 和总 KV Cache 随时间变化。

每类实验都会输出 CSV；可拟合的实验会额外保存散点图和一次函数拟合结果 CSV。

### KV Cache 与 CUDA memory 字段

CSV 中的 `*_bytes` 字段单位都是 byte；图中的纵轴使用 `MiB`，计算方式是
`bytes / 1024 / 1024`，不是 megabit。

`kv_cache_bytes` 直接递归统计当前 request 的 `DynamicCache` 中 key/value tensor
payload，因此它是按 request 隔离的 KV Cache 逻辑大小。`cuda_memory_allocated_bytes`
来自 `torch.cuda.memory_allocated()`，表示该节点 PyTorch 当前已分配显存总量，会包含
模型权重、其他 request、临时 tensor 等非 KV 内容。为了便于和 KV 增长趋势对照，worker
会在带 `telemetry_addr` 且开启 `trace_kv_cache` 的请求第一次进入每个分片 forward 前记录
`cuda_memory_baseline_bytes`，并在报告中额外给出
`cuda_memory_delta_bytes = cuda_memory_allocated_bytes - cuda_memory_baseline_bytes`。
当前图中的 CUDA 曲线使用 delta 字段，图例统一为 `cuda memory baseline delta`；
raw allocated 字段仍保留在 CSV 中用于排查单节点显存状态，但不会画到图上，避免纵轴比例被拉大。

除全 pipeline 求和图外，实验脚本还会为每个节点额外生成 per-node 图，文件名形如
`<experiment>_per_node_<node_id>_shards_<shards_start>~<shards_end>.png`。这些图不会跨节点
求和，会画出该节点的 `kv_cache_bytes` 和 `cuda_memory_delta_bytes`。

`test/pipeline_test.py` 的主菜单第 6 项为 `run simultaneous pair forward measurement`。
该测试会用相同 prompt 跑两个不带测量 flag 的 warm-up request，等它们完成后再等待 2 秒，
让运行时状态稳定，然后把两个相同 prompt 作为一个 batch 发送给 first node/owner。
batch 入口会先把两个请求都放入 first-stage queue，再开始推进队列，避免第一个请求先被
forward、第二个请求还没进入 active queue 的实验偏差。测试会收集每个 request 在每个 node 上的
prefill forward 耗时、DynamicCache 增量和
`torch.cuda.memory_allocated()` 前后差值。结果保存为
`simultaneous_pair_forward_summary.csv` 和
`simultaneous_pair_forward_memory_long.csv`，并额外输出三张柱状图用于快速查看。

主菜单第 7 项为 `run two-round pipeline latency experiments`。该测试会临时把模型链
切到 2/3/4/5/6/7 台设备，并按 Llama 3.2 3B 的 28 层均匀切分：
2 台设备为 `0~14/14~28`，3 台设备为 `0~9/9~18/18~28`，4 台设备为
`0~7/7~14/14~21/21~28`，5 台设备为 `0~6/6~12/12~18/18~23/23~28`，6 台设备为
`0~4/4~9/9~14/14~18/18~23/23~28`，7 台设备为
`0~4/4~8/8~12/12~16/16~20/20~24/24~28`。
6 台设备时，4 GiB Orin Nano（`node5`，`172.16.0.3`）会被固定放在倒数第二个
stage，加载 `18~23` 层；其 `can_receive_user_request=False`，因此不会加载
tokenizer/embedding。末 stage 由 `node4`（`172.16.0.2`）承载并加载 LM head。
7 台设备时，`node5` 仍是中间 stage，加载 `20~24` 层；新增的 `node6`
（`172.16.0.8`）作为末 stage，加载 `24~28` 层与 LM head。
每个场景会在发送新 config 后等待所有节点回传当前 `config_id` 的 ready report；
`Config-ready timeout seconds` 默认 60 秒，超时会直接停止该场景，避免节点还在加载 layer
时就触发 warm-up。确认 ready 后会跑一条固定 warm-up 请求：其输入严格为 16 tokens，
且 token 序列不会与任何正式请求的前 16 个 token 相同。该请求设置 `max_new_tokens=17` 并启用
EOS 跳过，因此先经过一次 prefill，再恰好执行 16 次 decode；warm-up 超时会直接停止场景。
当前子场景包括 3 节点 3 请求、2 节点 3 请求、4 节点 3 请求、4 节点 4 请求、3 节点 4 请求、
5 节点 4 请求、5 节点 5 请求和 5 节点 6 请求；
每个请求 `max_new_tokens=2`。`max_active_requests` 等于节点数，因此 2 节点 3 请求场景中
第 3 个请求应进入 pending queue，5 节点 6 请求场景中第 6 个请求也应进入 pending queue。结果会输出
`<node_count>node_<request_count>request_2round_forward_reports.csv`、`done_reports.csv`、
`critical_path_forward_rows.csv` 和 `summary.json`；summary 中的 `total_complete_time`
是从测试脚本发出 batch 到收到所有 done report 的总时延，`inference_time` 是由
forward report DAG critical path 累加得到的推理关键路径时延，`communication_and_noncompute_time`
是二者差值。测试还会输出 latency breakdown 饼图，以及各 request 的
`done_received_elapsed_ms` 横向条形图。每个 stage 还会输出一张
`<prefix>_forward_elapsed_<node_id>_shards_<start>~<end>.png`：每条线对应一个 request，
横轴为 prefill 和后续 decode 轮次，纵轴为该 stage 的 `forward_elapsed_ms`。另有
`<prefix>_forward_elapsed_total.png`，对同一 request、同一轮的所有 stage
`forward_elapsed_ms` 求和；缺失任一 stage report 的点不会纳入总和。

该菜单还提供一个自定义等长不同内容测试项：节点数可输入 2~7，请求数可输入 2~7，
每个请求仍为 `max_new_tokens=2`。该测试参考 `utils/node_profiler.py` 的 prompt 构造方式：
先构造足够长的文本，tokenize 后直接截取固定长度的 `input_ids`；不同 request 使用不同顺序的
fragment，因此内容不同，但输入 token length 由截断后的 `input_ids` 保证一致。结果文件名前缀为
`<node_count>node_<request_count>request_distinct_same_len<input_token_length>_2round`。
另一个自定义测试项使用和固定子场景 1~8 相同的请求内容构造方式：所有 request 发送同一个 prompt，
但节点数可输入 2~7，请求数可输入 2~7。结果文件名前缀为
`<node_count>node_<request_count>request_same_prompt_custom_2round`。
另外还有两个对应的可变输出长度测试项：一个沿用“不同内容但等长 input_ids”的构造方式，
另一个沿用“相同 prompt”的构造方式；二者都可额外输入 `max_new_tokens`。当 `max_new_tokens`
不是 2 时，结果文件名前缀会使用 `maxnew<max_new_tokens>`，并且 forward report 会按
`prefill step=0` 加 `decode step=1..max_new_tokens-1` 收集。
为保证这些 latency 场景的 decode 轮数固定，menu 7 发出的测量请求会带
`ignore_eos_for_measurement=True`：如果生成过程中出现 EOS，owner 会记录 `eos_seen` 和
`first_eos_step`，但仍继续 decode 到 `max_new_tokens`。这种情况下 CSV 中
`semantic_output_valid=False`，表示生成文本不应再用于语义分析，但 forward latency/KV cache
测量仍可使用。

## 调试建议

* 如果消息没有继续向下游走，先检查每个节点的 `node_addr` 是否是可连接地址。
* 如果 first node 收不到 owner 的下一轮 decode，检查 owner config 中的 `first_node_addr`。
* 如果请求结束后显存不降，确认 `pipeline_clear` 是否绕链回到 first node；末节点 `dst_addr` 应指向 first node。
