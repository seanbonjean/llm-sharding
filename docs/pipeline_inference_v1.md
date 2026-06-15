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

末节点的 `dst_addr` 仍建议指回 first node，因为 `pipeline_clear` 需要沿模型链绕一圈后停止。

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
该测试会先用相同 prompt 跑两个不带测量 flag 的 warm-up request，等它们完成后再等待
2 秒，让运行时状态稳定，然后把两个相同 prompt 作为一个 batch 发送给 first node/owner。
batch 入口会先把两个请求都放入 first-stage queue，再开始推进队列，避免第一个请求先被
forward、第二个请求还没进入 active queue 的实验偏差。测试会收集每个 request 在每个 node 上的
prefill forward 耗时、DynamicCache 增量和
`torch.cuda.memory_allocated()` 前后差值。结果保存为
`simultaneous_pair_forward_summary.csv` 和
`simultaneous_pair_forward_memory_long.csv`，并额外输出三张柱状图用于快速查看。

主菜单第 7 项为 `run two-round pipeline latency experiments`。该测试会临时把模型链
切到前 2/3/4 台设备，并按 Llama 3.2 3B 的 28 层均匀切分：
2 台设备为 `0~14/14~28`，3 台设备为 `0~9/9~18/18~28`，4 台设备为
`0~7/7~14/14~21/21~28`。每个场景会先跑一次同形状 warm-up batch；发送新 config 后会先等待
`Config settle seconds`，默认 30 秒，避免节点还在加载 layer 时就触发 warm-up。
当前子场景包括 3 节点 3 请求、2 节点 3 请求、4 节点 3 请求、4 节点 4 请求和 3 节点 4 请求；
每个请求 `max_new_tokens=2`。`max_active_requests` 等于节点数，因此 2 节点 3 请求场景中
第 3 个请求应进入 pending queue。结果会输出
`<node_count>node_<request_count>request_2round_forward_reports.csv`、`done_reports.csv` 和
`summary.csv`；summary 中的 `total_process_elapsed_ms` 是从测试脚本发出 batch 到收到所有
done report 的总时延。

## 调试建议

* 如果消息没有继续向下游走，先检查每个节点的 `node_addr` 是否是可连接地址。
* 如果 first node 收不到 owner 的下一轮 decode，检查 owner config 中的 `first_node_addr`。
* 如果请求结束后显存不降，确认 `pipeline_clear` 是否绕链回到 first node；末节点 `dst_addr` 应指向 first node。
