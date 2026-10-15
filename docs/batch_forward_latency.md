# Batch 请求数与 prompt 长度的前向时延

`analysis/batch_forward_latency.py` 测量同一设备、模型、精度、attention 后端和层数下，batch 请求数 B 与每条请求 token 长度 L 对 prefill Transformer 前向时延的影响。运行后保存原始数据、两组变化曲线、B/L 时延热力图，以及等时延推荐规则。

## 运行

在 Ubuntu 服务器的项目根目录、已有 PyTorch 和 Transformers 环境中执行：

```bash
python analysis/batch_forward_latency.py \
  --shards-path shards/Llama-3___2-3B-Instruct_float16 \
  --layer-count 6 --device cuda:0 --dtype float16 \
  --batch-sizes 1 2 4 8 \
  --prompt-lengths 128 256 512 1024 2048 4096 \
  --warmup 2 --repeats 7 --relative-tolerance 0.05
```

也可以从本地 Hugging Face 格式 Llama 模型目录选择 Transformer 前缀：

```bash
python analysis/batch_forward_latency.py \
  --model-path weights/Llama-3___2-1B \
  --layer-count 6 --device cuda:0 --dtype float16
```

模型和分词器只从本地加载。分片模式只加载 `embedding.pth`、配置、分词器和 `block_0.pth` 至 `block_<N-1>.pth`；完整模型目录模式会先在 CPU 加载基础模型，再保留选择的前缀。接口按项目依赖中的 Transformers 4.53.2 编写。

默认 `--attention-backend eager` 与 batching 示例一致；指定 `--attention-backend sdpa` 可以测试 PyTorch SDPA。SDPA 实际选择的内核由设备、精度、形状和 PyTorch 环境决定，因此推荐规则只用于其记录的环境。

CPU 实验可指定 `--device cpu --dtype float32 --cpu-threads 4`。`--layer-count` 是从零开始实际加载的 Transformer 层数。`--allow-tf32` 显式启用 TF32；默认关闭。

## 输入与计时

- B 是一次前向中的独立请求数，L 是每条请求实际送入模型的严格 token 数。
- 流式读取 `datasets/LongBench-Pro/longbench_pro.json`，默认从第零条记录开始选择不同 context。`--dataset` 和 `--sample-start` 可覆盖。
- 使用当前模型分词器截取原始 context 的 token 前缀；不使用数据集标签中的 Qwen token 数。长度不够的记录会跳过；数据集不足时明确报错，不拼接或重复文本。
- 所有请求等长，输入是连续的 `[B,L]` 张量；这样先隔离 B/L 的影响，不引入不同 padding 比例。同一 B/L 扫描复用相同的 token 前缀，较小 B 使用相同样本列表的前 B 条。
- 模型处于评估和 inference mode。每次前向创建空的独立 `DynamicCache`，计时包含所选 Transformer 层的计算和本轮 KV 写入。
- 计时从第一层调用前开始，到最后一层完成并同步后结束。分词、数据集读取、CPU/GPU 输入搬运、embedding、RoPE、掩码准备、final norm、LM head、通信与写文件均在计时外。
- eager 使用显式因果掩码；等长空 KV 的 SDPA 使用其 `is_causal` 路径。
- 每种形状先独立预热，再正式重复测量；形状顺序按 `--seed` 打乱。轮次之间保留 CUDA allocator 的可复用块，每轮 KV 内容独立；OOM 后回收并继续其他形状。

结果是所选 Transformer 前缀的 prefill 前向时延，不是完整请求完成时间，也不包含 pipeline 等待或传输。不同层数、attention 后端、设备、精度需要分别标定。

## 输出目录

每次执行新建：

```text
results/batch_forward_latency/<北京时间时间戳>-<运行标识>/
```

可用 `--output-root` 指定其他根目录。所有相对输入/输出路径均以项目根目录为基准。

| 文件 | 内容 |
| --- | --- |
| `trials.csv` | 每次预热、正式测量、OOM 和上下文超限记录；每行及时刷新 |
| `run.json` | 参数、执行顺序、设备/软件环境、执行状态和进行中的形状 |
| `input_manifest.json` | 源样本索引、ID、截取 token 数及 token 前缀 SHA-256 |
| `summary.csv` | 每个 B/L 的成功状态、重复次数、中位数、p10/p90、CV 与峰值显存 |
| `latency_vs_batch_size.png` | 横轴 B，不同曲线对应不同 L |
| `latency_vs_prompt_length.png` | 横轴 L，不同曲线对应不同 B |
| `latency_map.png` | B/L 网格上的时延中位数，标注失败或未完成的格子 |
| `balance_rule.json` | 实测表、候选模型验证误差，以及通过验证后的函数系数 |
| `equivalent_batches.csv` | B 与 L 反向变化、实测时延满足容差的 batch 对 |
| `analysis_summary.json` | 成功形状数、匹配数量和拟合采纳状态 |

曲线只绘制正式重复次数完整、没有失败的形状；中位数为折线，p10 至 p90 为描述性波动范围，不是置信区间。OOM、超限、部分完成和未测点不参与拟合，不补点连线。峰值 allocated 包含常驻权重、预备输入和 KV；reserved 还可能包含此前形状留下的 allocator 块。

中断后可只分析已完成数据，无需加载模型：

```bash
python analysis/batch_forward_latency.py \
  --analyze-only results/batch_forward_latency/<运行目录>
```

只分析需要 NumPy 和 matplotlib。若需重新选择近似相等容差，可增加 `--relative-tolerance`。

## 推荐函数与实测对照

近似相等按两个时延中位数的对称相对差定义：

$$
\frac{|t_A-t_B|}{(t_A+t_B)/2}\le\varepsilon.
$$

默认 $\varepsilon=0.05$。参与实测推荐的两个测量点还必须各自满足 $\mathrm{CV}\le\varepsilon/2$，避免高波动的中位数碰巧接近。此规则描述一次标定运行中的接近程度，不是统计等效性检验。

拟合比较非负系数模型 `t = c0 + cBL*B*L` 与 `t = c0 + B*(cB + cBL*L + cBL2*L**2)`。至少需要八种完整成功形状，覆盖两个 B 和三个 L。每次验证留出整个 B/L 形状，其全部重复都不进入训练；所有验证点的相对预测误差必须不超过 $\varepsilon/2$，且测量 CV 也不超过该门槛。多个模型通过时，优先选择参数更少的模型。

函数通过后，`balance_rule.json` 保存如下可逆规则的实际系数：

```python
t_ms = c0 + B * (cB + cBL * L + cBL2 * L**2)
B_continuous = (target_ms - c0) / (cB + cBL * L + cBL2 * L**2)
```

请求数必须为整数；推荐函数检查相邻整数，并限制在已有成功测量覆盖的范围。未直接测过的推荐明确标记为 `model_estimate` 和 `requires_measurement=True`。已有实测值优先于预测值，模型不能把实测不匹配的形状重新推荐为匹配。

可在项目根目录使用以下接口，选择实际测过的参考 batch，并查询另一 L 对应的 B：

```python
import json
from pathlib import Path
from analysis.batch_latency_balance import recommend_equivalent_batch

rule = json.loads(Path("results/batch_forward_latency/<运行目录>/balance_rule.json").read_text())
recommendation = recommend_equivalent_batch(
    rule,
    reference_batch_size=4,
    reference_prompt_tokens=512,
    target_prompt_tokens=1024,
)
print(recommendation)
```

调用中的 `4, 512, 1024` 是查询参数，实际是否接近由该运行的测量决定。返回 `source="measured"` 且 `within_tolerance=True` 时，候选有实测支持；`source="model_estimate"` 时需要补测确认。若没有满足条件的整数 B，则 `within_tolerance=False`。

拟合不通过时，同一接口仍优先在实测数据中查找匹配，`equivalent_batches.csv` 提供可直接查看的 B/L 对；`summary.csv` 保留完整扫描表。当前网格没有匹配时不虚构等时延组合，可缩小 B/L 步长或增加重复次数后再标定。

## 静态检查与回归测试

测试文件为 `test/unittest_batch_forward_latency.py`，检查数据读取、严格 token 截取、预热和失败排除、拟合质量门槛、实测优先及整数推荐。需要 NumPy，不需要模型权重或 GPU。

```bash
python -B test/unittest_batch_forward_latency.py
```

这些逻辑测试与实际模型的性能测量分开。实际前向、OOM 恢复和图形布局需在目标服务器执行后核对。
