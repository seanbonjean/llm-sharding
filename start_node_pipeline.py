"""
Pipeline inference 节点启动入口

运行方式和旧 start_node.py 基本一致：
    python start_node_pipeline.py --port 40700

该进程启动后会等待中控设备或 send_config_pipeline.py 发送 config
pipeline 版本在旧 config 基础上新增以下字段：
    node_addr: 当前节点可被其他节点 connect 的地址，例如 tcp://172.16.0.2:40800
    first_node_addr: 模型链首分片地址
    pipeline_depth: 模型链 stage 数，例如：4 台设备各一个 stage 时填 4
    max_active_requests: 最大 active 请求数，默认建议等于 pipeline_depth
    node_id: 用于实现日志和 request_id 前缀，不参与路由
"""

import sys
import torch
from utils.node_worker_pipeline import PipelineNodeController


def parse_port(argv: list[str]) -> int:
    """解析传入参数：controller 接收 config 的端口，兼容旧脚本的位置参数写法"""

    port = 40700  # 未传入参数时的默认端口
    if len(argv) > 2 and argv[1] == "--port":
        port = int(argv[2])
    elif len(argv) > 2 and argv[1] != "--port":
        raise ValueError(f"Unexpected argument {argv[1]}, expected '--port'")
    elif len(argv) > 1:  # 如果省略了 --port ，只有一个参数，则认为它是端口号
        print(
            f"[WARNING] Port argument provided without \"--port\" flag, interpreting \"{argv[1]}\" as port number."
        )
        port = int(argv[1])
    return port


controller = PipelineNodeController(
    "shards/Llama-3___2-3B-Instruct_float16",
    device="cuda:0",
    dtype=torch.float16,
    listen_port=parse_port(sys.argv),
)
controller.run_worker_loop(max_new_tokens=512)
