"""
Pipeline inference 节点启动入口

运行方式和旧 start_node.py 基本一致：
    python pipeline_start_node.py --port 40700
    python pipeline_start_node.py --port 40700 --cli-pages

该进程启动后会等待中控设备或 pipeline_debug.py / pipeline_send_config.py 发送 config
pipeline 版本在旧 config 基础上新增以下字段：
    node_addr: 当前节点可被其他节点 connect 的地址，例如 tcp://172.16.0.2:40800
    first_node_addr: 模型链首分片地址
    pipeline_depth: 模型链 stage 数，例如：4 台设备各一个 stage 时填 4
    max_active_requests: 最大 active 请求数，默认建议等于 pipeline_depth
    node_id: 用于实现日志和 request_id 前缀，不参与路由

用户请求入口复用 node_addr 对应的数据端口，默认是 40800。外部客户端发送
type="user_request" 的 torch 序列化 dict 后，本节点会在本地调用 receive_request()。

默认使用普通 print 输出。传入 --cli-pages 后，会启用可选 curses 多页面视图：
LOG 页显示运行日志，每个 request_id 一个实时 token 页。
"""

import sys
import torch
from utils.pipeline_node_worker import PipelineNodeController


def parse_args(argv: list[str]) -> tuple[int, bool]:
    """解析传入参数：controller 接收 config 的端口，兼容旧脚本的位置参数写法"""

    port = 40700  # 未传入参数时的默认端口
    enable_cli_pages = False
    args = argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--port":
            if i + 1 >= len(args):
                raise ValueError("--port requires a port number")
            port = int(args[i + 1])
            i += 2
        elif arg == "--cli-pages":
            enable_cli_pages = True
            i += 1
        elif arg.isdigit():
            print(
                f"[WARNING] Port argument provided without \"--port\" flag, interpreting \"{arg}\" as port number."
            )
            port = int(arg)
            i += 1
        else:
            raise ValueError(f"Unexpected argument {arg}")
    return port, enable_cli_pages


port, enable_cli_pages = parse_args(sys.argv)


controller = PipelineNodeController(
    "shards/Llama-3___2-3B-Instruct_float16",
    device="cuda:0",
    dtype=torch.float16,
    listen_port=port,
    enable_cli_pages=enable_cli_pages,
)
controller.run_worker_loop(max_new_tokens=512)
