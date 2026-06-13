"""
手动发送 pipeline config 的示例脚本。

实际实验时，请把 NODE_IPS、src_addr、dst_addr、node_addr 改成 Jetson 上的真实
IP/端口。每个 ConfigSenderPipeline 实例需要保持存活，否则 PUSH socket 可能在
接收方处理前断开。
"""

import time

import zmq


class ConfigSenderPipeline:
    """
    Pipeline 版本 config sender。

    这个类只用于手动测试；未来中控设备实现后，可以直接发送同样字段的 JSON。
    """

    def __init__(self, node_port: int = 40700):
        self.node_port = node_port
        self.node_addr = ""
        self.config: dict = {}
        self.context = zmq.Context.instance()
        self.send_socket = self.context.socket(zmq.PUSH)

    def build_config(
        self,
        shards_start: int,
        shards_end: int,
        can_receive_user_request: bool,
        src_addr: str,
        dst_addr: str,
        node_addr: str,
        first_node_addr: str,
        pipeline_depth: int,
        max_active_requests: int | None = None,
        node_id: str = "",
    ) -> None:
        """
        构造 pipeline 节点配置。

        :param src_addr: 节点本地 bind 地址，通常为 tcp://*:40800。
        :param dst_addr: 模型链下一 stage 的可连接地址；末节点用于 clear 绕回首节点。
        :param node_addr: 当前节点可被其他节点 connect 的地址，不能使用 tcp://*:port。
        :param first_node_addr: 模型链首分片可连接地址。
        :param pipeline_depth: 模型链 stage 数。
        :param max_active_requests: 最大 active 请求数，默认等于 pipeline_depth。
        :param node_id: request_id/log 前缀；为空时使用 node_addr。
        """

        if "*" in node_addr:
            raise ValueError(
                "node_addr must be connectable, for example tcp://172.16.0.2:40800"
            )
        if not first_node_addr:
            raise ValueError("first_node_addr cannot be empty")
        if pipeline_depth <= 0:
            raise ValueError("pipeline_depth must be positive")

        self.config = {
            "src_addr": src_addr,
            "dst_addr": dst_addr,
            "node_addr": node_addr,
            "first_node_addr": first_node_addr,
            "can_receive_user_request": can_receive_user_request,
            "shards_start": shards_start,
            "shards_end": shards_end,
            "pipeline_depth": pipeline_depth,
            "max_active_requests": max_active_requests or pipeline_depth,
            "node_id": node_id or node_addr,
        }

    def send_config(self, node_ip: str) -> None:
        """向某个 pipeline_start_node.py 进程发送 config。"""

        if self.node_addr:
            self.send_socket.disconnect(self.node_addr)
        self.node_addr = "tcp://" + node_ip + ":" + str(self.node_port)
        self.send_socket.connect(self.node_addr)
        self.send_socket.send_json(self.config)


if __name__ == "__main__":
    # 示例：Llama 3.2 3B 有 28 个 layer，4 个 stage 各放 7 层。
    # 末 stage 的 dst_addr 指回 first node，仅用于 pipeline_clear 绕回首节点。
    # 请按你的 Jetson IP 和切片边界修改。
    pipeline_depth = 4
    max_active_requests = pipeline_depth
    first_node_addr = "tcp://172.16.0.4:40800"

    senders = [ConfigSenderPipeline(node_port=40700) for _ in range(pipeline_depth)]

    senders[0].build_config(
        shards_start=0,
        shards_end=7,
        can_receive_user_request=True,
        src_addr="tcp://*:40800",
        dst_addr="tcp://172.16.0.5:40800",
        node_addr="tcp://172.16.0.4:40800",
        first_node_addr=first_node_addr,
        pipeline_depth=pipeline_depth,
        max_active_requests=max_active_requests,
        node_id="node0",
    )
    senders[0].send_config(node_ip="172.16.0.4")

    senders[1].build_config(
        shards_start=7,
        shards_end=14,
        can_receive_user_request=True,
        src_addr="tcp://*:40800",
        dst_addr="tcp://172.16.0.6:40800",
        node_addr="tcp://172.16.0.5:40800",
        first_node_addr=first_node_addr,
        pipeline_depth=pipeline_depth,
        max_active_requests=max_active_requests,
        node_id="node1",
    )
    senders[1].send_config(node_ip="172.16.0.5")

    senders[2].build_config(
        shards_start=14,
        shards_end=21,
        can_receive_user_request=True,
        src_addr="tcp://*:40800",
        dst_addr="tcp://172.16.0.7:40800",
        node_addr="tcp://172.16.0.6:40800",
        first_node_addr=first_node_addr,
        pipeline_depth=pipeline_depth,
        max_active_requests=max_active_requests,
        node_id="node2",
    )
    senders[2].send_config(node_ip="172.16.0.6")

    senders[3].build_config(
        shards_start=21,
        shards_end=28,
        can_receive_user_request=True,
        src_addr="tcp://*:40800",
        dst_addr="tcp://172.16.0.4:40800",
        node_addr="tcp://172.16.0.7:40800",
        first_node_addr=first_node_addr,
        pipeline_depth=pipeline_depth,
        max_active_requests=max_active_requests,
        node_id="node3",
    )
    senders[3].send_config(node_ip="172.16.0.7")

    print("[CONFIG] pipeline configs sent. Keep this process alive to retain sockets.")
    while True:
        time.sleep(3600)
