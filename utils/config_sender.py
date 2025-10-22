import zmq


class ConfigSender:
    """
    配置发送器，向节点控制器发送配置文件以设置一个节点
    使用时，需要每个节点都对应一个 ConfigSender 实例，并且对应节点退出前不能释放对应的 ConfigSender 实例，以保留 socket
    """

    def __init__(self, node_port: int = 40700):
        self.node_port = node_port
        self.node_addr = ""
        self.config = dict()
        send_context = zmq.Context()
        self.send_socket = send_context.socket(zmq.PUSH)

    def build_config(self, shards_start: int, shards_end: int,
                     can_receive_user_request: bool,
                     dst_ip: str, dst_port: int = 40801,
                     src_addr: str = "tcp://*:40800",
                     first_node_addr: str = ""):
        """
        :param shards_start: 加载的起始层
        :param shards_end: 加载的最终层（不包括本身）
        :param can_receive_user_request: 见 node_worker.py 的 NodeWorker 类
        :param first_node_addr: 当 can_receive_user_request = True 时，为其指定一个模型链的第一个 node
        :param dst_ip: 见 node_worker.py 的 Communicator 类
        :param dst_port: 见 node_worker.py 的 Communicator 类
        :param src_addr: 见 node_worker.py 的 Communicator 类
        """
        if can_receive_user_request:
            if first_node_addr == "":
                raise ValueError("first_node_addr cannot be empty when can_receive_user_request = True")

        dst_addr = "tcp://" + dst_ip + ":" + str(dst_port)
        self.config = {
            "src_addr": src_addr,
            "dst_addr": dst_addr,
            "can_receive_user_request": can_receive_user_request,
            "first_node_addr": first_node_addr,
            "shards_start": shards_start,
            "shards_end": shards_end
        }

    def send_config(self, node_ip: str) -> None:
        if self.node_addr != "":
            self.send_socket.disconnect(self.node_addr)
        self.node_addr = "tcp://" + node_ip + ":" + str(self.node_port)
        self.send_socket.connect(self.node_addr)
        self.send_socket.send_json(self.config)


if __name__ == "__main__":
    sender = ConfigSender()
    sender.build_config(shards_start=0,
                        shards_end=10,
                        can_receive_user_request=True,
                        dst_ip="100.115.1.2",
                        first_node_addr="tcp://100.115.1.1:40800",
                        )
    sender.send_config(node_ip="100.115.1.1")
    # 测试中发现如果发送完毕后进程直接结束，会主动断开连接 [TCP RST]，导致接收节点来不及接收配置文件
    # 所以这里添加一个 while True: 循环，保持运行，保留 socket
    while True:
        pass
    # 实际使用时因为 sender 实例一直都在，所以不用停顿
