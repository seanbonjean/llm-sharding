from utils.config_sender import ConfigSender

# sender = ConfigSender()

senders = list()
for i in range(3):
    senders.append(ConfigSender())

senders[0].build_config(shards_start=0,
                        shards_end=6,
                        can_receive_user_request=True,
                        src_addr="tcp://*:40800",
                        dst_addr="tcp://172.16.0.3:40800",
                        first_node_addr="tcp://172.16.0.2:40800",
                        )
senders[0].send_config(node_ip="172.16.0.2")

senders[1].build_config(shards_start=6,
                        shards_end=7,
                        can_receive_user_request=True,
                        src_addr="tcp://*:40800",
                        dst_addr="tcp://172.16.0.1:40800",
                        first_node_addr="tcp://172.16.0.2:40800",
                        )
senders[1].send_config(node_ip="172.16.0.3")

senders[2].build_config(shards_start=7,
                        shards_end=32,
                        can_receive_user_request=True,
                        src_addr="tcp://*:40800",
                        dst_addr="tcp://172.16.0.2:40800",
                        first_node_addr="tcp://172.16.0.2:40800",
                        )
senders[2].send_config(node_ip="172.16.0.1")

senders.append(ConfigSender(node_port=40701))
senders[3].build_config(shards_start=17,
                        shards_end=19,
                        can_receive_user_request=False,
                        src_addr="tcp://*:40801",
                        dst_addr="tcp://172.16.0.1:40801",
                        first_node_addr="tcp://172.16.0.2:40801",
                        )
senders[3].send_config(node_ip="172.16.0.2")

# 保持运行，保留 socket
while True:
    pass
