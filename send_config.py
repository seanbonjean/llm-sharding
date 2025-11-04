from utils.config_sender import ConfigSender

# sender = ConfigSender()

senders = list()
for i in range(3):
    senders.append(ConfigSender())

senders[0].build_config(shards_start=0,
                        shards_end=7,
                        can_receive_user_request=True,
                        src_addr="tcp://*:40800",
                        dst_addr="tcp://100.115.1.2:40800",
                        first_node_addr="tcp://100.115.1.1:40800",
                        )
senders[0].send_config(node_ip="100.115.1.1")

senders[1].build_config(shards_start=7,
                        shards_end=9,
                        can_receive_user_request=True,
                        src_addr="tcp://*:40800",
                        dst_addr="tcp://100.89.105.75:40800",
                        first_node_addr="tcp://100.115.1.1:40800",
                        )
senders[1].send_config(node_ip="100.115.1.2")

senders[2].build_config(shards_start=9,
                        shards_end=32,
                        can_receive_user_request=True,
                        src_addr="tcp://*:40800",
                        dst_addr="tcp://100.115.1.1:40800",
                        first_node_addr="tcp://100.115.1.1:40800",
                        )
senders[2].send_config(node_ip="100.89.105.75")

# 保持运行，保留 socket
while True:
    pass
