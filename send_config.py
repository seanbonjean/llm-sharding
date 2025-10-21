from utils.config_sender import ConfigSender

sender = ConfigSender()
sender.build_config(shards_start=0,
                    shards_end=10,
                    can_receive_user_request=True,
                    dst_ip="100.115.1.2",
                    first_node_addr="tcp://100.115.1.1:40800",
                    )
sender.send_config(node_ip="100.115.1.1")

# 保持运行，保留 socket
while True:
    pass
