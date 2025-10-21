import os
import gc
import zmq
import torch
from transformers import LlamaConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from utils.shard_loader import LlamaShardPart
from utils.forwarding_utils import build_position_ids


class Communicator:
    def __init__(self, src_addr: str, dst_addr: str):
        """
        :param src_addr: 从该地址接收 隐藏层 / next token id 的数据（使用 socket.bind，应为本机地址，如 tcp://*:port 或指定一个本机ip）
        :param dst_addr: 向该地址发送 隐藏层 / next token id 的数据（使用 socket.connect，应为对方地址）
        """
        self.src_addr = src_addr
        recv_context = zmq.Context()
        self.recv_socket = recv_context.socket(zmq.PULL)
        self.recv_socket.bind(self.src_addr)

        self.dst_addr = dst_addr
        send_context = zmq.Context()
        self.send_socket = send_context.socket(zmq.PUSH)
        self.send_socket.connect(self.dst_addr)

    def change_src_addr(self, new_src_addr: str) -> str:
        self.recv_socket.unbind(self.src_addr)
        self.recv_socket.bind(new_src_addr)
        self.src_addr = new_src_addr
        return self.src_addr

    def change_dst_addr(self, new_dst_addr: str) -> str:
        self.send_socket.disconnect(self.dst_addr)
        self.send_socket.connect(new_dst_addr)
        self.dst_addr = new_dst_addr
        return self.dst_addr

    def transfer_data(self, data: torch.Tensor | dict,
                      data_path: str = "results/send_data.pt", keep_data: bool = False) -> None | str:
        # ! 若此处有修改，请一并修改 NodeController 类中的 _forward_request 方法
        os.makedirs(os.path.dirname(data_path), exist_ok=True)  # 确保父目录存在
        torch.save(data, data_path)
        with open(data_path, "rb") as f:
            self.send_socket.send(f.read())
        if not keep_data:
            os.remove(data_path)
            return None
        return data_path

    def receive_data(self, no_block: bool = False,
                     data_path: str = "results/recv_data.pt", keep_data: bool = False) -> torch.Tensor | dict:
        if no_block:
            received_data = self.recv_socket.recv(flags=zmq.NOBLOCK)
        else:
            received_data = self.recv_socket.recv()
        with open(data_path, "wb") as f:
            f.write(received_data)
        data = torch.load(data_path)
        if not keep_data:
            os.remove(data_path)
        return data


class NodeWorker:
    # 每个 node 上运行的 client
    def __init__(self, src_addr: str, dst_addr: str,
                 can_receive_user_request: bool, shards_path: str, device="cpu", dtype=torch.float16):
        """
        :param can_receive_user_request: 是否接收用户请求（关系到是否加载分词器、嵌入层）
        :param shards_path: 切片路径
        :param device: "cpu" 或 "cuda:0" 等
        :param dtype: torch.float32 / torch.float16 等
        """
        self.communicator = Communicator(src_addr=src_addr, dst_addr=dst_addr)
        self.can_receive_user_request = can_receive_user_request
        self.shards_path = shards_path
        self.device = torch.device(device)
        self.dtype = dtype
        self.config = LlamaConfig.from_pretrained(self.shards_path)
        self.layer_num = self.config.num_hidden_layers
        # self.num_heads = self.config.num_attention_heads
        # self.hidden_size = self.config.hidden_size
        # self.head_dim = self.hidden_size // self.num_heads

        self.tokenizer = None  # 分词器
        self.embed_tokens = None  # 嵌入层
        self.rope = None  # 旋转位置编码（RoPE）
        self.shard = None  # （对应分片）隐藏层权重
        self.past_key_value = None  # KV cache
        self.lm_head = None

        # 运行时的缓存数据
        self.start = 0
        self.end = 0
        if can_receive_user_request:
            self.batch_size = 0
            self.generated_ids = []
            self._load_embedding()

    def _load_embedding(self) -> None:
        """
        加载嵌入层
        :return: None
        """
        print("[INFO] loading tokenizer...")
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.shards_path)
        print("[INFO] tokenizer loaded.")
        # 加载嵌入层
        print("[INFO] loading embedding layer...")
        self.embed_tokens = torch.nn.Embedding(
            self.config.vocab_size, self.config.hidden_size).to(self.device, dtype=self.dtype)
        self.embed_tokens.load_state_dict(
            torch.load(os.path.join(self.shards_path, "embedding.pth"), map_location=self.device))
        print("[INFO] embedding layer loaded.")

    def load_shards(self, start: int, end: int) -> None:
        """
        加载切片并删除旧切片（如有），从start到end（不包括end）
        :param start:
        :param end:
        :return: None
        """
        if start < 0 or start >= end or end > self.layer_num:
            raise ValueError("[ERROR] start or end is invalid")
        self.start = start
        self.end = end

        try:
            del self.rope
            del self.shard
            del self.lm_head
        except AttributeError:
            pass
        gc.collect()  # 强制 Python 做一次垃圾回收
        if self.device.type == "cuda":
            torch.cuda.empty_cache()  # 清空 CUDA 缓存池，让 nvidia-smi 立刻下降

        if self.start == 0:
            # 加载旋转位置编码（RoPE）
            print("[INFO] loading RoPE...")
            self.rope = LlamaRotaryEmbedding(config=self.config, device=self.device).to(self.device)
            print("[INFO] RoPE loaded.")

        if self.end == self.layer_num:
            add_final_norm = True
            final_norm_weight = "final_norm.pth"
            # 加载 lm_head
            print("[INFO] loading lm_head...")
            self.lm_head = torch.nn.Linear(
                self.config.hidden_size, self.config.vocab_size, bias=False).to(self.device, dtype=self.dtype)
            self.lm_head.load_state_dict(
                torch.load(os.path.join(self.shards_path, "lm_head.pth"), map_location=self.device))
            print("[INFO] lm_head loaded.")
        else:
            add_final_norm = False
            final_norm_weight = None

        print(f"[INFO] loading hidden layer {start}~{end}(end excluded)...")
        self.shard = LlamaShardPart(
            self.shards_path,
            ["block_" + str(i) + ".pth" for i in range(start, end)],
            start, end,
            device=self.device,
            dtype=self.dtype,
            add_final_norm=add_final_norm,
            final_norm_weight=final_norm_weight
        )
        self.shard.eval()
        print(f"[INFO] hidden layer {start}~{end}(end excluded) loaded.")

        # 初始化 KV cache
        print("[INFO] loading KV cache...")
        self.past_key_value = DynamicCache()
        print("[INFO] KV cache loaded.")

    def receive_user_request(self, request: str = "Write a poem about the blue sky.") -> dict:
        """
        接收用户请求
        :return: input_token_info: dict，包含隐藏层参数和用于 RoPE 的 batch_size & seq_len
        """
        if not self.can_receive_user_request:
            raise RuntimeError("[ERROR] this node does not have embedding layer while receiving user request.")

        input_text = request
        print("[INFO] input: " + input_text)

        # 分词器tokenize
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]  # 初始 prompt 张量化后的 token id 序列
        self.generated_ids = [input_ids]  # 存储所有 input 和 output 的 token id 序列

        # 经过嵌入层
        hidden_states = self.embed_tokens(input_ids)  # [B, S, H]
        batch_size, seq_len, _ = hidden_states.shape
        self.batch_size = batch_size

        input_token_info = {
            "hidden_states": hidden_states,
            "batch_size": batch_size,
            "seq_len": seq_len,
        }
        return input_token_info

    def pass_through_shard(self, state_info: dict) -> torch.Tensor | dict:
        """
        对隐藏层前向传播
        :param state_info: 传递hidden_states。若存在batch_size和seq_len：先计算RoPE的cos,sin；若存在cos,sin：直接传入shard
        :return: next_token_id: torch.Tensor | next_state_info: dict
        """
        if "batch_size" in state_info and "seq_len" in state_info:
            if self.start != 0:
                raise RuntimeError("[ERROR] after embedding layer, the states should first passing hidden layer 0!")
            # 旋转位置编码（RoPE）获取 cos/sin 表
            position_ids = build_position_ids(self.past_key_value, state_info["seq_len"], device=self.device,
                                              batch_size=state_info["batch_size"])
            cos, sin = self.rope(state_info["hidden_states"], position_ids)  # [B, S, Hd] each; hidden_states 只是用作参考张量 x
        else:
            cos, sin = state_info["cos"], state_info["sin"]
        hidden_states = state_info["hidden_states"]

        hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
        next_hidden_states = self.shard(
            hidden_states,
            # attention_mask=inputs["attention_mask"],
            past_key_value=self.past_key_value,
            rotary_emb=(cos, sin)
        )

        if self.end == self.layer_num:
            # lm_head 输出 logits
            logits = self.lm_head(next_hidden_states)  # [batch, seq_len, vocab_size]
            # 取最后一个位置的预测 token
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)  # [1]
            return next_token_id
        else:
            next_state_info = {
                "hidden_states": next_hidden_states,
                "cos": cos,
                "sin": sin,
            }
            return next_state_info

    def receive_next_token(self, next_token_id: torch.Tensor) -> tuple:
        """
        接收最后一层的输出token id
        :param next_token_id:
        :return: (reached_eos: bool, input_token_info: dict|None): tuple 如果reached_eos=True，则input_token_info=None
        """
        if not self.can_receive_user_request:
            raise RuntimeError(
                '[ERROR] this node does not store the "generated token id" list, but received a next_token_id.')
        # 拼接到已有序列
        self.generated_ids.append(next_token_id.unsqueeze(-1))
        # 解码
        next_token = self.tokenizer.decode(next_token_id.item())
        print(repr(next_token), end=" ", flush=True)

        reached_eos = (next_token == self.tokenizer.eos_token)
        if reached_eos:
            # 解码最终结果
            print()
            final_ids = torch.cat(self.generated_ids, dim=-1)  # [B, seq_len + out_token_num]
            print("output: ", self.tokenizer.decode(final_ids[0]))
            return reached_eos, None
        else:
            # 更新输入
            hidden_states = self.embed_tokens(next_token_id.unsqueeze(0))  # [B, 1, H]
            seq_len = 1
            input_token_info = {
                "hidden_states": hidden_states,
                "batch_size": self.batch_size,
                "seq_len": seq_len,
            }
            return reached_eos, input_token_info


class NodeController:
    """
    节点控制器，根据主控节点 master node 发送的配置文件自动运行 NodeWorker
    """

    def __init__(self, shards_path: str, device: str, dtype: torch.dtype,
                 listen_port: int = 40700):
        self.shards_path = shards_path
        self.device = device
        self.dtype = dtype

        # 初始化接收配置文件的 socket 以接收主控节点的配置文件
        self.listen_addr = "tcp://*:" + str(listen_port)
        recv_config_context = zmq.Context()
        self.recv_config_socket = recv_config_context.socket(zmq.PULL)
        self.recv_config_socket.bind(self.listen_addr)

        # 接收配置文件
        self.received_config = self._receive_config()

        # 若 can_receive_user_request=True，则需要向 first_node_addr 发送 “经过嵌入层处理后的”用户请求（从而保护用户隐私）
        if self.received_config["can_receive_user_request"]:
            self.first_node_addr = self.received_config["first_node_addr"]
            send_request_context = zmq.Context()
            self.send_request_socket = send_request_context.socket(zmq.PUSH)
            self.send_request_socket.connect(self.first_node_addr)

        # 创建节点实例并加载分片
        self.node_worker = NodeWorker(
            src_addr=self.received_config["src_addr"],
            dst_addr=self.received_config["dst_addr"],
            can_receive_user_request=self.received_config["can_receive_user_request"],
            shards_path=self.shards_path,
            device=self.device,
            dtype=self.dtype
        )
        self.node_worker.load_shards(self.received_config["shards_start"], self.received_config["shards_end"])
        print("[INFO] Node is ready.")

    def _receive_config(self, no_block: bool = False) -> dict | None:
        if no_block:
            print("[CONFIG] Checking if there has a new configuration file from master node...")
            try:
                received_config = self.recv_config_socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                return None
        else:
            print("[CONFIG] Waiting for configuration file from master node...")
            received_config = self.recv_config_socket.recv_json()
        print("[CONFIG] Received configuration file from master node:")
        for k, v in received_config.items():
            print(f"  - {k}: {v}")
        return received_config

    def _change_first_node_addr(self, new_first_node_addr: str) -> str:
        self.send_request_socket.disconnect(self.first_node_addr)
        self.send_request_socket.connect(new_first_node_addr)
        self.first_node_addr = new_first_node_addr
        return self.first_node_addr

    def check_new_config(self) -> None:
        """
        更改配置文件，并重新加载分片
        """
        # 接收配置文件
        new_config = self._receive_config(no_block=True)
        # 如果没有发来新的 config，直接退出该检查函数
        if not new_config:
            return
        # 如果 can_receive_user_request 被更改，只能重新创建节点实例
        if new_config["can_receive_user_request"] != self.received_config["can_receive_user_request"]:
            del self.node_worker
            # 创建节点实例并加载分片
            self.node_worker = NodeWorker(
                src_addr=new_config["src_addr"],
                dst_addr=new_config["dst_addr"],
                can_receive_user_request=new_config["can_receive_user_request"],
                shards_path=self.shards_path,
                device=self.device,
                dtype=self.dtype
            )
            self.node_worker.load_shards(new_config["shards_start"], new_config["shards_end"])
        else:
            # 无需重新创建节点实例，调用方法更改配置即可
            self.node_worker.communicator.change_src_addr(new_config["src_addr"])
            self.node_worker.communicator.change_dst_addr(new_config["dst_addr"])
            self._change_first_node_addr(new_config["first_node_addr"])
            self.node_worker.load_shards(new_config["shards_start"], new_config["shards_end"])
        self.received_config = new_config
        print("[INFO] The new configuration node is ready.")

    def _forward_request(self, data: dict,
                         data_path: str = "results/send_request.pt", keep_data: bool = False) -> None | str:
        os.makedirs(os.path.dirname(data_path), exist_ok=True)  # 确保父目录存在
        torch.save(data, data_path)
        with open(data_path, "rb") as f:
            self.send_request_socket.send(f.read())
        if not keep_data:
            os.remove(data_path)
            return None
        return data_path

    def receive_request(self, request: str = "Write a poem about the blue sky.") -> None:
        if not self.node_worker.can_receive_user_request:
            raise RuntimeError("[ERROR] this node cannot receive user request.")
        token_info = self.node_worker.receive_user_request(request)
        self._forward_request(token_info)

    def run_worker_loop(self) -> None:
        """
        常驻监听有无传入数据待处理，若无则检查是否有新请求进入
        """
        skip_this_transfer = False  # 用于在得到 <EOS> 后，通过不继续传递 state 的方式来结束该推理任务

        # TODO 临时实现仅发送一次请求
        # request_not_send = True
        request_not_send = False

        while True:
            try:
                # 监听有无传入数据待处理
                received_data = self.node_worker.communicator.receive_data(no_block=True)
                # 若为从模型链末尾传回的 next_token_id
                if type(received_data) == torch.Tensor:
                    if self.node_worker.start != 0:
                        raise RuntimeError(
                            "[ERROR] I'm not the first node in the model chain, but received \"next_token_id\".")
                    # 需要先解码并输出 token，然后产生下一个 state
                    reached_eos, received_data = self.node_worker.receive_next_token(received_data)
                    if reached_eos:
                        skip_this_transfer = True
                elif hasattr(received_data, "batch_size") and hasattr(received_data, "seq_len"):
                    if self.node_worker.start != 0:
                        raise RuntimeError(
                            "[ERROR] I'm not the first node in the model chain, but received \"input_token_info\".")
                elif hasattr(received_data, "cos") and hasattr(received_data, "sin"):
                    if self.node_worker.start == 0:
                        raise RuntimeError(
                            "[ERROR] I'm the first node in the model chain, but received \"next_state_info\".")
                else:
                    attrs = [a for a in dir(received_data) if
                             not a.startswith("__")]  # 报错时，同时打印对象的属性，并过滤掉 __xxx__ 这种内置属性
                    raise RuntimeError(
                        f"[ERROR] Received unknown data type: {type(received_data)}; "
                        f"Attributes: {attrs if attrs else 'No attributes found'}"
                    )
                processed_data = self.node_worker.pass_through_shard(received_data)
                if not skip_this_transfer:
                    self.node_worker.communicator.transfer_data(processed_data)
                else:
                    skip_this_transfer = False
            # 若无传入数据，则静默失败
            except zmq.Again:
                pass

            # 检查是否有新请求进入
            # TODO 替换为实际的用户请求仿真
            if request_not_send:
                self.receive_request()
                request_not_send = False

            # 检查是否有新的配置文件
            self.check_new_config()


if __name__ == "__main__":
    controller = NodeController(
        "shards/Llama-2-7b-chat-hf_float16",
        device="cuda:0",
        dtype=torch.float16
    )
    controller.run_worker_loop()
