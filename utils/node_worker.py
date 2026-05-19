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
        # 将形如 tcp://*:40800 转为形如 tcp://100.115.1.1:40800
        self.actual_src_addr = self.recv_socket.getsockopt_string(zmq.LAST_ENDPOINT)

        self.dst_addr = dst_addr
        send_context = zmq.Context()
        self.send_socket = send_context.socket(zmq.PUSH)
        self.send_socket.connect(self.dst_addr)

    def change_src_addr(self, new_src_addr: str) -> str:
        self.recv_socket.unbind(self.actual_src_addr)
        self.recv_socket.bind(new_src_addr)
        self.src_addr = new_src_addr
        self.actual_src_addr = self.recv_socket.getsockopt_string(zmq.LAST_ENDPOINT)
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
    CLEAR_KV_CACHE_COMMAND = "clear_KV_cache"
    CLEAR_KV_CACHE_ORIGIN_KEY = "origin_node"  # 发起 clear KV cache 的源节点标志

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
        self.batch_size = 0  # 假设模型链是：A->B->C，如果请求是从B发起并传入A的，此时首节点A也需要一个 batch_size 属性来保存 batch_size，用于后续 receive_next_token 方法中封装新的 input_token_info
        if can_receive_user_request:
            self.generated_ids = []
            self._load_embedding()
            self.input_token_length = None  # (仅记录) 输入 token 长度

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

    @torch.inference_mode()  # 避免产生计算图
    def receive_user_request(self, request: str = "Write a poem about the blue sky.",
                             input_ids: torch.Tensor | None = None) -> dict:
        """
        接收用户请求
        :param request: 用户请求字符串
        :param input_ids: (默认不使用) 用户请求对应的 token id 序列 (仅在 profile 阶段测试不同 token 长度下的计算能力时使用)
        :return: input_token_info: dict，包含隐藏层参数和用于 RoPE 的 batch_size & seq_len
        """
        if not self.can_receive_user_request:
            raise RuntimeError("[ERROR] this node does not have embedding layer while receiving user request.")

        if input_ids is None:  # 实际只走这条路径
            input_text = request
            print("[INFO] input: " + input_text)
            # 分词器tokenize
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]  # 初始 prompt 张量化后的 token id 序列
        else:  # 仅 profile 时使用
            if input_ids.ndim != 2:
                raise ValueError("[ERROR] input_ids must be a 2D tensor with shape [batch_size, seq_len].")
            input_ids = input_ids.to(device=self.device, dtype=torch.long)
            print("[INFO] input: inputted from direct token ids.")
        self.input_token_length = input_ids.shape[1]
        print("[INFO] input token number: " + str(self.input_token_length))
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

    @torch.inference_mode()
    def pass_through_shard(self, state_info: dict) -> torch.Tensor | dict:
        """
        对隐藏层前向传播
        :param state_info: 传递hidden_states。若存在batch_size和seq_len：先计算RoPE的cos,sin；若存在cos,sin：直接传入shard
        :return: next_token_id: torch.Tensor | next_state_info: dict
        """
        if self.is_input_token_info(state_info):
            if self.start != 0:
                raise RuntimeError("[ERROR] after embedding layer, the states should first passing hidden layer 0!")
            # 在模型链的首节点中也保存一份 batch_size，用于后续 receive_next_token 方法中封装新的 input_token_info
            self.batch_size = state_info["batch_size"]
            # 旋转位置编码（RoPE）获取 cos/sin 表
            position_ids = build_position_ids(self.past_key_value, state_info["seq_len"], device=self.device,
                                              batch_size=state_info["batch_size"])
            cos, sin = self.rope(state_info["hidden_states"], position_ids)  # [B, S, Hd] each; hidden_states 只是用作参考张量 x
        elif self.is_next_state_info(state_info):
            cos, sin = state_info["cos"], state_info["sin"]
        else:
            attrs = [a for a in dir(state_info) if not a.startswith("__")]
            raise RuntimeError(
                f"[ERROR] Received unknown state_info type: {type(state_info)}; "
                f"Attributes: {attrs if attrs else 'No attributes found'}"
            )
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

    @torch.inference_mode()
    def receive_next_token(self, next_token_id: torch.Tensor, max_new_tokens: int = 1024) -> tuple:
        """
        接收最后一层的输出token id
        :param next_token_id:
        :return: (reached_end: bool, input_token_info: dict|None): tuple 如果reached_end=True，则input_token_info=None
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
        reached_max_new_tokens = len(self.generated_ids) > max_new_tokens  # 优化自 len(self.generated_ids) - 1 >= max_new_tokens ，因为 generated_ids[0] 是输入 prompt，不被计入 output
        reached_end = reached_eos or reached_max_new_tokens
        if reached_end:
            # 解码最终结果
            print()
            final_ids = torch.cat(self.generated_ids, dim=-1)  # [B, seq_len + out_token_num]
            print("output: ", self.tokenizer.decode(final_ids[0]))
            print("\n[INFO] output token number: " + str(len(self.generated_ids) - 1))
            return reached_end, None
        else:
            # 更新输入
            hidden_states = self.embed_tokens(next_token_id.unsqueeze(0))  # [B, 1, H]
            seq_len = 1
            input_token_info = {
                "hidden_states": hidden_states,
                "batch_size": self.batch_size,
                "seq_len": seq_len,
            }
            return reached_end, input_token_info

    @staticmethod
    def is_input_token_info(data: object) -> bool:
        return isinstance(data, dict) and "batch_size" in data and "seq_len" in data

    @staticmethod
    def is_next_state_info(data: object) -> bool:
        return isinstance(data, dict) and "cos" in data and "sin" in data

    def clear_KV_cache(self) -> None:
        """
        清除上一次用户请求留下的推理状态

        保留已经加载好的 tokenizer / embedding / RoPE / shard / lm_head 等模型资源，
        只重置一次请求内会增长或被覆盖的运行时状态
        """
        old_past_key_value = self.past_key_value
        self.past_key_value = None
        self.batch_size = 0

        if self.can_receive_user_request:
            self.generated_ids = []
            self.input_token_length = None

        if old_past_key_value is not None:
            # 尝试通过 reset 方法重置缓存
            reset_cache = getattr(old_past_key_value, "reset", None)
            if callable(reset_cache):
                try:
                    reset_cache()
                except NotImplementedError:
                    pass
            # 尝试通过常见的内部容器清理缓存
            for cache_attr in ("key_cache", "value_cache"):
                cache_storage = getattr(old_past_key_value, cache_attr, None)
                if hasattr(cache_storage, "clear"):
                    cache_storage.clear()
            del old_past_key_value

        gc.collect()  # 强制 Python 做一次垃圾回收
        if self.device.type == "cuda":
            torch.cuda.empty_cache()  # 清空 CUDA 缓存池，让 nvidia-smi 立刻下降

        if self.shard is not None:
            self.past_key_value = DynamicCache()
        print("\n[INFO] KV cache and all states caused by prev user are cleared.")

    def _get_clear_KV_cache_origin(self) -> dict:
        """
        通过节点的一些配置信息，识别“发起 clear KV cache 命令”的源节点，以停止继续往模型链后方节点传递 clear KV cache 命令
        """
        return {
            "src_addr": self.communicator.src_addr,
            "dst_addr": self.communicator.dst_addr,
            "shards_start": self.start,
            "shards_end": self.end,
        }

    def build_clear_KV_cache_command(self) -> dict:
        return {
            "command": self.CLEAR_KV_CACHE_COMMAND,
            self.CLEAR_KV_CACHE_ORIGIN_KEY: self._get_clear_KV_cache_origin(),
        }

    @classmethod
    def is_clear_KV_cache_command(cls, data: object) -> bool:
        return isinstance(data, dict) and data.get("command") == cls.CLEAR_KV_CACHE_COMMAND

    def is_clear_KV_cache_command_origin(self, data: object) -> bool:
        return (
            self.is_clear_KV_cache_command(data)
            and data.get(self.CLEAR_KV_CACHE_ORIGIN_KEY) == self._get_clear_KV_cache_origin()
        )


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
            # print("[CONFIG] Checking if there has a new configuration file from master node...")
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

    def run_worker_loop(self, max_new_tokens: int = 1024) -> None:
        """
        常驻监听有无传入数据待处理，若无则检查是否有新请求进入
        """
        # TODO 临时实现仅发送一次请求
        # request_not_send = True
        request_not_send = False

        while True:
            try:
                should_pass_through_shard = True  # 得到 <EOS> 后，不再进行计算和传递数据
                # 监听有无传入数据待处理
                received_data = self.node_worker.communicator.receive_data(no_block=True)
                # 若为 clear KV cache 命令
                if self.node_worker.is_clear_KV_cache_command(received_data):
                    # 且不是发起 clear KV cache 命令的源节点 (否则停止继续传递，防止重复清除和死循环)
                    if not self.node_worker.is_clear_KV_cache_command_origin(received_data):
                        # 源节点发起 clear KV Cache 命令后，顺着模型链进行一次命令传递，命令回到源节点后中止传递任何数据，通过不继续传递信息的方式来结束此次推理任务
                        self.node_worker.communicator.transfer_data(received_data)
                        self.node_worker.clear_KV_cache()
                    continue
                # 若为从模型链末尾传回的 next_token_id
                if isinstance(received_data, torch.Tensor):
                    if self.node_worker.start != 0:
                        raise RuntimeError(
                            "[ERROR] I'm not the first node in the model chain, but received \"next_token_id\".")
                    # 需要先解码并输出 token，然后产生下一个 state
                    reached_end, received_data = self.node_worker.receive_next_token(received_data, max_new_tokens)
                    if reached_end:
                        clear_KV_cache_command = self.node_worker.build_clear_KV_cache_command()
                        self.node_worker.communicator.transfer_data(clear_KV_cache_command)  # 发起 clear KV Cache 命令，通知模型链后续节点也清除KV Cache
                        self.node_worker.clear_KV_cache()
                        should_pass_through_shard = False
                elif self.node_worker.is_input_token_info(received_data):
                    if self.node_worker.start != 0:
                        raise RuntimeError(
                            "[ERROR] I'm not the first node in the model chain, but received \"input_token_info\".")
                elif self.node_worker.is_next_state_info(received_data):
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
                if should_pass_through_shard:
                    processed_data = self.node_worker.pass_through_shard(received_data)
                    self.node_worker.communicator.transfer_data(processed_data)
                    del received_data
                    del processed_data
                    if self.node_worker.start != 0:
                        print('*', end=' ', flush=True)  # 当有数据传过时，如果不是首节点，不会输出 token，这里打印一个 * 表示有数据经过该节点
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
