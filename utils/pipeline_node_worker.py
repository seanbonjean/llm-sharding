from __future__ import annotations

import gc
import io
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque

import torch
import zmq
from transformers import AutoTokenizer, LlamaConfig
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from utils.forwarding_utils import build_position_ids
from utils.shard_loader import LlamaShardPart


class PipelineProtocol:
    """
    Pipeline 推理消息协议

    所有跨节点消息都保留 owner_addr 和 first_node_addr，原因是：
    - owner_addr 用于末节点把 next token 点对点发回接收用户请求的节点
    - first_node_addr 用于 owner 把下一轮 decode 输入直接送回模型链首节点
    """

    TYPE_KEY = "type"

    PIPELINE_INPUT = "pipeline_input"
    PIPELINE_STATE = "pipeline_state"
    PIPELINE_TOKEN = "pipeline_token"
    PIPELINE_DONE = "pipeline_done"
    PIPELINE_CLEAR = "pipeline_clear"
    USER_REQUEST = "user_request"
    USER_REQUEST_ACK = "user_request_ack"
    KV_CACHE_QUERY = "kv_cache_query"
    KV_CACHE_REPORT = "kv_cache_report"
    PIPELINE_DONE_REPORT = "pipeline_done_report"

    TELEMETRY_FIELDS = (
        "client_request_id",
        "telemetry_addr",
        "trace_kv_cache",
        "trace_label",
    )

    PHASE_PREFILL = "prefill"
    PHASE_DECODE = "decode"
    PHASE_DONE = "done"
    PHASE_CLEAR = "clear"

    @classmethod
    def build_user_request(
        cls,
        prompt: str,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        """
        构造外部客户端发给 owner 节点的用户请求消息。

        这不是模型链内部的 hidden state 消息，而是中控/CLI 用来触发某个节点
        调用 receive_request() 的请求入口。节点收到后会在本地完成
        tokenizer/embedding，再生成正常的 pipeline_input 发往 first node。
        """

        return {
            cls.TYPE_KEY: cls.USER_REQUEST,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        }

    @classmethod
    def copy_telemetry_fields(
        cls, source_message: dict[str, Any], target_message: dict[str, Any]
    ) -> dict[str, Any]:
        """
        复制测试/观测用的可选字段。

        这些字段只在 pipeline_test.py 发起 KV cache 实验时存在；普通推理消息
        不带这些字段，因此不会产生额外回传，也不会改变原有 pipeline 行为。
        """

        for field in cls.TELEMETRY_FIELDS:
            if field in source_message:
                target_message[field] = source_message[field]
        return target_message

    @classmethod
    def base_message(
        cls,
        message_type: str,
        request_id: str,
        phase: str,
        step: int,
        owner_addr: str,
        first_node_addr: str,
    ) -> dict[str, Any]:
        """
        构造所有 pipeline 消息共享的字段

        phase (prefill / decode / done / clear) 表示该消息对应的请求阶段：
        - pipeline_input / pipeline_state / pipeline_token 使用 prefill 或 decode
          prefill 表示第一次把完整 prompt 送入模型链；decode 表示后续每次
          只把上一个 next token 的 embedding 送回模型链
        - pipeline_done 使用 done，pipeline_clear 使用 clear。它们不是模型
          前向阶段，只是复用 phase 字段方便日志和统一协议处理

        step 表示该请求当前推进到第几次模型链输入：prefill 为 0，
        第一个 decode token 输入为 1，之后依次递增。它主要用于日志和
        调试乱序问题，不用于路由
        """

        return {
            cls.TYPE_KEY: message_type,
            "request_id": request_id,
            "phase": phase,
            "step": step,
            "owner_addr": owner_addr,
            "first_node_addr": first_node_addr,
        }

    @classmethod
    def build_input(
        cls,
        request_id: str,
        phase: str,
        step: int,
        owner_addr: str,
        first_node_addr: str,
        hidden_states: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> dict[str, Any]:
        """
        构造 owner 发往首节点的输入消息

        prefill 阶段 hidden_states 的 seq_len 是 prompt 长度；decode 阶段
        seq_len 固定为 1。首节点收到后会根据 request_id 找到该请求自己的
        KV cache，并据此生成正确的 position_ids
        """

        message = cls.base_message(
            cls.PIPELINE_INPUT,
            request_id,
            phase,
            step,
            owner_addr,
            first_node_addr,
        )
        message.update(
            {
                "hidden_states": hidden_states,
                "batch_size": batch_size,
                "seq_len": seq_len,
            }
        )
        return message

    @classmethod
    def build_state(
        cls,
        source_message: dict[str, Any],
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> dict[str, Any]:
        """
        构造分片之间传递的 hidden state 消息

        cos/sin 由首分片按该请求自己的 KV cache 长度计算，后续分片复用同一
        组 RoPE 张量，保持与原始单请求链路的实现方式一致
        """

        message = cls.base_message(
            cls.PIPELINE_STATE,
            source_message["request_id"],
            source_message["phase"],
            source_message["step"],
            source_message["owner_addr"],
            source_message["first_node_addr"],
        )
        message.update(
            {
                "hidden_states": hidden_states,
                "cos": cos,
                "sin": sin,
            }
        )
        return cls.copy_telemetry_fields(source_message, message)

    @classmethod
    def build_token(
        cls,
        source_message: dict[str, Any],
        next_token_id: torch.Tensor,
    ) -> dict[str, Any]:
        """
        构造末节点发往 owner 的 token 消息

        这条消息不沿模型链空转一轮，而是由末节点直接 send_to(owner_addr)
        owner 收到后负责解码、判断结束，以及生成下一轮 decode 输入
        """

        message = cls.base_message(
            cls.PIPELINE_TOKEN,
            source_message["request_id"],
            source_message["phase"],
            source_message["step"],
            source_message["owner_addr"],
            source_message["first_node_addr"],
        )
        message["next_token_id"] = next_token_id
        return cls.copy_telemetry_fields(source_message, message)

    @classmethod
    def build_done(
        cls,
        source_message: dict[str, Any],
        reason: str,
        output_token_count: int,
    ) -> dict[str, Any]:
        """
        构造 owner 发往首节点的完成消息 (若仍未完成，发送的是 pipeline_input，并且 phase=decode)

        首节点收到后释放 active slot，并发起 pipeline_clear，让链上每个
        分片只清理该 request_id 对应的 KV cache
        """

        message = cls.base_message(
            cls.PIPELINE_DONE,
            source_message["request_id"],
            cls.PHASE_DONE,
            source_message["step"],
            source_message["owner_addr"],
            source_message["first_node_addr"],
        )
        message.update(
            {
                "reason": reason,
                "output_token_count": output_token_count,
            }
        )
        return cls.copy_telemetry_fields(source_message, message)

    @classmethod
    def build_clear(
        cls,
        source_message: dict[str, Any],
        clear_origin_addr: str,
    ) -> dict[str, Any]:
        """
        构造首节点沿模型链广播的单请求清理消息

        clear_origin_addr 是首节点自己的可连接地址，消息绕链一圈回到该地址
        后停止转发，防止 clear 命令无限循环
        """

        message = cls.base_message(
            cls.PIPELINE_CLEAR,
            source_message["request_id"],
            cls.PHASE_CLEAR,
            source_message.get("step", 0),
            source_message["owner_addr"],
            source_message["first_node_addr"],
        )
        message["clear_origin_addr"] = clear_origin_addr
        return cls.copy_telemetry_fields(source_message, message)

    @classmethod
    def is_type(cls, data: object, message_type: str) -> bool:
        """判断一个对象是否为指定 pipeline 消息类型"""

        return isinstance(data, dict) and data.get(cls.TYPE_KEY) == message_type

    @classmethod
    def is_pipeline_message(cls, data: object) -> bool:
        """判断一个对象是否为 pipeline 协议消息"""

        return isinstance(data, dict) and data.get(cls.TYPE_KEY) in {
            cls.PIPELINE_INPUT,
            cls.PIPELINE_STATE,
            cls.PIPELINE_TOKEN,
            cls.PIPELINE_DONE,
            cls.PIPELINE_CLEAR,
            cls.USER_REQUEST,
            cls.KV_CACHE_QUERY,
        }


class PipelineCommunicator:
    """
    Pipeline 通信器

    和旧 Communicator 不同，这里不再把数据写到固定 send_data.pt/recv_data.pt，
    而是用 BytesIO 在内存里序列化，避免并发消息互相覆盖临时文件
    """

    def __init__(self, src_addr: str, dst_addr: str):
        """
        :param src_addr: 本节点 PULL socket 绑定地址，例如 tcp://*:40800
        :param dst_addr: 模型链下一个节点的可连接地址，用于普通 state 转发
        """

        self.src_addr = src_addr
        self.dst_addr = dst_addr
        self.context = zmq.Context.instance()

        self.recv_socket = self.context.socket(zmq.PULL)
        self.recv_socket.bind(self.src_addr)
        self.actual_src_addr = self.recv_socket.getsockopt_string(zmq.LAST_ENDPOINT)

        self._send_sockets: dict[str, zmq.Socket] = {}
        if self.dst_addr:
            self._get_send_socket(self.dst_addr)

    @staticmethod
    def _serialize(data: Any) -> bytes:
        buffer = io.BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def _deserialize(payload: bytes) -> Any:
        buffer = io.BytesIO(payload)
        try:
            return torch.load(buffer, weights_only=False)
        except TypeError:
            buffer.seek(0)
            return torch.load(buffer)

    def _get_send_socket(self, addr: str) -> zmq.Socket:
        """
        返回 communicator 中相应的发送 socket；如果没有对应 socket，就创建一个新的连接
        """
        if addr not in self._send_sockets:
            send_socket = self.context.socket(zmq.PUSH)
            send_socket.connect(addr)
            self._send_sockets[addr] = send_socket
        return self._send_sockets[addr]

    def change_src_addr(self, new_src_addr: str) -> str:
        """重新绑定本节点接收地址。通常只在收到新 config 时调用"""

        self.recv_socket.unbind(self.actual_src_addr)
        self.recv_socket.bind(new_src_addr)
        self.src_addr = new_src_addr
        self.actual_src_addr = self.recv_socket.getsockopt_string(zmq.LAST_ENDPOINT)
        return self.src_addr

    def change_dst_addr(self, new_dst_addr: str) -> str:
        """更新模型链下一个节点地址；旧 socket 会保留给可能的直连 owner 回路，这样 send _to 时就可能不用再创建新的 socket"""

        self.dst_addr = new_dst_addr
        if self.dst_addr:
            self._get_send_socket(self.dst_addr)
        return self.dst_addr

    def send_to(self, addr: str, data: Any) -> None:
        """
        点对点发送消息到指定地址

        末节点向 owner 返回 token、owner 向 first node 返回 decode 输入，都走这个方法；
        平常的 hidden state 流动也复用该函数，但地址固定为 dst_addr
        """

        if not addr:
            raise ValueError("[ERROR] send_to requires a non-empty target address.")
        self._get_send_socket(addr).send(self._serialize(data))

    def transfer_data(self, data: Any) -> None:
        """把消息发送给 config 中配置的模型链下一个节点"""

        self.send_to(self.dst_addr, data)

    def receive_data(self, no_block: bool = False) -> Any:
        """
        接收一个 pipeline 消息

        no_block=True 时，如果没有消息会抛出 zmq.Again，调用方负责决定是否
        继续处理本地队列或检查新 config
        """

        flags = zmq.NOBLOCK if no_block else 0
        payload = self.recv_socket.recv(flags=flags)
        return self._deserialize(payload)


@dataclass
class PipelineSession:
    """
    单个 request_id 在某个节点上的运行态

    同一个节点可能同时是 shard 节点和 owner 节点：
    - shard session 负责保存该请求在本分片上的 DynamicCache
    - owner session 负责保存 tokenizer 输出、generated_ids 和结束条件

    两类状态放在同一个对象里，是为了处理“owner 恰好也是首节点/中间节点”
    的情况，避免同一个 request_id 在本节点拆成两份状态
    """

    request_id: str
    past_key_value: DynamicCache | None = None
    batch_size: int = 0
    step: int = 0

    generated_ids: list[torch.Tensor] = field(default_factory=list)
    input_token_length: int | None = None
    max_new_tokens: int = 1024
    finished: bool = False


class PipelineNodeWorker:
    """
    Pipeline 版本的节点 worker

    该类只负责模型资源、per-request session 和单条消息的计算/构造；
    是否准入、是否排队、何时转发由 PipelineNodeController 负责
    """

    def __init__(
        self,
        src_addr: str,
        dst_addr: str,
        node_addr: str,
        first_node_addr: str,
        node_id: str,
        can_receive_user_request: bool,
        shards_path: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        """
        :param src_addr: 本节点 PULL 绑定地址。
        :param dst_addr: 模型链下一个分片节点地址。
        :param node_addr: 本节点可被其他节点连接的地址，用作 owner_addr。
        :param first_node_addr: 模型链首分片地址。
        :param node_id: 只用于 request_id 和日志，不参与路由。
        :param can_receive_user_request: 本节点是否加载 tokenizer/embedding。
        :param shards_path: 本地分片权重目录。
        """

        self.communicator = PipelineCommunicator(src_addr=src_addr, dst_addr=dst_addr)
        self.node_addr = node_addr
        self.first_node_addr = first_node_addr
        self.node_id = node_id
        self.can_receive_user_request = can_receive_user_request
        self.shards_path = shards_path
        self.device = torch.device(device)
        self.dtype = dtype

        self.config = LlamaConfig.from_pretrained(self.shards_path)
        self.layer_num = self.config.num_hidden_layers

        self.tokenizer = None
        self.embed_tokens = None
        self.rope = None
        self.shard = None
        self.lm_head = None

        self.start = 0
        self.end = 0
        self._request_seq = 0
        self.sessions: dict[str, PipelineSession] = {}

        if self.can_receive_user_request:
            self._load_embedding()

    @property
    def is_first_stage(self) -> bool:
        """当前节点是否承载模型链首分片"""

        return self.start == 0

    @property
    def is_last_stage(self) -> bool:
        """当前节点是否承载模型链末分片并负责产生 token。"""

        return self.end == self.layer_num

    def _load_embedding(self) -> None:
        print("[LOADER] loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.shards_path)
        print("[LOADER] tokenizer loaded.")

        print("[LOADER] loading embedding layer...")
        self.embed_tokens = torch.nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
        ).to(self.device, dtype=self.dtype)
        self.embed_tokens.load_state_dict(
            torch.load(
                os.path.join(self.shards_path, "embedding.pth"),
                map_location=self.device,
            )
        )
        print("[LOADER] embedding layer loaded.")

    def load_shards(self, start: int, end: int) -> None:
        """
        加载当前节点负责的连续分片。

        pipeline 版本不创建全局 KV cache；KV cache 会在每个 request_id 第一次
        到达本节点时创建，并在该请求结束后的 pipeline_clear 中单独释放。
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
        self.sessions.clear()
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        if self.is_first_stage:
            print("[LOADER] loading RoPE...")
            self.rope = LlamaRotaryEmbedding(config=self.config, device=self.device).to(
                self.device
            )
            print("[LOADER] RoPE loaded.")

        if self.is_last_stage:
            add_final_norm = True
            final_norm_weight = "final_norm.pth"
            print("[LOADER] loading lm_head...")
            self.lm_head = torch.nn.Linear(
                self.config.hidden_size,
                self.config.vocab_size,
                bias=False,
            ).to(self.device, dtype=self.dtype)
            self.lm_head.load_state_dict(
                torch.load(
                    os.path.join(self.shards_path, "lm_head.pth"),
                    map_location=self.device,
                )
            )
            print("[LOADER] lm_head loaded.")
        else:
            add_final_norm = False
            final_norm_weight = None

        print(f"[LOADER] loading hidden layer {start}~{end}(end excluded)...")
        self.shard = LlamaShardPart(
            self.shards_path,
            ["block_" + str(i) + ".pth" for i in range(start, end)],
            start,
            end,
            device=self.device,
            dtype=self.dtype,
            add_final_norm=add_final_norm,
            final_norm_weight=final_norm_weight,
        )
        self.shard.eval()
        print(f"[LOADER] hidden layer {start}~{end}(end excluded) loaded.")

    def _new_request_id(self) -> str:
        self._request_seq += 1
        now_ms = int(time.time() * 1000)
        return f"{self.node_id}-{now_ms}-{self._request_seq}"

    def _get_or_create_session(self, request_id: str) -> PipelineSession:
        session = self.sessions.get(request_id)
        if session is None:
            session = PipelineSession(request_id=request_id)
            self.sessions[request_id] = session
        if session.past_key_value is None:
            session.past_key_value = DynamicCache()
        return session

    @staticmethod
    def _iter_tensors(data: object):
        """
        递归遍历 DynamicCache 内部结构中的 tensor。

        HuggingFace 不同版本的 cache 容器结构可能略有差异，因此这里不假设
        key_cache/value_cache 一定是简单 list，而是递归展开 dict/list/tuple。
        """

        if torch.is_tensor(data):
            yield data
        elif isinstance(data, dict):
            for value in data.values():
                yield from PipelineNodeWorker._iter_tensors(value)
        elif isinstance(data, (list, tuple)):
            for value in data:
                yield from PipelineNodeWorker._iter_tensors(value)

    @classmethod
    def _dynamic_cache_payload_stats(
        cls, past_key_value: DynamicCache | None
    ) -> dict[str, int]:
        """
        统计 DynamicCache 中 key/value tensor 的逻辑 payload 大小。

        bytes/numel 只来自 KV cache tensor 本体，不包含 Python 容器开销；
        token_length 取 cache tensor 中能观察到的最大 seq_len，用于辅助检查
        prefill/decode 后 cache 是否按预期增长。
        """

        if past_key_value is None:
            return {
                "kv_cache_bytes": 0,
                "kv_cache_numel": 0,
                "kv_cache_tensor_count": 0,
                "kv_cache_token_length": 0,
            }

        total_bytes = 0
        total_numel = 0
        tensor_count = 0
        token_length = 0

        def visit(cache_storage: object) -> None:
            nonlocal total_bytes, total_numel, tensor_count, token_length
            for tensor in cls._iter_tensors(cache_storage):
                tensor_count += 1
                total_numel += tensor.numel()
                total_bytes += tensor.numel() * tensor.element_size()
                if tensor.ndim >= 2:
                    token_length = max(token_length, int(tensor.shape[-2]))

        for cache_attr in ("key_cache", "value_cache"):
            visit(getattr(past_key_value, cache_attr, None))

        if tensor_count == 0:
            layers = getattr(past_key_value, "layers", None)
            if layers is not None:
                for layer_cache in layers:
                    for cache_attr in (
                        "key_cache",
                        "value_cache",
                        "keys",
                        "values",
                        "key_states",
                        "value_states",
                    ):
                        visit(getattr(layer_cache, cache_attr, None))

        return {
            "kv_cache_bytes": int(total_bytes),
            "kv_cache_numel": int(total_numel),
            "kv_cache_tensor_count": int(tensor_count),
            "kv_cache_token_length": int(token_length),
        }

    def _cuda_memory_allocated_bytes(self) -> int | None:
        """返回当前 CUDA memory_allocated；非 CUDA 设备返回 None。"""

        if self.device.type != "cuda":
            return None
        torch.cuda.synchronize(self.device)
        return int(torch.cuda.memory_allocated(self.device))

    def build_kv_cache_report(
        self,
        request_id: str,
        source_message: dict[str, Any] | None = None,
        event: str = "query",
    ) -> dict[str, Any]:
        """
        构造当前节点某个 request_id 的 KV cache 测量结果。

        report 会发回 pipeline_test.py 绑定的 telemetry socket；这里不修改
        session，不触发 clear，只做只读统计。
        """

        session = self.sessions.get(request_id)
        cache_stats = self._dynamic_cache_payload_stats(
            session.past_key_value if session is not None else None
        )
        report: dict[str, Any] = {
            PipelineProtocol.TYPE_KEY: PipelineProtocol.KV_CACHE_REPORT,
            "event": event,
            "request_id": request_id,
            "node_id": self.node_id,
            "node_addr": self.node_addr,
            "shards_start": self.start,
            "shards_end": self.end,
            "has_session": session is not None,
            "session_step": session.step if session is not None else None,
            "timestamp": time.time(),
            "cuda_memory_allocated_bytes": self._cuda_memory_allocated_bytes(),
        }
        report.update(cache_stats)
        if source_message is not None:
            report.update(
                {
                    "phase": source_message.get("phase"),
                    "step": source_message.get("step"),
                    "forward_seq_len": source_message.get("seq_len"),
                    "client_request_id": source_message.get("client_request_id"),
                    "trace_label": source_message.get("trace_label"),
                }
            )
        else:
            report.update(
                {
                    "phase": None,
                    "step": session.step if session is not None else None,
                    "forward_seq_len": None,
                    "client_request_id": None,
                    "trace_label": None,
                }
            )
        return report

    def maybe_emit_kv_cache_report(
        self,
        request_id: str,
        source_message: dict[str, Any],
        event: str = "post_shard_forward",
    ) -> None:
        """
        如果消息带 telemetry_addr，则把当前节点 KV cache snapshot 发回测试脚本。

        普通推理消息没有 telemetry_addr，所以这里是零侵入的观测钩子。
        """

        telemetry_addr = source_message.get("telemetry_addr")
        if not telemetry_addr or not bool(source_message.get("trace_kv_cache", False)):
            return
        report = self.build_kv_cache_report(
            request_id=request_id,
            source_message=source_message,
            event=event,
        )
        self.communicator.send_to(telemetry_addr, report)

    @torch.inference_mode()
    def receive_user_request(
        self,
        request: str = "Write a poem about the blue sky.",
        max_new_tokens: int = 1024,
        input_ids: torch.Tensor | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        接收用户请求并构造发往 first node 的 pipeline_input。

        只有 can_receive_user_request=True 的 owner 节点会调用本方法。它负责
        tokenizer/embedding，并创建 owner session；真正的 shard KV cache 会在
        pipeline_input 进入各分片时按 request_id 懒创建。
        """

        if not self.can_receive_user_request:
            raise RuntimeError("[ERROR] this node cannot receive user request.")

        if input_ids is None:
            print("[REQUEST] input: " + request)
            inputs = self.tokenizer(request, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]
        else:
            if input_ids.ndim != 2:
                raise ValueError(
                    "[ERROR] input_ids must be a 2D tensor with shape [batch_size, seq_len]."
                )
            input_ids = input_ids.to(device=self.device, dtype=torch.long)
            print("[REQUEST] input: inputted from direct token ids.")

        if input_ids.shape[0] != 1:
            raise ValueError(
                "[ERROR] pipeline v1 expects batch_size=1 for each user request."
            )

        request_id = self._new_request_id()
        session = PipelineSession(
            request_id=request_id,
            generated_ids=[input_ids],
            input_token_length=input_ids.shape[1],
            max_new_tokens=max_new_tokens,
        )
        self.sessions[request_id] = session

        print(
            f"[REQUEST] request_id={request_id} input token number: {session.input_token_length}"
        )
        hidden_states = self.embed_tokens(input_ids)
        batch_size, seq_len, _ = hidden_states.shape

        message = PipelineProtocol.build_input(
            request_id=request_id,
            phase=PipelineProtocol.PHASE_PREFILL,
            step=0,
            owner_addr=self.node_addr,
            first_node_addr=self.first_node_addr,
            hidden_states=hidden_states,
            batch_size=batch_size,
            seq_len=seq_len,
        )
        if request_metadata:
            PipelineProtocol.copy_telemetry_fields(request_metadata, message)
        return message

    @torch.inference_mode()
    def pass_through_shard(self, message: dict[str, Any]) -> dict[str, Any]:
        """
        处理一个 pipeline_input 或 pipeline_state，并返回下一跳消息。

        - 首分片只接收 pipeline_input，按 request_id 的 KV cache 计算 position_ids。
        - 中间/末分片接收 pipeline_state，复用首分片传来的 cos/sin。
        - 末分片返回 pipeline_token，由 controller 直接发给 owner。
        """

        if PipelineProtocol.is_type(message, PipelineProtocol.PIPELINE_INPUT):
            if not self.is_first_stage:
                raise RuntimeError(
                    "[ERROR] pipeline_input should only be processed by the first stage."
                )
            hidden_states = message["hidden_states"].to(
                device=self.device, dtype=self.dtype
            )
            session = self._get_or_create_session(message["request_id"])
            session.batch_size = int(message["batch_size"])
            session.step = int(message["step"])

            position_ids = build_position_ids(
                session.past_key_value,
                int(message["seq_len"]),
                device=self.device,
                batch_size=session.batch_size,
            )
            cos, sin = self.rope(hidden_states, position_ids)
        elif PipelineProtocol.is_type(message, PipelineProtocol.PIPELINE_STATE):
            if self.is_first_stage:
                raise RuntimeError(
                    "[ERROR] first stage should not receive pipeline_state."
                )
            hidden_states = message["hidden_states"].to(
                device=self.device, dtype=self.dtype
            )
            cos = message["cos"].to(device=self.device, dtype=self.dtype)
            sin = message["sin"].to(device=self.device, dtype=self.dtype)
            session = self._get_or_create_session(message["request_id"])
            session.step = int(message["step"])
        else:
            raise RuntimeError(
                f"[ERROR] unsupported message for shard forward: {message.get('type')}"
            )

        next_hidden_states = self.shard(
            hidden_states,
            past_key_value=session.past_key_value,
            rotary_emb=(cos, sin),
        )
        self.maybe_emit_kv_cache_report(message["request_id"], message)

        if self.is_last_stage:
            logits = self.lm_head(next_hidden_states)
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return PipelineProtocol.build_token(message, next_token_id)

        return PipelineProtocol.build_state(message, next_hidden_states, cos, sin)

    @torch.inference_mode()
    def receive_next_token(self, message: dict[str, Any]) -> dict[str, Any]:
        """
        owner 节点处理末分片返回的 pipeline_token。

        若请求结束，返回 pipeline_done；否则把 token embedding 成下一轮
        pipeline_input，并由 controller 发回 first_node_addr。
        """

        request_id = message["request_id"]
        session = self.sessions.get(request_id)
        if session is None or not session.generated_ids:
            raise RuntimeError(
                f"[ERROR] owner session not found for request_id={request_id}."
            )
        if self.embed_tokens is None or self.tokenizer is None:
            raise RuntimeError(
                "[ERROR] owner node must have tokenizer and embedding layer."
            )

        next_token_id = message["next_token_id"].to(
            device=self.device, dtype=torch.long
        )
        session.generated_ids.append(next_token_id.unsqueeze(-1))

        token_id = int(next_token_id.item())
        next_token = self.tokenizer.decode(token_id)
        print(f"[{request_id}] {next_token!r}", end=" ", flush=True)

        output_token_count = len(session.generated_ids) - 1
        reached_eos = self.tokenizer.eos_token_id is not None and token_id == int(
            self.tokenizer.eos_token_id
        )
        reached_max_new_tokens = output_token_count >= session.max_new_tokens

        if reached_eos or reached_max_new_tokens:
            session.finished = True
            reason = "eos" if reached_eos else "max_new_tokens"
            print()
            final_ids = torch.cat(session.generated_ids, dim=-1)
            print(
                f"[REQUEST] request_id={request_id} output: {self.tokenizer.decode(final_ids[0])}"
            )
            print(
                f"[REQUEST] request_id={request_id} output token number: {output_token_count}"
            )
            return PipelineProtocol.build_done(message, reason, output_token_count)

        session.step = output_token_count
        hidden_states = self.embed_tokens(next_token_id.unsqueeze(0))
        next_input = PipelineProtocol.build_input(
            request_id=request_id,
            phase=PipelineProtocol.PHASE_DECODE,
            step=session.step,
            owner_addr=message["owner_addr"],
            first_node_addr=message["first_node_addr"],
            hidden_states=hidden_states,
            batch_size=1,
            seq_len=1,
        )
        return PipelineProtocol.copy_telemetry_fields(message, next_input)

    def clear_request_state(self, request_id: str) -> None:
        """
        只清理一个 request_id 的本地状态。

        这是 pipeline 并发的关键点：一个用户结束后不能调用全局 clear，否则会
        把其他仍在模型链中的请求 KV cache 一并删除。
        """

        session = self.sessions.pop(request_id, None)
        if session is None:
            return

        old_past_key_value = session.past_key_value
        session.generated_ids.clear()
        session.past_key_value = None

        if old_past_key_value is not None:
            reset_cache = getattr(old_past_key_value, "reset", None)
            if callable(reset_cache):
                try:
                    reset_cache()
                except NotImplementedError:
                    pass
            for cache_attr in ("key_cache", "value_cache"):
                cache_storage = getattr(old_past_key_value, cache_attr, None)
                if hasattr(cache_storage, "clear"):
                    cache_storage.clear()
            del old_past_key_value

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[PIPELINE] request_id={request_id} local session cleared.")


class PipelineNodeController:
    """
    Pipeline 节点控制器

    首节点额外承担 admission scheduler：限制 active request 数量，并保证
    prefill 不会越过已经 active 的 decode。非首节点只按上游消息 FIFO 处理
    """

    def __init__(
        self,
        shards_path: str,
        device: str,
        dtype: torch.dtype,
        listen_port: int = 40700,
    ):
        """
        :param shards_path: 分片权重目录。
        :param device: 运行设备，例如 cuda:0。
        :param dtype: 模型权重 dtype。
        :param listen_port: 接收中控 config 的端口。
        """

        self.shards_path = shards_path
        self.device = device
        self.dtype = dtype

        self.listen_addr = "tcp://*:" + str(listen_port)
        self.config_context = zmq.Context.instance()
        self.recv_config_socket = self.config_context.socket(zmq.PULL)
        self.recv_config_socket.bind(self.listen_addr)

        self.received_config = self._receive_config()
        self._normalize_config(self.received_config)
        self.accepting_user_requests = bool(self.received_config["can_receive_user_request"])

        self.node_worker = self._create_worker(self.received_config)
        self.node_worker.load_shards(
            self.received_config["shards_start"],
            self.received_config["shards_end"],
        )

        self.pipeline_depth = int(self.received_config["pipeline_depth"])
        self.max_active_requests = int(self.received_config["max_active_requests"])
        self.default_max_new_tokens = 1024
        self.active_request_ids: set[str] = (
            set()
        )  # 已被首节点准入、正在模型链中推理的 request 的 request_id 集合
        self.pending_prefill_queue: Deque[dict[str, Any]] = (
            deque()
        )  # active request 数量超过 max_active_requests 数量限制后，新进来的 request 在这个队列中等待前序任务处理完
        self.first_stage_input_queue: Deque[dict[str, Any]] = (
            deque()
        )  # 也就是 active queue，存放首节点准入的待执行的 request 的消息流：包含刚已准入的等待 prefill 的任务，和正在 decode 的任务
        self.clear_in_flight_request_ids: set[str] = (
            set()
        )  # 首节点已发出、尚未沿旧模型链回到 origin 的 clear 请求
        self.reconfig_pending = False  # 收到新 config 但仍有旧 pipeline 请求未完成时置 True，期间不再准入新的 prefill
        self.deferred_config: dict[str, Any] | None = (
            None  # 等旧请求 drain 完毕后再真正应用的 config
        )
        print("[CONTROLLER] Pipeline node is ready.")

    @property
    def is_first_stage(self) -> bool:
        """当前 controller 是否运行在模型链首分片节点上"""

        return self.node_worker.is_first_stage

    def _normalize_config(self, config: dict[str, Any]) -> None:
        if not config.get("node_addr"):
            raise ValueError(
                "[ERROR] pipeline config requires node_addr, for example tcp://172.16.0.2:40800."
            )
        if "*" in config["node_addr"]:
            raise ValueError(
                "[ERROR] node_addr must be connectable by other devices; do not use tcp://*:port."
            )

        config.setdefault(
            "node_id", config["node_addr"]
        )  # 如果没有显式提供 node_id，就用 node_addr 作为默认值，保证日志里至少有个可区分的标识
        if config.get("shards_start") == 0:
            config.setdefault("first_node_addr", config["node_addr"])
        elif not config.get("first_node_addr"):
            raise ValueError(
                "[ERROR] first_node_addr is required for non-first pipeline nodes."
            )

        config.setdefault("pipeline_depth", 1)
        config.setdefault("max_active_requests", config["pipeline_depth"])
        if int(config["pipeline_depth"]) <= 0:
            raise ValueError("[ERROR] pipeline_depth must be positive.")
        if int(config["max_active_requests"]) <= 0:
            raise ValueError("[ERROR] max_active_requests must be positive.")

    def _create_worker(
        self, config: dict[str, Any], keep_owner_runtime: bool = False
    ) -> PipelineNodeWorker:
        worker_can_receive_user_request = (
            bool(config["can_receive_user_request"]) or keep_owner_runtime
        )
        return PipelineNodeWorker(
            src_addr=config["src_addr"],
            dst_addr=config["dst_addr"],
            node_addr=config["node_addr"],
            first_node_addr=config["first_node_addr"],
            node_id=str(config["node_id"]),
            can_receive_user_request=worker_can_receive_user_request,
            shards_path=self.shards_path,
            device=self.device,
            dtype=self.dtype,
        )

    def _receive_config(self, no_block: bool = False) -> dict[str, Any] | None:
        if no_block:
            try:
                received_config = self.recv_config_socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                return None
        else:
            print(
                "[CONFIG] Waiting for pipeline configuration file from master node..."
            )
            received_config = self.recv_config_socket.recv_json()

        print("[CONFIG] Received pipeline configuration:")
        for k, v in received_config.items():
            print(f"  - {k}: {v}")
        return received_config

    def check_new_config(self) -> None:
        """
        非阻塞检查新 config。

        如果本节点仍有旧 pipeline 请求状态，先暂存新 config 并进入
        reconfig_pending 状态。首节点在该状态下不会再准入新的 prefill，
        只允许已经 active 的请求继续 decode，直到旧请求 drain 完毕后再切换。
        """

        new_config = self._receive_config(no_block=True)
        if not new_config:
            self._whether_apply_deferred_config()
            return

        self._normalize_config(new_config)
        if self._has_pipeline_work_in_progress(new_config):
            self.deferred_config = new_config
            self.reconfig_pending = True
            print(
                "[CONFIG] Deferred new config until current pipeline requests are drained. "
                "New prefill requests will stay in pending_prefill_queue."
            )
            return

        self.deferred_config = None
        self.reconfig_pending = False
        self._apply_config(new_config)
        self._release_pending_prefill_after_reconfig()

    def _has_pipeline_work_in_progress(self, next_config: dict[str, Any] | None = None) -> bool:
        """
        判断本节点是否仍保有旧 config 下的 pipeline 工作

        pending_prefill_queue 不算“正在推理”，因为这些请求还没进入模型链；
        它们可以在重配置完成后按新 config 放行。

        owner-only session 也不阻止重配置。若新 config 不再允许本节点接收
        新用户请求，_apply_config 会临时保留 tokenizer/embedding 只服务这些
        已接收请求，但 controller 会拒绝新的 receive_request。
        """

        return bool(
            self.active_request_ids
            or self.first_stage_input_queue
            or self.clear_in_flight_request_ids
            or self._has_shard_sessions()
        )

    def _has_shard_sessions(self) -> bool:
        """
        判断本节点是否有已经进入旧分片计算的 session

        owner 刚收到用户请求时会先保存 generated_ids，但 pending prefill 还没有
        进入模型链，past_key_value 仍为 None；这种 owner-only session 不应阻止
        重配置，否则 pending 请求会让 deferred config 永远无法应用
        """

        return any(
            session.past_key_value is not None
            for session in self.node_worker.sessions.values()
        )

    def _preserve_owner_only_sessions_for_reconfig(self) -> dict[str, PipelineSession]:
        """
        保存尚未进入分片计算的 owner session

        这些请求还没有 KV cache，只需要保留 generated_ids/max_new_tokens，等
        新 config 应用后继续等待 pending_prefill_queue 放行
        """

        return {
            request_id: session
            for request_id, session in self.node_worker.sessions.items()
            if session.generated_ids
            and session.past_key_value is None
            and not session.finished
        }

    def _whether_apply_deferred_config(self) -> None:
        """如果已经进入 drain 状态且旧请求清空，则应用暂存的新 config"""

        if not self.reconfig_pending or self.deferred_config is None:
            return
        if self._has_pipeline_work_in_progress(self.deferred_config):
            return

        config = self.deferred_config
        self.deferred_config = None
        self.reconfig_pending = False
        self._apply_config(config)
        self._release_pending_prefill_after_reconfig()

    def _apply_config(self, new_config: dict[str, Any]) -> None:
        """真正切换到新 config；调用前必须保证旧请求已经 drain 完毕"""

        owner_only_sessions = self._preserve_owner_only_sessions_for_reconfig()
        keep_owner_runtime = bool(owner_only_sessions)
        del self.node_worker
        gc.collect()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.received_config = new_config
        self.accepting_user_requests = bool(new_config["can_receive_user_request"])
        self.node_worker = self._create_worker(
            new_config,
            keep_owner_runtime=keep_owner_runtime,
        )
        self.node_worker.load_shards(
            new_config["shards_start"], new_config["shards_end"]
        )
        self.node_worker.sessions.update(owner_only_sessions)
        self.pipeline_depth = int(new_config["pipeline_depth"])
        self.max_active_requests = int(new_config["max_active_requests"])
        if keep_owner_runtime and not self.accepting_user_requests:
            print(
                "[CONFIG] kept tokenizer/embedding for existing owner sessions; "
                "new user requests are still rejected by the controller."
            )
        print("[CONTROLLER] Pipeline node reconfigured.")

    def _release_pending_prefill_after_reconfig(self) -> None:
        """
        重配置完成后放行 pending prefill。

        如果本节点重配置后仍是首节点，就按新的 max_active_requests 准入；
        如果本节点不再是首节点，则把尚未进入模型链的 prefill 转发给新首节点。
        """

        if self.is_first_stage:
            self._admit_pending_prefill()
            return

        while self.pending_prefill_queue:
            message = self.pending_prefill_queue.popleft()
            message["first_node_addr"] = self.node_worker.first_node_addr
            self.node_worker.communicator.send_to(message["first_node_addr"], message)
            print(
                f"[SCHEDULER] forwarded pending request_id={message['request_id']} "
                f"to new first node {message['first_node_addr']}."
            )

    def receive_request(
        self,
        request: str = "Write a poem about the blue sky.",
        max_new_tokens: int | None = None,
        input_ids: torch.Tensor | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        接收一个用户请求并提交到 pipeline

        如果当前节点就是首分片，消息进入本地 scheduler；否则点对点发送到
        first_node_addr。返回 request_id，方便测试脚本记录
        """

        if not self.accepting_user_requests:
            raise RuntimeError(
                "[ERROR] this node is not accepting new user requests under current config."
            )

        message = self.node_worker.receive_user_request(
            request=request,
            max_new_tokens=max_new_tokens or self.default_max_new_tokens,
            input_ids=input_ids,
            request_metadata=request_metadata,
        )
        request_id = message["request_id"]
        self._submit_input_to_first_stage(message)
        return request_id

    def run_worker_loop(self, max_new_tokens: int = 1024) -> None:
        """
        常驻 worker loop

        此处的 max_new_tokens 是 receive_request 的 max_new_tokens 未显式传入时的默认值；
        接收实际请求时，可以在调用 receive_request 时为每个请求设置不同上限
        """

        self.default_max_new_tokens = max_new_tokens

        while True:
            try:
                received_data = self.node_worker.communicator.receive_data(
                    no_block=True
                )
                self._handle_message(received_data)
            except zmq.Again:
                pass

            # 首节点可能已经积累了待执行的 pipeline_input；即使本轮没有新网络消息，
            # 也要继续推进队列，才能填满 pipeline。
            self._process_first_stage_input_once()
            self.check_new_config()

    def _submit_input_to_first_stage(self, message: dict[str, Any]) -> None:
        if self.is_first_stage:
            self._handle_first_stage_input(message)
        else:
            self.node_worker.communicator.send_to(message["first_node_addr"], message)

    def _handle_message(self, data: Any) -> None:
        if not PipelineProtocol.is_pipeline_message(data):
            raise RuntimeError(
                f"[ERROR] received unknown pipeline data type: {type(data)}"
            )

        if PipelineProtocol.is_type(data, PipelineProtocol.PIPELINE_INPUT):
            if self.is_first_stage:
                self._handle_first_stage_input(data)
            else:
                raise RuntimeError("[ERROR] non-first node received pipeline_input.")
        elif PipelineProtocol.is_type(data, PipelineProtocol.PIPELINE_STATE):
            # 非首节点没有维护任务队列，每次读入一个 request 就处理一个，转发后直接进入下一个 while 读入下一个 request
            # 堆积的 request 会在 ZMQ socket 里排队
            processed = self.node_worker.pass_through_shard(data)
            self._route_processed_message(processed)
        elif PipelineProtocol.is_type(data, PipelineProtocol.PIPELINE_TOKEN):
            self._handle_pipeline_token(data)
        elif PipelineProtocol.is_type(data, PipelineProtocol.PIPELINE_DONE):
            self._handle_pipeline_done(data)
        elif PipelineProtocol.is_type(data, PipelineProtocol.PIPELINE_CLEAR):
            self._handle_pipeline_clear(data)
        elif PipelineProtocol.is_type(data, PipelineProtocol.USER_REQUEST):
            self._handle_user_request(data)
        elif PipelineProtocol.is_type(data, PipelineProtocol.KV_CACHE_QUERY):
            self._handle_kv_cache_query(data)

    def _handle_user_request(self, message: dict[str, Any]) -> None:
        """
        处理外部客户端发来的用户请求。

        该入口复用节点的 40800 PULL socket，消息类型为 user_request。为了
        避免一次错误外部请求直接杀掉长驻 worker，这里捕获异常并打印；正常的
        pipeline 内部状态错误仍会在对应分支 fail-fast。
        """

        response_addr = message.get("response_addr") or message.get("telemetry_addr")
        client_request_id = message.get("client_request_id")
        request_metadata = {
            field: message[field]
            for field in PipelineProtocol.TELEMETRY_FIELDS
            if field in message
        }

        raw_input_ids = message.get("input_ids")
        input_ids = None
        if raw_input_ids is not None:
            input_ids = raw_input_ids if torch.is_tensor(raw_input_ids) else torch.as_tensor(raw_input_ids)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)

        try:
            request_id = self.receive_request(
                request=message.get("prompt", "Write a poem about the blue sky."),
                max_new_tokens=int(
                    message.get("max_new_tokens", self.default_max_new_tokens)
                ),
                input_ids=input_ids,
                request_metadata=request_metadata,
            )
        except Exception as exc:
            print(f"[REQUEST ERROR] failed to submit user request: {exc}")
            if response_addr:
                self.node_worker.communicator.send_to(
                    response_addr,
                    {
                        PipelineProtocol.TYPE_KEY: PipelineProtocol.USER_REQUEST_ACK,
                        "ok": False,
                        "client_request_id": client_request_id,
                        "request_id": None,
                        "node_id": self.node_worker.node_id,
                        "node_addr": self.node_worker.node_addr,
                        "error": str(exc),
                        "timestamp": time.time(),
                    },
                )
            return

        print(f"[REQUEST] submitted request_id={request_id}")
        if response_addr:
            session = self.node_worker.sessions.get(request_id)
            self.node_worker.communicator.send_to(
                response_addr,
                {
                    PipelineProtocol.TYPE_KEY: PipelineProtocol.USER_REQUEST_ACK,
                    "ok": True,
                    "client_request_id": client_request_id,
                    "request_id": request_id,
                    "node_id": self.node_worker.node_id,
                    "node_addr": self.node_worker.node_addr,
                    "input_token_length": (
                        session.input_token_length if session is not None else None
                    ),
                    "timestamp": time.time(),
                },
            )

    def _handle_kv_cache_query(self, message: dict[str, Any]) -> None:
        """
        处理测试脚本发来的 KV cache 查询。

        查询不参与模型链调度，也不会清理 session；它只是把本节点当前保存的
        某个 request_id 的 DynamicCache 统计结果回传到 response_addr。
        """

        response_addr = message.get("response_addr") or message.get("telemetry_addr")
        if not response_addr:
            print("[REQUEST ERROR] kv_cache_query missing response_addr.")
            return

        request_id = message.get("request_id")
        if request_id and request_id != "*":
            request_ids = [request_id]
        else:
            request_ids = list(self.node_worker.sessions.keys())
            if not request_ids:
                request_ids = [request_id or "*"]

        for current_request_id in request_ids:
            report = self.node_worker.build_kv_cache_report(
                request_id=current_request_id,
                source_message=None,
                event="query",
            )
            report.update(
                {
                    "client_request_id": message.get("client_request_id"),
                    "trace_label": message.get("trace_label"),
                    "query_id": message.get("query_id"),
                }
            )
            self.node_worker.communicator.send_to(response_addr, report)

    def _handle_first_stage_input(self, message: dict[str, Any]) -> None:
        request_id = message["request_id"]
        phase = message["phase"]

        # 首节点是全链唯一的 admission scheduler。active 已满或正在等待
        # 重配置时，新 prefill 只能在 pending_prefill_queue 等待；已经 active
        # 的请求返回 decode 时，直接进入 first_stage_input_queue
        if phase == PipelineProtocol.PHASE_PREFILL:
            if request_id in self.active_request_ids:
                raise RuntimeError(
                    f"[ERROR] received duplicate prefill for active request_id={request_id}; "
                    "pipeline scheduler state is inconsistent."
                )
            if self.reconfig_pending:
                self.pending_prefill_queue.append(message)
                print(
                    f"[SCHEDULER] queued request_id={request_id} during reconfig pending: "
                    "requests in the active queue are draining off to clear the way for new config; "
                    "new coming requests are pending until the new config is applied; "
                    f"pending queue={len(self.pending_prefill_queue)}"
                )
            elif len(self.active_request_ids) < self.max_active_requests:
                self.active_request_ids.add(request_id)
                self.first_stage_input_queue.append(message)
                print(
                    f"[SCHEDULER] admitted request_id={request_id}; "
                    f"active queue={len(self.active_request_ids)}/{self.max_active_requests}"
                )
            else:
                self.pending_prefill_queue.append(message)
                print(
                    f"[SCHEDULER] queued request_id={request_id}; "
                    f"pending queue={len(self.pending_prefill_queue)}"
                )
            return

        if phase == PipelineProtocol.PHASE_DECODE:
            if request_id in self.active_request_ids:
                self.first_stage_input_queue.append(message)
            else:
                raise RuntimeError(
                    f"[ERROR] received decode for inactive request_id={request_id}; "
                    "pipeline scheduler state is inconsistent."
                )
            return

        raise RuntimeError(
            f"[ERROR] unsupported first-stage input phase={phase!r} "
            f"for request_id={request_id}."
        )

    def _process_first_stage_input_once(self) -> None:
        if not self.is_first_stage or not self.first_stage_input_queue:
            return
        message = self.first_stage_input_queue.popleft()
        processed = self.node_worker.pass_through_shard(message)
        self._route_processed_message(processed)

    def _route_processed_message(self, message: dict[str, Any]) -> None:
        if PipelineProtocol.is_type(message, PipelineProtocol.PIPELINE_STATE):
            self.node_worker.communicator.transfer_data(message)
        elif PipelineProtocol.is_type(message, PipelineProtocol.PIPELINE_TOKEN):
            owner_addr = message["owner_addr"]
            if owner_addr == self.node_worker.node_addr:
                self._handle_pipeline_token(message)
            else:
                self.node_worker.communicator.send_to(owner_addr, message)
        else:
            raise RuntimeError(
                f"[ERROR] cannot route processed message type={message.get('type')}."
            )

    def _handle_pipeline_token(self, message: dict[str, Any]) -> None:
        # 如果 token 的 owner_addr 不是本节点，说明是发错了，重新转发到相应节点
        if message["owner_addr"] != self.node_worker.node_addr:
            self.node_worker.communicator.send_to(message["owner_addr"], message)
            return

        next_message = self.node_worker.receive_next_token(message)
        if PipelineProtocol.is_type(next_message, PipelineProtocol.PIPELINE_DONE):
            if self.is_first_stage:
                self._handle_pipeline_done(next_message)
            else:
                self.node_worker.communicator.send_to(
                    next_message["first_node_addr"], next_message
                )
        else:
            self._submit_input_to_first_stage(next_message)

    def _handle_pipeline_done(self, message: dict[str, Any]) -> None:
        # 如果 token 的 first_node_addr 不是本节点，说明是发错了，重新转发到相应节点
        if not self.is_first_stage:
            self.node_worker.communicator.send_to(message["first_node_addr"], message)
            return

        request_id = message["request_id"]
        self.active_request_ids.discard(request_id)
        print(
            f"[SCHEDULER] completed request_id={request_id} reason={message.get('reason')}; "
            f"active={len(self.active_request_ids)}/{self.max_active_requests}"
        )
        self._emit_pipeline_done_report(message)

        self._start_pipeline_clear(message)
        if self.reconfig_pending:
            self._whether_apply_deferred_config()
        else:
            self._admit_pending_prefill()

    def _emit_pipeline_done_report(self, message: dict[str, Any]) -> None:
        """在测试模式下通知 pipeline_test.py 某个 request 已经完成。"""

        telemetry_addr = message.get("telemetry_addr")
        if not telemetry_addr:
            return
        self.node_worker.communicator.send_to(
            telemetry_addr,
            {
                PipelineProtocol.TYPE_KEY: PipelineProtocol.PIPELINE_DONE_REPORT,
                "request_id": message["request_id"],
                "client_request_id": message.get("client_request_id"),
                "trace_label": message.get("trace_label"),
                "reason": message.get("reason"),
                "output_token_count": message.get("output_token_count"),
                "node_id": self.node_worker.node_id,
                "node_addr": self.node_worker.node_addr,
                "timestamp": time.time(),
            },
        )

    def _start_pipeline_clear(self, done_message: dict[str, Any]) -> None:
        request_id = done_message["request_id"]
        self.node_worker.clear_request_state(request_id)

        if self.pipeline_depth <= 1:
            return

        clear_message = PipelineProtocol.build_clear(
            done_message,
            clear_origin_addr=self.node_worker.node_addr,
        )
        self.clear_in_flight_request_ids.add(request_id)
        self.node_worker.communicator.transfer_data(clear_message)

    def _handle_pipeline_clear(self, message: dict[str, Any]) -> None:
        request_id = message["request_id"]

        if message.get("clear_origin_addr") == self.node_worker.node_addr:
            # clear 命令已经绕模型链一圈回到首节点。首节点在发起 clear 前已经
            # 清理过本地状态，因此这里停止转发即可。
            self.clear_in_flight_request_ids.discard(request_id)
            print(
                f"[PIPELINE] request_id={request_id} clear command returned to origin."
            )
            self._whether_apply_deferred_config()
            return

        self.node_worker.clear_request_state(request_id)
        self.node_worker.communicator.transfer_data(message)
        self._whether_apply_deferred_config()

    def _admit_pending_prefill(self) -> None:
        if self.reconfig_pending:
            return
        while (
            self.pending_prefill_queue
            and len(self.active_request_ids) < self.max_active_requests
        ):
            message = self.pending_prefill_queue.popleft()
            request_id = message["request_id"]
            self.active_request_ids.add(request_id)
            message["first_node_addr"] = self.node_worker.first_node_addr
            self.first_stage_input_queue.append(message)
            print(
                f"[SCHEDULER] admitted pending request_id={request_id}; "
                f"active={len(self.active_request_ids)}/{self.max_active_requests}; "
                f"pending={len(self.pending_prefill_queue)}"
            )
