"""
Pipeline inference debug CLI.

This script is intended to run on the controller/debug machine. It can:
1. send or re-send pipeline config to all Jetson nodes;
2. submit one or more user requests to a selected owner node;
3. change simple config knobs such as max_active_requests;
4. run KV cache growth experiments and save CSV/plots;
5. run the manual scenarios listed in docs/pipeline_inference_v1.md.

The node processes should already be running with:
    python pipeline_start_node.py --port 40700
"""

from __future__ import annotations

import csv
import io
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import zmq


USER_REQUEST = "user_request"
USER_REQUEST_BATCH = "user_request_batch"
USER_REQUEST_ACK = "user_request_ack"
KV_CACHE_QUERY = "kv_cache_query"
KV_CACHE_REPORT = "kv_cache_report"
SHARD_FORWARD_REPORT = "shard_forward_report"
PIPELINE_DONE_REPORT = "pipeline_done_report"
PIPELINE_NODE_READY = "pipeline_node_ready"
PIPELINE_ADMISSION_REPORT = "pipeline_admission_report"
PIPELINE_TOKEN_REPORT = "pipeline_token_report"

TOKENIZER_PATH = "shards/Llama-3___2-3B-Instruct_float16"
RESULT_ROOT = Path("results") / "pipeline_kv_cache"
DEFAULT_TELEMETRY_HOST = "172.16.0.1"
DEFAULT_TELEMETRY_PORT = 40900
DEFAULT_PREFILL_TOKEN_LENGTHS = [8, 16, 32, 64, 128, 256, 512]
DEFAULT_DECODE_OUTPUT_TOKEN_LENGTHS = [8, 16, 32, 64, 128, 256, 512]
OVERLAP_TIME_SAMPLE_INTERVAL_S = 1.0
CONCURRENCY_SWEEP_NODE_COUNT = 4
CONCURRENCY_SWEEP_REQUEST_COUNT = 12
CONCURRENCY_SWEEP_MAX_ACTIVE_VALUES = [1, 2, 3, 4, 5, 6, 8]
CONCURRENCY_SWEEP_INPUT_TOKEN_LENGTH = 128
CONCURRENCY_SWEEP_OUTPUT_TOKEN_LENGTH = 64
CONCURRENCY_SWEEP_REPEATS = 5
CONCURRENCY_SWEEP_WARMUP_INPUT_TOKEN_LENGTH = 64
CONCURRENCY_SWEEP_WARMUP_OUTPUT_TOKEN_LENGTH = 16
PIPELINE_LATENCY_WARMUP_INPUT_TOKEN_LENGTH = 16
PIPELINE_LATENCY_WARMUP_DECODE_STEPS = 16
PIPELINE_LATENCY_WARMUP_MAX_NEW_TOKENS = (
    PIPELINE_LATENCY_WARMUP_DECODE_STEPS + 1
)
PIPELINE_LATENCY_WARMUP_TEXTS = [
    "Reserved warmup context initializes pipeline kernels and cache paths before timing. ",
    "Independent calibration request prepares each inference stage without reusing test input. ",
    "Warmup-only token sequence exercises decode routing before the measured workload begins. ",
]
PROMPT_FRAGMENT = (
    "Distributed inference splits a language model across multiple edge devices so "
    "that each device processes part of the network while cooperating with the others. "
)
DISTINCT_PROMPT_FRAGMENTS = [
    "Dense routing sends hidden states across pipeline stages while preserving request order. ",
    "Careful scheduling admits active requests first and leaves extra work in a pending queue. ",
    "Memory telemetry records cache tensors, CUDA allocation deltas, and shard timing reports. ",
    "Network links move intermediate activations between devices before the next layer runs. ",
    "Owner nodes keep generated token history and return decode inputs to the first stage. ",
    "Layer shards compute transformer blocks in sequence without changing model semantics. ",
    "Warm-up requests stabilize kernels, caches, and communication sockets before measurement. ",
    "Controller scripts collect acknowledgements, forward reports, and completion timestamps. ",
]


@dataclass
class PipelineNodeSpec:
    node_id: str
    ip: str
    shards_start: int
    shards_end: int
    can_receive_user_request: bool = True
    config_port: int = 40700
    data_port: int = 40800

    @property
    def node_addr(self) -> str:
        return f"tcp://{self.ip}:{self.data_port}"

    @property
    def config_addr(self) -> str:
        return f"tcp://{self.ip}:{self.config_port}"


AVAILABLE_NODES = [
    PipelineNodeSpec("node0", "172.16.0.4", 0, 7),
    PipelineNodeSpec("node1", "172.16.0.5", 7, 14),
    PipelineNodeSpec("node2", "172.16.0.6", 14, 21),
    PipelineNodeSpec("node3", "172.16.0.7", 21, 28),
    PipelineNodeSpec("node4", "172.16.0.2", 23, 28),
    # node5 is a 4 GiB Orin Nano. In every topology that includes it, it stays
    # an intermediate stage and must not load tokenizer/embedding or lm_head.
    PipelineNodeSpec("node5", "172.16.0.3", 23, 28, can_receive_user_request=False),
    PipelineNodeSpec("node6", "172.16.0.8", 23, 28),
]
DEFAULT_NODES = AVAILABLE_NODES[:4]
DEFAULT_LAYER_COUNT = 28

DEFAULT_PROMPTS = [
    "Write a poem about the blue sky.",
    "Explain edge computing in three short paragraphs.",
    "Give me a concise story about a robot learning to garden.",
    "List five practical tips for reducing GPU memory usage during inference.",
    "Describe pipeline parallel inference in simple terms.",
    "Write a short travel diary from a rainy mountain town.",
]


class PipelineDebugClient:
    """Small ZMQ client for sending pipeline configs and user requests."""

    def __init__(self, nodes: list[PipelineNodeSpec]):
        self.nodes = nodes
        self.pipeline_depth = len(nodes)
        self.max_active_requests = self.pipeline_depth
        self.context = zmq.Context.instance()
        self._json_sockets: dict[str, zmq.Socket] = {}
        self._data_sockets: dict[str, zmq.Socket] = {}
        self._request_seq = 0
        self._config_seq = 0
        self.last_config_id: str | None = None
        self.telemetry_public_addr: str | None = None
        self.telemetry_bind_addr: str | None = None
        self._telemetry_socket: zmq.Socket | None = None
        self._telemetry_poller: zmq.Poller | None = None
        self._event_backlog: list[dict] = []

    def _json_socket(self, addr: str) -> zmq.Socket:
        if addr not in self._json_sockets:
            socket = self.context.socket(zmq.PUSH)
            socket.connect(addr)
            self._json_sockets[addr] = socket
        return self._json_sockets[addr]

    def _data_socket(self, addr: str) -> zmq.Socket:
        if addr not in self._data_sockets:
            socket = self.context.socket(zmq.PUSH)
            socket.connect(addr)
            self._data_sockets[addr] = socket
        return self._data_sockets[addr]

    @staticmethod
    def _serialize(data: object) -> bytes:
        buffer = io.BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def _deserialize(payload: bytes) -> object:
        buffer = io.BytesIO(payload)
        try:
            return torch.load(buffer, weights_only=False)
        except TypeError:
            buffer.seek(0)
            return torch.load(buffer)

    def _new_client_request_id(self) -> str:
        self._request_seq += 1
        return f"client-{int(time.time() * 1000)}-{self._request_seq}"

    def _new_config_id(self) -> str:
        """Create an ID that ties node-ready reports to one config broadcast."""

        self._config_seq += 1
        return f"config-{int(time.time() * 1000)}-{self._config_seq}"

    def _build_user_request_payload(
        self,
        prompt: str,
        max_new_tokens: int,
        input_ids: torch.Tensor | None = None,
        trace_kv_cache: bool = False,
        trace_forward_measurement: bool = False,
        telemetry_only: bool = False,
        trace_label: str = "",
        ignore_eos_for_measurement: bool = False,
    ) -> tuple[str, dict]:
        client_request_id = self._new_client_request_id()
        payload = {
            "type": USER_REQUEST,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "client_request_id": client_request_id,
        }
        if ignore_eos_for_measurement:
            payload["ignore_eos_for_measurement"] = True
        if input_ids is not None:
            payload["input_ids"] = input_ids.cpu()
        if trace_kv_cache or trace_forward_measurement or telemetry_only:
            if not self.telemetry_public_addr:
                raise RuntimeError("[ERROR] telemetry must be configured before tracing.")
            payload["telemetry_addr"] = self.telemetry_public_addr
            payload["trace_label"] = trace_label
            if trace_kv_cache:
                payload["trace_kv_cache"] = True
            if trace_forward_measurement:
                payload["trace_forward_measurement"] = True
        return client_request_id, payload

    def build_config(self, index: int, config_id: str | None = None) -> dict:
        """
        Build one node's pipeline config.

        If telemetry is configured, its PULL endpoint also receives optional
        controller-plane node-ready reports. This reuse is test-only plumbing:
        request telemetry remains unchanged and normal configs may omit it.
        """

        node = self.nodes[index]
        next_node = self.nodes[(index + 1) % self.pipeline_depth]
        first_node = self.nodes[0]
        config = {
            "src_addr": f"tcp://*:{node.data_port}",
            "dst_addr": next_node.node_addr,
            "node_addr": node.node_addr,
            "first_node_addr": first_node.node_addr,
            "can_receive_user_request": node.can_receive_user_request,
            "shards_start": node.shards_start,
            "shards_end": node.shards_end,
            "pipeline_depth": self.pipeline_depth,
            "max_active_requests": self.max_active_requests,
            "node_id": node.node_id,
        }
        if config_id is not None:
            config["config_id"] = config_id
        if self.telemetry_public_addr:
            config["controller_addr"] = self.telemetry_public_addr
        return config

    def send_config(self) -> str:
        """
        Send current config to all pipeline nodes and return its config_id.

        Nodes echo this ID in their optional pipeline_node_ready reports, so a
        caller can wait for the exact broadcast instead of relying on a fixed
        layer-load delay.
        """

        config_id = self._new_config_id()
        self.last_config_id = config_id

        for index, node in enumerate(self.nodes):
            config = self.build_config(index, config_id=config_id)
            self._json_socket(node.config_addr).send_json(config)
            print(
                f"[CONFIG] sent {node.node_id} {node.ip}: "
                f"layers {node.shards_start}~{node.shards_end}, "
                f"dst={config['dst_addr']}, can_receive={node.can_receive_user_request}"
            )
        print("[CONFIG] all configs sent; keep this CLI alive while nodes receive them.")
        return config_id

    def configure_telemetry(
        self,
        public_host: str = DEFAULT_TELEMETRY_HOST,
        port: int = DEFAULT_TELEMETRY_PORT,
    ) -> None:
        """
        Bind a local PULL socket for worker telemetry.

        public_host 必须是 Jetson 节点可以连到的控制端 IP；本地 bind 仍使用
        tcp://*:port，方便在不同网卡上接收回传。
        """

        if self._telemetry_socket is not None:
            return

        self.telemetry_public_addr = f"tcp://{public_host}:{port}"
        self.telemetry_bind_addr = f"tcp://*:{port}"
        self._telemetry_socket = self.context.socket(zmq.PULL)
        self._telemetry_socket.bind(self.telemetry_bind_addr)
        self._telemetry_poller = zmq.Poller()
        self._telemetry_poller.register(self._telemetry_socket, zmq.POLLIN)
        print(
            f"[TEST] telemetry listening on {self.telemetry_bind_addr}; "
            f"workers will connect to {self.telemetry_public_addr}"
        )

    def receive_event(self, timeout_s: float = 1.0) -> dict | None:
        if self._event_backlog:
            return self._event_backlog.pop(0)
        if self._telemetry_socket is None or self._telemetry_poller is None:
            raise RuntimeError("[ERROR] telemetry is not configured.")

        events = dict(self._telemetry_poller.poll(int(timeout_s * 1000)))
        if self._telemetry_socket not in events:
            return None
        data = self._deserialize(self._telemetry_socket.recv())
        return data if isinstance(data, dict) else {"type": "unknown", "raw": data}

    def wait_for_event(
        self,
        predicate: Callable[[dict], bool],
        timeout_s: float,
    ) -> dict | None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            for index, event in enumerate(self._event_backlog):
                if predicate(event):
                    return self._event_backlog.pop(index)

            if self._telemetry_socket is None or self._telemetry_poller is None:
                raise RuntimeError("[ERROR] telemetry is not configured.")
            events = dict(
                self._telemetry_poller.poll(
                    int(max(0.05, min(0.5, deadline - time.time())) * 1000)
                )
            )
            if self._telemetry_socket not in events:
                continue
            data = self._deserialize(self._telemetry_socket.recv())
            event = data if isinstance(data, dict) else {"type": "unknown", "raw": data}
            if predicate(event):
                return event
            self._event_backlog.append(event)
        return None

    def drain_events(self) -> None:
        while True:
            event = self.receive_event(timeout_s=0.01)
            if event is None:
                return

    def wait_for_nodes_ready(
        self,
        config_id: str,
        timeout_s: float,
    ) -> list[dict]:
        """
        Wait until every configured node confirms that it loaded config_id.

        This validates layer loading after a topology reconfiguration. Reports
        from another config_id remain in the event backlog for their own caller;
        they cannot satisfy this wait accidentally.
        """

        expected_node_ids = {node.node_id for node in self.nodes}
        reports_by_node: dict[str, dict] = {}
        deadline = time.time() + timeout_s
        while (
            set(reports_by_node) != expected_node_ids
            and time.time() < deadline
        ):
            report = self.wait_for_event(
                lambda event: (
                    event.get("type") == PIPELINE_NODE_READY
                    and event.get("config_id") == config_id
                ),
                timeout_s=max(0.05, deadline - time.time()),
            )
            if report is None:
                break
            node_id = str(report.get("node_id"))
            if node_id in expected_node_ids:
                reports_by_node[node_id] = report

        missing_node_ids = sorted(expected_node_ids - set(reports_by_node))
        if missing_node_ids:
            raise TimeoutError(
                "[ERROR] timed out waiting for node-ready reports for "
                f"config_id={config_id}; missing={missing_node_ids}"
            )

        reports = [reports_by_node[node.node_id] for node in self.nodes]
        print(
            f"[CONTROLLER] all {len(reports)} nodes confirmed ready for "
            f"config_id={config_id}."
        )
        return reports

    def submit_request(
        self,
        owner_index: int,
        prompt: str,
        max_new_tokens: int,
        input_ids: torch.Tensor | None = None,
        trace_kv_cache: bool = False,
        trace_forward_measurement: bool = False,
        telemetry_only: bool = False,
        trace_label: str = "",
    ) -> str:
        """Send one user request to the selected owner node."""

        owner = self.nodes[owner_index]
        client_request_id, payload = self._build_user_request_payload(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            input_ids=input_ids,
            trace_kv_cache=trace_kv_cache,
            trace_forward_measurement=trace_forward_measurement,
            telemetry_only=telemetry_only,
            trace_label=trace_label,
        )
        self._data_socket(owner.node_addr).send(self._serialize(payload))
        print(
            f"[REQUEST] sent to {owner.node_id} ({owner.node_addr}), "
            f"client_request_id={client_request_id}, max_new_tokens={max_new_tokens}"
        )
        return client_request_id

    def submit_burst_requests(
        self,
        owner_index: int,
        prompts: list[str],
        max_new_tokens: int,
        input_ids_list: list[torch.Tensor] | None = None,
        trace_forward_measurement: bool = False,
        telemetry_only: bool = False,
        trace_label_prefix: str = "burst",
        ignore_eos_for_measurement: bool = False,
    ) -> list[str]:
        """Send several requests as one batch so the owner enqueues them in one loop turn."""

        owner = self.nodes[owner_index]
        if input_ids_list is not None and len(input_ids_list) != len(prompts):
            raise ValueError("[ERROR] input_ids_list length must match prompts length.")
        payloads: list[tuple[str, dict]] = []
        for index, prompt in enumerate(prompts, start=1):
            payloads.append(
                self._build_user_request_payload(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    input_ids=(
                        input_ids_list[index - 1]
                        if input_ids_list is not None
                        else None
                    ),
                    trace_forward_measurement=trace_forward_measurement,
                    telemetry_only=telemetry_only,
                    trace_label=f"{trace_label_prefix}_request{index}",
                    ignore_eos_for_measurement=ignore_eos_for_measurement,
                )
            )
        socket = self._data_socket(owner.node_addr)
        batch_payload = {
            "type": USER_REQUEST_BATCH,
            "requests": [payload for _, payload in payloads],
        }
        started = time.perf_counter()
        socket.send(self._serialize(batch_payload))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        client_request_ids = [client_request_id for client_request_id, _ in payloads]
        print(
            f"[REQUEST] batch sent {len(payloads)} requests to {owner.node_id} "
            f"({owner.node_addr}) in {elapsed_ms:.3f} ms; "
            f"client_request_ids={client_request_ids}"
        )
        return client_request_ids

    def wait_for_ack(self, client_request_id: str, timeout_s: float = 15.0) -> dict:
        ack = self.wait_for_event(
            lambda event: (
                event.get("type") == USER_REQUEST_ACK
                and event.get("client_request_id") == client_request_id
            ),
            timeout_s=timeout_s,
        )
        if ack is None:
            raise TimeoutError(f"[ERROR] timed out waiting for ack of {client_request_id}")
        if not ack.get("ok", False):
            raise RuntimeError(f"[ERROR] request rejected: {ack.get('error')}")
        return ack

    def wait_for_done(self, client_request_id: str, timeout_s: float = 60.0) -> dict | None:
        return self.wait_for_event(
            lambda event: (
                event.get("type") == PIPELINE_DONE_REPORT
                and event.get("client_request_id") == client_request_id
            ),
            timeout_s=timeout_s,
        )

    def query_kv_cache(self, request_id: str, trace_label: str = "") -> str:
        if not self.telemetry_public_addr:
            raise RuntimeError("[ERROR] telemetry must be configured before querying KV cache.")
        query_id = f"query-{int(time.time() * 1000)}"
        for node in self.nodes:
            payload = {
                "type": KV_CACHE_QUERY,
                "request_id": request_id,
                "telemetry_addr": self.telemetry_public_addr,
                "query_id": query_id,
                "trace_label": trace_label,
            }
            self._data_socket(node.node_addr).send(self._serialize(payload))
        return query_id

    def show_topology(self) -> None:
        print("\nCurrent pipeline topology:")
        for index, node in enumerate(self.nodes):
            next_node = self.nodes[(index + 1) % self.pipeline_depth]
            print(
                f"  {index}: {node.node_id} {node.ip} "
                f"layers {node.shards_start}~{node.shards_end} "
                f"-> {next_node.node_id} {next_node.ip}; "
                f"can_receive={node.can_receive_user_request}"
            )
        print(f"  first_node_addr={self.nodes[0].node_addr}")
        print(f"  max_active_requests={self.max_active_requests}\n")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{prompt} {suffix}: ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def ask_int_list(prompt: str, default: list[int]) -> list[int]:
    raw_default = ",".join(str(value) for value in default)
    raw = input(f"{prompt} [{raw_default}]: ").strip()
    if not raw:
        return default
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def ask_int(
    prompt: str,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        value = default if not raw else int(raw)
        if minimum is not None and value < minimum:
            print(f"Value must be >= {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Value must be <= {maximum}.")
            continue
        return value


def ask_node(client: PipelineDebugClient, prompt: str = "Choose owner node") -> int:
    client.show_topology()
    return ask_int(prompt, 0, minimum=0) % client.pipeline_depth


def build_even_split_nodes(node_count: int) -> list[PipelineNodeSpec]:
    """
    Build a contiguous, even layer split for a menu-7 topology.

    The regular topologies use the order in AVAILABLE_NODES. For six nodes,
    node5 (172.16.0.3) is deliberately placed before node4: node5 is the 4 GiB
    Orin Nano and therefore remains an intermediate shard without embedding or
    LM-head weights, while node4 remains the final stage. The seven-node order
    naturally keeps node5 intermediate and uses node6 as the final stage.
    """
    if node_count < 1 or node_count > len(AVAILABLE_NODES):
        raise ValueError(f"[ERROR] node_count must be in [1, {len(AVAILABLE_NODES)}].")

    topology_nodes = AVAILABLE_NODES[:node_count]
    if node_count == 6:
        topology_nodes = [
            *AVAILABLE_NODES[:4],
            AVAILABLE_NODES[5],
            AVAILABLE_NODES[4],
        ]

    nodes: list[PipelineNodeSpec] = []
    for index, base_node in enumerate(topology_nodes):
        start = (DEFAULT_LAYER_COUNT * index) // node_count
        end = (DEFAULT_LAYER_COUNT * (index + 1)) // node_count
        nodes.append(
            PipelineNodeSpec(
                node_id=base_node.node_id,
                ip=base_node.ip,
                shards_start=start,
                shards_end=end,
                can_receive_user_request=base_node.can_receive_user_request,
                config_port=base_node.config_port,
                data_port=base_node.data_port,
            )
        )
    return nodes


def configure_even_split_topology(client: PipelineDebugClient, node_count: int) -> str:
    """
    临时把测试 CLI 的拓扑切到前 node_count 台设备，并发送对应 config。

    该 helper 只影响测试脚本管理的 pipeline config；没有修改 worker 主逻辑。
    """

    client.nodes = build_even_split_nodes(node_count)
    client.pipeline_depth = node_count
    client.max_active_requests = node_count
    print(
        f"[TEST] applying {node_count}-node topology with "
        f"max_active_requests={client.max_active_requests}."
    )
    client.show_topology()
    return client.send_config()


def ask_prompt(default: str = DEFAULT_PROMPTS[0]) -> str:
    raw = input(f"Prompt [{default}]: ").strip()
    return raw or default


def ask_telemetry(client: PipelineDebugClient) -> None:
    if client.telemetry_public_addr:
        return
    host = input(
        f"Telemetry callback host/IP visible to Jetson nodes [{DEFAULT_TELEMETRY_HOST}]: "
    ).strip() or DEFAULT_TELEMETRY_HOST
    port = ask_int("Telemetry callback port", DEFAULT_TELEMETRY_PORT, minimum=1)
    client.configure_telemetry(public_host=host, port=port)


def bytes_to_mib(byte_size: int | float | None) -> float:
    return 0.0 if byte_size is None else float(byte_size) / (1024 ** 2)


def make_result_dir() -> Path:
    result_dir = RESULT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        print(f"[TEST] no rows to write for {path}")
        return
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[TEST] csv saved to {path}")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[TEST] json saved to {path}")


def percentile(values: list[float], percent: float) -> float | None:
    usable_values = sorted(value for value in values if value is not None)
    if not usable_values:
        return None
    if len(usable_values) == 1:
        return usable_values[0]
    rank = (len(usable_values) - 1) * percent / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return usable_values[int(rank)]
    lower_value = usable_values[lower]
    upper_value = usable_values[upper]
    return lower_value + (upper_value - lower_value) * (rank - lower)


def mean_or_none(values: list[float]) -> float | None:
    usable_values = [value for value in values if value is not None]
    if not usable_values:
        return None
    return sum(usable_values) / len(usable_values)


class TegrastatsLogger:
    """
    Optional local tegrastats capture for runs launched on a Jetson controller.

    This does not SSH into worker devices. It only records the local machine's
    tegrastats output when the CLI operator explicitly enables it.
    """

    def __init__(self, output_path: Path, interval_ms: int = 1000):
        self.output_path = output_path
        self.interval_ms = interval_ms
        self._file = None
        self._process: subprocess.Popen | None = None

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", encoding="utf-8", errors="replace")
        try:
            self._process = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=self._file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print(f"[TEST] tegrastats logging to {self.output_path}")
        except FileNotFoundError:
            print("[WARNING] tegrastats command not found; skip tegrastats logging.")
            self.stop()

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5.0)
        self._process = None
        if self._file is not None:
            self._file.close()
            self._file = None


def linear_fit(xs: list[float], ys: list[float]) -> dict[str, float]:
    if len(xs) != len(ys):
        raise ValueError("[ERROR] xs and ys must have the same length.")
    if len(xs) < 2:
        return {
            "slope": math.nan,
            "intercept": math.nan,
            "rmse": math.nan,
            "r_squared": math.nan,
        }
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        slope = 0.0
        intercept = mean_y
    else:
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
        intercept = mean_y - slope * mean_x
    fitted = [slope * x + intercept for x in xs]
    rmse = math.sqrt(sum((y - y_hat) ** 2 for y, y_hat in zip(ys, fitted)) / len(ys))
    total = sum((y - mean_y) ** 2 for y in ys)
    residual = sum((y - y_hat) ** 2 for y, y_hat in zip(ys, fitted))
    r_squared = math.nan if total == 0 else 1 - residual / total
    return {
        "slope": slope,
        "intercept": intercept,
        "rmse": rmse,
        "r_squared": r_squared,
    }


def plot_scatter_with_fit(
    rows: list[dict],
    x_key: str,
    y_key: str,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str = "KV cache size (MiB)",
    group_key: str | None = None,
    vertical_line_x: float | None = None,
    scatter_rows: list[dict] | None = None,
    extra_y_series: list[tuple[str, str]] | None = None,
) -> list[dict]:
    if not rows:
        print(f"[TEST] no rows to plot for {output_path}")
        return []

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups: dict[str, list[dict]] = {}
    if group_key is None:
        groups["all"] = rows
    else:
        for row in rows:
            groups.setdefault(str(row.get(group_key, "unknown")), []).append(row)

    scatter_source_rows = rows if scatter_rows is None else scatter_rows
    scatter_groups: dict[str, list[dict]] = {}
    if group_key is None:
        scatter_groups["all"] = scatter_source_rows
    else:
        for row in scatter_source_rows:
            scatter_groups.setdefault(str(row.get(group_key, "unknown")), []).append(row)

    fit_rows = []
    plt.figure(figsize=(8, 5))
    y_series = [(y_key, "KV cache")]
    if extra_y_series:
        y_series.extend(extra_y_series)

    for group_name, group_rows in groups.items():
        sampled_group_rows = scatter_groups.get(group_name, [])
        for series_key, series_label in y_series:
            usable_rows = [
                row for row in group_rows if row.get(series_key) is not None
            ]
            if not usable_rows:
                continue

            xs = [float(row[x_key]) for row in usable_rows]
            ys_mib = [bytes_to_mib(row[series_key]) for row in usable_rows]
            usable_sampled_rows = [
                row for row in sampled_group_rows if row.get(series_key) is not None
            ]
            scatter_xs = [float(row[x_key]) for row in usable_sampled_rows]
            scatter_ys_mib = [
                bytes_to_mib(row[series_key])
                for row in usable_sampled_rows
            ]
            if group_key is None:
                scatter_label = f"{series_label} sampled"
                fit_label = f"{series_label} linear fit"
            else:
                scatter_label = f"{group_name} {series_label} sampled"
                fit_label = f"{group_name} {series_label} fit"
            if scatter_xs:
                plt.scatter(scatter_xs, scatter_ys_mib, label=scatter_label)
            if len(xs) >= 2:
                fit = linear_fit(xs, ys_mib)
                sorted_xs = sorted(xs)
                fitted_ys = [
                    fit["slope"] * x_value + fit["intercept"]
                    for x_value in sorted_xs
                ]
                plt.plot(sorted_xs, fitted_ys, label=fit_label)
                fit_rows.append(
                    {
                        "group": group_name,
                        "metric": series_key,
                        "metric_label": series_label,
                        **fit,
                    }
                )

    if vertical_line_x is not None:
        plt.axvline(vertical_line_x, color="tab:red", linestyle="--", linewidth=1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[TEST] plot saved to {output_path}")
    return fit_rows


CUDA_MEMORY_BASELINE_DELTA_LABEL = "cuda memory baseline delta"

PER_NODE_MEMORY_SERIES = [
    ("cuda_memory_delta_bytes", CUDA_MEMORY_BASELINE_DELTA_LABEL),
]


def safe_filename_part(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_", "~") else "_" for ch in text)


def node_plot_filename(prefix: str, node_id: object, shards_start: object, shards_end: object) -> str:
    return (
        f"{prefix}_{safe_filename_part(node_id)}_"
        f"shards_{safe_filename_part(shards_start)}~{safe_filename_part(shards_end)}.png"
    )


def plot_per_node_scatter_with_fit(
    rows: list[dict],
    output_dir: Path,
    filename_prefix: str,
    x_key: str,
    y_key: str,
    title_prefix: str,
    x_label: str,
    y_label: str = "Memory size (MiB)",
    group_key: str | None = None,
    vertical_line_x: float | None = None,
    scatter_rows: list[dict] | None = None,
    extra_y_series: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """
    为每个 node_id 单独生成一张图，并把 node_id 与 shard 范围写入文件名。

    聚合图适合观察整条 pipeline 的总 KV Cache；per-node 图只画 baseline delta，
    避免 raw torch.cuda.memory_allocated() 把纵轴比例拉得过大。
    """

    rows_by_node: dict[tuple[str, int, int], list[dict]] = {}
    for row in rows:
        if row.get("node_id") is None:
            continue
        key = (
            str(row.get("node_id")),
            int(row.get("shards_start") or 0),
            int(row.get("shards_end") or 0),
        )
        rows_by_node.setdefault(key, []).append(row)

    scatter_by_node: dict[tuple[str, int, int], list[dict]] = {}
    if scatter_rows is not None:
        for row in scatter_rows:
            if row.get("node_id") is None:
                continue
            key = (
                str(row.get("node_id")),
                int(row.get("shards_start") or 0),
                int(row.get("shards_end") or 0),
            )
            scatter_by_node.setdefault(key, []).append(row)

    all_fit_rows: list[dict] = []
    for (node_id, shards_start, shards_end), node_rows in sorted(rows_by_node.items()):
        output_path = output_dir / node_plot_filename(
            filename_prefix,
            node_id,
            shards_start,
            shards_end,
        )
        node_scatter_rows = (
            scatter_by_node.get((node_id, shards_start, shards_end))
            if scatter_rows is not None
            else None
        )
        fit_rows = plot_scatter_with_fit(
            node_rows,
            x_key=x_key,
            y_key=y_key,
            output_path=output_path,
            title=f"{title_prefix} - {node_id} shards {shards_start}~{shards_end}",
            x_label=x_label,
            y_label=y_label,
            group_key=group_key,
            vertical_line_x=vertical_line_x,
            scatter_rows=node_scatter_rows,
            extra_y_series=extra_y_series,
        )
        for fit_row in fit_rows:
            fit_row.update(
                {
                    "node_id": node_id,
                    "shards_start": shards_start,
                    "shards_end": shards_end,
                    "plot_file": output_path.name,
                }
            )
        all_fit_rows.extend(fit_rows)
    return all_fit_rows


def plot_pair_forward_bars(
    rows: list[dict],
    value_key: str,
    output_path: Path,
    title: str,
    y_label: str,
    value_transform: Callable[[float], float] | None = None,
) -> None:
    if not rows:
        print(f"[TEST] no rows to plot for {output_path}")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    node_keys = sorted(
        {
            (
                str(row.get("node_id")),
                int(row.get("shards_start") or 0),
                int(row.get("shards_end") or 0),
            )
            for row in rows
        },
        key=lambda item: item[1],
    )
    request_orders = sorted({int(row.get("request_order") or 0) for row in rows})
    values_by_key = {
        (
            str(row.get("node_id")),
            int(row.get("shards_start") or 0),
            int(row.get("shards_end") or 0),
            int(row.get("request_order") or 0),
        ): row.get(value_key)
        for row in rows
    }

    x_positions = list(range(len(node_keys)))
    width = 0.8 / max(1, len(request_orders))
    plt.figure(figsize=(9, 5))
    for offset_index, request_order in enumerate(request_orders):
        bar_positions = [
            x + (offset_index - (len(request_orders) - 1) / 2) * width
            for x in x_positions
        ]
        bar_values = []
        for node_id, shards_start, shards_end in node_keys:
            raw_value = values_by_key.get(
                (node_id, shards_start, shards_end, request_order)
            )
            if raw_value is None:
                bar_values.append(0.0)
            else:
                value = float(raw_value)
                bar_values.append(value_transform(value) if value_transform else value)
        plt.bar(
            bar_positions,
            bar_values,
            width=width,
            label=f"request {request_order}",
        )

    plt.xticks(
        x_positions,
        [f"{node_id}\n{start}~{end}" for node_id, start, end in node_keys],
    )
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[TEST] plot saved to {output_path}")


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def analyze_forward_critical_path(forward_rows: list[dict]) -> dict:
    """
    Build a DAG from menu-7 forward reports and return the compute critical path.

    Edges model two constraints:
    1. One request must pass through phase/step/stages in order.
    2. One physical node executes its own forward calls serially; this order uses
       only timestamps from that node, so no cross-device clock sync is required.
    """

    if not forward_rows:
        return {
            "inference_time": None,
            "critical_path_rows": [],
            "critical_path_cycle_detected": False,
            "critical_path_node_count": 0,
        }

    indexed_rows = list(enumerate(forward_rows))
    row_by_index = {index: row for index, row in indexed_rows}
    weights = {
        index: _float_or_none(row.get("forward_elapsed_ms")) or 0.0
        for index, row in indexed_rows
    }
    successors: dict[int, set[int]] = {index: set() for index, _ in indexed_rows}
    predecessors: dict[int, set[int]] = {index: set() for index, _ in indexed_rows}

    def add_edge(src: int, dst: int) -> None:
        if src == dst or dst in successors[src]:
            return
        successors[src].add(dst)
        predecessors[dst].add(src)

    def row_phase_order(row: dict) -> int:
        explicit_order = row.get("phase_step_order")
        if explicit_order is not None:
            return int(explicit_order)
        phase = str(row.get("phase") or "")
        step = int(row.get("step") or 0)
        return 1 if phase == "prefill" else step + 1

    def request_sort_key(item: tuple[int, dict]) -> tuple[int, int, int]:
        _, row = item
        return (
            row_phase_order(row),
            int(row.get("shards_start") or 0),
            int(row.get("shards_end") or 0),
        )

    rows_by_request: dict[str, list[tuple[int, dict]]] = {}
    for item in indexed_rows:
        _, row = item
        request_key = str(row.get("client_request_id") or row.get("request_id"))
        rows_by_request.setdefault(request_key, []).append(item)
    for request_rows in rows_by_request.values():
        ordered = sorted(request_rows, key=request_sort_key)
        for (src, _), (dst, _) in zip(ordered, ordered[1:]):
            add_edge(src, dst)

    def node_sort_key(item: tuple[int, dict]) -> tuple[float, float, int, int, int]:
        _, row = item
        started = _float_or_none(row.get("started_timestamp"))
        finished = _float_or_none(row.get("finished_timestamp"))
        return (
            started if started is not None else math.inf,
            finished if finished is not None else math.inf,
            int(row.get("request_order") or 0),
            row_phase_order(row),
            int(row.get("shards_start") or 0),
        )

    rows_by_node: dict[str, list[tuple[int, dict]]] = {}
    for item in indexed_rows:
        _, row = item
        node_key = str(row.get("node_id") or row.get("node_addr") or "unknown")
        rows_by_node.setdefault(node_key, []).append(item)
    for node_rows in rows_by_node.values():
        ordered = sorted(node_rows, key=node_sort_key)
        for (src, _), (dst, _) in zip(ordered, ordered[1:]):
            add_edge(src, dst)

    indegree = {index: len(preds) for index, preds in predecessors.items()}
    ready = sorted(index for index, degree in indegree.items() if degree == 0)
    topo_order: list[int] = []
    while ready:
        current = ready.pop(0)
        topo_order.append(current)
        for successor in sorted(successors[current]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()

    cycle_detected = len(topo_order) != len(indexed_rows)
    if cycle_detected:
        print("[WARNING] critical path graph has a cycle; inference_time is unavailable.")
        return {
            "inference_time": None,
            "critical_path_rows": [],
            "critical_path_cycle_detected": True,
            "critical_path_node_count": 0,
        }

    distance = {index: weights[index] for index, _ in indexed_rows}
    previous: dict[int, int | None] = {index: None for index, _ in indexed_rows}
    for current in topo_order:
        for successor in successors[current]:
            candidate = distance[current] + weights[successor]
            if candidate > distance[successor]:
                distance[successor] = candidate
                previous[successor] = current

    end_index = max(distance, key=lambda index: distance[index])
    critical_path_indices = []
    current_index: int | None = end_index
    while current_index is not None:
        critical_path_indices.append(current_index)
        current_index = previous[current_index]
    critical_path_indices.reverse()

    cumulative = 0.0
    critical_path_rows = []
    for path_index, row_index in enumerate(critical_path_indices, start=1):
        row = row_by_index[row_index]
        elapsed = weights[row_index]
        cumulative += elapsed
        critical_path_rows.append(
            {
                "critical_path_index": path_index,
                "critical_path_cumulative_ms": cumulative,
                "forward_elapsed_ms": elapsed,
                "request_order": row.get("request_order"),
                "client_request_id": row.get("client_request_id"),
                "request_id": row.get("request_id"),
                "node_id": row.get("node_id"),
                "node_addr": row.get("node_addr"),
                "shards_start": row.get("shards_start"),
                "shards_end": row.get("shards_end"),
                "phase": row.get("phase"),
                "step": row.get("step"),
                "phase_step_order": row.get("phase_step_order"),
                "started_timestamp": row.get("started_timestamp"),
                "finished_timestamp": row.get("finished_timestamp"),
            }
        )

    return {
        "inference_time": distance[end_index],
        "critical_path_rows": critical_path_rows,
        "critical_path_cycle_detected": False,
        "critical_path_node_count": len(critical_path_rows),
    }


def plot_latency_breakdown_pie(summary: dict, output_path: Path) -> None:
    inference_time = _float_or_none(summary.get("inference_time"))
    residual_time = _float_or_none(summary.get("communication_and_noncompute_time"))
    total_time = _float_or_none(summary.get("total_complete_time_preferred_ms"))
    if total_time is None:
        total_time = _float_or_none(summary.get("total_complete_time"))
    if inference_time is None or residual_time is None or total_time is None:
        print(f"[TEST] no complete latency breakdown to plot for {output_path}")
        return
    if residual_time < 0:
        print(
            f"[TEST] negative communication/non-forward time; "
            f"skip latency breakdown pie for {output_path}"
        )
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [max(0.0, inference_time), residual_time]
    if sum(values) <= 0:
        print(f"[TEST] invalid latency breakdown values for {output_path}")
        return

    labels = ["critical-path inference", "communication + non-forward"]

    def autopct(pct: float) -> str:
        absolute = pct * sum(values) / 100.0
        return f"{pct:.1f}%\n{absolute:.1f} ms"

    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=labels, autopct=autopct, startangle=90)
    plt.title(f"Latency breakdown, total={total_time:.1f} ms")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[TEST] plot saved to {output_path}")


def plot_request_completion_elapsed_barh(
    done_rows: list[dict],
    output_path: Path,
) -> None:
    """Plot per-request completion elapsed time with an explicit preferred source.

    The preferred value is owner-local batch elapsed time because it is measured
    on the owner node's monotonic clock. The collector-side receive elapsed time
    is kept only as a fallback for older worker versions or incomplete reports.
    """

    usable_rows = [
        row
        for row in done_rows
        if (
            _float_or_none(row.get("owner_local_batch_elapsed_ms")) is not None
            or _float_or_none(row.get("collector_done_received_elapsed_ms")) is not None
        )
    ]
    if not usable_rows:
        print(f"[TEST] no done rows to plot for {output_path}")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered_rows = sorted(usable_rows, key=lambda row: int(row.get("request_order") or 0))
    labels = [f"request {int(row.get('request_order') or 0)}" for row in ordered_rows]
    values: list[float] = []
    sources: list[str] = []
    for row in ordered_rows:
        owner_elapsed = _float_or_none(row.get("owner_local_batch_elapsed_ms"))
        if owner_elapsed is not None:
            values.append(owner_elapsed)
            sources.append("owner_local_batch_elapsed_ms")
            continue
        values.append(
            _float_or_none(row.get("collector_done_received_elapsed_ms")) or 0.0
        )
        sources.append("collector_done_received_elapsed_ms")

    height = max(4.0, 0.45 * len(ordered_rows) + 1.5)
    plt.figure(figsize=(9, height))
    plt.barh(labels, values)
    plt.xlabel("request_completion_elapsed_preferred_ms")
    plt.ylabel("request order")
    plt.title("Request completion elapsed time (preferred source)")
    plt.grid(axis="x", alpha=0.3)
    for index, (value, source) in enumerate(zip(values, sources)):
        plt.text(value, index, f" {value:.1f} ms\n {source}", va="center", fontsize=8)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[TEST] plot saved to {output_path}")


def _menu7_phase_step_order(row: dict) -> int:
    """Return the menu-7 x-axis order: prefill first, then decode steps."""

    explicit_order = row.get("phase_step_order")
    if explicit_order is not None:
        return int(explicit_order)
    return 1 if row.get("phase") == "prefill" else int(row.get("step") or 0) + 1


def plot_menu7_forward_elapsed_lines(
    forward_rows: list[dict],
    output_dir: Path,
    prefix: str,
    expected_phase_steps: list[tuple[str, int]] | None = None,
    dense_output_axis: bool = False,
) -> None:
    """
    Plot menu-7 forward elapsed time for every stage and for their per-step sum.

    Each request is one line. The x-axis is the request-local sequence of one
    prefill followed by decode rounds, not wall-clock time. A total point is
    plotted only when every stage reported that request/round, preventing a
    partial report from being mistaken for the full pipeline forward time.
    """

    usable_rows = [
        row
        for row in forward_rows
        if _float_or_none(row.get("forward_elapsed_ms")) is not None
    ]
    if not usable_rows:
        print("[TEST] no forward elapsed rows available for menu-7 line plots.")
        return

    trend_output_dir = output_dir / "forward_delay_trend"
    trend_output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stage_keys = sorted(
        {
            (
                str(row.get("node_id")),
                int(row.get("shards_start") or 0),
                int(row.get("shards_end") or 0),
            )
            for row in usable_rows
        },
        key=lambda item: (item[1], item[2], item[0]),
    )
    request_orders = sorted(
        {int(row.get("request_order") or 0) for row in usable_rows}
    )
    if expected_phase_steps is not None:
        phase_orders = list(range(1, len(expected_phase_steps) + 1))
    else:
        phase_orders = sorted({_menu7_phase_step_order(row) for row in usable_rows})

    values_by_stage_request_step: dict[tuple[str, int, int, int, int], float] = {}
    for row in usable_rows:
        node_id = str(row.get("node_id"))
        shards_start = int(row.get("shards_start") or 0)
        shards_end = int(row.get("shards_end") or 0)
        request_order = int(row.get("request_order") or 0)
        phase_order = _menu7_phase_step_order(row)
        values_by_stage_request_step[
            (node_id, shards_start, shards_end, request_order, phase_order)
        ] = _float_or_none(row.get("forward_elapsed_ms")) or 0.0

    x_positions = list(range(len(phase_orders)))
    for node_id, shards_start, shards_end in stage_keys:
        plt.figure(figsize=(9, 5))
        for request_order in request_orders:
            values = [
                values_by_stage_request_step.get(
                    (node_id, shards_start, shards_end, request_order, phase_order),
                    math.nan,
                )
                for phase_order in phase_orders
            ]
            plt.plot(
                x_positions,
                values,
                linewidth=0.5 if dense_output_axis else 0.8,
                label=f"request {request_order}",
            )
        plt.xticks([])
        plt.xlabel("Forward round (prefill followed by decode)")
        plt.ylabel("forward_elapsed_ms (ms)")
        plt.title(f"Forward elapsed - {node_id} shards {shards_start}~{shards_end}")
        plt.legend()
        plt.tight_layout()
        output_path = trend_output_dir / (
            f"{prefix}_forward_elapsed_{safe_filename_part(node_id)}_"
            f"shards_{shards_start}~{shards_end}.png"
        )
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"[TEST] plot saved to {output_path}")

    plt.figure(figsize=(9, 5))
    missing_total_points = 0
    for request_order in request_orders:
        total_values = []
        for phase_order in phase_orders:
            stage_values = [
                values_by_stage_request_step.get(
                    (node_id, shards_start, shards_end, request_order, phase_order)
                )
                for node_id, shards_start, shards_end in stage_keys
            ]
            if any(value is None for value in stage_values):
                total_values.append(math.nan)
                missing_total_points += 1
            else:
                total_values.append(sum(stage_values))
        plt.plot(
            x_positions,
            total_values,
            linewidth=0.5 if dense_output_axis else 0.8,
            label=f"request {request_order}",
        )
    plt.xticks([])
    plt.xlabel("Forward round (prefill followed by decode)")
    plt.ylabel("sum of forward_elapsed_ms across stages (ms)")
    plt.title("Total forward elapsed across pipeline stages")
    plt.legend()
    plt.tight_layout()
    total_output_path = trend_output_dir / f"{prefix}_forward_elapsed_total.png"
    plt.savefig(total_output_path, dpi=150)
    plt.close()
    print(f"[TEST] plot saved to {total_output_path}")
    if missing_total_points:
        print(
            f"[WARNING] omitted {missing_total_points} incomplete request/round "
            "points from the total forward-elapsed plot."
        )


def plot_sweep_metric_lines(
    rows: list[dict],
    metric_specs: list[tuple[str, str]],
    output_path: Path,
    title: str,
    y_label: str,
    value_transform: Callable[[float], float] | None = None,
) -> None:
    if not rows:
        print(f"[TEST] no rows to plot for {output_path}")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_active_values = sorted(
        {int(row["max_active_requests"]) for row in rows if row.get("max_active_requests") is not None}
    )
    if not max_active_values:
        print(f"[TEST] no max_active_requests values for {output_path}")
        return

    plt.figure(figsize=(9, 5))
    for metric_key, label in metric_specs:
        xs = []
        ys = []
        for max_active in max_active_values:
            raw_values = [
                _float_or_none(row.get(metric_key))
                for row in rows
                if int(row.get("max_active_requests") or 0) == max_active
            ]
            mean_value = mean_or_none([
                value for value in raw_values if value is not None
            ])
            if mean_value is None:
                continue
            xs.append(max_active)
            ys.append(value_transform(mean_value) if value_transform else mean_value)
        if xs:
            plt.plot(xs, ys, marker="o", linewidth=1.4, label=label)

    plt.xlabel("max_active_requests")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[TEST] plot saved to {output_path}")


def plot_stage_time_utilization(
    rows: list[dict],
    output_path: Path,
) -> None:
    if not rows:
        print(f"[TEST] no stage utilization rows to plot for {output_path}")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stage_keys = sorted(
        {
            (
                str(row.get("node_id")),
                int(row.get("shards_start") or 0),
                int(row.get("shards_end") or 0),
            )
            for row in rows
        },
        key=lambda item: (item[1], item[2], item[0]),
    )
    max_active_values = sorted(
        {int(row.get("max_active_requests") or 0) for row in rows}
    )

    plt.figure(figsize=(9, 5))
    for node_id, shards_start, shards_end in stage_keys:
        xs = []
        ys = []
        for max_active in max_active_values:
            values = [
                _float_or_none(row.get("stage_time_utilization"))
                for row in rows
                if str(row.get("node_id")) == node_id
                and int(row.get("shards_start") or 0) == shards_start
                and int(row.get("shards_end") or 0) == shards_end
                and int(row.get("max_active_requests") or 0) == max_active
            ]
            mean_value = mean_or_none([
                value for value in values if value is not None
            ])
            if mean_value is None:
                continue
            xs.append(max_active)
            ys.append(mean_value * 100.0)
        if xs:
            plt.plot(
                xs,
                ys,
                marker="o",
                linewidth=1.4,
                label=f"{node_id} {shards_start}~{shards_end}",
            )

    plt.xlabel("max_active_requests")
    plt.ylabel("stage time utilization (%)")
    plt.title(
        "Stage time utilization (forward time / total completion time; lower means more idle)"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[TEST] plot saved to {output_path}")


def plot_memory_peak_by_node(
    rows: list[dict],
    metric_key: str,
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    if not rows:
        print(f"[TEST] no memory peak rows to plot for {output_path}")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    node_keys = sorted(
        {
            (
                str(row.get("node_id")),
                int(row.get("shards_start") or 0),
                int(row.get("shards_end") or 0),
            )
            for row in rows
        },
        key=lambda item: (item[1], item[2], item[0]),
    )
    max_active_values = sorted(
        {int(row.get("max_active_requests") or 0) for row in rows}
    )

    plt.figure(figsize=(9, 5))
    for node_id, shards_start, shards_end in node_keys:
        xs = []
        ys = []
        for max_active in max_active_values:
            values = [
                _float_or_none(row.get(metric_key))
                for row in rows
                if str(row.get("node_id")) == node_id
                and int(row.get("shards_start") or 0) == shards_start
                and int(row.get("shards_end") or 0) == shards_end
                and int(row.get("max_active_requests") or 0) == max_active
            ]
            mean_value = mean_or_none([
                value for value in values if value is not None
            ])
            if mean_value is None:
                continue
            xs.append(max_active)
            ys.append(bytes_to_mib(mean_value))
        if xs:
            plt.plot(
                xs,
                ys,
                marker="o",
                linewidth=1.4,
                label=f"{node_id} {shards_start}~{shards_end}",
            )

    plt.xlabel("max_active_requests")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[TEST] plot saved to {output_path}")


def sample_rows_by_x_interval(
    rows: list[dict],
    x_key: str,
    interval: float,
    group_key: str | None = None,
) -> list[dict]:
    """
    按横轴数值间隔采样散点，保留完整 rows 给拟合使用。

    overlap 图的横轴是 elapsed_s，因此 interval 的单位就是秒。每个 group 内
    保留首尾点，并在相邻采样点的 elapsed_s 至少相差 interval 时保留新点。
    """

    if interval <= 0:
        return rows

    def sample_one(group_rows: list[dict]) -> list[dict]:
        sorted_rows = sorted(group_rows, key=lambda row: float(row[x_key]))
        if len(sorted_rows) <= 2:
            return sorted_rows

        sampled = [sorted_rows[0]]
        last_sample_x = float(sorted_rows[0][x_key])
        for row in sorted_rows[1:-1]:
            current_x = float(row[x_key])
            if current_x - last_sample_x >= interval:
                sampled.append(row)
                last_sample_x = current_x
        if sampled[-1] is not sorted_rows[-1]:
            sampled.append(sorted_rows[-1])
        return sampled

    if group_key is None:
        return sample_one(rows)

    sampled_rows: list[dict] = []
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get(group_key, "unknown")), []).append(row)
    for group_rows in groups.values():
        sampled_rows.extend(sample_one(group_rows))
    return sampled_rows


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_PATH)


def build_input_ids_for_lengths(tokenizer, token_lengths: list[int]) -> dict[int, torch.Tensor]:
    max_length = max(token_lengths)
    text = PROMPT_FRAGMENT
    while len(tokenizer(text, return_tensors="pt")["input_ids"][0]) < max_length:
        text += PROMPT_FRAGMENT
    long_input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    return {
        token_length: long_input_ids[:, :token_length].clone()
        for token_length in token_lengths
    }


def build_pipeline_latency_warmup_input_ids(
    tokenizer,
    actual_prompts: list[str],
    actual_input_ids_list: list[torch.Tensor] | None,
) -> torch.Tensor:
    """
    Build a fixed-length warm-up input that cannot reuse a measured request.

    Menu 7 warm-up must exercise a stable 16-token prefill plus 16 decode rounds
    without sharing the measured request's prompt/input IDs. Candidate texts are
    tokenized and truncated exactly like the profiler-style input construction;
    the first sequence that does not match any measured request's first 16 token
    IDs is selected.
    """

    actual_prefixes: set[tuple[int, ...]] = set()
    if actual_input_ids_list is not None:
        actual_prefixes.update(
            tuple(
                int(token_id)
                for token_id in input_ids[0, :PIPELINE_LATENCY_WARMUP_INPUT_TOKEN_LENGTH].tolist()
            )
            for input_ids in actual_input_ids_list
        )
    else:
        actual_prefixes.update(
            tuple(
                int(token_id)
                for token_id in tokenizer(prompt, return_tensors="pt")["input_ids"][
                    0, :PIPELINE_LATENCY_WARMUP_INPUT_TOKEN_LENGTH
                ].tolist()
            )
            for prompt in actual_prompts
        )

    for seed_text in PIPELINE_LATENCY_WARMUP_TEXTS:
        text = seed_text
        while (
            len(tokenizer(text, return_tensors="pt")["input_ids"][0])
            < PIPELINE_LATENCY_WARMUP_INPUT_TOKEN_LENGTH
        ):
            text += seed_text
        warmup_input_ids = tokenizer(text, return_tensors="pt")["input_ids"][
            :, :PIPELINE_LATENCY_WARMUP_INPUT_TOKEN_LENGTH
        ].clone()
        warmup_sequence = tuple(int(token_id) for token_id in warmup_input_ids[0].tolist())
        if warmup_sequence not in actual_prefixes:
            return warmup_input_ids

    raise RuntimeError(
        "[ERROR] failed to construct a warm-up input distinct from all measured requests."
    )


def build_distinct_input_ids_for_same_length(
    tokenizer,
    token_length: int,
    request_count: int,
) -> list[torch.Tensor]:
    """
    Profiler-style equal-length prompt construction for different requests.

    This mirrors the profiler path: build text long enough for the target,
    tokenize it, then slice exact-length input ids. The only addition is that
    each request starts from a different fragment order, so token length is
    identical while the token-id sequence is different.
    """

    if token_length < 8:
        raise ValueError(
            "[ERROR] token_length must be at least 8 for distinct same-length prompts."
        )
    if request_count < 1:
        raise ValueError("[ERROR] request_count must be positive.")

    input_ids_list: list[torch.Tensor] = []
    for request_index in range(request_count):
        text = f"Profiler-style distinct request seed {request_index + 1}. "
        fragment_index = request_index
        while len(tokenizer(text, return_tensors="pt")["input_ids"][0]) < token_length:
            text += DISTINCT_PROMPT_FRAGMENTS[
                fragment_index % len(DISTINCT_PROMPT_FRAGMENTS)
            ]
            fragment_index += 1
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"][
            :, :token_length
        ].clone()
        input_ids_list.append(input_ids)

    unique_sequences = {tuple(input_ids[0].tolist()) for input_ids in input_ids_list}
    if len(unique_sequences) != request_count:
        raise RuntimeError(
            "[ERROR] generated duplicate input_ids; increase token_length or adjust fragments."
        )
    return input_ids_list


def aggregate_reports(reports_by_node: dict[str, dict]) -> dict:
    reports = list(reports_by_node.values())
    total_bytes = sum(int(report.get("kv_cache_bytes") or 0) for report in reports)
    total_numel = sum(int(report.get("kv_cache_numel") or 0) for report in reports)
    total_tensors = sum(int(report.get("kv_cache_tensor_count") or 0) for report in reports)
    max_cache_tokens = max(int(report.get("kv_cache_token_length") or 0) for report in reports)

    def sum_optional_int(key: str) -> int | None:
        values = [
            int(report[key])
            for report in reports
            if report.get(key) is not None
        ]
        return sum(values) if values else None

    total_cuda_memory_allocated = sum_optional_int("cuda_memory_allocated_bytes")
    total_cuda_memory_baseline = sum_optional_int("cuda_memory_baseline_bytes")
    total_cuda_memory_delta = sum_optional_int("cuda_memory_delta_bytes")
    total_mem_info_free = sum_optional_int("cuda_mem_get_info_free_bytes")
    total_mem_info_used = sum_optional_int("cuda_mem_get_info_used_bytes")
    total_mem_info_total = sum_optional_int("cuda_mem_get_info_total_bytes")
    total_mem_info_free_baseline = sum_optional_int(
        "cuda_mem_get_info_free_baseline_bytes"
    )
    total_mem_info_used_baseline = sum_optional_int(
        "cuda_mem_get_info_used_baseline_bytes"
    )
    total_mem_info_used_delta = sum_optional_int(
        "cuda_mem_get_info_used_delta_bytes"
    )
    total_mem_info_free_delta = sum_optional_int(
        "cuda_mem_get_info_free_delta_bytes"
    )
    timestamps = [float(report.get("timestamp") or 0.0) for report in reports]
    first = reports[0]
    return {
        "client_request_id": first.get("client_request_id"),
        "request_id": first.get("request_id"),
        "trace_label": first.get("trace_label"),
        "phase": first.get("phase"),
        "step": first.get("step"),
        "forward_seq_len": first.get("forward_seq_len"),
        "node_count": len(reports),
        "kv_cache_bytes": total_bytes,
        "kv_cache_mib": bytes_to_mib(total_bytes),
        "kv_cache_numel": total_numel,
        "kv_cache_tensor_count": total_tensors,
        "kv_cache_token_length": max_cache_tokens,
        "cuda_memory_allocated_bytes": total_cuda_memory_allocated,
        "cuda_memory_allocated_mib": (
            bytes_to_mib(total_cuda_memory_allocated)
            if total_cuda_memory_allocated is not None
            else None
        ),
        "cuda_memory_baseline_bytes": total_cuda_memory_baseline,
        "cuda_memory_baseline_mib": (
            bytes_to_mib(total_cuda_memory_baseline)
            if total_cuda_memory_baseline is not None
            else None
        ),
        "cuda_memory_delta_bytes": total_cuda_memory_delta,
        "cuda_memory_delta_mib": (
            bytes_to_mib(total_cuda_memory_delta)
            if total_cuda_memory_delta is not None
            else None
        ),
        "cuda_mem_get_info_free_bytes": total_mem_info_free,
        "cuda_mem_get_info_free_mib": (
            bytes_to_mib(total_mem_info_free)
            if total_mem_info_free is not None
            else None
        ),
        "cuda_mem_get_info_used_bytes": total_mem_info_used,
        "cuda_mem_get_info_used_mib": (
            bytes_to_mib(total_mem_info_used)
            if total_mem_info_used is not None
            else None
        ),
        "cuda_mem_get_info_total_bytes": total_mem_info_total,
        "cuda_mem_get_info_total_mib": (
            bytes_to_mib(total_mem_info_total)
            if total_mem_info_total is not None
            else None
        ),
        "cuda_mem_get_info_free_baseline_bytes": total_mem_info_free_baseline,
        "cuda_mem_get_info_free_baseline_mib": (
            bytes_to_mib(total_mem_info_free_baseline)
            if total_mem_info_free_baseline is not None
            else None
        ),
        "cuda_mem_get_info_used_baseline_bytes": total_mem_info_used_baseline,
        "cuda_mem_get_info_used_baseline_mib": (
            bytes_to_mib(total_mem_info_used_baseline)
            if total_mem_info_used_baseline is not None
            else None
        ),
        "cuda_mem_get_info_used_delta_bytes": total_mem_info_used_delta,
        "cuda_mem_get_info_used_delta_mib": (
            bytes_to_mib(total_mem_info_used_delta)
            if total_mem_info_used_delta is not None
            else None
        ),
        "cuda_mem_get_info_free_delta_bytes": total_mem_info_free_delta,
        "cuda_mem_get_info_free_delta_mib": (
            bytes_to_mib(total_mem_info_free_delta)
            if total_mem_info_free_delta is not None
            else None
        ),
        "first_timestamp": min(timestamps),
        "last_timestamp": max(timestamps),
    }


def collect_complete_aggregate(
    client: PipelineDebugClient,
    client_request_id: str,
    phase: str,
    step: int,
    timeout_s: float = 60.0,
) -> tuple[dict, list[dict]]:
    expected_node_ids = {node.node_id for node in client.nodes}
    reports_by_node: dict[str, dict] = {}
    held_events: list[dict] = []
    deadline = time.time() + timeout_s
    try:
        while time.time() < deadline and set(reports_by_node) != expected_node_ids:
            event = client.receive_event(timeout_s=max(0.05, min(0.5, deadline - time.time())))
            if event is None:
                continue
            if (
                event.get("type") == KV_CACHE_REPORT
                and event.get("client_request_id") == client_request_id
                and event.get("phase") == phase
                and int(event.get("step") or 0) == step
                and event.get("event") == "post_shard_forward"
            ):
                reports_by_node[str(event.get("node_id"))] = event
            else:
                held_events.append(event)
    finally:
        client._event_backlog[:0] = held_events

    missing = expected_node_ids - set(reports_by_node)
    if missing:
        raise TimeoutError(
            f"[ERROR] timed out waiting for {phase} step={step} reports; missing={sorted(missing)}"
        )
    return aggregate_reports(reports_by_node), list(reports_by_node.values())


def submit_one(client: PipelineDebugClient) -> None:
    owner_index = ask_node(client)
    prompt = ask_prompt()
    max_new_tokens = ask_int("max_new_tokens", 128, minimum=1)
    client.submit_request(owner_index, prompt, max_new_tokens)


def submit_many(client: PipelineDebugClient) -> None:
    count = ask_int("How many requests", 4, minimum=1)
    max_new_tokens = ask_int("max_new_tokens for each request", 128, minimum=1)
    round_robin = ask_yes_no("Round-robin owners across all nodes?", default=True)
    owner_index = 0 if round_robin else ask_node(client)

    for i in range(count):
        prompt = DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)]
        if count <= len(DEFAULT_PROMPTS):
            custom = input(f"Prompt {i + 1} [{prompt}]: ").strip()
            prompt = custom or prompt
        target = i % client.pipeline_depth if round_robin else owner_index
        client.submit_request(target, prompt, max_new_tokens)
        time.sleep(0.05)


def change_max_active(client: PipelineDebugClient) -> None:
    client.max_active_requests = ask_int(
        "max_active_requests",
        client.max_active_requests,
        minimum=1,
    )
    client.send_config()


def collect_decode_aggregates(
    client: PipelineDebugClient,
    client_request_id: str,
    max_decode_step: int,
    timeout_s: float,
) -> tuple[list[dict], list[dict], dict | None]:
    expected_node_ids = {node.node_id for node in client.nodes}
    reports_by_key: dict[tuple[str, int], dict[str, dict]] = {}
    completed_keys: set[tuple[str, int]] = set()
    aggregate_rows: list[dict] = []
    per_node_rows: list[dict] = []
    done_event = None
    held_events: list[dict] = []
    deadline = time.time() + timeout_s

    wanted_keys = {("prefill", 0)}
    wanted_keys.update(("decode", step) for step in range(1, max_decode_step + 1))

    try:
        while time.time() < deadline:
            if all(key in completed_keys for key in wanted_keys):
                break
            event = client.receive_event(timeout_s=max(0.05, min(0.5, deadline - time.time())))
            if event is None:
                continue
            if (
                event.get("type") == PIPELINE_DONE_REPORT
                and event.get("client_request_id") == client_request_id
            ):
                done_event = event
                continue
            if (
                event.get("type") != KV_CACHE_REPORT
                or event.get("client_request_id") != client_request_id
                or event.get("event") != "post_shard_forward"
            ):
                held_events.append(event)
                continue

            phase = str(event.get("phase"))
            step = int(event.get("step") or 0)
            key = (phase, step)
            if key not in wanted_keys:
                continue
            per_node_rows.append(event)
            reports_by_key.setdefault(key, {})[str(event.get("node_id"))] = event
            if key not in completed_keys and set(reports_by_key[key]) == expected_node_ids:
                completed_keys.add(key)
                aggregate = aggregate_reports(reports_by_key[key])
                aggregate["output_token_length"] = 0 if phase == "prefill" else step
                aggregate_rows.append(aggregate)
    finally:
        client._event_backlog[:0] = held_events

    missing = sorted(wanted_keys - completed_keys)
    if missing:
        print(f"[WARNING] decode collection ended before all steps arrived: missing={missing[:10]}")
    aggregate_rows.sort(key=lambda row: (row["output_token_length"], row["phase"]))
    return aggregate_rows, per_node_rows, done_event


def run_prefill_kv_experiment(client: PipelineDebugClient, result_dir: Path) -> None:
    print("\n[TEST] Prefill KV cache size vs input token length")
    ask_telemetry(client)
    owner_index = ask_node(client)
    token_lengths = ask_int_list("Input token lengths", DEFAULT_PREFILL_TOKEN_LENGTHS)
    repeats = ask_int("Repeat count per length", 1, minimum=1)
    tokenizer = load_tokenizer()
    input_ids_by_length = build_input_ids_for_lengths(tokenizer, token_lengths)

    summary_rows: list[dict] = []
    per_node_rows: list[dict] = []
    client.drain_events()

    for token_length in token_lengths:
        for repeat_index in range(repeats):
            trace_label = f"prefill_len{token_length}_rep{repeat_index + 1}"
            client_request_id = client.submit_request(
                owner_index=owner_index,
                prompt="",
                input_ids=input_ids_by_length[token_length],
                max_new_tokens=1,
                trace_kv_cache=True,
                trace_label=trace_label,
            )
            ack = client.wait_for_ack(client_request_id)
            aggregate, reports = collect_complete_aggregate(
                client,
                client_request_id=client_request_id,
                phase="prefill",
                step=0,
                timeout_s=60.0,
            )
            aggregate.update(
                {
                    "experiment": "prefill_by_input_length",
                    "target_input_token_length": token_length,
                    "actual_input_token_length": ack.get("input_token_length"),
                    "repeat_index": repeat_index + 1,
                }
            )
            for report in reports:
                report.update(
                    {
                        "experiment": "prefill_by_input_length",
                        "target_input_token_length": token_length,
                        "actual_input_token_length": ack.get("input_token_length"),
                        "repeat_index": repeat_index + 1,
                    }
                )
            summary_rows.append(aggregate)
            per_node_rows.extend(reports)
            client.wait_for_done(client_request_id, timeout_s=60.0)
            cuda_info = ""
            if aggregate.get("cuda_memory_delta_bytes") is not None:
                cuda_info = (
                    f", CUDA delta="
                    f"{bytes_to_mib(aggregate['cuda_memory_delta_bytes']):.6f} MiB"
                )
            print(
                "[TEST] prefill sample: "
                f"tokens={ack.get('input_token_length')}, "
                f"KV={aggregate['kv_cache_mib']:.6f} MiB"
                f"{cuda_info}"
            )

    write_csv(result_dir / "prefill_kv_summary.csv", summary_rows)
    write_csv(result_dir / "prefill_kv_per_node.csv", per_node_rows)
    fit_rows = plot_scatter_with_fit(
        summary_rows,
        x_key="actual_input_token_length",
        y_key="kv_cache_bytes",
        output_path=result_dir / "prefill_kv_fit.png",
        title="Pipeline KV cache after prefill",
        x_label="Input token length",
        y_label="Memory size (MiB)",
        extra_y_series=[("cuda_memory_delta_bytes", CUDA_MEMORY_BASELINE_DELTA_LABEL)],
    )
    write_csv(result_dir / "prefill_kv_fit.csv", fit_rows)
    per_node_fit_rows = plot_per_node_scatter_with_fit(
        per_node_rows,
        output_dir=result_dir,
        filename_prefix="prefill_kv_per_node",
        x_key="actual_input_token_length",
        y_key="kv_cache_bytes",
        title_prefix="Pipeline KV cache after prefill",
        x_label="Input token length",
        extra_y_series=PER_NODE_MEMORY_SERIES,
    )
    write_csv(result_dir / "prefill_kv_per_node_fit.csv", per_node_fit_rows)


def run_decode_kv_experiment(client: PipelineDebugClient, result_dir: Path) -> None:
    print("\n[TEST] Decode KV cache size vs generated output token length")
    ask_telemetry(client)
    owner_index = ask_node(client)
    sample_steps = ask_int_list("Output token checkpoints", DEFAULT_DECODE_OUTPUT_TOKEN_LENGTHS)
    max_step = max(sample_steps)
    prompt = ask_prompt("The capital of France is")

    client.drain_events()
    client_request_id = client.submit_request(
        owner_index=owner_index,
        prompt=prompt,
        max_new_tokens=max_step + 1,
        trace_kv_cache=True,
        trace_label="decode_growth",
    )
    ack = client.wait_for_ack(client_request_id)
    aggregate_rows, per_node_rows, done_event = collect_decode_aggregates(
        client,
        client_request_id=client_request_id,
        max_decode_step=max_step,
        timeout_s=max(120.0, max_step * 5.0),
    )
    for row in aggregate_rows:
        row.update(
            {
                "experiment": "decode_by_output_length",
                "input_token_length": ack.get("input_token_length"),
            }
        )
    for row in per_node_rows:
        row.update(
            {
                "experiment": "decode_by_output_length",
                "input_token_length": ack.get("input_token_length"),
                "output_token_length": (
                    0 if row.get("phase") == "prefill" else int(row.get("step") or 0)
                ),
            }
        )
    if done_event is None:
        done_event = client.wait_for_done(client_request_id, timeout_s=30.0)

    decode_rows = [row for row in aggregate_rows if row.get("phase") == "decode"]
    sampled_decode_rows = [
        row
        for row in decode_rows
        if int(row.get("output_token_length") or 0) in sample_steps
    ]
    write_csv(result_dir / "decode_kv_summary.csv", aggregate_rows)
    write_csv(result_dir / "decode_kv_per_node.csv", per_node_rows)
    fit_rows = plot_scatter_with_fit(
        decode_rows,
        x_key="output_token_length",
        y_key="kv_cache_bytes",
        output_path=result_dir / "decode_kv_fit.png",
        title="Pipeline KV cache during decode",
        x_label="Generated output token length",
        y_label="Memory size (MiB)",
        scatter_rows=sampled_decode_rows,
        extra_y_series=[("cuda_memory_delta_bytes", CUDA_MEMORY_BASELINE_DELTA_LABEL)],
    )
    write_csv(result_dir / "decode_kv_fit.csv", fit_rows)
    decode_node_rows = [row for row in per_node_rows if row.get("phase") == "decode"]
    sampled_decode_node_rows = [
        row
        for row in decode_node_rows
        if int(row.get("output_token_length") or 0) in sample_steps
    ]
    per_node_fit_rows = plot_per_node_scatter_with_fit(
        decode_node_rows,
        output_dir=result_dir,
        filename_prefix="decode_kv_per_node",
        x_key="output_token_length",
        y_key="kv_cache_bytes",
        title_prefix="Pipeline KV cache during decode",
        x_label="Generated output token length",
        scatter_rows=sampled_decode_node_rows,
        extra_y_series=PER_NODE_MEMORY_SERIES,
    )
    write_csv(result_dir / "decode_kv_per_node_fit.csv", per_node_fit_rows)
    if done_event:
        write_csv(result_dir / "decode_done.csv", [done_event])


def run_sequential_same_prompt_experiment(client: PipelineDebugClient, result_dir: Path) -> None:
    print("\n[TEST] Sequential same-prompt KV cache consistency")
    ask_telemetry(client)
    owner_index = ask_node(client)
    token_lengths = ask_int_list("Input token lengths to compare", [8, 16, 32, 64, 128])
    tokenizer = load_tokenizer()
    input_ids_by_length = build_input_ids_for_lengths(tokenizer, token_lengths)

    summary_rows: list[dict] = []
    per_node_rows: list[dict] = []
    comparison_rows: list[dict] = []
    client.drain_events()

    for token_length in token_lengths:
        pair_rows = []
        for order in (1, 2):
            trace_label = f"sequential_len{token_length}_order{order}"
            client_request_id = client.submit_request(
                owner_index=owner_index,
                prompt="",
                input_ids=input_ids_by_length[token_length],
                max_new_tokens=1,
                trace_kv_cache=True,
                trace_label=trace_label,
            )
            ack = client.wait_for_ack(client_request_id)
            aggregate, reports = collect_complete_aggregate(
                client,
                client_request_id=client_request_id,
                phase="prefill",
                step=0,
                timeout_s=60.0,
            )
            aggregate.update(
                {
                    "experiment": "sequential_same_prompt",
                    "target_input_token_length": token_length,
                    "actual_input_token_length": ack.get("input_token_length"),
                    "order": order,
                }
            )
            for report in reports:
                report.update(
                    {
                        "experiment": "sequential_same_prompt",
                        "target_input_token_length": token_length,
                        "actual_input_token_length": ack.get("input_token_length"),
                        "order": order,
                    }
                )
            summary_rows.append(aggregate)
            per_node_rows.extend(reports)
            pair_rows.append(aggregate)
            client.wait_for_done(client_request_id, timeout_s=60.0)

        first, second = pair_rows
        first_bytes = int(first["kv_cache_bytes"])
        second_bytes = int(second["kv_cache_bytes"])
        first_cuda_allocated_bytes = first.get("cuda_memory_allocated_bytes")
        second_cuda_allocated_bytes = second.get("cuda_memory_allocated_bytes")
        cuda_allocated_difference_bytes = (
            None
            if first_cuda_allocated_bytes is None or second_cuda_allocated_bytes is None
            else int(second_cuda_allocated_bytes) - int(first_cuda_allocated_bytes)
        )
        first_cuda_delta_bytes = first.get("cuda_memory_delta_bytes")
        second_cuda_delta_bytes = second.get("cuda_memory_delta_bytes")
        cuda_delta_difference_bytes = (
            None
            if first_cuda_delta_bytes is None or second_cuda_delta_bytes is None
            else int(second_cuda_delta_bytes) - int(first_cuda_delta_bytes)
        )
        comparison_rows.append(
            {
                "target_input_token_length": token_length,
                "actual_input_token_length": first.get("actual_input_token_length"),
                "first_kv_cache_bytes": first_bytes,
                "second_kv_cache_bytes": second_bytes,
                "delta_bytes": second_bytes - first_bytes,
                "relative_delta": (
                    0.0 if first_bytes == 0 else (second_bytes - first_bytes) / first_bytes
                ),
                "first_cuda_memory_allocated_bytes": first_cuda_allocated_bytes,
                "second_cuda_memory_allocated_bytes": second_cuda_allocated_bytes,
                "cuda_memory_allocated_difference_bytes": cuda_allocated_difference_bytes,
                "first_cuda_memory_delta_bytes": first_cuda_delta_bytes,
                "second_cuda_memory_delta_bytes": second_cuda_delta_bytes,
                "cuda_memory_delta_difference_bytes": cuda_delta_difference_bytes,
            }
        )
        print(
            "[TEST] sequential compare: "
            f"tokens={first.get('actual_input_token_length')}, "
            f"first={bytes_to_mib(first_bytes):.6f} MiB, "
            f"second={bytes_to_mib(second_bytes):.6f} MiB"
            + (
                ""
                if first_cuda_delta_bytes is None or second_cuda_delta_bytes is None
                else (
                    f", CUDA delta first={bytes_to_mib(first_cuda_delta_bytes):.6f} MiB, "
                    f"CUDA delta second={bytes_to_mib(second_cuda_delta_bytes):.6f} MiB"
                )
            )
        )

    write_csv(result_dir / "sequential_same_prompt_summary.csv", summary_rows)
    write_csv(result_dir / "sequential_same_prompt_per_node.csv", per_node_rows)
    write_csv(result_dir / "sequential_same_prompt_comparison.csv", comparison_rows)
    fit_rows = plot_scatter_with_fit(
        summary_rows,
        x_key="actual_input_token_length",
        y_key="kv_cache_bytes",
        output_path=result_dir / "sequential_same_prompt_fit.png",
        title="Sequential same-prompt prefill KV cache",
        x_label="Input token length",
        y_label="Memory size (MiB)",
        group_key="order",
        extra_y_series=[("cuda_memory_delta_bytes", CUDA_MEMORY_BASELINE_DELTA_LABEL)],
    )
    write_csv(result_dir / "sequential_same_prompt_fit.csv", fit_rows)
    per_node_fit_rows = plot_per_node_scatter_with_fit(
        per_node_rows,
        output_dir=result_dir,
        filename_prefix="sequential_same_prompt_per_node",
        x_key="actual_input_token_length",
        y_key="kv_cache_bytes",
        title_prefix="Sequential same-prompt prefill KV cache",
        x_label="Input token length",
        group_key="order",
        extra_y_series=PER_NODE_MEMORY_SERIES,
    )
    write_csv(result_dir / "sequential_same_prompt_per_node_fit.csv", per_node_fit_rows)


def run_overlap_same_prompt_experiment(client: PipelineDebugClient, result_dir: Path) -> None:
    print("\n[TEST] Overlapped same-prompt KV cache growth over time")
    ask_telemetry(client)
    owner_index = ask_node(client)
    input_token_length = ask_int("Input token length", 64, minimum=1)
    second_delay_s = float(input("Delay before second request seconds [10]: ").strip() or "10")
    max_new_tokens = ask_int("max_new_tokens for each request", 256, minimum=2)
    collect_after_second_s = float(
        input("Collect seconds after second request [30]: ").strip() or "30"
    )

    tokenizer = load_tokenizer()
    input_ids = build_input_ids_for_lengths(tokenizer, [input_token_length])[input_token_length]
    if client.max_active_requests < 2:
        client.max_active_requests = 2
        client.send_config()
    client.drain_events()

    start_time = time.time()
    first_client_id = client.submit_request(
        owner_index=owner_index,
        prompt="",
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        trace_kv_cache=True,
        trace_label="overlap_first",
    )
    first_ack = client.wait_for_ack(first_client_id)

    expected_node_ids = {node.node_id for node in client.nodes}
    reports_by_key: dict[tuple[str, str, int], dict[str, dict]] = {}
    request_rows: list[dict] = []
    per_node_rows: list[dict] = []
    total_rows: list[dict] = []
    node_total_rows: list[dict] = []
    done_requests: set[str] = set()
    client_ids = {first_client_id: "first"}
    latest_bytes_by_client: dict[str, int] = {}
    latest_bytes_by_node_client: dict[tuple[str, str], int] = {}
    cuda_initial_baseline_by_node: dict[str, int] = {}
    second_client_id = None
    second_submit_elapsed = second_delay_s

    def process_event(event: dict) -> None:
        nonlocal second_submit_elapsed
        if event.get("type") == PIPELINE_DONE_REPORT and event.get("client_request_id") in client_ids:
            current_client_id = str(event.get("client_request_id"))
            done_requests.add(current_client_id)
            latest_bytes_by_client[current_client_id] = 0
            for node_id in expected_node_ids:
                latest_bytes_by_node_client[(node_id, current_client_id)] = 0
            elapsed = float(event.get("timestamp") or time.time()) - start_time
            total_rows.append(
                {
                    "elapsed_s": elapsed,
                    "total_kv_cache_bytes": sum(latest_bytes_by_client.values()),
                    "total_kv_cache_mib": bytes_to_mib(sum(latest_bytes_by_client.values())),
                    "total_cuda_memory_allocated_bytes": None,
                    "total_cuda_memory_allocated_mib": None,
                    "total_cuda_memory_initial_baseline_bytes": None,
                    "total_cuda_memory_initial_baseline_mib": None,
                    "total_cuda_memory_delta_bytes": None,
                    "total_cuda_memory_delta_mib": None,
                    "event": "done",
                    "client_request_id": current_client_id,
                    "request_order": client_ids[current_client_id],
                    "second_submit_elapsed_s": second_submit_elapsed,
                }
            )
            for node in client.nodes:
                node_total_bytes = sum(
                    value
                    for (node_id, _), value in latest_bytes_by_node_client.items()
                    if node_id == node.node_id
                )
                node_total_rows.append(
                    {
                        "node_id": node.node_id,
                        "shards_start": node.shards_start,
                        "shards_end": node.shards_end,
                        "elapsed_s": elapsed,
                        "node_total_kv_cache_bytes": node_total_bytes,
                        "node_total_kv_cache_mib": bytes_to_mib(node_total_bytes),
                        "cuda_memory_allocated_bytes": None,
                        "cuda_memory_delta_from_start_bytes": None,
                        "event": "done",
                        "client_request_id": current_client_id,
                        "request_order": client_ids[current_client_id],
                        "second_submit_elapsed_s": second_submit_elapsed,
                    }
                )
            return

        if (
            event.get("type") != KV_CACHE_REPORT
            or event.get("client_request_id") not in client_ids
            or event.get("event") != "post_shard_forward"
        ):
            return

        current_client_id = str(event.get("client_request_id"))
        phase = str(event.get("phase"))
        step = int(event.get("step") or 0)
        key = (current_client_id, phase, step)
        reports_by_key.setdefault(key, {})[str(event.get("node_id"))] = event
        if set(reports_by_key[key]) != expected_node_ids:
            return

        current_reports = reports_by_key[key]
        for node_id, report in current_reports.items():
            if report.get("cuda_memory_baseline_bytes") is not None:
                cuda_initial_baseline_by_node.setdefault(
                    node_id,
                    int(report["cuda_memory_baseline_bytes"]),
                )
        current_cuda_allocated_by_node = {
            node_id: int(report["cuda_memory_allocated_bytes"])
            for node_id, report in current_reports.items()
            if report.get("cuda_memory_allocated_bytes") is not None
        }
        if (
            set(current_cuda_allocated_by_node) == expected_node_ids
            and set(cuda_initial_baseline_by_node) == expected_node_ids
        ):
            total_cuda_memory_allocated = sum(current_cuda_allocated_by_node.values())
            total_cuda_memory_initial_baseline = sum(
                cuda_initial_baseline_by_node.values()
            )
            total_cuda_memory_delta = (
                total_cuda_memory_allocated - total_cuda_memory_initial_baseline
            )
        else:
            total_cuda_memory_allocated = None
            total_cuda_memory_initial_baseline = None
            total_cuda_memory_delta = None

        aggregate = aggregate_reports(reports_by_key[key])
        output_tokens = 0 if phase == "prefill" else step
        elapsed = float(aggregate["last_timestamp"]) - start_time
        row = {
            **aggregate,
            "experiment": "overlap_same_prompt",
            "request_order": client_ids[current_client_id],
            "elapsed_s": elapsed,
            "output_token_length": output_tokens,
            "input_token_length": first_ack.get("input_token_length"),
            "second_submit_elapsed_s": second_submit_elapsed,
        }
        request_rows.append(row)
        for node_id, report in current_reports.items():
            node_row = {
                **report,
                "experiment": "overlap_same_prompt",
                "request_order": client_ids[current_client_id],
                "elapsed_s": elapsed,
                "output_token_length": output_tokens,
                "input_token_length": first_ack.get("input_token_length"),
                "second_submit_elapsed_s": second_submit_elapsed,
                "node_request_group": f"{node_id}:{client_ids[current_client_id]}",
            }
            per_node_rows.append(node_row)
            latest_bytes_by_node_client[(node_id, current_client_id)] = int(
                report.get("kv_cache_bytes") or 0
            )
            node_total_bytes = sum(
                value
                for (current_node_id, _), value in latest_bytes_by_node_client.items()
                if current_node_id == node_id
            )
            cuda_allocated_bytes = (
                int(report["cuda_memory_allocated_bytes"])
                if report.get("cuda_memory_allocated_bytes") is not None
                else None
            )
            cuda_initial_baseline = cuda_initial_baseline_by_node.get(node_id)
            cuda_delta_from_start = (
                cuda_allocated_bytes - cuda_initial_baseline
                if cuda_allocated_bytes is not None and cuda_initial_baseline is not None
                else None
            )
            node_total_rows.append(
                {
                    "node_id": node_id,
                    "shards_start": report.get("shards_start"),
                    "shards_end": report.get("shards_end"),
                    "elapsed_s": elapsed,
                    "node_total_kv_cache_bytes": node_total_bytes,
                    "node_total_kv_cache_mib": bytes_to_mib(node_total_bytes),
                    "cuda_memory_allocated_bytes": cuda_allocated_bytes,
                    "cuda_memory_delta_from_start_bytes": cuda_delta_from_start,
                    "event": f"{client_ids[current_client_id]}_{phase}_{step}",
                    "client_request_id": current_client_id,
                    "request_order": client_ids[current_client_id],
                    "second_submit_elapsed_s": second_submit_elapsed,
                }
            )
        latest_bytes_by_client[current_client_id] = int(aggregate["kv_cache_bytes"])
        total_bytes = sum(latest_bytes_by_client.values())
        total_rows.append(
            {
                "elapsed_s": elapsed,
                "total_kv_cache_bytes": total_bytes,
                "total_kv_cache_mib": bytes_to_mib(total_bytes),
                "total_cuda_memory_allocated_bytes": total_cuda_memory_allocated,
                "total_cuda_memory_allocated_mib": bytes_to_mib(
                    total_cuda_memory_allocated
                )
                if total_cuda_memory_allocated is not None
                else None,
                "total_cuda_memory_initial_baseline_bytes": (
                    total_cuda_memory_initial_baseline
                ),
                "total_cuda_memory_initial_baseline_mib": bytes_to_mib(
                    total_cuda_memory_initial_baseline
                )
                if total_cuda_memory_initial_baseline is not None
                else None,
                "total_cuda_memory_delta_bytes": total_cuda_memory_delta,
                "total_cuda_memory_delta_mib": bytes_to_mib(total_cuda_memory_delta)
                if total_cuda_memory_delta is not None
                else None,
                "event": f"{client_ids[current_client_id]}_{phase}_{step}",
                "client_request_id": current_client_id,
                "request_order": client_ids[current_client_id],
                "second_submit_elapsed_s": second_submit_elapsed,
            }
        )

    while time.time() - start_time < second_delay_s:
        event = client.receive_event(timeout_s=0.2)
        if event is not None:
            process_event(event)

    second_client_id = client.submit_request(
        owner_index=owner_index,
        prompt="",
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        trace_kv_cache=True,
        trace_label="overlap_second",
    )
    second_submit_elapsed = time.time() - start_time
    client_ids[second_client_id] = "second"
    client.wait_for_ack(second_client_id)

    deadline = time.time() + collect_after_second_s
    while time.time() < deadline and len(done_requests) < len(client_ids):
        event = client.receive_event(timeout_s=0.2)
        if event is not None:
            process_event(event)

    write_csv(result_dir / "overlap_same_prompt_requests.csv", request_rows)
    write_csv(result_dir / "overlap_same_prompt_per_node.csv", per_node_rows)
    write_csv(result_dir / "overlap_same_prompt_total.csv", total_rows)
    write_csv(result_dir / "overlap_same_prompt_total_per_node.csv", node_total_rows)
    sampled_request_rows = sample_rows_by_x_interval(
        request_rows,
        x_key="elapsed_s",
        interval=OVERLAP_TIME_SAMPLE_INTERVAL_S,
        group_key="request_order",
    )
    request_fit_rows = plot_scatter_with_fit(
        request_rows,
        x_key="elapsed_s",
        y_key="kv_cache_bytes",
        output_path=result_dir / "overlap_same_prompt_per_request.png",
        title="Overlapped requests KV cache by request",
        x_label="Elapsed time (s)",
        y_label="Memory size (MiB)",
        group_key="request_order",
        vertical_line_x=second_submit_elapsed,
        scatter_rows=sampled_request_rows,
        extra_y_series=[("cuda_memory_delta_bytes", CUDA_MEMORY_BASELINE_DELTA_LABEL)],
    )
    write_csv(result_dir / "overlap_same_prompt_per_request_fit.csv", request_fit_rows)
    sampled_node_request_rows = sample_rows_by_x_interval(
        per_node_rows,
        x_key="elapsed_s",
        interval=OVERLAP_TIME_SAMPLE_INTERVAL_S,
        group_key="node_request_group",
    )
    per_node_request_fit_rows = plot_per_node_scatter_with_fit(
        per_node_rows,
        output_dir=result_dir,
        filename_prefix="overlap_same_prompt_request_per_node",
        x_key="elapsed_s",
        y_key="kv_cache_bytes",
        title_prefix="Overlapped requests KV cache by request",
        x_label="Elapsed time (s)",
        group_key="request_order",
        vertical_line_x=second_submit_elapsed,
        scatter_rows=sampled_node_request_rows,
        extra_y_series=PER_NODE_MEMORY_SERIES,
    )
    write_csv(
        result_dir / "overlap_same_prompt_request_per_node_fit.csv",
        per_node_request_fit_rows,
    )

    growth_total_rows = [row for row in total_rows if row.get("event") != "done"]
    before_rows = [
        row for row in growth_total_rows if float(row["elapsed_s"]) < second_submit_elapsed
    ]
    after_rows = [
        row for row in growth_total_rows if float(row["elapsed_s"]) >= second_submit_elapsed
    ]
    total_fit_input_rows = [
        {**row, "segment": "before_second"} for row in before_rows
    ] + [
        {**row, "segment": "after_second"} for row in after_rows
    ]
    sampled_total_rows = sample_rows_by_x_interval(
        total_fit_input_rows,
        x_key="elapsed_s",
        interval=OVERLAP_TIME_SAMPLE_INTERVAL_S,
        group_key="segment",
    )
    total_fit_rows = plot_scatter_with_fit(
        total_fit_input_rows,
        x_key="elapsed_s",
        y_key="total_kv_cache_bytes",
        output_path=result_dir / "overlap_same_prompt_total.png",
        title="Total KV cache before/after second request",
        x_label="Elapsed time (s)",
        y_label="Memory size (MiB)",
        group_key="segment",
        vertical_line_x=second_submit_elapsed,
        scatter_rows=sampled_total_rows,
        extra_y_series=[
            ("total_cuda_memory_delta_bytes", CUDA_MEMORY_BASELINE_DELTA_LABEL)
        ],
    )
    write_csv(result_dir / "overlap_same_prompt_total_fit.csv", total_fit_rows)

    node_growth_total_rows = [
        row for row in node_total_rows if row.get("event") != "done"
    ]
    node_total_fit_input_rows = []
    for row in node_growth_total_rows:
        segment = (
            "before_second"
            if float(row["elapsed_s"]) < second_submit_elapsed
            else "after_second"
        )
        node_total_fit_input_rows.append(
            {
                **row,
                "segment": segment,
                "node_segment_group": f"{row.get('node_id')}:{segment}",
            }
        )
    sampled_node_total_rows = sample_rows_by_x_interval(
        node_total_fit_input_rows,
        x_key="elapsed_s",
        interval=OVERLAP_TIME_SAMPLE_INTERVAL_S,
        group_key="node_segment_group",
    )
    per_node_total_fit_rows = plot_per_node_scatter_with_fit(
        node_total_fit_input_rows,
        output_dir=result_dir,
        filename_prefix="overlap_same_prompt_total_per_node",
        x_key="elapsed_s",
        y_key="node_total_kv_cache_bytes",
        title_prefix="Total KV cache before/after second request",
        x_label="Elapsed time (s)",
        group_key="segment",
        vertical_line_x=second_submit_elapsed,
        scatter_rows=sampled_node_total_rows,
        extra_y_series=[
            ("cuda_memory_delta_from_start_bytes", CUDA_MEMORY_BASELINE_DELTA_LABEL),
        ],
    )
    write_csv(
        result_dir / "overlap_same_prompt_total_per_node_fit.csv",
        per_node_total_fit_rows,
    )


def run_kv_cache_experiments(client: PipelineDebugClient) -> None:
    result_dir = make_result_dir()
    print(
        "\nKV cache experiments:\n"
        "\n"
        "Experiment descriptions:\n"
        "  1. Prefill length sweep: send prompts with exact token lengths such as "
        "8/16/32/... and collect the KV cache size immediately after prefill "
        "finishes on every pipeline shard. This checks whether KV cache size is "
        "linear in input token length.\n"
        "  2. Decode growth sweep: run one request for many generated tokens, "
        "collect every decode-step KV cache snapshot for fitting, and only draw "
        "the configured output-token checkpoints on the figure. This checks "
        "whether each generated token adds a constant KV cache increment.\n"
        "  3. Sequential same-prompt consistency: submit two identical prompts "
        "one after another with max_new_tokens=1, compare their post-prefill KV "
        "cache sizes, then repeat for several input lengths. This checks whether "
        "per-request clear/reuse leaves no residual state.\n"
        "  4. Overlapped same-prompt timeline: submit a second identical prompt "
        "after a delay while the first request is still decoding, then record "
        "per-request and total KV cache over time. This checks how the total "
        "growth slope changes before and after concurrent requests overlap.\n"
        "  5. Run all experiments above in order and save all CSV/plots under "
        "one timestamped result directory.\n"
        "\n"
        "Select an experiment option:\n"
        "  1. prefill KV cache vs input token length\n"
        "  2. decode KV cache vs output token length\n"
        "  3. sequential same-prompt prefill consistency\n"
        "  4. overlapped same-prompt KV cache over time\n"
        "  5. run all experiments\n"
    )
    choice = input("Experiment: ").strip()
    if choice == "1":
        run_prefill_kv_experiment(client, result_dir)
    elif choice == "2":
        run_decode_kv_experiment(client, result_dir)
    elif choice == "3":
        run_sequential_same_prompt_experiment(client, result_dir)
    elif choice == "4":
        run_overlap_same_prompt_experiment(client, result_dir)
    elif choice == "5":
        run_prefill_kv_experiment(client, result_dir)
        run_decode_kv_experiment(client, result_dir)
        run_sequential_same_prompt_experiment(client, result_dir)
        run_overlap_same_prompt_experiment(client, result_dir)
    else:
        print("Unknown experiment.")
    print(f"[TEST] results directory: {result_dir}")


def collect_shard_forward_reports(
    client: PipelineDebugClient,
    client_request_ids: list[str],
    timeout_s: float,
) -> tuple[list[dict], list[tuple[str, str]]]:
    expected_node_ids = {node.node_id for node in client.nodes}
    expected_pairs = {
        (client_request_id, node_id)
        for client_request_id in client_request_ids
        for node_id in expected_node_ids
    }
    reports_by_pair: dict[tuple[str, str], dict] = {}
    held_events: list[dict] = []
    deadline = time.time() + timeout_s

    try:
        while time.time() < deadline and set(reports_by_pair) != expected_pairs:
            event = client.receive_event(
                timeout_s=max(0.05, min(0.5, deadline - time.time()))
            )
            if event is None:
                continue
            if (
                event.get("type") == SHARD_FORWARD_REPORT
                and event.get("client_request_id") in client_request_ids
                and event.get("event") == "shard_forward_measurement"
                and event.get("phase") == "prefill"
                and int(event.get("step") or 0) == 0
            ):
                key = (str(event.get("client_request_id")), str(event.get("node_id")))
                reports_by_pair[key] = event
            else:
                held_events.append(event)
    finally:
        client._event_backlog[:0] = held_events

    missing = sorted(expected_pairs - set(reports_by_pair))
    reports = list(reports_by_pair.values())
    reports.sort(
        key=lambda row: (
            client_request_ids.index(str(row.get("client_request_id"))),
            int(row.get("shards_start") or 0),
        )
    )
    return reports, missing


def collect_forward_reports_and_done(
    client: PipelineDebugClient,
    client_request_ids: list[str],
    expected_phase_steps: list[tuple[str, int]],
    process_started_perf: float,
    timeout_s: float,
) -> tuple[
    list[dict],
    list[dict],
    dict[str, dict],
    list[tuple[str, str, str, int]],
    list[str],
    list[str],
]:
    expected_node_ids = {node.node_id for node in client.nodes}
    expected_report_keys = {
        (client_request_id, node_id, phase, step)
        for client_request_id in client_request_ids
        for node_id in expected_node_ids
        for phase, step in expected_phase_steps
    }
    expected_done_ids = set(client_request_ids)
    expected_ack_ids = set(client_request_ids)
    reports_by_key: dict[tuple[str, str, str, int], dict] = {}
    done_by_client_id: dict[str, dict] = {}
    ack_by_client_id: dict[str, dict] = {}
    held_events: list[dict] = []
    deadline = time.time() + timeout_s

    try:
        while time.time() < deadline:
            if (
                set(ack_by_client_id) == expected_ack_ids
                and set(reports_by_key) == expected_report_keys
                and set(done_by_client_id) == expected_done_ids
            ):
                break
            event = client.receive_event(
                timeout_s=max(0.05, min(0.5, deadline - time.time()))
            )
            if event is None:
                continue
            event_type = event.get("type")
            client_request_id = str(event.get("client_request_id"))
            if event_type == USER_REQUEST_ACK and client_request_id in expected_ack_ids:
                ack_by_client_id[client_request_id] = event
                continue
            if (
                event_type == SHARD_FORWARD_REPORT
                and client_request_id in expected_done_ids
                and event.get("event") == "shard_forward_measurement"
            ):
                phase = str(event.get("phase"))
                step = int(event.get("step") or 0)
                key = (
                    client_request_id,
                    str(event.get("node_id")),
                    phase,
                    step,
                )
                if key in expected_report_keys:
                    reports_by_key[key] = event
                continue
            if event_type == PIPELINE_DONE_REPORT and client_request_id in expected_done_ids:
                event["collector_done_received_elapsed_ms"] = (
                    time.perf_counter() - process_started_perf
                ) * 1000.0
                done_by_client_id[client_request_id] = event
                continue
            held_events.append(event)
    finally:
        client._event_backlog[:0] = held_events

    phase_step_order = {
        phase_step: index for index, phase_step in enumerate(expected_phase_steps)
    }
    reports = list(reports_by_key.values())
    reports.sort(
        key=lambda row: (
            client_request_ids.index(str(row.get("client_request_id"))),
            phase_step_order.get((str(row.get("phase")), int(row.get("step") or 0)), 999),
            int(row.get("shards_start") or 0),
        )
    )
    done_reports = list(done_by_client_id.values())
    done_reports.sort(
        key=lambda row: client_request_ids.index(str(row.get("client_request_id")))
    )
    missing_reports = sorted(expected_report_keys - set(reports_by_key))
    missing_done = sorted(expected_done_ids - set(done_by_client_id))
    missing_ack = sorted(expected_ack_ids - set(ack_by_client_id))
    return reports, done_reports, ack_by_client_id, missing_reports, missing_done, missing_ack


def wait_for_warmup_batch(
    client: PipelineDebugClient,
    owner_index: int,
    prompt: str,
    request_count: int,
    max_new_tokens: int,
    timeout_s: float,
    trace_label_prefix: str,
    prompts: list[str] | None = None,
    input_ids_list: list[torch.Tensor] | None = None,
    ignore_eos_for_measurement: bool = False,
    require_completion: bool = False,
) -> None:
    warmup_prompts = prompts if prompts is not None else [prompt] * request_count
    warmup_client_ids = client.submit_burst_requests(
        owner_index=owner_index,
        prompts=warmup_prompts,
        max_new_tokens=max_new_tokens,
        input_ids_list=input_ids_list,
        telemetry_only=True,
        trace_label_prefix=trace_label_prefix,
        ignore_eos_for_measurement=ignore_eos_for_measurement,
    )
    for warmup_client_id in warmup_client_ids:
        client.wait_for_ack(warmup_client_id, timeout_s=timeout_s)
    missing_warmups = []
    for warmup_client_id in warmup_client_ids:
        warmup_done = client.wait_for_done(warmup_client_id, timeout_s=timeout_s)
        if warmup_done is None:
            missing_warmups.append(warmup_client_id)
    if missing_warmups:
        if require_completion:
            raise TimeoutError(
                "[ERROR] warm-up requests did not report done before timeout: "
                f"{missing_warmups}"
            )
        print(
            f"[WARNING] warm-up requests did not report done before timeout: "
            f"{missing_warmups}; continuing with measured run."
        )
    else:
        print(f"[TEST] warm-up requests completed: {warmup_client_ids}.")


def collect_concurrency_sweep_events(
    client: PipelineDebugClient,
    client_request_ids: list[str],
    expected_phase_steps: list[tuple[str, int]],
    process_started_perf: float,
    timeout_s: float,
) -> dict:
    expected_node_ids = {node.node_id for node in client.nodes}
    expected_report_keys = {
        (client_request_id, node_id, phase, step)
        for client_request_id in client_request_ids
        for node_id in expected_node_ids
        for phase, step in expected_phase_steps
    }
    expected_client_ids = set(client_request_ids)
    reports_by_key: dict[tuple[str, str, str, int], dict] = {}
    done_by_client_id: dict[str, dict] = {}
    ack_by_client_id: dict[str, dict] = {}
    admission_by_client_id: dict[str, dict] = {}
    first_token_by_client_id: dict[str, dict] = {}
    held_events: list[dict] = []
    deadline = time.time() + timeout_s

    try:
        while time.time() < deadline:
            if (
                set(ack_by_client_id) == expected_client_ids
                and set(admission_by_client_id) == expected_client_ids
                and set(first_token_by_client_id) == expected_client_ids
                and set(reports_by_key) == expected_report_keys
                and set(done_by_client_id) == expected_client_ids
            ):
                break
            event = client.receive_event(
                timeout_s=max(0.05, min(0.5, deadline - time.time()))
            )
            if event is None:
                continue
            received_elapsed_ms = (time.perf_counter() - process_started_perf) * 1000.0
            event_type = event.get("type")
            client_request_id = str(event.get("client_request_id"))

            if event_type == USER_REQUEST_ACK and client_request_id in expected_client_ids:
                event["collector_ack_received_elapsed_ms"] = received_elapsed_ms
                ack_by_client_id[client_request_id] = event
                continue

            if (
                event_type == PIPELINE_ADMISSION_REPORT
                and client_request_id in expected_client_ids
            ):
                event["collector_admission_received_elapsed_ms"] = received_elapsed_ms
                admission_by_client_id[client_request_id] = event
                continue

            if (
                event_type == PIPELINE_TOKEN_REPORT
                and client_request_id in expected_client_ids
            ):
                event["collector_token_received_elapsed_ms"] = received_elapsed_ms
                if (
                    bool(event.get("is_first_token"))
                    and client_request_id not in first_token_by_client_id
                ):
                    event["collector_first_token_received_elapsed_ms"] = (
                        received_elapsed_ms
                    )
                    if event.get("owner_local_ttft_ms") is not None:
                        event["first_token_elapsed_preferred_ms"] = event.get(
                            "owner_local_ttft_ms"
                        )
                        event["first_token_elapsed_source"] = "owner_local_ttft_ms"
                    else:
                        event["first_token_elapsed_preferred_ms"] = received_elapsed_ms
                        event["first_token_elapsed_source"] = (
                            "collector_first_token_received_elapsed_ms"
                        )
                    first_token_by_client_id[client_request_id] = event
                continue

            if (
                event_type == SHARD_FORWARD_REPORT
                and client_request_id in expected_client_ids
                and event.get("event") == "shard_forward_measurement"
            ):
                phase = str(event.get("phase"))
                step = int(event.get("step") or 0)
                key = (
                    client_request_id,
                    str(event.get("node_id")),
                    phase,
                    step,
                )
                if key in expected_report_keys:
                    event["collector_received_elapsed_ms"] = received_elapsed_ms
                    reports_by_key[key] = event
                continue

            if event_type == PIPELINE_DONE_REPORT and client_request_id in expected_client_ids:
                event["collector_done_received_elapsed_ms"] = received_elapsed_ms
                done_by_client_id[client_request_id] = event
                continue

            held_events.append(event)
    finally:
        client._event_backlog[:0] = held_events

    phase_step_order = {
        phase_step: index for index, phase_step in enumerate(expected_phase_steps)
    }
    reports = list(reports_by_key.values())
    reports.sort(
        key=lambda row: (
            client_request_ids.index(str(row.get("client_request_id"))),
            phase_step_order.get((str(row.get("phase")), int(row.get("step") or 0)), 999),
            int(row.get("shards_start") or 0),
        )
    )
    done_reports = list(done_by_client_id.values())
    done_reports.sort(
        key=lambda row: client_request_ids.index(str(row.get("client_request_id")))
    )
    return {
        "reports": reports,
        "done_reports": done_reports,
        "ack_by_client_id": ack_by_client_id,
        "admission_by_client_id": admission_by_client_id,
        "first_token_by_client_id": first_token_by_client_id,
        "missing_ack": sorted(expected_client_ids - set(ack_by_client_id)),
        "missing_admission": sorted(expected_client_ids - set(admission_by_client_id)),
        "missing_first_token": sorted(expected_client_ids - set(first_token_by_client_id)),
        "missing_reports": sorted(expected_report_keys - set(reports_by_key)),
        "missing_done": sorted(expected_client_ids - set(done_by_client_id)),
    }


def add_run_memory_baselines(forward_rows: list[dict]) -> None:
    allocated_baseline_by_node: dict[str, int] = {}
    mem_info_used_baseline_by_node: dict[str, int] = {}
    for row in forward_rows:
        node_id = str(row.get("node_id"))
        allocated_before = row.get("cuda_memory_before_bytes")
        if allocated_before is not None:
            allocated_baseline_by_node[node_id] = min(
                allocated_baseline_by_node.get(node_id, int(allocated_before)),
                int(allocated_before),
            )
        mem_info_used_before = row.get("cuda_mem_get_info_used_before_bytes")
        if mem_info_used_before is not None:
            mem_info_used_baseline_by_node[node_id] = min(
                mem_info_used_baseline_by_node.get(node_id, int(mem_info_used_before)),
                int(mem_info_used_before),
            )

    for row in forward_rows:
        node_id = str(row.get("node_id"))
        allocated_after = row.get("cuda_memory_after_bytes")
        allocated_baseline = allocated_baseline_by_node.get(node_id)
        allocated_delta = (
            int(allocated_after) - allocated_baseline
            if allocated_after is not None and allocated_baseline is not None
            else None
        )
        row["cuda_memory_allocated_run_baseline_bytes"] = allocated_baseline
        row["cuda_memory_allocated_run_delta_bytes"] = allocated_delta
        row["cuda_memory_allocated_run_delta_mib"] = bytes_to_mib(allocated_delta)

        mem_info_used_after = row.get("cuda_mem_get_info_used_after_bytes")
        mem_info_used_baseline = mem_info_used_baseline_by_node.get(node_id)
        mem_info_used_delta = (
            int(mem_info_used_after) - mem_info_used_baseline
            if mem_info_used_after is not None and mem_info_used_baseline is not None
            else None
        )
        row["cuda_mem_get_info_used_run_baseline_bytes"] = mem_info_used_baseline
        row["cuda_mem_get_info_used_run_delta_bytes"] = mem_info_used_delta
        row["cuda_mem_get_info_used_run_delta_mib"] = bytes_to_mib(mem_info_used_delta)


def summarize_memory_peaks(
    forward_rows: list[dict],
    done_rows: list[dict],
    client_nodes: list[PipelineNodeSpec],
) -> tuple[list[dict], dict]:
    latest_kv_by_node_request: dict[tuple[str, str], int] = {}
    peak_dynamic_by_node: dict[str, int] = {node.node_id: 0 for node in client_nodes}
    peak_alloc_delta_by_node: dict[str, int | None] = {
        node.node_id: None for node in client_nodes
    }
    peak_mem_info_delta_by_node: dict[str, int | None] = {
        node.node_id: None for node in client_nodes
    }
    peak_dynamic_total = 0

    timeline_events: list[tuple[float, str, dict]] = []
    # These events are ordered by the collector's telemetry receive time. This
    # is useful for a conservative active-KV estimate, but it is not a synced
    # cross-device execution timeline and should not be mixed with owner-local
    # latency fields.
    for row in forward_rows:
        elapsed = _float_or_none(row.get("collector_received_elapsed_ms"))
        if elapsed is not None:
            timeline_events.append((elapsed, "forward", row))
    for row in done_rows:
        elapsed = _float_or_none(row.get("collector_done_received_elapsed_ms"))
        if elapsed is not None:
            timeline_events.append((elapsed, "done", row))
    timeline_events.sort(key=lambda item: item[0])

    for _, event_type, row in timeline_events:
        if event_type == "done":
            client_request_id = str(row.get("client_request_id"))
            for node in client_nodes:
                latest_kv_by_node_request[(node.node_id, client_request_id)] = 0
            continue

        node_id = str(row.get("node_id"))
        client_request_id = str(row.get("client_request_id"))
        latest_kv_by_node_request[(node_id, client_request_id)] = int(
            row.get("kv_cache_after_bytes") or 0
        )
        node_dynamic_total = sum(
            value
            for (current_node_id, _), value in latest_kv_by_node_request.items()
            if current_node_id == node_id
        )
        peak_dynamic_by_node[node_id] = max(
            peak_dynamic_by_node.get(node_id, 0),
            node_dynamic_total,
        )
        peak_dynamic_total = max(
            peak_dynamic_total,
            sum(latest_kv_by_node_request.values()),
        )

        allocated_delta = row.get("cuda_memory_allocated_run_delta_bytes")
        if allocated_delta is not None:
            previous = peak_alloc_delta_by_node.get(node_id)
            peak_alloc_delta_by_node[node_id] = max(
                int(allocated_delta),
                int(previous) if previous is not None else int(allocated_delta),
            )
        mem_info_delta = row.get("cuda_mem_get_info_used_run_delta_bytes")
        if mem_info_delta is not None:
            previous = peak_mem_info_delta_by_node.get(node_id)
            peak_mem_info_delta_by_node[node_id] = max(
                int(mem_info_delta),
                int(previous) if previous is not None else int(mem_info_delta),
            )

    node_spec_by_id = {node.node_id: node for node in client_nodes}
    memory_rows = []
    for node_id in sorted(peak_dynamic_by_node, key=lambda item: node_spec_by_id[item].shards_start):
        node = node_spec_by_id[node_id]
        allocated_delta = peak_alloc_delta_by_node.get(node_id)
        mem_info_delta = peak_mem_info_delta_by_node.get(node_id)
        dynamic_peak = peak_dynamic_by_node[node_id]
        memory_rows.append(
            {
                "node_id": node_id,
                "node_addr": node.node_addr,
                "shards_start": node.shards_start,
                "shards_end": node.shards_end,
                "peak_ordering_basis": "collector_telemetry_receive_order",
                "peak_dynamiccache_total_bytes": dynamic_peak,
                "peak_dynamiccache_total_mib": bytes_to_mib(dynamic_peak),
                "peak_cuda_memory_allocated_run_delta_bytes": allocated_delta,
                "peak_cuda_memory_allocated_run_delta_mib": bytes_to_mib(allocated_delta),
                "peak_cuda_mem_get_info_used_run_delta_bytes": mem_info_delta,
                "peak_cuda_mem_get_info_used_run_delta_mib": bytes_to_mib(mem_info_delta),
            }
        )

    allocated_values = [
        value for value in peak_alloc_delta_by_node.values() if value is not None
    ]
    mem_info_values = [
        value for value in peak_mem_info_delta_by_node.values() if value is not None
    ]
    memory_summary = {
        "memory_peak_ordering_basis": "collector_telemetry_receive_order",
        "peak_dynamiccache_total_bytes": peak_dynamic_total,
        "peak_dynamiccache_total_mib": bytes_to_mib(peak_dynamic_total),
        "sum_peak_dynamiccache_by_node_bytes": sum(peak_dynamic_by_node.values()),
        "sum_peak_dynamiccache_by_node_mib": bytes_to_mib(
            sum(peak_dynamic_by_node.values())
        ),
        "max_peak_dynamiccache_by_node_bytes": max(peak_dynamic_by_node.values())
        if peak_dynamic_by_node
        else None,
        "max_peak_dynamiccache_by_node_mib": bytes_to_mib(
            max(peak_dynamic_by_node.values()) if peak_dynamic_by_node else None
        ),
        "sum_peak_cuda_memory_allocated_run_delta_bytes": sum(allocated_values)
        if allocated_values
        else None,
        "sum_peak_cuda_memory_allocated_run_delta_mib": bytes_to_mib(
            sum(allocated_values) if allocated_values else None
        ),
        "max_peak_cuda_memory_allocated_run_delta_bytes": max(allocated_values)
        if allocated_values
        else None,
        "max_peak_cuda_memory_allocated_run_delta_mib": bytes_to_mib(
            max(allocated_values) if allocated_values else None
        ),
        "sum_peak_cuda_mem_get_info_used_run_delta_bytes": sum(mem_info_values)
        if mem_info_values
        else None,
        "sum_peak_cuda_mem_get_info_used_run_delta_mib": bytes_to_mib(
            sum(mem_info_values) if mem_info_values else None
        ),
        "max_peak_cuda_mem_get_info_used_run_delta_bytes": max(mem_info_values)
        if mem_info_values
        else None,
        "max_peak_cuda_mem_get_info_used_run_delta_mib": bytes_to_mib(
            max(mem_info_values) if mem_info_values else None
        ),
    }
    return memory_rows, memory_summary


def build_concurrency_sweep_result_rows(
    *,
    experiment_name: str,
    run_index: int,
    repeat_index: int,
    max_active_requests: int,
    input_token_length: int,
    output_token_length: int,
    client: PipelineDebugClient,
    client_request_ids: list[str],
    collected: dict,
    process_started_perf: float,
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    dict,
]:
    expected_phase_steps = [("prefill", 0)] + [
        ("decode", step) for step in range(1, output_token_length)
    ]
    phase_step_order = {
        phase_step: index + 1 for index, phase_step in enumerate(expected_phase_steps)
    }
    request_order_by_client_id = {
        client_request_id: index
        for index, client_request_id in enumerate(client_request_ids, start=1)
    }
    ack_by_client_id = collected["ack_by_client_id"]
    admission_by_client_id = collected["admission_by_client_id"]
    first_token_by_client_id = collected["first_token_by_client_id"]

    request_id_by_client_id = {
        client_request_id: ack.get("request_id")
        for client_request_id, ack in ack_by_client_id.items()
    }

    forward_rows: list[dict] = []
    for report in collected["reports"]:
        client_request_id = str(report.get("client_request_id"))
        phase = str(report.get("phase"))
        step = int(report.get("step") or 0)
        request_order = request_order_by_client_id[client_request_id]
        row = {
            "experiment": experiment_name,
            "run_index": run_index,
            "repeat_index": repeat_index,
            "topology_node_count": client.pipeline_depth,
            "request_count": len(client_request_ids),
            "max_active_requests": max_active_requests,
            "input_token_length": input_token_length,
            "max_new_tokens": output_token_length,
            "request_order": request_order,
            "client_request_id": client_request_id,
            "request_id": request_id_by_client_id.get(client_request_id),
            "node_id": report.get("node_id"),
            "node_addr": report.get("node_addr"),
            "shards_start": report.get("shards_start"),
            "shards_end": report.get("shards_end"),
            "phase": phase,
            "step": step,
            "phase_step_order": phase_step_order.get((phase, step)),
            "started_timestamp": report.get("started_timestamp"),
            "finished_timestamp": report.get("finished_timestamp"),
            "collector_received_elapsed_ms": report.get("collector_received_elapsed_ms"),
            "forward_elapsed_ms": report.get("forward_elapsed_ms"),
            "shard_forward_elapsed_ms": report.get("shard_forward_elapsed_ms"),
            "owner_embedding_elapsed_ms": report.get("owner_embedding_elapsed_ms"),
            "kv_cache_before_bytes": report.get("kv_cache_before_bytes"),
            "kv_cache_after_bytes": report.get("kv_cache_after_bytes"),
            "kv_cache_delta_bytes": report.get("kv_cache_delta_bytes"),
            "kv_cache_delta_mib": bytes_to_mib(report.get("kv_cache_delta_bytes")),
            "cuda_memory_before_bytes": report.get("cuda_memory_before_bytes"),
            "cuda_memory_after_bytes": report.get("cuda_memory_after_bytes"),
            "cuda_memory_forward_delta_bytes": report.get(
                "cuda_memory_forward_delta_bytes"
            ),
            "cuda_memory_forward_delta_mib": bytes_to_mib(
                report.get("cuda_memory_forward_delta_bytes")
            ),
            "cuda_mem_get_info_free_before_bytes": report.get(
                "cuda_mem_get_info_free_before_bytes"
            ),
            "cuda_mem_get_info_total_before_bytes": report.get(
                "cuda_mem_get_info_total_before_bytes"
            ),
            "cuda_mem_get_info_used_before_bytes": report.get(
                "cuda_mem_get_info_used_before_bytes"
            ),
            "cuda_mem_get_info_free_after_bytes": report.get(
                "cuda_mem_get_info_free_after_bytes"
            ),
            "cuda_mem_get_info_total_after_bytes": report.get(
                "cuda_mem_get_info_total_after_bytes"
            ),
            "cuda_mem_get_info_used_after_bytes": report.get(
                "cuda_mem_get_info_used_after_bytes"
            ),
            "cuda_mem_get_info_used_forward_delta_bytes": report.get(
                "cuda_mem_get_info_used_forward_delta_bytes"
            ),
            "cuda_mem_get_info_used_forward_delta_mib": bytes_to_mib(
                report.get("cuda_mem_get_info_used_forward_delta_bytes")
            ),
            "cuda_mem_get_info_free_forward_delta_bytes": report.get(
                "cuda_mem_get_info_free_forward_delta_bytes"
            ),
        }
        forward_rows.append(row)
    add_run_memory_baselines(forward_rows)

    done_rows: list[dict] = []
    for done_report in collected["done_reports"]:
        client_request_id = str(done_report.get("client_request_id"))
        request_order = request_order_by_client_id[client_request_id]
        done_rows.append(
            {
                "experiment": experiment_name,
                "run_index": run_index,
                "repeat_index": repeat_index,
                "topology_node_count": client.pipeline_depth,
                "request_count": len(client_request_ids),
                "max_active_requests": max_active_requests,
                "input_token_length": input_token_length,
                "max_new_tokens": output_token_length,
                "request_order": request_order,
                "client_request_id": client_request_id,
                "request_id": request_id_by_client_id.get(client_request_id),
                "reason": done_report.get("reason"),
                "output_token_count": done_report.get("output_token_count"),
                "ignore_eos_for_measurement": done_report.get(
                    "ignore_eos_for_measurement"
                ),
                "eos_seen": done_report.get("eos_seen"),
                "first_eos_step": done_report.get("first_eos_step"),
                "semantic_output_valid": done_report.get("semantic_output_valid"),
                "collector_done_received_elapsed_ms": done_report.get(
                    "collector_done_received_elapsed_ms"
                ),
                "owner_local_request_elapsed_ms": done_report.get(
                    "owner_local_request_elapsed_ms"
                ),
                "owner_local_batch_elapsed_ms": done_report.get(
                    "owner_local_batch_elapsed_ms"
                ),
                "owner_request_start_timestamp": done_report.get(
                    "owner_request_start_timestamp"
                ),
                "owner_batch_start_timestamp": done_report.get(
                    "owner_batch_start_timestamp"
                ),
                "owner_batch_id": done_report.get("owner_batch_id"),
            }
        )

    request_latency_rows: list[dict] = []
    admission_rows: list[dict] = []
    first_token_rows: list[dict] = []
    for client_request_id in client_request_ids:
        request_order = request_order_by_client_id[client_request_id]
        admission = admission_by_client_id.get(client_request_id, {})
        first_token = first_token_by_client_id.get(client_request_id, {})
        done = next(
            (
                row for row in done_rows
                if row.get("client_request_id") == client_request_id
            ),
            {},
        )
        admission_row = {
            "experiment": experiment_name,
            "run_index": run_index,
            "repeat_index": repeat_index,
            "request_order": request_order,
            "client_request_id": client_request_id,
            "request_id": request_id_by_client_id.get(client_request_id),
            "max_active_requests": max_active_requests,
            "source": admission.get("source"),
            "pending_wait_ms": admission.get("pending_wait_ms"),
            "collector_admission_received_elapsed_ms": admission.get(
                "collector_admission_received_elapsed_ms"
            ),
            "active_request_count": admission.get("active_request_count"),
            "pending_prefill_count": admission.get("pending_prefill_count"),
        }
        first_token_row = {
            "experiment": experiment_name,
            "run_index": run_index,
            "repeat_index": repeat_index,
            "request_order": request_order,
            "client_request_id": client_request_id,
            "request_id": request_id_by_client_id.get(client_request_id),
            "max_active_requests": max_active_requests,
            "first_token_elapsed_preferred_ms": first_token.get(
                "first_token_elapsed_preferred_ms"
            ),
            "first_token_elapsed_source": first_token.get(
                "first_token_elapsed_source"
            ),
            "owner_local_ttft_ms": first_token.get("owner_local_ttft_ms"),
            "owner_local_batch_ttft_ms": first_token.get("owner_local_batch_ttft_ms"),
            "collector_first_token_received_elapsed_ms": first_token.get(
                "collector_first_token_received_elapsed_ms"
            ),
            "collector_token_received_elapsed_ms": first_token.get(
                "collector_token_received_elapsed_ms"
            ),
            "output_token_index": first_token.get("output_token_index"),
            "owner_batch_id": first_token.get("owner_batch_id"),
        }
        request_completion_elapsed_preferred_ms = (
            done.get("owner_local_request_elapsed_ms")
            if done.get("owner_local_request_elapsed_ms") is not None
            else done.get("collector_done_received_elapsed_ms")
        )
        request_complete_elapsed_source = (
            "owner_local_request_elapsed_ms"
            if done.get("owner_local_request_elapsed_ms") is not None
            else (
                "collector_done_received_elapsed_ms"
                if done.get("collector_done_received_elapsed_ms") is not None
                else None
            )
        )
        latency_row = {
            "experiment": experiment_name,
            "run_index": run_index,
            "repeat_index": repeat_index,
            "topology_node_count": client.pipeline_depth,
            "request_count": len(client_request_ids),
            "max_active_requests": max_active_requests,
            "input_token_length": input_token_length,
            "max_new_tokens": output_token_length,
            "request_order": request_order,
            "client_request_id": client_request_id,
            "request_id": request_id_by_client_id.get(client_request_id),
            "pending_wait_ms": admission.get("pending_wait_ms"),
            "first_token_elapsed_preferred_ms": first_token.get(
                "first_token_elapsed_preferred_ms"
            ),
            "first_token_elapsed_source": first_token.get(
                "first_token_elapsed_source"
            ),
            "owner_local_ttft_ms": first_token.get("owner_local_ttft_ms"),
            "collector_first_token_received_elapsed_ms": first_token.get(
                "collector_first_token_received_elapsed_ms"
            ),
            "request_completion_elapsed_preferred_ms": (
                request_completion_elapsed_preferred_ms
            ),
            "request_completion_elapsed_source": request_complete_elapsed_source,
            "owner_local_request_elapsed_ms": done.get(
                "owner_local_request_elapsed_ms"
            ),
            "owner_local_batch_elapsed_ms": done.get("owner_local_batch_elapsed_ms"),
            "collector_done_received_elapsed_ms": done.get(
                "collector_done_received_elapsed_ms"
            ),
            "output_token_count": done.get("output_token_count"),
        }
        admission_rows.append(admission_row)
        first_token_rows.append(first_token_row)
        request_latency_rows.append(latency_row)

    owner_batch_elapsed_values = [
        float(row["owner_local_batch_elapsed_ms"])
        for row in done_rows
        if row.get("owner_local_batch_elapsed_ms") is not None
    ]
    done_received_elapsed_values = [
        float(row["collector_done_received_elapsed_ms"])
        for row in done_rows
        if row.get("collector_done_received_elapsed_ms") is not None
    ]
    if done_rows and not collected["missing_done"] and owner_batch_elapsed_values:
        total_elapsed_ms = max(owner_batch_elapsed_values)
        total_elapsed_source = "owner_local_batch_elapsed_ms"
    elif done_rows and not collected["missing_done"] and done_received_elapsed_values:
        total_elapsed_ms = max(done_received_elapsed_values)
        total_elapsed_source = "collector_done_received_elapsed_ms"
    else:
        total_elapsed_ms = None
        total_elapsed_source = None
    critical_path_analysis = analyze_forward_critical_path(forward_rows)
    inference_time = critical_path_analysis["inference_time"]
    communication_and_noncompute_time = (
        total_elapsed_ms - inference_time
        if total_elapsed_ms is not None and inference_time is not None
        else None
    )

    stage_rows: list[dict] = []
    for node in client.nodes:
        node_forward_values = [
            _float_or_none(row.get("forward_elapsed_ms")) or 0.0
            for row in forward_rows
            if str(row.get("node_id")) == node.node_id
        ]
        busy_ms = sum(node_forward_values)
        utilization = (
            busy_ms / total_elapsed_ms
            if total_elapsed_ms is not None and total_elapsed_ms > 0
            else None
        )
        stage_rows.append(
            {
                "experiment": experiment_name,
                "run_index": run_index,
                "repeat_index": repeat_index,
                "topology_node_count": client.pipeline_depth,
                "request_count": len(client_request_ids),
                "max_active_requests": max_active_requests,
                "input_token_length": input_token_length,
                "max_new_tokens": output_token_length,
                "node_id": node.node_id,
                "node_addr": node.node_addr,
                "shards_start": node.shards_start,
                "shards_end": node.shards_end,
                "stage_forward_busy_ms": busy_ms,
                "total_complete_time_preferred_ms": total_elapsed_ms,
                "total_complete_time_ms": total_elapsed_ms,
                "total_complete_time_source": total_elapsed_source,
                "stage_time_utilization": utilization,
                "stage_time_utilization_percent": (
                    utilization * 100.0 if utilization is not None else None
                ),
            }
        )

    memory_rows, memory_summary = summarize_memory_peaks(
        forward_rows,
        done_rows,
        client.nodes,
    )
    for row in memory_rows:
        row.update(
            {
                "experiment": experiment_name,
                "run_index": run_index,
                "repeat_index": repeat_index,
                "topology_node_count": client.pipeline_depth,
                "request_count": len(client_request_ids),
                "max_active_requests": max_active_requests,
                "input_token_length": input_token_length,
                "max_new_tokens": output_token_length,
            }
        )

    request_completion_values = [
        value for value in (
            _float_or_none(row.get("request_completion_elapsed_preferred_ms"))
            for row in request_latency_rows
        )
        if value is not None
    ]
    first_token_values = [
        value for value in (
            _float_or_none(row.get("first_token_elapsed_preferred_ms"))
            for row in request_latency_rows
        )
        if value is not None
    ]
    pending_values = [
        value for value in (
            _float_or_none(row.get("pending_wait_ms")) for row in request_latency_rows
        )
        if value is not None
    ]
    stage_util_values = [
        value for value in (
            _float_or_none(row.get("stage_time_utilization")) for row in stage_rows
        )
        if value is not None
    ]

    summary = {
        "experiment": experiment_name,
        "run_index": run_index,
        "repeat_index": repeat_index,
        "topology_node_count": client.pipeline_depth,
        "request_count": len(client_request_ids),
        "max_active_requests": max_active_requests,
        "input_token_length": input_token_length,
        "max_new_tokens": output_token_length,
        "time_unit": "ms",
        "total_complete_time_preferred_ms": total_elapsed_ms,
        "total_complete_time_ms": total_elapsed_ms,
        "total_complete_time_source": total_elapsed_source,
        "throughput_requests_per_s": (
            len(client_request_ids) / (total_elapsed_ms / 1000.0)
            if total_elapsed_ms is not None and total_elapsed_ms > 0
            else None
        ),
        "request_completion_latency_preferred_mean_ms": mean_or_none(
            request_completion_values
        ),
        "request_completion_latency_preferred_p50_ms": percentile(
            request_completion_values, 50
        ),
        "request_completion_latency_preferred_p95_ms": percentile(
            request_completion_values, 95
        ),
        "request_completion_latency_preferred_max_ms": (
            max(request_completion_values) if request_completion_values else None
        ),
        "first_token_latency_preferred_mean_ms": mean_or_none(first_token_values),
        "first_token_latency_preferred_p50_ms": percentile(first_token_values, 50),
        "first_token_latency_preferred_p95_ms": percentile(first_token_values, 95),
        "first_token_latency_preferred_max_ms": (
            max(first_token_values) if first_token_values else None
        ),
        "pending_wait_mean_ms": mean_or_none(pending_values),
        "pending_wait_p50_ms": percentile(pending_values, 50),
        "pending_wait_p95_ms": percentile(pending_values, 95),
        "pending_wait_max_ms": max(pending_values) if pending_values else None,
        "stage_time_utilization_mean": mean_or_none(stage_util_values),
        "stage_time_utilization_min": min(stage_util_values) if stage_util_values else None,
        "stage_time_utilization_max": max(stage_util_values) if stage_util_values else None,
        "inference_time_ms": inference_time,
        "communication_and_noncompute_time_ms": communication_and_noncompute_time,
        "critical_path_cycle_detected": critical_path_analysis[
            "critical_path_cycle_detected"
        ],
        "critical_path_node_count": critical_path_analysis[
            "critical_path_node_count"
        ],
        "expected_forward_report_count": (
            len(client_request_ids) * client.pipeline_depth * len(expected_phase_steps)
        ),
        "collected_forward_report_count": len(forward_rows),
        "missing_ack_count": len(collected["missing_ack"]),
        "missing_admission_count": len(collected["missing_admission"]),
        "missing_first_token_count": len(collected["missing_first_token"]),
        "missing_forward_report_count": len(collected["missing_reports"]),
        "missing_done_count": len(collected["missing_done"]),
        "collector_process_elapsed_ms": (
            time.perf_counter() - process_started_perf
        ) * 1000.0,
        **memory_summary,
    }
    return (
        forward_rows,
        done_rows,
        request_latency_rows,
        admission_rows,
        first_token_rows,
        stage_rows,
        memory_rows,
        critical_path_analysis["critical_path_rows"],
        summary,
    )


def run_simultaneous_pair_forward_experiment(client: PipelineDebugClient) -> None:
    """
    Back-to-back 提交两个相同 prompt，收集每个 request 在每个 node 的前向耗时和 KV 增量。

    该实验只采 prefill step=0：max_new_tokens=1 会让请求在首个 token 产生后立即结束，
    因此不会混入后续 decode forward report。
    """

    result_dir = make_result_dir()
    print(
        "\n[TEST] Simultaneous same-prompt shard forward measurement\n"
        "This test sends two identical prompts back-to-back to the first node as owner, "
        "then collects one prefill forward report per request per node.\n"
    )
    ask_telemetry(client)
    owner_index = 0
    owner = client.nodes[owner_index]
    if owner.shards_start != 0:
        print(
            f"[WARNING] node index 0 is configured as first_node_addr but shards_start={owner.shards_start}; "
            "the test will still send to index 0."
        )
    if client.max_active_requests < 2:
        client.max_active_requests = 2
        client.send_config()

    prompt = ask_prompt(DEFAULT_PROMPTS[4])
    timeout_s = float(input("Forward report timeout seconds [120]: ").strip() or "120")
    client.drain_events()

    print("[TEST] running two warm-up requests without measurement...")
    warmup_client_ids = client.submit_burst_requests(
        owner_index=owner_index,
        prompts=[prompt, prompt],
        max_new_tokens=1,
        telemetry_only=True,
        trace_label_prefix="simultaneous_pair_warmup",
    )
    for warmup_client_id in warmup_client_ids:
        client.wait_for_ack(warmup_client_id, timeout_s=timeout_s)
    missing_warmups = []
    for warmup_client_id in warmup_client_ids:
        warmup_done = client.wait_for_done(warmup_client_id, timeout_s=timeout_s)
        if warmup_done is None:
            missing_warmups.append(warmup_client_id)
    if missing_warmups:
        print(
            f"[WARNING] warm-up requests did not report done before timeout: "
            f"{missing_warmups}; continuing with the measured burst."
        )
    else:
        print(f"[TEST] warm-up requests completed: {warmup_client_ids}.")
    print("[TEST] waiting 2 seconds after warm-up for runtime state to settle...")
    time.sleep(2.0)
    client.drain_events()

    client_request_ids = client.submit_burst_requests(
        owner_index=owner_index,
        prompts=[prompt, prompt],
        max_new_tokens=1,
        trace_forward_measurement=True,
        trace_label_prefix="simultaneous_pair",
    )
    ack_by_client_id = {
        client_request_id: client.wait_for_ack(client_request_id, timeout_s=15.0)
        for client_request_id in client_request_ids
    }
    request_id_by_client_id = {
        client_request_id: ack.get("request_id")
        for client_request_id, ack in ack_by_client_id.items()
    }

    reports, missing = collect_shard_forward_reports(
        client,
        client_request_ids=client_request_ids,
        timeout_s=timeout_s,
    )

    summary_rows: list[dict] = []
    memory_rows: list[dict] = []
    request_order_by_client_id = {
        client_request_id: index
        for index, client_request_id in enumerate(client_request_ids, start=1)
    }
    for report in reports:
        client_request_id = str(report.get("client_request_id"))
        kv_delta_bytes = report.get("kv_cache_delta_bytes")
        cuda_delta_bytes = report.get("cuda_memory_forward_delta_bytes")
        row = {
            "experiment": "simultaneous_pair_forward_measurement",
            "request_order": request_order_by_client_id[client_request_id],
            "client_request_id": client_request_id,
            "request_id": request_id_by_client_id.get(client_request_id),
            "node_id": report.get("node_id"),
            "node_addr": report.get("node_addr"),
            "shards_start": report.get("shards_start"),
            "shards_end": report.get("shards_end"),
            "phase": report.get("phase"),
            "step": report.get("step"),
            "input_token_length": ack_by_client_id[client_request_id].get(
                "input_token_length"
            ),
            "started_timestamp": report.get("started_timestamp"),
            "finished_timestamp": report.get("finished_timestamp"),
            "forward_elapsed_ms": report.get("forward_elapsed_ms"),
            "shard_forward_elapsed_ms": report.get("shard_forward_elapsed_ms"),
            "owner_embedding_elapsed_ms": report.get("owner_embedding_elapsed_ms"),
            "kv_cache_before_bytes": report.get("kv_cache_before_bytes"),
            "kv_cache_after_bytes": report.get("kv_cache_after_bytes"),
            "kv_cache_delta_bytes": kv_delta_bytes,
            "kv_cache_delta_mib": bytes_to_mib(kv_delta_bytes),
            "cuda_memory_before_bytes": report.get("cuda_memory_before_bytes"),
            "cuda_memory_after_bytes": report.get("cuda_memory_after_bytes"),
            "cuda_memory_forward_delta_bytes": cuda_delta_bytes,
            "cuda_memory_forward_delta_mib": bytes_to_mib(cuda_delta_bytes),
        }
        summary_rows.append(row)
        for method, value in (
            ("dynamiccache", kv_delta_bytes),
            ("cuda_memory_allocated", cuda_delta_bytes),
        ):
            memory_rows.append(
                {
                    "experiment": "simultaneous_pair_forward_measurement",
                    "request_order": row["request_order"],
                    "client_request_id": client_request_id,
                    "request_id": row["request_id"],
                    "node_id": row["node_id"],
                    "shards_start": row["shards_start"],
                    "shards_end": row["shards_end"],
                    "method": method,
                    "delta_bytes": value,
                    "delta_mib": bytes_to_mib(value),
                }
            )

    write_csv(result_dir / "simultaneous_pair_forward_summary.csv", summary_rows)
    write_csv(result_dir / "simultaneous_pair_forward_memory_long.csv", memory_rows)
    plot_pair_forward_bars(
        summary_rows,
        value_key="forward_elapsed_ms",
        output_path=result_dir / "simultaneous_pair_forward_elapsed_ms.png",
        title="Two same-prompt requests: forward elapsed by node",
        y_label="Elapsed time (ms)",
    )
    plot_pair_forward_bars(
        summary_rows,
        value_key="kv_cache_delta_bytes",
        output_path=result_dir / "simultaneous_pair_dynamiccache_delta_mib.png",
        title="Two same-prompt requests: DynamicCache delta by node",
        y_label="DynamicCache delta (MiB)",
        value_transform=bytes_to_mib,
    )
    plot_pair_forward_bars(
        summary_rows,
        value_key="cuda_memory_forward_delta_bytes",
        output_path=result_dir / "simultaneous_pair_cuda_delta_mib.png",
        title="Two same-prompt requests: cuda memory baseline delta by node",
        y_label="cuda memory baseline delta (MiB)",
        value_transform=bytes_to_mib,
    )

    for client_request_id in client_request_ids:
        client.wait_for_done(client_request_id, timeout_s=30.0)

    if missing:
        print(f"[WARNING] missing forward reports: {missing}")
    print(f"[TEST] collected {len(summary_rows)} shard forward reports.")
    print(f"[TEST] results directory: {result_dir}")


def run_two_round_pipeline_latency_scenario(
    client: PipelineDebugClient,
    result_dir: Path,
    node_count: int,
    request_count: int,
    prompt: str,
    timeout_s: float,
    config_ready_timeout_s: float,
    max_new_tokens: int = 2,
    ignore_eos_for_measurement: bool = True,
    prompts: list[str] | None = None,
    input_ids_list: list[torch.Tensor] | None = None,
    prefix: str | None = None,
    experiment_name: str = "two_round_pipeline_latency",
) -> None:
    if max_new_tokens < 1:
        raise ValueError("[ERROR] max_new_tokens must be >= 1.")
    expected_phase_steps = [("prefill", 0)] + [
        ("decode", step) for step in range(1, max_new_tokens)
    ]
    if prefix is None:
        prefix = (
            f"{node_count}node_{request_count}request_2round"
            if max_new_tokens == 2
            else f"{node_count}node_{request_count}request_maxnew{max_new_tokens}"
        )
    request_prompts = prompts if prompts is not None else [prompt] * request_count
    if len(request_prompts) != request_count:
        raise ValueError("[ERROR] prompts length must match request_count.")
    if input_ids_list is not None and len(input_ids_list) != request_count:
        raise ValueError("[ERROR] input_ids_list length must match request_count.")
    target_input_lengths = (
        [int(input_ids.shape[-1]) for input_ids in input_ids_list]
        if input_ids_list is not None
        else [None] * request_count
    )

    print(
        f"\n[TEST] {node_count} nodes, {request_count} simultaneous requests, "
        f"max_new_tokens={max_new_tokens}"
    )
    config_id = configure_even_split_topology(client, node_count)
    print(
        f"[TEST] waiting up to {config_ready_timeout_s:.1f} seconds for all "
        "nodes to confirm layer loading..."
    )
    client.wait_for_nodes_ready(config_id, timeout_s=config_ready_timeout_s)
    client.drain_events()

    warmup_tokenizer = load_tokenizer()
    warmup_input_ids = build_pipeline_latency_warmup_input_ids(
        tokenizer=warmup_tokenizer,
        actual_prompts=request_prompts,
        actual_input_ids_list=input_ids_list,
    )
    print(
        "[TEST] running one fixed warm-up request without measurement: "
        f"input_tokens={PIPELINE_LATENCY_WARMUP_INPUT_TOKEN_LENGTH}, "
        f"decode_steps={PIPELINE_LATENCY_WARMUP_DECODE_STEPS}."
    )
    wait_for_warmup_batch(
        client=client,
        owner_index=0,
        prompt="",
        request_count=1,
        max_new_tokens=PIPELINE_LATENCY_WARMUP_MAX_NEW_TOKENS,
        timeout_s=timeout_s,
        trace_label_prefix=f"{prefix}_warmup",
        prompts=[""],
        input_ids_list=[warmup_input_ids],
        ignore_eos_for_measurement=True,
        require_completion=True,
    )
    print("[TEST] waiting 2 seconds after warm-up for runtime state to settle...")
    time.sleep(2.0)
    client.drain_events()

    process_started_perf = time.perf_counter()
    client_request_ids = client.submit_burst_requests(
        owner_index=0,
        prompts=request_prompts,
        max_new_tokens=max_new_tokens,
        input_ids_list=input_ids_list,
        trace_forward_measurement=True,
        trace_label_prefix=prefix,
        ignore_eos_for_measurement=ignore_eos_for_measurement,
    )
    (
        reports,
        done_reports,
        ack_by_client_id,
        missing_reports,
        missing_done,
        missing_ack,
    ) = collect_forward_reports_and_done(
        client=client,
        client_request_ids=client_request_ids,
        expected_phase_steps=expected_phase_steps,
        process_started_perf=process_started_perf,
        timeout_s=timeout_s,
    )
    request_id_by_client_id = {
        client_request_id: ack.get("request_id")
        for client_request_id, ack in ack_by_client_id.items()
    }
    actual_input_lengths = [
        ack_by_client_id.get(client_request_id, {}).get("input_token_length")
        for client_request_id in client_request_ids
    ]
    if input_ids_list is not None and actual_input_lengths != target_input_lengths:
        print(
            "[WARNING] owner-reported input token lengths do not match targets: "
            f"target={target_input_lengths}, actual={actual_input_lengths}"
        )

    request_order_by_client_id = {
        client_request_id: index
        for index, client_request_id in enumerate(client_request_ids, start=1)
    }
    phase_step_order = {
        phase_step: index + 1 for index, phase_step in enumerate(expected_phase_steps)
    }
    forward_rows: list[dict] = []
    for report in reports:
        client_request_id = str(report.get("client_request_id"))
        phase = str(report.get("phase"))
        step = int(report.get("step") or 0)
        request_order = request_order_by_client_id[client_request_id]
        forward_rows.append(
            {
                "experiment": experiment_name,
                "topology_node_count": node_count,
                "request_count": request_count,
                "max_active_requests": client.max_active_requests,
                "max_new_tokens": max_new_tokens,
                "request_order": request_order,
                "client_request_id": client_request_id,
                "request_id": request_id_by_client_id.get(client_request_id),
                "node_id": report.get("node_id"),
                "node_addr": report.get("node_addr"),
                "shards_start": report.get("shards_start"),
                "shards_end": report.get("shards_end"),
                "phase": phase,
                "step": step,
                "phase_step_order": phase_step_order.get((phase, step)),
                "input_token_length": ack_by_client_id.get(client_request_id, {}).get(
                    "input_token_length"
                ),
                "target_input_token_length": target_input_lengths[request_order - 1],
                "started_timestamp": report.get("started_timestamp"),
                "finished_timestamp": report.get("finished_timestamp"),
                "forward_elapsed_ms": report.get("forward_elapsed_ms"),
                "shard_forward_elapsed_ms": report.get("shard_forward_elapsed_ms"),
                "owner_embedding_elapsed_ms": report.get("owner_embedding_elapsed_ms"),
                "kv_cache_delta_bytes": report.get("kv_cache_delta_bytes"),
                "kv_cache_delta_mib": bytes_to_mib(report.get("kv_cache_delta_bytes")),
                "cuda_memory_forward_delta_bytes": report.get(
                    "cuda_memory_forward_delta_bytes"
                ),
                "cuda_memory_forward_delta_mib": bytes_to_mib(
                    report.get("cuda_memory_forward_delta_bytes")
                ),
            }
        )

    done_rows: list[dict] = []
    for done_report in done_reports:
        client_request_id = str(done_report.get("client_request_id"))
        request_order = request_order_by_client_id[client_request_id]
        done_rows.append(
            {
                "experiment": experiment_name,
                "topology_node_count": node_count,
                "request_count": request_count,
                "max_active_requests": client.max_active_requests,
                "max_new_tokens": max_new_tokens,
                "request_order": request_order,
                "client_request_id": client_request_id,
                "request_id": request_id_by_client_id.get(client_request_id),
                "target_input_token_length": target_input_lengths[request_order - 1],
                "input_token_length": ack_by_client_id.get(client_request_id, {}).get(
                    "input_token_length"
                ),
                "reason": done_report.get("reason"),
                "output_token_count": done_report.get("output_token_count"),
                "ignore_eos_for_measurement": done_report.get(
                    "ignore_eos_for_measurement"
                ),
                "eos_seen": done_report.get("eos_seen"),
                "first_eos_step": done_report.get("first_eos_step"),
                "semantic_output_valid": done_report.get("semantic_output_valid"),
                "collector_done_received_elapsed_ms": done_report.get(
                    "collector_done_received_elapsed_ms"
                ),
                "owner_local_request_elapsed_ms": done_report.get(
                    "owner_local_request_elapsed_ms"
                ),
                "owner_local_batch_elapsed_ms": done_report.get(
                    "owner_local_batch_elapsed_ms"
                ),
                "owner_request_start_timestamp": done_report.get(
                    "owner_request_start_timestamp"
                ),
                "owner_batch_start_timestamp": done_report.get(
                    "owner_batch_start_timestamp"
                ),
                "owner_batch_id": done_report.get("owner_batch_id"),
            }
        )

    owner_batch_elapsed_values = [
        float(row["owner_local_batch_elapsed_ms"])
        for row in done_rows
        if row.get("owner_local_batch_elapsed_ms") is not None
    ]
    done_received_elapsed_values = [
        float(row["collector_done_received_elapsed_ms"])
        for row in done_rows
        if row.get("collector_done_received_elapsed_ms") is not None
    ]
    if done_rows and not missing_done and owner_batch_elapsed_values:
        total_elapsed_ms = max(owner_batch_elapsed_values)
        total_elapsed_source = "owner_local_batch_elapsed_ms"
    elif done_rows and not missing_done and done_received_elapsed_values:
        total_elapsed_ms = max(done_received_elapsed_values)
        total_elapsed_source = "collector_done_received_elapsed_ms"
    else:
        total_elapsed_ms = None
        total_elapsed_source = None
    critical_path_analysis = analyze_forward_critical_path(forward_rows)
    inference_time = critical_path_analysis["inference_time"]
    communication_and_noncompute_time = (
        total_elapsed_ms - inference_time
        if total_elapsed_ms is not None and inference_time is not None
        else None
    )
    eos_seen_values = [row.get("eos_seen") for row in done_rows]
    first_eos_steps = [row.get("first_eos_step") for row in done_rows]
    expected_report_count = (
        request_count * node_count * len(expected_phase_steps)
    )
    summary = {
        "experiment": experiment_name,
        "topology_node_count": node_count,
        "request_count": request_count,
        "max_active_requests": client.max_active_requests,
        "max_new_tokens": max_new_tokens,
        "time_unit": "ms",
        "total_complete_time_preferred_ms": total_elapsed_ms,
        "total_complete_time": total_elapsed_ms,
        "total_complete_time_source": total_elapsed_source,
        "inference_time": inference_time,
        "communication_and_noncompute_time": communication_and_noncompute_time,
        "total_process_elapsed_ms": total_elapsed_ms,
        "critical_path_cycle_detected": critical_path_analysis[
            "critical_path_cycle_detected"
        ],
        "critical_path_node_count": critical_path_analysis[
            "critical_path_node_count"
        ],
        "ignore_eos_for_measurement": ignore_eos_for_measurement,
        "any_eos_seen": any(bool(value) for value in eos_seen_values),
        "first_eos_steps": first_eos_steps,
        "all_semantic_outputs_valid": all(
            bool(row.get("semantic_output_valid", True)) for row in done_rows
        ),
        "target_input_token_lengths": target_input_lengths,
        "actual_input_token_lengths": actual_input_lengths,
        "expected_forward_report_count": expected_report_count,
        "collected_forward_report_count": len(forward_rows),
        "ack_report_count": len(ack_by_client_id),
        "missing_ack_count": len(missing_ack),
        "missing_forward_report_count": len(missing_reports),
        "done_report_count": len(done_rows),
        "missing_done_count": len(missing_done),
        "missing_ack_client_request_ids": missing_ack,
        "missing_forward_reports": missing_reports,
        "missing_done_client_request_ids": missing_done,
    }

    write_csv(result_dir / f"{prefix}_forward_reports.csv", forward_rows)
    write_csv(result_dir / f"{prefix}_done_reports.csv", done_rows)
    write_csv(
        result_dir / f"{prefix}_critical_path_forward_rows.csv",
        critical_path_analysis["critical_path_rows"],
    )
    write_json(result_dir / f"{prefix}_summary.json", summary)
    plot_menu7_forward_elapsed_lines(
        forward_rows=forward_rows,
        output_dir=result_dir,
        prefix=prefix,
        expected_phase_steps=expected_phase_steps,
        dense_output_axis=max_new_tokens > 64,
    )
    plot_latency_breakdown_pie(
        summary,
        result_dir / f"{prefix}_latency_breakdown_pie.png",
    )
    plot_request_completion_elapsed_barh(
        done_rows,
        result_dir / f"{prefix}_request_completion_elapsed_preferred_barh.png",
    )

    if missing_ack:
        print(f"[WARNING] missing ack reports for {node_count}-node scenario: {missing_ack}")
    if missing_reports:
        print(f"[WARNING] missing forward reports for {node_count}-node scenario: {missing_reports}")
    if missing_done:
        print(f"[WARNING] missing done reports for {node_count}-node scenario: {missing_done}")
    if any(bool(value) for value in eos_seen_values):
        print(
            "[WARNING] EOS was generated during this measurement; "
            "ignore_eos_for_measurement kept decoding until max_new_tokens."
        )
    if communication_and_noncompute_time is not None and communication_and_noncompute_time < 0:
        print(
            "[WARNING] communication_and_noncompute_time is negative; "
            "check critical path assumptions and measurement overhead."
        )
    print(
        f"[TEST] {node_count}-node scenario total elapsed: "
        f"{total_elapsed_ms if total_elapsed_ms is not None else 'N/A'} ms"
    )


def run_two_round_distinct_same_length_custom_scenario(
    client: PipelineDebugClient,
    result_dir: Path,
    custom_max_new_tokens: bool = False,
    allow_single_request: bool = False,
) -> None:
    print(
        "\n[TEST] Custom two-round latency with different requests but equal input token length\n"
        "This scenario builds each request with the profiler-style method: construct "
        "a long text, tokenize it, then slice exact-length input ids. Each request "
        "uses a different fragment order, so the token ids differ while length stays equal.\n"
    )
    node_count = ask_int("Node count", 2, minimum=2, maximum=7)
    request_count = ask_int(
        "Request count",
        1 if allow_single_request else node_count,
        minimum=1 if allow_single_request else 2,
        maximum=7,
    )
    input_token_length = ask_int("Input token length for every request", 64, minimum=8)
    max_new_tokens = (
        ask_int("max_new_tokens", 2, minimum=1)
        if custom_max_new_tokens
        else 2
    )
    timeout_s = float(input("Overall timeout seconds [180]: ").strip() or "180")
    config_ready_timeout_s = float(
        input("Config-ready timeout seconds after topology config [60]: ").strip() or "60"
    )
    ask_telemetry(client)

    tokenizer = load_tokenizer()
    input_ids_list = build_distinct_input_ids_for_same_length(
        tokenizer=tokenizer,
        token_length=input_token_length,
        request_count=request_count,
    )
    prompts = [""] * request_count
    prefix = (
        f"{node_count}node_{request_count}request_"
        f"distinct_same_len{input_token_length}_"
        f"{'2round' if max_new_tokens == 2 else f'maxnew{max_new_tokens}'}"
    )
    print(
        f"[TEST] generated {request_count} distinct input_id sequences; "
        f"target_input_token_length={input_token_length}."
    )
    run_two_round_pipeline_latency_scenario(
        client=client,
        result_dir=result_dir,
        node_count=node_count,
        request_count=request_count,
        prompt="",
        timeout_s=timeout_s,
        config_ready_timeout_s=config_ready_timeout_s,
        max_new_tokens=max_new_tokens,
        prompts=prompts,
        input_ids_list=input_ids_list,
        prefix=prefix,
        experiment_name=(
            "two_round_distinct_same_length_latency"
            if max_new_tokens == 2
            else "custom_new_tokens_distinct_same_length_latency"
        ),
    )


def run_two_round_same_prompt_custom_scenario(
    client: PipelineDebugClient,
    result_dir: Path,
    custom_max_new_tokens: bool = False,
) -> None:
    print(
        "\n[TEST] Custom two-round latency with identical request content\n"
        "This scenario uses the same prompt for every request, matching scenarios 1-8, "
        "but lets you choose node_count and request_count interactively.\n"
    )
    node_count = ask_int("Node count", 2, minimum=2, maximum=7)
    request_count = ask_int("Request count", node_count, minimum=2, maximum=7)
    prompt = ask_prompt(DEFAULT_PROMPTS[4])
    max_new_tokens = (
        ask_int("max_new_tokens", 2, minimum=1)
        if custom_max_new_tokens
        else 2
    )
    timeout_s = float(input("Overall timeout seconds [180]: ").strip() or "180")
    config_ready_timeout_s = float(
        input("Config-ready timeout seconds after topology config [60]: ").strip() or "60"
    )
    ask_telemetry(client)

    prefix = (
        f"{node_count}node_{request_count}request_same_prompt_custom_"
        f"{'2round' if max_new_tokens == 2 else f'maxnew{max_new_tokens}'}"
    )
    run_two_round_pipeline_latency_scenario(
        client=client,
        result_dir=result_dir,
        node_count=node_count,
        request_count=request_count,
        prompt=prompt,
        timeout_s=timeout_s,
        config_ready_timeout_s=config_ready_timeout_s,
        max_new_tokens=max_new_tokens,
        prefix=prefix,
        experiment_name=(
            "two_round_same_prompt_custom_latency"
            if max_new_tokens == 2
            else "custom_new_tokens_same_prompt_latency"
        ),
    )


def run_two_round_pipeline_latency_experiment(client: PipelineDebugClient) -> None:
    result_dir = make_result_dir()
    print(
        "\nTwo-round pipeline latency experiments:\n"
        "  1. 3 nodes, 3 simultaneous requests, max_new_tokens=2\n"
        "  2. 2 nodes, 3 simultaneous requests, max_new_tokens=2; request 3 should wait in pending queue\n"
        "  3. 4 nodes, 3 simultaneous requests, max_new_tokens=2\n"
        "  4. 4 nodes, 4 simultaneous requests, max_new_tokens=2\n"
        "  5. 3 nodes, 4 simultaneous requests, max_new_tokens=2\n"
        "  6. 5 nodes, 4 simultaneous requests, max_new_tokens=2\n"
        "  7. 5 nodes, 5 simultaneous requests, max_new_tokens=2\n"
        "  8. 5 nodes, 6 simultaneous requests, max_new_tokens=2\n"
        "  9. run scenarios 6, 7, and 8\n"
        "  10. run all scenarios\n"
        "  11. custom nodes/requests with different requests but equal input token length\n"
        "  12. custom nodes/requests with identical request content\n"
        "  13. custom different requests with equal input token length, custom max_new_tokens; one request allowed\n"
        "  14. custom identical requests with custom max_new_tokens\n"
    )
    choice = input("Experiment: ").strip()
    if choice == "11":
        run_two_round_distinct_same_length_custom_scenario(client, result_dir)
        print(f"[TEST] results directory: {result_dir}")
        return
    if choice == "12":
        run_two_round_same_prompt_custom_scenario(client, result_dir)
        print(f"[TEST] results directory: {result_dir}")
        return
    if choice == "13":
        run_two_round_distinct_same_length_custom_scenario(
            client,
            result_dir,
            custom_max_new_tokens=True,
            allow_single_request=True,
        )
        print(f"[TEST] results directory: {result_dir}")
        return
    if choice == "14":
        run_two_round_same_prompt_custom_scenario(
            client, result_dir, custom_max_new_tokens=True
        )
        print(f"[TEST] results directory: {result_dir}")
        return

    prompt = ask_prompt(DEFAULT_PROMPTS[4])
    timeout_s = float(input("Overall timeout seconds [180]: ").strip() or "180")
    config_ready_timeout_s = float(
        input("Config-ready timeout seconds after each topology config [60]: ").strip() or "60"
    )
    ask_telemetry(client)

    if choice == "1":
        run_two_round_pipeline_latency_scenario(
            client, result_dir, 3, 3, prompt, timeout_s, config_ready_timeout_s
        )
    elif choice == "2":
        run_two_round_pipeline_latency_scenario(
            client, result_dir, 2, 3, prompt, timeout_s, config_ready_timeout_s
        )
    elif choice == "3":
        run_two_round_pipeline_latency_scenario(
            client, result_dir, 4, 3, prompt, timeout_s, config_ready_timeout_s
        )
    elif choice == "4":
        run_two_round_pipeline_latency_scenario(
            client, result_dir, 4, 4, prompt, timeout_s, config_ready_timeout_s
        )
    elif choice == "5":
        run_two_round_pipeline_latency_scenario(
            client, result_dir, 3, 4, prompt, timeout_s, config_ready_timeout_s
        )
    elif choice == "6":
        run_two_round_pipeline_latency_scenario(
            client, result_dir, 5, 4, prompt, timeout_s, config_ready_timeout_s
        )
    elif choice == "7":
        run_two_round_pipeline_latency_scenario(
            client, result_dir, 5, 5, prompt, timeout_s, config_ready_timeout_s
        )
    elif choice == "8":
        run_two_round_pipeline_latency_scenario(
            client, result_dir, 5, 6, prompt, timeout_s, config_ready_timeout_s
        )
    elif choice == "9":
        run_two_round_pipeline_latency_scenario(
            client, result_dir, 5, 4, prompt, timeout_s, config_ready_timeout_s
        )
        run_two_round_pipeline_latency_scenario(
            client, result_dir, 5, 5, prompt, timeout_s, config_ready_timeout_s
        )
        run_two_round_pipeline_latency_scenario(
            client, result_dir, 5, 6, prompt, timeout_s, config_ready_timeout_s
        )
    elif choice == "10":
        run_two_round_pipeline_latency_scenario(
            client, result_dir, 3, 3, prompt, timeout_s, config_ready_timeout_s
        )
        run_two_round_pipeline_latency_scenario(
            client, result_dir, 2, 3, prompt, timeout_s, config_ready_timeout_s
        )
        run_two_round_pipeline_latency_scenario(
            client, result_dir, 4, 3, prompt, timeout_s, config_ready_timeout_s
        )
        run_two_round_pipeline_latency_scenario(
            client, result_dir, 4, 4, prompt, timeout_s, config_ready_timeout_s
        )
        run_two_round_pipeline_latency_scenario(
            client, result_dir, 3, 4, prompt, timeout_s, config_ready_timeout_s
        )
        run_two_round_pipeline_latency_scenario(
            client, result_dir, 5, 4, prompt, timeout_s, config_ready_timeout_s
        )
        run_two_round_pipeline_latency_scenario(
            client, result_dir, 5, 5, prompt, timeout_s, config_ready_timeout_s
        )
        run_two_round_pipeline_latency_scenario(
            client, result_dir, 5, 6, prompt, timeout_s, config_ready_timeout_s
        )
    else:
        print("Unknown experiment.")
    print(f"[TEST] results directory: {result_dir}")


def run_concurrency_sweep_motivation_experiment(client: PipelineDebugClient) -> None:
    result_dir = make_result_dir()
    experiment_name = "concurrency_sweep_motivation"
    print(
        "\nConcurrency sweep motivation experiment:\n"
        "This experiment fixes a 4-stage Llama 3.2 3B pipeline and submits "
        "12 distinct requests with identical input token length to the first "
        "node. It sweeps max_active_requests to show utilization, latency, "
        "TTFT/pending wait, and memory-pressure tradeoffs.\n"
    )
    use_tegrastats = ask_yes_no(
        "Is this CLI running on a Jetson device and should it record local tegrastats?",
        default=False,
    )
    input_token_length = ask_int(
        "Input token length for every request",
        CONCURRENCY_SWEEP_INPUT_TOKEN_LENGTH,
        minimum=8,
    )
    output_token_length = ask_int(
        "Output token count / max_new_tokens",
        CONCURRENCY_SWEEP_OUTPUT_TOKEN_LENGTH,
        minimum=1,
    )
    repeats = ask_int(
        "Repeat count per max_active_requests",
        CONCURRENCY_SWEEP_REPEATS,
        minimum=1,
    )
    max_active_values = ask_int_list(
        "max_active_requests values",
        CONCURRENCY_SWEEP_MAX_ACTIVE_VALUES,
    )
    max_active_values = sorted({value for value in max_active_values if value >= 1})
    if not max_active_values:
        print("[ERROR] max_active_requests values must contain at least one value >= 1.")
        return
    timeout_default = max(
        300,
        int(output_token_length * CONCURRENCY_SWEEP_REQUEST_COUNT * 2.0),
    )
    timeout_s = float(
        input(f"Overall timeout seconds per run [{timeout_default}]: ").strip()
        or str(timeout_default)
    )
    config_ready_timeout_s = float(
        input("Config-ready timeout seconds [60]: ").strip() or "60"
    )
    ask_telemetry(client)

    if output_token_length > 64:
        print(
            "[TEST] output_token_length > 64: token-step line plots will use dense "
            "style when generated. This experiment's summary plots use "
            "max_active_requests on the x-axis, so no summary plot uses output "
            "token count as a plotted dimension."
        )

    tokenizer = load_tokenizer()
    input_ids_list = build_distinct_input_ids_for_same_length(
        tokenizer=tokenizer,
        token_length=input_token_length,
        request_count=CONCURRENCY_SWEEP_REQUEST_COUNT,
    )
    prompts = [""] * CONCURRENCY_SWEEP_REQUEST_COUNT

    client.nodes = build_even_split_nodes(CONCURRENCY_SWEEP_NODE_COUNT)
    client.pipeline_depth = CONCURRENCY_SWEEP_NODE_COUNT
    client.max_active_requests = CONCURRENCY_SWEEP_NODE_COUNT
    print(
        f"[TEST] applying {CONCURRENCY_SWEEP_NODE_COUNT}-node topology for "
        "concurrency sweep."
    )
    client.show_topology()
    config_id = client.send_config()
    client.wait_for_nodes_ready(config_id, timeout_s=config_ready_timeout_s)
    client.drain_events()

    warmup_input_ids = build_input_ids_for_lengths(
        tokenizer,
        [CONCURRENCY_SWEEP_WARMUP_INPUT_TOKEN_LENGTH],
    )[CONCURRENCY_SWEEP_WARMUP_INPUT_TOKEN_LENGTH]
    print(
        "[TEST] running one warm-up request without measurement: "
        f"input_tokens={CONCURRENCY_SWEEP_WARMUP_INPUT_TOKEN_LENGTH}, "
        f"output_tokens={CONCURRENCY_SWEEP_WARMUP_OUTPUT_TOKEN_LENGTH}."
    )
    wait_for_warmup_batch(
        client=client,
        owner_index=0,
        prompt="",
        request_count=1,
        max_new_tokens=CONCURRENCY_SWEEP_WARMUP_OUTPUT_TOKEN_LENGTH,
        timeout_s=timeout_s,
        trace_label_prefix="concurrency_sweep_warmup",
        prompts=[""],
        input_ids_list=[warmup_input_ids],
        ignore_eos_for_measurement=True,
        require_completion=True,
    )
    print("[TEST] waiting 2 seconds after warm-up for runtime state to settle...")
    time.sleep(2.0)
    client.drain_events()

    tegrastats_logger = (
        TegrastatsLogger(result_dir / "tegrastats_raw.log")
        if use_tegrastats
        else None
    )

    summary_rows: list[dict] = []
    all_forward_rows: list[dict] = []
    all_done_rows: list[dict] = []
    all_request_latency_rows: list[dict] = []
    all_admission_rows: list[dict] = []
    all_first_token_rows: list[dict] = []
    all_stage_rows: list[dict] = []
    all_memory_rows: list[dict] = []
    all_critical_path_rows: list[dict] = []

    run_plan = [
        (max_active, repeat_index)
        for max_active in max_active_values
        for repeat_index in range(1, repeats + 1)
    ]
    random.shuffle(run_plan)
    write_json(
        result_dir / "concurrency_sweep_plan.json",
        {
            "experiment": experiment_name,
            "node_count": CONCURRENCY_SWEEP_NODE_COUNT,
            "request_count": CONCURRENCY_SWEEP_REQUEST_COUNT,
            "input_token_length": input_token_length,
            "output_token_length": output_token_length,
            "max_active_values": max_active_values,
            "repeat_count_per_max_active": repeats,
            "randomized_run_plan": [
                {"max_active_requests": max_active, "repeat_index": repeat_index}
                for max_active, repeat_index in run_plan
            ],
        },
    )

    try:
        if tegrastats_logger is not None:
            tegrastats_logger.start()

        expected_phase_steps = [("prefill", 0)] + [
            ("decode", step) for step in range(1, output_token_length)
        ]
        for run_index, (max_active, repeat_index) in enumerate(run_plan, start=1):
            print(
                f"\n[TEST] concurrency sweep run {run_index}/{len(run_plan)}: "
                f"max_active_requests={max_active}, repeat={repeat_index}/{repeats}"
            )
            client.max_active_requests = max_active
            config_id = client.send_config()
            client.wait_for_nodes_ready(config_id, timeout_s=config_ready_timeout_s)
            client.drain_events()

            prefix = f"sweep_run{run_index:03d}_M{max_active}_rep{repeat_index}"
            process_started_perf = time.perf_counter()
            client_request_ids = client.submit_burst_requests(
                owner_index=0,
                prompts=prompts,
                max_new_tokens=output_token_length,
                input_ids_list=input_ids_list,
                trace_forward_measurement=True,
                trace_label_prefix=prefix,
                ignore_eos_for_measurement=True,
            )
            collected = collect_concurrency_sweep_events(
                client=client,
                client_request_ids=client_request_ids,
                expected_phase_steps=expected_phase_steps,
                process_started_perf=process_started_perf,
                timeout_s=timeout_s,
            )
            (
                forward_rows,
                done_rows,
                request_latency_rows,
                admission_rows,
                first_token_rows,
                stage_rows,
                memory_rows,
                critical_path_rows,
                summary,
            ) = build_concurrency_sweep_result_rows(
                experiment_name=experiment_name,
                run_index=run_index,
                repeat_index=repeat_index,
                max_active_requests=max_active,
                input_token_length=input_token_length,
                output_token_length=output_token_length,
                client=client,
                client_request_ids=client_request_ids,
                collected=collected,
                process_started_perf=process_started_perf,
            )
            summary_rows.append(summary)
            all_forward_rows.extend(forward_rows)
            all_done_rows.extend(done_rows)
            all_request_latency_rows.extend(request_latency_rows)
            all_admission_rows.extend(admission_rows)
            all_first_token_rows.extend(first_token_rows)
            all_stage_rows.extend(stage_rows)
            all_memory_rows.extend(memory_rows)
            for path_row in critical_path_rows:
                path_row.update(
                    {
                        "experiment": experiment_name,
                        "run_index": run_index,
                        "repeat_index": repeat_index,
                        "max_active_requests": max_active,
                    }
                )
            all_critical_path_rows.extend(critical_path_rows)

            print(
                "[TEST] run summary: "
                f"makespan={summary.get('total_complete_time_preferred_ms')} ms, "
                f"throughput={summary.get('throughput_requests_per_s')} req/s, "
                f"stage_util_mean={summary.get('stage_time_utilization_mean')}"
            )
            if (
                summary["missing_ack_count"]
                or summary["missing_admission_count"]
                or summary["missing_first_token_count"]
                or summary["missing_forward_report_count"]
                or summary["missing_done_count"]
            ):
                print(
                    "[WARNING] run has missing telemetry: "
                    f"ack={summary['missing_ack_count']}, "
                    f"admission={summary['missing_admission_count']}, "
                    f"first_token={summary['missing_first_token_count']}, "
                    f"forward={summary['missing_forward_report_count']}, "
                    f"done={summary['missing_done_count']}"
                )
    finally:
        if tegrastats_logger is not None:
            tegrastats_logger.stop()

    write_csv(result_dir / "concurrency_sweep_run_summary.csv", summary_rows)
    write_csv(result_dir / "concurrency_sweep_request_latency.csv", all_request_latency_rows)
    write_csv(result_dir / "concurrency_sweep_admission_reports.csv", all_admission_rows)
    write_csv(result_dir / "concurrency_sweep_first_token_reports.csv", all_first_token_rows)
    write_csv(result_dir / "concurrency_sweep_done_reports.csv", all_done_rows)
    write_csv(result_dir / "concurrency_sweep_forward_reports.csv", all_forward_rows)
    write_csv(result_dir / "concurrency_sweep_stage_time_utilization.csv", all_stage_rows)
    write_csv(result_dir / "concurrency_sweep_memory_peaks_per_node.csv", all_memory_rows)
    write_csv(result_dir / "concurrency_sweep_critical_path_forward_rows.csv", all_critical_path_rows)

    plot_sweep_metric_lines(
        summary_rows,
        [("throughput_requests_per_s", "throughput")],
        result_dir / "concurrency_sweep_throughput.png",
        "Throughput vs max_active_requests",
        "Throughput (requests/s)",
    )
    plot_sweep_metric_lines(
        summary_rows,
        [
            ("total_complete_time_preferred_ms", "makespan"),
            (
                "request_completion_latency_preferred_mean_ms",
                "request completion mean",
            ),
            (
                "request_completion_latency_preferred_p95_ms",
                "request completion P95",
            ),
        ],
        result_dir / "concurrency_sweep_completion_latency.png",
        "Completion latency vs max_active_requests",
        "Latency (ms)",
    )
    plot_sweep_metric_lines(
        summary_rows,
        [
            ("pending_wait_mean_ms", "pending wait mean"),
            ("pending_wait_p95_ms", "pending wait P95"),
            ("first_token_latency_preferred_mean_ms", "first token mean"),
            ("first_token_latency_preferred_p95_ms", "first token P95"),
        ],
        result_dir / "concurrency_sweep_first_token_pending_wait.png",
        "First-token latency and pending wait vs max_active_requests",
        "Latency (ms)",
    )
    plot_sweep_metric_lines(
        summary_rows,
        [
            ("total_complete_time_preferred_ms", "makespan"),
            ("inference_time_ms", "critical-path inference"),
            ("communication_and_noncompute_time_ms", "communication + non-forward"),
        ],
        result_dir / "concurrency_sweep_latency_breakdown_lines.png",
        "Latency breakdown vs max_active_requests",
        "Latency (ms)",
    )
    plot_stage_time_utilization(
        all_stage_rows,
        result_dir / "concurrency_sweep_stage_time_utilization.png",
    )
    plot_sweep_metric_lines(
        summary_rows,
        [
            ("peak_dynamiccache_total_bytes", "collector-ordered total peak"),
            ("max_peak_dynamiccache_by_node_bytes", "max node peak"),
        ],
        result_dir / "concurrency_sweep_dynamiccache_peak_summary.png",
        "DynamicCache peak vs max_active_requests",
        "DynamicCache peak (MiB)",
        value_transform=bytes_to_mib,
    )
    plot_memory_peak_by_node(
        all_memory_rows,
        "peak_dynamiccache_total_bytes",
        result_dir / "concurrency_sweep_dynamiccache_peak_by_node.png",
        "DynamicCache peak by node",
        "DynamicCache peak (MiB)",
    )
    plot_sweep_metric_lines(
        summary_rows,
        [
            (
                "max_peak_cuda_memory_allocated_run_delta_bytes",
                "max node baseline delta",
            ),
            (
                "sum_peak_cuda_memory_allocated_run_delta_bytes",
                "sum of node peak deltas",
            ),
        ],
        result_dir / "concurrency_sweep_cuda_memory_allocated_peak_summary.png",
        "torch.cuda.memory_allocated() baseline delta peak",
        "Memory delta (MiB)",
        value_transform=bytes_to_mib,
    )
    plot_memory_peak_by_node(
        all_memory_rows,
        "peak_cuda_memory_allocated_run_delta_bytes",
        result_dir / "concurrency_sweep_cuda_memory_allocated_peak_by_node.png",
        "torch.cuda.memory_allocated() baseline delta by node",
        "Memory delta (MiB)",
    )
    plot_sweep_metric_lines(
        summary_rows,
        [
            (
                "max_peak_cuda_mem_get_info_used_run_delta_bytes",
                "max node used baseline delta",
            ),
            (
                "sum_peak_cuda_mem_get_info_used_run_delta_bytes",
                "sum of node used peak deltas",
            ),
        ],
        result_dir / "concurrency_sweep_cuda_mem_get_info_peak_summary.png",
        "torch.cuda.mem_get_info() used baseline delta peak",
        "Memory delta (MiB)",
        value_transform=bytes_to_mib,
    )
    plot_memory_peak_by_node(
        all_memory_rows,
        "peak_cuda_mem_get_info_used_run_delta_bytes",
        result_dir / "concurrency_sweep_cuda_mem_get_info_peak_by_node.png",
        "torch.cuda.mem_get_info() used baseline delta by node",
        "Memory delta (MiB)",
    )

    print(f"[TEST] results directory: {result_dir}")


def run_scenario(client: PipelineDebugClient) -> None:
    print(
        "\nScenarios:\n"
        "  1. single request with max_active_requests=1\n"
        "  2. owner is non-first node\n"
        "  3. four concurrent requests\n"
        "  4. queue six requests with max_active_requests=4\n"
        "  5. early stop with small max_new_tokens\n"
    )
    choice = input("Scenario: ").strip()

    if choice == "1":
        client.max_active_requests = 1
        client.send_config()
        client.submit_request(0, DEFAULT_PROMPTS[0], 128)
    elif choice == "2":
        client.max_active_requests = client.pipeline_depth
        client.send_config()
        client.submit_request(1, DEFAULT_PROMPTS[1], 128)
    elif choice == "3":
        client.max_active_requests = client.pipeline_depth
        client.send_config()
        for i in range(4):
            client.submit_request(i % client.pipeline_depth, DEFAULT_PROMPTS[i], 128)
            time.sleep(0.05)
    elif choice == "4":
        client.max_active_requests = client.pipeline_depth
        client.send_config()
        for i in range(6):
            client.submit_request(i % client.pipeline_depth, DEFAULT_PROMPTS[i], 128)
            time.sleep(0.05)
    elif choice == "5":
        client.max_active_requests = client.pipeline_depth
        client.send_config()
        client.submit_request(0, DEFAULT_PROMPTS[0], 8)
    else:
        print("Unknown scenario.")


def menu_loop(client: PipelineDebugClient) -> None:
    while True:
        print(
            "\nPipeline debug menu:\n"
            "  1. send/re-send config\n"
            "  2. submit one request\n"
            "  3. submit multiple requests\n"
            "  4. change max_active_requests and send config\n"
            "  5. run KV cache growth experiments\n"
            "  6. run simultaneous pair forward measurement\n"
            "  7. run two-round pipeline latency experiments\n"
            "  8. run concurrency sweep motivation experiment\n"
            "  9. run predefined Jetson test scenario\n"
            "  10. show topology/config\n"
            "  q. quit\n"
        )
        choice = input("Select: ").strip().lower()
        if choice == "1":
            client.send_config()
        elif choice == "2":
            submit_one(client)
        elif choice == "3":
            submit_many(client)
        elif choice == "4":
            change_max_active(client)
        elif choice == "5":
            run_kv_cache_experiments(client)
        elif choice == "6":
            run_simultaneous_pair_forward_experiment(client)
        elif choice == "7":
            run_two_round_pipeline_latency_experiment(client)
        elif choice == "8":
            run_concurrency_sweep_motivation_experiment(client)
        elif choice == "9":
            run_scenario(client)
        elif choice == "10":
            client.show_topology()
        elif choice in {"q", "quit", "exit"}:
            return
        else:
            print("Unknown choice.")


def main() -> None:
    client = PipelineDebugClient(DEFAULT_NODES)
    client.show_topology()
    if ask_yes_no("Send initial config now?", default=True):
        ask_telemetry(client)
        config_ready_timeout_s = float(
            input("Initial config-ready timeout seconds [60]: ").strip() or "60"
        )
        config_id = client.send_config()
        print(
            f"[TEST] waiting up to {config_ready_timeout_s:.1f} seconds for all "
            "nodes to confirm initial layer loading..."
        )
        client.wait_for_nodes_ready(config_id, timeout_s=config_ready_timeout_s)
        client.drain_events()
    menu_loop(client)


if __name__ == "__main__":
    main()
