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
import math
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

TOKENIZER_PATH = "shards/Llama-3___2-3B-Instruct_float16"
RESULT_ROOT = Path("results") / "pipeline_kv_cache"
DEFAULT_TELEMETRY_HOST = "172.16.0.1"
DEFAULT_TELEMETRY_PORT = 40900
DEFAULT_PREFILL_TOKEN_LENGTHS = [8, 16, 32, 64, 128, 256, 512]
DEFAULT_DECODE_OUTPUT_TOKEN_LENGTHS = [8, 16, 32, 64, 128, 256, 512]
OVERLAP_TIME_SAMPLE_INTERVAL_S = 1.0
PROMPT_FRAGMENT = (
    "Distributed inference splits a language model across multiple edge devices so "
    "that each device processes part of the network while cooperating with the others. "
)


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


DEFAULT_NODES = [
    PipelineNodeSpec("node0", "172.16.0.4", 0, 7),
    PipelineNodeSpec("node1", "172.16.0.5", 7, 14),
    PipelineNodeSpec("node2", "172.16.0.6", 14, 21),
    PipelineNodeSpec("node3", "172.16.0.7", 21, 28),
]
DEFAULT_LAYER_COUNT = DEFAULT_NODES[-1].shards_end

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

    def _build_user_request_payload(
        self,
        prompt: str,
        max_new_tokens: int,
        input_ids: torch.Tensor | None = None,
        trace_kv_cache: bool = False,
        trace_forward_measurement: bool = False,
        telemetry_only: bool = False,
        trace_label: str = "",
    ) -> tuple[str, dict]:
        client_request_id = self._new_client_request_id()
        payload = {
            "type": USER_REQUEST,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "client_request_id": client_request_id,
        }
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

    def build_config(self, index: int) -> dict:
        node = self.nodes[index]
        next_node = self.nodes[(index + 1) % self.pipeline_depth]
        first_node = self.nodes[0]
        return {
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

    def send_config(self) -> None:
        """Send current config to all pipeline nodes."""

        for index, node in enumerate(self.nodes):
            config = self.build_config(index)
            self._json_socket(node.config_addr).send_json(config)
            print(
                f"[CONFIG] sent {node.node_id} {node.ip}: "
                f"layers {node.shards_start}~{node.shards_end}, "
                f"dst={config['dst_addr']}, can_receive={node.can_receive_user_request}"
            )
        print("[CONFIG] all configs sent; keep this CLI alive while nodes receive them.")

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
        trace_forward_measurement: bool = False,
        telemetry_only: bool = False,
        trace_label_prefix: str = "burst",
    ) -> list[str]:
        """Send several requests as one batch so the owner enqueues them in one loop turn."""

        owner = self.nodes[owner_index]
        payloads: list[tuple[str, dict]] = []
        for index, prompt in enumerate(prompts, start=1):
            payloads.append(
                self._build_user_request_payload(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    trace_forward_measurement=trace_forward_measurement,
                    telemetry_only=telemetry_only,
                    trace_label=f"{trace_label_prefix}_request{index}",
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


def ask_int(prompt: str, default: int, minimum: int | None = None) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        value = default if not raw else int(raw)
        if minimum is None or value >= minimum:
            return value
        print(f"Value must be >= {minimum}.")


def ask_node(client: PipelineDebugClient, prompt: str = "Choose owner node") -> int:
    client.show_topology()
    return ask_int(prompt, 0, minimum=0) % client.pipeline_depth


def build_even_split_nodes(node_count: int) -> list[PipelineNodeSpec]:
    if node_count < 1 or node_count > len(DEFAULT_NODES):
        raise ValueError(f"[ERROR] node_count must be in [1, {len(DEFAULT_NODES)}].")

    nodes: list[PipelineNodeSpec] = []
    for index, base_node in enumerate(DEFAULT_NODES[:node_count]):
        start = (DEFAULT_LAYER_COUNT * index) // node_count
        end = (DEFAULT_LAYER_COUNT * (index + 1)) // node_count
        nodes.append(
            PipelineNodeSpec(
                node_id=base_node.node_id,
                ip=base_node.ip,
                shards_start=start,
                shards_end=end,
                can_receive_user_request=True,
                config_port=base_node.config_port,
                data_port=base_node.data_port,
            )
        )
    return nodes


def configure_even_split_topology(client: PipelineDebugClient, node_count: int) -> None:
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
    client.send_config()


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
                event["done_received_elapsed_ms"] = (
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
) -> None:
    warmup_client_ids = client.submit_burst_requests(
        owner_index=owner_index,
        prompts=[prompt] * request_count,
        max_new_tokens=max_new_tokens,
        telemetry_only=True,
        trace_label_prefix=trace_label_prefix,
    )
    for warmup_client_id in warmup_client_ids:
        client.wait_for_ack(warmup_client_id, timeout_s=15.0)
    missing_warmups = []
    for warmup_client_id in warmup_client_ids:
        warmup_done = client.wait_for_done(warmup_client_id, timeout_s=timeout_s)
        if warmup_done is None:
            missing_warmups.append(warmup_client_id)
    if missing_warmups:
        print(
            f"[WARNING] warm-up requests did not report done before timeout: "
            f"{missing_warmups}; continuing with measured run."
        )
    else:
        print(f"[TEST] warm-up requests completed: {warmup_client_ids}.")


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
        client.wait_for_ack(warmup_client_id, timeout_s=15.0)
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
    prompt: str,
    timeout_s: float,
) -> None:
    request_count = 3
    max_new_tokens = 2
    expected_phase_steps = [
        ("prefill", 0),
        ("decode", 1),
    ]
    prefix = f"{node_count}node_3request_2round"

    print(
        f"\n[TEST] {node_count} nodes, 3 simultaneous requests, max_new_tokens=2"
    )
    configure_even_split_topology(client, node_count)
    print("[TEST] waiting 2 seconds after config for nodes to settle...")
    time.sleep(2.0)
    client.drain_events()

    print("[TEST] running topology-matched warm-up batch without measurement...")
    wait_for_warmup_batch(
        client=client,
        owner_index=0,
        prompt=prompt,
        request_count=request_count,
        max_new_tokens=max_new_tokens,
        timeout_s=timeout_s,
        trace_label_prefix=f"{prefix}_warmup",
    )
    print("[TEST] waiting 2 seconds after warm-up for runtime state to settle...")
    time.sleep(2.0)
    client.drain_events()

    process_started_perf = time.perf_counter()
    client_request_ids = client.submit_burst_requests(
        owner_index=0,
        prompts=[prompt] * request_count,
        max_new_tokens=max_new_tokens,
        trace_forward_measurement=True,
        trace_label_prefix=prefix,
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
        forward_rows.append(
            {
                "experiment": "two_round_pipeline_latency",
                "topology_node_count": node_count,
                "request_count": request_count,
                "max_active_requests": client.max_active_requests,
                "max_new_tokens": max_new_tokens,
                "request_order": request_order_by_client_id[client_request_id],
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
        done_rows.append(
            {
                "experiment": "two_round_pipeline_latency",
                "topology_node_count": node_count,
                "request_count": request_count,
                "max_active_requests": client.max_active_requests,
                "max_new_tokens": max_new_tokens,
                "request_order": request_order_by_client_id[client_request_id],
                "client_request_id": client_request_id,
                "request_id": request_id_by_client_id.get(client_request_id),
                "reason": done_report.get("reason"),
                "output_token_count": done_report.get("output_token_count"),
                "done_received_elapsed_ms": done_report.get(
                    "done_received_elapsed_ms"
                ),
            }
        )

    total_elapsed_ms = (
        max(float(row["done_received_elapsed_ms"]) for row in done_rows)
        if done_rows and not missing_done
        else None
    )
    expected_report_count = (
        request_count * node_count * len(expected_phase_steps)
    )
    summary_rows = [
        {
            "experiment": "two_round_pipeline_latency",
            "topology_node_count": node_count,
            "request_count": request_count,
            "max_active_requests": client.max_active_requests,
            "max_new_tokens": max_new_tokens,
            "expected_forward_report_count": expected_report_count,
            "collected_forward_report_count": len(forward_rows),
            "ack_report_count": len(ack_by_client_id),
            "missing_ack_count": len(missing_ack),
            "missing_forward_report_count": len(missing_reports),
            "done_report_count": len(done_rows),
            "missing_done_count": len(missing_done),
            "total_process_elapsed_ms": total_elapsed_ms,
            "missing_ack_client_request_ids": repr(missing_ack),
            "missing_forward_reports": repr(missing_reports),
            "missing_done_client_request_ids": repr(missing_done),
        }
    ]

    write_csv(result_dir / f"{prefix}_forward_reports.csv", forward_rows)
    write_csv(result_dir / f"{prefix}_done_reports.csv", done_rows)
    write_csv(result_dir / f"{prefix}_summary.csv", summary_rows)

    if missing_ack:
        print(f"[WARNING] missing ack reports for {node_count}-node scenario: {missing_ack}")
    if missing_reports:
        print(f"[WARNING] missing forward reports for {node_count}-node scenario: {missing_reports}")
    if missing_done:
        print(f"[WARNING] missing done reports for {node_count}-node scenario: {missing_done}")
    print(
        f"[TEST] {node_count}-node scenario total elapsed: "
        f"{total_elapsed_ms if total_elapsed_ms is not None else 'N/A'} ms"
    )


def run_two_round_pipeline_latency_experiment(client: PipelineDebugClient) -> None:
    result_dir = make_result_dir()
    print(
        "\nTwo-round pipeline latency experiments:\n"
        "  1. 3 nodes, 3 simultaneous requests, max_new_tokens=2\n"
        "  2. 2 nodes, 3 simultaneous requests, max_new_tokens=2; request 3 should wait in pending queue\n"
        "  3. run both scenarios\n"
    )
    choice = input("Experiment: ").strip()
    prompt = ask_prompt(DEFAULT_PROMPTS[4])
    timeout_s = float(input("Overall timeout seconds [180]: ").strip() or "180")
    ask_telemetry(client)

    if choice == "1":
        run_two_round_pipeline_latency_scenario(client, result_dir, 3, prompt, timeout_s)
    elif choice == "2":
        run_two_round_pipeline_latency_scenario(client, result_dir, 2, prompt, timeout_s)
    elif choice == "3":
        run_two_round_pipeline_latency_scenario(client, result_dir, 3, prompt, timeout_s)
        run_two_round_pipeline_latency_scenario(client, result_dir, 2, prompt, timeout_s)
    else:
        print("Unknown experiment.")
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
            "  8. run predefined Jetson test scenario\n"
            "  9. show topology/config\n"
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
            run_scenario(client)
        elif choice == "9":
            client.show_topology()
        elif choice in {"q", "quit", "exit"}:
            return
        else:
            print("Unknown choice.")


def main() -> None:
    client = PipelineDebugClient(DEFAULT_NODES)
    client.show_topology()
    if ask_yes_no("Send initial config now?", default=True):
        client.send_config()
    menu_loop(client)


if __name__ == "__main__":
    main()
