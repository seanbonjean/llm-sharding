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
USER_REQUEST_ACK = "user_request_ack"
KV_CACHE_QUERY = "kv_cache_query"
KV_CACHE_REPORT = "kv_cache_report"
PIPELINE_DONE_REPORT = "pipeline_done_report"

TOKENIZER_PATH = "shards/Llama-3___2-3B-Instruct_float16"
RESULT_ROOT = Path("results") / "pipeline_kv_cache"
DEFAULT_TELEMETRY_HOST = "127.0.0.1"
DEFAULT_TELEMETRY_PORT = 40900
DEFAULT_PREFILL_TOKEN_LENGTHS = [8, 16, 32, 64, 128, 256, 512]
DEFAULT_DECODE_OUTPUT_TOKEN_LENGTHS = [8, 16, 32, 64, 128, 256, 512]
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
        trace_label: str = "",
    ) -> str:
        """Send one user request to the selected owner node."""

        owner = self.nodes[owner_index]
        client_request_id = self._new_client_request_id()
        payload = {
            "type": USER_REQUEST,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "client_request_id": client_request_id,
        }
        if input_ids is not None:
            payload["input_ids"] = input_ids.cpu()
        if trace_kv_cache:
            if not self.telemetry_public_addr:
                raise RuntimeError("[ERROR] telemetry must be configured before tracing KV cache.")
            payload.update(
                {
                    "response_addr": self.telemetry_public_addr,
                    "telemetry_addr": self.telemetry_public_addr,
                    "trace_kv_cache": True,
                    "trace_label": trace_label,
                }
            )
        self._data_socket(owner.node_addr).send(self._serialize(payload))
        print(
            f"[REQUEST] sent to {owner.node_id} ({owner.node_addr}), "
            f"client_request_id={client_request_id}, max_new_tokens={max_new_tokens}"
        )
        return client_request_id

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
                "response_addr": self.telemetry_public_addr,
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

    fit_rows = []
    plt.figure(figsize=(8, 5))
    for group_name, group_rows in groups.items():
        xs = [float(row[x_key]) for row in group_rows]
        ys_mib = [bytes_to_mib(row[y_key]) for row in group_rows]
        label = "measured" if group_key is None else f"{group_name} measured"
        plt.scatter(xs, ys_mib, label=label)
        if len(xs) >= 2:
            fit = linear_fit(xs, ys_mib)
            sorted_xs = sorted(xs)
            fitted_ys = [
                fit["slope"] * x_value + fit["intercept"]
                for x_value in sorted_xs
            ]
            fit_label = (
                "linear fit"
                if group_key is None
                else f"{group_name} fit"
            )
            plt.plot(sorted_xs, fitted_ys, label=fit_label)
            fit_rows.append({"group": group_name, **fit})

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
            print(
                "[TEST] prefill sample: "
                f"tokens={ack.get('input_token_length')}, "
                f"KV={aggregate['kv_cache_mib']:.6f} MiB"
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
    )
    write_csv(result_dir / "prefill_kv_fit.csv", fit_rows)


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
            }
        )
    if done_event is None:
        done_event = client.wait_for_done(client_request_id, timeout_s=30.0)

    decode_rows = [row for row in aggregate_rows if row.get("phase") == "decode"]
    write_csv(result_dir / "decode_kv_summary.csv", aggregate_rows)
    write_csv(result_dir / "decode_kv_per_node.csv", per_node_rows)
    fit_rows = plot_scatter_with_fit(
        decode_rows,
        x_key="output_token_length",
        y_key="kv_cache_bytes",
        output_path=result_dir / "decode_kv_fit.png",
        title="Pipeline KV cache during decode",
        x_label="Generated output token length",
    )
    write_csv(result_dir / "decode_kv_fit.csv", fit_rows)
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
            aggregate, _ = collect_complete_aggregate(
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
            summary_rows.append(aggregate)
            pair_rows.append(aggregate)
            client.wait_for_done(client_request_id, timeout_s=60.0)

        first, second = pair_rows
        first_bytes = int(first["kv_cache_bytes"])
        second_bytes = int(second["kv_cache_bytes"])
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
            }
        )
        print(
            "[TEST] sequential compare: "
            f"tokens={first.get('actual_input_token_length')}, "
            f"first={bytes_to_mib(first_bytes):.6f} MiB, "
            f"second={bytes_to_mib(second_bytes):.6f} MiB"
        )

    write_csv(result_dir / "sequential_same_prompt_summary.csv", summary_rows)
    write_csv(result_dir / "sequential_same_prompt_comparison.csv", comparison_rows)
    fit_rows = plot_scatter_with_fit(
        summary_rows,
        x_key="actual_input_token_length",
        y_key="kv_cache_bytes",
        output_path=result_dir / "sequential_same_prompt_fit.png",
        title="Sequential same-prompt prefill KV cache",
        x_label="Input token length",
        group_key="order",
    )
    write_csv(result_dir / "sequential_same_prompt_fit.csv", fit_rows)


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
    total_rows: list[dict] = []
    done_requests: set[str] = set()
    client_ids = {first_client_id: "first"}
    latest_bytes_by_client: dict[str, int] = {}
    second_client_id = None
    second_submit_elapsed = second_delay_s

    def process_event(event: dict) -> None:
        nonlocal second_submit_elapsed
        if event.get("type") == PIPELINE_DONE_REPORT and event.get("client_request_id") in client_ids:
            current_client_id = str(event.get("client_request_id"))
            done_requests.add(current_client_id)
            latest_bytes_by_client[current_client_id] = 0
            elapsed = float(event.get("timestamp") or time.time()) - start_time
            total_rows.append(
                {
                    "elapsed_s": elapsed,
                    "total_kv_cache_bytes": sum(latest_bytes_by_client.values()),
                    "total_kv_cache_mib": bytes_to_mib(sum(latest_bytes_by_client.values())),
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
        latest_bytes_by_client[current_client_id] = int(aggregate["kv_cache_bytes"])
        total_bytes = sum(latest_bytes_by_client.values())
        total_rows.append(
            {
                "elapsed_s": elapsed,
                "total_kv_cache_bytes": total_bytes,
                "total_kv_cache_mib": bytes_to_mib(total_bytes),
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
    write_csv(result_dir / "overlap_same_prompt_total.csv", total_rows)
    request_fit_rows = plot_scatter_with_fit(
        request_rows,
        x_key="elapsed_s",
        y_key="kv_cache_bytes",
        output_path=result_dir / "overlap_same_prompt_per_request.png",
        title="Overlapped requests KV cache by request",
        x_label="Elapsed time (s)",
        group_key="request_order",
        vertical_line_x=second_submit_elapsed,
    )
    write_csv(result_dir / "overlap_same_prompt_per_request_fit.csv", request_fit_rows)

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
    total_fit_rows = plot_scatter_with_fit(
        total_fit_input_rows,
        x_key="elapsed_s",
        y_key="total_kv_cache_bytes",
        output_path=result_dir / "overlap_same_prompt_total.png",
        title="Total KV cache before/after second request",
        x_label="Elapsed time (s)",
        group_key="segment",
        vertical_line_x=second_submit_elapsed,
    )
    write_csv(result_dir / "overlap_same_prompt_total_fit.csv", total_fit_rows)


def run_kv_cache_experiments(client: PipelineDebugClient) -> None:
    result_dir = make_result_dir()
    print(
        "\nKV cache experiments:\n"
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
            "  6. run predefined Jetson test scenario\n"
            "  7. show topology/config\n"
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
            run_scenario(client)
        elif choice == "7":
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
