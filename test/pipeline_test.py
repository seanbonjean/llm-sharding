"""
Pipeline inference debug CLI.

This script is intended to run on the controller/debug machine. It can:
1. send or re-send pipeline config to all Jetson nodes;
2. submit one or more user requests to a selected owner node;
3. change simple config knobs such as max_active_requests and can_receive_user_request;
4. run the manual scenarios listed in docs/pipeline_inference_v1.md.

The node processes should already be running with:
    python pipeline_start_node.py --port 40700
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass

import torch
import zmq


USER_REQUEST = "user_request"


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

    def submit_request(self, owner_index: int, prompt: str, max_new_tokens: int) -> None:
        """Send one user request to the selected owner node."""

        owner = self.nodes[owner_index]
        payload = {
            "type": USER_REQUEST,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        }
        self._data_socket(owner.node_addr).send(self._serialize(payload))
        print(
            f"[REQUEST] sent to {owner.node_id} ({owner.node_addr}), "
            f"max_new_tokens={max_new_tokens}"
        )

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


def change_receive_flag(client: PipelineDebugClient) -> None:
    index = ask_node(client, prompt="Choose node to toggle can_receive_user_request")
    node = client.nodes[index]
    node.can_receive_user_request = not node.can_receive_user_request
    print(f"[CONFIG] {node.node_id} can_receive_user_request={node.can_receive_user_request}")
    client.send_config()


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
            "  5. toggle can_receive_user_request and send config\n"
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
            change_receive_flag(client)
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
