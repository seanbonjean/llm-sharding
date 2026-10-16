"""CPU correctness tests against a tiny, randomly initialized official Llama.

Run: python -B test/unittest_shard_loader.py
使用本地 PyTorch/Transformers 和随机权重进行 CPU 数值验证，不下载模型。
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.shard_loader import LlamaShardPart


class ShardCausalAttentionTests(unittest.TestCase):
    """Compare loaded shards, independent caches and official causal forward.

    对照已加载分片、独立缓存与官方因果前向。
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Create matching full-model and on-disk shard fixtures.

        Args:
            cls: Test class owning the shared CPU fixtures. 持有共享 CPU 测试模型的类。
        """
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(17)
        cls.directory = tempfile.TemporaryDirectory()
        cls.weights_path = Path(cls.directory.name)
        config = LlamaConfig(
            vocab_size=37, hidden_size=24, intermediate_size=48,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64, attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        cls.model = LlamaForCausalLM(config).eval()
        config.save_pretrained(cls.weights_path)
        for index, layer in enumerate(cls.model.model.layers):
            torch.save(layer.state_dict(), cls.weights_path / f"block_{index}.pth")
        torch.save(cls.model.model.norm.state_dict(), cls.weights_path / "final_norm.pth")
        cls.boundaries = (2, 3)
        cls.shards = [
            LlamaShardPart(str(cls.weights_path), ["block_0.pth", "block_1.pth"], 0, 2).eval(),
            LlamaShardPart(
                str(cls.weights_path), ["block_2.pth"], 2, 3,
                add_final_norm=True, final_norm_weight="final_norm.pth",
            ).eval(),
        ]

    @classmethod
    def tearDownClass(cls) -> None:
        """Release saved weights and restore the process thread setting.

        Args:
            cls: Test class owning temporary resources. 持有临时资源的测试类。
        """
        cls.directory.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def set_backend(self, backend: str) -> None:
        """Select the same attention implementation in reference and shards.

        Args:
            self: Test instance sharing matching model weights. 共享同一组模型权重的测试实例。
            backend: eager or sdpa implementation to compare. 待比较的 eager 或 sdpa 后端。
        """
        self.model.config._attn_implementation = backend
        for shard in self.shards:
            shard.config._attn_implementation = backend

    def forward_shards(
        self, input_ids: torch.Tensor, caches: list[DynamicCache] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the real shard API with per-stage relative cache indices.

        Args:
            self: Test instance holding embedding, RoPE, shards and LM head. 持有推理组件的实例。
            input_ids: Current token IDs to embed and forward. 本轮输入 token ID。
            caches: Separate cache for each shard, or None for uncached execution.
                各分片的独立缓存；None 表示无缓存执行。
            attention_mask: Optional full-history padding or prepared mask.
                覆盖历史和当前输入的可选掩码。
        """
        with torch.inference_mode():
            hidden_states = self.model.model.embed_tokens(input_ids)
            past_length = caches[0].get_seq_length() if caches is not None else 0
            position_ids = torch.arange(past_length, past_length + input_ids.shape[1]).unsqueeze(0)
            rotary_emb = self.model.model.rotary_emb(hidden_states, position_ids)
            boundary_states = []
            for index, shard in enumerate(self.shards):
                hidden_states = shard(
                    hidden_states, attention_mask=attention_mask,
                    past_key_value=caches[index] if caches is not None else None,
                    rotary_emb=rotary_emb,
                )
                boundary_states.append(hidden_states)
            return self.model.lm_head(hidden_states), boundary_states

    def assert_matches_reference(
        self, input_ids: torch.Tensor, reference_cache: DynamicCache | None = None,
        shard_caches: list[DynamicCache] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compare valid positions, shard boundaries, logits and every cached layer.

        Args:
            self: Test instance providing model fixtures and assertions. 提供模型和断言的实例。
            input_ids: Current prefill or decode token IDs. 本轮 prefill 或 decode 输入。
            reference_cache: Full model cache for this request. 此请求的完整模型缓存。
            shard_caches: Per-shard caches for this request. 此请求的各分片独立缓存。
            attention_mask: Optional 2D padding mask covering history and current tokens.
                覆盖历史和当前输入的可选二维 padding 掩码。
        """
        with torch.inference_mode():
            reference = self.model(
                input_ids, attention_mask=attention_mask, past_key_values=reference_cache,
                use_cache=reference_cache is not None, output_hidden_states=True,
            )
            actual_logits, actual_states = self.forward_shards(input_ids, shard_caches, attention_mask)
        valid_queries = (
            attention_mask[:, -input_ids.shape[1]:].bool()
            if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
        )
        torch.testing.assert_close(
            actual_logits[valid_queries], reference.logits[valid_queries], rtol=1e-5, atol=1e-6,
        )
        for boundary, actual in zip(self.boundaries, actual_states):
            torch.testing.assert_close(
                actual[valid_queries], reference.hidden_states[boundary][valid_queries],
                rtol=1e-5, atol=1e-6,
            )
        if shard_caches is not None:
            for shard, cache in zip(self.shards, shard_caches):
                for relative_index in range(shard.end - shard.start):
                    global_index = shard.start + relative_index
                    self.assertEqual(cache.get_seq_length(relative_index), reference_cache.get_seq_length(global_index))
                    for actual, expected in zip(cache[relative_index], reference_cache[global_index]):
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        return actual_logits

    def test_prefill_matches_official_model(self) -> None:
        """Match both backends across a nonzero shard start; self holds fixtures.
        对照两种后端及非零起始层分片；self 持有测试模型。
        """
        tokens = torch.tensor([[3, 4, 5, 6, 7], [8, 9, 10, 11, 12]])
        for backend in ("eager", "sdpa"):
            with self.subTest(backend=backend):
                self.set_backend(backend)
                self.assert_matches_reference(tokens)
                self.assert_matches_reference(tokens, DynamicCache(), [DynamicCache(), DynamicCache()])

    def test_future_tokens_do_not_change_prefix(self) -> None:
        """Changing a suffix must preserve all earlier outputs; self holds fixtures.
        改变后文不得改变前文输出；self 持有测试模型。
        """
        original = torch.tensor([[3, 4, 5, 6, 7]])
        changed = torch.tensor([[3, 4, 5, 20, 21]])
        for backend in ("eager", "sdpa"):
            with self.subTest(backend=backend):
                self.set_backend(backend)
                first, _ = self.forward_shards(original)
                second, _ = self.forward_shards(changed)
                torch.testing.assert_close(first[:, :3], second[:, :3], rtol=1e-5, atol=1e-6)

    def test_cached_single_and_multi_token_decode(self) -> None:
        """Check cache offsets against both full-sequence and incremental references.
        self 为测试实例；验证单 token 和多 token decode 的历史偏移。
        """
        tokens = torch.tensor([[3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14]])
        for backend in ("eager", "sdpa"):
            with self.subTest(backend=backend):
                self.set_backend(backend)
                with torch.inference_mode():
                    full_logits = self.model(tokens, use_cache=False).logits
                reference_cache = DynamicCache()
                shard_caches = [DynamicCache(), DynamicCache()]
                outputs, offset = [], 0
                for count in (3, 1, 2):
                    outputs.append(self.assert_matches_reference(
                        tokens[:, offset:offset + count], reference_cache, shard_caches,
                    ))
                    offset += count
                torch.testing.assert_close(torch.cat(outputs, dim=1), full_logits, rtol=1e-5, atol=1e-6)

    def test_padding_is_respected_in_prefill_and_decode(self) -> None:
        """Cover left/right padding and changing PAD contents; self holds fixtures.
        验证左右 padding、PAD 内容不影响有效位置，以及带 padding 历史的 decode。
        """
        tokens = torch.tensor([[0, 0, 4, 5, 6], [7, 8, 9, 0, 0]])
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 0, 0]])
        changed = tokens.masked_fill(mask == 0, 31)
        for backend in ("eager", "sdpa"):
            with self.subTest(backend=backend):
                self.set_backend(backend)
                original_logits = self.assert_matches_reference(tokens, attention_mask=mask)
                changed_logits = self.assert_matches_reference(changed, attention_mask=mask)
                torch.testing.assert_close(original_logits[mask.bool()], changed_logits[mask.bool()])
                reference_cache = DynamicCache()
                shard_caches = [DynamicCache(), DynamicCache()]
                self.assert_matches_reference(tokens, reference_cache, shard_caches, mask)
                next_tokens = torch.tensor([[10, 11], [12, 13]])
                extended_mask = torch.cat((mask, torch.ones((2, 2), dtype=torch.long)), dim=1)
                self.assert_matches_reference(next_tokens, reference_cache, shard_caches, extended_mask)

    def test_interleaved_requests_keep_separate_histories(self) -> None:
        """Interleave unequal request histories through shared layers.
        self 为测试实例；验证共享模型层时各请求仍使用独立 KV 历史。
        """
        for backend in ("eager", "sdpa"):
            with self.subTest(backend=backend):
                self.set_backend(backend)
                reference_caches = [DynamicCache(), DynamicCache()]
                shard_caches = [[DynamicCache(), DynamicCache()] for _ in range(2)]
                for request, tokens in ((0, [3, 4, 5]), (1, [6, 7]), (0, [8, 9]), (1, [10])):
                    self.assert_matches_reference(
                        torch.tensor([tokens]), reference_caches[request], shard_caches[request],
                    )

    def test_chunk_mask_uses_past_length_and_padding(self) -> None:
        """Inspect visibility of cached, current and future keys.
        self 为测试实例；逐格检查历史、当前和未来位置的可见性。
        """
        for backend in ("eager", "sdpa"):
            with self.subTest(backend=backend):
                self.set_backend(backend)
                caches = [DynamicCache(), DynamicCache()]
                self.forward_shards(torch.tensor([[0, 3, 4]]), caches, torch.tensor([[0, 1, 1]]))
                hidden_states = torch.zeros((1, 2, 24))
                mask, positions = self.shards[0].prepare_attention_mask(
                    hidden_states, torch.tensor([[0, 1, 1, 1, 1]]), caches[0],
                )
                self.assertEqual(positions.tolist(), [3, 4])
                allowed = mask if mask.dtype == torch.bool else mask == 0
                expected = torch.tensor([[[[False, True, True, True, False],
                                           [False, True, True, True, True]]]])
                torch.testing.assert_close(allowed, expected)
                self.assertEqual(caches[0].get_seq_length(), 3)

    def test_decode_rejects_mask_missing_history(self) -> None:
        """Reject a query-only padding mask before cache mutation.
        self 为测试实例；拒绝漏掉历史部分的二维掩码。
        """
        self.set_backend("eager")
        caches = [DynamicCache(), DynamicCache()]
        self.forward_shards(torch.tensor([[3, 4, 5]]), caches)
        with self.assertRaisesRegex(ValueError, "past and current"):
            self.forward_shards(torch.tensor([[6]]), caches, torch.ones((1, 1)))
        self.assertEqual(caches[0].get_seq_length(), 3)


if __name__ == "__main__":
    unittest.main()
