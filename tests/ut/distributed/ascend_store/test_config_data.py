#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Hybrid C128 external-transfer tests for the ascend_store metadata module.

The legacy config_data module was renamed to metadata (with its test suite in
test_metadata.py). This module keeps the hybrid C128 transfer tests adapted to
the merged metadata interfaces.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
    ChunkedTokenDatabase,
    HybridCacheC128Config,
    KeyMetadata,
    PoolKey,
    resolve_hybrid_cache_c128_config,
)


class TestPoolKey(unittest.TestCase):
    def setUp(self):
        self.meta = KeyMetadata("llama", 1, 3, 0)

    def test_default_key_string_is_unchanged(self):
        k = PoolKey(self.meta, "hash1")
        self.assertEqual(
            k.to_string(),
            "llama@dcp:3@head_or_tp_rank:1@pp_rank:0@group:0"
            "@cache_role:kv@cache_family:default@hash1",
        )

    def test_transfer_namespace_and_range_are_part_of_key(self):
        transfer_meta = KeyMetadata(
            "llama",
            1,
            3,
            0,
            transfer_namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            slot_start=4,
            slot_end=8,
        )
        key = PoolKey(transfer_meta, "hash1")
        self.assertIn("@transfer:hybrid_c128_chunk_v1@range:4_8@hash1", key.to_string())
        self.assertNotEqual(hash(key), hash(PoolKey(self.meta, "hash1")))


class TestChunkedTokenDatabase(unittest.TestCase):
    def setUp(self):
        self.meta = KeyMetadata("llama", 0, 0, 0)

    def test_disabled_transfer_chunks_preserve_compressed_block_mapping(self):
        db = ChunkedTokenDatabase(
            [self.meta],
            block_size=[8],
            partitions=None,
            hash_block_size=8,
        )
        db.set_group_buffers(
            {0: [1000]},
            {0: [80]},
            group_cache_families={0: "c4"},
        )
        db.cache_coordinator = MagicMock()
        db.cache_coordinator.cacheable_group_ids = [0]
        db.cache_coordinator.transfer_value_block_range.side_effect = AssertionError(
            "disabled transfer path must preserve legacy block-id mapping"
        )

        chunks = list(
            db.process_transfer_chunks_with_block_ids(
                64,
                [f"h{index}" for index in range(8)],
                [7, 11],
            )
        )

        self.assertEqual([chunk.raw_start for chunk in chunks], [48, 56])
        self.assertEqual([chunk.block_id for chunk in chunks], [7, 11])
        addrs, sizes, block_id = db.prepare_transfer_value(chunks[1], [7, 11])
        self.assertEqual((addrs, sizes, block_id), ([1000 + 11 * 80], [80], 11))

    def test_hybrid_c128_transfer_chunks_use_non_cumulative_c128_ranges(self):
        config = HybridCacheC128Config(
            enabled=True,
            chunk_tokens=512,
            namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            c128_group_id=1,
            c128_slots_per_page=128,
        )
        metadata = [
            KeyMetadata("hybrid-model", 0, 0, 0, kv_cache_group_id=0),
            KeyMetadata("hybrid-model", 0, 0, 0, kv_cache_group_id=1),
        ]
        db = ChunkedTokenDatabase(
            metadata,
            block_size=[128, 128],
            partitions=None,
            hash_block_size=128,
            hybrid_cache_c128_config=config,
        )
        db.set_group_buffers(
            {0: [1000], 1: [2000]},
            {0: [1280], 1: [1280]},
            group_cache_families={0: "c4", 1: "c128"},
        )
        hashes = [f"h{index}" for index in range(132)]

        chunks = list(db.process_transfer_chunks(16896, hashes, kv_cache_group_id=1))

        self.assertEqual(len(chunks), 33)
        self.assertEqual((chunks[0].value_start, chunks[0].value_end), (0, 4))
        self.assertEqual((chunks[1].value_start, chunks[1].value_end), (4, 8))
        self.assertEqual((chunks[31].value_start, chunks[31].value_end), (124, 128))
        self.assertEqual((chunks[32].value_start, chunks[32].value_end), (0, 4))
        self.assertEqual(chunks[31].target_block_index, 0)
        self.assertEqual(chunks[32].target_block_index, 1)
        self.assertIn("@range:0_4", chunks[0].key.to_string())
        self.assertIn("@range:4_8", chunks[1].key.to_string())

    def test_hybrid_c128_non_c128_chunk_prepares_one_complete_block(self):
        config = HybridCacheC128Config(
            enabled=True,
            chunk_tokens=512,
            namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            c128_group_id=1,
            c128_slots_per_page=128,
        )
        metadata = [
            KeyMetadata("hybrid-model", 0, 0, 0, kv_cache_group_id=0),
            KeyMetadata("hybrid-model", 0, 0, 0, kv_cache_group_id=1),
        ]
        db = ChunkedTokenDatabase(
            metadata,
            block_size=[128, 128],
            partitions=None,
            hash_block_size=128,
            hybrid_cache_c128_config=config,
        )
        db.set_group_buffers(
            {0: [1000], 1: [2000]},
            {0: [1280], 1: [1280]},
            group_cache_families={0: "c4", 1: "c128"},
        )
        hashes = ["h0", "h1", "h2", "h3"]
        chunk = next(iter(db.process_transfer_chunks_with_block_ids(512, hashes, [7], kv_cache_group_id=0)))

        addrs, sizes, block_id = db.prepare_transfer_value(chunk, [7], kv_cache_group_id=0)

        self.assertEqual(block_id, 7)
        self.assertEqual(addrs, [1000 + 7 * 1280])
        self.assertEqual(sizes, [1280])
        self.assertNotIn("@range:", chunk.key.to_string())

    def test_hybrid_c128_block_size_64_uses_two_authoritative_slots_per_chunk(self):
        config = HybridCacheC128Config(
            enabled=True,
            chunk_tokens=256,
            namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            c128_group_id=1,
            c128_slots_per_page=64,
        )
        metadata = [
            KeyMetadata("hybrid-model", 0, 0, 0, kv_cache_group_id=0),
            KeyMetadata("hybrid-model", 0, 0, 0, kv_cache_group_id=1),
        ]
        db = ChunkedTokenDatabase(
            metadata,
            block_size=[64, 64],
            partitions=None,
            hash_block_size=64,
            hybrid_cache_c128_config=config,
        )
        db.set_group_buffers(
            {0: [1000], 1: [2000]},
            {0: [640], 1: [640]},
            group_cache_families={0: "c4", 1: "c128"},
        )
        hashes = [f"h{index}" for index in range(8)]

        chunks = list(db.process_transfer_chunks(512, hashes, kv_cache_group_id=1))

        self.assertEqual(len(chunks), 2)
        self.assertEqual([(chunk.value_start, chunk.value_end) for chunk in chunks], [(0, 2), (2, 4)])
        self.assertEqual([chunk.target_block_index for chunk in chunks], [0, 0])
        self.assertIn("@range:0_2", chunks[0].key.to_string())
        self.assertIn("@range:2_4", chunks[1].key.to_string())

    def test_hybrid_c128_block_ids_map_across_20k_prefix_pages(self):
        config = HybridCacheC128Config(
            enabled=True,
            chunk_tokens=512,
            namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            c128_group_id=0,
            c128_slots_per_page=128,
        )
        db = ChunkedTokenDatabase(
            [KeyMetadata("hybrid-model", 0, 0, 0)],
            block_size=[128],
            partitions=None,
            hash_block_size=128,
            hybrid_cache_c128_config=config,
        )
        db.set_group_buffers(
            {0: [1000]},
            {0: [1280]},
            group_cache_families={0: "c128"},
        )
        hashes = [f"h{index}" for index in range(160)]

        chunks = list(
            db.process_transfer_chunks_with_block_ids(
                20 * 1024,
                hashes,
                [10, 11],
            )
        )

        self.assertEqual(len(chunks), 40)
        self.assertEqual([chunk.block_id for chunk in chunks[:32]], [10] * 32)
        self.assertEqual([chunk.block_id for chunk in chunks[32:]], [11] * 8)
        self.assertEqual((chunks[32].value_start, chunks[32].value_end), (0, 4))
        self.assertEqual(chunks[32].target_block_index, 1)

    def test_hybrid_c128_tail_block_ids_and_mask_num_map_second_page(self):
        config = HybridCacheC128Config(
            enabled=True,
            chunk_tokens=512,
            namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            c128_group_id=0,
            c128_slots_per_page=128,
        )
        db = ChunkedTokenDatabase(
            [KeyMetadata("hybrid-model", 0, 0, 0)],
            block_size=[128],
            partitions=None,
            hash_block_size=128,
            hybrid_cache_c128_config=config,
        )
        db.set_group_buffers(
            {0: [1000]},
            {0: [1280]},
            group_cache_families={0: "c128"},
        )
        hashes = [f"h{index}" for index in range(256)]

        chunks = list(
            db.process_transfer_chunks_with_block_ids(
                32 * 1024,
                hashes,
                [21],
                mask_num=17 * 1024,
            )
        )

        self.assertEqual(len(chunks), 30)
        self.assertEqual(chunks[0].raw_start, 17 * 1024)
        self.assertEqual((chunks[0].value_start, chunks[0].value_end), (8, 12))
        self.assertTrue(all(chunk.target_block_index == 1 for chunk in chunks))
        self.assertTrue(all(chunk.block_id == 21 for chunk in chunks))

    def test_hybrid_c128_skip_null_blocks_keeps_valid_second_page(self):
        config = HybridCacheC128Config(
            enabled=True,
            chunk_tokens=512,
            namespace=HYBRID_CACHE_C128_TRANSFER_NAMESPACE,
            c128_group_id=0,
            c128_slots_per_page=128,
        )
        db = ChunkedTokenDatabase(
            [KeyMetadata("hybrid-model", 0, 0, 0)],
            block_size=[128],
            partitions=None,
            hash_block_size=128,
            hybrid_cache_c128_config=config,
        )
        db.set_group_buffers(
            {0: [1000]},
            {0: [1280]},
            group_cache_families={0: "c128"},
        )
        hashes = [f"h{index}" for index in range(256)]

        chunks = list(
            db.process_transfer_chunks_with_block_ids(
                32 * 1024,
                hashes,
                [0, 22],
                skip_null_blocks=True,
            )
        )

        self.assertEqual(len(chunks), 32)
        self.assertEqual(chunks[0].raw_start, 16 * 1024)
        self.assertTrue(all(chunk.target_block_index == 1 for chunk in chunks))
        self.assertTrue(all(chunk.block_id == 22 for chunk in chunks))


class TestHybridCacheC128Config(unittest.TestCase):
    @staticmethod
    def _config(enabled=True, *, backend="mooncake", load_async=False):
        extra = {
            "backend": backend,
            "load_async": load_async,
            "hybrid_cache_c128_chunk": enabled,
        }
        return SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(),
                hf_config=SimpleNamespace(),
            ),
            kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
        )

    def _resolve(self, config, **overrides):
        kwargs = {
            "use_layerwise": False,
            "group_block_sizes": [128, 128],
            "group_cache_families": ["c4", "c128"],
            "hash_block_size": 128,
            "discard_partial_chunks": True,
        }
        kwargs.update(overrides)
        return resolve_hybrid_cache_c128_config(config, **kwargs)

    def test_missing_option_preserves_default_path(self):
        config = self._config()
        del config.kv_transfer_config.kv_connector_extra_config["hybrid_cache_c128_chunk"]
        self.assertFalse(self._resolve(config).enabled)

    def test_false_option_preserves_default_path(self):
        self.assertFalse(self._resolve(self._config(False)).enabled)

    def test_block_size_128_uses_512_tokens(self):
        resolved = self._resolve(self._config())
        self.assertTrue(resolved.enabled)
        self.assertEqual(resolved.chunk_tokens, 512)
        self.assertEqual(resolved.c128_group_id, 1)
        self.assertEqual(resolved.c128_slots_per_page, 128)

    def test_block_size_64_uses_256_tokens(self):
        resolved = self._resolve(
            self._config(),
            group_block_sizes=[64, 64],
            group_cache_families=["c4", "c128"],
            hash_block_size=64,
        )
        self.assertEqual(resolved.chunk_tokens, 256)
        self.assertEqual(resolved.c128_slots_per_page, 64)

    def test_load_async_is_supported(self):
        resolved = self._resolve(self._config(load_async=True))
        self.assertTrue(resolved.enabled)

    def test_invalid_values_fail_fast(self):
        for value in (1, "true", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self._resolve(self._config(value))

    def test_unsupported_execution_modes_fail_fast(self):
        cases = (
            (self._config(backend="memcache"), {}),
            (self._config(), {"use_layerwise": True}),
            (self._config(), {"discard_partial_chunks": False}),
            (self._config(), {"group_block_sizes": [256, 128]}),
            (self._config(), {"group_block_sizes": [128], "group_cache_families": ["default"]}),
        )
        for config, overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self._resolve(config, **overrides)


if __name__ == "__main__":
    unittest.main()
