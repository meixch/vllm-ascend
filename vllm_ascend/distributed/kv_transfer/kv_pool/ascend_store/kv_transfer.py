import queue
import threading
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from vllm.distributed.kv_events import BlockStored
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend

# isort: off
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    ChunkedTokenDatabase,
    aggregate_c128_page_chunks,
    LayerMultiBlockReqMeta,
    ReqMeta,
    TransferChunkWithBlockId,
    get_block_hashes,
)
# isort: on

_KVPOOL_RANGE_DEBUG_PREFIX = "[KVPOOL_RANGE_DEBUG]"
_TRANSFER_SHUTDOWN_POLL_S = 0.1


def _build_range_debug_payload(
    direction: str,
    layer_id: int,
    sizes: list[list[int]],
    object_offsets: list[list[int]],
    results: list[int],
) -> dict[str, Any]:
    nested_sizes = [[int(size) for size in key_sizes] for key_sizes in sizes]
    return {
        "event": "range",
        "direction": direction,
        "layer_id": int(layer_id),
        "key_count": len(results),
        "requested_bytes": [sum(key_sizes) for key_sizes in nested_sizes],
        "sizes": nested_sizes,
        "object_offsets": [[int(offset) for offset in key_offsets] for key_offsets in object_offsets],
        "results": [int(result) for result in results],
    }


def _emit_range_debug_event(
    direction: str,
    layer_id: int,
    sizes: list[list[int]],
    object_offsets: list[list[int]],
    results: list[int],
) -> None:
    try:
        if not get_ascend_config().kv_pool_range_debug:
            return
        payload = _build_range_debug_payload(direction, layer_id, sizes, object_offsets, results)
        logger.info("%s %s", _KVPOOL_RANGE_DEBUG_PREFIX, json.dumps(payload, separators=(",", ":")))
    except Exception:
        # Debug instrumentation must never affect the transfer path.
        pass


def _emit_commit_debug_event(layer_id: int, key_count: int, results: list[int]) -> None:
    try:
        if not get_ascend_config().kv_pool_range_debug:
            return
        payload = {
            "event": "commit",
            "layer_id": int(layer_id),
            "key_count": int(key_count),
            "results": [int(result) for result in results],
        }
        logger.info("%s %s", _KVPOOL_RANGE_DEBUG_PREFIX, json.dumps(payload, separators=(",", ":")))
    except Exception:
        pass


def _circular_shift(lst: list, offset: int) -> list:
    if not lst or offset == 0:
        return lst
    return lst[offset:] + lst[:offset]


@dataclass(frozen=True)
class _LayerRevokeTask:
    keys: tuple[str, ...]


class LayerBatchBuilder:
    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        page_size_bytes: int,
        num_layers: int,
        group_id: int = 0,
        layer_byte_offset: int = 0,
    ) -> None:
        self.page_size_bytes = page_size_bytes
        self.num_layers = num_layers
        self.group_id = group_id
        self.layer_byte_offset = layer_byte_offset
        self._block_len_np = np.asarray(token_database.group_block_len[group_id], dtype=np.int64)
        self._kv_caches_base_addr_np = np.asarray(
            token_database.group_kv_caches_base_addr[group_id],
            dtype=np.int64,
        )
        group_block_stride = token_database.group_block_stride.get(group_id, token_database.group_block_len[group_id])
        self._block_stride_np = np.asarray(group_block_stride, dtype=np.int64)
        layer_cache_entry_offsets = token_database.group_layer_cache_entry_offsets.get(group_id)
        if layer_cache_entry_offsets is None:
            caches_per_layer = max(1, self._block_len_np.shape[0] // max(1, num_layers))
            layer_cache_entry_offsets = [
                min(layer * caches_per_layer, self._block_len_np.shape[0]) for layer in range(num_layers + 1)
            ]
        self._layer_cache_entry_offsets_np = np.asarray(layer_cache_entry_offsets, dtype=np.int64)
        self._block_ids_buf: np.ndarray | None = None
        self._block_gvas_buf: np.ndarray | None = None

    def _ensure_buf(self, capacity: int) -> tuple[np.ndarray, np.ndarray]:
        if self._block_ids_buf is None or len(self._block_ids_buf) < capacity:
            self._block_ids_buf = np.empty(capacity, dtype=np.int64)
            self._block_gvas_buf = np.empty(capacity, dtype=np.int64)
        assert self._block_ids_buf is not None and self._block_gvas_buf is not None
        return self._block_ids_buf[:capacity], self._block_gvas_buf[:capacity]

    @staticmethod
    def _dedupe_transfer_blocks(
        block_ids_arr: np.ndarray,
        block_gvas_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if block_ids_arr.size <= 1:
            return block_ids_arr, block_gvas_arr

        block_transfer_array = np.column_stack((block_ids_arr, block_gvas_arr))
        _, unique_indices = np.unique(
            block_transfer_array,
            axis=0,
            return_index=True,
        )
        if unique_indices.size == block_ids_arr.size:
            return block_ids_arr, block_gvas_arr

        return (
            block_ids_arr[unique_indices],
            block_gvas_arr[unique_indices],
        )

    def _build_transfer_arrays(
        self,
        block_ids_arr: np.ndarray,
        base_gvas_arr: np.ndarray,
        layer_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base_offset = int(self._layer_cache_entry_offsets_np[layer_id])
        end_offset = int(self._layer_cache_entry_offsets_np[layer_id + 1])
        layer_base_addrs = self._kv_caches_base_addr_np[base_offset:end_offset]
        layer_block_len = self._block_len_np[base_offset:end_offset]
        layer_block_stride = self._block_stride_np[base_offset:end_offset]
        # Per-cache inner offsets within one layer's page: [0, len0, len0+len1, ...].
        layer_inner_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(layer_block_len[:-1], dtype=np.int64))
        )
        rank_layer_offset = self.layer_byte_offset + int(self._block_len_np[:base_offset].sum())
        if base_gvas_arr.size > 0 and np.any(base_gvas_arr <= 0):
            # Zero GVAs here are expected: blocks already committed in the
            # pool or excluded by masks are intentionally not allocated.
            # Real batch_alloc failures are logged as errors by the caller.
            zero_count = int(np.sum(base_gvas_arr <= 0))
            logger.debug(
                "[KVPOOL] build_transfer layer=%d detected %d zero/negative base_gvas "
                "(base_gvas_sample=%s); these blocks will be skipped in batch_copy",
                layer_id,
                zero_count,
                base_gvas_arr[:5].tolist(),
            )
        logger.debug(
            "[KVPOOL] build_transfer layer=%d page_size=%d caches_per_layer=%d "
            "rank_layer_offset=%d layer_block_len=%s layer_inner_offsets=%s "
            "base_gvas=%s",
            layer_id,
            self.page_size_bytes,
            end_offset - base_offset,
            rank_layer_offset,
            layer_block_len.tolist(),
            layer_inner_offsets.tolist(),
            base_gvas_arr.tolist(),
        )

        addr_arr = layer_base_addrs[None, :] + block_ids_arr[:, None] * layer_block_stride[None, :]
        size_arr = np.broadcast_to(layer_block_len, addr_arr.shape)
        gvas_arr = base_gvas_arr[:, None] + rank_layer_offset + layer_inner_offsets[None, :]

        return (
            addr_arr.ravel(),
            size_arr.ravel(),
            gvas_arr.ravel(),
        )

    def _require_request_arrays(
        self,
        block_range: LayerBlockRange,
        is_save: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        request = block_range.request
        group_id = self.group_id
        block_ids_np: np.ndarray | None
        block_gvas_np: np.ndarray | None
        if is_save:
            group_block_ids = request.block_ids_by_group_np
            group_block_gvas = request.block_gvas_by_group_np
            if (
                group_block_ids is not None
                and group_block_gvas is not None
                and group_id < len(group_block_ids)
                and group_id < len(group_block_gvas)
            ):
                block_ids_np = group_block_ids[group_id]
                block_gvas_np = group_block_gvas[group_id]
            else:
                block_ids_np = request.block_ids_np
                block_gvas_np = request.block_gvas_np
        else:
            group_block_ids = request.block_ids_by_group_np
            group_block_gvas = request.load_block_gvas_by_group_np
            if (
                group_block_ids is not None
                and group_block_gvas is not None
                and group_id < len(group_block_ids)
                and group_id < len(group_block_gvas)
            ):
                block_ids_np = group_block_ids[group_id]
                block_gvas_np = group_block_gvas[group_id]
            else:
                block_ids_np = request.block_ids_np
                block_gvas_np = request.load_block_gvas_np
        if block_ids_np is None or block_gvas_np is None:
            raise RuntimeError(
                f"ReqMeta {'save' if is_save else 'load'} block metadata"
                f" is not initialized for request {request.req_id}"
            )
        return block_ids_np, block_gvas_np

    def build_shared(self, task: LayerTransferTask, is_save: bool = True) -> SharedBlockData | None:
        """Pre-compute shared block data that is identical across all layers."""
        if not task.block_ranges:
            return None

        if task.use_key_major_ranges:
            return self._build_key_major_shared(task, is_save)

        total = 0
        for block_range in task.block_ranges:
            total += block_range.end_block - block_range.start_block
            if block_range.partial_block_index is not None:
                total += 1

        block_ids_arr, block_gvas_arr = self._ensure_buf(total)
        req_ids: list[str] = []
        is_last_chunks: list[bool | None] = []
        all_save_keys: list[str] = []
        all_load_keys: list[str] = []
        collected_load_keys_request_ids: set[int] = set()
        offset = 0

        for block_range in task.block_ranges:
            request = block_range.request
            req_ids.append(request.req_id)
            is_last_chunks.append(request.is_last_chunk)
            if request.save_keys:
                all_save_keys.extend(request.save_keys)
            if request.load_keys and id(request) not in collected_load_keys_request_ids:
                # Avoid collecting the same request's load_keys multiple times
                collected_load_keys_request_ids.add(id(request))
                all_load_keys.extend(request.load_keys)
            block_ids_np, block_gvas_np = self._require_request_arrays(block_range, is_save)
            gva_block_offset = request.gva_block_offset if is_save else request.load_gva_block_offset

            num_blocks = block_range.end_block - block_range.start_block
            if num_blocks > 0:
                gva_start = block_range.start_block - gva_block_offset
                gva_end = block_range.end_block - gva_block_offset
                if gva_start < 0 or gva_end > len(block_gvas_np):
                    raise RuntimeError(
                        "ReqMeta GVA metadata does not cover requested block "
                        f"range [{block_range.start_block}, {block_range.end_block}) "
                        f"with offset {gva_block_offset}"
                    )
                end = offset + num_blocks
                block_ids_arr[offset:end] = block_ids_np[block_range.start_block : block_range.end_block]
                block_gvas_arr[offset:end] = block_gvas_np[gva_start:gva_end]
                offset = end

            if block_range.partial_block_index is not None:
                partial_block_gva = None
                partial_gva_per_group = (
                    request.partial_save_gva_per_group if is_save else request.partial_load_gva_per_group
                )
                if task.group_id < len(partial_gva_per_group):
                    partial_block_gva = partial_gva_per_group[task.group_id]
                if partial_block_gva is None:
                    partial_block_gva = request.last_block_gva
                assert partial_block_gva is not None
                block_ids_arr[offset] = block_ids_np[block_range.partial_block_index]
                block_gvas_arr[offset] = partial_block_gva
                offset += 1

        block_ids_slice = block_ids_arr[:offset]
        block_gvas_slice = block_gvas_arr[:offset]
        valid_mask = block_gvas_slice > 0
        if not np.all(valid_mask):
            # Zero GVAs are the designed sentinel for blocks that are not
            # transfer targets this step (already committed in the pool,
            # hit-prefix blocks, or masked-out blocks). Real batch_alloc
            # failures are logged as errors by _alloc_gvas_for_save.
            skip_count = int(np.sum(~valid_mask))
            logger.debug(
                "[KVPOOL] build_shared skipping %d blocks with invalid gva (gva<=0)",
                skip_count,
            )
            block_ids_slice = block_ids_slice[valid_mask]
            block_gvas_slice = block_gvas_slice[valid_mask]

        block_ids_arr, block_gvas_arr = self._dedupe_transfer_blocks(block_ids_slice, block_gvas_slice)

        logger.debug(
            "[KVPOOL] build_shared req_ids=%s block_gvas_arr=%s block_ids_arr=%s",
            req_ids,
            block_gvas_arr.tolist(),
            block_ids_arr.tolist(),
        )
        return SharedBlockData(
            block_ids_arr=block_ids_arr,
            block_gvas_arr=block_gvas_arr,
            req_ids=req_ids,
            is_last_chunks=is_last_chunks,
            save_keys=all_save_keys,
            load_keys=all_load_keys,
        )

    @staticmethod
    def _request_block_keys(
        request: ReqMeta,
        is_save: bool,
    ) -> tuple[list[str | None], int, str | None]:
        if is_save:
            return (
                request.save_block_keys,
                request.save_key_block_offset,
                request.save_last_block_key,
            )
        return (
            request.load_block_keys,
            request.load_key_block_offset,
            request.load_last_block_key,
        )

    def _request_block_ids(self, request: ReqMeta) -> list[int]:
        if request.block_ids_by_group_np is not None and self.group_id < len(request.block_ids_by_group_np):
            return request.block_ids_by_group_np[self.group_id].tolist()
        if request.block_ids_np is not None:
            return request.block_ids_np.tolist()
        if self.group_id < len(request.block_ids_by_group):
            return request.block_ids_by_group[self.group_id]
        return request.block_ids

    def _build_key_major_shared(
        self,
        task: LayerTransferTask,
        is_save: bool,
    ) -> SharedBlockData:
        """Build block-key rows shared by all backend layer ranges."""
        block_ids: list[int] = []
        block_keys: list[str] = []
        req_ids: list[str] = []
        is_last_chunks: list[bool | None] = []
        all_load_keys: list[str] = []
        seen_save_keys: set[str] = set()

        for block_range in task.block_ranges:
            request = block_range.request
            req_ids.append(request.req_id)
            is_last_chunks.append(request.is_last_chunk)
            all_load_keys.extend(request.load_keys)
            request_block_ids = self._request_block_ids(request)
            if (
                block_range.start_block < 0
                or block_range.end_block < block_range.start_block
                or block_range.end_block > len(request_block_ids)
            ):
                raise RuntimeError(
                    "ReqMeta block metadata does not cover requested block range "
                    f"[{block_range.start_block}, {block_range.end_block})"
                )

            request_keys, key_block_offset, last_block_key = self._request_block_keys(request, is_save)
            key_start = block_range.start_block - key_block_offset
            key_end = block_range.end_block - key_block_offset
            if key_start < 0 or key_end > len(request_keys):
                raise RuntimeError(
                    f"ReqMeta {'save' if is_save else 'load'} block key metadata "
                    f"does not cover requested range [{block_range.start_block}, "
                    f"{block_range.end_block}) with offset {key_block_offset}"
                )

            for block_id, key in zip(
                request_block_ids[block_range.start_block : block_range.end_block],
                request_keys[key_start:key_end],
                strict=True,
            ):
                if key is None or (is_save and key in seen_save_keys):
                    continue
                block_ids.append(block_id)
                block_keys.append(key)
                if is_save:
                    seen_save_keys.add(key)

            partial_block_index = block_range.partial_block_index
            if partial_block_index is not None:
                if partial_block_index < 0 or partial_block_index >= len(request_block_ids):
                    raise RuntimeError(f"ReqMeta block metadata does not cover partial block {partial_block_index}")
                if last_block_key is not None and (not is_save or last_block_key not in seen_save_keys):
                    block_ids.append(request_block_ids[partial_block_index])
                    block_keys.append(last_block_key)
                    if is_save:
                        seen_save_keys.add(last_block_key)

        return SharedBlockData(
            block_ids_arr=np.asarray(block_ids, dtype=np.int64),
            block_gvas_arr=None,
            block_keys=block_keys,
            req_ids=req_ids,
            is_last_chunks=is_last_chunks,
            load_keys=all_load_keys,
        )

    def build_addrs(
        self,
        shared: SharedBlockData,
        layer_id: int,
    ) -> LayerBatchReqMeta | LayerRangeReqMeta:
        """Compute per-layer addresses from pre-computed shared block data."""
        if shared.block_keys is not None:
            base_offset = int(self._layer_cache_entry_offsets_np[layer_id])
            end_offset = int(self._layer_cache_entry_offsets_np[layer_id + 1])
            layer_base_addrs = self._kv_caches_base_addr_np[base_offset:end_offset]
            layer_block_len = self._block_len_np[base_offset:end_offset]
            layer_block_stride = self._block_stride_np[base_offset:end_offset]
            layer_inner_offsets = np.concatenate(
                (np.zeros(1, dtype=np.int64), np.cumsum(layer_block_len[:-1], dtype=np.int64))
            )
            layer_object_offset = int(self._block_len_np[:base_offset].sum())
            offsets = (layer_object_offset + layer_inner_offsets).tolist()
            sizes = layer_block_len.tolist()
            all_buffers = [
                (layer_base_addrs + block_id * layer_block_stride).tolist() for block_id in shared.block_ids_arr
            ]
            return LayerRangeReqMeta(
                req_ids=shared.req_ids,
                layer_id=layer_id,
                block_ids=shared.block_ids_arr.tolist(),
                keys=shared.block_keys,
                all_buffers=all_buffers,
                all_sizes=[sizes.copy() for _ in shared.block_ids_arr],
                all_offsets=[offsets.copy() for _ in shared.block_ids_arr],
                load_keys=shared.load_keys,
            )

        assert shared.block_gvas_arr is not None
        addr_array, size_array, gvas_array = self._build_transfer_arrays(
            shared.block_ids_arr, shared.block_gvas_arr, layer_id
        )

        return LayerBatchReqMeta(
            req_ids=shared.req_ids,
            layer_id=layer_id,
            is_last_chunks=shared.is_last_chunks,
            addr_array=addr_array,
            size_array=size_array,
            gvas_array=gvas_array,
            load_keys=shared.load_keys,
        )

    def build(
        self,
        task: LayerTransferTask,
        is_save: bool = True,
    ) -> LayerBatchReqMeta | LayerRangeReqMeta | None:
        """Full build: shared data + per-layer addresses (backward compat)."""
        shared = self.build_shared(task, is_save)
        if shared is None:
            return None
        return self.build_addrs(shared, task.layer_idx_in_group)


class KVTransferThread(threading.Thread):
    _invalid_block_ids: set[int]
    _invalid_block_ids_lock: threading.Lock

    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        ready_event: threading.Event,
        name: str,
    ):
        super().__init__(daemon=True, name=name)
        self.m_store = m_store
        self.ready_event = ready_event
        self.block_size = block_size
        self.tp_rank = tp_rank
        self.dcp_size = dcp_size
        self.token_database = token_database
        self.done_task_lock = threading.Lock()
        # Serialize request submission with the transition to a fatal state.
        # Otherwise a producer can enqueue immediately after the failed
        # consumer has drained the queue, leaving work with no live consumer.
        self._request_submit_lock = threading.Lock()
        self._stopping = threading.Event()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        # TODO(jianzs): make this configurable
        self.executor = ThreadPoolExecutor(max_workers=32)
        self.finished_requests: set[str] = set()
        self.failed_requests: set[str] = set()
        self.kv_event_lock = threading.Lock()
        self.kv_events: list[BlockStored] = []
        self._fatal_error: BaseException | None = None
        self.save_disabled = threading.Event()
        self.load_disabled = threading.Event()

    def _mark_failed_requests(self, requests: list[ReqMeta]) -> None:
        logger.error("Memcache transfer failed for requests=%s", [request.req_id for request in requests])
        for request in requests:
            request.can_save = False
        with self.done_task_lock:
            self.failed_requests.update(request.req_id for request in requests if len(request.block_ids_by_group) > 1)
        self.save_disabled.set()
        with self._invalid_block_ids_lock:
            self._invalid_block_ids.update(
                block_id
                for request in requests
                if len(request.block_ids_by_group) == 1
                for group in request.block_ids_by_group
                for block_id in group
                if block_id > 0
            )

    def get_and_clear_failed_requests(self) -> set[str]:
        with self.done_task_lock:
            failed = self.failed_requests.copy()
            self.failed_requests.clear()
        return failed

    def _get_block_size(self, kv_cache_group_id: int = 0) -> int:
        if isinstance(self.block_size, list):
            if kv_cache_group_id >= len(self.block_size):
                return self.block_size[0]
            return self.block_size[kv_cache_group_id]
        return self.block_size

    def add_request(
        self,
        request: ReqMeta | LayerMultiBlockReqMeta,
    ) -> torch.Tensor:
        self.request_queue.put(request)

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.done_task_lock:
            finished_requests = self.finished_requests.copy()
            self.finished_requests.clear()
        return finished_requests

    def set_finished_request(self, req_id):
        with self.done_task_lock:
            self.finished_requests.add(req_id)

    @staticmethod
    def _split_transfer_packets(
        gvas: np.ndarray,
        addrs: np.ndarray,
        sizes: np.ndarray,
        max_transfer_bytes: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if max_transfer_bytes <= 0:
            return gvas, addrs, sizes

        split_counts: np.ndarray = (sizes + max_transfer_bytes - 1) // max_transfer_bytes
        total_splits = int(split_counts.sum())
        if total_splits == sizes.shape[0]:
            return gvas, addrs, sizes

        split_indices: np.ndarray = np.arange(int(split_counts.max()), dtype=np.int64)
        split_mask = split_indices[:, None] < split_counts[None, :]
        entry_indices = np.broadcast_to(
            np.arange(sizes.shape[0], dtype=np.int64),
            split_mask.shape,
        )[split_mask]
        transfer_offsets = np.broadcast_to(
            split_indices[:, None] * max_transfer_bytes,
            split_mask.shape,
        )[split_mask]

        split_gvas = gvas[entry_indices] + transfer_offsets
        split_addrs = addrs[entry_indices] + transfer_offsets
        split_sizes = np.minimum(
            max_transfer_bytes,
            sizes[entry_indices] - transfer_offsets,
        )
        return split_gvas, split_addrs, split_sizes

    def _batch_copy_with_limits(
        self,
        gvas: np.ndarray,
        addrs: np.ndarray,
        sizes: np.ndarray,
        direction: int,
        max_transfer_blocks: int,
        max_transfer_bytes: int,
    ) -> int:
        if len(gvas) == 0:
            return 0

        # direction: 0/SMEMB_COPY_L2G = save (write), 1/SMEMB_COPY_G2L = load (read)
        dir_name = "save(L2G)" if direction == 0 else "load(G2L)" if direction == 1 else f"dir{direction}"
        logger.debug(
            "[KVPOOL] batch_copy %s gvas=%d total_bytes=%d",
            dir_name,
            len(gvas),
            int(sizes.sum()) if len(sizes) else 0,
        )

        max_transfer_addrs = 0
        if max_transfer_blocks > 0:
            max_transfer_addrs = max_transfer_blocks * self.num_addrs_per_block
        if max_transfer_addrs <= 0:
            max_transfer_addrs = len(gvas)

        assert self.m_store.store is not None
        for start in range(0, len(gvas), max_transfer_addrs):
            end = start + max_transfer_addrs
            split_gvas, split_addrs, split_sizes = self._split_transfer_packets(
                gvas[start:end],
                addrs[start:end],
                sizes[start:end],
                max_transfer_bytes,
            )
            logger.debug(
                "[KVPOOL] batch_copy %s split_gvas=%s split_sizes=%s",
                dir_name,
                split_gvas.tolist(),
                split_sizes.tolist(),
            )
            res = self.m_store.store.batch_copy(
                split_gvas.tolist(),
                split_addrs.tolist(),
                split_sizes.tolist(),
                direction,
            )
            if res != 0:
                logger.error("[KVPOOL] batch_copy %s FAILED res=%d", dir_name, res)
                return res
        return 0

    @staticmethod
    def _range_transfer_batches(
        keys: list[str],
        all_buffers: list[list[int]],
        all_sizes: list[list[int]],
        all_offsets: list[list[int]],
        max_transfer_blocks: int,
        max_transfer_bytes: int,
    ) -> list[tuple[list[str], list[list[int]], list[list[int]], list[list[int]]]]:
        """Split backend range calls by key count and per-segment byte size."""
        row_count = len(keys)
        if not (len(all_buffers) == len(all_sizes) == len(all_offsets) == row_count):
            raise ValueError("Layerwise range metadata must contain one buffer/size/offset row per key")

        # Default configuration (both limits unset) means one batch holding every
        # key and no per-segment splitting, so every row passes through as-is
        # unless it carries an empty or negative segment. Detect that case and
        # hand back the caller's own lists: this runs once per key, per layer,
        # per group, and rebuilding them element-wise was the receive thread's
        # largest single block of exposed time.
        if max_transfer_blocks <= 0 and max_transfer_bytes <= 0:
            for buffers, sizes, offsets in zip(all_buffers, all_sizes, all_offsets, strict=True):
                if not (len(buffers) == len(sizes) == len(offsets)):
                    raise ValueError("Layerwise range rows must align buffers, sizes, and offsets")
                if sizes and min(sizes) >= 0 and 0 not in sizes:
                    continue
                break
            else:
                return [(keys, all_buffers, all_sizes, all_offsets)]

        normalized_buffers: list[list[int]] = []
        normalized_sizes: list[list[int]] = []
        normalized_offsets: list[list[int]] = []
        for buffers, sizes, offsets in zip(all_buffers, all_sizes, all_offsets, strict=True):
            if not (len(buffers) == len(sizes) == len(offsets)):
                raise ValueError("Layerwise range rows must align buffers, sizes, and offsets")
            split_buffers: list[int] = []
            split_sizes: list[int] = []
            split_offsets: list[int] = []
            for buffer, size, offset in zip(buffers, sizes, offsets, strict=True):
                if size < 0:
                    raise ValueError(f"Layerwise range size must be non-negative, got {size}")
                if size == 0:
                    continue
                packet_size = max_transfer_bytes if max_transfer_bytes > 0 else size
                for packet_offset in range(0, size, packet_size):
                    split_buffers.append(buffer + packet_offset)
                    split_sizes.append(min(packet_size, size - packet_offset))
                    split_offsets.append(offset + packet_offset)
            normalized_buffers.append(split_buffers)
            normalized_sizes.append(split_sizes)
            normalized_offsets.append(split_offsets)

        rows_per_batch = max_transfer_blocks if max_transfer_blocks > 0 else max(1, row_count)
        return [
            (
                keys[start:end],
                normalized_buffers[start:end],
                normalized_sizes[start:end],
                normalized_offsets[start:end],
            )
            for start in range(0, row_count, rows_per_batch)
            for end in [min(start + rows_per_batch, row_count)]
        ]

    def _set_os_thread_name(self) -> None:
        try:
            libc = ctypes.CDLL("libc.so.6")
            # Linux task comm is limited to 15 visible bytes plus NUL.
            libc.prctl(15, self.name[:15].encode(), 0, 0, 0)
        except Exception:
            pass

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        self.m_store.set_device()
        self.ready_event.set()
        while True:
            try:
                request_data = self.request_queue.get()
                if request_data is None or self._stopping.is_set():
                    self.request_queue.task_done()
                    if request_data is None:
                        return
                    continue
                self._handle_request(request_data)
            except Exception as e:
                with self._request_submit_lock:
                    self._fatal_error = e
                    try:
                        self._handle_request_exception(request_data)
                    except Exception:
                        logger.exception("Failed to clean up KV transfer request after an unexpected dispatch error")
                logger.error(
                    "Error in KVCacheTransferThread(%s). type=%s, error=%s. Check thread state and request processing.",
                    self.name,
                    type(e).__name__,
                    e,
                )

    def _handle_request(self, req_meta: Any):
        pass

    def lookup(
        self,
        keys: list[str],
    ) -> list[bool]:
        """
        Check the existence of all keys from the cache engine.
        :return: A bool list where True means the key exists in store.
        """
        try:
            res = self.m_store.exists(keys)  # type: ignore[assignment]
            exists_list = [False] * len(keys)
            for index, value in enumerate(res):  # type: ignore[arg-type]
                exists_list[index] = value == 1
            return exists_list
        except Exception as e:
            logger.error(
                "Remote connection failed in lookup. type=%s, error=%s. Check network and remote store.",
                type(e).__name__,
                e,
            )
            return [False] * len(keys)

    def update_kv_event(self, event: list[BlockStored]):
        with self.kv_event_lock:
            self.kv_events.extend(event)

    def get_kv_events(self) -> list[BlockStored]:
        with self.kv_event_lock:
            events = self.kv_events.copy()
            self.kv_events.clear()
        return events

    @staticmethod
    def _skip_null_blocks(req_meta: ReqMeta, group_id: int, cache_role: str = "kv") -> bool:
        if cache_role != "kv":
            return False
        skip_flags = req_meta.skip_null_blocks_by_group
        return group_id < len(skip_flags) and skip_flags[group_id] if skip_flags else False

    def _process_tokens_with_block_ids(
        self,
        token_len: int,
        block_hashes,
        block_ids: list[int],
        mask_num: int = 0,
        kv_cache_group_id: int = 0,
        skip_null_blocks: bool = False,
        cache_role: str = "kv",
    ):
        process_with_block_ids = getattr(self.token_database, "process_tokens_with_block_ids", None)
        if process_with_block_ids is not None:
            return process_with_block_ids(
                token_len,
                block_hashes,
                block_ids,
                mask_num,
                kv_cache_group_id=kv_cache_group_id,
                skip_null_blocks=skip_null_blocks,
                cache_role=cache_role,
            )

        def iter_with_legacy_process_tokens():
            try:
                token_iter = self.token_database.process_tokens(token_len, block_hashes, mask_num)
            except TypeError:
                token_iter = self.token_database.process_tokens(token_len, block_hashes)
            group_block_size = self._get_block_size(kv_cache_group_id)
            for start, end, key in token_iter:
                block_idx = start // group_block_size
                if block_idx >= len(block_ids):
                    continue
                block_id = block_ids[block_idx]
                if skip_null_blocks and cache_role == "kv" and block_id <= 0:
                    continue
                yield start, end, key, block_id

        return iter_with_legacy_process_tokens()

    def _prepare_value(
        self,
        start: int,
        end: int,
        block_ids: list[int],
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
        block_id: int | None = None,
    ):
        try:
            return self.token_database.prepare_value(
                start,
                end,
                block_ids,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
                block_id=block_id,
            )
        except TypeError:
            return self.token_database.prepare_value(start, end, block_ids)

    def _process_transfer_chunks_with_block_ids(
        self,
        token_len: int,
        block_hashes,
        block_ids: list[int],
        mask_num: int = 0,
        kv_cache_group_id: int = 0,
        skip_null_blocks: bool = False,
        cache_role: str = "kv",
    ):
        process_transfer_chunks = getattr(
            self.token_database,
            "process_transfer_chunks_with_block_ids",
            None,
        )
        if process_transfer_chunks is not None:
            return process_transfer_chunks(
                token_len,
                block_hashes,
                block_ids,
                mask_num,
                kv_cache_group_id=kv_cache_group_id,
                skip_null_blocks=skip_null_blocks,
                cache_role=cache_role,
            )

        def iter_legacy_chunks():
            group_block_size = self._get_block_size(kv_cache_group_id)
            for start, end, key, block_id in self._process_tokens_with_block_ids(
                token_len,
                block_hashes,
                block_ids,
                mask_num,
                kv_cache_group_id=kv_cache_group_id,
                skip_null_blocks=skip_null_blocks,
                cache_role=cache_role,
            ):
                yield TransferChunkWithBlockId(
                    raw_start=start,
                    raw_end=end,
                    value_start=start,
                    value_end=end,
                    target_block_index=start // group_block_size,
                    key=key,
                    block_id=block_id,
                )

        return iter_legacy_chunks()

    def _prepare_transfer_value(
        self,
        chunk: TransferChunkWithBlockId,
        block_ids: list[int],
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
    ):
        prepare_transfer_value = getattr(self.token_database, "prepare_transfer_value", None)
        if prepare_transfer_value is not None:
            return prepare_transfer_value(
                chunk,
                block_ids,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
            )
        return self._prepare_value(
            chunk.value_start,
            chunk.value_end,
            block_ids,
            kv_cache_group_id=kv_cache_group_id,
            cache_role=cache_role,
            block_id=chunk.block_id,
        )

    def _decode_adaptor_prefill_pp(
        self,
        keys: list[str],
        addrs: list[list[int]],
        sizes: list[list[int]],
        kv_cache_group_id: int = 0,
        cache_role: str = "kv",
    ):
        try:
            return self.token_database.decode_adaptor_prefill_pp(
                keys,
                addrs,
                sizes,
                kv_cache_group_id=kv_cache_group_id,
                cache_role=cache_role,
            )
        except TypeError:
            return self.token_database.decode_adaptor_prefill_pp(keys, addrs, sizes)

    def _store_mask(self, req_meta: ReqMeta) -> tuple[list[bool], ...] | None:
        store_mask = getattr(self.token_database, "store_mask", None)
        if store_mask is None:
            return None
        try:
            return store_mask(req_meta.token_len_chunk, req_meta.num_prompt_tokens)
        except AssertionError as exc:
            logger.debug("Skip AscendStore store mask for unaligned request %s: %s", req_meta.req_id, exc)
            return None

    def _load_mask(self, req_meta: ReqMeta, token_len: int) -> tuple[list[bool], ...] | None:
        load_mask = getattr(self.token_database, "load_mask", None)
        if load_mask is None:
            return None
        return load_mask(req_meta.block_hashes, token_len)

    def _mask_allows_chunk(
        self,
        masks: tuple[list[bool], ...] | None,
        group_id: int,
        start: int,
    ) -> bool:
        mask_allows_chunk = getattr(self.token_database, "mask_allows_chunk", None)
        if mask_allows_chunk is None:
            return True
        return mask_allows_chunk(masks, group_id, start)


class KVCacheStoreSendingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        put_step: int,
        kv_role: str,
        ready_event: threading.Event,
        group_uses_align_state: list[bool],
        enable_kv_event: bool = False,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheSendingThread"
        )
        self.put_step = put_step
        self.kv_role = kv_role
        self.stored_requests = defaultdict[str, int](int)
        self.group_uses_align_state = group_uses_align_state
        self.enable_kv_event = enable_kv_event
        self.completed_events_lock = threading.Lock()
        self.completed_events: dict[int, int] = {}

    def lookup_store_keys(self, keys: list[str], kv_cache_group_id: int = 0) -> list[bool] | None:
        """Return existence in the logical Decode-key domain.

        A Decode consumer may expand one logical block into one key per
        Prefill PP rank. The block is reusable only when every expanded key is
        present; checking only ``@pp_rank:0`` would turn a partial PUT into a
        false hit and permanently skip the missing partitions on retry.
        Return None after logging malformed results; callers must fail the job.
        """
        partitions = getattr(self.token_database, "partitions", None)
        if self.kv_role != "kv_consumer" or not keys or partitions is None or len(partitions) <= 1:
            return [bool(value) for value in self.lookup(keys)]

        empty_buffers: list[list[int]] = [[] for _ in keys]
        expanded_keys, _, _ = self._decode_adaptor_prefill_pp(
            keys,
            empty_buffers,
            empty_buffers,
            kv_cache_group_id=kv_cache_group_id,
        )
        if len(expanded_keys) % len(keys) != 0:
            logger.error(
                "Prefill-PP key adaptor expanded %s logical keys to an invalid count %s", len(keys), len(expanded_keys)
            )
            return None
        partitions_per_key = len(expanded_keys) // len(keys)
        if partitions_per_key == 0:
            logger.error("Prefill-PP key adaptor returned no store keys")
            return None
        expanded_exists = self.lookup(expanded_keys)
        if len(expanded_exists) != len(expanded_keys):
            logger.error(
                "KV store returned %s lookup results for %s Prefill-PP keys", len(expanded_exists), len(expanded_keys)
            )
            return None
        return [
            all(expanded_exists[start : start + partitions_per_key])
            for start in range(0, len(expanded_exists), partitions_per_key)
        ]

    def add_stored_request(self, req_meta: ReqMeta) -> None:
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]

    def mark_completed_events(self, event_id: int | None) -> None:
        if event_id is not None:
            with self.completed_events_lock:
                self.completed_events[event_id] = 1

    def get_saved_offset(self, req_id: str) -> int:
        with self.done_task_lock:
            return self._saved_offsets.get(req_id, 0)

    def record_saved_offset(self, req_meta: ReqMeta, token_len: int) -> None:
        with self.done_task_lock:
            if req_meta.store_job_id in self.stored_requests.get(req_meta.req_id, ()):
                self._saved_offsets[req_meta.req_id] = max(self._saved_offsets.get(req_meta.req_id, 0), token_len)

    def reset_saved_request(self, req_id: str) -> None:
        with self.done_task_lock:
            self._saved_offsets.pop(req_id, None)
            self._retry_token_ids.pop(req_id, None)

    def get_event_token_ids(self, req_meta: ReqMeta, start: int, end: int) -> list[int] | None:
        if req_meta.token_ids is None:
            return None
        token_ids_start = req_meta.save_start_token
        token_ids = req_meta.token_ids
        with self.done_task_lock:
            if req_meta.store_job_id not in self.stored_requests.get(req_meta.req_id, ()):
                return []
            retry = self._retry_token_ids.get(req_meta.req_id)
            if retry is not None:
                retry_start, retry_ids = retry
                if retry_start + len(retry_ids) == token_ids_start:
                    token_ids_start = retry_start
                    token_ids = retry_ids + token_ids
        token_ids_end = token_ids_start + len(token_ids)
        if start < token_ids_start or end > token_ids_end:
            return []
        return token_ids[start - token_ids_start : end - token_ids_start]

    def _remember_retry_token_ids(self, req_meta: ReqMeta) -> None:
        if not self.enable_kv_event or req_meta.token_ids is None:
            return
        token_ids_start = req_meta.save_start_token
        token_ids = req_meta.token_ids.copy()
        with self.done_task_lock:
            if req_meta.store_job_id not in self.stored_requests.get(req_meta.req_id, ()):
                return
            retry = self._retry_token_ids.get(req_meta.req_id)
            if retry is not None:
                retry_start, retry_ids = retry
                if retry_start + len(retry_ids) == token_ids_start:
                    token_ids_start = retry_start
                    token_ids = retry_ids + token_ids
            self._retry_token_ids[req_meta.req_id] = (token_ids_start, token_ids)

    def _clear_retry_token_ids(self, req_meta: ReqMeta) -> None:
        with self.done_task_lock:
            if req_meta.store_job_id in self.stored_requests.get(req_meta.req_id, ()):
                self._retry_token_ids.pop(req_meta.req_id, None)

    def get_completed_events(self):
        with self.completed_events_lock:
            if not self.completed_events:
                return None
            completed_events = self.completed_events.copy()
            self.completed_events.clear()
        return completed_events

    def _handle_request(self, req_meta: ReqMeta):
        token_len = req_meta.token_len_chunk
        req_id = req_meta.req_id
        current_event = req_meta.current_event
        try:
            if req_id not in self.stored_requests:
                self.request_queue.task_done()
                return

            store_masks = self._store_mask(req_meta)
            for group_id in req_meta.kv_cache_group_ids or [0]:
                block_ids = req_meta.block_ids_by_group[group_id]
                group_block_size = self._get_block_size(group_id)
                transfer_config = getattr(self.token_database, "hybrid_cache_c128_config", None)
                event_granularity = (
                    transfer_config.chunk_tokens
                    if transfer_config is not None and transfer_config.enabled
                    else group_block_size
                )
                assert event_granularity is not None
                group_block_hashes = get_block_hashes(
                    req_meta.block_hashes,
                    event_granularity,
                    getattr(self.token_database, "hash_block_size", group_block_size),
                )
                chunks = [
                    chunk
                    for chunk in self._process_transfer_chunks_with_block_ids(
                        token_len,
                        req_meta.block_hashes,
                        block_ids,
                        kv_cache_group_id=group_id,
                        skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
                    )
                    if self._mask_allows_chunk(store_masks, group_id, chunk.raw_start)
                ]

                if (
                    not self.dcp_size > 1
                    and not req_meta.disable_tp_key_sharding
                    and not self.group_uses_align_state[group_id]
                ):
                    chunks = chunks[self.tp_rank % self.put_step :: self.put_step]

                if not chunks:
                    continue

                keys = [chunk.key.to_string() for chunk in chunks]
                exists_states = self.lookup(keys)
                missing_indices = [index for index, exists in enumerate(exists_states) if not exists]

                if not missing_indices:
                    continue

                chunks = [chunks[index] for index in missing_indices]
                keys = [keys[index] for index in missing_indices]

                logger.info(
                    "Storing KV cache for %d out of %d blocks (missing_count=%d) for request %s in group %d",
                    len(keys),
                    token_len // group_block_size,
                    len(missing_indices),
                    req_id,
                    group_id,
                )
                addrs = []
                sizes = []
                stored_events: list[BlockStored] = []
                prev_key = None
                new_block_hashes = [
                    maybe_convert_block_hash(
                        group_block_hashes[chunk.raw_start // event_granularity]
                    )
                    for chunk in chunks
                ]
                for index, chunk in enumerate(chunks):
                    addr, size, _ = self._prepare_transfer_value(
                        chunk,
                        block_ids,
                        kv_cache_group_id=group_id,
                    )
                    addrs.append(addr)
                    sizes.append(size)

                    # Create KV event
                    if self.enable_kv_event:
                        token_ids = (
                            req_meta.token_ids[chunk.raw_start : chunk.raw_end]
                            if req_meta.token_ids is not None
                            else None
                        )
                        block_size = (
                            transfer_config.chunk_tokens
                            if transfer_config is not None and transfer_config.enabled
                            else (
                                req_meta.original_block_size[group_id]
                                if isinstance(req_meta.original_block_size, list)
                                else req_meta.original_block_size
                            )
                        )
                        if block_size is not None:
                            stored_event = BlockStored(
                                block_hashes=[new_block_hashes[index]],
                                parent_block_hash=prev_key,
                                token_ids=token_ids,
                                block_size=block_size,
                                lora_id=None,
                                medium="cpu",
                                lora_name=None,
                            )
                            stored_events.append(stored_event)
                            prev_key = new_block_hashes[index]
                            logger.debug("Added kv cache event '%s' to kv cache events queue", stored_event)

                if self.kv_role == "kv_consumer":
                    keys, addrs, sizes = self._decode_adaptor_prefill_pp(
                        keys,
                        addrs,
                        sizes,
                        kv_cache_group_id=group_id,
                    )

                if current_event is not None:
                    current_event.synchronize()
                self.m_store.put(keys, addrs, sizes)

                # TODO Query specific replica info to update the event
                if self.enable_kv_event and stored_events is not None:
                    self.update_kv_event(stored_events)
        finally:
            # always free blocks
            self.mark_completed_events(req_meta.event_id)
        self.dec_stored_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        ready_event: threading.Event,
        invalid_block_ids: set[int],
        invalid_block_ids_lock: threading.Lock,
        get_c128_staging_value: Callable[[int], tuple[list[int], list[int]]] | None = None,
        merge_c128_staging_chunk: Callable[[int, TransferChunkWithBlockId], None] | None = None,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheStoreRecvingThread"
        )
        self._invalid_block_ids = invalid_block_ids
        self._invalid_block_ids_lock = invalid_block_ids_lock
        self._get_c128_staging_value = get_c128_staging_value
        self._merge_c128_staging_chunk = merge_c128_staging_chunk

    def _handle_request(self, req_meta: ReqMeta):
        transfer_config = getattr(self.token_database, "hybrid_cache_c128_config", None)
        if transfer_config is not None and transfer_config.enabled:
            self._handle_hybrid_c128_request(req_meta)
            return

        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        req_id = req_meta.req_id
        addr_list = []
        size_list = []
        key_list = []
        block_id_list: list[int] = []
        group_ids = req_meta.kv_cache_group_ids or [0]
        load_masks = self._load_mask(req_meta, token_len)
        for group_id in group_ids:
            block_ids = req_meta.block_ids_by_group[group_id]
            group_block_size = self._get_block_size(group_id)
            mask_num = (
                req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
                // group_block_size
                * group_block_size
            )
            for start, end, key, block_id in self._process_tokens_with_block_ids(
                token_len,
                req_meta.block_hashes,
                block_ids,
                mask_num,
                kv_cache_group_id=group_id,
                skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
            ):
                if not self._mask_allows_chunk(load_masks, group_id, start):
                    continue
                addr, size, block_id = self._prepare_value(
                    start,
                    end,
                    block_ids,
                    kv_cache_group_id=group_id,
                    block_id=block_id,
                )
                key_list.append(key.to_string())
                addr_list.append(addr)
                size_list.append(size)
                block_id_list.append(block_id)
        if not key_list:
            self.set_finished_request(req_id)
            self.request_queue.task_done()
            return
        key_list_c = key_list[self.tp_rank % len(key_list) :] + key_list[: self.tp_rank % len(key_list)]
        addr_list_c = addr_list[self.tp_rank % len(addr_list) :] + addr_list[: self.tp_rank % len(addr_list)]
        size_list_c = size_list[self.tp_rank % len(size_list) :] + size_list[: self.tp_rank % len(size_list)]
        block_id_list_c = (
            block_id_list[self.tp_rank % len(block_id_list) :] + block_id_list[: self.tp_rank % len(block_id_list)]
        )
        logger.debug(
            "KV pool async recv calls backend get request=%s token_len=%d groups=%s keys=%d sample_keys=%s",
            req_id,
            token_len,
            req_meta.kv_cache_group_ids or [0],
            len(key_list_c),
            key_list_c[:3],
        )
        ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
        if ret is not None and any(r != 0 for r in ret):
            missing_block_ids = record_failed_blocks(
                block_id_list_c,
                ret,
            )
            if len(req_meta.block_ids_by_group) == 1:
                with self._invalid_block_ids_lock:
                    self._invalid_block_ids.update(missing_block_ids)
            elif missing_block_ids:
                logger.error(
                    "KV load failed for hybrid request %s. "
                    "Skip invalid-block fallback to avoid scheduler crash. "
                    "failed_blocks=%s",
                    req_id,
                    missing_block_ids,
                )
        elif ret is None:
            missing_block_ids = record_failed_blocks(
                block_id_list_c,
                [1] * len(block_id_list_c),
            )
            if len(req_meta.block_ids_by_group) == 1:
                with self._invalid_block_ids_lock:
                    self._invalid_block_ids.update(missing_block_ids)
            elif missing_block_ids:
                logger.error(
                    "KV load failed for hybrid request %s. "
                    "Skip invalid-block fallback to avoid scheduler crash. "
                    "failed_blocks=%s",
                    req_id,
                    missing_block_ids,
                )
        logger.debug(
            "KV pool async recv backend get returned request=%s token_len=%d groups=%s keys=%d",
            req_id,
            token_len,
            req_meta.kv_cache_group_ids or [0],
            len(key_list_c),
        )
        self.set_finished_request(req_id)
        self.request_queue.task_done()

    def _record_load_failures(
        self,
        req_meta: ReqMeta,
        block_ids: list[int],
        results: list[int] | None,
    ) -> None:
        if results is None:
            results = [1] * len(block_ids)
        if not any(result != 0 for result in results):
            return
        missing_block_ids = record_failed_blocks(block_ids, results)
        if len(req_meta.block_ids_by_group) == 1:
            with self._invalid_block_ids_lock:
                self._invalid_block_ids.update(missing_block_ids)
        elif missing_block_ids:
            logger.error(
                "KV load failed for hybrid request %s. "
                "Skip invalid-block fallback to avoid scheduler crash. "
                "failed_blocks=%s",
                req_meta.req_id,
                missing_block_ids,
            )

    def _handle_hybrid_c128_request(self, req_meta: ReqMeta) -> None:
        transfer_config = self.token_database.hybrid_cache_c128_config
        assert transfer_config.chunk_tokens is not None

        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        req_id = req_meta.req_id
        group_ids = req_meta.kv_cache_group_ids or [0]
        load_masks = self._load_mask(req_meta, token_len)
        addr_list: list[list[int]] = []
        size_list: list[list[int]] = []
        key_list: list[str] = []
        block_id_list: list[int] = []
        c128_chunks: list[tuple[int, TransferChunkWithBlockId]] = []
        mask_num = (
            req_meta.load_spec.vllm_cached_tokens  # type: ignore[union-attr]
            // transfer_config.chunk_tokens
            * transfer_config.chunk_tokens
        )

        for group_id in group_ids:
            if group_id >= len(req_meta.block_ids_by_group):
                continue
            block_ids = req_meta.block_ids_by_group[group_id]
            for chunk in self._process_transfer_chunks_with_block_ids(
                token_len,
                req_meta.block_hashes,
                block_ids,
                mask_num,
                kv_cache_group_id=group_id,
                skip_null_blocks=self._skip_null_blocks(req_meta, group_id),
            ):
                if not self._mask_allows_chunk(load_masks, group_id, chunk.raw_start):
                    continue
                if self.token_database.is_c128_group(group_id):
                    c128_chunks.append((group_id, chunk))
                    continue
                addr, size, block_id = self._prepare_transfer_value(
                    chunk,
                    block_ids,
                    kv_cache_group_id=group_id,
                )
                key_list.append(chunk.key.to_string())
                addr_list.append(addr)
                size_list.append(size)
                block_id_list.append(block_id)

        if not key_list and not c128_chunks:
            self.set_finished_request(req_id)
            self.request_queue.task_done()
            return

        if key_list:
            rotation = self.tp_rank % len(key_list)
            key_list = key_list[rotation:] + key_list[:rotation]
            addr_list = addr_list[rotation:] + addr_list[:rotation]
            size_list = size_list[rotation:] + size_list[:rotation]
            block_id_list = block_id_list[rotation:] + block_id_list[:rotation]
            results = self.m_store.get(key_list, addr_list, size_list)
            self._record_load_failures(req_meta, block_id_list, results)

        c128_groups: dict[int, list[TransferChunkWithBlockId]] = defaultdict(list)
        for group_id, chunk in c128_chunks:
            c128_groups[group_id].append(chunk)
        for group_id, chunks in c128_groups.items():
            page_chunks = aggregate_c128_page_chunks(chunks, transfer_config.c128_slots_per_page)
            page_keys: list[str] = []
            page_addrs: list[list[int]] = []
            page_sizes: list[list[int]] = []
            page_block_ids: list[int] = []
            for chunk in page_chunks:
                block_ids = list(chunk.block_ids) or [chunk.block_id]
                addr, size, block_id = self._prepare_transfer_value(
                    chunk,
                    block_ids,
                    kv_cache_group_id=group_id,
                )
                if not addr:
                    continue
                page_keys.append(chunk.key.to_string())
                page_addrs.append(addr)
                page_sizes.append(size)
                page_block_ids.append(block_id)
            if not page_keys:
                continue
            rotation = self.tp_rank % len(page_keys)
            page_keys = page_keys[rotation:] + page_keys[:rotation]
            page_addrs = page_addrs[rotation:] + page_addrs[:rotation]
            page_sizes = page_sizes[rotation:] + page_sizes[:rotation]
            page_block_ids = page_block_ids[rotation:] + page_block_ids[:rotation]
            results = self.m_store.get(page_keys, page_addrs, page_sizes)
            self._record_load_failures(req_meta, page_block_ids, results)

        self.set_finished_request(req_id)
        self.request_queue.task_done()


class KVCacheStoreLayerSendingThread(KVCacheStoreLayerSendingThreadBase):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        put_step: int,
        ready_event: threading.Event,
        num_layers: int,
        enable_kv_event: bool = False,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheStoreLayerSendingThread"
        )
        self.final_layer_id = num_layers - 1
        self.put_step = put_step
        self.enable_kv_event = enable_kv_event
        self.layerwise_event_starts: dict[str, set[int]] = defaultdict(set)
        self.stored_requests: dict[str, int] = defaultdict(int)
        self.done_task_lock = threading.Lock()
        self.layerwise_event_lock = threading.Lock()

    def add_stored_request(self, req_id: str):
        with self.done_task_lock:
            self.stored_requests[req_id] += 1

    def dec_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                self.stored_requests[req_id] -= 1

    def delete_finished_stored_request(self, req_id: str):
        with self.done_task_lock:
            if req_id in self.stored_requests:
                del self.stored_requests[req_id]
        with self.layerwise_event_lock:
            self.layerwise_event_starts.pop(req_id, None)

    def _record_layerwise_event_starts(self, req_meta: LayerMultiBlockReqMeta, starts: list[int]) -> None:
        if self.enable_kv_event:
            with self.layerwise_event_lock:
                self.layerwise_event_starts[req_meta.req_id].update(starts)

    def _build_stored_events(self, req_meta: LayerMultiBlockReqMeta) -> list[BlockStored]:
        if not self.enable_kv_event or req_meta.layer_id != self.final_layer_id:
            return []
        block_size = (
            req_meta.original_block_size[req_meta.kv_cache_group_id]
            if isinstance(req_meta.original_block_size, list)
            else req_meta.original_block_size
        )
        if block_size is None:
            return []
        stored_events: list[BlockStored] = []
        group_block_size = self._get_block_size(req_meta.kv_cache_group_id)
        new_block_hashes = [maybe_convert_block_hash(bh) for bh in req_meta.block_hashes]
        with self.layerwise_event_lock:
            starts_set = self.layerwise_event_starts.pop(req_meta.req_id, set())
        for start in sorted(starts_set):
            block_idx = start // group_block_size
            if block_idx >= len(new_block_hashes):
                continue
            block_hash = new_block_hashes[block_idx]
            parent_block_hash = new_block_hashes[block_idx - 1] if block_idx > 0 else None
            end = min(start + group_block_size, len(req_meta.token_ids or []))
            token_ids = req_meta.token_ids[start:end] if req_meta.token_ids is not None else None
            stored_event = BlockStored(
                block_hashes=[block_hash],
                parent_block_hash=parent_block_hash,
                token_ids=token_ids,
                block_size=block_size,
                lora_id=None,
                medium="cpu",
                lora_name=None,
            )
            stored_events.append(stored_event)
            logger.debug("Added layerwise kv cache event '%s' to kv cache events queue", stored_event)
        return stored_events

    def add_request(  # type: ignore[override]
        self, req_meta: ReqMeta
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(  # type: ignore[override]
        self, req_meta: LayerMultiBlockReqMeta
    ):
        starts = req_meta.starts
        ends = req_meta.ends
        keys = req_meta.keys
        layer_id = req_meta.layer_id
        current_event = req_meta.current_event
        total_block = len(keys)
        is_last_chunk = req_meta.is_last_chunk
        with self.done_task_lock:
            if req_meta.req_id not in self.stored_requests:
                self.request_queue.task_done()
                return
        if not self.dcp_size > 1:
            starts = starts[self.tp_rank % self.put_step :: self.put_step]
            ends = ends[self.tp_rank % self.put_step :: self.put_step]
            keys = keys[self.tp_rank % self.put_step :: self.put_step]

        if not keys:
            if layer_id == self.final_layer_id:
                stored_events = self._build_stored_events(req_meta)
                if stored_events:
                    self.update_kv_event(stored_events)
            if is_last_chunk and layer_id == self.final_layer_id:
                self.dec_stored_request(req_meta.req_id)
                self.set_finished_request(req_meta.req_id)
            self.request_queue.task_done()
            return

        key_list = []
        for key in keys:
            key_list.append(key.to_string())

        exists_states = self.lookup(key_list)
        missing_indices = [index for index, exists in enumerate(exists_states) if not exists]

        if not missing_indices:
            if layer_id == self.final_layer_id:
                stored_events = self._build_stored_events(req_meta)
                if stored_events:
                    self.update_kv_event(stored_events)
            if is_last_chunk and layer_id == self.final_layer_id:
                self.dec_stored_request(req_meta.req_id)
                self.set_finished_request(req_meta.req_id)
            self.request_queue.task_done()
            return

        starts = [starts[index] for index in missing_indices]
        ends = [ends[index] for index in missing_indices]
        key_list = [key_list[index] for index in missing_indices]

        addr_list = []
        size_list = []
        for index, key in enumerate(key_list):
            addr, size, _ = self.token_database.prepare_value_layer(
                starts[index], ends[index], req_meta.block_ids_by_group[0], layer_id
            )
            addr_list.append(addr)
            size_list.append(size)

        if current_event is not None:
            current_event.synchronize()
        self.m_store.put(key_list, addr_list, size_list)
        self._record_layerwise_event_starts(req_meta, starts)
        stored_events = self._build_stored_events(req_meta)
        if stored_events:
            self.update_kv_event(stored_events)

        if layer_id == self.final_layer_id and is_last_chunk:
            with self.layerwise_event_lock:
                self.layerwise_event_starts.pop(req_meta.req_id, None)
            self.dec_stored_request(req_meta.req_id)
            self.set_finished_request(req_meta.req_id)
        self.request_queue.task_done()

        if layer_id == self.final_layer_id:
            logger.info(
                "Storing KV cache layerwise for %d out of %d blocks (missing_count=%d) for request %s, layer %d",
                len(key_list),
                total_block,
                len(missing_indices),
                req_meta.req_id,
                layer_id,
            )
        else:
            logger.debug(
                "Storing KV cache layerwise for %d out of %d blocks (missing_count=%d) for request %s, layer %d",
                len(key_list),
                total_block,
                len(missing_indices),
                req_meta.req_id,
                layer_id,
            )


class KVCacheStoreLayerRecvingThread(KVTransferThread):
    def __init__(
        self,
        m_store: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int | list[int],
        tp_rank: int,
        dcp_size: int,
        ready_event: threading.Event,
        get_event: threading.Event,
        invalid_block_ids: set[int],
        invalid_block_ids_lock: threading.Lock,
    ):
        super().__init__(
            m_store, token_database, block_size, tp_rank, dcp_size, ready_event, name="KVCacheStoreLayerRecvingThread"
        )
        self.get_event = get_event
        self._invalid_block_ids = invalid_block_ids
        self._invalid_block_ids_lock = invalid_block_ids_lock

    def add_request(  # type: ignore[override]
        self, req_meta: LayerMultiBlockReqMeta
    ) -> torch.Tensor:
        self.request_queue.put(req_meta)

    def _handle_request(  # type: ignore[override]
        self, req_meta: LayerMultiBlockReqMeta
    ):
        addr_list = []
        size_list = []
        key_list = []
        block_id_list = []
        for index, key in enumerate(req_meta.keys):
            addr, size, block_id = self.token_database.prepare_value_layer(
                req_meta.starts[index], req_meta.ends[index], req_meta.block_ids_by_group[0], req_meta.layer_id
            )
            key_list.append(key.to_string())
            addr_list.append(addr)
            size_list.append(size)
            block_id_list.append(block_id)

        offset = self.tp_rank % len(key_list)
        key_list_c = key_list[offset:] + key_list[:offset]
        addr_list_c = addr_list[offset:] + addr_list[:offset]
        size_list_c = size_list[offset:] + size_list[:offset]
        block_id_list_c = block_id_list[offset:] + block_id_list[:offset]
        ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)

        if ret is not None and any(r != 0 for r in ret):
            missing_block_ids = record_failed_blocks(
                block_id_list_c,
                ret,
            )
            with self._invalid_block_ids_lock:
                self._invalid_block_ids.update(missing_block_ids)
        elif ret is None:
            with self._invalid_block_ids_lock:
                self._invalid_block_ids.update(block_id_list_c)

        self.request_queue.task_done()
        self.get_event.set()


def record_failed_blocks(
    block_ids: list[int],
    ret_codes: list[int],
) -> set[int]:
    failed_blocks: set[int] = set()
    for block_id, code in zip(block_ids, ret_codes):
        if code != 0:
            failed_blocks.add(block_id)
    if failed_blocks:
        logger.error(
            "Failed to load blocks. failed_count=%d, failed_blocks=%s. Check block availability and memory state.",
            len(failed_blocks),
            failed_blocks,
        )
    return failed_blocks
