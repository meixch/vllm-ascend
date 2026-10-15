# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Print-level precision probes for the DeepSeek V4.1 A5 operator stack.

Enable with ``VLLM_ASCEND_PRECISION_PROBE=1``. Every probe prints one line with
a stable tag plus shape/dtype/min/max/mean/sum/head-value summaries of the
tensors crossing a logical pipeline boundary, prefixed by a monotonically
increasing call counter. Run the identical workload once before and once after
swapping operators to the dsnext implementations, then diff the two logs by
counter and tag.

Stable tags and their boundaries:

- ``IDX_K_LATENT``/``IDX_K_PROJ``/``IDX_K_ROPE``/``IDX_K_CACHE_IN``/
  ``IDX_K_CACHE_OUT``/``IDX_K_FOLD_OUT``: index K pipeline
  (raw latent -> wk projection + norm -> partial RoPE -> separated cache
  planes -> folded twin). The fused ``indexer_prologue_k`` replacement must
  reproduce every intermediate by the ``IDX_K_CACHE_OUT``/``IDX_K_FOLD_OUT``
  boundaries.
- ``IDX_Q_PROJ``/``IDX_Q_ROPE``/``IDX_Q_WEIGHTS``/``IDX_Q_QUANT``: index query
  pipeline (wq_b projection -> partial RoPE -> weights -> MXFP4 quantization).
- ``QLI_META_CONSUME``/``QLI_IN``/``QLI_OUT``/``IDX_SELECT_POST``: QLI.
- ``QSLI_META_IN``/``QSLI_META_OUT``/``QSLI_IN``/``QSLI_OUT``: QSLI.
- ``SMLA_META_IN``/``SMLA_META_OUT``/``SMLA_META_CONSUME``/``SMLA_IN``/
  ``SMLA_OUT``: mixed-quant sparse flash MLA (MQSMLA).
- ``KV_WRITE_IN``/``KV_WRITE_OUT``: quantized win/cmp KV cache writes.
- ``CMP_R1_IN``/``CMP_R1_OUT``/``CMP_R2_IN``/``CMP_R2_POOLED``/``CMP_R2_OUT``:
  C1 projection and C2 ring compressor.
- ``HC_PRE_IN``/``HC_PRE_OUT``/``HC_POST_IN``/``HC_POST_OUT``/
  ``HC_COLLAPSE_IN``/``HC_COLLAPSE_OUT``: manifold hyper-connections; the
  decoder layer index is logged as the first value.
- ``RMS_NORM_IN``/``RMS_NORM_OUT``: fused RMSNorm cast feeding MoE routing.

The probe is eager-mode oriented: under ACLGraph capture the Python prints only
execute at capture time, so run the probe workload with cudagraph_mode EAGER.
"""

import torch

_ENABLED: bool | None = None
_COUNTER = 0
PROBE_HEAD_ELEMENTS = 8


def _enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        from vllm_ascend.envs import env_variables

        _ENABLED = bool(env_variables["VLLM_ASCEND_PRECISION_PROBE"]())
        if _ENABLED:
            print("[PP #000000] PRECISION_PROBE enabled", flush=True)
    return _ENABLED


def _head(t: torch.Tensor, n: int = PROBE_HEAD_ELEMENTS) -> list | None:
    """Sample the first ``n`` elements without materializing a full copy."""
    try:
        if t.is_contiguous():
            sample = t.view(-1)[:n]
        else:
            block = tuple(slice(0, min(size, 2)) for size in t.shape)
            sample = t[block].reshape(-1)[:n]
        if sample.is_floating_point():
            return sample.to(torch.float32).tolist()
        return sample.tolist()
    except Exception:  # noqa: BLE001
        return None


def _describe(value) -> str:
    if value is None:
        return "None"
    if not isinstance(value, torch.Tensor):
        return f"{type(value).__name__}={value!r}"[:160]
    header = f"tensor(shape={tuple(value.shape)}, dtype={value.dtype}"
    try:
        t = value.detach()
        if t.numel() == 0:
            return header + ", empty)"
        head = _head(t)
        # Reduce on the original tensor with an explicit accumulator dtype so
        # no full-size temporary is ever allocated (KV caches are huge and the
        # NPU is already nearly full from cache allocation).
        if t.is_floating_point():
            total = t.sum(dtype=torch.float32).item()
            stats = (
                f"min={t.min().item():.6g}, max={t.max().item():.6g}"
                + f", mean={total / t.numel():.6g}, sum={total:.6g}"
            )
        else:
            total = t.sum(dtype=torch.int64).item()
            stats = f"min={t.min().item()}, max={t.max().item()}, sum={total}"
        return header + f", {stats}, head={head})"
    except Exception as exc:  # noqa: BLE001
        return header + f", probe-error={exc!r})"


def plog(tag: str, *values) -> None:
    """Print one probe line summarizing ``values`` at a stable boundary."""
    global _COUNTER
    if not _enabled():
        return
    _COUNTER += 1
    body = " | ".join(_describe(v) for v in values)
    print(f"[PP #{_COUNTER:06d}] {tag} :: {body}", flush=True)
