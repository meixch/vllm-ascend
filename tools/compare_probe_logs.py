#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare two DeepSeek V4.1 precision-probe logs (before/after operator swap).

Parses ``[PP #nnnnnn] TAG :: v1 | v2 | ...`` lines (optionally prefixed by a
vLLM worker tag), matches them by (worker, tag, occurrence index) and prints a
compact per-tag drift report so only the result needs to be transferred back.

Usage:
    python3 tools/compare_probe_logs.py /tmp/probe_before.log /tmp/probe_after.log \
        [--top 20] [--out /tmp/compare_report.txt]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

LINE_RE = re.compile(
    r"(?:\((?P<worker>Worker_\S+)\s+pid=\d+\)\s*)?\[PP\s+#(?P<cnt>\d+)\]\s+(?P<tag>\S+)\s+::\s?(?P<body>.*)"
)
TENSOR_RE = re.compile(r"^tensor\(shape=\((?P<shape>[^)]*)\),\s*dtype=(?P<dtype>[\w.]+),\s?(?P<rest>.*)\)$")
STAT_RE = re.compile(r"\b(min|max|mean|sum)=(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|-?inf|nan)")
SAMPLED_RE = re.compile(r"sampled=(\d+)/(\d+)")
HEAD_RE = re.compile(r"head=\[([^\]]*)\]")
SCALAR_RE = re.compile(r"^(?P<name>\w+)=(?P<value>.+)$", re.S)
INT_RE = re.compile(r"^-?\d+$")

# Structural changes that are direct consequences of the operator swap.
EXPECTED_ABSENT_AFTER = {
    "IDX_K_PROJ",  # wk projection + norm absorbed into indexer_prologue_k
    "IDX_K_ROPE",  # partial RoPE absorbed into indexer_prologue_k
}
EXPECTED_ARITY = {
    "IDX_K_CACHE_IN": "before(cache,scale,slots,key=4) -> after(cache,scale,slots=3)",
    "CMP_R2_IN": "before(kv,scores,ring_meta=3) -> after(hidden,ring_meta=2)",
}
# Tags whose content difference is expected and non-indicative; the
# downstream OUT tags are the authoritative comparison points.
INFORMATIONAL_TAGS = {
    "QLI_META_CONSUME",  # opaque launch-table blob, wrapper generation changed
    "QSLI_META_OUT",
    "SMLA_META_OUT",
    "SMLA_META_CONSUME",
    "IDX_Q_QUANT",  # quantizer changed: Triton -> npu_dynamic_mx_quant
}

TINY_REL = 1e-6  # below the %.6g printing resolution: indistinguishable
SMALL_REL = 1e-3


def parse_num(text: str):
    text = text.strip()
    if INT_RE.match(text):
        return int(text)
    return float(text)


def parse_value(text: str) -> dict:
    text = text.strip()
    if text == "None":
        return {"kind": "none"}
    m = TENSOR_RE.match(text)
    if m:
        rest = m.group("rest")
        value = {
            "kind": "tensor",
            "shape": tuple(int(x) for x in m.group("shape").split(",") if x.strip()),
            "dtype": m.group("dtype"),
        }
        if rest.startswith("empty"):
            value["empty"] = True
            return value
        if "probe-error=" in rest:
            value["error"] = True
            return value
        value["stats"] = {k: float(v) for k, v in STAT_RE.findall(rest)}
        sampled = SAMPLED_RE.search(rest)
        value["sampled"] = (int(sampled.group(1)), int(sampled.group(2))) if sampled else None
        head = HEAD_RE.search(rest)
        value["head"] = [parse_num(x) for x in head.group(1).split(",")] if head and head.group(1).strip() else []
        return value
    m = SCALAR_RE.match(text)
    if m:
        return {"kind": "scalar", "name": m.group("name"), "value": m.group("value")}
    return {"kind": "raw", "value": text}


def rel_diff(a: float, b: float) -> float:
    if a == b:
        return 0.0
    denom = max(abs(a), abs(b))
    if denom == 0:
        return float("inf")
    return abs(a - b) / denom


def compare_value(before: dict, after: dict) -> dict:
    """Compare one parsed value; returns drift info for the report."""
    out = {"cls": "EXACT", "max_rel": 0.0, "int_abs": 0, "detail": []}
    if before["kind"] != after["kind"]:
        out.update(cls="STRUCT", detail=[f"kind {before['kind']} vs {after['kind']}"])
        return out
    if before["kind"] == "none":
        return out
    if before["kind"] in ("scalar", "raw"):
        bv = before.get("value")
        av = after.get("value")
        if bv == av:
            return out
        try:
            rel = rel_diff(float(bv), float(av))
            out.update(cls="SMALL" if rel <= SMALL_REL else "LARGE", max_rel=rel, detail=[f"scalar {bv} -> {av}"])
        except (TypeError, ValueError):
            out.update(cls="TEXT", detail=[f"{bv} -> {av}"])
        return out
    # tensor
    if before.get("empty") or after.get("empty") or before.get("error") or after.get("error"):
        if before.get("empty") == after.get("empty") and before.get("error") == after.get("error"):
            out.update(cls="SKIP", detail=["empty/probe-error on both sides"])
        else:
            out.update(cls="STRUCT", detail=["empty/probe-error flags differ"])
        return out
    if before["shape"] != after["shape"] or before["dtype"] != after["dtype"]:
        out.update(
            cls="STRUCT",
            detail=[f"shape/dtype {before['shape']}/{before['dtype']} vs {after['shape']}/{after['dtype']}"],
        )
        return out
    if before.get("sampled") != after.get("sampled"):
        out["detail"].append(f"sampled prefix {before.get('sampled')} vs {after.get('sampled')}")
    for key in sorted(set(before["stats"]) | set(after["stats"])):
        vb = before["stats"].get(key)
        va = after["stats"].get(key)
        if vb is None or va is None:
            out["detail"].append(f"{key}: missing on one side")
            out.update(cls="STRUCT")
            continue
        if vb == va:
            continue
        if isinstance(vb, float) and vb.is_integer() and isinstance(va, float) and va.is_integer():
            out["int_abs"] = max(out["int_abs"], abs(vb - va))
            out["detail"].append(f"{key}: {vb:g} -> {va:g}")
        else:
            rel = rel_diff(vb, va)
            out["max_rel"] = max(out["max_rel"], rel)
            out["detail"].append(f"{key}: {vb:g} -> {va:g} (rel {rel:.2g})")
    hb, ha = before.get("head", []), after.get("head", [])
    if len(hb) != len(ha):
        out["detail"].append(f"head len {len(hb)} vs {len(ha)}")
    for i, (x, y) in enumerate(zip(hb, ha)):
        if x == y:
            continue
        if isinstance(x, int) and isinstance(y, int):
            out["int_abs"] = max(out["int_abs"], abs(x - y))
            out["detail"].append(f"head[{i}]: {x} -> {y}")
        else:
            rel = rel_diff(float(x), float(y))
            out["max_rel"] = max(out["max_rel"], rel)
            out["detail"].append(f"head[{i}]: {x:g} -> {y:g} (rel {rel:.2g})")
    if out["int_abs"] or out["max_rel"] > SMALL_REL:
        out["cls"] = "LARGE"
    elif out["max_rel"] > TINY_REL:
        out["cls"] = "SMALL"
    if before.get("sampled") != after.get("sampled") and out["cls"] == "EXACT":
        out["cls"] = "NOTE"
    return out


def classify_line(results: list[dict]) -> dict:
    order = {"STRUCT": 5, "LARGE": 4, "TEXT": 3, "NOTE": 2, "SMALL": 1, "SKIP": 0, "EXACT": 0}
    cls = "EXACT"
    for r in results:
        if order[r["cls"]] > order[cls]:
            cls = r["cls"]
    return {
        "cls": cls,
        "max_rel": max((r["max_rel"] for r in results), default=0.0),
        "int_abs": max((r["int_abs"] for r in results), default=0),
        "details": [d for r in results for d in r["detail"]][:6],
    }


def parse_log(path: str):
    seq: dict[tuple[str, str], list] = defaultdict(list)
    workers: set[str] = set()
    total = 0
    with open(path, errors="replace") as handle:
        for line in handle:
            m = LINE_RE.search(line)
            if not m:
                continue
            worker = m.group("worker") or "main"
            tag = m.group("tag")
            if tag == "PRECISION_PROBE":
                continue
            values = [parse_value(chunk) for chunk in m.group("body").split(" | ")]
            seq[(worker, tag)].append(values)
            workers.add(worker)
            total += 1
    return seq, workers, total


def compare_tag(before_seq: dict, after_seq: dict, workers: list[str], tag: str) -> dict:
    """Match occurrences per worker; leftover lines pair across workers."""
    per_worker = []
    leftover_before: list = []
    leftover_after: list = []
    for worker in workers:
        b = before_seq.get((worker, tag), [])
        a = after_seq.get((worker, tag), [])
        for idx in range(min(len(b), len(a))):
            per_worker.append((worker, idx, b[idx], a[idx]))
        leftover_before.extend([(worker, len(b) - len(a) + i, v) for i, v in enumerate(b[min(len(b), len(a)) :])])
        leftover_after.extend([(worker, len(a) - len(b) + i, v) for i, v in enumerate(a[min(len(b), len(a)) :])])
    cross = []
    if leftover_before or leftover_after:
        for i in range(min(len(leftover_before), len(leftover_after))):
            cross.append((leftover_before[i][0], f"+{i}", leftover_before[i][2], leftover_after[i][2]))
    unmatched = abs(len(leftover_before) - len(leftover_after))
    return {"pairs": per_worker + cross, "unmatched": unmatched}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", help="probe_before.log")
    parser.add_argument("after", help="probe_after.log")
    parser.add_argument("--top", type=int, default=20, help="number of drift lines to detail")
    parser.add_argument("--out", default="", help="also write the full report to this file")
    args = parser.parse_args()

    before_seq, before_workers, n_before = parse_log(args.before)
    after_seq, after_workers, n_after = parse_log(args.after)
    workers = sorted(before_workers | after_workers)
    tags = sorted({tag for _, tag in before_seq} | {tag for _, tag in after_seq})

    lines = []
    lines.append(f"before: {args.before}  lines={n_before}  workers={len(before_workers)}")
    lines.append(f"after : {args.after}  lines={n_after}  workers={len(after_workers)}")
    lines.append("")

    header = (
        f"{'tag':<22}{'n_b':>6}{'n_a':>6}{'exact':>7}{'small':>7}{'large':>7}"
        f"{'struct':>8}{'max_rel':>10}{'int_abs':>9}  note"
    )
    lines.append(header)
    lines.append("-" * len(header))

    drift_lines = []
    struct_patterns: dict[tuple, dict] = {}
    excluded_from_details = set(EXPECTED_ARITY) | INFORMATIONAL_TAGS | EXPECTED_ABSENT_AFTER
    for tag in tags:
        info = compare_tag(before_seq, after_seq, workers, tag)
        n_b = sum(len(before_seq.get((w, tag), [])) for w in workers)
        n_a = sum(len(after_seq.get((w, tag), [])) for w in workers)
        counts = {"EXACT": 0, "SMALL": 0, "LARGE": 0, "STRUCT": 0, "TEXT": 0, "NOTE": 0, "SKIP": 0}
        max_rel, int_abs = 0.0, 0
        for worker, idx, bvals, avals in info["pairs"]:
            if len(bvals) != len(avals):
                res = {
                    "cls": "STRUCT",
                    "max_rel": 0.0,
                    "int_abs": 0,
                    "details": [f"arity {len(bvals)} -> {len(avals)} ({EXPECTED_ARITY.get(tag, 'UNEXPECTED')})"],
                }
            else:
                res = classify_line([compare_value(x, y) for x, y in zip(bvals, avals)])
            counts[res["cls"]] += 1
            max_rel = max(max_rel, res["max_rel"])
            int_abs = max(int_abs, res["int_abs"])
            if res["cls"] == "STRUCT" and tag not in excluded_from_details:
                key = (tag, tuple(res["details"]))
                group = struct_patterns.setdefault(key, {"count": 0, "worker": worker, "idx": idx})
                group["count"] += 1
            elif res["cls"] in ("LARGE", "TEXT") and tag not in excluded_from_details and len(drift_lines) < 20000:
                drift_lines.append((res["cls"], tag, worker, idx, res["max_rel"], res["int_abs"], res["details"]))
        note = ""
        if tag in EXPECTED_ABSENT_AFTER:
            note = "expected absent after (fused)"
        elif tag in EXPECTED_ARITY:
            note = "arity change expected"
        elif tag in INFORMATIONAL_TAGS:
            note = "informational (generation diff)"
        if info["unmatched"]:
            note = (note + "; " if note else "") + f"unmatched={info['unmatched']}"
        if counts["SKIP"]:
            note = (note + "; " if note else "") + f"skip={counts['SKIP']} (probe-error/empty both sides)"
        lines.append(
            f"{tag:<22}{n_b:>6}{n_a:>6}{counts['EXACT']:>7}{counts['SMALL']:>7}{counts['LARGE']:>7}"
            f"{counts['STRUCT']:>8}{max_rel:>10.2g}{int_abs:>9g}  {note}"
        )

    lines.append("")
    lines.append("==== drift details (LARGE, worst first) ====")
    drift_lines.sort(key=lambda item: (0 if item[0] == "LARGE" else 1, -item[5], -item[4]))
    for cls, tag, worker, idx, rel, iabs, details in drift_lines[: args.top]:
        lines.append(f"[{cls}] {tag} w={worker} occ={idx} max_rel={rel:.2g} int_abs={iabs:g}")
        for d in details:
            lines.append(f"    {d}")

    lines.append("")
    lines.append("==== structural patterns (deduped) ====")
    for (tag, details), group in sorted(struct_patterns.items(), key=lambda item: -item[1]["count"])[
        : max(args.top, 20)
    ]:
        lines.append(f"[STRUCT x{group['count']}] {tag} e.g. w={group['worker']} occ={group['idx']}")
        for d in details:
            lines.append(f"    {d}")

    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(report + "\n")
        print(f"\nfull report written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
