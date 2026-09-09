#!/usr/bin/env python3
"""Export allowlisted benchmark metrics for the public report.

Raw benchmark directories are intentionally gitignored. This exporter copies
only scalar measurements needed by the public charts and tables.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


PUBLIC_VARIANTS = {
    "uzu-m-spec": "Uzu Mirai-M",
    "omlx-optiq-dflash": "oMLX OptiQ + DFlash 4-bit",
    "omlx-mxfp4-dflash-q4": "oMLX MXFP4 + DFlash 4-bit",
    "omlx-mxfp4-vlm-mtp": "oMLX MXFP4 + VLM MTP 4-bit",
}
PUBLIC_STATUSES = {"ok", "ram_limit"}
CONTINUOUS_FIELDS = (
    "attempt",
    "round",
    "variant",
    "context_tokens",
    "output_tokens",
    "completion_tokens",
    "decode_tps",
    "ttft_seconds",
    "end_to_end_tps",
    "spec_acceptance_percent",
    "target_pass_reduction_percent",
    "tokens_per_spec_cycle",
    "tokens_per_target_verify",
    "status",
)
FULL_FIELDS = (
    "variant",
    "prompt_id",
    "prompt_tokens",
    "target_output_tokens",
    "completion_tokens",
    "repetition",
    "decode_tps",
    "ttft_seconds",
    "end_to_end_tps",
    "spec_acceptance_percent",
    "target_pass_reduction_percent",
    "tokens_per_spec_cycle",
    "tokens_per_target_verify",
    "system_memory_free_after_percent",
    "system_swap_delta_bytes",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def allowlist(row: dict[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    return {field: row.get(field) for field in fields}


def continuous_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    public_rows = [
        allowlist(row, CONTINUOUS_FIELDS)
        for row in rows
        if row.get("variant") in PUBLIC_VARIANTS
        and row.get("status") in PUBLIC_STATUSES
    ]
    public_rows.sort(key=lambda row: int(row["attempt"]))
    return {
        "schema_version": 1,
        "benchmark": {
            "model_family": "Qwen3.6-27B",
            "machine": "Apple M5 Mac, 32 GiB unified memory",
            "output_tokens": 512,
            "sampling": "temperature 0, top-p 1, top-k 1, thinking disabled",
            "cache_policy": "request, prefix, paged, SSD, and DFlash caches disabled",
            "snapshot_attempt": max(
                (int(row["attempt"]) for row in public_rows), default=None
            ),
        },
        "series": [
            {"id": variant, "label": label}
            for variant, label in PUBLIC_VARIANTS.items()
        ],
        "rows": public_rows,
    }


def write_full_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FULL_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(allowlist(row, FULL_FIELDS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--continuous-source",
        type=Path,
        default=Path("continuous-results/main/events.jsonl"),
    )
    parser.add_argument(
        "--continuous-output",
        type=Path,
        default=Path("docs/data/continuous-results.json"),
    )
    parser.add_argument(
        "--full-source",
        type=Path,
        default=Path("results/20260907-221333/results.jsonl"),
    )
    parser.add_argument(
        "--full-output",
        type=Path,
        default=Path("docs/data/full-benchmark.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = continuous_payload(read_jsonl(args.continuous_source))
    args.continuous_output.parent.mkdir(parents=True, exist_ok=True)
    args.continuous_output.write_text(json.dumps(payload, indent=2) + "\n")
    write_full_csv(args.full_output, read_jsonl(args.full_source))
    print(f"wrote {len(payload['rows'])} continuous rows to {args.continuous_output}")
    print(f"wrote full benchmark metrics to {args.full_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
