#!/usr/bin/env python3
"""Archive selected continuous-benchmark rows without renumbering attempts."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import continuous_bench


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--campaign-dir", required=True)
    result.add_argument("--label", required=True)
    result.add_argument("--variants", required=True)
    result.add_argument("--reason", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    campaign = Path(args.campaign_dir).expanduser().resolve()
    events_path = campaign / "events.jsonl"
    events = continuous_bench.read_events(events_path)
    variants = set(args.variants.split(","))
    selected = [
        row
        for row in events
        if row.get("variant") in variants and row.get("status") != "archived"
    ]
    if not selected:
        raise SystemExit("no matching active events to archive")

    archive = campaign / "archive" / args.label
    archive.mkdir(parents=True, exist_ok=False)
    continuous_bench.atomic_write(
        archive / "events.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
    )
    continuous_bench.atomic_write(
        archive / "events.full-before.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in events),
    )
    if (campaign / "chart.html").is_file():
        shutil.copy2(campaign / "chart.html", archive / "chart-before.html")
    metadata = {
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "variants": sorted(variants),
        "rows": len(selected),
        "reason": args.reason,
        "raw_artifacts": "retained at the paths recorded in events.jsonl",
    }
    continuous_bench.atomic_write(
        archive / "archive.json", json.dumps(metadata, indent=2) + "\n"
    )

    archive_ref = str(archive / "events.jsonl")
    rewritten = []
    for row in events:
        if row in selected:
            rewritten.append(
                {
                    key: row[key]
                    for key in (
                        "attempt",
                        "round",
                        "slot",
                        "variant",
                        "context_tokens",
                        "output_tokens",
                        "sampling_segment",
                        "sampling_seed",
                        "sampling_min_context",
                        "sampling_max_context",
                    )
                    if key in row
                }
                | {
                    "status": "archived",
                    "detail": args.reason,
                    "archive_file": archive_ref,
                }
            )
        else:
            rewritten.append(row)
    continuous_bench.atomic_write(
        events_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rewritten),
    )
    print(f"archived {len(selected)} rows to {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
