#!/usr/bin/env python3
"""Deterministic, resumable continuous context benchmark orchestrator."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import specbench


ROOT = Path(__file__).resolve().parent
LEGACY_VARIANTS = (
    "uzu-m-spec",
    "omlx-optiq-dflash",
    "omlx-mxfp4-dflash-q4",
)
ACTIVE_VARIANTS = (
    "uzu-m-spec",
    "omlx-mxfp4-vlm-mtp",
)
CHART_VARIANTS = (
    "uzu-m-spec",
    "omlx-optiq-dflash",
    "omlx-optiq-vlm-mtp",
    "omlx-mxfp4-dflash-q4",
    "omlx-mxfp4-vlm-mtp",
)
# Public alias for callers that need the current run roster.
VARIANTS = ACTIVE_VARIANTS
LABELS = {
    "uzu-m-spec": "Uzu Mirai-M",
    "omlx-optiq-dflash": "oMLX OptiQ + DFlash 4-bit",
    "omlx-optiq-vlm-mtp": "oMLX OptiQ + VLM MTP 4-bit",
    "omlx-mxfp4-dflash-q4": "oMLX MXFP4 + DFlash 4-bit",
    "omlx-mxfp4-vlm-mtp": "oMLX MXFP4 + VLM MTP 4-bit",
}
COLORS = {
    "uzu-m-spec": "#4f8cff",
    "omlx-optiq-dflash": "#f29b61",
    "omlx-optiq-vlm-mtp": "#52c7a5",
    "omlx-mxfp4-dflash-q4": "#d878b2",
    "omlx-mxfp4-vlm-mtp": "#9b8cff",
}
SYSTEM = "Use the supplied reference context. Answer only the final request."
UNIT = "The worker reads one block, validates its checksum, updates the index, and records latency. "
SUFFIX = (
    "\n\nFinal request: write an exhaustive numbered sequence of independent "
    "performance-diagnostic checks, each with a concrete explanation and "
    "calculation. Continue adding new checks until the output limit interrupts "
    "you. Never conclude early."
)
CHAT_OVERHEAD_AND_SUFFIX_TOKENS = 69
UNIT_TOKENS = 18
RAM_STOP_RE = re.compile(r"RAM safety stop: (?P<detail>.+)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def exact_prompt(target_tokens: int, attempt: int) -> dict[str, Any]:
    if target_tokens < CHAT_OVERHEAD_AND_SUFFIX_TOKENS:
        raise ValueError(
            f"context must be at least {CHAT_OVERHEAD_AND_SUFFIX_TOKENS} tokens"
        )
    remaining = target_tokens - CHAT_OVERHEAD_AND_SUFFIX_TOKENS
    repeats, pad_repeats = divmod(remaining, UNIT_TOKENS)
    return {
        "id": f"continuous-{attempt:08d}-{target_tokens}",
        "expected_prompt_tokens": target_tokens,
        "synthetic_context": {
            "system": SYSTEM,
            "unit": UNIT,
            "repeats": repeats,
            "pad_unit": " x",
            "pad_repeats": pad_repeats,
            "suffix": SUFFIX,
        },
    }


def context_for_round(
    seed: int, round_index: int, minimum: int, maximum: int
) -> int:
    rng = random.Random(seed)
    value = minimum
    for _ in range(round_index + 1):
        value = rng.randint(minimum, maximum)
    return value


def variant_for(
    variants: list[str] | tuple[str, ...], round_index: int, slot: int
) -> str:
    return variants[(round_index + slot) % len(variants)]


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid event JSON on line {line_number}") from exc
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def parse_result(run_dir: Path | None) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    for name in ("results.jsonl", "results.partial.jsonl"):
        path = run_dir / name
        if path.is_file():
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            if rows:
                return rows[-1]
    return None


def run_attempt(
    command: list[str], log_path: Path
) -> tuple[int, str, Path | None]:
    output: list[str] = []
    run_dir: Path | None = None
    with log_path.open("w") as log:
        log.write("command: " + json.dumps(command) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                output.append(line)
                if line.startswith("run: "):
                    run_dir = Path(line.removeprefix("run: ").strip())
            return proc.wait(), "".join(output), run_dir
        except KeyboardInterrupt:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=15)
            raise


def classify(returncode: int, output: str) -> tuple[str, str | None]:
    match = RAM_STOP_RE.search(output)
    if match:
        return "ram_limit", match.group("detail")
    if returncode == 0:
        return "ok", None
    if "memory did not recover" in output:
        return "memory_recovery_timeout", "memory did not recover before timeout"
    return "error", f"benchmark exited with status {returncode}"


def linear_fit(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    if len(points) < 2:
        return None
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    return slope, mean_y - slope * mean_x


def render_chart(
    path: Path,
    events: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    segments = manifest["segments"]
    current_segment_index, current_segment = segment_for_attempt(
        manifest, len(events)
    )
    sampling_text = (
        f"collecting {int(current_segment['min_context']):,}–"
        f"{int(current_segment['max_context']):,} context"
    )
    if current_segment_index + 1 < len(segments):
        upcoming = segments[current_segment_index + 1]
        sampling_text += (
            f" · next full round {int(upcoming['min_context']):,}–"
            f"{int(upcoming['max_context']):,}"
        )
    maximum_context = max(
        [int(segment["max_context"]) for segment in segments]
        + [int(row.get("context_tokens", 0)) for row in events]
    )
    clean = [
        event
        for event in events
        if event.get("status") == "ok" and event.get("decode_tps") is not None
    ]
    maximum_tps = max((float(row["decode_tps"]) for row in clean), default=20.0)
    y_max = max(5.0, math.ceil(maximum_tps * 1.15))
    width, height = 1100, 620
    left, right, top, bottom = 76, 28, 86, 154
    plot_width = width - left - right
    plot_height = height - top - bottom

    def sx(value: float) -> float:
        return left + (value / maximum_context) * plot_width

    def sy(value: float) -> float:
        return top + plot_height - (value / y_max) * plot_height

    grid: list[str] = []
    for index in range(6):
        value = y_max * index / 5
        y = sy(value)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>'
            f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end">{value:.0f}</text>'
        )
    for index in range(6):
        value = maximum_context * index / 5
        x = sx(value)
        grid.append(
            f'<text x="{x:.1f}" y="{top+plot_height+25}" text-anchor="middle">{value/1000:.0f}k</text>'
        )

    marks: list[str] = []
    legend: list[str] = []
    for variant in manifest.get("chart_variants", CHART_VARIANTS):
        color = COLORS[variant]
        points = [
            (float(row["context_tokens"]), float(row["decode_tps"]))
            for row in clean
            if row["variant"] == variant
        ]
        fit = linear_fit(points)
        if fit and points:
            slope, intercept = fit
            x1, x2 = min(x for x, _ in points), max(x for x, _ in points)
            y1 = max(0.0, min(y_max, slope * x1 + intercept))
            y2 = max(0.0, min(y_max, slope * x2 + intercept))
            marks.append(
                f'<line x1="{sx(x1):.1f}" y1="{sy(y1):.1f}" '
                f'x2="{sx(x2):.1f}" y2="{sy(y2):.1f}" '
                f'stroke="{color}" class="trend"/>'
            )
        for row in clean:
            if row["variant"] != variant:
                continue
            title = html.escape(
                f"{LABELS[variant]} · {row['context_tokens']:,} context · "
                f"{row['decode_tps']:.2f} tok/s · TTFT {row['ttft_seconds']:.1f}s"
            )
            marks.append(
                f'<circle cx="{sx(float(row["context_tokens"])):.1f}" '
                f'cy="{sy(float(row["decode_tps"])):.1f}" r="5.5" fill="{color}">'
                f"<title>{title}</title></circle>"
            )
        legend.append(
            f'<span><i style="background:{color}"></i>{html.escape(LABELS[variant])}</span>'
        )

    ram_stops = [row for row in events if row.get("status") == "ram_limit"]
    for row in ram_stops:
        x = sx(float(row["context_tokens"]))
        y = top + plot_height - 9
        title = html.escape(
            f"RAM stop · {LABELS[row['variant']]} · {row['context_tokens']:,} context · "
            f"{row.get('detail') or 'safety guard'}"
        )
        marks.append(
            f'<g stroke="{COLORS[row["variant"]]}" class="ram-stop">'
            f'<line x1="{x-6:.1f}" y1="{y-6:.1f}" x2="{x+6:.1f}" y2="{y+6:.1f}"/>'
            f'<line x1="{x+6:.1f}" y1="{y-6:.1f}" x2="{x-6:.1f}" y2="{y+6:.1f}"/>'
            f"<title>{title}</title></g>"
        )

    recent_rows: list[str] = []
    for row in reversed(events[-15:]):
        speed = (
            f"{float(row['decode_tps']):.2f}"
            if row.get("decode_tps") is not None
            else "—"
        )
        ttft = (
            f"{float(row['ttft_seconds']):.1f}s"
            if row.get("ttft_seconds") is not None
            else "—"
        )
        recent_rows.append(
            "<tr>"
            f"<td>{int(row['attempt'])}</td>"
            f"<td>{int(row['round'])}</td>"
            f"<td>{html.escape(LABELS[row['variant']])}</td>"
            f"<td>{int(row['context_tokens']):,}</td>"
            f"<td>{speed}</td><td>{ttft}</td>"
            f"<td class=\"status-{html.escape(row['status'])}\">{html.escape(row['status'])}</td>"
            "</tr>"
        )
    last = events[-1] if events else None
    last_text = (
        f"Last attempt: {LABELS[last['variant']]} at {last['context_tokens']:,} tokens — {last['status']}"
        if last
        else "Waiting for the first benchmark"
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="15"><title>Continuous local-model benchmark</title>
<style>
:root{{color-scheme:light dark;font-family:ui-sans-serif,system-ui,sans-serif;background:#111;color:#eee}}
body{{margin:24px;max-width:1180px}}h1{{font-size:22px;margin:0 0 6px}}p{{color:#aaa;margin:4px 0 16px}}
.legend{{display:flex;gap:20px;flex-wrap:wrap;margin:8px 0 10px}}.legend span{{display:flex;align-items:center;gap:7px}}
.legend i{{width:18px;height:3px;display:inline-block}}svg{{width:100%;height:auto;display:block;background:#151515}}
svg text{{fill:#bbb;font-size:12px}}.grid{{stroke:#333;stroke-width:1}}.frame{{fill:none;stroke:#444}}
.trend{{stroke-width:2;stroke-dasharray:6 4;opacity:.7}}circle{{stroke:#111;stroke-width:1.5}}.ram-stop{{stroke-width:3}}
.axis-title{{fill:#eee;font-size:13px}}table{{width:100%;border-collapse:collapse;margin-top:18px;font-size:13px}}
th,td{{padding:7px 9px;border-bottom:1px solid #333;text-align:right}}th:nth-child(3),td:nth-child(3),th:last-child,td:last-child{{text-align:left}}
.status-ok{{color:#83d39b}}.status-ram_limit,.status-error,.status-memory_recovery_timeout{{color:#ff8d86}}
@media(prefers-color-scheme:light){{:root{{background:#fff;color:#171717}}p{{color:#666}}svg{{background:#fafafa}}svg text{{fill:#555}}.grid{{stroke:#ddd}}.frame{{stroke:#bbb}}circle{{stroke:#fff}}th,td{{border-color:#ddd}}.axis-title{{fill:#171717}}}}
</style></head><body>
<h1>Continuous decode speed vs context</h1>
<p>{html.escape(last_text)} · {html.escape(sampling_text)} · seed {int(current_segment['seed'])} · fixed {int(current_segment['output_tokens'])}-token output · segment {current_segment_index + 1} · refreshes every 15s</p>
<div class="legend">{''.join(legend)}<span>× RAM safety stop</span></div>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Decode throughput scatter plot with per-engine trend lines">
<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" class="frame"/>
{''.join(grid)}{''.join(marks)}
<text x="{left+plot_width/2:.1f}" y="{top+plot_height+58}" text-anchor="middle" class="axis-title">Prompt context (tokens)</text>
<text x="20" y="{top+plot_height/2:.1f}" text-anchor="middle" transform="rotate(-90 20 {top+plot_height/2:.1f})" class="axis-title">Decode speed (tokens/s)</text>
</svg>
<table><thead><tr><th>Attempt</th><th>Round</th><th>Engine</th><th>Context</th><th>tok/s</th><th>TTFT</th><th>Status</th></tr></thead>
<tbody>{''.join(recent_rows)}</tbody></table>
</body></html>"""
    atomic_write(path, document)


def requested_segment(
    args: argparse.Namespace, start_attempt: int, start_round: int
) -> dict[str, Any]:
    return {
        "start_attempt": start_attempt,
        "start_round": start_round,
        "seed": args.seed,
        "min_context": max(CHAT_OVERHEAD_AND_SUFFIX_TOKENS, args.min_context),
        "max_context": args.max_context,
        "output_tokens": args.output_tokens,
        "delay_seconds": args.delay_seconds,
        "variants": list(ACTIVE_VARIANTS),
    }


def same_segment_settings(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = (
        "seed",
        "min_context",
        "max_context",
        "output_tokens",
        "delay_seconds",
        "variants",
    )
    return all(left[key] == right[key] for key in keys)


def migrate_manifest(actual: dict[str, Any]) -> dict[str, Any]:
    schema = actual.get("schema_version")
    if schema == 1:
        variants = actual["variants"]
        segments = [
            {
                "start_attempt": 0,
                "start_round": 0,
                "seed": actual["seed"],
                "min_context": actual["min_context"],
                "max_context": actual["max_context"],
                "output_tokens": actual["output_tokens"],
                "delay_seconds": actual["delay_seconds"],
                "variants": list(variants),
            }
        ]
    elif schema == 2:
        variants = actual["variants"]
        segments = [
            dict(segment, variants=list(variants)) for segment in actual["segments"]
        ]
    elif schema == 3:
        segments = actual["segments"]
    else:
        raise RuntimeError("unsupported campaign manifest schema")
    return {
        "schema_version": 3,
        "chart_variants": list(CHART_VARIANTS),
        "segments": segments,
    }


def segment_for_attempt(
    manifest: dict[str, Any], attempt: int
) -> tuple[int, dict[str, Any]]:
    selected_index = 0
    for index, segment in enumerate(manifest["segments"]):
        if int(segment["start_attempt"]) > attempt:
            break
        selected_index = index
    return selected_index, manifest["segments"][selected_index]


def prepare_campaign(
    args: argparse.Namespace,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    campaign = Path(args.campaign_dir).expanduser().resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    for name in ("attempts", "prompts", "raw"):
        (campaign / name).mkdir(exist_ok=True)
    settings_path = campaign / "campaign.json"
    events = read_events(campaign / "events.jsonl")
    if settings_path.exists():
        manifest = migrate_manifest(json.loads(settings_path.read_text()))
        latest = manifest["segments"][-1]
        requested = requested_segment(
            args, int(latest["start_attempt"]), int(latest["start_round"])
        )
        if not same_segment_settings(latest, requested):
            if int(latest["start_attempt"]) >= len(events):
                manifest["segments"][-1] = requested
            elif latest["variants"] != requested["variants"]:
                next_round = max(
                    (int(event.get("round", -1)) for event in events), default=-1
                ) + 1
                manifest["segments"].append(
                    requested_segment(args, len(events), next_round)
                )
            else:
                roster_size = len(latest["variants"])
                completed = len(events) - int(latest["start_attempt"])
                completed_rounds = math.ceil(completed / roster_size)
                next_attempt = (
                    int(latest["start_attempt"]) + completed_rounds * roster_size
                )
                next_round = int(latest["start_round"]) + completed_rounds
                manifest["segments"].append(
                    requested_segment(args, next_attempt, next_round)
                )
    else:
        manifest = {
            "schema_version": 3,
            "chart_variants": list(CHART_VARIANTS),
            "segments": [requested_segment(args, 0, 0)],
        }
    atomic_write(settings_path, json.dumps(manifest, indent=2) + "\n")
    return campaign, events, manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--campaign-dir", default=str(ROOT / "continuous-results" / "main")
    )
    result.add_argument("--config", default=str(ROOT / "benchmark.json"))
    result.add_argument("--seed", type=int, default=20260908)
    result.add_argument("--min-context", type=int, default=69)
    result.add_argument("--max-context", type=int, default=50000)
    result.add_argument("--output-tokens", type=int, default=512)
    result.add_argument("--delay-seconds", type=float, default=60.0)
    result.add_argument(
        "--max-rounds", type=int, default=0, help="zero means run until interrupted"
    )
    result.add_argument(
        "--prepare-only",
        action="store_true",
        help="update campaign metadata and chart without running inference",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    minimum = max(CHAT_OVERHEAD_AND_SUFFIX_TOKENS, args.min_context)
    if args.max_context < minimum:
        raise SystemExit(f"--max-context must be at least {minimum}")
    if args.output_tokens < 2 or args.delay_seconds < 0 or args.max_rounds < 0:
        raise SystemExit("output tokens must be >=2; delay and max rounds must be >=0")
    campaign, events, manifest = prepare_campaign(args)
    events_path = campaign / "events.jsonl"
    chart_path = campaign / "chart.html"
    render_chart(chart_path, events, manifest)
    attempt = len(events)
    print(f"campaign: {campaign}")
    print(f"chart: {chart_path}")
    if args.prepare_only:
        print(f"prepared at attempt {attempt}; no inference was run")
        return 0
    print(f"resuming at attempt {attempt}; Ctrl-C stops after cleaning up the active server")
    try:
        while True:
            segment_index, segment = segment_for_attempt(manifest, attempt)
            variants = segment["variants"]
            relative_attempt = attempt - int(segment["start_attempt"])
            relative_round, slot = divmod(relative_attempt, len(variants))
            round_index = int(segment["start_round"]) + relative_round
            if args.max_rounds and round_index >= args.max_rounds:
                break
            context_tokens = context_for_round(
                int(segment["seed"]),
                round_index,
                int(segment["min_context"]),
                int(segment["max_context"]),
            )
            variant = variant_for(variants, round_index, slot)
            prompt = exact_prompt(context_tokens, attempt)
            prompt_path = campaign / "prompts" / f"{attempt:08d}.jsonl"
            atomic_write(prompt_path, json.dumps(prompt) + "\n")
            attempt_log = campaign / "attempts" / f"{attempt:08d}-{variant}.log"
            command = [
                sys.executable,
                str(ROOT / "specbench.py"),
                "--config",
                str(Path(args.config).expanduser().resolve()),
                "run",
                "--prompts",
                str(prompt_path),
                "--output-dir",
                str(campaign / "raw"),
                "--variants",
                variant,
                "--output-tokens",
                str(segment["output_tokens"]),
                "--repetitions",
                "1",
            ]
            before = specbench.memory_snapshot(os.getpid())
            started = utc_now()
            print(
                f"\nround {round_index} attempt {attempt}: {variant} at "
                f"{context_tokens} context tokens",
                flush=True,
            )
            returncode, output, run_dir = run_attempt(command, attempt_log)
            status, detail = classify(returncode, output)
            result = parse_result(run_dir)
            after = specbench.memory_snapshot(os.getpid())
            event: dict[str, Any] = {
                "attempt": attempt,
                "round": round_index,
                "slot": slot,
                "variant": variant,
                "context_tokens": context_tokens,
                "output_tokens": segment["output_tokens"],
                "sampling_segment": segment_index,
                "sampling_seed": segment["seed"],
                "sampling_min_context": segment["min_context"],
                "sampling_max_context": segment["max_context"],
                "status": status,
                "detail": detail,
                "started_at": started,
                "finished_at": utc_now(),
                "returncode": returncode,
                "attempt_log": str(attempt_log),
                "run_dir": str(run_dir) if run_dir else None,
                "memory_before": before,
                "memory_after": after,
            }
            if result:
                for key in (
                    "decode_tps",
                    "ttft_seconds",
                    "end_to_end_tps",
                    "elapsed_seconds",
                    "prompt_tokens",
                    "completion_tokens",
                    "spec_acceptance_percent",
                    "spec_cycles",
                    "tokens_per_spec_cycle",
                    "target_pass_reduction_percent",
                    "tokens_per_target_verify",
                    "system_memory_free_after_percent",
                    "system_swap_delta_bytes",
                    "output_sha256",
                ):
                    if key in result:
                        event[key] = result[key]
            append_jsonl(events_path, event)
            events.append(event)
            render_chart(chart_path, events, manifest)
            print(f"status: {status}; chart updated: {chart_path}", flush=True)
            attempt += 1
            if segment["delay_seconds"]:
                time.sleep(float(segment["delay_seconds"]))
    except KeyboardInterrupt:
        print(f"\nstopped; resume with the same command\nchart: {chart_path}")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
