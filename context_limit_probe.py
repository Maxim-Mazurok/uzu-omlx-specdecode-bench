#!/usr/bin/env python3
"""Find an approximate clean context limit using guarded benchmark runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import continuous_bench
import specbench


ROOT = Path(__file__).resolve().parent
DEFAULT_VARIANT = "omlx-mxfp4-turboquant"
MINIMUM_EXACT_CONTEXT = continuous_bench.CHAT_OVERHEAD_AND_SUFFIX_TOKENS


def rounded_midpoint(passing_context: int, failing_context: int, step: int) -> int:
    midpoint = (passing_context + failing_context) // 2
    rounded = (midpoint // step) * step
    return max(passing_context + step, min(rounded, failing_context - step))


def next_context(
    observations: list[dict[str, Any]],
    maximum_context: int,
    required_context: int,
    minimum_context: int,
    precision: int,
) -> int | None:
    outcomes = {
        int(observation["context_tokens"]): observation["status"]
        for observation in observations
    }
    if required_context not in outcomes:
        return required_context
    if outcomes[required_context] == "ok" and maximum_context not in outcomes:
        return maximum_context
    if outcomes.get(maximum_context) == "ok":
        return None

    passing = [context for context, status in outcomes.items() if status == "ok"]
    failing = [
        context for context, status in outcomes.items() if status == "ram_limit"
    ]
    if not passing:
        candidate = max(minimum_context, required_context // 2)
        while candidate in outcomes and candidate > minimum_context:
            candidate = max(minimum_context, candidate // 2)
        return None if candidate in outcomes else candidate

    passing_context = max(passing)
    higher_failures = [context for context in failing if context > passing_context]
    if not higher_failures:
        return None
    failing_context = min(higher_failures)
    if failing_context - passing_context <= precision:
        return None
    return rounded_midpoint(passing_context, failing_context, precision)


def load_observations(path: Path, variant: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    observations = continuous_bench.read_events(path)
    mismatched = {
        str(observation.get("variant"))
        for observation in observations
        if observation.get("variant") != variant
    }
    if mismatched:
        raise RuntimeError(
            f"existing probe contains other variants: {', '.join(sorted(mismatched))}"
        )
    return observations


def write_prompt(path: Path, context_tokens: int, attempt: int) -> None:
    prompt = continuous_bench.exact_prompt(context_tokens, attempt)
    path.write_text(json.dumps(prompt) + "\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default=str(ROOT / "benchmark.json"))
    result.add_argument("--variant", default=DEFAULT_VARIANT)
    result.add_argument(
        "--output-dir", default=str(ROOT / "context-limit-results" / DEFAULT_VARIANT)
    )
    result.add_argument("--maximum-context", type=int, default=256 * 1024)
    result.add_argument("--required-context", type=int, default=128 * 1024)
    result.add_argument("--minimum-context", type=int, default=32 * 1024)
    result.add_argument("--precision", type=int, default=8 * 1024)
    result.add_argument("--output-tokens", type=int, default=16)
    result.add_argument("--warmup-tokens", type=int, default=0)
    result.add_argument("--request-timeout-seconds", type=float, default=2 * 60 * 60)
    result.add_argument("--max-swap-growth-gib", type=float, default=0.5)
    result.add_argument("--memory-watch-interval-seconds", type=float, default=0.5)
    result.add_argument("--max-attempts", type=int, default=12)
    result.add_argument(
        "--prepare-only",
        action="store_true",
        help="print next context without loading a model",
    )
    return result


def validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.minimum_context < MINIMUM_EXACT_CONTEXT:
        raise ValueError(
            f"--minimum-context must be at least {MINIMUM_EXACT_CONTEXT}"
        )
    if not (
        arguments.minimum_context
        <= arguments.required_context
        <= arguments.maximum_context
    ):
        raise ValueError("context bounds must satisfy minimum <= required <= maximum")
    if arguments.precision < 1:
        raise ValueError("--precision must be positive")
    if arguments.output_tokens < 2 or arguments.warmup_tokens < 0:
        raise ValueError("output tokens must be >= 2 and warmup tokens must be >= 0")
    if arguments.request_timeout_seconds <= 0:
        raise ValueError("--request-timeout-seconds must be positive")
    if arguments.max_swap_growth_gib <= 0:
        raise ValueError("--max-swap-growth-gib must be positive")
    if arguments.memory_watch_interval_seconds <= 0:
        raise ValueError("--memory-watch-interval-seconds must be positive")
    if arguments.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")


def summarize(observations: list[dict[str, Any]], required_context: int) -> str:
    passing = sorted(
        int(observation["context_tokens"])
        for observation in observations
        if observation["status"] == "ok"
    )
    failing = sorted(
        int(observation["context_tokens"])
        for observation in observations
        if observation["status"] == "ram_limit"
    )
    if not passing:
        return "No clean context found."
    passing_context = passing[-1]
    verdict = "PASS" if passing_context >= required_context else "BAD"
    if failing:
        return (
            f"{verdict}: clean through {passing_context:,}; "
            f"RAM boundary at or below {min(context for context in failing if context > passing_context):,}."
            if any(context > passing_context for context in failing)
            else f"{verdict}: clean through {passing_context:,}."
        )
    return f"{verdict}: clean through at least {passing_context:,}."


def main() -> int:
    arguments = parser().parse_args()
    validate_arguments(arguments)
    output_directory = Path(arguments.output_dir).expanduser().resolve()
    prompts_directory = output_directory / "prompts"
    attempts_directory = output_directory / "attempts"
    raw_directory = output_directory / "raw"
    for directory in (prompts_directory, attempts_directory, raw_directory):
        directory.mkdir(parents=True, exist_ok=True)
    events_path = output_directory / "events.jsonl"
    observations = load_observations(events_path, arguments.variant)
    print(f"probe: {output_directory}")

    while len(observations) < arguments.max_attempts:
        context_tokens = next_context(
            observations,
            arguments.maximum_context,
            arguments.required_context,
            arguments.minimum_context,
            arguments.precision,
        )
        if context_tokens is None:
            break
        print(f"next context: {context_tokens:,}", flush=True)
        if arguments.prepare_only:
            return 0

        attempt = len(observations)
        prompt_path = prompts_directory / f"{attempt:03d}-{context_tokens}.jsonl"
        write_prompt(prompt_path, context_tokens, attempt)
        log_path = attempts_directory / f"{attempt:03d}-{context_tokens}.log"
        command = [
            sys.executable,
            str(ROOT / "specbench.py"),
            "--config",
            str(Path(arguments.config).expanduser().resolve()),
            "run",
            "--prompts",
            str(prompt_path),
            "--output-dir",
            str(raw_directory),
            "--variants",
            arguments.variant,
            "--output-tokens",
            str(arguments.output_tokens),
            "--warmup-tokens",
            str(arguments.warmup_tokens),
            "--cooldown-seconds",
            "0",
            "--request-timeout-seconds",
            str(arguments.request_timeout_seconds),
            "--max-swap-growth-gib",
            str(arguments.max_swap_growth_gib),
            "--memory-watch-interval-seconds",
            str(arguments.memory_watch_interval_seconds),
            "--repetitions",
            "1",
        ]
        return_code, output, run_directory = continuous_bench.run_attempt(
            command, log_path
        )
        status, detail = continuous_bench.classify(return_code, output)
        result = continuous_bench.parse_result(run_directory)
        observation: dict[str, Any] = {
            "attempt": attempt,
            "variant": arguments.variant,
            "context_tokens": context_tokens,
            "status": status,
            "detail": detail,
            "returncode": return_code,
            "run_dir": str(run_directory) if run_directory else None,
            "attempt_log": str(log_path),
        }
        if result:
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "ttft_seconds",
                "decode_tps",
                "system_memory_free_after_percent",
                "system_swap_delta_bytes",
            ):
                if key in result:
                    observation[key] = result[key]
        continuous_bench.append_jsonl(events_path, observation)
        observations.append(observation)
        print(f"status: {status}", flush=True)
        if status not in ("ok", "ram_limit"):
            print(f"probe stopped: {detail}", file=sys.stderr)
            return 1

    print(summarize(observations, arguments.required_context))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())