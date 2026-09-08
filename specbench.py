#!/usr/bin/env python3
"""Thin Uzu/oMLX speculative-decoding benchmark runner (stdlib only)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
MTP_RE = re.compile(
    r"MTP\[[^]]+\] finish=(?P<finish>\S+) tokens=(?P<tokens>\d+) "
    r"cycles=(?P<cycles>\d+) tok/cycle=(?P<tpc>[\d.]+) "
    r"accept=(?P<accepted>\d+)/(?P<drafted>\d+) \((?P<rate>[\d.]+)%\)"
)
DFLASH_RE = re.compile(
    r"DFlash generation complete: (?P<tokens>\d+) tokens, "
    r"(?P<tps>[\d.]+) tok/s, acceptance=(?P<rate>[\d.]+)%, "
    r"cycles=(?P<cycles>\d+)"
)
MTP_NORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "pre_fc_norm_hidden.weight",
    "pre_fc_norm_embedding.weight",
    "norm.weight",
)


def expand(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


def sample_stdev(values: list[float]) -> float:
    """Return sample standard deviation, or zero for a single observation."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError("unsupported config schema_version")
    return data


def load_prompts(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        synthetic = row.pop("synthetic_context", None)
        if synthetic:
            content = (
                synthetic["unit"] * int(synthetic["repeats"])
                + synthetic.get("pad_unit", "") * int(synthetic.get("pad_repeats", 0))
                + synthetic["suffix"]
            )
            row["messages"] = [
                {"role": "system", "content": synthetic["system"]},
                {"role": "user", "content": content},
            ]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("prompt IDs must be unique")
    return rows


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def wait_ready(base_url: str, proc: subprocess.Popen[Any], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not ready"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with status {proc.returncode}")
        try:
            request_json(f"{base_url}/v1/models", None, 3)
            return
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1)
    raise TimeoutError(f"server did not become ready: {last_error}")


def omlx_settings(
    variant: dict[str, Any], dflash_draft: Path, vlm_mtp_draft: Path
) -> dict[str, Any]:
    spec = variant["speculative"]
    settings: dict[str, Any] = {
        "max_tokens": 32768,
        "force_sampling": False,
        "enable_thinking": False,
        "thinking_budget_enabled": False,
        "guided_grammar_enabled": False,
        "turboquant_kv_enabled": False,
        "qwen35_ane_prefill_enabled": False,
        "specprefill_enabled": False,
        "dflash_enabled": spec == "dflash",
        "dflash_in_memory_cache": False,
        "dflash_ssd_cache": False,
        "mtp_enabled": spec == "mtp",
        "vlm_mtp_enabled": spec == "vlm-mtp",
        "is_pinned": False,
        "is_default": False,
    }
    if spec == "dflash":
        draft_bits = variant.get("draft_quant_bits")
        settings.update(
            {
                "dflash_draft_model": str(dflash_draft),
                "dflash_draft_quant_enabled": draft_bits is not None,
                "dflash_draft_quant_weight_bits": int(draft_bits or 4),
                "dflash_draft_quant_activation_bits": 16,
                "dflash_draft_quant_group_size": 64,
                "dflash_block_size": 16,
                "dflash_verify_mode": "adaptive",
            }
        )
    if spec == "mtp":
        settings["mtp_num_draft_tokens"] = int(variant.get("mtp_draft_tokens", 2))
    if spec == "vlm-mtp":
        settings.update(
            {
                "vlm_mtp_draft_model": str(vlm_mtp_draft),
                "vlm_mtp_draft_block_size": int(
                    variant.get("vlm_mtp_draft_block_size", 2)
                ),
            }
        )
    return {"version": 1, "models": {variant["model"]: settings}}


def find_omlx_model(model_dir: Path, model_id: str) -> Path:
    candidates = list(model_dir.glob(f"*/{model_id}")) + list(model_dir.glob(model_id))
    candidates = [path for path in candidates if (path / "config.json").is_file()]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one oMLX model named {model_id}, found {len(candidates)}"
        )
    return candidates[0]


def safetensor_keys(path: Path) -> list[str]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    return [key for key in header if key != "__metadata__"]


def _float_to_bfloat16(value: float) -> int:
    """Round a Python float to bfloat16 using round-to-nearest-even."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def patch_raw_mtp_norms(source: Path, target: Path) -> list[str]:
    """Copy an OptiQ MTP sidecar and convert raw HF RMSNorms to MLX form.

    OptiQ's bundled head is a raw-HF sidecar next to an already-sanitized MLX
    backbone. oMLX's indexed-shard path only repairs a subset of these norms.
    Patch the per-run copy so every MTP norm follows MLX's ``weight + 1``
    convention, without mutating the downloaded model.
    """
    shutil.copy2(source, target)
    with target.open("r+b") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
        data_start = 8 + header_size
        patched: list[str] = []
        for key, metadata in header.items():
            if key == "__metadata__" or not key.startswith("mtp."):
                continue
            if not key.endswith(MTP_NORM_SUFFIXES):
                continue
            if metadata.get("dtype") != "BF16" or len(metadata.get("shape", [])) != 1:
                raise RuntimeError(f"unsupported MTP norm encoding for {key}")
            start, end = metadata["data_offsets"]
            handle.seek(data_start + int(start))
            raw = handle.read(int(end) - int(start))
            if len(raw) % 2:
                raise RuntimeError(f"invalid BF16 payload size for {key}")
            values = struct.unpack(f"<{len(raw) // 2}H", raw)
            shifted = []
            for bits in values:
                value = struct.unpack("<f", struct.pack("<I", bits << 16))[0]
                shifted.append(_float_to_bfloat16(value + 1.0))
            handle.seek(data_start + int(start))
            handle.write(struct.pack(f"<{len(shifted)}H", *shifted))
            patched.append(key)
    return patched


def omlx_model_view(
    model_dir: Path, variant: dict[str, Any], destination: Path
) -> Path:
    """Create a symlink view containing only the target; patch OptiQ's MTP hint."""
    source = find_omlx_model(model_dir, variant["model"])
    target = destination / variant["model"]
    target.mkdir(parents=True, exist_ok=True)
    patched_names = {"config.json"}
    if variant["speculative"] == "mtp":
        patched_names.update({"model.safetensors.index.json", "mtp.safetensors"})
    for child in source.iterdir():
        if child.name in patched_names:
            continue
        link = target / child.name
        if not link.exists():
            if child.is_file():
                os.link(child, link)
            else:
                link.symlink_to(child, target_is_directory=True)
    config = json.loads((source / "config.json").read_text())
    if variant["speculative"] == "mtp":
        text_config = config.setdefault("text_config", {})
        text_config["mtp_num_hidden_layers"] = 1
        quantization = config.setdefault("quantization", {})
        mtp_modules = (
            "mlp.down_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "self_attn.k_proj",
            "self_attn.o_proj",
            "self_attn.q_proj",
            "self_attn.v_proj",
        )
        for module in mtp_modules:
            quantization[f"language_model.mtp.layers.0.{module}"] = {
                "bits": 4,
                "group_size": 64,
            }
        index_path = source / "model.safetensors.index.json"
        mtp_path = source / "mtp.safetensors"
        if not index_path.is_file() or not mtp_path.is_file():
            raise RuntimeError("OptiQ native MTP requires an index and mtp.safetensors")
        patched_norms = patch_raw_mtp_norms(mtp_path, target / "mtp.safetensors")
        if len(patched_norms) != 7:
            raise RuntimeError(
                f"expected seven Qwen3.6 MTP norms, patched {len(patched_norms)}"
            )
        index = json.loads(index_path.read_text())
        for key in safetensor_keys(mtp_path):
            index.setdefault("weight_map", {})[key] = "mtp.safetensors"
        (target / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2) + "\n"
        )
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return destination


@contextmanager
def server(
    config: dict[str, Any], variant: dict[str, Any], run_dir: Path
) -> Iterator[tuple[str, Path, int]]:
    engine = variant["engine"]
    variant_dir = run_dir / variant["id"]
    variant_dir.mkdir(parents=True, exist_ok=True)
    log_path = variant_dir / "server.log"
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    binaries = config["binaries"]
    paths = config["paths"]
    if engine == "uzu":
        model_path = expand(paths["uzu_model"])
        cmd = [
            str(expand(binaries["uzu"])),
            "server",
            "--model",
            str(model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-prefix-cache",
        ]
    elif engine == "omlx":
        base_path = variant_dir / "omlx-state"
        base_path.mkdir(parents=True, exist_ok=True)
        model_view = omlx_model_view(
            expand(paths["omlx_model_dir"]), variant, variant_dir / "omlx-models"
        )
        (base_path / "model_settings.json").write_text(
            json.dumps(
                omlx_settings(
                    variant,
                    expand(paths["dflash_draft"]),
                    expand(paths.get("vlm_mtp_draft", paths["dflash_draft"])),
                ),
                indent=2,
            )
            + "\n"
        )
        cmd = [
            str(expand(binaries["omlx"])),
            "serve",
            "--model-dir",
            str(model_view),
            "--base-path",
            str(base_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--max-concurrent-requests",
            "1",
            "--memory-guard",
            "off",
            "--no-cache",
            "--no-hf-cache",
            "--log-level",
            "info",
            "--sse-keepalive-mode",
            "off",
        ]
    else:
        raise ValueError(f"unknown engine: {engine}")
    (variant_dir / "server-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    with log_path.open("ab", buffering=0) as log:
        try:
            proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            wait_ready(
                base_url,
                proc,
                float(config["common"]["server_start_timeout_seconds"]),
            )
            yield base_url, log_path, proc.pid
        except Exception:
            tail = log_path.read_text(errors="replace")[-8000:]
            raise RuntimeError(f"{variant['id']} failed; log tail:\n{tail}")
        finally:
            if "proc" in locals() and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)


def chat_body(
    variant: dict[str, Any],
    prompt: dict[str, Any],
    max_tokens: int,
    common: dict[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": variant["model"],
        "messages": prompt["messages"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": common["temperature"],
        "top_p": common["top_p"],
        "top_k": common["top_k"],
    }
    if variant["engine"] == "uzu":
        body["enable_thinking"] = common["enable_thinking"]
    return body


def stream_chat(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    started = time.perf_counter()
    first_content: float | None = None
    last_content: float | None = None
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    with response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if choices:
                finish_reason = choices[0].get("finish_reason") or finish_reason
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    now = time.perf_counter()
                    first_content = first_content or now
                    last_content = now
                    text_parts.append(content)
    ended = time.perf_counter()
    if first_content is None:
        raise RuntimeError("stream contained no content")
    completion = int(usage.get("completion_tokens") or 0)
    decode_seconds = max(0.0, (last_content or ended) - first_content)
    return {
        "usage": usage,
        "finish_reason": finish_reason,
        "text": "".join(text_parts),
        "elapsed_seconds": ended - started,
        "ttft_seconds": first_content - started,
        "decode_seconds": decode_seconds,
        "decode_tps": (
            (completion - 1) / decode_seconds
            if completion > 1 and decode_seconds
            else None
        ),
        "end_to_end_tps": completion / (ended - started) if completion else None,
    }


def parse_spec_log(chunk: str) -> dict[str, Any]:
    mtp = list(MTP_RE.finditer(chunk))
    if mtp:
        value = mtp[-1].groupdict()
        return {
            "spec_acceptance_percent": float(value["rate"]),
            "spec_accepted_tokens": int(value["accepted"]),
            "spec_drafted_tokens": int(value["drafted"]),
            "spec_cycles": int(value["cycles"]),
            "tokens_per_spec_cycle": float(value["tpc"]),
        }
    dflash = list(DFLASH_RE.finditer(chunk))
    if dflash:
        value = dflash[-1].groupdict()
        return {
            "spec_acceptance_percent": float(value["rate"]),
            "spec_cycles": int(value["cycles"]),
            "tokens_per_spec_cycle": int(value["tokens"]) / int(value["cycles"]),
            "engine_reported_tps": float(value["tps"]),
        }
    return {}


def memory_snapshot(pid: int) -> dict[str, int | None]:
    rss_bytes: int | None = None
    swap_bytes: int | None = None
    free_percent: int | None = None
    try:
        value = subprocess.run(
            ["/bin/ps", "-o", "rss=", "-p", str(pid)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        rss_bytes = int(value) * 1024 if value else None
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    try:
        value = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        match = re.search(r"used = ([\d.]+)([MG])", value)
        if match:
            multiplier = 1024**2 if match.group(2) == "M" else 1024**3
            swap_bytes = int(float(match.group(1)) * multiplier)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    try:
        value = subprocess.run(
            ["/usr/bin/memory_pressure"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        match = re.search(r"memory free percentage: (\d+)%", value)
        if match:
            free_percent = int(match.group(1))
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return {
        "server_rss_bytes": rss_bytes,
        "system_swap_used_bytes": swap_bytes,
        "system_memory_free_percent": free_percent,
    }


def wait_for_memory(common: dict[str, Any], phase: str) -> dict[str, int | None]:
    minimum = int(common.get("min_memory_free_percent", 0))
    timeout = float(common.get("memory_recovery_timeout_seconds", 0))
    deadline = time.monotonic() + timeout
    while True:
        snapshot = memory_snapshot(os.getpid())
        free_percent = snapshot["system_memory_free_percent"]
        if free_percent is None or free_percent >= minimum:
            return snapshot
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"memory did not recover before {phase}: {free_percent}% free "
                f"(requires {minimum}%)"
            )
        print(
            f"  RAM gate before {phase}: {free_percent}% free; waiting for {minimum}%",
            flush=True,
        )
        time.sleep(5)


def enforce_memory_limits(
    snapshot: dict[str, int | None],
    common: dict[str, Any],
    campaign_swap_start: int | None,
) -> None:
    free_percent = snapshot["system_memory_free_percent"]
    abort_below = int(common.get("abort_memory_free_percent", 0))
    if free_percent is not None and free_percent < abort_below:
        raise RuntimeError(
            f"RAM safety stop: only {free_percent}% memory free (floor {abort_below}%)"
        )
    current_swap = snapshot["system_swap_used_bytes"]
    max_growth = float(common.get("max_swap_growth_gib", 0)) * 1024**3
    if (
        campaign_swap_start is not None
        and current_swap is not None
        and max_growth
        and current_swap - campaign_swap_start > max_growth
    ):
        growth = (current_swap - campaign_swap_start) / 1024**3
        raise RuntimeError(
            f"RAM safety stop: swap grew {growth:.2f} GiB "
            f"(limit {max_growth / 1024**3:.2f} GiB)"
        )


def run_one(
    base_url: str,
    log_path: Path,
    log_offset: int,
    variant: dict[str, Any],
    prompt: dict[str, Any],
    max_tokens: int,
    common: dict[str, Any],
    server_pid: int,
    campaign_swap_start: int | None,
) -> tuple[dict[str, Any], int]:
    memory_before = memory_snapshot(server_pid)
    watch_stop = threading.Event()
    memory_breach: list[RuntimeError] = []

    def watch_memory() -> None:
        interval = float(common.get("memory_watch_interval_seconds", 2))
        while not watch_stop.wait(interval):
            snapshot = memory_snapshot(server_pid)
            try:
                enforce_memory_limits(snapshot, common, campaign_swap_start)
            except RuntimeError as exc:
                memory_breach.append(exc)
                try:
                    os.killpg(server_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                return

    watcher = threading.Thread(target=watch_memory, daemon=True)
    watcher.start()
    try:
        result = stream_chat(
            base_url,
            chat_body(variant, prompt, max_tokens, common),
            float(common["request_timeout_seconds"]),
        )
    except Exception:
        watch_stop.set()
        watcher.join(timeout=5)
        if memory_breach:
            raise memory_breach[0]
        raise
    finally:
        watch_stop.set()
    watcher.join(timeout=5)
    if memory_breach:
        raise memory_breach[0]
    time.sleep(0.2)
    memory_after = memory_snapshot(server_pid)
    enforce_memory_limits(memory_after, common, campaign_swap_start)
    with log_path.open("rb") as log:
        log.seek(log_offset)
        chunk = log.read()
        new_offset = log.tell()
    result.update(parse_spec_log(chunk.decode(errors="replace")))
    result.update(
        {
            "server_rss_before_bytes": memory_before["server_rss_bytes"],
            "server_rss_after_bytes": memory_after["server_rss_bytes"],
            "system_swap_before_bytes": memory_before["system_swap_used_bytes"],
            "system_swap_after_bytes": memory_after["system_swap_used_bytes"],
            "system_memory_free_before_percent": memory_before[
                "system_memory_free_percent"
            ],
            "system_memory_free_after_percent": memory_after[
                "system_memory_free_percent"
            ],
        }
    )
    if (
        memory_before["system_swap_used_bytes"] is not None
        and memory_after["system_swap_used_bytes"] is not None
    ):
        result["system_swap_delta_bytes"] = (
            memory_after["system_swap_used_bytes"]
            - memory_before["system_swap_used_bytes"]
        )
    usage = result.pop("usage")
    completion = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    result.update(
        {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion,
            "total_tokens": int(usage.get("total_tokens") or prompt_tokens + completion),
            "spec_verify_count": usage.get("spec_verify_ct"),
            "output_sha256": hashlib.sha256(result["text"].encode()).hexdigest(),
        }
    )
    verify = usage.get("spec_verify_ct")
    if verify and completion:
        result["tokens_per_target_verify"] = completion / int(verify)
        result["target_pass_reduction_percent"] = 100 * (1 - int(verify) / completion)
    return result, new_offset


def selected_variants(
    config: dict[str, Any], names: str | None
) -> list[dict[str, Any]]:
    variants = config["variants"]
    if not names:
        return [
            row
            for row in variants
            if row.get("enabled", True) and row.get("default_run", True)
        ]
    wanted = names.split(",")
    by_id = {row["id"]: row for row in variants}
    missing = [name for name in wanted if name not in by_id]
    if missing:
        raise ValueError(f"unknown variants: {', '.join(missing)}")
    disabled = [name for name in wanted if not by_id[name].get("enabled", True)]
    if disabled:
        details = "; ".join(
            f"{name}: {by_id[name].get('unavailable_reason', 'disabled')}"
            for name in disabled
        )
        raise ValueError(f"disabled variants requested: {details}")
    return [by_id[name] for name in wanted]


def doctor(config: dict[str, Any]) -> int:
    paths = config["paths"]
    variants = config["variants"]
    checks: list[tuple[str, bool, str]] = []
    for name in ("uzu", "omlx", "hf"):
        path = expand(config["binaries"][name])
        checks.append(
            (f"binary:{name}", path.is_file() and os.access(path, os.X_OK), str(path))
        )
    uzu = expand(paths["uzu_model"])
    checks.extend(
        [
            ("uzu target", (uzu / "model.safetensors").is_file(), str(uzu)),
            (
                "uzu speculator",
                (uzu / "speculator/model.safetensors").is_file(),
                str(uzu / "speculator"),
            ),
        ]
    )
    model_dir = expand(paths["omlx_model_dir"])
    seen_models: set[str] = set()
    for variant in variants:
        if variant["engine"] == "omlx":
            if variant["model"] in seen_models:
                continue
            seen_models.add(variant["model"])
            found = list(model_dir.glob(f"*/{variant['model']}/config.json"))
            found += list(model_dir.glob(f"{variant['model']}/config.json"))
            checks.append(
                (f"omlx target:{variant['model']}", bool(found), str(model_dir))
            )
    optiq = next(model_dir.glob("*/Qwen3.6-27B-OptiQ-4bit"), None)
    checks.append(
        (
            "OptiQ MTP weights",
            bool(optiq and (optiq / "mtp.safetensors").is_file()),
            str(optiq or model_dir),
        )
    )
    if optiq:
        optiq_config = json.loads((optiq / "config.json").read_text())
        mtp_hint = (optiq_config.get("text_config") or {}).get(
            "mtp_num_hidden_layers"
        )
        print(
            "INFO OptiQ config MTP hint: "
            + (str(mtp_hint) if mtp_hint else "missing; runner patches per-run view to 1")
        )
    mxfp = next(model_dir.glob("*/Qwen3.6-27B-MXFP4"), None)
    checks.append(
        (
            "MXFP4 has no native MTP (expected)",
            bool(mxfp and not (mxfp / "mtp.safetensors").exists()),
            str(mxfp or model_dir),
        )
    )
    draft = expand(paths["dflash_draft"])
    checks.append(
        ("DFlash draft", (draft / "model.safetensors").is_file(), str(draft))
    )
    vlm_mtp_draft = expand(paths["vlm_mtp_draft"])
    checks.append(
        (
            "Qwen3.6 VLM MTP draft",
            (vlm_mtp_draft / "model.safetensors").is_file(),
            str(vlm_mtp_draft),
        )
    )
    for name, ok, detail in checks:
        print(f"{'OK  ' if ok else 'MISS'} {name}: {detail}")
    print("\nParity policy: temperature/top_p/top_k shared; neutral penalties; seed omitted")
    print(
        "Cache policy: Uzu --no-prefix-cache; oMLX --no-cache; "
        "DFlash RAM/SSD caches false"
    )
    return 0 if all(ok for _, ok, _ in checks) else 1


def download_draft(config: dict[str, Any], path_key: str, repo_key: str) -> int:
    target = expand(config["paths"][path_key])
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(expand(config["binaries"]["hf"])),
        "download",
        config[repo_key],
        "--local-dir",
        str(target),
    ]
    return subprocess.call(cmd)


def system_manifest() -> dict[str, Any]:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "physical_memory_bytes": os.sysconf("SC_PAGE_SIZE")
        * os.sysconf("SC_PHYS_PAGES"),
    }
    manifest.update(memory_snapshot(os.getpid()))
    return manifest


def validate_parity(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["prompt_id"], row["target_output_tokens"], row["repetition"])
        groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        output_counts = {row["completion_tokens"] for row in group}
        prompt_counts = {row["prompt_tokens"] for row in group}
        if len(output_counts) != 1 or next(iter(output_counts), 0) != key[1]:
            errors.append(
                f"{key}: output token counts differ or missed cap: {sorted(output_counts)}"
            )
        if len(prompt_counts) != 1:
            errors.append(f"{key}: prompt token counts differ: {sorted(prompt_counts)}")
    return errors


def write_reports(
    run_dir: Path, rows: list[dict[str, Any]], parity: list[str]
) -> None:
    (run_dir / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    columns = sorted({key for row in rows for key in row if key != "text"})
    with (run_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)

    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
    variant_defs = {
        row["id"]: row
        for row in json.loads((run_dir / "effective-config.json").read_text())[
            "variants"
        ]
    }
    lines = [
        "# Benchmark report",
        "",
        "| Variant | Spec | Mean decode tok/s | Median | Std dev | Mean TTFT | Spec efficiency | Min free RAM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    means: dict[str, float] = {}
    for name, group in by_variant.items():
        tps_values = [
            row["decode_tps"] for row in group if row.get("decode_tps") is not None
        ]
        ttft_values = [row["ttft_seconds"] for row in group]
        accept_values = [
            row["spec_acceptance_percent"]
            for row in group
            if row.get("spec_acceptance_percent") is not None
        ]
        means[name] = statistics.fmean(tps_values)
        if accept_values:
            efficiency = f"{statistics.fmean(accept_values):.1f}% acceptance"
        else:
            verify = [row.get("spec_verify_count") for row in group]
            if all(value is not None for value in verify):
                tokens = sum(int(row["completion_tokens"]) for row in group)
                passes = sum(int(value) for value in verify)
                efficiency = (
                    f"{100 * (1 - passes / tokens):.1f}% fewer target passes; "
                    f"{tokens / passes:.2f} tok/verify"
                )
            else:
                efficiency = "n/a"
        free_values = [
            row["system_memory_free_after_percent"]
            for row in group
            if row.get("system_memory_free_after_percent") is not None
        ]
        minimum_free = f"{min(free_values)}%" if free_values else "n/a"
        lines.append(
            f"| {name} | {variant_defs[name]['speculative']} | "
            f"{means[name]:.2f} | {statistics.median(tps_values):.2f} | "
            f"{sample_stdev(tps_values):.2f} | "
            f"{statistics.fmean(ttft_values):.3f}s | {efficiency} | {minimum_free} |"
        )
    lines.extend(
        [
            "",
            "## Speculative speedup",
            "",
            "| Variant | Baseline | Speedup | Gain |",
            "|---|---|---:|---:|",
        ]
    )
    for name, definition in variant_defs.items():
        baseline = definition.get("baseline")
        if baseline and name in means and baseline in means:
            ratio = means[name] / means[baseline]
            lines.append(
                f"| {name} | {baseline} | {ratio:.3f}x | {(ratio - 1) * 100:+.1f}% |"
            )
    lines.extend(
        [
            "",
            "## Per-case means",
            "",
            "| Prompt | Output | Variant | Decode tok/s | Std dev | TTFT | Spec efficiency |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    cases: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["prompt_id"], row["target_output_tokens"], row["variant"])
        cases.setdefault(key, []).append(row)
    for (prompt_id, target_tokens, name), group in cases.items():
        tps = [row["decode_tps"] for row in group]
        acceptance_values = [
            row["spec_acceptance_percent"]
            for row in group
            if row.get("spec_acceptance_percent") is not None
        ]
        if acceptance_values:
            efficiency = f"{statistics.fmean(acceptance_values):.1f}% acceptance"
        elif all(row.get("spec_verify_count") is not None for row in group):
            tokens = sum(int(row["completion_tokens"]) for row in group)
            passes = sum(int(row["spec_verify_count"]) for row in group)
            efficiency = f"{100 * (1 - passes / tokens):.1f}% fewer target passes"
        else:
            efficiency = "n/a"
        lines.append(
            f"| {prompt_id} | {target_tokens} | {name} | "
            f"{statistics.fmean(tps):.2f} | "
            f"{sample_stdev(tps):.2f} | "
            f"{statistics.fmean(row['ttft_seconds'] for row in group):.3f}s | "
            f"{efficiency} |"
        )
    initial = json.loads((run_dir / "system.json").read_text())
    swap_start = initial.get("system_swap_used_bytes")
    swap_values = [
        row["system_swap_after_bytes"]
        for row in rows
        if row.get("system_swap_after_bytes") is not None
    ]
    free_values = [
        row["system_memory_free_after_percent"]
        for row in rows
        if row.get("system_memory_free_after_percent") is not None
    ]
    lines.extend(["", "## Memory safety", ""])
    if free_values:
        lines.append(f"- Minimum measured system memory free: {min(free_values)}%.")
    if swap_start is not None and swap_values:
        growth_mib = (max(swap_values) - swap_start) / 1024**2
        lines.append(f"- Maximum campaign swap growth: {growth_mib:+.0f} MiB.")
    positive_request_swap = [
        row["system_swap_delta_bytes"]
        for row in rows
        if row.get("system_swap_delta_bytes", 0) > 0
    ]
    lines.append(
        "- Requests with positive before/after swap growth: "
        f"{len(positive_request_swap)} of {len(rows)}."
    )
    lines.extend(["", "## Parity", ""])
    if parity:
        lines.extend(f"- {item}" for item in parity)
    else:
        lines.append(
            "PASS: prompt and completion token counts match across all selected variants."
        )
    lines.extend(
        [
            "",
            "`decode_tps` is measured from first to last streamed content using "
            "`(completion_tokens - 1) / elapsed`.",
            "",
            "oMLX acceptance comes from native MTP/DFlash logs. Uzu exposes "
            "verification-pass count, not true draft acceptance; its per-row "
            "target-pass reduction is therefore reported separately in JSON/CSV.",
        ]
    )
    (run_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def run_benchmark(args: argparse.Namespace, config: dict[str, Any]) -> int:
    variants = selected_variants(config, args.variants)
    prompts = load_prompts(expand(args.prompts))
    if args.prompt_ids:
        wanted = set(args.prompt_ids.split(","))
        prompts = [row for row in prompts if row["id"] in wanted]
    common = dict(config["common"])
    output_tokens = (
        [int(x) for x in args.output_tokens.split(",")]
        if args.output_tokens
        else common["output_tokens"]
    )
    repetitions = args.repetitions or int(common["repetitions"])
    if args.smoke:
        prompts, output_tokens, repetitions = prompts[:1], [32], 1
        common["warmup_tokens"] = 8
        common["cooldown_seconds"] = 0
    common["output_tokens"] = output_tokens
    common["repetitions"] = repetitions
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = expand(args.output_dir) / run_id
    run_dir.mkdir(parents=True)
    effective = dict(config)
    effective["variants"] = variants
    effective["common"] = common
    (run_dir / "effective-config.json").write_text(
        json.dumps(effective, indent=2) + "\n"
    )
    (run_dir / "system.json").write_text(
        json.dumps(system_manifest(), indent=2) + "\n"
    )
    rows: list[dict[str, Any]] = []
    campaign_memory = wait_for_memory(common, "campaign start")
    campaign_swap_start = campaign_memory["system_swap_used_bytes"]
    print(f"run: {run_dir}", flush=True)
    for index, variant in enumerate(variants):
        wait_for_memory(common, variant["id"])
        print(f"\n[{index + 1}/{len(variants)}] {variant['id']}", flush=True)
        with server(effective, variant, run_dir) as (base_url, log_path, server_pid):
            offset = log_path.stat().st_size
            warmup_tokens = int(common["warmup_tokens"])
            if warmup_tokens:
                print(
                    f"  untimed kernel warmup: {warmup_tokens} tokens "
                    "(fresh request; request/KV caches remain disabled)",
                    flush=True,
                )
                _, offset = run_one(
                    base_url,
                    log_path,
                    offset,
                    variant,
                    prompts[0],
                    warmup_tokens,
                    common,
                    server_pid,
                    campaign_swap_start,
                )
            for prompt in prompts:
                for target in output_tokens:
                    for repetition in range(1, repetitions + 1):
                        print(
                            f"  {prompt['id']} output={target} rep={repetition}",
                            flush=True,
                        )
                        result, offset = run_one(
                            base_url,
                            log_path,
                            offset,
                            variant,
                            prompt,
                            target,
                            common,
                            server_pid,
                            campaign_swap_start,
                        )
                        result.update(
                            {
                                "variant": variant["id"],
                                "engine": variant["engine"],
                                "model": variant["model"],
                                "speculative": variant["speculative"],
                                "prompt_id": prompt["id"],
                                "target_output_tokens": target,
                                "repetition": repetition,
                            }
                        )
                        expected_prompt_tokens = prompt.get("expected_prompt_tokens")
                        if (
                            expected_prompt_tokens is not None
                            and result["prompt_tokens"] != int(expected_prompt_tokens)
                        ):
                            raise RuntimeError(
                                f"{prompt['id']} produced {result['prompt_tokens']} prompt "
                                f"tokens; expected {expected_prompt_tokens}"
                            )
                        rows.append(result)
                        (run_dir / "results.partial.jsonl").write_text(
                            "".join(
                                json.dumps(row, sort_keys=True) + "\n" for row in rows
                            )
                        )
                        print(
                            f"    {result['decode_tps']:.2f} tok/s, "
                            f"TTFT {result['ttft_seconds']:.3f}s",
                            flush=True,
                        )
        if index + 1 < len(variants) and common["cooldown_seconds"]:
            time.sleep(float(common["cooldown_seconds"]))
    parity = validate_parity(rows)
    write_reports(run_dir, rows, parity)
    (run_dir / "results.partial.jsonl").unlink(missing_ok=True)
    print(f"\nreport: {run_dir / 'REPORT.md'}")
    if parity and common["strict_exact_output"]:
        print("strict parity validation failed", file=sys.stderr)
        return 1
    return 0


def plan(config: dict[str, Any]) -> int:
    print("Variants:")
    for variant in config["variants"]:
        baseline = f"; baseline={variant['baseline']}" if variant.get("baseline") else ""
        status = "" if variant.get("enabled", True) else "; disabled"
        if not variant.get("default_run", True):
            status += "; opt-in"
        print(
            f"  {variant['id']}: {variant['engine']} / {variant['model']} / "
            f"{variant['speculative']}{baseline}{status}"
        )
        if variant.get("unavailable_reason"):
            print(f"    reason: {variant['unavailable_reason']}")
    common = config["common"]
    print(f"Prompts: {len(load_prompts(ROOT / 'prompts.jsonl'))}")
    print(f"Output caps: {common['output_tokens']}; repetitions: {common['repetitions']}")
    print("Caches: disabled at every exposed engine layer")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default=str(ROOT / "benchmark.json"))
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "doctor", help="verify binaries and model artifacts without loading models"
    )
    sub.add_parser("plan", help="print the benchmark matrix without loading models")
    sub.add_parser(
        "download-draft", help="download the configured oMLX DFlash drafter"
    )
    sub.add_parser(
        "download-vlm-mtp",
        help="download the configured Qwen3.6 external VLM MTP drafter",
    )
    run = sub.add_parser("run", help="run the benchmark matrix")
    run.add_argument("--prompts", default=str(ROOT / "prompts.jsonl"))
    run.add_argument("--output-dir", default=str(ROOT / "results"))
    run.add_argument("--variants", help="comma-separated variant IDs")
    run.add_argument("--prompt-ids", help="comma-separated prompt IDs")
    run.add_argument("--output-tokens", help="comma-separated exact output caps")
    run.add_argument("--repetitions", type=int)
    run.add_argument(
        "--smoke", action="store_true", help="one prompt, 32 output tokens, one repetition"
    )
    return result


def main() -> int:
    args = parser().parse_args()
    config = load_config(expand(args.config))
    if args.command == "doctor":
        return doctor(config)
    if args.command == "plan":
        return plan(config)
    if args.command == "download-draft":
        return download_draft(config, "dflash_draft", "draft_repo")
    if args.command == "download-vlm-mtp":
        return download_draft(config, "vlm_mtp_draft", "vlm_mtp_draft_repo")
    if args.command == "run":
        return run_benchmark(args, config)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
