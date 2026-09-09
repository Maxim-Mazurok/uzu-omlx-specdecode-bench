# Uzu vs oMLX speculative-decoding benchmark

A small, stdlib-only harness for comparing local Qwen3.6-27B inference on
Apple silicon. It keeps prompts, output caps, sampling, cache policy, warmup,
and memory guards aligned across [Uzu](https://github.com/trymirai/uzu) and
[oMLX](https://github.com/jundot/omlx).

**[Explore the interactive results](https://maxim-mazurok.github.io/uzu-omlx-specdecode-bench/)**
· [Read the findings](RESULTS.md) · [Inspect the methodology](#methodology)

> **Bottom line:** Uzu was the fastest practical configuration in these tests.
> oMLX gained substantially from speculation, and MXFP4 + external VLM MTP
> came within 6.7% of Uzu's decode speed over eight matched random contexts.
> At long context, prefill and unified-memory pressure mattered more than the
> modest decline in steady-state decode speed.

## Headline results

Measured on a base Apple M5 Mac with 32 GiB unified memory. These are local
single-stream measurements, not universal model or engine rankings.

| Finding | Result |
|---|---:|
| Fastest formal-campaign configuration | **Uzu, 20.18 decode tok/s** |
| Fastest oMLX configuration | **OptiQ + 4-bit DFlash, 15.37 tok/s** |
| OptiQ + DFlash gain vs the same target | **+140.1%** |
| OptiQ + repaired native MTP gain | **+111.9%** |
| 10k-context decode lead, Uzu vs OptiQ + DFlash | **+19.0%** |
| Latest matched-context lead, Uzu vs MXFP4 + VLM MTP | **+6.7%** |
| Largest clean fixed-output context under the RAM guard | **Uzu: 40k** |

See [RESULTS.md](RESULTS.md) for the synthesis and caveats, or drill into the
individual campaigns:

- [Formal 108-generation campaign](BENCHMARK_RESULTS.md)
- [Fixed context-length sweep](CONTEXT_SWEEP_RESULTS.md)
- [10k comparison and 50k safety stop](LONG_CONTEXT_RESULTS.md)
- [VLM-MTP block-size tuning](VLM_MTP_TUNING_RESULTS.md)
- [Curated machine-readable data](docs/data/)

## What is compared

| Target checkpoint | Target-only baseline | Speculative paths |
|---|---|---|
| Uzu Mirai-M-4 | Not exposed by tested CLI | Bundled Mirai-M speculator |
| oMLX MXFP4 | Yes | DFlash; external VLM MTP |
| oMLX OptiQ-4bit | Yes | DFlash; repaired native Lightning MTP; external VLM MTP |

The tested Uzu 0.5.26 CLI requires its bundled speculator and has no public
switch to disable it. The harness therefore reports Uzu verification efficiency
but does **not** invent a target-only baseline or a speculative speedup.

The tested oMLX 0.6.4 installation exposes acceptance and cycle counters.
OptiQ's native MTP sidecar needed a temporary, per-run metadata/norm repair;
the downloaded checkpoint is never modified. External VLM MTP uses
`mlx-community/Qwen3.6-27B-MTP-4bit` with block size 3, selected by the included
block sweep.

## Quick start

Requirements:

- An Apple-silicon Mac with enough unified memory for the selected target and
  drafter. The full matrix here was designed around 32 GiB.
- Python 3.11 or newer.
- Working `uzu`, `omlx`, and Hugging Face CLI installations.
- Local target checkpoints matching the names in [`benchmark.json`](benchmark.json).

Clone and inspect the plan without loading a model:

```sh
git clone https://github.com/Maxim-Mazurok/uzu-omlx-specdecode-bench.git
cd uzu-omlx-specdecode-bench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python specbench.py doctor
python specbench.py plan
```

The configuration uses `~`-relative paths, so it does not assume another
user's home directory. Edit [`benchmark.json`](benchmark.json) if your model
names or locations differ.

Download the optional draft checkpoints:

```sh
python3 specbench.py download-draft
python3 specbench.py download-vlm-mtp
```

Run a small functional check first:

```sh
python3 specbench.py run --smoke
```

Then run the default formal matrix:

```sh
python3 specbench.py run
```

Select a narrower paired comparison when memory is limited:

```sh
python3 specbench.py run \
  --variants omlx-optiq-base,omlx-optiq-dflash \
  --prompt-ids code-stream \
  --output-tokens 128,512 \
  --repetitions 3
```

## Continuous context sweep

The continuous runner chooses one context length per round and presents it to
each active engine. The schedule and engine rotation are deterministic, and a
restart resumes the existing append-only campaign.

```sh
caffeinate -dimsu python3 -u continuous_bench.py \
  --campaign-dir continuous-results/main \
  --seed 20260908 \
  --min-context 69 \
  --max-context 35000 \
  --output-tokens 512 \
  --delay-seconds 30
```

The current active roster is Uzu plus MXFP4 + VLM MTP. Historical DFlash
measurements remain on the chart; invalid pre-validation VLM rows and the
RAM-heavy OptiQ + external VLM configuration are excluded.

Running the same command keeps the ledger and chart. Changing the range creates
a new campaign segment without rewriting old observations. Add
`--prepare-only` to migrate metadata and redraw without loading a model.

To refresh the allowlisted public snapshot after collecting new results:

```sh
python3 export_public_results.py
```

Raw runs remain ignored. The exporter publishes only benchmark metrics—never
generated text, hashes, absolute paths, hostnames, IP addresses, or server logs.

## Methodology

- Identical messages and exact completion-token caps for every variant.
- Greedy decoding: temperature 0, top-p 1, top-k 1, thinking disabled.
- Neutral penalty settings; unsupported request fields are not sent.
- One request and one loaded model/drafter pair at a time.
- Uzu prefix cache disabled; oMLX paged/SSD cache and model discovery disabled;
  DFlash RAM/SSD prefix caches disabled.
- Fresh requests for each output length, so one generation never inherits the
  previous generation's KV context.
- Identical excluded 512-token warmups before measurement.
- Strict prompt/completion token-count validation.
- Live memory watchdog: stop the active server below 12% free RAM or above
  4 GiB campaign swap growth.

Decode throughput is measured from the first to last streamed content event as
`(completion_tokens - 1) / seconds`. TTFT and end-to-end throughput are stored
separately.

## Reading speculative efficiency

The project deliberately keeps two concepts separate:

1. **Wall-clock gain:** speculative decode throughput divided by the matching
   target-only throughput.
2. **Draft efficiency:** oMLX draft acceptance/tokens per cycle, or Uzu output
   tokens per target verification and estimated target-pass reduction.

High acceptance does not guarantee a speedup. Draft cost, verification width,
target-pass cost, and memory traffic can outweigh the saved target passes.

## Repository layout

| Path | Purpose |
|---|---|
| [`specbench.py`](specbench.py) | Paired benchmark runner and report generator |
| [`continuous_bench.py`](continuous_bench.py) | Deterministic resumable context sweep |
| [`benchmark.json`](benchmark.json) | Engines, checkpoints, controls, and RAM limits |
| [`prompts.jsonl`](prompts.jsonl) | Short benchmark prompts |
| [`context_sweep_prompts.jsonl`](context_sweep_prompts.jsonl) | Exact context checkpoints |
| [`export_public_results.py`](export_public_results.py) | Privacy-preserving public-data exporter |
| [`docs/`](docs/) | Static GitHub Pages report and curated datasets |
| [`tests/`](tests/) | Unit tests for parsing, guards, scheduling, and export |

## Scope and limitations

- Results apply to the named checkpoints, engine versions, OS, and hardware.
- Uzu and oMLX use different target quantizations; cross-engine comparisons are
  practical system comparisons, not isolated runtime comparisons.
- The one-repetition context sweep maps curve shape and memory limits; the
  formal short-context campaign provides repeated measurements.
- macOS memory reclamation can make uncached prefill latency noisy.
- RAM-stop points mark this project's fairness boundary, not a model's absolute
  context limit.
- This is an independent project and is not affiliated with Uzu, Mirai, oMLX,
  Alibaba, Qwen, or the checkpoint publishers.

## Contributing and security

Benchmark contributions should include exact model identifiers, software and
hardware versions, complete controls, and raw-to-curated provenance. See
[CONTRIBUTING.md](CONTRIBUTING.md).

Please do not commit raw server logs or result directories. If you find a
security or privacy issue, follow [SECURITY.md](SECURITY.md).

Released under the [MIT License](LICENSE).
