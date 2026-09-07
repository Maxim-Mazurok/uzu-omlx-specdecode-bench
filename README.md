# Uzu vs oMLX speculative-decoding benchmark

A thin, stdlib-only CLI harness for an apples-to-apples comparison of Uzu and
oMLX on Qwen3.6-27B. It measures speculative decoding against a no-speculation
baseline for the **same target checkpoint**, then reports throughput gain and
available speculative-efficiency counters.

The default matrix is:

| Target checkpoint | Baseline | Speculative path |
|---|---|---|
| Uzu Mirai-M-4 | unavailable in public CLI | bundled Mirai-M speculator |
| oMLX MXFP4 | target only | `z-lab/Qwen3.6-27B-DFlash` |
| oMLX OptiQ-4bit | target only | DFlash and checkpoint's native Lightning MTP |

Uzu 0.5.26 treats the bundled speculator as a required checkpoint artifact and
has no public no-speculation switch. The harness therefore runs Uzu Mirai-M and
reports its verification efficiency, but deliberately disables the impossible
`uzu-m-base` variant instead of presenting a false speedup. The variant remains
in the config as an explicit capability gap.

MXFP4 does not ship `mtp.*` weights, so it cannot use native MTP. The installed
OptiQ checkpoint does ship `mtp.safetensors`; no separate drafter is needed for
that pair. Its converted weight index omits the separate sidecar, so the runner
creates a per-run hardlink view and maps the MTP tensors and their OptiQ
quantization metadata into the view's config/index. The downloaded sidecar also
retains raw Qwen RMSNorm weights while the backbone is already in MLX
convention. oMLX's indexed-shard loader repairs only four of seven norms on
this path, which caused 0% draft acceptance. The runner copies only the 300 MB
MTP sidecar into the temporary view, shifts all seven norms to MLX convention,
and leaves the downloaded checkpoint untouched. All 19 GB target shards remain
hardlinked.

Native MTP uses depth 2, the model author's documented empirical sweet spot.
The DFlash path provides an independent speculative comparison for the same
OptiQ target.

## Fairness controls

- Identical messages and exact `max_tokens` caps for every variant.
- Strict validation that prompt-token and completion-token counts match.
- Greedy decoding: temperature `0`, top-p `1`, top-k `1`, thinking off.
- Neutral repetition/presence/min-p settings; they are recorded but not sent
  because Uzu does not expose those request fields.
- Seed omitted because Uzu does not expose a seed. Greedy decoding removes the
  practical need for it.
- One request at a time and one loaded server/model at a time.
- Uzu prefix cache disabled; oMLX paged/SSD cache and Hugging Face cache
  discovery disabled; DFlash's private RAM and SSD prefix caches disabled.
- Every output length is a fresh request. Longer output therefore grows only
  that request's KV context instead of inheriting prior generated context.
- A fresh 512-token Metal/kernel warmup is excluded for each loaded variant.
  Request/KV caches remain disabled during warmup and measurement. Raw server
  logs, full responses, hashes, timings, and effective settings are retained
  under `results/`.
- Per-request server RSS and system swap deltas are captured, making memory
  pressure visible instead of silently counting swap-throttled runs as normal.
- A live watchdog checks memory during long prefills and terminates only the
  active benchmark server if free RAM or campaign swap growth crosses the
  configured safety limit.

Throughput is measured from the first to last streamed content event as
`(completion_tokens - 1) / seconds`. End-to-end throughput and TTFT are also
stored. The report compares each speculative variant only to its paired target
baseline—never across different quantizations.

oMLX exposes true MTP/DFlash acceptance in its logs. Uzu 0.5.26 exposes
`spec_verify_ct`, not accepted/drafted counts, so the harness reports tokens per
target verification and approximate target-pass reduction for Uzu, without
mislabeling it as draft acceptance.

## Usage

No model is loaded by these commands:

```sh
python3 specbench.py plan
python3 specbench.py doctor
```

Download the external DFlash drafter (3.5 GB):

```sh
python3 specbench.py download-draft
```

Run a functional six-variant smoke test:

```sh
python3 specbench.py run --smoke
```

Run the default benchmark (three prompts, 128 and 512 output tokens, three
repetitions per case):

```sh
python3 specbench.py run
```

Or select a paired comparison:

```sh
python3 specbench.py run \
  --variants omlx-optiq-base,omlx-optiq-mtp \
  --prompt-ids code-stream \
  --output-tokens 128,512,1024 \
  --repetitions 3
```

Run the two fastest configurations with exact 10k- and 50k-token prompts:

```sh
python3 specbench.py run \
  --prompts long_context_prompts.jsonl \
  --variants uzu-m-spec,omlx-optiq-dflash \
  --output-tokens 512 \
  --repetitions 3
```

The long-context prompt definitions are compact repeat specifications expanded
in memory by the runner. Their expected token counts are checked against each
server response before a row is accepted.

Map decode speed over exact context checkpoints with fixed 512-token outputs:

```sh
python3 specbench.py run \
  --prompts context_sweep_prompts.jsonl \
  --prompt-ids context-1k,context-5k,context-10k,context-20k,context-30k,context-40k \
  --variants uzu-m-spec,omlx-optiq-dflash \
  --output-tokens 512 \
  --repetitions 1
```

See [`CONTEXT_SWEEP_RESULTS.md`](CONTEXT_SWEEP_RESULTS.md) for the 32 GB Mac
results and practical RAM limits.

Each timestamped run contains `REPORT.md`, CSV/JSONL results, an effective
configuration, system manifest, exact server commands, and per-variant logs.

## Interpretation

Speculative efficiency is reported in two distinct ways:

1. **Wall-clock gain:** speculative decode tok/s divided by its paired baseline.
2. **Draft efficiency:** oMLX acceptance percentage and tokens per cycle; Uzu
   tokens per verification pass and target-pass reduction.

High acceptance does not guarantee a speedup—the draft and verification work
can cost more than the target passes they save. Treat wall-clock gain as the
deciding result and the efficiency counters as the explanation.

The OptiQ DFlash variant loads its draft at 4-bit to stay within a 32 GB unified
memory budget. MXFP4 keeps the BF16 draft because that pair fits comfortably.
