# Results synthesis

This document combines the formal short-context benchmark, repeated 10k run,
fixed context sweep, VLM-MTP tuning, and the latest deterministic continuous
campaign. All measurements were made locally on a base Apple M5 Mac with
32 GiB unified memory.

## Executive summary

Uzu was the fastest practical Qwen3.6-27B configuration tested. Its advantage
was largest on short prompts and in end-to-end long-context latency. oMLX
benefited strongly from speculative decoding, but the fastest speculative path
depended on the target quantization and available memory.

The latest MXFP4 + VLM MTP results materially narrow the decode-speed gap. In
eight clean, matched-context pairs between 2,547 and 23,561 prompt tokens, Uzu
averaged 17.37 decode tok/s and MXFP4 + VLM MTP averaged 16.29 tok/s. Uzu's mean
and median paired advantage were both 6.7%. MXFP4 + VLM MTP had 11.2% lower
mean TTFT across the same pairs, although this small local sample should not be
treated as a general prefill ranking.

Long context changed the practical ranking more through prefill time and memory
headroom than through steady-state decode degradation. Uzu stayed within the
project's no-swap fairness guard through 40k context. MXFP4 + 4-bit DFlash was
clean through 30k, and OptiQ + DFlash through 10k.

## Formal campaign: 108 measured generations

Three prompts × two output lengths × three repetitions were measured for six
configurations after a shared excluded warmup.

| Configuration | Mean decode tok/s | Gain vs same target | Efficiency | Minimum free RAM |
|---|---:|---:|---:|---:|
| Uzu Mirai-M + bundled speculator | **20.18** | unavailable | 4.23 tok/verify; 76.3% fewer target passes | 34% |
| oMLX OptiQ + 4-bit DFlash | **15.37** | **+140.1%** | 65.4% acceptance | 23% |
| oMLX OptiQ + repaired native MTP | 13.56 | +111.9% | **85.5% acceptance** | 18% |
| oMLX MXFP4 + BF16 DFlash | 12.85 | +57.0% | 65.6% acceptance | 25% |
| oMLX MXFP4 target only | 8.19 | — | — | 36% |
| oMLX OptiQ target only | 6.40 | — | — | 26% |

The fastest oMLX path did not have the highest acceptance. Native MTP accepted
85.5% of draft tokens but was slower than the 65.4%-acceptance DFlash path.
Acceptance measures how often the draft is useful; it does not include the
cost of producing and verifying that draft.

The formal metric-only rows are available as
[`docs/data/full-benchmark.csv`](docs/data/full-benchmark.csv).

## Fixed-context comparison

| Context | Uzu decode | OptiQ + DFlash | MXFP4 + DFlash 4-bit | Clean result |
|---:|---:|---:|---:|---|
| 1k | 20.08 | 14.79 | 13.73 | all |
| 5k | 16.63 | 15.27 | 13.80 | all |
| 10k | 16.00 | 14.71 | 13.63 | all |
| 20k | 15.74 | RAM boundary | 13.30 | Uzu, MXFP4 |
| 30k | 13.30 | not attempted | 11.54 | Uzu, MXFP4 |
| 40k | 13.57 | not attempted | RAM boundary | Uzu |

At the largest mutually clean context, 10k, Uzu decoded 8.8% faster than
OptiQ + DFlash and 17.3% faster than MXFP4 + DFlash. MXFP4 had the fastest TTFT
at that point: 46.7 seconds, versus 51.7 for Uzu and 73.4 for OptiQ.

The separate repeated 10k campaign produced a larger Uzu lead: 15.92 versus
13.38 mean decode tok/s (+19.0%), 52.05 versus 141.63 seconds mean TTFT, and
roughly 2× end-to-end throughput. oMLX decode remained stable while its uncached
prefill latency varied, consistent with macOS reclaiming and faulting mapped
model pages.

See [`context_sweep_results.csv`](context_sweep_results.csv) for the compact
machine-readable sweep.

## External VLM MTP

MXFP4's external 4-bit VLM MTP drafter was tested at block sizes 2, 3, 4, and 6.
Block 3 was fastest at 16.96 decode tok/s in the controlled tuning sweep.

| Block | Decode tok/s | Acceptance | Target rounds | Tokens/round |
|---:|---:|---:|---:|---:|
| 2 | 13.35 | 78.3% | 143 | 1.78 |
| 3 | **16.96** | 70.3% | 106 | 2.41 |
| 4 | 15.46 | 53.7% | 98 | 2.61 |
| 6 | 10.54 | 36.9% | 90 | 2.84 |

Larger blocks reduced target rounds but added more low-value draft and wider
verification work. Block 3 gave the best wall-clock tradeoff and matches the
checkpoint's declared native block size.

OptiQ + external VLM MTP was functionally validated but left too little memory
headroom for a trustworthy 32 GiB continuous campaign. It is therefore
available only as an explicit manual variant and is excluded from the public
chart.

## Continuous snapshot

The interactive chart contains 149 allowlisted observations through attempt
217. It includes successful rows and RAM safety stops for Uzu, historical
OptiQ/MXFP4 DFlash, and validated MXFP4 VLM MTP. Sixty-four superseded
pre-validation VLM rows, five generic errors, raw logs, paths, and generated
model text are excluded.

The latest matched Uzu/MXFP4-VLM segment currently has eight fully clean pairs.
That sample is useful evidence that VLM MTP closes most of the decode gap, but
it is not yet a replacement for the larger repeated formal campaign.

Download the curated snapshot from
[`docs/data/continuous-results.json`](docs/data/continuous-results.json).

## What can and cannot be concluded

Supported by the measurements:

- Uzu is the fastest tested practical configuration on this machine.
- Speculative decoding gives large oMLX speedups against matching target-only
  baselines.
- Draft acceptance alone does not rank speculative implementations.
- Long-context prefill and memory pressure can dominate overall task time.
- A quantized drafter is necessary for useful DFlash memory/performance on a
  32 GiB machine.

Not established by the measurements:

- A within-Uzu speculation speedup; the tested public CLI cannot disable the
  bundled speculator.
- A runtime-only Uzu/oMLX comparison; the target checkpoints use different
  quantizations and layouts.
- Universal rankings across Apple chips, OS versions, engine versions, prompt
  distributions, sampling modes, or concurrent workloads.
- Absolute model context limits; RAM stops enforce this project's fairness
  policy rather than the engines' theoretical limits.

## Reproducibility notes

- Tested engines: Uzu 0.5.26 and oMLX 0.6.4.
- Target family: Qwen3.6-27B.
- Hardware: base Apple M5 Mac, 32 GiB unified memory.
- Sampling: temperature 0, top-p 1, top-k 1, thinking disabled.
- Caches: request/KV, prefix, paged/SSD, and DFlash caches disabled where
  applicable.
- Warmup: fresh 512-token kernel warmup for each loaded variant, excluded.
- Concurrency: one request and one target/drafter pair at a time.
- Safety: stop below 12% free memory or above 4 GiB campaign swap growth.

For command lines, prompt definitions, token validation, and metric semantics,
see the [README](README.md#methodology) and the individual result documents.
