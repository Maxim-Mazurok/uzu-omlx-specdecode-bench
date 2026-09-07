# Full benchmark results

Run `20260907-221333` completed on a base Apple M5 Mac with 32 GiB unified
memory. The campaign used three prompts, exact 128- and 512-token output caps,
three repetitions per case, and one excluded 512-token kernel warmup per loaded
variant: 108 measured generations in total. All prompt/completion token parity
checks passed.

## Overall result

| Variant | Mean decode tok/s | Speculative gain | Spec efficiency | Minimum free RAM |
|---|---:|---:|---:|---:|
| Uzu Mirai-M-4 + bundled speculator | **20.18** | no public Uzu baseline | 76.3% fewer target passes; 4.23 tok/verify | 34% |
| oMLX MXFP4 baseline | 8.19 | — | — | 36% |
| oMLX MXFP4 + BF16 DFlash | 12.85 | **+57.0%** | 65.6% acceptance | 25% |
| oMLX OptiQ baseline | 6.40 | — | — | 26% |
| oMLX OptiQ + 4-bit DFlash | **15.37** | **+140.1%** | 65.4% acceptance | 23% |
| oMLX OptiQ + repaired native MTP | 13.56 | **+111.9%** | **85.5% acceptance** | 18% |

Uzu was the fastest tested configuration overall, but Uzu 0.5.26 exposes no
way to disable its required bundled speculator, so a trustworthy within-Uzu
speedup cannot be calculated. Among oMLX configurations, OptiQ plus a 4-bit
DFlash draft was fastest despite lower draft acceptance than native MTP. Draft
acceptance alone is not speedup: draft cost, verify width, and target-pass cost
also matter.

## Paired oMLX results

| Target/speculator | Prompt | Output | Decode tok/s | Gain vs same target | Acceptance |
|---|---|---:|---:|---:|---:|
| MXFP4 + DFlash | code | 128 | 11.65 | +41.6% | 62.5% |
| MXFP4 + DFlash | code | 512 | 13.26 | +62.4% | 66.8% |
| MXFP4 + DFlash | prose | 128 | 10.35 | +26.0% | 55.5% |
| MXFP4 + DFlash | prose | 512 | 10.71 | +31.3% | 58.0% |
| MXFP4 + DFlash | reasoning | 128 | 15.51 | +88.8% | 74.2% |
| MXFP4 + DFlash | reasoning | 512 | 15.64 | +91.7% | 76.4% |
| OptiQ + DFlash | code | 128 | 14.48 | +125.2% | 63.3% |
| OptiQ + DFlash | code | 512 | 17.37 | +173.1% | 70.3% |
| OptiQ + DFlash | prose | 128 | 12.52 | +95.0% | 57.0% |
| OptiQ + DFlash | prose | 512 | 12.03 | +88.8% | 56.4% |
| OptiQ + DFlash | reasoning | 128 | 17.83 | +177.6% | 74.2% |
| OptiQ + DFlash | reasoning | 512 | **17.97** | **+181.3%** | 71.5% |
| OptiQ + native MTP | code | 128 | 13.50 | +109.9% | 83.9% |
| OptiQ + native MTP | code | 512 | 13.85 | +117.8% | 88.7% |
| OptiQ + native MTP | prose | 128 | 12.74 | +98.3% | 78.7% |
| OptiQ + native MTP | prose | 512 | 12.28 | +92.6% | 76.1% |
| OptiQ + native MTP | reasoning | 128 | 14.77 | +130.0% | 94.2% |
| OptiQ + native MTP | reasoning | 512 | 14.24 | +123.0% | 91.4% |

## Uzu efficiency

| Prompt | Output | Decode tok/s | Target-pass reduction | Tokens/verify |
|---|---:|---:|---:|---:|
| code | 128 | 19.32 | 74.7% | 3.96 |
| code | 512 | 24.09 | 80.7% | 5.17 |
| prose | 128 | 12.03 | 60.9% | 2.56 |
| prose | 512 | 14.92 | 68.9% | 3.22 |
| reasoning | 128 | 23.58 | 79.4% | 4.86 |
| reasoning | 512 | **27.13** | **82.9%** | **5.84** |

Uzu reports verification-pass count rather than accepted/drafted token counts,
so target-pass reduction is not labeled as draft acceptance.

## Memory and context-length finding

The campaign started with 82% system memory free and 1.67 GiB swap in use.
The minimum measured free-memory reading was 18%, and maximum swap growth from
campaign start was 632 MiB. Zero of 108 requests had positive before/after swap
growth. No model combination tripped the 12% free-memory floor or 4 GiB swap
growth stop.

There was no 128-to-512 baseline collapse: MXFP4 moved from 8.22 to 8.16 tok/s
and OptiQ from 6.43 to 6.38 tok/s. At these output lengths, one isolated model
at a time does not support the hypothesis that growing KV context causes disk
swap and throughput loss on this 32 GiB machine.

The initial 16-token warmup was insufficient: MXFP4 rose from roughly 5.2 to
8.2 tok/s only after its first long decode. Those cold results were discarded,
and the full campaign used an identical 512-token untimed warmup for every
variant. Prefix, KV, response, DFlash RAM/SSD, and model-discovery caches stayed
disabled; compiled Metal kernels were deliberately warmed before timing.

Raw data and the generated report are stored under
`results/20260907-221333/`.
