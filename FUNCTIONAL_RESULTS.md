# Superseded functional validation results

These cold-start exploratory checks are superseded by
[`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md). They remain here as the failure
record that led to the native-MTP norm repair and full warm-run methodology.

These are **exploratory functional checks, not a publishable benchmark**. The
matrix was stopped before completion to avoid further memory pressure. Every
completed row used the same `code-stream` prompt (77 prompt tokens), greedy
sampling, disabled prefix caches, and an exact output-token cap.

## Completed checks

| Variant | Output | Decode tok/s | Spec efficiency |
|---|---:|---:|---:|
| Uzu Mirai-M | 32 | 8.95 | 3.20 tokens/verify; 68.75% fewer target verify passes |
| Uzu Mirai-M | 128 | 18.69 | 4.27 tokens/verify; 76.56% fewer target verify passes |
| Uzu Mirai-M | 512 | 23.85 | 5.63 tokens/verify; 82.23% fewer target verify passes |
| oMLX MXFP4 baseline | 128 | 7.45 | n/a |
| oMLX MXFP4 + DFlash | 128 | 11.85 | 62.5% acceptance |
| oMLX MXFP4 baseline | 512 | 7.65 | n/a |
| oMLX MXFP4 + DFlash | 512 | 12.43 | 66.8% acceptance |
| oMLX OptiQ baseline | 32 | 3.98 | n/a |
| oMLX OptiQ + native MTP | 32 | 2.74 | 0/27 accepted (0%) |
| oMLX OptiQ + 4-bit DFlash draft | 32 | 10.69 | 68.8% acceptance |
| oMLX OptiQ baseline | 128 | 3.52 | n/a |

For the paired sustained MXFP4 checks, DFlash improved decode throughput by
58.9% at 128 tokens and 62.5% at 512 tokens. The 32-token OptiQ checks suggest
that the quantized DFlash draft works while this native MTP sidecar does not,
but those tiny, separately started trials are only functional evidence.

## Memory finding

Loading OptiQ with the BF16 DFlash draft drove system swap usage to roughly
20.5 GiB and stalled an 8-token warmup. That combination was terminated and is
not represented as a result. Configuring the OptiQ draft for 4-bit loading made
the same functional path complete normally.

System swap was already elevated during portions of the longer partial run and
changed independently of individual requests. Those rows are therefore useful
for validating the harness and approximate speed, but not for final ranking.
A clean formal run should start after a reboot and use at least three
repetitions per output length.

## Uzu baseline limitation

Uzu 0.5.26 exposes no way to disable its bundled speculator. Its model resolver
also treats the sidecar as a required checkpoint artifact, so removing it makes
the target unavailable before session creation. The harness reports Uzu's
verification-pass efficiency but does not invent a speculation-off speedup.
