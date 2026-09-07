# Context-length decoding sweep

Run on a 32 GB Apple-silicon Mac on 2026-09-08. The comparison uses Uzu
Mirai-M, oMLX OptiQ-4bit, and oMLX MXFP4. The primary oMLX comparisons use the
same 4-bit DFlash drafter.

## Result

Uzu is the practical winner on this machine. It remained clean through a
40,000-token prompt. OptiQ+DFlash crossed the 4 GiB campaign swap-growth guard
during the 20,000-token request. MXFP4 with the same 4-bit DFlash configuration
remained clean through 30,000 tokens and crossed the guard at 40,000.

| Context | Uzu decode | Uzu TTFT | OptiQ+DFlash decode | OptiQ+DFlash TTFT | Status |
|---:|---:|---:|---:|---:|---|
| 1,000 | 20.08 tok/s | 5.3s | 14.79 tok/s | 4.9s | clean |
| 5,000 | 16.63 tok/s | 25.4s | 15.27 tok/s | 24.0s | clean |
| 10,000 | 16.00 tok/s | 51.7s | 14.71 tok/s | 73.4s | clean |
| 20,000 | 15.74 tok/s | 110.4s | 13.10 tok/s | 369.8s | OptiQ RAM boundary |
| 30,000 | 13.30 tok/s | 177.9s | — | — | OptiQ not attempted |
| 40,000 | 13.57 tok/s | 254.5s | — | — | OptiQ not attempted |

### MXFP4 with 4-bit DFlash

| Context | Decode | TTFT | Acceptance | Status |
|---:|---:|---:|---:|---|
| 1,000 | 13.73 tok/s | 4.9s | 63.9% | clean |
| 5,000 | 13.80 tok/s | 24.0s | 62.9% | clean |
| 10,000 | 13.63 tok/s | 46.7s | 62.7% | clean |
| 20,000 | 13.30 tok/s | 98.6s | 64.6% | clean |
| 30,000 | 11.54 tok/s | 183.8s | 64.3% | clean |
| 40,000 | 12.50 tok/s | 498.5s | 66.8% | RAM boundary |

At the largest mutually clean point for all three targets (10,000 tokens), Uzu
decoded 8.8% faster than OptiQ and 17.3% faster than MXFP4. MXFP4's prefill was
the fastest of the three there: 46.7s versus Uzu's 51.7s and OptiQ's 73.4s.

The OptiQ 20k request completed all 512 output tokens at the same instant the
watchdog fired. Its server log reports 13.1 decode tok/s, 369.796s prefill, and
65.0% DFlash acceptance, but the row is classified as a boundary observation
rather than a clean runner result. The safe, fully recorded OptiQ range ends at
10k.

## What slows down

- Uzu decode fell 20.3% from 1k to 10k and 32.4% from 1k to 40k.
- Uzu's decode curve was fairly flat from 5k through 20k, then stepped down
  around 30k. The small 30k-to-40k uptick is normal one-run/speculation noise.
- OptiQ decode was almost flat through 10k, but its prefill cost accelerated
  sharply. The boundary 20k request spent roughly 370 seconds before decode.
- MXFP4 with a 4-bit DFlash draft stayed nearly flat through 20k, then fell to
  11.54 tok/s at 30k. Its 40k request completed at 12.5 tok/s, but prefill took
  498.5 seconds and swap growth crossed 4.64 GiB.
- Long-context prefill and memory pressure matter more than the modest decline
  in steady-state decoding speed. A single very long generation would mix these
  effects with changing speculative acceptance, so fixed 512-token outputs are
  the cleaner experiment.

## Speculative efficiency

Uzu exposes target verification passes, so its efficiency is reported as fewer
target passes: 70.3%–76.4% across the sweep (3.37–4.23 output tokens per target
verification). oMLX exposes true DFlash draft acceptance. OptiQ measured
63.3%–65.0%; MXFP4 with the 4-bit draft measured 62.7%–66.8%. Acceptance was
therefore not the main cause of MXFP4's remaining decode-speed gap. These oMLX
acceptance percentages should not be directly compared with Uzu's target-pass
reduction metric.

The initial MXFP4 sweep used the BF16 DFlash checkpoint and decoded only
8.66–11.73 tok/s before reaching the RAM boundary at 30k. Native phase logs
showed BF16 drafting taking about 5.3–9.5 seconds per 512-token response, versus
roughly 0.2 seconds after quantizing the same draft to 4-bit. The BF16 series is
retained as diagnostic evidence but is not the apples-to-apples comparison.

## Controls

- Exact prompt sizes: 1k, 5k, 10k, 20k, 30k, 40k; a 50k prompt is retained but
  not rerun after the earlier Uzu 50k prefill reached the RAM/swap boundary.
- Exactly 512 output tokens per completed clean row.
- Temperature 0, top-p 1, top-k 1, repetition penalty 1, no presence penalty,
  thinking disabled, and no seed sent.
- Uzu prefix cache disabled; oMLX request cache and DFlash memory/SSD caches
  disabled.
- One measured repetition per point after a 512-token warm-up. This sweep maps
  the shape and practical memory ceiling; the earlier 10k campaign provides
  three-repetition confirmation.
- Automatic stop below 12% free memory or above 4 GiB campaign swap growth.

Raw clean Uzu data are in `results/20260908-002910`; clean OptiQ data and the
20k safety-stop log are in `results/20260908-004419`. MXFP4 BF16-draft data are
in `results/20260908-012406`; MXFP4 4-bit-draft data are in
`results/20260908-013829`.
