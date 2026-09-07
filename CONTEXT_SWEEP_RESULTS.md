# Context-length decoding sweep

Run on a 32 GB Apple-silicon Mac on 2026-09-08. The comparison uses Uzu
Mirai-M and oMLX OptiQ-4bit with the 4-bit DFlash drafter.

## Result

Uzu is the practical winner on this machine. It remained usable through a
40,000-token prompt, while OptiQ+DFlash crossed the 4 GiB campaign swap-growth
guard during the 20,000-token request. Across the largest mutually clean point
(10,000 tokens), Uzu decoded 8.8% faster and reached the first token 21.7
seconds sooner.

| Context | Uzu decode | Uzu TTFT | OptiQ+DFlash decode | OptiQ+DFlash TTFT | Status |
|---:|---:|---:|---:|---:|---|
| 1,000 | 20.08 tok/s | 5.3s | 14.79 tok/s | 4.9s | clean |
| 5,000 | 16.63 tok/s | 25.4s | 15.27 tok/s | 24.0s | clean |
| 10,000 | 16.00 tok/s | 51.7s | 14.71 tok/s | 73.4s | clean |
| 20,000 | 15.74 tok/s | 110.4s | 13.10 tok/s | 369.8s | OptiQ RAM boundary |
| 30,000 | 13.30 tok/s | 177.9s | — | — | OptiQ not attempted |
| 40,000 | 13.57 tok/s | 254.5s | — | — | OptiQ not attempted |

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
- Long-context prefill and memory pressure matter more than the modest decline
  in steady-state decoding speed. A single very long generation would mix these
  effects with changing speculative acceptance, so fixed 512-token outputs are
  the cleaner experiment.

## Speculative efficiency

Uzu exposes target verification passes, so its efficiency is reported as fewer
target passes: 70.3%–76.4% across the sweep (3.37–4.23 output tokens per target
verification). oMLX exposes true DFlash draft acceptance: 63.3%–65.0%. These
percentages describe different mechanisms and should not be compared as if they
were the same metric.

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
20k safety-stop log are in `results/20260908-004419`.
