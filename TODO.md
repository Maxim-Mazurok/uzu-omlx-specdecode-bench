# Benchmark TODO

## Qwen3.6-27B external VLM MTP

Status: wired as opt-in benchmark variants and as the active continuous roster;
the 4-bit drafter is downloaded and detected, but no VLM MTP inference has run.

- [x] Download `mlx-community/Qwen3.6-27B-MTP-4bit` with
  `python3 specbench.py download-vlm-mtp`.
- [x] Run `python3 specbench.py doctor` and confirm the 4-bit external drafter is
  detected without loading either target.
- [ ] Smoke-test `omlx-mxfp4-vlm-mtp` and `omlx-optiq-vlm-mtp` separately under the
  existing RAM/swap guard.
- [ ] Confirm greedy output parity against each target-only baseline.
- [ ] Determine whether oMLX 0.6.4 logs VLM MTP acceptance/cycle counters. Extend
  result parsing if those counters are available.
- [ ] Compare VLM MTP, 4-bit DFlash, native OptiQ MTP, and target-only decode speed,
  TTFT, end-to-end latency, speculative efficiency, and peak memory on the same
  short prompts.
- [ ] Repeat at fixed 1k, 5k, 10k, and the largest safe context. VLM MTP's smaller
  drafter makes long-context memory behavior a key part of the comparison.
- [x] Confirm the existing continuous campaign starts a new roster segment
  without rewriting its historical DFlash ledger.
- [ ] Functionally validate both VLM MTP variants before treating continuous
  results as trustworthy.
