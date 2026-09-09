# VLM-MTP validation and block-size tuning

Measured 2026-09-09 with oMLX 0.6.4, greedy decoding, caches disabled, a
77-token `code-stream` prompt, a 32-token warmup, and a 256-token measured
generation. The RAM and swap guards remained enabled.

## MXFP4 block sweep

| Block size | Decode tok/s | Acceptance | Target rounds | Tokens/round | Live RAM free |
|---:|---:|---:|---:|---:|---:|
| 2 | 13.35 | 78.3% | 143 | 1.78 | 33% |
| 3 | **16.96** | 70.3% | 106 | 2.41 | 31% |
| 4 | 15.46 | 53.7% | 98 | 2.61 | 30% |
| 6 | 10.54 | 36.9% | 90 | 2.84 | 27% |

Block 3 is the best tradeoff. Block 2 wastes target passes; blocks 4 and 6
save progressively fewer additional target passes than the extra drafting and
wider verification cost. Two later block-3 confirmations produced 13.15 and
14.63 tok/s with identical output and speculative counters, but only 18% live
RAM free, indicating system memory pressure affected wall-clock speed. The
same-headroom sweep above is the selection basis.

## OptiQ validation

After excluding OptiQ's unrelated native `mtp.safetensors` sidecar from the
temporary external-MTP target view, oMLX loaded the VLM engine and attached the
external assistant successfully.

| Block size | Decode tok/s | Acceptance | Target rounds | Tokens/round | RAM guard |
|---:|---:|---:|---:|---:|---|
| 2 | 8.2 | 80.3% | 142 | 1.80 | stopped at 10% free |
| 3 | **10.3** | 69.6% | 107 | 2.39 | stopped at 11% free |

Both requests completed before the guard shut down their servers, proving the
combination works. Block 3 is clearly better and matches the checkpoint's
declared native size. Larger OptiQ blocks were not attempted because the model
already crossed the 12%-free safety floor at a tiny prompt.

## Historical-data disposition

All 64 earlier VLM-MTP continuous rows were removed from active plotting and
preserved in a local, gitignored archive. They are excluded from the curated
public dataset because the runs predate strict backend validation.
