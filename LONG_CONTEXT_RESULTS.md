# Long-context benchmark results

Run `20260908-000023` compared the two fastest configurations from the full
benchmark using an exactly 10,000-token prompt and an exact 512-token output
cap. Each engine received one excluded 512-token warmup followed by three fresh,
uncached repetitions. Tokenizer files were byte-identical, and prompt/completion
token parity passed for all six measured rows.

## 10k result

| Variant | Decode tok/s, mean | Decode median | TTFT, mean | TTFT values | End-to-end tok/s, mean | Spec efficiency |
|---|---:|---:|---:|---|---:|---:|
| Uzu Mirai-M-4 | **15.92** | **15.19** | **52.05s** | 52.04, 52.10, 52.00s | **6.07** | 72.6% fewer target passes; 3.65 tok/verify |
| oMLX OptiQ + 4-bit DFlash | 13.38 | 13.46 | 141.63s | 87.28, 186.11, 151.50s | 3.02 | 64.6% acceptance |

Uzu led decode throughput by 19.0% using arithmetic means and 12.8% using the
more outlier-resistant medians. Its mean time to first token was 63.3% lower,
and its mean end-to-end throughput was approximately 2.0 times higher.

oMLX decode was stable (13.22–13.46 tok/s), but uncached prefill latency varied
substantially as macOS reclaimed and faulted mapped model pages between
requests. Uzu's TTFT remained tightly grouped around 52 seconds.

The lowest measured free-memory readings were 35% for Uzu and 26% for
OptiQ+DFlash. The completed 10k campaign stayed within the configured RAM and
swap limits.

## 50k safety result

A 50,000-token Uzu request was attempted first but deliberately terminated
during prefill before producing a benchmark row. System memory fell to the 12%
safety boundary and swap rose from 2.16 GiB at campaign start to 6.59 GiB while
the process was winding down—a 4.43 GiB increase. Running two more Uzu repeats
or loading OptiQ+DFlash at 50k would not have been a clean no-swap comparison on
this 32 GiB desktop, so the paired 50k campaign was not run.

The result is therefore: 10k fits and is benchmarkable; 50k does not meet this
project's RAM/no-swap fairness criterion on the current 32 GiB system with the
normal desktop workload present.

Raw 10k data and the generated report are in `results/20260908-000023/`.
