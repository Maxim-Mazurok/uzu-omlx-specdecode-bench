# Contributing

Contributions that improve reproducibility, engine support, parsing, or
analysis are welcome.

## Before opening a pull request

1. Run `python3 -m unittest discover -s tests -v`.
2. Run `python3 specbench.py plan` and `python3 specbench.py doctor` if your
   change affects configuration or model discovery.
3. Keep the runner Python-stdlib-only unless a dependency is essential and
   justified.
4. Preserve apples-to-apples controls: identical prompts, output caps,
   sampling, warmup, cache state, and single-model concurrency.
5. State exact engine, checkpoint, macOS, chip, and memory versions for new
   benchmark claims.

## Result contributions

Do not commit `results/`, `continuous-results/`, server logs, model output, or
machine-local paths. Add a strict allowlist to `export_public_results.py` for
any new public metric, then export a compact dataset under `docs/data/`.

Document incomplete and RAM-stopped runs explicitly. Never treat a safety-stop
boundary as a completed benchmark point or compare speculative variants against
a different target checkpoint.

## Pull requests

Keep changes focused and explain methodology changes separately from result
changes. A result-changing pull request should include:

- the hypothesis;
- exact commands and controls;
- the curated measurements;
- limitations and unexpected failures;
- tests for new parsers, guards, or schedule behavior.
