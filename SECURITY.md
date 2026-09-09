# Security and privacy

## Reporting a problem

Please use GitHub's private vulnerability-reporting form in the repository's
**Security** tab. Do not open a public issue for credentials, private network
information, path disclosure, unsafe process handling, or a reproducible way
to make the runner execute unintended commands.

## Supported version

Security and privacy fixes are applied to the current `main` branch.

## Data policy

Raw benchmark outputs are intentionally excluded from version control because
server logs can contain usernames, absolute paths, local addresses, and model
output. Only allowlisted scalar metrics produced by
`export_public_results.py` belong under `docs/data/`.

The benchmark starts local HTTP servers bound to loopback and terminates only
processes it launches. Review `benchmark.json` and run `specbench.py plan`
before executing a new or modified configuration.
