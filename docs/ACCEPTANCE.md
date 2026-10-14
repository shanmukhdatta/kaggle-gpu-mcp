# Acceptance and test protocol

## What the automated tests prove

Run `python -m pytest -q` to verify deterministic code-path tests with a simulated Kaggle CLI adapter. These cover request approval, source validation, private metadata, workflow dependencies, recovery, job completion, metrics check, hash manifest, and simulated mail delivery.

**They do not prove access to a real Kaggle GPU.** The real acceptance must be completed from your own Kaggle account.

## Real end-to-end sequence

1. Install project dependencies and Kaggle CLI.
2. Authenticate with Kaggle's supported mechanism, and set `KGM_KAGGLE_USERNAME`.
3. Run `kaggle-gpu-mcp check`. Confirm account access.
4. Run `kaggle-gpu-mcp demo` to copy example scripts.
5. In terminal A: `kaggle-gpu-mcp worker`.
6. In terminal B: `kaggle-gpu-mcp run-demo --approve`.
7. Poll with `kaggle-gpu-mcp jobs` or `kaggle-gpu-mcp job JOB_ID`.
8. Confirm prep CPU notebook finishes and saves `merged.csv`.
9. Confirm train GPU notebook starts, mounts the prep output, and saves `metrics.json`, `run_manifest.json`, `model.pt`.
10. Confirm `run_manifest.json` says `gpu_verified: true` with an actual CUDA device.
11. Confirm local manifest contains SHA-256 hashes and model checkpoint.
12. Restart the worker and verify jobs remain available, without duplicate notification emails.
13. Configure SMTP, submit a separate approved job and validate a **real** email delivered.
14. Examine Kaggle notebook execution logs and error recovery on a separate intentionally failing test.

## Known limitations

The demo uses a synthetic dataset if no Kaggle CSV input is supplied. It should not be confused with a validated production dataset. `model.pt` is a demo checkpoint; never run untrusted pickle deserialization. Raw results and GPU quotas depend on Kaggle and can change.

## Windows smoke-test checklist

- [ ] Kaggle CLI authenticated
- [ ] First private CPU Kaggle notebook submitted
- [ ] CPU output downloaded and verified
- [ ] Dependent GPU Kaggle notebook submitted
- [ ] GPU CUDA computation verified
- [ ] PyTorch checkpoint downloaded
- [ ] Validation metrics read
- [ ] Notification delivered
- [ ] Persistence survives restart
- [ ] MCP host successfully invoked the tools
