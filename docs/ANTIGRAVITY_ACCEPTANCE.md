# Windows and Antigravity acceptance plan

## 1. Set up (user-run; do not share credentials)

From PowerShell after cloning `main`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
$env:KGM_WORKSPACE = (Join-Path (Get-Location) "workspace")
$env:KGM_KAGGLE_USERNAME = "YOUR_KAGGLE_USERNAME"
kaggle-gpu-mcp doctor
kaggle-gpu-mcp check
```

Follow the official Kaggle CLI authentication instructions to configure credentials on this machine. Do not paste any credential into a PR, AI chat, public issue, or tracked file.

## 2. Offline integration tests

Ensure `pytest -q` passes. Examine the job package created by `kaggle-gpu-mcp demo` and check that the generated Kaggle notebook metadata is private, has correct CPU/GPU mode, and uses `enable_internet=false` by default.

## 3. Real CPU -> GPU Kaggle acceptance

Run only after checking the notebook contents and confirming your Kaggle quota:

```powershell
$env:KGM_ALLOW_SUBMISSIONS = "1"
kaggle-gpu-mcp demo
kaggle-gpu-mcp run-demo --approve
```

In a second terminal with the same environment and activated venv:

```powershell
kaggle-gpu-mcp worker
```

Inspect `kaggle-gpu-mcp jobs`, `kaggle-gpu-mcp job <TRAINING_JOB_ID>`, and the saved logs and manifest. The test is successful only when the CPU preparation job completes, the GPU training job reports actual `gpu_verified=true` and `device=cuda`, and `model.pt`, `metrics.json`, `run_manifest.json` and `manifest.json` are retrieved. Notebook submission is not proof of GPU execution.

## 4. Notification acceptance

Set approved SMTP environment variables in the **worker** shell, restart worker and run an approved notification test through the MCP host. Verify receipt in the recipient inbox. Check the `kaggle_notification_inbox` output and use `kaggle_retry_notification` only for skipped or exhausted notices.

## 5. Failures to exercise

Check no-GPU, invalid Kaggle authentication, source tampering, malformed CSV, disconnected network, worker restart during a job, uncertain submission, and duplicate worker startup. A submission timeout is `SUBMISSION_UNKNOWN`, not a signal to submit it again. For evidence, capture sanitized terminal output, job ID, redacted errors, relevant log excerpts and commit SHA. Never capture tokens or untrusted dataset contents.

## Prompt for Antigravity

> Clone https://github.com/shanmukhdatta/kaggle-gpu-mcp.git at main. Review README, docs/ARCHITECTURE.md, docs/ACCEPTANCE.md and this file. Create a fresh local venv, install dependencies, run pytest and CLI doctor, inspect registered MCP tools, and report findings without changing production code. Ask me for any required Kaggle authentication or explicit GPU submission approval using the local shell; never request credentials in chat or write secrets to the repository. After I authorize the test, run the demo CPU→GPU pipeline, inspect actual Kaggle states, save redacted acceptance evidence, test email notification delivery with my SMTP configuration, and provide a precise pass/fail matrix. Do not claim real GPU results without actual Kaggle output.
