# Kaggle GPU MCP

A locally hosted, **persistent, approval-gated Kaggle CPU/GPU job system** for MCP clients.

Submit scripts and notebooks, run a two-stage CPU data-preparation -> GPU PyTorch training workflow, track Kaggle status, download logs and artifacts, compare experiments, retry with revised code, and receive SMTP completion emails. The repository includes **offline tests using a simulated Kaggle provider**. Actual Kaggle availability depends on your Kaggle account, credentials, runtime quotas and provider changes.

> **Security**: Kaggle runs executable code. Only submit code you inspected and authorized. Set `approved=true` for every remote execution. Never put Kaggle tokens, passwords or private data into notebooks or public Git repositories.

## Features

- MCP SDK v2 tools over **local stdio**, plus independent, persistent polling worker.
- Kaggle notebook/script packaging and unique private Kaggle kernel per job.
- Automatic CPU/GPU *recommendation* (not a guaranteed allocation); T4 default GPU request.
- Kaggle CLI `kernels push/status/logs/output` adapter with subprocess timeouts.
- SQLite job state and event history; uncertain submissions require reconciliation instead of blind retry.
- Durable two-stage workflow: Kaggle CPU preparation -> attach saved kernel outputs -> Kaggle GPU training.
- Approved CSV combining/deduplication and creation/versioning of **private** Kaggle datasets.
- Local/Kaggle dataset source provenance catalog, SHA-256 fingerprinting, bounded CSV profiling and atomic CSV merge manifests.
- Metrics, model files and SHA-256 artifact manifest with actual CUDA verification.
- SMTP STARTTLS or SMTP_SSL notifications via a durable outbox, bounded retries, and manual recovery; no repeated emails after successful delivery.
- New approved experiments can attach previous Kaggle outputs, enabling AI-guided iteration.
- Demo scripts for synthetic CSV preparation and a genuine PyTorch CUDA smoke test.
- Automated offline pytest tests and CI. Real GPU tests require manually running the demo.

## Requirements

- Python 3.11+, `pip`, Git, Kaggle account with CLI credentials and permission to use GPU.
- An MCP-compatible host that can execute a local stdio server.
- A worker process running while jobs need to be monitored. Kaggle may continue jobs after your computer shuts down, but **the worker cannot send completion emails while it is off**.
- For email: an authorized SMTP provider and sender account (never commit its password).

## Windows PowerShell quickstart

```powershell
# From the repository directory
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

# Authenticate using Kaggle's official CLI instructions.
# Never paste API tokens into chat or the repository.
$env:KGM_WORKSPACE = (Join-Path (Get-Location) "workspace")
$env:KGM_KAGGLE_USERNAME = "YOUR_KAGGLE_USERNAME"
# Opt in to authorized Kaggle execution only after reviewing the code:
$env:KGM_ALLOW_SUBMISSIONS = "1"

kaggle-gpu-mcp doctor
kaggle-gpu-mcp check
kaggle-gpu-mcp demo
python -m pytest -q

# Open 2 separate PowerShell terminals with the same environment and venv:
kaggle-gpu-mcp worker
# In the other terminal, connect the MCP host to:
kaggle-gpu-mcp serve
```

For an easy command-line end-to-end trial (without an MCP host):

```powershell
kaggle-gpu-mcp run-demo --approve
kaggle-gpu-mcp jobs
kaggle-gpu-mcp job JOB_ID_FROM_OUTPUT
```

The worker queues a Kaggle **CPU** script to create `merged.csv`, downloads its output, then queues a **GPU** script attaching the previous Kaggle notebook's outputs as an input. The GPU script trains a tiny PyTorch neural network, writes `model.pt`, `metrics.json`, and `run_manifest.json`, and the worker validates the actual CUDA run. It generates synthetic data only when no Kaggle dataset CSV is attached. This is a connectivity smoke test, **not a trained production model**.

## MCP client configuration (local stdio)

Use absolute executable paths from your virtual environment. Example MCP configuration:

```json
{
  "mcpServers": {
    "kaggle-gpu-mcp": {
      "command": "C:\\path\\to\\kaggle-gpu-mcp\\.venv\\Scripts\\kaggle-gpu-mcp.exe",
      "args": ["serve"],
      "env": {
        "KGM_WORKSPACE": "C:\\path\\to\\kaggle-gpu-mcp\\workspace",
        "KGM_KAGGLE_USERNAME": "YOUR_KAGGLE_USERNAME",
        "KGM_ALLOW_SUBMISSIONS": "1"
      }
    }
  }
}
```

MCP configuration syntax depends on the host. The background worker must be run separately. This project intentionally **does not expose an unauthenticated public HTTP MCP endpoint**. ChatGPT client access depends on its currently supported MCP integrations and does not imply that Kaggle can proactively write a message into an existing chat.

## Tools

| Tool | Description |
|---|---|
| `kaggle_check_connection` | Check Kaggle CLI and account access |
| `resource_select` | CPU/GPU heuristic and accelerator preference |
| `demo_install` | Copy synthetic data-prep and GPU-training scripts into workspace |
| `kaggle_submit_job` | Queue an approved `.py` or `.ipynb` file, with optional sources |
| `kaggle_workflow_start` | Chain Kaggle CPU preparation and Kaggle GPU training |
| `kaggle_get_job` | Inspect durable status, events and result metrics |
| `kaggle_list_jobs` | List stored jobs |
| `kaggle_get_logs` | Read last downloaded Kaggle logs |
| `kaggle_refresh_logs` | Request current remote logs (where available) |
| `kaggle_get_artifacts` | Read artifact manifest and metrics |
| `kaggle_job_events` | Fetch state transition audit log |
| `kaggle_reconcile_job` | Reconcile unknown pushes without auto-duplicating |
| `kaggle_publish_dataset` | Create/version a private Kaggle dataset from permitted local files (explicit license required) |
| `kaggle_merge_local_csv` | Merge/deduplicate explicitly chosen, same-schema CSV files |
| `kaggle_compare_jobs` | Compare completed experiment metrics |
| `kaggle_followup_experiment` | Submit approved revised code with previous outputs attached |
| `kaggle_test_notification` | Send approved test email to configured SMTP address |
| `kaggle_register_dataset_source` | Record an approved source reference, license claim and provenance |
| `kaggle_list_dataset_sources` | Review saved dataset source records |
| `kaggle_profile_local_csv` | Inspect bounded row/schema/missing-value statistics without exposing raw rows |
| `kaggle_workflow_status` | Read combined CPU preparation and GPU training state |
| `kaggle_notification_inbox` | Inspect saved completion notices and email delivery status |
| `kaggle_retry_notification` | Requeue unsent terminal-job email with approval |

Submission also requires a user-set `KGM_ALLOW_SUBMISSIONS=1` environment variable. All submitted code and source files must be inside `KGM_WORKSPACE`; symlink escapes are rejected. The client can create or edit scripts inside this workspace (through its own permitted editing capability). Remote uploads **always require explicit approval**.

## SMTP notification setup

Set environment variables in the **worker's environment**, not just the MCP client:

```powershell
$env:KGM_NOTIFY_EMAIL = "you@example.com"
$env:KGM_SMTP_HOST = "smtp.example.com"
$env:KGM_SMTP_PORT = "587"
$env:KGM_SMTP_USERNAME = "smtp-account@example.com"
$env:KGM_SMTP_PASSWORD = "YOUR_SMTP_APP_PASSWORD"
$env:KGM_SMTP_SENDER = "smtp-account@example.com"
kaggle-gpu-mcp worker
```

Prefer entering secrets through a secure OS credential mechanism instead of saved shell scripts. The worker uses STARTTLS on port 587 by default. For TLS-at-connect, set `KGM_SMTP_USE_SSL=1` and your provider's SSL port.

**Notifications** are sent after outputs are retrieved and validated, or after an explicit job failure. The durable notification inbox is always available from the MCP even when SMTP is unavailable. Delivery retries use backoff; unsent notices can be requeued with approval. SMTP may be unavailable for some accounts. Email is an external notification, not a direct unsolicited ChatGPT chat message.

## Using a real dataset

1. Discover and approve Kaggle datasets after checking the license, privacy, provenance and labels.
2. Pass Kaggle IDs such as `owner/dataset-slug` as `dataset_sources` to `kaggle_workflow_start`.
3. The prep script combines CSVs with matching columns. **Inspect and customize** the preparation script for different schemas and leakage-safe splits before real training.
4. The example GPU trainer requires numeric `f1`, `f2` and binary `label`. Replace the trainer with your actual model when ready.
5. For local CSVs: use `kaggle_merge_local_csv`, then `kaggle_publish_dataset` (with license rights) and attach its Kaggle reference to a workflow.
6. Check `kaggle_get_job` and `kaggle_get_artifacts` for output and provenance.
7. Your AI agent can analyze the metrics, write a revised training script and submit it using `kaggle_followup_experiment` with approval.

**Limitations:** This server does not automatically search/scrape arbitrary Internet data, train an LLM to autonomously solve a project, force GPU capacity, or send new ChatGPT messages on its own. Kaggle saved notebook outputs are not a permanent unlimited storage volume. Dataset publishing requires explicit user permission and legitimate data rights.

## Testing

```bash
python -m pip install -e '.[dev]'
pytest -q
python -m compileall -q src
```

The GitHub CI quality matrix validates MCP SDK v2 tool registration and the Kaggle CLI adapter command contract. Local `pytest` may skip the MCP contract test if the SDK is not installed. Run the **real Kaggle GPU acceptance test** manually with `kaggle-gpu-mcp run-demo --approve` after setting credentials. A local unit-test pass cannot prove real Kaggle authentication, GPU allocation, notebook completion or SMTP delivery. For additional detail read [architecture](docs/ARCHITECTURE.md), [dataset provenance](docs/DATASETS.md), [notifications](docs/NOTIFICATIONS.md), [Windows/Antigravity testing](docs/ANTIGRAVITY_ACCEPTANCE.md), and [acceptance checklist](docs/ACCEPTANCE.md).

## Publish to GitHub

Create a new empty GitHub repository named `kaggle-gpu-mcp`, then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/kaggle-gpu-mcp.git
git push -u origin main
```

If the downloaded project has no `.git` directory, run `git init -b main && git add . && git commit -m "Initial Kaggle GPU MCP"` first.

## Safety and operational notes

- Use one worker process per workspace. Multi-worker distributed leasing is not included.
- Automatic Kaggle submission retry is forbidden after unknown upload outcomes.
- Dataset merge works with compatible CSV schemas only and fits within configured local limits.
- Remote executable code, logs, checkpoints and dataset files should be treated as untrusted.
- SHA-256 checksums detect accidental corruption but do not prove that downloaded models are benign.
- Never load untrusted Python pickle model artifacts outside a sandbox.
- Tests use simulated notebook outputs, including a placeholder checkpoint. Only Kaggle can prove the real run.

Development is organized into phase-specific pull requests; see [phase tracker](docs/PHASE_TRACKER.md).

MIT licensed.
