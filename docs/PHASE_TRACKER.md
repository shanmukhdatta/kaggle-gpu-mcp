# Kaggle GPU MCP implementation tracker

The codebase targets one **local stdio MCP server** plus one **persistent local worker**. These milestones describe implemented code, not real-account integration proof.

| Phase | Deliverables | Verification |
|---|---|---|
| 1 | MCP server, Kaggle CLI adapter, SQLite jobs, CPU/GPU flow, model outputs, SMTP outbox | Offline tests and GitHub CI |
| 2 | Exclusive worker lock, frozen package hashes, terminal state protection, queue starvation regression | Unit tests and CI |
| 3 | Dataset source catalog, CSV quality profile, input/output SHA-256 provenance, atomic merge | Unit tests and CI |
| 4 | Notification retries and inbox, workflow status reporting, SQLite migration | Unit tests and CI |
| 5 | Kaggle adapter command tests, MCP SDK contract test, CLI doctor, Antigravity acceptance runbook | Unit tests and CI |

## Not yet account-verified

- Kaggle authentication, actual GPU allocation and successful remote training.
- Kaggle output-to-input dependency mounting in a live two-stage run.
- SMTP deliverability with the user's provider.
- Compatibility of the chosen AI host's MCP integration.

## Not part of this release

- Automatic discovery/scraping of every possible dataset source, or automatic right-to-use verification.
- Unlimited free GPU availability and always-on cloud execution.
- An unattended general-purpose LLM agent that autonomously rewrites models or starts new external GPU jobs without user authorization.
- An externally exposed unauthenticated MCP HTTP endpoint or unsolicited messages into an existing ChatGPT conversation.

These require separate products, provider authorization and real-world acceptance checks. The existing `kaggle_followup_experiment` tool enables an AI **host** to propose and submit a revision with explicit approval.
