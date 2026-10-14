# Durable completion notifications

The Kaggle worker polls asynchronous notebooks, downloads/verifies artifacts and records terminal states (`COMPLETED` or `FAILED`). Each terminal job gets one SQLite outbox entry, visible through the MCP `kaggle_notification_inbox` tool. An MCP client may read this inbox later even if email was not configured when the job finished.

## Email delivery

Configure SMTP on the **worker's** environment (`KGM_NOTIFY_EMAIL`, `KGM_SMTP_HOST`, `KGM_SMTP_SENDER`, optional TLS credentials). The outbox sends after terminal status, with STARTTLS or direct SSL. Exceptions are stored as *sanitized errors*, not raw SMTP exceptions. Automatic retry backs off from 60 seconds and stops after three failed attempts (`EXHAUSTED`). Unconfigured email is marked `SKIPPED`; it does not block job completion.

To requeue a skipped or exhausted notification, configure SMTP, then call `kaggle_retry_notification(job_id, approved=true)`. Already-sent notices cannot be resent through this method, preventing accidental duplicates. The scheduler must be running for email delivery; Kaggle itself does not send messages through this MCP.

## Combined workflow status

`kaggle_workflow_status(training_job_id)` returns preparation state, training state, Kaggle references and downloaded artifacts when present. This is a *read-only* tool for AI clients that reconnect after a long run. AI clients cannot be assumed to receive a proactive message inside an existing ChatGPT conversation: they should poll this status or use external email notifications.

## Operating modes

The local stdio MCP server can be started and stopped independently of the worker. For notifications while your Windows laptop is off, deploy the same worker and its persistent SQLite workspace to a trusted always-on host. The worker uses an OS-level exclusive lock to prevent concurrent workers against one workspace; SQLite on an ephemeral volume is not adequate for reliable long-term state.
