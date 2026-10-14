# Security model and disclosure

## Default trust assumptions

This server is designed for a single local user working in a private workspace. It is not a hardened multi-tenant SaaS platform. Do **not** bind stdio-launched MCP to public network exposure, share Kaggle tokens, or execute untrusted third-party notebooks without review.

## Defenses implemented

- Code upload requires `approved=true` and a separately configured `KGM_ALLOW_SUBMISSIONS=1` host opt-in and input files within the workspace.
- Inputs are checked with `Path.resolve`; paths escaping via symlink are rejected.
- Kaggle notebook/kernel references are validated against restrictive patterns.
- Kaggle CLI invocations are argument arrays, never a shell command.
- Remote notebooks are always private; dataset create defaults to private.
- The orchestrator refuses naive retries when upload status is unknown.
- Local outputs are verified with a configurable byte limit and SHA-256 hashes.
- Email uses verified TLS or SSL; secrets reside outside repository files.
- Git ignores `.env`, credentials, SQLite databases and checkpoints.

## Residual risks

- Kaggle notebooks execute code with Kaggle-account permissions. The AI model should be given permission to submit only after human review.
- Notebook logs and downloaded files can contain personally identifying data or secrets. This implementation saves Kaggle logs locally; protect access to that directory.
- Resource limits are applied **after** downloading; a maliciously large remote output could temporarily exhaust disk space.
- Model `.pt` artifacts can contain unsafe pickle objects. Never automatically deserialize external artifacts; use isolated validation with safe serialization where possible.
- This implementation assumes **one worker process**. Running two workers concurrently can cause duplicated actions; introduce a distributed lease before production hosting.
- No built-in CSRF/authentication is included for HTTP because remote HTTP is deliberately not supported yet.
- SMTP app passwords are environment-supplied; use an operating-system secret manager for stronger deployment.
- Dataset license defaults only for user-owned, rights-cleared data. The server does not independently verify licenses.

## Responsible deployment

Keep the runtime on a machine that you control, use Kaggle account permissions consistent with its rules, configure strict quota limits, monitor generated notebooks, and routinely delete unneeded remote Kaggle notebooks and local artifacts after retention needs end.
