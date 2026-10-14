# Dataset provenance and preparation

Phase 3 deliberately uses **explicitly approved input sources**. The MCP server does not spider the Internet or download arbitrary URLs. This avoids transferring private data or violating dataset licenses without user review.

## Register sources

- `kaggle_register_dataset_source(source_type="kaggle", reference="owner/dataset", license_name="CC0-1.0", approved=true)` records an *unverified user-supplied Kaggle reference*. Kaggle account permissions are checked only when a remote job uses the reference.
- `kaggle_register_dataset_source(source_type="local_csv", reference="data/example.csv", license_name="CC0-1.0", approved=true)` records a workspace CSV and its current SHA-256, schema, row count and missing-value summary.
- `kaggle_list_dataset_sources` lists immutable source records. Re-registering a modified CSV creates a fresh record rather than rewriting history.
- `kaggle_profile_local_csv` inspects a workspace CSV without publishing rows.

A license identifier records the operator's claim; it does **not** verify legal rights or guarantee that an upstream dataset permits redistribution. Review ownership, source terms, privacy and data labels before publication.

## Combine local CSV files

Use `kaggle_merge_local_csv(["data/part1.csv", "data/part2.csv"], "combined.csv")` for compatible schemas. The operation writes into a temporary file, deduplicates exact rows, and writes both `datasets/merged/combined.csv` and `datasets/merged/combined.csv.manifest.json` with input/output hashes. Mismatched headers or malformed rows fail without leaving a partial merged CSV.

This is **not** an automatic schema harmonizer or leakage-safe ML split. Inspect the training task and write custom preparation code for different schemas, overlapping IDs, duplicates with conflicts, and dataset-specific validity rules.

## Publish and train

Publishing requires a separate explicit authorization and `KGM_ALLOW_SUBMISSIONS=1` in the worker environment. `kaggle_publish_dataset` uploads local allowed files to a **private** Kaggle dataset by default. To preprocess data on Kaggle without first uploading it locally, attach permitted Kaggle dataset refs directly to `kaggle_workflow_start`. Its CPU preparation notebook saves intermediate files; a dependent GPU notebook consumes the saved Kaggle output. Kaggle output storage is quota-limited and not a substitute for unrestricted permanent cloud storage.
