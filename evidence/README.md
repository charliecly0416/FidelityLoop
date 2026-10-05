# Evidence ledger

The JSON and CSV files here are compact, reviewable records extracted from the locked execution package used by the paper. They preserve result identities, per-cell predictions, and source hashes without including the full raw GPU event archive.

- `E2_RESULTS_32.csv` and `E2_CONTINUOUS_32_RESULTS.json`: physical 32-cell campaign results.
- `E2_PREDICTION_RECHECK.json`: feasibility agreement and cost-error recheck for C and C30.
- `E2_RESULTS_MANIFEST.json`: source archive identity and handoff-file hashes.
- `E2_PACKAGE_MANIFEST.json`: file-level manifest for the execution package.
- `HISTORICAL_FACADE_VALIDATION.json`: CPU exact-replay and conservation validation receipt.
- `V7_CLAIM_LOCK.json`: paper-claim strings checked against the evidence.

The original 220 MB handoff archive and raw GPU journals remain outside this Git repository. Their hash is recorded so a privately retained copy can be compared without making redistribution claims.
