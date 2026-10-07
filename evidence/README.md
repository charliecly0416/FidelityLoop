# Evidence ledger

The JSON and CSV files here are compact, reviewable records extracted from the locked execution package used by the paper. They preserve result identities, per-cell predictions, and source hashes without including the full raw GPU event archive.

- `E2_RESULTS_32.csv` and `E2_CONTINUOUS_32_RESULTS.json`: physical 32-cell campaign results.
- `E2_PREDICTION_RECHECK.json`: feasibility agreement and cost-error recheck for C and C30.
- `E2_RESULTS_MANIFEST.json`: source archive identity and handoff-file hashes.
- `E2_PACKAGE_MANIFEST.json`: file-level manifest for the execution package.
- `HISTORICAL_FACADE_VALIDATION.json`: CPU exact-replay and conservation validation receipt.
- `E1_DECISION_REPLAY_SUMMARY.json` and `E1_DECISION_REPLAY_AUDIT.json`: unchanged historical recovery records for 29,580 decision ticks across 16 E1 runs, including two smoke runs and two all2 capacity anchors. The audit binds the summary hash and reports the independent review outcome. These are audit receipts; the underlying raw logs are outside the compact release. Their historical schema names mention the earlier E2 microbenchmark, which is distinct from the later 32-cell E2 policy campaign.
- `V7_CLAIM_LOCK.json`: paper-claim strings checked against the evidence.

The original 220 MB handoff archive and raw GPU journals remain outside this Git repository. Their hash is recorded so a privately retained copy can be compared without making redistribution claims.
