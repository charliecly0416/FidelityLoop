# Implementation footprint

This audit counts selected frozen physical-policy implementations and their shared integration code. It is an implementation inventory, not a measurement of development time, porting effort, or total framework size.

```sh
python -B count_footprint.py
python -B -m unittest -v test_count_footprint.py
```

The first command verifies source hashes and prints JSON identical to `IMPLEMENTATION_FOOTPRINT.json`. It uses only the Python standard library; the source snapshots are inspected, not imported or executed. They are not a standalone deployment package.

The count is the number of nonblank physical Python lines carrying code tokens, excluding comments and module/class/function docstrings. Decorators and multiline expressions retain their code lines; multiple statements on one line count once. File hashes, symbol ranges, and counted lines are recorded. Shared code appears once, and supporting training/runtime modules are itemized separately.

PPO's 247 policy lines include 136 lines of reused network implementation; its 382 integration lines cover the registered loading, observation, replay, and deployment paths. The guard also depends on the inherited target policy and service estimator. These components are not independent complete systems or estimates of code newly written per policy.

Historical backend, ledger, and Llama onboarding changes are unknown because complete before/after attribution is unavailable; unknown is not zero. One inherited GPU launcher contains private deployment paths and is retained as a hash-only provenance reference, without its source or line count. The selected snapshots therefore do not constitute a full dependency closure.
