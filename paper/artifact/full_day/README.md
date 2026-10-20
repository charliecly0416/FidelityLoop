# Full-day anonymous CPU audit

This artifact accompanies the continuous 24-hour arrival and variable-length study in FidelityLoop. It contains projections of all eight physical runs, all 24 primary host/day prediction records, and the previously reported retrospective deadline-score and startup-statistic checks. No GPU, model weights, external packages, network access, or original experiment directory is needed to audit the included evidence. Python 3.10 or newer is sufficient.

Run from this directory:

```text
python -B tools/recompute.py
python -B -m unittest discover -s tests -v
```

The first command verifies the file inventory and common identity-leak patterns, then independently reconstructs physical feasibility, full request denominators, occupied-GPU time, window costs, control coverage, frozen prediction classifications, candidate selection, and the two retrospective sensitivity summaries. It compares the reconstructed values with the displayed reference results. Optional `--output <new-file>` writes full results **outside** this directory. The scripts do not modify this artifact.

## Evidence and retained semantics

- `data/inputs`: every planned arrival, relative arrival/deadline, online/offline class, and numeric input/output budget. Request IDs are replaced with fixed lexicographic ranks within each day. The mapping preserves source-ID ordering. Prompt token IDs and raw source-row identity are omitted because this is a ledger audit, not a new inference workload.
- `data/runs`: full planned-population release/dispatch/terminal events; all 86,700 control ticks per run; actual control/actuation timing; every generation's launch, readiness, shutdown and verified release; and API accepted token budgets. Times are integer nanoseconds relative to the observation origin. Negative setup and positive post-cutoff cleanup times are retained. No absolute clock or wall date is needed.
- `data/calibration`: both sites' frozen service models, lifecycle samples, shared guard safety, and software/runtime settings. Site-A and Site-B preserve the same-host comparison and repeat assignments.
- `data/predictions`: per-request completion/dispatch projections for two days, four predictors and three policies. Strict feasibility is recomputed from every request. Prediction cost values are retained **frozen replay scalars** used by the original selection rule; this package does not independently rerun the simulator or reconstruct predictive resource-cost trajectories.
- `data/sensitivity`: all 20 retrospective startup-statistic request predictions. The 48 fixed-trajectory deadline-score cases are recomputed from physical events. Changing scoring allowance does not change controller deadlines, actions, or arrivals.
- `protocol/PROTOCOL.json`: prices, scope, fixed candidate pool, run identities, pairing, reference cost scalars and the sensitivity grid.
- `expected/RECOMPUTED_RESULTS.json`: reference results checked against the projections; this file is a comparison target, never used as the input evidence for metric computation.

The audit checks that generations do not overlap on the same GPU and occupy at most two GPUs over the observation interval. Occupied time runs from launch through verified release, clipped to the window; setup and post-cutoff cleanup intervals are retained separately in the lifecycle evidence and excluded from window cost. The projection preserves the original `release_verified` evidence flag, rather than including the private process/GPU inspection records behind that flag. The package cannot authenticate hardware execution from scratch.

## Study scope

There are two source days, six primary runs and two separately reported diagnostic repeats. Primary day and host are coupled. The D1 H repeat uses Site-B and has no same-site D1 all2 anchor; it is not assigned paired savings. All negative outcomes remain included. Control coverage and repeated blocks are descriptive trajectory evidence, not independent statistical trials.

The four primary dynamic runs all meet their online deadlines; one meets the joint online/offline contract. This is a **fixed-trajectory metric projection**: it does not remove offline work, retrain a policy, or rerun inference. It shows which observed violations online-only scoring would omit. The strict all-arrival contract and original outcome labels remain unchanged.

The inputs preserve retained BurstGPT timestamps and map source numeric lengths to clipped Qwen token budgets. They do not contain source user prompts or reconstruct its tokenizer. Offline work is constructed; the API is a finite-capacity delay-only sink. The audit recomputes these recorded runs; population-wide reliability requires additional workloads and deployments.

## Anonymity and attribution

Export uses a positive field allowlist. Account directories, device identifiers, process records, absolute clocks, machine names, original request hashes and private provenance links are omitted. Public model/software names and scientifically necessary timing, pairing and parameters remain. Source bindings and original-to-anonymous associations are retained privately by the authors and are not part of this package. The files in this directory provide the full-day companion records referenced in the paper.

Source arrival/length data are adapted from **BurstGPT**, HPMLL/BurstGPT, under CC BY 4.0. See `LICENSE_BurstGPT.txt` and https://github.com/HPMLL/BurstGPT . Changes include deterministic thinning, clipping and numeric length mapping, synthetic offline additions, anonymous request identifiers, and projection to the fields needed for audit. No prompt corpus text, tokenizer files, model weights or original private machine logs are distributed here.
