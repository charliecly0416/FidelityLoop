# E1 fixed-event resource and scenario-cost repricing

This standalone CPU artifact recomputes the E1 resource and scenario-cost results from reduced, previously collected records. It runs without the parent repository, network access, GPU, model weights, or extra Python packages. It is distributed with the paper release as `FidelityLoop_E1_CPU_Artifact_v7r8_20261007.zip`.

## Run locally

Requirements: Python 3.9 or newer, standard library only. No package installation, network, GPU, weights, credentials, or workload download is required. From this directory:

    python -B verify.py
    python -B -m unittest -v test_analysis
    python -B analysis.py

The verifier checks the exact file inventory and recomputes four window/accounting results against full-precision reference values. Tests cover independent arithmetic and rejected input mutations. The last command prints JSON to standard output. Included data are not changed. Keep generated output outside this directory when verifying its exact inventory. Floating-point comparisons use absolute and relative tolerances of 1e-12; source numerical cells are exported without rounding.

## Data and scope

- data/ledger.csv: 14 reduced ledgers, of which 12 compare two complete policies in two known windows with three repetitions per arm/window. Two capacity-gate rows are checked for bookkeeping consistency but excluded from comparison averages.
- data/prices.json: six numerical base rates for local GPU occupation, synthetic API input/output token budgets, startup, shutdown, and offline deadline misses. Units are scenario USD, not provider bills.
- data/requests.json: 1,564 steady-window and 454 recovery-window requests, containing anonymous IDs, online/offline type, and integer input/output token budgets only.
- data/runs.json: fixed routes and phase components for the 12 comparisons. No raw controller timeline, prompt, prompt token sequence, upstream row ID, or original timestamp is included.
- reference_results.json: unrounded reference results for window-only and full-deployment accounting.

The labels steady and recovery identify the two known evaluation windows. B-HPA is the baseline and H+guard is the compared complete policy. Only symbolic identities have been replaced. Exported numeric ledger cells, base rates, token budgets, phase components, and routing choices retain their source precision and meaning. Local route labels gpu0/gpu1 are abstract device labels, not host identities.

## Three distinct evidence layers

1. Recomputed ledger arithmetic: the verifier checks phase component sums, GPU-time charges, startup/shutdown event charges, setup + window + cleanup = full deployment, three-run arm means, savings, price-boundary roots, and signs around each root.
2. Recomputed token pricing: fixed routed request IDs are matched to the included input/output budgets, reconstructing synthetic API charges and acceptance counts. This is budget-based scenario pricing, not measured provider billing or actual generated-token metering.
3. Inherited physical-run acceptance: completion/timeliness counts, accepted route decisions, phase measurements, and acceptance flags are inputs from previously accepted reduced records. Their consistency is checked here, but deadlines and execution authenticity are not re-audited from full raw request/controller logs. The package does not recreate a physical run or independently re-establish its original acceptance.

For each window and accounting phase, differences are H+guard minus B-HPA means of three runs. At local GPU multiplier x and joint API input/output multiplier y:

    Delta C(x,y) = x * Delta C_G + y * Delta C_A + Delta C_E

The coefficients are base-price scenario-USD cost differences. Delta C_E is startup plus shutdown, held fixed while x and y vary. Local occupied GPU seconds are reported separately. Analytical roots and their sign changes are recomputed separately for each window. The optional summary averages within-window savings with equal weight; it is not a pooled-run cost ratio or an averaged break-even point.

## Workload lineage and attribution

The original E1 online arrival patterns derive from the successful-request subset of BurstGPT. The study uses controlled local prompts and registered token budgets rather than original users' prompts or semantic tasks. Offline jobs are constructed workloads, with 120 jobs per window at 256-input/256-output budgets. The included request table is a minimal derived projection; it does not redistribute the public raw trace or the prompt corpus and cannot reconstruct arrival selection or the full transformation pipeline. Azure workloads elsewhere in the broader study are outside this E1 artifact.

Public source attribution: Yuxin Wang et al. (2024), BurstGPT: A Real-world Workload Dataset to Optimize LLM Serving Systems, arXiv:2401.17644. This source attribution is retained despite removing internal paths and source-row identifiers. Removing internal provenance is not a claim that these arrival patterns or underlying datasets were authored by the artifact preparer.

## Interpretation limits

This is retrospective repricing of fixed events. Routes, lifecycle events, and outcomes do not change with prices. The two known windows provide no new held-out evidence. A complete-policy comparison does not isolate a causal guard contribution. Less local GPU occupation does not measure external API compute and cannot establish lower total compute, energy, or actual bills. The artifact cannot reproduce the full framework, model fitting, workload generation, GPU experiments, or the complete original provenance audit.

## Implementation and license

The arithmetic implementation and tests were written independently for this reduced format; pre-existing analysis code was used as a numerical reference. This release preserves the accepted analysis, tests, data, and full-precision reference values. Packaging changes update this README and include the MIT license for code and documentation.

The reduced records contain experiment ledger measurements and constructed request budgets. Original prompts, raw production traces, model weights, and the full GPU event archive are not redistributed. Upstream datasets retain their own terms and the BurstGPT attribution above. The manifest is a byte-integrity inventory, not a digital signature or independent proof of physical execution.
