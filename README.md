# FidelityLoop

**Closed-loop validation of elastic LLM-serving policies.**

FidelityLoop is the research artifact accompanying the paper *FidelityLoop: Closed-Loop Validation of Elastic LLM-Serving Policies*. It connects replay predictions, policy decisions, lifecycle events, request identities, and deadline-aware cost accounting. The public CPU package is designed for inspection and deterministic interface checks; the physical campaign reported in the paper was run separately on two NVIDIA A100 GPUs.

## What is included

- `src/fidelityloop/framework/`: bounded CPU facade with injectable service estimators, target/guard policy adapters, lifecycle simulation, and PPO action mapping.
- `src/fidelityloop/bridge/`: protocol, metric, and artifact-integrity helpers used by the registered evaluation contract.
- `src/fidelityloop/legacy/`: the small historical replay core required for exact two-device compatibility.
- `experiments/`: a self-contained CPU validation and its configuration.
- `evidence/`: compact E2 result ledger, prediction recheck, source hashes, and the registered claim lock. These files contain the paper's reported 32-cell results without shipping multi-gigabyte runtime logs.
- `artifact/`: protocol/configuration evidence and a small derived workload sample. Full production traces, model weights, and GPU event journals are not redistributed here.
- `paper/`: anonymous paper source, figures, bibliography, and the compiled main paper and supplement.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate                 # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -e .
PYTHONPATH=src python experiments/run_framework_validation.py
python -m pytest -q
```

The validation should report `status: PASS`. It uses only a synthetic workload and CPU execution; it does not contact an API, start a GPU worker, or train PPO.

## Rebuilding the paper

The anonymous main entry point is `paper/main.tex`; the independent supplement is `paper/supplement.tex`. The checked-in PDFs were built with the included MLSys style files. Compile from the `paper/` directory with pdfLaTeX and BibTeX. The source uses the anonymous author and facility placeholders required for review; authors should fill those only for a non-anonymous version.

## Evidence and scope

`evidence/E2_PREDICTION_RECHECK.json` records the locked comparison: model C agrees with the physical feasibility outcome on 32/32 cells, while C30 agrees on 20/32. `evidence/E2_RESULTS_32.csv` and `evidence/E2_CONTINUOUS_32_RESULTS.json` contain the compact per-cell result ledger. The original archive hash is retained in `evidence/E2_RESULTS_MANIFEST.json`; the original archive is not required to run the public CPU checks.

The CPU facade demonstrates interface extensibility and conservation. It is not a claim that arbitrary heterogeneous GPUs, model weights, or unseen workloads are accurately emulated. Reproducing the physical A100 campaign requires the registered serving stack, checkpoint files, and the separately maintained execution environment described in the paper.

## License

Code and documentation are released under the MIT License in `LICENSE`. Dataset and model artifacts retain the terms of their original sources; the small derived sample is included for validation only.
