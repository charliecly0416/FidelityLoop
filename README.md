# FidelityLoop

**Closed-loop validation of elastic LLM-serving policies.**

FidelityLoop is the research artifact accompanying the paper *FidelityLoop: Closed-Loop Validation of Elastic LLM-Serving Policies*. It connects replay predictions, policy decisions, lifecycle events, request identities, and deadline-aware cost accounting. The public CPU package supports interface checks, fixed-input replay, and result recomputation; the physical campaign reported in the paper was run separately on two NVIDIA A100 GPUs.

Current paper: [main PDF](paper/main.pdf) · [supplement](paper/supplement.pdf) · [Overleaf ZIP](https://github.com/charliecly0416/FidelityLoop/releases/download/paper-v9-20261014/FidelityLoop_Overleaf_v9_20261014.zip).

## What is included

- `src/fidelityloop/framework/`: bounded CPU facade with injectable service estimators, target/guard policy adapters, lifecycle simulation, and PPO action mapping.
- `src/fidelityloop/bridge/`: protocol, metric, and artifact-integrity helpers used by the registered evaluation contract.
- `src/fidelityloop/legacy/`: the small historical replay core required for exact two-device compatibility.
- `experiments/`: a self-contained CPU validation and its configuration; `python -m fidelityloop.framework.validate` also reproduces the historical facade compatibility checks.
- `evidence/`: compact E2/E3 result ledgers, prediction recheck, source hashes, and the current manuscript claim checks. These files contain the paper's reported deployment outcomes without shipping multi-gigabyte runtime logs.
- `artifact/`: protocol/configuration evidence, a small derived workload sample, an implementation-footprint audit, and the standalone `e1_repricing/` verifier for the paper's fixed-event resource and cost analysis. Full production traces, model weights, and GPU event journals are not redistributed here.
- `traces/`: standalone paired prediction/execution corpus: 197 accepted physical runs, 239 unique predictions, 555 explicit pairs, and standard-library loaders and metric examples. Prompts and machine identifiers are omitted; scientific failures remain included.
- `cpu_replay/`: optional native Vidur service backend with fixed Qwen profiles, all 12 retrospective policy inputs, and portable CPU replay checks.
- `paper/`: anonymous paper source, figures, bibliography, and the compiled main paper and supplement.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate                 # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -e .
python experiments/run_framework_validation.py
python -m fidelityloop.framework.validate
python -B artifact/e1_repricing/verify.py
```

The two framework checks should report `status: PASS`; the E1 verifier should report `PASS_REDUCED_LEDGER_REPRICING` and 169 checked numerical values. These commands use CPU execution without an API, GPU worker, or PPO training. See [REPRODUCING.md](REPRODUCING.md) for tests, expected results, and the distinction between portable checks and the full historical archive.

## Paired deployment traces

From `traces/`, run `python -B verify.py` to check the complete corpus and recompute its window ledgers. `python -B evaluate.py --metric feasibility` and `--metric cost` give per-campaign examples. The [standalone ZIP](https://github.com/charliecly0416/FidelityLoop/releases/download/paper-v8r12-20261013/FidelityLoop_Paired_Traces_20261013.zip) has author-neutral contents for independent use. See [schema and scope](traces/SCHEMA.md); repeated pairs are not independent experiments, and the data cannot rerun model inference or establish unseen-workload generalization.

## External service model and result checks

The paper includes service validation and closed-loop comparison with a Vidur
backend. [CPU replay instructions](cpu_replay/README.md) regenerate the existing
12 predictions; [paper artifact instructions](paper/README.md) provide fast
standard-library checks against 36 recorded physical runs. These retrospective
predictions do not increase the original 197/239/555 corpus counts.

Reader-facing model, study and checkpoint names are mapped to the unchanged
archive IDs in [DISPLAY_NAMES.json](paper/DISPLAY_NAMES.json) and the supplement.

## Quantized deployment validation

The [AWQ-Marlin artifact](paper/artifact/awq_formal/README.md) contains 12 physical
runs and 18 frozen predictions, separate from the 197/239/555 corpus above.
The three-predictor comparison uses eight policy runs; four capacity anchors
are excluded. Run `python -B paper/artifact/awq_formal/verify.py` to recompute
feasibility agreement and resource/cost errors from the released evidence.

## Rebuilding the paper

The anonymous main entry point is `paper/main.tex`; the independent supplement is `paper/supplement.tex`. The checked-in PDFs were built with the included MLSys style files. Compile from the `paper/` directory with pdfLaTeX and BibTeX. The source uses the anonymous author and facility placeholders required for review; authors should fill those only for a non-anonymous version.

## Evidence and scope

`evidence/E2_PREDICTION_RECHECK.json` records the locked comparison: Lookup-69 (archive C) agrees with the physical feasibility outcome on 32/32 cells, while Lookup-30 (archive C30) agrees on 20/32. `evidence/E2_RESULTS_32.csv` and `evidence/E2_CONTINUOUS_32_RESULTS.json` contain the compact per-cell result ledger. The original archive hash is retained in `evidence/E2_RESULTS_MANIFEST.json`; the original archive is not required to run the public CPU checks.

`evidence/E3_SCREENING_EVIDENCE.json` records the follow-up: six new checkpoints all pass their four physical runs, including three rejected by C. Across 24 policy cells, C/C30 feasibility agreement is 18/24 and 24/24, and cost MAE is 0.3112/0.0163 scenario USD. E3 uses separate window allocations; it is not pooled with E2 and does not establish a generally superior screening model.

The CPU facade demonstrates interface extensibility and conservation. It is not a claim that arbitrary heterogeneous GPUs, model weights, or unseen workloads are accurately emulated. Reproducing the physical A100 campaign requires the registered serving stack, checkpoint files, and the separately maintained execution environment described in the paper.

## License

Code and documentation are released under the MIT License in `LICENSE`. Dataset and model artifacts retain the terms of their original sources; the small derived sample is included for validation only.
