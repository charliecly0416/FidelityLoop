# FidelityLoop: CPU replay with a Vidur service backend

This optional package regenerates the paper's 12 retrospective Qwen2.5-7B
policy predictions. Vidur supplies native service scheduling and token
execution; FidelityLoop supplies admissions, lifecycle transitions, H/guard
decisions, and accounting. Measured operator profiles and all fixed inputs
are included. No GPU, model weights, API key, or new profiling is needed.

## Run

Use a separate **Python 3.13** environment (validated with 3.13.5). Install
the pinned CPU dependencies, then run these stages in order:

```text
python -m pip install -r requirements.txt
python -B run.py verify
python -B run.py engineering
python -B run.py historical
python -B run.py check
python -B run.py test
```

Dependency installation requires Internet access or a local wheel cache.
After installation, replay and validation run locally without downloads.
The historical runner uses POSIX alarms (Linux/macOS; use WSL on Windows)
and a 180-second per-case bound. Use a fresh extracted directory for each
rerun: existing result directories are rejected, never overwritten.

`engineering` exercises seven lifecycle/admission/causality fixtures;
`historical` regenerates all 12 fixed policy/window combinations.
`check` compares every request record, decision and summary with the accepted
predictions, including unsuccessful outcomes. Expected output is
`PASS_PORTABLE_NATIVE_REPLAY`, 12 cases and 25,200 decisions. `test` runs the
10 integration and seven ledger tests, including the 36-group, 90-request
service regression. The service fixtures are separate from the 12 policy cases.

Results and native event records are written to `runtime/`. Expected outputs
are read by the checker after prediction; they are not predictor inputs.
This is a portability check of existing experiments, not a new evaluation
campaign or an accuracy claim for new models and workloads.

## Contents and provenance

- `src/`: accepted control adapter and Qwen service configuration. Only the
  import locations changed for packaging; no policy or simulator math changed.
- `inputs/`: original 12 scheduling inputs, fixed lifecycle/guard parameters,
  measured profiles and service fixtures. Machine-local source paths are omitted.
- `vendor/vidur/`: native Python source pinned to commit
  `abae7f63aa857300f5cdc6f5e0d27860cd24721b`, with its MIT license and the
  existing deterministic serialization patch. Profiling and GPU execution
  dependencies are not installed by this CPU recipe.
- `vendor/fidelityloop/`: fixed policy and accounting implementation, MIT licensed.
- `PROVENANCE.json`: source hashes and packaging changes.
- `FILES_SHA256.json`: file integrity manifest.

For fast recomputation of the 12-prediction/36-physical-run comparison, use
the companion paper artifact `vidur_control/verify.py`. That standard-library
checker does not need the dependencies above. Neither entry point reruns
physical inference. The original paired corpus remains 197 physical runs,
239 original predictions and 555 pairs; these retrospective predictions
remain separate.
