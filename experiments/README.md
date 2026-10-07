# Experiments

The checked-in experiment is a bounded CPU validation of the public framework interface. It checks request, lifecycle-state, and cost conservation for two and three device synthetic scenarios and exercises the PPO action adapter. It does not claim GPU scaling accuracy and does not train a new policy.

```bash
python -m pip install -e .
python experiments/run_framework_validation.py
python -m fidelityloop.framework.validate
```

The first check uses three synthetic configurations with five requests each. The second reproduces the four historical policy-compatibility cases and two extension cases, each with three synthetic requests; its output is `results/facade/VALIDATION.json` and matches `evidence/HISTORICAL_FACADE_VALIDATION.json`. These are distinct checks.

The paper's physical E2 campaign was executed on two A100 GPUs. Its compact result ledger is in `evidence/`; the full runtime logs are intentionally kept outside this Git repository because they are large execution records rather than a portable source dependency. The ledger records the source archive hash and all 32-cell prediction checks. See the root `REPRODUCING.md` for the portable E1 accounting verifier and tests.
