# Experiments

The checked-in experiment is a bounded CPU validation of the public framework interface. It checks request, lifecycle-state, and cost conservation for two and three device synthetic scenarios and exercises the PPO action adapter. It does not claim GPU scaling accuracy and does not train a new policy.

```bash
PYTHONPATH=src python experiments/run_framework_validation.py
```

The paper's physical E2 campaign was executed on two A100 GPUs. Its compact result ledger is in `evidence/`; the full runtime logs are intentionally kept outside this Git repository because they are large execution records rather than a portable source dependency. The ledger records the source archive hash and all 32-cell prediction checks.
