"""Package entrypoint for the V4 W1 baseline contracts.

The implementation lives in ``scripts.maxopt_v4_w1_baselines`` so it can also
be loaded by the legacy hyphenated ``mock-schedule`` test harness.  Keeping a
single implementation prevents the two CPU entrypoints from drifting.
"""

from scripts.maxopt_v4_w1_baselines import (  # noqa: F401
    HPA_Q_TARGETS,
    HPA_STABILIZATIONS,
    MAX_TARGET,
    MMC_RHO_MAX,
    PRED_ALPHAS,
    PRED_BETAS,
    PRED_HORIZONS,
    PRED_MARGINS,
    UPDATE_SECONDS,
    Decision,
    HPAController,
    MMCController,
    PredictiveController,
    State,
    all_parameter_cells,
    decision_dict,
    pred_grid,
    validate_contracts,
)

__all__ = [name for name in globals() if not name.startswith("_")]
