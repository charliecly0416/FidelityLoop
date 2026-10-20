"""Standard-library recomputation of the reduced E6 startup diagnostic."""

from pathlib import Path
from statistics import mean
import csv, hashlib, json, math

ROOT = Path(__file__).resolve().parent
MODELS = {"S14", "L14", "T30-14"}
METRICS = (
    "occupied_absolute_error_gpu_seconds",
    "scenario_cost_absolute_error",
    "occupied_trajectory_l1_gpu_seconds",
    "target_disagreement_fraction",
)


def read_json(name):
    return json.loads((ROOT / name).read_text())


def read_csv(name):
    return list(csv.DictReader((ROOT / name).open(newline="")))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def boolean(value):
    require(value in ("True", "False"), "noncanonical boolean")
    return value == "True"


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-9)


def reduce_pairs(rows):
    require(bool(rows), "empty policy population")
    true = [boolean(r["physical_P1plus"]) for r in rows]
    predicted = [boolean(r["predicted_P1plus"]) for r in rows]
    fa = sum(p and not t for p, t in zip(predicted, true))
    fr = sum(not p and t for p, t in zip(predicted, true))
    pos = sum(true)
    neg = len(true) - pos
    return dict(
        agreement=len(rows) - fa - fr,
        denominator=len(rows),
        false_accepts=fa,
        false_rejects=fr,
        actual_feasible=pos,
        actual_infeasible=neg,
        false_accept_rate=fa / neg if neg else None,
        false_reject_rate=fr / pos if pos else None,
        **{k: mean(float(r[k]) for r in rows) for k in METRICS},
    )


def check_summary(expected, actual):
    for key, value in actual.items():
        require(
            (expected[key] == "" if value is None else close(expected[key], value)),
            "summary differs: " + key,
        )


def main():
    manifest = read_json("MANIFEST.json")
    for name, digest in manifest["files"].items():
        p = ROOT / name
        require(
            p.resolve().is_relative_to(ROOT) and p.is_file(), "unsafe or missing member"
        )
        require(
            hashlib.sha256(p.read_bytes()).hexdigest() == digest,
            "file changed: " + name,
        )
    params = read_json("PARAMETERS.json")
    models = params["models"]
    require(set(models) == MODELS, "predictor population differs")
    require(models["T30-14"]["startup_seconds"] == 30.0, "fixed 30s contrast changed")
    require(
        models["S14"]["startup_seconds"] == 69.219185665, "historical startup changed"
    )
    require(
        models["L14"]["startup_seconds"] == 46.067357474999994,
        "measured startup changed",
    )
    common = {k: v for k, v in models["L14"].items() if k != "startup_seconds"}
    require(
        all(
            {k: v for k, v in m.items() if k != "startup_seconds"} == common
            for m in models.values()
        ),
        "contrast changes a non-startup parameter",
    )
    require(params["guard_model"] == models["L14"], "fixed guard model changed")
    require(
        params["guard_safety"]["startup_seconds"] == models["L14"]["startup_seconds"],
        "guard safety binding changed",
    )
    predictions = read_json("PREDICTION_INDEX.json")
    identity = lambda r: (r["model"], r["policy"], r["window"])
    index = {identity(r): r for r in predictions}
    require(
        len(index) == len(predictions) == 18, "duplicate/missing prediction identities"
    )
    require({r["model"] for r in predictions} == MODELS, "missing prediction model")
    windows = {r["window"] for r in predictions}
    policies = {r["policy"] for r in predictions if r["policy"] != "all2"}
    require(
        len(windows) == 2 and len(policies) == 2, "window or policy population differs"
    )
    require(
        set(index)
        == {(m, p, w) for m in MODELS for p in policies | {"all2"} for w in windows},
        "incomplete prediction Cartesian product",
    )
    require(
        all(
            r["timing"]
            == (
                "post_hoc"
                if r["model"] == "T30-14"
                else "reproduction_of_pre_execution"
            )
            for r in predictions
        ),
        "post-hoc/pre-execution timing mixed",
    )
    pairs = read_csv("POLICY_PAIRS.csv")
    anchors = read_csv("ANCHOR_PAIRS.csv")
    require(len(pairs) == 24 and len(anchors) == 12, "pair denominators differ")
    require(
        len({(r["model"], r["run_id"]) for r in pairs}) == 24, "duplicate policy pair"
    )
    require(len({r["run_id"] for r in pairs}) == 8, "physical policy count differs")
    require(
        len({(r["model"], r["run_id"]) for r in anchors}) == 12, "duplicate anchor pair"
    )
    require(len({r["run_id"] for r in anchors}) == 4, "physical anchor count differs")
    physical = {}
    for rows, anchor in ((pairs, False), (anchors, True)):
        for r in rows:
            require(
                (r["policy"] == "all2") == anchor, "anchor mixed into policy estimate"
            )
            require(identity(r) in index, "unbound model/policy/window pair")
            pred = index[identity(r)]
            require(
                boolean(r["predicted_P1plus"]) is pred["P1plus"],
                "prediction truth changed",
            )
            require(
                close(r["predicted_occupied"], pred["occupied"])
                and close(r["predicted_cost"], pred["cost"]),
                "prediction scalar changed across repeats",
            )
            require(
                close(
                    r[METRICS[0]],
                    abs(float(r["physical_occupied"]) - float(r["predicted_occupied"])),
                ),
                "occupied absolute error does not reproduce",
            )
            require(
                close(
                    r[METRICS[1]],
                    abs(float(r["physical_cost"]) - float(r["predicted_cost"])),
                ),
                "cost absolute error does not reproduce",
            )
            require(
                close(
                    r["target_disagreement_fraction"],
                    int(r["target_disagreement_ticks"]) / 2100,
                ),
                "target error fraction differs",
            )
            require(
                all(math.isfinite(float(r[k])) and float(r[k]) >= 0 for k in METRICS),
                "invalid metric",
            )
            fact = tuple(
                r[k]
                for k in (
                    "window",
                    "policy",
                    "physical_P1plus",
                    "physical_occupied",
                    "physical_cost",
                )
            )
            require(
                physical.setdefault(r["run_id"], fact) == fact,
                "physical result changes by model",
            )
        for key in {identity(r) for r in rows}:
            require(
                sum(identity(r) == key for r in rows) == 2,
                "two physical repeats must reuse one prediction",
            )
    for w in windows:
        a = [index[m, "all2", w] for m in MODELS]
        require(
            all(
                r["P1plus"]
                and close(r["occupied"], a[0]["occupied"])
                and close(r["cost"], a[0]["cost"])
                for r in a
            ),
            "all2 startup invariance failed",
        )
    summary = {r["model"]: r for r in read_csv("MODEL_COMPARISON.csv")}
    per_window = {(r["model"], r["window"]): r for r in read_csv("PER_WINDOW.csv")}
    require(
        set(summary) == MODELS and len(per_window) == 6, "summary population differs"
    )
    original = read_json("FROZEN_S14_L14_SUMMARY.json")
    computed = {}
    for model in sorted(MODELS):
        rows = [r for r in pairs if r["model"] == model]
        require(
            len({(r["policy"], r["window"]) for r in rows}) == 4,
            "unique prediction denominator differs",
        )
        per = [
            reduce_pairs([r for r in rows if r["window"] == w]) for w in sorted(windows)
        ]
        for w, result in zip(sorted(windows), per):
            check_summary(per_window[model, w], result)
        result = reduce_pairs(rows)
        result.update({k: mean(r[k] for r in per) for k in METRICS})
        check_summary(summary[model], result)
        computed[model] = result
        if model in original:
            old = original[model]
            require(
                result["agreement"] == old["feasibility"]["correct"]
                and result["false_accepts"] == old["feasibility"]["false_pass"]
                and result["false_rejects"] == old["feasibility"]["false_block"],
                "original S/L classification changed",
            )
            require(
                all(close(result[k], old["equal_window_means"][k]) for k in METRICS),
                "original S/L metric changed",
            )
    print(
        json.dumps(
            dict(
                status="PASS",
                unique_policy_predictions_per_model=4,
                physical_policy_runs=8,
                physical_anchor_runs=4,
                summary=computed,
                scope="Reduced metric recomputation; not inference replay or raw lifecycle reconstruction.",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
