"""Recompute online row isolation and the fixed engine-study cost comparison.

Standard library only. This checks released projections, not original prompts
or full GPU journals. Run from any directory; no historical archive is required.
"""
import hashlib
import itertools
import json
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_rows():
    data = json.loads((HERE / "INPUT_ROWS.json").read_text())
    contract_path = HERE.parents[1] / "PPO_TRAINING_CONTRACT.json"
    assert digest(contract_path) == data["training_contract_sha256"]
    expected = json.loads(contract_path.read_text())["selection_contract"]["data_sha256"]
    sets = {}
    validation_matches = 0
    for item in data["sources"]:
        rows = item["online_rows"]
        hashes = [r[0] for r in rows]
        times = [r[1] for r in rows]
        assert all(len(x) == 64 and all(c in "0123456789abcdef" for c in x) for x in hashes)
        assert len(rows) == item["included_online_row_count"]
        assert len(set(hashes)) == item["unique_included_raw_row_hashes"]
        assert min(times) == item["raw_timestamp_s_min"]
        assert max(times) == item["raw_timestamp_s_max"]
        assert item["id"] not in sets
        sets[item["id"]] = set(hashes)
        if item["id"] in expected:
            assert item["sha256"] == expected[item["id"]]
            validation_matches += 1
    assert len(sets) == 8 and validation_matches == 3
    names = list(sets)
    matrix = [[len(sets[a] & sets[b]) for b in names] for a in names]
    assert all(not sets[a] & sets[b] for a, b in itertools.combinations(names, 2))
    return {"source_order": names, "overlap_matrix": matrix,
            "disjoint_source_pairs": 28, "validation_hash_matches": validation_matches,
            "scope": data["scope"]}


def fixed_cost_pair():
    projection = json.loads((HERE / "ENGINE_ROUTE_COUNTS.json").read_text())
    price_path = HERE.parent / "decision_price" / "INPUTS.json"
    assert digest(price_path) == projection["price_input_sha256"]
    rows = json.loads(price_path.read_text())["rows"]
    evidence = {r["record_id"]: r for r in rows if r["campaign"] == "E4" and r["kind"] == "physical"}
    selected = "C30_s105_e0035"
    comparator = "hysteresis_u8_d0_c60__guard"
    windows = ("recovery", "steady")
    groups = {}
    seen = set()
    for item in projection["runs"]:
        assert item["record_id"] not in seen
        seen.add(item["record_id"])
        row = evidence[item["record_id"]]
        assert row["record_sha256"] == item["record_sha256"]
        assert row["policy"] == item["policy"] and row["window"] == item["window"]
        assert row["P1plus"]
        expected_online = {"recovery": 358, "steady": 944}[item["window"]]
        assert item["online_arrivals"] == expected_online
        assert item["terminal_count"] == expected_online + 120
        assert 0 <= item["api_terminal_count"] <= expected_online
        assert abs(sum(row["costs"].values()) - row["cost_total"]) < 1e-9
        groups.setdefault((row["policy"], row["window"]), []).append((item, row))
    assert set(groups) == set(itertools.product((selected, comparator), windows))
    assert all(len(g) == 2 for g in groups.values())
    assert all({item["repeat"] for item, _ in g} == {1, 2} for g in groups.values())
    means, combinations = {}, {}
    for policy in (selected, comparator):
        means[policy] = {
            "cost": mean(mean(row["cost_total"] for _, row in groups[policy, w]) for w in windows),
            "occupied_gpu_seconds": mean(mean(row["costs"]["gpu"] / projection["gpu_price_per_second"]
                                             for _, row in groups[policy, w]) for w in windows),
            "online_api_fraction": mean(mean(item["api_terminal_count"] / item["online_arrivals"]
                                            for item, _ in groups[policy, w]) for w in windows),
        }
        combinations[policy] = [mean(row["cost_total"] for _, row in pair)
                                for pair in itertools.product(*(groups[policy, w] for w in windows))]
    ratios = [100 * (a / b - 1) for a, b in itertools.product(combinations[selected], combinations[comparator])]
    nominal = 100 * (means[selected]["cost"] / means[comparator]["cost"] - 1)
    assert len(ratios) == 16 and round(nominal, 2) == 58.95
    assert round(min(ratios), 2) == 55.44 and round(max(ratios), 2) == 62.62
    assert [round(means[p]["occupied_gpu_seconds"], 2) for p in (selected, comparator)] == [1759.39, 920.79]
    assert [round(100 * means[p]["online_api_fraction"], 2) for p in (selected, comparator)] == [27.83, 61.23]
    return {"fixed_pair": [selected, comparator], "means": means, "nominal_gap_pct": nominal,
            "observed_repeat_envelope_pct": [min(ratios), max(ratios)], "ratio_count": len(ratios),
            "interpretation": "Descriptive envelope of existing repeats for two fixed policies; not a confidence interval."}


def main():
    print(json.dumps({"status": "PASS", "source_row_isolation": source_rows(),
                      "fixed_engine_pair": fixed_cost_pair()}, indent=2))


if __name__ == "__main__":
    main()
