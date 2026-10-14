"""Regression checks for deadline semantics and evidence tampering."""
import unittest
from unittest.mock import patch

import verify


class EvidenceTests(unittest.TestCase):
    def test_unfinished_request_is_infeasible_despite_legacy_flag(self):
        path = next((verify.ROOT / "accounting").glob(
            "hysteresis_u8_d0_c60__bridge_locked_test_steady_v1__r1*"))
        account = verify.read(path)
        self.assertTrue(account["metrics"]["contract_feasible"])
        self.assertFalse(verify.strict_feasibility(account))

    def test_false_timely_label_is_rejected(self):
        path = next((verify.ROOT / "accounting").glob(
            "hysteresis_u8_d0_c60__bridge_locked_test_steady_v1__r1*"))
        account = verify.read(path)
        next(r for r in account["requests"].values() if r["unfinished"])["timely"] = True
        with self.assertRaises(AssertionError):
            verify.strict_feasibility(account)

    def test_wrong_false_reject_denominator_is_rejected(self):
        original = verify.read

        def altered(path):
            value = original(path)
            if path.name == "SUMMARY.json":
                value["model_comparisons"]["AWQ-Hist"]["feasibility"]["false_reject_rate"] = 2 / 8
            return value

        with patch.object(verify, "read", altered), self.assertRaises(AssertionError):
            verify.recompute()

    def test_prediction_hash_tamper_is_rejected(self):
        original = verify.read

        def altered(path):
            value = original(path)
            if path.name == "CPU_PREDICTION_RECEIPT.json":
                value["cells"][0]["sha256"] = "0" * 64
            return value

        with patch.object(verify, "read", altered), self.assertRaises(AssertionError):
            verify.recompute()


if __name__ == "__main__":
    unittest.main()
