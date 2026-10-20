"""Small semantic checks, independent of archived numeric reference summaries."""
import copy
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from verify import verify
from load_trace import measure


def fixture():
    return {"horizon_s": 10, "prices": {"gpu_second": 2, "startup_event": 3, "shutdown_event": 5, "api_input_token": 0.1, "api_output_token": 0.2, "offline_deadline_miss": 7},
        "requests": [
          {"request_id": "a", "job_type": "online", "arrival_s": 0, "deadline_s": 4, "input_tokens": 2, "max_output_tokens": 3, "status": "completed", "completed_at_s": 4},
          {"request_id": "b", "job_type": "offline", "arrival_s": 1, "deadline_s": 9, "input_tokens": 2, "max_output_tokens": 3, "status": "completed", "completed_at_s": 9}],
        "resource_intervals": [{"gpu": 0, "generation": 1, "start_s": -2, "end_s": 6}, {"gpu": 0, "generation": 2, "start_s": 9, "end_s": 12}],
        "ledger_events": [{"time_s": -2, "kind": "start", "request_id": None}, {"time_s": 9, "kind": "start", "request_id": None}, {"time_s": 6, "kind": "stop", "request_id": None}, {"time_s": 10, "kind": "stop", "request_id": None}, {"time_s": 0, "kind": "api_accept", "request_id": "a"}]}

class TraceTests(unittest.TestCase):
    def test_window_clipping_and_half_open_charges(self):
        out = measure(fixture())
        self.assertEqual(out["gpu_occupied_seconds"], 7)
        self.assertAlmostEqual(out["cost_total"], 14 + 3 + 5 + 0.8)
        self.assertTrue(out["P1plus"])  # deadline equality is on time

    def test_late_and_unfinished_retained(self):
        r = fixture(); r["requests"][1]["completed_at_s"] = 9.5
        out = measure(r)
        self.assertFalse(out["P1plus"])
        self.assertEqual(out["populations"]["offline"]["unfinished"], 0)
        self.assertEqual(out["costs"]["offline_miss"], 7)
        r["requests"][1]["completed_at_s"] = 11
        out = measure(r)
        self.assertEqual(out["populations"]["offline"]["unfinished"], 1)
        self.assertEqual(out["populations"]["offline"]["arrivals"], 1)

    def test_duplicate_requests_rejected(self):
        r = fixture(); r["requests"].append(copy.deepcopy(r["requests"][0]))
        with self.assertRaises(ValueError): measure(r)

    def test_overlap_and_duplicate_api_rejected(self):
        r = fixture(); r["resource_intervals"][1]["start_s"] = 5
        with self.assertRaises(ValueError): measure(r)
        r = fixture(); r["ledger_events"].append(copy.deepcopy(r["ledger_events"][-1]))
        with self.assertRaises(ValueError): measure(r)

    def test_unknown_request_not_silently_dropped(self):
        r = fixture(); r["ledger_events"][-1]["request_id"] = "unknown"
        with self.assertRaises(ValueError): measure(r)

class ReleaseInventoryTests(unittest.TestCase):
    def check_inventory(self, git_directory, extra_file=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "MANIFEST.json").write_text(json.dumps({"files": []}))
            if git_directory:
                (root / ".git").mkdir()
                (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
            else:
                (root / ".git").write_text("gitdir: metadata\n")
            if extra_file:
                (root / "unexpected.txt").write_text("not in the release")
            # Stop after inventory validation; no scientific corpus is needed here.
            with patch("verify.read_index", side_effect=RuntimeError("inventory passed")):
                if extra_file:
                    with self.assertRaisesRegex(ValueError, "release inventory mismatch"):
                        verify(root)
                else:
                    with self.assertRaisesRegex(RuntimeError, "inventory passed"):
                        verify(root)

    def test_clone_metadata_is_not_a_release_member(self):
        self.check_inventory(git_directory=True)

    def test_worktree_metadata_is_not_a_release_member(self):
        self.check_inventory(git_directory=False)

    def test_extra_release_member_is_still_rejected(self):
        self.check_inventory(git_directory=True, extra_file=True)


if __name__ == "__main__": unittest.main()
