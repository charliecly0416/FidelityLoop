"""Reject corrupted saved ledgers without importing the simulator."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from verify_ledger import audit

SOURCE = Path(__file__).resolve().parents[1] / 'runtime/static_equivalence'


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        for name in ['summary.json', 'records.json', 'decisions.jsonl', 'events.jsonl']:
            shutil.copy2(SOURCE / name, self.path / name)

    def events(self):
        return [json.loads(l) for l in (self.path / 'events.jsonl').read_text().splitlines()]

    def save_events(self, events):
        (self.path / 'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))

    def test_valid_native_case(self):
        self.assertEqual(audit(self.path)['status'], 'PASS')

    def test_duplicate_dispatch_rejected(self):
        events = self.events()
        i = next(i for i,e in enumerate(events) if e['kind'] == 'dispatch')
        events.insert(i+1, dict(events[i])); self.save_events(events)
        with self.assertRaises(AssertionError): audit(self.path)

    def test_generation_crossing_rejected(self):
        events = self.events()
        e = next(e for e in events if e['kind'] == 'complete')
        e['generation'] += 1; self.save_events(events)
        with self.assertRaises(AssertionError): audit(self.path)

    def test_missing_completion_rejected(self):
        records = json.loads((self.path / 'records.json').read_text())
        records[next(iter(records))]['completed_at'] = 1e6
        (self.path / 'records.json').write_text(json.dumps(records))
        with self.assertRaises(AssertionError): audit(self.path)

    def test_false_resource_total_rejected(self):
        summary = json.loads((self.path / 'summary.json').read_text())
        summary['occupied_seconds'] += 1
        (self.path / 'summary.json').write_text(json.dumps(summary))
        with self.assertRaises(AssertionError): audit(self.path)

    def test_missing_noop_tick_rejected(self):
        path=self.path/'decisions.jsonl'
        rows=path.read_text().splitlines()
        path.write_text('\n'.join(rows[:-1])+'\n')
        with self.assertRaises(AssertionError): audit(self.path)

    def test_future_field_rejected(self):
        rows = [json.loads(l) for l in (self.path / 'decisions.jsonl').read_text().splitlines()]
        rows[0]['observation']['ready_at'] = 8
        (self.path / 'decisions.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in rows))
        with self.assertRaises(AssertionError): audit(self.path)


if __name__ == '__main__': unittest.main()
