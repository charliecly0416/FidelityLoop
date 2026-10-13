"""Ensure reduced evidence is checked independently of its expected summaries."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import verify

class ReducedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for f in verify.ROOT.iterdir():
            if f.is_file():shutil.copy2(f,self.root/f.name)
        old=verify.ROOT;verify.ROOT=self.root
        self.addCleanup(setattr,verify,'ROOT',old)

    def rewrite(self,name,change):
        with gzip.open(self.root/name,'rt',newline='') as f:rows=list(csv.DictReader(f))
        fields=list(rows[0]);change(rows)
        with gzip.open(self.root/name,'wt',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        manifest=json.loads((self.root/'FILES_SHA256.json').read_text())
        manifest[name]=hashlib.sha256((self.root/name).read_bytes()).hexdigest()
        (self.root/'FILES_SHA256.json').write_text(json.dumps(manifest))

    def test_complete_evidence(self):self.assertEqual(verify.recompute()['status'],'PASS')

    def test_missing_control_tick(self):
        self.rewrite('targets.csv.gz',lambda rows:rows.pop())
        with self.assertRaises(AssertionError):verify.recompute()

    def test_changed_occupation(self):
        self.rewrite('occupation_intervals.csv.gz',lambda rows:rows[0].update(end_s=float(rows[0]['end_s'])-0.01))
        with self.assertRaises(AssertionError):verify.recompute()

    def test_disguised_unfinished_request(self):
        self.rewrite('requests.csv.gz',lambda rows:rows[0].update(completed_at='',status='unfinished'))
        with self.assertRaises(AssertionError):verify.recompute()

if __name__=='__main__':unittest.main()
