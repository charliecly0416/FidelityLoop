"""Build a portable CPU-to-GPU bundle, excluding frozen research worktrees."""
import argparse
import hashlib
import zipfile
from pathlib import Path

from .common import ROOT,DOC,OUT,read,digest,save,verify_protection
from .freeze import verify_lock


def build(destination):
    locked=OUT/'s4';verification=verify_lock(locked);verify_protection()
    destination=Path(destination)
    files=set()
    # All V3 code and only bound read-only V2 dependencies; no original paper or
    # old experiment tree is put in an overwriteable package.
    lock=read(locked/'prediction_lock.json')
    files.update(ROOT/p for p in lock['code'])
    files.update(p for p in locked.rglob('*') if p.is_file())
    files.update(p for p in DOC.glob('*.md'))
    files.update(p for p in (ROOT/'tests/maxopt_v3').glob('*.py'))
    # Test fixtures imported by the compact V3 suite.
    files.add(ROOT/'tests/maxopt_v2/test_calibrated.py')
    files.add(ROOT/'docs/max_optimization_v2_20260915/N1_FEASIBILITY_CONTRACT.json')
    files.update(p for stage in ('s0','s1','s2','s3') for p in (OUT/stage).glob('*.json'))
    manifest={str(p.relative_to(ROOT)):digest(p) for p in sorted(files)}
    destination.parent.mkdir(parents=True,exist_ok=True)
    import json
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted(files):archive.write(p,str(p.relative_to(ROOT)))
        archive.writestr('V3_BUNDLE_MANIFEST.json',json.dumps(dict(schema='maxopt-v3-outbound-v1',files=manifest,
            extraction='NEW EMPTY DIRECTORY ONLY; never extract over V1/V2',gpu_executed=False,
            prediction_lock=verification),indent=2)+'\n')
    # Verify compressed bytes and each archive member, not only an outer SHA.
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:raise ValueError('archive CRC failure')
        for name,expected in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=expected:raise ValueError(name)
    receipt=dict(status='PASS',archive=str(destination),bytes=destination.stat().st_size,
                 sha256=digest(destination),files=len(files),prediction_lock=verification,
                 new_gpu_runs=0,protected_files=verify_protection())
    save(destination.with_suffix(destination.suffix+'.receipt.json'),receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    build(p.parse_args().output)
