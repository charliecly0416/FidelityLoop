"""Build and verify an explicit, portable GPU preflight snapshot. No GPU work."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

MANIFEST = 'PAYLOAD_MANIFEST.json'


def safe_relative(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or any(x in ('.', '..', '') for x in name.split('/')) or '\\' in name or ':' in name:
        raise ValueError('unsafe relative path: ' + name)
    if any(x in ('.git', '.env', 'run.sh', '__pycache__') or x.endswith('.pyc') for x in p.parts):
        raise ValueError('unwanted private/generated file: ' + name)
    return p


def identity(data):
    return dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def build_snapshot(root, files, output):
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError('preserve existing bundle; choose a new output path')
    names = sorted(files)
    if len(names) != len(set(names)) or MANIFEST in names:
        raise ValueError('duplicate or reserved payload path')
    payload = {}
    for name in names:
        safe_relative(name)
        p = root / name
        if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError('payload path is not a local regular file: ' + name)
        payload[name] = p.read_bytes()
    if not payload:
        raise ValueError('empty payload')
    manifest = dict(schema='maxopt-bridge-portable-payload-v1',
                    purpose='development GPU preflight only; no formal test inputs or PPO training',
                    files={n: identity(b) for n, b in payload.items()})
    payload[MANIFEST] = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 30, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, data)
    verify_snapshot(output)
    return dict(path=str(output), **identity(output.read_bytes()), files=len(names),
                manifest_sha256=identity(payload[MANIFEST])['sha256'])


def verify_snapshot(path):
    path = Path(path)
    if path.is_dir():
        manifest = json.loads((path / MANIFEST).read_text())
        found = {p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()}
        if found != set(manifest['files']) | {MANIFEST}:
            raise ValueError('directory inventory differs from manifest; use a clean extraction and external output directory')
        def fetch(name):
            p = path / name
            if p.is_symlink() or not p.resolve().is_relative_to(path.resolve()):
                raise ValueError('payload escapes snapshot')
            return p.read_bytes()
    else:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if len(names) != len(set(names)):
                raise ValueError('duplicate ZIP member')
            for info in z.infolist():
                safe_relative(info.filename)
                if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError('unexpected directory/symlink member')
            payload = {name: z.read(name) for name in names}
        manifest = json.loads(payload[MANIFEST])
        if set(payload) != set(manifest['files']) | {MANIFEST}:
            raise ValueError('ZIP inventory differs from manifest')
        fetch = payload.__getitem__
    if manifest['schema'] != 'maxopt-bridge-portable-payload-v1':
        raise ValueError('unknown payload schema')
    for name, expected in manifest['files'].items():
        safe_relative(name)
        if identity(fetch(name)) != expected:
            raise ValueError('payload checksum mismatch: ' + name)
    return dict(status='PASS_PAYLOAD_INTEGRITY_ONLY', files=len(manifest['files']),
                scope='not scientific or GPU acceptance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_snapshot(args.verify), indent=2))


if __name__ == '__main__':
    main()
