"""Download explicitly pinned public inputs, never model weights."""
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / 'artifacts/max_optimization_v2_20260916/sources'
BURST_REV = 'd895a53bb7b8ec137d0d2fe203b335835a78c10a'
TOKENIZER_REV = 'a09a35458c702b33eeacc393d103063234e8bc28'
EXPECTED = {
    'BurstGPT_without_fails_1.csv': 'a4d068a7113ec0290e74063a1b3447dc6001a30e4298eb313581b71006dda1f4',
    'burst_README.md': '9d9061ef8548e96a854df7c317b4cdf286745c1a6da2f711f288f5e24aa09ef3',
    'burst_LICENSE': '9e5f1b3c610b9c2da5c313bf81d577a7d1acec686bdb0384edefa6df0f90cd94',
    'python_LICENSE': '78b12c3a81360b357002334f0e70ea0e92eebf7a9b358805c03c48484945f3bb',
    'prompt_train.rst': 'e5da06f2db7559a6539f9a8f48f3ceaa4395f82800160f48f8c4208f4be57c27',
    'prompt_validation.rst': '3c2b7a87519b3e7b1513f5fbb12c9d37ba715c402a6bf032f1b584d555eb979b',
    'prompt_locked_test.rst': '13e550fdc65948d88196eda0985221df240a1dce40dc456e858ef3a55611a606',
    'tokenizer.json': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
    'tokenizer_config.json': '5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583',
    'qwen_LICENSE': '832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e',
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def acquire():
    STORE.mkdir(parents=True, exist_ok=True)
    urls = {
        'BurstGPT_without_fails_1.csv': 'https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_without_fails_1.csv',
        'burst_README.md': f'https://raw.githubusercontent.com/HPMLL/BurstGPT/{BURST_REV}/README.md',
        'burst_LICENSE': f'https://raw.githubusercontent.com/HPMLL/BurstGPT/{BURST_REV}/LICENSE',
        'python_LICENSE': 'https://raw.githubusercontent.com/python/cpython/v3.13.0/LICENSE',
        'prompt_train.rst': 'https://raw.githubusercontent.com/python/cpython/v3.13.0/Doc/tutorial/controlflow.rst',
        'prompt_validation.rst': 'https://raw.githubusercontent.com/python/cpython/v3.13.0/Doc/tutorial/datastructures.rst',
        'prompt_locked_test.rst': 'https://raw.githubusercontent.com/python/cpython/v3.13.0/Doc/tutorial/inputoutput.rst',
        'tokenizer.json': f'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/{TOKENIZER_REV}/tokenizer.json',
        'tokenizer_config.json': f'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/{TOKENIZER_REV}/tokenizer_config.json',
        'qwen_LICENSE': f'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/{TOKENIZER_REV}/LICENSE',
    }
    records = {}
    for name, url in urls.items():
        path = STORE / name
        if not path.exists():
            if shutil.disk_usage(STORE).free < 2 * 1024**3:
                raise RuntimeError('less than 2 GiB disk reserve')
            part = path.with_suffix(path.suffix + '.part')
            with urllib.request.urlopen(url, timeout=40) as response, part.open('xb') as output:
                if response.status != 200:
                    raise ValueError(f'partial download rejected: {name}')
                shutil.copyfileobj(response, output, 1024 * 1024)
                if response.headers.get('Content-Length') and output.tell() != int(response.headers['Content-Length']):
                    raise ValueError(f'truncated download: {name}')
            if digest(part) != EXPECTED[name]:
                raise ValueError(f'wrong pinned content: {name}; partial retained')
            part.rename(path)
        if digest(path) != EXPECTED[name]:
            raise ValueError(f'changed pinned source: {name}')
        records[name] = {'url': url, 'bytes': path.stat().st_size, 'sha256': digest(path)}
        print(name, records[name]['bytes'], records[name]['sha256'], flush=True)
    manifest = {'schema': 'maxopt-sources-v1', 'records': records,
                'weights_downloaded': False, 'burst_revision': BURST_REV,
                'tokenizer_revision': TOKENIZER_REV, 'python_release': 'v3.13.0'}
    target = STORE / 'manifest.json'
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + '\n'
    if target.exists() and target.read_text() != serialized:
        raise ValueError('source inventory changed; use a new version')
    if not target.exists():
        target.write_text(serialized)
    return manifest


if __name__ == '__main__':
    acquire()
