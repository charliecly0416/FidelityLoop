"""Public literature verification for Claude R1; keep the first review sources."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import urllib.parse

from .n9_sources import fetch

ARXIV = {
    'miao2023spotserve': '2311.15566', 'griggs2024melange': '2404.14527',
    'zhang2025aapa': '2507.05653', 'cui2026opscale': '2608.13499',
    'patel2024splitwise': '2311.18677', 'zhong2024distserve': '2401.09670',
    'fu2024serverlessllm': '2401.14351', 'agrawal2023sarathi': '2308.16369',
    'zheng2023sglang': '2312.07104', 'sheng2023flexgen': '2303.06865',
    'qin2024mooncake': '2407.00079', 'wan2025bros': '2504.09590',
    'zhang2025kiss': '2507.07932', 'li2023alpaserve': '2302.11665',
    'agrawal2024sarathiserve': '2403.02310',
}
CONTROL = {'forssell1999closedloop': 'Closed-loop identification revisited',
           'hjalmarsson1998ift': 'Iterative feedback tuning: theory and applications'}
DOC = Path(__file__).resolve().parents[2] / 'docs/maxopt_n9_paper_20260920'


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for key, result in pool.map(fetch, [(k, 'https://arxiv.org/abs/' + v) for k, v in ARXIV.items()]):
            save(output / (key + '.json'), result)
            print(key, result.get('metadata', {}).get('citation_title'), result.get('error', ''), flush=True)
    for key, title in CONTROL.items():
        url = 'https://api.crossref.org/works?query.title=' + urllib.parse.quote(title) + '&rows=2'
        with urllib.request.urlopen(url, timeout=35) as response:
            data = response.read()
        matches = json.loads(data)['message']['items']
        exact = [r for r in matches if r['title'][0].lower().replace(':', '') == title.lower().replace(':', '')]
        if len(exact) != 1:
            raise ValueError(('control metadata ambiguous', title, [r['title'] for r in matches]))
        save(output / (key + '.json'), dict(url=url, retrieved_utc=datetime.now(timezone.utc).isoformat(),
             response_sha256=hashlib.sha256(data).hexdigest(), metadata=exact[0], scope='publisher-deposited Crossref metadata'))
        print(key, exact[0]['DOI'], flush=True)
    for key in ('vidur2024', 'charon2026'):
        metadata = json.loads((DOC / 'literature_sources' / (key + '.json')).read_text())
        url = metadata['metadata']['citation_pdf_url'][0]
        with urllib.request.urlopen(url, timeout=45) as response:
            data = response.read()
        assert data.startswith(b'%PDF')
        pdf = output / (key + '.pdf')
        pdf.write_bytes(data)
        subprocess.run(['pdftotext', '-layout', str(pdf), str(pdf.with_suffix('.txt'))], check=True)
        save(output / (key + '_fulltext.json'), dict(url=url, sha256=hashlib.sha256(data).hexdigest(),
             retrieved_utc=datetime.now(timezone.utc).isoformat(), scope='official proceedings full PDF'))
    save(output / 'manifest.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(output.iterdir()) if p.is_file()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
