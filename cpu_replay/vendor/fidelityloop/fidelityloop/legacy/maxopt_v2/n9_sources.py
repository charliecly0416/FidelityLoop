"""Collect public primary-source metadata for the V2 manuscript's bounded claims."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

from bs4 import BeautifulSoup

SOURCES={
 'vidur2024':'https://proceedings.mlsys.org/paper_files/paper/2024/hash/b74a8de47d2b3c928360e0a011f48351-Abstract-Conference.html',
 'charon2026':'https://proceedings.mlsys.org/paper_files/paper/2026/hash/dbc8ce0fdfcd55172d73fb05dbae07fc-Abstract-Conference.html',
 'kwon2023vllm':'https://arxiv.org/abs/2309.06180',
 'wang2024burstgpt':'https://arxiv.org/abs/2401.17644v3',
 'qiu2025hygen':'https://arxiv.org/abs/2501.14808',
 'sun2024llumnix':'https://arxiv.org/abs/2406.03243',
 'chen2023frugalgpt':'https://arxiv.org/abs/2305.05176',
 'gujarati2020clockwork':'https://www.usenix.org/conference/osdi20/presentation/gujarati',
 'yu2022orca':'https://www.usenix.org/conference/osdi22/presentation/yu',
 'gujarati2020clockwork_arxiv':'https://arxiv.org/abs/2006.02464',
 'mlsys2027_cfp':'https://mlsys.org/Conferences/2027/CallForPapers',
 'mlsys2027_dates':'https://mlsys.org/Conferences/2027/Dates',
 'mlsys2026_cfp':'https://mlsys.org/Conferences/2026/CallForPapers',
}


def fetch(item):
    key,url=item
    try:
        with urllib.request.urlopen(url,timeout=35) as response:
            blob=response.read();final=response.url
    except Exception as exc:
        return key,dict(url=url,status='UNAVAILABLE',error=str(exc))
    soup=BeautifulSoup(blob,'html.parser')
    metadata={}
    for tag in soup.find_all('meta'):
        name=tag.get('name','')
        if name.startswith('citation_'):metadata.setdefault(name,[]).append(tag.get('content'))
    for tag in soup(['script','style','nav']):tag.decompose()
    body=soup.find('main') or soup.find('article') or soup
    return key,dict(url=url,final_url=final,retrieved_utc=datetime.now(timezone.utc).isoformat(),
                    response_sha256=hashlib.sha256(blob).hexdigest(),metadata=metadata,
                    text=body.get_text(' ',strip=True),scope='primary landing page and abstract; not a full-paper systematic review')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--only',nargs='+',choices=list(SOURCES))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise ValueError('output must be empty')
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(fetch,[(k,v) for k,v in SOURCES.items() if not a.only or k in a.only]))
    for key,result in results:
        (a.output/(key+'.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        print(key, result.get('metadata',{}).get('citation_title'), result.get('metadata',{}).get('citation_author'),result.get('error','OK'))
    (a.output/'manifest.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(a.output.glob('*.json'))},indent=2)+'\n')
