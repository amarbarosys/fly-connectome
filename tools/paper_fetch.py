"""Fetch every paper a paper_pull table lists, through the local crawl4ai server, into one markdown file each.

    python tools/paper_fetch.py docs/research/plasticity_papers.md OUT_DIR [LIMIT]   (OUT_DIR must not exist yet)

Per DOI: OpenAlex gives the open-access URLs (if any) and the abstract. An open-access PDF is downloaded and turned into
text by pdftotext (the crawler's browser cannot open PDF downloads); otherwise crawl4ai fetches the HTML page, else the DOI
landing page. Writes <out>/<n>.md (abstract + fetched text) and <out>/index.json (status per paper: full text or abstract
only, characters, source). A paywalled landing page is kept but marked; never counted as full text.
"""
import json
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_pull import CRAWL, auth  # noqa: E402

FULL_TEXT_CHARS = 15000  # a landing page with only the abstract is far shorter
UA = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) fly-connectome paper_fetch'}


def pdf_text(url):
    data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()
    if not data.startswith(b'%PDF'): raise ValueError('not a PDF (paywall or HTML page)')
    with tempfile.NamedTemporaryFile(suffix='.pdf') as f:
        f.write(data); f.flush()
        r = subprocess.run(['pdftotext', '-enc', 'UTF-8', f.name, '-'], capture_output=True, text=True, timeout=120)
    if r.returncode: raise ValueError(f'pdftotext: {r.stderr.strip()[:80]}')
    return r.stdout


def crawl(url, f='fit'):
    req = urllib.request.Request(CRAWL, json.dumps({'url': url, 'f': f, 'c': '0'}).encode(),
                                 {'Content-Type': 'application/json', 'Authorization': auth()})
    return json.loads(urllib.request.urlopen(req, timeout=180).read()).get('markdown') or ''


def openalex(doi):
    url = 'https://api.openalex.org/works/doi:' + urllib.parse.quote(doi.removeprefix('https://doi.org/'), safe='/')
    md = crawl(url + '?select=title,open_access,best_oa_location,abstract_inverted_index', 'raw')
    return json.loads(md[md.index('{'):md.rindex('}') + 1])


def abstract(inv):
    if not inv: return ''
    return ' '.join(w for _, w in sorted((i, w) for w, idx in inv.items() for i in idx))


def main(table, out, limit=None):
    out = Path(out)
    out.mkdir(parents=True)  # a fresh directory per run: never overwrite an earlier run's papers
    rows, topic = [], None
    for line in Path(table).read_text().splitlines():
        if line.startswith('## '): topic = line[3:]
        m = re.search(r'\((https://doi\.org/[^)]+)\)', line)
        if m and line.startswith('| '): rows.append((topic, line.split('|')[3].strip(), m.group(1)))
    rows = rows[:int(limit)] if limit else rows
    index, failed = [], []
    for n, (topic, title, doi) in enumerate(rows, 1):
        rec = {'n': n, 'topic': topic, 'title': title, 'doi': doi}
        try:
            meta = openalex(doi)
            loc = meta.get('best_oa_location') or {}
            tried, text, src = [], '', None
            tries = [(pdf_text, u) for u in [loc.get('pdf_url')] if u] + [(crawl, u) for u in dict.fromkeys(
                u for u in (loc.get('landing_page_url'), doi) if u)]
            for get, url in tries:
                try: got = get(url)
                except Exception as exc:
                    tried.append(f'{url}: {str(exc)[:60]}'); continue
                if len(got) > len(text): text, src = got, url
                if len(got) >= FULL_TEXT_CHARS: break
            if src is None: raise RuntimeError('; '.join(tried))
            ab = abstract(meta.get('abstract_inverted_index'))
            rec.update(source=src, open_access=bool(loc), chars=len(text), abstract=bool(ab), full_text=len(text) >= FULL_TEXT_CHARS)
            (out / f'{n:02d}.md').write_text(f'# {title}\n\n{topic} · {doi} · source {rec["source"]}\n\n'
                                             f'## Abstract\n\n{ab}\n\n## Fetched text\n\n{text}\n')
        except Exception as exc:
            rec.update(error=str(exc)[:200])
            failed.append(n)
        index.append(rec)
        tag = 'FULL' if rec.get('full_text') else 'PART' if 'chars' in rec else 'FAIL'
        print(f"{n:02d} {tag} {rec.get('chars', 0):>7} {title[:60]}", flush=True)
    (out / 'index.json').write_text(json.dumps(index, indent=1))
    full = sum(r.get('full_text', False) for r in index)
    print(f'{full}/{len(index)} full text, {len(index) - full - len(failed)} partial, {len(failed)} failed')
    if failed: sys.exit(f'FAIL {len(failed)}/{len(index)} {failed}')


if __name__ == '__main__':
    main(*sys.argv[1:])
