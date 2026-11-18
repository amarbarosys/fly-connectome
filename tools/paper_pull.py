"""Pull the most-cited relevant papers per topic from OpenAlex, fetched through the local crawl4ai server (private note CLAUDE §4c).

    python tools/paper_pull.py private note OUT "label=query" ["label=query" ...]

Per topic: OpenAlex relevance search, 25 hits, the 10 most cited kept, written as a markdown table with DOI links.
"""
import json
import pathlib
import sys
import urllib.parse
import urllib.request

CRAWL = 'http://127.0.0.1:11235/md'


def auth():
    # The same header Claude Code's crawl4ai MCP entry sends; read at run time, never printed or stored.
    def find(o):
        if isinstance(o, dict):
            if 'crawl4ai' in o and isinstance(o['crawl4ai'], dict): return o['crawl4ai'].get('headers', {}).get('Authorization')
            return next((h for h in map(find, o.values()) if h), None)
    return find(json.load(open(pathlib.Path.home() / '.claude.json')))
FIELDS = 'title,publication_year,doi,cited_by_count,authorships,primary_location'


def fetch(query, n=25):
    url = 'https://api.openalex.org/works?' + urllib.parse.urlencode({'search': query, 'per-page': n, 'select': FIELDS})
    req = urllib.request.Request(CRAWL, json.dumps({'url': url, 'f': 'raw', 'c': '0'}).encode(), {'Content-Type': 'application/json', 'Authorization': auth()})
    md = json.loads(urllib.request.urlopen(req, timeout=120).read())['markdown']
    body = md[md.index('{'):md.rindex('}') + 1]
    return json.loads(body)['results']


def row(w):
    names = [a['author']['display_name'] for a in w.get('authorships', [])]
    who = names[0].split()[-1] + (' et al.' if len(names) > 1 else '') if names else '?'
    venue = ((w.get('primary_location') or {}).get('source') or {}).get('display_name') or ''
    link = f"[doi]({w['doi']})" if w.get('doi') else ''
    return f"| {w.get('publication_year')} | {who} | {(w.get('title') or '').replace('|', '/')} | {venue} | {w.get('cited_by_count')} | {link} |"


def main(out, topics):
    parts, failed = [], []
    for t in topics:
        label, query = t.split('=', 1)
        try: hits = sorted(fetch(query), key=lambda w: -(w.get('cited_by_count') or 0))[:10]
        except Exception as exc:
            failed.append(label); print(f'FAIL {label}: {exc}', file=sys.stderr); continue
        parts += [f'## {label}', f'Query: `{query}` (OpenAlex relevance, top 25, 10 most cited kept)', '',
                  '| Year | First author | Title | Venue | Cited | Link |', '|---|---|---|---|---|---|', *map(row, hits), '']
        print(f'ok {label}: {len(hits)}')
    open(out, 'w').write('\n'.join(parts) + '\n')
    if failed: sys.exit(f'FAIL {len(failed)}/{len(topics)} {failed}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
