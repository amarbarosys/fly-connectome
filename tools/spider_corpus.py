#!/usr/bin/env python3
"""Build a provenance-preserving, isolated spider literature corpus.

Source text is imported from the existing local spider-docs cache. This first
stage deliberately records unavailable sources without treating them as fetched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCS = Path('/home/<user>/Projects/science/fly-connectome/.work/spider-docs')
DEFAULT_OUT = Path('/home/<user>/Projects/science/fly-connectome/.work/papers/spider')

# Curated starting packet. IDs are stable corpus identifiers; DOI/PMCID provide
# deduplication keys. Unavailable records remain visible and unchecked.
SOURCES = [
    dict(id='nabawy2018', title='Jumping performance of Phidippus regius', species=['Phidippus regius'],
         doi='10.1038/s41598-018-25227-9', pmcid='PMC5940701', filename='PMC5940701.txt', kind='primary'),
    dict(id='deagro2024', title='Multi-eye motion in jumping spiders', species=['Salticidae'],
         doi=None, pmcid='PMC11418026', filename='PMC11418026.txt', kind='primary'),
    dict(id='rossler2022', title='REM sleep in jumping spiders', species=['Salticidae'],
         doi=None, pmcid='PMC9388130', filename='PMC9388130.txt', kind='primary'),
    dict(id='dahl2025', title='Individual recognition in Phidippus regius', species=['Phidippus regius'],
         doi=None, pmcid='PMC12604854', filename='PMC12604854.txt', kind='primary'),
    dict(id='aguilararguello2020', title='Detour distance in Portia spiders', species=['Portia'],
         doi=None, pmcid='PMC7234261', filename='PMC7234261.txt', kind='primary'),
    dict(id='girard2011', title='Vibrational courtship in Maratus volans', species=['Maratus volans'],
         doi=None, pmcid='PMC3181266', filename='PMC3181266.txt', kind='primary'),
    dict(id='jin2023', title='Spider brain single-cell atlas', species=['spider'],
         doi=None, pmcid='PMC10697844', filename='PMC10697844.txt', kind='primary'),
    dict(id='artiushin2026', title='Uloborus brain atlas', species=['Uloborus'],
         doi=None, pmcid='PMC12919972', filename='PMC12919972.txt', kind='primary'),
    dict(id='hung2026', title='Vibration-guided prey capture in Uloborus', species=['Uloborus'],
         doi=None, pmcid='PMC13592698', filename='PMC13592698.txt', kind='primary'),
    dict(id='barth2021', title='Spider sensory guidance and slit sensilla', species=['spider'],
         doi=None, pmcid='PMC8046691', filename='PMC8046691.txt', kind='review'),
    dict(id='winsor2023', title='Distributed Vision in Spiders, chapter 10', species=['Salticidae', 'spider'],
         doi='10.1007/978-3-031-23216-9_10', pmcid=None, filename='winsor2023.txt', kind='review'),
    dict(id='cross2016', title='Portia detour planning', species=['Portia'],
         doi='10.1002/jeab.189', pmcid='PMC4751061', filename=None, xml_file='PMC4751061.xml', kind='primary'),
    dict(id='steck2024', title='Secondary-eye colour vision', species=['Salticidae'],
         doi=None, pmcid='PMC11756594', filename=None, kind='primary'),
]

REF_HEAD = re.compile(r'(?im)^\s*(references|bibliography|literature cited)\s*$')
HEADING = re.compile(r'(?m)^\s*(?:[A-Z][A-Za-z0-9 ,:;()/-]{2,90}|\d+(?:\.\d+)*\s+.{2,80})\s*$')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_text(text: str) -> tuple[str, bool]:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    match = REF_HEAD.search(text)
    if match:
        return text[:match.start()].strip() + '\n', True
    return text.strip() + '\n', False


def jats_body(xml_path: Path) -> str:
    """Extract only JATS body text, with section titles and figure/table captions."""
    root = ET.parse(xml_path).getroot()
    body = next((e for e in root.iter() if e.tag.rsplit('}', 1)[-1] == 'body'), None)
    if body is None:
        # NCBI BioC export: section labels and offsets are carried on each passage.
        if root.tag.rsplit('}', 1)[-1] == 'collection':
            lines = []
            for passage in (e for e in root.iter() if e.tag.rsplit('}', 1)[-1] == 'passage'):
                infons = {e.attrib.get('key'): (e.text or '') for e in passage if e.tag.rsplit('}', 1)[-1] == 'infon'}
                section = infons.get('section_type', infons.get('section_title', 'body'))
                if section.casefold() in {'ref', 'references', 'bibliography'}:
                    continue
                value = next(((e.text or '').strip() for e in passage if e.tag.rsplit('}', 1)[-1] == 'text'), '')
                if value:
                    lines.extend(['', section, '', value])
            if not lines:
                raise ValueError(f'BioC passages missing: {xml_path}')
            return '\n\n'.join(lines).strip() + '\n'
        raise ValueError(f'JATS body missing: {xml_path}')

    def tag(e: ET.Element) -> str:
        return e.tag.rsplit('}', 1)[-1]

    def text(e: ET.Element) -> str:
        return ' '.join(' '.join(e.itertext()).split())

    lines: list[str] = []

    def visit(parent: ET.Element) -> None:
        for child in list(parent):
            name = tag(child)
            if name == 'sec':
                title = next((text(x) for x in list(child) if tag(x) == 'title'), 'Unlabeled section')
                if title.casefold().strip() in {'references', 'bibliography', 'literature cited'}:
                    continue
                lines.extend(['', title, ''])
                visit(child)
            elif name in ('p', 'caption'):
                value = text(child)
                if value:
                    lines.append(value)
            elif name in ('fig', 'table-wrap'):
                caption = next((x for x in child.iter() if tag(x) == 'caption'), None)
                if caption is not None and text(caption):
                    lines.extend(['', f'Caption: {text(caption)}'])
    visit(body)
    return '\n\n'.join(x for x in lines if x.strip()).strip() + '\n'


def chunks(text: str, source_id: str) -> list[dict]:
    """Keep section labels as locators; avoid splitting a labeled section blindly."""
    matches = list(HEADING.finditer(text))
    sections: list[tuple[str, int, int]] = []
    if not matches:
        sections = [('body', 0, len(text))]
    else:
        if matches[0].start() > 0:
            sections.append(('body', 0, matches[0].start()))
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            title = m.group().strip()
            if len(text[m.end():end].strip()) > 0:
                sections.append((title, m.start(), end))
    out = []
    for label, start, end in sections:
        body = text[start:end].strip()
        # Bounded chunks with a stable section + ordinal locator.
        paras = re.split(r'\n\s*\n', body)
        part, size, n = [], 0, 0
        for para in paras:
            if len(para) > 2200:
                units = [para[i:i + 2000] for i in range(0, len(para), 2000)]
            else:
                units = [para]
            for unit in units:
                if part and size + len(unit) > 2200:
                    out.append(dict(source_id=source_id, section=label, locator=f'{label} / passage {n + 1}', text='\n\n'.join(part)))
                    n += 1
                    part, size = [], 0
                part.append(unit)
                size += len(unit)
        if part:
            out.append(dict(source_id=source_id, section=label, locator=f'{label} / passage {n + 1}', text='\n\n'.join(part)))
    return out


def build(docs: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    textdir = out / 'clean'
    textdir.mkdir(exist_ok=True)
    manifest = []
    passages = []
    seen: dict[str, str] = {}
    for source in SOURCES:
        row = dict(source)
        keys = [k.lower() for k in (source.get('doi'), source.get('pmcid')) if k]
        duplicate = next((seen[k] for k in keys if k in seen), None)
        if duplicate:
            raise ValueError(f"duplicate DOI/PMCID for {source['id']}: {duplicate}")
        for k in keys:
            seen[k] = source['id']
        src = docs / source['filename'] if source.get('filename') else None
        xml_path = docs / source['xml_file'] if source.get('xml_file') else (
            docs / source['filename'].replace('.txt', '.xml') if source.get('filename') else None)
        input_path = src if src is not None and src.is_file() else xml_path
        if input_path is None or not input_path.is_file():
            row.update(status='unavailable', source_sha256=None, cleaned=False, passage_count=0)
        else:
            raw = input_path.read_bytes()
            if xml_path and xml_path.is_file():
                txt, references_removed = jats_body(xml_path), True
            else:
                txt, references_removed = clean_text(raw.decode('utf-8', errors='replace'))
            dest = textdir / f"{source['id']}.txt"
            dest.write_text(txt, encoding='utf-8')
            ps = chunks(txt, source['id'])
            passages.extend(ps)
            row.update(status='checked', source_sha256=digest(raw), clean_sha256=digest(txt.encode()),
                       source_path=str(input_path), cleaned=references_removed, passage_count=len(ps))
        manifest.append(row)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    (out / 'passages.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in passages))
    return {'sources': len(manifest), 'checked': sum(x['status'] == 'checked' for x in manifest),
            'unavailable': sum(x['status'] == 'unavailable' for x in manifest), 'passages': len(passages)}


def check(out: Path) -> dict:
    manifest = json.loads((out / 'manifest.json').read_text())
    ids, keys = set(), {}
    for r in manifest:
        assert r['id'] not in ids, f"duplicate id {r['id']}"
        ids.add(r['id'])
        for key in (r.get('doi'), r.get('pmcid')):
            if key:
                k = key.casefold()
                assert k not in keys, f"duplicate {key}: {keys.get(k)} / {r['id']}"
                keys[k] = r['id']
        if r['status'] == 'checked':
            assert len(r['source_sha256']) == 64 and len(r['clean_sha256']) == 64
            source_path = Path(r['source_path'])
            assert source_path.is_file() and digest(source_path.read_bytes()) == r['source_sha256'], f"source hash changed: {r['id']}"
            clean = (out / 'clean' / f"{r['id']}.txt").read_text()
            assert digest(clean.encode()) == r['clean_sha256'], f"clean hash changed: {r['id']}"
            assert not REF_HEAD.search(clean), f"reference section retained in {r['id']}"
        else:
            assert r['status'] == 'unavailable' and r['passage_count'] == 0
    count = 0
    with (out / 'passages.jsonl').open() as f:
        for line in f:
            p = json.loads(line)
            assert p['source_id'] in ids and p['locator'] and p['text'].strip()
            assert not REF_HEAD.search(p['text']), f"reference passage: {p['locator']}"
            count += 1
    assert count == sum(r['passage_count'] for r in manifest)
    return {'sources': len(manifest), 'checked': sum(x['status'] == 'checked' for x in manifest),
            'unavailable': sum(x['status'] == 'unavailable' for x in manifest), 'passages': count,
            'dedup_keys': len(keys), 'stable_hashes': 'PASS', 'reference_filter': 'PASS', 'locators': 'PASS'}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['build', 'check'])
    ap.add_argument('--docs', type=Path, default=DEFAULT_DOCS)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    result = build(args.docs, args.out) if args.command == 'build' else check(args.out)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
