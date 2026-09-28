"""Build paired literal/correction-aware research artifacts from reviewed local recipes."""

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from research.extraction.prepare_sft import MODEL, REVISION, completion_tokens, prior

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'data/research/extraction/sft-smoke-v1'
DEFAULT = ROOT / 'data/research/extraction/sft-correction-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def apply_edits(original, text, edits):
    result = copy.deepcopy(original)
    seen = set()
    for edit in edits:
        row, field = edit['row'] - 1, edit['field']
        if (row, field) in seen or field not in ('name', 'value', 'unit', 'reference_range'):
            raise ValueError('Duplicate or unsupported edit')
        seen.add((row, field))
        if result['results'][row][field] != edit['before']:
            raise ValueError('Edit before-value mismatch')
        if not edit['reason'] or not edit['spans'] or edit['rule'] not in (2, 3, 4, 5):
            raise ValueError('Missing review evidence')
        for span in edit['spans']:
            if not 0 <= span['start'] < span['end'] <= len(text) or text[span['start']:span['end']] != span['text']:
                raise ValueError('Evidence span mismatch')
        result['results'][row][field] = edit['after']
    check_pair(original, result)
    return result


def translate_warnings(original, edits):
    result = copy.deepcopy(original)
    seen = set()
    for edit in edits:
        index = edit['index']
        if index in seen or not 0 <= index < len(result['errors']) or result['errors'][index] != edit['before']:
            raise ValueError('Warning translation mismatch')
        seen.add(index)
        if not edit['reason'] or not edit['after'] or len(edit['after']) > 160:
            raise ValueError('Invalid warning translation')
        if re.findall(r'\d+(?:[.,]\d+)*', edit['before']) != re.findall(r'\d+(?:[.,]\d+)*', edit['after']):
            raise ValueError('Warning identifying numbers changed')
        result['errors'][index] = edit['after']
    return result


def check_pair(literal, corrected):
    if literal['date'] != corrected['date'] or literal['errors'] != corrected['errors']:
        raise ValueError('Date/warning policy changed')
    if len(literal['results']) != len(corrected['results']):
        raise ValueError('Observation coverage changed')
    for old, new in zip(literal['results'], corrected['results'], strict=True):
        if old['unit'] is None and new['unit'] is not None:
            raise ValueError('Missing unit invented')
        if old['unit'] and new['unit']:
            exponent = r'^(?:\*|x)?10(?:\^|\*|E)(\d+)'
            a, b = re.match(exponent, old['unit']), re.match(exponent, new['unit'])
            if bool(a) != bool(b) or (a and a.group(1) != b.group(1)):
                raise ValueError('Unit exponent changed')
        for field in ('name', 'value', 'reference_range'):
            a, b = old[field], new[field]
            if a == b:
                continue
            if field == 'reference_range' and b is None:
                continue  # Documented abstention, never a reconstructed numeric interval.
            if a is None or b is None:
                raise ValueError('Unsupported missing-field change')
            if re.findall(r'\d+(?:[.,]\d+)*|[<>≤≥+−/:-]', a) != re.findall(r'\d+(?:[.,]\d+)*|[<>≤≥+−/:-]', b):
                raise ValueError('Numeric/sign content changed')
            if field == 'value' and re.search(r'\d', a) and a != b:
                raise ValueError('Numeric measurement edited')
            if field == 'value':
                negative = r'(?i)(?:\bне\b|\bнет\b|\bHeT\b|отри)'
                if bool(re.search(negative, a)) != bool(re.search(negative, b)):
                    raise ValueError('Qualitative polarity changed')


def check_specs(specs, sources):
    families, groups, identifiers, fingerprints = {}, {}, set(), set()
    for item in specs:
        if item['split'] not in ('train', 'validation'):
            raise ValueError('Invalid split')
        for mapping, key in ((families, item['family']), (groups, item['source_group'])):
            if key in mapping and mapping[key] != item['split']:
                raise ValueError('Split leakage')
            mapping[key] = item['split']
        fingerprint = hashlib.sha256(''.join(item['input'].split()).encode()).hexdigest()
        if item['id'] in identifiers or fingerprint in fingerprints:
            raise ValueError('Duplicate synthetic example')
        identifiers.add(item['id'])
        fingerprints.add(fingerprint)
        if any(sources[x]['role'] != 'development' for x in item['development_sources']):
            raise ValueError('Reserved source in synthetic provenance')
        if hashlib.sha256(item['input'].encode()).hexdigest() != item['input_sha256']:
            raise ValueError('Synthetic input hash mismatch')
        if item['review_status'] != 'assistant-reviewed; awaiting user acceptance' or not item['review_notes']:
            raise ValueError('Missing review')
        check_pair(item['literal'], item['corrected'])
    if Counter(s['split'] for s in specs) != {'train': 16, 'validation': 8}:
        raise ValueError('Expected 16 training / 8 validation')
    if Counter(families.values()) != {'train': 8, 'validation': 2}:
        raise ValueError('Family count mismatch')
    for family, split in families.items():
        if sum(s['family'] == family for s in specs) != (2 if split == 'train' else 4):
            raise ValueError('Family membership mismatch')


def build(directory):
    from transformers import AutoTokenizer

    preserved = read(directory / 'baseline_hashes.json')
    for path, expected in preserved.items():
        if digest(BASE / path) != expected:
            raise ValueError(f'Baseline changed: {path}')
    policy_hash = digest(directory / 'policy.md')
    if policy_hash != (directory / 'policy.sha256').read_text().strip():
        raise ValueError('Policy changed after annotation')
    for path, expected in read(directory / 'prompt_hashes.json').items():
        if digest(directory / path) != expected:
            raise ValueError('Prompt changed after annotation')
    for path, expected in read(BASE / 'preserved.json').items():
        if digest(ROOT / path) != expected:
            raise ValueError(f'Historical source changed: {path}')
    sources = {s['id']: s for s in read(BASE / 'sources.json')}
    schema = read(BASE / 'schema.json')
    write(directory / 'schema.json', schema)
    provenance = read(BASE / 'tokenizer/provenance.json')
    if provenance['model'] != MODEL or provenance['revision'] != REVISION:
        raise ValueError('Wrong tokenizer revision')
    for name, expected in provenance['files'].items():
        if digest(BASE / 'tokenizer' / name) != expected:
            raise ValueError('Tokenizer hash mismatch')
    tokenizer = AutoTokenizer.from_pretrained(str(BASE / 'tokenizer'), local_files_only=True)
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != provenance['chat_template_sha256']:
        raise ValueError('Chat template changed')
    write(directory / 'tokenizer_provenance.json', provenance)
    systems = {mode: (directory / f'prompts/{mode}.txt').read_text() for mode in ('literal', 'corrected')}

    def messages(text, target, mode):
        return [{'role': 'system', 'content': systems[mode]},
                {'role': 'user', 'content': 'Extract laboratory observations from this parser output:\n<report>\n' + text + '\n</report>\n'},
                {'role': 'assistant', 'content': json.dumps(target, ensure_ascii=False, indent=2)}]

    specs = read(directory / 'synthetic_specs.json')
    check_specs(specs, sources)
    stats = []
    for mode in systems:
        tokenized = []
        for split in ('train', 'validation'):
            lines = []
            for item in specs:
                if item['split'] != split:
                    continue
                target = item[mode]
                prior.strict_validate(json.dumps(target, ensure_ascii=False), schema)
                msgs = messages(item['input'], target, mode)
                ids, labels, prefix = completion_tokens(tokenizer, msgs)
                lines.append(json.dumps({'id': item['id'], 'messages': msgs}, ensure_ascii=False))
                tokenized.append(json.dumps(dict(id=item['id'], split=split, input_ids=ids, labels=labels)))
                stats.append(dict(id=item['id'], mode=mode, split=split, input_tokens=prefix, total_tokens=len(ids)))
            path = directory / f'synthetic/{mode}/{split}.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('\n'.join(lines) + '\n')
        (directory / f'synthetic/{mode}/tokenized.jsonl').write_text('\n'.join(tokenized) + '\n')
    write(directory / 'synthetic/token_counts.json', stats)
    write(directory / 'synthetic/provenance.json', [{k: v for k, v in s.items() if k not in ('input', 'literal', 'corrected')} for s in specs])
    records = read(directory / 'real_edits.json')
    expected_ids = {i for i in sources if (BASE / f'real/{i}/target.json').exists()}
    if {r['id'] for r in records} != expected_ids or len(records) != len(expected_ids):
        raise ValueError('Real report coverage mismatch')
    real_stats, changes = [], []
    for record in records:
        identifier = record['id']
        if record['role'] != sources[identifier]['role'] or record['status'] != 'assistant-reviewed; awaiting user acceptance':
            raise ValueError('Real review status/role mismatch')
        src, dst = BASE / f'real/{identifier}', directory / f'real/{identifier}'
        for name, key in [('input.txt', 'input_sha256'), ('target.json', 'literal_target_sha256')]:
            if digest(src / name) != record[key]:
                raise ValueError('Real source hash mismatch')
        text = (src / 'input.txt').read_text()
        literal = translate_warnings(read(src / 'target.json'), record.get('warning_edits', []))
        corrected = apply_edits(literal, text, record['edits'])
        prior.strict_validate(json.dumps(corrected, ensure_ascii=False), schema)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / 'input.txt').write_bytes((src / 'input.txt').read_bytes())
        (dst / 'historical-target.json').write_bytes((src / 'target.json').read_bytes())
        write(dst / 'literal-target.json', literal)
        write(dst / 'target.json', corrected)
        write(dst / 'review.json', dict(record, policy_sha256=policy_hash, target_sha256=digest(dst / 'target.json')))
        changes.extend(dict(id=identifier, **edit) for edit in record['edits'])
        for mode, target in [('literal', literal), ('corrected', corrected)]:
            ids = tokenizer.apply_chat_template(messages(text, target, mode), tokenize=True, add_generation_prompt=False)
            ids = ids['input_ids'] if hasattr(ids, 'keys') else ids
            real_stats.append(dict(id=identifier, role=sources[identifier]['role'], mode=mode, total_tokens=len(ids), fits_8192=len(ids) <= 8192))
    write(directory / 'real_token_counts.json', real_stats)
    write(directory / 'changes.json', changes)
    write(directory / 'summary.json', dict(status='assistant-reviewed; awaiting user acceptance', real_reports=len(records),
          real_observations=sum(len(read(directory / f"real/{r['id']}/target.json")['results']) for r in records),
          changed_fields=len(changes), changed_reports=len({c['id'] for c in changes}),
          changed_observations=len({(c['id'], c['row']) for c in changes}),
          changes_by_field=dict(Counter(c['field'] for c in changes)),
          abstentions=sum(c['after'] is None for c in changes), paired_synthetic_examples=len(specs),
          changed_synthetic_targets=sum(s['literal'] != s['corrected'] for s in specs),
          max_synthetic_tokens=max(s['total_tokens'] for s in stats), baseline_files_verified=len(preserved),
          excluded_reports=['D0001', 'D0013'], layout_only=['D0010'], new_api_calls=0, training_runs=0))
    print((directory / 'summary.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    args = parser.parse_args()
    build(args.directory)


if __name__ == '__main__':
    main()
