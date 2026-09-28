"""Prepare a bounded, private SFT packet without changing historical experiments."""

import argparse
import asyncio
import fcntl
import json
import time
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from research.extraction import evaluate_image_extraction as prior

ROOT = prior.ROOT
DEFAULT = ROOT / 'data/research/extraction/sft-smoke-v1'
REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'
MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
RESERVED = {'KDL', 'CitiLab', 'Helix'}
PILOT = ROOT / 'data/extraction_poc/image-extraction-pilot-v1'
ORIGINAL = ROOT / 'data/extraction_poc/qwen-eval-packet-v1'
read, save, digest, encoded, Halt = prior.read, prior.save, prior.digest, prior.encoded, prior.Halt


def messages(text):
    user, system, _ = prior.PromptEngine(ROOT / 'backend/prompts').render(
        prior.PromptTypeEnum.laboratory_test_processor, 'v4', input_text=text)
    return [{'role': 'system', 'content': system['system_instruction']}, {'role': 'user', 'content': user}]


def organization(identifier):
    if identifier[0] in 'ABC':
        return {'A': 'Invitro', 'B': 'KDL', 'C': 'Gemotest'}[identifier[0]]
    n = int(identifier[1:])
    return next(org for org, ids in {
        'CMD': [1, 6, 7], 'Lab4U': [2, 11, 12], 'CitiLab': [3, 4, 5],
        'Helix': [8, 9, 10], 'UniLab': [13, 14, 15, 16], 'KDL': [17, 18],
    }.items() if n in ids)


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=False)
    prompt = ROOT / 'backend/prompts/laboratory_test_processor/v4.yaml'
    if digest(prompt.read_bytes()) != prior.PROMPT_HASH:
        raise Halt('Frozen prompt changed')
    save(directory / 'prompt.yaml', prompt.read_bytes())
    save(directory / 'schema.json', read(PILOT / 'schema.json'))
    sources, preserved = [], {}
    for p in (ROOT / 'data/extraction_poc').rglob('*'):
        if p.is_file():
            preserved[str(p.relative_to(ROOT))] = digest(p.read_bytes())
    for item in read(ORIGINAL / 'manifest.json')['documents']:
        identifier = item['document_id']
        org = organization(identifier)
        entry = dict(id=identifier, organization=org, role='reserved' if org in RESERVED else 'development',
                     source_group=identifier, template_group=org, input_kind='raw-parser',
                     exposure='Historical contract and Gemini review; B has no small-model exposure.',
                     original_mapping=item['sources'], duplicate_paths=item.get('duplicate_pdf_paths', []),
                     status='historical-reviewed', pages=[])
        for kind, key in [('input.txt', 'input'), ('target.json', 'target')]:
            raw = (ORIGINAL / item[key]).read_bytes()
            save(directory / f'real/{identifier}/{kind}', raw)
        sources.append(entry)
    for n in range(1, 19):
        identifier = f'D{n:04d}'
        org = organization(identifier)
        entry = dict(id=identifier, organization=org,
                     role='layout-only' if n == 10 else ('reserved' if org in RESERVED else 'development'),
                     source_group=identifier, template_group=org, input_kind='raw-parser',
                     exposure='Prior visual inventory; D0002/D0007/D0016 also used in extraction pilot.',
                     status='excluded-layout' if n == 10 else 'pending', pages=[])
        candidates = sorted((ROOT / 'data/approved_image_redactions').glob(f'{identifier}-*.json'))
        seen = set()
        for p in candidates:
            m = read(p)
            png = p.with_suffix('.png')
            if m['approval_status'] != 'kept' or digest(png.read_bytes()) != m['export_sha256']:
                raise Halt(f'Unapproved or changed export: {p.name}')
            page = m.get('page_number', 1)
            if page in seen:
                raise Halt(f'Multiple approved versions require explicit selection: {identifier}/{page}')
            seen.add(page)
            stem = f'{identifier}-p{page:03d}'
            save(directory / f'images/{stem}.png', png.read_bytes())
            save(directory / f'images/{stem}.json', p.read_bytes())
            entry['pages'].append(dict(id=stem, number=page, path=f'images/{stem}.png',
                                       sha256=m['export_sha256'], source=str(png.relative_to(ROOT))))
            preserved[str(png.relative_to(ROOT))] = digest(png.read_bytes())
            preserved[str(p.relative_to(ROOT))] = digest(p.read_bytes())
        if not entry['pages'] or sorted(seen) != list(range(1, len(seen) + 1)):
            raise Halt(f'Missing pages: {identifier}')
        entry['pages'].sort(key=lambda x: x['number'])
        if identifier in prior.SOURCES:
            save(directory / f'real/{identifier}/input.txt', (PILOT / f'parse/{identifier}/input.txt').read_bytes())
            save(directory / f'real/{identifier}/historical-corrected.txt', (PILOT / f'corrected/{identifier}.txt').read_bytes())
            save(directory / f'real/{identifier}/historical-corrected-target.json', (PILOT / f'targets/{identifier}.json').read_bytes())
            entry['status'] = 'raw-target-review-pending'
        sources.append(entry)
    save(directory / 'sources.json', sources)
    save(directory / 'preserved.json', preserved)
    save(directory / 'protocol.json', dict(
        created=prior.now(), status='awaiting-user-acceptance', prompt_sha256=prior.PROMPT_HASH,
        parser=prior.PARSER_SETTINGS, parser_sdk='0.6.94', google_sdk='1.75.0',
        gemini=read(PILOT / 'protocol.json')['gemini'],
        limits=dict(parse=22, gemini=14, total_usd=5),
        prices=dict(parser_page_usd=0.00125, gemini_input_per_million=0.25, gemini_output_per_million=1.5),
        price_sources=['https://developers.llamaindex.ai/llamaparse/general/pricing/',
                       'https://ai.google.dev/gemini-api/docs/pricing'],
        price_checked='2026-09-27', tokenizer=dict(model=MODEL, revision=REVISION),
        user_authorization='2026-09-27: implement the full 24-example preparation plan; no training or Qwen requests.',
    ))
    save(directory / 'ledger.json', dict(attempts=[], reserved_usd=0))
    hashes = {str(p.relative_to(directory)): digest(p.read_bytes()) for p in directory.rglob('*')
              if p.is_file() and p.name != 'ledger.json'}
    save(directory / 'initial_hashes.json', hashes)


def check_integrity(directory):
    for name, expected in read(directory / 'initial_hashes.json').items():
        if digest((directory / name).read_bytes()) != expected:
            raise Halt(f'Packet input changed: {name}')
    if digest((ROOT / 'backend/prompts/laboratory_test_processor/v4.yaml').read_bytes()) != prior.PROMPT_HASH:
        raise Halt('Active frozen prompt changed')


def reserve(directory, stage, identifier, cost, source_hash):
    ledger = read(directory / 'ledger.json')
    old = next((x for x in ledger['attempts'] if (x['stage'], x['id']) == (stage, identifier)), None)
    if old:
        if old['source_sha256'] != source_hash:
            raise Halt('Request input changed')
        if old['status'] == 'success' or (stage == 'parse' and old.get('job_id') and old['status'] != 'service_failed'):
            return old
        raise Halt(f'Uncertain or failed request cannot be resubmitted: {identifier}')
    limits = read(directory / 'protocol.json')['limits']
    if sum(x['stage'] == stage for x in ledger['attempts']) >= limits[stage]:
        raise Halt('Request ceiling reached')
    if ledger['reserved_usd'] + cost > limits['total_usd']:
        raise Halt('Cost ceiling reached')
    entry = dict(stage=stage, id=identifier, status='submission_uncertain', source_sha256=source_hash,
                 reserved_usd=cost, started=prior.now())
    ledger['attempts'].append(entry)
    ledger['reserved_usd'] += cost
    save(directory / 'ledger.json', ledger, replace=True)
    return entry


def update(directory, entry, **changes):
    ledger = read(directory / 'ledger.json')
    entry.update(changes)
    for i, old in enumerate(ledger['attempts']):
        if (old['stage'], old['id']) == (entry['stage'], entry['id']):
            ledger['attempts'][i] = entry
    save(directory / 'ledger.json', ledger, replace=True)


def access(directory):
    check_integrity(directory)
    record = read(directory / 'access.json')
    if record['status'] != 'verified' or record['parser_available_pages'] < 22 or record['gemini_available_requests'] < 14:
        raise Halt('Insufficient documented access/quota')
    for package, key in [('llama-parse', 'parser_sdk'), ('google-genai', 'google_sdk')]:
        if version(package) != read(directory / 'protocol.json')[key]:
            raise Halt(f'Unexpected SDK: {package}')


def join_pages(pages):
    ordered = sorted(pages, key=lambda x: x['number'])
    if [p['number'] for p in ordered] != list(range(1, len(pages) + 1)):
        raise Halt('Missing/duplicate page numbers')
    text, boundaries = '', []
    for p in ordered:
        if boundaries:
            text += '\n'
        start = len(text)
        text += p['text']
        boundaries.append(dict(number=p['number'], start=start, end=len(text)))
    return text, boundaries


async def parse(directory):
    import llama_cloud_services.parse.base as parser_module
    from llama_parse import LlamaParse

    from configs.models import LlamaParseSettings

    access(directory)
    client = LlamaParse(api_key=LlamaParseSettings().api_key.get_secret_value(), **prior.PARSER_SETTINGS)
    if client.disable_ocr or client.fast_mode or client.max_pages is not None:
        raise Halt('Unexpected OCR/page settings')

    async def one_request(http, method, url, timeout=60, **kwargs):
        response = await http.request(method, url, timeout=timeout, **kwargs)
        response.raise_for_status()
        return response

    async with client.aclient as http:
        for source in read(directory / 'sources.json'):
            if source['status'] != 'pending':
                continue
            parsed = []
            for page in source['pages']:
                path = directory / page['path']
                entry = reserve(directory, 'parse', page['id'], .00125, digest(path.read_bytes()))
                out = directory / f"parse/{page['id']}"
                if entry['status'] != 'success':
                    try:
                        if not entry.get('job_id'):
                            with patch.object(parser_module, 'make_api_request', one_request):
                                job = await client._create_job(str(path))
                            update(directory, entry, job_id=job, status='submitted')
                        deadline = time.monotonic() + 2000
                        while True:
                            response = await http.get(f"/api/parsing/job/{entry['job_id']}", timeout=60)
                            response.raise_for_status()
                            status = response.json()['status']
                            if status == 'SUCCESS':
                                break
                            if status != 'PENDING':
                                update(directory, entry, status='service_failed')
                                raise Halt(f'Parser status {status}')
                            if time.monotonic() >= deadline:
                                raise Halt('Parser polling deadline exceeded')
                            await asyncio.sleep(5)
                        response = await http.get(f"/api/parsing/job/{entry['job_id']}/result/text", timeout=60)
                        response.raise_for_status()
                        result = response.json()
                        save(out / 'response.json', result)
                        text = result['text']
                        if not isinstance(text, str):
                            raise Halt('Parser text missing')
                        separator = client.page_separator or parser_module._DEFAULT_SEPARATOR
                        text = '\n'.join(text.split(separator)) if client.split_by_page else text
                        save(out / 'input.txt', text.encode())
                        update(directory, entry, status='success', finished=prior.now(), text_sha256=digest(text.encode()), response_sha256=digest((out / 'response.json').read_bytes()))
                    except BaseException as exc:
                        update(directory, entry, error_type=type(exc).__name__)
                        raise Halt(f"Parser stopped: {page['id']} ({type(exc).__name__}); no resubmission") from None
                text = (out / 'input.txt').read_text()
                if digest(text.encode()) != entry['text_sha256']:
                    raise Halt('Saved parser output changed')
                parsed.append(dict(number=page['number'], text=text))
                print('parsed', page['id'], flush=True)
            text, boundaries = join_pages(parsed)
            save(directory / f"real/{source['id']}/input.txt", text.encode())
            save(directory / f"real/{source['id']}/pages.json", boundaries)


def annotate(directory):
    from google import genai
    from google.genai import types

    from configs.models import GoogleGeminiSettings

    access(directory)
    if not (directory / 'annotation_rules.sha256').exists():
        raise Halt('Freeze development-derived annotation rules first')
    if digest((directory / 'annotation_rules.md').read_bytes()) != (directory / 'annotation_rules.sha256').read_text().strip():
        raise Halt('Annotation rules changed after freeze')
    protocol = read(directory / 'protocol.json')
    config = dict(protocol['gemini'])
    model = config.pop('model')
    config['system_instruction'] = messages('')[0]['content']
    with genai.Client(api_key=GoogleGeminiSettings(model_name=model).api_key.get_secret_value(),
                      http_options=types.HttpOptions(timeout=600000, retry_options=types.HttpRetryOptions(attempts=1))) as client:
        sources = sorted(read(directory / 'sources.json'), key=lambda x: (x['role'] == 'reserved', x['id']))
        for source in sources:
            if source['status'] != 'pending':
                continue
            identifier = source['id']
            text = (directory / f'real/{identifier}/input.txt').read_text()
            msgs = messages(text)
            bound = sum(len(m['content'].encode()) for m in msgs) + len(encoded(config)) + 1024
            cost = (bound * .25 + 16384 * 1.5) / 1_000_000
            attempts = read(directory / 'ledger.json')['attempts']
            existing = next((x for x in attempts if x['stage'] == 'gemini' and x['id'] == identifier), None)
            if existing is None:
                quota = read(directory / 'access.json')
                interval = max(10, 60 / quota['gemini_rpm'], 60 * (bound + 16384) / quota['gemini_tpm'])
                previous = [x for x in attempts if x['stage'] == 'gemini']
                if previous:
                    remaining = interval - (time.time() - prior.datetime.fromisoformat(previous[-1]['started']).timestamp())
                    while remaining > 0:
                        time.sleep(min(remaining, 30))
                        remaining = interval - (time.time() - prior.datetime.fromisoformat(previous[-1]['started']).timestamp())
            entry = reserve(directory, 'gemini', identifier, cost, digest(text.encode()))
            if entry['status'] == 'success':
                continue
            out = directory / f'real/{identifier}/teacher'
            save(out / 'request.json', dict(model=model, contents=msgs[1]['content'], config=config))
            try:
                response = client.models.generate_content(model=model, contents=msgs[1]['content'], config=config)
                raw = response.model_dump(mode='json')
                save(out / 'response.json', raw)
                candidates = raw.get('candidates', [])
                if len(candidates) != 1 or candidates[0].get('finish_reason') != 'STOP':
                    raise Halt('Incomplete teacher output')
                parts = candidates[0].get('content', {}).get('parts', [])
                content = ''.join(p.get('text', '') for p in parts if not p.get('thought'))
                save(out / 'draft.txt', content.encode())
                target = prior.strict_validate(content, read(directory / 'schema.json'))
                save(out / 'draft.json', target)
                update(directory, entry, status='success', finished=prior.now(), usage=raw.get('usage_metadata'), response_sha256=digest((out / 'response.json').read_bytes()), draft_sha256=digest((out / 'draft.json').read_bytes()))
            except BaseException as exc:
                update(directory, entry, error_type=type(exc).__name__)
                raise Halt(f'Teacher stopped: {identifier} ({type(exc).__name__}); no retry') from None
            print('annotated', identifier, flush=True)


def check_review(text, target, review, schema):
    prior.strict_validate(json.dumps(target, ensure_ascii=False), schema)
    if review['input_sha256'] != digest(text.encode()) or review['target_sha256'] != digest(encoded(target)):
        raise Halt('Review hash mismatch')
    if review['status'] != 'assistant-reviewed' or len(review['rows']) != len(target['results']):
        raise Halt('Incomplete review')
    for row, evidence in zip(target['results'], review['rows'], strict=True):
        for field, value in row.items():
            item = evidence[field]
            if not item['reason']:
                raise Halt('Missing field evidence reason')
            for span in item['spans']:
                if not 0 <= span['start'] < span['end'] <= len(text) or text[span['start']:span['end']] != span['text']:
                    raise Halt('Invalid field evidence span')
            if value is not None and not item['spans']:
                raise Halt('Non-null field without source evidence')
    if not review.get('date_evidence') or not review.get('coverage') or 'corrections' not in review:
        raise Halt('Date/coverage/correction review missing')


def review_stage(directory):
    schema = read(directory / 'schema.json')
    for source in read(directory / 'sources.json'):
        if source['status'] in ('historical-reviewed', 'excluded-layout'):
            continue
        root = directory / f"real/{source['id']}"
        review = read(root / 'review.json')
        if review['status'] == 'excluded-unresolved':
            if review.get('input_sha256') != digest((root / 'input.txt').read_bytes()) or not review.get('unresolved') or not review.get('coverage'):
                raise Halt('Exclusion requires input-bound reasons and coverage review')
            continue
        check_review((root / 'input.txt').read_text(), read(root / 'target.json'), review, schema)
    print('All 17 raw-input cases have source-bound reviews or explicit unresolved exclusions.')


def completion_tokens(tokenizer, msgs):
    prefix = tokenizer.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    if hasattr(prefix, 'keys'):
        prefix = prefix['input_ids']
    if hasattr(full, 'keys'):
        full = full['input_ids']
    if full[:len(prefix)] != prefix:
        raise Halt('Assistant boundary is not a token prefix')
    if len(full) > 8192:
        raise Halt('Full conversation exceeds 8192 tokens; no truncation')
    labels = [-100] * len(prefix) + full[len(prefix):]
    if len(full) == len(prefix) or tokenizer.eos_token_id not in full[len(prefix):]:
        raise Halt('Assistant completion/end token missing')
    return full, labels, len(prefix)


def synthetic(directory):
    check_integrity(directory)
    sources = {s['id']: s for s in read(directory / 'sources.json')}
    specs = read(directory / 'synthetic/specs.json')
    schema = read(directory / 'schema.json')
    outputs = {'train': [], 'validation': []}
    groups, inputs, identifiers, source_groups = {}, set(), set(), {}
    for spec in specs:
        split, family = spec['split'], spec['family']
        if split not in outputs or (family in groups and groups[family] != split):
            raise Halt('Template-family split leakage')
        groups[family] = split
        if spec['id'] in identifiers:
            raise Halt('Duplicate synthetic ID')
        identifiers.add(spec['id'])
        source_group = spec['source_group']
        if source_group in source_groups and source_groups[source_group] != split:
            raise Halt('Source-group split leakage')
        source_groups[source_group] = split
        for identifier in spec['development_sources']:
            if sources[identifier]['role'] != 'development' or sources[identifier]['organization'] in RESERVED:
                raise Halt('Reserved source in synthetic provenance')
        fingerprint = digest(''.join(spec['input'].split()).encode())
        if fingerprint in inputs:
            raise Halt('Duplicate synthetic input')
        inputs.add(fingerprint)
        check_review(spec['input'], spec['target'], spec['review'], schema)
        msgs = messages(spec['input']) + [{'role': 'assistant', 'content': encoded(spec['target']).decode().strip()}]
        outputs[split].append(dict(id=spec['id'], messages=msgs))
    if {s: len(v) for s, v in outputs.items()} != {'train': 16, 'validation': 8}:
        raise Halt('Expected 16/8 smoke examples')
    if {s: sum(v == s for v in groups.values()) for s in outputs} != {'train': 8, 'validation': 2}:
        raise Halt('Expected eight training/two validation families')
    for family, split in groups.items():
        if sum(s['family'] == family for s in specs) != (2 if split == 'train' else 4):
            raise Halt('Unexpected smoke family size')
    for split, rows in outputs.items():
        save(directory / f'synthetic/{split}.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode())
    save(directory / 'synthetic/provenance.json', [{k: v for k, v in s.items() if k not in ('input', 'target', 'review')} for s in specs])


def validate(directory, synthetic_only=False):
    from transformers import AutoTokenizer

    check_integrity(directory)
    if not synthetic_only:
        review_stage(directory)
    synthetic(directory)  # Recheck reviews/provenance and byte identity of published JSONL.
    provenance = read(directory / 'tokenizer/provenance.json')
    if provenance['model'] != MODEL or provenance['revision'] != REVISION:
        raise Halt('Wrong tokenizer checkpoint')
    for name, expected in provenance['files'].items():
        if digest((directory / 'tokenizer' / name).read_bytes()) != expected:
            raise Halt('Tokenizer file changed')
    tokenizer = AutoTokenizer.from_pretrained(str(directory / 'tokenizer'), local_files_only=True)
    if digest(tokenizer.chat_template.encode()) != provenance['chat_template_sha256']:
        raise Halt('Tokenizer chat template changed')
    stats, tokenized = [], []
    for split in ('train', 'validation'):
        for line in (directory / f'synthetic/{split}.jsonl').read_text().splitlines():
            row = json.loads(line)
            full, labels, prefix = completion_tokens(tokenizer, row['messages'])
            stats.append(dict(id=row['id'], split=split, input_tokens=prefix, target_tokens=len(full)-prefix, total_tokens=len(full)))
            tokenized.append(dict(id=row['id'], split=split, input_ids=full, labels=labels))
    save(directory / 'synthetic/token_counts.json', stats)
    real_stats = []
    for source in ([] if synthetic_only else read(directory / 'sources.json')):
        if source['role'] == 'layout-only':
            continue
        base = directory / f"real/{source['id']}"
        if (base / 'review.json').exists() and read(base / 'review.json')['status'] == 'excluded-unresolved':
            real_stats.append(dict(id=source['id'], organization=source['organization'], role=source['role'], status='excluded-unresolved'))
            continue
        msgs = messages((base / 'input.txt').read_text())
        prefix = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(msgs + [{'role': 'assistant', 'content': encoded(read(base / 'target.json')).decode().strip()}], tokenize=True, add_generation_prompt=False)
        prefix = prefix['input_ids'] if hasattr(prefix, 'keys') else prefix
        full = full['input_ids'] if hasattr(full, 'keys') else full
        real_stats.append(dict(id=source['id'], organization=source['organization'], role=source['role'], input_tokens=len(prefix), target_tokens=len(full)-len(prefix), total_tokens=len(full), fits_8k=len(full)<=8192))
    if not synthetic_only:
        save(directory / 'real_token_counts.json', real_stats)
    save(directory / 'synthetic/tokenized.jsonl', ''.join(json.dumps(r) + '\n' for r in tokenized).encode())
    for name, expected in read(directory / 'preserved.json').items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise Halt(f'Historical artifact changed: {name}')
    validation_name = 'synthetic/validation.json' if synthetic_only else 'validation.json'
    ledger = read(directory / 'ledger.json')
    if abs(sum(x['reserved_usd'] for x in ledger['attempts']) - ledger['reserved_usd']) > 1e-9:
        raise Halt('Ledger cost arithmetic mismatch')
    limits = read(directory / 'protocol.json')['limits']
    if ledger['reserved_usd'] > limits['total_usd']:
        raise Halt('Ledger exceeds cost ceiling')
    for stage in ('parse', 'gemini'):
        if sum(x['stage'] == stage for x in ledger['attempts']) > limits[stage]:
            raise Halt('Ledger exceeds request ceiling')
    for entry in ledger['attempts']:
        if not synthetic_only and entry['status'] != 'success':
            raise Halt('Unresolved request in full packet')
        if entry['status'] == 'success':
            base = directory / (f"parse/{entry['id']}" if entry['stage'] == 'parse' else f"real/{entry['id']}/teacher")
            if digest((base / 'response.json').read_bytes()) != entry['response_sha256']:
                raise Halt('Provider response changed')
    save(directory / validation_name, dict(status='passed', scope='synthetic-only' if synthetic_only else 'complete-packet', examples=len(stats), max_tokens=max(r['total_tokens'] for r in stats),
         preserved_artifacts=len(read(directory / 'preserved.json')), review_status='assistant-reviewed; awaiting user acceptance'))
    print(json.dumps(read(directory / validation_name)))


def package(directory):
    if read(directory / 'validation.json')['status'] != 'passed':
        raise Halt('Validation required')
    validate(directory)
    files = {str(p.relative_to(directory)): digest(p.read_bytes()) for p in directory.rglob('*')
             if p.is_file() and p.name not in ('SHA256SUMS.json', '.lock')}
    save(directory / 'SHA256SUMS.json', files, replace=True)
    print(f'Packaged {len(files)} artifacts; awaiting user acceptance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'parse', 'annotate', 'review', 'synthetic', 'validate', 'package'])
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    parser.add_argument('--synthetic-only', action='store_true', help='Validate only smoke data; cannot mark the full packet complete.')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.directory)
        return
    with (args.directory / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'parse':
            asyncio.run(parse(args.directory))
        elif args.stage == 'validate':
            validate(args.directory, synthetic_only=args.synthetic_only)
        else:
            {'annotate': annotate, 'review': review_stage, 'synthetic': synthetic,
             'validate': validate, 'package': package}[args.stage](args.directory)


if __name__ == '__main__':
    main()
