"""Bounded local Qwen CA1 development baseline; no reserved reports or teacher calls."""

import argparse
import fcntl
import json
import shlex
import time
from pathlib import Path

from research.extraction import evaluate_local_extraction as local
from research.extraction.prepare_sft import completion_tokens
from research.extraction.scoring import compare, summarize

ROOT = local.ROOT
DATA = ROOT / 'data/research/extraction/sft-correction-v1'
OUTPUT = ROOT / 'data/research/extraction/qwen-ca1-untuned-v1'
RUNTIME_VERSION = '0.34.3'
read, write, sha = local.read_json, local.write_json, local.sha


def request_for(messages, prefix):
    if not 0 < prefix < 8192:
        raise ValueError('Invalid input context length')
    return dict(model=local.MODEL, messages=messages, stream=False, think=False, keep_alive='10m',
                options=dict(local.OPTIONS, num_ctx=8192, num_predict=8192-prefix))


def prepare(output):
    from transformers import AutoTokenizer

    if output.exists():
        raise FileExistsError(output)
    for path, expected in read(DATA / 'SHA256SUMS.json').items():
        if sha((DATA / path).read_bytes()) != expected:
            raise ValueError(f'Approved packet changed: {path}')
    tokenizer_path = DATA.parent / 'sft-smoke-v1/tokenizer'
    provenance = read(DATA / 'tokenizer_provenance.json')
    for name, expected in provenance['files'].items():
        if sha((tokenizer_path / name).read_bytes()) != expected:
            raise ValueError('Tokenizer changed')
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    if sha(tokenizer.chat_template.encode()) != provenance['chat_template_sha256']:
        raise ValueError('Tokenizer template changed')
    system = (DATA / 'prompts/corrected.txt').read_text()
    inputs = []
    for record in read(DATA / 'real_edits.json'):
        if record['role'] != 'development':
            continue
        identifier = record['id']
        text = (DATA / f'real/{identifier}/input.txt').read_text()
        target = read(DATA / f'real/{identifier}/target.json')
        msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content':
                'Extract laboratory observations from this parser output:\n<report>\n' + text + '\n</report>\n'}]
        inputs.append((identifier, 'real-development', msgs, target))
    for line in (DATA / 'synthetic/corrected/validation.jsonl').read_text().splitlines():
        row = json.loads(line)
        inputs.append((row['id'], 'synthetic-validation', row['messages'][:-1], json.loads(row['messages'][-1]['content'])))
    if len(inputs) != 21 or sum(group == 'real-development' for _, group, _, _ in inputs) != 13:
        raise ValueError('Unexpected authorized development scope')
    output.mkdir(parents=True)
    write(output / 'schema.json', read(DATA / 'schema.json'))
    records = []
    for identifier, group, messages, target in inputs:
        full = messages + [{'role': 'assistant', 'content': json.dumps(target, ensure_ascii=False, indent=2)}]
        _, _, prefix = completion_tokens(tokenizer, full)
        request = request_for(messages, prefix)
        path = output / identifier
        path.mkdir()
        write(path / 'request.json', request)
        write(path / 'target.json', target)
        records.append(dict(id=identifier, group=group, status='not_attempted', input_tokens=prefix,
                            request_sha256=sha((path / 'request.json').read_bytes()),
                            target_sha256=sha((path / 'target.json').read_bytes())))
    write(output / 'manifest.json', dict(created_at=local.now(), model=local.MODEL, digest=local.DIGEST,
          ollama_version=RUNTIME_VERSION, endpoint=local.ENDPOINT, prompt_sha256=sha(system.encode()),
          approved_policy_sha256=sha((DATA / 'policy.md').read_bytes()),
          approval='User approved policy/prompt and requested untuned Qwen re-evaluation. Scope: development only; reserved reports remain closed.',
          source_packet_checksums_sha256=sha((DATA / 'SHA256SUMS.json').read_bytes()),
          max_submissions=21, decoding='unconstrained; no response schema supplied, local schema validation only',
          context_tokens=8192, tokenizer=provenance, records=records))
    print('Prepared 13 development reports and 8 synthetic validation cases.', flush=True)


def parameter_values(text):
    result = {}
    for line in text.splitlines():
        parts = shlex.split(line)
        if parts:
            result.setdefault(parts[0], []).append(parts[1:])
    return result


def run(output):
    with (output / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read(output / 'manifest.json')
        if any(r['status'] in ('submitted', 'uncertain_failure') for r in manifest['records']):
            raise ValueError('Uncertain prior submission; no automatic retry')
        # This run explicitly pins the inspected newer runtime; historical runner files stay unchanged.
        local.VERSION = RUNTIME_VERSION
        runtime_path = output / 'runtime.json'
        if runtime_path.exists():
            baseline = read(runtime_path)
            details = local.metadata('/api/show', {'model': local.MODEL})
            if parameter_values(details['parameters']) != parameter_values(baseline['show']['parameters']):
                raise ValueError('Effective model parameters changed')
            # A host restart can reorder independent parameter lines without changing their values.
            local.PARAMETERS_HASH = sha(details['parameters'].encode())
            runtime_path = output / f'runtime-resume-{local.now().replace(":", "-")}.json'
        runtime = local.preflight()
        write(runtime_path, runtime)
        started = time.monotonic()
        for record in manifest['records']:
            if record['status'] != 'not_attempted':
                continue
            if time.monotonic() - started > 2700:
                break
            path = output / record['id']
            for name in ('request', 'target'):
                if sha((path / f'{name}.json').read_bytes()) != record[f'{name}_sha256']:
                    raise ValueError('Frozen request/target changed')
            payload = read(path / 'request.json')
            record.update(status='submitted', started_at=local.now())
            write(output / 'manifest.json', manifest)
            print('Submitted ' + record['id'], flush=True)
            try:
                response_bytes = local.request_bytes('/api/chat', payload, timeout=600)
                (path / 'response.json').write_bytes(response_bytes)
                response = json.loads(response_bytes)
                content = response.get('message', {}).get('content', '')
                (path / 'content.txt').write_text(content)
                record['response_metadata'] = {k: v for k, v in response.items() if k != 'message'}
                if response.get('model') != local.MODEL or response.get('error'):
                    raise RuntimeError('Unexpected model or provider error')
                if response.get('message', {}).get('thinking') or response.get('message', {}).get('tool_calls'):
                    raise RuntimeError('Unexpected thinking/tools')
                if not response.get('done') or response.get('done_reason') != 'stop':
                    record.update(status='incomplete', error='Generation did not stop normally')
                elif response.get('prompt_eval_count', 0) + response.get('eval_count', 0) > 8192:
                    record.update(status='context_overflow', error='Measured prompt plus output exceeds 8192')
                else:
                    try:
                        prediction = local.parse_and_validate_extraction(content, read(output / 'schema.json'))
                        write(path / 'prediction.json', prediction)
                        record['status'] = 'success'
                    except (ValueError, TypeError) as error:
                        record.update(status='invalid_output', error=str(error))
                record['response_sha256'] = sha(response_bytes)
            except Exception as error:
                record.update(status='uncertain_failure', error=f'{type(error).__name__}: {error}')
                record['finished_at'] = local.now()
                write(output / 'manifest.json', manifest)
                print(record['id'] + ': uncertain failure; stopped without retry', flush=True)
                return
            record['finished_at'] = local.now()
            write(output / 'manifest.json', manifest)
            print(record['id'] + ': ' + record['status'], flush=True)
        manifest['finished_at'] = local.now()
        write(output / 'manifest.json', manifest)


def score(output):
    manifest = read(output / 'manifest.json')
    scores = []
    for record in manifest['records']:
        path = output / record['id']
        prediction = read(path / 'prediction.json') if record['status'] == 'success' else None
        target = read(path / 'target.json')
        result = compare(target, prediction)
        result.update(id=record['id'], group=record['group'], status=record['status'])
        if prediction and prediction['errors'] == target['errors']:
            result['warning_status'] = 'pass'
            result['strict_pass'] = result['date_exact'] and result['ordered_results_exact']
        result['numeric_value_matches_by_position'] = sum(
            a['value'] == b['value'] for a, b in zip(target['results'], prediction['results'] if prediction else [], strict=False)
            if a['value'] is not None and any(c.isdigit() for c in a['value']))
        result['numeric_value_denominator'] = sum(r['value'] is not None and any(c.isdigit() for c in r['value']) for r in target['results'])
        scores.append(result)
    write(output / 'scores.json', dict(records=scores, by_group={
        group: summarize([s for s in scores if s['group'] == group]) for group in ('real-development', 'synthetic-validation')},
        note='Exact positional and record-multiset metrics; omissions/repairs need source-aligned manual review. Nonidentical warnings remain pending semantic review. Failed/unattempted cases stay in denominators.'))
    print(json.dumps(read(output / 'scores.json')['by_group']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run', 'score'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'score': score}[args.stage](args.output)


if __name__ == '__main__':
    main()
