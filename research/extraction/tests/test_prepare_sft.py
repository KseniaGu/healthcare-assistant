"""Offline integrity and training-boundary checks with fictitious inputs."""

import pytest

from research.extraction import prepare_sft as prep


def test_join_pages_orders_and_retains_empty_continuation():
    text, spans = prep.join_pages([dict(number=2, text=''), dict(number=1, text='test 1')])
    assert text == 'test 1\n'
    assert spans == [dict(number=1, start=0, end=6), dict(number=2, start=7, end=7)]
    with pytest.raises(prep.Halt):
        prep.join_pages([dict(number=2, text='gap')])



def test_join_pages_preserves_leading_empty_pages():
    text, spans = prep.join_pages([
        dict(number=3, text='result'), dict(number=1, text=''), dict(number=2, text='')])
    assert text == '\n\nresult'
    assert spans == [dict(number=1, start=0, end=0),
                     dict(number=2, start=1, end=1),
                     dict(number=3, start=2, end=8)]


def test_uncertain_submission_never_repeated(tmp_path):
    prep.save(tmp_path / 'protocol.json', dict(limits=dict(parse=1, gemini=1, total_usd=1)))
    prep.save(tmp_path / 'ledger.json', dict(attempts=[], reserved_usd=0))
    entry = prep.reserve(tmp_path, 'parse', 'page', .01, 'source')
    with pytest.raises(prep.Halt, match='resubmitted'):
        prep.reserve(tmp_path, 'parse', 'page', .01, 'source')
    prep.update(tmp_path, entry, job_id='job', status='submitted')
    assert prep.reserve(tmp_path, 'parse', 'page', .01, 'source')['job_id'] == 'job'
    with pytest.raises(prep.Halt, match='input changed'):
        prep.reserve(tmp_path, 'parse', 'page', .01, 'different')
    with pytest.raises(prep.Halt, match='ceiling'):
        prep.reserve(tmp_path, 'parse', 'other', .01, 'source')


def test_input_hash_change_is_rejected(tmp_path):
    prep.save(tmp_path / 'input.txt', b'old')
    prep.save(tmp_path / 'initial_hashes.json', {'input.txt': prep.digest(b'old')})
    (tmp_path / 'input.txt').write_text('new')
    with pytest.raises(prep.Halt, match='changed'):
        prep.check_integrity(tmp_path)


class Tokenizer:
    eos_token_id = 9

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1, 2, 3] if add_generation_prompt else [1, 2, 3, 4, 9]


def test_only_assistant_and_end_tokens_are_supervised():
    full, labels, prefix = prep.completion_tokens(Tokenizer(), [{}, {}, {}])
    assert prefix == 3
    assert labels == [-100, -100, -100, 4, 9]
    assert len(full) == len(labels)


def test_no_truncation():
    class LongTokenizer(Tokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return [1] if add_generation_prompt else [1] * 8192 + [9]
    with pytest.raises(prep.Halt, match='8192'):
        prep.completion_tokens(LongTokenizer(), [{}, {}])


def test_raw_target_cannot_claim_image_only_evidence(tmp_path):
    target = dict(date=None, results=[dict(name='Test', value='2', unit=None, reference_range=None)], errors=[])
    review = dict(status='assistant-reviewed', input_sha256=prep.digest(b'Test 2'),
                  target_sha256=prep.digest(prep.encoded(target)), date_evidence='absent', coverage='one row', corrections=[],
                  rows=[{k: dict(reason='visible or absent', spans=[]) for k in target['results'][0]}])
    schema = {'type': 'object'}
    with pytest.raises(prep.Halt, match='without source'):
        prep.check_review('Test 2', target, review, schema)
    for field in ('name', 'value'):
        value = target['results'][0][field]
        start = 'Test 2'.index(value)
        review['rows'][0][field]['spans'] = [dict(start=start, end=start+len(value), text=value)]
    prep.check_review('Test 2', target, review, schema)
    review['rows'][0]['value']['spans'][0]['text'] = '3'
    with pytest.raises(prep.Halt, match='span'):
        prep.check_review('Test 2', target, review, schema)


def test_reserved_provenance_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, 'check_integrity', lambda directory: None)
    prep.save(tmp_path / 'sources.json', [dict(id='B1', organization='KDL', role='reserved')])
    prep.save(tmp_path / 'schema.json', {})
    prep.save(tmp_path / 'synthetic/specs.json', [dict(id='one', source_group='one', split='train', family='one', development_sources=['B1'])])
    with pytest.raises(prep.Halt, match='Reserved'):
        prep.synthetic(tmp_path)


def test_prompt_contains_exact_report_text():
    msgs = prep.messages('Fictitious input 2.00')
    assert all(isinstance(m['content'], str) for m in msgs)
    assert '<report>\nFictitious input 2.00\n</report>' in msgs[1]['content']


def test_tokenizer_mapping_return_type():
    class MappingTokenizer(Tokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return {'input_ids': super().apply_chat_template(messages, tokenize, add_generation_prompt)}
    assert prep.completion_tokens(MappingTokenizer(), [{}, {}, {}])[1] == [-100, -100, -100, 4, 9]


def test_cross_partition_family_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, 'check_integrity', lambda directory: None)
    monkeypatch.setattr(prep, 'check_review', lambda *args: None)
    prep.save(tmp_path / 'sources.json', [])
    prep.save(tmp_path / 'schema.json', {})
    specs = [dict(id=str(n), source_group=str(n), family='same', split=split, development_sources=[],
                  input=str(n), target={}, review={}) for n, split in enumerate(['train','validation'])]
    prep.save(tmp_path / 'synthetic/specs.json', specs)
    with pytest.raises(prep.Halt, match='family split'):
        prep.synthetic(tmp_path)


def test_duplicate_input_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, 'check_integrity', lambda directory: None)
    monkeypatch.setattr(prep, 'check_review', lambda *args: None)
    prep.save(tmp_path / 'sources.json', [])
    prep.save(tmp_path / 'schema.json', {})
    prep.save(tmp_path / 'synthetic/specs.json', [dict(id=str(n), source_group=str(n), family=str(n), split='train',
              development_sources=[], input='Same input', target={}, review={}) for n in range(2)])
    with pytest.raises(prep.Halt, match='Duplicate synthetic input'):
        prep.synthetic(tmp_path)


def test_multipage_parse_resume_without_new_uploads(tmp_path, monkeypatch):
    import asyncio

    import httpx

    from research.extraction.tests.test_image_extraction import parser_http

    monkeypatch.setattr(prep, 'access', lambda directory: None)
    prep.save(tmp_path / 'protocol.json', dict(limits=dict(parse=2, gemini=1, total_usd=1)))
    prep.save(tmp_path / 'ledger.json', dict(attempts=[], reserved_usd=0))
    for n in (1,2):
        prep.save(tmp_path / f'images/p{n}.png', b'fictitious image')
    prep.save(tmp_path / 'sources.json', [dict(id='X', status='pending', pages=[
        dict(id=f'p{n}', number=n, path=f'images/p{n}.png') for n in (1,2)])])
    uploads=[]

    def handle(request):
        if request.method=='POST':
            uploads.append(request)
            return httpx.Response(200, json={'id':f'job{len(uploads)}'})
        if request.url.path.endswith('/result/text'):
            return httpx.Response(200, json={'text':'Test 2.00' if 'job1' in request.url.path else ''})
        return httpx.Response(200, json={'status':'SUCCESS'})

    parser_http(monkeypatch, handle)
    asyncio.run(prep.parse(tmp_path))
    assert len(uploads)==2
    assert (tmp_path/'real/X/input.txt').read_text()=='Test 2.00\n'
    asyncio.run(prep.parse(tmp_path))
    assert len(uploads)==2
    assert all(x['status']=='success' for x in prep.read(tmp_path/'ledger.json')['attempts'])
