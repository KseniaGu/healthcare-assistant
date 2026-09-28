import copy

import pytest

from research.extraction.prepare_correction_variant import apply_edits, check_pair, check_specs


def target():
    return {'date': None, 'errors': [], 'results': [
        {'name': 'Тест', 'value': '1.00', 'unit': 'r/π', 'reference_range': '0–2'}]}


def test_reviewed_unit_repair_and_evidence():
    edit = dict(row=1, field='unit', before='r/π', after='г/л', rule=3,
                reason='Recognizable unit glyphs', spans=[dict(start=0, end=3, text='r/π')])
    assert apply_edits(target(), 'r/π', [edit])['results'][0]['unit'] == 'г/л'
    with pytest.raises(ValueError, match='Evidence'):
        apply_edits(target(), 'г/л', [edit])
    with pytest.raises(ValueError, match='before-value'):
        apply_edits(target(), 'r/π', [dict(edit, before='bad')])


@pytest.mark.parametrize('field,after', [('value', '1.0'), ('value', '10.00'), ('reference_range', '0–3'), ('name', 'Тест 2')])
def test_numeric_information_is_not_repaired(field, after):
    updated = copy.deepcopy(target())
    updated['results'][0][field] = after
    with pytest.raises(ValueError):
        check_pair(target(), updated)


def test_date_and_missing_value_cannot_be_inferred():
    updated = copy.deepcopy(target())
    updated['date'] = '2030-01-01'
    with pytest.raises(ValueError, match='Date'):
        check_pair(target(), updated)
    original = target()
    original['results'][0]['value'] = None
    with pytest.raises(ValueError):
        check_pair(original, target())


def test_qualitative_repair_and_optional_abstention():
    original = target()
    original['results'][0]['value'] = 'Отрищательный'
    corrected = copy.deepcopy(original)
    corrected['results'][0].update(value='Отрицательный', unit=None, reference_range=None)
    check_pair(original, corrected)


def test_reserved_provenance_rejected_before_materialization():
    spec = dict(id='x', family='f', source_group='f', split='train', input='text', development_sources=['B1'])
    with pytest.raises(ValueError, match='Reserved'):
        check_specs([spec], {'B1': {'role': 'reserved'}})


def test_family_leakage_rejected():
    import hashlib
    first = dict(id='a', family='f', source_group='f', split='train', input='one',
                 development_sources=[], input_sha256=hashlib.sha256(b'one').hexdigest(),
                 review_status='assistant-reviewed; awaiting user acceptance', review_notes=['reviewed'],
                 literal=target(), corrected=target())
    second = dict(first, id='b', input='two', split='validation')
    with pytest.raises(ValueError, match='Split leakage'):
        check_specs([first, second], {})


def test_qualitative_negation_cannot_be_removed():
    original = target()
    original['results'][0]['value'] = 'Отрищательный'
    corrected = copy.deepcopy(original)
    corrected['results'][0]['value'] = 'Положительный'
    with pytest.raises(ValueError, match='polarity'):
        check_pair(original, corrected)


def test_missing_unit_and_visible_exponent_cannot_be_invented():
    original = target()
    corrected = copy.deepcopy(original)
    original['results'][0]['unit'] = None
    with pytest.raises(ValueError, match='Missing unit'):
        check_pair(original, corrected)
    original['results'][0]['unit'] = '10^9/π'
    corrected['results'][0]['unit'] = '10^12/л'
    with pytest.raises(ValueError, match='exponent'):
        check_pair(original, corrected)


def test_warning_translation_preserves_details_and_observations():
    from research.extraction.prepare_correction_variant import translate_warnings
    original = target()
    original['errors'] = ['Пропущена 1 строка: 7.30.']
    edit = dict(index=0, before=original['errors'][0], after='Omitted 1 row: 7.30.', reason='English wording')
    translated = translate_warnings(original, [edit])
    assert translated['errors'] == ['Omitted 1 row: 7.30.']
    assert translated['results'] == original['results']
    assert original['errors'] == ['Пропущена 1 строка: 7.30.']
    with pytest.raises(ValueError, match='numbers'):
        translate_warnings(original, [dict(edit, after='Omitted 2 rows: 7.30.')])
    with pytest.raises(ValueError, match='mismatch'):
        translate_warnings(original, [dict(edit, before='Other')])
