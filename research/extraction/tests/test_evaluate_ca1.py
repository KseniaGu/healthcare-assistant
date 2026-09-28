import json

import pytest

from research.extraction import evaluate_ca1 as runner


def test_free_generation_context_budget():
    request = runner.request_for([{'role': 'user', 'content': 'fictitious input'}], 2500)
    assert 'format' not in request
    assert request['options']['num_predict'] == 5692
    assert request['options']['num_ctx'] == 8192
    assert request['think'] is False
    with pytest.raises(ValueError):
        runner.request_for([], 8192)


@pytest.mark.parametrize('status', ['submitted', 'uncertain_failure'])
def test_uncertain_run_never_resubmits(tmp_path, monkeypatch, status):
    (tmp_path / 'manifest.json').write_text(json.dumps({'records': [{'status': status}]}))
    def no_request(*args, **kwargs):
        pytest.fail('No provider request allowed')
    monkeypatch.setattr(runner.local, 'preflight', no_request)
    with pytest.raises(ValueError, match='Uncertain'):
        runner.run(tmp_path)


def test_parameter_order_is_not_a_configuration_change():
    a = 'temperature 0.7\ntop_k 20\nstop "<|im_start|>"\nstop "<|im_end|>"'
    b = 'top_k 20\nstop "<|im_start|>"\nstop "<|im_end|>"\ntemperature 0.7'
    assert runner.parameter_values(a) == runner.parameter_values(b)
    assert runner.parameter_values(a) != runner.parameter_values(b.replace('0.7', '0.8'))
    assert runner.parameter_values(a) != runner.parameter_values(b.replace('<|im_end|>', 'other'))
