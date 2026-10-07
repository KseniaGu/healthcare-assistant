"""Fictitious offline checks for CA1 case selection, generation, and resume."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from research.extraction.datasets import packet
from research.extraction.evaluation import evaluate_ca1_adapter as evaluation
from research.extraction.evaluation.scoring import compare
from research.extraction.tests.test_ca1_packet import fixture_packet


def test_synthetic_selection_uses_validation_only(tmp_path, monkeypatch):
    root, _, _ = fixture_packet(tmp_path, monkeypatch)
    records = evaluation.cases(root, None, "synthetic")
    assert [(identifier, group) for identifier, group, _, _ in records] == [
        ("validation-fiction", "synthetic-validation")
    ]


def test_real_selection_requires_exact_role_only_membership(tmp_path, monkeypatch):
    root, _, _ = fixture_packet(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluation, "DEVELOPMENT_IDS", frozenset({"F1"}))
    monkeypatch.setattr(evaluation, "RESERVED_IDS", frozenset())
    real = tmp_path / "real-data"
    (real / "real/F1").mkdir(parents=True)
    (real / "real/F1/input.txt").write_text("Invented result: 4")
    packet.dump(real / "real/F1/target.json", {"date": None, "results": [], "errors": []})
    packet.dump(real / "real_edits.json", [{"id": "F1", "role": "development"}])

    records = evaluation.cases(root, real, "development")
    assert len(records) == 1 and records[0][0] == "F1"
    assert records[0][2][0]["content"] == (root / "prompt.txt").read_text()

    packet.dump(real / "real_edits.json", [{"id": "F1", "role": "development", "medical_details": "extra"}])
    with pytest.raises(ValueError, match="only ID and role"):
        evaluation.cases(root, real, "development")


def test_long_context_budget_and_eos():
    import torch

    class Tokenizer:
        eos_token_id = 99
        pad_token_id = 7

        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": torch.full((1, 9000), 11), "attention_mask": torch.ones((1, 9000))}

        def decode(self, *args, **kwargs):
            return '{"date": null, "results": [], "errors": []}'

    class Model:
        device = torch.device("cpu")

        def generate(self, input_ids, **kwargs):
            assert kwargs["max_new_tokens"] == 16384 - 9000
            return torch.cat((input_ids, torch.tensor([[99]])), dim=1)

    validator = Draft202012Validator({"type": "object"})

    with pytest.raises(ValueError, match="8192"):
        evaluation.one_response(Model(), Tokenizer(), [], validator)
    result = evaluation.one_response(Model(), Tokenizer(), [], validator, token_limit=16384)
    assert result["eos_reached"] and result["input_tokens"] == 9000


def test_saved_case_rejects_changed_identity_and_score(tmp_path):
    target = {"date": None, "results": [], "errors": []}
    path = tmp_path / "case.json"
    saved = {
        "id": "F1",
        "group": "synthetic-validation",
        "content": json.dumps(target),
        "prediction": target,
        "schema_error": None,
        "input_tokens": 12,
        "generated_tokens": 8,
        "eos_reached": True,
        "score": compare(target, target),
    }
    packet.dump(path, saved)
    validator = Draft202012Validator({"type": "object"})
    evaluation.validate_saved_case(path, "F1", "synthetic-validation", target, validator)

    saved["id"] = "F2"
    packet.dump(path, saved)
    with pytest.raises(ValueError, match="identity"):
        evaluation.validate_saved_case(path, "F1", "synthetic-validation", target, validator)

    saved["id"] = "F1"
    saved["score"]["strict_pass"] = False
    packet.dump(path, saved)
    with pytest.raises(ValueError, match="score"):
        evaluation.validate_saved_case(path, "F1", "synthetic-validation", target, validator)
    saved["score"]["strict_pass"] = True
    saved["content"] = '{"date": "changed"}'
    packet.dump(path, saved)
    with pytest.raises(ValueError, match="generated content"):
        evaluation.validate_saved_case(path, "F1", "synthetic-validation", target, validator)
    saved["content"] = json.dumps(target)
    saved["input_tokens"] = 8192
    packet.dump(path, saved)
    with pytest.raises(ValueError, match="token budget"):
        evaluation.validate_saved_case(path, "F1", "synthetic-validation", target, validator)


def test_saved_invalid_case_rechecks_content_and_eos(tmp_path):
    target = {"date": None, "results": [], "errors": []}
    validator = Draft202012Validator({"type": "object", "required": ["date", "results", "errors"]})
    path = tmp_path / "case.json"
    saved = {
        "id": "F1",
        "group": "synthetic-validation",
        "content": json.dumps(target),
        "prediction": None,
        "schema_error": "old failure message",
        "input_tokens": 12,
        "generated_tokens": 8,
        "eos_reached": True,
        "score": compare(target, None),
    }
    packet.dump(path, saved)
    with pytest.raises(ValueError, match="invalid generation"):
        evaluation.validate_saved_case(path, "F1", "synthetic-validation", target, validator)

    saved["content"] = '{"date": null}'
    packet.dump(path, saved)
    evaluation.validate_saved_case(path, "F1", "synthetic-validation", target, validator)
    saved["content"] = "{"
    packet.dump(path, saved)
    evaluation.validate_saved_case(path, "F1", "synthetic-validation", target, validator)
    saved["content"] = json.dumps(target)
    saved["eos_reached"] = False
    packet.dump(path, saved)
    evaluation.validate_saved_case(path, "F1", "synthetic-validation", target, validator)


def test_atomic_evaluation_write_preserves_previous_record(tmp_path, monkeypatch):
    path = tmp_path / "evaluation.json"
    original = '{"status": "running"}\n'
    path.write_text(original)

    def failed_replace(self, target):
        assert target == path
        assert path.read_text() == original
        raise OSError("replace interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", failed_replace)
        with pytest.raises(OSError, match="replace interrupted"):
            evaluation._write_json_atomic(path, {"status": "completed"})
    assert path.read_text() == original
    evaluation._write_json_atomic(path, {"status": "completed"})
    assert packet.read_json(path) == {"status": "completed"}


@pytest.mark.parametrize("dataset", ["synthetic", "development", "reserved"])
def test_completion_limitation_applies_only_to_reserved(tmp_path, monkeypatch, dataset):
    root, evidence, reference = fixture_packet(tmp_path, monkeypatch)
    packet.accept(root, "model_facing_parity", evidence, reference)
    adapter = tmp_path / "run/checkpoint-80"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"fictitious")
    packet.dump(
        adapter.parent / "run.json",
        {
            "status": "completed",
            "best_checkpoint": adapter.name,
            "recipe": {
                "model": packet.MODEL,
                "revision": packet.REVISION,
                "packet_sha256": packet.sha(root / "SHA256SUMS.json"),
            },
        },
    )
    monkeypatch.setattr(evaluation, "cases", lambda *args: [])
    monkeypatch.setattr(evaluation, "model_with_adapter", lambda *args: pytest.fail("Model loaded for empty cases"))
    args = SimpleNamespace(
        packet=root,
        adapter=adapter,
        real_data=None,
        output=tmp_path / "output",
        device="cuda",
        dataset=dataset,
        resume=False,
    )
    evaluation.evaluate(args, None)
    report = packet.read_json(args.output / "evaluation.json")
    assert report["status"] == "completed"
    assert ("limitation" in report) == (dataset == "reserved")
    if dataset != "reserved":
        report["limitation"] = "stale reserved warning"
        packet.dump(args.output / "evaluation.json", report)
        args.resume = True
        evaluation.evaluate(args, None)
        assert "limitation" not in packet.read_json(args.output / "evaluation.json")


def test_resume_validates_case_before_model_load(tmp_path, monkeypatch):
    root, evidence, reference = fixture_packet(tmp_path, monkeypatch)
    packet.accept(root, "model_facing_parity", evidence, reference)
    adapter = tmp_path / "run/checkpoint-80"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"fictitious")
    packet.dump(
        adapter.parent / "run.json",
        {
            "status": "completed",
            "best_checkpoint": adapter.name,
            "recipe": {
                "model": packet.MODEL,
                "revision": packet.REVISION,
                "packet_sha256": packet.sha(root / "SHA256SUMS.json"),
            },
        },
    )
    output = tmp_path / "output"
    (output / "base").mkdir(parents=True)
    target = {"date": None, "results": [], "errors": []}
    records = [("F1", "synthetic-validation", [], target)]
    monkeypatch.setattr(evaluation, "cases", lambda *args: records)
    args = SimpleNamespace(
        packet=root, adapter=adapter, real_data=None, output=output, device="cuda", dataset="synthetic", resume=True
    )
    packet.dump(
        output / "evaluation.json",
        {"settings": evaluation.evaluation_settings(args, records, adapter.parent / "run.json")},
    )
    (output / "base/F1.json").write_text("{}")
    monkeypatch.setattr(
        evaluation, "model_with_adapter", lambda *args: pytest.fail("Model loaded before saved-case validation")
    )
    with pytest.raises(ValueError, match="Incomplete saved case"):
        evaluation.evaluate(args, None)
