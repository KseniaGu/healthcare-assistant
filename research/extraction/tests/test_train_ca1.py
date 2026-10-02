"""Offline checks for the accepted CA1 training loader and collator."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from research.extraction import train_ca1


def accepted_packet():
    if not train_ca1.PACKET.exists():
        pytest.skip("Private accepted packet is absent")
    return train_ca1.PACKET


def copy_packet(tmp_path):
    destination = tmp_path / "packet"
    shutil.copytree(accepted_packet(), destination)
    return destination


def reseal(packet, filename):
    path = packet / filename
    checksums = train_ca1.read_json(packet / "SHA256SUMS.json")
    checksums[filename] = train_ca1.sha(path)
    (packet / "SHA256SUMS.json").write_text(json.dumps(checksums, indent=2) + "\n")


def test_training_entrypoint_works_with_research_only_bundle(tmp_path):
    source = train_ca1.ROOT / "research/extraction/train_ca1.py"
    destination = tmp_path / "research/extraction/train_ca1.py"
    destination.parent.mkdir(parents=True)
    shutil.copy2(source, destination)
    result = subprocess.run(
        [sys.executable, "-m", "research.extraction.train_ca1", "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_train_command_accepts_wandb_reporting(tmp_path, monkeypatch):
    received = []
    monkeypatch.setattr(sys, "argv", ["train_ca1", "train", "--output", str(tmp_path / "run"), "--report-to", "wandb"])
    monkeypatch.setattr(train_ca1, "load_packet", lambda *_: (None, None, None))
    monkeypatch.setattr(train_ca1, "train", lambda args, *_: received.append(args.report_to))
    train_ca1.main()
    assert received == ["wandb"]


def test_accepted_packet_loads_with_disjoint_splits():
    splits, tokenizer, manifest = train_ca1.load_packet(accepted_packet())
    assert {name: len(rows) for name, rows in splits.items()} == {"train": 320, "validation": 64}
    assert set(row["id"] for row in splits["train"]).isdisjoint(row["id"] for row in splits["validation"])
    assert max(len(row["input_ids"]) for rows in splits.values() for row in rows) == 4742
    assert tokenizer.pad_token_id != tokenizer.eos_token_id
    assert manifest["status"] == "user-accepted"


def test_labels_cover_assistant_through_eos_only():
    row = {"id": "fixture", "input_ids": [11, 12, 21, 22, 99, 198], "labels": [-100, -100, 21, 22, 99, -100]}
    assert train_ca1.validate_labels(row, 99, [198]) == 2
    for labels in (
        [11, -100, 21, 22, 99, -100],
        [-100, -100, 21, -100, 99, -100],
        [-100, -100, 21, 22, -100, -100],
        [-100, -100, 21, 22, 99, 198],
    ):
        with pytest.raises(ValueError):
            train_ca1.validate_labels(dict(row, labels=labels), 99, [198])
    with pytest.raises(ValueError):
        train_ca1.validate_labels(dict(row, input_ids=[11] * 8193, labels=[-100] * 8193), 99, [198])


def test_dynamic_padding_masks_labels_and_attention():
    rows = [
        dict(input_ids=[11, 21, 99, 198], labels=[-100, 21, 99, -100]),
        dict(input_ids=[11, 99, 198], labels=[-100, 99, -100]),
    ]
    batch = train_ca1.collate_batch(rows, pad_id=7)
    assert batch["input_ids"].tolist() == [[11, 21, 99, 198], [11, 99, 198, 7]]
    assert batch["labels"].tolist() == [[-100, 21, 99, -100], [-100, 99, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]


def test_loader_rejects_changed_checksum(tmp_path):
    packet = copy_packet(tmp_path)
    with (packet / "train.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        train_ca1.load_packet(packet)


def test_loader_rejects_wrong_count_after_reseal(tmp_path, monkeypatch):
    packet = copy_packet(tmp_path)
    path = packet / "train.jsonl"
    path.write_text("".join(path.read_text().splitlines(keepends=True)[:-1]))
    reseal(packet, "train.jsonl")
    monkeypatch.setattr(train_ca1, "PACKET_SHA256", train_ca1.sha(packet / "SHA256SUMS.json"))
    with pytest.raises(ValueError, match="Wrong train count"):
        train_ca1.load_packet(packet)


def test_loader_rejects_split_leak_after_reseal(tmp_path, monkeypatch):
    packet = copy_packet(tmp_path)
    path = packet / "tokenized.jsonl"
    rows = train_ca1.read_jsonl(path)
    rows[0]["split"] = "validation"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    reseal(packet, "tokenized.jsonl")
    monkeypatch.setattr(train_ca1, "PACKET_SHA256", train_ca1.sha(packet / "SHA256SUMS.json"))
    with pytest.raises(ValueError, match="Split mismatch"):
        train_ca1.load_packet(packet)


def test_loader_rejects_family_leak_after_reseal(tmp_path, monkeypatch):
    packet = copy_packet(tmp_path)
    path = packet / "manifest.json"
    manifest = train_ca1.read_json(path)
    training_family = next(item["family"] for item in manifest["examples"] if item["split"] == "train")
    next(item for item in manifest["examples"] if item["split"] == "validation")["family"] = training_family
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    reseal(packet, "manifest.json")
    monkeypatch.setattr(train_ca1, "PACKET_SHA256", train_ca1.sha(packet / "SHA256SUMS.json"))
    with pytest.raises(ValueError, match="family isolation"):
        train_ca1.load_packet(packet)


def test_evaluation_default_uses_only_held_out_synthetic_reports():
    from research.extraction.evaluate_ca1_adapter import cases

    packet = accepted_packet()
    records = cases(packet, train_ca1.CORRECTION, include_development=False)
    expected = {row["id"] for row in train_ca1.read_jsonl(packet / "validation.jsonl")}
    assert len(records) == 64
    assert {identifier for identifier, _, _, _ in records} == expected
    assert {group for _, group, _, _ in records} == {"synthetic-validation"}


def test_development_evaluation_uses_approved_ids_without_generator_import(monkeypatch):
    from research.extraction.evaluate_ca1_adapter import cases

    monkeypatch.setitem(sys.modules, "research.extraction.generate_sft_pilot", None)
    records = cases(accepted_packet(), train_ca1.CORRECTION, include_development=True)
    development_ids = {identifier for identifier, group, _, _ in records if group == "real-development"}
    assert development_ids == {
        "A1",
        "A2",
        "A3",
        "A4",
        "C1",
        "D0002",
        "D0006",
        "D0007",
        "D0011",
        "D0012",
        "D0014",
        "D0015",
        "D0016",
    }
    assert len(records) == 77


def test_development_only_evaluation_selects_exact_eligible_reports():
    from research.extraction.evaluate_ca1_adapter import DEVELOPMENT_IDS, cases

    records = cases(accepted_packet(), train_ca1.CORRECTION, include_development=False, development_only=True)
    assert len(records) == 13
    assert {identifier for identifier, _, _, _ in records} == DEVELOPMENT_IDS
    assert {group for _, group, _, _ in records} == {"real-development"}


def test_reserved_only_evaluation_selects_all_and_separates_long_report():
    from research.extraction.evaluate_ca1_adapter import cases

    records = cases(accepted_packet(), train_ca1.CORRECTION, include_development=False, reserved_only=True)
    groups = {identifier: group for identifier, group, _, _ in records}
    assert len(records) == 11
    assert {identifier for identifier, group in groups.items() if group == "real-reserved"} == {
        "B1",
        "B2",
        "B3",
        "B4",
        "D0003",
        "D0004",
        "D0005",
        "D0008",
        "D0009",
        "D0017",
    }
    assert {identifier for identifier, group in groups.items() if group == "real-reserved-long"} == {"D0018"}


def test_evaluator_accepts_long_context_only_with_explicit_budget():
    import torch
    from jsonschema import Draft202012Validator

    from research.extraction.evaluate_ca1_adapter import one_response

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

    tokenizer = Tokenizer()
    validator = Draft202012Validator(train_ca1.read_json(train_ca1.CORRECTION / "schema.json"))
    with pytest.raises(ValueError, match="8192"):
        one_response(Model(), tokenizer, [], validator)
    result = one_response(Model(), tokenizer, [], validator, token_limit=16384)
    assert result["eos_reached"]
    assert result["input_tokens"] == 9000
    assert result["prediction"] == {"date": None, "results": [], "errors": []}


def test_matched_evaluator_requires_eos_and_schema():
    import torch
    from jsonschema import Draft202012Validator

    from research.extraction.evaluate_ca1_adapter import one_response

    class FakeTokenizer:
        eos_token_id = 99
        pad_token_id = 7

        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[11, 12]]), "attention_mask": torch.tensor([[1, 1]])}

        def decode(self, *args, **kwargs):
            return '{"date": null, "results": [], "errors": []}'

    class FakeModel:
        device = torch.device("cpu")

        def __init__(self, end):
            self.end = end

        def generate(self, **kwargs):
            return torch.tensor([[11, 12, 21, self.end]])

    validator = Draft202012Validator(train_ca1.read_json(train_ca1.CORRECTION / "schema.json"))
    valid = one_response(FakeModel(99), FakeTokenizer(), [], validator)
    assert valid["eos_reached"] and valid["prediction"] == {"date": None, "results": [], "errors": []}
    capped = one_response(FakeModel(22), FakeTokenizer(), [], validator)
    assert capped["prediction"] is None and "without EOS" in capped["schema_error"]


def test_loader_rejects_resealed_replacement(tmp_path):
    packet = copy_packet(tmp_path)
    path = packet / "README.md"
    path.write_text(path.read_text() + "\n")
    reseal(packet, "README.md")
    with pytest.raises(ValueError, match="user-accepted sealed"):
        train_ca1.load_packet(packet)
