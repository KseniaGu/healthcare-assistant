"""Offline integrity and acceptance checks using a wholly fictitious packet."""

import json

import pytest
from jsonschema import ValidationError

from research.extraction.datasets import packet


class FakeTokenizer:
    eos_token_id = 1000000
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        prefix = "".join(message["content"] for message in messages[:2]) + "<assistant>"
        ids = [ord(char) for char in prefix]
        if not add_generation_prompt:
            ids += [ord(char) for char in messages[2]["content"]] + [self.eos_token_id, 10]
        return ids


def fixture_packet(tmp_path, monkeypatch):
    """Builds a small sealed packet whose acceptance can be varied by each test."""
    monkeypatch.setattr(packet, "COUNTS", {"train": 1, "validation": 1})
    provenance = {"model": packet.MODEL, "revision": packet.REVISION, "files": {}, "chat_template_sha256": "0" * 64}
    monkeypatch.setattr(packet, "tokenizer_and_hashes", lambda path: (FakeTokenizer(), provenance))

    root = tmp_path / "packet"
    (root / "tokenizer").mkdir(parents=True)
    packet.dump(root / "tokenizer/provenance.json", provenance)
    prompt = "Fictitious extraction prompt"
    (root / "prompt.txt").write_text(prompt)
    schema = {
        "type": "object",
        "required": ["date", "results", "errors"],
        "properties": {"date": {"type": "null"}, "results": {"type": "array"}, "errors": {"type": "array"}},
    }
    packet.dump(root / "schema.json", schema)

    target = json.dumps({"date": None, "results": [], "errors": []})
    examples = []
    tokens = []
    for split in ("train", "validation"):
        identifier = f"{split}-fiction"
        text = f"Fictitious {split} report"
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": packet.WRAPPER_START + text + packet.WRAPPER_END},
            {"role": "assistant", "content": target},
        ]
        ids, labels, _, _ = packet.tokenize(FakeTokenizer(), messages)
        (root / f"{split}.jsonl").write_text(json.dumps({"id": identifier, "messages": messages}) + "\n")
        examples.append(
            {
                "id": identifier,
                "split": split,
                "family": split,
                "input_sha256": packet.sha(text),
                "target_sha256": packet.sha(target),
            }
        )
        tokens.append({"id": identifier, "split": split, "input_ids": ids, "labels": labels})

    (root / "tokenized.jsonl").write_text("".join(json.dumps(row) + "\n" for row in tokens))
    packet.dump(
        root / "manifest.json",
        {
            "model": packet.MODEL,
            "revision": packet.REVISION,
            "source_commit": "a" * 40,
            "prompt_sha256": packet.sha(root / "prompt.txt"),
            "schema_sha256": packet.sha(root / "schema.json"),
            "tokenizer_provenance": provenance,
            "examples": examples,
        },
    )
    content_sha256 = packet.write_checksums(root)

    evidence = tmp_path / "evidence.json"
    reference = "b" * 64
    packet.dump(
        evidence,
        {
            "basis": "model_facing_parity",
            "result": "pass",
            "content_sha256": content_sha256,
            "previous_packet_sha256": reference,
            "example_count": 2,
            "compared": ["all model-facing fields"],
        },
    )
    return root, evidence, reference


def test_candidate_integrity_is_independent_of_acceptance(tmp_path, monkeypatch):
    root, evidence, reference = fixture_packet(tmp_path, monkeypatch)
    assert packet.verify_integrity(root) == packet.sha(root / "SHA256SUMS.json")
    with pytest.raises(ValueError, match="not accepted"):
        packet.load_packet(root)
    packet.accept(root, "model_facing_parity", evidence, reference)
    splits, _, _ = packet.load_packet(root)
    assert {name: len(rows) for name, rows in splits.items()} == {"train": 1, "validation": 1}


def test_integrity_rejects_changed_missing_and_extra_files(tmp_path, monkeypatch):
    root, _, _ = fixture_packet(tmp_path, monkeypatch)
    (root / "prompt.txt").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        packet.verify_integrity(root)
    (root / "prompt.txt").unlink()
    with pytest.raises(ValueError, match="checksum"):
        packet.verify_integrity(root)
    (root / "extra.txt").write_text("extra")
    with pytest.raises(ValueError, match="checksum"):
        packet.verify_integrity(root)


def test_acceptance_rejects_stale_and_unsupported_evidence(tmp_path, monkeypatch):
    root, evidence, reference = fixture_packet(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="reference"):
        packet.accept(root, "model_facing_parity", evidence, "c" * 64)
    data = packet.read_json(evidence)
    data["result"] = "fail"
    packet.dump(evidence, data)
    with pytest.raises(ValueError, match="evidence"):
        packet.accept(root, "model_facing_parity", evidence, reference)
    data["result"] = "pass"
    packet.dump(evidence, data)
    packet.accept(root, "model_facing_parity", evidence, reference)
    acceptance = packet.read_json(root / "acceptance.json")
    acceptance["basis"] = "unsupported"
    packet.dump(root / "acceptance.json", acceptance)
    with pytest.raises(ValueError, match="acceptance"):
        packet.load_packet(root)


def test_accepted_content_change_invalidates_record(tmp_path, monkeypatch):
    root, evidence, reference = fixture_packet(tmp_path, monkeypatch)
    packet.accept(root, "model_facing_parity", evidence, reference)
    manifest = packet.read_json(root / "manifest.json")
    manifest["source_hashes"] = {"changed": "c" * 64}
    packet.dump(root / "manifest.json", manifest)
    packet.write_checksums(root)
    with pytest.raises(ValueError, match="acceptance"):
        packet.load_packet(root)


def test_loader_rejects_resealed_wrong_counts_and_labels(tmp_path, monkeypatch):
    root, evidence, reference = fixture_packet(tmp_path, monkeypatch)
    (root / "validation.jsonl").write_text("")
    packet.write_checksums(root)
    data = packet.read_json(evidence)
    data["content_sha256"] = packet.sha(root / "SHA256SUMS.json")
    packet.dump(evidence, data)
    packet.accept(root, "model_facing_parity", evidence, reference)
    with pytest.raises(ValueError, match="Wrong validation count"):
        packet.load_packet(root)


def test_validate_target_rejects_fences_and_bad_schema():
    schema = {"type": "object", "required": ["date"]}
    assert packet.validate_target('{"date": null}', schema) == {"date": None}
    with pytest.raises(json.JSONDecodeError):
        packet.validate_target('```json\n{"date": null}\n```', schema)
    with pytest.raises(ValidationError):
        packet.validate_target('{"results": []}', schema)


def reseal_and_accept(root, evidence, reference):
    data = packet.read_json(evidence)
    data["content_sha256"] = packet.write_checksums(root)
    packet.dump(evidence, data)
    packet.accept(root, "model_facing_parity", evidence, reference)


@pytest.mark.parametrize(
    "change,problem",
    [
        ("duplicate", "Duplicate or unknown report ID"),
        ("family", "family isolation"),
        ("prompt", "prompt or schema"),
        ("schema", "prompt or schema"),
        ("labels", "boundary"),
    ],
)
def test_loader_rejects_invalid_resealed_content(tmp_path, monkeypatch, change, problem):
    root, evidence, reference = fixture_packet(tmp_path, monkeypatch)
    if change == "duplicate":
        path = root / "validation.jsonl"
        row = json.loads(path.read_text())
        row["id"] = "train-fiction"
        path.write_text(json.dumps(row) + "\n")
    elif change == "family":
        manifest = packet.read_json(root / "manifest.json")
        manifest["examples"][1]["family"] = "train"
        packet.dump(root / "manifest.json", manifest)
    elif change in ("prompt", "schema"):
        path = root / ("prompt.txt" if change == "prompt" else "schema.json")
        path.write_text(path.read_text() + " ")
    else:
        path = root / "tokenized.jsonl"
        rows = packet.read_jsonl(path)
        supervised = rows[0]["labels"].index(next(label for label in rows[0]["labels"] if label != -100))
        rows[0]["labels"][supervised] = -100
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    reseal_and_accept(root, evidence, reference)
    with pytest.raises(ValueError, match=problem):
        packet.load_packet(root)


def test_loader_rejects_tokenizer_drift(tmp_path, monkeypatch):
    root, evidence, reference = fixture_packet(tmp_path, monkeypatch)
    packet.accept(root, "model_facing_parity", evidence, reference)
    monkeypatch.setattr(packet, "tokenizer_and_hashes", lambda path: (FakeTokenizer(), {"changed": True}))
    with pytest.raises(ValueError, match="Tokenizer provenance"):
        packet.load_packet(root)


def test_reviewed_changes_requires_each_case_review(tmp_path, monkeypatch):
    root, evidence, _ = fixture_packet(tmp_path, monkeypatch)
    packet.dump(
        evidence,
        {
            "basis": "reviewed_changes",
            "result": "pass",
            "content_sha256": packet.sha(root / "SHA256SUMS.json"),
            "changed_examples": ["train-fiction", "validation-fiction"],
            "reviews": {"train-fiction": "reviewed"},
        },
    )
    with pytest.raises(ValueError, match="incomplete"):
        packet.accept(root, "reviewed_changes", evidence)


def test_integrity_rejects_symlinked_payload(tmp_path, monkeypatch):
    root, _, _ = fixture_packet(tmp_path, monkeypatch)
    (root / "linked.txt").symlink_to(tmp_path / "outside.txt")
    with pytest.raises(ValueError, match="symlink"):
        packet.verify_integrity(root)
