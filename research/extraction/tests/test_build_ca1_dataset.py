"""Builder source gates use only invented files and hashes."""

import pytest

from research.extraction.datasets import build_ca1_dataset as builder
from research.extraction.datasets import packet
from research.extraction.datasets.synthetic.reports import ELIGIBLE


def source_fixture(tmp_path, monkeypatch):
    correction = tmp_path / "correction"
    smoke = tmp_path / "smoke"
    tokenizer = smoke / "tokenizer"
    tokenizer.mkdir(parents=True)
    names = [
        "prompts/corrected.txt",
        "schema.json",
        "synthetic_specs.json",
        "synthetic/corrected/train.jsonl",
        "synthetic/corrected/validation.jsonl",
        "real_edits.json",
        "verification.json",
        "synthetic_review.md",
        "policy.md",
    ]
    names.extend(f"real/{identifier}/input.txt" for identifier in ELIGIBLE)
    for name in names:
        path = correction / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status": "passed"}' if name == "verification.json" else "Invented source")
    packet.dump(correction / "SHA256SUMS.json", {name: packet.sha(correction / name) for name in names})
    (smoke / "sources.json").write_text("[]")
    packet.dump(smoke / "SHA256SUMS.json", {"sources.json": packet.sha(smoke / "sources.json")})
    (tokenizer / "provenance.json").write_text("{}")
    monkeypatch.setattr(builder.packet, "tokenizer_and_hashes", lambda path: (None, {"files": {}}))
    return correction, smoke / "sources.json", tokenizer


def test_source_hashes_record_exact_consumed_inputs(tmp_path, monkeypatch):
    correction, source_index, tokenizer = source_fixture(tmp_path, monkeypatch)
    hashes = builder.source_hashes_for(correction, source_index, tokenizer)
    assert len([key for key in hashes if key.startswith("real/")]) == 13
    assert hashes["source_index/sources.json"] == packet.sha(source_index)
    assert hashes["verification.json"] == packet.sha(correction / "verification.json")


def test_changed_or_unreviewed_source_rejected(tmp_path, monkeypatch):
    correction, source_index, tokenizer = source_fixture(tmp_path, monkeypatch)
    (correction / "real/A1/input.txt").write_text("Changed invented source")
    with pytest.raises(ValueError, match="checksum"):
        builder.source_hashes_for(correction, source_index, tokenizer)
    checksums = packet.read_json(correction / "SHA256SUMS.json")
    checksums["real/A1/input.txt"] = packet.sha(correction / "real/A1/input.txt")
    packet.dump(correction / "SHA256SUMS.json", checksums)
    (correction / "verification.json").write_text('{"status": "failed"}')
    checksums["verification.json"] = packet.sha(correction / "verification.json")
    packet.dump(correction / "SHA256SUMS.json", checksums)
    with pytest.raises(ValueError, match="review"):
        builder.source_hashes_for(correction, source_index, tokenizer)


def test_build_does_not_accept_source_revision_override(tmp_path):
    with pytest.raises(TypeError, match="source_commit"):
        builder.build(
            tmp_path / "packet", tmp_path / "source", tmp_path / "index", tmp_path / "tokenizer", source_commit="fake"
        )


def test_source_revision_tracks_every_synthetic_module(monkeypatch):
    checked = []

    def output(command, **kwargs):
        return "a" * 40 if command[-1] == "HEAD" else ""

    monkeypatch.setattr(builder.subprocess, "check_output", output)
    monkeypatch.setattr(builder.subprocess, "check_call", lambda command, **kwargs: checked.append(command[-1]))

    assert builder.source_revision() == "a" * 40
    assert {
        "research/extraction/datasets/synthetic/__init__.py",
        "research/extraction/datasets/synthetic/reports.py",
        "research/extraction/datasets/synthetic/layouts.py",
        "research/extraction/datasets/synthetic/types.py",
    } <= set(checked)
