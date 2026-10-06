"""Offline trainer checks without private packets, CUDA, or model weights."""

import sys
from types import SimpleNamespace

import pytest
import torch

from research.extraction.datasets import packet
from research.extraction.tests.test_ca1_packet import fixture_packet
from research.extraction.training import train_ca1


def test_dynamic_padding_masks_labels_and_attention():
    rows = [
        {"input_ids": [11, 21, 99, 198], "labels": [-100, 21, 99, -100]},
        {"input_ids": [11, 99, 198], "labels": [-100, 99, -100]},
    ]
    batch = train_ca1.collate_batch(rows, pad_id=7)
    assert batch["input_ids"].tolist() == [[11, 21, 99, 198], [11, 99, 198, 7]]
    assert batch["labels"].tolist() == [[-100, 21, 99, -100], [-100, 99, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]


@pytest.mark.parametrize("stage", ["check", "preflight", "train"])
def test_every_stage_rejects_candidate_before_model_or_output(tmp_path, monkeypatch, stage):
    root, _, _ = fixture_packet(tmp_path, monkeypatch)
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train_ca1", stage, "--packet", str(root), "--output", str(output)])
    monkeypatch.setattr(train_ca1, "load_model", lambda *args: pytest.fail("Model loaded for candidate"))
    with pytest.raises(ValueError, match="not accepted"):
        train_ca1.main()
    assert not output.exists()


def test_check_accepts_self_contained_packet(tmp_path, monkeypatch, capsys):
    root, evidence, reference = fixture_packet(tmp_path, monkeypatch)
    packet.accept(root, "model_facing_parity", evidence, reference)
    monkeypatch.setattr(sys, "argv", ["train_ca1", "check", "--packet", str(root)])
    train_ca1.main()
    assert "1 train, 1 validation" in capsys.readouterr().out


def test_train_seeds_before_creating_adapter(tmp_path, monkeypatch):
    class ModelReached(Exception):
        pass

    packet_dir = tmp_path / "packet"
    packet_dir.mkdir()
    (packet_dir / "SHA256SUMS.json").write_text("{}\n")
    packet.dump(packet_dir / "manifest.json", {"source_commit": "a" * 40})
    monkeypatch.setattr(train_ca1, "current_source_commit", lambda: "a" * 40)
    monkeypatch.setattr(train_ca1, "cuda_dtype", lambda: torch.bfloat16)
    version = train_ca1.importlib.metadata.version

    def installed_version(name):
        try:
            return version(name)
        except train_ca1.importlib.metadata.PackageNotFoundError:
            return "0.0.0"

    monkeypatch.setattr(train_ca1.importlib.metadata, "version", installed_version)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "test GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 9))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(total_memory=16 * 2**30))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    initial_seeds = []

    def observe_model(dtype):
        initial_seeds.append(torch.initial_seed())
        raise ModelReached

    monkeypatch.setattr(train_ca1, "load_model", observe_model)
    args = SimpleNamespace(packet=packet_dir, output=tmp_path / "run", resume=None, report_to="none")
    previous_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(7)
        with pytest.raises(ModelReached):
            train_ca1.train(args, {}, None)
    finally:
        torch.random.set_rng_state(previous_state)
    assert initial_seeds == [42]
