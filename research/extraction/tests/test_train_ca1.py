"""Offline trainer checks without private packets, CUDA, or model weights."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from research.extraction.datasets import packet
from research.extraction.tests.test_ca1_packet import fixture_packet
from research.extraction.training import train_ca1


@pytest.mark.parametrize("status", [" M unrelated.py\n", "M  unrelated.py\n"])
def test_source_commit_rejects_tracked_changes_anywhere(monkeypatch, status):
    calls = []

    def git_output(command, *, cwd=None, text=None):
        calls.append((command, cwd, text))
        return status

    monkeypatch.setattr(train_ca1.subprocess, "check_output", git_output)
    with pytest.raises(ValueError, match="uncommitted tracked changes"):
        train_ca1.current_source_commit()
    assert calls == [(["git", "status", "--porcelain", "--untracked-files=no"], packet.ROOT, True)]


def test_source_commit_uses_repository_root(monkeypatch):
    calls = []

    def git_output(command, *, cwd=None, text=None):
        calls.append((command, cwd, text))
        return "" if command[1] == "status" else "a" * 40 + "\n"

    monkeypatch.setattr(train_ca1.subprocess, "check_output", git_output)
    assert train_ca1.current_source_commit() == "a" * 40
    assert all(cwd == packet.ROOT and text is True for _, cwd, text in calls)
    assert calls[-1][0] == ["git", "rev-parse", "HEAD"]


def test_preflight_and_train_check_source_before_cuda(tmp_path, monkeypatch):
    def dirty_source():
        raise ValueError("uncommitted tracked changes")

    monkeypatch.setattr(train_ca1, "current_source_commit", dirty_source)
    monkeypatch.setattr(train_ca1, "cuda_dtype", lambda: pytest.fail("CUDA checked before source"))
    with pytest.raises(ValueError, match="uncommitted tracked changes"):
        train_ca1.preflight([], None)
    args = SimpleNamespace(packet=tmp_path, output=tmp_path / "run", resume=None, report_to="none")
    with pytest.raises(ValueError, match="uncommitted tracked changes"):
        train_ca1.train(args, {}, None)
    assert not args.output.exists()


def test_run_file_replacement_preserves_previous_record(tmp_path, monkeypatch):
    run_file = tmp_path / "run.json"
    original = '{"status": "previous"}\n'
    run_file.write_text(original)

    def failed_replace(self, target):
        assert target == run_file
        assert run_file.read_text() == original
        raise OSError("replace interrupted")

    monkeypatch.setattr(Path, "replace", failed_replace)
    with pytest.raises(OSError, match="replace interrupted"):
        train_ca1.write_run_file(run_file, {"status": "running"})
    assert run_file.read_text() == original


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
    recipe = packet.read_json(args.output / "run.json")["recipe"]
    assert (recipe["seed"], recipe["epochs"], recipe["eval_steps"], recipe["lora_rank"]) == (42, 3, 40, 16)


def test_checkpoint_selection_uses_only_finite_epoch_losses(tmp_path):
    for step in (40, 80, 120):
        (tmp_path / f"checkpoint-{step}").mkdir()
    history = [
        {"step": 10, "eval_loss": 0.001},
        {"step": 40, "eval_loss": 0.3},
        {"step": 80, "eval_loss": 0.2},
        {"step": 120, "eval_loss": 0.4},
    ]
    checkpoint, best_loss, losses = train_ca1.select_checkpoint(history, tmp_path)
    assert checkpoint.name == "checkpoint-80"
    assert best_loss == 0.2
    assert [entry["step"] for entry in losses] == [40, 80, 120]
    history[2]["eval_loss"] = float("nan")
    with pytest.raises(ValueError, match="finite validation losses"):
        train_ca1.select_checkpoint(history, tmp_path)
