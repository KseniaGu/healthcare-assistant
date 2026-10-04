"""Current research CLIs import without historical modules or private data."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("module", [
    "research.extraction.datasets.build_ca1_dataset",
    "research.extraction.datasets.packet",
    "research.extraction.training.train_ca1",
    "research.extraction.evaluation.evaluate_ca1_adapter",
])
def test_cli_help_from_research_only_bundle(tmp_path, module):
    source = Path(__file__).resolve().parents[3]
    research = tmp_path / "research"
    (research / "extraction").mkdir(parents=True)
    shutil.copy2(source / "research/__init__.py", research / "__init__.py")
    shutil.copy2(source / "research/extraction/__init__.py", research / "extraction/__init__.py")
    for package in ("datasets", "training", "evaluation"):
        shutil.copytree(source / "research/extraction" / package, research / "extraction" / package,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.md", "*.txt"))
    result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=tmp_path,
                            env={**os.environ, "PYTHONPATH": str(tmp_path)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
