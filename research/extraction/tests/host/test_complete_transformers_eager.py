"""Offline checks for the one-time eager completion runner."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.extraction.archive import complete_transformers_eager as completion
from research.extraction.archive import run_transformers_eager_short as short


class CompletionTests(unittest.TestCase):
    def test_short_report_budget_is_separate_and_exactly_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(short, "OUTPUT", root):
                self.assertEqual(short.reserve("A4"), 1)
                self.assertEqual(short.reserve("C1"), 2)
                saved = json.loads((root / "execution.json").read_text())
                self.assertEqual(saved["max_attempts"], 2)
                self.assertEqual([a["document_id"] for a in saved["attempts"]], ["A4", "C1"])
                with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                    short.reserve("A4")

    def test_three_attempt_budget_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "execution.json"
            for identifier in completion.DOCUMENTS:
                completion.reserve(ledger, identifier)
            saved = json.loads(ledger.read_text())
            self.assertEqual(saved["max_attempts"], 3)
            self.assertEqual([a["document_id"] for a in saved["attempts"]], list(completion.DOCUMENTS))
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                completion.reserve(ledger, "A3")

    def test_runtime_failure_retains_partial_and_stops(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            entries = {}
            for identifier in completion.DOCUMENTS:
                (dataset / f"{identifier}.txt").write_text("fixture")
                entries[identifier] = {"document_id": identifier, "input": f"{identifier}.txt"}
            messages = [{"role": "system", "content": "frozen"}, {"role": "user", "content": "fixture"}]
            provenance = {
                "model": completion.exp.HF_MODEL, "revision": "a" * 40, "selected_dtype": "torch.bfloat16",
                "formatted_inputs": {identifier: {"messages_sha256": completion.prior.sha(
                    completion.prior.json_bytes(messages))} for identifier in completion.DOCUMENTS},
            }
            eager = root / "eager"
            eager.mkdir()
            (eager / "provenance.json").write_text("{}")
            called = []

            def fail(_request, directory):
                called.append(directory.name)
                (directory / "content.partial.txt").write_text("{")
                raise RuntimeError("device failure")

            with (patch.object(completion, "OUTPUT", root / "completion"),
                  patch.object(completion.exp, "EAGER_OUTPUT", eager),
                  patch.object(completion.exp, "DATASET", dataset),
                  patch.object(completion, "preflight", return_value=(entries, provenance, {})),
                  patch.object(completion.exp, "messages_for", return_value=messages),
                  patch.object(completion.exp, "memory", return_value={}),
                  patch.object(completion.exp, "generate_hf", side_effect=fail)):
                self.assertEqual(completion.run(), 1)
                with self.assertRaises(FileExistsError):
                    completion.run()
            self.assertEqual(called, ["A3"])
            manifest = json.loads((root / "completion/manifest.json").read_text())
            self.assertEqual([r["status"] for r in manifest["records"]],
                             ["runtime_failure", "not_attempted", "not_attempted"])
            self.assertEqual(len(json.loads((root / "completion/execution.json").read_text())["attempts"]), 1)
            self.assertEqual((root / "completion/A3/content.partial.txt").read_text(), "{")


if __name__ == "__main__":
    unittest.main()
