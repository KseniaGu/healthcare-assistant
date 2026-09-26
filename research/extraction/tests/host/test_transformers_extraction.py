"""Offline guards for the bounded host Qwen diagnostic."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from research.extraction import evaluate_transformers_extraction as diagnostic


class DiagnosticTests(unittest.TestCase):
    def test_ollama_request_has_no_output_constraint(self):
        messages = [{"role": "system", "content": "frozen"}, {"role": "user", "content": "report"}]
        request = diagnostic.ollama_request(messages)
        self.assertEqual(request["messages"], messages)
        self.assertNotIn("format", request)
        self.assertEqual(request["options"], diagnostic.prior.OPTIONS)
        self.assertEqual((request["think"], request["stream"]), (False, True))

    def test_ledger_counts_before_generation_and_blocks_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "execution.json"
            for condition in ("transformers", "ollama"):
                for identifier in diagnostic.IDS:
                    diagnostic.reserve(ledger, condition, identifier)
            self.assertEqual(len(json.loads(ledger.read_text())["attempts"]), 12)
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                diagnostic.reserve(ledger, "ollama", "A1")

    def test_orb_off_rerun_has_six_attempts_and_requires_stopped_services(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / "execution.json"
            with patch.object(diagnostic, "OUTPUT", root), patch.object(diagnostic, "RERUN_OUTPUT", root):
                for identifier in diagnostic.IDS:
                    diagnostic.reserve(ledger, "transformers", identifier)
                self.assertEqual(json.loads(ledger.read_text())["max_attempts"], 6)
                with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                    diagnostic.reserve(ledger, "transformers", "A1")
                with self.assertRaisesRegex(ValueError, "Transformers only"):
                    diagnostic.run("ollama")
        with patch.object(diagnostic.subprocess, "run", return_value=SimpleNamespace(stdout=" 123 /Applications/OrbStack.app/bin/orb\n")):
            with self.assertRaisesRegex(RuntimeError, "still running"):
                diagnostic.stopped_services()
        with patch.object(diagnostic.subprocess, "run", return_value=SimpleNamespace(stdout=" 123 /usr/bin/python\n")):
            self.assertEqual(diagnostic.stopped_services()["matching_processes"], [])

    def test_eager_rerun_has_separate_six_attempt_ledger(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / "execution.json"
            with patch.object(diagnostic, "OUTPUT", root), patch.object(diagnostic, "EAGER_OUTPUT", root):
                for identifier in diagnostic.IDS:
                    diagnostic.reserve(ledger, "transformers", identifier)
                attempts = json.loads(ledger.read_text())
                self.assertEqual(attempts["max_attempts"], 6)
                self.assertEqual([a["document_id"] for a in attempts["attempts"]], list(diagnostic.IDS))
                with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                    diagnostic.reserve(ledger, "transformers", "A1")
                with self.assertRaisesRegex(ValueError, "Transformers only"):
                    diagnostic.run("ollama")

    def fixture(self, root):
        dataset, output = root / "packet", root / "diagnostic"
        dataset.mkdir()
        output.mkdir()
        (dataset / "manifest.json").write_text("{}")
        (dataset / "schema.json").write_text(json.dumps({
            "type": "object", "required": ["date", "results", "errors"],
            "properties": {"date": {"type": "string", "format": "date"},
                           "results": {"type": "array"}, "errors": {"type": "array"}},
        }))
        entries = []
        for identifier in ("smoke", "A1"):
            (dataset / f"{identifier}.txt").write_text("fixture")
            (dataset / f"{identifier}.json").write_text(json.dumps({"date": "2024-01-02", "results": [], "errors": []}))
            entries.append({"document_id": identifier, "input": f"{identifier}.txt", "target": f"{identifier}.json"})
        (output / "provenance.json").write_text(json.dumps({
            "packet_manifest_sha256": diagnostic.prior.sha((dataset / "manifest.json").read_bytes()),
            "revision": "a" * 40,
        }))
        return dataset, output, entries

    def test_complete_invalid_json_is_retained_and_next_document_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset, output, entries = self.fixture(Path(temporary))
            called = []

            def generate(request_path, directory):
                self.assertTrue((output / "execution.json").exists())
                called.append(directory.name)
                (directory / "content.txt").write_text("{invalid" if directory.name == "smoke" else
                                                       '{"date":"2024-01-02","results":[],"errors":[]}')
                return {"finish_reason": "stop"}

            with (patch.object(diagnostic, "DATASET", dataset), patch.object(diagnostic, "OUTPUT", output),
                  patch.object(diagnostic, "packet_entries", return_value=entries),
                  patch.object(diagnostic, "messages_for", return_value=[
                      {"role": "system", "content": "frozen"}, {"role": "user", "content": "fixture"}]),
                  patch.object(diagnostic, "ollama_preflight", return_value={}),
                  patch.object(diagnostic, "memory", return_value={})):
                self.assertEqual(diagnostic.run("ollama", generate=generate), 1)
            self.assertEqual(called, ["smoke", "A1"])
            manifest = json.loads((output / "ollama/manifest.json").read_text())
            self.assertEqual([r["status"] for r in manifest["records"]], ["invalid_output", "success"])
            self.assertEqual((output / "ollama/smoke/content.txt").read_text(), "{invalid")
            self.assertFalse((output / "ollama/smoke/prediction.json").exists())
            self.assertTrue((output / "ollama/A1/prediction.json").exists())

    def test_runtime_failure_stops_condition_without_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset, output, entries = self.fixture(Path(temporary))
            def fail(_request, directory):
                (directory / "content.partial.txt").write_text("{")
                raise TimeoutError("generation timeout")
            with (patch.object(diagnostic, "DATASET", dataset), patch.object(diagnostic, "OUTPUT", output),
                  patch.object(diagnostic, "packet_entries", return_value=entries),
                  patch.object(diagnostic, "messages_for", return_value=[
                      {"role": "system", "content": "frozen"}, {"role": "user", "content": "fixture"}]),
                  patch.object(diagnostic, "ollama_preflight", return_value={}),
                  patch.object(diagnostic, "memory", return_value={})):
                self.assertEqual(diagnostic.run("ollama", generate=fail), 1)
            manifest = json.loads((output / "ollama/manifest.json").read_text())
            self.assertEqual(manifest["attempts"], 1)
            self.assertEqual([r["status"] for r in manifest["records"]], ["runtime_failure", "not_attempted"])
            self.assertEqual((output / "ollama/smoke/content.partial.txt").read_text(), "{")


if __name__ == "__main__":
    unittest.main()
