"""Offline Gemini baseline tests with fictitious inputs and mocked requests."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from google.genai import types
from google.genai.errors import ClientError

from backend.models.llm_clients.gemini import GeminiClient
from backend.utils.prompt_engine import PromptEngine
from configs.constants import ROOT_DIR
from research.extraction.run_extraction_baseline import run_baseline, sha256

SAMPLES = ROOT_DIR / "tests/samples/extraction_poc"
ENGINE = PromptEngine(ROOT_DIR / "backend/prompts")


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.client = GeminiClient.__new__(GeminiClient)
        self.client.config = SimpleNamespace(
            temperature=0.0,
            max_tokens=16384,
            use_grounding=False,
            response_mime_type="application/json",
            model_name="gemini-3.1-flash-lite",
        )
        self.client.params = self.client._normalize_parameters()
        self.client.path_settings = SimpleNamespace(llm_outputs_dir=self.directory / "raw")
        self.generate = Mock()
        self.client.client = SimpleNamespace(models=SimpleNamespace(generate_content=self.generate))
        self.inputs = [SAMPLES / f"{name}.txt" for name in ("scalars", "references", "omissions")]
        self.expected = [json.loads(path.with_suffix(".expected.json").read_text()) for path in self.inputs]
        self.budget = self.directory / "budget.json"

    @staticmethod
    def response(output, finish_reason=types.FinishReason.STOP):
        text = output if isinstance(output, str) else json.dumps(output)
        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(content=types.Content(parts=[types.Part(text=text)]), finishReason=finish_reason)
            ],
            usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=10, candidates_token_count=20),
        )

    def run_batch(self, name="run", max_attempts=3, version="v4", interval=0):
        return run_baseline(
            self.inputs,
            self.directory / name,
            version,
            self.client,
            ENGINE,
            budget_file=self.budget,
            max_attempts=max_attempts,
            interval_seconds=interval,
        )

    def test_runner_retains_failed_raw_outputs_and_records_actual_model(self):
        self.client.config.model_name = "explicit-model-override"
        self.generate.side_effect = [
            self.response(self.expected[0]),
            self.response('{"results":'),
            self.response(self.expected[2]),
        ]
        self.assertEqual(self.run_batch(), 1)
        manifest = json.loads((self.directory / "run/manifest.json").read_text())
        self.assertEqual([r["status"] for r in manifest["records"]], ["success", "failed", "success"])
        self.assertEqual(manifest["model"], "explicit-model-override")
        self.assertEqual(manifest["generation_settings"]["max_output_tokens"], 16384)
        self.assertEqual(manifest["records"][0]["usage"]["prompt_token_count"], 10)
        self.assertEqual(manifest["records"][0]["input_sha256"], sha256(self.inputs[0].read_bytes()))
        self.assertTrue(all(r["schema_sha256"] for r in manifest["records"]))
        self.assertFalse((self.directory / "run/002/prediction.json").exists())
        self.assertEqual(len(manifest["records"][1]["raw_paths"]), 1)
        self.assertTrue((self.directory / "run/prompt.yaml").exists())
        for call in self.generate.call_args_list:
            self.assertEqual(call.kwargs["model"], "explicit-model-override")

    def test_invalid_schema_and_truncation_are_not_predictions(self):
        self.generate.side_effect = [
            self.response({"date": None, "results": []}),
            self.response('{"date":', types.FinishReason.MAX_TOKENS),
            self.response(self.expected[2]),
        ]
        self.assertEqual(self.run_batch(), 1)
        manifest = json.loads((self.directory / "run/manifest.json").read_text())
        for index in (0, 1):
            self.assertEqual(manifest["records"][index]["status"], "failed")
            self.assertEqual(len(manifest["records"][index]["raw_paths"]), 1)
            self.assertFalse((self.directory / f"run/{index + 1:03d}/prediction.json").exists())

    def test_preflight_rejects_duplicate_inputs_and_existing_output_without_calls(self):
        self.inputs[1] = self.inputs[0]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_batch()
        self.assertFalse((self.directory / "run").exists())
        (self.directory / "run").mkdir()
        with self.assertRaises(FileExistsError):
            self.run_batch()
        self.generate.assert_not_called()

    def test_shared_budget_and_pacing_across_invocations(self):
        self.generate.return_value = self.response(self.expected[0])
        self.budget.write_text(json.dumps({"attempts": 7, "stopped": None, "last_attempt_at": 100}))
        with (
            patch("research.extraction.run_extraction_baseline.time.time", return_value=105),
            patch("research.extraction.run_extraction_baseline.time.sleep") as sleep,
        ):
            self.assertEqual(self.run_batch(max_attempts=1, version="v3", interval=10), 1)
            self.assertEqual(self.run_batch(name="revision", interval=10), 1)
        self.assertEqual(self.generate.call_count, 2)
        self.assertEqual(sleep.call_args_list[0].args, (5,))
        self.assertEqual(json.loads(self.budget.read_text())["attempts"], 9)
        with self.assertRaisesRegex(ValueError, "exhausted"):
            self.run_batch(name="forbidden")
        self.assertFalse((self.directory / "forbidden").exists())

    def test_quota_or_billing_error_stops_current_and_later_batches(self):
        for code in (429, 402, 403):
            with self.subTest(code=code):
                self.budget = self.directory / f"budget-{code}.json"
                self.generate.reset_mock()
                self.generate.side_effect = ClientError(code, {"error": {"message": "Synthetic quota/billing failure"}})
                with self.assertLogs("Laboratory results processing", level="ERROR"):
                    self.assertEqual(self.run_batch(name=str(code)), 1)
                self.assertEqual(self.generate.call_count, 1)
                manifest = json.loads((self.directory / str(code) / "manifest.json").read_text())
                self.assertEqual(
                    [r["status"] for r in manifest["records"]], ["failed", "not_attempted", "not_attempted"]
                )
                with self.assertRaisesRegex(ValueError, "stopped"):
                    self.run_batch(name=f"again-{code}")


