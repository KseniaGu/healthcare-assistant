"""Offline extraction regressions; no processor initialization, APIs, or database."""

import asyncio
import copy
import json
import tempfile
import unittest
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from google.genai import types
from google.genai.errors import ClientError

from backend.data_processing.laboratory_test import LaboratoryTestProcessor
from backend.data_processing.run_extraction_baseline import run_baseline, sha256
from backend.models.llm_clients.gemini import GeminiClient
from backend.utils.exceptions import LaboratoryTestProcessingError
from backend.utils.extraction import parse_and_validate_extraction
from backend.utils.prompt_engine import PromptEngine
from configs.data import DataSettings
from configs.enums import PromptTypeEnum
from configs.models import ModelSettings

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "tests/samples/extraction_poc"
ENGINE = PromptEngine(ROOT / "backend/prompts")
SCHEMA = ENGINE.render(PromptTypeEnum.laboratory_test_processor, "v4", input_text="synthetic")[1][
    "response_json_schema"
]


class ExtractionTests(unittest.TestCase):
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

    def test_schema_accepts_faithful_outputs_and_rejects_contract_violations(self):
        for output in self.expected:
            self.assertEqual(parse_and_validate_extraction(json.dumps(output), SCHEMA), output)
        mutations = [
            (lambda x: x.update(date="2025-02-30"), "$.date"),
            (lambda x: x.pop("errors"), "$"),
            (lambda x: x["results"][0].update(name=" \t"), "name"),
            (lambda x: x["results"][0].update(name=None), "name"),
            (lambda x: x["results"][0].update(value=4.5), "value"),
            (lambda x: x["results"][0].update(value_min=0.2), "results"),
            (lambda x: x.update(flag="Normal"), "$"),
            (lambda x: x.update(errors=["same", "same"]), "errors"),
            (lambda x: x.update(errors=["x" * 161]), "errors"),
            (lambda x: x.update(errors=[str(i) for i in range(6)]), "errors"),
        ]
        for mutate, location in mutations:
            output = copy.deepcopy(self.expected[0])
            mutate(output)
            with self.subTest(location=location), self.assertRaisesRegex(ValueError, "schema violation") as raised:
                parse_and_validate_extraction(json.dumps(output), SCHEMA)
            self.assertIn(location, str(raised.exception))
        with self.assertRaises(json.JSONDecodeError):
            parse_and_validate_extraction('{"date":', SCHEMA)

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
            patch("backend.data_processing.run_extraction_baseline.time.time", return_value=105),
            patch("backend.data_processing.run_extraction_baseline.time.sleep") as sleep,
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

    def processor(self):
        processor = LaboratoryTestProcessor.__new__(LaboratoryTestProcessor)
        processor.model_settings = ModelSettings()
        processor.data_settings = DataSettings()
        processor.data_processor = self.client
        processor.prompt_engine = ENGINE
        processor.logger = Mock()
        processor.logger.start_as_current_observation.return_value = nullcontext(Mock())
        return processor

    def test_application_validates_fresh_results_and_preserves_legacy_cache(self):
        processor = self.processor()
        output_path = self.directory / "processed.json"
        process = LaboratoryTestProcessor.process_data.__wrapped__
        self.generate.return_value = self.response({"results": []})
        with self.assertRaisesRegex(LaboratoryTestProcessingError, "schema violation"):
            process(processor, self.inputs[0], output_path, "override-model")
        self.assertFalse(output_path.exists())
        self.assertEqual(len(list(self.directory.rglob("*.txt"))), 1)
        self.generate.return_value = self.response(self.expected[0])
        step, output = process(processor, self.inputs[0], output_path, "override-model")
        self.assertEqual(step.model_version, "override-model")
        self.assertEqual(output, self.expected[0])
        output_path.write_text(json.dumps(self.expected[0]["results"]))
        self.generate.reset_mock()
        step, legacy = process(processor, self.inputs[0], output_path, "override-model")
        self.assertIsNone(step)
        self.assertIsInstance(legacy, list)
        self.generate.assert_not_called()

    def test_persistence_keeps_report_units_dates_warnings_and_safe_flags(self):
        processor = self.processor()
        processor.database = SimpleNamespace(
            add_report=AsyncMock(return_value=SimpleNamespace(id=1, uuid="synthetic-report")),
            get_test_catalog_id=AsyncMock(return_value=2),
            add_test_observations=AsyncMock(),
        )
        rows = [
            {"name": f"Marker {i}", "value": "4.50", "unit": "мкМЕ/мл", "unit_normalized": "mIU/L", "flag": flag}
            for i, flag in enumerate((None, "unexpected", "Normal", "High", "Low"))
        ]
        rows.append({"name": "No flag", "value": 0, "unit": "%"})
        rows.extend({"name": name, "value": "7"} for name in (None, "", " \t"))
        rows.extend({"name": name, "value": value} for name, value in (("Qualitative", "negative"), ("Range", "0-2")))
        report = {"date": "2025-04-10", "results": rows, "errors": ["Synthetic warning"]}
        path = self.directory / "report.json"
        path.write_text(json.dumps(report))
        with self.assertLogs("Laboratory results processing", level="WARNING"):
            asyncio.run(
                LaboratoryTestProcessor.save_report.__wrapped__(
                    processor, 1, 1, Path("synthetic.pdf"), path, report, datetime.now(UTC), 1
                )
            )
        saved = processor.database.add_report.call_args.args[2]
        self.assertEqual(saved.raw_json, report)
        self.assertEqual(saved.report_date.date().isoformat(), report["date"])
        observations = processor.database.add_test_observations.call_args.args[0]
        self.assertEqual(len(observations), 8)
        self.assertEqual(
            [r.flag for r in observations],
            ["unknown", "unknown", "normal", "high", "low", "unknown", "unknown", "unknown"],
        )
        self.assertEqual(observations[0].unit, "мкМЕ/мл")
        self.assertEqual(observations[0].observed_value, 4.5)
        self.assertEqual(observations[0].observed_value_text, "4.50")
        self.assertEqual([row.observed_value_text for row in observations[-2:]], ["negative", "0-2"])
        self.assertTrue(all(row.observed_value is None for row in observations[-2:]))
        self.assertEqual(processor.database.get_test_catalog_id.call_args_list[0].args[0].preferred_unit, "mIU/L")
        self.assertTrue(
            all(call.args[0].canonical_name.strip() for call in processor.database.get_test_catalog_id.call_args_list)
        )
        asyncio.run(
            LaboratoryTestProcessor.save_report.__wrapped__(
                processor, 1, 1, Path("synthetic.pdf"), path, rows[:1], datetime.now(UTC), 1
            )
        )
        self.assertEqual(processor.database.add_report.call_args.args[2].raw_json, {"results": rows[:1]})


if __name__ == "__main__":
    unittest.main()
