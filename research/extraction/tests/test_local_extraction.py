"""Local evaluation regressions using fictitious packets and mocked Ollama HTTP."""

import copy
import http.client
import io
import json
import shutil
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from research.extraction import evaluate_local_extraction as experiment


class LocalExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "packet"
        self.output = self.root / "run"
        self.targets = {}
        review = {
            "user_decisions": [{"id": f"D{i}", "status": "approved"} for i in range(1, 7)],
            "documents": [],
            "source_sha256": {},
        }
        historical = {"reference_target_sha256": "old_review_hash", "documents": []}
        for identifier, count in experiment.COUNTS.items():
            target = {
                "date": "2025-04-18",
                "results": [
                    {"name": f"Marker {i}", "value": "4.50", "unit": "мкМЕ/л", "reference_range": None}
                    for i in range(count)
                ],
                "errors": [],
            }
            self.targets[identifier] = target
            sources = {"pdf": f"sources/{identifier}.pdf", "parsed_text": f"sources/{identifier}.txt"}
            for role, relative in sources.items():
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"FICTITIOUS {role} {identifier}")
                review["source_sha256"][relative] = experiment.sha(path.read_bytes())
            review["documents"].append(
                {
                    "document_id": identifier,
                    "organization_alias": identifier[0],
                    "sources": sources,
                    "proposed_target": target,
                    "row_evidence": [
                        {
                            "row": i + 1,
                            "parsed_lines": [i + 1, i + 1],
                            "review_status": "assistant_reviewed_user_pending",
                            "notes": [],
                        }
                        for i in range(count)
                    ],
                    "date_evidence_parsed_lines": [1],
                    "supplemental_context": {},
                    "excluded_non_result_rows": [],
                    "duplicate_pdf_paths": [],
                }
            )
            prediction = f"sources/{identifier}.json"
            experiment.write_json(self.root / prediction, target)
            historical["documents"].append({"document_id": identifier, "prediction": prediction})
        review_dir = self.root / "data/review"
        review_dir.mkdir(parents=True)
        self.review_path = review_dir / "structured_extraction_labels.proposed.json"
        experiment.write_json(self.review_path, review)
        experiment.write_json(review_dir / "structured_extraction_v4_real_results.json", historical)
        for relative in (
            "backend/prompts/laboratory_test_processor/v4.yaml",
            "tests/samples/extraction_poc/omissions.txt",
            "tests/samples/extraction_poc/omissions.expected.json",
        ):
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(experiment.ROOT / relative, destination)
        self.smoke = experiment.read_json(self.root / "tests/samples/extraction_poc/omissions.expected.json")
        self.allocation = {
            "models": [
                {
                    "name": experiment.MODEL,
                    "digest": experiment.DIGEST,
                    "context_length": 32768,
                    "size_vram": 100,
                    "size": 100,
                }
            ]
        }

    def freeze(self):
        experiment.freeze(self.dataset, self.root)

    def responses(self):
        return [self.response(self.smoke)] + [self.response(self.targets[i]) for i in experiment.DEVELOPMENT]

    @staticmethod
    def response(prediction, **changes):
        result = {
            "model": experiment.MODEL,
            "message": {"content": json.dumps(prediction)},
            "done": True,
            "done_reason": "stop",
            "eval_count": 20,
            "eval_duration": 2000000000,
        }
        result.update(changes)
        return experiment.json_bytes(result)

    def run_mocked(self, responses, allocation=None):
        with (
            patch.object(experiment, "preflight", return_value={"pinned": True}),
            patch.object(experiment, "metadata", return_value=allocation or self.allocation),
            patch.object(experiment, "request_bytes", side_effect=responses) as transport,
        ):
            status = experiment.run(self.dataset, self.output)
        return status, transport

    def test_freeze_preserves_approvals_and_reserves_b(self):
        before = self.review_path.read_bytes()
        self.freeze()
        packet = experiment.load_packet(self.dataset)
        self.assertEqual(before, self.review_path.read_bytes())
        dev = [d for d in packet["documents"] if d["split"] == "development"]
        self.assertEqual([d["document_id"] for d in dev], list(experiment.DEVELOPMENT))
        self.assertEqual(sum(len(d["row_evidence"]) for d in dev), 123)
        self.assertTrue(
            all(r["review_status"] == "user_approved" for d in packet["documents"] for r in d["row_evidence"])
        )
        self.assertFalse((self.dataset / "gemini/B1.json").exists())
        with self.assertRaises(FileExistsError):
            self.freeze()
        (self.dataset / "inputs/A1.txt").write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            experiment.load_packet(self.dataset)

    def test_freeze_rejects_unapproved_duplicate_and_changed_sources(self):
        original = experiment.read_json(self.review_path)
        for kind in ("approval", "duplicate", "hash", "count"):
            review = copy.deepcopy(original)
            if kind == "approval":
                review["user_decisions"][0]["status"] = "pending"
            elif kind == "duplicate":
                review["documents"][1]["sources"]["parsed_text"] = review["documents"][0]["sources"]["parsed_text"]
            elif kind == "hash":
                review["source_sha256"]["sources/A1.txt"] = "incorrect"
            else:
                review["documents"].pop()
            experiment.write_json(self.review_path, review)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.freeze()
            self.assertFalse(self.dataset.exists())

    def test_successful_run_uses_exact_requests_and_cannot_repeat(self):
        self.freeze()
        status, transport = self.run_mocked(self.responses())
        self.assertEqual(status, 0)
        self.assertEqual(transport.call_count, 6)
        for call in transport.call_args_list:
            request = call.args[1]
            self.assertEqual(call.args[0], "/api/chat")
            self.assertEqual(request["options"], experiment.OPTIONS)
            self.assertFalse(request["think"])
            self.assertFalse(request["stream"])
            self.assertEqual(len(request["messages"]), 2)
            self.assertNotIn("FICTITIOUS parsed_text B", request["messages"][1]["content"])
        scores = experiment.score(self.dataset, self.output)
        self.assertEqual(scores["summary"]["qwen"]["pooled"]["strict_pass"], 5)
        self.assertEqual(scores["summary"]["qwen"]["pooled"]["field_matches"], 492)
        self.assertEqual(scores["smoke"]["warning_status"], "pending_review")
        with self.assertRaises(FileExistsError):
            experiment.run(self.dataset, self.output)
        self.output = self.root / "second-run"
        with self.assertRaisesRegex(ValueError, "already has an attempted run"):
            self.run_mocked(self.responses())

    def test_completed_invalid_output_continues_and_preserves_raw(self):
        self.freeze()
        responses = self.responses()
        bad = copy.deepcopy(self.targets["A1"])
        bad["date"] = "2025-02-30"
        responses[1] = self.response(bad)
        responses[2] = self.response(self.targets["A2"], done_reason="length")
        responses[3] = self.response(self.targets["A3"], message={"content": "{bad"})
        responses[4] = self.response(self.targets["A4"], message={"content": "{}", "tool_calls": [{"name": "ignored"}]})
        status, transport = self.run_mocked(responses)
        self.assertEqual((status, transport.call_count), (1, 6))
        for identifier in ("A1", "A2", "A3", "A4"):
            self.assertTrue((self.output / identifier / "response.raw.json").exists())
            self.assertFalse((self.output / identifier / "prediction.json").exists())
        scores = experiment.score(self.dataset, self.output)
        pooled = scores["summary"]["qwen"]["pooled"]
        self.assertEqual((pooled["reports"], pooled["expected_rows"], pooled["strict_pass"]), (5, 123, 1))

    def test_smoke_failure_stops_before_real_reports(self):
        self.freeze()
        status, transport = self.run_mocked([self.response({}, done=False)])
        self.assertEqual((status, transport.call_count), (1, 1))
        records = experiment.read_json(self.output / "manifest.json")["records"]
        self.assertTrue(all(r["status"] == "not_attempted" for r in records[1:]))

    def test_timeout_partial_response_and_interrupt_are_not_retried(self):
        self.freeze()
        for i, error in enumerate(
            (TimeoutError("timeout"), http.client.IncompleteRead(b"partial"), KeyboardInterrupt())
        ):
            packet = self.root / f"packet-{i}"
            shutil.copytree(self.dataset, packet)
            with (
                patch.object(experiment, "preflight", return_value={}),
                patch.object(experiment, "metadata", return_value=self.allocation),
                patch.object(experiment, "request_bytes", side_effect=error) as transport,
            ):
                output = self.root / f"run-{i}"
                self.assertEqual(experiment.run(packet, output), 1)
                self.assertEqual(transport.call_count, 1)
            if isinstance(error, http.client.IncompleteRead):
                self.assertEqual((output / "smoke/error.raw").read_bytes(), b"partial")

    def test_unexpected_loaded_context_stops_batch(self):
        self.freeze()
        allocation = copy.deepcopy(self.allocation)
        allocation["models"][0]["context_length"] = 4096
        status, transport = self.run_mocked(self.responses(), allocation)
        self.assertEqual((status, transport.call_count), (1, 1))
        self.assertFalse((self.output / "smoke/prediction.json").exists())

    def test_preflight_rejects_remote_changed_model_and_version(self):
        for change in ("remote", "digest", "version"):
            model = {"name": experiment.MODEL, "digest": experiment.DIGEST}
            version = experiment.VERSION
            if change == "remote":
                model["remote_host"] = "https://example.org"
            elif change == "digest":
                model["digest"] = "changed"
            else:
                version = "changed"
            with (
                patch.object(experiment, "metadata", side_effect=[{"version": version}, {"models": [model]}]),
                self.subTest(change=change),
                self.assertRaises(ValueError),
            ):
                experiment.preflight()

    def test_preflight_checks_native_template_parameters_and_quantization(self):
        details = {
            "details": {"quantization_level": "Q8_0"},
            "template": "native",
            "parameters": "stops",
            "model_info": {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen2"},
        }
        for change in (None, "template", "parameters", "quantization"):
            show = copy.deepcopy(details)
            if change == "quantization":
                show["details"]["quantization_level"] = "Q4_K_M"
            elif change:
                show[change] = "changed"
            responses = [
                {"version": experiment.VERSION},
                {"models": [{"name": experiment.MODEL, "digest": experiment.DIGEST}]},
                show,
            ]
            with (
                patch.object(experiment, "TEMPLATE_HASH", experiment.sha(b"native")),
                patch.object(experiment, "PARAMETERS_HASH", experiment.sha(b"stops")),
                patch.object(experiment, "metadata", side_effect=responses),
            ):
                if change:
                    with self.assertRaises(ValueError):
                        experiment.preflight()
                else:
                    self.assertEqual(experiment.preflight()["show"], details)

    def test_http_redirect_is_a_transport_failure(self):
        with self.assertRaises(urllib.error.URLError):
            experiment.NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://example.org")

    def test_http_failure_retains_error_and_stops(self):
        self.freeze()
        error = urllib.error.HTTPError("http://local", 500, "runtime failure", {}, io.BytesIO(b"runtime evidence"))
        status, transport = self.run_mocked([error])
        self.assertEqual((status, transport.call_count), (1, 1))
        self.assertEqual((self.output / "smoke/error.raw").read_bytes(), b"runtime evidence")

    def test_malformed_transport_envelope_stops_and_reserved_scoring_is_rejected(self):
        self.freeze()
        status, transport = self.run_mocked([b"not an HTTP JSON envelope"])
        self.assertEqual((status, transport.call_count), (1, 1))
        manifest = experiment.read_json(self.output / "manifest.json")
        manifest["records"][-1]["document_id"] = "B1"
        experiment.write_json(self.output / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "B scoring is forbidden"):
            experiment.score(self.dataset, self.output)

    def test_scoring_precision_nulls_missing_extra_duplicates_order_and_warnings(self):
        target = copy.deepcopy(self.targets["A4"])
        for field, value in (("value", "4.5"), ("reference_range", ""), ("unit", "мкМЕ/Л")):
            prediction = copy.deepcopy(target)
            prediction["results"][0][field] = value
            result = experiment.compare(target, prediction)
            self.assertEqual(result["field_matches"], 7)
            self.assertFalse(result["strict_pass"])
        prediction = copy.deepcopy(target)
        prediction["results"].reverse()
        result = experiment.compare(target, prediction)
        self.assertTrue(result["order_only_difference"])
        self.assertEqual(result["record_recall"], 1)
        prediction["results"].pop()
        self.assertEqual(experiment.compare(target, prediction)["missing_exact_records"], 1)
        prediction = copy.deepcopy(target)
        prediction["results"].append(copy.deepcopy(prediction["results"][0]))
        result = experiment.compare(target, prediction)
        self.assertEqual(result["extra_duplicate_records"], 1)
        self.assertAlmostEqual(result["record_precision"], 2 / 3)
        prediction = copy.deepcopy(target)
        prediction["errors"] = ["Unexpected warning"]
        self.assertEqual(experiment.compare(target, prediction)["warning_status"], "pending_review")

    def test_warning_review_is_bound_to_raw_response(self):
        self.freeze()
        self.run_mocked(self.responses())
        manifest = experiment.read_json(self.output / "manifest.json")
        review = {
            "smoke": {
                "raw_sha256": manifest["records"][0]["raw_sha256"],
                "status": "pass",
                "reason": "Matches omission checklist.",
                "reviewer": "test reviewer",
            }
        }
        experiment.write_json(self.output / "warning_review.json", review)
        scores = experiment.score(self.dataset, self.output)
        self.assertEqual(scores["smoke"]["warning_status"], "pass")
        self.assertTrue(scores["smoke"]["strict_pass"])
        review["smoke"]["raw_sha256"] = "stale"
        experiment.write_json(self.output / "warning_review.json", review)
        with self.assertRaisesRegex(ValueError, "Warning review does not match"):
            experiment.score(self.dataset, self.output)

    def test_oversized_input_and_altered_prediction_are_rejected(self):
        self.freeze()
        with self.assertRaisesRegex(ValueError, "exceeds context"):
            experiment.make_request(self.dataset, "Я" * 32768)
        self.run_mocked(self.responses())
        (self.output / "A1/prediction.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Changed response"):
            experiment.score(self.dataset, self.output)

    def test_followup_profiles_preserve_prompt_and_pin_settings(self):
        self.freeze()
        original, _ = experiment.make_request(self.dataset, "Fictitious input")
        for profile in experiment.PROFILES[1:]:
            request, bound = experiment.make_request(self.dataset, "Fictitious input", profile)
            self.assertEqual(request["messages"], original["messages"])
            expected_schema = copy.deepcopy(original["format"])
            if profile in ("qwen35-4b-compat", "qwen35-4b-thinking"):
                del expected_schema["properties"]["results"]["items"]["properties"]["name"]["pattern"]
            self.assertEqual(request["format"], expected_schema)
            self.assertEqual(request["options"]["num_ctx"], 32768)
            self.assertLess(bound + request["options"]["num_predict"], 32768)
            thinking = profile.endswith("-thinking")
            self.assertEqual(request["think"], thinking)
            self.assertEqual(request["options"]["presence_penalty"], 1.5 if thinking else 0)
            self.assertEqual(request["options"]["temperature"], 1.0 if thinking else 0)
        with self.assertRaises(ValueError):
            experiment.configuration("deepseek-v4-flash:cloud")

    def test_followup_ledgers_preserve_original_and_share_thinking_budget(self):
        self.freeze()
        self.run_mocked(self.responses())
        original_ledger = self.dataset.parent / f".{self.dataset.name}.execution.json"
        before = original_ledger.read_bytes()
        for profile in ("qwen35-4b", "qwen35-9b", "qwen35-9b-thinking"):
            config = experiment.configuration(profile)
            responses = [json.loads(r) for r in self.responses()]
            for response in responses:
                response["model"] = config["model"]
                if config["think"]:
                    response["message"]["thinking"] = "Fictitious reasoning"
            allocation = copy.deepcopy(self.allocation)
            allocation["models"][0].update(name=config["model"], digest=config["digest"])
            output = self.root / profile
            with (
                patch.object(experiment, "preflight", return_value={}),
                patch.object(experiment, "metadata", return_value=allocation),
                patch.object(
                    experiment, "request_bytes", side_effect=list(map(experiment.json_bytes, responses))
                ) as transport,
            ):
                self.assertEqual(experiment.run(self.dataset, output, profile), 0)
                self.assertEqual(transport.call_count, 6)
            scores = experiment.score(self.dataset, output)
            self.assertEqual(scores["summary"]["qwen"]["pooled"]["strict_pass"], 5)
        self.assertEqual(original_ledger.read_bytes(), before)
        with patch.object(experiment, "preflight", return_value={}):
            with self.assertRaisesRegex(ValueError, "already has an attempted run"):
                experiment.run(self.dataset, self.root / "second-thinking", "qwen35-4b-thinking")

    def test_thinking_profile_rejects_silently_disabled_thinking(self):
        self.freeze()
        config = experiment.configuration("qwen35-9b-thinking")
        allocation = copy.deepcopy(self.allocation)
        allocation["models"][0].update(name=config["model"], digest=config["digest"])
        with (
            patch.object(experiment, "preflight", return_value={}),
            patch.object(experiment, "metadata", return_value=allocation),
            patch.object(
                experiment, "request_bytes", return_value=self.response(self.smoke, model=config["model"])
            ) as transport,
        ):
            self.assertEqual(experiment.run(self.dataset, self.output, "qwen35-9b-thinking"), 1)
            self.assertEqual(transport.call_count, 1)
        self.assertTrue((self.output / "smoke/response.raw.json").exists())
        self.assertFalse((self.output / "smoke/prediction.json").exists())

    def test_mlx_compatibility_keeps_full_post_validation(self):
        self.freeze()
        config = experiment.configuration("qwen35-4b-compat")
        allocation = copy.deepcopy(self.allocation)
        allocation["models"][0].update(name=config["model"], digest=config["digest"])
        invalid = copy.deepcopy(self.smoke)
        invalid["results"][0]["name"] = " "
        with (
            patch.object(experiment, "preflight", return_value={}),
            patch.object(experiment, "metadata", return_value=allocation),
            patch.object(
                experiment, "request_bytes", return_value=self.response(invalid, model=config["model"])
            ) as transport,
        ):
            self.assertEqual(experiment.run(self.dataset, self.output, "qwen35-4b-compat"), 1)
            self.assertEqual(transport.call_count, 1)
        self.assertFalse((self.output / "smoke/prediction.json").exists())


if __name__ == "__main__":
    unittest.main()
