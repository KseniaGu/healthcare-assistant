"""Offline regressions: python -m unittest discover -s tests/unit -v."""

import io
import json
import runpy
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from alembic.config import Config
from google.genai import types

from backend.models.llm_clients.gemini import GeminiClient
from backend.utils.common_funcs import get_logger
from backend.utils.exceptions import IncompleteLLMResponseError


class ResponseHandlingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.client = GeminiClient.__new__(GeminiClient)
        self.client.config = SimpleNamespace(
            temperature=0.0,
            max_tokens=16384,
            use_grounding=False,
            response_mime_type="application/json",
            model_name="test-model",
        )
        self.client.params = self.client._normalize_parameters()
        self.client.path_settings = SimpleNamespace(llm_outputs_dir=Path(self.directory.name))
        self.generate = Mock()
        self.client.client = SimpleNamespace(models=SimpleNamespace(generate_content=self.generate))

    @staticmethod
    def response(parts, finish_reason=types.FinishReason.STOP):
        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(content=types.Content(parts=parts), finish_reason=finish_reason),
            ]
        )

    def test_json_request_and_complete_text_parts(self):
        self.generate.return_value = self.response(
            [
                types.Part(text="Internal reasoning", thought=True),
                types.Part(text='{"date": null,'),
                types.Part(text='"results": []}'),
            ]
        )
        schema = {"type": "object", "properties": {"results": {"type": "array", "items": {"type": "object"}}}}
        output, _, path = self.client.generate(
            "Synthetic report",
            system_instruction="Return JSON",
            response_json_schema=schema,
        )
        params = self.generate.call_args.kwargs["config"]
        self.assertEqual(params.response_mime_type, "application/json")
        self.assertEqual(params.response_json_schema, schema)
        self.assertEqual(params.max_output_tokens, 16384)
        self.assertEqual(json.loads(output), {"date": None, "results": []})
        self.assertEqual(path.read_text(), output)
        # One cached client must not retain a previous request's schema/instructions.
        self.client.generate("Another synthetic report")
        self.assertIsNone(self.generate.call_args.kwargs["config"].response_json_schema)
        self.assertIsNot(params, self.client.params)
        self.assertIsNone(self.client.params.system_instruction)

    def test_truncated_response_is_not_returned_as_success(self):
        self.generate.return_value = self.response(
            [types.Part(text='```json\n[{"test_name": "Synthetic')],
            types.FinishReason.MAX_TOKENS,
        )
        with self.assertRaises(IncompleteLLMResponseError) as raised:
            self.client.generate("Synthetic report")
        self.assertIn("max_output_tokens=16384", str(raised.exception))
        self.assertIn("cut off", raised.exception.user_message)
        self.assertFalse(raised.exception.is_retryable)
        self.assertEqual(len(list(Path(self.directory.name).rglob("*.txt"))), 1)

    def test_migrations_keep_application_errors_visible(self):
        root = Path(__file__).resolve().parents[2]
        logger, handler = get_logger("response-regression")
        self.addCleanup(logger.removeHandler, handler)
        stream = io.StringIO()
        config = Config(str(root / ".devcontainer/alembic.ini"))
        with (
            patch("sys.stderr", stream),
            patch("alembic.context.config", config, create=True),
            patch("alembic.context.is_offline_mode", return_value=True),
            patch("alembic.context.configure"),
            patch("alembic.context.begin_transaction", return_value=nullcontext()),
            patch("alembic.context.run_migrations"),
        ):
            runpy.run_path(str(root / "src/databases/postgres/migrations/alembic/env.py"))
            self.assertFalse(logger.disabled)
            logger.error("Synthetic extraction failure")
        self.assertIn("Synthetic extraction failure", stream.getvalue())
        self.assertTrue(any("Synthetic extraction failure" in message for message in handler.get_logs()))


if __name__ == "__main__":
    unittest.main()
