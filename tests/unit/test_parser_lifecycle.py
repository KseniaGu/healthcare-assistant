"""Exercise real SDK parsing/error handling with only the API work replaced."""

import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llama_parse import LlamaParse
from pydantic import SecretStr

from backend.models.llm_clients.llama_parse import LlamaParseClient
from backend.utils.exceptions import DocumentParsingError
from backend.utils.helpers import process_test_value


class ParserLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            api_key=SecretStr("synthetic-test-key"),
            result_type="text",
            max_pages=None,
            language="ru",
            parse_mode="parse_page_without_llm",
            disable_image_extraction=True,
            fast_mode=None,
        )
        self.client = LlamaParseClient(self.config)

    def test_repeated_reports_use_separate_closed_connections(self):
        connections = []
        owners = {}

        async def parse_one(parser, file_path, *args, **kwargs):
            connection = parser.aclient
            loop = asyncio.get_running_loop()
            if connection in owners and owners[connection] is not loop:
                raise RuntimeError("Event loop is closed")
            owners[connection] = loop
            connections.append(connection)
            self.assertFalse(connection.is_closed)
            self.assertFalse(parser.ignore_errors)
            return [("synthetic-job", {"text": f"Parsed {file_path.name}"})]

        with patch.object(LlamaParse, "_parse_one", parse_one):
            for name in ("first.pdf", "second.pdf"):
                # Match Streamlit's fresh asyncio.run + worker thread per report.
                result = asyncio.run(asyncio.to_thread(self.client.parse, Path(name)))
                self.assertEqual(result, f"Parsed {name}")
                self.assertTrue(connections[-1].is_closed)
                self.assertTrue(owners[connections[-1]].is_closed())
        self.assertIsNot(connections[0], connections[1])
        self.assertEqual(self.config.result_type, "text")

    def test_failure_preserves_cause_and_closes_connection(self):
        connections = []
        failure = RuntimeError("Synthetic upstream failure")

        async def parse_one(parser, *args, **kwargs):
            connections.append(parser.aclient)
            raise failure

        with (
            patch.object(LlamaParse, "_parse_one", parse_one),
            patch("builtins.print"),
            self.assertLogs("Laboratory results processing", level="ERROR"),
            self.assertRaises(DocumentParsingError) as raised,
        ):
            asyncio.run(asyncio.to_thread(self.client.parse, Path("synthetic.pdf")))
        self.assertIs(raised.exception.__cause__, failure)
        self.assertTrue(connections[0].is_closed)
        self.assertIn("Synthetic upstream failure", str(raised.exception))


class ObservationValueTests(unittest.TestCase):
    def test_json_values_and_inequalities(self):
        for value, expected in (
            (3.5, (3.5, "")),
            (0, (0.0, "")),
            (-2, (-2.0, "")),
            (None, (None, "")),
            ("3.5", (3.5, "")),
            ("< 3.0", (3.0, "<")),
            ("≥ 2.5", (2.5, "≥")),
        ):
            with self.subTest(value=value):
                self.assertEqual(process_test_value(value, ("<", ">", "≤", "≥", "~", "≈")), expected)
        for value in (True, "pending"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                process_test_value(value, ("<", ">"))


if __name__ == "__main__":
    unittest.main()
