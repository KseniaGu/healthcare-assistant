"""Exercise SDK retries with a local HTTP transport, without credentials or network."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from google import genai
from pydantic import SecretStr

from backend.models.llm_clients.gemini import GeminiClient
from backend.utils.exceptions import BaseLLMError


class GeminiRetryTests(unittest.TestCase):
    def test_transient_recovery_exhaustion_and_non_retryable_errors(self):
        for statuses in ([503, 200], [503, 503, 503], [400]):
            with self.subTest(statuses=statuses), tempfile.TemporaryDirectory() as directory:
                calls = []

                def respond(request, statuses=statuses, calls=calls):
                    status = statuses[len(calls)]
                    calls.append(status)
                    payload = (
                        {"candidates": [{"content": {"parts": [{"text": "{}"}]}, "finishReason": "STOP"}]}
                        if status == 200
                        else {"error": {"code": status, "message": "Synthetic API error"}}
                    )
                    return httpx.Response(status, json=payload)

                real_client = genai.Client
                with httpx.Client(transport=httpx.MockTransport(respond)) as transport:

                    def create_client(real_client=real_client, transport=transport, **kwargs):
                        kwargs["http_options"].httpx_client = transport
                        return real_client(**kwargs)

                    config = SimpleNamespace(
                        api_key=SecretStr("synthetic-key"),
                        model_name="gemini-3.8-flash",
                        temperature=0.0,
                        max_tokens=16384,
                        use_grounding=False,
                        response_mime_type="application/json",
                    )
                    with patch("backend.models.llm_clients.gemini.genai.Client", side_effect=create_client):
                        client = GeminiClient(config, SimpleNamespace(llm_outputs_dir=Path(directory)))
                    try:
                        with patch("time.sleep"):
                            if statuses[-1] == 200:
                                self.assertEqual(client.generate("Synthetic report")[0], "{}")
                            else:
                                with (
                                    self.assertRaises(BaseLLMError),
                                    self.assertLogs("Laboratory results processing", level="ERROR"),
                                ):
                                    client.generate("Synthetic report")
                        self.assertEqual(calls, statuses)
                    finally:
                        client.client.close()
