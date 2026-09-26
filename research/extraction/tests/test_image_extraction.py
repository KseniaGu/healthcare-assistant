"""Synthetic fixtures only: no medical inputs, downloads, or live API calls."""

import asyncio
import io
import json

import httpx
import pytest
from PIL import Image
from pydantic import SecretStr

from research.extraction import evaluate_image_extraction as pilot


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline tests must not open network connections")

    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("socket.socket.connect", forbidden)


@pytest.fixture
def packet(tmp_path, monkeypatch):
    source = tmp_path / "sources"
    source.mkdir()
    inventory = {}
    for number, (identifier, (organization, stem, _)) in enumerate(pilot.SOURCES.items()):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (number, 0, 0)).save(buffer, format="PNG")
        raw = buffer.getvalue()
        expected = pilot.digest(raw)
        inventory[identifier] = (organization, stem, expected)
        (source / (stem + ".png")).write_bytes(raw)
        pilot.save(source / (stem + ".json"), dict(id=identifier, approval_status="kept", export_sha256=expected))
    monkeypatch.setattr(pilot, "SOURCES", inventory)
    root = tmp_path / "experiment"
    pilot.initialize(root, source)
    return root


def allow(packet):
    pilot.save(
        packet / "access_review.json",
        dict(
            status="verified",
            reviewer="test",
            evidence="fictitious quota",
            checked_at=pilot.now(),
            parser_available_pages=3,
            gemini_available_requests=3,
            gemini_rpm=10,
            gemini_tpm=100000,
        ),
        replace=True,
    )


def row(value="1.00", unit=None):
    return dict(name="Fictitious marker", value=value, unit=unit, reference_range=None)


def target():
    return dict(date="2025-02-03", results=[row()], errors=[])


def reviewed(packet):
    _, ledger = pilot.verify(packet)
    source_review, target_review = {}, {}
    for identifier in pilot.SOURCES:
        image_hash = pilot.digest((packet / f"inputs/{identifier}.png").read_bytes())
        entry = pilot.reserve(packet, ledger, "parse", identifier, 0.00125, image_hash)
        entry["status"] = "success"
        raw = b"Fictitious marker 1.00; 2025-02-03"
        pilot.artifact(packet, ledger, f"parse/{identifier}/input.txt", raw)
        source_review[identifier] = dict(
            status="pass",
            reviewer="test",
            observations=["all fictitious rows checked"],
            input_sha256=pilot.digest(raw),
            image_sha256=image_hash,
        )
        entry = pilot.reserve(packet, ledger, "gemini", identifier, 0.03, pilot.digest(raw))
        response = {"fictitious_response": True}
        pilot.artifact(packet, ledger, f"gemini/{identifier}/response.json", response)
        pilot.finish_prediction(packet, ledger, entry, json.dumps(target()), True)
        pilot.save(packet / f"targets/{identifier}.json", target())
        target_review[identifier] = dict(
            status="pass",
            reviewer="test",
            evidence="fictitious source",
            corrections=[],
            input_sha256=pilot.digest(raw),
            target_sha256=pilot.digest(pilot.encoded(target())),
            gemini_response_sha256=pilot.digest(pilot.encoded(response)),
        )
    pilot.save(packet / "source_review.json", source_review)
    pilot.save(packet / "target_review.json", target_review)
    return ledger


def parser_http(monkeypatch, handler):
    import llama_cloud_services.parse.base as sdk

    from configs.models import LlamaParseSettings

    real = sdk.LlamaParse

    def construct(**kwargs):
        instance = real(**kwargs, custom_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        assert not instance.disable_ocr
        assert instance.parse_mode == "parse_page_without_llm"
        assert instance.max_pages is None and not instance.fast_mode
        return instance

    monkeypatch.setattr("llama_parse.LlamaParse", construct)
    monkeypatch.setattr("configs.models.LlamaParseSettings", lambda: LlamaParseSettings(api_key=SecretStr("fake")))


def test_allowlist_hashes_and_pending_access(packet):
    protocol, ledger = pilot.verify(packet)
    assert protocol["documents"] == ["D0007", "D0002", "D0016"]
    with pytest.raises(pilot.Halt, match="quota verification"):
        pilot.access(packet, ledger, "parse")
    (packet / "inputs/D0007.png").write_bytes(b"changed")
    with pytest.raises(pilot.Halt, match="hash mismatch"):
        pilot.verify(packet)


def test_reject_bad_approval(packet, tmp_path):
    source = tmp_path / "sources"
    stem = pilot.SOURCES["D0007"][1]
    approval = pilot.read(source / (stem + ".json"))
    approval["approval_status"] = "skipped"
    pilot.save(source / (stem + ".json"), approval, replace=True)
    with pytest.raises(pilot.Halt, match="approval/hash"):
        pilot.initialize(tmp_path / "bad", source)
    assert not (tmp_path / "bad").exists()


def test_existing_dir_and_no_stage_bypass(packet):
    with pytest.raises(pilot.Halt, match="requires --resume"):
        pilot.main(["--experiment-dir", str(packet), "--stage", "parse"])
    _, ledger = pilot.verify(packet)
    with pytest.raises(pilot.Halt, match="parser results"):
        pilot.source_gate(packet, ledger)
    for identifier in ["B1", "D0001"]:
        with pytest.raises(pilot.Halt, match="Unauthorized"):
            pilot.reserve(packet, ledger, "parse", identifier, 0, "x")


def test_attempt_and_cost_limits(packet):
    _, ledger = pilot.verify(packet)
    pilot.reserve(packet, ledger, "parse", "D0007", 0.8, "x")
    with pytest.raises(pilot.Halt, match="cannot be repeated"):
        pilot.reserve(packet, ledger, "parse", "D0007", 0, "x")
    with pytest.raises(pilot.Halt, match="ceiling"):
        pilot.reserve(packet, ledger, "gemini", "D0002", 0.3, "x")
    assert len(ledger["attempts"]) == 1


def test_prompt_settings_and_frozen_targets(packet):
    ledger = reviewed(packet)
    pilot.freeze_targets(packet, ledger)
    request, bound = pilot.qwen_request(packet, "D0007")
    assert "format" not in request
    assert request["options"] == pilot.OPTIONS
    assert request["think"] is False and request["stream"] is False
    assert request["messages"] == pilot.messages(packet, "D0007")
    assert bound + 16384 <= 32768
    (packet / "targets/D0007.json").write_text(json.dumps(dict(target(), date=None)))
    with pytest.raises(pilot.Halt, match="hash mismatch"):
        pilot.verify(packet)


def test_correction_history_required(packet):
    ledger = reviewed(packet)
    review = pilot.read(packet / "target_review.json")
    del review["D0002"]["corrections"]
    pilot.save(packet / "target_review.json", review, replace=True)
    with pytest.raises(pilot.Halt, match="Correction history"):
        pilot.freeze_targets(packet, ledger)


@pytest.mark.parametrize(
    "content,complete",
    [
        ('{"date":"2025-02-30","results":[],"errors":[]}', True),
        ('{"date":null}', True),
        ("not json", True),
        ("```json\n{}\n```", True),
        (json.dumps(target()), False),
    ],
)
def test_raw_content_retained_on_output_failure(packet, content, complete):
    _, ledger = pilot.verify(packet)
    entry = pilot.reserve(packet, ledger, "qwen", "D0007", 0, "x")
    pilot.finish_prediction(packet, ledger, entry, content, complete)
    assert entry["status"] == "output_failed"
    assert (packet / "qwen/D0007/content.txt").read_text() == content
    assert not (packet / "qwen/D0007/prediction.json").exists()


def test_exact_scoring_and_denominators(packet):
    ledger = reviewed(packet)
    scores = pilot.score_stage(packet, ledger)
    assert scores["qwen"]["pooled"]["reports"] == 3
    assert scores["qwen"]["pooled"]["expected_rows"] == 3
    assert scores["qwen"]["pooled"]["field_matches"] == 0
    assert scores["gemini"]["pooled"]["strict_pass"] == 3
    baseline = target()
    for change in [row("1.0"), row("1.00", ""), row("<1.00")]:
        result = pilot.compare(baseline, dict(baseline, results=[change]))
        assert result["ordered_record_matches"] == 0
    duplicated = dict(baseline, results=[row(), row()])
    assert pilot.compare(baseline, duplicated)["extra_duplicate_records"] == 1
    two = dict(baseline, results=[row(), row("2")])
    assert pilot.compare(two, dict(two, results=list(reversed(two["results"]))))["order_only_difference"]
    assert pilot.compare(two, baseline)["missing_exact_records"] == 1
    assert pilot.compare(baseline, dict(baseline, errors=["unsupported"]))["warning_status"] == "pending_review"


def test_warning_review_bound_to_response(packet):
    ledger = reviewed(packet)
    pilot.save(
        packet / "warning_review.json",
        {"gemini": {"D0007": dict(status="pass", reviewer="test", reason="checked", response_sha256="wrong")}},
    )
    with pytest.raises(pilot.Halt, match="Warning review response"):
        pilot.score_stage(packet, ledger)


def test_parse_three_uploads_and_cross_host_resume(packet, monkeypatch):
    allow(packet)
    calls = []

    def handler(request):
        calls.append(request)
        if request.method == "POST":
            body = request.read()
            assert b"disable_ocr" not in body
            assert b"parse_page_without_llm" in body
            assert b"disable_image_extraction" in body
            return httpx.Response(200, json={"id": "job-" + str(len(calls))})
        if request.url.path.endswith("/text"):
            return httpx.Response(200, json={"text": "fictitious\n---\nsecond page"})
        return httpx.Response(200, json={"status": "SUCCESS"})

    parser_http(monkeypatch, handler)
    _, ledger = pilot.verify(packet)
    asyncio.run(pilot.parse_stage(packet, ledger))
    assert sum(r.method == "POST" for r in calls) == 3
    assert all(a["status"] == "success" for a in ledger["attempts"])
    before = len(calls)
    asyncio.run(pilot.parse_stage(packet, ledger))
    assert len(calls) == before


def test_uncertain_upload_no_retry_and_preserves_error_body(packet, monkeypatch):
    allow(packet)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, text="fictitious service unavailable")

    parser_http(monkeypatch, handler)
    _, ledger = pilot.verify(packet)
    with pytest.raises(pilot.Halt, match="LlamaParse failed"):
        asyncio.run(pilot.parse_stage(packet, ledger))
    assert len(calls) == 1
    assert ledger["attempts"][0]["status"] == "submission_uncertain"
    assert (packet / "parse/D0007/http-0001.body").read_text() == "fictitious service unavailable"
    with pytest.raises(pilot.Halt, match="cannot be repeated"):
        asyncio.run(pilot.parse_stage(packet, ledger))
    assert len(calls) == 1


def test_existing_job_retrieved_without_reupload(packet, monkeypatch):
    allow(packet)
    _, ledger = pilot.verify(packet)
    entry = pilot.reserve(packet, ledger, "parse", "D0007", 0.00125, "x")
    entry.update(status="retrieval_failed", job_id="existing")
    calls = []

    def handler(request):
        calls.append(request)
        if request.method == "POST":
            raise httpx.ConnectTimeout("mock next-upload interruption")
        if request.url.path.endswith("/text"):
            return httpx.Response(200, json={"text": "fictitious recovered input"})
        return httpx.Response(200, json={"status": "SUCCESS"})

    parser_http(monkeypatch, handler)
    with pytest.raises(pilot.Halt):
        asyncio.run(pilot.parse_stage(packet, ledger))
    assert calls[0].url.path == "/api/parsing/job/existing"
    assert entry["status"] == "success"
    assert len(ledger["attempts"]) == 2
    assert ledger["attempts"][1]["status"] == "submission_uncertain"


def test_ollama_identity_changes_and_redirects_rejected():
    for response in [
        httpx.Response(200, json={"version": "changed"}),
        httpx.Response(302, headers={"Location": "https://example.com"}),
    ]:

        def handler(request, response=response):
            if request.url.path == "/api/version":
                return response
            return httpx.Response(200, json={"models": []})

        with httpx.Client(base_url="http://localhost", transport=httpx.MockTransport(handler)) as client:
            with pytest.raises((pilot.Halt, httpx.HTTPStatusError)):
                pilot.ollama_identity(client)


def qwen_http(monkeypatch, handler):
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setattr(pilot, "ollama_identity", lambda http: {"verified": "fictitious"})


def qwen_response(content=None, reason="stop"):
    return dict(
        model=pilot.MODEL,
        done=True,
        done_reason=reason,
        message=dict(role="assistant", content=json.dumps(target()) if content is None else content),
        load_duration=1000000000,
        eval_count=10,
        eval_duration=2000000000,
    )


def test_qwen_three_once_completed_failures_continue(packet, monkeypatch):
    ledger = reviewed(packet)
    generations = []

    def handler(request):
        if request.url.path == "/api/ps":
            return httpx.Response(
                200, json={"models": [dict(digest=pilot.DIGEST, context_length=32768, size_vram=100)]}
            )
        data = json.loads(request.read())
        assert "format" not in data
        assert data["options"] == pilot.OPTIONS
        generations.append(data)
        return httpx.Response(200, json=qwen_response("bad json" if len(generations) == 1 else None))

    qwen_http(monkeypatch, handler)
    pilot.qwen_stage(packet, ledger)
    entries = [a for a in ledger["attempts"] if a["stage"] == "qwen"]
    assert [a["status"] for a in entries] == ["output_failed", "success", "success"]
    assert entries[-1]["generation_tokens_per_second"] == 5
    pilot.qwen_stage(packet, ledger)
    assert len(generations) == 3
    assert (packet / "qwen/D0007/response.body").exists()


def test_qwen_timeout_partial_body_and_no_resume_retry(packet, monkeypatch):
    ledger = reviewed(packet)

    class Partial(httpx.SyncByteStream):
        def __iter__(self):
            yield b'{"partial":'
            raise httpx.ReadTimeout("fictitious timeout")

    calls = []

    def handler(request):
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        calls.append(request)
        return httpx.Response(200, stream=Partial())

    qwen_http(monkeypatch, handler)
    with pytest.raises(pilot.Halt, match="runtime/transport"):
        pilot.qwen_stage(packet, ledger)
    assert (packet / "qwen/D0007/response.body").read_bytes() == b'{"partial":'
    assert len([a for a in ledger["attempts"] if a["stage"] == "qwen"]) == 1
    with pytest.raises(pilot.Halt, match="no retry"):
        pilot.qwen_stage(packet, ledger)
    assert len(calls) == 1


def test_qwen_wrong_context_stops_after_raw_saved(packet, monkeypatch):
    ledger = reviewed(packet)

    def handler(request):
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [dict(digest=pilot.DIGEST, context_length=4096, size_vram=100)]})
        return httpx.Response(200, json=qwen_response())

    qwen_http(monkeypatch, handler)
    with pytest.raises(pilot.Halt):
        pilot.qwen_stage(packet, ledger)
    assert (packet / "qwen/D0007/content.txt").exists()
    assert not (packet / "qwen/D0007/prediction.json").exists()


def test_gemini_exact_settings_and_once_only(packet, monkeypatch):
    from types import SimpleNamespace

    from google import genai

    from configs.models import GoogleGeminiSettings

    ledger = reviewed(packet)
    ledger["attempts"] = [a for a in ledger["attempts"] if a["stage"] != "gemini"]
    ledger["reserved_usd"] = sum(a["reserved_usd"] for a in ledger["attempts"])
    # Delete only synthetic model artifacts to construct a pre-Gemini fixture.
    for path in list((packet / "gemini").rglob("*.json")) + list((packet / "gemini").rglob("*.txt")):
        ledger["artifact_hashes"].pop(str(path.relative_to(packet)), None)
        path.unlink()
    allow(packet)
    requests = []

    def dump(value):
        return SimpleNamespace(model_dump=lambda **kwargs: value)

    def generate(**kwargs):
        requests.append(kwargs)
        assert kwargs["model"] == pilot.GEMINI_MODEL
        config = kwargs["config"]
        assert config["temperature"] == 0 and config["max_output_tokens"] == 16384
        assert config["response_mime_type"] == "application/json"
        assert config["response_json_schema"] == pilot.read(packet / "schema.json")
        assert "tools" not in config
        return dump(
            dict(
                candidates=[dict(finish_reason="STOP", content=dict(parts=[dict(text=json.dumps(target()))]))],
                usage_metadata=dict(prompt_token_count=10, candidates_token_count=20),
            )
        )

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["http_options"].timeout == 600000
            assert kwargs["http_options"].retry_options.attempts == 1
            self.models = SimpleNamespace(
                get=lambda **kwargs: dump({"name": pilot.GEMINI_MODEL}), generate_content=generate
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(genai, "Client", Client)
    monkeypatch.setattr(
        "configs.models.GoogleGeminiSettings",
        lambda **kwargs: GoogleGeminiSettings(api_key=SecretStr("fake"), **kwargs),
    )
    # Simulated pacing advances elapsed wall clock; there is no real sleeping.
    clock = [1000.0]
    monkeypatch.setattr(pilot.time, "time", lambda: clock[0])
    monkeypatch.setattr(pilot.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    protocol = pilot.read(packet / "protocol.json")
    pilot.gemini_stage(packet, ledger, protocol)
    assert clock[0] >= 1020
    assert len(requests) == 3
    pilot.gemini_stage(packet, ledger, protocol)
    assert len(requests) == 3
    assert (packet / "gemini/D0007/response.json").exists()


def test_parser_quota_independent_of_pending_gemini_daily_limit(packet):
    allow(packet)
    review = pilot.read(packet / "access_review.json")
    review["gemini_available_requests"] = 0
    pilot.save(packet / "access_review.json", review, replace=True)
    _, ledger = pilot.verify(packet)
    pilot.access(packet, ledger, "parse")
    with pytest.raises(pilot.Halt, match="daily-request"):
        pilot.access(packet, ledger, "gemini")
    review["gemini_available_requests"] = 3
    pilot.save(packet / "access_review.json", review, replace=True)
    pilot.access(packet, ledger, "gemini")
    assert pilot.read(packet / "access_review.parse.frozen.json")["gemini_available_requests"] == 0


def test_clean_input_amendment_keeps_raw_and_binds_derivative(packet):
    ledger = reviewed(packet)
    amendment = dict(
        authorization="user_approved_clean_input_diagnostic",
        protocol_sha256=pilot.digest((packet / "protocol.json").read_bytes()),
        inputs={},
    )
    for identifier in pilot.SOURCES:
        raw = (packet / f"parse/{identifier}/input.txt").read_bytes()
        corrected = raw + b" CORRECTED"
        pilot.artifact(packet, ledger, f"corrected/{identifier}.txt", corrected)
        amendment["inputs"][identifier] = dict(raw_sha256=pilot.digest(raw), corrected_sha256=pilot.digest(corrected))
    pilot.artifact(packet, ledger, "clean_input_amendment.json", amendment)
    request, _ = pilot.qwen_request(packet, "D0007")
    assert "CORRECTED" in request["messages"][1]["content"]
    assert b"CORRECTED" not in (packet / "parse/D0007/input.txt").read_bytes()
    (packet / "corrected/D0007.txt").write_text("changed")
    with pytest.raises(pilot.Halt, match="Corrected input hash"):
        pilot.model_input(packet, "D0007")


def test_native_parameter_order_is_not_model_identity():
    original = 'temperature 0.7\ntop_k 20\ntop_p 0.8\nrepeat_penalty 1\nstop "<|im_start|>"\nstop "<|im_end|>"'
    assert pilot.native_parameters_match(original)
    assert pilot.native_parameters_match("\n".join(reversed(original.splitlines())))
    assert not pilot.native_parameters_match(original.replace("0.7", "0.8"))
    assert not pilot.native_parameters_match(original.replace("<|im_end|>", "different"))
    assert not pilot.native_parameters_match(original + '\nstop "extra"')
