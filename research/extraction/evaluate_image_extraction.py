"""Bounded three-image pilot. Review artifacts, not model outputs, open stage gates."""

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import platform
import shlex
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from backend.utils.extraction import parse_and_validate_extraction
from backend.utils.prompt_engine import PromptEngine
from configs.enums import PromptTypeEnum
from research.extraction.evaluate_local_extraction import (
    DIGEST,
    MODEL,
    OPTIONS,
    PROMPT_HASH,
    TEMPLATE_HASH,
)
from research.extraction.scoring import compare, summarize

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "D0007": ("CMD", "D0007-v003-f0a7fb8f1955", "f0a7fb8f1955918606f8cbb0b649dd9541c476be7b9b875ae1e1f1c77236ba4b"),
    "D0002": ("Lab4U", "D0002-v004-5748327bd960", "5748327bd9607f4d262332eadc528d2a5abf5153e07fda29afbcb684e5fd096d"),
    "D0016": ("UniLab", "D0016-v002-7255547b2ed8", "7255547b2ed8af384d5e50ce338b5d45e85279a18b734c487af440b23bdec061"),
}
PARSER_SETTINGS = dict(
    result_type="text",
    language="ru",
    parse_mode="parse_page_without_llm",
    disable_image_extraction=True,
    max_timeout=2000,
    ignore_errors=False,
    check_for_updates=False,
)
GEMINI_MODEL = "gemini-3.1-flash-lite"
OLLAMA_VERSION = "0.34.3"
STAGES = ("parse", "gemini", "qwen")


class Halt(RuntimeError):
    pass


def now():
    return datetime.now(UTC).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def save(path, data, *, replace=False):
    """Publish durable artifacts; immutable by default, atomic replacement for the ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = data if isinstance(data, bytes) else encoded(data)
    if not replace and path.exists():
        if path.read_bytes() != raw:
            raise Halt(f"Refusing to overwrite {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def strict_validate(content, schema):
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    json.loads(content)  # Reject fences/prose before the shared application validator.
    return parse_and_validate_extraction(content, schema)


def initialize(directory, source_dir=None):
    from PIL import Image

    source_dir = source_dir or ROOT / "data/approved_image_redactions"
    inputs = {}
    for identifier, (organization, stem, expected) in SOURCES.items():
        image, approval = source_dir / (stem + ".png"), source_dir / (stem + ".json")
        manifest = read(approval)
        if (manifest["id"], manifest["approval_status"], manifest["export_sha256"]) != (
            identifier,
            "kept",
            expected,
        ) or digest(image.read_bytes()) != expected:
            raise Halt(f"Keep approval/hash mismatch: {identifier}")
        with Image.open(image) as im:
            if im.format != "PNG" or im.mode != "RGB" or im.n_frames != 1 or im.info:
                raise Halt(f"Unexpected image format/metadata: {identifier}")
        inputs[identifier] = dict(organization=organization, image=image.read_bytes(), approval=approval.read_bytes())
    prompt = (ROOT / "backend/prompts/laboratory_test_processor/v4.yaml").read_bytes()
    if digest(prompt) != PROMPT_HASH:
        raise Halt("Frozen v4.1 prompt changed")
    directory.mkdir(parents=True, exist_ok=False)
    artifacts = {"prompts/laboratory_test_processor/v4.yaml": prompt}
    _, system, _ = PromptEngine(ROOT / "backend/prompts").render(
        PromptTypeEnum.laboratory_test_processor, "v4", input_text=""
    )
    artifacts["schema.json"] = encoded(system["response_json_schema"])
    for identifier, item in inputs.items():
        artifacts[f"inputs/{identifier}.png"] = item["image"]
        artifacts[f"inputs/{identifier}.approval.json"] = item["approval"]
    for name, raw in artifacts.items():
        save(directory / name, raw)
    protocol = dict(
        created=now(),
        documents=list(SOURCES),
        organizations={k: v[0] for k, v in SOURCES.items()},
        organization_nonoverlap="User confirmed all three brands are new; B remains reserved.",
        source_duplicate_groups="Unknown; no split assignment",
        approved_scope="Only supplied image pages",
        hashes={name: digest(raw) for name, raw in artifacts.items()},
        parser=PARSER_SETTINGS,
        parser_sdk="0.6.94",
        google_sdk="1.75.0",
        ocr_enabled=True,
        gemini=dict(
            model=GEMINI_MODEL,
            temperature=0,
            max_output_tokens=16384,
            response_mime_type="application/json",
            response_json_schema=system["response_json_schema"],
        ),
        qwen=dict(
            model=MODEL,
            digest=DIGEST,
            version=OLLAMA_VERSION,
            options=OPTIONS,
            think=False,
            stream=False,
            keep_alive="10m",
            format="omitted",
        ),
        limits=dict(parse=3, gemini=3, qwen=3, total_usd=1),
        prices=dict(parser_page_usd=0.00125, gemini_input_per_million=0.25, gemini_output_per_million=1.50),
        price_sources=[
            "https://developers.llamaindex.ai/llamaparse/general/pricing/",
            "https://ai.google.dev/gemini-api/docs/pricing",
        ],
        scoring="Shared exact compare/summarize; no normalization; all three documents in denominator; warnings separate",
    )
    save(directory / "protocol.json", protocol)
    save(directory / "protocol.sha256", (digest(encoded(protocol)) + "\n").encode())
    save(directory / "ledger.json", {"attempts": [], "artifact_hashes": {}, "reserved_usd": 0})
    save(
        directory / "access_review.json",
        dict(
            status="pending",
            reviewer="",
            evidence="",
            checked_at="",
            parser_available_pages=0,
            gemini_available_requests=0,
            gemini_rpm=0,
            gemini_tpm=0,
        ),
    )


def verify(directory):
    protocol = read(directory / "protocol.json")
    if digest((directory / "protocol.json").read_bytes()) != (directory / "protocol.sha256").read_text().strip():
        raise Halt("Protocol hash mismatch")
    if protocol["documents"] != list(SOURCES) or protocol["limits"] != dict(parse=3, gemini=3, qwen=3, total_usd=1):
        raise Halt("Unexpected scope/allowance")
    ledger = read(directory / "ledger.json")
    for name, expected in (protocol["hashes"] | ledger["artifact_hashes"]).items():
        if digest((directory / name).read_bytes()) != expected:
            raise Halt(f"Artifact hash mismatch: {name}")
    for stage in STAGES:
        ids = [a["id"] for a in ledger["attempts"] if a["stage"] == stage]
        if len(ids) != len(set(ids)) or set(ids) - set(SOURCES) or len(ids) > 3:
            raise Halt("Invalid attempt inventory")
    if abs(sum(a["reserved_usd"] for a in ledger["attempts"]) - ledger["reserved_usd"]) > 1e-9:
        raise Halt("Cost ledger inconsistent")
    if ledger["reserved_usd"] > 1:
        raise Halt("Budget exceeded")
    return protocol, ledger


def artifact(directory, ledger, name, data):
    save(directory / name, data)
    ledger["artifact_hashes"][name] = digest((directory / name).read_bytes())
    save(directory / "ledger.json", ledger, replace=True)


def reserve(directory, ledger, stage, identifier, cost, input_hash):
    if stage not in STAGES or identifier not in SOURCES or cost < 0:
        raise Halt("Unauthorized attempt")
    if any(a["stage"] == stage and a["id"] == identifier for a in ledger["attempts"]):
        raise Halt("Existing/uncertain attempt cannot be repeated")
    if ledger["reserved_usd"] + cost > 1:
        raise Halt("US$1 ceiling would be exceeded")
    entry = dict(
        stage=stage,
        id=identifier,
        status="attempted",
        started=now(),
        started_epoch=time.time(),
        reserved_usd=cost,
        input_sha256=input_hash,
        host=platform.system(),
    )
    ledger["attempts"].append(entry)
    ledger["reserved_usd"] += cost
    save(directory / "ledger.json", ledger, replace=True)
    return entry


def access(directory, ledger, stage, token_bound=0):
    review = read(directory / "access_review.json")
    if review["status"] != "verified" or not all(review.get(k) for k in ("reviewer", "evidence", "checked_at")):
        raise Halt("Account access/quota verification pending: access_review.json")
    if stage == "parse" and review["parser_available_pages"] < 3:
        raise Halt("Insufficient verified parser allowance")
    if stage == "gemini" and review["gemini_available_requests"] < 3:
        raise Halt("Gemini daily-request allowance verification pending")
    if stage == "gemini" and (review["gemini_rpm"] <= 0 or review["gemini_tpm"] < token_bound):
        raise Halt("Gemini request exceeds verified token/rate quota")
    artifact(directory, ledger, f"access_review.{stage}.frozen.json", review)
    return review


def model_input(directory, identifier):
    """Use a hash-bound, approved derivative only under an explicit protocol amendment."""
    if identifier not in SOURCES:
        raise Halt("Unauthorized document")
    amendment_path = directory / "clean_input_amendment.json"
    if not amendment_path.exists():
        return directory / f"parse/{identifier}/input.txt"
    amendment = read(amendment_path)
    if amendment.get("authorization") != "user_approved_clean_input_diagnostic" or set(amendment["inputs"]) != set(
        SOURCES
    ):
        raise Halt("Invalid clean-input amendment")
    if amendment["protocol_sha256"] != digest((directory / "protocol.json").read_bytes()):
        raise Halt("Amendment belongs to a different protocol")
    path = directory / f"corrected/{identifier}.txt"
    if digest(path.read_bytes()) != amendment["inputs"][identifier]["corrected_sha256"]:
        raise Halt("Corrected input hash mismatch")
    if (
        digest((directory / f"parse/{identifier}/input.txt").read_bytes())
        != amendment["inputs"][identifier]["raw_sha256"]
    ):
        raise Halt("Original parser input hash mismatch")
    return path


def messages(directory, identifier):
    text = model_input(directory, identifier).read_text()
    user, system, _ = PromptEngine(directory / "prompts").render(
        PromptTypeEnum.laboratory_test_processor, "v4", input_text=text
    )
    return [{"role": "system", "content": system["system_instruction"]}, {"role": "user", "content": user}]


def qwen_request(directory, identifier):
    msgs = messages(directory, identifier)
    bound = sum(len(m["content"].encode()) for m in msgs) + 1024
    if bound + OPTIONS["num_predict"] > OPTIONS["num_ctx"]:
        raise Halt(f"Conservative UTF-8 token bound does not fit: {identifier}")
    return dict(model=MODEL, messages=msgs, options=OPTIONS.copy(), think=False, stream=False, keep_alive="10m"), bound


def source_gate(directory, ledger):
    if any(
        not any(a["stage"] == "parse" and a["id"] == i and a["status"] == "success" for a in ledger["attempts"])
        for i in SOURCES
    ):
        raise Halt("All three parser results must succeed before source review")
    review_name = (
        "source_review.corrected.json" if (directory / "clean_input_amendment.json").exists() else "source_review.json"
    )
    review = read(directory / review_name)
    if set(review) != set(SOURCES):
        raise Halt("Source review must cover all three pages")
    for identifier, row in review.items():
        if row.get("status") != "pass" or not row.get("reviewer") or not row.get("observations"):
            raise Halt(f"Source/contract review required: {identifier}")
        for key, name in (
            ("input_sha256", str(model_input(directory, identifier).relative_to(directory))),
            ("image_sha256", f"inputs/{identifier}.png"),
        ):
            if row.get(key) != digest((directory / name).read_bytes()):
                raise Halt("Source review hash mismatch")
    artifact(directory, ledger, "source_review.frozen.json", review)


def freeze_targets(directory, ledger):
    source_gate(directory, ledger)
    if any(
        not any(
            a["stage"] == "gemini" and a["id"] == i and a["status"] in ("success", "output_failed")
            for a in ledger["attempts"]
        )
        for i in SOURCES
    ):
        raise Halt("Complete Gemini attempts before freezing targets")
    schema = read(directory / "schema.json")
    review = read(directory / "target_review.json")
    if set(review) != set(SOURCES):
        raise Halt("Target review must cover all three pages")
    for identifier, item in review.items():
        raw = (directory / f"targets/{identifier}.json").read_bytes()
        strict_validate(raw, schema)
        if item.get("status") != "pass" or not item.get("reviewer") or not item.get("evidence"):
            raise Halt("Target review incomplete")
        if not isinstance(item.get("corrections"), list):
            raise Halt("Correction history required (empty list if none)")
        if item.get("target_sha256") != digest(raw) or item.get("input_sha256") != digest(
            model_input(directory, identifier).read_bytes()
        ):
            raise Halt("Target review hash mismatch")
        original = directory / f"gemini/{identifier}/response.json"
        expected = digest(original.read_bytes()) if original.exists() else None
        if item.get("gemini_response_sha256") != expected:
            raise Halt("Target review must bind the original Gemini response/failure")
        artifact(directory, ledger, f"targets/{identifier}.json", raw)
    artifact(directory, ledger, "target_review.frozen.json", review)


async def parse_stage(directory, ledger):
    import llama_cloud_services.parse.base as parser_module
    from llama_parse import LlamaParse

    from configs.models import LlamaParseSettings

    for package in ("llama-parse", "llama-cloud-services"):
        if version(package) != "0.6.94":
            raise Halt(f"Unexpected {package} version; host must match 0.6.94")
    access(directory, ledger, "parse")
    key = LlamaParseSettings().api_key
    if not key:
        raise Halt("Missing LLAMAPARSE_API_KEY")
    client = LlamaParse(api_key=key.get_secret_value(), **PARSER_SETTINGS)
    if client.disable_ocr or client.fast_mode or client.max_pages is not None:
        raise Halt("Unexpected effective OCR configuration")

    # Reuse SDK multipart construction while removing its hidden upload retries.
    async def one_request(http, method, url, timeout=60, **kwargs):
        response = await http.request(method, url, timeout=timeout, **kwargs)
        response.raise_for_status()
        return response

    async with client.aclient as http:
        for identifier in SOURCES:
            entry = next((a for a in ledger["attempts"] if a["stage"] == "parse" and a["id"] == identifier), None)
            if entry and entry["status"] == "success":
                continue
            if entry and (not entry.get("job_id") or entry["status"] == "service_failed"):
                raise Halt("Existing submission cannot be repeated; reconcile status before continuation")
            prefix = f"parse/{identifier}"
            sequence = len(list((directory / prefix).glob("http-*.body")))

            async def capture(response, prefix=prefix):
                nonlocal sequence
                sequence += 1
                body = bytearray()
                try:
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                finally:
                    artifact(directory, ledger, f"{prefix}/http-{sequence:04d}.body", bytes(body))
                response._content = bytes(body)  # Preserve SDK access after streaming the response hook.
                artifact(
                    directory,
                    ledger,
                    f"{prefix}/http-{sequence:04d}.json",
                    dict(
                        status=response.status_code,
                        method=response.request.method,
                        path=response.request.url.path,
                        timestamp=now(),
                        content_type=response.headers.get("content-type"),
                    ),
                )

            http.event_hooks = {"response": [capture]}
            start = time.monotonic()
            try:
                if entry is None:
                    entry = reserve(
                        directory,
                        ledger,
                        "parse",
                        identifier,
                        0.00125,
                        digest((directory / f"inputs/{identifier}.png").read_bytes()),
                    )
                    with patch.object(parser_module, "make_api_request", one_request):
                        job = await client._create_job(str(directory / f"inputs/{identifier}.png"))
                    entry.update(job_id=job, status="submitted")
                    save(directory / "ledger.json", ledger, replace=True)
                # Polling repeats status reads, never uploads, and fails on the first HTTP error.
                deadline = time.monotonic() + 2000
                while True:
                    response = await http.get(
                        f"/api/parsing/job/{entry['job_id']}", timeout=max(1, deadline - time.monotonic())
                    )
                    response.raise_for_status()
                    status = response.json()["status"]
                    if status == "SUCCESS":
                        break
                    if status != "PENDING":
                        entry["status"] = "service_failed"
                        raise Halt(f"Parser job ended with status {status}")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("2000-second parser deadline exceeded")
                    await asyncio.sleep(min(5, max(0, deadline - time.monotonic())))
                response = await http.get(f"/api/parsing/job/{entry['job_id']}/result/text")
                response.raise_for_status()
                result = response.json()
                artifact(directory, ledger, f"{prefix}/result.json", result)
                text = result["text"]
                if not isinstance(text, str) or not text.strip():
                    entry["status"] = "service_failed"
                    raise Halt("Parser returned empty text")
                # Same split/join as the application's aload_data path.
                separator = client.page_separator or parser_module._DEFAULT_SEPARATOR
                text = "\n".join(text.split(separator)) if client.split_by_page else text
                artifact(directory, ledger, f"{prefix}/input.txt", text.encode())
                entry.update(
                    status="success",
                    finished=now(),
                    wall_seconds=time.monotonic() - start,
                    cost_usd_upper_bound=0.00125,
                    actual_charge_usd=None,
                )
                save(directory / "ledger.json", ledger, replace=True)
            except BaseException as exc:
                if entry is not None:
                    if entry["status"] != "service_failed":
                        entry["status"] = "retrieval_failed" if entry.get("job_id") else "submission_uncertain"
                    entry.update(
                        error_type=type(exc).__name__,
                        error=str(exc).replace(key.get_secret_value(), "[REDACTED]"),
                        finished=now(),
                        wall_seconds=time.monotonic() - start,
                    )
                    save(directory / "ledger.json", ledger, replace=True)
                # Error bodies are already retained by the response hook; no secrets in diagnostic.
                raise Halt(
                    f"LlamaParse failed ({type(exc).__name__}); stop live execution and diagnose saved evidence"
                ) from exc
    if not (directory / "source_review.json").exists():
        save(
            directory / "source_review.json",
            {
                i: dict(
                    status="pending",
                    reviewer="",
                    observations=[],
                    issues=[],
                    input_sha256=digest((directory / f"parse/{i}/input.txt").read_bytes()),
                    image_sha256=digest((directory / f"inputs/{i}.png").read_bytes()),
                )
                for i in SOURCES
            },
        )


def finish_prediction(directory, ledger, entry, content, complete):
    prefix = f"{entry['stage']}/{entry['id']}"
    artifact(directory, ledger, f"{prefix}/content.txt", content.encode())
    try:
        if not complete:
            raise ValueError("Incomplete response, tool call, or unexpected finish reason")
        prediction = strict_validate(content, read(directory / "schema.json"))
        artifact(directory, ledger, f"{prefix}/prediction.json", prediction)
        entry["status"] = "success"
    except (ValueError, TypeError) as exc:
        entry.update(status="output_failed", validation_error=str(exc))
    entry["finished"] = now()
    save(directory / "ledger.json", ledger, replace=True)


def gemini_stage(directory, ledger, protocol):
    from google import genai
    from google.genai import types

    from configs.models import GoogleGeminiSettings

    source_gate(directory, ledger)
    if version("google-genai") != protocol["google_sdk"]:
        raise Halt("Unexpected google-genai version")
    key = GoogleGeminiSettings(model_name=GEMINI_MODEL).api_key
    if not key:
        raise Halt("Missing GEMINI_API_KEY")
    with genai.Client(
        api_key=key.get_secret_value(),
        http_options=types.HttpOptions(timeout=600000, retry_options=types.HttpRetryOptions(attempts=1)),
    ) as client:
        # Metadata only; no content leaves the machine until quota and review gates pass.
        access(directory, ledger, "gemini")
        metadata = client.models.get(model=GEMINI_MODEL)
        artifact(directory, ledger, "gemini/model.json", metadata.model_dump(mode="json"))
        for identifier in SOURCES:
            old = next((a for a in ledger["attempts"] if a["stage"] == "gemini" and a["id"] == identifier), None)
            if old:
                if old["status"] not in ("success", "output_failed"):
                    raise Halt("Uncertain Gemini submission; no retry")
                continue
            msgs = messages(directory, identifier)
            bound = sum(len(m["content"].encode()) for m in msgs) + len(encoded(protocol["gemini"])) + 1024
            allowance = access(directory, ledger, "gemini", bound + 16384)
            previous = [a for a in ledger["attempts"] if a["stage"] == "gemini"]
            if previous:
                delay = max(10, 60 / allowance["gemini_rpm"], 60 * (bound + 16384) / allowance["gemini_tpm"])
                remaining = delay - (time.time() - previous[-1]["started_epoch"])
                if remaining > 0:
                    while remaining > 0:
                        time.sleep(min(remaining, 30))
                        remaining = delay - (time.time() - previous[-1]["started_epoch"])
            config = {k: v for k, v in protocol["gemini"].items() if k != "model"}
            config["system_instruction"] = msgs[0]["content"]
            request = dict(model=GEMINI_MODEL, contents=msgs[1]["content"], config=config)
            artifact(directory, ledger, f"gemini/{identifier}/request.json", request)
            cost = (bound * 0.25 + 16384 * 1.5) / 1_000_000
            entry = reserve(
                directory,
                ledger,
                "gemini",
                identifier,
                cost,
                digest(model_input(directory, identifier).read_bytes()),
            )
            start = time.monotonic()
            try:
                response = client.models.generate_content(**request)
                raw = response.model_dump(mode="json")
                artifact(directory, ledger, f"gemini/{identifier}/response.json", raw)
                entry.update(
                    wall_seconds=time.monotonic() - start,
                    usage=raw.get("usage_metadata"),
                    completion_reasons=[c.get("finish_reason") for c in raw.get("candidates", [])],
                )
                usage = raw.get("usage_metadata") or {}
                entry["estimated_list_cost_usd"] = (
                    usage.get("prompt_token_count", 0) * 0.25
                    + ((usage.get("candidates_token_count") or 0) + (usage.get("thoughts_token_count") or 0)) * 1.5
                ) / 1_000_000
                entry["actual_charge_usd"] = None  # Account credit application is not in the SDK response.
                candidates = raw.get("candidates") or []
                parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
                complete = len(candidates) == 1 and candidates[0].get("finish_reason") == "STOP"
                complete = complete and not any(p.get("function_call") for p in parts)
                content = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                finish_prediction(directory, ledger, entry, content, complete)
            except BaseException as exc:
                entry.update(status="runtime_failed", error_type=type(exc).__name__, finished=now())
                body = getattr(exc, "response_json", None)
                if body is not None:
                    artifact(directory, ledger, f"gemini/{identifier}/error.json", body)
                save(directory / "ledger.json", ledger, replace=True)
                raise Halt("Gemini runtime/transport failure; no further requests") from exc


def native_parameters_match(text):
    # Ollama can reorder the displayed parameter lines after a restart.
    expected = [
        ["temperature", "0.7"],
        ["top_k", "20"],
        ["top_p", "0.8"],
        ["repeat_penalty", "1"],
        ["stop", "<|im_start|>"],
        ["stop", "<|im_end|>"],
    ]
    return sorted(shlex.split(line) for line in text.splitlines() if line.strip()) == sorted(expected)


def ollama_identity(http):
    ver = http.get("/api/version")
    ver.raise_for_status()
    tags = http.get("/api/tags")
    tags.raise_for_status()
    models = [m for m in tags.json()["models"] if m["name"] == MODEL]
    if ver.json().get("version") != OLLAMA_VERSION or len(models) != 1 or models[0]["digest"] != DIGEST:
        raise Halt("Unexpected Ollama/model identity")
    show = http.post("/api/show", json={"model": MODEL})
    show.raise_for_status()
    data = show.json()
    if data.get("remote_host") or data.get("remote_model") or data["details"].get("quantization_level") != "Q8_0":
        raise Halt("Cloud or unexpected quantization")
    if digest(data["template"].encode()) != TEMPLATE_HASH or not native_parameters_match(data["parameters"]):
        raise Halt("Unexpected template/model parameters")
    return dict(version=ver.json(), model=models[0], show=data)


def allocation(http):
    response = http.get("/api/ps")
    response.raise_for_status()
    data = response.json()
    return dict(
        timestamp=now(), kind="allocation snapshot, not peak memory", host_memory_pressure="unavailable", data=data
    )


def check_allocation(snapshot):
    models = [m for m in snapshot["data"]["models"] if m.get("digest") == DIGEST]
    if len(models) != 1 or models[0].get("context_length") != 32768 or not models[0].get("size_vram"):
        raise Halt("Unexpected effective context/GPU allocation")


def qwen_stage(directory, ledger):
    import httpx

    freeze_targets(directory, ledger)
    endpoint = "http://127.0.0.1:11434" if platform.system() == "Darwin" else "http://host.orb.internal:11434"
    with httpx.Client(base_url=endpoint, timeout=600, follow_redirects=False, trust_env=False) as http:
        identity = ollama_identity(http)
        artifact(directory, ledger, "qwen/identity.json", identity)
        for identifier in SOURCES:
            old = next((a for a in ledger["attempts"] if a["stage"] == "qwen" and a["id"] == identifier), None)
            if old:
                if old["status"] not in ("success", "output_failed"):
                    raise Halt("Uncertain Qwen attempt; no retry")
                continue
            ollama_identity(http)
            request, bound = qwen_request(directory, identifier)
            prefix = f"qwen/{identifier}"
            artifact(directory, ledger, f"{prefix}/before.json", allocation(http))
            artifact(directory, ledger, f"{prefix}/request.json", request)
            entry = reserve(
                directory,
                ledger,
                "qwen",
                identifier,
                0,
                digest(model_input(directory, identifier).read_bytes()),
            )
            entry["input_token_upper_bound"] = bound
            entry["endpoint"] = endpoint
            save(directory / "ledger.json", ledger, replace=True)
            start, body = time.monotonic(), bytearray()
            try:
                with http.stream("POST", "/api/chat", json=request) as response:
                    try:
                        for chunk in response.iter_bytes():
                            body.extend(chunk)
                    finally:
                        artifact(directory, ledger, f"{prefix}/response.body", bytes(body))
                    response.raise_for_status()
                try:
                    raw = json.loads(body)
                    if not isinstance(raw, dict):
                        raise ValueError("Ollama envelope is not an object")
                except ValueError:
                    snapshot = allocation(http)
                    artifact(directory, ledger, f"{prefix}/after.json", snapshot)
                    check_allocation(snapshot)
                    finish_prediction(directory, ledger, entry, bytes(body).decode("utf-8", errors="replace"), False)
                    continue
                artifact(directory, ledger, f"{prefix}/response.json", raw)
                entry.update(
                    wall_seconds=time.monotonic() - start,
                    metrics={k: v for k, v in raw.items() if k.endswith("duration") or k.endswith("count")},
                    completion_reason=raw.get("done_reason"),
                )
                durations = entry["metrics"]
                entry["loading_seconds"] = durations.get("load_duration", 0) / 1e9
                entry["generation_tokens_per_second"] = (
                    durations.get("eval_count", 0) * 1e9 / durations["eval_duration"]
                    if durations.get("eval_duration")
                    else None
                )
                message = raw.get("message") or {}
                content = message.get("content", "")
                artifact(directory, ledger, f"{prefix}/content.txt", content.encode())
                snapshot = allocation(http)
                artifact(directory, ledger, f"{prefix}/after.json", snapshot)
                check_allocation(snapshot)
                complete = (
                    raw.get("done") is True and raw.get("done_reason") == "stop" and not message.get("tool_calls")
                )
                finish_prediction(directory, ledger, entry, content, complete)
            except BaseException as exc:
                entry.update(status="runtime_failed", error_type=type(exc).__name__, finished=now())
                save(directory / "ledger.json", ledger, replace=True)
                raise Halt("Qwen runtime/transport failure; remaining reports unattempted") from exc


def score_stage(directory, ledger):
    freeze_targets(directory, ledger)
    reviews = read(directory / "warning_review.json") if (directory / "warning_review.json").exists() else {}
    scores, lines = (
        {},
        ["# Three-image pilot comparison", "", "Exact strings; three-document denominator for each model.", ""],
    )
    for model in ("gemini", "qwen"):
        rows = {}
        for identifier in SOURCES:
            path = directory / f"{model}/{identifier}/prediction.json"
            entry = next((a for a in ledger["attempts"] if (a["stage"], a["id"]) == (model, identifier)), None)
            prediction = (
                strict_validate(path.read_bytes(), read(directory / "schema.json"))
                if (entry and entry["status"] == "success")
                else None
            )
            row = compare(read(directory / f"targets/{identifier}.json"), prediction)
            row["runtime"] = entry or {"status": "unattempted"}
            review = reviews.get(model, {}).get(identifier)
            if review:
                raw = directory / f"{model}/{identifier}/response.json"
                if prediction is None or review.get("response_sha256") != digest(raw.read_bytes()):
                    raise Halt("Warning review response mismatch")
                if (
                    review.get("status") not in ("pass", "fail")
                    or not review.get("reason")
                    or not review.get("reviewer")
                ):
                    raise Halt("Incomplete warning adjudication")
                row["warning_status"] = review["status"]
                row["strict_pass"] = row["date_exact"] and row["ordered_results_exact"] and review["status"] == "pass"
            rows[identifier] = row
            lines += [
                f"## {model} / {identifier} ({SOURCES[identifier][0]})",
                "",
                f"Status: {row['runtime']['status']}; date: {row['date_exact']}; "
                f"records: {row['ordered_record_matches']}/{row['expected_rows']}; warnings: {row['warning_status']}.",
                "",
                "Evidence: frozen input, source review and target correction history.",
                "",
                "```json",
                json.dumps(row["differences"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        scores[model] = dict(
            documents=rows,
            pooled=summarize(list(rows.values())),
            organizations={SOURCES[i][0]: summarize([row]) for i, row in rows.items()},
        )
    save(directory / "scores.json", scores, replace=True)
    save(directory / "discrepancies.md", "\n".join(lines).encode(), replace=True)
    if (directory / "assessment.md").exists():
        return scores
    save(
        directory / "assessment.md",
        (
            b"# Assessment status\n\nOffline scores and all discrepancies are available. Assistant adjudication is required "
            b"to distinguish parser losses, extraction mistakes, contract ambiguities and runtime failures, "
            b"and to assess critical value/unit/inequality errors.\n\n"
            b"No training/deployment conclusion is automatic. B was neither generated nor scored.\n"
        ),
    )
    return scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=(*STAGES, "score"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    directory = args.experiment_dir.resolve()
    if not directory.exists():
        if args.resume or args.stage != "parse":
            raise Halt("Initialize with --stage parse without --resume")
        initialize(directory)
    elif not args.resume:
        raise Halt("Existing experiment requires --resume; never create a replacement ledger")
    with (directory / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol, ledger = verify(directory)
        if args.stage == "parse":
            asyncio.run(parse_stage(directory, ledger))
        elif args.stage == "gemini":
            gemini_stage(directory, ledger, protocol)
        elif args.stage == "qwen":
            qwen_stage(directory, ledger)
        else:
            score_stage(directory, ledger)
        if args.stage in STAGES:
            completed = [a for a in ledger["attempts"] if a["stage"] == args.stage and a["status"] == "success"]
            return 0 if len(completed) == 3 else 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Halt, FileNotFoundError, BlockingIOError) as error:
        print(f"STOP: {error}")
        raise SystemExit(2) from None
