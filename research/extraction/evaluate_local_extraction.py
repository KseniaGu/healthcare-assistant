"""Frozen, local-only Qwen extraction experiment; no application or Gemini calls."""

import argparse
import copy
import fcntl
import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from backend.utils.extraction import parse_and_validate_extraction
from backend.utils.prompt_engine import PromptEngine
from configs.enums import PromptTypeEnum
from research.extraction.scoring import FIELDS, compare, summarize

ROOT = Path(__file__).resolve().parents[2]
MODEL = "qwen3:4b-instruct-2507-q8_0"
DIGEST = "aa7252f68dda4d25dfffa65b3760af6d2c3231a140c0060c78d444686d98a374"
VERSION = "0.33.2"
ENDPOINT = "http://host.orb.internal:11434"
PROMPT_HASH = "5016ee3952d7198a79b5cb50c41da8132d14cd7ca368f6487fb7f7df13b2818b"
TEMPLATE_HASH = "40c21f34cf67d8c760ef72f8ad3ae5afad514299d4b06e91dd9a8d705af7b541"
PARAMETERS_HASH = "2454acecd583100a159248b0f6b45de8cf7031a230cdb98be4feb81c5939aa46"
DEVELOPMENT = ("A1", "A2", "A3", "A4", "C1")
COUNTS = {"A1": 20, "A2": 32, "A3": 45, "A4": 2, "B1": 6, "B2": 16, "B3": 21, "B4": 17, "C1": 24}
OPTIONS = dict(temperature=0, seed=42, repeat_penalty=1, top_k=20, top_p=0.8, min_p=0, num_ctx=32768, num_predict=16384)
QWEN35 = {
    "qwen35-4b": {
        "model": "qwen3.5:4b-mlx",
        "digest": "61aa3858e9d3022e8fca725550089addd9289c4446f33bd09dfd12f95f2a6792",
        "quantization": "nvfp4",
        "parameters_sha256": "124e6770a2c900e2803d80477676a7fa6b6e3fabd744bfcef60f5b0f46779420",
    },
    "qwen35-9b": {
        "model": "qwen3.5:9b",
        "digest": "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7",
        "quantization": "Q4_K_M",
        "parameters_sha256": "6290a4f3b88445e37918439f7730e11dfb1e5f5012138755a4c705a4aebe3185",
    },
}
PROFILES = ("qwen3", "qwen35-4b", "qwen35-9b", "qwen35-4b-compat", "qwen35-4b-thinking", "qwen35-9b-thinking")


def configuration(profile="qwen3"):
    if profile not in PROFILES:
        raise ValueError("Unknown authorized experiment profile.")
    config = dict(
        model=MODEL,
        digest=DIGEST,
        quantization="Q8_0",
        template_sha256=TEMPLATE_HASH,
        parameters_sha256=PARAMETERS_HASH,
        options=OPTIONS.copy(),
        think=False,
    )
    if profile != "qwen3":
        config.update(QWEN35[profile.removesuffix("-thinking").removesuffix("-compat")])
        config["template_sha256"] = "b507b9c2f6ca642bffcd06665ea7c91f235fd32daeefdf875a0f938db05fb315"
        config["options"]["presence_penalty"] = 0
        config["think"] = profile.endswith("-thinking")
        if config["think"]:
            config["options"].update(temperature=1.0, top_p=0.95, presence_penalty=1.5)
    return config


RULES = {
    "version": 1,
    "comparison": "Parsed JSON; exact strings/nulls, no numeric, Unicode or field-whitespace normalization.",
    "fields": list(FIELDS),
    "position_alignment": "Expected row position; missing positions receive zero field/record credit.",
    "secondary": "Multiset exact-record precision/recall; duplicates count; no fuzzy alignment.",
    "failures": "Failed/unattempted reports receive zero credit and remain in all expected denominators.",
    "warnings": "Real development targets require []; unexpected warnings require review.",
    "smoke_warning_checklist": [
        "One concise warning identifies two unnamed omitted rows.",
        "Includes visible values 1.23 10*9/л and 27.5 % and incomplete references.",
        "No inferred cause or warnings about pending tests, cover referrals or other routine exclusions.",
    ],
    "holdout": "B is reserved from subsequent model adaptation, not blind to prior contract development.",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def now() -> str:
    return datetime.now(UTC).isoformat()


def schema_at(directory: Path):
    return PromptEngine(directory).render(PromptTypeEnum.laboratory_test_processor, "v4", input_text="")[1][
        "response_json_schema"
    ]


def freeze(output: Path, root: Path = ROOT):
    """Snapshot reviewed labels, preserving the original review packet and decisions."""
    if output.exists():
        raise FileExistsError(output)
    review_path = root / "data/review/structured_extraction_labels.proposed.json"
    evaluation_path = root / "data/review/structured_extraction_v4_real_results.json"
    review, evaluation = read_json(review_path), read_json(evaluation_path)
    decisions = review["user_decisions"]
    if {d["id"] for d in decisions if d["status"] == "approved"} != {f"D{i}" for i in range(1, 7)}:
        raise ValueError("All D1-D6 approvals are required.")
    for name, digest in review["source_sha256"].items():
        if sha((root / name).read_bytes()) != digest:
            raise ValueError(f"Source hash mismatch: {name}")
    prompt = (root / "backend/prompts/laboratory_test_processor/v4.yaml").read_bytes()
    if sha(prompt) != PROMPT_HASH:
        raise ValueError("Frozen v4.1 prompt changed.")
    schema = schema_at(root / "backend/prompts")
    blobs = {
        "prompts/laboratory_test_processor/v4.yaml": prompt,
        "schema.json": json_bytes(schema),
        "scoring.json": json_bytes(RULES),
        "review.original.json": review_path.read_bytes(),
        "gemini-review.original.json": evaluation_path.read_bytes(),
    }
    documents, seen = [], set()
    for document in review["documents"]:
        identifier = document["document_id"]
        target = document["proposed_target"]
        parse_and_validate_extraction(json.dumps(target), schema)
        if identifier not in COUNTS or len(target["results"]) != COUNTS[identifier]:
            raise ValueError(f"Unexpected document/row count: {identifier}")
        if document["organization_alias"] != identifier[0]:
            raise ValueError(f"Organization mismatch: {identifier}")
        source = root / document["sources"]["parsed_text"]
        content = source.read_bytes()
        if not content.decode("utf-8").strip() or sha(content) in seen:
            raise ValueError(f"Empty or duplicate parser input: {identifier}")
        seen.add(sha(content))
        entry = copy.deepcopy(document)
        entry.update(
            split="development" if identifier in DEVELOPMENT else "reserved",
            duplicate_group=identifier,
            approval_source="Existing user approvals D1-D6 and corpus acceptance",
            review_status="user_approved",
            input=f"inputs/{identifier}.txt",
            target=f"targets/{identifier}.json",
        )
        entry.pop("proposed_target")
        entry["source_hashes"] = {name: review["source_sha256"][name] for name in document["sources"].values()}
        for row in entry["row_evidence"]:
            row["historical_review_status"] = row["review_status"]
            row["review_status"] = "user_approved"
        blobs[entry["input"]], blobs[entry["target"]] = content, json_bytes(target)
        if identifier in DEVELOPMENT:
            if target["errors"]:
                raise ValueError(f"Unexpected warning target: {identifier}")
            prior = next(x for x in evaluation["documents"] if x["document_id"] == identifier)
            prediction = (root / prior["prediction"]).read_bytes()
            parsed = parse_and_validate_extraction(prediction.decode(), schema)
            if parsed["date"] != target["date"] or parsed["results"] != target["results"]:
                raise ValueError(f"Reviewed Gemini core mismatch: {identifier}")
            entry.update(gemini=f"gemini/{identifier}.json", gemini_source=prior["prediction"])
            blobs[entry["gemini"]] = prediction
        documents.append(entry)
    if len(documents) != 9 or {d["document_id"] for d in documents} != set(COUNTS):
        raise ValueError("Expected nine distinct reviewed documents.")
    for suffix in ("txt", "expected.json"):
        blobs[f"smoke/omissions.{suffix}"] = (root / f"tests/samples/extraction_poc/omissions.{suffix}").read_bytes()
    parse_and_validate_extraction(blobs["smoke/omissions.expected.json"].decode(), schema)
    manifest = {
        "packet_version": 1,
        "created_at": now(),
        "documents": documents,
        "approvals": decisions,
        "model": MODEL,
        "model_digest": DIGEST,
        "ollama_version": VERSION,
        "template_sha256": TEMPLATE_HASH,
        "parameters_sha256": PARAMETERS_HASH,
        "review_sha256": sha(review_path.read_bytes()),
        "historical_evaluation_review_sha256": evaluation["reference_target_sha256"],
        "provenance_note": "Historical whole-packet hash differs after review updates; original hashes retained. "
        "Current development core targets were checked against saved Gemini predictions.",
        "files": {name: sha(content) for name, content in blobs.items()},
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, content in blobs.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    write_json(output / "manifest.json", manifest)
    (output / "manifest.sha256").write_text(sha((output / "manifest.json").read_bytes()) + "\n")


def load_packet(directory: Path):
    raw = (directory / "manifest.json").read_bytes()
    if sha(raw) != (directory / "manifest.sha256").read_text().strip():
        raise ValueError("Packet manifest hash mismatch.")
    packet = json.loads(raw)
    for name, digest in packet["files"].items():
        if sha((directory / name).read_bytes()) != digest:
            raise ValueError(f"Packet hash mismatch: {name}")
    if packet["model_digest"] != DIGEST or packet["template_sha256"] != TEMPLATE_HASH:
        raise ValueError("Unexpected packet model identity.")
    if sha((directory / "prompts/laboratory_test_processor/v4.yaml").read_bytes()) != PROMPT_HASH:
        raise ValueError("Unexpected packet prompt.")
    if read_json(directory / "scoring.json") != RULES:
        raise ValueError("Scoring rules changed.")
    if read_json(directory / "schema.json") != schema_at(directory / "prompts"):
        raise ValueError("Schema differs from prompt.")
    documents = packet["documents"]
    if len(documents) != 9 or {d["document_id"] for d in documents} != set(COUNTS):
        raise ValueError("Unexpected packet inventory.")
    for d in documents:
        expected = "development" if d["document_id"] in DEVELOPMENT else "reserved"
        if d["split"] != expected:
            raise ValueError("Reserved split changed.")
    return packet


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError("Local evaluation refuses HTTP redirects.")


def request_bytes(route: str, payload=None, *, timeout=10) -> bytes:
    # A fixed host and disabled proxies/redirects prevent accidental provider routing.
    request = urllib.request.Request(
        ENDPOINT + route,
        data=None if payload is None else json_bytes(payload),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        return response.read()


def metadata(route: str, payload=None):
    return json.loads(request_bytes(route, payload))


def preflight(profile="qwen3"):
    config = configuration(profile)
    version, tags = metadata("/api/version"), metadata("/api/tags")
    if version.get("version") != VERSION:
        raise ValueError("Ollama version changed; review before running.")
    model = next((m for m in tags["models"] if m["name"] == config["model"]), None)
    if not model or model["digest"] != config["digest"] or model.get("remote_host") or model.get("remote_model"):
        raise ValueError("Required pinned local model is unavailable.")
    details = metadata("/api/show", {"model": config["model"]})
    if details.get("remote_host") or details.get("remote_model"):
        raise ValueError("Remote model metadata is forbidden.")
    if details["details"].get("quantization_level") != config["quantization"]:
        raise ValueError("Quantization changed.")
    if (
        sha(details["template"].encode()) != config["template_sha256"]
        or sha(details["parameters"].encode()) != config["parameters_sha256"]
    ):
        raise ValueError("Native model template/parameters changed.")
    info = details["model_info"]
    tokenizer_evidence = {}
    if config["quantization"] == "nvfp4":
        # MLX /show omits tokenizer internals. Verify the previously downloaded public metadata offline.
        directory = ROOT / "data/extraction_poc/qwen35-metadata-v1"
        hashes = {
            "registry-manifest.json": config["digest"],
            "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
        }
        for name, digest in hashes.items():
            if sha((directory / name).read_bytes()) != digest:
                raise ValueError("MLX tokenizer provenance changed.")
        tokenizer_evidence = {
            "files": hashes,
            "directory": str(directory),
            "chat_template": read_json(directory / "tokenizer_config.json")["chat_template"],
        }
    elif info.get("tokenizer.ggml.model") != "gpt2" or info.get("tokenizer.ggml.pre") != (
        "qwen2" if profile == "qwen3" else "qwen35"
    ):
        raise ValueError("Byte-level tokenizer assumption no longer holds.")
    return {"version": version, "model": model, "show": details, "tokenizer_evidence": tokenizer_evidence}


def make_request(dataset: Path, text: str, profile="qwen3"):
    config = configuration(profile)
    prompt, system, _ = PromptEngine(dataset / "prompts").render(
        PromptTypeEnum.laboratory_test_processor, "v4", input_text=text
    )
    bound = len((system["system_instruction"] + prompt).encode("utf-8")) + 1024
    if bound + config["options"]["num_predict"] > config["options"]["num_ctx"]:
        raise ValueError("Conservative input bound plus output allowance exceeds context.")
    decoding_schema = copy.deepcopy(system["response_json_schema"])
    if profile in ("qwen35-4b-compat", "qwen35-4b-thinking"):
        # MLX/XGrammar treats this unanchored JSON Schema pattern as a whole string.
        # Keep the original pattern in post-validation; only decoding omits it.
        del decoding_schema["properties"]["results"]["items"]["properties"]["name"]["pattern"]
    return {
        "model": config["model"],
        "messages": [{"role": "system", "content": system["system_instruction"]}, {"role": "user", "content": prompt}],
        "format": decoding_schema,
        "options": config["options"].copy(),
        "stream": False,
        "think": config["think"],
        "keep_alive": "10m",
    }, bound


def run(dataset: Path, output: Path, profile="qwen3") -> int:
    config = configuration(profile)
    if output.exists():
        raise FileExistsError(output)
    packet = load_packet(dataset)
    runtime = preflight(profile)
    entries = [{"document_id": "smoke", "input": "smoke/omissions.txt"}] + [
        next(d for d in packet["documents"] if d["document_id"] == name) for name in DEVELOPMENT
    ]
    prepared = [make_request(dataset, (dataset / d["input"]).read_text(), profile) for d in entries]
    # The original ledger remains intact. Both thinking candidates share one additional six-call budget.
    scope = "" if profile == "qwen3" else "." + ("qwen35-thinking" if config["think"] else profile)
    ledger_path = dataset.parent / f".{dataset.name}{scope}.execution.json"
    with ledger_path.open("a+", encoding="utf-8") as ledger:
        fcntl.flock(ledger, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger.seek(0)
        previous = ledger.read()
        if previous and json.loads(previous)["attempts"]:
            raise ValueError("This packet already has an attempted run; do not bypass the six-request scope.")
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "runtime.json", runtime)
        manifest = {
            "started_at": now(),
            "profile": profile,
            "think": config["think"],
            "dataset_sha256": sha((dataset / "manifest.json").read_bytes()),
            "model": config["model"],
            "digest": config["digest"],
            "ollama_version": VERSION,
            "endpoint": ENDPOINT,
            "prompt_sha256": PROMPT_HASH,
            "schema_sha256": sha(json_bytes(read_json(dataset / "schema.json"))),
            "decoding_schema_sha256": sha(json_bytes(prepared[0][0]["format"])),
            "template_sha256": config["template_sha256"],
            "options": config["options"],
            "attempts": 0,
            "hardware_user_reported": "MacBook Air M5, 24 GB unified memory",
            "host_peak_memory_and_pressure": "unavailable from container; allocation snapshots only",
            "records": [
                {
                    "document_id": d["document_id"],
                    "status": "not_attempted",
                    "input_sha256": sha((dataset / d["input"]).read_bytes()),
                }
                for d in entries
            ],
        }
        stopped = None
        write_json(output / "manifest.json", manifest)
        for entry, record, (request, bound) in zip(entries, manifest["records"], prepared, strict=True):
            if stopped:
                record["error"] = stopped
                continue
            directory = output / entry["document_id"]
            directory.mkdir()
            write_json(directory / "request.json", request)
            record.update(
                request_sha256=sha(json_bytes(request)),
                input_token_upper_bound=bound,
                bound_method="UTF-8 message bytes + 1024 chat-format tokens; not exact token count",
            )
            try:
                record["allocation_before"] = metadata("/api/ps")
            except Exception as error:
                stopped = record["error"] = f"Runtime pre-request failure: {error}"
                break
            manifest["attempts"] += 1
            ledger.seek(0)
            ledger.truncate()
            ledger.write(json.dumps({"attempts": manifest["attempts"], "run_dir": str(output.resolve())}))
            ledger.flush()
            os.fsync(ledger.fileno())
            record.update(status="started", started_at=now())
            write_json(output / "manifest.json", manifest)
            start = time.monotonic()
            runtime_failure = False
            try:
                raw = request_bytes("/api/chat", request, timeout=600)
                (directory / "response.raw.json").write_bytes(raw)
                record["raw_sha256"] = sha(raw)
                try:
                    response = json.loads(raw)
                except (ValueError, UnicodeError):
                    runtime_failure = True
                    raise
                if not isinstance(response, dict) or not isinstance(response.get("message", {}), dict):
                    runtime_failure = True
                    raise ValueError("Malformed Ollama response envelope.")
                message = response.get("message", {})
                content = message.get("content", "")
                (directory / "content.txt").write_text(content, encoding="utf-8")
                record["response_metadata"] = {k: v for k, v in response.items() if k != "message"}
                record["timing"] = {
                    "load_seconds": response.get("load_duration", 0) / 1e9,
                    "prompt_seconds": response.get("prompt_eval_duration", 0) / 1e9,
                    "generation_seconds": response.get("eval_duration", 0) / 1e9,
                    "generation_tokens_per_second": response.get("eval_count", 0) * 1e9 / response["eval_duration"]
                    if response.get("eval_duration")
                    else None,
                }
                if response.get("error"):
                    runtime_failure = True
                    raise ValueError(response["error"])
                if response.get("model") != config["model"]:
                    runtime_failure = True
                    raise ValueError("Response model identity differs.")
                if not response.get("done") or response.get("done_reason") != "stop":
                    raise ValueError("Incomplete response or output limit reached.")
                if message.get("tool_calls") or (message.get("thinking") and not config["think"]):
                    raise ValueError("Unexpected tools or thinking output.")
                record["thinking_characters"] = len(message.get("thinking", ""))
                if config["think"] and not message.get("thinking"):
                    runtime_failure = True
                    raise ValueError("Requested thinking mode produced no separate thinking; verify runtime support.")
                prediction = parse_and_validate_extraction(content, read_json(dataset / "schema.json"))
                write_json(directory / "prediction.json", prediction)
                record.update(status="success", prediction_sha256=sha((directory / "prediction.json").read_bytes()))
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as error:
                runtime_failure = True
                if isinstance(error, urllib.error.HTTPError):
                    (directory / "error.raw").write_bytes(error.read())
                elif isinstance(error, http.client.IncompleteRead):
                    (directory / "error.raw").write_bytes(error.partial)
                record.update(status="failed", error=f"{type(error).__name__}: {error}")
            except (ValueError, TypeError, KeyError) as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}")
            except KeyboardInterrupt:
                record.update(status="failed", error="Interrupted; no automatic retry.")
                runtime_failure = True
            finally:
                record.update(wall_seconds=time.monotonic() - start, finished_at=now())
                try:
                    record["allocation_after"] = metadata("/api/ps")
                    loaded = next(m for m in record["allocation_after"]["models"] if m["name"] == config["model"])
                    if (
                        loaded.get("context_length") != config["options"]["num_ctx"]
                        or loaded.get("digest") != config["digest"]
                    ):
                        raise ValueError("Loaded context/model differs from requested configuration.")
                except Exception as error:
                    runtime_failure = True
                    record["allocation_error"] = str(error)
                if runtime_failure:
                    record["failure_category"] = "runtime"
                    record["status"] = "failed"
                    (directory / "prediction.json").unlink(missing_ok=True)
                elif record["status"] == "failed":
                    record["failure_category"] = "model_output"
                if runtime_failure or (entry["document_id"] == "smoke" and record["status"] != "success"):
                    stopped = "Batch stopped after runtime or compatibility failure; no retries."
                write_json(output / "manifest.json", manifest)
        for record in manifest["records"]:
            if record["status"] == "not_attempted":
                record.setdefault("error", stopped or "Not attempted")
        manifest.update(finished_at=now(), stopped=stopped)
        write_json(output / "manifest.json", manifest)
    return int(any(r["status"] != "success" for r in manifest["records"]))


def apply_warning_review(metrics: dict, record: dict, reviews: dict):
    review = reviews.get(record["document_id"])
    if review is None:
        return
    if record["status"] != "success" or review["raw_sha256"] != record["raw_sha256"]:
        raise ValueError("Warning review does not match a successful raw response.")
    if review["status"] not in ("pass", "fail") or not review.get("reason") or not review.get("reviewer"):
        raise ValueError("Warning review needs pass/fail, reason and reviewer.")
    metrics["warning_status"] = review["status"]
    metrics["warning_review"] = review
    metrics["strict_pass"] = metrics["date_exact"] and metrics["ordered_results_exact"] and review["status"] == "pass"


def score(dataset: Path, run_dir: Path):
    packet = load_packet(dataset)
    manifest = read_json(run_dir / "manifest.json")
    profile = manifest.get("profile", "qwen3")
    config = configuration(profile)
    if manifest["dataset_sha256"] != sha((dataset / "manifest.json").read_bytes()):
        raise ValueError("Run belongs to a different packet.")
    if (manifest["digest"], manifest["ollama_version"], manifest["prompt_sha256"], manifest["options"]) != (
        config["digest"],
        VERSION,
        PROMPT_HASH,
        config["options"],
    ):
        raise ValueError("Run configuration differs from the frozen experiment.")
    if (
        manifest["model"] != config["model"]
        or manifest["template_sha256"] != config["template_sha256"]
        or manifest.get("think", False) != config["think"]
    ):
        raise ValueError("Run model/template/thinking differs from its profile.")
    if [r["document_id"] for r in manifest["records"]] != ["smoke", *DEVELOPMENT]:
        raise ValueError("Unexpected run inventory; B scoring is forbidden.")
    schema = read_json(dataset / "schema.json")
    review_path = run_dir / "warning_review.json"
    reviews = read_json(review_path) if review_path.exists() else {}
    if set(reviews) - {"smoke", *DEVELOPMENT}:
        raise ValueError("Warning review contains an unexpected/reserved document.")
    scores = {"dataset_sha256": manifest["dataset_sha256"], "qwen": [], "gemini": [], "smoke": None}
    for record in manifest["records"]:
        identifier = record["document_id"]
        input_path = (
            "smoke/omissions.txt"
            if identifier == "smoke"
            else next(d["input"] for d in packet["documents"] if d["document_id"] == identifier)
        )
        if record["input_sha256"] != sha((dataset / input_path).read_bytes()):
            raise ValueError("Run input differs from frozen packet.")
        if record.get("request_sha256"):
            request_raw = (run_dir / identifier / "request.json").read_bytes()
            expected_request, _ = make_request(dataset, (dataset / input_path).read_text(), profile)
            if sha(request_raw) != record["request_sha256"] or json.loads(request_raw) != expected_request:
                raise ValueError("Saved request differs from the frozen experiment.")
        prediction = None
        if record["status"] == "success":
            directory = run_dir / identifier
            raw = (directory / "response.raw.json").read_bytes()
            saved = (directory / "prediction.json").read_bytes()
            if sha(raw) != record["raw_sha256"] or sha(saved) != record["prediction_sha256"]:
                raise ValueError(f"Changed response/prediction: {identifier}")
            prediction = parse_and_validate_extraction(saved.decode(), schema)
            if parse_and_validate_extraction(json.loads(raw)["message"]["content"], schema) != prediction:
                raise ValueError("Saved prediction differs from raw model content.")
        if identifier == "smoke":
            scores["smoke"] = compare(read_json(dataset / "smoke/omissions.expected.json"), prediction)
            scores["smoke"].update(warning_checklist=RULES["smoke_warning_checklist"], runtime=record)
            apply_warning_review(scores["smoke"], record, reviews)
            continue
        document = next(d for d in packet["documents"] if d["document_id"] == identifier)
        target = read_json(dataset / document["target"])
        for provider, result in (("qwen", prediction), ("gemini", read_json(dataset / document["gemini"]))):
            if result is not None:
                parse_and_validate_extraction(json.dumps(result), schema)
            metrics = compare(target, result)
            metrics.update(
                document_id=identifier,
                organization=document["organization_alias"],
                source=document["sources"],
                row_evidence=document["row_evidence"],
                date_evidence=document["date_evidence_parsed_lines"],
                upstream_evidence=document["supplemental_context"],
            )
            if provider == "qwen":
                metrics["runtime"] = record
                apply_warning_review(metrics, record, reviews)
            scores[provider].append(metrics)
    scores["summary"] = {
        provider: {
            "pooled": summarize(scores[provider]),
            "by_organization": {
                org: summarize([r for r in scores[provider] if r["organization"] == org]) for org in ("A", "C")
            },
        }
        for provider in ("qwen", "gemini")
    }
    if any(r["status"] != "success" for r in manifest["records"]):
        recommendation = "Investigate runtime/model-output failures before attributing the gap to a need for SFT."
    elif all(r["strict_pass"] for r in scores["qwen"]):
        recommendation = (
            "Gather unseen reports; development targets are met. Review the separate synthetic warning before SFT."
        )
    else:
        recommendation = (
            "Review source-visible extraction discrepancies as potential SFT targets; no claim about reserved B."
        )
    scores["recommendation"] = recommendation
    write_json(run_dir / "scores.json", scores)
    lines = [
        "# Local Qwen development evaluation",
        "",
        recommendation,
        "",
        "A+C development corpus only; B was neither inferred nor scored. All nine reports previously informed the contract.",
        "Position-aligned errors can cascade after an omission; multiset scores are separate diagnostics.",
        "Missing/extra exact records count substitutions as a missing expected and an extra predicted record.",
        "",
        "| Provider | Report | Schema | Date | Ordered rows | Fields | Warnings | Strict pass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for provider in ("qwen", "gemini"):
        for row in scores[provider]:
            lines.append(
                f"| {provider} | {row['document_id']} | {row['schema_valid']} | {row['date_exact']} | "
                f"{row['ordered_record_matches']}/{row['expected_rows']} | {row['field_matches']}/{row['expected_fields']} | "
                f"{row['warning_status']} | {row['strict_pass']} |"
            )
    lines += [
        "",
        "## Totals and organization scores",
        "",
        "```json",
        json.dumps(scores["summary"], indent=2),
        "```",
        "",
        "## Synthetic compatibility request (not pooled)",
        "",
        "Warning semantics use the frozen checklist and hash-bound manual reviews; no prose equality pass.",
        "```json",
        json.dumps(scores["smoke"], ensure_ascii=False, indent=2),
        "```",
    ]
    for row in scores["qwen"]:
        lines += [
            "",
            f"## {row['document_id']}",
            "",
            f"Sources: {json.dumps(row['source'], ensure_ascii=False)}",
            "",
            "### Runtime",
            "",
            "Allocation snapshots are not peak memory. Host memory pressure was not measured.",
            "```json",
            json.dumps(row["runtime"], ensure_ascii=False, indent=2),
            "```",
            "",
            "### Extraction discrepancies",
            "",
        ]
        for difference in row["differences"]:
            evidence = next((e for e in row["row_evidence"] if e["row"] == difference.get("row")), None)
            if not row["schema_valid"]:
                assessment = "No valid prediction: runtime/model-output failure, not individual extraction errors."
            elif difference.get("field") == "errors":
                assessment = "Unexpected warning; review meaning against source and frozen exclusions."
            else:
                assessment = (
                    "Mismatch against approved parser-visible target; inspect alignment before assigning cause."
                )
            lines += [
                "```json",
                json.dumps(
                    {**difference, "evidence": evidence or row["date_evidence"], "assessment": assessment},
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
            ]
        if not row["differences"]:
            lines.append("None.")
        lines += [
            "",
            "### Upstream parser evidence",
            "",
            "Preserve parser spelling; do not count approved PDF-only corrections as errors.",
            "```json",
            json.dumps(
                {"context": row["upstream_evidence"], "noted_rows": [r for r in row["row_evidence"] if r.get("notes")]},
                ensure_ascii=False,
                indent=2,
            ),
            "```",
        ]
    (run_dir / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("freeze")
    export.add_argument("--output-dir", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--profile", choices=PROFILES, default="qwen3")
    execute.add_argument("--dataset", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, required=True)
    evaluate = commands.add_parser("score")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze(args.output_dir)
    elif args.command == "run":
        return run(args.dataset, args.output_dir, args.profile)
    else:
        score(args.dataset, args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
