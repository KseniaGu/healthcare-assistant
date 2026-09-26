"""Bounded host-only Qwen3 BF16 versus Ollama Q8 extraction diagnostic."""

import argparse
import fcntl
import importlib.metadata
import json
import os
import psutil
import platform
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from research.extraction import evaluate_local_extraction as prior
from backend.utils.prompt_engine import PromptEngine
from configs.enums import PromptTypeEnum

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/extraction_poc/qwen-eval-packet-v1"
ORIGINAL_OUTPUT = ROOT / "data/extraction_poc/qwen-transformers-diagnostic-v1"
RERUN_OUTPUT = ROOT / "data/extraction_poc/qwen-transformers-orb-off-v1"
EAGER_OUTPUT = ROOT / "data/extraction_poc/qwen-transformers-eager-v1"
OUTPUT = (EAGER_OUTPUT if os.environ.get("QWEN_EAGER_RERUN") == "1" else
          RERUN_OUTPUT if os.environ.get("QWEN_ORB_OFF_RERUN") == "1" else ORIGINAL_OUTPUT)
ATTENTION = "eager" if OUTPUT == EAGER_OUTPUT else "sdpa"
print("ATTENTION:", ATTENTION)
HF_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
IDS = ("smoke", *prior.DEVELOPMENT)
MAX_NEW = 16384
CONTEXT = 32768
TIMEOUT = 1800
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_OPTIONS = prior.OPTIONS.copy()

process = psutil.Process(os.getpid())
import torch

def report_memory(label):
    print(
        label,
        f"RSS={process.memory_info().rss / 1024**3:.2f} GiB",
        f"MPS current={torch.mps.current_allocated_memory() / 1024**3:.2f} GiB",
        f"MPS driver={torch.mps.driver_allocated_memory() / 1024**3:.2f} GiB",
        flush=True,
    )

from transformers import StoppingCriteria, StoppingCriteriaList
class MemoryReporter(StoppingCriteria):
    def __init__(self, prompt_len, every=128, empty_cache_at=None):
        self.prompt_len = prompt_len
        self.every = every
        self.empty_cache_at = empty_cache_at
        self.early_steps = {1, 2, 4, 8, 16, 32, 64, 128}
        self.reported = set()

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids.shape[-1] - self.prompt_len

        should_report = (
            generated in self.early_steps
            or (generated > 128 and generated % self.every == 0)
        )

        if should_report and generated not in self.reported:
            self.reported.add(generated)

            torch.mps.synchronize()
            report_memory(f"generated={generated}")

            if generated == self.empty_cache_at:
                torch.mps.empty_cache()

                print(
                    "after empty_cache:",
                    f"current={torch.mps.current_allocated_memory() / 1024**3:.2f} GiB",
                    f"driver={torch.mps.driver_allocated_memory() / 1024**3:.2f} GiB",
                )

        if torch.mps.driver_allocated_memory() > 14 * 1024**3:
            torch.mps.empty_cache()

            print(
                "after decode empty_cache:",
                f"current={torch.mps.current_allocated_memory()/1024**3:.2f}",
                f"driver={torch.mps.driver_allocated_memory()/1024**3:.2f}",
            )

        return False


def stamp():
    return datetime.now(UTC).isoformat()


def memory():
    """Capture labeled host pressure and VM counters; neither is process peak RSS."""
    return {
        "captured_at": stamp(),
        "memory_pressure_Q": subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout,
        "vm_stat": subprocess.run(["vm_stat"], capture_output=True, text=True).stdout,
    }


def hardware():
    report = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True).stdout
    labels = ("Model Name:", "Model Identifier:", "Chip:", "Total Number of Cores:", "Memory:")
    return [line.strip() for line in report.splitlines() if line.strip().startswith(labels)]


def stopped_services():
    """Record that OrbStack and Ollama have no running host processes."""
    process_list = subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True, text=True, check=True).stdout
    running = [line.strip() for line in process_list.splitlines() if (
        "/Applications/OrbStack.app/" in line
        or "/Applications/Ollama.app/" in line
        or "ollama serve" in line
    )]
    if running:
        raise RuntimeError("OrbStack or Ollama is still running; keep the rerun background condition fixed")
    return {"checked_at": stamp(), "matching_processes": running}


def packet_entries():
    packet = prior.load_packet(DATASET)
    live_prompt = ROOT / "backend/prompts/laboratory_test_processor/v4.yaml"
    if prior.sha(live_prompt.read_bytes()) != prior.PROMPT_HASH:
        raise ValueError("Active v4.1 prompt differs from frozen packet")
    by_id = {d["document_id"]: d for d in packet["documents"]}
    entries = [{"document_id": "smoke", "input": "smoke/omissions.txt", "target": "smoke/omissions.expected.json"}]
    entries += [by_id[name] for name in prior.DEVELOPMENT]
    for entry in entries:
        target = prior.read_json(DATASET / entry["target"])
        strict_validate(json.dumps(target), prior.read_json(DATASET / "schema.json"))
    for entry in entries[1:]:
        gemini = strict_validate((DATASET / entry["gemini"]).read_text(), prior.read_json(DATASET / "schema.json"))
        target = prior.read_json(DATASET / entry["target"])
        if gemini["date"] != target["date"] or gemini["results"] != target["results"]:
            raise ValueError("Saved Gemini core differs from approved target")
    historical = prior.read_json(DATASET.parent / "qwen-untuned-v1/manifest.json")
    if historical["digest"] != prior.DIGEST or historical["attempts"] != 6:
        raise ValueError("Historical Qwen identity or attempt count differs")
    for record in historical["records"]:
        path = DATASET.parent / "qwen-untuned-v1" / record["document_id"] / "response.raw.json"
        if prior.sha(path.read_bytes()) != record["raw_sha256"]:
            raise ValueError("Historical Qwen raw response changed")
    return entries


def strict_validate(content, schema):
    """Validate the complete original schema without markdown or prose cleanup."""
    parsed = json.loads(content)
    issue = next(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(parsed), None)
    if issue:
        raise ValueError(f"Schema violation at {issue.json_path}: {issue.message}")
    return parsed


def messages_for(entry):
    source = (DATASET / entry["input"]).read_text(encoding="utf-8")
    user, system, _ = PromptEngine(DATASET / "prompts").render(
        PromptTypeEnum.laboratory_test_processor, "v4", input_text=source
    )
    return [{"role": "system", "content": system["system_instruction"]}, {"role": "user", "content": user}]


def ollama_request(messages):
    return {
        "model": prior.MODEL,
        "messages": messages,
        "options": OLLAMA_OPTIONS.copy(),
        "think": False,
        "stream": True,
        "keep_alive": "10m",
    }


def prepare():
    entries = packet_entries()
    import huggingface_hub
    import torch
    import transformers
    import safetensors

    if platform.system() != "Darwin" or platform.machine() != "arm64" or not torch.backends.mps.is_available():
        raise RuntimeError("Requires native macOS arm64 with available PyTorch MPS")
    from huggingface_hub import HfApi, snapshot_download
    from safetensors import safe_open
    from transformers import AutoTokenizer

    if OUTPUT in (RERUN_OUTPUT, EAGER_OUTPUT):
        previous = prior.read_json(ORIGINAL_OUTPUT / "provenance.json")
        revision = previous["revision"]
        model_path = Path(previous["model_path"])
        services = stopped_services()
    else:
        revision = HfApi().model_info(HF_MODEL).sha
        model_path = Path(snapshot_download(
            HF_MODEL, revision=revision,
            allow_patterns=["*.json", "*.safetensors", "*.model", "vocab.*", "merges.txt"],
        ))
        services = None
    if not revision or len(revision) != 40:
        raise RuntimeError("Could not pin a full Hugging Face commit")
    if OUTPUT in (RERUN_OUTPUT, EAGER_OUTPUT) and not model_path.exists():
        raise RuntimeError("Pinned checkpoint is missing from the host cache")
    config = prior.read_json(model_path / "config.json")
    if config.get("model_type") != "qwen3" or config.get("torch_dtype") != "bfloat16":
        raise RuntimeError("Checkpoint is not the expected Qwen3 BF16 model")
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise RuntimeError("Original Safetensors weights absent")
    dtypes = set()
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            dtypes.update(handle.get_slice(key).get_dtype() for key in handle.keys())
    if dtypes - {"BF16", "F32"} or "BF16" not in dtypes:
        raise RuntimeError(f"Unexpected checkpoint weight dtypes: {dtypes}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    template = tokenizer.chat_template
    if not template:
        raise RuntimeError("Checkpoint has no native chat template")
    formatted_inputs = {}
    for entry in entries:
        messages = messages_for(entry)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(token_ids) + MAX_NEW > CONTEXT:
            raise RuntimeError(f"{entry['document_id']} exceeds 32K formatted context budget")
        formatted_inputs[entry["document_id"]] = {
            "messages_sha256": prior.sha(prior.json_bytes(messages)),
            "rendered_sha256": prior.sha(rendered.encode()),
            "input_tokens": len(token_ids), "token_ids_sha256": prior.sha(prior.json_bytes(token_ids)),
        }
    provenance = {
        "prepared_at": stamp(), "packet_manifest_sha256": prior.sha((DATASET / "manifest.json").read_bytes()),
        "historical_manifest_sha256": prior.sha((DATASET.parent / "qwen-untuned-v1/manifest.json").read_bytes()),
        "model": HF_MODEL, "revision": revision, "model_path": str(model_path),
        "safetensors_files": {p.name: prior.sha(p.read_bytes()) for p in shards},
        "tokenizer_files": {p.name: prior.sha(p.read_bytes()) for p in model_path.glob("tokenizer*") if p.is_file()},
        "tokenizer_class": type(tokenizer).__name__, "chat_template": template,
        "chat_template_sha256": prior.sha(template.encode()),
        "generation_config": prior.read_json(model_path / "generation_config.json"),
        "checkpoint_dtypes": sorted(dtypes), "selected_dtype": "torch.bfloat16",
        "attn_implementation": ATTENTION,
        "os": platform.platform(), "architecture": platform.machine(), "hardware": hardware(), "python": sys.version,
        "torch": torch.__version__, "mps_available": torch.backends.mps.is_available(),
        "transformers": transformers.__version__, "huggingface_hub": huggingface_hub.__version__,
        "safetensors_version": safetensors.__version__, "memory_before": memory(),
        "package_versions": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "huggingface-hub", "safetensors", "tokenizers",
            "jsonschema", "Jinja2", "PyYAML", "requests", "pydantic", "langdetect",
        )},
        "documents": [e["document_id"] for e in entries],
        "formatted_inputs": formatted_inputs,
    }
    if OUTPUT in (RERUN_OUTPUT, EAGER_OUTPUT):
        for key in ("revision", "safetensors_files", "tokenizer_files", "chat_template_sha256", "formatted_inputs", "package_versions"):
            if provenance[key] != previous[key]:
                raise RuntimeError(f"Rerun configuration differs from original: {key}")
        provenance["rerun_of"] = str(ORIGINAL_OUTPUT)
        provenance["original_provenance_sha256"] = prior.sha((ORIGINAL_OUTPUT / "provenance.json").read_bytes())
        provenance["background_services"] = services
    prior.write_json(OUTPUT / "provenance.json", provenance)
    print(json.dumps({k: provenance[k] for k in ("model", "revision", "selected_dtype", "mps_available")}))


def ollama_preflight():
    import requests

    with requests.Session() as session:
        session.trust_env = False
        version = session.get(OLLAMA_URL + "/api/version", timeout=10).json()["version"]
        tags = session.get(OLLAMA_URL + "/api/tags", timeout=10).json()["models"]
        model = next((m for m in tags if m["name"] == prior.MODEL), None)
        if not model or model["digest"] != prior.DIGEST or model.get("remote_host"):
            raise RuntimeError("Pinned local Ollama model absent or changed")
        show = session.post(OLLAMA_URL + "/api/show", json={"model": prior.MODEL}, timeout=10).json()
        if show.get("remote_host") or show["details"]["quantization_level"] != "Q8_0":
            raise RuntimeError("Unexpected Ollama model provenance")
        ps = session.get(OLLAMA_URL + "/api/ps", timeout=10).json()
    return {
        "version": version, "historical_version": prior.VERSION,
        "runtime_drift": version != prior.VERSION, "digest": model["digest"],
        "quantization": show["details"]["quantization_level"],
        "template": show["template"], "template_sha256": prior.sha(show["template"].encode()),
        "parameters": show["parameters"], "parameters_sha256": prior.sha(show["parameters"].encode()),
        "loaded_models_before": ps, "checked_at": stamp(),
    }


def _timeout_exit(partial):
    partial.with_suffix(".timeout").write_text(f"Generation exceeded {TIMEOUT} seconds\n")
    os._exit(124)


def hf_one(request_path, output_dir):
    """Run one generation in its own process so a timeout can release MPS memory."""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

    provenance = prior.read_json(OUTPUT / "provenance.json")
    model_path = provenance["model_path"]
    request = prior.read_json(request_path)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    report_memory("before model load")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, use_safetensors=True,
        local_files_only=True, attn_implementation=ATTENTION
    )
    report_memory("after HF load")
    model = model.to("mps").eval()
    report_memory("after MPS transfer")
    load_seconds = time.monotonic() - started
    if model.config._attn_implementation != ATTENTION:
        raise RuntimeError("Loaded attention backend differs from frozen protocol")
    if any(p.device.type != "mps" or p.dtype not in (torch.bfloat16, torch.float32) for p in model.parameters()):
        raise RuntimeError("Model did not remain in 16-bit MPS execution")
    rendered = tokenizer.apply_chat_template(request["messages"], tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.apply_chat_template(
        request["messages"], tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    input_count = tokens.shape[-1]
    if input_count + MAX_NEW > CONTEXT:
        raise ValueError(f"Formatted input {input_count} plus output allowance exceeds {CONTEXT}")
    frozen = provenance["formatted_inputs"][output_dir.name]
    if input_count != frozen["input_tokens"] or prior.sha(rendered.encode()) != frozen["rendered_sha256"]:
        raise RuntimeError("Tokenizer rendering differs from preparation")
    (output_dir / "rendered_prompt.txt").write_text(rendered, encoding="utf-8")
    (output_dir / "input_token_ids.json").write_text(json.dumps(tokens[0].tolist()))
    before = memory()
    partial = output_dir / "content.partial.txt"

    class SaveStreamer(TextStreamer):
        def on_finalized_text(self, text, stream_end=False):
            with partial.open("a", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()

    streamer = SaveStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    watchdog = threading.Timer(TIMEOUT, _timeout_exit, args=(partial,))
    watchdog.daemon = True
    start = time.monotonic()
    watchdog.start()
    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids=tokens.to("mps"), max_new_tokens=MAX_NEW,
                do_sample=False, repetition_penalty=1.0,
                cache_implementation="dynamic", streamer=streamer,
                stopping_criteria=StoppingCriteriaList([MemoryReporter(
                    prompt_len=input_count,
                    every=128,
                    empty_cache_at=2,
                )]),
            )
        torch.mps.synchronize()
        report_memory("after generation")
    finally:
        watchdog.cancel()

    generated = output[0, input_count:].tolist()
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    raw = tokenizer.decode(generated, skip_special_tokens=False)
    content_ids = generated[:-1] if generated and generated[-1] in eos_ids else generated
    content = tokenizer.decode(content_ids, skip_special_tokens=False)
    (output_dir / "generated.raw.txt").write_text(raw, encoding="utf-8")
    (output_dir / "content.txt").write_text(content, encoding="utf-8")
    (output_dir / "output_token_ids.json").write_text(json.dumps(generated))
    prior.write_json(output_dir / "generation.json", {
        "load_seconds": load_seconds, "generation_seconds": time.monotonic() - start,
        "input_tokens": input_count, "output_tokens": len(generated),
        "finish_reason": "eos" if generated and generated[-1] in eos_ids else "length_or_other",
        "eos_token_ids": sorted(eos_ids), "chat_template": tokenizer.chat_template,
        "chat_template_sha256": prior.sha(tokenizer.chat_template.encode()),
        "rendered_prompt_sha256": prior.sha(rendered.encode()),
        "tokenizer_class": type(tokenizer).__name__, "model_class": type(model).__name__,
        "dtype": str(next(model.parameters()).dtype), "device": str(next(model.parameters()).device),
        "attn_implementation": model.config._attn_implementation,
        "mps_allocated_bytes_after": torch.mps.current_allocated_memory(),
        "mps_driver_bytes_after": torch.mps.driver_allocated_memory(),
        "host_memory_before": before, "host_memory_after": memory(),
    })


def generate_hf(request_path, directory):
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    command = [sys.executable, "-m", "research.extraction.evaluate_transformers_extraction", "_hf_one", str(request_path), str(directory)]
    try:
        result = subprocess.run(command, env=env, capture_output=True, timeout=TIMEOUT + 900)
    except subprocess.TimeoutExpired as error:
        (directory / "worker.partial.stdout").write_bytes(error.stdout or b"")
        (directory / "worker.partial.stderr").write_bytes(error.stderr or b"")
        raise TimeoutError("Transformers worker exceeded loading plus generation deadline") from error
    (directory / "worker.stdout").write_bytes(result.stdout)
    (directory / "worker.stderr").write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f"Transformers worker exit {result.returncode}")
    return prior.read_json(directory / "generation.json")


def generate_ollama(request_path, directory):
    import requests

    request = prior.read_json(request_path)
    started = time.monotonic()
    chunks = []
    final = None
    def on_timeout(_signum, _frame):
        raise TimeoutError(f"Ollama generation exceeded {TIMEOUT} seconds")
    previous = signal.signal(signal.SIGALRM, on_timeout)
    signal.setitimer(signal.ITIMER_REAL, TIMEOUT)
    try:
        with requests.Session() as session:
            session.trust_env = False
            with session.post(
                OLLAMA_URL + "/api/chat", json=request, stream=True,
                timeout=(10, TIMEOUT), allow_redirects=False,
            ) as response:
                if 300 <= response.status_code < 400:
                    raise RuntimeError("Local Ollama endpoint redirected the request")
                response.raise_for_status()
                with (directory / "response.stream.jsonl").open("wb") as raw, (directory / "content.partial.txt").open("w", encoding="utf-8") as partial:
                    for line in response.iter_lines():
                        if not line:
                            continue
                        raw.write(line + b"\n")
                        raw.flush()
                        event = json.loads(line)
                        if event.get("error"):
                            raise RuntimeError(event["error"])
                        fragment = event.get("message", {}).get("content", "")
                        partial.write(fragment)
                        partial.flush()
                        chunks.append(fragment)
                        if event.get("done"):
                            final = event
                            break
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    content = "".join(chunks)
    (directory / "content.txt").write_text(content, encoding="utf-8")
    if not final or final.get("model") != prior.MODEL or final.get("done_reason") != "stop":
        raise RuntimeError("Incomplete or changed Ollama response")
    metadata = {k: v for k, v in final.items() if k != "message"}
    metadata.update(
        wall_seconds=time.monotonic() - started,
        load_seconds=final.get("load_duration", 0) / 1e9,
        generation_seconds=final.get("eval_duration", 0) / 1e9,
        input_tokens=final.get("prompt_eval_count"),
        output_tokens=final.get("eval_count"),
        finish_reason=final.get("done_reason"), host_memory_after=memory(),
    )
    prior.write_json(directory / "generation.json", metadata)
    return metadata


def reserve(ledger_path, condition, identifier):
    """Persist a new attempt before generation; never erase or reuse an attempt."""
    with ledger_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        content = handle.read()
        limit = 6 if OUTPUT in (RERUN_OUTPUT, EAGER_OUTPUT) else 12
        ledger = json.loads(content) if content else {"max_attempts": limit, "attempts": []}
        if ledger["max_attempts"] != limit:
            raise RuntimeError("Diagnostic ledger has an unexpected budget")
        if len(ledger["attempts"]) >= limit or any(
            a["condition"] == condition and a["document_id"] == identifier for a in ledger["attempts"]
        ):
            raise RuntimeError("Diagnostic attempt budget exhausted or duplicate attempt")
        if sum(a["condition"] == condition for a in ledger["attempts"]) >= 6:
            raise RuntimeError("Condition attempt budget exhausted")
        ledger["attempts"].append({"condition": condition, "document_id": identifier, "reserved_at": stamp()})
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(ledger, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        return len(ledger["attempts"])


def run(condition, generate=None):
    if condition not in ("transformers", "ollama"):
        raise ValueError(condition)
    if OUTPUT in (RERUN_OUTPUT, EAGER_OUTPUT) and condition != "transformers":
        raise ValueError("Transformers rerun authorizes Transformers only")
    entries = packet_entries()
    provenance = prior.read_json(OUTPUT / "provenance.json")
    if provenance["packet_manifest_sha256"] != prior.sha((DATASET / "manifest.json").read_bytes()):
        raise ValueError("Packet changed after preparation")
    condition_dir = OUTPUT / condition
    if condition_dir.exists():
        raise FileExistsError("Condition already initialized; no automatic retry")
    if condition == "ollama":
        runtime = ollama_preflight()
    else:
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")
        if OUTPUT in (RERUN_OUTPUT, EAGER_OUTPUT):
            services = stopped_services()
            loaded = {"models": [], "daemon": "stopped"}
        else:
            import requests
            with requests.Session() as session:
                session.trust_env = False
                loaded = session.get(OLLAMA_URL + "/api/ps", timeout=10).json()
            services = None
        if loaded.get("models"):
            raise RuntimeError("Unload idle Ollama models before Transformers inference")
        runtime = {"mps_available": True, "ollama_loaded_models_before": loaded, "background_services": services, "memory_before": memory()}
    condition_dir.mkdir()
    prior.write_json(condition_dir / "runtime.json", runtime)
    manifest = {"condition": condition, "started_at": stamp(), "attempts": 0, "records": [
        {"document_id": e["document_id"], "status": "not_attempted", "input_sha256": prior.sha((DATASET / e["input"]).read_bytes())}
        for e in entries
    ]}
    prior.write_json(condition_dir / "manifest.json", manifest)
    generate = generate or (generate_hf if condition == "transformers" else generate_ollama)
    stopped = False
    for entry, record in zip(entries, manifest["records"], strict=True):
        if stopped:
            break
        directory = condition_dir / entry["document_id"]
        directory.mkdir()
        messages = messages_for(entry)
        request = ollama_request(messages) if condition == "ollama" else {"messages": messages, "model": HF_MODEL, "revision": provenance["revision"]}
        if condition == "ollama":
            input_bound = len((messages[0]["content"] + messages[1]["content"]).encode()) + 1024
            if input_bound + MAX_NEW > CONTEXT:
                raise ValueError("Conservative Ollama input bound exceeds 32K total context")
            record["input_token_upper_bound"] = input_bound
        prior.write_json(directory / "request.json", request)
        record["request_sha256"] = prior.sha((directory / "request.json").read_bytes())
        record["attempt_number"] = reserve(OUTPUT / "execution.json", condition, entry["document_id"])
        manifest["attempts"] += 1
        record.update(status="started", started_at=stamp(), host_memory_before=memory())
        prior.write_json(condition_dir / "manifest.json", manifest)
        start = time.monotonic()
        try:
            generation = generate(directory / "request.json", directory)
            record["generation"] = generation
            content = (directory / "content.txt").read_text(encoding="utf-8")
            record["content_sha256"] = prior.sha(content.encode())
            if generation["finish_reason"] not in ("eos", "stop"):
                raise ValueError("Output did not finish at native EOS/stop")
            prediction = strict_validate(content, prior.read_json(DATASET / "schema.json"))
            prior.write_json(directory / "prediction.json", prediction)
            record.update(status="success", prediction_sha256=prior.sha((directory / "prediction.json").read_bytes()))
        except (ValueError, json.JSONDecodeError) as error:
            record.update(status="invalid_output", error=f"{type(error).__name__}: {error}")
        except (Exception, KeyboardInterrupt) as error:
            record.update(status="runtime_failure", error=f"{type(error).__name__}: {error}")
            stopped = True
        finally:
            record.update(wall_seconds=time.monotonic() - start, finished_at=stamp())
            record["artifact_sha256"] = {
                p.name: prior.sha(p.read_bytes()) for p in directory.iterdir() if p.is_file()
            }
            raw_name = "generated.raw.txt" if condition == "transformers" else "response.stream.jsonl"
            record["raw_sha256"] = record["artifact_sha256"].get(raw_name)
            prior.write_json(condition_dir / "manifest.json", manifest)
    manifest.update(finished_at=stamp(), stopped=stopped)
    prior.write_json(condition_dir / "manifest.json", manifest)
    return int(stopped or any(r["status"] != "success" for r in manifest["records"]))


def score():
    entries = packet_entries()
    schema = prior.read_json(DATASET / "schema.json")
    review_path = OUTPUT / "warning_review.json"
    reviews = prior.read_json(review_path) if review_path.exists() else {}
    if set(reviews) - {"transformers", "ollama"}:
        raise ValueError("Unknown warning-review condition")
    result = {"scored_at": stamp(), "conditions": {}, "gemini": [], "historical": {}}
    conditions = ("transformers",) if OUTPUT in (RERUN_OUTPUT, EAGER_OUTPUT) else ("transformers", "ollama")
    for condition in conditions:
        run_dir = OUTPUT / condition
        manifest = prior.read_json(run_dir / "manifest.json") if run_dir.exists() else {"records": []}
        records = {r["document_id"]: r for r in manifest["records"]}
        rows = []
        for entry in entries:
            identifier = entry["document_id"]
            record = records.get(identifier, {"status": "not_attempted"})
            prediction = None
            if record.get("attempt_number"):
                directory = run_dir / identifier
                if prior.sha((directory / "request.json").read_bytes()) != record["request_sha256"]:
                    raise ValueError("Changed diagnostic request")
                for name, digest in record.get("artifact_sha256", {}).items():
                    if prior.sha((directory / name).read_bytes()) != digest:
                        raise ValueError(f"Changed diagnostic artifact: {identifier}/{name}")
            if record["status"] == "success":
                directory = run_dir / identifier
                content = (directory / "content.txt").read_text()
                if prior.sha(content.encode()) != record["content_sha256"]:
                    raise ValueError("Changed generated content")
                prediction = strict_validate(content, schema)
                if prior.sha((directory / "prediction.json").read_bytes()) != record["prediction_sha256"]:
                    raise ValueError("Changed prediction")
                if prediction != prior.read_json(directory / "prediction.json"):
                    raise ValueError("Prediction differs from raw content")
            metrics = prior.compare(prior.read_json(DATASET / entry["target"]), prediction)
            metrics.update(document_id=identifier, status=record["status"])
            condition_reviews = reviews.get(condition, {})
            if set(condition_reviews) - set(IDS):
                raise ValueError("Reserved or unknown document in warning review")
            if identifier in condition_reviews:
                prior.apply_warning_review(metrics, {"document_id": identifier, **record}, condition_reviews)
            rows.append(metrics)
        result["conditions"][condition] = {
            "smoke": rows[0], "reports": rows[1:], "summary": prior.summarize(rows[1:]),
        }
    for entry in entries[1:]:
        result["gemini"].append({"document_id": entry["document_id"], **prior.compare(
            prior.read_json(DATASET / entry["target"]),
            strict_validate((DATASET / entry["gemini"]).read_text(), schema),
        )})
    result["gemini_summary"] = prior.summarize(result["gemini"])
    old = prior.read_json(DATASET.parent / "qwen-untuned-v1/scores.json")
    if old["dataset_sha256"] != prior.sha((DATASET / "manifest.json").read_bytes()):
        raise ValueError("Historical score belongs to a different packet")
    result["historical"] = {
        "score_sha256": prior.sha((DATASET.parent / "qwen-untuned-v1/scores.json").read_bytes()),
        "summary": old["summary"]["qwen"]["pooled"],
        "reports": [{"document_id": r["document_id"], "date_exact": r["date_exact"],
                     "ordered_record_matches": r["ordered_record_matches"], "field_matches": r["field_matches"]}
                    for r in old["qwen"]],
    }
    prior.write_json(OUTPUT / "scores.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "score", "_hf_one"))
    parser.add_argument("argument", nargs="?")
    parser.add_argument("directory", nargs="?")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run":
        return run(args.argument)
    elif args.command == "score":
        score()
    else:
        hf_one(Path(args.argument), Path(args.directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
