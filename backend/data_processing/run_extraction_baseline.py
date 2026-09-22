"""Run explicit cached parser inputs through Gemini without PDF or database processing."""

import argparse
import fcntl
import hashlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from backend.models.llm_clients.gemini import GeminiClient
from backend.utils.extraction import parse_and_validate_extraction
from backend.utils.prompt_engine import PromptEngine
from configs.constants import DATA_DIR, ROOT_DIR
from configs.enums import PromptTypeEnum
from configs.models import ModelSettings

MAX_DEVELOPMENT_ATTEMPTS = 9


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def run_baseline(
    inputs: list[Path],
    output_dir: Path,
    prompt_version: str,
    client: GeminiClient,
    prompt_engine: PromptEngine,
    *,
    budget_file: Path,
    max_attempts: int,
    interval_seconds: float,
) -> int:
    """Capture fresh outputs; reserve attempts before sending, including failed requests.

    The caller must verify free-tier availability and remaining request/token allowance.
    A shared, locked budget file caps the whole development loop, across prompt versions.
    """
    if not inputs or not 1 <= max_attempts <= MAX_DEVELOPMENT_ATTEMPTS:
        raise ValueError("Provide inputs and a request budget between 1 and 9.")
    if not math.isfinite(interval_seconds) or interval_seconds < 0:
        raise ValueError("Request interval must be finite and nonnegative.")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    prompt_type = PromptTypeEnum.laboratory_test_processor
    template_path = prompt_engine.prompt_dir / prompt_type.value / f"{prompt_version}.yaml"
    template = template_path.read_bytes()
    prepared, seen = [], set()
    for path in inputs:
        content = path.read_bytes()
        digest = sha256(content)
        if digest in seen:
            raise ValueError(f"Duplicate input contents: {path}")
        seen.add(digest)
        text = content.decode("utf-8")
        if not text.strip():
            raise ValueError(f"Empty parser input: {path}")
        prompt, system, _ = prompt_engine.render(prompt_type, prompt_version, input_text=text)
        prepared.append((path, digest, prompt, system))

    budget_file.parent.mkdir(parents=True, exist_ok=True)
    # One local runner at a time keeps the nine-attempt budget and pacing meaningful.
    with budget_file.open("a+", encoding="utf-8") as budget_handle:
        fcntl.flock(budget_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget_handle.seek(0)
        existing = budget_handle.read()
        budget = json.loads(existing) if existing else {"attempts": 0, "stopped": None}
        if budget["stopped"] or budget["attempts"] >= MAX_DEVELOPMENT_ATTEMPTS:
            raise ValueError("Development budget is exhausted or stopped; do not reset it to bypass the limit.")

        def save_budget():
            budget_handle.seek(0)
            budget_handle.truncate()
            budget_handle.write(json.dumps(budget, indent=2) + "\n")
            budget_handle.flush()

        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "prompt.yaml").write_bytes(template)
        manifest = {
            "started_at": utc_now(),
            "prompt_version": prompt_version,
            "prompt_sha256": sha256(template),
            "model": client.config.model_name,
            "generation_settings": client.params.model_dump(mode="json", exclude_none=True),
            "sdk_attempts_per_request": 1,
            "max_attempts": max_attempts,
            "interval_seconds": interval_seconds,
            "budget_file": str(budget_file.resolve()),
            "records": [],
        }
        manifest_path = output_dir / "manifest.json"
        write_json(manifest_path, manifest)
        attempts = 0
        for index, (path, digest, prompt, system) in enumerate(prepared):
            record = {"input": str(path.resolve()), "input_sha256": digest, "status": "not_attempted"}
            manifest["records"].append(record)
            if budget["stopped"] or attempts >= max_attempts or budget["attempts"] >= MAX_DEVELOPMENT_ATTEMPTS:
                record["error"] = budget["stopped"] or "Request budget exhausted."
                continue
            # Persisted timestamp also paces consecutive v3/v4 invocations.
            delay = interval_seconds - (time.time() - budget.get("last_attempt_at", 0))
            if delay > 0:
                time.sleep(delay)
            directory = output_dir / f"{index + 1:03d}"
            directory.mkdir()
            request = {"model": client.config.model_name, "prompt": prompt, **system}
            write_json(directory / "request.json", request)
            record.update(
                started_at=utc_now(),
                request_path=str((directory / "request.json").relative_to(output_dir)),
                schema_sha256=sha256(json.dumps(system["response_json_schema"], sort_keys=True).encode()),
                status="started",
            )
            client.path_settings = SimpleNamespace(llm_outputs_dir=directory / "raw")
            budget["attempts"] += 1
            budget["last_attempt_at"] = time.time()
            attempts += 1
            save_budget()
            write_json(manifest_path, manifest)
            try:
                response, usage, _ = client.generate(
                    prompt, prompt_type, prompt_version, **system, model_name=client.config.model_name
                )
                record["usage"] = usage
                prediction = parse_and_validate_extraction(response, system["response_json_schema"])
                write_json(directory / "prediction.json", prediction)
                record.update(status="success", prediction_path=f"{index + 1:03d}/prediction.json")
            except Exception as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}")
                cause = error
                while cause is not None:
                    # Authentication/billing/permission errors also stop the batch, never trigger a fallback.
                    if getattr(cause, "code", None) in (401, 402, 403, 429):
                        budget["stopped"] = f"API HTTP {cause.code}; live testing stopped."
                        save_budget()
                        break
                    cause = cause.__cause__
            finally:
                record["raw_paths"] = [str(p.relative_to(output_dir)) for p in sorted(directory.rglob("*.txt"))]
                record["finished_at"] = utc_now()
                write_json(manifest_path, manifest)
        manifest.update(finished_at=utc_now(), attempts=attempts, development_attempts=budget["attempts"])
        write_json(manifest_path, manifest)
        return int(any(record["status"] != "success" for record in manifest["records"]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, dest="inputs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-version", choices=("v3", "v4"), default="v4")
    parser.add_argument("--model", help="Defaults to the configured processor model; no automatic fallback.")
    parser.add_argument("--budget-file", type=Path, default=DATA_DIR / "extraction_poc_budget.json")
    parser.add_argument("--max-attempts", type=int, choices=range(1, 10), default=3)
    parser.add_argument(
        "--interval-seconds", type=float, required=True, help="Pacing chosen from verified RPM/TPM limits."
    )
    parser.add_argument(
        "--quota-verified",
        action="store_true",
        required=True,
        help="Confirm free model access, remaining allowance, and input sizes fit current AI Studio limits.",
    )
    args = parser.parse_args(argv)
    config = ModelSettings().data_processor.model_copy()
    if args.model:
        config.model_name = args.model
    # A baseline must use plain extraction, even if the application's environment enables grounding.
    config.use_grounding = False
    client = GeminiClient(config, SimpleNamespace(llm_outputs_dir=args.output_dir), retry_attempts=1)
    try:
        return run_baseline(
            args.inputs,
            args.output_dir,
            args.prompt_version,
            client,
            PromptEngine(ROOT_DIR / "backend/prompts"),
            budget_file=args.budget_file,
            max_attempts=args.max_attempts,
            interval_seconds=args.interval_seconds,
        )
    finally:
        client.client.close()


if __name__ == "__main__":
    raise SystemExit(main())
