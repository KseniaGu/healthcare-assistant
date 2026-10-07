"""Matched base/adapter CA1 evaluation on synthetic and real report roles.

Model imports stay in generation paths so saved-case checks and CLI help work without adapter packages.
"""

import argparse
import importlib.metadata
import json
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, TypedDict, cast

from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from research.extraction.datasets.packet import (
    LIMIT,
    MODEL,
    PACKET,
    REVISION,
    WRAPPER_END,
    WRAPPER_START,
    CA1Target,
    ChatMessage,
    load_packet,
    read_json,
    read_jsonl,
    sha,
)
from research.extraction.evaluation.scoring import Score, compare, summarize

DEVELOPMENT_IDS = frozenset("A1 A2 A3 A4 C1 D0002 D0006 D0007 D0011 D0012 D0014 D0015 D0016".split())
RESERVED_IDS = frozenset("B1 B2 B3 B4 D0003 D0004 D0005 D0008 D0009 D0017 D0018".split())
LONG_CONTEXT_LIMIT = 16384


class EvaluationCase(NamedTuple):
    identifier: str
    group: str
    messages: list[ChatMessage]
    target: CA1Target


class SavedCase(TypedDict):
    id: str
    group: str
    content: str
    prediction: CA1Target | None
    schema_error: str | None
    input_tokens: int
    generated_tokens: int
    eos_reached: bool
    score: Score


def cases(packet: Path, real_data: Path | None, dataset: str) -> list[EvaluationCase]:
    """Selects one evaluation role without mixing real reports into the packet.

    Args:
        real_data: Separate real-report archive, required for development or reserved.
        dataset: Exactly one of synthetic, development, or reserved.

    Returns:
        Synthetic validation cases or the exact approved development/reserved IDs.
        Real inputs and targets are loaded from the separate evaluation archive.

    Raises:
        ValueError: The role is unknown, real data is missing, or the real-role
            index contains extra, missing, or reassigned cases.
    """
    if dataset not in ("synthetic", "development", "reserved"):
        raise ValueError(f"Unknown evaluation dataset: {dataset}")
    prompt = (packet / "prompt.txt").read_text()
    if dataset == "synthetic":
        return [
            EvaluationCase(
                row["id"], "synthetic-validation", row["messages"][:2], json.loads(row["messages"][2]["content"])
            )
            for row in read_jsonl(packet / "validation.jsonl")
        ]
    if real_data is None:
        raise ValueError(f"{dataset} evaluation requires --real-data")
    index = read_json(real_data / "real_edits.json")
    if not isinstance(index, list) or any(set(row) != {"id", "role"} for row in index):
        raise ValueError("Real evaluation index must contain only ID and role")
    for role, identifiers in (("development", DEVELOPMENT_IDS), ("reserved", RESERVED_IDS)):
        members = [row["id"] for row in index if row["role"] == role]
        if len(members) != len(identifiers) or set(members) != identifiers:
            raise ValueError(f"Real {role} membership changed")
    if len(index) != len(DEVELOPMENT_IDS) + len(RESERVED_IDS):
        raise ValueError("Real evaluation index has extra cases")

    selected = [row for row in index if row["role"] == dataset]
    validator = Draft202012Validator(read_json(packet / "schema.json"), format_checker=FormatChecker())
    records: list[EvaluationCase] = []
    for row in selected:
        identifier = row["id"]
        text = (real_data / "real" / identifier / "input.txt").read_text()
        target = read_json(real_data / "real" / identifier / "target.json")
        validator.validate(target)
        group = "real-reserved-long" if identifier == "D0018" else f"real-{dataset}"
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content=WRAPPER_START + text + WRAPPER_END),
        ]
        records.append(EvaluationCase(identifier, group, messages, target))
    return records


def model_with_adapter(adapter: Path, device: str) -> tuple[Any, str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype
        )
        base = AutoModelForCausalLM.from_pretrained(
            MODEL,
            revision=REVISION,
            dtype=dtype,
            quantization_config=quantization,
            device_map={"": torch.cuda.current_device()},
        )
        precision = f"NF4/{dtype}"
    elif device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Mac MPS is unavailable")
        base = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.bfloat16, device_map={"": "mps"}
        )
        precision = "BF16/MPS"
    else:
        raise ValueError("Choose cuda or mps")
    model = PeftModel.from_pretrained(base, adapter)
    model.config.use_cache = True
    model.eval()
    return model, precision


def one_response(
    model: Any,
    tokenizer: Any,
    messages: list[ChatMessage],
    validator: Draft202012Validator,
    *,
    token_limit: int = LIMIT,
) -> dict[str, Any]:
    """Generates one greedy completion and validates its exact JSON content.

    Generation stops on EOS or the remaining context budget, without sampling.

    Args:
        messages: System/user prompt without an assistant target.
        token_limit: Total input-plus-output budget; the reserved D0018 case
            receives a larger limit from its caller.

    Returns:
        Decoded content, parsed prediction or schema error, token counts, and
        whether EOS was reached. A valid JSON object without EOS remains invalid.

    Raises:
        ValueError: The input alone reaches the total token budget.

    """
    import torch

    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    prefix = encoded["input_ids"].shape[-1]
    if prefix >= token_limit:
        raise ValueError(f"Evaluation input exceeds {token_limit} tokens")

    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=token_limit - prefix,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    eos_reached = generated[0, -1].item() == tokenizer.eos_token_id
    content = tokenizer.decode(generated[0, prefix:], skip_special_tokens=True).strip()

    try:
        if not eos_reached:
            raise ValueError(f"Generation reached the {token_limit}-token limit without EOS")
        prediction = json.loads(content)
        validator.validate(prediction)
        error = None
    except (ValueError, TypeError, ValidationError) as exc:
        prediction, error = None, str(exc)

    return {
        "content": content,
        "prediction": prediction,
        "schema_error": error,
        "input_tokens": prefix,
        "generated_tokens": generated.shape[-1] - prefix,
        "eos_reached": eos_reached,
    }


def validate_saved_case(
    path: Path, identifier: str, group: str, target: CA1Target, validator: Draft202012Validator
) -> SavedCase:
    """Revalidates a saved case before resume skips generation.

    Invalid generations are checked by validity outcome, not error-message text.

    Args:
        group: Selects the ordinary or reserved long-context token budget.
        target: Current gold target used to recompute the stored score.

    Returns:
        The stored case only if identity, token budget, content/schema outcome,
        EOS status, and recomputed score still agree with the current inputs.

    Raises:
        ValueError: A saved field or derived outcome is stale or malformed.

    """
    saved = read_json(path)
    required = {
        "id",
        "group",
        "content",
        "prediction",
        "schema_error",
        "input_tokens",
        "generated_tokens",
        "eos_reached",
        "score",
    }
    if not isinstance(saved, dict) or not required.issubset(saved):
        raise ValueError(f"Incomplete saved case: {path}")

    if saved["id"] != identifier or saved["group"] != group:
        raise ValueError(f"Saved case identity changed: {path}")
    if (
        not isinstance(saved["content"], str)
        or not isinstance(saved["input_tokens"], int)
        or not isinstance(saved["generated_tokens"], int)
        or not isinstance(saved["eos_reached"], bool)
    ):
        raise ValueError(f"Malformed saved generation: {path}")
    limit = LONG_CONTEXT_LIMIT if group == "real-reserved-long" else LIMIT
    if (
        saved["input_tokens"] < 0
        or saved["generated_tokens"] < 1
        or saved["input_tokens"] + saved["generated_tokens"] > limit
    ):
        raise ValueError(f"Saved case token budget changed: {path}")

    if saved["prediction"] is None:
        if not isinstance(saved["schema_error"], str) or not saved["schema_error"]:
            raise ValueError(f"Missing saved schema error: {path}")
        if saved["eos_reached"]:
            try:
                validator.validate(json.loads(saved["content"]))
            except (ValueError, TypeError, ValidationError):
                pass
            else:
                raise ValueError(f"Saved invalid generation now satisfies the schema: {path}")
    else:
        validator.validate(saved["prediction"])
        if json.loads(saved["content"]) != saved["prediction"]:
            raise ValueError(f"Saved prediction differs from generated content: {path}")
        if not saved["eos_reached"] or saved["schema_error"] is not None:
            raise ValueError(f"Saved prediction lacks clean EOS/schema status: {path}")
    if saved["score"] != compare(target, saved["prediction"]):
        raise ValueError(f"Saved case score changed: {path}")
    return cast(SavedCase, saved)


def evaluation_settings(args: argparse.Namespace, records: list[EvaluationCase], run_file: Path) -> dict[str, Any]:
    """Fingerprints the inputs and decoding choices that must match on resume.

    Returns:
        Adapter, run, packet, and per-case hashes plus role, device, and
        decoding settings; a changed value requires a new output directory.
    """
    adapter_files = ("adapter_config.json", "adapter_model.safetensors")
    decoding = f"greedy, unconstrained, EOS stop, total context <={LIMIT}"
    if args.dataset == "reserved":
        decoding += f" except D0018 <={LONG_CONTEXT_LIMIT}"
    return {
        "model": MODEL,
        "revision": REVISION,
        "adapter": str(args.adapter.resolve()),
        "adapter_sha256": {name: sha(args.adapter / name) for name in adapter_files},
        "adapter_run_sha256": sha(run_file),
        "packet_sha256": sha(args.packet / "SHA256SUMS.json"),
        "prompt_sha256": sha(args.packet / "prompt.txt"),
        "schema_sha256": sha(args.packet / "schema.json"),
        "device": args.device,
        "dataset": args.dataset,
        "decoding": decoding,
        "case_ids": [identifier for identifier, _, _, _ in records],
        "case_content_sha256": {
            identifier: sha(json.dumps([group, messages, target], ensure_ascii=False, sort_keys=True))
            for identifier, group, messages, target in records
        },
    }


def _validate_existing_cases(
    output: Path, records: list[EvaluationCase], validator: Draft202012Validator, resume: bool
) -> None:
    for variant in ("base", "adapted"):
        for identifier, group, _, target in records:
            path = output / variant / f"{identifier}.json"
            if path.exists():
                if not resume:
                    raise FileExistsError(path)
                validate_saved_case(path, identifier, group, target, validator)


def _saved_summaries(
    output: Path, records: list[EvaluationCase], validator: Draft202012Validator
) -> dict[str, dict[str, dict[str, int | float]]]:
    summaries: dict[str, dict[str, dict[str, int | float]]] = {}
    for variant in ("base", "adapted"):
        group_summaries: dict[str, dict[str, int | float]] = {}
        for group in {record.group for record in records}:
            scores: list[Score] = []
            for record in records:
                if record.group != group:
                    continue
                path = output / variant / f"{record.identifier}.json"
                saved = validate_saved_case(path, record.identifier, group, record.target, validator)
                scores.append(saved["score"])
            group_summaries[group] = summarize(scores)
        summaries[variant] = group_summaries
    return summaries


def _write_json_atomic(path: Path, value: object, *, ensure_ascii: bool = True) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=ensure_ascii, indent=2) + "\n")
    temporary.replace(path)


def evaluate(args: argparse.Namespace, tokenizer: Any) -> None:
    """Evaluates matched base and adapted cases with resumable output.

    Only missing cases are generated; completed cases are rescored from saved
    content before final summaries are written.

    Raises:
        ValueError: The adapter is not selected for this packet, or a saved
            case fails the resume checks.
        jsonschema.ValidationError: A saved prediction fails the schema.
        FileExistsError: Existing output has different settings or resume was
            not requested.

    """
    run_file = args.adapter.parent / "run.json"
    run = read_json(run_file)
    expected_recipe = {"model": MODEL, "revision": REVISION, "packet_sha256": sha(args.packet / "SHA256SUMS.json")}
    if (
        run.get("status") != "completed"
        or run.get("best_checkpoint") != args.adapter.name
        or any(run["recipe"].get(key) != value for key, value in expected_recipe.items())
    ):
        raise ValueError("Adapter is not the selected checkpoint for this accepted packet")

    records = cases(args.packet, args.real_data, args.dataset)
    validator = Draft202012Validator(read_json(args.packet / "schema.json"), format_checker=FormatChecker())
    output = args.output
    manifest_path = output / "evaluation.json"
    settings = evaluation_settings(args, records, run_file)

    if output.exists() and any(output.iterdir()):
        if not args.resume or not manifest_path.exists() or read_json(manifest_path)["settings"] != settings:
            raise FileExistsError("Evaluation output exists with different settings; use a new directory")
        report = read_json(manifest_path)
    else:
        output.mkdir(parents=True, exist_ok=True)
        report = {"settings": settings, "started_at": datetime.now(UTC).isoformat(), "status": "running"}
        _write_json_atomic(manifest_path, report)

    _validate_existing_cases(output, records, validator, args.resume)
    missing = [
        (variant, identifier)
        for variant in ("base", "adapted")
        for identifier, _, _, _ in records
        if not (output / variant / f"{identifier}.json").exists()
    ]
    if missing:
        model, precision = model_with_adapter(args.adapter, args.device)
        report.update(
            precision=precision,
            versions={
                name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "peft")
            },
        )
        _write_json_atomic(manifest_path, report)
        for variant in ("base", "adapted"):
            (output / variant).mkdir(exist_ok=True)
            context = model.disable_adapter() if variant == "base" else nullcontext()
            with context:
                for identifier, group, messages, target in records:
                    path = output / variant / f"{identifier}.json"
                    if path.exists():
                        continue
                    token_limit = LONG_CONTEXT_LIMIT if group == "real-reserved-long" else LIMIT
                    result = one_response(model, tokenizer, messages, validator, token_limit=token_limit)
                    result.update(id=identifier, group=group, score=compare(target, result["prediction"]))
                    _write_json_atomic(path, result, ensure_ascii=False)
                    status = "valid" if result["prediction"] is not None else "invalid"
                    print(f"{variant} {identifier}: {status}", flush=True)

    summaries = _saved_summaries(output, records, validator)
    report.update(status="completed", finished_at=datetime.now(UTC).isoformat(), summaries=summaries)
    if args.dataset == "reserved":
        report["limitation"] = "Reserved reports were already evaluated historically; exact scores need source review."
    else:
        report.pop("limitation", None)
    _write_json_atomic(manifest_path, report, ensure_ascii=False)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=PACKET)
    parser.add_argument("--real-data", type=Path)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "mps"), required=True)
    parser.add_argument("--dataset", choices=("synthetic", "development", "reserved"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    _, tokenizer, _ = load_packet(args.packet)
    evaluate(args, tokenizer)


if __name__ == "__main__":
    main()
