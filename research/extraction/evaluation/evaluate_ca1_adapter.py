"""Matched base/adapter CA1 evaluation on synthetic and real report roles."""

import argparse
import importlib.metadata
import json
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from research.extraction.datasets.packet import (
    LIMIT,
    MODEL,
    PACKET,
    REVISION,
    WRAPPER_END,
    WRAPPER_START,
    load_packet,
    read_json,
    read_jsonl,
    sha,
)
from research.extraction.evaluation.scoring import compare, summarize

DEVELOPMENT_IDS = frozenset("A1 A2 A3 A4 C1 D0002 D0006 D0007 D0011 D0012 D0014 D0015 D0016".split())
RESERVED_IDS = frozenset("B1 B2 B3 B4 D0003 D0004 D0005 D0008 D0009 D0017 D0018".split())
LONG_CONTEXT_LIMIT = 16384


def cases(packet, real_data, dataset):
    """Select exactly one evaluation role without mixing real reports into the packet."""
    if dataset not in ("synthetic", "development", "reserved"):
        raise ValueError(f"Unknown evaluation dataset: {dataset}")
    prompt = (packet / "prompt.txt").read_text()
    if dataset == "synthetic":
        return [(row["id"], "synthetic-validation", row["messages"][:2],
                 json.loads(row["messages"][2]["content"]))
                for row in read_jsonl(packet / "validation.jsonl")]
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
    records = []
    for row in selected:
        identifier = row["id"]
        text = (real_data / "real" / identifier / "input.txt").read_text()
        target = read_json(real_data / "real" / identifier / "target.json")
        validator.validate(target)
        group = "real-reserved-long" if identifier == "D0018" else f"real-{dataset}"
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": WRAPPER_START + text + WRAPPER_END}]
        records.append((identifier, group, messages, target))
    return records


def model_with_adapter(adapter, device):
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


def one_response(model, tokenizer, messages, validator, *, token_limit=LIMIT):
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



def validate_saved_case(path, identifier, group, target, validator):
    """Reject incomplete or stale saved output before resume skips generation."""
    saved = read_json(path)
    required = {"id", "group", "content", "prediction", "schema_error", "input_tokens",
                "generated_tokens", "eos_reached", "score"}
    if not isinstance(saved, dict) or not required.issubset(saved):
        raise ValueError(f"Incomplete saved case: {path}")
    if saved["id"] != identifier or saved["group"] != group:
        raise ValueError(f"Saved case identity changed: {path}")
    if (not isinstance(saved["content"], str) or not isinstance(saved["input_tokens"], int)
            or not isinstance(saved["generated_tokens"], int) or not isinstance(saved["eos_reached"], bool)):
        raise ValueError(f"Malformed saved generation: {path}")
    limit = LONG_CONTEXT_LIMIT if group == "real-reserved-long" else LIMIT
    if saved["input_tokens"] < 0 or saved["generated_tokens"] < 1 or saved["input_tokens"] + saved["generated_tokens"] > limit:
        raise ValueError(f"Saved case token budget changed: {path}")
    if saved["prediction"] is None:
        if not isinstance(saved["schema_error"], str) or not saved["schema_error"]:
            raise ValueError(f"Missing saved schema error: {path}")
    else:
        validator.validate(saved["prediction"])
        if json.loads(saved["content"]) != saved["prediction"]:
            raise ValueError(f"Saved prediction differs from generated content: {path}")
        if not saved["eos_reached"] or saved["schema_error"] is not None:
            raise ValueError(f"Saved prediction lacks clean EOS/schema status: {path}")
    if saved["score"] != compare(target, saved["prediction"]):
        raise ValueError(f"Saved case score changed: {path}")
    return saved


def evaluation_settings(args, records, run_file):
    adapter_files = ("adapter_config.json", "adapter_model.safetensors")
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
        "decoding": (f"greedy, unconstrained, EOS stop, total context <={LIMIT} "
                     f"except D0018 <={LONG_CONTEXT_LIMIT}" if args.dataset == "reserved" else
                     f"greedy, unconstrained, EOS stop, total context <={LIMIT}"),
        "case_ids": [identifier for identifier, _, _, _ in records],
        "case_content_sha256": {
            identifier: sha(json.dumps([group, messages, target], ensure_ascii=False, sort_keys=True))
            for identifier, group, messages, target in records
        },
    }


def evaluate(args, tokenizer):
    run_file = args.adapter.parent / "run.json"
    run = read_json(run_file)
    expected_recipe = {"model": MODEL, "revision": REVISION, "packet_sha256": sha(args.packet / "SHA256SUMS.json")}
    if (run.get("status") != "completed" or run.get("best_checkpoint") != args.adapter.name
            or any(run["recipe"].get(key) != value for key, value in expected_recipe.items())):
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
        manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    for variant in ("base", "adapted"):
        for identifier, group, _, target in records:
            path = output / variant / f"{identifier}.json"
            if path.exists():
                if not args.resume:
                    raise FileExistsError(path)
                validate_saved_case(path, identifier, group, target, validator)
    missing = [(variant, identifier) for variant in ("base", "adapted")
               for identifier, _, _, _ in records if not (output / variant / f"{identifier}.json").exists()]
    if missing:
        model, precision = model_with_adapter(args.adapter, args.device)
        report.update(precision=precision, versions={name: importlib.metadata.version(name)
                      for name in ("torch", "transformers", "tokenizers", "peft")})
        manifest_path.write_text(json.dumps(report, indent=2) + "\n")
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
                    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                    print(f"{variant} {identifier}: {'valid' if result['prediction'] is not None else 'invalid'}", flush=True)
    summaries = {}
    for variant in ("base", "adapted"):
        summaries[variant] = {
            group: summarize([validate_saved_case(output / variant / f"{identifier}.json", identifier,
                                                   current_group, target, validator)["score"]
                              for identifier, current_group, _, target in records if current_group == group])
            for group in {group for _, group, _, _ in records}
        }
    report.update(status="completed", finished_at=datetime.now(UTC).isoformat(), summaries=summaries,
                  limitation="Reserved reports were already evaluated historically; exact scores need source review.")
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def main():
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
