"""Matched base/adapter CA1 evaluation on synthetic, development and reserved reports."""

import argparse
import importlib.metadata
import json
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from research.extraction.scoring import compare, summarize
from research.extraction.train_ca1 import (
    CORRECTION,
    LIMIT,
    MODEL,
    PACKET,
    REVISION,
    TOKENIZER,
    load_packet,
    read_json,
    read_jsonl,
    sha,
)

DEVELOPMENT_IDS = frozenset("A1 A2 A3 A4 C1 D0002 D0006 D0007 D0011 D0012 D0014 D0015 D0016".split())
RESERVED_IDS = frozenset("B1 B2 B3 B4 D0003 D0004 D0005 D0008 D0009 D0017 D0018".split())
LONG_CONTEXT_LIMIT = 16384


def cases(packet, correction, include_development, *, development_only=False, reserved_only=False):
    validation = []
    if not (development_only or reserved_only):
        for row in read_jsonl(packet / "validation.jsonl"):
            validation.append(
                (row["id"], "synthetic-validation", row["messages"][:2], json.loads(row["messages"][2]["content"]))
            )
    if not (include_development or development_only or reserved_only):
        return validation
    role = "reserved" if reserved_only else "development"
    expected_ids = RESERVED_IDS if reserved_only else DEVELOPMENT_IDS
    sources = [row for row in read_json(correction / "real_edits.json") if row["role"] == role]
    if {row["id"] for row in sources} != expected_ids or len(sources) != len(expected_ids):
        raise ValueError(f"Real {role} membership changed")
    prompt = (correction / "prompts/corrected.txt").read_text()
    for row in sources:
        identifier = row["id"]
        parser_text = (correction / "real" / identifier / "input.txt").read_text()
        target = read_json(correction / "real" / identifier / "target.json")
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": "Extract laboratory observations from this parser output:\n<report>\n"
                + parser_text
                + "\n</report>\n",
            },
        ]
        group = "real-reserved-long" if identifier == "D0018" and reserved_only else f"real-{role}"
        validation.append((identifier, group, messages, target))
    return validation


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


def evaluate(args, tokenizer):
    run_file = args.adapter.parent / "run.json"
    run = read_json(run_file)
    expected_recipe = {"model": MODEL, "revision": REVISION, "packet_sha256": sha(args.packet / "SHA256SUMS.json")}
    if (
        run.get("status") != "completed"
        or run.get("best_checkpoint") != args.adapter.name
        or any(run["recipe"].get(key) != value for key, value in expected_recipe.items())
    ):
        raise ValueError("Adapter is not the selected checkpoint for this accepted packet")
    records = cases(
        args.packet,
        args.correction,
        args.include_development,
        development_only=args.development_only,
        reserved_only=args.reserved_only,
    )
    schema = read_json(args.correction / "schema.json")
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    output = args.output
    manifest_path = output / "evaluation.json"
    settings = {
        "model": MODEL,
        "revision": REVISION,
        "adapter": str(args.adapter.resolve()),
        "adapter_run_sha256": sha(run_file),
        "packet_sha256": expected_recipe["packet_sha256"],
        "device": args.device,
        "include_development": args.include_development or args.development_only,
        "decoding": "greedy, unconstrained, EOS stop, total context <=8192",
        "case_ids": [identifier for identifier, _, _, _ in records],
    }
    if args.development_only:
        settings["development_only"] = True
    if args.reserved_only:
        settings.update(
            reserved_only=True,
            long_context_limit=LONG_CONTEXT_LIMIT,
            decoding=f"greedy, unconstrained, EOS stop, total context <={LIMIT} except D0018 <={LONG_CONTEXT_LIMIT}",
        )
    if output.exists() and any(output.iterdir()):
        if not args.resume or not manifest_path.exists() or read_json(manifest_path)["settings"] != settings:
            raise FileExistsError("Evaluation output exists with different settings; use a new directory")
        report = read_json(manifest_path)
    else:
        output.mkdir(parents=True, exist_ok=True)
        report = {"settings": settings, "started_at": datetime.now(UTC).isoformat(), "status": "running"}
        manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    model, precision = model_with_adapter(args.adapter, args.device)
    report.update(
        precision=precision,
        versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "peft")},
    )
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    for variant in ("base", "adapted"):
        (output / variant).mkdir(exist_ok=True)
        context = model.disable_adapter() if variant == "base" else nullcontext()
        with context:
            for identifier, group, messages, target in records:
                path = output / variant / f"{identifier}.json"
                if path.exists():
                    if not args.resume:
                        raise FileExistsError(path)
                    continue
                token_limit = LONG_CONTEXT_LIMIT if group == "real-reserved-long" else LIMIT
                result = one_response(model, tokenizer, messages, validator, token_limit=token_limit)
                result.update(id=identifier, group=group, score=compare(target, result["prediction"]))
                path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                print(
                    f"{variant} {identifier}: {'valid' if result['prediction'] is not None else 'invalid'}", flush=True
                )
    summaries = {}
    for variant in ("base", "adapted"):
        summaries[variant] = {
            group: summarize(
                [
                    read_json(output / variant / f"{identifier}.json")["score"]
                    for identifier, current_group, _, _ in records
                    if current_group == group
                ]
            )
            for group in {group for _, group, _, _ in records}
        }
    report.update(
        status="completed",
        finished_at=datetime.now(UTC).isoformat(),
        summaries=summaries,
        limitation=(
            "Reserved cases were not used for training or checkpoint selection; D0018 has a separate long-context budget. "
            "Exact scores and warnings require source-aligned manual review."
            if args.reserved_only
            else "Validation selected the checkpoint; exact scores and warnings require source-aligned manual review. "
            "Real development reports supplied layout conventions. Reserved reports were not used."
        ),
    )
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=PACKET)
    parser.add_argument("--correction", type=Path, default=CORRECTION)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "mps"), required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--include-development", action="store_true")
    selection.add_argument("--development-only", action="store_true")
    selection.add_argument("--reserved-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    _, tokenizer, _ = load_packet(args.packet, args.correction, args.tokenizer)
    evaluate(args, tokenizer)


if __name__ == "__main__":
    main()
