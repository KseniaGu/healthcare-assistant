"""Check an accepted CA1 packet and run its pinned CUDA fine-tuning recipe."""

import argparse
import importlib.metadata
import json
import math
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from research.extraction.datasets.packet import LIMIT, MODEL, PACKET, REVISION, load_packet, read_json, sha


def current_source_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

def collate_batch(rows, pad_id):
    """Right-pad cached conversations while ignoring every padded target token."""
    import torch

    width = max(len(row["input_ids"]) for row in rows)
    return {
        "input_ids": torch.tensor(
            [row["input_ids"] + [pad_id] * (width - len(row["input_ids"])) for row in rows], dtype=torch.long
        ),
        "labels": torch.tensor(
            [row["labels"] + [-100] * (width - len(row["labels"])) for row in rows], dtype=torch.long
        ),
        "attention_mask": torch.tensor(
            [[1] * len(row["input_ids"]) + [0] * (width - len(row["input_ids"])) for row in rows], dtype=torch.long
        ),
    }


class TokenRows:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {key: row[key] for key in ("input_ids", "labels")}


def load_model(dtype):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        quantization_config=quantization,
        dtype=dtype,
        device_map={"": torch.cuda.current_device()},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    return get_peft_model(
        model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
    )


def cuda_dtype():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires one CUDA GPU; this container has none")
    capability = torch.cuda.get_device_capability()
    cuda_version = torch.version.cuda
    if cuda_version is None:
        raise RuntimeError("CUDA runtime version is unavailable")
    cuda = tuple(int(part) for part in cuda_version.split("."))
    if capability[0] == 6 and cuda >= (12, 8):
        raise RuntimeError("P100/sm60 needs a bitsandbytes CUDA 11.8–12.6 wheel")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The matched CA1 recipe requires CUDA BF16")
    return torch.bfloat16


def preflight(rows, tokenizer):
    import torch

    dtype = cuda_dtype()
    model = load_model(dtype)
    longest = max(rows, key=lambda row: len(row["input_ids"]))
    batch = {key: value.cuda() for key, value in collate_batch([longest], tokenizer.pad_token_id).items()}
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    model.train()
    loss = model(**batch).loss
    if not torch.isfinite(loss):
        raise ValueError("Longest-example preflight loss is not finite")
    loss.backward()
    torch.cuda.synchronize()
    result = {
        "id": longest["id"],
        "tokens": len(longest["input_ids"]),
        "seconds": round(time.monotonic() - start, 2),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "gpu_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
    }
    print(json.dumps(result, indent=2))


def train(args, splits, tokenizer):
    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    dtype = cuda_dtype()
    output = args.output
    run_file = output / "run.json"
    recipe = {
        "model": MODEL,
        "revision": REVISION,
        "packet_sha256": sha(args.packet / "SHA256SUMS.json"),
        "seed": 42,
        "trainer_code_sha256": sha(Path(__file__)),
        "source_commit": current_source_commit(),
        "epochs": 3,
        "microbatch": 1,
        "accumulation": 8,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "scheduler": "cosine",
        "warmup_steps": 4,
        "eval_steps": 40,
        "save_steps": 10,
        "lora_target_modules": "all-linear",
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "quantization": "NF4+double",
        "compute_dtype": str(dtype),
        "report_to": args.report_to,
        "max_tokens": LIMIT,
    }
    if recipe["source_commit"] != read_json(args.packet / "manifest.json")["source_commit"]:
        raise ValueError("Training source revision differs from the accepted packet")
    if args.resume:
        if not run_file.exists() or read_json(run_file)["recipe"] != recipe:
            raise ValueError("Resume requires an existing run with the identical recipe and packet")
        if not args.resume.is_dir() or args.resume.resolve().parent != output.resolve():
            raise ValueError("Resume checkpoint must belong to this output directory")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    info = read_json(run_file) if args.resume else {"recipe": recipe, "started_at": datetime.now(UTC).isoformat()}
    runtime = {
        "gpu": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers", "peft", "accelerate", "bitsandbytes")
            + (("wandb",) if args.report_to == "wandb" else ())
        },
    }
    if args.resume and info["status"] == "completed":
        raise ValueError("Run is already complete")
    if args.resume and any(info[key] != value for key, value in runtime.items()):
        raise ValueError("Resume requires the same CUDA hardware and dependency versions")
    prior_elapsed = info.get("elapsed_seconds", 0)
    prior_peak = info.get("peak_allocated_gib", 0)
    info.update(status="running", **runtime)
    run_file.write_text(json.dumps(info, indent=2) + "\n")
    started = time.monotonic()
    try:
        set_seed(recipe["seed"])
        model = load_model(dtype)
        settings = TrainingArguments(
            output_dir=str(output),
            num_train_epochs=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_steps=4,
            optim="adamw_torch",
            weight_decay=0.01,
            fp16=dtype == torch.float16,
            bf16=dtype == torch.bfloat16,
            gradient_checkpointing=True,
            eval_strategy="steps",
            eval_steps=40,
            prediction_loss_only=True,
            save_strategy="steps",
            save_steps=10,
            logging_steps=10,
            save_total_limit=13,
            seed=42,
            data_seed=42,
            report_to=args.report_to,
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model,
            args=settings,
            train_dataset=TokenRows(splits["train"]),
            eval_dataset=TokenRows(splits["validation"]),
            data_collator=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
        )
        trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
        epochs = [
            entry for entry in trainer.state.log_history if "eval_loss" in entry and entry.get("step") in (40, 80, 120)
        ]
        if len(epochs) != 3 or any(not math.isfinite(entry["eval_loss"]) for entry in epochs):
            raise ValueError("Missing finite validation losses at epoch checkpoints")
        best = min(epochs, key=lambda entry: entry["eval_loss"])
        checkpoint = output / f"checkpoint-{best['step']}"
        if not checkpoint.is_dir():
            raise ValueError(f"Selected checkpoint missing: {checkpoint}")
        info.update(
            status="completed",
            best_checkpoint=checkpoint.name,
            best_validation_loss=best["eval_loss"],
            validation_losses=epochs,
        )
    finally:
        info.update(
            finished_at=datetime.now(UTC).isoformat(),
            elapsed_seconds=round(prior_elapsed + time.monotonic() - started, 1),
            peak_allocated_gib=max(prior_peak, round(torch.cuda.max_memory_allocated() / 2**30, 2)),
        )
        if info["status"] != "completed":
            info["status"] = "interrupted"
        run_file.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({"best_checkpoint": info["best_checkpoint"], "validation_loss": info["best_validation_loss"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "preflight", "train"))
    parser.add_argument("--packet", type=Path, default=PACKET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--report-to", choices=("none", "wandb"), default="none")
    args = parser.parse_args()
    splits, tokenizer, _ = load_packet(args.packet)
    if args.stage == "check":
        print(
            f"Accepted CA1 packet: {len(splits['train'])} train, {len(splits['validation'])} validation; "
            f"max {max(len(row['input_ids']) for rows in splits.values() for row in rows)} tokens"
        )
    elif args.stage == "preflight":
        preflight(splits["train"], tokenizer)
    elif args.output is None:
        parser.error("train requires --output")
    else:
        train(args, splits, tokenizer)


if __name__ == "__main__":
    main()
