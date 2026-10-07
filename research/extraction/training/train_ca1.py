"""Check an accepted CA1 packet and run its pinned CUDA fine-tuning recipe.

Model-stack imports stay in execution paths so packet checks and CLI help do not require the VM-only PEFT stack.
"""

import argparse
import importlib.metadata
import json
import math
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research.extraction.datasets.packet import (
    LIMIT,
    MODEL,
    PACKET,
    REVISION,
    ROOT,
    TokenRow,
    TrainingRow,
    load_packet,
    read_json,
    sha,
)

TRAINING_SEED = 42
EPOCHS = 3
MICROBATCH = 1
ACCUMULATION = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 4
EVAL_STEPS = 40
SAVE_STEPS = 10
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"
SCHEDULER = "cosine"
# 320 train rows at microbatch 1 and accumulation 8 give 40 optimizer steps per epoch.
EPOCH_CHECKPOINT_STEPS = (40, 80, 120)


def current_source_commit() -> str:
    """Reads HEAD only when tracked repository source is clean.

    Returns:
        Commit recorded in the run recipe and compared with packet provenance.

    Raises:
        ValueError: Any tracked repository file has uncommitted changes.
    """
    changed = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
    if changed:
        raise ValueError("Training source has uncommitted tracked changes")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def collate_batch(rows: list[TrainingRow], pad_id: int) -> dict[str, Any]:
    """Right-pad cached conversations without supervising padding.

    Returns:
        Long tensors of shape (batch, longest_sequence) for input IDs, labels,
        and attention mask. Padding labels are -100; padding attention is zero.
    """
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
    def __init__(self, rows: list[TokenRow]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> TrainingRow:
        row = self.rows[index]
        return {"input_ids": row["input_ids"], "labels": row["labels"]}


def load_model(dtype: Any) -> Any:
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
        model,
        LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        ),
    )


def cuda_dtype() -> Any:
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


def preflight(rows: list[TokenRow], tokenizer: Any) -> None:
    """Runs one training-mode forward/backward pass on the longest case.

    The clean-source gate runs before any model or CUDA work. Printed
    measurements report elapsed seconds and allocated/total GPU GiB.

    Raises:
        ValueError: Source is dirty or the loss is nonfinite.
        RuntimeError: CUDA or the required BF16 training path is unavailable.
    """
    current_source_commit()
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


def select_checkpoint(history: list[dict[str, Any]], output: Path) -> tuple[Path, float, list[dict[str, Any]]]:
    """Selects the lowest-loss checkpoint from eligible epoch-step log entries.

    Returns:
        Checkpoint path, its validation loss, and the three eligible loss entries.

    Raises:
        ValueError: The number of eligible entries is not three, a loss is
            nonfinite, or the chosen checkpoint directory is absent.
    """
    losses = [entry for entry in history if "eval_loss" in entry and entry.get("step") in EPOCH_CHECKPOINT_STEPS]
    if len(losses) != len(EPOCH_CHECKPOINT_STEPS) or any(not math.isfinite(entry["eval_loss"]) for entry in losses):
        raise ValueError("Missing finite validation losses at epoch checkpoints")
    best = min(losses, key=lambda entry: entry["eval_loss"])
    checkpoint = output / f"checkpoint-{best['step']}"
    if not checkpoint.is_dir():
        raise ValueError(f"Selected checkpoint missing: {checkpoint}")
    return checkpoint, best["eval_loss"], losses


def run_recipe(args: argparse.Namespace, dtype: Any, source_commit: str) -> dict[str, Any]:
    """Records selected settings and source identifiers used to validate resume.

    Returns:
        Recipe metadata binding the packet digest, trainer code hash, clean
        source commit, pinned hyperparameters, and selected reporting backend.
    """
    return {
        "model": MODEL,
        "revision": REVISION,
        "packet_sha256": sha(args.packet / "SHA256SUMS.json"),
        "seed": TRAINING_SEED,
        "trainer_code_sha256": sha(Path(__file__)),
        "source_commit": source_commit,
        "epochs": EPOCHS,
        "microbatch": MICROBATCH,
        "accumulation": ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": SCHEDULER,
        "warmup_steps": WARMUP_STEPS,
        "eval_steps": EVAL_STEPS,
        "save_steps": SAVE_STEPS,
        "lora_target_modules": LORA_TARGET_MODULES,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "quantization": "NF4+double",
        "compute_dtype": str(dtype),
        "report_to": args.report_to,
        "max_tokens": LIMIT,
    }


def write_run_file(path: Path, info: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(info, indent=2) + "\n")
    temporary.replace(path)


def train(args: argparse.Namespace, splits: dict[str, list[TokenRow]], tokenizer: Any) -> None:
    """Runs the pinned CA1 recipe and persists resumable run metadata.

    The selected checkpoint has the lowest finite validation loss at an epoch boundary.

    Args:
        args: Resume only from a checkpoint inside the same output directory.
        splits: Accepted, validated token rows; only their labels carry loss.

    Raises:
        ValueError: Resume recipe, packet, hardware, or dependency versions differ.
        FileExistsError: A new run would reuse a nonempty output directory.
        RuntimeError: The required CUDA/BF16 path is unavailable.

    """
    source_commit = current_source_commit()
    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    dtype = cuda_dtype()
    output = args.output
    run_file = output / "run.json"
    recipe = run_recipe(args, dtype, source_commit)
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
    write_run_file(run_file, info)

    started = time.monotonic()
    try:
        set_seed(recipe["seed"])
        model = load_model(dtype)
        settings = TrainingArguments(
            output_dir=str(output),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=MICROBATCH,
            per_device_eval_batch_size=MICROBATCH,
            gradient_accumulation_steps=ACCUMULATION,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type=SCHEDULER,
            warmup_steps=WARMUP_STEPS,
            optim="adamw_torch",
            weight_decay=WEIGHT_DECAY,
            fp16=dtype == torch.float16,
            bf16=dtype == torch.bfloat16,
            gradient_checkpointing=True,
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            prediction_loss_only=True,
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            logging_steps=10,
            save_total_limit=13,
            seed=TRAINING_SEED,
            data_seed=TRAINING_SEED,
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

        checkpoint, best_loss, epochs = select_checkpoint(trainer.state.log_history, output)
        info.update(
            status="completed",
            best_checkpoint=checkpoint.name,
            best_validation_loss=best_loss,
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
        write_run_file(run_file, info)

    print(json.dumps({"best_checkpoint": info["best_checkpoint"], "validation_loss": info["best_validation_loss"]}))


def main() -> None:
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
