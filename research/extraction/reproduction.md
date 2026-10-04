# Reproducing the maintained CA1 extraction experiment

**Status — 2026-10-04:** Section A completed locally: `ca1-sft-v3` was generated from committed source, matched the reviewed v2 packet model-facing content, and was accepted through separate parity evidence. Its offline `train_ca1 check` passes. Sections B and C remain manual cloud procedures; no refactored model training or adapter-dependent evaluation has run. Offline CLI and packet checks do not verify future VM execution.

This is the canonical **operational** guide. Dataset design, source roles, and the v1 audit belong in [datasets/dataset.md](datasets/dataset.md); training recipe and environment rationale belong in [training/training.md](training/training.md); new-run scores and interpretation belong in [evaluation/evaluations.md](evaluation/evaluations.md), with the recorded CA1 baseline in [historical evaluations](evaluation/historical_evaluations.md). The [implementation plan](implementation_plan.md) defines migration gates. All detailed medical data and model artifacts remain under gitignored `data/`.

## Fixed contract for this reproduction

- Model: `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
- Packet: 320 fictitious training and 64 fictitious validation reports, including the corrected 16/8 smoke subset. No real reports are in training JSONL. The existing CA1 prompt/schema, messages, targets, IDs/splits, cached IDs/labels, and 8,192-token limit remain model-facing parity targets.
- Training comparison platform: one NVIDIA L4 with CUDA BF16 and the pinned dependency stack. Training, longest-example preflight, and model-dependent evaluation run on the cloud VM, not in the development environment.
- Evaluation: matched base/adapter generation, greedy decoding and EOS stop. Development = 13 real reports; reserved = 11 real reports. D0018 uses 16,384 total tokens. The reserved set has already been evaluated historically; rerunning it compares implementations, not independent final performance.
- New packet and run paths must be immutable once accepted. `ca1-sft-v3` is the accepted local packet; `ca1-run-v3` and `ca1-eval-v3-*` are planned later cloud artifact names. Versioned artifacts are useful even though maintained Python modules are unversioned. A generated packet is a candidate until its separate `acceptance.json` binds the exact content digest to passing parity or completed changed-example review.

## A. Local preparation and deterministic checks

Section A is the only execution section for the refactor implementation session. These steps may generate and validate data, but they must not load model weights, perform a training/preflight pass, or run adapter-dependent evaluation.

### A1. Prepare the development environment

Run from the repository root with Python 3.11. Tracked `pyproject.toml` plus `poetry.lock` define the reproducible local development and dataset-generation environment; the optional `research` group pins the tokenizer stack and offline Torch tests:

```sh
set -euo pipefail
poetry install --with research --no-root
poetry run python -c 'from importlib.metadata import version; print({name: version(name) for name in ("jsonschema", "torch", "transformers", "tokenizers")})'
```

Dataset generation needs the pinned tokenizer but no model weights, CUDA, provider credentials, parser requests, or teacher calls. The optional `research` group must declare Transformers, tokenizers, and Torch for the small offline tensor tests; `jsonschema` is already a main dependency, and the existing non-optional `dev` group supplies pytest and Ruff. The locked Torch 2.14.0 stack supports CPU-only offline tests. The local Poetry environment is separate from the L4 environment: [training/requirements-training.txt](training/requirements-training.txt) pins its research packages, and Poetry must not replace the VM's known-good CUDA PyTorch installation. Do not install cloud-only bitsandbytes for packet construction. The configured type checker is `poetry run mypy`.

### A2. Validate private inputs before generating

The source directory is the retained private CA1 correction packet, and the tokenizer directory is the pinned local tokenizer cache. These may be absent in a clean checkout. The builder must check source hashes, role eligibility, prompt/schema, review metadata, and tokenizer provenance before writing output. A missing or changed source is a stop condition, not permission to regenerate parser/teacher data.

The following input and historical-packet checks passed locally for the accepted v3 build:

```sh
CA1_REPO_ROOT="$(pwd -P)"
CA1_SOURCE_DIR="$CA1_REPO_ROOT/data/research/extraction/sft-correction-v1"
CA1_SOURCE_INDEX="$CA1_REPO_ROOT/data/research/extraction/sft-smoke-v1/sources.json"
CA1_TOKENIZER_DIR="$CA1_REPO_ROOT/data/research/extraction/sft-smoke-v1/tokenizer"
CA1_OLD_PACKET="$CA1_REPO_ROOT/data/research/extraction/sft-pilot-v2"
CA1_BASELINE_PACKET_SHA256="bad1ed2e70de8886a1bb2a57e2ce9c782009e2e2c189912168dd1c570601b4e0"
CA1_PACKET="$CA1_REPO_ROOT/data/research/extraction/ca1-sft-v3"

test -f "$CA1_SOURCE_DIR/prompts/corrected.txt"
test -f "$CA1_SOURCE_DIR/schema.json"
test -f "$CA1_SOURCE_DIR/synthetic_specs.json"
test -f "$CA1_SOURCE_DIR/synthetic/corrected/train.jsonl"
test -f "$CA1_SOURCE_DIR/synthetic/corrected/validation.jsonl"
test -f "$CA1_SOURCE_DIR/real_edits.json"
test -f "$CA1_SOURCE_INDEX"
test -f "$CA1_TOKENIZER_DIR/provenance.json"
test -f "$CA1_OLD_PACKET/SHA256SUMS.json"
printf '%s  %s\n' "$CA1_BASELINE_PACKET_SHA256" \
  "$CA1_OLD_PACKET/SHA256SUMS.json" | sha256sum -c -
poetry run python - "$CA1_OLD_PACKET" <<'PYTHON'
import hashlib
import json
import sys
from pathlib import Path

packet = Path(sys.argv[1])
checksums = json.loads((packet / "SHA256SUMS.json").read_text())
files = {path.name for path in packet.iterdir() if path.is_file() and path.name != "SHA256SUMS.json"}
assert files == set(checksums), "Historical packet file set changed"
for name, expected in checksums.items():
    actual = hashlib.sha256((packet / name).read_bytes()).hexdigest()
    assert actual == expected, name
print("Historical packet payload hashes verified")
PYTHON
```

The old packet is required for the **migration parity check**, not as a permanent runtime dependency of the cleaned trainer. Both its recorded `SHA256SUMS.json` digest and all payload hashes must pass before it serves as the previously reviewed comparison source. Keep all source and old-packet files unchanged.

### A3. Generate and verify a new candidate packet

The builder refuses to overwrite `$CA1_PACKET` and requires the producing source revision to be committed:

```sh
poetry run python -m research.extraction.datasets.build_ca1_dataset \
  --source-dir "$CA1_SOURCE_DIR" \
  --source-index "$CA1_SOURCE_INDEX" \
  --tokenizer-dir "$CA1_TOKENIZER_DIR" \
  --output "$CA1_PACKET"

poetry run python -m research.extraction.datasets.build_ca1_dataset \
  --verify --output "$CA1_PACKET"
sha256sum "$CA1_PACKET/SHA256SUMS.json"
```

The builder writes an immutable candidate payload with `manifest.json` for provenance and a canonical `SHA256SUMS.json` for integrity; it does **not** write `acceptance.json`. The `--verify` command checks the complete payload file set and hashes but does not accept the candidate. Define the content digest as SHA-256 of the exact `SHA256SUMS.json` bytes. That checksum file covers every payload file by sorted relative path, excluding itself and `acceptance.json`, and is written as UTF-8 JSON with sorted keys, compact separators, and one trailing newline. The accepted packet records 320/64 reports, 4,742 maximum complete tokens, isolated families, exact source hashes, and producing Git revision `43294f56ff6f59ae57d7d8264d06159de3af2b16`. Commit and pin any future cleaned implementation before building a replacement; a dirty working tree is rejected.

### A4. Produce review evidence and accept the packet

The completed v3 migration used `model_facing_parity`. Compare parsed JSONL and the model-facing manifest fields; a serialization-only difference does not itself change the examples. The previous packet is anchored by the digest checked in A2. The command below writes a small passing evidence file **outside** the packet only after every assertion succeeds; its fields were checked against the accepted v3 packet.

```sh
CA1_PARITY_EVIDENCE="$CA1_REPO_ROOT/data/research/extraction/ca1-sft-v3-parity.json"

poetry run python - "$CA1_OLD_PACKET" "$CA1_PACKET" "$CA1_PARITY_EVIDENCE" <<'PYTHON'
import hashlib
import json
import sys
from pathlib import Path

old, new, evidence = map(Path, sys.argv[1:4])

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(root, name):
    return [json.loads(line) for line in (root / name).read_text().splitlines()]

for name in ("train.jsonl", "validation.jsonl", "tokenized.jsonl"):
    before, after = rows(old, name), rows(new, name)
    assert before == after, f"Model-facing difference in {name}"
    print(f"{name}: {len(after)} rows match")

before_manifest = json.loads((old / "manifest.json").read_text())
after_manifest = json.loads((new / "manifest.json").read_text())
for field in ("model", "revision", "prompt_sha256", "schema_sha256", "tokenizer_provenance"):
    assert before_manifest[field] == after_manifest[field], field
before_examples = {item["id"]: item for item in before_manifest["examples"]}
after_examples = {item["id"]: item for item in after_manifest["examples"]}
assert before_examples.keys() == after_examples.keys(), "Case membership changed"
for identifier in before_examples:
    before, after = before_examples[identifier], after_examples[identifier]
    for field in before.keys() | after.keys():
        if field == "review_status":  # Immutable old review metadata differs from a new candidate.
            continue
        assert before.get(field) == after.get(field), (identifier, field)
for name in ("statistics.json", "distribution.json"):
    assert (old / name).read_bytes() == (new / name).read_bytes(), name
assert json.loads((new / "statistics.json").read_text())["max_total"] == 4742

record = {
    "basis": "model_facing_parity",
    "content_sha256": digest(new / "SHA256SUMS.json"),
    "previous_packet_sha256": digest(old / "SHA256SUMS.json"),
    "result": "pass",
    "compared": ["train.jsonl", "validation.jsonl", "tokenized.jsonl",
                 "model/revision", "prompt/schema/tokenizer", "all example fields except review status",
                 "statistics/distribution bytes"],
    "example_count": len(after_examples),
}
if evidence.exists():
    raise FileExistsError(evidence)
evidence.write_text(json.dumps(record, sort_keys=True, indent=2) + chr(10))
print(f"Parity evidence: {evidence}")
PYTHON

sha256sum "$CA1_PARITY_EVIDENCE"
```

Compare distribution and coverage statistics separately and record any limitation. New code hashes, paths, review provenance, `acceptance.json`, and checksum-file bytes may differ without changing model-facing content. If any model-facing comparison fails, do **not** write or use parity evidence. Resolve accidental differences, or prepare a separate private `reviewed_changes` evidence file that identifies every intentional difference and its completed case-level review. That file must declare the same candidate content digest and a passing review result; it must not claim inherited approval from the previous packet.

Choose **one** acceptance basis. The following CLI and evidence fields match the implementation. The acceptance operation validates the evidence's basis, result, candidate digest, and required changed-example reviews; for parity it also compares the evidence's old-packet anchor with the historical digest supplied on the command line. It then atomically writes `acceptance.json`. It refuses to overwrite a different acceptance record.

For exact model-facing parity:

```sh
poetry run python -m research.extraction.datasets.packet accept \
  --packet "$CA1_PACKET" --basis model_facing_parity \
  --evidence "$CA1_PARITY_EVIDENCE" \
  --reference-sha256 "$CA1_BASELINE_PACKET_SHA256"
```

For intentionally changed examples **instead**, after their review evidence exists:

```sh
CA1_REVIEW_EVIDENCE="$CA1_REPO_ROOT/data/research/extraction/ca1-sft-v3-reviewed-changes.json"
poetry run python -m research.extraction.datasets.packet accept \
  --packet "$CA1_PACKET" --basis reviewed_changes \
  --evidence "$CA1_REVIEW_EVIDENCE"
```

The v3 record contains content digest `20d31cda44e955c60963b3da0f61add86cd20648e1041b69653d798a28af255f`, source commit `43294f56ff6f59ae57d7d8264d06159de3af2b16`, parity evidence digest `424231da3e22e7614416ea862480c872e17c70d70aa5d2dab8c1feb02841b435`, and the evidence basename. It is excluded from `SHA256SUMS.json` and the content digest. Re-verify both gates after acceptance:

```sh
test -f "$CA1_PACKET/acceptance.json"
poetry run python -m research.extraction.datasets.build_ca1_dataset \
  --verify --output "$CA1_PACKET"
poetry run python -m research.extraction.training.train_ca1 check --packet "$CA1_PACKET"
sha256sum "$CA1_PACKET/SHA256SUMS.json" "$CA1_PACKET/acceptance.json"
```

`--verify` checks payload integrity; offline `train_ca1 check` also verifies that `acceptance.json` refers to that exact content. A candidate, absent/malformed record, stale content digest, or unsupported basis must fail `check` before any model load. Retain the evidence file and its digest privately; cloud training does not need the old packet, old generator source paths, or the evidence file to recheck acceptance.

### A5. Run local quality checks

These commands do not load a cloud model:

```sh
poetry run ruff check research/extraction
poetry run mypy
poetry run python -m pytest --noconftest research/extraction/tests
git diff --check
```

The `train_ca1 check` command in A4 validates packet integrity and acceptance offline; it does not run the `preflight` or `train` stages. The current suite must run without private files, GPU, network, or application database; separate private-packet integration tests may skip when inputs are absent. The active modules are checked by configured mypy; the maintained offline suite has no host-test directory. On 2026-10-04, Poetry lock consistency, Ruff, mypy (nine source files), and 41 offline extraction tests passed with Python 3.11.13. The accepted packet passed both `--verify` and `train_ca1 check` (320/64 cases; maximum 4,742 complete tokens). Do not claim a future run passed because these local checks passed.

### A6. Prepare private transfer archives

After `acceptance.json` is written and `train_ca1 check` passes, create a **training-only** packet archive that includes the accepted packet and its acceptance record. Keep real-report evaluation inputs in a separate archive. The evaluation bundle contains only an ID/role index and the 24 approved `input.txt`/`target.json` pairs. The prompt and schema come from the accepted packet. Neither real-report text nor the detailed source `real_edits.json` belongs in the synthetic training JSONL or the transferred role index.

These archives were created and checked locally after acceptance:

```sh
CA1_PACKET_ARCHIVE="$CA1_REPO_ROOT/data/research/extraction/ca1-sft-v3.tar.gz"
CA1_EVAL_ARCHIVE="$CA1_REPO_ROOT/data/research/extraction/ca1-real-eval-v3.tar.gz"
CA1_EVAL_STAGE="$(mktemp -d)"
test ! -e "$CA1_PACKET_ARCHIVE"
test ! -e "$CA1_EVAL_ARCHIVE"

tar -C "$CA1_REPO_ROOT" -czf "$CA1_PACKET_ARCHIVE" data/research/extraction/ca1-sft-v3

poetry run python - "$CA1_SOURCE_DIR" "$CA1_EVAL_STAGE" <<'PYTHON'
import json
import shutil
import sys
from pathlib import Path

source, staging = map(Path, sys.argv[1:3])
expected = {
    "development": set("A1 A2 A3 A4 C1 D0002 D0006 D0007 D0011 D0012 D0014 D0015 D0016".split()),
    "reserved": set("B1 B2 B3 B4 D0003 D0004 D0005 D0008 D0009 D0017 D0018".split()),
}
rows = [
    {"id": row["id"], "role": row["role"]}
    for row in json.loads((source / "real_edits.json").read_text())
    if row["role"] in expected
]
for role, identifiers in expected.items():
    actual = [row["id"] for row in rows if row["role"] == role]
    assert len(actual) == len(identifiers) and set(actual) == identifiers, role

root = staging / "data/research/extraction/sft-correction-v1"
root.mkdir(parents=True)
(root / "real_edits.json").write_text(json.dumps(rows, indent=2) + "\n")
for row in rows:
    directory = root / "real" / row["id"]
    directory.mkdir(parents=True)
    for name in ("input.txt", "target.json"):
        shutil.copy2(source / "real" / row["id"] / name, directory / name)
PYTHON

tar -C "$CA1_EVAL_STAGE" -czf "$CA1_EVAL_ARCHIVE" data/research/extraction/sft-correction-v1
sha256sum "$CA1_PACKET_ARCHIVE" "$CA1_EVAL_ARCHIVE"
```

The evaluator accepted the staged role-only bundle offline: exactly 13 development and 11 reserved input/target pairs, with a 49-file archive (one index plus 48 case files). The packet archive has 29 files including acceptance. Packet archive SHA-256: `2f91300d9ebd61378d7b6ef299ecfcdd85b7de1c7149b6e205a96c70c655a74a`; real-evaluation archive SHA-256: `980122f4ca714eb54772ec358df178963e6cc89ed952cd85288bdaf27ca7bdee`. The temporary staging directory was removed after archive verification. Do not include credentials, source images, provider responses, detailed edit evidence, or base-model weights in these archives.

## B. Manual cloud training after refactoring

**Do not execute any command in Section B during the refactor implementation session.** This includes transfer, VM setup, dependency installation, model download, GPU preflight, training, and checkpoint handling. Section B is for the later manual run, after the cleaned code and accepted packet are finalized. These are planned instructions for a Linux CUDA cloud VM. The completed historical experiment used a Lightning Studio L4 at `/teamspace/studios/this_studio`; the commands below use that location. A different provider requires an explicitly documented path and image adjustment, not a silent change to the recipe. A VM with SSH can use `scp`; Lightning's file upload interface can place the same archives in the VM root instead.

### B1. Supply operator-specific values and transfer inputs

The source commit, archive names, and SHA-256 digests below are fixed by the completed local migration. Supply an accessible repository URL, a VM image with compatible CUDA PyTorch (historically 2.8.0+cu128), and a VM SSH address or private upload route before the manual run. Never use a moving branch name as the training revision.

**Planned post-refactor commands on the development machine, when SSH is available:**

```sh
CA1_VM_SSH='REPLACE_WITH_VM_SSH_TARGET'
CA1_VM_ROOT='/teamspace/studios/this_studio'

scp "$CA1_PACKET_ARCHIVE" "$CA1_EVAL_ARCHIVE" "$CA1_VM_SSH:$CA1_VM_ROOT/"
```

Use the provider's private upload interface if SSH is unavailable, then verify the exact local archive digests on the VM. The packet archive can be transferred before training; the real-evaluation archive is needed only after the checkpoint is selected.

### B2. Prepare the VM and obtain the pinned source

**Planned post-refactor commands on the VM:**

```sh
set -euo pipefail
CA1_VM_ROOT='/teamspace/studios/this_studio'
CA1_REPO_URL='REPLACE_WITH_ACCESSIBLE_REPOSITORY_URL'
CA1_REVISION='43294f56ff6f59ae57d7d8264d06159de3af2b16'
CA1_PACKET_ARCHIVE_SHA256='2f91300d9ebd61378d7b6ef299ecfcdd85b7de1c7149b6e205a96c70c655a74a'
CA1_EVAL_ARCHIVE_SHA256='980122f4ca714eb54772ec358df178963e6cc89ed952cd85288bdaf27ca7bdee'
CA1_VM_REPO="$CA1_VM_ROOT/healthcare-assistant"
CA1_PACKET="$CA1_VM_REPO/data/research/extraction/ca1-sft-v3"
CA1_RUN="$CA1_VM_ROOT/ca1-run-v3"
HF_HOME="$CA1_VM_ROOT/hf-cache"
export HF_HOME

git clone "$CA1_REPO_URL" "$CA1_VM_REPO"
git -C "$CA1_VM_REPO" checkout --detach "$CA1_REVISION"
git -C "$CA1_VM_REPO" rev-parse HEAD
cd "$CA1_VM_REPO"

printf '%s  %s\n' "$CA1_PACKET_ARCHIVE_SHA256" "$CA1_VM_ROOT/ca1-sft-v3.tar.gz" | sha256sum -c -
tar -xzf "$CA1_VM_ROOT/ca1-sft-v3.tar.gz" -C "$CA1_VM_REPO"
nvidia-smi
python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())'
```

The archive check must pass before extraction. Run B2–C3 in the same VM shell, or restore these variables after reconnecting. Require one visible L4, CUDA availability, and BF16 for the historical matched recipe. The previous run used PyTorch 2.8.0+cu128. If the VM image differs, record and resolve the deviation before training. Check free disk space for base weights, full checkpoints, and logs.

### B3. Install and record dependencies

The training requirements file moves with the trainer. The cloud image must already provide compatible CUDA PyTorch; `requirements-training.txt` does not pin Torch. Record `torch.__version__` and `torch.version.cuda` before and after installing those requirements, and stop if the known-good CUDA build changes. Never install the local Poetry environment on the VM in place of its CUDA Torch.

**Planned post-refactor commands on the VM:**

```sh
cd "$CA1_VM_REPO"
python3 -c 'import torch; print(torch.__version__, torch.version.cuda)'
python3 -m pip install -r research/extraction/training/requirements-training.txt
python3 -c 'import torch; print(torch.__version__, torch.version.cuda)'
python3 -m pip freeze > "$CA1_VM_ROOT/ca1-packages-v3.txt"
python3 -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-4B-Instruct-2507", revision="cdbee75f17c01a7cc42f958dc650907174af0554")'
python3 -m research.extraction.training.train_ca1 check --packet "$CA1_PACKET"
```

The current historical requirements pin Transformers 5.16.1, tokenizers 0.23.2, PEFT 0.21.0, Accelerate 1.15.0, bitsandbytes 0.50.2, and jsonschema 4.25.1. Confirm the moved requirements retain the intended pins. The `check` stage must re-verify payload integrity and `acceptance.json` after transfer, then report 320/64 cases and the verified maximum token length before any GPU preflight. It must not need the old packet, evidence file, or historical generator paths.

### B4. Preflight and train

Preflight is one forward/backward pass on the longest accepted example; it does not update model weights. Training runs in a **new** output directory.

**Planned post-refactor commands on the VM:**

```sh
cd "$CA1_VM_REPO"
python3 -m research.extraction.training.train_ca1 preflight --packet "$CA1_PACKET" \
  2>&1 | tee "$CA1_VM_ROOT/ca1-preflight-v3.log"

python3 -m research.extraction.training.train_ca1 train \
  --packet "$CA1_PACKET" \
  --output "$CA1_RUN" \
  --report-to none \
  2>&1 | tee "$CA1_VM_ROOT/ca1-train-v3.log"
```

The historical run logged to W&B; `--report-to none` changes logging, not the model-facing recipe. If W&B is required for the new run, install its pinned version, configure the API key as a VM secret, use `--report-to wandb`, and record that choice. Do not put secrets in commands, logs, or archives.

If interrupted, preserve the entire output directory and resume only from its latest **complete** checkpoint under the same directory, on the same GPU and package versions:

```sh
python3 -m research.extraction.training.train_ca1 train \
  --packet "$CA1_PACKET" \
  --output "$CA1_RUN" \
  --resume "$CA1_RUN/checkpoint-N" \
  --report-to none \
  2>&1 | tee -a "$CA1_VM_ROOT/ca1-train-v3.log"
```

Replace `checkpoint-N` with an existing complete checkpoint. Do not resume from the inference-only selected-adapter archive. The trainer must reject a completed run, changed packet/recipe, or incompatible hardware/dependencies.

### B5. Preserve the training result

Read `$CA1_RUN/run.json` only after it reports `completed`. It names the selected checkpoint; do not assume the new run selects step 80.

**Planned post-refactor commands on the VM:**

```sh
CA1_BEST="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "$CA1_RUN/run.json")"
test -f "$CA1_RUN/$CA1_BEST/adapter_config.json"
test -f "$CA1_RUN/$CA1_BEST/adapter_model.safetensors"

nvidia-smi -q > "$CA1_RUN/nvidia-smi.txt"
git -C "$CA1_VM_REPO" rev-parse HEAD > "$CA1_RUN/source_commit.txt"
cp "$CA1_VM_ROOT/ca1-packages-v3.txt" "$CA1_RUN/packages.txt"
tar -C "$CA1_RUN" -czf "$CA1_VM_ROOT/ca1-selected-adapter-v3.tar.gz" \
  run.json "$CA1_BEST/adapter_config.json" "$CA1_BEST/adapter_model.safetensors"
sha256sum "$CA1_VM_ROOT/ca1-selected-adapter-v3.tar.gz" \
  "$CA1_RUN/$CA1_BEST/adapter_model.safetensors"
```

Preserve `run.json`, the complete run/checkpoint directory (including optimizer, scheduler, and RNG state if resumption may be needed), selected-adapter archive, preflight and training logs, package list, source commit, packet digest, GPU/CUDA metadata, and archive hashes. The selected-adapter archive alone supports inference, not optimizer-state resumption. Transfer these artifacts back to private `data/` after verifying digests.

## C. Manual evaluation after cloud training

**Do not execute any command in Section C during the refactor implementation session.** Run it only after the later manual cloud training has produced a selected adapter and completed run record. These are planned commands for the same cloud VM and pinned source. No model-dependent evaluation was run while writing this guide.

### C1. Supply real evaluation inputs and verify membership

**Planned post-refactor commands on the VM:**

```sh
cd "$CA1_VM_REPO"
printf '%s  %s\n' "$CA1_EVAL_ARCHIVE_SHA256" "$CA1_VM_ROOT/ca1-real-eval-v3.tar.gz" | sha256sum -c -
tar -xzf "$CA1_VM_ROOT/ca1-real-eval-v3.tar.gz" -C "$CA1_VM_REPO"
CA1_REAL_DATA="$CA1_VM_REPO/data/research/extraction/sft-correction-v1"
test -f "$CA1_REAL_DATA/real_edits.json"
```

The evaluation archive check must pass before extraction. The evaluator must verify exact role membership: development = A1–A4, C1, D0002, D0006, D0007, D0011, D0012, D0014–D0016; reserved = B1–B4, D0003–D0005, D0008, D0009, D0017, D0018. D0001/D0013 remain excluded; D0010 is layout-only. No real report enters the training packet.

### C2. Run each maintained matched evaluation

Select the adapter named in the **new** `run.json`. Use fresh output directories. The evaluator must record adapter digest and settings, save each case before advancing, and validate saved cases before `--resume` skips them.

**Planned post-refactor commands on the VM:**

```sh
CA1_BEST="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "$CA1_RUN/run.json")"
CA1_ADAPTER="$CA1_RUN/$CA1_BEST"

python3 -m research.extraction.evaluation.evaluate_ca1_adapter \
  --packet "$CA1_PACKET" --adapter "$CA1_ADAPTER" \
  --output "$CA1_VM_ROOT/ca1-eval-v3-synthetic" \
  --device cuda --dataset synthetic \
  2>&1 | tee "$CA1_VM_ROOT/ca1-eval-v3-synthetic.log"

python3 -m research.extraction.evaluation.evaluate_ca1_adapter \
  --packet "$CA1_PACKET" --adapter "$CA1_ADAPTER" \
  --real-data "$CA1_REAL_DATA" \
  --output "$CA1_VM_ROOT/ca1-eval-v3-development" \
  --device cuda --dataset development \
  2>&1 | tee "$CA1_VM_ROOT/ca1-eval-v3-development.log"

python3 -m research.extraction.evaluation.evaluate_ca1_adapter \
  --packet "$CA1_PACKET" --adapter "$CA1_ADAPTER" \
  --real-data "$CA1_REAL_DATA" \
  --output "$CA1_VM_ROOT/ca1-eval-v3-reserved" \
  --device cuda --dataset reserved \
  2>&1 | tee "$CA1_VM_ROOT/ca1-eval-v3-reserved.log"
```

If interrupted, repeat **only that dataset's** command with `--resume` after checking the saved manifest and case files. A corrupt case, changed adapter, packet, prompt, device, case list, or decoding settings must stop rather than silently reuse output. Reserved results remain a historical comparison; do not select another checkpoint or change the generator based on them.

### C3. Preserve outputs and recompute summaries

Each output directory should contain `evaluation.json`, one base and one adapted JSON file per case, and recorded settings/versions. Verify 64 synthetic, 13 development, and 11 reserved cases; all expected case files must exist. Recompute aggregate metrics from saved per-case scores before reporting them.

**Planned post-refactor commands on the VM:**

```sh
for set_name in synthetic development reserved; do
  directory="$CA1_VM_ROOT/ca1-eval-v3-$set_name"
  test -f "$directory/evaluation.json"
  python3 - "$directory" "$CA1_PACKET" "$CA1_REAL_DATA" "$set_name" <<'PYTHON'
import sys
from pathlib import Path
from jsonschema import Draft202012Validator, FormatChecker
from research.extraction.datasets.packet import read_json, sha
from research.extraction.evaluation.evaluate_ca1_adapter import cases, validate_saved_case
from research.extraction.evaluation.scoring import summarize

root, packet, real_data = map(Path, sys.argv[1:4])
dataset = sys.argv[4]
report = read_json(root / "evaluation.json")
assert report["status"] == "completed"
assert report["settings"]["dataset"] == dataset
assert report["settings"]["packet_sha256"] == sha(packet / "SHA256SUMS.json")
records = cases(packet, None if dataset == "synthetic" else real_data, dataset)
assert len(records) == {"synthetic": 64, "development": 13, "reserved": 11}[dataset]
assert report["settings"]["case_ids"] == [identifier for identifier, *_ in records]
validator = Draft202012Validator(read_json(packet / "schema.json"), format_checker=FormatChecker())
for variant in ("base", "adapted"):
    groups = {}
    for identifier, group, _, target in records:
        saved = validate_saved_case(root / variant / f"{identifier}.json", identifier, group, target, validator)
        groups.setdefault(group, []).append(saved["score"])
    assert {group: summarize(scores) for group, scores in groups.items()} == report["summaries"][variant]
print(root.name, len(records), "cases and summaries verified")
PYTHON
  tar -C "$CA1_VM_ROOT" -czf "$CA1_VM_ROOT/ca1-eval-v3-$set_name.tar.gz" \
    "ca1-eval-v3-$set_name"
  sha256sum "$CA1_VM_ROOT/ca1-eval-v3-$set_name.tar.gz"
done
```

Verify 64, 13, and 11 printed cases respectively. The snippet uses the refactored saved-case fields, role selector, and `summarize` contract; it verifies each saved case and recomputes the aggregate without loading model weights. Its cloud execution remains pending. Copy archives and logs to private `data/` and verify digests after download.

### C4. Compare with the recorded baseline

Compare the new **base and adapted** arms separately, then compare the adapter's change relative to its own matched base. The historical exact-record baseline is:

| Set | Historical base | Historical adapted |
| --- | ---: | ---: |
| Synthetic validation | 1,034/1,251 | 1,219/1,251 |
| Real development | 94/292 | 131/292 |
| Real reserved, including D0018 | 45/147 | 58/147 |

Also compare schema-valid reports, exact dates, actual versus expected row counts, ordered matches, strict passes, full JSON equality, warning status, EOS completion, token budgets, and individual regressions. Use [historical evaluations](evaluation/historical_evaluations.md), [known issues](known_issues.md), and the private per-report review for the prior case-level errors; compare the same categories on the new outputs.

A refactored dataset with exact model-facing parity need not produce numerically identical training metrics: rerunning optimization can still vary. Record packet, source, hardware, package, recipe, selected checkpoint, and decoding differences before interpreting score changes. Keep historical results unchanged and publish the new result as a separate run. A reserved-set rerun cannot establish a fresh final accuracy claim.

## Final implementation-session closeout

The local packet, acceptance, typing, archive digests, and role-only evaluation bundle are complete. Section A and offline CLI checks passed; the Section C3 saved-output check was reviewed against the implemented contract without model loading. VM repository access, image, and transfer details remain operator-supplied. Sections B and C become verified only after later manual cloud training and adapter-dependent evaluation, with their commands, logs, and artifacts checked. Record new results in [evaluations](evaluation/evaluations.md), update [known issues](known_issues.md) and [next steps](next_steps.md) from evidence, and keep [research status](../README.md) current.
