# Reproducing the maintained CA1 extraction experiment

This guide covers current offline packet checks and the recorded v3 manual cloud procedure. The cloud commands pin the source commit that produced the accepted v3 packet. Training from a newer source revision requires a new candidate packet built from that clean committed revision, followed by separate acceptance. [Dataset](datasets/dataset.md) describes source roles and the packet contract; [training](training/training.md) records the recipe; [evaluation](evaluation/evaluations.md) records current results and provenance. Detailed medical data and model artifacts remain under gitignored `data/`.

## Fixed contract for this reproduction

- Model: `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
- Packet: 320 fictitious training and 64 fictitious validation reports, including the corrected 16/8 smoke subset. No real reports are in training JSONL. The CA1 prompt/schema, messages, targets, IDs/splits, cached IDs/labels, and 8,192-token limit define the model-facing packet.
- Training comparison platform: one NVIDIA L4 with CUDA BF16 and the pinned dependency stack. Training, longest-example preflight, and model-dependent evaluation run on the cloud VM, not in the development environment.
- Evaluation: matched base/adapter generation, greedy decoding and EOS stop. Development = 13 real reports; reserved = 11 real reports. D0018 uses 16,384 total tokens. The reserved set has already been evaluated; rerunning it does not establish independent final performance.
- New packet and run paths must be immutable once accepted. `ca1-sft-v3` is the accepted packet; `ca1-run-v3` and `ca1-eval-v3-*` name the completed private cloud artifacts. Versioned artifacts identify immutable runs even though maintained Python modules are unversioned. A generated packet is a candidate until its separate `acceptance.json` binds the exact content digest to passing parity or completed changed-example review.

## A. Local preparation and deterministic checks

Section A checks local inputs and packets without loading model weights, running preflight/training, or performing adapter-dependent evaluation.

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

Check the private inputs and reviewed comparison packet before rebuilding the accepted v3 packet:

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
if files != set(checksums):
    raise ValueError("Historical packet file set changed")
for name, expected in checksums.items():
    actual = hashlib.sha256((packet / name).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Historical packet hash changed: {name}")
print("Historical packet payload hashes verified")
PYTHON
```

The earlier reviewed packet is needed only when using model-facing parity as the acceptance basis. Verify its recorded `SHA256SUMS.json` digest and all payload hashes before comparing it with a candidate. The trainer uses only the accepted packet. Keep source and reviewed-packet files unchanged.

### A3. Generate and verify a new candidate packet

The builder refuses to overwrite an existing directory and requires the producing source revision to be committed and clean; its callable build path has no revision override. The source check includes `datasets/synthetic/__init__.py` and its `types.py`, `layouts.py`, and `reports.py` modules; candidate manifests record a hash for each of the three modules. Keep `$CA1_PACKET` as the immutable accepted v3 packet and choose a new, unused path for a candidate built from the current code:

```sh
CA1_CANDIDATE="$CA1_REPO_ROOT/data/research/extraction/ca1-sft-next-candidate"
test ! -e "$CA1_CANDIDATE"
poetry run python -m research.extraction.datasets.build_ca1_dataset \
  --source-dir "$CA1_SOURCE_DIR" \
  --source-index "$CA1_SOURCE_INDEX" \
  --tokenizer-dir "$CA1_TOKENIZER_DIR" \
  --output "$CA1_CANDIDATE"

poetry run python -m research.extraction.datasets.build_ca1_dataset \
  --verify --output "$CA1_CANDIDATE"
sha256sum "$CA1_CANDIDATE/SHA256SUMS.json"
```

The builder writes an immutable candidate payload with `manifest.json` for provenance and a canonical `SHA256SUMS.json` for integrity; it does **not** write `acceptance.json`. The `--verify` command checks the complete payload file set and hashes but does not accept the candidate. Define the content digest as SHA-256 of the exact `SHA256SUMS.json` bytes. That checksum file covers every payload file by sorted relative path, excluding itself and `acceptance.json`, and is written as UTF-8 JSON with sorted keys, compact separators, and one trailing newline. The accepted packet records 320/64 reports, 4,742 maximum complete tokens, isolated families, exact source hashes, and producing Git revision `43294f56ff6f59ae57d7d8264d06159de3af2b16`. Commit and pin any future cleaned implementation before building a replacement; a dirty working tree is rejected.

### A4. Produce review evidence and accept the packet

The accepted v3 packet used `model_facing_parity`. For a new candidate, compare parsed JSONL and the model-facing manifest fields; a serialization-only difference does not itself change the examples. The comparison packet is anchored by the digest checked in A2. The command below writes a passing evidence file **outside** the packet only after every comparison succeeds.

```sh
CA1_PARITY_EVIDENCE="$CA1_REPO_ROOT/data/research/extraction/ca1-sft-next-candidate-parity.json"

poetry run python - "$CA1_OLD_PACKET" "$CA1_CANDIDATE" "$CA1_PARITY_EVIDENCE" <<'PYTHON'
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
    if before != after:
        raise ValueError(f"Model-facing difference in {name}")
    print(f"{name}: {len(after)} rows match")

before_manifest = json.loads((old / "manifest.json").read_text())
after_manifest = json.loads((new / "manifest.json").read_text())
for field in ("model", "revision", "prompt_sha256", "schema_sha256", "tokenizer_provenance"):
    if before_manifest[field] != after_manifest[field]:
        raise ValueError(f"Model-facing manifest difference: {field}")
before_examples = {item["id"]: item for item in before_manifest["examples"]}
after_examples = {item["id"]: item for item in after_manifest["examples"]}
if before_examples.keys() != after_examples.keys():
    raise ValueError("Case membership changed")
for identifier in before_examples:
    before, after = before_examples[identifier], after_examples[identifier]
    for field in before.keys() | after.keys():
        if field == "review_status":  # Immutable old review metadata differs from a new candidate.
            continue
        if before.get(field) != after.get(field):
            raise ValueError(f"Case field changed: {identifier} {field}")
for name in ("statistics.json", "distribution.json"):
    if (old / name).read_bytes() != (new / name).read_bytes():
        raise ValueError(f"Packet summary changed: {name}")
if json.loads((new / "statistics.json").read_text())["max_total"] != 4742:
    raise ValueError("Maximum token count changed")

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
  --packet "$CA1_CANDIDATE" --basis model_facing_parity \
  --evidence "$CA1_PARITY_EVIDENCE" \
  --reference-sha256 "$CA1_BASELINE_PACKET_SHA256"
```

For intentionally changed examples **instead**, after their review evidence exists:

```sh
CA1_REVIEW_EVIDENCE="$CA1_REPO_ROOT/data/research/extraction/ca1-sft-next-candidate-reviewed-changes.json"
poetry run python -m research.extraction.datasets.packet accept \
  --packet "$CA1_CANDIDATE" --basis reviewed_changes \
  --evidence "$CA1_REVIEW_EVIDENCE"
```

The candidate acceptance record is excluded from `SHA256SUMS.json` and the content digest. Re-verify both gates after accepting the candidate:

```sh
test -f "$CA1_CANDIDATE/acceptance.json"
poetry run python -m research.extraction.datasets.build_ca1_dataset \
  --verify --output "$CA1_CANDIDATE"
poetry run python -m research.extraction.training.train_ca1 check --packet "$CA1_CANDIDATE"
sha256sum "$CA1_CANDIDATE/SHA256SUMS.json" "$CA1_CANDIDATE/acceptance.json"
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

The `train_ca1 check` command in A4 validates packet integrity and acceptance offline; it does not run the `preflight` or `train` stages. The current suite must run without private files, GPU, network, or application database; separate private-packet integration tests may skip when inputs are absent. The active modules are checked by configured mypy; the maintained offline suite has no host-test directory. The accepted packet contains 320/64 cases with a maximum of 4,742 complete tokens. Run the checks against the candidate and environment actually being used.

### A6. Recorded v3 transfer archives

The commands below document how the accepted v3 packet and its real-evaluation inputs were packaged. They use `$CA1_PACKET`, not the new `$CA1_CANDIDATE` from A3–A4, and stop if the recorded archive paths already exist. For a future accepted candidate, choose new archive names and record its own checksums. A training-only packet archive includes the accepted packet and its acceptance record. Keep real-report evaluation inputs in a separate archive. The evaluation bundle contains only an ID/role index and the 24 approved `input.txt`/`target.json` pairs. The prompt and schema come from the accepted packet. Neither real-report text nor the detailed source `real_edits.json` belongs in the synthetic training JSONL or the transferred role index.

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
    if len(actual) != len(identifiers) or set(actual) != identifiers:
        raise ValueError(f"Real evaluation role changed: {role}")

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

## B. Manual cloud training

The commands below are reproducible reference steps for the manual Linux CUDA procedure. The recorded run used a Lightning Studio L4 at `/teamspace/studios/this_studio`; downloaded artifacts verify its result and recipe, not each transfer command. A different provider requires a documented path and image adjustment. A VM with SSH can use `scp`; Lightning's file upload interface can place the same archives in the VM root instead.

### B1. Supply operator-specific values and transfer inputs

The source commit, archive names, and SHA-256 digests below identify the accepted v3 packet and completed run. The current trainer requires the source commit in a packet manifest to equal the clean checked-out HEAD, so the old v3 packet cannot train from a newer source commit. Supply an accessible repository URL, a VM image with compatible CUDA PyTorch (2.8.0+cu128 for the recorded run), and a VM SSH address or private upload route before the manual run. Never use a moving branch name as the training revision.

**Reference commands on the development machine, when SSH is available:**

```sh
CA1_VM_SSH='REPLACE_WITH_VM_SSH_TARGET'
CA1_VM_ROOT='/teamspace/studios/this_studio'

scp "$CA1_PACKET_ARCHIVE" "$CA1_EVAL_ARCHIVE" "$CA1_VM_SSH:$CA1_VM_ROOT/"
```

Use the provider's private upload interface if SSH is unavailable, then verify the exact local archive digests on the VM. The packet archive can be transferred before training; the real-evaluation archive is needed only after the checkpoint is selected.

### B2. Prepare the VM and obtain the pinned source

**Reference commands on the VM:**

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

The archive check must pass before extraction. Run B2–C3 in the same VM shell, or restore these variables after reconnecting. Require one visible L4, CUDA availability, and BF16 for the recorded matched recipe. The recorded run used PyTorch 2.8.0+cu128. If the VM image differs, record and resolve the deviation before training. Check free disk space for base weights, full checkpoints, and logs.

### B3. Install and record dependencies

The training requirements file is kept with the trainer. The cloud image must already provide compatible CUDA PyTorch; `requirements-training.txt` does not pin Torch. Record `torch.__version__` and `torch.version.cuda` before and after installing those requirements, and stop if the known-good CUDA build changes. Never install the local Poetry environment on the VM in place of its CUDA Torch.

**Reference commands on the VM:**

```sh
cd "$CA1_VM_REPO"
python3 -c 'import torch; print(torch.__version__, torch.version.cuda)'
python3 -m pip install -r research/extraction/training/requirements-training.txt
python3 -m pip install 'wandb==0.30.0'
python3 -c 'import torch; print(torch.__version__, torch.version.cuda)'
python3 -m pip freeze > "$CA1_VM_ROOT/ca1-packages-v3.txt"
python3 -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-4B-Instruct-2507", revision="cdbee75f17c01a7cc42f958dc650907174af0554")'
python3 -m research.extraction.training.train_ca1 check --packet "$CA1_PACKET"
```

The training requirements pin Transformers 5.16.1, tokenizers 0.23.2, PEFT 0.21.0, Accelerate 1.15.0, bitsandbytes 0.50.2, and jsonschema 4.25.1. W&B 0.30.0 is installed separately because the completed run chose W&B logging. Confirm the requirements retain the intended pins. The `check` stage must re-verify payload integrity and `acceptance.json` after transfer, then report 320/64 cases and the verified maximum token length before any GPU preflight. It does not need the comparison packet, evidence file, or source-generation paths.

### B4. Preflight and train

Preflight is one forward/backward pass on the longest accepted example; it does not update model weights. Training runs in a **new** output directory. Both commands require a clean tracked Git worktree at the repository root, including no staged or unstaged tracked changes, before CUDA use. The offline `check` stage can inspect an accepted packet without CUDA.

**Reference commands on the VM:**

```sh
cd "$CA1_VM_REPO"
test -n "${WANDB_API_KEY:-}"
export WANDB_ENTITY='gu-fuku-ro-'
export WANDB_PROJECT='ca1-pilot-v3'
python3 -m research.extraction.training.train_ca1 preflight --packet "$CA1_PACKET" \
  2>&1 | tee "$CA1_VM_ROOT/ca1-preflight-v3.log"

python3 -m research.extraction.training.train_ca1 train \
  --packet "$CA1_PACKET" \
  --output "$CA1_RUN" \
  --report-to wandb \
  2>&1 | tee "$CA1_VM_ROOT/ca1-train-v3.log"
```

The completed v3 run used `--report-to wandb`, W&B 0.30.0, and project `ca1-pilot-v3`; its [W&B record](https://wandb.ai/gu-fuku-ro-/ca1-pilot-v3/runs/qra140m7) is linked in the [training guide](training/training.md). For reproduction, install the pinned W&B version, set the project and entity as shown, and provide the API key through a VM secret. That logging choice does not change the model-facing recipe. Do not put secrets in commands, logs, or archives.

If interrupted, preserve the entire output directory and resume only from its latest **complete** checkpoint under the same directory, on the same GPU and package versions:

```sh
python3 -m research.extraction.training.train_ca1 train \
  --packet "$CA1_PACKET" \
  --output "$CA1_RUN" \
  --resume "$CA1_RUN/checkpoint-N" \
  --report-to wandb \
  2>&1 | tee -a "$CA1_VM_ROOT/ca1-train-v3.log"
```

Replace `checkpoint-N` with an existing complete checkpoint. Do not resume from the inference-only selected-adapter archive. The trainer must reject a completed run, changed packet/recipe, or incompatible hardware/dependencies.

### B5. Preserve the training result

Read `$CA1_RUN/run.json` only after it reports `completed`. It names the selected checkpoint; do not assume a future run selects the same step.

**Reference commands on the VM:**

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

Preserve `run.json`, the complete run/checkpoint directory (including optimizer, scheduler, and RNG state if resumption may be needed), selected-adapter archive, preflight and training logs, package list, source commit, packet digest, GPU/CUDA metadata, and archive hashes. The selected-adapter archive alone supports inference, not optimizer-state resumption. The downloaded v3 selected-adapter archive has SHA-256 `2ba47d57c91c8f9dfd93ea414af4ae335d250e6116373ce41f4a13fc270b7924` and contains completed `run.json` plus `checkpoint-40` adapter files. The preflight and training logs and package list were also downloaded. Full optimizer/checkpoint history, separate `nvidia-smi.txt` and `source_commit.txt` sidecars, and VM-side checksum records were not supplied; retrieve them if full resume or independent transfer provenance is required.

## C. Manual evaluation after cloud training

Section C was performed manually after the completed v3 run selected `checkpoint-40`. The three downloaded evaluation archives and logs were reviewed offline against the accepted packet and selected adapter. The commands below remain the reproducible procedure for the same cloud VM and pinned source.

### C1. Supply real evaluation inputs and verify membership

**Reference commands on the VM:**

```sh
cd "$CA1_VM_REPO"
printf '%s  %s\n' "$CA1_EVAL_ARCHIVE_SHA256" "$CA1_VM_ROOT/ca1-real-eval-v3.tar.gz" | sha256sum -c -
tar -xzf "$CA1_VM_ROOT/ca1-real-eval-v3.tar.gz" -C "$CA1_VM_REPO"
CA1_REAL_DATA="$CA1_VM_REPO/data/research/extraction/sft-correction-v1"
test -f "$CA1_REAL_DATA/real_edits.json"
```

The evaluation archive check must pass before extraction. The evaluator must verify exact role membership: development = A1–A4, C1, D0002, D0006, D0007, D0011, D0012, D0014–D0016; reserved = B1–B4, D0003–D0005, D0008, D0009, D0017, D0018. D0001/D0013 remain excluded; D0010 is layout-only. No real report enters the training packet.

### C2. Run each maintained matched evaluation

Select the adapter named in `run.json`. Use fresh output directories. The evaluator must record adapter digest and settings, save each case before advancing, and validate saved cases before `--resume` skips them.

**Reference commands on the VM:**

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

If interrupted, repeat **only that dataset's** command with `--resume` after checking the saved manifest and case files. The evaluator replaces each JSON file atomically and rechecks saved invalid responses against their content and EOS state before skipping them. A corrupt case, changed adapter, packet, prompt, device, case list, or decoding settings must stop rather than silently reuse output. Reserved results are limited to comparison; do not select another checkpoint or change the generator based on them.

### C3. Preserve outputs and recompute summaries

Each output directory should contain `evaluation.json`, one base and one adapted JSON file per case, and recorded settings/versions. Verify 64 synthetic, 13 development, and 11 reserved cases; all expected case files must exist. Recompute aggregate metrics from saved per-case scores before reporting them.

**Reference commands on the VM:**

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
if report["status"] != "completed":
    raise ValueError("Evaluation is incomplete")
if report["settings"]["dataset"] != dataset:
    raise ValueError("Evaluation dataset changed")
if report["settings"]["packet_sha256"] != sha(packet / "SHA256SUMS.json"):
    raise ValueError("Evaluation packet changed")
records = cases(packet, None if dataset == "synthetic" else real_data, dataset)
if len(records) != {"synthetic": 64, "development": 13, "reserved": 11}[dataset]:
    raise ValueError("Evaluation case count changed")
if report["settings"]["case_ids"] != [identifier for identifier, *_ in records]:
    raise ValueError("Evaluation case IDs changed")
validator = Draft202012Validator(read_json(packet / "schema.json"), format_checker=FormatChecker())
for variant in ("base", "adapted"):
    groups = {}
    for identifier, group, _, target in records:
        saved = validate_saved_case(root / variant / f"{identifier}.json", identifier, group, target, validator)
        groups.setdefault(group, []).append(saved["score"])
    if {group: summarize(scores) for group, scores in groups.items()} != report["summaries"][variant]:
        raise ValueError(f"Evaluation summary changed: {variant}")
print(root.name, len(records), "cases and summaries verified")
PYTHON
  tar -C "$CA1_VM_ROOT" -czf "$CA1_VM_ROOT/ca1-eval-v3-$set_name.tar.gz" \
    "ca1-eval-v3-$set_name"
  sha256sum "$CA1_VM_ROOT/ca1-eval-v3-$set_name.tar.gz"
done
```

Verify 64, 13, and 11 printed cases respectively. The snippet verifies each saved case and recomputes the aggregate without loading model weights. Archive digests and case-level findings are in [evaluation](evaluation/evaluations.md). Keep archives and logs in private `data/` and verify digests after transfer.

### C4. Interpret the matched comparison

Compare the base and adapted arms separately, then the adapter's change relative to its own matched base. Review schema-valid reports, exact dates, actual versus expected row counts, ordered matches, strict passes, full JSON equality, warning status, EOS completion, token budgets, and individual case errors. The [evaluation](evaluation/evaluations.md) records the accepted run's scores and case-level limitations; [known issues](known_issues.md) lists open interpretation risks.

Optimization can vary even when a candidate packet has exact model-facing parity. Record packet, source, hardware, package, recipe, selected checkpoint, and decoding differences before interpreting score changes. A reserved-set rerun cannot establish a fresh final accuracy claim.
