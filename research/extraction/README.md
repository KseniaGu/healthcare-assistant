# Extraction experiment tools

[Research status](../README.md) · [Results](evaluations.md) · [Proposed next step](plan.md)

Run commands from the repository root. These tools reuse the application's prompt
engine and validator; they do not start the UI, database or tracing. Production
code stays under `backend/` and `src/`.

## Tools and environments

| Module under `research.extraction` | Role | Environment |
| --- | --- | --- |
| `run_extraction_baseline` | Gemini baseline with explicit inputs and durable budget | Project Python/Poetry; live requests require a separately approved budget |
| `evaluate_local_extraction` | Freeze original packet, bounded Ollama run, offline scoring | Project Python/Poetry; host Ollama only for `run` |
| `evaluate_image_extraction` | Three-image parse/review/Gemini/target/Qwen pilot | Project Python/Poetry; completed packet's live allowance is exhausted |
| `evaluate_transformers_extraction` | Original BF16-versus-Ollama host diagnostic | Separate macOS environment with PyTorch/MPS, Transformers and psutil |
| `scoring` | Pure exact-record/field scoring used by the runners | No inference runtime |
| `archive/` | One-time completion scripts and detailed historical notes | Historical provenance only; do not replay completed jobs |

Do not install host-only dependencies into the application merely to collect their
tests. A Linux Dev Container cannot directly use macOS MPS. Ollama runs on the Mac;
container requests use `http://host.orb.internal:11434`, host requests use
`http://127.0.0.1:11434`. Each experiment pins its own runtime and model identity;
do not change historical pins to make a new runtime pass preflight.

The runners request 32,768 context tokens explicitly and check allocation. The
Ollama desktop slider is not evidence of the context used by these API requests.
Model allocation snapshots are not peak process memory. Keep the Mac awake and
avoid competing workloads for any future authorized run.

## Offline commands

Inspect interfaces without submitting requests:

```sh
poetry run python -m research.extraction.evaluate_local_extraction --help
poetry run python -m research.extraction.evaluate_image_extraction --help
poetry run python -m research.extraction.run_extraction_baseline --help
```

Recompute scores only when intended: these commands write derived score/review files
in the specified run directory. Preserve manually authored assessments and any
existing derived files before rescoring historical work.

```sh
poetry run python -m research.extraction.evaluate_local_extraction score \
  --dataset data/extraction_poc/qwen-eval-packet-v1 \
  --run-dir data/extraction_poc/qwen-untuned-v1
poetry run python -m research.extraction.evaluate_image_extraction \
  --experiment-dir data/extraction_poc/image-extraction-pilot-v1 --stage score --resume
```

The local evaluator's `freeze --output-dir …` is offline and requires a new directory.
Its `run --dataset … --output-dir …` makes requests. The image runner stages
`parse`, `gemini` and `qwen` also make requests, behind hash-bound reviews and
attempt/cost ledgers. **All existing live allowances are exhausted.** A fresh folder
or copied packet is not authorization to repeat an experiment. Preserve hidden
ledgers and never reset budgets to bypass a guard.

## Offline tests

The extraction tests use fictitious data and mocked requests. `--noconftest` avoids
the existing application integration hooks, including test-database cleanup.

```sh
poetry run python -m pytest --noconftest tests/unit research/extraction/tests \
  --ignore=research/extraction/tests/host
```

Run host diagnostic tests separately in the prepared macOS environment:

```sh
python -m pytest --noconftest research/extraction/tests/host
```

The [image redaction guide](../image_redaction/README.md) describes its separate
requirements and tests. Shared fictitious contract fixtures stay under
`tests/samples/extraction_poc/` because application validation tests also use them.

## Data and host continuation

Existing packets stay under gitignored `data/extraction_poc/`; source paths and
ledgers are unchanged. The Dev Container mounts a data volume over the host folder.
Copy the entire required packet, including hidden ledgers, when handing work to
the host. Credentials stay in the host environment, never in packets or commands.

For any future authorized parser continuation, stop after a container execution
failure and diagnose it first. LlamaParse OCR runs remotely. Host execution may
help local networking/dependency/filesystem issues, but not server OCR or quotas.
Retrieve existing job IDs; reconcile uncertain submissions before another upload.
Verify host SDK versions against the frozen protocol and retain its original limits.
The completed image pilot used llama-parse/llama-cloud-services 0.6.94 and
Google GenAI 1.75.0; it does not need a host retry.
