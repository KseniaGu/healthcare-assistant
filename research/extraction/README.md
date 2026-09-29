# Extraction experiment tools

[Research status](../README.md) · [Results](evaluations.md) · [Current dataset plan](plan.md)

Run commands from the repository root. These tools reuse the application's prompt
engine and validator; they do not start the UI, database or tracing. Production
code stays under `backend/` and `src/`.

## Report locations and per-report index

The [private report artifact index](../../data/research/report_inventory.md) links
all nine original reviewed reports (A1–A4, B1–B4, C1) and all 16 approved internet
examples (D0001–D0016), plus D0017/D0018 (two recovered originals, ten pages),
to their available artifacts, including the completed CA1 baseline. It stays
gitignored alongside the evidence and may be absent on other checkouts. It lists
known duplicates and distinguishes historical application outputs from reviewed
targets. It does not authorize new processing or release reserved B.

| Collection | Mapping and locations |
| --- | --- |
| Original nine | `data/extraction_poc/qwen-eval-packet-v1/manifest.json`: `documents` entries record `sources.pdf`, `sources.parsed_text`, `sources.processed_json`, duplicate paths and packet-relative `input`, `target`, and optional `gemini` paths |
| Original application artifacts | `data/anonymized/`, `data/parsed/`, `data/processed/`; use manifest mappings, not directory counts, to identify unique reports |
| Original reviewed targets | `data/extraction_poc/qwen-eval-packet-v1/targets/<ID>.json`; application JSON is not automatically the v4.1 gold target |
| Internet images | `data/approved_image_redactions/Dxxxx-vNNN-<hash>.png` and sibling `.json`; `overview.md` records organization/layout observations and review limitations |
| Recovered originals | `data/approved_image_redactions/D0017-pNNN-vNNN-<hash>.png` and `D0018-pNNN-vNNN-<hash>.png`, with sibling `.json`; 2 and 8 pages respectively. `originals-review-2026-09-26.md` links exact versions and the user-confirmed review outcomes. Raw page text and report targets now live in `data/research/extraction/sft-smoke-v1/real/<ID>/`. |
| Three-image pilot | `data/extraction_poc/image-extraction-pilot-v1/`: `parse/<ID>/input.txt`, `corrected/<ID>.txt`, `targets/<ID>.json`, `gemini/<ID>/prediction.json`, `qwen/<ID>/prediction.json` |

Only D0002, D0007 and D0016 have parser outputs and frozen targets in the image
pilot; those targets apply to **corrected** inputs. The later SFT preparation adds the other usable
raw-parser targets; see the preparation guide rather than treating this pilot as
the complete current collection. D0010 is an annotated layout guide per the
existing inventory. `protocol.json`, `clean_input_amendment.json` and frozen review
files record pilot provenance. D0017/D0018 now have approved page-image mappings;
the user resolved all redaction-review follow-ups on 2026-09-27. D0018-p008 has
no results. Private original PDF paths remain unassigned; user-confirmed aliases are
A = Invitro, B = KDL, C = Gemotest, so D0017/D0018 are reserved KDL reports; do not infer them from duplicates. Keep each report
together across pages and distinguish image approval from reviewed extraction targets.

## SFT preparation packet

The latest [correction-aware comparison guide](correction_policy.md) and
[review packet](../../data/research/extraction/sft-correction-v1/README.md) add paired
CA1 targets and revised synthetic validation without replacing the literal packet.
CA1 and the revised v2 synthetic packet are user-accepted. The literal variant
is historical comparison data. Training has not run.

The [preparation guide](preparation.md) documents the new staged command and
[private packet](../../data/research/extraction/sft-smoke-v1/README.md). The local
16/8 synthetic subset and all 14 new drafts are assistant-reviewed; twelve new
drafts have usable targets and D0001/D0013 are explicitly excluded. Including nine
historical and three raw-pilot targets gives 24 usable real targets. D0018 exceeds
8K (9,696 tokens), so ten reserved candidates fit the proposed 8K evaluation. All KDL sources
(B1–B4, D0017/D0018), CitiLab and Helix are withheld from synthetic generation.
No training has run. The completed untuned CA1 scores are linked below.

## CA1 synthetic learning packet

The [revised local packet](../../data/research/extraction/sft-pilot-v2/README.md)
contains 320 training and 64 validation reports, including the corrected 16/8
smoke subset. It was **assistant-reviewed and user-accepted on 2026-09-29**.
Its 304
new training inputs have median 2,591 characters / 47 lines, versus 746 / 21
in [v1](../../data/research/extraction/sft-pilot-v1/README.md) and 3,289 / 61
in the 13 eligible originals. The [audit](distribution_audit.md) explains why
v1 is retained only as failed historical evidence. V2 has 32 training and 10
validation families, four domains, 0 duplicate inputs, 0 detected split leaks,
and 0 cross-split near-duplicate candidates. All 384 chats fit 8,192 tokens;
the maximum is 4,742. The assistant reviewed all 64 validation and 85
stratified training pairs; see the packet review for fixes and limitations.
No training, new inference or external requests ran.

```sh
/opt/venv/bin/python -m research.extraction.generate_sft_pilot_v2 --output /tmp/sft-pilot-v2-rebuild
/opt/venv/bin/python -m research.extraction.generate_sft_pilot_v2 --verify
```

The build refuses to overwrite an existing packet; `--verify` checks the sealed
one. The pinned cached Qwen tokenizer and exact CA1 prompt/schema are used. The
combined tokenized file supervises assistant JSON through EOS and masks the
native post-EOS newline.

For spreadsheet review, the [13 eligible original user inputs](../../data/research/extraction/sft-pilot-v1-review-csv/original-user-inputs.csv),
[320 revised training inputs](../../data/research/extraction/sft-pilot-v2-review-csv/training-user-inputs.csv),
and [64 revised validation inputs](../../data/research/extraction/sft-pilot-v2-review-csv/validation-user-inputs.csv)
are UTF-8 CSV. Each row holds `id` and the exact multiline `user_content`,
including the `<report>` wrapper. The v2 exports sit beside the sealed packet.
The [v1 training input CSV](../../data/research/extraction/sft-pilot-v1-review-csv/training-user-inputs.csv)
remains historical comparison evidence. The small original set cannot prove a
matching deployment distribution. The
user accepted v2 after reviewing the revised inputs.

## Training pipeline handoff

Use the **accepted, sealed** [v2 packet](../../data/research/extraction/sft-pilot-v2/README.md)
in this workspace. The gitignored `data/` packet and cached tokenizer may be
absent from a fresh checkout, so check their presence before implementing a
loader. Run the offline integrity check first:

```sh
/opt/venv/bin/python -m research.extraction.generate_sft_pilot_v2 --verify
```

- `train.jsonl` has 320 records and `validation.jsonl` has 64. Each line has
  `id` and `messages`; the messages contain `role` and `content` for system,
  user and assistant in that order. Assistant content is serialized CA1 JSON.
  The 16/8 smoke pairs are **included** in these totals; do not append them
  again.
- `tokenized.jsonl` is the single combined cache. Each line has `id`, `split`,
  `input_ids` and `labels`. Labels are `-100` for the system/user prefix and the
  native post-EOS newline; assistant JSON through EOS is supervised. If a
  training collator adds padding, its labels must also be `-100`. Do not
  truncate: the largest complete chat is 4,742 tokens and the limit is 8,192.
- `manifest.json` pins the approved prompt and schema hashes, generator hashes,
  smoke hashes, model ID, tokenizer revision, split/family provenance and
  per-example evidence. The model/tokenizer is
  `Qwen/Qwen3-4B-Instruct-2507` at revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`; tokenizer files are under
  `data/research/extraction/sft-smoke-v1/tokenizer/`. Use the exact prompt at
  `data/research/extraction/sft-correction-v1/prompts/corrected.txt` and its
  sibling `schema.json`. Verify chat-template and labels if retokenizing.
- `review.json`, `statistics.json`, `distribution.json` and
  `SHA256SUMS.json` document acceptance, coverage, remaining realism limits and
  file integrity. The input-only CSVs are review exports, not training targets.
  The generator's `--output` command makes a new pending-review candidate;
  its output does not inherit this packet's later user-acceptance record.

Train only on the training split; use the validation split for development
checks. Real reports supplied layout conventions but are absent from these
JSONL files. The reserved KDL, CitiLab and Helix reports remain withheld from
adaptation and checkpoint selection. The completed untuned CA1 Q8/Ollama
baseline is useful diagnostic evidence, but a later base/tuned comparison needs
matched precision, runtime, prompt and decoding conditions. Dataset acceptance
did not select a training method, hardware, budget or evaluation run. The
user-accepted dataset status is already recorded; the next session need not
request acceptance of these same pairs again.

## Approved CA1 untuned baseline

The [final evaluation report](evaluations.md#approved-ca1-untuned-qwen-baseline--2026-09-28)
contains all 13 development reports and eight synthetic validation examples in one
set of tables, with per-example output links and reviewed metrics. The existing
Qwen3-4B-Instruct-2507 Q8_0 model used Ollama 0.34.3, the approved CA1 prompt,
English warnings and unconstrained generation within 8K. Reserved reports stay closed.
These results are not directly comparable with older schema-constrained scores or
BF16 training runs. All live allowances are consumed. `evaluate_ca1 score --output
PATH` recomputes positional scores offline; source alignment and warning semantics
also require the saved manual review. Preserve frozen artifacts when rescoring.

## Tools and environments

| Module under `research.extraction` | Role | Environment |
| --- | --- | --- |
| `prepare_correction_variant` | Rebuild reviewed historical CA1/literal packet | Offline project environment and cached tokenizer |
| `generate_sft_pilot_v2` | Generate or verify the revised 320/64 CA1 packet | `/opt/venv/bin/python`, offline cached tokenizer |
| `generate_sft_pilot` | Reproduce the failed historical v1 packet | `/opt/venv/bin/python`, offline cached tokenizer |
| `evaluate_ca1` | Frozen CA1 local Qwen baseline and offline scoring | Completed allowances; no automatic retries |
| `prepare_sft` | Bounded source preparation, real annotation, local smoke generation, validation and review packaging | Project dependencies plus pinned-tokenizer tooling; see preparation guide |
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

The older local/image runners request 32,768 context tokens; `evaluate_ca1` uses
8,192. Each run records its own settings. The
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

Packets stay under gitignored `data/extraction_poc/` and
`data/research/extraction/`; source paths and
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
