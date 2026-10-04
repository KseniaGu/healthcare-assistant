# CA1 extraction experiment refactor implementation plan

> **For the implementation session:** Complete the refactor and its offline validation gates in Stages 1–12 and 16. Stages 13–15 require documentation and command review only. Use the repository state and Git diff as the source of truth, and update the [reproduction guide](reproduction.md) against the refactored local code. No implementation, dataset generation, training, or model evaluation was performed while writing this plan.

**Goal:** Replace the branch's historical-script dependency chain with a small, reproducible CA1 dataset, training, and evaluation pipeline while preserving model-facing data and experimental semantics unless a correction is explicitly recorded.

**Architecture:** Dataset construction and packet validation live under `datasets/`. Training consumes the accepted packet; evaluation consumes the packet, selected adapter, and separately supplied real-report cases. Neither training nor evaluation imports a historical runner, and evaluation does not import training. Version numbers identify immutable packets and runs, not maintained Python modules.

**Tech stack:** Python 3.11, JSON/JSONL, `jsonschema`, pinned Qwen tokenizer and Transformers/PEFT/Accelerate/bitsandbytes stack, CUDA on one NVIDIA L4 for the recorded recipe. Development checks use Ruff, a type checker, and pytest. Private reports, packets, checkpoints, and review artifacts remain under gitignored `data/`.

**Status — 2026-10-04:** This remains the agreed migration specification. Local Stages 1–12 and 16 were implemented and verified; Stages 13–15 have completed procedure and command review, while their cloud training, adapter evaluation, and result comparison remain manual. The accepted `ca1-sft-v3` packet has content digest `20d31cda44e955c60963b3da0f61add86cd20648e1041b69653d798a28af255f` and was generated from clean source commit `43294f56ff6f59ae57d7d8264d06159de3af2b16`. [Reproduction](reproduction.md) records the executed local gates, private archive hashes, and pending cloud commands. The checkboxes below preserve the approved stage specification rather than serving as an execution log.

**Execution boundary for the refactor:** Local source validation, dataset/packet generation, packet acceptance, `train_ca1 check`, offline tests, linting, typing, and documentation checks are in scope. `train_ca1 check` validates packet data without loading or training a model. Do **not** run `train_ca1 preflight` (which performs a forward/backward pass), `train_ca1 train`, GPU/MPS model workloads, adapter-dependent evaluation, or any cloud-VM command in this implementation session. Stages 13–15 only document those later procedures. Manual cloud training may start after the refactor and packet are complete; matched evaluation must then wait for the newly trained adapter.

## Change classes

| Class | Scope and acceptance rule |
| --- | --- |
| Behavior-preserving refactor | Move current generation, packet validation, training, evaluation, and scoring logic into active modules. Keep model-facing rows, token labels, recipe, and scoring semantics equal; prove this with focused tests and parity checks. |
| Repository reorganization | Move the maintained scripts, requirements, tests, and documentation to the layout below. Update imports and commands; path and manifest provenance may change. |
| Intentional correction | Change an inaccurate target, data-quality rule, or implementation behavior only with an identified case, rationale, review evidence, and a separate before/after comparison. Do not fold such a change into structural parity. |
| Dataset regeneration | Build and review a new immutable packet with the cleaned code. The old packet remains private historical evidence; its digest and generator-source layout need not stay loadable by active code. |
| Retraining and reevaluation | Document the later manual cloud run and its three matched evaluations. They are outside this refactor session; report new metrics only after those procedures actually run, alongside the recorded baseline and identified deviations. |

At every stage, update affected imports, command examples, tests, and documentation in the same implementation change; retain an old entry point only until its replacement is verified. A stage gate requires its preserved behavior and identified risks to be checked before obsolete code is deleted. The final result must be generated from a clean, committed source revision so packet provenance identifies reproducible code.

## Invariants and boundaries

- Keep the CA1 research prompt and schema separate from the production Gemini v4 prompt. No adapter is deployed to the application.
- Preserve, unless a documented correction requires otherwise: 320 synthetic training and 64 validation reports (including the corrected 16/8 smoke pairs), IDs, splits, families, rendered input text, target JSON, chat messages, cached token IDs and labels, prompt/schema bytes, pinned Qwen revision `cdbee75f17c01a7cc42f958dc650907174af0554`, and the 8,192-token training limit. Assistant JSON through EOS is supervised; system/user, post-EOS newline, and padding labels are `-100`.
- Preserve the completed training recipe for comparison: one L4, NF4 with double quantization, BF16, LoRA rank 16/alpha 32/dropout 0.05 on all linear layers, microbatch 1, accumulation 8, three epochs, cosine schedule, learning rate 1e-4, warmup 4, seed 42, evaluation at steps 40/80/120, checkpoint saves every 10 steps. A changed dependency or hardware configuration must be recorded as a deviation.
- Preserve matched base/adapter evaluation: one pinned base model, adapter toggled on that model, greedy unconstrained generation, EOS stopping, same prompt/tokenizer/precision per pair, per-case output before advancing, exact scoring with no string normalization. D0018 alone receives 16,384 total tokens; the other cases use 8,192.
- Development cases are diagnostic because their layouts informed generation. The 11 reserved reports were already evaluated; they are a historical comparison, never a fresh holdout or a source for checkpoint selection. New final claims need new held-out reports.
- Do not launch provider requests, repeat exhausted parser/teacher jobs, reset request ledgers, or copy medical text into tracked files. Preserve all private source/review artifacts, old packets, responses, ledgers, model archives, and hashes under `data/`.
- Use Git history for obsolete **tracked** source code. The currently ignored `research/extraction/archive/` is not automatically recoverable from Git; inspect it for unique technical evidence before removing its references or local contents.
- Keep the existing `scoring.strict_pass` definition for historical comparability: it requires empty target and prediction warning lists. Its 47/64 synthetic result differs from 49/64 full JSON matches; do not silently relabel old scores.

## Target layout and module contracts

```text
research/extraction/
  README.md                         short current index
  implementation_plan.md            this migration specification
  reproduction.md                   canonical execution procedure
  known_issues.md                   current problems and uncertainties
  next_steps.md                     future work only
  datasets/
    __init__.py
    dataset.md                      source roles, CA1 contract, construction, audit
    packet.py                       shared packet paths, model/tokenizer pin, validation/load
    synthetic_reports.py            in-memory fictitious scenarios, rendering, visible targets
    build_ca1_dataset.py            source loading, generation, packet writing/verification CLI
  training/
    __init__.py
    training.md                     training-specific recipe and environment detail
    train_ca1.py                    check/preflight/train CLI and run record
    requirements-training.txt
  evaluation/
    __init__.py
    evaluations.md                  maintained methodology and new CA1 results
    historical_evaluations.md       previous CA1 run and older baselines/provenance
    evaluate_ca1_adapter.py         dataset selection, generation, resume, aggregation CLI
    scoring.py                      exact compare/summarize
  tests/                             focused current offline tests
```

`datasets.packet` owns the shared packet paths/settings, integrity checks, acceptance checks, and loader. A packet contains the prompt, schema, tokenizer files/provenance, messages, token cache, `manifest.json`, statistics, and `SHA256SUMS.json`; immutable review provenance may also remain in the payload. Real evaluation cases stay in a separate private role-indexed directory. Training reads a self-contained packet and does not need old correction/smoke directories or historical generator source files at their old paths.

The packet has four distinct states/operations:

1. **Generate:** Write an immutable candidate payload. `manifest.json` records dataset identity, source Git commit, and source hashes; it does not become an acceptance flag. Generation does not create `acceptance.json`.
2. **Verify integrity:** `SHA256SUMS.json` lists SHA-256 hashes of every payload file by sorted relative path, excluding itself and `acceptance.json`. Write it as UTF-8 JSON with sorted keys, compact separators, and one trailing newline. Recompute the complete file list and each hash; reject missing/extra files or mismatches. Define `content_sha256` as SHA-256 of the exact canonical `SHA256SUMS.json` bytes. This digest covers the immutable payload, including `manifest.json`, without referring to acceptance metadata.
3. **Establish evidence:** Retain a separate, hashed evidence artifact under gitignored `data/`. `model_facing_parity` requires a passing comparison against the previously reviewed packet anchored by its recorded digest. `reviewed_changes` requires a complete review of every intentional model-facing difference. The evidence identifies the candidate `content_sha256` and its basis; checksum validity alone is never review evidence.
4. **Accept:** A small operation validates the selected evidence against the candidate and writes `acceptance.json` atomically without changing the payload or `SHA256SUMS.json`. The record contains `content_sha256`, the producing source Git commit matching `manifest.json`, the basis (`model_facing_parity` or `reviewed_changes`), and the evidence SHA-256/reference; an acceptance timestamp is optional. Exclude `acceptance.json` from the content digest to avoid a self-reference cycle.

Integrity verification and acceptance verification are separate functions. The loader first verifies payload integrity, then requires a well-formed `acceptance.json` whose digest matches the exact content loaded, whose source commit matches the manifest, and whose basis/evidence fields are valid. The acceptance operation checks the evidence file and records its digest; the detailed evidence is retained for audit but is not a historical source-path dependency of cloud training. Do not hardcode a packet digest in Python merely to accept a new reviewed packet. A changed payload or rewritten checksums changes `content_sha256` and invalidates the existing acceptance record.

Planned CLI contracts, to be confirmed in implementation:

- `python -m research.extraction.datasets.build_ca1_dataset --source-dir PATH --source-index PATH --tokenizer-dir PATH --output PATH` generates a candidate; `--verify --output PATH` verifies payload integrity only. `--source-index` points to the retained `sft-smoke-v1/sources.json`, a build input rather than a runtime dependency.
- `python -m research.extraction.training.train_ca1 {check,preflight,train} --packet PATH`, with `--output PATH`, `--resume PATH`, and `--report-to {none,wandb}` for training.
- `python -m research.extraction.evaluation.evaluate_ca1_adapter --packet PATH --adapter PATH --output PATH --device cuda --dataset {synthetic,development,reserved}`; the two real sets additionally use `--real-data PATH`, and `--resume` retains its meaning.
- `python -m research.extraction.datasets.packet accept --packet PATH --basis {model_facing_parity,reviewed_changes} --evidence PATH` is the planned minimal acceptance command; the parity basis also supplies the recorded prior-packet digest as an argument. Stage 12 and `reproduction.md` define its gates. Final flag spelling may change during implementation, but the separate acceptance record and two bases may not be collapsed into generation or checksum verification.

Do not add a generalized configuration framework, plugin system, or role-specific subpackages. Keep names and signatures close to the current code unless a boundary above requires a change.

## Baseline to preserve as evidence

At `138c2961d3ac6771c4af06bc80b02417dcffa07b` (`llm-training`; merge base with `main`: `e40f9209f6571abc324b8ff71df6f63bca4c3d02`), the accepted `sft-pilot-v2/SHA256SUMS.json` has SHA-256 `bad1ed2e70de8886a1bb2a57e2ce9c782009e2e2c189912168dd1c570601b4e0`. The selected historical adapter is `checkpoint-80`: 120 optimizer steps, 4,990 seconds, 14.18 GiB peak allocated; validation losses at steps 40/80/120 were 0.01391129/0.01355948/0.01509854. The adapter archive digest is `e880cf8637d8df52f80a0073f0922ea176fd4d4fa5db56bef85ad621893f2344`.

| Matched exact records | Base | Adapted |
| --- | ---: | ---: |
| 64 synthetic validation reports | 1,034/1,251 | 1,219/1,251 |
| 13 real development reports | 94/292 | 131/292 |
| 11 real reserved reports, including D0018 | 45/147 | 58/147 |

The prior synthetic, development, and reserved evaluation archive digests are respectively `f82de82d12bd36fbc41c070eeb3e00d4c86b13f5f00806f64fdedccdbf594b77`, `d0ceeb1e2e8af2ff62e8e2ba3499d005a7309f76ead89eaeadd99e66c9e8826a`, and `fc31b47eae2d3f56d81e8a0ddfeeaba95383043028ea290a913a35a376475588`. They remain historical evidence even after source deletion. See the current `evaluations.md` and private `data/research/extraction/ca1-real-eval-review-v1.md` for case-level findings.

## Migrated documentation evidence

The current issue/evidence ledger is [known_issues.md](known_issues.md); future actions are in [next_steps.md](next_steps.md). Dataset policy, source roles, and the v1 distribution audit are in [datasets/dataset.md](datasets/dataset.md). Historical scores and provenance remain in [evaluation/historical_evaluations.md](evaluation/historical_evaluations.md), with the old tracked source available at commit `138c2961d3ac6771c4af06bc80b02417dcffa07b`. This section records the completed documentation migration rather than a second issue ledger.

## Staged tasks and gates

### Stage 1 — Record the behavioral and experimental baseline

**Files:** Read the current generators, trainer, evaluator, scorer, tests, tracked docs, private packet manifests/reviews, and `git diff main...HEAD`. Record evidence in this plan's baseline section and later in `evaluation/historical_evaluations.md`; do not edit implementation yet.

- [ ] Record the implementation commit, accepted packet digest, source/prompt/schema/tokenizer hashes, run recipe, selected checkpoint, archive digests, case membership, scores, and known limitations. Include the source index, corrected smoke JSONL, and real development texts that the current generator reads. Preserve the original separate Studio input-archive provenance; the later unified archive was assembled after the recorded runs.
- [ ] Run only offline baseline checks in the implementation session: existing focused tests, `train_ca1 check`, v2 `--verify`, and Ruff/type-check status. Capture failures rather than assuming the branch is lint-clean.
- [ ] Generate a fresh **old-code** candidate into a new temporary directory if the private source data and pinned tokenizer are present (historical entry point at the baseline revision: `python -m research.extraction.generate_sft_pilot_v2 --output PATH`; verify with the same module's `--verify --output PATH`). Retain it until parity is complete; do not overwrite the accepted packet. Record differences caused solely by subsequent review/acceptance metadata.

**Gate:** Baseline code revision and all comparison artifacts are identifiable. If private inputs are unavailable, stop the parity-dependent migration until they are restored; unit-level work may continue, but deletion and new acceptance cannot.

### Stage 2 — Introduce the package structure

**Files:** Create `datasets/__init__.py`, `training/__init__.py`, `evaluation/__init__.py`; update `pyproject.toml` and tracked `poetry.lock` for local research dependencies; prepare the target module and documentation paths. Keep old entry points temporarily during migration.

- [ ] Move only a working vertical slice at a time. Make all new commands runnable from the repository root with `python -m`; resolve the repository root once in `datasets.packet` or pass paths explicitly, since nesting adds one directory level to `__file__`.
- [ ] Update relative imports to absolute `research.extraction.*` imports. Avoid importing production application modules into the research-only training bundle.
- [ ] Track `poetry.lock` alongside `pyproject.toml` as the reproducible local development/research environment. Add an optional Poetry `research` group with the packet-generation pins `transformers==5.16.1` and `tokenizers==0.23.2`. Declare Torch directly for the offline tests that import it, using a version compatible with the existing lock (currently 2.14.0) and checking CPU execution; keep `jsonschema` in the main dependencies and pytest/Ruff in `dev`. Update `poetry.lock` without unrelated application dependency upgrades. The local setup command becomes `poetry install --with research --no-root`, with no separate pip install. This group is for local preparation and checks; `training/requirements-training.txt` remains the separate cloud L4 dependency specification, and Poetry must not replace that VM's known-good CUDA PyTorch installation.
- [ ] Verify the package paths and document the future entry points; add their import and `--help` smoke test once the modules exist in Stage 8.

**Gate:** The package paths import without loading historical modules. A clean Poetry install with the optional `research` group exposes the pinned tokenizer stack and runs the Torch-dependent offline tests without a GPU; inspect the lockfile diff for unrelated application changes. No generated data changes.

### Stage 3 — Extract current dataset logic from historical modules

**Files:** Create `datasets/synthetic_reports.py` and the generation-facing portion of `datasets/packet.py`; migrate relevant tests from `test_generate_sft_pilot.py`, `test_generate_sft_pilot_v2.py`, `test_prepare_sft.py`, and `test_correction_variant.py`.

- [ ] From `generate_sft_pilot.py`, move only behavior used by the current v2 generator: family catalog/assignment and bucket rules, panels and value/date generation, visible-cell target/evidence logic, split/provenance and near-duplicate checks, tokenizer-label boundaries, and statistics. Keep v1-only `raw_rows`, rendering, packet building, and v1 verification out of active code.
- [ ] From `prepare_sft.py`, move the model/revision pin and `completion_tokens` behavior. Replace the `evaluate_image_extraction.strict_validate` dependency with an unfenced `json.loads` parse followed by `Draft202012Validator` and `FormatChecker` in the active dataset code. Preserve the current rejection behavior for invalid content and schema errors; test the error contract. This removes the transitive image/local-evaluator import chain.
- [ ] Test visible evidence, numeric and qualitative-polarity preservation, missing/ambiguous cells, warnings, dates, family isolation, reserved-source rejection, token prefix/EOS/post-EOS masking, and the 8,192-token limit with fictitious data.

**Gate:** The current v2 generator can use the extracted functions with identical in-memory cases; old and new focused tests pass. Do not delete the old modules yet.

### Stage 4 — Migrate dataset generation

**Files:** Rename/replace `generate_sft_pilot_v2.py` with `datasets/build_ca1_dataset.py`; keep scenario definitions/rendering in `datasets/synthetic_reports.py`. Replace `tests/test_generate_sft_pilot_v2.py` with current builder/report tests.

- [ ] Keep the 14 current layout styles, fictional assay catalogs, seed, 32 training/10 validation families, 320/64 counts, and 16/8 smoke membership. Separate source loading, case creation, deterministic statistics, and packet writing into a few named operations; do not introduce a generic template engine.
- [ ] Add explicit `--source-dir`, `--source-index`, `--tokenizer-dir`, `--output`, and `--verify` paths. `--source-dir` is the retained correction packet; `--source-index` is the retained smoke packet's `sources.json`. Refuse to overwrite an existing packet. Bundle prompt, schema, and pinned tokenizer provenance/files with the new training packet; keep real evaluation inputs separate.
- [ ] Record exact source input hashes and the clean producing Git revision in immutable `manifest.json`. Write canonical `SHA256SUMS.json` over the complete immutable payload, excluding itself and the absent `acceptance.json`. A new build remains a candidate; no acceptance status is flipped in `manifest.json` or a payload review file.

**Gate:** Old and new generators produce identical model-facing messages, targets, IDs/splits, token IDs, and labels under the same inputs, or every difference is documented as a proposed intentional correction before proceeding.

### Stage 5 — Migrate the packet contract

**Files:** Complete `datasets/packet.py`; move loader/label tests from `tests/test_train_ca1.py` to `tests/test_ca1_packet.py`.

- [ ] Move `load_packet`, label validation, common paths/model pin, JSONL parsing, prompt/schema/tokenizer checks, membership/family checks, and the chat wrapper into the packet module. Implement separate payload-integrity and acceptance-verification functions using the contract above; the loader calls both.
- [ ] Make the packet sufficient for training/check/preflight without `sft-correction-v1` or `sft-smoke-v1` siblings. Keep their hashes and the producing source commit as manifest provenance, not as files that the trainer must locate. The accepted content digest lives in `acceptance.json`, not a Python constant.
- [ ] Test changed payload bytes, missing/extra payload files, changed checksums, wrong counts, duplicate IDs, split/family leakage, prompt/schema/tokenizer drift, malformed assistant labels, and a clean-clone path with fictitious fixtures. Test a candidate without `acceptance.json`, malformed/unsupported basis, missing evidence digest, source-commit mismatch, and an acceptance digest left stale after a payload/checksum change. Integrity may pass for a candidate; loading for training must still fail.

**Gate:** Payload integrity can be checked independently of acceptance. The loader rejects tampering and every unaccepted/invalid acceptance state, and imports no old preparation/evaluation modules. Do not weaken validation to make parity pass.

### Stage 6 — Migrate training

**Files:** Move `train_ca1.py` and `requirements-training.txt` into `training/`; keep training-only tests in `tests/test_train_ca1.py`.

- [ ] Import packet validation/settings from `datasets.packet`. Preserve collation, longest-example CUDA preflight, NF4/LoRA model setup, training arguments, checkpoint selection, resume guard, and `run.json` fields. Remove only redundant packet-loading code.
- [ ] Make `check` offline and `preflight`/`train` explicitly CUDA-only. All three stages must call the same accepted-packet loader before model loading or output creation, so a candidate, missing record, mismatched content digest, or invalid record is rejected. Preserve `--report-to` and resume semantics. Record the **new** source commit and package/hardware versions in the run record.
- [ ] Update the research-only bundle test to include `datasets.packet`, not just the trainer source file. Keep `training/requirements-training.txt` for the L4 image's CUDA PyTorch 2.8.0+cu128 environment; do not treat the local Poetry group's Torch version as the cloud training pin.

**Gate:** Offline trainer/loader tests pass, and `check` rejects the still-unaccepted regenerated packet. Stage 12 verifies that `check` accepts it only after valid `acceptance.json` is created. Preflight and training remain cloud steps.

### Stage 7 — Migrate evaluation and scoring

**Files:** Move `evaluate_ca1_adapter.py` and `scoring.py` into `evaluation/`; split evaluation tests from the current `test_train_ca1.py` and retain scoring tests from `test_local_extraction.py`.

- [ ] Import packet/settings, including the prompt and schema, from `datasets.packet` and `compare/summarize` from `evaluation.scoring`. Keep the synthetic, development, and reserved roles; use one explicit dataset selector. The separate real-data directory supplies only the role index, input text, and reviewed targets. Keep real role membership and D0018's distinct budget.
- [ ] Preserve matched base/adapter precision, prompt, decoding, EOS handling, case output, aggregate keys, and `strict_pass` meaning. Add saved-case identity/completeness validation before `--resume` skips it; record adapter artifact digest so changed weights cannot silently reuse output.
- [ ] Test malformed/non-EOS output, D0018 limit, exact scoring with duplicates/order/warnings, role membership, interrupted output, and resume with a corrupt or mismatched case.

**Gate:** Offline evaluator tests pass with fictitious cases; no model loading or adapter evaluation is needed to complete this stage.

### Stage 8 — Migrate and simplify tests

**Files:** Keep focused `tests/test_ca1_packet.py`, `test_synthetic_reports.py`, `test_train_ca1.py`, `test_evaluate_ca1_adapter.py`, and `test_scoring.py` (a separate builder test may be kept if it improves clarity). Remove old-script tests only after their current behavior is covered.

- [ ] Replace module-import-time reads of private `data/` with small fictitious schema and tokenizer fixtures. Keep private accepted-packet checks as separate integration tests that skip clearly when artifacts are absent.
- [ ] Move useful assertions from correction, v1 generator, source preparation, and local evaluation tests; do not carry tests of exhausted request machinery into the current suite.
- [ ] Add a smoke test for imports and `--help` of all three current CLIs from a research-only bundle. Ensure `pytest --noconftest` on a clean checkout runs the current unit suite without private data, GPU, network, or application database.

**Gate:** The maintained suite covers the current pipeline and runs independently of historical source modules.

### Stage 9 — Verify model-facing dataset parity

**Files:** Write comparison output under gitignored `data/research/extraction/` or temporary storage; summarize only hashes/counts/differences in tracked docs. Do not change the old accepted packet.

- [ ] Verify the old accepted packet's checksum-manifest digest and every listed payload hash before using it as the parity reference. Compare old and new `train.jsonl`, `validation.jsonl`, and `tokenized.jsonl` by parsed content and, where serialization matches, bytes. Compare IDs, roles, split/family membership, prompt, user text, assistant JSON, target hashes, token IDs/labels, coverage statistics, and the 4,742-token historical maximum.
- [ ] Ignore expected new code hashes, file paths, acceptance metadata, review timestamps, and checksum-manifest bytes during **model-facing** parity; still record them for provenance. Write a passing parity evidence artifact only after all model-facing comparisons pass; bind it to the new `content_sha256` and the old accepted packet digest.
- [ ] Investigate every content mismatch. Either correct the refactor to restore parity or document the intentional correction, rationale, affected cases, new review requirement, and expected effect before proceeding.

**Gate:** A saved parity report is reviewable. If model-facing differences remain, prepare `reviewed_changes` evidence covering each changed example instead of labeling the packet parity-equivalent. No historical code deletion or new packet acceptance occurs with unexplained differences.

### Stage 10 — Remove obsolete tracked implementation

**Files to delete after gates:** `generate_sft_pilot.py`, `prepare_sft.py`, `prepare_correction_variant.py`, `evaluate_ca1.py`, `evaluate_image_extraction.py`, `evaluate_local_extraction.py`, `evaluate_transformers_extraction.py`, `run_extraction_baseline.py`, and tests dedicated to them, including `tests/host/` if it has no current consumer.

- [ ] Search all Python imports, command strings, tests, docs, and bundle manifests for each candidate. Retain only behavior actively used by the new modules; migrate any missing rule/test/evidence first.
- [ ] Keep `research/image_redaction/`, application extraction code, and shared application fixtures; they have independent responsibilities.
- [ ] Do not add tracked source archives. Record the old commit and relevant hashes in historical documentation. Inspect ignored `archive/` for unique notes before removing references or its ignore rule.

**Gate:** No current code or maintained command imports a deleted module; tests and docs still locate historical results through artifacts and Git history.

### Stage 11 — Consolidate maintained documentation

**Files:** Shorten `README.md`; create `datasets/dataset.md`, `training/training.md`, `evaluation/evaluations.md`, `evaluation/historical_evaluations.md`, `known_issues.md`, and `next_steps.md`. Merge/remove root `plan.md`, `preparation.md`, `correction_policy.md`, `distribution_audit.md`, `training.md`, and `evaluations.md` after migrating their unique content. Update `research/README.md`, main `README.md`, project/application/personal guides, sample-fixture links, and `AGENTS.md` references where affected.

- [ ] Put current navigation in `README.md`; operational commands only in `reproduction.md`; dataset rules and v1 failure evidence in `datasets/dataset.md`; training recipe detail in `training/training.md`; new-run methodology/results in `evaluation/evaluations.md`; the recorded v2 CA1 run and older pre-SFT scores/provenance in `evaluation/historical_evaluations.md`. Until the new run exists, `evaluation/evaluations.md` may point to the previous CA1 baseline without presenting it as a new result.
- [ ] Keep current unresolved annotation/data quality, model regressions, evaluation limits, and implementation uncertainties in `known_issues.md`. Keep future experiments and improvement ideas in `next_steps.md`. Results may describe observations and link to the issue ledger, but must not maintain competing issue/task lists.
- [ ] Remove conversational/personal phrasing from maintained prose. Do not edit sealed private historical artifacts to refresh old status wording. Update every link and command affected by new paths. The untracked `sft_experiment_handoff.md` and ignored `archive/` are context, not Git-backed active documentation; migrate unique evidence before considering either for removal.

**Gate:** All maintained links resolve in a clean clone except explicitly labeled private artifacts; no stale claim that training has not run remains in current docs.

### Stage 12 — Generate and accept the new packet

**Files/artifacts:** New private immutable candidate packet under `data/research/extraction/`; `manifest.json`, canonical `SHA256SUMS.json`, and later `acceptance.json` in that packet; parity or reviewed-change evidence retained separately under gitignored `data/`. No accepted digest is added to training Python source.

1. **Generate a candidate.** Build into a new directory and record source hashes, clean producing Git commit, model/tokenizer revision, counts, and payload provenance in `manifest.json`. Do not write `acceptance.json` or change a payload status to accepted.
2. **Verify integrity.** Recompute the complete payload file list and SHA-256 values in `SHA256SUMS.json`; check prompt/schema/tokenizer, split isolation, labels, and source provenance. Compute `content_sha256 = SHA256(SHA256SUMS.json bytes)`. Integrity alone does not authorize training.
3. **Produce evidence.** For unchanged model-facing content, save passing `model_facing_parity` evidence tied to this `content_sha256` and the old accepted packet digest. If content changed intentionally, save `reviewed_changes` evidence identifying every changed example and its completed review. A failed or incomplete comparison cannot be relabeled as parity.
4. **Run the acceptance operation.** Supply the packet path, one of the two bases, and the evidence file. It validates the evidence result, packet digest, source commit, a caller-supplied previously reviewed packet digest against parity evidence, and changed-example coverage/review for intentional changes. The prior digest is provenance supplied at acceptance time, not a Python constant. Reject missing, stale, failed, or incompatible evidence.
5. **Create `acceptance.json`.** Write it atomically only after step 4 passes, with the content digest, producing source commit, selected basis, evidence digest/reference, and optional timestamp. Refuse to overwrite a different existing acceptance. Do not rewrite `manifest.json`, payload files, or `SHA256SUMS.json`.
6. **Re-verify the accepted packet.** Run integrity verification again and the separate acceptance check. Confirm the content digest still matches `acceptance.json`; preserve the old packet and its hashes.
7. **Confirm the training gate.** Run offline `train_ca1 check --packet PATH` and confirm that it accepts this packet. Keep tests proving that `check`, `preflight`, and `train` reject candidates and invalid records before any model load. Do not execute preflight or training in the refactor session.

**Gate:** One identified new packet is accepted through its artifact-level record and is fully verifiable from the cleaned code. The exact CLI spelling and packet directory name are implementation-time details to finalize in `reproduction.md`; the acceptance mechanism above is fixed.

### Stage 13 — Finalize cloud training procedure

This stage is documentation only. Do not execute cloud setup or transfer, GPU preflight, model training, or any command in Section B of `reproduction.md` during the implementation session.

- [ ] Verify that the documented training CLI matches the refactored code.
- [ ] Verify `--help`, argument names, paths, packet contract, and expected outputs using offline CLI inspection only; do not load model weights or run preflight.
- [ ] Finalize the cloud-VM setup and training instructions in `reproduction.md`.
- [ ] Leave operator-specific values clearly marked where they cannot yet be known.

**Gate:** The cloud training procedure is complete and consistent with the refactored implementation. Training remains pending manual execution on the cloud VM.

### Stage 14 — Finalize post-training evaluation procedure

This stage is documentation only. Do not execute adapter-dependent evaluation or any command in Section C of `reproduction.md` during the implementation session.

- [ ] Verify the evaluator CLI and expected input/output contracts.
- [ ] Finalize commands for synthetic, development, and reserved evaluation.
- [ ] Document how saved outputs and summaries should be verified.
- [ ] Document the dependency on the adapter produced by cloud training.

**Gate:** The post-training evaluation procedure is documented and consistent with the refactored implementation. Execution remains pending until cloud training artifacts are available.

### Stage 15 — Document the post-training baseline-comparison procedure

**Files:** Finalize the comparison procedure in `reproduction.md` and ensure
`evaluation/historical_evaluations.md` contains the recorded baseline needed
for a later comparison. Do not update current-result documents with new scores
until cloud training and model-dependent evaluation have actually been run.

This stage defines how future results produced from the refactored pipeline
must be compared with the recorded CA1 baseline. It does not execute training,
model generation, or model-dependent evaluation.

- [ ] Document the comparison sequence that should be followed after the new
  cloud-trained adapter and evaluation artifacts are available.

- [ ] Require comparison of experiment inputs and provenance before comparing model quality, including:
  - packet identity and digest;
  - source Git revision;
  - model/tokenizer revision;
  - training recipe;
  - dependency versions;
  - hardware/CUDA environment;
  - selected checkpoint;
  - decoding and evaluation settings;
  - any intentional dataset or implementation corrections.

- [ ] Document how to compare the new matched base and adapted arms with the recorded historical results for:
  - synthetic validation;
  - real development;
  - real reserved.

- [ ] Preserve the historical adapted exact-record results in `evaluation/historical_evaluations.md`:
  - synthetic: 1,219 / 1,251;
  - development: 131 / 292;
  - reserved: 58 / 147.

  Preserve the corresponding historical base-arm results as well so that the new adapter's improvement relative to its own matched base can be compared, rather than comparing adapted scores in isolation.

- [ ] Document the metrics and case-level evidence that should be compared, including where applicable:
  - exact records;
  - expected versus produced row counts;
  - schema-valid outputs;
  - exact dates;
  - ordered matches;
  - warning differences;
  - `strict_pass`;
  - full JSON equality;
  - EOS completion;
  - token-budget behavior;
  - known regressions and omissions.

- [ ] Require material case-level differences to be reviewed against the retained source evidence and existing private review artifacts rather than interpreted only from aggregate scores.

- [ ] Document how results should be recorded after manual cloud execution:
  - keep the historical results unchanged;
  - write the new run's results separately in `evaluation/evaluations.md`;
  - update `known_issues.md` only with currently supported findings;
  - update `next_steps.md` only with resulting future work;
  - update `research/README.md` only after the corresponding evidence exists.

**Gate:** The post-training comparison procedure is complete, unambiguous, and consistent with the refactored artifact and evaluation contracts. No new training or evaluation result is required to complete this stage. Actual comparison remains pending until the manually executed cloud training and post-training evaluation artifacts are available.

### Stage 16 — Final repository checks

**Files:** All tracked changes.

- [ ] Run Ruff, the configured type checker, current offline tests, packet verification/check, link and command checks, `git diff --check`, and a readability review of `git diff main...HEAD`. Verify the working tree contains no private reports, credentials, model weights, generated packets, or ignored archive copies staged for commit.
- [ ] Confirm no imports or documented active commands reference deleted modules and the production application still uses its existing Gemini path. Verify applicable Section A commands locally; check Section B/C command names and arguments through offline `--help` and source inspection without executing those sections. Label cloud training and adapter-dependent evaluation as pending manual work.
- [ ] Record any remaining blocked environment-dependent checks in `known_issues.md` and the research status, without claiming they passed.
- [ ] Review `git diff main...HEAD` for readability and maintainability, focusing on:
  - unnecessary complexity;
  - duplicated logic;
  - accidental abstraction;
  - unclear module ownership;
  - stale compatibility code;
  - unnecessary historical dependencies;
  - documentation duplication.

  Resolve material issues without introducing unrelated cleanup.
- [ ] Finalize `reproduction.md` against the implemented local interfaces:
  - replace all locally resolvable implementation-time placeholders;
  - verify local command names, paths, options, and expected outputs;
  - verify the syntax of documented cloud commands against the refactored CLI;
  - clearly retain operator-specific values such as repository URL, VM access, commit SHA, artifact digests, and cloud image details as values to be supplied at execution time;
  - clearly mark cloud training and post-training evaluation as pending manual execution.
- [ ] Confirm that the maintained documentation is internally consistent:
  - `README.md` describes the current structure;
  - `reproduction.md` contains the canonical operational workflow;
  - dataset details live in `datasets/dataset.md`;
  - training-specific explanation lives in `training/training.md`;
  - current evaluation methodology/results live in `evaluation/evaluations.md`;
  - historical results remain in `evaluation/historical_evaluations.md`;
  - current issues and future work are not duplicated across documents.

**Gate:** The refactored repository is locally reviewable and internally consistent; all available local checks pass or have explicit documented blockers; obsolete historical implementation dependencies are removed; the canonical reproduction instructions match the implemented local interfaces; and cloud-dependent training, evaluation, and result comparison are clearly documented as pending manual execution.

## Decisions to resolve during implementation, not guess now

1. The new packet's immutable directory name, final `accept` CLI spelling, and exact `acceptance.json`/evidence field names. The separate content digest and two acceptance bases above are settled.
2. The minimal private real-evaluation bundle layout and verified `--real-data` contract. It must carry exact ID/role membership and approved `input.txt`/`target.json` pairs without including real cases in the training JSONL; the prompt and schema come from the accepted packet.
3. Resolved: mypy 1.16.1 is pinned in the development group, configured in `pyproject.toml` for the active extraction modules, and run with `poetry run mypy`.
4. Cloud VM location and image identifier, exact CUDA/PyTorch environment, repository access URL, and private transfer route. The commands in `reproduction.md` use explicit variables for operator-supplied values; the completed recipe used PyTorch 2.8.0+cu128 on an L4.
