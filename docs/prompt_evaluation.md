# Structured extraction: prompt contract and evaluation

Status as of 2026-09-22: **v4 is the application default**, and its evaluated revision
is v4.1 in `v4.yaml`. Prompt development and real-report baseline evaluation are
complete for the current development corpus. The first untuned local Qwen evaluation is recorded below; no small-model training has run.
This work establishes a target and comparison baseline for the
[training experiment](llm_training_experiment.md); application persistence is covered
separately in the [v4 update guide](v4_application_update.md).

## Scope and executable contract

The extraction boundary is **LlamaParse text → structured laboratory results**.
LlamaParse remains unchanged. The YAML prompt's JSON Schema is the executable
contract; [v4.yaml](../backend/prompts/laboratory_test_processor/v4.yaml) is the
source of truth.

- Require `date`, `results`, and `errors`; reject additional fields.
- Each result has a nonblank `name` and string-or-null `value`, `unit`, and
  `reference_range`. Preserve parser spelling, precision, qualitative values,
  measured ranges, inequalities and literal units.
- A reference is the direct reference cell or pointer. Do not substitute allergy
  classes, historical results or commentary intervals.
- Choose collection date, then examination/performance date, then referral/order
  header date, otherwise null. Date provenance remains in review annotations.
- Exclude unnamed observations and describe their omission in `errors`. Exclude
  pending tests and cover referrals while retaining actual attached measurements.
- Allow at most five distinct warnings, each at most 160 characters. Do not emit
  inferred flags, normalized units, or measurement bounds.

Fresh application responses receive local schema/date validation before processed
JSON is written. Failures preserve raw responses and identify the validation location.
Cached historical outputs are not revalidated or counted as fresh baseline results.

| Consumer | Selected prompt |
| --- | --- |
| Application pipeline and Gemini behavior test | `DataSettings.prompt_version = "v4"` |
| Standalone extraction runner | `--prompt-version`, default `v4`; permits `v3` comparisons |
| Direct `GeminiClient.generate()` without a version argument | `v1` |

The configured processor model defaults to `gemini-3.1-flash-lite`; explicit model
selection is recorded in output provenance. The evaluated prompt file is frozen at
SHA-256 `5016ee3952d7198a79b5cb50c41da8132d14cd7ca368f6487fb7f7df13b2818b`.
Its `_meta.description` refers to the original experimental rollout, when the app
still used v3; that historical sentence does not select the active prompt.

## Recorded prompt-development loop — 2026-09-21

Three entirely synthetic parser-text fixtures were paired with expected outputs
before testing. Their [fixture guide](../tests/samples/extraction_poc/README.md)
describes the cases. The loop made three v3 requests, three initial v4 requests,
and three requests after one warning-focused v4 revision: **9/9 allowed attempts**.

All responses were schema-valid. V3 exhibited date-role, measurement-bound and
unit-normalization problems, and included unnamed/pending observations. Initial v4
matched the dates and all ten named records across the fixtures, but mishandled
omission warnings. Final v4.1 retained those correct records and described missing
names/incomplete references without routine-exclusion warnings. It still omitted
the visible values requested in the synthetic omission warning. This remaining
instruction-compliance limitation means the synthetic suite was not a complete pass.

Recorded usage was 7,296 input and 2,677 output tokens, with no quota errors. The
project/model limits verified at that time were 15 RPM, 250,000 input TPM and 500 RPD;
requests used ten-second pacing with automatic retries disabled. These are historical
observations, not current quota guarantees. The development ledger is exhausted.

## Recorded real-report evaluation — 2026-09-21

After the user resolved the review decisions and accepted the residual anonymization
findings for this experiment, a separate authorized run processed the nine distinct
report contents. The corpus comprises four reports from organization A, four from B,
and one from C. Duplicates and the pending/completed-version discrepancy were
reconciled; no missing report is still inferred. The attached allergen panel's shared
collection date was confirmed. These approvals concern this corpus, not future data.

| Measure | Recorded result |
| --- | ---: |
| Schema-valid outputs | 9/9 |
| Exact report dates | 9/9 |
| Exact named result records | 183/183 |
| Exact result fields | 732/732 |
| Missing or extra named results | 0 |
| Correct warning semantics | 9/9 |
| Whole JSON equality, including warning wording | 8/9 |

The remaining whole-object difference was acceptable warning wording. Nine requests
used 20,489 input and 10,674 output tokens. Inputs, prompt/schema hashes, raw responses
and predictions were checked against the reviewed targets. Parser spelling damage
was deliberately retained; this is not proof of PDF-to-result correctness.

All nine documents informed contract development. This is a **development-corpus
result**, not an unseen-organization generalization result. Repeated allergy rows
are correlated observations. Historical v3 artifacts do not have sufficient request
provenance to count as a controlled fresh-v3 comparison on these real reports.
One successful warning does not establish deterministic compliance.

Detailed evidence stays in the gitignored local paths:

- `data/review/structured_extraction_prompt_comparison.md`
- `data/review/structured_extraction_labels.proposed.json`
- `data/review/structured_extraction_v4_real_results.md` and its JSON companion
- `data/extraction_poc/v4-real-reports/manifest.json`

These paths may be absent in a clean checkout. Do not copy medical records into
shared documentation to make those references public.

## Fresh-run tooling

`backend.data_processing.run_extraction_baseline` reads explicit parser-text paths
and calls the prompt engine/Gemini client directly, without initializing PDF
processing or the database. It rejects identical input contents and existing output
directories. Inputs are processed sequentially; failures and unattempted inputs are
recorded, and any incomplete run exits unsuccessfully.

Each output directory contains a prompt snapshot and manifest. Numbered directories
hold requests, raw responses and validated predictions. Provenance includes actual
model/settings, input/prompt/schema hashes, timestamps and available token usage.
Invalid or truncated responses never receive a validated prediction file.

Before any newly authorized run, verify model access and current project RPM/TPM/RPD
limits and remaining allowance in AI Studio, including other clients' usage. The
`--quota-verified` flag is an operator attestation, not automatic quota discovery.
If remaining allowance cannot be established, defer live calls. This runner disables
SDK retries and grounding; application 503 retries remain unchanged.

Command shape for a separately scoped, quota-verified evaluation (replace all
uppercase placeholders with the reviewed paths and verified limits):

```sh
poetry run python -m backend.data_processing.run_extraction_baseline \
  --input PARSED_TEXT_PATH \
  --prompt-version v4 \
  --output-dir NEW_OUTPUT_DIRECTORY \
  --budget-file AUTHORIZED_EVALUATION_LEDGER \
  --max-attempts REQUEST_COUNT \
  --interval-seconds VERIFIED_INTERVAL_SECONDS \
  --quota-verified
```

Repeat `--input` for each distinct parser text; use `--model` only for an explicitly
chosen comparison. `--max-attempts` permits 1–9 and cannot exceed the available
allowance. The runner caps a ledger at nine attempts including failures, locks it
against concurrent runs, and persists pacing across invocations. HTTP 401/402/403/429
stops the ledger; there is no automatic model/project/paid fallback.

Use the same ledger for every batch within an experiment. **Do not reset a ledger or
create a new one to bypass an exhausted or stopped budget.** The original synthetic
loop and the later real-report evaluation were distinct authorized runs; their
completion is not permission to repeat them. Keep the frozen prompt out of further
tuning against the intended small-model holdout.

## Local untuned Qwen evaluation

The standalone `backend.data_processing.evaluate_local_extraction` CLI freezes the
approved corpus, evaluates the installed Qwen model through host Ollama, and scores
saved outputs offline. It does not initialize PDF processing, Gemini, tracing or the
database. Run from the repository's Poetry environment (container Python is
`/opt/venv/bin/python`), not the container's bare system Python.

```sh
poetry run python -m backend.data_processing.evaluate_local_extraction freeze \
  --output-dir data/extraction_poc/qwen-eval-packet-v1
poetry run python -m backend.data_processing.evaluate_local_extraction run \
  --dataset data/extraction_poc/qwen-eval-packet-v1 \
  --output-dir data/extraction_poc/qwen-untuned-v1
poetry run python -m backend.data_processing.evaluate_local_extraction score \
  --dataset data/extraction_poc/qwen-eval-packet-v1 \
  --run-dir data/extraction_poc/qwen-untuned-v1
```

`freeze` and `score` are offline. `run` performs at most six **local** generation
requests: the existing omissions fixture, then A1, A2, A3, A4 and C1. B is reserved
and cannot be selected by this CLI. The five real reports contain 123 named results
and 492 fields. The synthetic request is reported separately, not pooled.

The packet snapshots exact inputs, approved targets, schema, v4.1 prompt, scoring
rules and development Gemini predictions. It reconciles stale row approvals only
in the export, retaining historical statuses and the original review packet.
Historical target-packet hash differences remain documented; no historical hash is
silently replaced. All 46 recorded source artifact hashes are checked on export.

Pinned settings: `qwen3:4b-instruct-2507-q8_0`, digest
`aa7252f68dda4d25dfffa65b3760af6d2c3231a140c0060c78d444686d98a374`,
Ollama `0.33.2`, and `http://host.orb.internal:11434`. The native template and
parameters are hash-pinned. Requests use the unchanged v4.1 messages and schema,
`temperature=0`, `seed=42`, `repeat_penalty=1`, `top_k=20`, `top_p=0.8`, `min_p=0`,
`num_ctx=32768`, `num_predict=16384`, `think=false`, `stream=false`, and
`keep_alive="10m"`; native stops remain in effect. Timeout is 600 seconds.
The fixed endpoint bypasses proxies and rejects redirects; cloud models and changed
model/runtime identities are refused. There is no pull, fallback or automatic retry.

For the verified byte-level tokenizer, a conservative input bound uses UTF-8 message
bytes plus 1,024 chat-format tokens. Bound plus output allowance must fit in 32K.
This avoids a tokenizer download and is explicitly not an exact token count.
The model's allocated context is checked after each request. GPU allocation is
recorded, but allocation snapshots do not measure peak process memory or host swap.

Outputs include `manifest.json`, runtime/template metadata, exact requests, raw
responses, validated predictions, `scores.json` and `review.md`. Raw and available
partial/error responses survive failure. Completed invalid model outputs fail that
report; transport/runtime or compatibility failure stops the batch. Failures and
unattempted reports stay in score denominators. A sibling `.execution.json` ledger
locks the packet and rejects a second attempted run. Do not delete it or export a
new packet to bypass the six-request scope. A failed run may still be scored offline.

Scores use exact parsed strings/nulls, ordered records and position-aligned fields;
secondary multiset precision/recall accounts for repeated records. A substitution
counts as one missing exact record and one extra exact record. Position scores may
cascade after a missing row. Real development targets require no warnings. Synthetic
warning meaning is manually reviewed against the frozen checklist; wording equality
is not a semantic pass. Review findings belong in a separate local assessment file
so re-running the scorer does not erase them. Optional `warning_review.json` maps
`smoke` or development document IDs to `raw_sha256`, `status` (`pass`/`fail`),
`reason` and `reviewer`. The scorer verifies each judgment against its raw response
and applies it to warning/strict-pass metrics; missing judgments remain pending.
Reserved document judgments are rejected.

The implementation's quality target is five correct dates, 123 ordered records,
492 exact fields and appropriate warnings. A quality failure is a valid experiment
outcome. These already-reviewed development reports cannot establish blind
organization generalization, and the repeated A3 panel rows are correlated.


### Recorded local result — 2026-09-22

The authorized six-request run completed without transport, schema, context or
output-limit failures. Real-report results: 4/5 dates, 61/123 exact records,
301/492 fields, 0/5 warning-semantic passes and 0/5 strict report passes. Saved
Gemini predictions on the identical A+C subset score 5/5, 123/123, 492/492, 5/5
and 5/5. Qwen retained 80 named rows, omitting 43 attached-panel observations.
Its exact-record precision/recall are 76.25%/49.59%; substitutions account for
19 of the 62 missing exact records and all 19 extra exact records. They are not
19 invented tests. The synthetic fixture's date/records are correct but warnings
fail the frozen checklist. All warnings received source-backed manual review.

The five real requests took 171.26 seconds total, median 42.05 seconds. Allocation
snapshots showed 32K context and 9.33 GB on GPU; no peak-memory/swap measurement
was made. Faster completion on the attached-panel report reflects omitted output,
so latency must be interpreted alongside quality. These are development results
with correlated panel rows, not a blind-generalization benchmark.

Local evidence: `data/extraction_poc/qwen-eval-packet-v1/` and
`data/extraction_poc/qwen-untuned-v1/`, especially `assessment.md`, `scores.json`,
`warning_review.json` and the raw responses. All six attempts are exhausted for
this packet. Offline scoring can be repeated without inference. The final offline
regression suite passed 34 tests; no PostgreSQL test was needed for this isolated CLI.
