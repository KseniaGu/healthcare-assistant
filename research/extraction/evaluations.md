# Completed extraction evaluations

[Research status and next steps](../README.md) · [Runner instructions](README.md)

## Matched CA1 real-report evaluations — 2026-10-01

The user completed separate Lightning CUDA comparisons on all 13 real
development reports and 11 reserved reports using the same pinned Qwen base,
selected `checkpoint-80` adapter, NF4/BF16 precision, prompt, greedy decoding,
and exact scorer. The [development archive](../../data/research/extraction/ca1-eval-development-v1.tar.gz)
has SHA-256 `d0ceeb1e2e8af2ff62e8e2ba3499d005a7309f76ead89eaeadd99e66c9e8826a`;
the [reserved archive](../../data/research/extraction/ca1-eval-reserved-v1.tar.gz)
has SHA-256 `fc31b47eae2d3f56d81e8a0ddfeeaba95383043028ea290a913a35a376475588`.
The completed runs used the original separate private input archives:
[training](../../data/research/extraction/ca1-lightning-preflight.tar.gz)
SHA-256 `ef7552ffd25a576dd2f43bccc5969666d1eeac6b95efaf03023b3391d29eb5fa`,
[development](../../data/research/extraction/ca1-development-eval-input.tar.gz)
SHA-256 `9308b51eaf0a3beedb55cba5f916c0fbea4b44fa7a1fdb4a6485906189ca6c7a`,
and [reserved](../../data/research/extraction/ca1-reserved-eval-input.tar.gz)
SHA-256 `e9db5134b4b013bc31bb8ec3e696db3df1069ca4811ed516b3cf78f4e09c04e5`.
The later [single Studio input archive](../../data/research/extraction/ca1-lightning-input-v1.tar.gz)
combines the same necessary code and data for simpler setup; it was assembled
after these evaluations and did not produce their saved outputs.

Both manifests say `completed` and match the accepted packet and selected run
record. All 48 saved responses reached EOS. Case membership, individual scores,
and aggregate summaries were reproduced against the local targets. The
[private per-report review](../../data/research/extraction/ca1-real-eval-review-v1.md)
records each pair and selected source-aligned errors.

| Set and measure | Base | Adapted |
| --- | ---: | ---: |
| Development: schema-valid / exact dates | 13/13 / 11/13 | 13/13 / 13/13 |
| Development: exact records / 292 | 94 (32.2%) | 131 (44.9%) |
| Development: exact-record precision | 39.0% | 57.7% |
| Development: fully matching parsed JSON | 0/13 | 1/13 |
| Reserved, 10 standard: schema-valid / exact dates | 8/10 / 7/10 | 10/10 / 8/10 |
| Reserved, 10 standard: exact records / 130 | 42 (32.3%) | 56 (43.1%) |
| Reserved, long D0018: exact records / 17 | 3 (17.6%) | 2 (11.8%) |
| Reserved, all 11: exact records / 147 | 45 (30.6%) | 58 (39.5%) |
| Reserved, all 11: exact-record precision | 35.4% | 40.0% |
| Reserved, all 11: fully matching parsed JSON | 0/11 | 0/11 |

The adapter gains 37 exact development rows and 13 exact reserved rows over
the matched base. The improvement is uneven: it improves exact rows on five
development reports and three reserved reports, loses ground on four and two
respectively, and ties on the rest. Its development output contains 227 actual
rows against 292 expected. The entire 65-row count shortfall occurs in four
reports: A3 misses 43, A2 misses 12, and D0014/D0016 each miss five. A2 is a
new truncation-like omission despite normal EOS; both variants largely omit
A3's visible panel and the paired absolute-count columns in D0014/D0016.
In contrast, the adapted reserved output contains 145 rows against 147
expected: the low exact score there mainly reflects wrong names, values,
units, and reference ranges, not missing row count. `missing_exact_records`
counts all nonmatching rows, so it must not be read as an omission count.

Source checks found consequential content errors: D0005 loses qualitative
polarity, D0015 loses strength/titre detail, and D0008 omits the overall positive
confirmation result; D0003 and D0018 mislabel hormone units; and D0008/D0009
assign administrative dates where targets have no eligible date. The adapted
outputs omit required warnings in B3, D0008, and D0009. The long D0018 case
completed with a 16,384-token budget but regressed from three to two exact
records and omitted one result. Schema validity, normal EOS, and near-correct
row counts therefore do not establish safe extraction. Exact matching also
penalizes harmless formatting differences, so the scores are not a complete
semantic assessment; the cited omissions, polarity, date, and unit errors are
confirmed against the approved targets and selected source lines.

The 64-case synthetic validation recall of 97.4% did not transfer to these
real reports. The development set is exploratory because its layouts informed
the packet. The reserved set was evaluated with the fixed checkpoint, but has
only 11 heterogeneous reports and cannot establish deployment accuracy. This
adapter is **not ready for unattended clinical extraction** or replacement of
the production Gemini path. Keep these reserved results as the historical test;
model or prompt changes informed by them require a new held-out set for a fresh
final evaluation. The next research step is to review the development failures
by source layout and define acceptance criteria before considering another
training or extraction change.

## Matched CA1 synthetic validation generation — 2026-10-01

The user ran the selected `checkpoint-80` adapter and its base model in one
Lightning CUDA evaluation. The downloaded [archive](../../data/research/extraction/ca1-eval-full.tar.gz)
has SHA-256 `f82de82d12bd36fbc41c070eeb3e00d4c86b13f5f00806f64fdedccdbf594b77`.
Its manifest identifies the pinned Qwen revision and accepted v2 packet, and
its run-record hash matches the locally verified selected adapter. It records
NF4/BF16, greedy decoding, EOS stopping, and the same 8,192-token context
limit for both variants. All 64 accepted validation IDs have one base and one
adapted response. All 128 reached EOS, and the largest recorded total context
was 4,008 tokens. I parsed every response, rescored it against the frozen
targets, and reproduced both aggregate summaries. The 13 real development and
all reserved reports were absent from this cloud evaluation.

| Measure | Base | Adapted |
| --- | ---: | ---: |
| Schema-valid reports | 63/64 | 64/64 |
| Exact dates | 40/64 | 63/64 |
| Exact records, ignoring row order | 1,034/1,251 | 1,219/1,251 |
| Record precision / recall | 86.1% / 82.7% | 99.6% / 97.4% |
| Exact records in the correct row position | 949/1,251 | 1,101/1,251 |
| Matching fields at the same row position | 4,290/5,004 | 4,440/5,004 |
| Scorer `strict_pass` | 11/64 | 47/64 |
| Entire parsed JSON equal to target | 11/64 | 49/64 |

| Validation family | Base strict | Adapted strict |
| --- | ---: | ---: |
| Transposed smoke | 2/4 | 3/4 |
| Numbered-block smoke | 0/4 | 2/4 |
| Chemistry | 2/14 | 6/14 |
| Hematology | 4/14 | 11/14 |
| Urine | 1/14 | 12/14 |
| Qualitative | 2/14 | 13/14 |

On paired reports, 36 changed from strict failure to pass, 11 passed under
both variants, and none changed from pass to failure. The scorer's
`strict_pass` requires *empty* warning lists in both target and prediction;
two adapted outputs exactly match targets that contain required warnings, so
the 47/64 strict count understates full JSON equality by two. This is a
scoring convention, not two extraction errors.

The adapted model still misses 28 actual result rows in four two-column
chemistry reports (`v-chemistry-02-01` through `-04`), including 12/29 rows in
`-04`. These account for 28 of its 32 missing **exact** records; the other
four are field substitutions. All seven reports in this two-column chemistry
subfamily fail the strict check. The model also adds one pending confirmation
as a result in `v-chemistry-02-07`, which should have been excluded. Two
chemistry reports contain all correct records but in the wrong order. The one
remaining wrong date is `v-urine-01-07`, where a print date was selected even
though no eligible date is visible. Seven adapted warning lists differ from
their targets: three are wording or punctuation differences for the same issue,
three omit required warnings, and one falsely claims a missing value. The four
field substitutions are a missing unit, an unsupported unit, an added reference
range, and a copied dash where a urine value is missing. Exact-record counts
fell in two chemistry reports despite the aggregate improvement. The 64
synthetic reports show a strong gain in this matched setup, but these failures
prevent a reliability claim for complete CA1 extraction.

This validation split selected the checkpoint by loss, so the comparison is
exploratory. The reports are fictitious and generator-related; they do not
establish performance on real documents. The historical Ollama Q8 baseline is
not a matched control. The subsequent separate real-report evaluations are
reported above.

## CA1 synthetic validation loss after Lightning training — 2026-09-30

The user supplied the completed `run.json` and terminal output for a fresh
three-epoch QLoRA run on one NVIDIA L4. The record pins the accepted v2 packet
SHA-256 `bad1ed2e70de8886a1bb2a57e2ce9c782009e2e2c189912168dd1c570601b4e0`
and Qwen revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Its
trainer-code SHA-256 matches the current `train_ca1.py`. All 120 optimizer
steps completed in 4,990 seconds; peak allocated VRAM was 14.18 GiB. W&B
[recorded the run](https://wandb.ai/gu-fuku-ro-/ca1-pilot-v2/0aw4uf5v).
The selected-adapter [archive](../../data/research/extraction/ca1-best-adapter.tar.gz)
was copied and verified locally (SHA-256
`e880cf8637d8df52f80a0073f0922ea176fd4d4fa5db56bef85ad621893f2344`).
It contains exactly the completed `run.json`, `checkpoint-80` adapter config,
and a readable 132,187,888-byte safetensors file with 504 LoRA tensors. The
full optimizer/checkpoint history remains in Lightning and was not copied.

| Epoch | Optimizer step | Synthetic validation loss |
| ---: | ---: | ---: |
| 1 | 40 | 0.01391129 |
| 2 | 80 | **0.01355948** |
| 3 | 120 | 0.01509854 |

The runner selected `checkpoint-80`, the lowest of the three recorded losses.
Loss rose after epoch two, so the final checkpoint is not the selected
candidate. These are token-level losses on the 64 fictitious validation
reports used for checkpoint selection. Generated-output evidence is reported
above; loss alone does not measure schema compliance or extraction accuracy.
The user's Mac passed a PyTorch 2.8.0 MPS/BF16 tensor check, but model loading
and generation there remain untested. Real and reserved reports were not
training records or checkpoint-selection data; their later generation results
are reported above.

## Approved CA1 untuned Qwen baseline — 2026-09-28

Final results for **all 13 real development reports and eight synthetic validation
examples**, evaluated with the approved CA1 prompt. Each example appears once.
[Reviewed metrics and field differences](../../data/research/extraction/qwen-ca1-d0012-retry-v1/combined_assessment.json)
provide the evidence behind the tables; individual outputs are linked below.

Qwen3-4B-Instruct-2507 Q8_0 on Ollama 0.34.3; unconstrained decoding with local
schema validation, 8,192-token context, temperature 0 and seed 42. All 21 saved
responses stopped normally, matched pinned input-token counts and fit the context
limit. Reserved reports were not evaluated. At the time of this baseline, no
training had run.

| Measure | Real development | Synthetic validation |
| --- | ---: | ---: |
| Saved responses | 13/13 | 8/8 |
| Schema-valid responses | 11/13 | 8/8 |
| Complete reports | 0/13 | 2/8 |
| Exact aligned records / all target rows | 91/292 (31.2%) | 14/20 (70%) |
| Exact aligned records / schema-valid-report target rows | 91/238 (38.2%) | 14/20 (70%) |
| Actual omissions in valid responses | 5/238 | 0/20 |
| Successful repair fields in schema-valid reports | 3/86 | 3/5 |
| Correct abstentions in schema-valid reports | 0/19 | 1/2 |

A complete report requires correct date, all result records, and supported English
warnings. Exact records require all four fields to match; a spelling/unit mismatch
is not necessarily a wrong measurement. The 54 target rows in schema-invalid
A3/D0015 receive zero scoring credit, but are **not 54 observed omissions**.

### Real development reports

| Report / raw output | Schema valid | Exact records / target rows | Complete report |
| --- | --- | ---: | --- |
| [A1](../../data/research/extraction/qwen-ca1-untuned-v1/A1/content.txt) | Yes | 17/20 | Fail |
| [A2](../../data/research/extraction/qwen-ca1-untuned-v1/A2/content.txt) | Yes | 29/32 | Fail |
| [A3](../../data/research/extraction/qwen-ca1-untuned-v1/A3/content.txt) | No | 0/45 | Fail |
| [A4](../../data/research/extraction/qwen-ca1-untuned-v1/A4/content.txt) | Yes | 0/2 | Fail |
| [C1](../../data/research/extraction/qwen-ca1-untuned-v1/C1/content.txt) | Yes | 15/24 | Fail |
| [D0002](../../data/research/extraction/qwen-ca1-untuned-v1/D0002/content.txt) | Yes | 8/12 | Fail |
| [D0006](../../data/research/extraction/qwen-ca1-untuned-v1/D0006/content.txt) | Yes | 0/18 | Fail |
| [D0007](../../data/research/extraction/qwen-ca1-untuned-v1/D0007/content.txt) | Yes | 9/22 | Fail |
| [D0011](../../data/research/extraction/qwen-ca1-untuned-v1/D0011/content.txt) | Yes | 7/26 | Fail |
| [D0012](../../data/research/extraction/qwen-ca1-d0012-retry-v1/D0012/content.txt) | Yes | 6/40 | Fail |
| [D0014](../../data/research/extraction/qwen-ca1-untuned-v1/D0014/content.txt) | Yes | 0/20 | Fail |
| [D0015](../../data/research/extraction/qwen-ca1-untuned-v1/D0015/content.txt) | No | 0/9 | Fail |
| [D0016](../../data/research/extraction/qwen-ca1-untuned-v1/D0016/content.txt) | Yes | 0/22 | Fail |

### Synthetic validation examples

| Example / raw output | Schema valid | Exact records / target rows | Complete report |
| --- | --- | ---: | --- |
| [v01-transposed-01](../../data/research/extraction/qwen-ca1-untuned-v1/v01-transposed-01/content.txt) | Yes | 2/3 | Fail |
| [v01-transposed-02](../../data/research/extraction/qwen-ca1-untuned-v1/v01-transposed-02/content.txt) | Yes | 3/3 | Pass |
| [v01-transposed-03](../../data/research/extraction/qwen-ca1-untuned-v1/v01-transposed-03/content.txt) | Yes | 1/2 | Fail |
| [v01-transposed-04](../../data/research/extraction/qwen-ca1-untuned-v1/v01-transposed-04/content.txt) | Yes | 3/3 | Pass |
| [v02-numbered-blocks-01](../../data/research/extraction/qwen-ca1-untuned-v1/v02-numbered-blocks-01/content.txt) | Yes | 1/2 | Fail |
| [v02-numbered-blocks-02](../../data/research/extraction/qwen-ca1-untuned-v1/v02-numbered-blocks-02/content.txt) | Yes | 2/2 | Fail |
| [v02-numbered-blocks-03](../../data/research/extraction/qwen-ca1-untuned-v1/v02-numbered-blocks-03/content.txt) | Yes | 0/2 | Fail |
| [v02-numbered-blocks-04](../../data/research/extraction/qwen-ca1-untuned-v1/v02-numbered-blocks-04/content.txt) | Yes | 2/3 | Fail |

### Main findings and limitations

D0012 returned the correct date and all 40 observations, with 6/40 exact records
and 124/160 exact fields. It retained 27 `HeT` values, missed other text repairs,
kept microscopy unit text in value fields, and added five disallowed warnings.
D0014 omitted five absolute-count subcolumn results. Other recurring failures were
missed name/unit repairs, copied flags/dashes, altered codes/references and
unsupported warnings. A3/D0015 exceeded warning length limits and selected wrong
dates; raw D0015 also misassigned a result.

Among schema-valid real reports, 149/164 numeric-containing value strings match
exactly. The difference comprises five omitted rows, eight retained flag/prefix
strings and two values containing unit text. The latter also cause two numeric-token
screen differences because the OCR unit contains a digit. This textual diagnostic
is not a claim of clinical safety. All 14 numeric-containing synthetic values match;
all 11 valid real outputs and eight synthetic outputs select the target date.

This is a small development baseline, not an SFT result or final holdout evaluation.
Correlated examples cannot establish whether training will help. Differences in
prompt, target, decoding and runtime prevent causal comparison with historical
v4.1 scores. A future base/tuned comparison must match precision, runtime, prompt,
context and decoding conditions; tuned BF16 versus this Q8 baseline is insufficient.

## Correction-aware preparation — 2026-09-27

[CA1 preparation](correction_policy.md) adds 24 paired real targets and 16/8 matched
synthetic pairs for a future comparison. There are 250 proposed field changes and
20 optional-field abstentions; numeric measurements are unchanged. This is annotation
work, not a new evaluation result. No model predictions, score changes, training or
API calls occurred. Historical results below remain unchanged.

## SFT data preparation — 2026-09-27

The [review packet](../../data/research/extraction/sft-smoke-v1/README.md) contains
16 synthetic training and 8 validation pairs, plus 24 usable real targets with 439
observations. All 22 parser submissions and 14 teacher calls succeeded; unchanged
drafts and source-bound corrections are retained. D0001/D0013 are explicitly
excluded. Ten reserved candidates fit 8K; D0018 needs 9,696 tokens and is retained
for longer-context evaluation. The maximum synthetic conversation is 4,092 tokens.
36 offline tests pass; 658 historical/source hashes remain unchanged. This is data
preparation, not a new model score or evidence of SFT improvement. At that
date the original preparation packet awaited acceptance. The later
[v2 synthetic packet](../../data/research/extraction/sft-pilot-v2/README.md)
was accepted on 2026-09-29. All historical evaluations below are unchanged.

## Contract and scoring

The executable contract is [v4.yaml](../../backend/prompts/laboratory_test_processor/v4.yaml),
evaluated revision v4.1, SHA-256
`5016ee3952d7198a79b5cb50c41da8132d14cd7ca368f6487fb7f7df13b2818b`.
It requires `date`, `results` and `errors`. Rows contain a nonblank `name` and
string-or-null `value`, `unit` and `reference_range`. Preserve precision, parser
spelling, qualitative values, inequalities and direct reference cells/pointers.
Do not infer clinical flags, normalize units or substitute allergy classes.

Choose collection date, then examination/performance date, then referral/order
header date, otherwise null. Omit unnamed observations with a warning; exclude
pending tests and cover referrals. At most five distinct warnings, each at most
160 characters, are allowed. The schema and calendar date are validated locally;
raw failed responses are retained. Schema validity alone does not imply correctness.

| Consumer | Prompt selection |
| --- | --- |
| Application pipeline and Gemini behavior test | `DataSettings.prompt_version = "v4"` |
| Standalone Gemini baseline | `--prompt-version`, default `v4`; historical `v3` comparison supported |
| Direct `GeminiClient.generate()` without a version | `v1` |

Exact scoring compares dates, ordered records and four position-aligned fields.
A secondary multiset score counts duplicates. Missing rows can shift later position
scores; an exact-record substitution counts as both missing and extra. Strings and
nulls are never normalized. Invalid/unattempted responses stay in the denominator
and earn no extraction credit. Warnings receive separate semantic review.

## Original development corpus

Nine distinct reports contain 183 named results: A has four reports, B four, C one.
All informed contract development. Historical Gemini matched 9/9 dates, 183/183
records and 732/732 fields, with correct warning semantics. This was not a blind
benchmark. B is now reserved from small-model generation/scoring and training.

Small-model development uses **A1–A4 and C1**, totaling 123 results and 492 fields.
A3 contains 45 results, including one correlated 43-row allergy panel.

| Configuration | Exact dates / 5 | Ordered records / 123 | Fields / 492 |
| --- | ---: | ---: | ---: |
| Saved Gemini | 5 | 123 | 492 |
| Qwen3 4B Q8_0, schema-constrained Ollama | 4 | 61 | 301 |
| Qwen3.5 4B MLX, original constraints (confounded) | 3 | 0 | 206 |
| Qwen3.5 4B MLX, compatible constraints | 3 | 70 | 310 |
| Qwen3.5 9B Q4_K_M, non-thinking | 5 | 64 | 379 |
| Qwen3.5 4B MLX, thinking | 5 | 57 | 268 |
| Qwen3 4B Q8_0, unconstrained Ollama | 3 | 77 | 350 |
| Qwen3 4B BF16, unconstrained Transformers/MPS | 4 | 106 | 475 |

No small-model configuration passed every complete development report.
The initial MLX decoder restricted names to one character; its score does not
measure extraction capability fairly. A decoding-schema amendment retained the
original post-validation contract. Thinking did not solve the overall task.

The original Qwen3 constrained run omitted the entire allergy panel. Unconstrained
runs returned it. Ollama's invalid A2 warning invalidated the whole response even
though its raw result rows matched Transformers; A1, A4 and C1 rows also matched.
Thus the official score gap is not purely a difference in extracted rows. BF16
returned all 123 observations but made date, spelling, reference and warning errors.

The runtime diagnostic is confounded by Q8_0 versus BF16, templates, Ollama changing
from 0.33.2 to 0.34.3 and amended MPS cache management. Transformers totals combine
separate attempts, including manual A3 with substantial swapping. They are not one
uninterrupted benchmark or proof that Ollama alone caused poor extraction.
The omission warning's missing-unit detail was subsequently accepted as non-critical;
historical scores remain unchanged and this decision is not reopened.

## Three-image pilot: completed 2026-09-26

Selected approved pages: D0007/CMD (22 observations), D0002/Lab4U (12),
D0016/UniLab (22). The user confirmed these organizations are new to the corpus.
All three parser jobs succeeded inside the container. Review found all 56 measured
values and three dates, but 22 damaged/missing unit cells. The user authorized
separate source-backed corrected parser copies; raw output remains immutable.
Targets were independently reviewed and frozen before Qwen evaluation.

Both models used the same corrected inputs and frozen contract. Qwen used
unconstrained Qwen3-4B-Instruct-2507 Q8_0 through Ollama 0.34.3.

| Measure | Gemini | Qwen3 4B Q8_0 |
| --- | ---: | ---: |
| Schema-valid responses | 3/3 | 3/3 |
| Correct dates | 3/3 | 3/3 |
| Observations returned | 51/56 | 51/56 |
| Ordered exact records | 34/56 | 31/56 |
| Position-aligned exact fields | 166/224 | 171/224 |
| Multiset exact records | 34/56 | 32/56 |
| Complete report passes | 2/3 | 1/3 |

Both omitted five UniLab `#`-column measurements. Qwen retained three CMD abnormality
triangles inside values. Lab4U was exact for both. Abbreviation and reference-text
mismatches also reduced strict scores; zero exact ordered UniLab rows does not mean
all returned measurements were wrong. Source-aligned review found no changed numeric
digits or invented units/reference thresholds among returned rows. Inequalities
were not exercised. Empty warnings were correct for all six responses.

Qwen wall times were 53.08, 20.47 and 41.40 seconds at 27.00–28.54 generated tokens/s.
All post-request snapshots showed context 32768 and 9,333,144,288 bytes allocated
on GPU. These are allocations, not peak host-memory measurements. Initial model
loading took 2.56 seconds; later requests reused the model.

The full quality target was not met. This is development evidence on manually
corrected inputs, not natural-parser robustness, a generalization benchmark or
proof that SFT will improve the task. All 3 parser + 3 Gemini + 3 Qwen slots are used.

## Evidence index

These private paths may be absent on another checkout. Do not commit their medical
contents. Failed attempts, corrected derivatives, raw outputs and ledgers remain
part of the evidence, not cleanup candidates.

| Evidence | Local path under `data/` |
| --- | --- |
| Original label and Gemini review | `review/structured_extraction_v4_real_results.md` |
| Frozen original packet | `extraction_poc/qwen-eval-packet-v1/` |
| Original Qwen assessment | `extraction_poc/qwen-untuned-v1/assessment.md` |
| Qwen3.5 comparisons | `extraction_poc/qwen35-comparison-v1/assessment.md` |
| Consolidated Transformers diagnostic | `extraction_poc/qwen-transformers-eager-short-v1/assessment.md` |
| Manual A3 provenance | `extraction_poc/qwen-transformers-eager-a3-manual-v1/provenance_review.json` |
| Image inventory and remaining review flags | `approved_image_redactions/overview.md` |
| Final image pilot | `extraction_poc/image-extraction-pilot-v1/assessment.md`, `scores.json`, `discrepancies.md` |

For historical request settings, amendments and earlier intermediate states, see
[archived evaluation notes](archive/legacy_evaluations.md) and
[archived research notes](archive/legacy_plan.md). Current conclusions live here
and in the [research summary](../README.md), not in those chronological notes.
