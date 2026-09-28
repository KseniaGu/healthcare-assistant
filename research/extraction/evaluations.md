# Completed extraction evaluations

[Research status and next steps](../README.md) · [Runner instructions](README.md)

## Approved CA1 untuned Qwen baseline — 2026-09-28

Final results for **all 13 real development reports and eight synthetic validation
examples**, evaluated with the approved CA1 prompt. Each example appears once.
[Reviewed metrics and field differences](../../data/research/extraction/qwen-ca1-d0012-retry-v1/combined_assessment.json)
provide the evidence behind the tables; individual outputs are linked below.

Qwen3-4B-Instruct-2507 Q8_0 on Ollama 0.34.3; unconstrained decoding with local
schema validation, 8,192-token context, temperature 0 and seed 42. All 21 saved
responses stopped normally, matched pinned input-token counts and fit the context
limit. Reserved reports were not evaluated. No training has run.

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
preparation, not a new model score or evidence of SFT improvement. The packet awaits
user acceptance; all historical evaluations below are unchanged.

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
