# Completed extraction evaluations

[Research status and next steps](../README.md) · [Runner instructions](README.md)

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
