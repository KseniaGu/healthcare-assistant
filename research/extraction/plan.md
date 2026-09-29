# Correction-aware synthetic dataset for the SFT pilot

**Revision accepted — 2026-09-29.** The [v2 320/64 packet](../../data/research/extraction/sft-pilot-v2/README.md)
was generated, assistant-reviewed and then explicitly accepted by the user in
response to the [v1 distribution audit](distribution_audit.md). V1 remains
failed historical evidence. CA1 policy/prompt approval and real annotation
are complete. Training, GPU provisioning, new model
evaluations and external parser/teacher requests are separate tasks. This plan
replaces superseded literal-only and four-condition proposals.

## Task and size

Input: fictitious **parser-like text**. Output: JSON following the approved
[CA1 contract](correction_policy.md), with English extraction warnings. Do not
rewrite the production prompt or train on real reports in this task.

| Dataset | Training reports | Validation reports | Purpose |
| --- | ---: | ---: | --- |
| Existing CA1 smoke subset | 16 | 8 | Technical training/tokenization checks |
| Revised learning pilot, user-accepted | 320 total | 64 total | Source for the next training-pipeline task |

Counts are complete reports, not rows. Include the existing corrected smoke pairs
within these totals, preserving IDs and split membership. Keep any corrected variant
of an existing example in the same split and record the change; do not rewrite the
frozen packet. The 320/64 v1 dataset is retained for audit, not training;
v2 is the accepted replacement. Generate CA1 targets only; there is no
requirement to generate or train a second literal dataset.

## Allowed source evidence

Read the role manifest as metadata, then inspect only eligible development content.
The [inventory](../../data/research/report_inventory.md) maps all artifacts.

| Role | Sources | Use |
| --- | --- | --- |
| Development | A1–A4 (Invitro), C1 (Gemotest); D0002/D0011/D0012 (Lab4U); D0006/D0007 (CMD); D0014–D0016 (UniLab) | Conventions, realistic text/layout characteristics and development checks |
| Reserved | B1–B4 and D0017/D0018 (KDL); D0003–D0005 (CitiLab); D0008/D0009 (Helix) | Later fixed evaluation only; no generation, training or checkpoint selection |
| Excluded | D0001, D0013 | No eligible target; do not use as generation evidence |
| Layout-only | D0010 (Helix) | No scoring or generation; its organization is reserved |

Development evidence is under `data/research/extraction/sft-correction-v1/real/<ID>/`:
`input.txt`, `target.json`, `review.json`. Real reports remain outside synthetic
train/validation JSONL. Do not reuse real result vectors or patient/admin text.
Keep source reports, pages, panels, duplicates and transformed variants grouped.
Different file hashes do not establish independence. Reserved data have prior
project exposure; describe them as withheld from adaptation, not never inspected.

## Generation requirements

- Generate reproducibly from structured fictitious records and explicitly designed
  layouts. Use a fixed seed and a compact manifest. No synthetic PDFs or API calls.
- Design splits **before** variants. A layout family and base scenario/source group
  belong to one split. Preserve the smoke's eight training and two validation
  family memberships; add independent families where useful. Changing values,
  dates or lab labels alone does not create a new layout family.
- Avoid filling validation with near-clones. Vary row counts, panels, date roles,
  missing information, alignment, wrapping and artifact combinations within each
  family. Validation may share CA1 rules but not training templates/base scenarios.
  Document layout inspiration separately from shared annotation conventions;
  validation need not pretend to have a real source layout.
- Include mostly ordinary reports, clean controls and a smaller difficult subset.
  Suggested buckets: 192 ordinary, 80 layout/parser variation, 48 targeted difficult
  training reports; 38/16/10 validation. These are design defaults, not production
  prevalence. Cover chemistry, hematology, urine and qualitative tests.
- Cross-cover bounded OCR repairs and ambiguity/nulls; name-plus-code cells; paired
  percentage/absolute columns; wrapped cells; precision, commas, inequalities and
  measured ranges; missing/dash units; flags versus qualitative +/-; date precedence;
  direct references versus commentary; attached panels; pending/history exclusions;
  unnamed observations and supported English warnings.
- Derive targets from **the rendered visible text under CA1**. Keep a recoverable
  spelling repair, but never restore deleted numbers, units or names from the hidden
  generating record. A missing value becomes null with the applicable warning; an
  unnamed row is omitted with the required warning. Preserve numeric information
  and qualitative polarity. Record evidence and reasons for repairs/abstentions.

## Implementation and output

Reuse the helpers listed in [preparation.md](preparation.md#reusable-code-for-the-next-dataset).
Implement the smallest research-only generator/validator needed. Existing builders
are frozen-packet materializers, not a general 320-example generator; do not just
raise their count assertions or repeat their templates with different numbers.

The v1 output is retained as failed historical evidence. The replacement was
written to `data/research/extraction/sft-pilot-v2/` without overwriting v1. Its
required contents are:

- `train.jsonl`, `validation.jsonl`: ID plus system/user/assistant messages; the
  assistant content is a JSON string. Use the exact CA1 system prompt and existing
  report wrapper. Keep provenance out of model messages.
- One manifest holding seed, IDs, splits, families, source groups, smoke membership,
  eligible source references, transformations, review status, and prompt/schema/
  tokenizer/code hashes. Avoid a directory or JSON file for every synthetic row.
- Token counts (input, supervised, total; p50/p95/max), coverage and duplicate checks,
  review findings, checksums and one readable README. Save tokenized IDs/labels only
  once if needed for downstream training; they are derivable from the JSONL.

Use Qwen/Qwen3-4B-Instruct-2507 tokenizer revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, cached at
`data/research/extraction/sft-smoke-v1/tokenizer/`. Verify its provenance hashes.
The complete native chat, including JSON completion and end token, must fit **8,192
tokens** with no truncation. Supervise only assistant completion/end tokens;
system, user and later padding labels are -100. SFT has no `response_json_schema`
API argument: validate targets locally, without adding schema text to the prompt.

## Acceptance and handoff

Check every pair for schema/date validity, row/subcolumn coverage, required warnings,
visible evidence, number/polarity preservation, duplicate inputs, near-duplicate
families across splits, excluded provenance, token fit and completion boundaries.
Tests should cover these failure modes with small fictitious fixtures; structural
validation alone is not independent evidence that labels are correct.

Manually review all 64 validation pairs and at least 64 stratified training pairs,
including every family/transformation, every smoke pair, and every rare or ambiguous
case (expand the sample if necessary). A systematic defect requires fixing and
rechecking its entire family/transformation. Mark assistant review separately from
user acceptance. Deliver unresolved cases explicitly rather than guessing.

The [packet review](../../data/research/extraction/sft-pilot-v2/review.json)
records all 64 validation and 85 stratified training pairs reviewed, the fixes,
remaining limitations and the user's acceptance. The [tool guide](README.md#training-pipeline-handoff)
contains exact artifact and label contracts for the next session. No new
inference was part of dataset preparation. Future training design should include
a smoke check before a full pilot and use fresh source weights. Assess untuned
versus tuned models under identical CA1, precision/runtime and decoding
conditions, with reserved reports kept out of tuning.
Hardware, training hyperparameters and spending should be settled in that later task.
