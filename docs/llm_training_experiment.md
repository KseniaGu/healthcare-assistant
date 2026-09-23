# Small-model structured-extraction experiment

Status as of 2026-09-23: **Gemini and bounded local Qwen/Qwen3.5 comparisons complete;
training not started**. The evaluation and organization split are implemented;
subsequent synthetic-data generation and SFT remain proposed work.

## Question and scope

Can a relatively small open-weight model extract laboratory results from LlamaParse
text well enough to replace the proprietary extraction stage, potentially enabling
local inference later?

The target is **LlamaParse output → the v4 structured-result contract**. Keep the
parser fixed, including its visible spelling errors. The first experiment does not
train on PDFs or change anonymization, perform clinical interpretation, infer flags,
normalize units, introduce RL/DPO/GRPO, or optimize deployment. A local extractor
would still leave LlamaParse as an external service; fully local PDF processing is
outside this experiment.

The [prompt work](prompt_evaluation.md) establishes the output contract, reviewed
targets, fresh-run provenance and Gemini comparison. The [application update](v4_application_update.md)
allows the app to retain all supported result types. Neither changes model weights
or demonstrates that fine-tuning is necessary.

## Evidence available

The reconciled development corpus contains nine distinct reports from three
organizations (four A, four B, one C), with 183 named results. Duplicate files are
not additional examples. All documents informed the annotation contract and have
been inspected. The frozen Gemini v4 candidate matches the reviewed dates and result
records on this corpus; warning wording is scored separately.

That supports a small feasibility experiment, not a reliable estimate of broad
medical-document accuracy. A single urine report from C also differs in test type
from much of A/B, so an organization holdout can conflate layout and content shifts.
A large repeated panel does not provide dozens of independent layouts. The already
reviewed corpus cannot become a fully blind benchmark by being relabeled as a test set.

## Proposed minimal sequence

1. **Freeze data and scoring.** Export approved targets with source/parser hashes,
   organization and duplicate-group identifiers, provenance and exclusion notes.
   Keep the original review packet intact. Resolve any export ambiguity before
   building training data; do not use unchecked Gemini outputs as gold.
2. **Evaluate one untuned small model.** Select an instruction-tuned open-weight model
   after checking license, Russian support, context requirements and available
   hardware. Use the same parser inputs and contract as Gemini. Record the model
   revision, prompt, decoding settings and raw responses. The initial model is selected below.
3. **Generate a small synthetic dataset only if there is a useful gap to close.**
   Base templates on two organizations, vary rows, values, units, dates, optional
   cells and exclusions, and derive exact targets from the generating records.
   Start with a few hundred examples as a planning range, then inspect learning
   behavior before increasing volume. This size is a proposal, not a proven requirement.
4. **Run one supervised adapter fine-tune.** Use SFT with LoRA; use QLoRA only if
   memory requires it. Keep one base model, one fixed split and a small training
   configuration. Select the checkpoint on validation data from the training
   organizations, never on the held-out organization's scores.
5. **Compare and decide.** Evaluate the untuned and tuned versions with identical
   inference settings and compare both with the frozen Gemini baseline. Add more
   training complexity only if the results identify a concrete reason.

## Dataset and split strategy

Split by organization and source/template family **before** generating synthetic
variants. Keep duplicates, near-duplicates, pages and attached panels in the same
split. The third organization's parser examples and layout must not inform synthetic
training templates, validation, checkpoint selection or later prompt revisions.
Organization B is now reserved, as recorded below.

The first target consumes parser text, so synthetic parser-text/JSON pairs are the
simplest starting point. They should preserve realistic column order, shared units,
wrapping and parser artifacts visible in the two training organizations. If creating
synthetic PDFs is necessary to reproduce those artifacts, process them with the
unchanged LlamaParse configuration and inspect the resulting text/label alignment.
That is data preparation, not PDF-to-JSON model training; budget its API use separately.

Use fully fictitious contents and avoid copying residual identifiers from real
reports. Include ordinary cases as well as precision, inequalities, measured ranges,
qualitative values, missing names, incomplete references, historical columns, pending
tests and attached panels. Do not let synthetic examples become only a collection
of adversarial edge cases.

Reserve distinct template variants for validation within the two training
organizations. Keep real organization-held-out reports for a final small-model
comparison. Because those reports already informed the contract, describe them as
held out from small-model training, not unseen to the overall project. Additional
previously unseen reports are needed for a stronger prospective test.

## Comparisons and evaluation

| Comparison | Purpose |
| --- | --- |
| Frozen Gemini v4 | Current proprietary extraction reference |
| Untuned small model | Measures what is achievable without training |
| The same small model after SFT | Isolates the value of training |
| v3 artifacts | Historical diagnosis; not a fresh controlled baseline unless rerun with provenance |

Score schema validity and report dates, missing/extra rows, and exact name, value,
unit and direct-reference strings/nulls. Preserve precision and inequalities when
scoring. Report per-document and per-organization results alongside pooled totals;
a long panel must not conceal failures on shorter reports. Distinguish strict
whole-report correctness from row/field correctness and assess warning semantics
separately from literal warning wording.

Inspect every error in this small corpus. Separate parser-visible extraction errors
from upstream OCR/redaction loss. Record any invented result, changed magnitude or
inequality, unit corruption, reference substitution or date-role error explicitly.
Measure basic inference time and memory on stated hardware for feasibility; do not
start quantization or deployment optimization in this first experiment.

## Success criteria and decisions still needed

Before training, agree SFT acceptance thresholds, training hardware and final
dataset size. The initial model and organization split are selected below; the existing review decisions about report inventory, grouping, panel-date
association and accepted residual anonymization are resolved for the current corpus.

The POC should demonstrate that SFT improves the same model over its untuned baseline
on held-out reports, without introducing critical value/unit/inequality errors. If
the untuned model already meets the agreed quality target, stop and document that
fine-tuning is unnecessary. If the tuned model only improves training-layout results,
do not claim unseen-layout generalization.

Practical value also requires acceptable quality relative to Gemini and plausible
resource needs for the intended local hardware. Matching Gemini on these few reports
is a feasibility signal, not clinical validation or proof of edge readiness. A
negative result is useful if it shows that data diversity, parser quality or model
capacity is the limiting factor. No deployment change follows automatically from
this POC.

## Selected first experiment: local Qwen, A+C development

The user selected `qwen3:4b-instruct-2507-q8_0` on the M5 MacBook Air with 24 GB
unified memory, served by native host Ollama. The installed model and runtime are
pinned by the local evaluator. Initial inference covers A1-A4 and C1 (five reports,
123 results); B1-B4 (four reports, 60 results) are reserved for the later paired
untuned-versus-SFT comparison. This decision supersedes the previously open model
and organization-split choices above. No training has run.

Freeze all approved targets before inference. Develop future synthetic templates,
validation and checkpoint choices from A+C only. Do not inspect new B predictions
until those choices are frozen. B already informed contract development, so describe
it as withheld from subsequent adaptation, never as a fully blind benchmark.

The dedicated local CLI, request limits and scoring definitions are documented in
[prompt evaluation](prompt_evaluation.md#local-untuned-qwen-evaluation). It reuses the
frozen prompt and reviewed targets and compares against saved Gemini predictions;
it does not spend or reset the exhausted Gemini development budget.

A single existing synthetic omissions fixture tests local compatibility and warning
behavior. It is not a newly generated training dataset. No changes to prompt,
precision or inference settings should be selected from this run and silently mixed
into an SFT comparison. Runtime failures need diagnosis before training; meaningful
source-visible errors can justify SFT, while a successful development baseline
motivates collecting more unseen real reports first. Any later training/export must
retain a comparable untuned checkpoint and use the same evaluation settings.


## Recorded untuned outcome — 2026-09-22

The bounded local run completed all six requests: one existing synthetic fixture
and five A+C reports. No B inference, Gemini calls, downloads or training occurred.
All real outputs validated, but Qwen matched only 4/5 dates, 61/123 exact records
and 301/492 fields; saved Gemini matched 5/5, 123/123 and 492/492 respectively.
Qwen omitted the complete 43-row attached allergy panel, mishandled some value
markers/units/references, and produced unsupported warnings on all five reports.
No report passed the complete target. This demonstrates a useful adaptation gap
under the frozen configuration, not a claim about Qwen's maximum performance.

The local source-backed assessment is
`data/extraction_poc/qwen-untuned-v1/assessment.md`; detailed scores and hash-bound
warning adjudications are alongside it. The subsequent model comparison is recorded
below. B stays reserved. The baseline is finished and must not
be rerun merely to obtain a better answer or bypass its execution ledger.

## Follow-up before a training decision

The user approved a bounded comparison of installed Qwen3.5 4B MLX and 9B models
before committing to SFT. This supersedes the immediate next-step suggestion to
start training after the first Qwen3 result. Use the same A+C development packet;
B remains reserved. Two deterministic runs and at most one thinking-enabled run
of the better usable candidate are allowed, with no prompt edits or new datasets.
The protocol, settings and independent request ledgers are documented in
[prompt evaluation](prompt_evaluation.md#authorized-qwen35-comparison).


### Completed comparison and training recommendation — 2026-09-23

The follow-up completed 24 local generation requests: 4B MLX with original
constraints, 9B non-thinking, compatibility-corrected 4B, and selected 4B thinking.
The initial MLX result was confounded by one-character names caused by decoding
pattern handling. A separately recorded amendment removed only that decoding
constraint; full frozen-schema validation and all labels remained unchanged.

Corrected 4B non-thinking matched 70/123 records and 3/5 dates; 9B matched 64/123
and 5/5. 9B recovered all 123 named observations but added a cover row and confused
allergy classes with references. Thinking matched 57/123 ordered records (67/123
as a multiset), 5/5 dates, and correct empty warnings on all real reports. It still
omitted all 43 allergy observations and introduced literal-name, duplication,
unit/reference and qualitative-text errors. None passed a complete real report.
The five real thinking requests took 1,509.82 seconds, versus 142.60 seconds for
corrected 4B and 387.76 seconds for 9B. These are development results, not evidence
of each model's maximum capability or of thinking alone causing the differences.

**Retain the SFT proposal as a small research pilot.** Residual errors are visible
in the input and match specific extraction rules, so task adaptation remains a
reasonable hypothesis. Training benefit is not proven. Verify a supported training
checkpoint and reproducible export/quantization path before committing resources;
an installed NVFP4 inference model does not establish training/export compatibility.
Keep B reserved and compare future tuned weights against a usable untuned baseline
with the same inference settings. Do not train to compensate for the decoding bug.

The complete assessment, discrepancy evidence, timings and one future warning-rule
clarification are in `data/extraction_poc/qwen35-comparison-v1/assessment.md` and
`review_items.json`. No labels were changed; no training or new training data was
created. All allowed inference batches are complete and must not be rerun merely
to improve their scores.
