# Proposed next step: prepare a bounded SFT pilot

[Research status](../README.md) · [Completed evaluations](evaluations.md)

**Proposal, not an executed or authorized training run.** The first deliverable is
an inspectable seed-data and evaluation packet, not a synthetic dataset or model.

## Why consider SFT?

The untuned models still make contract and coverage errors on readable inputs.
Source-backed OCR corrections did not prevent either model from missing UniLab's
five secondary-column measurements. Qwen also retained abnormality markers inside
values. These are plausible supervised-learning targets, but table representation
and deterministic marker handling remain alternative explanations or remedies.
SFT will not repair upstream missing text or the host's MPS memory pressure.

## Prepare before generating data

- **Annotation policy:** specify paired percentage/absolute columns without invented
  names or units, direct reference cells, printed abbreviations, marker removal and
  warning semantics. Version any future policy change; retain all historical labels
  and scores. The previously accepted missing-unit detail in an omission warning
  is non-critical and is not a new blocker.
- **Seed inventory:** review privacy, reuse provenance, legibility and source/template
  relationships. Group duplicate pages and reports. The sixteen approved images
  are layout evidence; three now have reviewed targets on corrected parser text.
  The selected new organizations do not overlap the original ones, per the user.
- **Holdout design:** propose organization/template groups before creating variants.
  Keep B reserved; never use its small-model errors for prompt, model or training
  selection. B already informed the contract. Additional withheld real reports are
  needed for stronger prospective evidence.
- **Stable comparison:** pick one base checkpoint and reproducible inference settings.
  Compare untuned and SFT versions using the same runtime, precision, template,
  prompt, constraints, context and decoding. Quantization/runtime diagnostics are
  separate experiments, not evidence of SFT improvement.

## Small training experiment, if approved afterward

Start with fully fictitious parser-text/JSON pairs derived from permitted layouts.
This directly matches the training boundary and is simpler than generating PDFs.
Include ordinary reports as well as missing cells, qualitative values, ranges,
inequalities, date roles and paired columns. Keep validation variants separate
from training and keep every template family together.

Generate PDFs only if a specific data gap requires sampling realistic parser
artifacts. Budget parsing separately and independently check the resulting labels.
Never assume Gemini responses are gold or copy real result patterns into synthetic
training examples without a separate review.

Use one small LoRA SFT configuration; consider QLoRA only if required by training
memory. Select checkpoints on training-organization validation data. No RL,
DPO/GRPO, end-to-end image/PDF training or deployment optimization is proposed.
Dataset size, hardware, cost and acceptance thresholds must be agreed before a run.

## How to decide whether it worked

Keep exact historical metrics, and predeclare a separate source-aligned assessment
of missing observations, values, units, inequalities, references and dates. This
prevents presentation mismatches from obscuring substantive errors without relaxing
exact scores after seeing results. Assess warning meaning separately.

Report paired untuned/SFT changes by report and organization, including complete
report passes, failure rates and runtime. Long correlated panels must remain
visible separately. Improvement on training layouts alone is insufficient. Require
meaningful held-out improvement without new critical value/unit/inequality or
invented-reference errors. Decide numerical thresholds before training; the current
small corpus cannot support a deployment claim even if every example passes.

Stop or revise the proposal if inference is unstable, labels are ambiguous, data
are unsuitable, or the untuned model already meets the agreed target on additional
withheld reports. A negative result is useful; a substantial training run is not
the default next step.
