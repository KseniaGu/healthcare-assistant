# Laboratory extraction research

**Current status — 2026-09-26:** Untuned-model evaluation is complete for the
current development examples. No training has occurred. A small supervised
fine-tuning (SFT) pilot remains worth investigating, but data and evaluation
preparation must come first. There is no deployment recommendation.

## What are we researching?

Can a small open-weight model reliably turn **LlamaParse text into laboratory-result
JSON**, and would SFT improve it enough to make local extraction useful?

The application currently uses Gemini for this step. The research keeps LlamaParse
and the frozen v4.1 extraction contract unchanged. The contract preserves literal
names, values, units and direct references, with a report date and extraction
warnings. This is not PDF-to-JSON training or medical interpretation. A local
extractor would still rely on an external parser.

## What have we learned?

| Completed work | Main finding |
| --- | --- |
| Contract and Gemini baseline | Reviewed targets cover nine reports and 183 results. Gemini matched their dates and result records. All nine informed contract development. |
| Untuned Qwen3/Qwen3.5 comparisons | Small models made omissions, date mistakes and copying errors. No tested configuration met the full development quality target. |
| Ollama versus Transformers diagnostic | Unconstrained generation recovered the omitted allergy panel. Several result tables matched across runtimes; invalid warnings explain part of the official score gap. Runtime, precision and memory differences prevent a clean causal comparison. |
| Local image redaction | Sixteen approved exports provide a layout inventory. Human approval and export checks do not establish training suitability or reuse permission. |
| Three-image extraction pilot | OCR damaged units; the user approved separate source-backed corrected copies. On those inputs, both Gemini and Qwen omitted five UniLab subcolumn results. Gemini passed 2/3 complete reports; Qwen passed 1/3. |

The residual errors justify a research question about SFT; they do not show that
SFT will solve it. Parser errors, table representation, annotation conventions and
runtime constraints need separate treatment. The image pilot tested corrected
inputs, not natural parser robustness. The Transformers diagnostic combined several
attempts, including a manual A3 run with substantial swapping.

[Experiment results and limitations](extraction/evaluations.md) contains the scores
and evidence links. [The proposed next experiment](extraction/plan.md) explains what
would make an SFT pilot informative.

## What happens next?

The next small task is **reviewed seed-data and evaluation-protocol preparation**:

1. Settle future annotation rules for paired columns, abnormality symbols and
   literal names/references. Keep existing targets and scores unchanged.
2. Check source permission, privacy, near-duplicates and template relationships
   before selecting reusable layouts. Images alone are not training labels.
3. Specify organization/template separation, a stable inference setup and a paired
   untuned-versus-SFT evaluation before generating any synthetic variants.
4. Review that packet and the training budget before authorizing synthetic generation
   or a small adapter SFT run.

**B1–B4 remain reserved for small-model evaluation and training.** They were reviewed
and evaluated by Gemini historically, so they are not blind to the whole project.
Do not generate or score new B predictions, assign new splits, train, change the
prompt or make new live requests as a consequence of this documentation update.
The completed image pilot used all three parser, three Gemini and three Qwen slots.

## Where things belong

| Location | Contents |
| --- | --- |
| [extraction/](extraction/README.md) | Runners, exact scoring, offline tests and execution instructions |
| [image_redaction/](image_redaction/README.md) | Standalone local image preparation tool, fictitious fixtures and tests |
| [extraction/archive/](extraction/archive/README.md) | Detailed historical notes and one-time continuation scripts |
| `backend/`, `src/`, `docs/` | Application code, shared prompt/validation code and application guides |
| `data/extraction_poc/` | Existing private packets, raw responses, targets, scores, ledgers and assessments |
| `data/review/`, `data/approved_image_redactions/` | Existing private review evidence and approved image inventory |

Historical data paths stay in place: manifests, hashes and exhausted ledgers depend
on them. Application data folders also stay in place. For a future separately
approved study, prefer `data/research/extraction/datasets/` and `runs/`; do not move
old packets just to match that convention. All medical artifacts remain gitignored.
A clean checkout does not contain them, and the container's data volume can mask
host files.

## Maintaining this page

After a meaningful research step, update the status, findings and next action here.
Link detailed evidence; do not append a transcript or copy medical content into Git.
Keep completed work separate from proposals, and retain limitations that affect the
conclusions. Put new experiment totals in `extraction/evaluations.md` and operational
changes in the relevant tool guide. Preserve earlier results and provenance.

Latest update: research code, tests and documentation were grouped here on
2026-09-26. This was an organizational change; it added no experimental evidence.
Offline verification passed 62 tests and 46 subtests; 650 historical artifact hashes
were unchanged. Host-only tests require dependencies absent from the container.
An unusable copied Mac environment was removed from the container (about 655 MiB);
its dependency metadata and the refactor verification remain in
`data/research/reorganization-2026-09-26/`. The host environment was not modified.
