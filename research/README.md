# Laboratory extraction research

**Current status — 2026-09-28:** Prepare a larger correction-aware synthetic dataset
for a supervised fine-tuning (SFT) pilot. The CA1 policy and research prompt are
user-approved; no training has run. The production application still uses Gemini
with prompt v4. The research task takes parser text as input and returns laboratory
JSON; it does not interpret images or provide medical interpretation.

## Start here

| Need | Document |
| --- | --- |
| Next dataset task, source roles, acceptance checks | [Current dataset plan](extraction/plan.md) |
| Approved correction behavior and exact prompt/schema links | [CA1 policy guide](extraction/correction_policy.md) |
| Existing builders, tokenization and completed annotation workflow | [Preparation guide](extraction/preparation.md) |
| Real report locations and per-report artifacts | [Private inventory](../data/research/report_inventory.md) |
| Evaluation evidence, when needed | [Completed results](extraction/evaluations.md) |
| Runner commands and environments | [Tool guide](extraction/README.md) |
| Local masking workflow | [Image redaction guide](image_redaction/README.md) |

## What exists

- 24 usable real-report targets (439 observations), with source roles fixed.
  Thirteen reports are development evidence. Eleven are reserved; D0018 requires
  longer-context handling. D0001/D0013 are excluded; D0010 is a layout guide.
- [CA1 packet](../data/research/extraction/sft-correction-v1/README.md): corrected
  targets, unchanged raw inputs, approved prompt, schema and **16 training / 8
  validation** synthetic examples. These check mechanics, not SFT efficacy.
- [Completed untuned CA1 baseline](extraction/evaluations.md#approved-ca1-untuned-qwen-baseline--2026-09-28):
  all 13 development reports have saved responses; 11 are schema-valid and none
  passes completely. Synthetic validation passes 2/8. This identifies behaviors
  to teach; it does not predict whether fine-tuning will work.

The next dataset defaults to **320 synthetic training / 64 validation reports**,
using development evidence for conventions and realism, not real result vectors.
Keep reserved reports out of generation, training and checkpoint selection. Generate
locally; no new parser/teacher calls are needed. Dataset review precedes training.
A later base/tuned experiment needs matched inference conditions and a separately
specified training environment and budget; the old Q8 baseline is not a BF16 control.

## Evidence and cleanup

Private data belong under gitignored `data/`; code and concise operating guidance
belong under `research/`. Frozen packets, target variants, raw provider outputs,
hashes and exhausted ledgers remain reproducibility evidence. Their historical
“pending” or “proposal” labels are snapshots, not the current policy-approval state.
Current decisions are recorded here and in the linked plan/policy guide.

The 2026-09-28 cleanup consolidated superseded planning prose and removed disposable
Python caches. Existing builders, tests and frozen evidence were retained because
they support validation/reproduction. No production default or historical score was
changed. Do not replay completed live workflows as part of dataset generation.

Pre-commit review fixed preservation of separators before empty leading parser pages
and aligned redaction setup commands with the persistent Python 3.11 environment.
Historical packets were not regenerated.
