# Laboratory extraction research

**Current status — 2026-09-29:** The user **accepted** the revised 320/64 CA1
synthetic [v2 packet](../data/research/extraction/sft-pilot-v2/README.md) after
assistant review. Its parser inputs are longer and more varied than the failed
historical v1 candidate; distribution matching remains uncertain with only 13
eligible originals. CA1 policy and research prompt are approved. **No training or
new model inference has run.** The next task is to design and implement the
training pipeline. The production application still uses Gemini with prompt v4.

## Start here

| Need | Document |
| --- | --- |
| Dataset design, source roles and acceptance checks | [Completed dataset plan](extraction/plan.md) |
| Accepted packet and next-session loader contract | [Training pipeline handoff](extraction/README.md#training-pipeline-handoff) |
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
- [Revised 320/64 CA1 pilot packet](../data/research/extraction/sft-pilot-v2/README.md):
  304 new training and 56 new validation fictitious reports plus the unchanged
  16/8 smoke subset. All 384 pairs pass schema, split and chat-label checks.
  The 360 new pairs also pass rendered visible-cell checks. The longest chat
  is 4,742 tokens. The assistant reviewed all 64 validation and 85 stratified
  training pairs. The 13 eligible
  originals have 3,289 median input characters; new v2 training has 2,591,
  compared with 746 for v1. The user accepted this packet on 2026-09-29.
- [Historical v1 packet](../data/research/extraction/sft-pilot-v1/README.md):
  preserved because the [distribution audit](extraction/distribution_audit.md)
  found it too short and regular. Do not train on v1.
- [Completed untuned CA1 baseline](extraction/evaluations.md#approved-ca1-untuned-qwen-baseline--2026-09-28):
  all 13 development reports have saved responses; 11 are schema-valid and none
  passes completely. Synthetic validation passes 2/8. This identifies behaviors
  to teach; it does not predict whether fine-tuning will work.

V2 uses 32 training and 10 validation families across chemistry, hematology,
urine and qualitative reports. It has 0 detected duplicate inputs, split leaks or
cross-split near-duplicate candidates. The 13 eligible development reports are
too few to prove distribution equivalence; 17 new v2 reports remain below the
shortest original by character count. The frozen smoke examples remain technical
checks. The next action is a training-pipeline implementation plan using the
[accepted packet and handoff](extraction/README.md#training-pipeline-handoff).
Keep reserved reports out of adaptation and checkpoint selection. Select the
training method, hardware, budget and matched evaluation protocol in that next
task; the old Q8 baseline is not a BF16 control.

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
