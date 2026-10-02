# Laboratory extraction research

**Status — 2026-10-02:** The accepted CA1 v2 packet had one completed run with the
research-only QLoRA pipeline. `checkpoint-80` is the selected adapter. Matched
base/adapter evaluations are complete on synthetic validation, 13 real
development reports, and 11 reserved reports. The adapter improved exact
record matching over the base but remains unreliable on real reports:
131/292 exact development records and 58/147 exact reserved records, with no
fully correct reserved report. It is not ready for unattended clinical
extraction. The production Gemini path remains unchanged. See the
[Lightning training guide](extraction/training.md) for the verified single
private input archive and Studio commands, and [evaluation results](extraction/evaluations.md)
for evidence and limits. The combined archive was assembled after the completed
runs; their original inputs remain recorded separately.

## Research map

| Need | Document |
| --- | --- |
| Dataset design, source roles, and acceptance checks | [Dataset plan](extraction/plan.md) |
| Accepted packet and loader contract | [Extraction tool guide](extraction/README.md#training-pipeline-handoff) |
| Lightning workflow and other compute options | [Training guide](extraction/training.md) |
| Completed comparisons and selected checkpoint | [Evaluations](extraction/evaluations.md) |
| Correction behavior and prompt/schema links | [CA1 policy](extraction/correction_policy.md) |
| Dataset preparation and annotation workflow | [Preparation guide](extraction/preparation.md) |
| Private report locations and per-report evidence | [Report inventory](../data/research/report_inventory.md) |
| Local masking workflow | [Image redaction guide](image_redaction/README.md) |

## Data and next step

The research set has 24 usable real-report targets: 13 development and 11
reserved. D0018 needs a longer evaluation context; D0001 and D0013 are
excluded, and D0010 is a layout guide. The accepted fictitious v2 packet has
320 training and 64 validation reports. Frozen packet files and detailed
medical inputs, outputs, and reviews remain under gitignored `data/`.

The next step is to examine development errors against their source layouts
and define acceptance criteria for any later experiment. Development layouts
informed packet preparation, so those reports are diagnostic. Preserve this
reserved evaluation as the historical test; a revised model or prompt needs
new held-out reports for a fresh final assessment.
