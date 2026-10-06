# Laboratory extraction research

**Status — 2026-10-06:** The accepted `ca1-sft-v3` packet was trained manually on one L4 from committed source, and three matched base/adapter evaluations completed. The new selected adapter is `checkpoint-40`. Its adapted exact-record results are 1,212/1,251 synthetic, 145/292 real development, and 63/147 real reserved; the matched base remains 1,034/1,251, 94/292, and 45/147. Compared with the historical v2 adapter, the new run loses seven synthetic exact records and gains 14 development and five reserved records. Material case regressions include reserved D0004 (15/26→6/26). The adapter is research-only and is not deployed.

| Need | Document |
| --- | --- |
| Operational local checks and manual cloud procedure | [Reproduction](extraction/reproduction.md) |
| Dataset roles, CA1 rules, v1 audit | [Dataset](extraction/datasets/dataset.md) |
| L4 recipe and dependency boundary | [Training](extraction/training/training.md) |
| New-run methodology and later results | [Evaluations](extraction/evaluation/evaluations.md) |
| Completed v2 run, older comparisons, artifact provenance | [Historical evaluations](extraction/evaluation/historical_evaluations.md) |
| Current findings and limitations | [Known issues](extraction/known_issues.md) |
| Future research | [Next steps](extraction/next_steps.md) |
| Private report locations | [Report inventory](../data/research/report_inventory.md) |
| Local masking | [Image redaction](image_redaction/README.md) |

Private reports, packet reviews, hashes, and model artifacts remain under gitignored `data/`. D0001/D0013 are excluded, D0010 is layout-only, and the reserved set was already evaluated. Any new final performance claim needs new held-out reports. The production Gemini v4 path remains unchanged.
