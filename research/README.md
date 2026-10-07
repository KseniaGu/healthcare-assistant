# Laboratory extraction research

**Current status:** The accepted `ca1-sft-v3` packet was trained manually on one L4 from committed source. The selected research adapter is `checkpoint-40`; it is not deployed. Its exact-record results are 1,212/1,251 on synthetic validation, 145/292 on real development, and 63/147 on real reserved reports. The matched base results are 1,034/1,251, 94/292, and 45/147, respectively. See [evaluations](extraction/evaluation/evaluations.md) for the method, artifact provenance, and case-level limits.

| Need | Document |
| --- | --- |
| Local checks and manual cloud procedure | [Reproduction](extraction/reproduction.md) |
| Dataset roles, CA1 policy, and packet construction | [Dataset](extraction/datasets/dataset.md) |
| L4 recipe and dependency boundary | [Training](extraction/training/training.md) |
| Evaluation method and current results | [Evaluations](extraction/evaluation/evaluations.md) |
| Current findings and limitations | [Known issues](extraction/known_issues.md) |
| Future research | [Next steps](extraction/next_steps.md) |
| Private report locations | [Report inventory](../data/research/report_inventory.md) |
| Local masking | [Image redaction](image_redaction/README.md) |

Private reports, packet reviews, hashes, and model artifacts remain under gitignored `data/`. D0001/D0013 are excluded, D0010 is layout-only, and the reserved set has already been evaluated. A new final performance claim needs new held-out reports. The production Gemini v4 path is separate from this research adapter.
