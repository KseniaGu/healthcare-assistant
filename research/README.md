# Laboratory extraction research

**Status — 2026-10-04:** The accepted CA1 v2 packet produced one completed L4 QLoRA run and three matched base/adapter evaluations. The selected historical adapter is `checkpoint-80`. Its adapted exact-record results were 1,219/1,251 synthetic, 131/292 real development, and 58/147 real reserved; the matched base scores were 1,034/1,251, 94/292, and 45/147. These results remain historical evidence, and the adapter is not deployed to the application. The refactor produced and accepted self-contained `ca1-sft-v3` with model-facing parity to v2, plus maintained training/evaluation commands. The v3 packet passed offline integrity and trainer checks; its content digest is `20d31cda44e955c60963b3da0f61add86cd20648e1041b69653d798a28af255f`, from source commit `43294f56ff6f59ae57d7d8264d06159de3af2b16`. Local development and dataset generation use tracked `pyproject.toml` and `poetry.lock`; cloud L4 training uses separate pinned requirements while retaining the VM CUDA PyTorch. No new cloud training or model-dependent evaluation has run.

| Need | Document |
| --- | --- |
| Operational local checks and later manual cloud steps | [Reproduction](extraction/reproduction.md) |
| Dataset roles, CA1 rules, v1 audit | [Dataset](extraction/datasets/dataset.md) |
| L4 recipe and dependency boundary | [Training](extraction/training/training.md) |
| New-run methodology and later results | [Evaluations](extraction/evaluation/evaluations.md) |
| Completed v2 run, older comparisons, artifact provenance | [Historical evaluations](extraction/evaluation/historical_evaluations.md) |
| Current findings and limitations | [Known issues](extraction/known_issues.md) |
| Future research | [Next steps](extraction/next_steps.md) |
| Private report locations | [Report inventory](../data/research/report_inventory.md) |
| Local masking | [Image redaction](image_redaction/README.md) |

Private reports, packet reviews, hashes, and model artifacts remain under gitignored `data/`. D0001/D0013 are excluded, D0010 is layout-only, and the reserved set was already evaluated. Any new final performance claim needs new held-out reports. The production Gemini v4 path remains unchanged.
