# CA1 extraction research

The research pipeline builds a self-contained fictitious CA1 packet, trains from an accepted packet, and evaluates a selected adapter against synthetic and separately supplied real reports. The adapter is research-only; the production Gemini extraction path is separate.

- [Reproduction procedure](reproduction.md): local packet checks and manual cloud training and evaluation steps.
- [Dataset contract](datasets/dataset.md): source roles, visible-cell rules, generation, and validation.
- [Training recipe](training/training.md): model, optimizer, environment, and run records.
- [Evaluation method and results](evaluation/evaluations.md).
- [Known issues](known_issues.md) and [next steps](next_steps.md).

The accepted `ca1-sft-v3` packet has SHA256SUMS content digest `20d31cda44e955c60963b3da0f61add86cd20648e1041b69653d798a28af255f`. It passed packet integrity and offline acceptance checks. Manual L4 training selected `checkpoint-40`; three matched base/adapter evaluations are complete. Private reports, source reviews, packets, and model artifacts remain under gitignored `data/`. A future run requires a candidate built from clean committed source and its own acceptance evidence.
