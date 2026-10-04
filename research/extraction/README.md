# CA1 extraction research

The maintained research pipeline builds a self-contained fictitious CA1 packet, trains from an accepted packet, and evaluates a selected adapter against synthetic and separately supplied real reports. It does not change the production Gemini extraction path.

- [Reproduction procedure](reproduction.md): local dataset and packet checks, then documented manual cloud training and matched evaluation.
- [Dataset contract and historical audit](datasets/dataset.md): source roles, visible-cell rules, construction, and the failed v1 distribution check.
- [Training recipe](training/training.md): model, optimizer, environment, and run records.
- [Current evaluation method](evaluation/evaluations.md) and [historical results](evaluation/historical_evaluations.md).
- [Known issues](known_issues.md) and [future work](next_steps.md).
- [Migration specification](implementation_plan.md): stage gates and preserved invariants.

Private reports, source reviews, packets, and model artifacts remain under gitignored `data/`. The previously accepted `sft-pilot-v2` packet and its run are historical evidence. The `ca1-sft-v3` packet passed model-facing parity and was accepted through a separate `acceptance.json` record. Its SHA256SUMS content digest is `20d31cda44e955c60963b3da0f61add86cd20648e1041b69653d798a28af255f`; cloud training and evaluation remain pending.
