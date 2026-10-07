# CA1 dataset contract

The research prompt and schema are separate from the production Gemini v4 prompt. The current builder reads the retained correction source, smoke source index, and pinned Qwen tokenizer, then writes an immutable candidate packet. Every build checks the committed, clean tracked source revision; callers cannot supply a revision override. The accepted `ca1-sft-v3` packet has content digest `20d31cda44e955c60963b3da0f61add86cd20648e1041b69653d798a28af255f`; [reproduction](../reproduction.md) records the commands and acceptance checks.

## Source roles

| Role | IDs | Use |
| --- | --- | --- |
| Development | A1–A4, C1, D0002, D0006, D0007, D0011, D0012, D0014–D0016 | Layout and vocabulary conventions; diagnostic evaluation |
| Reserved | B1–B4, D0003–D0005, D0008, D0009, D0017, D0018 | Evaluation only; never generation or checkpoint selection |
| Excluded | D0001, D0013 | Unresolved targets |
| Layout only | D0010 | No scoring or generation |

The 13 development layouts informed generation, so their evaluation is diagnostic. The 11 reserved reports were already evaluated and are not a fresh holdout. Real text and targets remain in private `data/`, outside the synthetic JSONL. The private [report inventory](../../../data/research/report_inventory.md) locates source and review files.

## CA1 visible-text policy

The exact private `sft-correction-v1/prompts/corrected.txt` and `schema.json` govern the contract. The output has `date`, `results`, and `errors`; each result has `name`, `value`, `unit`, and `reference_range`. Non-name fields may be null. Preserve measured digits, punctuation, trailing zeros, signs, inequalities, reference bounds, and qualitative polarity. Repair identifiable spelling or glyph errors only when the visible text supports the repair. Ambiguous units and references become null; missing values become null with a warning; unnamed, pending, and historical rows are excluded under the policy. Preserve row and subcolumn order, including paired measurements. Choose dates in collection, performance, then order priority; print/registration dates do not substitute. Warnings are concise English text and remain exact strings for scoring. Evidence for repairs and abstentions stays outside model messages.

The private [frozen policy](../../../data/research/extraction/sft-correction-v1/policy.md) governs the full CA1 rules. The corrected 16/8 smoke pairs are included in the 320/64 packet.

## Construction and validation

Generation uses fixed fictional assays, fourteen layouts, seed 20260929, 32 training and 10 validation families, and exact bucket quotas (training 192 ordinary/80 layout/48 difficult; validation 38/16/10). The maintained `datasets/synthetic/` package keeps shared case contracts in `types.py`, fictional generation and layout rendering in `layouts.py`, and visible-target derivation and validation in `reports.py`. The builder imports these modules through the package. Families and source groups cannot cross splits. Targets are derived from rendered visible cells, not hidden generating records. The builder checks source hashes and roles, prompt/schema, tokenizer provenance, duplicate and near-duplicate inputs, evidence spans, coverage, target schema, token boundaries, and the 8,192-token limit. The full assistant JSON through EOS is supervised; system/user, post-EOS newline, and padding are masked.

The packet bundles `prompt.txt`, `schema.json`, pinned tokenizer files, train/validation messages, token cache, manifest, statistics, distribution, and canonical `SHA256SUMS.json`. Candidate manifests hash all three synthetic modules alongside the builder, while the clean-source check also requires the package initializer to be tracked. Acceptance is separate: its evidence binds a verified model-facing parity comparison or a completed review of every changed example to the exact content digest. Training requires both payload integrity and a matching `acceptance.json`; the private source directories are provenance inputs for generation, not training dependencies.
