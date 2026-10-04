# CA1 dataset contract and audit

The research prompt and schema are separate from the production Gemini v4 prompt. The current builder reads the retained correction source, smoke source index, and pinned Qwen tokenizer, then writes a new immutable *candidate* packet. [Reproduction](../reproduction.md) gives commands. The accepted v2 packet remains historical evidence.

## Source roles

| Role | IDs | Use |
| --- | --- | --- |
| Development | A1–A4, C1, D0002, D0006, D0007, D0011, D0012, D0014–D0016 | Layout and vocabulary conventions; diagnostic evaluation |
| Reserved | B1–B4, D0003–D0005, D0008, D0009, D0017, D0018 | Historical comparison only; never generation or checkpoint selection |
| Excluded | D0001, D0013 | Unresolved targets |
| Layout only | D0010 | No scoring or generation |

The 13 development layouts informed generation, so their evaluation is diagnostic. The 11 reserved reports were already evaluated and are not a fresh holdout. Real text and targets remain in private `data/`, outside the synthetic JSONL. The original separate Studio input archives identify what the completed run consumed; the later unified archive was assembled afterward. The private [report inventory](../../../data/research/report_inventory.md) locates source and review files.

## CA1 visible-text policy

The exact private `sft-correction-v1/prompts/corrected.txt` and `schema.json` govern the contract. The output has `date`, `results`, and `errors`; each result has `name`, `value`, `unit`, and `reference_range`. Non-name fields may be null. Preserve measured digits, punctuation, trailing zeros, signs, inequalities, reference bounds, and qualitative polarity. Repair identifiable spelling or glyph errors only when the visible text supports the repair. Ambiguous units and references become null; missing values become null with a warning; unnamed, pending, and historical rows are excluded under the policy. Preserve row and subcolumn order, including paired measurements. Choose dates in collection, performance, then order priority; print/registration dates do not substitute. Warnings are concise English text and remain exact strings for scoring. Evidence for repairs and abstentions stays outside model messages.

The private [frozen policy](../../../data/research/extraction/sft-correction-v1/policy.md) and [packet review](../../../data/research/extraction/sft-pilot-v2/review.json) preserve the original full rules and reviews. Their superseded proposal wording remains unchanged as historical evidence. The corrected 16/8 smoke pairs remain unchanged within the 320/64 packet.

## Construction and validation

Generation uses fixed fictional assays, fourteen layouts, seed 20260929, 32 training and 10 validation families, and exact bucket quotas (training 192 ordinary/80 layout/48 difficult; validation 38/16/10). Families and source groups cannot cross splits. Targets are derived from rendered visible cells, not hidden generating records. The builder checks source hashes and roles, prompt/schema, tokenizer provenance, duplicate and near-duplicate inputs, evidence spans, coverage, target schema, token boundaries, and the 8,192-token limit. The full assistant JSON through EOS is supervised; system/user, post-EOS newline, and padding are masked.

The packet bundles `prompt.txt`, `schema.json`, pinned tokenizer files, train/validation messages, token cache, manifest, statistics, distribution, and canonical `SHA256SUMS.json`. Acceptance is separate: its evidence binds a reviewed old-packet comparison or completed review of every changed example to the exact content digest. Training requires both payload integrity and a matching `acceptance.json`; the old source directories are provenance inputs for generation, not training dependencies.

## Historical v1 distribution failure

The first 320/64 synthetic packet passed label and token checks but was too short and uniform for the 13 eligible original layouts: median training input length 718 characters/21 lines versus 3,289 characters/61 lines for originals. It used one universal header and lacked parser artifacts seen in originals. It was rejected for training. The reviewed v2 replacement reached 2,591 characters/47 lines for 304 new training inputs and 2,537/49 for 56 new validation inputs. All 64 validation and 85 stratified training pairs were reviewed, then v2 was accepted on 2026-09-29. Seventeen generated inputs remain shorter than the shortest original; 13 originals cannot define production prevalence. The old distribution audit remains available with `git show 138c2961d3ac6771c4af06bc80b02417dcffa07b:research/extraction/distribution_audit.md`; the private v1/v2 packets and CSV reviews retain measurements and provenance.
