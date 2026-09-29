# Approved correction-aware extraction policy — CA1

The user approved the CA1 policy and research prompt. **CA1 is the active dataset
and pilot task.** Production remains on v4; historical literal data remain intact.
A second literal adapter is an optional future ablation, not a prerequisite.
See the [current dataset plan](plan.md) for source roles, size and implementation.

## Authoritative artifacts

All paths are private/gitignored and may be absent in a clean checkout:

- [Exact system prompt](../../data/research/extraction/sft-correction-v1/prompts/corrected.txt).
- [Frozen policy rules](../../data/research/extraction/sft-correction-v1/policy.md).
- [JSON schema](../../data/research/extraction/sft-correction-v1/schema.json).
- [Corrected synthetic smoke pairs](../../data/research/extraction/sft-correction-v1/synthetic/corrected/).
- [Reviewed synthetic specifications](../../data/research/extraction/sft-correction-v1/synthetic_specs.json).

The frozen policy/packet still contain their original “proposal/awaiting acceptance”
labels and a four-condition experiment suggestion. Those are historical snapshots:
policy/prompt approval is recorded here; the current dataset plan supersedes that
experiment suggestion. Do not change those hashed bytes to refresh their status.
The [v2 synthetic packet](../../data/research/extraction/sft-pilot-v2/README.md)
was later reviewed and explicitly accepted on 2026-09-29. Future generated
examples still require their own review and user acceptance.

## Contract summary

Return `date`, `results`, `errors`; each result has exactly `name`, `value`, `unit`,
`reference_range`. Non-name fields may be null; names must be identifiable. Numbers
are strings. The exact prompt/schema, not this summary, govern boundary cases.

- Repair identifiable spelling, mixed-script glyphs and broken words in names,
  codes, units and qualitative values. Preserve wording and name-plus-code behavior;
  no synonym expansion, indiscriminate transliteration or case-only normalization.
- Never change measured digits, decimal punctuation, trailing zeros, signs,
  inequalities, numeric reference bounds or qualitative polarity. Never convert
  units or reconstruct information from typical medical values.
- A surviving unit must identify the repair; ambiguous units/references become null.
  Do not restore text visible only in a source image or hidden generating record.
- Preserve row/subcolumn order and both paired measurements. Exclude pending/history
  and unnamed rows as instructed; warn for unnamed rows and missing required values.
- Dates follow collection → performance → order. No registration/creation fallback.
- References are direct cells/pointers, not allergy classes or adjacent commentary.
  No comment field or demographic/time-of-day selection is introduced.
- `errors` are concise English warnings with source quotations/values preserved.
  No warnings for successful repairs or missing/ambiguous optional units/references.
  At most five distinct warnings, each at most 160 characters. Judge supported
  meaning, identifying details and language separately from exact wording.

Track repair/abstention evidence outside model targets. “Corrected” means bounded
extraction from visible parser text, not recovery of complete source-image truth.
Evaluation should distinguish omissions, numeric/polarity errors, name/unit repairs,
abstentions, warnings and schema failures; an exact record mismatch alone does not
mean the measurement is wrong. No training has run; [results](evaluations.md) contain
the completed untuned baseline and its limitations.
