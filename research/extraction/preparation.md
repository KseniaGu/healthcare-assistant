# Dataset preparation tools

For the next task, follow the [current dataset plan](plan.md) and
[approved CA1 contract](correction_policy.md). The **320/64 expansion is not yet
implemented**. The existing 16/8 corrected subset is a technical smoke dataset.
No new LlamaParse/Gemini quota or inference request is needed to generate text locally.

## Existing data

| Location under `data/research/extraction/` | Purpose |
| --- | --- |
| `sft-smoke-v1/sources.json` | Initial roles and source/group metadata |
| `sft-smoke-v1/real/` | Raw parser text, unchanged teacher drafts, original literal targets and reviews |
| `sft-smoke-v1/tokenizer/` | Pinned tokenizer/config and provenance; no model weights |
| `sft-correction-v1/real/<ID>/` | Same raw text plus CA1 target/review and preserved literal variants |
| `sft-correction-v1/synthetic/corrected/` | Current 16/8 CA1 train/validation JSONL |
| `sft-correction-v1/synthetic_specs.json` | Reviewed fictitious smoke inputs, target variants, family/split membership and provenance |
| `sft-correction-v1/synthetic/provenance.json` | Smoke provenance without input/target content |

Use only the development IDs allowed in the plan. Do not bulk-read reserved real
content while gathering examples. The [private inventory](../../data/research/report_inventory.md)
links per-report artifacts if a development mapping needs clarification.

`status: pending` in `sources.json` records initialization, not a role-approval
request. Current outcomes are in packet review/status artifacts. The preparation
completed 22 parser submissions and 14 teacher calls; all allowances are consumed.
Twenty-four real targets are usable; D0001/D0013 are excluded and D0010 is layout-only.
CA1 policy approval does not silently accept every future generated pair.

## Reusable code for the next dataset

- [prepare_sft.py](prepare_sft.py): `completion_tokens(tokenizer, messages)` uses the
  native template, verifies the assistant prefix/end token, returns token IDs and
  labels, and rejects total length over 8,192. `MODEL`/`REVISION` pin the tokenizer.
- [evaluate_image_extraction.py](evaluate_image_extraction.py): `strict_validate`
  is the existing JSON/schema/date validator used by the preparation builders.
- [prepare_correction_variant.py](prepare_correction_variant.py): study the message
  serialization, prompt/tokenizer hash checks and `check_pair` numeric/polarity
  safeguards. Its `build` recreates a reviewed historical packet; `check_specs`
  deliberately enforces 16/8 and is not a configurable expansion generator.
- [Preparation tests](tests/test_prepare_sft.py) and
  [correction tests](tests/test_correction_variant.py) demonstrate existing checks.

Reuse helpers where their contracts fit. Do not invoke the old builders against
frozen output folders, weaken their fixed-count guards, or transplant row-specific
real edits into an automatic repair algorithm. A new local generator should derive
labels from fictitious visible evidence and write a new packet. Existing code is
needed for reproducibility, so this cleanup did not delete it as “unused.”

The exact system prompt is `sft-correction-v1/prompts/corrected.txt`. Existing user
messages wrap the text as:

```text
Extract laboratory observations from this parser output:
<report>
[rendered parser-like text]
</report>
```

Assistant content is serialized JSON: parse it with `json.loads`, not
`ast.literal_eval`. The schema validates locally; it is not an extra system message
or an SFT response-schema parameter. Provenance/review metadata stay outside messages.

## Environment and checks

Run from the repository root. `/opt/venv/bin/python` is the prepared container
interpreter. Tokenization previously used Transformers 5.16.1 / tokenizers 0.23.2;
record actual versions and use the pinned local tokenizer, not a new `main` revision.
No model weights or new training dependencies are needed for dataset generation.

```sh
/opt/venv/bin/python -m pytest --noconftest \
  research/extraction/tests/test_prepare_sft.py \
  research/extraction/tests/test_correction_variant.py
```

`--noconftest` avoids application integration hooks. Add focused tests for new
rendering/labeling and split checks. Offline schema/token tests do not replace
manual review against rendered inputs.

## Completed workflow, retained for reproduction

`prepare_sft` supplied inventory, parse, annotate, review, synthetic, validate and
package stages. `prepare_correction_variant` materialized reviewed literal/CA1
variants. Their private [original](../../data/research/extraction/sft-smoke-v1/README.md)
and [CA1](../../data/research/extraction/sft-correction-v1/README.md) packet reports
retain exact provenance, settings and outcomes. Do not replay provider stages or
regenerate a historical packet merely to start the next task. Keep all raw outputs,
hashes and attempt ledgers; directory copies do not create new request allowances.
