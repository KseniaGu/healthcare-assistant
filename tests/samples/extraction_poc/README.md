# Synthetic extraction development fixtures

These three parser-text examples contain no patient documents or organization
layouts. Expected outputs were authored before live testing. They are development
cases, not evidence of held-out generalization.

| Case | Checks |
| --- | --- |
| `scalars` | Collection-date precedence, historical columns, raw units, precision, decimal commas and inequalities |
| `references` | Examination-date fallback, measured/reference intervals, qualitative results, pointers, shared units and allergy classes |
| `omissions` | Referral-date fallback, unnamed rows, incomplete references, pending tests and a cover referral to panel measurements |

Compare predictions with each `.expected.json`: dates, ordered records and all
result fields should match. Warning wording may vary, but the omission warning must
identify missing names and incomplete references without inventing causes. Schema
validity alone is insufficient.

These fixtures are development cases and do not authorize new provider requests.
Consult the research protocol before creating a new evaluation ledger.

For the maintained CA1 procedure, see the
[reproduction guide](../../../research/extraction/reproduction.md).
