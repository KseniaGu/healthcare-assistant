# AGENTS.md

Guidance for AI coding agents working in this repository.

## General Code Style

- Keep changes closely scoped to the task.
- Avoid speculative helpers, compatibility layers, or configuration unless the current code path needs them.
- Use clear names and straightforward control flow before adding comments.
- Add comments only when they explain intent, boundary conditions, or non-obvious tradeoffs.

## Code quality

Write production-quality Python consistent with the surrounding code.

Prefer:
- small functions with one clear responsibility;
- named domain types/dataclasses over loosely structured dictionaries;
- explicit helper functions when validation or transformation has a distinct purpose;
- descriptive names over comments explaining complicated code;
- early returns and simple control flow over deeply nested logic.

Avoid:
- monolithic functions that mix validation, I/O, configuration, and execution;
- unnecessary abstraction or helpers used only to make code look modular;
- duplicated validation logic;
- large inline comprehensions when a named operation would be clearer.

Before finishing:
- run Ruff, the type checker, and relevant tests;
- review the diff for readability and simplify unnecessarily dense code.

## Docstrings

Write docstrings when they help a future maintainer understand the role or contract of code. Do not add a docstring to every small helper by default.

Preferred style:

- Use concise triple-quoted docstrings.
- Start with a short summary sentence.
- Add a second paragraph only when it explains why the function exists or how it fits into the pipeline.
- Use `Args:` and `Returns:` when the argument contract is not obvious.
- Keep docstrings accurate after refactors. An outdated docstring is worse than no docstring.

Examples:

```python
class PaddleOCRPipeline:
    """Pipeline for OCR processing of PDFs and images using PaddleOCR."""

    def parse_pdf(self, input_file: str | Path) -> Path | None:
        """Parses a PDF file and converts it to Markdown format.

        Args:
            input_file: Path to the input PDF file.

        Returns:
            Path to the generated markdown file, or None if the PDF has too many pages.
        """
```

Another existing style example:

```python
class KnowledgeGraphIndexer:
    """Manages knowledge graph indexing, processing, and retrieval operations.

    This class handles the creation, processing, and querying of property graph indexes,
    including implicit graph processing, entity/relation extraction, and vector embeddings.
    """
```


## External Library Documentation

Consider using Context7 when code review, refactoring, feature development, or another task depends on the current or version-specific behavior of an external library.

It is especially useful for checking API contracts, return types, configuration options, deprecations, and recommended framework usage. When practical, identify the installed dependency version from `pyproject.toml` or `uv.lock` and prefer matching documentation.

Use repository evidence, tests, and runtime behavior as the primary source of truth. Context7 should support decisions, not override established project patterns or justify unnecessary changes.

## Documentation Synchronization

When changing a runtime default or selecting a new active prompt, update the
corresponding documentation in the same change. At minimum, check:

- `README.md` for user-facing prerequisites and runtime defaults;
- `docs/personal_guide.md` for personal developer instructions covering environment
  setup and management, launching the app, configuring Codex, and other day-to-day
  development tasks.

When changing an active prompt version, search for every role that selects that
prompt and update the explicit prompt-version tables in project documentation;
do not infer one shared version for roles with separate settings.

## Research documentation

After completing a meaningful research task or intermediate experiment step, update
`research/README.md` in the same change: current status, what was learned, relevant
limitations, evidence links and the next concrete action. Distinguish completed work
from proposals and pending approvals; replace superseded status instead of appending
a conversation log. Update `research/extraction/evaluation/evaluations.md` for new results and
the relevant tool guide for command or environment changes. Keep detailed medical
inputs, outputs and review artifacts in gitignored `data/`; preserve historical
scores, provenance and exhausted ledgers. Documentation updates do not authorize
new requests, use of reserved B reports, synthetic generation or training.
