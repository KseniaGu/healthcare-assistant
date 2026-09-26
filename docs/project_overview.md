# Healthcare Assistant: project overview

Healthcare Assistant is a personal laboratory-report processing application under
active development. It accepts PDF reports, helps redact personally identifiable
information (PII), extracts laboratory observations, and charts results over time.
The current Streamlit UI uses one seeded patient; it has no account or patient
selection flow.

## Architecture and data flow

The application runs as a Streamlit process calling Python backend classes directly.
There is no separate HTTP backend or job queue. PostgreSQL stores results and
processing metadata; files remain on local disk. Database operations are async,
while blocking anonymization and API calls run through `asyncio.to_thread`.
Streamlit caches the processor and keeps the pending review context in session state.

```text
PDF upload → local redaction → user reviews PDF → LlamaParse API
           → Gemini API → PostgreSQL reports/observations → charts
```

1. **Local redaction:** `LaboratoryTestProcessor.anonymize()` records the source
   artifact and checks for an existing report by source checksum. PyMuPDF extracts
   text; unreadable text is retried with `ftfy`, then full-page local Tesseract OCR
   (`rus+eng`, 300 DPI). GLiNER detects PII in overlapping text chunks. PyMuPDF
   redacts matching areas and images; OCR-derived text uses OCR coordinates.
2. **Review:** the UI displays the redacted PDF. Only **Confirm & Analyze** calls
   `run()` for external document processing. Rejecting deletes the uploaded and
   redacted files. This gate belongs to the UI: the module's development `main()`
   invokes both stages directly and does not provide interactive review.
3. **Extraction:** LlamaParse converts the reviewed PDF to text. A versioned YAML
   prompt, rendered with Jinja2, supplies Gemini's instructions and response schema.
   The processor parses and locally validates fresh responses against the selected
   prompt schema before saving processed JSON. Raw failed responses are retained.
4. **Persistence and display:** the backend saves the report and individual
   observations, including values, inequalities, units, reference ranges, and flags.
   The dynamics page filters by date and test name and displays pandas/Plotly charts.

## Where to work

| Area | Responsibility |
| --- | --- |
| [src/app.py](../src/app.py) | Upload/review UI, model selection, dynamics, in-memory system logs |
| [backend/data_processing/laboratory_test.py](../backend/data_processing/laboratory_test.py) | Pipeline orchestration, redaction/OCR, artifact tracking, report persistence |
| [backend/models/llm_clients](../backend/models/llm_clients) | LlamaParse and Gemini API adapters; Gemini response/usage handling |
| [backend/prompts/laboratory_test_processor](../backend/prompts/laboratory_test_processor) | Versioned instructions and JSON schemas |
| [backend/utils](../backend/utils) | Pydantic transfer schemas, text validation, artifact metadata, file/log helpers |
| [src/databases/postgres](../src/databases/postgres) | SQLAlchemy models, async repository methods, transaction decorator, Alembic seeds |
| [configs](../configs) | Pydantic settings for models, paths, processing, database, and tests |

The stack is Python 3.11–3.13 with Poetry, [Streamlit](https://streamlit.io/), [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)/[Tesseract](https://github.com/tesseract-ocr/tesseract), [GLiNER](https://github.com/urchade/GLiNER) and [Transformers](https://github.com/huggingface/transformers), [LlamaParse](https://www.llamaindex.ai/llamaparse), [Google GenAI](https://github.com/googleapis/python-genai), [PostgreSQL](https://www.postgresql.org/)/[SQLAlchemy](https://www.sqlalchemy.org/)/[psycopg](https://www.psycopg.org/), and [Langfuse](https://github.com/langfuse/langfuse) tracing. [pyproject.toml](../pyproject.toml) defines dependency constraints. OpenAI settings and [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR?ysclid=mjelp06ui4625881594) references are future-facing; the active pipeline instantiates LlamaParse and Gemini directly.

Current extraction defaults are explicit in the code:

| Consumer | Prompt selection | Model default |
| --- | --- | --- |
| Pipeline and Gemini behavior test | `DataSettings.prompt_version = "v4"` | `gemini-3.1-flash-lite` |
| Standalone Gemini baseline runner | CLI `--prompt-version`, default `v4` | Configured processor model; explicit `--model` override |
| Local Qwen experiment | Frozen `v4` revision 4.1 | Original Qwen3 4B Q8_0; separately pinned Qwen3.5 4B MLX / 9B comparison profiles through host Ollama |
| Direct `GeminiClient.generate()` calls omitting arguments | `prompt_version="v1"` | Configured Gemini model |

The UI can override the generation model, but prompt selection is not model-specific.
When changing extraction behavior, check the prompt schema, `save_report()`, and
model/integration tests together.

## Persistence and operational behavior

`Patient`, `Report`, `TestCatalog`, and `TestObservation` hold medical results.
`Pipeline`, `PipelineRun`, `PipelineStep`, and `Artifact` track executions, tools,
inputs, outputs, and file checksums. The database stores file references rather
than PDF contents. The connection decorator commits per repository call, so an
entire processing run is not one atomic transaction.

[PathSettings](../configs/paths.py) creates directories under `data/` for raw,
redacted, parsed, processed, and raw LLM outputs. Existing output paths can cause
stages to be skipped; account for cached artifacts when retesting prompts or OCR.
The standalone `research.extraction.run_extraction_baseline` CLI always uses a
new output directory, explicit parser inputs, local schema validation, and a shared
nine-attempt development budget. It bypasses PDF processing and database caching.
See [evaluation runner usage](../research/extraction/README.md).
The separate `research.extraction.evaluate_local_extraction` CLI freezes reviewed
targets, runs a bounded local Qwen experiment, and scores saved responses offline.
Its A+C development split reserves B from inference and scoring. It leaves the
application defaults and Gemini budget unchanged; see the
[local experiment record](../research/extraction/evaluations.md).

Reports retain complete extraction JSON, including dates and errors. Observation
persistence uses `unit`, catalog preferences use `unit_normalized`, unnamed rows
are skipped, and absent/unrecognized flags become unknown. Observations preserve original
values in `observed_value_text` and project only scalar numbers into `observed_value`.
Qualitative results and measured ranges are stored with a null numeric projection.
The results table shows all value types; charts include only exact scalar measurements
with known dates and keep different units in separate series. Unknown flags do not
count as high/low results. Catalog identity is the unique test name, independent of
optional normalized-unit preferences.

The additive `c84d120a6e31` migration adds the nullable text column. The standalone
`backend.data_processing.restore_observations` command previews recovery from stored
JSON; `--apply` commits it. It restores missing rows and raw text/units, repairs
identifiable legacy placeholder names, and skips ambiguous/unnamed/pending rows.
Existing numeric measurements, flags, dates, and report JSON remain unchanged.
This repair never calls an LLM and is not a fresh v4 extraction baseline.

Successful UI processing deletes the temporary upload but retains downstream
artifacts. There is no general retention/cleanup policy implemented.

Database startup defaults to `ENV=DEV`: it creates the `uuid-ossp` extension and
ORM tables, then runs Alembic when `POSTGRES_AUTO_MIGRATE` is enabled. The checked-in
revisions seed a pipeline and patient and add the observation text column; they are
not a complete initial-schema migration history. Outside DEV, this constructor does not initialize the schema.

## Development and setup

The [Dev Container](../.devcontainer/devcontainer.json) uses Python 3.11 at
`/opt/venv/bin/python`, PostgreSQL 16, separate development/test databases, and
persistent volumes for database data, working documents, and model weights. Its
`data/` volume masks the host directory. The separate [deployment Compose stack](../deploy/docker/docker-compose.yml)
runs the app on port 8501; it persists PostgreSQL and mounts model weights, but does
not declare a persistent app `data/` volume.

Configure database values and Gemini/LlamaParse API credentials through the root
`.env`/environment; inspect settings definitions for names (`POSTGRES_*`,
`GEMINI_*`, `LLAMAPARSE_*`). The database client builds its URL from `POSTGRES_*`,
not `DATABASE_URL`. Langfuse uses its own client environment configuration.
Startup may download GLiNER/tokenizer weights unless they are already cached.

From the repository root, after configuring services:

```sh
poetry install --with dev --no-root --no-interaction
poetry run streamlit run src/app.py --server.address=0.0.0.0 --server.port=8501
```

`poetry.lock` is currently gitignored. If installation reports it is stale after
manifest edits, Poetry 1.8 supports `poetry lock --no-update`, followed by reinstalling.
A failed post-create install can leave a rebuilt container without `streamlit`.
Both Dockerfiles install Tesseract with Russian/English language data. See the
[application guide](application_guide.md) for setup prerequisites and daily usage.

[VS Code launch configurations](../.vscode/launch.json) support the current Python
file and the Streamlit app. Both set the workspace root as `cwd` and `PYTHONPATH`
to resolve `backend` imports.

## Tests and current limits

Offline regressions run with `LANGFUSE_TRACING_ENABLED=false HF_HUB_OFFLINE=1 poetry run python -m unittest discover -s tests/unit -v`; they mock external work and avoid processor initialization.

Run `poetry run pytest` only with an isolated test database configured through
`POSTGRES_TEST_DATABASE_NAME`. Model behavior tests call real APIs; the integration
test mocks processing responses but still initializes the processor and uses
PostgreSQL. Privacy tests exercise local redaction. Fixtures delete generated
files, truncate tables, and drop the test schema after a successful session.
Importing the pipeline or collecting tests can load environment settings and create
directories; these are not side-effect-free offline checks.

- **Manual review is essential to the current workflow.** OCR/NER failures can leave
  text unmasked; the code logs and continues to review rather than aborting every
  unreadable page. OCR itself is local, but later parsing/extraction use external
  services, and Langfuse instrumentation is a separate telemetry path.
- **Language support is narrower than OCR's language list.** Text validation expects
  Russian using stopwords and language detection; English OCR support does not make
  the full pipeline language-independent.
- **Image masking is unconditional.** A scanned page stored as one image may be
  fully blacked out. The OCR fallback does not replace this policy.
- **Results need validation.** Observation conversion errors are logged and skipped,
  and test-name unification remains a TODO. Extraction and charting are still under
  development, not a validated clinical decision system.

## Related guides

- [Application setup and usage](application_guide.md)
- [v4 storage update, migration and historical recovery](v4_application_update.md)
- [Research status and next steps](../research/README.md)
- [Prompt contract and evaluation results](../research/extraction/evaluations.md)
- [Proposed small-model training experiment](../research/extraction/plan.md)
