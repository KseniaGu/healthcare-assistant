# Application guide

[Back to README](../README.md)

## Development setup

The Dev Container is the documented route for local development. It supplies
Python 3.11, Poetry, PostgreSQL 16, and Tesseract with Russian and English language
data. Install Docker and the VS Code Dev Containers extension first.

1. Create `.env` in the repository root using `.env.example` as a starting point.
   Configure `GEMINI_API_KEY` and `LLAMAPARSE_API_KEY`. For the bundled development
   database, use the following local settings (not production credentials):

   ```dotenv
   POSTGRES_USERNAME=healthcare
   POSTGRES_PASSWORD=healthcare_dev
   POSTGRES_DATABASE_NAME=healthcare_dev
   POSTGRES_TEST_DATABASE_NAME=healthcare_test
   POSTGRES_HOST=database
   POSTGRES_PORT=5432
   POSTGRES_AUTO_MIGRATE=True
   ENV=DEV
   ```

2. Run **Dev Containers: Reopen in Container**. The post-create step installs the
   project dependencies. Initial setup requires network access.
3. Start the application from the repository root:

   ```sh
   poetry run streamlit run src/app.py --server.address=0.0.0.0 --server.port=8501
   ```

4. Open forwarded port 8501. In DEV, startup creates ORM tables and, with automatic
   migration enabled, applies migrations and seed data. The UI currently uses the
   seeded patient rather than offering account or patient selection.

The container mounts the Alembic configuration automatically. It keeps database
contents, working documents, and Hugging Face models in separate Docker volumes.
Its `data/` volume masks the host checkout's `data/` folder, so the host and container
may show different files. Do not delete volumes if you need to retain these records.

GLiNER and tokenizer weights download on first use unless cached. The container
uses its own persistent model cache; copying host weights is unnecessary. For the
separate deployment Compose setup, `models_cache/` can be populated from the host
Hugging Face cache in advance. See the [architecture/setup notes](project_overview.md)
for the distinction between these environments. A native installation must provide
Python 3.11–3.13, PostgreSQL, Poetry, Tesseract and its `rus`/`eng` data, plus a root
Alembic configuration; the container handles these prerequisites for you.

## Processing a report

On **Process Report**, upload a PDF and click **Start Anonymization**. Redaction
runs locally with GLiNER and PyMuPDF. Unreadable text is retried through local
Tesseract OCR at 300 DPI. OCR does not send the document to a remote service.

Inspect every page of the preview. OCR and entity recognition can miss information;
a page stored as a single image may also become fully masked. Use **Reject / Delete**
when the preview is unsuitable. **Confirm & Analyze** sends the reviewed PDF to
LlamaParse, then the parsed text to Gemini, and saves the resulting report.
Langfuse instrumentation is a separate telemetry path when configured.

Read the returned results and `errors` warnings. A valid JSON response does not
by itself establish that extraction is correct. Unnamed observations are omitted
with a warning; pending tests are excluded from completed results.

Uploading an already processed document may return the stored report. Changing the
prompt does not automatically re-extract old reports. Developers needing a fresh
comparison should use the [evaluation runner](prompt_evaluation.md#fresh-run-tooling).

## Viewing results

**Lab Test Dynamics** shows original result text, units, references, dates and flags.
Select tests and a date range to explore trends. Undated results remain visible in
the table but cannot be placed on a timeline.

Qualitative values, measured ranges and inequalities remain visible as reported.
Only exact numeric measurements with known dates appear as chart points. Different
units have separate series; the app does not perform unit conversion. New v4 results
have unknown flags because extraction no longer infers clinical classifications.
Historical flags may still be present; unknown does not mean normal.

## Settings and troubleshooting

- The application defaults to `gemini-3.1-flash-lite` and prompt v4; the UI can select
  another configured model. Full role-specific defaults are in the
  [project overview](project_overview.md#architecture-and-data-flow).
- Settings are read from `.env` and the environment. After changes, restart the app;
  its processor is cached. In the container, open a new terminal or recreate the
  container if Compose-provided environment values still take precedence.
- The response limit defaults to 16,384 tokens (`GEMINI_MAX_TOKENS`). Truncated or
  schema-invalid fresh responses are rejected, with raw text kept in
  `data/llm_outputs/`. Inspect the error before retrying; increasing the response
  budget or splitting a long report may help with truncation.
- Temporary Gemini HTTP 503 errors receive up to three total attempts. If they all
  fail, wait and use **Confirm & Analyze** again; the reviewed document stays available.
- If `streamlit` is missing after container setup, inspect the post-create install
  failure and rerun `poetry install --with dev --no-root --no-interaction` after fixing
  it. Rebuild the container after Dockerfile or system-dependency changes.
- An existing database needs the v4 observation-text migration before this app version
  runs. Follow the [upgrade and recovery guide](v4_application_update.md), rather
  than re-uploading every document.

The current text validation expects Russian reports. English OCR availability does
not imply full English extraction support. See [current limits](project_overview.md#tests-and-current-limits)
for additional operational constraints.
