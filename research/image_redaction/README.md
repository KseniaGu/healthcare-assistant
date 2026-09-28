# Local image redaction review

[Research status](../README.md)

Run commands from the repository root. The fictitious fixtures are in `fixtures/`;
they are test assets, not training data.

This standalone tool accepts **one explicitly named image or PDF** at a time. Images
can be JPEG, PNG or single-frame WebP; `propose-pdf` renders each PDF page locally
at 300 DPI and creates a separate image candidate. It runs
Tesseract OCR (`rus+eng`), GLiNER on CPU, narrow contact/ID rules, and local QR/
barcode region detection. It fills proposed areas with opaque black pixels in a
new RGB PNG. It does not initialize the application, database, tracing, parser or
LLM clients. It never submits report content to a service. The model loader is
offline by default; `--allow-model-download` is only for fetching public model
weights during initial setup.

Offline runs load complete cached GLiNER weights and the separate
`microsoft/mdeberta-v3-base` tokenizer and encoder config through a local
temporary bundle. They do not ask the model hub to resolve either `main`
revision. If either component is missing, the tool reports a setup error and
requires an explicit model download flag.

This is assisted redaction. OCR, names in unusual layouts, handwriting, seals,
signatures, faint text and codes can be missed. Inspect every candidate at full
resolution before choosing Keep. A clean image can still identify someone through
dates, rare result patterns or other context. Public availability does not grant
reuse permission. Keep source/reuse terms in your private records.

## Host setup

Use a terminal **outside a coding-assistant tool session** for private inputs.
Python 3.11, Tesseract with `eng` and `rus`, and an isolated environment are
required. On macOS with Homebrew:

```sh
brew install python@3.11 tesseract tesseract-lang
"$(brew --prefix python@3.11)/bin/python3.11" -m venv "$HOME/.venvs/lab-redactor-py311"
PY="$HOME/.venvs/lab-redactor-py311/bin/python"
"$PY" -m pip install -r research/image_redaction/requirements.txt
tesseract --list-langs
```

The model ID is `urchade/gliner_multi-v2.1`, threshold 0.4. Its public weights
are about 1.16 GB (Apache 2.0), with a separate tokenizer of about 4.3 MB (MIT),
downloaded only when you pass
`--allow-model-download`. Tesseract and its language data use Apache 2.0. Set
`HF_HUB_OFFLINE=1` on subsequent runs or simply omit the flag. The isolated
environment is deliberately separate from the app's Poetry environment.

## Host cache and Python troubleshooting

A host error naming `/home/vscode/.cache/huggingface` indicates a container cache
location is being used on macOS. Before launching a fresh Python process, set:

```sh
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
mkdir -p "$HF_HUB_CACHE"
```

Do not create `/home/vscode` or use sudo to work around this on the Mac. The CLI
inherits cache settings; it does not load application settings or `.env`. The
container intentionally uses that Linux location in `.devcontainer/compose.yaml`.
The exact source of a host override must be checked in the host shell/IDE settings.
Hugging Face reads these settings on import, so restart the command after changing
them. See [cache environment variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).

If a traceback or warning names `lib/python3.14`, that environment was created with
Python 3.14 even if it is named `lab-redactor`. Use the documented Python 3.11
interpreter explicitly to create a separate environment; activation cannot change
an environment's interpreter. A persistent location avoids `/private/tmp` cleanup:

```sh
brew install python@3.11
"$(brew --prefix python@3.11)/bin/python3.11" -m venv "$HOME/.venvs/lab-redactor-py311"
PY="$HOME/.venvs/lab-redactor-py311/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r research/image_redaction/requirements.txt
"$PY" --version
```

The reported `cmsOpenProfileFromMem failed` message concerns PDF color-profile
handling. In the observed host attempt, processing continued to model cache lookup;
it was not the terminating error. Inspect eventual rendered candidates for visual
fidelity rather than assuming the profile warning proves either success or failure.
No private PDF was opened to diagnose this message.

## Private candidate and review flow

Keep originals and the private workspace in different directories. Keep the
workspace **outside the repository and any shared/synced folder**. The approved
directory below is gitignored by the repository. Run each explicit source path
locally; do not paste private paths, OCR or screenshots into a coding-assistant
session.

```sh
PY="$HOME/.venvs/lab-redactor-py311/bin/python"
WORK="$HOME/.local/share/lab-redaction-private"
APPROVED=data/approved_image_redactions

$PY research/image_redaction/local_image_redactor.py propose '/private/path/report.jpg' \
  --work "$WORK" --id D0001 --allow-model-download
$PY research/image_redaction/local_image_redactor.py review --work "$WORK" --approved "$APPROVED"
```

Open the printed `127.0.0.1` URL in **your own browser**. The page shows each
private candidate, its exact SHA-256, mask counts and review requirements. The
server binds only to loopback and uses a random session token. The candidate
stays in `WORK`; it is not an approved export. Inspect text at full size,
including page margins, codes, signatures and seals. Decide whether collection/
performance dates and values are safe for your particular sharing purpose.
These fields remain visible by default. For layout-only use, remove
patient-specific dates/values with manual rectangles.

Select **Keep exact candidate** only after reviewing the final image. Keep copies
the exact hashed PNG to `APPROVED` with a shareable allowlisted JSON manifest.
Select **Skip** to leave it private; an optional rejection reason stays in the
private workspace. Record the source and reuse-permission note privately in the
review form. Silence does not approve. Changed images require a new
version and fresh Keep. Approved versions are never overwritten in place.

For a missing mask, click the top-left and bottom-right corners on the candidate
in the local review page. It displays a pixel rectangle as `--mask X0 Y0 X1 Y1`
in the normalized image coordinates. Create a new immutable version:

```sh
$PY research/image_redaction/local_image_redactor.py revise \
  --work "$WORK" --id D0001 --mask 120 60 400 95
```

You may repeat `--mask` for multiple rectangles. Earlier manual rectangles are
retained in the new version. `revise` reads the original source path from the
private record when you omit it; the path is never printed. A manual revision
reuses the earlier automatic suggestions without rerunning OCR or GLiNER. Fix
recurring detection failures in the pipeline, then use `revise --redetect` to
regenerate all automatic suggestions; prior versions and decisions stay in
place. A later version does not revoke a prior export automatically, so remove
earlier approved exports from any proposed sharing packet if the review found
them unsafe.

If a suggestion covers part of a laboratory value, **Skip that candidate**.
In the local review page, click the affected pixels to see the zero-based mask
index (several indices may overlap). Use the index from the latest version:

```sh
$PY research/image_redaction/local_image_redactor.py revise --work "$WORK" --id D0001 --drop-mask 3
```

Repeat `--drop-mask` if multiple masks cover the value. This removes each
selected mask and creates a new candidate from the unchanged original. Check
the whole affected area for PII before Keep; removing a mask can expose text.
The earlier candidate and any Skip decision remain intact. A revised candidate
needs its own review and hash-bound Keep. `--drop-mask` cannot be combined with
`--redetect`, since a fresh detector run assigns new mask indices.

Before sharing or asking an assistant to inspect approved outputs, verify their
current bytes against the approval record:

```sh
$PY research/image_redaction/local_image_redactor.py verify --work "$WORK" --approved "$APPROVED"
```

An edited or missing export or manifest is reported as invalid. The review page
also marks a changed Keep export invalid; the earlier approval never applies to
changed bytes.

The review form accepts neutral organization/template/page/duplicate codes and
blank/completed/other status. Blank forms are layout examples, not completed
reports with null results. Do not assign training/test splits here. Keep any
reserved organization reserved. The shareable manifest has no source path,
source hash, OCR text, detected text, URLs, detailed mask coordinates or rejection
reason. Private records in `WORK` contain source mapping and masks. Do not move
those records into the approved directory. Back up this private workspace under
your own retention policy if `/private/tmp` may be cleared.

All four codes are optional labels **you assign**, not identifiers to transcribe
from a report. Leave organization or template blank when you cannot tell. Use a
page group only when images are pages of the same report, and a duplicate group
only when they are copies of the same source. Select `completed` only when actual
results appear, `blank` for an unfilled form, or `other` when neither fits or you
are unsure. The private source/reuse note can say `permission unknown`; the
rejection reason is useful only for Skip. Do not put personal details in codes.

The output is a flattened, single-frame RGB PNG; mask pixels are replaced, and
source EXIF, thumbnails and other inherited metadata are discarded. The tool
reopens each candidate to check its format, frame count, size and metadata. It
does not assert that review or a second OCR pass guarantees anonymization.

## PDF reports: local page rendering and image review

PDF input is supported by `propose-pdf`. Native PDF editing/export is not needed
for this workflow: the user accepted image-based parsing for the two remaining
original reports. The original PDF stays private and unchanged. PyMuPDF renders
one page at a time in memory; no persistent unmasked page PNGs are added. The existing
local OCR step uses a temporary PNG and removes it afterward. OCR, suggestions,
masking and review use the existing image workflow. No LlamaParse or Gemini is called.

Run from the repository root in your own host terminal. Install the updated isolated
requirements (now including PyMuPDF), then replace the example PDF path locally:

```sh
PY="$HOME/.venvs/lab-redactor-py311/bin/python"
WORK="$HOME/.local/share/lab-redaction-private"
APPROVED=data/approved_image_redactions
PDF='/absolute/private/path/to/remaining-report.pdf'

"$PY" -m pip install -r research/image_redaction/requirements.txt
"$PY" research/image_redaction/local_image_redactor.py propose-pdf "$PDF" \
  --work "$WORK" --id D0017
"$PY" research/image_redaction/local_image_redactor.py review \
  --work "$WORK" --approved "$APPROVED"
```

Use a fresh neutral report ID (for example D0018 for the second report); do not
reuse an image/report ID already in the workspace or approved collection. Add
`--allow-model-download` to `propose-pdf` only if the cached detector is not yet
available. The original host setup above is needed if that Python environment
no longer exists. Keep originals outside WORK, and keep WORK outside the repository
and synced folders. A temporary WORK directory must be backed up privately if you
want to retain editable revisions and decisions across cleanup/reboots.

The completed original-report exports are D0017 (2 pages) and D0018 (8 pages),
with all review follow-ups resolved by user confirmation on 2026-09-27. Exact
current PNG/JSON paths and checks are in the [private review](../../data/approved_image_redactions/originals-review-2026-09-26.md)
and [report inventory](../../data/research/report_inventory.md). D0018-p008 has no
results and remains grouped with D0018. The user confirmed KDL = original group B;
both reports remain reserved from adaptation in the [SFT preparation workflow](../extraction/preparation.md). The three-page example below illustrates
naming only; it does not describe D0017's actual page count.

A three-page PDF creates `D0017-p001`, `D0017-p002`, `D0017-p003` in WORK. These
are page IDs, not three independent reports. Review every page in your browser.
Use the displayed pixel rectangles and mask indices for the relevant page:

```sh
"$PY" research/image_redaction/local_image_redactor.py revise \
  --work "$WORK" --id D0017-p001 --mask 120 60 400 95
"$PY" research/image_redaction/local_image_redactor.py revise \
  --work "$WORK" --id D0017-p001 --drop-mask 3
```

The numbers above are illustrative; use the coordinates/indices from your candidate.
Revisions re-render the same original PDF page at the same DPI and retain the source
hash check. Dropping a mask restores original pixels in a new candidate; it cannot
recover content from a PDF that was already destructively redacted before import.
All revised candidates need their own Keep decision. Older exports are not revoked
automatically; remove unsafe/superseded versions from any proposed sharing packet.

**Only Keep-approved pages go into `data/approved_image_redactions/`.** Exports are
named `D0017-p001-v001-<hash>.png` with a matching `.json`. PDF-page manifests add
`source_document_id`, one-based `page_number`, `source_type: "pdf"`, `render_dpi: 300`,
and a fixed `page_group` equal to the neutral report ID. They contain no private
PDF path, source hash, OCR text or mask coordinates. Approval remains per page;
before downstream use, ensure every required page has its final approved version.
Keep pages in order and group them as one report for annotation and evaluation.

```sh
"$PY" research/image_redaction/local_image_redactor.py verify \
  --work "$WORK" --approved "$APPROVED"
```

PDF preparation accepts unencrypted PDFs up to 40 MB and 50 pages; each rendered
page must fit the existing 30-million-pixel / 10,000-pixel-side limits. Rotation is
rendered into the image, so manual masks use displayed pixel coordinates. A failure
may leave earlier page candidates private; it never approves them. A repeated
`propose-pdf` with existing page IDs is rejected. Review/revise existing candidates;
for a missing page, use `propose '/private/path/report.pdf' --work "$WORK" --id
D0017-p002` with the appropriate page ID rather than overwriting earlier versions.

Downstream parser text may differ from the native PDF condition. Record this input
as a rasterized PDF derivative, retain the private original mapping, and do not
reuse historical text/targets blindly. No real PDF was processed during development.

## Fictitious verification

```sh
"$HOME/.venvs/lab-redactor-py311/bin/python" -m unittest discover -s research/image_redaction/tests -v
"$HOME/.venvs/lab-redactor-py311/bin/python" research/image_redaction/make_fictitious_lab_images.py /private/tmp/lab-redactor-fixtures
"$HOME/.venvs/lab-redactor-py311/bin/python" research/image_redaction/local_image_redactor.py propose \
  /private/tmp/lab-redactor-fixtures/invented_report.png \
  --work /private/tmp/lab-redactor-demo-private --id D9001
```

The demo generator uses invented people, identifiers and values. It is not a
training dataset. For an offline smoke test, cache the public GLiNER checkpoint
first using the explicit download flag; later runs can omit it.

## Reviewed pilot outcome — 2026-09-26

The private review workspace contains 40 decided candidate versions from 16
explicitly named examples: 16 Keep and 24 Skip. All 16 final approved exports
passed `verify`, and an independent local check confirmed unique SHA-256 hashes,
single-frame RGB PNG format and empty inherited metadata. This does not measure
missed PII. Several approved files have visible code/identifier regions or
pre-existing smears that warrant another private review before redistribution.

The neutral-ID [approved inventory](../../data/approved_image_redactions/overview.md)
records visual organization/layout groups, document status, legibility and
specific re-review flags. The [research summary](../README.md) records the current
findings and next steps. Keep-approved means the
user approved an exact candidate for local inspection; it is not a license to
redistribute it or use its real values for training.
