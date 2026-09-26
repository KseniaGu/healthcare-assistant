# Local image redaction review

[Research status](../README.md)

Run commands from the repository root. The fictitious fixtures are in `fixtures/`;
they are test assets, not training data.

This standalone tool accepts **one explicitly named JPEG, PNG or single-frame WebP** at a time. It runs
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
brew install tesseract tesseract-lang
python3.11 -m venv /private/tmp/lab-redactor-venv
/private/tmp/lab-redactor-venv/bin/pip install -r research/image_redaction/requirements.txt
tesseract --list-langs
```

The model ID is `urchade/gliner_multi-v2.1`, threshold 0.4. Its public weights
are about 1.16 GB (Apache 2.0), with a separate tokenizer of about 4.3 MB (MIT),
downloaded only when you pass
`--allow-model-download`. Tesseract and its language data use Apache 2.0. Set
`HF_HUB_OFFLINE=1` on subsequent runs or simply omit the flag. The isolated
environment is deliberately separate from the app's Poetry environment.

## Private candidate and review flow

Keep originals and the private workspace in different directories. Keep the
workspace **outside the repository and any shared/synced folder**. The approved
directory below is gitignored by the repository. Run each explicit source path
locally; do not paste private paths, OCR or screenshots into a coding-assistant
session.

```sh
PY=/private/tmp/lab-redactor-venv/bin/python
WORK=/private/tmp/lab-redaction-private
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

## Fictitious verification

```sh
/private/tmp/lab-redactor-venv/bin/python -m unittest discover -s research/image_redaction/tests -v
/private/tmp/lab-redactor-venv/bin/python research/image_redaction/make_fictitious_lab_images.py /private/tmp/lab-redactor-fixtures
/private/tmp/lab-redactor-venv/bin/python research/image_redaction/local_image_redactor.py propose \
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
