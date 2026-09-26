"""Local, review-gated redaction of single-frame laboratory-report images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

TOOL_VERSION = "1.4"
MODEL_ID = "urchade/gliner_multi-v2.1"
TOKENIZER_ID = "microsoft/mdeberta-v3-base"
TOKENIZER_FILES = ("config.json", "tokenizer_config.json", "spm.model")
LABELS = ("person", "phone number", "email address", "date of birth", "age", "gender", "location")
THRESHOLD = 0.4
MAX_BYTES = 40_000_000
MAX_PIXELS = 30_000_000
MAX_SIDE = 10_000
PADDING = 3
ID_RE = re.compile(r"D[0-9]{4,8}\Z")
EMAIL = re.compile(r"(?<![\w.])[\w.+-]+@[\w.-]+\.[A-Za-zА-Яа-я]{2,}(?!\w)")
PHONE = re.compile(r"(?<!\w)(?:\+?7|8)[\s()\-]*(?:\d[\s()\-]*){10}(?!\w)")
URL = re.compile(r"(?i)\b(?:https?://|www\.)[^\s]+")
ID_LINE = re.compile(r"(?i)(?:№|\b(?:id|order|sample|заказ|пациент|образец|проб[аы]|направлени[ея]|штрих.?код)\b)\s*[:№#-]?\s*([A-ZА-Я0-9][A-ZА-Я0-9/-]{3,})")
DOB_LINE = re.compile(r"(?i)(?:дата\s+рождени[яе]|рождени[яе]|dob)\s*[:\s]*(\d{2}[./-]\d{2}[./-]\d{4})")
SEX_LINE = re.compile(r"(?i)(?:пол|sex|gender)\s*[:\s]+(муж\.?|жен\.?|м|ж|male|female)\b")
NAME_LINE = re.compile(r"(?i)^\s*(?:фио|пациент|patient|врач|doctor)\s*[:\s]+([^\n]+)")
ADDRESS_LINE = re.compile(r"(?i)^\s*(?:адрес|address)\s*[:\s]+([^\n]+)")
COLLECTION_LINE = re.compile(
    r"(?i)(?:collection|collected|performance|performed|report\s+date|"
    r"дата\s+(?:взятия|забора|сбора|исследования|выполнения|выдачи|поступления|анализа))"
)
EXCLUDED = {"organization", "lab result", "measurement"}


class RedactionError(Exception):
    """A safe, content-free error suitable for CLI output."""


@dataclass(frozen=True)
class Word:
    text: str
    start: int
    end: int
    box: tuple[int, int, int, int]
    line: tuple[int, int, int]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
    except OSError:
        raise RedactionError("hashing failed") from None
    return hasher.hexdigest()


def load_image(path: Path) -> Image.Image:
    try:
        if path.stat().st_size > MAX_BYTES:
            raise RedactionError("input exceeds size limit")
        with Image.open(path) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise RedactionError("unsupported image format")
            if getattr(source, "n_frames", 1) != 1:
                raise RedactionError("multi-frame image is unsupported")
            if source.width * source.height > MAX_PIXELS:
                raise RedactionError("input exceeds pixel limit")
            if max(source.size) > MAX_SIDE:
                raise RedactionError("input exceeds dimension limit")
            source.load()
            oriented = ImageOps.exif_transpose(source)
            image = Image.new("RGB", oriented.size, "white")
            if oriented.mode in {"RGBA", "LA"} or "transparency" in oriented.info:
                image.paste(oriented.convert("RGBA"), mask=oriented.convert("RGBA").getchannel("A"))
            else:
                image.paste(oriented.convert("RGB"))
            return image
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise RedactionError("unreadable image") from None


def ocr_words(image: Image.Image) -> tuple[str, list[Word]]:
    """Return OCR text and offset-bound word boxes in normalized image coordinates."""
    with tempfile.TemporaryDirectory(prefix="image-redactor-ocr-") as temporary:
        path = Path(temporary) / "input.png"
        image.save(path, format="PNG")
        try:
            result = subprocess.run(
                ["tesseract", str(path), "stdout", "-l", "rus+eng", "--psm", "3", "tsv"],
                check=True, capture_output=True, timeout=120,
                env={**os.environ, "OMP_THREAD_LIMIT": "2"},
            )
        except (OSError, subprocess.SubprocessError):
            raise RedactionError("local OCR failed") from None
    rows = csv.DictReader(io.StringIO(result.stdout.decode("utf-8", errors="replace")), delimiter="\t")
    output: list[str] = []
    words: list[Word] = []
    previous_line = None
    offset = 0
    for row in rows:
        value = row.get("text", "").strip()
        if not value:
            continue
        try:
            line = (int(row["block_num"]), int(row["par_num"]), int(row["line_num"]))
            x, y, width, height = (int(row[key]) for key in ("left", "top", "width", "height"))
        except (KeyError, ValueError):
            raise RedactionError("local OCR produced invalid coordinates") from None
        if width <= 0 or height <= 0:
            continue
        if x < 0 or y < 0 or x + width > image.width or y + height > image.height:
            raise RedactionError("local OCR produced invalid coordinates")
        if output:
            separator = "\n" if line != previous_line else " "
            output.append(separator)
            offset += len(separator)
        start = offset
        output.append(value)
        offset += len(value)
        words.append(Word(value, start, start + len(value), (x, y, x + width, y + height), line))
        previous_line = line
    return "".join(output), words


def spans_to_boxes(words: list[Word], start: int, end: int, size: tuple[int, int]) -> list[list[int]]:
    """Map one occurrence to separate line boxes; never search for matching text globally."""
    if not 0 <= start < end:
        raise RedactionError("invalid detector offsets")
    selected = [word for word in words if word.start < end and word.end > start]
    if not selected:
        return []
    groups: dict[tuple[int, int, int], list[Word]] = {}
    for word in selected:
        groups.setdefault(word.line, []).append(word)
    boxes = []
    for group in groups.values():
        x0 = max(0, min(word.box[0] for word in group) - PADDING)
        y0 = max(0, min(word.box[1] for word in group) - PADDING)
        x1 = min(size[0], max(word.box[2] for word in group) + PADDING)
        y1 = min(size[1], max(word.box[3] for word in group) + PADDING)
        boxes.append([x0, y0, x1, y1])
    return boxes


def detect_entities(text: str, *, allow_download: bool = False) -> list[dict]:
    if not text.strip():
        return []
    os.environ["HF_HUB_OFFLINE"] = "0" if allow_download else "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "0" if allow_download else "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    # Some native dependencies create session files relative to cwd on import.
    # Keep any such files in a disposable private directory.
    previous_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="image-redactor-runtime-") as runtime_dir:
        try:
            os.chdir(runtime_dir)
            return _predict_local_entities(text, allow_download)
        finally:
            os.chdir(previous_cwd)


def cached_snapshot(model_id: str, required_files: tuple[str, ...]) -> Path:
    """Find a complete public model snapshot without contacting the model hub."""
    roots = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    for root in dict.fromkeys(roots):
        repository = root / ("models--" + model_id.replace("/", "--"))
        snapshots = repository / "snapshots"
        if not snapshots.is_dir():
            continue
        main_ref = repository / "refs" / "main"
        preferred = main_ref.read_text(encoding="utf-8").strip() if main_ref.is_file() else ""
        candidates = ([snapshots / preferred] if preferred else []) + sorted(snapshots.iterdir(), reverse=True)
        for candidate in candidates:
            if all((candidate / name).is_file() for name in required_files):
                return candidate
    raise RedactionError("model component unavailable offline; use --allow-model-download once")


def cached_model_dir() -> Path:
    return cached_snapshot(MODEL_ID, ("gliner_config.json", "model.safetensors"))


def local_model_bundle(model_dir: Path, tokenizer_dir: Path) -> Path:
    """Give GLiNER local tokenizer and encoder config paths without hub lookup."""
    bundle = Path.cwd() / "model_bundle"
    bundle.mkdir()
    config = json.loads((model_dir / "gliner_config.json").read_text(encoding="utf-8"))
    if config.get("model_name") != TOKENIZER_ID:
        raise RedactionError("unexpected GLiNER encoder model")
    config["model_name"] = str(bundle)
    (bundle / "gliner_config.json").write_text(json.dumps(config), encoding="utf-8")
    (bundle / "model.safetensors").symlink_to((model_dir / "model.safetensors").resolve())
    for name in TOKENIZER_FILES:
        (bundle / name).symlink_to((tokenizer_dir / name).resolve())
    return bundle


def _predict_local_entities(text: str, allow_download: bool) -> list[dict]:
    stage = "dependency import"
    try:
        import onnxruntime

        onnxruntime.disable_telemetry_events()
        from huggingface_hub import snapshot_download
        from gliner import GLiNER

        stage = "model cache lookup"
        if allow_download:
            # The repository also contains a second 1.16 GB pickle checkpoint;
            # only the safetensors weights and config are needed for inference.
            model_dir = snapshot_download(
                repo_id=MODEL_ID,
                allow_patterns=["gliner_config.json", "model.safetensors"],
            )
            tokenizer_dir = snapshot_download(repo_id=TOKENIZER_ID, allow_patterns=list(TOKENIZER_FILES))
        else:
            model_dir = cached_model_dir()
            tokenizer_dir = cached_snapshot(TOKENIZER_ID, TOKENIZER_FILES)
        stage = "model loading"
        model = GLiNER.from_pretrained(local_model_bundle(Path(model_dir), Path(tokenizer_dir)))
        model.to("cpu")
        entities = []
        # Keep windows bounded, and retain global offsets for repeated occurrences.
        cursor = 0
        while cursor < len(text):
            stage = "prediction"
            end = min(len(text), cursor + 1100)
            if end < len(text):
                boundary = text.rfind("\n", cursor + 500, end)
                if boundary > cursor:
                    end = boundary + 1
            chunk = text[cursor:end]
            for entity in model.predict_entities(chunk, list(LABELS), threshold=THRESHOLD):
                if "start" not in entity or "end" not in entity:
                    raise RedactionError("detector returned no offsets")
                entities.append({"start": cursor + entity["start"], "end": cursor + entity["end"], "kind": entity["label"]})
            if end == len(text):
                break
            cursor = max(cursor + 1, end - 100)
        return entities
    except RedactionError:
        raise
    except Exception as error:
        raise RedactionError(f"local entity detector failed during {stage} ({type(error).__name__})") from None


def pattern_entities(text: str) -> list[dict]:
    found = []
    for kind, expression in (("email", EMAIL), ("phone", PHONE), ("portal_link", URL)):
        found.extend({"start": match.start(), "end": match.end(), "kind": kind} for match in expression.finditer(text))
    offset = 0
    for line in text.splitlines(keepends=True):
        for kind, expression in (("identifier", ID_LINE), ("date_of_birth", DOB_LINE), ("sex", SEX_LINE),
                                 ("named_person_field", NAME_LINE), ("address", ADDRESS_LINE)):
            found.extend({"start": offset + match.start(1), "end": offset + match.end(1), "kind": kind} for match in expression.finditer(line))
        offset += len(line)
    return found


def code_boxes(image: Image.Image) -> tuple[list[list[int]], list[str]]:
    """Detect QR and supported barcodes without decoding or logging their payloads."""
    requirements = ["Inspect all QR codes and barcodes visually", "Inspect signatures, seals and handwriting visually"]
    try:
        import cv2
        import numpy as np

        pixels = np.asarray(image)
        detector = cv2.QRCodeDetector()
        found, points = detector.detectMulti(pixels)
        polygons = list(points) if found and points is not None else []
        if not polygons:
            found, points = detector.detect(pixels)
            if found and points is not None:
                polygons = [points[0]]
        if hasattr(cv2, "barcode_BarcodeDetector"):
            try:
                barcode = cv2.barcode_BarcodeDetector()
                found, points = barcode.detect(pixels)
                if found and points is not None:
                    polygons.extend(points)
            except Exception:
                requirements.append("Barcode detection failed; inspect every barcode")
        else:
            requirements.append("Barcode detector unavailable; inspect every barcode")
        boxes = []
        for polygon in polygons:
            xs = polygon[:, 0]
            ys = polygon[:, 1]
            boxes.append([max(0, int(xs.min()) - 5), max(0, int(ys.min()) - 5),
                          min(image.width, int(xs.max()) + 6), min(image.height, int(ys.max()) + 6)])
        return boxes, requirements
    except Exception:
        requirements.append("Code detection failed; inspect every QR code and barcode")
        return [], requirements


def normalize_boxes(boxes: list[list[int]], size: tuple[int, int]) -> list[list[int]]:
    result = []
    for box in boxes:
        if len(box) != 4 or not all(isinstance(value, int) for value in box):
            raise RedactionError("invalid manual mask")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
            raise RedactionError("manual mask outside image")
        if box not in result:
            result.append(box)
    return result


def render(image: Image.Image, masks: list[dict]) -> Image.Image:
    output = Image.new("RGB", image.size, "white")
    output.paste(image)
    draw = ImageDraw.Draw(output)
    for mask in masks:
        draw.rectangle(mask["box"], fill=(0, 0, 0))
    return output


def save_clean(image: Image.Image, path: Path) -> None:
    with path.open("xb") as stream:
        image.save(stream, format="PNG")
    with Image.open(path) as check:
        if check.format != "PNG" or check.mode != "RGB" or check.size != image.size or check.n_frames != 1 or check.info:
            raise RedactionError("export verification failed")


def private_record(work: Path, doc_id: str, version: int) -> tuple[Path, dict]:
    path = work / doc_id / f"v{version:03d}" / "record.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    path.chmod(0o600)


def propose(source: Path, work: Path, doc_id: str, *, manual: list[list[int]] | None = None,
            drop_masks: list[int] | None = None, allow_download: bool = False, redetect: bool = False) -> dict:
    if not ID_RE.fullmatch(doc_id):
        raise RedactionError("document ID must match D plus 4-8 digits")
    source = source.resolve(strict=True)
    work = work.resolve()
    if work == source.parent or work in source.parents or source.parent in work.parents:
        raise RedactionError("private workspace must be separate from originals")
    source_hash = digest(source)
    image = load_image(source)
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    work.chmod(0o700)
    folder = work / doc_id
    existing = sorted(folder.glob("v[0-9][0-9][0-9]")) if folder.exists() else []
    version = len(existing) + 1
    new_manual = manual or []
    prior = None
    if existing:
        _, prior = private_record(work, doc_id, version - 1)
        if prior["source_path"] != str(source) or prior["source_sha256"] != source_hash:
            raise RedactionError("document ID belongs to a different source")
    drop_indices = drop_masks or []
    if drop_indices and (prior is None or redetect or len(set(drop_indices)) != len(drop_indices)
                         or any(index < 0 or index >= len(prior["masks"]) for index in drop_indices)):
        raise RedactionError("invalid mask index for revision")
    manual_boxes = normalize_boxes(
        ([mask["box"] for index, mask in enumerate(prior["masks"])
          if mask["origin"] == "manual" and index not in drop_indices] if prior else []) + new_manual,
        image.size,
    )
    reuse_suggestions = prior is not None and bool(new_manual or drop_indices) and not redetect
    if reuse_suggestions:
        masks = [mask.copy() for index, mask in enumerate(prior["masks"])
                 if mask["origin"] == "automatic" and index not in drop_indices]
        requirements = list(prior["requirements"])
        ocr_word_count = prior["ocr_word_count"]
    else:
        text, words = ocr_words(image)
        entities = detect_entities(text, allow_download=allow_download) + pattern_entities(text)
        masks = []
        seen = set()
        unmapped = 0
        for entity in entities:
            if entity["kind"] in EXCLUDED:
                continue
            if entity["kind"] == "date of birth":
                start_of_line = text.rfind("\n", 0, entity["start"]) + 1
                end_of_line = text.find("\n", entity["end"])
                if end_of_line < 0:
                    end_of_line = len(text)
                if COLLECTION_LINE.search(text[start_of_line:end_of_line]):
                    continue
            boxes = spans_to_boxes(words, entity["start"], entity["end"], image.size)
            if not boxes:
                unmapped += 1
            for box in boxes:
                key = (entity["kind"], tuple(box))
                if key not in seen:
                    masks.append({"kind": entity["kind"], "box": box, "origin": "automatic"})
                    seen.add(key)
        codes, requirements = code_boxes(image)
        if unmapped:
            requirements.append(f"{unmapped} detected spans had no OCR box; inspect visually")
        for box in codes:
            masks.append({"kind": "code", "box": box, "origin": "automatic"})
        requirements.extend(["Inspect names, contacts, IDs and all image edges",
                             "Choose whether collection/performance dates and result patterns are safe for this use"])
        ocr_word_count = len(words)
    for box in manual_boxes:
        masks.append({"kind": "manual", "box": box, "origin": "manual"})
    target = folder / f"v{version:03d}"
    if digest(source) != source_hash:
        raise RedactionError("source changed during processing")
    folder.mkdir(exist_ok=True, mode=0o700)
    folder.chmod(0o700)
    target.mkdir(parents=True, exist_ok=False, mode=0o700)
    candidate = target / "candidate.png"
    save_clean(render(image, masks), candidate)
    record = {"id": doc_id, "version": version, "source_path": str(source), "source_sha256": source_hash,
              "candidate_sha256": digest(candidate), "size": image.size, "masks": masks,
              "removed_mask_count": (prior.get("removed_mask_count", 0) if prior and reuse_suggestions else 0)
              + len(drop_indices),
              "requirements": requirements, "ocr_word_count": ocr_word_count, "status": "pending",
              "tool_version": TOOL_VERSION, "model": MODEL_ID, "created_at": datetime.now(timezone.utc).isoformat()}
    write_json_new(target / "record.json", record)
    return record


def decisions_path(work: Path) -> Path:
    return work / "decisions.json"


def read_decisions(work: Path) -> dict:
    path = decisions_path(work)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def approval_valid(approved: Path, decision: dict) -> bool:
    if decision["choice"] != "keep":
        return False
    name = decision.get("export_name", "")
    if not re.fullmatch(r"D[0-9]{4,8}-v[0-9]{3}-[0-9a-f]{12}\.png", name):
        return False
    image_path = approved / name
    manifest_path = image_path.with_suffix(".json")
    if not image_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return (digest(image_path) == decision["sha256"] == manifest["export_sha256"]
                and digest(manifest_path) == decision["manifest_sha256"])
    except (OSError, ValueError, KeyError):
        return False


def decide(work: Path, approved: Path, doc_id: str, version: int, sha256: str, choice: str,
           reason: str = "", codes: dict | None = None, private_source_note: str = "") -> dict:
    if not ID_RE.fullmatch(doc_id) or choice not in {"keep", "skip"}:
        raise RedactionError("invalid decision")
    path, record = private_record(work, doc_id, version)
    candidate = path.parent / "candidate.png"
    if digest(candidate) != sha256 or sha256 != record["candidate_sha256"]:
        raise RedactionError("candidate hash changed; review again")
    key = f"{doc_id}/v{version:03d}"
    decisions = read_decisions(work)
    if key in decisions:
        raise RedactionError("candidate already decided")
    decision = {"choice": choice, "sha256": sha256, "reason": reason[:500],
                "private_source_note": private_source_note[:1000],
                "decided_at": datetime.now(timezone.utc).isoformat()}
    if choice == "keep":
        if (codes or {}).get("document_status", "other") not in {"blank", "completed", "other"}:
            raise RedactionError("invalid document status")
        for field in ("organization_code", "template_code", "page_group", "duplicate_group"):
            if not re.fullmatch(r"[A-Za-z0-9_-]{0,40}", (codes or {}).get(field, "")):
                raise RedactionError("invalid shareable code")
        approved.mkdir(parents=True, exist_ok=True)
        name = f"{doc_id}-v{version:03d}-{sha256[:12]}.png"
        destination = approved / name
        if destination.exists():
            raise RedactionError("approved export already exists")
        with candidate.open("rb") as source_stream, destination.open("xb") as output_stream:
            shutil.copyfileobj(source_stream, output_stream)
        if digest(destination) != sha256:
            destination.unlink()
            raise RedactionError("approved export hash mismatch")
        counts = Counter(mask["kind"] for mask in record["masks"] if mask["origin"] == "automatic")
        manifest = {"id": doc_id, "version": version, "organization_code": (codes or {}).get("organization_code", ""),
                    "template_code": (codes or {}).get("template_code", ""), "page_group": (codes or {}).get("page_group", ""),
                    "duplicate_group": (codes or {}).get("duplicate_group", ""), "document_status": (codes or {}).get("document_status", "other"),
                    "automatic_mask_counts": dict(counts), "manual_mask_count": sum(mask["origin"] == "manual" for mask in record["masks"]),
                    "removed_mask_count": record.get("removed_mask_count", 0),
                    "tool_version": TOOL_VERSION, "model": MODEL_ID, "threshold": THRESHOLD, "padding_pixels": PADDING,
                    "export_sha256": sha256, "approval_status": "kept"}
        manifest_path = approved / name.replace(".png", ".json")
        write_json_new(manifest_path, manifest)
        decision["export_name"] = name
        decision["manifest_sha256"] = digest(manifest_path)
    decisions[key] = decision
    temporary = decisions_path(work).with_suffix(".tmp")
    temporary.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(decisions_path(work))
    return decision


def review_server(work: Path, approved: Path, port: int) -> None:
    if work == approved or work in approved.parents or approved in work.parents:
        raise RedactionError("private and approved directories must be separate")
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def _send(self, payload: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline' 'self'; "
                             "script-src 'unsafe-inline' 'self'; form-action 'self'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parse_qs(parsed.query).get("token") != [token]:
                self._send(b"Not found", "text/plain", 404)
                return
            parts = parsed.path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "image" and ID_RE.fullmatch(parts[1]) and re.fullmatch(r"v[0-9]{3}", parts[2]):
                path = work / parts[1] / parts[2] / "candidate.png"
                if path.is_file():
                    self._send(path.read_bytes(), "image/png")
                    return
            if parsed.path != "/":
                self._send(b"Not found", "text/plain", 404)
                return
            decisions = read_decisions(work)
            sections = []
            for path in sorted(work.glob("D[0-9]*/v[0-9][0-9][0-9]/record.json")):
                record = json.loads(path.read_text(encoding="utf-8"))
                doc_id, version, sha = record["id"], record["version"], record["candidate_sha256"]
                key = f"{doc_id}/v{version:03d}"
                decision = decisions.get(key, {})
                state = decision.get("choice", "pending")
                if state == "keep" and not approval_valid(approved, decision):
                    state = "changed; approval invalid"
                requirements = "".join(f"<li>{html.escape(item)}</li>" for item in record["requirements"])
                counts = Counter(mask["kind"] for mask in record["masks"])
                boxes = ", ".join(f"{kind}: {count}" for kind, count in sorted(counts.items())) or "none"
                form = ""
                if state == "pending":
                    form = (f'<form method="post" action="/decision"><input type="hidden" name="id" value="{doc_id}">'
                            f'<input type="hidden" name="version" value="{version}">'
                            f'<input type="hidden" name="sha256" value="{sha}">'
                            f'<input type="hidden" name="token" value="{token}">'
                            '<p><strong>Optional labels you assign:</strong> Do not copy values from the report. '
                            'Leave any unknown field empty. Reuse a code only when you know two images belong together.</p>'
                            '<p>Organization code <input name="organization_code" placeholder="optional"> '
                            'Template code <input name="template_code" placeholder="optional"></p>'
                            '<p>Page group <input name="page_group" placeholder="optional"> '
                            'Duplicate group <input name="duplicate_group" placeholder="optional"></p>'
                            '<p>Status <select name="document_status"><option>other</option><option>blank</option>'
                            '<option>completed</option></select> '
                            '(blank form, report with actual results, or other/uncertain)</p>'
                            '<p>Source/reuse permission note (private, optional) '
                            '<input name="private_source_note" placeholder="e.g. permission unknown"></p>'
                            '<p>Rejection reason (private, optional; for Skip) '
                            '<input name="reason" placeholder="e.g. missed seal"></p>'
                            '<p><button name="choice" value="keep">Keep exact candidate</button> '
                            '<button name="choice" value="skip">Skip</button></p></form>')
                sections.append(f'<section><h2>{doc_id} v{version:03d} — {state}</h2><p>SHA-256: <code>{sha}</code></p>'
                                f'<p>Image: {record["size"][0]} × {record["size"][1]}. Masks: {html.escape(boxes)}</p>'
                                f'<img src="/image/{doc_id}/v{version:03d}?token={token}" '
                                f'data-key="{doc_id}-v{version:03d}" '
                                f'data-masks="{html.escape(json.dumps([mask["box"] for mask in record["masks"]]))}" '
                                f'alt="Private candidate">'
                                f'<p>Click a masked area to find its mask index: '
                                f'<code id="hit-{doc_id}-v{version:03d}">none</code>. '
                                f'Use <code>revise --drop-mask INDEX</code> to remove that mask in a new version.</p>'
                                f'<details><summary>Mask indices and pixel boxes</summary><ol start="0">'
                                + "".join(f'<li>{html.escape(mask["kind"])} {html.escape(str(mask["box"]))}</li>'
                                          for mask in record["masks"])
                                + '</ol></details>'
                                f'<p>For a missing mask, click its upper-left and lower-right corners. '
                                f'Coordinates: <code id="rect-{doc_id}-v{version:03d}">none</code></p>'
                                f'<h3>Review requirements</h3><ul>{requirements}</ul>{form}</section>')
            page = ('<!doctype html><meta charset="utf-8"><title>Private image review</title>'
                    '<style>body{font:16px system-ui;max-width:1100px;margin:2em auto}section{border-top:2px solid #444;padding:1em 0}'
                    'img{max-width:100%;border:1px solid #888}input{margin:.3em}code{overflow-wrap:anywhere}</style>'
                    '<h1>Private image review</h1><p>Use your own browser on this device. Inspect every pixel before Keep. '
                    'Candidate images may still contain PII. Keep copies the exact hash shown; Skip shares nothing. '
                    'For corrections, use revise to create a new version, then review it separately.</p>'
                    + "".join(sections)
                    + """<script>
let first = null;
document.querySelectorAll('img[data-key]').forEach(image => {
  image.addEventListener('click', event => {
    const bounds = image.getBoundingClientRect();
    const x = Math.round((event.clientX - bounds.left) * image.naturalWidth / bounds.width);
    const y = Math.round((event.clientY - bounds.top) * image.naturalHeight / bounds.height);
    const hits = JSON.parse(image.dataset.masks).flatMap((box, index) =>
      x >= box[0] && x <= box[2] && y >= box[1] && y <= box[3] ? [index] : []);
    document.getElementById('hit-' + image.dataset.key).textContent = hits.length ? hits.join(', ') : 'none';
    const label = document.getElementById('rect-' + image.dataset.key);
    if (!first || first.image !== image) {
      first = {image, x, y};
      label.textContent = 'first corner: ' + x + ' ' + y;
    } else {
      const box = [Math.min(first.x, x), Math.min(first.y, y),
                   Math.max(first.x, x), Math.max(first.y, y)];
      label.textContent = '--mask ' + box.join(' ');
      first = null;
    }
  });
});
</script>""")
            self._send(page.encode(), "text/html; charset=utf-8")

        def do_POST(self) -> None:
            if self.path != "/decision" or int(self.headers.get("Content-Length", "0")) > 10000:
                self._send(b"Invalid request", "text/plain", 400)
                return
            form = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
            value = lambda key: form.get(key, [""])[0]
            if value("token") != token:
                self._send(b"Invalid request", "text/plain", 400)
                return
            codes = {key: value(key) for key in ("organization_code", "template_code", "page_group", "duplicate_group", "document_status")}
            if codes["document_status"] not in {"blank", "completed", "other"}:
                self._send(b"Invalid status", "text/plain", 400)
                return
            try:
                decide(work, approved, value("id"), int(value("version")), value("sha256"), value("choice"),
                       value("reason"), codes, value("private_source_note"))
            except (RedactionError, ValueError, FileNotFoundError):
                self._send(b"Decision rejected. Refresh and review again.", "text/plain", 400)
                return
            self.send_response(303)
            self.send_header("Location", f"/?token={token}")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Private review at http://127.0.0.1:{port}/?token={token}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local, hash-bound review of laboratory report images")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("propose", "revise"):
        item = sub.add_parser(command)
        item.add_argument("source", type=Path, nargs="?" if command == "revise" else None,
                          help="one explicit JPEG, PNG or WebP path; revise can reuse the private source mapping")
        item.add_argument("--work", type=Path, required=True, help="private workspace outside the repository")
        item.add_argument("--id", required=True)
        item.add_argument("--mask", nargs=4, type=int, action="append", metavar=("X0", "Y0", "X1", "Y1"), default=[])
        if command == "revise":
            item.add_argument("--redetect", action="store_true", help="rerun OCR and GLiNER after a pipeline correction")
            item.add_argument("--drop-mask", type=int, action="append", default=[], metavar="INDEX",
                              help="remove a mask by its zero-based index from the latest candidate")
        item.add_argument("--allow-model-download", action="store_true")
    item = sub.add_parser("review")
    item.add_argument("--work", type=Path, required=True)
    item.add_argument("--approved", type=Path, required=True)
    item.add_argument("--port", type=int, default=8765)
    item = sub.add_parser("verify")
    item.add_argument("--work", type=Path, required=True)
    item.add_argument("--approved", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in {"propose", "revise"}:
            if args.command == "propose" and (args.work / args.id).exists():
                raise RedactionError("document ID already exists; use revise")
            if args.command == "revise" and not (args.work / args.id).exists():
                raise RedactionError("document ID does not exist; use propose")
            source = args.source
            if args.command == "revise" and source is None:
                versions = sorted((args.work / args.id).glob("v[0-9][0-9][0-9]"))
                if not versions:
                    raise RedactionError("document ID has no candidate")
                _, previous = private_record(args.work, args.id, int(versions[-1].name[1:]))
                source = Path(previous["source_path"])
            record = propose(source, args.work, args.id, manual=args.mask,
                             drop_masks=getattr(args, "drop_mask", []),
                             allow_download=args.allow_model_download, redetect=getattr(args, "redetect", False))
            print(f'{record["id"]} v{record["version"]:03d} candidate SHA-256 {record["candidate_sha256"]}; review required')
        elif args.command == "review":
            review_server(args.work.resolve(), args.approved.resolve(), args.port)
        else:
            decisions = read_decisions(args.work)
            invalid = []
            for key, decision in sorted(decisions.items()):
                if decision["choice"] == "keep" and not approval_valid(args.approved, decision):
                    invalid.append(key)
            print(f"Kept {sum(d['choice'] == 'keep' for d in decisions.values())}; "
                  f"skipped {sum(d['choice'] == 'skip' for d in decisions.values())}; invalid {len(invalid)}")
            for key in invalid:
                print(f"{key}: hash or manifest mismatch")
            return 1 if invalid else 0
    except RedactionError as error:
        print(f"Rejected: {error}", file=sys.stderr)
        return 2
    except Exception:
        print("Redaction request failed; check input, setup and private workspace.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
