import hashlib
import math
import mimetypes
import os
import re
import tempfile

from langdetect import LangDetectException, detect_langs

from backend.utils.schemas import *
from configs.enums import TestFlagTypeEnum

try:
    import requests
except Exception:
    requests = None

CHUNK = 65536


def _is_pdf_header(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"%PDF"
    except Exception:
        return False


def get_hash_and_size(path: Path) -> tuple[str, int]:
    sha = hashlib.sha256()
    total = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            sha.update(chunk)
            total += len(chunk)
    return sha.hexdigest(), total


def generate_artifact_meta(
    source_path: Path | str,
    source: Any = None,
    storage_type: str = None,
    exception_to_raise: type[Exception] = BaseException,
) -> ArtifactSchema:
    """Generates artifact metadata."""
    tmp_path = None
    source_path_is_url = str(source_path).lower().startswith(("http://", "https://"))
    try:
        if os.path.exists(source_path):
            local_path = source_path
            if storage_type is None:
                storage_type = "local"
        elif isinstance(source, (bytes, bytearray)):
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp_path = tmp.name
            tmp.write(source)
            tmp.close()
            local_path = tmp_path
        elif source_path_is_url:
            if requests is None:
                raise exception_to_raise("requests required to fetch URLs")
            resp = requests.get(source_path, stream=True, timeout=30)
            resp.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp_path = tmp.name
            for chunk in resp.iter_content(chunk_size=CHUNK):
                if chunk:
                    tmp.write(chunk)
            tmp.close()
            local_path = tmp_path
            if storage_type is None:
                storage_type = "http"
        else:
            raise exception_to_raise(
                """
                Unable to identify source: source type is not bytes/bytearray 
                or source_path is neither local nor valid remote url.
                """
            )

        if storage_type is None and source_path_is_url:
            storage_type = "http"

        filename = os.path.basename(source_path)
        checksum, size = get_hash_and_size(local_path)
        is_pdf = _is_pdf_header(local_path)
        mime, _ = mimetypes.guess_type(filename or local_path)
        mime_type = mime or ("application/pdf" if is_pdf else "application/octet-stream")

        meta = ArtifactSchema(
            storage_type=storage_type,
            storage_path=str(source_path),
            checksum=checksum,
            checksum_algorithm="sha256",
            mime_type=mime_type,
            size=size,
        )

        return meta
    except Exception as e:
        raise exception_to_raise(
            f"Error encountered while trying to get source ({source_path}) artifact metadata: {str(e)}"
        )
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def clean_json_markdown(text: str) -> str:
    """Removes Markdown code blocks."""
    # Remove `json at start and ``` at end, handling newlines
    pattern = r"^`{3}(?:json)?\s*(.*?)\s*`{3}$"
    match = re.search(pattern, text.strip(), re.DOTALL)
    if match:
        return match.group(1)
    return text.strip()


def is_valid_text(text: str, confidence_threshold: float = 0.8, common_words: set = ()) -> bool:
    """Returns True if the provided text is in a natural language and False if it is a garbage/encoding error."""
    if not text or len(text.strip()) < 3:
        return False

    # First, run statistical stopword check
    words = set(re.findall(r"\b[а-яё]{1,10}\b", text.lower()))
    intersection = words.intersection(common_words)

    # Calculate ratio: if text is long, we expect at least a few common words
    if len(text) > 50 and len(intersection) == 0:
        return False

    # Second, try algorithmic detection
    try:
        langs = detect_langs(text)
        for language in langs:
            if language.lang == "ru" and language.prob >= confidence_threshold:
                return True
        return False

    except LangDetectException:
        return False


def process_test_value(test_value: str | float | int | None, common_inequalities: tuple) -> tuple[float | None, str]:
    """Project a scalar or numeric limit; leave qualitative values and ranges as text.

    The caller stores the original value separately. Full-string matching prevents
    ranges or descriptive text from becoming misleading numeric chart points.
    """
    if test_value is None:
        return None, ""
    if isinstance(test_value, bool) or not isinstance(test_value, (str, int, float)):
        raise ValueError("Test value must be a string, number or null.")
    if isinstance(test_value, (int, float)):
        return (float(test_value), "") if math.isfinite(test_value) else (None, "")
    qualifiers = "|".join(re.escape(value) for value in sorted(set(common_inequalities) | {"<=", ">="}, key=len, reverse=True))
    number = r"[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?"
    match = re.fullmatch(rf"\s*({qualifiers})?\s*({number})\s*", test_value.replace("−", "-"))
    if match is None:
        return None, ""
    value = float(match[2].replace(",", "."))
    return (value, match[1] or "") if math.isfinite(value) else (None, "")


def observation_values(result: dict, common_inequalities: tuple) -> dict:
    """Keeps original result text and derive only a conservative numeric projection."""
    value = result.get("value")
    numeric, qualifier = process_test_value(value, common_inequalities)
    flag = str(result.get("flag") or "unknown").strip().lower()
    return {
        "observed_value": numeric,
        "observed_value_text": None if value is None else str(value),
        "observed_value_inequality": qualifier,
        "unit": result.get("unit"),
        "reference_range": result.get("reference_range"),
        "flag": flag if flag in TestFlagTypeEnum.__members__ else "unknown",
    }
