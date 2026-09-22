"""Local validation shared by application extraction and baseline runs."""

import json

from jsonschema import Draft202012Validator, FormatChecker

from backend.utils.helpers import clean_json_markdown


def parse_and_validate_extraction(text: str, schema: dict) -> dict:
    """Reject invalid outputs without coercing values or repairing model responses."""
    output = json.loads(clean_json_markdown(text))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    error = next(validator.iter_errors(output), None)
    if error is not None:
        raise ValueError(f"Extraction schema violation at {error.json_path}: {error.message}") from error
    return output
