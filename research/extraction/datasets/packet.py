"""Self-contained CA1 packet integrity, acceptance, and model-facing validation."""

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, TypedDict

from jsonschema import Draft202012Validator, FormatChecker


class ChatMessage(TypedDict):
    role: str
    content: str


class TrainingRow(TypedDict):
    input_ids: list[int]
    labels: list[int]


class TokenRow(TrainingRow):
    id: str
    split: str


class CA1Result(TypedDict):
    name: str
    value: str | None
    unit: str | None
    reference_range: str | None


class CA1Target(TypedDict):
    date: str | None
    results: list[CA1Result]
    errors: list[str]


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data/research/extraction"
PACKET = BASE / "ca1-sft-v3"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
LIMIT = 8192
COUNTS = {"train": 320, "validation": 64}
WRAPPER_START = "Extract laboratory observations from this parser output:\n<report>\n"
WRAPPER_END = "\n</report>\n"
BASES = {"model_facing_parity", "reviewed_changes"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def sha(path_or_text: Path | str) -> str:
    value = path_or_text.read_bytes() if isinstance(path_or_text, Path) else path_or_text.encode()
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def validate_target(content: str, schema: dict[str, Any]) -> Any:
    """Accepts only unfenced JSON satisfying the CA1 schema and date formats."""
    target = json.loads(content)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(target)
    return target


def completion_tokens(tokenizer: Any, messages: list[ChatMessage]) -> tuple[list[int], list[int], int]:
    """Builds the complete chat and locates the assistant token boundary.

    Args:
        messages: System/user prompt followed by the complete assistant target.

    Returns:
        Full token IDs, labels masked through the generation prompt, and the
        number of prefix tokens. Assistant tokens after the boundary remain visible.

    Raises:
        ValueError: The chat template changes the prefix, exceeds 8,192 tokens,
            or omits the assistant completion or EOS token. Input is never truncated.
    """
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if hasattr(prefix, "keys"):
        prefix = prefix["input_ids"]
    if hasattr(full, "keys"):
        full = full["input_ids"]
    if full[: len(prefix)] != prefix:
        raise ValueError("Assistant boundary is not a token prefix")
    if len(full) > LIMIT:
        raise ValueError("Full conversation exceeds 8192 tokens; no truncation")
    labels = [-100] * len(prefix) + full[len(prefix) :]
    if len(full) == len(prefix) or tokenizer.eos_token_id not in full[len(prefix) :]:
        raise ValueError("Assistant completion/end token missing")
    return full, labels, len(prefix)


def tokenize(tokenizer: Any, messages: list[ChatMessage]) -> tuple[list[int], list[int], int, int]:
    """Masks every token outside the assistant completion through EOS.

    Returns:
        Token IDs, labels, prefix length, and supervised token count including EOS.
        Any post-EOS newline is masked with -100.

    Raises:
        ValueError: The chat boundary, length, EOS count, or assistant-only
            label mask violates the packet contract.
    """
    ids, labels, prefix = completion_tokens(tokenizer, messages)
    completion = ids[prefix:]
    if completion.count(tokenizer.eos_token_id) != 1:
        raise ValueError("Expected one assistant end token")
    end = prefix + completion.index(tokenizer.eos_token_id)
    labels[end + 1 :] = [-100] * (len(ids) - end - 1)
    if (
        labels[:prefix] != [-100] * prefix
        or labels[prefix : end + 1] != ids[prefix : end + 1]
        or any(label != -100 for label in labels[end + 1 :])
    ):
        raise ValueError("Assistant-only labels invalid")
    return ids, labels, prefix, end - prefix + 1


def tokenizer_and_hashes(path: Path) -> tuple[Any, dict[str, Any]]:
    """Loads the pinned tokenizer from local files after checking provenance.

    Returns:
        The tokenizer and its recorded provenance, including file hashes.

    Raises:
        ValueError: The model identity, tokenizer files, or chat template differ
            from the retained provenance.
    """
    from transformers import AutoTokenizer

    provenance = read_json(path / "provenance.json")
    if (provenance["model"], provenance["revision"]) != (MODEL, REVISION):
        raise ValueError("Tokenizer identity changed")
    if any(sha(path / name) != digest for name, digest in provenance["files"].items()):
        raise ValueError("Tokenizer files changed")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    if sha(tokenizer.chat_template) != provenance["chat_template_sha256"]:
        raise ValueError("Chat template changed")
    return tokenizer, provenance


def _payload_files(packet: Path) -> dict[str, Path]:
    return {
        str(path.relative_to(packet)): path
        for path in packet.rglob("*")
        if path.is_file() and path.relative_to(packet).as_posix() not in ("SHA256SUMS.json", "acceptance.json")
    }


def write_checksums(packet: Path) -> str:
    """Writes the canonical checksum manifest for candidate payload files.

    Returns:
        SHA-256 of SHA256SUMS.json. The checksum file itself and the separate
        acceptance record are excluded from the payload file list.
    """
    files = {name: sha(path) for name, path in _payload_files(packet).items()}
    (packet / "SHA256SUMS.json").write_bytes(
        (json.dumps(files, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    )
    return sha(packet / "SHA256SUMS.json")


def verify_integrity(packet: Path) -> str:
    """Verifies the complete packet payload against canonical checksums.

    Returns:
        SHA-256 of the exact SHA256SUMS.json bytes, used as the content digest.

    Raises:
        ValueError: The payload contains symlinks, missing or extra files,
            noncanonical checksums, or a mismatched file hash.
    """
    if any(path.is_symlink() for path in packet.rglob("*")):
        raise ValueError("Packet payload contains a symlink")

    raw = (packet / "SHA256SUMS.json").read_bytes()
    checksums = json.loads(raw)
    canonical = (json.dumps(checksums, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if not isinstance(checksums, dict) or raw != canonical:
        raise ValueError("Noncanonical checksum manifest")

    files = _payload_files(packet)
    if set(checksums) != set(files):
        raise ValueError("Packet checksum mismatch")
    for name, digest in checksums.items():
        if not isinstance(digest, str) or not SHA256.fullmatch(digest) or sha(files[name]) != digest:
            raise ValueError("Packet checksum mismatch")
    return hashlib.sha256(raw).hexdigest()


def verify_acceptance(packet: Path, content_sha256: str | None = None) -> dict[str, Any]:
    """Checks that acceptance is bound to the current packet content.

    Args:
        content_sha256: Already verified payload digest, when the caller has one;
            otherwise packet integrity is checked here.

    Returns:
        The acceptance record with matching content digest and source commit.

    Raises:
        ValueError: Acceptance is absent, malformed, or bound to another payload.
    """
    content_sha256 = content_sha256 or verify_integrity(packet)
    path = packet / "acceptance.json"
    if not path.is_file():
        raise ValueError("Packet is not accepted")

    record = read_json(path)
    manifest = read_json(packet / "manifest.json")
    if (
        record.get("content_sha256") != content_sha256
        or record.get("source_commit") != manifest.get("source_commit")
        or not isinstance(record.get("source_commit"), str)
        or not COMMIT.fullmatch(record["source_commit"])
        or record.get("basis") not in BASES
        or not isinstance(record.get("evidence_sha256"), str)
        or not SHA256.fullmatch(record["evidence_sha256"])
        or not isinstance(record.get("evidence_reference"), str)
        or not record["evidence_reference"]
    ):
        raise ValueError("Invalid packet acceptance record")
    return record


def accept(packet: Path, basis: str, evidence_path: Path, reference_sha256: str | None = None) -> dict[str, Any]:
    """Binds reviewed evidence to one immutable candidate packet.

    Args:
        basis: Either model-facing parity with a recorded prior packet or
            completed review of changed examples.
        reference_sha256: Required prior-packet digest for parity acceptance.

    Returns:
        The acceptance record bound to the candidate content and source commit.

    Raises:
        ValueError: Evidence is incomplete or does not match the candidate.
        FileExistsError: A different acceptance record already exists.
    """
    content_sha256 = verify_integrity(packet)
    manifest = read_json(packet / "manifest.json")
    evidence = read_json(evidence_path)
    if (
        basis not in BASES
        or evidence.get("basis") != basis
        or evidence.get("result") != "pass"
        or evidence.get("content_sha256") != content_sha256
    ):
        raise ValueError("Acceptance evidence does not match this candidate")
    if not isinstance(manifest.get("source_commit"), str) or not COMMIT.fullmatch(manifest["source_commit"]):
        raise ValueError("Packet lacks a committed source revision")
    if basis == "model_facing_parity":
        if (
            not reference_sha256
            or not SHA256.fullmatch(reference_sha256)
            or evidence.get("previous_packet_sha256") != reference_sha256
        ):
            raise ValueError("Parity evidence lacks the recorded reference digest")
        if evidence.get("example_count") != sum(COUNTS.values()) or not evidence.get("compared"):
            raise ValueError("Parity evidence is incomplete")
    else:
        changed = evidence.get("changed_examples")
        reviews = evidence.get("reviews")
        if (
            not isinstance(changed, list)
            or not changed
            or len(changed) != len(set(changed))
            or not isinstance(reviews, dict)
            or set(reviews) != set(changed)
            or any(not isinstance(review, str) or not review.strip() for review in reviews.values())
        ):
            raise ValueError("Changed-example review evidence is incomplete")

    record = {
        "content_sha256": content_sha256,
        "source_commit": manifest["source_commit"],
        "basis": basis,
        "evidence_sha256": sha(evidence_path),
        "evidence_reference": evidence_path.name,
    }
    destination = packet / "acceptance.json"
    if destination.exists():
        if read_json(destination) != record:
            raise FileExistsError("Different acceptance record already exists")
        return record

    with tempfile.NamedTemporaryFile("w", dir=packet, delete=False) as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, destination)
    return record


def validate_labels(row: TokenRow, eos_id: int, newline_ids: list[int]) -> int:
    """Checks the cached assistant-only label span and post-EOS mask.

    Args:
        newline_ids: Expected unsupervised token IDs after the single EOS token.

    Returns:
        Index of the first supervised assistant token.

    Raises:
        ValueError: IDs or labels violate length, contiguity, EOS, or masking
            requirements.
    """
    ids, labels = row["input_ids"], row["labels"]
    if not ids or len(ids) != len(labels) or len(ids) > LIMIT:
        raise ValueError(f"Invalid chat length: {row['id']}")
    supervised = [index for index, label in enumerate(labels) if label != -100]
    if not supervised:
        raise ValueError(f"No assistant labels: {row['id']}")
    start, end = supervised[0], supervised[-1]
    if (
        start == 0
        or supervised != list(range(start, end + 1))
        or labels[start : end + 1] != ids[start : end + 1]
        or labels[end] != eos_id
        or ids[start : end + 1].count(eos_id) != 1
        or ids[end + 1 :] != newline_ids
    ):
        raise ValueError(f"Assistant boundary or post-EOS mask changed: {row['id']}")
    return start


def load_packet(packet: Path = PACKET) -> tuple[dict[str, list[TokenRow]], Any, dict[str, Any]]:
    """Loads an accepted packet using only its immutable payload.

    Returns:
        Validated token rows by split, the pinned tokenizer, and the manifest.
        Cached IDs and labels must reproduce the bundled messages exactly.

    Raises:
        ValueError: Integrity, acceptance, provenance, split isolation, target,
            token-boundary, or cached-label checks fail.
    """
    content_sha256 = verify_integrity(packet)
    verify_acceptance(packet, content_sha256)

    manifest = read_json(packet / "manifest.json")
    if (manifest.get("model"), manifest.get("revision")) != (MODEL, REVISION):
        raise ValueError("Model revision changed")
    prompt_path, schema_path = packet / "prompt.txt", packet / "schema.json"
    if sha(prompt_path) != manifest["prompt_sha256"] or sha(schema_path) != manifest["schema_sha256"]:
        raise ValueError("CA1 prompt or schema changed")

    tokenizer, provenance = tokenizer_and_hashes(packet / "tokenizer")
    if provenance != manifest["tokenizer_provenance"]:
        raise ValueError("Tokenizer provenance changed")
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Pinned tokenizer lacks pad or EOS token")

    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    schema = read_json(schema_path)
    prompt = prompt_path.read_text()
    examples = {item["id"]: item for item in manifest["examples"]}
    if len(examples) != len(manifest["examples"]) or len(examples) != sum(COUNTS.values()):
        raise ValueError("Manifest membership changed")
    token_rows = read_jsonl(packet / "tokenized.jsonl")
    tokens = {row["id"]: row for row in token_rows}
    if len(tokens) != len(token_rows):
        raise ValueError("Duplicate token IDs")

    splits: dict[str, list[TokenRow]] = {}
    families: dict[str, set[str]] = {}
    seen: set[str] = set()
    for split, count in COUNTS.items():
        messages = read_jsonl(packet / f"{split}.jsonl")
        if len(messages) != count:
            raise ValueError(f"Wrong {split} count")
        splits[split], families[split] = [], set()
        for item in messages:
            identifier = item["id"]
            if identifier in seen or identifier not in examples or identifier not in tokens:
                raise ValueError(f"Duplicate or unknown report ID: {identifier}")
            seen.add(identifier)
            example, row = examples[identifier], tokens[identifier]
            if example["split"] != split or row["split"] != split:
                raise ValueError(f"Split mismatch: {identifier}")
            families[split].add(example["family"])

            chat = item["messages"]
            if (
                [part["role"] for part in chat] != ["system", "user", "assistant"]
                or chat[0]["content"] != prompt
                or not chat[1]["content"].startswith(WRAPPER_START)
                or not chat[1]["content"].endswith(WRAPPER_END)
                or sha(chat[1]["content"][len(WRAPPER_START) : -len(WRAPPER_END)]) != example["input_sha256"]
            ):
                raise ValueError(f"Prompt, wrapper, or roles changed: {identifier}")
            if sha(chat[2]["content"]) != example["target_sha256"]:
                raise ValueError(f"Target changed: {identifier}")
            validate_target(chat[2]["content"], schema)

            prefix = validate_labels(row, tokenizer.eos_token_id, newline_ids)
            prefix_ids = tokenizer.apply_chat_template(chat[:2], tokenize=True, add_generation_prompt=True)
            if hasattr(prefix_ids, "keys"):
                prefix_ids = prefix_ids["input_ids"]
            if row["input_ids"][:prefix] != prefix_ids:
                raise ValueError(f"Cached chat-template boundary changed: {identifier}")
            expected_ids, expected_labels, _, _ = tokenize(tokenizer, chat)
            if row["input_ids"] != expected_ids or row["labels"] != expected_labels:
                raise ValueError(f"Cached conversation or assistant labels changed: {identifier}")
            splits[split].append(row)

    if seen != set(examples) or seen != set(tokens) or families["train"] & families["validation"]:
        raise ValueError("Split membership or family isolation changed")
    return splits, tokenizer, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("accept",))
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--basis", choices=sorted(BASES), required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--reference-sha256")
    args = parser.parse_args()
    accept(args.packet, args.basis, args.evidence, args.reference_sha256)


if __name__ == "__main__":
    main()
