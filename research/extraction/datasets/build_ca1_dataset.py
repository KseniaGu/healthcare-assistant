"""Build and verify a candidate CA1 dataset packet from retained private inputs."""

import argparse
import json
import random
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Literal, TypedDict

from research.extraction.datasets import packet
from research.extraction.datasets.synthetic import layouts, reports, types

TokenLengthField = Literal["input", "supervised", "total"]
TOKEN_LENGTH_FIELDS: tuple[TokenLengthField, ...] = ("input", "supervised", "total")

REALISM_MIN_MEDIAN_RATIO = 0.6
REALISM_MAX_OPENING_DIVISOR = 5


class TokenCount(TypedDict):
    split: reports.Split
    input: int
    supervised: int
    total: int


def source_revision() -> str:
    """Requires a clean, tracked generating source before recording HEAD.

    Returns:
        The source commit recorded in a new candidate manifest.

    Raises:
        ValueError: Any tracked repository file has uncommitted changes.
        subprocess.CalledProcessError: A required generating module is untracked.
    """
    root = packet.ROOT
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    changed = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True)
    if changed:
        raise ValueError("Commit the cleaned source before building an immutable packet")
    for name in (
        "datasets/packet.py",
        "datasets/synthetic/__init__.py",
        "datasets/synthetic/types.py",
        "datasets/synthetic/layouts.py",
        "datasets/synthetic/reports.py",
        "datasets/build_ca1_dataset.py",
    ):
        subprocess.check_call(
            ["git", "ls-files", "--error-unmatch", f"research/extraction/{name}"], cwd=root, stdout=subprocess.DEVNULL
        )
    return revision


def source_hashes_for(source_dir: Path, source_index: Path, tokenizer_dir: Path) -> dict[str, str]:
    """Verifies retained inputs and records the sources consumed by generation.

    Returns:
        Hashes of the reviewed correction inputs, smoke index, and pinned
        tokenizer files used to reproduce the candidate.

    Raises:
        ValueError: Source checksums, review status, or smoke-index checksum
            no longer match the retained records.
    """
    checksums = packet.read_json(source_dir / "SHA256SUMS.json")
    files = {
        str(path.relative_to(source_dir)): path
        for path in source_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    if set(files) != set(checksums) or any(packet.sha(files[name]) != digest for name, digest in checksums.items()):
        raise ValueError("Correction source checksum mismatch")

    inputs = [
        "prompts/corrected.txt",
        "schema.json",
        "synthetic_specs.json",
        "synthetic/corrected/train.jsonl",
        "synthetic/corrected/validation.jsonl",
        "real_edits.json",
    ]
    inputs.extend(f"real/{identifier}/input.txt" for identifier in sorted(reports.ELIGIBLE))
    review = packet.read_json(source_dir / "verification.json")
    if not str(review.get("status", "")).startswith("passed"):
        raise ValueError("Correction source review is not complete")

    smoke_checksums = packet.read_json(source_index.parent / "SHA256SUMS.json")
    if packet.sha(source_index) != smoke_checksums.get(source_index.name):
        raise ValueError("Source index checksum mismatch")

    source_hashes = {name: checksums[name] for name in inputs}
    for name in ("verification.json", "synthetic_review.md", "policy.md"):
        source_hashes[name] = checksums[name]
    source_hashes["correction/SHA256SUMS.json"] = packet.sha(source_dir / "SHA256SUMS.json")
    source_hashes["source_index/sources.json"] = packet.sha(source_index)
    source_hashes["source_index/SHA256SUMS.json"] = packet.sha(source_index.parent / "SHA256SUMS.json")
    _, provenance = packet.tokenizer_and_hashes(tokenizer_dir)
    source_hashes.update({f"tokenizer/{name}": digest for name, digest in provenance["files"].items()})
    source_hashes["tokenizer/provenance.json"] = packet.sha(tokenizer_dir / "provenance.json")
    return source_hashes


def _smoke_cases(
    source_dir: Path, smoke_specs: list[dict[str, Any]], prompt: str, schema: dict[str, Any]
) -> list[reports.DatasetCase]:
    smoke_rows = {
        split: [
            json.loads(line) for line in (source_dir / f"synthetic/corrected/{split}.jsonl").read_text().splitlines()
        ]
        for split in reports.BUCKETS
    }
    if {split: len(group) for split, group in smoke_rows.items()} != {"train": 16, "validation": 8}:
        raise ValueError("Frozen smoke counts changed")

    smoke_messages = {row["id"]: row["messages"] for group in smoke_rows.values() for row in group}
    items: list[reports.DatasetCase] = []
    for spec in smoke_specs:
        messages = smoke_messages[spec["id"]]
        report_text = messages[1]["content"].split("<report>\n", 1)[1].split("\n</report>", 1)[0]
        if messages[0]["content"] != prompt or packet.sha(report_text) != spec["input_sha256"]:
            raise ValueError("Frozen smoke message changed")
        packet.validate_target(messages[2]["content"], schema)
        items.append(
            dict(
                id=spec["id"],
                split=spec["split"],
                family=spec["family"],
                source_group=spec["source_group"],
                domain="smoke",
                bucket=reports.smoke_bucket(spec),
                clean="clean-control" in spec["tags"],
                smoke=True,
                tags=spec["tags"],
                transformations=spec["tags"],
                development_sources=spec["development_sources"],
                input=spec["input"],
                input_sha256=spec["input_sha256"],
                target_sha256=packet.sha(messages[2]["content"]),
                evidence=[],
                review_notes=spec["review_notes"],
                messages=messages,
                review_status="historical-smoke",
            )
        )
    return items


def _generated_cases(
    families: list[reports.FamilySpec],
    assignments: dict[tuple[str, int], tuple[reports.Bucket, bool]],
    prompt: str,
    schema: dict[str, Any],
) -> list[reports.DatasetCase]:
    # One RNG spans every generated family; reseeding per family changes the packet.
    rng = random.Random(reports.SEED)
    items: list[reports.DatasetCase] = []
    for family in families:
        for variant in range(family["count"]):
            bucket, clean = assignments[family["family"], variant]
            rows, transformations = reports.rows_for(family, variant, rng, bucket, clean)
            dates, date_tag = reports.dates_for(family, variant, bucket)
            text, spans = reports.render(family, variant, rows, dates)
            target, evidence = reports.target_from_visible(rows, spans, dates)
            reports.check_pair(text, rows, target, evidence, schema)
            serialized = json.dumps(target, ensure_ascii=False, indent=2)
            messages = [
                packet.ChatMessage(role="system", content=prompt),
                packet.ChatMessage(role="user", content=packet.WRAPPER_START + text + packet.WRAPPER_END),
                packet.ChatMessage(role="assistant", content=serialized),
            ]
            items.append(
                dict(
                    id=f"{family['family']}-{variant + 1:02d}",
                    split=family["split"],
                    family=family["family"],
                    source_group=family["family"],
                    domain=family["domain"],
                    bucket=bucket,
                    clean=clean,
                    smoke=False,
                    tags=[date_tag],
                    transformations=transformations,
                    development_sources=family["development_sources"],
                    input=text,
                    input_sha256=packet.sha(text),
                    target_sha256=packet.sha(serialized),
                    evidence=evidence,
                    messages=messages,
                    review_status="pending-assistant-review",
                )
            )
    return items


def create_cases(
    source_dir: Path, source_index: Path, prompt: str, schema: dict[str, Any]
) -> tuple[list[reports.DatasetCase], dict[str, str]]:
    """Combines frozen smoke pairs with deterministic fictitious cases.

    Returns:
        Cases in packet order and the family-to-split map used for isolation checks.

    Raises:
        ValueError: Source IDs, split quotas, source provenance, or family
            isolation violate the CA1 dataset contract.
    """
    source_rows = packet.read_json(source_index)
    sources = {item["id"]: item for item in source_rows}
    if len(sources) != len(source_rows):
        raise ValueError("Duplicate source index ID")

    smoke_specs = packet.read_json(source_dir / "synthetic_specs.json")
    families = reports.catalog()
    assignments = reports.assignments(smoke_specs, families)
    items = _smoke_cases(source_dir, smoke_specs, prompt, schema)
    items.extend(_generated_cases(families, assignments, prompt, schema))

    split_families = reports.ensure_split_and_provenance(items, sources)
    if Counter(item["split"] for item in items) != packet.COUNTS:
        raise ValueError("Report counts changed")
    for split, quotas in reports.BUCKETS.items():
        if Counter(item["bucket"] for item in items if item["split"] == split) != quotas:
            raise ValueError("Bucket counts changed")
    return items, split_families


def statistics_for(
    items: list[reports.DatasetCase], split_families: dict[str, str], token_counts: list[TokenCount]
) -> dict[str, Any]:
    """Summarizes exact split, coverage, duplicate, and token statistics."""
    statistics = {
        "counts": dict(Counter(item["split"] for item in items)),
        "buckets": {
            split: dict(Counter(item["bucket"] for item in items if item["split"] == split))
            for split in reports.BUCKETS
        },
        "domains": dict(Counter(item["domain"] for item in items)),
        "families": {split: sum(value == split for value in split_families.values()) for split in reports.BUCKETS},
        "clean": {split: sum(item["clean"] for item in items if item["split"] == split) for split in reports.BUCKETS},
        "transformations": dict(Counter(tag for item in items for tag in item["transformations"])),
        "tokens": {
            split: {
                field: reports.quantiles([row[field] for row in token_counts if row["split"] == split])
                for field in TOKEN_LENGTH_FIELDS
            }
            for split in reports.BUCKETS
        },
        "max_total": max(row["total"] for row in token_counts),
        "duplicate_inputs": 0,
        "split_leakage": 0,
        "near_duplicate_candidates": reports.near_duplicate_candidates(items),
    }
    return statistics


def write_candidate(
    output: Path,
    source_dir: Path,
    tokenizer_dir: Path,
    items: list[reports.DatasetCase],
    token_rows: list[packet.TokenRow],
    manifest: dict[str, Any],
    statistics: dict[str, Any],
    distribution: dict[str, Any],
) -> None:
    """Writes the complete immutable candidate payload and its canonical checksums."""
    output.mkdir(parents=True)
    shutil.copy2(source_dir / "prompts/corrected.txt", output / "prompt.txt")
    shutil.copy2(source_dir / "schema.json", output / "schema.json")
    shutil.copytree(tokenizer_dir, output / "tokenizer", ignore=shutil.ignore_patterns(".lock"))

    for split in reports.BUCKETS:
        rows = (
            json.dumps({"id": item["id"], "messages": item["messages"]}, ensure_ascii=False) + "\n"
            for item in items
            if item["split"] == split
        )
        (output / f"{split}.jsonl").write_text("".join(rows))
    (output / "tokenized.jsonl").write_text("".join(json.dumps(row) + "\n" for row in token_rows))

    packet.dump(output / "manifest.json", manifest)
    packet.dump(output / "statistics.json", statistics)
    packet.dump(output / "distribution.json", distribution)
    packet.write_checksums(output)


def _tokenize_cases(items: list[reports.DatasetCase], tokenizer: Any) -> tuple[list[packet.TokenRow], list[TokenCount]]:
    token_rows: list[packet.TokenRow] = []
    token_counts: list[TokenCount] = []
    for item in items:
        ids, labels, prefix, supervised = packet.tokenize(tokenizer, item["messages"])
        if len(ids) > packet.LIMIT:
            raise ValueError(f"8192-token limit exceeded: {item['id']} ({len(ids)})")
        token_rows.append(packet.TokenRow(id=item["id"], split=item["split"], input_ids=ids, labels=labels))
        token_counts.append(TokenCount(split=item["split"], input=prefix, supervised=supervised, total=len(ids)))
    return token_rows, token_counts


def build(output: Path, source_dir: Path, source_index: Path, tokenizer_dir: Path) -> None:
    """Writes one immutable candidate from reviewed inputs and clean source.

    Acceptance is a separate step; this function does not authorize training.

    Args:
        output: New directory; an existing candidate is never overwritten.
        source_dir: Retained, checksummed correction source, including frozen smoke pairs.
        source_index: Reviewed source-role index used for provenance checks.
        tokenizer_dir: Local pinned tokenizer with its provenance record.

    Raises:
        FileExistsError: The output directory already exists.
        ValueError: Source, tokenizer, generation, or packet invariants fail.

    """
    if output.exists():
        raise FileExistsError(f"Packet exists: {output}")

    source_commit = source_revision()
    source_hashes = source_hashes_for(source_dir, source_index, tokenizer_dir)
    prompt = (source_dir / "prompts/corrected.txt").read_text()
    schema = packet.read_json(source_dir / "schema.json")
    items, split_families = create_cases(source_dir, source_index, prompt, schema)

    original_texts = [
        (source_dir / "real" / identifier / "input.txt").read_text() for identifier in sorted(reports.ELIGIBLE)
    ]
    new_train = [item["input"] for item in items if item["split"] == "train" and not item["smoke"]]
    new_validation = [item["input"] for item in items if item["split"] == "validation" and not item["smoke"]]
    original_profile = reports.profile(original_texts)
    train_profile = reports.profile(new_train)
    validation_profile = reports.profile(new_validation)
    short_characters = (
        train_profile["characters"]["median"] < original_profile["characters"]["median"] * REALISM_MIN_MEDIAN_RATIO
    )
    short_lines = train_profile["lines"]["median"] < original_profile["lines"]["median"] * REALISM_MIN_MEDIAN_RATIO
    repeated_openings = train_profile["max_opening_count"] > len(new_train) // REALISM_MAX_OPENING_DIVISOR
    if short_characters or short_lines or repeated_openings:
        raise ValueError("Synthetic input realism gate failed")

    tokenizer, provenance = packet.tokenizer_and_hashes(tokenizer_dir)
    # Generation records package versions; payload verification does not need these optional imports.
    import tokenizers as tokenizers_package
    import transformers

    token_rows, token_counts = _tokenize_cases(items, tokenizer)
    manifest = {
        "seed": reports.SEED,
        "model": packet.MODEL,
        "revision": packet.REVISION,
        "source_commit": source_commit,
        "source_hashes": source_hashes,
        "prompt_sha256": packet.sha(source_dir / "prompts/corrected.txt"),
        "schema_sha256": packet.sha(source_dir / "schema.json"),
        "builder_sha256": packet.sha(Path(__file__)),
        "synthetic_reports_sha256": packet.sha(Path(reports.__file__)),
        "synthetic_layouts_sha256": packet.sha(Path(layouts.__file__)),
        "synthetic_types_sha256": packet.sha(Path(types.__file__)),
        "tokenizer_provenance": provenance,
        "smoke_sha256": {
            name: packet.sha(source_dir / name)
            for name in (
                "synthetic_specs.json",
                "synthetic/corrected/train.jsonl",
                "synthetic/corrected/validation.jsonl",
            )
        },
        "versions": {"transformers": transformers.__version__, "tokenizers": tokenizers_package.__version__},
        "examples": [{key: value for key, value in item.items() if key not in ("input", "messages")} for item in items],
    }
    statistics = statistics_for(items, split_families, token_counts)
    distribution = {
        "original": original_profile,
        "new_training": train_profile,
        "new_validation": validation_profile,
        "below_original_min_characters": {
            "train": sum(len(text) < original_profile["characters"]["min"] for text in new_train),
            "validation": sum(len(text) < original_profile["characters"]["min"] for text in new_validation),
        },
        "below_original_min_lines": {
            "train": sum(len(text.splitlines()) < original_profile["lines"]["min"] for text in new_train),
            "validation": sum(len(text.splitlines()) < original_profile["lines"]["min"] for text in new_validation),
        },
    }

    write_candidate(output, source_dir, tokenizer_dir, items, token_rows, manifest, statistics, distribution)
    print(json.dumps(dict(statistics=statistics, distribution=distribution), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--source-index", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(f"Verified candidate payload: {packet.verify_integrity(args.output)}")
    else:
        if not all((args.source_dir, args.source_index, args.tokenizer_dir)):
            parser.error("generation requires --source-dir, --source-index, and --tokenizer-dir")
        build(args.output, args.source_dir, args.source_index, args.tokenizer_dir)


if __name__ == "__main__":
    main()
