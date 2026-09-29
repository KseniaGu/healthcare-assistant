"""Generate and verify the local CA1 laboratory SFT pilot packet."""

import argparse
import difflib
import hashlib
import json
import random
import re
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from statistics import median

from research.extraction.evaluate_image_extraction import strict_validate
from research.extraction.prepare_sft import MODEL, REVISION, completion_tokens

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/research/extraction"
CORRECTION = BASE / "sft-correction-v1"
SMOKE = BASE / "sft-smoke-v1"
OUTPUT = BASE / "sft-pilot-v1"
SEED = 20260928
STATUS = "assistant-reviewed; awaiting user acceptance"
ELIGIBLE = frozenset("A1 A2 A3 A4 C1 D0002 D0006 D0007 D0011 D0012 D0014 D0015 D0016".split())
BUCKETS = {"train": {"ordinary": 192, "layout": 80, "difficult": 48},
           "validation": {"ordinary": 38, "layout": 16, "difficult": 10}}
DOMAINS = ("chemistry", "hematology", "urine", "qualitative")

# These are fictional panels, not copied report vectors. Each tuple is name, unit,
# direct reference text; measurements are generated independently below.
PANELS = {
    "chemistry": [
        ("Глюкоза", "ммоль/л", "3,1–6,2"), ("Креатинин", "мкмоль/л", "45–101"),
        ("Альбумин", "г/л", "33–52"), ("Фосфор", "ммоль/л", "0,6–1,8"),
        ("Мочевина", "ммоль/л", "2,1–8,3"), ("Билирубин общий", "мкмоль/л", "<22"),
        ("Кальций", "ммоль/л", "2,0–2,7"), ("Общий белок", "г/л", "58–83"),
        ("Железо", "мкмоль/л", "8–32"), ("Триглицериды", "ммоль/л", "<2,0"),
        ("АЛТ", "Ед/л", "<45"), ("АСТ", "Ед/л", "<41"),
        ("Холестерин общий", "ммоль/л", "<5,0"), ("Магний", "ммоль/л", "0,6–1,1"),
        ("Натрий", "ммоль/л", "135–148"), ("Калий", "ммоль/л", "3,5–5,4")],
    "hematology": [
        ("Гемоглобин", "г/л", "109–169"), ("Лейкоциты", "10^9/л", "3,1–11,2"),
        ("Эритроциты", "10^12/л", "3,5–5,9"), ("Гематокрит", "%", "33–53"),
        ("Тромбоциты", "10^9/л", "125–410"), ("MCV", "фл", "78–102"),
        ("MCH", "пг", "24–36"), ("Нейтрофилы, %", "%", "39–76"),
        ("Лимфоциты, %", "%", "17–49"), ("СОЭ", "мм/час", "1–24"),
        ("Базофилы, %", "%", "0–2"), ("Эозинофилы, %", "%", "0–6"),
        ("Моноциты, %", "%", "2–13"), ("RDW", "%", "11–16"),
        ("MPV", "фл", "7–12"), ("Ретикулоциты, %", "%", "0,2–2,0")],
    "urine": [
        ("pH мочи", None, "4,5–8,0"), ("Удельный вес", None, "1,005–1,030"),
        ("Белок мочи", "г/л", "<0,15"), ("Глюкоза мочи", "ммоль/л", "отрицательно"),
        ("Эритроциты в поле зрения", "#", "0–3"), ("Лейкоциты в поле зрения", "#", "0–5"),
        ("Кетоны", None, "отрицательно"), ("Нитриты", None, "отрицательно"),
        ("Эпителий плоский", "#", "0–6"), ("Билирубин мочи", None, "отрицательно"),
        ("Уробилиноген", "мкмоль/л", "0–17"), ("Бактерии в поле зрения", "#", "0–1"),
        ("Цилиндры гиалиновые", "#", "0–2"), ("Кристаллы", "#", "0–3"),
        ("Дрожжевые клетки", "#", "0–1"), ("Клетки переходного эпителия", "#", "0–3")],
    "qualitative": [
        ("Антиген А", None, "отрицательно"), ("Антитела к маркеру B", None, "отрицательно"),
        ("ПЦР маркер C", None, "не обнаружено"), ("IgM маркер D", None, "отрицательно"),
        ("IgG маркер E", "Ед/мл", "<12"), ("Индекс маркера F", None, "<1,0"),
        ("Антиген G", None, "отрицательно"), ("ПЦР маркер H", None, "не обнаружено"),
        ("Реакция I", None, "отрицательно"), ("Антитела J", None, "отрицательно"),
        ("Антитела K", None, "отрицательно"), ("Антиген L (AgL)", None, "отрицательно"),
        ("ПЦР маркер M (M1)", None, "не обнаружено"), ("IgM маркер N", None, "отрицательно"),
        ("IgG маркер O", "Ед/мл", "<15"), ("Индекс маркера P", None, "<1,2")],
}
SOURCE_HINTS = {
    "chemistry": ("A1", "A2", "C1", "D0014"),
    "hematology": ("A3", "D0007", "D0011", "D0016"),
    "urine": ("A4", "D0012", "D0014", "D0016"),
    "qualitative": ("D0002", "D0006", "D0015", "C1"),
}


def sha(data):
    if isinstance(data, Path):
        data = data.read_bytes()
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load(path):
    return json.loads(path.read_text())


def family_catalog():
    """Fix source groups, layouts and splits before drawing any variants."""
    families = []
    for domain in DOMAINS:
        for style in range(6):
            index = len(families)
            families.append(dict(family=f"t-{domain}-{style+1:02d}", split="train",
                                 domain=domain, style=style, count=13 if index < 16 else 12,
                                 development_sources=[SOURCE_HINTS[domain][style % 4]]))
    for domain in DOMAINS:
        for style in range(2):
            families.append(dict(family=f"v-{domain}-{style+1:02d}", split="validation",
                                 domain=domain, style=6 + DOMAINS.index(domain) * 2 + style,
                                 count=7, development_sources=[]))
    return families


def smoke_bucket(item):
    tags = set(item["tags"])
    if tags & {"missing-name", "missing-value", "missing-date", "ambiguous-unit-abstention"}:
        return "difficult"
    if tags & {"paired-columns", "attached-panel", "long-panel", "wrapped-name", "history-exclusion"}:
        return "layout"
    return "ordinary"


def assignments(specs, catalog):
    """Meet the three exact bucket counts after accounting for frozen smoke IDs."""
    remaining = {split: Counter(quotas) for split, quotas in BUCKETS.items()}
    for item in specs:
        remaining[item["split"]][smoke_bucket(item)] -= 1
    slots = [(f, n) for f in catalog for n in range(f["count"])]
    rng = random.Random(SEED)
    rng.shuffle(slots)
    output = {}
    for split in BUCKETS:
        own = [(f, n) for f, n in slots if f["split"] == split]
        bucket_list = [b for b in ("ordinary", "layout", "difficult") for _ in range(remaining[split][b])]
        assert len(own) == len(bucket_list) and all(v >= 0 for v in remaining[split].values())
        rng.shuffle(bucket_list)
        clean_needed = 48 if split == "train" else 12
        clean_assigned = 0
        for (f, n), bucket in zip(own, bucket_list, strict=True):
            clean = bucket == "ordinary" and clean_assigned < clean_needed
            clean_assigned += clean
            output[f["family"], n] = bucket, clean
        assert clean_assigned == clean_needed
    return output


def value_for(domain, name, reference, rng, index):
    """Draw a fictional measurement near its own reference scale."""
    if name in ("Кетоны", "Нитриты", "Билирубин мочи") or (domain == "qualitative" and not re.search(r"\d", reference)):
        return ("Отрицательный", "Положительный", "не обнаружено", "+", "−")[index % 5]
    if name == "Удельный вес":
        return f"1,{rng.randrange(6, 31):03d}"
    if name in ("Эритроциты в поле зрения", "Лейкоциты в поле зрения", "Эпителий плоский",
                "Бактерии в поле зрения", "Цилиндры гиалиновые", "Кристаллы",
                "Дрожжевые клетки", "Клетки переходного эпителия"):
        n = rng.randrange(0, 9)
        return f"{n}–{n+2}" if index % 4 == 0 else str(n)
    numbers = [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", reference)]
    if not numbers:
        return ("Отрицательный", "Положительный", "не обнаружено", "+", "−")[index % 5]
    low, high = (numbers[0], numbers[-1]) if len(numbers) > 1 else (numbers[0] * 0.15, numbers[0] * 1.3)
    if name == "pH мочи":
        return f"{rng.uniform(4.8, 7.8):.1f}".replace(".", ",")
    scale = max((high - low) / 5, 0.05)
    measured = min(high + scale, max(low / 2, rng.uniform(low - scale, high + scale)))
    if index % 11 == 0 and reference.startswith("<"):
        places = len(reference.split(",", 1)[1]) if "," in reference else 0
        limit = f"{numbers[0] / 2:.{places}f}"
        return "<" + (limit.replace(".", ",") if places else limit)
    decimals = 0 if high >= 50 and "," not in reference else 2 if index % 3 == 0 else 1
    rendered = f"{measured:.{decimals}f}"
    return rendered.replace(".", ",") if index % 2 else rendered


def raw_rows(family, variant, rng, bucket, clean):
    domain = family["domain"]
    panel = PANELS[domain]
    count = (3, 5, 7, 9, 16)[variant % 5]
    start = (variant * 3 + family["style"]) % len(panel)
    selected = [panel[(start + j) % len(panel)] for j in range(count)]
    rows = []
    for j, (name, unit, ref) in enumerate(selected):
        rows.append(dict(name=name, value=value_for(domain, name, ref, rng, variant + j),
                         unit=unit or "—", reference_range=ref, state="current"))
    if bucket != "ordinary" and variant % 2 == 0:
        rows.insert(1, dict(name="Контроль предыдущего исследования", value="8,8", unit="—",
                            reference_range="—", state="history"))
    if bucket != "ordinary" and variant % 3 == 0:
        rows.append(dict(name="Повторное исследование", value="ожидается", unit="—",
                         reference_range="—", state="pending"))
    transformations = []
    if not clean:
        candidates = [r for r in rows if r["state"] == "current"]
        if bucket == "difficult":
            row = candidates[variant % len(candidates)]
            mode = variant % 4
            if mode == 0:
                row["name"] = "—"
                transformations.append("unnamed-row")
            elif mode == 1:
                row["value"] = "—"
                transformations.append("missing-value")
            elif mode == 2:
                row["unit"] = "10°/π"
                transformations.append("ambiguous-unit")
            else:
                row["reference_range"] = "—"
                transformations.append("missing-reference")
        if bucket != "ordinary" and variant % 3 == 1:
            row = candidates[-1]
            row["value"] += " *" if row["value"] not in ("+", "−", "—") else ""
            transformations.append("separate-flag")
        if variant % 4 == 1:
            row = candidates[0]
            if row["unit"] == "г/л":
                row["unit"] = "г/π"
                transformations.append("unit-glyph-repair")
            elif row["unit"] == "фл":
                row["unit"] = "φπ"
                transformations.append("unit-glyph-repair")
        if variant % 4 == 2:
            row = candidates[0]
            if "Глюкоза" in row["name"]:
                row["name"] = row["name"].replace("Глюкоза", "Глюкoза")
                transformations.append("name-glyph-repair")
            elif row["value"] == "Отрицательный":
                row["value"] = "Отрищательный"
                transformations.append("qualitative-repair")
    return rows, sorted(set(transformations))


REPAIRS = {"Глюкoза": "Глюкоза", "Отрищательный": "Отрицательный",
           "г/π": "г/л", "φπ": "фл"}


def visible_value(field, raw):
    """Interpret only a rendered cell; hidden source-record fields are unavailable."""
    if raw is None or raw.strip() in ("", "—", "–", "-"):
        return None
    raw = raw.strip()
    if field == "unit" and raw == "10°/π":
        return None
    if field == "value" and raw.endswith(" *"):
        raw = raw[:-2]
    for before, after in REPAIRS.items():
        raw = raw.replace(before, after)
    return raw


def date_lines(variant, bucket):
    day = date(2023, 1, 1) + timedelta(days=variant * 17 + 80)
    fmt = lambda d: d.strftime("%d.%m.%Y")
    lines = [f"Дата печати: {fmt(day + timedelta(days=3))}"]
    if bucket == "difficult" and variant % 7 == 0:
        return lines, None, "print-only-date"
    lines.append(f"Дата направления: {fmt(day)}")
    if variant % 3 != 0:
        lines.append(f"Дата выполнения: {fmt(day + timedelta(days=1))}")
    if variant % 2 == 0:
        lines.append(f"Дата взятия образца: {fmt(day - timedelta(days=1))}")
    return lines, None, "date-precedence"


def select_date(lines):
    for label in ("Дата взятия образца", "Дата выполнения", "Дата направления"):
        for line in lines:
            if line.startswith(label + ": "):
                return date.fromisoformat("-".join(reversed(line.split(": ", 1)[1].split(".")))).isoformat()
    return None


def format_row(row, style, index):
    n, v, u, r = (row[k] for k in ("name", "value", "unit", "reference_range"))
    if style == 0:
        return f"[{index}] {n}\n    Результат {v} | Ед. {u} | Норма {r}"
    if style == 1:
        return f"{n} :: {v}\n    единица={u}; референс={r}"
    if style == 2:
        return f"{index:02d}  {n:<32} {v:>13}  {u:<12}  {r}"
    if style == 3:
        return f"<анализ> {n} </анализ>\n<результат> {v} </результат> <ед> {u} </ед> <интервал> {r} </интервал>"
    if style == 4:
        return f"{n}\n  значение: {v}\n  измерение: {u}\n  предел: {r}"
    if style == 5:
        return f"{index}) {n}  /  {v}  /  {u}  /  {r}"
    if style == 6:
        return f"{n}; показатель={v}; ед={u}; границы={r}"
    if style == 7:
        return f"Реф. {r} ← {n} → {v} [{u}]"
    if style == 8:
        return f"{n}\n    {v}\n    ({u}; {r})"
    if style == 9:
        return f"Итог {index}: {v} / {n}\nНорматив {r}; размерность {u}"
    if style == 10:
        return f"{index}|{n}|{r}|{u}|{v}"
    if style == 11:
        return f"{n} ....... {v}\n    {r} ....... {u}"
    if style == 12:
        return f"Название={n}\nДанные={v}; Единицы={u}; Справочно={r}"
    return f"({index}) {n} => {v} ; [{r}] ; ед: {u}"


def render(family, variant, rows, dates):
    """Return parser text and cell spans from the final visible representation."""
    style = family["style"]
    label = {"chemistry": "Биохимическое исследование", "hematology": "Клеточный состав",
             "urine": "Исследование мочи", "qualitative": "Маркеры и реакции"}[family["domain"]]
    title = f"ЛАБОРАТОРНЫЙ БЛАНК / {label}"
    accession = 100000 + (int(sha(family["family"])[:8], 16) + variant * 7919) % 900000
    parts = [title, f"Номер бланка: {accession}", *dates,
             "Исследование завершено", "Результаты:" ]
    spans = []
    section = 0
    for i, row in enumerate(rows, 1):
        if i == 1 or (i > 1 and (i + variant) % 4 == 0):
            section += 1
            parts.append(f"Раздел {section}: {label}")
        line = format_row(row, style, i)
        if row["state"] == "history":
            line = "Архивное сравнение (не текущий результат): " + line
        if row["state"] == "pending":
            line = "В процессе / ожидается: " + line
        offset = sum(len(p) + 1 for p in parts)
        parts.append(line)
        cell_spans = {}
        cursor = 0
        for field in ("name", "value", "unit", "reference_range"):
            raw = row[field]
            pos = line.find(raw, cursor)
            if pos < 0:
                pos = line.find(raw)
            assert pos >= 0
            cell_spans[field] = dict(start=offset + pos, end=offset + pos + len(raw), text=raw)
            cursor = pos + len(raw)
        spans.append(cell_spans)
        if i == len(rows) // 2 and variant % 3 == 0:
            parts.append("Продолжение результатов на следующем листе")
    if variant % 4 == 0:
        parts.append("Комментарий: ориентировочные границы зависят от метода; использовать прямую графу нормы.")
    text = "\n".join(parts)
    for row_spans in spans:
        for span in row_spans.values():
            assert text[span["start"]:span["end"]] == span["text"]
    return text, spans


def target_from_visible(text, rows, spans, dates):
    result, evidence, unnamed, incomplete_unnamed, missing = [], [], [], 0, []
    for row, cell_spans in zip(rows, spans, strict=True):
        if row["state"] != "current":
            continue
        cells = {field: visible_value(field, row[field]) for field in
                 ("name", "value", "unit", "reference_range")}
        if cells["name"] is None:
            unnamed.append(row["value"])
            incomplete_unnamed += visible_value("reference_range", row["reference_range"]) is None
            evidence.append(dict(kind="omission", reason="Visible name absent", spans=cell_spans))
            continue
        if cells["value"] is None:
            missing.append(cells["name"])
        result.append(cells)
        for field, before in row.items():
            if field == "state" or cells[field] == before:
                continue
            rule = 2 if field == "name" else 3 if field == "unit" else 4 if field == "value" else 5
            reason = ("Visible cell is absent or ambiguous" if cells[field] is None else
                      "Identifiable mixed-script name glyph" if field == "name" else
                      "Surviving unit glyphs identify the unit" if field == "unit" else
                      "Separately printed abnormality flag" if before.endswith(" *") else
                      "Identifiable qualitative spelling with polarity preserved")
            evidence.append(dict(kind="repair" if cells[field] is not None else "abstention",
                                 result_index=len(result) - 1, field=field, before=before,
                                 after=cells[field], rule=rule, reason=reason,
                                 span=cell_spans[field]))
    warnings = []
    if unnamed:
        quote = ", ".join(f"'{value}'" for value in unnamed[:3])
        detail = f" {incomplete_unnamed} reference cell(s) incomplete." if incomplete_unnamed else ""
        warnings.append(f"Omitted {len(unnamed)} unnamed row(s); visible value(s): {quote}.{detail}")
    if missing:
        quote = ", ".join(f"'{name}'" for name in missing[:2])
        warnings.append(f"Missing result value for {len(missing)} named observation(s): {quote}.")
    chosen_date = select_date(dates)
    if chosen_date is None:
        warnings.append("No report collection, performance or order date is visible.")
    target = dict(date=chosen_date, results=result, errors=warnings)
    assert all(len(w) <= 160 for w in warnings)
    return target, evidence


def check_pair(text, rows, spans, target, evidence, schema):
    """Check the target against final visible cells and source spans."""
    strict_validate(json.dumps(target, ensure_ascii=False), schema)
    expected_count = sum(r["state"] == "current" and visible_value("name", r["name"]) is not None for r in rows)
    if len(target["results"]) != expected_count:
        raise ValueError("Visible observation coverage mismatch")
    for item in evidence:
        source_spans = item.get("spans", {}).values() if item["kind"] == "omission" else [item["span"]]
        if any(text[s["start"]:s["end"]] != s["text"] for s in source_spans):
            raise ValueError("Evidence span mismatch")
    current = [r for r in rows if r["state"] == "current" and visible_value("name", r["name"]) is not None]
    for source, result in zip(current, target["results"], strict=True):
        for field in ("name", "value", "unit", "reference_range"):
            before, after = source[field], result[field]
            if after != visible_value(field, before):
                raise ValueError("Target differs from visible-cell CA1 reading")
            if field in ("value", "reference_range") and after is not None:
                numbers = lambda s: re.findall(r"\d+(?:[.,]\d+)*|[<>≤≥+−]", s)
                if numbers(before.removesuffix(" *")) != numbers(after):
                    raise ValueError("Measured number, bound or polarity changed")
    if any(re.search(r"[А-Яа-я]", re.sub(r"'[^']*'", "", warning)) for warning in target["errors"]):
        raise ValueError("Non-English warning text")


def ensure_split_and_provenance(items, sources):
    families, groups, ids, inputs, normalized = {}, {}, set(), set(), set()
    for item in items:
        if item["id"] in ids:
            raise ValueError("Duplicate ID")
        ids.add(item["id"])
        for collection, key in ((families, "family"), (groups, "source_group")):
            old = collection.setdefault(item[key], item["split"])
            if old != item["split"]:
                raise ValueError("Family/source-group split leakage")
        for source in item["development_sources"]:
            if source not in ELIGIBLE or sources[source]["role"] != "development":
                raise ValueError("Excluded or reserved provenance")
        digest = item["input_sha256"]
        compact = sha("".join(item["input"].split()))
        if digest in inputs or compact in normalized:
            raise ValueError("Duplicate input")
        inputs.add(digest)
        normalized.add(compact)
    return families


def near_duplicate_candidates(items):
    """Screen representative parser layouts across splits for manual adjudication."""
    representatives = {}
    for item in items:
        representatives.setdefault(item["family"], item)
    def skeleton(item):
        lines = []
        for line in item["input"].splitlines():
            if line.startswith(("Дата ", "Номер бланка", "Раздел ", "ЛАБОРАТОРНЫЙ")):
                continue
            if line in ("Результаты:", "Исследование завершено"):
                continue
            line = re.sub(r"[\wА-Яа-я]+", "X", line)
            lines.append(line)
        return "\n".join(lines[:22])
    train = [item for item in representatives.values() if item["split"] == "train"]
    validation = [item for item in representatives.values() if item["split"] == "validation"]
    candidates = []
    for left in train:
        for right in validation:
            score = difflib.SequenceMatcher(None, skeleton(left), skeleton(right)).ratio()
            if score >= 0.85:
                candidates.append(dict(train_family=left["family"], validation_family=right["family"],
                                       structural_similarity=round(score, 3)))
    return sorted(candidates, key=lambda x: -x["structural_similarity"])


def tokenizer_and_hashes():
    from transformers import AutoTokenizer
    provenance = load(SMOKE / "tokenizer/provenance.json")
    if (provenance["model"], provenance["revision"]) != (MODEL, REVISION):
        raise ValueError("Tokenizer identity changed")
    for name, expected in provenance["files"].items():
        if sha(SMOKE / "tokenizer" / name) != expected:
            raise ValueError("Tokenizer file changed")
    tokenizer = AutoTokenizer.from_pretrained(str(SMOKE / "tokenizer"), local_files_only=True)
    if sha(tokenizer.chat_template) != provenance["chat_template_sha256"]:
        raise ValueError("Chat template changed")
    return tokenizer, provenance


def tokenize(tokenizer, messages):
    ids, labels, prefix = completion_tokens(tokenizer, messages)
    completion = ids[prefix:]
    if completion.count(tokenizer.eos_token_id) != 1:
        raise ValueError("Expected one assistant end token")
    end = prefix + completion.index(tokenizer.eos_token_id)
    labels[end + 1:] = [-100] * (len(ids) - end - 1)
    if (labels[:prefix] != [-100] * prefix or labels[prefix:end + 1] != ids[prefix:end + 1]
            or any(x != -100 for x in labels[end + 1:])):
        raise ValueError("Assistant-only labels invalid")
    return ids, labels, prefix, end - prefix + 1


def quantiles(values):
    ordered = sorted(values)
    return {"p50": median(ordered), "p95": ordered[(len(ordered) * 95 + 99) // 100 - 1], "max": ordered[-1]}


def build(output=OUTPUT):
    if output.exists():
        raise FileExistsError(f"Packet exists: {output}; preserve reviewed artifacts")
    prompt = (CORRECTION / "prompts/corrected.txt").read_text()
    schema = load(CORRECTION / "schema.json")
    sources = {x["id"]: x for x in load(SMOKE / "sources.json")}
    smoke_specs = load(CORRECTION / "synthetic_specs.json")
    smoke_rows = {split: [json.loads(line) for line in
                          (CORRECTION / f"synthetic/corrected/{split}.jsonl").read_text().splitlines()]
                  for split in BUCKETS}
    if {split: len(rows) for split, rows in smoke_rows.items()} != {"train": 16, "validation": 8}:
        raise ValueError("Smoke counts changed")
    smoke_messages = {row["id"]: row["messages"] for rows in smoke_rows.values() for row in rows}
    catalog = family_catalog()
    assigned = assignments(smoke_specs, catalog)
    rng = random.Random(SEED)
    items = []
    for spec in smoke_specs:
        messages = smoke_messages[spec["id"]]
        if (messages[0]["content"] != prompt or
                sha(messages[1]["content"].split("<report>\n", 1)[1].split("\n</report>", 1)[0]) != spec["input_sha256"]):
            raise ValueError("Smoke prompt or input changed")
        target = json.loads(messages[2]["content"])
        strict_validate(messages[2]["content"], schema)
        items.append(dict(id=spec["id"], split=spec["split"], family=spec["family"],
                          source_group=spec["source_group"], domain="smoke", bucket=smoke_bucket(spec),
                          clean="clean-control" in spec["tags"], smoke=True, tags=spec["tags"],
                          transformations=spec["tags"], development_sources=spec["development_sources"],
                          input=spec["input"], input_sha256=spec["input_sha256"],
                          target_sha256=sha(messages[2]["content"]), evidence=[], review_notes=spec["review_notes"],
                          messages=messages, review_status=STATUS))
    for family in catalog:
        for variant in range(family["count"]):
            bucket, clean = assigned[family["family"], variant]
            rows, transformations = raw_rows(family, variant, rng, bucket, clean)
            dates, _, date_tag = date_lines(variant + int(sha(family["family"])[:4], 16) % 32, bucket)
            text, spans = render(family, variant, rows, dates)
            target, evidence = target_from_visible(text, rows, spans, dates)
            check_pair(text, rows, spans, target, evidence, schema)
            serialized = json.dumps(target, ensure_ascii=False, indent=2)
            messages = [{"role": "system", "content": prompt},
                        {"role": "user", "content": "Extract laboratory observations from this parser output:\n<report>\n" + text + "\n</report>\n"},
                        {"role": "assistant", "content": serialized}]
            identifier = f"{family['family']}-{variant+1:02d}"
            items.append(dict(id=identifier, split=family["split"], family=family["family"],
                              source_group=family["family"], domain=family["domain"], bucket=bucket,
                              clean=clean, smoke=False, tags=[date_tag], transformations=transformations,
                              development_sources=family["development_sources"], input=text,
                              input_sha256=sha(text), target_sha256=sha(serialized), evidence=evidence,
                              messages=messages, review_status="pending-assistant-review"))
    families = ensure_split_and_provenance(items, sources)
    similarity = near_duplicate_candidates(items)
    if Counter(x["split"] for x in items) != {"train": 320, "validation": 64}:
        raise ValueError("Report counts mismatch")
    for split, quotas in BUCKETS.items():
        if Counter(x["bucket"] for x in items if x["split"] == split) != quotas:
            raise ValueError("Bucket quotas mismatch")
    tokenizer, provenance = tokenizer_and_hashes()
    import tokenizers as tokenizers_package
    import transformers
    token_rows, counts = [], []
    for item in items:
        ids, labels, prefix, supervised = tokenize(tokenizer, item["messages"])
        token_rows.append(dict(id=item["id"], split=item["split"], input_ids=ids, labels=labels))
        counts.append(dict(id=item["id"], split=item["split"], input=prefix,
                           supervised=supervised, total=len(ids)))
    output.mkdir(parents=True)
    for split in BUCKETS:
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps({"id": x["id"], "messages": x["messages"]}, ensure_ascii=False) + "\n"
                                for x in items if x["split"] == split))
    (output / "tokenized.jsonl").write_text("".join(json.dumps(x) + "\n" for x in token_rows))
    manifest = dict(seed=SEED, status="pending-assistant-review", model=MODEL, revision=REVISION,
                    prompt_sha256=sha(CORRECTION / "prompts/corrected.txt"),
                    schema_sha256=sha(CORRECTION / "schema.json"),
                    code_sha256=sha(Path(__file__)), tokenizer_provenance=provenance,
                    smoke_sha256={name: sha(CORRECTION / name) for name in (
                        "synthetic_specs.json", "synthetic/corrected/train.jsonl",
                        "synthetic/corrected/validation.jsonl")},
                    versions=dict(transformers=transformers.__version__, tokenizers=tokenizers_package.__version__),
                    examples=[{k: v for k, v in x.items() if k not in ("input", "messages")} for x in items])
    dump(output / "manifest.json", manifest)
    stats = dict(counts=dict(Counter(x["split"] for x in items)),
                 buckets={s: dict(Counter(x["bucket"] for x in items if x["split"] == s)) for s in BUCKETS},
                 domains=dict(Counter(x["domain"] for x in items)),
                 families={s: sum(v == s for v in families.values()) for s in BUCKETS},
                 clean={s: sum(x["clean"] for x in items if x["split"] == s) for s in BUCKETS},
                 transformations=dict(Counter(t for x in items for t in x["transformations"])),
                 tokens={s: {field: quantiles([x[field] for x in counts if x["split"] == s])
                             for field in ("input", "supervised", "total")} for s in BUCKETS},
                 max_total=max(x["total"] for x in counts), duplicate_inputs=0, split_leakage=0,
                 near_duplicate_candidates=similarity)
    dump(output / "statistics.json", stats)
    dump(output / "review.json", dict(status="pending-assistant-review", reviewed_validation=[],
                                     reviewed_training=[], findings=[], unresolved=[]))
    (output / "README.md").write_text("# CA1 synthetic SFT pilot packet\n\n"
        "Status: pending assistant review. User acceptance is a separate step.\n\n"
        f"Seed: {SEED}. Train: 320 reports. Validation: 64 reports. Includes frozen corrected 16/8 smoke subset.\n\n"
        "Build from the repository root: `/opt/venv/bin/python -m research.extraction.generate_sft_pilot`.\n\n"
        "Inputs are fictitious parser-like text. Assistant targets follow the exact CA1 prompt and schema. "
        "The single tokenized file masks system, user and post-EOS tokens. No model inference or training was run.\n\n"
        "`manifest.json` contains source roles, family/split membership and visible-cell repair evidence. "
        "`statistics.json` contains token and coverage summaries. `review.json` records manual review.\n")
    checksums = {p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != "SHA256SUMS.json"}
    dump(output / "SHA256SUMS.json", checksums)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def verify_packet(output=OUTPUT):
    """Check the sealed packet without regenerating or mutating its files."""
    manifest = load(output / "manifest.json")
    checksums = load(output / "SHA256SUMS.json")
    actual_files = {p.name for p in output.iterdir() if p.is_file() and p.name != "SHA256SUMS.json"}
    if set(checksums) != actual_files or any(sha(output / name) != digest for name, digest in checksums.items()):
        raise ValueError("Packet checksum mismatch")
    if manifest["code_sha256"] != sha(Path(__file__)):
        raise ValueError("Generator code changed")
    if manifest["prompt_sha256"] != sha(CORRECTION / "prompts/corrected.txt") or manifest["schema_sha256"] != sha(CORRECTION / "schema.json"):
        raise ValueError("Prompt or schema changed")
    if any(sha(CORRECTION / name) != digest for name, digest in manifest["smoke_sha256"].items()):
        raise ValueError("Smoke source changed")
    tokenizer, provenance = tokenizer_and_hashes()
    if provenance != manifest["tokenizer_provenance"]:
        raise ValueError("Tokenizer provenance changed")
    examples = {x["id"]: x for x in manifest["examples"]}
    schema = load(CORRECTION / "schema.json")
    tokenized = {x["id"]: x for x in (json.loads(line) for line in (output / "tokenized.jsonl").read_text().splitlines())}
    seen = set()
    for split, expected in (("train", 320), ("validation", 64)):
        rows = [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
        if len(rows) != expected:
            raise ValueError("Packet count mismatch")
        for row in rows:
            identifier, messages = row["id"], row["messages"]
            if identifier in seen or identifier not in examples or examples[identifier]["split"] != split:
                raise ValueError("Packet membership mismatch")
            seen.add(identifier)
            if [m["role"] for m in messages] != ["system", "user", "assistant"]:
                raise ValueError("Message roles changed")
            if messages[0]["content"] != (CORRECTION / "prompts/corrected.txt").read_text():
                raise ValueError("System prompt changed")
            text = messages[1]["content"]
            if not text.startswith("Extract laboratory observations from this parser output:\n<report>\n") or not text.endswith("\n</report>\n"):
                raise ValueError("Report wrapper changed")
            strict_validate(messages[2]["content"], schema)
            if sha(messages[2]["content"]) != examples[identifier]["target_sha256"]:
                raise ValueError("Target hash mismatch")
            ids, labels, _, _ = tokenize(tokenizer, messages)
            if tokenized[identifier] != dict(id=identifier, split=split, input_ids=ids, labels=labels):
                raise ValueError("Tokenized labels changed")
    if seen != set(examples) or seen != set(tokenized):
        raise ValueError("Missing manifest or tokenized row")
    print(f"Verified {len(seen)} CA1 pairs, hashes, schema, chat boundaries and labels.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true", help="Validate an existing sealed packet")
    args = parser.parse_args()
    if args.verify:
        verify_packet(args.output)
    else:
        build(args.output)


if __name__ == "__main__":
    main()
