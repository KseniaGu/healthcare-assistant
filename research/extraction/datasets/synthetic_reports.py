"""Deterministic fictitious CA1 scenarios, visible targets, and layout rendering."""

import difflib
import json
import random
import re
from collections import Counter
from datetime import date, timedelta
from statistics import median
from typing import Any

from research.extraction.datasets.packet import sha, validate_target

Assay = tuple[str, str | None, str]

ELIGIBLE = frozenset("A1 A2 A3 A4 C1 D0002 D0006 D0007 D0011 D0012 D0014 D0015 D0016".split())


BUCKETS = {"train": {"ordinary": 192, "layout": 80, "difficult": 48},
           "validation": {"ordinary": 38, "layout": 16, "difficult": 10}}


DOMAINS = ("chemistry", "hematology", "urine", "qualitative")


PANELS: dict[str, list[Assay]] = {
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


def family_catalog():
    """Fix source groups, layouts and splits before drawing any variants."""
    families: list[dict[str, Any]] = []
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
    rng = random.Random(20260928)
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


def base_value_for(domain, name, reference, rng, index):
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
    def fmt(d):
        return d.strftime("%d.%m.%Y")
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
    validate_target(json.dumps(target, ensure_ascii=False), schema)
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
                def numbers(s):
                    return re.findall(r"\d+(?:[.,]\d+)*|[<>≤≥+−]", s)
                if numbers(before.removesuffix(" *")) != numbers(after):
                    raise ValueError("Measured number, bound or polarity changed")
    if any(re.search(r"[А-Яа-я]", re.sub(r"'[^']*'", "", warning)) for warning in target["errors"]):
        raise ValueError("Non-English warning text")


def ensure_split_and_provenance(items, sources):
    families: dict[str, str] = {}
    groups: dict[str, str] = {}
    ids: set[str] = set()
    inputs: set[str] = set()
    normalized: set[str] = set()
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
    representatives: dict[str, dict[str, Any]] = {}
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


def quantiles(values):
    ordered = sorted(values)
    return {"p50": median(ordered), "p95": ordered[(len(ordered) * 95 + 99) // 100 - 1], "max": ordered[-1]}


SEED = 20260929


SOURCE_LAYOUTS = (("A1", "A2", "A3", "A4"), ("C1",), ("D0002", "D0011", "D0012"),
                  ("D0006", "D0007"), ("D0014", "D0015", "D0016"), ("A3", "D0012", "D0016"))


LABELS = {"chemistry": "Биохимическое исследование крови", "hematology": "Клиническое исследование крови",
          "urine": "Общий анализ мочи", "qualitative": "Молекулярные и серологические исследования"}


VOCAB_SOURCES = {"chemistry": ("A3", "A4"), "hematology": ("A1", "A2", "D0007", "D0011", "D0016"),
                 "urine": ("C1", "D0012"), "qualitative": ("D0002", "D0015")}


EXTRA: dict[str, list[Assay]] = {
    "chemistry": [("Щелочная фосфатаза", "Ед/л", "35–128"), ("ГГТ", "Ед/л", "<55"),
                  ("С-реактивный белок", "мг/л", "<5"), ("ЛПВП", "ммоль/л", ">1,0"),
                  ("ЛПНП", "ммоль/л", "<3,0"), ("Билирубин прямой", "мкмоль/л", "<5"),
                  ("Хлориды", "ммоль/л", "98–108"), ("Креатинкиназа", "Ед/л", "25–190")],
    "hematology": [("MCHC", "г/л", "310–360"), ("Нейтрофилы, абс.", "10^9/л", "1,5–7,5"),
                    ("Лимфоциты, абс.", "10^9/л", "1,0–4,0"), ("Моноциты, абс.", "10^9/л", "0,1–1,0"),
                    ("Эозинофилы, абс.", "10^9/л", "0–0,6"), ("Базофилы, абс.", "10^9/л", "0–0,2"),
                    ("Тромбокрит", "%", "0,1–0,5"), ("Ретикулоциты, абс.", "10^9/л", "20–100")],
    "urine": [("Плотность мочи", None, "1,005–1,030"), ("Белок", "г/л", "<0,15"),
              ("Слизь", None, "не обнаружена"), ("Дрожжеподобные грибы", None, "не обнаружены"),
              ("Эпителий переходный", "клеток/ПЗ", "0–2"), ("Фосфаты", "клеток/ПЗ", "0–2"),
              ("Оксалаты", "клеток/ПЗ", "0–2"), ("Аскорбиновая кислота", "ммоль/л", "<0,6")],
}


MORE: dict[str, list[Assay]] = {
    "chemistry": [("Мочевая кислота", "мкмоль/л", "140–420"), ("ЛДГ", "Ед/л", "120–245"),
                  ("Амилаза", "Ед/л", "25–125"), ("Липаза", "Ед/л", "13–60"),
                  ("Ферритин", "мкг/л", "15–220"), ("Трансферрин", "г/л", "2,0–3,6"),
                  ("Цинк", "мкмоль/л", "10–19"), ("Витамин B12", "пмоль/л", "130–670")],
    "hematology": [("PDW", "%", "9–18"), ("P-LCR", "%", "13–43"),
                    ("Незрелые гранулоциты, %", "%", "0–1"), ("Незрелые гранулоциты, абс.", "10^9/л", "0–0,1"),
                    ("Нормобласты, %", "%", "0–1"), ("Нормобласты, абс.", "10^9/л", "0–0,1"),
                    ("Средний объем ретикулоцита", "фл", "80–120"), ("Гемоглобин ретикулоцитов", "пг", "25–35")],
    "urine": [("Лейкоцитарная эстераза", None, "отрицательно"), ("Альбумин мочи", "мг/л", "<30"),
              ("Креатинин мочи", "ммоль/л", "3–25"), ("Цилиндры зернистые", "клеток/ПЗ", "0–1"),
              ("Цилиндры эпителиальные", "клеток/ПЗ", "0–1"), ("Аморфные кристаллы", "клеток/ПЗ", "0–2"),
              ("Кристаллы оксалата кальция", "клеток/ПЗ", "0–2"),
              ("Кристаллы трипельфосфатов", "клеток/ПЗ", "0–2")],
}


ASSAYS: tuple[list[Assay], ...] = (
    [("РНК вируса гриппа A, ПЦР", None, "не обнаружено"), ("РНК вируса гриппа B, ПЦР", None, "не обнаружено"),
     ("РНК SARS-CoV-2, ПЦР", None, "не обнаружено"), ("Антиген аденовируса", None, "отрицательно"),
     ("РНК респираторно-синцитиального вируса", None, "не обнаружено"), ("РНК коронавируса OC43", None, "не обнаружено"),
     ("РНК энтеровируса, ПЦР", None, "не обнаружено"), ("Антиген вируса гриппа A", None, "отрицательно")],
    [("HBsAg, качественно", None, "отрицательно"), ("anti-HCV, суммарные", None, "отрицательно"),
     ("Антитела к ВИЧ 1/2 и антиген p24", None, "отрицательно"), ("Антитела к T. pallidum", None, "отрицательно"),
     ("Антитела IgM к T. pallidum", None, "отрицательно"), ("Антитела anti-HBs", "мМЕ/мл", "<10"),
     ("Антитела IgG к вирусу гепатита A", "Ед/мл", "<20"), ("Антитела к вирусу гепатита E", None, "отрицательно")],
    [("ДНК Chlamydia trachomatis", None, "не обнаружено"), ("ДНК Mycoplasma genitalium", None, "не обнаружено"),
     ("ДНК Ureaplasma spp.", None, "не обнаружено"), ("ДНК Neisseria gonorrhoeae", None, "не обнаружено"),
     ("ДНК Trichomonas vaginalis", None, "не обнаружено"), ("ДНК вируса простого герпеса 1", None, "не обнаружено"),
     ("ДНК вируса простого герпеса 2", None, "не обнаружено"), ("ДНК ВПЧ высокого риска", None, "не обнаружено")],
    [("Антитела IgM к цитомегаловирусу", None, "отрицательно"), ("Антитела IgG к цитомегаловирусу", "Ед/мл", "<12"),
     ("Антитела IgM к вирусу краснухи", None, "отрицательно"), ("Антитела IgG к вирусу краснухи", "МЕ/мл", "<10"),
     ("Антитела IgM к ВПГ 1/2", None, "отрицательно"), ("Антитела IgG к ВПГ 1/2", "Ед/мл", "<12"),
     ("Антитела IgM к Toxoplasma gondii", None, "отрицательно"), ("Антитела IgG к Toxoplasma gondii", "МЕ/мл", "<10")],
    [("Антиген Helicobacter pylori", None, "отрицательно"), ("Антиген ротавируса", None, "отрицательно"),
     ("Антиген аденовируса", None, "отрицательно"), ("РНК норовируса", None, "не обнаружено"),
     ("Антиген Giardia lamblia", None, "отрицательно"), ("ДНК Salmonella spp.", None, "не обнаружено"),
     ("ДНК Shigella spp.", None, "не обнаружено"), ("РНК энтеровируса", None, "не обнаружено")],
    [("Антитела IgM к вирусу Эпштейна–Барр", None, "отрицательно"),
     ("Антитела IgG к вирусу Эпштейна–Барр", "Ед/мл", "<20"),
     ("Антитела IgM к вирусу кори", None, "отрицательно"), ("Антитела IgG к вирусу кори", "МЕ/мл", "<10"),
     ("Антитела IgM к вирусу паротита", None, "отрицательно"), ("Антитела IgG к вирусу паротита", "Ед/мл", "<12"),
     ("Антитела IgM к VZV", None, "отрицательно"), ("Антитела IgG к VZV", "Ед/мл", "<12")],
    [("ДНК Bordetella pertussis", None, "не обнаружено"), ("ДНК Mycoplasma pneumoniae", None, "не обнаружено"),
     ("ДНК Chlamydophila pneumoniae", None, "не обнаружено"), ("РНК вируса парагриппа 1", None, "не обнаружено"),
     ("РНК вируса парагриппа 2", None, "не обнаружено"), ("РНК риновируса", None, "не обнаружено"),
     ("РНК метапневмовируса", None, "не обнаружено"), ("ДНК аденовируса", None, "не обнаружено")],
    [("HBsAg", None, "отрицательно"), ("anti-HBc IgM", None, "отрицательно"),
     ("anti-HBc total", None, "отрицательно"), ("anti-HBs", "мМЕ/мл", "<10"),
     ("anti-HCV", None, "отрицательно"), ("anti-HAV IgM", None, "отрицательно"),
     ("anti-HAV IgG", "Ед/мл", "<20"), ("anti-HEV IgM", None, "отрицательно")],
)


VALIDATION_ASSAYS: tuple[list[Assay], ...] = (
    [("ДНК Bordetella parapertussis", None, "не обнаружено"),
     ("ДНК Bordetella pertussis", None, "не обнаружено"),
     ("ДНК Mycoplasma pneumoniae", None, "не обнаружено"),
     ("ДНК Chlamydophila pneumoniae", None, "не обнаружено"),
     ("РНК вируса парагриппа 3", None, "не обнаружено"),
     ("РНК вируса парагриппа 4", None, "не обнаружено"),
     ("РНК метапневмовируса", None, "не обнаружено"),
     ("РНК риновируса A/B", None, "не обнаружено"),
     ("РНК коронавируса NL63", None, "не обнаружено"),
     ("РНК коронавируса HKU1", None, "не обнаружено"),
     ("ДНК Haemophilus influenzae", None, "не обнаружено"),
     ("ДНК Streptococcus pneumoniae", None, "не обнаружено")],
    [("HBsAg, скрининг", None, "отрицательно"),
     ("HBeAg", None, "отрицательно"),
     ("anti-HBe", None, "отрицательно"),
     ("anti-HBc IgM", None, "отрицательно"),
     ("anti-HBc total", None, "отрицательно"),
     ("anti-HBs, количественно", "мМЕ/мл", "<10"),
     ("ДНК HBV, ПЦР", None, "не обнаружено"),
     ("anti-HCV, скрининг", None, "отрицательно"),
     ("РНК HCV, ПЦР", None, "не обнаружено"),
     ("anti-HAV IgM", None, "отрицательно"),
     ("anti-HEV IgM", None, "отрицательно"),
     ("anti-HEV IgG", None, "отрицательно")],
)


ASSAY_ADDITIONS: dict[int, list[Assay]] = {
    2: [("ДНК вируса папилломы человека, тип 16", None, "не обнаружено"),
        ("ДНК вируса папилломы человека, тип 18", None, "не обнаружено"),
        ("ДНК Mycoplasma hominis", None, "не обнаружено"),
        ("ДНК Ureaplasma parvum", None, "не обнаружено")],
    4: [("Антиген Clostridioides difficile", None, "отрицательно"),
        ("ДНК Campylobacter spp.", None, "не обнаружено"),
        ("Антиген Cryptosporidium spp.", None, "отрицательно"),
        ("ДНК Yersinia enterocolitica", None, "не обнаружено")],
}


def catalog():
    families = family_catalog()
    for family in families:
        if family["split"] == "train":
            family["development_sources"] = list(dict.fromkeys(SOURCE_LAYOUTS[family["style"]] + VOCAB_SOURCES[family["domain"]]))
    return families


def panel_for(family):
    domain, style = family["domain"], family["style"]
    if domain == "qualitative":
        if style in (12, 13):
            return VALIDATION_ASSAYS[style - 12]
        group = style if style < 6 else style - 6
        companion = (6, 7, 2, 5, 4, 3, 0, 1)[group]
        additional = ASSAY_ADDITIONS.get(group, ASSAYS[companion])
        return list(dict((name, (name, unit, ref)) for name, unit, ref in
                         ASSAYS[group] + additional).values())
    panel = PANELS[domain] + EXTRA[domain] + MORE[domain]
    # Domain-specific rotations alter panel composition without copying a result vector.
    domain_offset = 0 if domain == "chemistry" else 2 if domain == "hematology" else 5
    rotation = ((style * 4 + domain_offset) if family["split"] == "train"
                else (25 + 2 * (style - 6) + domain_offset)) % len(panel)
    return panel[rotation:] + panel[:rotation]


def value_for(domain, name, reference, rng, index):
    if domain == "qualitative" and not re.search(r"\d", reference):
        return ("не обнаружено", "Отрицательный", "отрицательно", "Положительный", "+", "−")[index % 6]
    if name in ("Плотность мочи", "Удельный вес"):
        return f"1,{rng.randrange(6, 31):03d}"
    if domain == "urine" and (name.startswith(("Цилиндры", "Кристаллы", "Эпителий")) or name in ("Фосфаты", "Оксалаты", "Аморфные кристаллы")):
        return str(rng.randrange(0, 6))
    if name == "Слизь":
        return ("не обнаружена", "следы", "умеренно")[index % 3]
    if name == "Дрожжеподобные грибы":
        return ("не обнаружены", "единичные")[index % 2]
    return base_value_for(domain, name, reference, rng, index)


def rows_for(family, variant, rng, bucket, clean):
    domain = family["domain"]
    panel = panel_for(family)
    if domain == "qualitative":
        count = 8 + (variant * 2 + family["style"]) % 5
    else:
        count = 22 + (variant * 3 + family["style"]) % 9
    start = (variant * 3 + family["style"]) % len(panel)
    chosen = [panel[(start + index) % len(panel)] for index in range(count)]
    rows = [dict(name=name, value=value_for(domain, name, ref, rng, variant + index),
                 unit=unit or "—", reference_range=ref, state="current")
            for index, (name, unit, ref) in enumerate(chosen)]
    transformations = []
    if bucket != "ordinary" and variant % 2 == 0:
        source = rows[0]
        rows.insert(1, dict(name=source["name"], value=value_for(domain, source["name"], source["reference_range"], rng, variant + 50),
                            unit=source["unit"], reference_range="—", state="history"))
    if bucket != "ordinary" and variant % 3 == 0:
        upcoming = panel[(start + count) % len(panel)]
        rows.append(dict(name=upcoming[0], value="ожидается", unit=upcoming[1] or "—",
                         reference_range="—", state="pending"))
    if not clean:
        current = [row for row in rows if row["state"] == "current"]
        if bucket == "difficult":
            row = current[(variant + family["style"]) % len(current)]
            mode = (variant + family["style"]) % 4
            field, replacement, tag = (("name", "—", "unnamed-row"), ("value", "—", "missing-value"),
                                       ("unit", "10°/π", "ambiguous-unit"),
                                       ("reference_range", "—", "missing-reference"))[mode]
            row[field] = replacement
            transformations.append(tag)
        if bucket != "ordinary" and variant % 3 == 1:
            row = current[-1]
            if row["value"] not in ("+", "−", "—"):
                row["value"] += " *"
                transformations.append("separate-flag")
        if variant % 5 == 1:
            for row in current:
                if row["unit"] == "г/л":
                    row["unit"] = "г/π"
                    transformations.append("unit-glyph-repair")
                    break
        if variant % 7 == 2:
            for row in current:
                if "Глюкоза" in row["name"]:
                    row["name"] = row["name"].replace("Глюкоза", "Глюкoза")
                    transformations.append("name-glyph-repair")
                    break
    return rows, transformations


def dates_for(family, variant, bucket):
    lines, _, tag = date_lines(variant + int(sha(family["family"])[:4], 16) % 50, bucket)
    return lines, tag


def metadata(family, variant, dates):
    """Use distinct fictitious document structures, not one universal header."""
    style = family["style"]
    label = LABELS[family["domain"]]
    order = 100000 + (int(sha(family["family"])[:8], 16) + variant * 7919) % 800000
    person = (variant + style) % 5 in (0, 2)
    stamp = (variant + 2 * style) % 5 in (0, 1)
    patient = "<PERSON>" if person else "обезличено"
    organization = "<ORGANIZATION>" if person and style % 2 == 0 else f"Лабораторная служба № {20 + style}"
    lines = []
    if style == 0:
        if stamp:
            lines.extend(["IMAGE", ""])
        lines.extend([f"{patient:<40} {organization}", "Пол: <GENDER>             Возраст: <AGE>                 Тел.: <PHONE NUMBER>",
                      f"Код заказа: {order}                            Пункт: <LOCATION>",
                      f"Материал: {'кровь' if family['domain'] != 'urine' else 'моча'}", *dates,
                      "Дата поступления образца: отмечена в журнале", "", "Исследование                    Результат        Единицы       Референсные значения    Комментарий", label])
    elif style == 1:
        lines.extend(["IMAGE" if stamp else "", f"ЗАКАЗ № {order}                 Пациент: {patient}",
                      "Дата регистрации                Код подразделения              Направляющий врач",
                      "Пол пациента: <GENDER>           Страховой полис:                 Отделение:",
                      "Адрес пациента:                   Номер истории болезни:",
                      "Проба зарегистрирована                    Биоматериал: " + ("моча" if family["domain"] == "urine" else "кровь"),
                      *dates, "", "РЕЗУЛЬТАТЫ ИССЛЕДОВАНИЙ", label,
                      "Исследование                         Значение         Ед. изм.         Нормальные значения"])
    elif style == 2:
        lines.extend(["Клинико-диагностическое подразделение", f"Протокол № {order}   {label}", "",
                      "Ист.бол./Амб.карта:                        Отделение:", f"Ф.И.О.: {patient:<36} Ф.И.О. врача: —",
                      "Дата рождения:                              Пол: <GENDER>",
                      "Номер направления:                           Номер образца:", *dates,
                      "Дата доставки материала: см. журнал регистрации", "www.clinic.example" if variant % 2 == 0 else "Адрес пункта: <LOCATION>", "", "Результаты анализов", "Наименование                 Результат     Ед. изм.        Референс"])
    elif style == 3:
        lines.extend(["ЦЕНТР ЛАБОРАТОРНОЙ ДИАГНОСТИКИ", "Контакт: <PHONE NUMBER>  info@laboratory.example",
                      "www.laboratory.example" if variant % 2 == 0 else "Адрес пункта: <LOCATION>",
                      f"Заказ № {order}                   Пациент: {patient}",
                      "Пол: <GENDER>                      Дата рождения: <DATE OF BIRTH>",
                      f"{label.upper()}", *dates,
                      "Штрих-код образца: ||| || |||  Материал: " + ("моча" if family["domain"] == "urine" else "венозная кровь"),
                      "Результат по текущему заказу", "Параметр                         Результат       Ед. изм.            Референс"])
    elif style == 4:
        lines.extend(["Единая справочная лабораторной службы <PHONE NUMBER>",
                      "Консультация по вопросам оформления бланка: www.assay.example" if variant % 2 == 0 else "Пункты взятия: <LOCATION>, <LOCATION>",
                      f"            {organization}                         № {order}",
                      "", "СВЕДЕНИЯ О ПАЦИЕНТЕ                         Место забора         Место выдачи",
                      f"Ф.И.О. {patient:<40} Дата рождения/возраст        Пол <GENDER>",
                      "№ заявки                              Отделение           Врач:", *dates,
                      "", f"РЕЗУЛЬТАТЫ ИССЛЕДОВАНИЙ: {label}",
                      "№  Наименование                 Значение            Ед. изм.          Референс"])
    elif style == 5:
        lines.extend([f"Электронная карта анализа / № {order}", "Лабораторное подразделение: <ORGANIZATION>" if stamp else "Лабораторное подразделение: центральный участок",
                      f"Пациент: {patient}             Направление: электронное", "Тип материала: зарегистрирован",
                      *dates, "Статус заказа: выполнен", "", f"Показатели — {label}",
                      "Название показателя                Результат          Единицы             Референсный интервал"])
    elif style == 6:
        lines.extend(["ПРИЛОЖЕНИЕ К ПРОТОКОЛУ ИССЛЕДОВАНИЯ", f"Регистрационный № {order}",
                      "Биоматериал: исследуемая проба", f"Пациент: {patient}        Код: {order % 10000:04d}",
                      *dates, "", f"Раздел: {label}", "Показатель   |  Результат  |  Ед.  |  Реф. интервал"])
    elif style == 7:
        lines.extend(["Список измерений по образцу", f"Образец № {order % 100000:05d}             Заказ № {order}",
                      f"Сведения о пациенте: {patient}", "Пункт приема: <LOCATION>", *dates,
                      "", f"{label}", "Параметр / результат / единица / допустимые границы"])
    elif style == 8:
        lines.extend(["ОТЧЕТ ЛАБОРАТОРИИ: биологический материал", f"Внутренний код исследования {order}",
                      "Вид пробы и условия хранения указаны в направлении", f"Пациент: {patient}", *dates,
                      "Показатели исследования:", label, "Название анализа       Результат       Ед. измерения        Норма"])
    elif style == 9:
        lines.extend(["Протокол измерений / клиническое подразделение", "Адрес подразделения: <LOCATION>",
                      f"Заказ {order}         Образец {order % 10000:04d}", "Врач, направивший образец: —",
                      *dates, "", f"{label.upper()}", "Наименование                  Результат           Ед.          Референс"])
    elif style == 10:
        lines.extend(["Результат обработки пробы", f"Номер исследования: {order}",
                      "Ф.И.О.: " + patient, "Документ зарегистрирован в электронной системе", *dates,
                      "", f"Лабораторная группа: {label}", "Показатель                     Значение           Единица          Границы"])
    elif style == 11:
        lines.extend(["Ведомость лабораторных исследований", f"Направление № {order}",
                      "Пациент: " + patient, "Место регистрации материала: <LOCATION>",
                      *dates, "", f"Тесты / {label}", "Наименование             Результат          Ед. изм.        Нормальные значения"])
    elif style == 12:
        lines.extend(["Служба лабораторной диагностики / выписка", f"Номер пробы: {order}",
                      "Пол пациента: <GENDER>          Возраст: <AGE>", "Заказчик: медицинское подразделение",
                      *dates, "", label, "Тест                   Результат            Единицы           Интервал"])
    else:
        lines.extend(["Лист исследования биоматериала", f"Код образца {order}   Пациент {patient}",
                      "Отделение: амбулаторное", "Тип исследования: лабораторное", *dates,
                      "", f"{label}", "Параметр            Ответ             Ед. изм.          Референс"])
    if style >= 6:
        additional = {
            6: ["Лабораторная сеть: участок приема № 4", "Направление зарегистрировано в журнале",
                "Категория заказа: амбулаторный", "Дата доставки: указана в карте пробы",
                "Исследовательский материал: зарегистрирован", "Сведения об исполнителе: электронная отметка",
                "Лист содержит только текущие результаты", ""],
            7: ["Сведения о пробе: первичная", "Номер транспортного контейнера: см. учетную карту",
                "Материал поступил в лабораторию", "Статус пробы: пригодна",
                "Направляющая организация: обезличено", "Приемный пункт: клиническое подразделение",
                "Результаты выдаются для указанного образца", ""],
            8: ["Исследовательское подразделение: лаборатория № 8", "Место взятия материала: <LOCATION>",
                "Дата регистрации заказа: в системе учета", "Состояние контейнера: удовлетворительно",
                "Медицинская карта: код не отображен", "Номер бланка и образца совпадают",
                "Предыдущие результаты не входят в текущий лист", ""],
            9: ["Служба регистрации проб", "Пациентские сведения скрыты в электронном экземпляре",
                "Тип документа: отчет о завершенном исследовании", "Способ доставки: курьерская передача",
                "Время получения образца: учетная запись", "Материал принят без замечаний",
                "Отделение: общий прием", ""],
            10: ["Контроль идентификации образца выполнен", "Врач, оформивший направление: обезличено",
                 "Пункт приема: районное подразделение", "Категория исследования: плановая",
                 "Дополнительные листы: по запросу", "Протокол создан в лабораторной системе",
                 "Дата доставки образца: в журнале", ""],
            11: ["Центр лабораторных измерений", "Заявитель: медицинское подразделение",
                 "Форма документа: электронная выписка", "Тип образца: отмечен в направлении",
                 "Контейнер принят, номер сохранен", "Код исследования: по внутреннему каталогу",
                 "Статус заказа: завершен", ""],
            12: ["Регистрационная карта образца", "Номер задания: совпадает с номером пробы",
                 "Отправитель: <LOCATION>", "Тип материала: указан при взятии",
                 "Сведения о транспортировке: без замечаний", "Метод и аппарат: по внутреннему протоколу",
                 "Результат относится к текущему заказу", ""],
            13: ["Лабораторное подразделение / реестр исследований", "Место выдачи: амбулаторное отделение",
                 "Номер страницы: 1", "Образец: зарегистрирован",
                 "Данные пациента: скрыты", "Условия приема: без особых отметок",
                 "Методика измерения: отражена в карте", ""],
        }[style]
        lines[2:2] = additional
        if style in (6, 8, 10, 12) and variant % 3 == 0:
            lines.insert(1, "<ORGANIZATION>                 Отдел лабораторной диагностики")
        if style in (7, 9, 11, 13) and variant % 2 == 0:
            lines.insert(2, "www.research-lab.example     Контакт: <PHONE NUMBER>")
        if style in (7, 8, 11) and variant % 3 == 1:
            lines.insert(0, "IMAGE")
    # Sparse OCR artifacts follow the kind of document; they do not alter measurements.
    if stamp and style in (1, 4, 6, 10, 12):
        lines.insert(0, "IMAGE")
    if style in (2, 3, 4, 5, 8, 11) and variant % 3 == 0:
        lines.insert(max(2, len(lines) // 2), "")
    return lines


def row_line(row, style, index, variant):
    n, v, u, ref = (row[key] for key in ("name", "value", "unit", "reference_range"))
    if style in (0, 1, 4, 9, 11, 13):
        prefix = f"{index:02d} " if style in (4, 11) else ""
        if len(n) > 30 and (index + variant) % 3 == 0:
            return f"{prefix}{n}\n{' ':<34}{v:>12}     {u:<12} {ref}"
        return f"{prefix}{n:<35} {v:>12}     {u:<12} {ref}"
    if style in (2, 8):
        return f"{n:<33}  {v:<15}  {u:<12}   {ref}"
    if style in (3, 12):
        return f"{n:<35}  {v:>12}   {u:<11} {ref}"
    if style in (5, 10):
        return f"{n}\n   результат: {v}       единица: {u}       референс: {ref}"
    if style == 6:
        return f"{n} | {v} | {u} | {ref}"
    if style == 7:
        return f"{n:<32} {v:>12}   {u:<10} {ref}"
    return f"{n:<35} {v:>12}     {u:<11} {ref}"


def footer(family, variant):
    style = family["style"]
    if style == 0:
        return ["", "Исполнитель: <PERSON>, врач клинической лабораторной диагностики" if variant % 2 == 0 else "Исполнитель: специалист лаборатории",
                "* Звездочка относится к показателю с отметкой за пределами референса.",
                "Результаты исследования не заменяют консультацию специалиста.",
                "Электронный экземпляр документа сформирован для данного заказа.",
                "IMAGE" if variant % 3 == 0 else "Подпись исполнителя: зарегистрирована", "стр. 1 из 1"]
    if style == 1:
        return ["", "Заведующий лабораторией: <PERSON>" if variant % 2 == 0 else "Заведующий лабораторией: подпись в электронном документе", "ПЕЧАТЬ: время формирования указано в учетной записи",
                "Сведения в графе нормы относятся к указанному методу исследования.", "Страница 1 из 1"]
    if style == 2:
        return ["", "Врач КЛД: <PERSON>" if variant % 2 == 0 else "Врач КЛД: подпись в электронной карте", "Дата выдачи: см. электронную запись",
                "Результат относится к текущему образцу; предыдущее измерение приводится отдельно.", "Стр. 1 из 2"]
    if style == 3:
        return ["", "Метод исследования и контроль качества зафиксированы в лаборатории.",
                "Результат выдал: <PERSON>" if variant % 2 == 0 else "Результат выдал: специалист лаборатории", "ПРОМЕЖУТОЧНЫЙ РЕЗУЛЬТАТ" if variant % 4 == 0 else "ИТОГОВЫЙ РЕЗУЛЬТАТ",
                "Страница 1 из 2", "www.laboratory.example" if variant % 2 == 0 else "Контакт: <PHONE NUMBER>"]
    if style == 4:
        return ["", "Примечание: результаты сравниваются с нормой в соответствующей графе.",
                "Заведующая лабораторией", "          Расшифровка подписи", "Контактный пункт: <LOCATION>"]
    if style == 5:
        return ["", "Методика: указана в регистрационной карте образца.", "Проверил: специалист лаборатории",
                "Документ действителен для обозначенного заказа.", "Конец выписки"]
    suffix = ["", "Ответственный специалист: <PERSON>" if variant % 3 == 0 else "Ответственный специалист: подпись в электронной карте",
              "Примечание к референсу: применяется только к строке с результатом.",
              f"Лист {1 + (variant % 2)} из {1 + (variant % 2)}"]
    if style in (6, 9, 13):
        suffix.extend(["Приложение к заказу: дополнительные сведения хранятся в карте образца.",
                       "Завершено / запись проверена"])
    return suffix


def locate(line, row, offset, start=0):
    spans = {}
    cursor = start
    for field in ("name", "value", "unit", "reference_range"):
        raw = row[field]
        pos = line.find(raw, cursor)
        if pos < 0:
            pos = line.find(raw, start)
        if pos < 0:
            raise ValueError((field, raw, line))
        spans[field] = dict(start=offset + pos, end=offset + pos + len(raw), text=raw)
        cursor = pos + len(raw)
    return spans, cursor


def render(family, variant, rows, dates):
    """Render rows and locate every current or excluded cell in final visible text."""
    style = family["style"]
    parts = metadata(family, variant, dates)
    spans = []
    index = 0
    while index < len(rows):
        row = rows[index]
        if index == len(rows) // 2 and variant % 4 == 0 and style in (0, 2, 3, 4, 6, 9, 13):
            parts.extend(["", "Страница 1 из 2 / продолжение на следующем листе",
                          "Страница 2 / продолжение результатов", "IMAGE" if style in (0, 4) else "",
                          "Исследование                 Результат         Ед. изм.        Референс"])
        if style == 7 and index + 1 < len(rows) and all(r["state"] == "current" for r in rows[index:index+2]):
            left = row_line(rows[index], style, index + 1, variant)
            right = row_line(rows[index+1], style, index + 2, variant)
            line = left + "       |       " + right
            offset = sum(len(part) + 1 for part in parts)
            parts.append(line)
            left_spans, cursor = locate(line, rows[index], offset)
            right_spans, _ = locate(line, rows[index+1], offset, cursor)
            spans.extend([left_spans, right_spans])
            index += 2
            continue
        line = row_line(row, style, index + 1, variant)
        if family["domain"] == "qualitative" and style not in (5, 10) and (len(row["name"]) > 28 or (variant + style) % 2 == 0 or index % 3 == 0):
            name_pos = line.find(row["name"])
            if name_pos >= 0:
                line = line[:name_pos] + row["name"] + "\n" + " " * 6 + line[name_pos + len(row["name"]):].lstrip()
        if row["state"] == "history":
            line = "АРХИВ / прошлый заказ: " + line
        if row["state"] == "pending":
            line = "Ожидается отдельное подтверждение: " + line
        offset = sum(len(part) + 1 for part in parts)
        parts.append(line)
        cell_spans, _ = locate(line, row, offset)
        spans.append(cell_spans)
        if family["domain"] == "qualitative" and style in (5, 10) and row["state"] == "current":
            method = "ПЦР" if row["name"].startswith(("ДНК", "РНК")) else "иммунохимический анализ"
            parts.append(f"   Метод исследования: {method}")
        if family["domain"] == "qualitative" and index in (3, 7):
            parts.append("Следующая группа исследований / продолжение панели")
        if index % 6 == 3 and variant % 5 == 0 and style in (1, 3, 4, 10):
            parts.append("Комментарий к странице: референсные значения приведены справа от результата.")
        index += 1
    ending = footer(family, variant)
    if variant % 4 == 0 and style in (0, 2, 3, 4, 6, 9, 13):
        ending = [line.replace("стр. 1 из 1", "стр. 2 из 2").replace("Стр. 1 из 2", "Стр. 2 из 2").replace("Страница 1 из 2", "Страница 2 из 2").replace("Лист 1 из 1", "Лист 2 из 2")
                  for line in ending]
    parts.extend(ending)
    text = "\n".join(parts)
    for cells in spans:
        for span in cells.values():
            if text[span["start"]:span["end"]] != span["text"]:
                raise ValueError("Visible cell span mismatch")
    return text, spans


def profile(texts):
    lengths = [len(text) for text in texts]
    lines = [len(text.splitlines()) for text in texts]
    first = [next((line for line in text.splitlines() if line.strip() and line.strip() != "IMAGE"), "") for text in texts]
    return dict(count=len(texts), characters=dict(min=min(lengths), median=median(lengths), max=max(lengths)),
                lines=dict(min=min(lines), median=median(lines), max=max(lines)),
                with_image=sum("IMAGE" in text for text in texts),
                with_person=sum("<PERSON>" in text for text in texts),
                with_organization=sum("<ORGANIZATION>" in text for text in texts),
                with_website=sum("www." in text.lower() for text in texts),
                unique_openings=len(set(first)), max_opening_count=max(Counter(first).values()))
