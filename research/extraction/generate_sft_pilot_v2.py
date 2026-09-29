"""Generate the second CA1 pilot from fictitious, full-document parser layouts."""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from research.extraction import generate_sft_pilot as prior
from research.extraction.evaluate_image_extraction import strict_validate
from research.extraction.prepare_sft import MODEL, REVISION

OUTPUT = prior.BASE / "sft-pilot-v2"
SEED = 20260929
SOURCE_LAYOUTS = (("A1", "A2", "A3", "A4"), ("C1",), ("D0002", "D0011", "D0012"),
                  ("D0006", "D0007"), ("D0014", "D0015", "D0016"), ("A3", "D0012", "D0016"))
LABELS = {"chemistry": "Биохимическое исследование крови", "hematology": "Клиническое исследование крови",
          "urine": "Общий анализ мочи", "qualitative": "Молекулярные и серологические исследования"}
VOCAB_SOURCES = {"chemistry": ("A3", "A4"), "hematology": ("A1", "A2", "D0007", "D0011", "D0016"),
                 "urine": ("C1", "D0012"), "qualitative": ("D0002", "D0015")}
EXTRA = {
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
MORE = {
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
ASSAYS = (
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
VALIDATION_ASSAYS = (
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
ASSAY_ADDITIONS = {
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
    families = prior.family_catalog()
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
    panel = prior.PANELS[domain] + EXTRA[domain] + MORE[domain]
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
    return prior.value_for(domain, name, reference, rng, index)


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
    lines, _, tag = prior.date_lines(variant + int(prior.sha(family["family"])[:4], 16) % 50, bucket)
    return lines, tag


def metadata(family, variant, dates):
    """Use distinct fictitious document structures, not one universal header."""
    style = family["style"]
    label = LABELS[family["domain"]]
    order = 100000 + (int(prior.sha(family["family"])[:8], 16) + variant * 7919) % 800000
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
    from statistics import median
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


def build(output=OUTPUT):
    """Write a new packet; never overwrite reviewed or historical artifacts."""
    if output.exists():
        raise FileExistsError(f"Packet exists: {output}")
    prompt = (prior.CORRECTION / "prompts/corrected.txt").read_text()
    schema = prior.load(prior.CORRECTION / "schema.json")
    sources = {item["id"]: item for item in prior.load(prior.SMOKE / "sources.json")}
    smoke_specs = prior.load(prior.CORRECTION / "synthetic_specs.json")
    smoke_rows = {split: [json.loads(line) for line in
                          (prior.CORRECTION / f"synthetic/corrected/{split}.jsonl").read_text().splitlines()]
                  for split in prior.BUCKETS}
    if {split: len(group) for split, group in smoke_rows.items()} != {"train": 16, "validation": 8}:
        raise ValueError("Frozen smoke counts changed")
    smoke_messages = {row["id"]: row["messages"] for group in smoke_rows.values() for row in group}
    families = catalog()
    assignments = prior.assignments(smoke_specs, families)
    rng = random.Random(SEED)
    items = []
    for spec in smoke_specs:
        messages = smoke_messages[spec["id"]]
        if messages[0]["content"] != prompt or prior.sha(messages[1]["content"].split("<report>\n", 1)[1].split("\n</report>", 1)[0]) != spec["input_sha256"]:
            raise ValueError("Frozen smoke message changed")
        strict_validate(messages[2]["content"], schema)
        items.append(dict(id=spec["id"], split=spec["split"], family=spec["family"],
                          source_group=spec["source_group"], domain="smoke", bucket=prior.smoke_bucket(spec),
                          clean="clean-control" in spec["tags"], smoke=True, tags=spec["tags"],
                          transformations=spec["tags"], development_sources=spec["development_sources"],
                          input=spec["input"], input_sha256=spec["input_sha256"],
                          target_sha256=prior.sha(messages[2]["content"]), evidence=[],
                          review_notes=spec["review_notes"], messages=messages,
                          review_status="historical-smoke"))
    for family in families:
        for variant in range(family["count"]):
            bucket, clean = assignments[family["family"], variant]
            rows, transformations = rows_for(family, variant, rng, bucket, clean)
            dates, date_tag = dates_for(family, variant, bucket)
            text, spans = render(family, variant, rows, dates)
            target, evidence = prior.target_from_visible(text, rows, spans, dates)
            prior.check_pair(text, rows, spans, target, evidence, schema)
            serialized = json.dumps(target, ensure_ascii=False, indent=2)
            messages = [{"role": "system", "content": prompt},
                        {"role": "user", "content": "Extract laboratory observations from this parser output:\n<report>\n" + text + "\n</report>\n"},
                        {"role": "assistant", "content": serialized}]
            items.append(dict(id=f"{family['family']}-{variant+1:02d}", split=family["split"],
                              family=family["family"], source_group=family["family"],
                              domain=family["domain"], bucket=bucket, clean=clean, smoke=False,
                              tags=[date_tag], transformations=transformations,
                              development_sources=family["development_sources"], input=text,
                              input_sha256=prior.sha(text), target_sha256=prior.sha(serialized),
                              evidence=evidence, messages=messages,
                              review_status="pending-assistant-review"))
    split_families = prior.ensure_split_and_provenance(items, sources)
    if Counter(item["split"] for item in items) != {"train": 320, "validation": 64}:
        raise ValueError("Report counts changed")
    for split, quotas in prior.BUCKETS.items():
        if Counter(item["bucket"] for item in items if item["split"] == split) != quotas:
            raise ValueError("Bucket counts changed")
    original_texts = [(prior.CORRECTION / "real" / identifier / "input.txt").read_text()
                      for identifier in sorted(prior.ELIGIBLE)]
    new_train = [item["input"] for item in items if item["split"] == "train" and not item["smoke"]]
    new_validation = [item["input"] for item in items if item["split"] == "validation" and not item["smoke"]]
    original_profile = profile(original_texts)
    train_profile = profile(new_train)
    validation_profile = profile(new_validation)
    if (train_profile["characters"]["median"] < original_profile["characters"]["median"] * 0.6
            or train_profile["lines"]["median"] < original_profile["lines"]["median"] * 0.6
            or train_profile["max_opening_count"] > len(new_train) // 5):
        raise ValueError("Synthetic input realism gate failed")
    tokenizer, provenance = prior.tokenizer_and_hashes()
    import tokenizers as tokenizers_package
    import transformers
    token_rows, token_counts = [], []
    for item in items:
        ids, labels, prefix, supervised = prior.tokenize(tokenizer, item["messages"])
        if len(ids) > 8192:
            raise ValueError(f"8192-token limit exceeded: {item['id']} ({len(ids)})")
        token_rows.append(dict(id=item["id"], split=item["split"], input_ids=ids, labels=labels))
        token_counts.append(dict(id=item["id"], split=item["split"], input=prefix,
                                 supervised=supervised, total=len(ids)))
    output.mkdir(parents=True)
    for split in prior.BUCKETS:
        (output / f"{split}.jsonl").write_text("".join(json.dumps({"id": item["id"], "messages": item["messages"]},
                                                               ensure_ascii=False) + "\n"
                                                    for item in items if item["split"] == split))
    (output / "tokenized.jsonl").write_text("".join(json.dumps(row) + "\n" for row in token_rows))
    manifest = dict(seed=SEED, status="pending-assistant-review", model=MODEL, revision=REVISION,
                    prompt_sha256=prior.sha(prior.CORRECTION / "prompts/corrected.txt"),
                    schema_sha256=prior.sha(prior.CORRECTION / "schema.json"),
                    code_sha256=prior.sha(Path(__file__)),
                    prior_code_sha256=prior.sha(Path(prior.__file__)),
                    tokenizer_provenance=provenance,
                    smoke_sha256={name: prior.sha(prior.CORRECTION / name) for name in
                                  ("synthetic_specs.json", "synthetic/corrected/train.jsonl",
                                   "synthetic/corrected/validation.jsonl")},
                    versions=dict(transformers=transformers.__version__, tokenizers=tokenizers_package.__version__),
                    examples=[{key: value for key, value in item.items() if key not in ("input", "messages")}
                              for item in items])
    prior.dump(output / "manifest.json", manifest)
    statistics = dict(counts=dict(Counter(item["split"] for item in items)),
                      buckets={split: dict(Counter(item["bucket"] for item in items if item["split"] == split))
                               for split in prior.BUCKETS},
                      domains=dict(Counter(item["domain"] for item in items)),
                      families={split: sum(value == split for value in split_families.values())
                                for split in prior.BUCKETS},
                      clean={split: sum(item["clean"] for item in items if item["split"] == split)
                             for split in prior.BUCKETS},
                      transformations=dict(Counter(tag for item in items for tag in item["transformations"])),
                      tokens={split: {field: prior.quantiles([row[field] for row in token_counts if row["split"] == split])
                                      for field in ("input", "supervised", "total")}
                              for split in prior.BUCKETS},
                      max_total=max(row["total"] for row in token_counts),
                      duplicate_inputs=0, split_leakage=0,
                      near_duplicate_candidates=prior.near_duplicate_candidates(items))
    prior.dump(output / "statistics.json", statistics)
    prior.dump(output / "distribution.json", dict(original=original_profile,
                                                  new_training=train_profile,
                                                  new_validation=validation_profile,
                                                  below_original_min_characters={"train": sum(len(text) < original_profile["characters"]["min"] for text in new_train),
                                                                                 "validation": sum(len(text) < original_profile["characters"]["min"] for text in new_validation)},
                                                  below_original_min_lines={"train": sum(len(text.splitlines()) < original_profile["lines"]["min"] for text in new_train),
                                                                            "validation": sum(len(text.splitlines()) < original_profile["lines"]["min"] for text in new_validation)}))
    prior.dump(output / "review.json", dict(status="pending-assistant-review", reviewed_validation=[],
                                             reviewed_training=[], findings=[], unresolved=[]))
    (output / "README.md").write_text("# CA1 synthetic SFT pilot v2\n\n"
        "**Status: pending assistant review; do not train.** This replacement responds to the v1 distribution audit. "
        "It keeps the corrected 16/8 smoke pairs unchanged and adds fictitious full-document parser reports. "
        "All targets follow visible CA1 evidence. No external parser/model, training, download or GPU was used.\n\n"
        "Build from the repository root with `/opt/venv/bin/python -m research.extraction.generate_sft_pilot_v2`. "
        "The command refuses to overwrite an existing packet. See `distribution.json` for the original/new input comparison, "
        "`manifest.json` for provenance and visible-cell evidence, and `statistics.json` for coverage and token counts.\n")
    prior.dump(output / "SHA256SUMS.json", {path.name: prior.sha(path) for path in output.iterdir()
                                              if path.is_file() and path.name != "SHA256SUMS.json"})
    print(json.dumps(dict(statistics=statistics, distribution=prior.load(output / "distribution.json")),
                     ensure_ascii=False, indent=2))


def verify_packet(output=OUTPUT):
    manifest = prior.load(output / "manifest.json")
    checksums = prior.load(output / "SHA256SUMS.json")
    files = {path.name for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS.json"}
    if set(checksums) != files or any(prior.sha(output / name) != digest for name, digest in checksums.items()):
        raise ValueError("Packet checksum mismatch")
    if manifest["code_sha256"] != prior.sha(Path(__file__)) or manifest["prior_code_sha256"] != prior.sha(Path(prior.__file__)):
        raise ValueError("Generator changed")
    if manifest["prompt_sha256"] != prior.sha(prior.CORRECTION / "prompts/corrected.txt") or manifest["schema_sha256"] != prior.sha(prior.CORRECTION / "schema.json"):
        raise ValueError("Approved prompt or schema changed")
    if any(prior.sha(prior.CORRECTION / name) != digest for name, digest in manifest["smoke_sha256"].items()):
        raise ValueError("Smoke source changed")
    tokenizer, provenance = prior.tokenizer_and_hashes()
    if provenance != manifest["tokenizer_provenance"]:
        raise ValueError("Tokenizer provenance changed")
    schema = prior.load(prior.CORRECTION / "schema.json")
    examples = {item["id"]: item for item in manifest["examples"]}
    tokens = {row["id"]: row for row in map(json.loads, (output / "tokenized.jsonl").read_text().splitlines())}
    seen = set()
    for split, expected in (("train", 320), ("validation", 64)):
        rows = [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
        if len(rows) != expected:
            raise ValueError("Split count mismatch")
        for row in rows:
            identifier, messages = row["id"], row["messages"]
            if identifier in seen or identifier not in examples or examples[identifier]["split"] != split:
                raise ValueError("ID or split mismatch")
            seen.add(identifier)
            if [message["role"] for message in messages] != ["system", "user", "assistant"]:
                raise ValueError("Role mismatch")
            if messages[0]["content"] != (prior.CORRECTION / "prompts/corrected.txt").read_text():
                raise ValueError("Prompt mismatch")
            if not messages[1]["content"].startswith("Extract laboratory observations from this parser output:\n<report>\n") or not messages[1]["content"].endswith("\n</report>\n"):
                raise ValueError("Wrapper mismatch")
            strict_validate(messages[2]["content"], schema)
            if prior.sha(messages[2]["content"]) != examples[identifier]["target_sha256"]:
                raise ValueError("Target mismatch")
            ids, labels, _, _ = prior.tokenize(tokenizer, messages)
            if tokens[identifier] != dict(id=identifier, split=split, input_ids=ids, labels=labels):
                raise ValueError("Assistant label mismatch")
    if seen != set(examples) or seen != set(tokens):
        raise ValueError("Membership mismatch")
    print(f"Verified {len(seen)} CA1 pairs, hashes, schema, chat boundaries and labels.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    verify_packet(args.output) if args.verify else build(args.output)


if __name__ == "__main__":
    main()
