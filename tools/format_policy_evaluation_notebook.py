import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "policy_evaluation.ipynb"
OUT_PATH = ROOT / "formatted_notebooks" / "06_policy_evaluation.ipynb"

STYLE_INFO = "max-width: 99%; padding:10px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"
STYLE_SUCCESS = "background-color:#e6ffe6; max-width: 99%; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"
STYLE_WARN = "background-color:#fff8e6; max-width: 99%; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"
STYLE_NOTE = "background-color:#f8f8fb; max-width: 99%; padding:10px; border-width:2px; border-color:#efe6ef; border-style:solid; border-radius:6px"


SECTION_ANCHORS = {
    "## 0. Импорты и настройки": "policy-imports",
    "## 1. Загрузка предсказанных скоров": "policy-load",
    "## 2. Экономическая постановка задачи": "policy-economics",
    "## 3. Семь стратегий коммуникации": "policy-strategies",
    "## 4. Кривые \"бюджет - предотвращенные дефолты\"": "policy-budget-curves",
    "## 5. Сравнение стратегий при фиксированных уровнях охвата": "policy-fixed-coverage",
    "## 6. ROI-кривая: при каком охвате стратегия окупается?": "policy-roi",
    "## 7. Multi-treatment policy: какой канал выбрать для каждого клиента?": "policy-multitreatment",
    "## 8. Сегментация клиентов: Persuadables, Sure Things, Sleeping Dogs, Lost Causes": "policy-segments",
    "## 9. Bootstrap доверительные интервалы": "policy-bootstrap",
    "## 10. Итоговые выводы и практические рекомендации": "policy-conclusions",
}

SUBSECTION_ANCHORS = {
    "### Выводы по разделу 4": "policy-budget-summary",
    "### Выводы по разделу 5": "policy-fixed-summary",
    "### Выводы по разделу 6": "policy-roi-summary",
    "### Выводы по разделу 7": "policy-multitreatment-summary",
    "### Выводы по разделу 8": "policy-segments-summary",
    "### Выводы по разделу 9": "policy-bootstrap-summary",
}


def source_lines(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    return text.splitlines(keepends=True)


def box(kind: str, body: str) -> str:
    styles = {
        "info": STYLE_INFO,
        "success": STYLE_SUCCESS,
        "warning": STYLE_WARN,
        "note": STYLE_NOTE,
    }
    classes = {
        "info": "alert alert-info",
        "success": "alert alert-success",
        "warning": "alert alert-warning",
        "note": "alert alert-secondary",
    }
    return f'<div class="{classes[kind]}" style="{styles[kind]}">\n\n{body.strip()}\n\n</div>'


def top_cell() -> dict:
    body = f"""
# <h1><center id="policy-top">Оценка бизнес-ценности политик коммуникации</center></h1>

### Оглавление
* <a href="#policy-imports">0. Импорты и настройки</a>
* <a href="#policy-load">1. Загрузка предсказанных скоров</a>
* <a href="#policy-economics">2. Экономическая постановка</a>
* <a href="#policy-strategies">3. Стратегии коммуникации</a>
* <a href="#policy-budget-curves">4. Бюджет и предотвращенные дефолты</a>
* <a href="#policy-fixed-coverage">5. Фиксированные уровни охвата</a>
* <a href="#policy-roi">6. ROI-кривая</a>
* <a href="#policy-multitreatment">7. Multi-treatment policy</a>
* <a href="#policy-segments">8. Сегментация клиентов</a>
* <a href="#policy-bootstrap">9. Bootstrap-интервалы</a>
* <a href="#policy-conclusions">10. Практические рекомендации</a>

{box("info", """
<b>Роль ноутбука.</b> Этот финальный слой переводит модельные скоры в бизнес-решения: кого контактировать, каким каналом, при каком бюджете и какой чистый эффект можно ожидать.
""")}

{box("success", """
<b>Как читать результаты.</b> Метрики качества моделей здесь становятся управленческими показателями: предотвращенные дефолты, стоимость контактов, ROI, сегменты клиентов и доверительные интервалы.
""")}

{box("warning", """
<b>Ограничение.</b> Экономические параметры сценарные. Они показывают метод расчета, но перед внедрением должны быть заменены на реальные стоимости каналов, LGD/PD-экономику и операционные ограничения банка.
""")}
"""
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(body)}


def heading_from_text(text: str) -> tuple[str | None, str]:
    lines = text.strip().splitlines()
    if not lines:
        return None, ""
    first = lines[0].strip()
    rest = "\n".join(lines[1:]).strip()
    if first in SECTION_ANCHORS:
        return f'<h2 id="{SECTION_ANCHORS[first]}">{first.removeprefix("## ").strip()}</h2>', rest
    if first in SUBSECTION_ANCHORS:
        return f'<h3 id="{SUBSECTION_ANCHORS[first]}">{first.removeprefix("### ").strip()}</h3>', rest
    return None, text.strip()


def format_markdown(text: str, index: int) -> str:
    stripped = text.strip()
    if not stripped:
        return text
    if index == 0:
        return ""

    heading, rest = heading_from_text(stripped)
    if heading:
        kind = "info"
        if "Выводы" in heading or "Итоговые выводы" in heading:
            kind = "success"
        if "Экономическая постановка" in heading or "ROI" in heading:
            kind = "note"
        return f"{heading}\n\n{box(kind, rest)}" if rest else heading

    return stripped


def inserted_after_original(index: int) -> str | None:
    if index == 7:
        return box(
            "note",
            "<b>Как читать экономические параметры.</b> Стоимость канала и ценность предотвращенного дефолта задают сценарий. При изменении этих параметров ранжирование моделей может остаться прежним, а оптимальный охват и ROI изменятся.",
        )
    if index == 12:
        return box(
            "warning",
            "<b>Кандидат на уплотнение вывода.</b> Кривые бюджет-эффект — самый тяжелый output ноутбука. Для финальной версии можно оставить одну итоговую фигуру и таблицу значений при 10%, 20%, 30% охвата.",
        )
    if index == 18:
        return box(
            "warning",
            "<b>Кандидат на уплотнение вывода.</b> ROI-график полезен, но лучше читается вместе с одной строкой: где у каждой стратегии максимум чистого эффекта и после какого охвата предельная выгода падает.",
        )
    if index == 23:
        return box(
            "note",
            "<b>Практический смысл multi-treatment.</b> Если модель почти всегда выбирает один канал, это сигнал не только о модели, но и о данных: альтернативные каналы могут быть плохо покрыты исторической политикой.",
        )
    if index == 28:
        return box(
            "warning",
            "<b>Кандидат на уплотнение вывода.</b> Сегментационная визуализация хороша для объяснения, но ее можно сжать до долей сегментов и top-20% концентрации Persuadables.",
        )
    if index == 32:
        return box(
            "note",
            "<b>Как читать bootstrap.</b> Здесь важны не только средние значения, но и пересечение интервалов. Если интервалы двух стратегий сильно перекрываются, бизнес-разницу стоит трактовать осторожно.",
        )
    return None


def main() -> None:
    original = json.loads(SRC_PATH.read_text(encoding="utf-8"))
    formatted = copy.deepcopy(original)

    new_cells = [top_cell()]
    for index, cell in enumerate(formatted["cells"]):
        if index == 0 and cell.get("cell_type") == "markdown":
            continue

        cell = copy.deepcopy(cell)
        if cell.get("cell_type") == "markdown":
            cell["source"] = source_lines(format_markdown("".join(cell.get("source", [])), index))
        new_cells.append(cell)

        inserted = inserted_after_original(index)
        if inserted:
            new_cells.append({"cell_type": "markdown", "metadata": {}, "source": source_lines(inserted)})

    formatted["cells"] = new_cells
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(formatted, ensure_ascii=False, indent=1), encoding="utf-8")

    original_code = [cell for cell in original["cells"] if cell.get("cell_type") == "code"]
    formatted_code = [cell for cell in formatted["cells"] if cell.get("cell_type") == "code"]
    if original_code != formatted_code:
        raise RuntimeError("Code cells changed during formatting")

    print(f"written: {OUT_PATH}")
    print(f"original cells: {len(original['cells'])}; formatted cells: {len(formatted['cells'])}")
    print(f"code cells preserved: {len(original_code)}")


if __name__ == "__main__":
    main()
