import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "analytic_dataset.ipynb"
OUT_PATH = ROOT / "formatted_notebooks" / "02_analytic_dataset.ipynb"

STYLE_INFO = "max-width: 99%; padding:10px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"
STYLE_SUCCESS = "background-color:#e6ffe6; max-width: 99%; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"
STYLE_WARN = "background-color:#fff8e6; max-width: 99%; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"
STYLE_NOTE = "background-color:#f8f8fb; max-width: 99%; padding:10px; border-width:2px; border-color:#efe6ef; border-style:solid; border-radius:6px"


SECTION_ANCHORS = {
    "## 1. Загрузка набора данных": "audit-load",
    "## 2. Верификация синтетически сгенерированных переменных": "audit-verification",
    "## 3. Финальные выводы о качестве синтетического набора данных": "audit-final",
}

SUBSECTION_ANCHORS = {
    "### 1.1 Постановка причинно-следственной задачи": "audit-causal-task",
    "### 2.1 Анализ BASE_PD — базовой вероятности дефолта": "audit-base-pd",
    "### 2.2 Анализ CONTACT_PROPENSITY — склонность к контакту": "audit-contact-propensity",
    "### 2.3 Анализ COMMUNICATION — типа воздействия (систематическое смещение отбора)": "audit-communication",
    "### 2.4 Анализ TRUE_UPLIFT — истинного каузального эффекта": "audit-true-uplift",
    "### 2.5 Анализ систематического смещения отбора": "audit-selection-bias",
    "### 2.6 Проверка потенциальных исходов": "audit-potential-outcomes",
    "### 2.7 Oracle policy: две верхние границы качества": "audit-oracle",
    "### 2.8 Аудит утечки данных": "audit-leakage",
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
# <h1><center id="audit-top">Аудит синтетического uplift-набора данных</center></h1>

### Оглавление
* <a href="#audit-load">1. Загрузка набора данных</a>
* <a href="#audit-causal-task">1.1 Причинно-следственная постановка</a>
* <a href="#audit-verification">2. Верификация синтетических переменных</a>
* <a href="#audit-base-pd">2.1 BASE_PD: базовый риск</a>
* <a href="#audit-contact-propensity">2.2 CONTACT_PROPENSITY: склонность к контакту</a>
* <a href="#audit-communication">2.3 COMMUNICATION: неслучайное назначение канала</a>
* <a href="#audit-true-uplift">2.4 TRUE_UPLIFT: истинный эффект</a>
* <a href="#audit-selection-bias">2.5 Selection bias</a>
* <a href="#audit-potential-outcomes">2.6 Потенциальные исходы</a>
* <a href="#audit-oracle">2.7 Oracle policy</a>
* <a href="#audit-leakage">2.8 Аудит утечки</a>
* <a href="#audit-final">3. Финальные выводы</a>

{box("info", """
<b>Задача ноутбука.</b> Проверить, что датасет из `prepare_dataset.ipynb` действительно подходит для uplift-эксперимента: в нем есть базовый кредитный риск, неслучайная политика контакта, неоднородный эффект коммуникаций, потенциальные исходы и явная защита от leakage.
""")}

{box("success", """
<b>Логика чтения.</b> Аудит идет от причинной постановки к диагностике переменных: сначала риск без коммуникации, затем склонность к контакту, фактические каналы, истинный эффект, смещение отбора, oracle-ориентиры и запретные признаки.
""")}

{box("warning", """
<b>Ограничение.</b> Большие графические outputs сохранены как часть исследовательского следа. Я отдельно отмечаю места, где для финальной учебной версии можно заменить перегруженный вывод более компактной сводкой.
""")}
"""
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(body)}


def format_heading(text: str) -> str | None:
    stripped = text.strip()
    if stripped in SECTION_ANCHORS:
        anchor = SECTION_ANCHORS[stripped]
        title = stripped.removeprefix("## ").strip()
        return f'<h2 id="{anchor}">{title}</h2>'
    if stripped in SUBSECTION_ANCHORS:
        anchor = SUBSECTION_ANCHORS[stripped]
        title = stripped.removeprefix("### ").strip()
        return f'<h3 id="{anchor}">{title}</h3>'
    return None


def format_markdown(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return text

    first_line = stripped.splitlines()[0].strip()
    heading = format_heading(first_line)
    rest = "\n".join(stripped.splitlines()[1:]).strip()

    if heading:
        if rest:
            kind = "info"
            if "Финальные выводы" in heading:
                kind = "success"
            return f"{heading}\n\n{box(kind, rest)}"
        return heading

    if stripped.startswith("### Выводы по"):
        return box("success", stripped)

    if stripped.startswith("**Ожидаемый результат.**"):
        return box("success", stripped)

    if stripped.startswith("**Рисунок"):
        return box("note", stripped)

    if stripped.startswith("**Интерпретация.**") or stripped.startswith("**Что показывает heatmap.**"):
        return box("info", stripped)

    if stripped.startswith("**Почему это важно.**") or stripped.startswith("**Вывод по oracle policy.**"):
        return box("warning", stripped)

    if stripped.startswith("**Средние значения") or stripped.startswith("**Распределение клиентов") or stripped.startswith("**Кросс-табуляция") or stripped.startswith("**Средний TRUE_UPLIFT"):
        return box("note", stripped)

    if stripped.startswith("**Ключевые итоги аудита:**"):
        return box("success", stripped)

    if "Зафиксируем зерно генератора" in stripped:
        return box(
            "warning",
            "<b>Фиксация случайности.</b> Эта ячейка закрепляет генератор NumPy, чтобы диагностические расчеты и визуализации воспроизводились при повторном запуске.",
        )

    if "Загрузим подготовленный набор данных" in stripped:
        return box(
            "info",
            "<b>Входной артефакт.</b> Загружается `data/processed/uplift-dataset.csv`, созданный в ноутбуке подготовки данных. Все дальнейшие проверки относятся именно к этому файлу.",
        )

    return stripped


def inserted_after_original(index: int) -> str | None:
    if index == 12:
        return box(
            "warning",
            "<b>Кандидат на уплотнение вывода.</b> Полный просмотр `df` полезен для первичного аудита, но в финальной версии его можно заменить на компактную сводку: размерность, количество типов признаков и список ключевых uplift-полей.",
        )
    if index == 58:
        return box(
            "success",
            "<b>Статистическая диагностика.</b> Следующий блок проверяет различия между contacted/control-группами до воздействия. Значимые p-value подтверждают, что сравнивать outcome напрямую нельзя: группы уже отличаются по исходному риску.",
        )
    if index == 59:
        return box(
            "warning",
            "<b>Кандидат на уплотнение вывода.</b> Рисунок 12 очень информативный, но тяжелый. Для защиты или статьи можно оставить 2-3 ключевых распределения, а полный набор вынести в приложение.",
        )
    if index == 78:
        return box(
            "success",
            "<b>Практический контроль.</b> Leakage audit должен стать обязательным переходом к следующим ноутбукам: baseline-модели и uplift-модели обязаны использовать только разрешенные признаки.",
        )
    return None


def main() -> None:
    original = json.loads(SRC_PATH.read_text(encoding="utf-8"))
    formatted = copy.deepcopy(original)

    new_cells = [top_cell()]
    for index, cell in enumerate(formatted["cells"]):
        cell = copy.deepcopy(cell)
        if cell.get("cell_type") == "markdown":
            text = "".join(cell.get("source", []))
            cell["source"] = source_lines(format_markdown(text))
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
