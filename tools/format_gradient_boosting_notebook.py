import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "gradient_boosting.ipynb"
OUT_PATH = ROOT / "formatted_notebooks" / "03_gradient_boosting.ipynb"

STYLE_INFO = "max-width: 99%; padding:10px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"
STYLE_SUCCESS = "background-color:#e6ffe6; max-width: 99%; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"
STYLE_WARN = "background-color:#fff8e6; max-width: 99%; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"
STYLE_NOTE = "background-color:#f8f8fb; max-width: 99%; padding:10px; border-width:2px; border-color:#efe6ef; border-style:solid; border-radius:6px"


SECTION_ANCHORS = {
    "## 1. Загрузка набора данных": "gb-load",
    "## 2. Первичный осмотр данных": "gb-inspection",
    "## 3. Конструирование производных признаков": "gb-features",
    "## 4. Предобработка данных": "gb-preprocess",
    "## 5. Разделение данных на обучающую, тестовую и вневременную выборки": "gb-split",
    "## 6. Отбор признаков": "gb-selection",
    "## 7. Базовая модель CatBoost": "gb-base-model",
    "## 8. Подбор гиперпараметров методом байесовской оптимизации": "gb-optuna",
    "## 9. Обучение финальной модели и кросс-валидационная оценка": "gb-final-model",
    "## 10. Сохранение артефактов модели": "gb-artifacts",
    "## 13. Оценка как стратегии упреждающего воздействия: Qini-кривая и AUUC": "gb-qini",
    "## 14. Кривая обучения: зависимость качества от объема данных": "gb-learning-curve",
    "## 15. Итоговые выводы": "gb-conclusions",
}

SUBSECTION_ANCHORS = {
    "### 6.1 Оценка значимости признаков": "gb-selection-importance",
    "### 6.2 Фильтрация по корреляции": "gb-selection-correlation",
    "### 6.3 Переоценка значимости на очищенном признаковом пространстве": "gb-selection-reestimate",
    "### 6.4 Последовательный отбор признаков": "gb-selection-forward",
    "### 7.1 SHAP-интерпретация базовой модели": "gb-shap",
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
# <h1><center id="gb-top">CatBoost baseline для кредитного риска</center></h1>

### Оглавление
* <a href="#gb-load">1. Загрузка набора данных</a>
* <a href="#gb-inspection">2. Первичный осмотр данных</a>
* <a href="#gb-features">3. Производные признаки</a>
* <a href="#gb-preprocess">4. Предобработка и защита от утечки</a>
* <a href="#gb-split">5. Train/Test/OOT-разбиение</a>
* <a href="#gb-selection">6. Отбор признаков</a>
* <a href="#gb-base-model">7. Базовая модель CatBoost</a>
* <a href="#gb-optuna">8. Подбор гиперпараметров</a>
* <a href="#gb-final-model">9. Финальная модель и CV</a>
* <a href="#gb-artifacts">10. Сохранение артефактов</a>
* <a href="#gb-qini">13. Qini/AUUC как risk-based стратегия</a>
* <a href="#gb-learning-curve">14. Кривая обучения</a>
* <a href="#gb-conclusions">15. Итоговые выводы</a>

{box("info", """
<b>Роль ноутбука.</b> Здесь строится сильный риск-ориентированный baseline на CatBoost. Модель прогнозирует вероятность дефолта, а не индивидуальный эффект коммуникации, поэтому дальше она нужна как честный соперник uplift-подхода.
""")}

{box("success", """
<b>Как читать результаты.</b> Сначала проверяется безопасное признаковое пространство, затем признаки отбираются и интерпретируются через CatBoost/SHAP, после чего модель оценивается на train, test и OOT и сравнивается с uplift-логикой через Qini/AUUC.
""")}

{box("warning", """
<b>Граница интерпретации.</b> Высокий Gini означает хорошее ранжирование риска. Это не доказывает, что выбранные клиенты сильнее реагируют на коммуникацию: для этого нужны uplift-модели и policy evaluation.
""")}
"""
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(body)}


def heading_from_text(text: str) -> tuple[str | None, str, bool]:
    lines = text.strip().splitlines()
    had_rule = False
    while lines and lines[0].strip() == "---":
        had_rule = True
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines = lines[1:]
    if not lines:
        return None, "", had_rule
    first = lines[0].strip()
    rest = "\n".join(lines[1:]).strip()
    if first in SECTION_ANCHORS:
        return f'<h2 id="{SECTION_ANCHORS[first]}">{first.removeprefix("## ").strip()}</h2>', rest, had_rule
    if first in SUBSECTION_ANCHORS:
        return f'<h3 id="{SUBSECTION_ANCHORS[first]}">{first.removeprefix("### ").strip()}</h3>', rest, had_rule
    return None, text.strip(), had_rule


def format_markdown(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return text

    heading, rest, had_rule = heading_from_text(stripped)
    if heading:
        kind = "info"
        if "Итоговые выводы" in heading:
            kind = "success"
        prefix = "---\n\n" if had_rule else ""
        return prefix + (f"{heading}\n\n{box(kind, rest)}" if rest else heading)

    if stripped.startswith("> **Предупреждение об утечке данных.**"):
        clean = stripped.replace("> ", "")
        return box("warning", clean)

    if stripped.startswith("**Проверка на утечку данных"):
        return box("warning", stripped)

    if stripped.startswith("**Рисунок"):
        return box("note", stripped)

    if stripped.startswith("---"):
        return stripped

    return stripped


def inserted_after_original(index: int) -> str | None:
    if index == 7:
        return box(
            "warning",
            "<b>Кандидат на уплотнение вывода.</b> Полный вывод `df` сохранен для аудита, но в финальной читательской версии его можно заменить на компактную таблицу: размерность, число признаков по типам, target-rate и список исключенных leakage-полей.",
        )
    if index == 61:
        return box(
            "note",
            "<b>Как читать forward selection.</b> Важна не каждая строка логов, а форма кривой: где тестовый Gini выходит на плато и где начинает расти разрыв train/test. Именно это определяет разумное число признаков.",
        )
    if index == 77:
        return box(
            "warning",
            "<b>Кандидат на уплотнение вывода.</b> SHAP beeswarm информативен, но тяжелый. Для финальной версии можно оставить сам график и краткий список 5-7 главных драйверов риска, а технический SVG/output не пересчитывать без необходимости.",
        )
    if index == 102:
        return box(
            "note",
            "<b>Методологический смысл.</b> Этот блок показывает предел risk-based стратегии: CatBoost может ранжировать клиентов по риску, но Qini/AUUC проверяет, насколько такое ранжирование совпадает с эффектом коммуникации.",
        )
    if index == 104:
        return box(
            "warning",
            "<b>Кандидат на уплотнение вывода.</b> Learning curve полезна для диссертационного сравнения объема данных, но вывод большой. Можно оставить итоговую кривую и таблицу ключевых точек: 0.5%, 10%, 35%, 70%, 100%.",
        )
    return None


def main() -> None:
    original = json.loads(SRC_PATH.read_text(encoding="utf-8"))
    formatted = copy.deepcopy(original)

    new_cells = [top_cell()]
    for index, cell in enumerate(formatted["cells"]):
        cell = copy.deepcopy(cell)
        if cell.get("cell_type") == "markdown":
            cell["source"] = source_lines(format_markdown("".join(cell.get("source", []))))
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
