import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "logistic_regression.ipynb"
OUT_PATH = ROOT / "formatted_notebooks" / "04_logistic_regression.ipynb"

STYLE_INFO = "max-width: 99%; padding:10px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"
STYLE_SUCCESS = "background-color:#e6ffe6; max-width: 99%; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"
STYLE_WARN = "background-color:#fff8e6; max-width: 99%; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"
STYLE_NOTE = "background-color:#f8f8fb; max-width: 99%; padding:10px; border-width:2px; border-color:#efe6ef; border-style:solid; border-radius:6px"


SECTION_ANCHORS = {
    "## 1. Загрузка набора данных": "lr-load",
    "## 2. Первичный осмотр данных": "lr-inspection",
    "## 3. Генерация производных признаков": "lr-features",
    "## 4. Предобработка набора данных": "lr-preprocess",
    "## 5. Разделение данных на обучающую, тестовую и вневременную выборки": "lr-split",
    "## 6. Отбор признаков": "lr-selection",
    "## 8. Обучение базовой логистической регрессии": "lr-base-model",
    "## 9. Подбор гиперпараметров логистической регрессии": "lr-optuna",
    "## 10. Оценка качества финальной модели": "lr-final-quality",
    "## 11. Валидация устойчивости на кросс-валидации": "lr-cv",
    "## 12. Сохранение результатов": "lr-artifacts",
    "## 13. Qini-кривая и AUUC": "lr-qini",
    "## 14. Зависимость качества модели от объема обучающих данных": "lr-learning-curve",
    "## 15. Итоговые выводы": "lr-conclusions",
}

SUBSECTION_ANCHORS = {
    "### 6.1 Информационная ценность (IV) и Solo Gini": "lr-iv-gini",
    "### 7.2 WoE-кодирование и фильтрация по корреляции": "lr-woe-correlation",
    "### 7.3 Пошаговый отбор признаков": "lr-stepwise",
    "### 7.4 Итоговый список отобранных признаков": "lr-selected-features",
    "### 9.1 Подбор гиперпараметров биннинга WoE (BinningProcess)": "lr-binning-optuna",
    "### 9.2 Подбор гиперпараметров модели LogisticRegression (Optuna, TPE, 20 испытаний)": "lr-model-optuna",
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
# <h1><center id="lr-top">Logistic Regression + WoE baseline</center></h1>

### Оглавление
* <a href="#lr-load">1. Загрузка набора данных</a>
* <a href="#lr-inspection">2. Первичный осмотр данных</a>
* <a href="#lr-features">3. Производные признаки</a>
* <a href="#lr-preprocess">4. Предобработка и leakage-контроль</a>
* <a href="#lr-split">5. Train/Test/OOT-разбиение</a>
* <a href="#lr-selection">6. Отбор признаков</a>
* <a href="#lr-iv-gini">6.1 IV и Solo Gini</a>
* <a href="#lr-woe-correlation">7.2 WoE и корреляционная фильтрация</a>
* <a href="#lr-stepwise">7.3 Пошаговый отбор</a>
* <a href="#lr-selected-features">7.4 Итоговые признаки</a>
* <a href="#lr-base-model">8. Базовая логистическая регрессия</a>
* <a href="#lr-optuna">9. Подбор гиперпараметров</a>
* <a href="#lr-final-quality">10. Финальная оценка качества</a>
* <a href="#lr-cv">11. Кросс-валидация</a>
* <a href="#lr-artifacts">12. Сохранение результатов</a>
* <a href="#lr-qini">13. Qini/AUUC как risk-based стратегия</a>
* <a href="#lr-learning-curve">14. Кривая обучения</a>
* <a href="#lr-conclusions">15. Итоговые выводы</a>

{box("info", """
<b>Роль ноутбука.</b> Здесь строится классический банковский baseline: логистическая регрессия с WoE-кодированием. Она уступает CatBoost по гибкости, но выигрывает в прозрачности, устойчивости и простоте объяснения.
""")}

{box("success", """
<b>Как читать результаты.</b> Главная линия ноутбука: безопасное признаковое пространство → IV/Solo Gini → WoE-биннинг → корреляционная фильтрация → компактная модель → сравнение с CatBoost и uplift-подходом.
""")}

{box("warning", """
<b>Граница интерпретации.</b> Как и CatBoost, LogReg прогнозирует риск дефолта, а не причинный эффект коммуникации. Qini/AUUC здесь показывают только то, насколько risk-score случайно совпадает с uplift-ранжированием.
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


def format_markdown(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return text

    heading, rest = heading_from_text(stripped)
    if heading:
        kind = "info"
        if "Итоговые выводы" in heading:
            kind = "success"
        return f"{heading}\n\n{box(kind, rest)}" if rest else heading

    if stripped.startswith("> **Примечание.**"):
        clean = stripped.replace("> ", "")
        return box("warning", clean)

    if stripped.startswith("**Матрица корреляций"):
        return box("warning", stripped)

    if stripped.startswith("**Рисунок"):
        return box("note", stripped)

    return stripped


def inserted_after_original(index: int) -> str | None:
    if index == 7:
        return box(
            "warning",
            "<b>Кандидат на уплотнение вывода.</b> Полный вывод `df` оставлен для аудита, но в финальной версии его можно заменить сводкой: размерность, target-rate, число признаков по типам и список исключенных uplift/leakage-полей.",
        )
    if index == 52:
        return box(
            "note",
            "<b>Как читать WoE-биннинг.</b> Важны не все технические строки `BinningProcess`, а устойчивость бинов и отсутствие слишком мелких групп. Подробные таблицы полезны для аудита, но в тексте лучше оставить краткое резюме по финальным признакам.",
        )
    if index == 71:
        return box(
            "note",
            "<b>Как читать пошаговый отбор.</b> Смысл блока — найти точку, где добавление переменных почти перестает повышать test Gini. Для LogReg это особенно важно: компактность модели напрямую влияет на интерпретируемость.",
        )
    if index == 78:
        return box(
            "warning",
            "<b>Кандидат на уплотнение вывода.</b> Большая корреляционная/диагностическая визуализация полезна для проверки, но для читателя достаточно оставить финальный список признаков и 2-3 ключевых WoE-графика.",
        )
    if index == 114:
        return box(
            "note",
            "<b>Методологический смысл.</b> Qini/AUUC здесь не превращают LogReg в uplift-модель. Они показывают, насколько простая риск-ориентация может конкурировать с индивидуальной оценкой эффекта.",
        )
    if index == 116:
        return box(
            "warning",
            "<b>Кандидат на уплотнение вывода.</b> Learning curve можно оставить как итоговую картинку и короткую таблицу точек насыщения. Полный output тяжелый и мало добавляет к интерпретации.",
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
