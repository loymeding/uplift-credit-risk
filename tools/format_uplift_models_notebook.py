import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "uplift_models.ipynb"
OUT_PATH = ROOT / "formatted_notebooks" / "05_uplift_models.ipynb"

STYLE_INFO = "max-width: 99%; padding:10px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"
STYLE_SUCCESS = "background-color:#e6ffe6; max-width: 99%; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"
STYLE_WARN = "background-color:#fff8e6; max-width: 99%; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"
STYLE_NOTE = "background-color:#f8f8fb; max-width: 99%; padding:10px; border-width:2px; border-color:#efe6ef; border-style:solid; border-radius:6px"


SECTION_ANCHORS = {
    "## 0. Импорты и настройки воспроизводимости": "uplift-imports",
    "## Загрузка данных и первичная проверка": "uplift-load",
    "## Производные признаки": "uplift-features",
    "## Разбиение на обучающую, тестовую и вневременную выборки": "uplift-split",
    "## Базовые модели риска": "uplift-risk-baselines",
    "## Диагностика сопоставимости групп": "uplift-overlap",
    "## Метрики Qini и AUUC": "uplift-qini-metrics",
    "## Базовые стратегии отбора клиентов": "uplift-basic-policies",
    "## Единая модель с признаком воздействия (S-Learner)": "uplift-s-learner",
    "## Две модели для групп воздействия и контроля (T-Learner)": "uplift-t-learner",
    "## Уточненная двухэтапная модель (X-Learner)": "uplift-x-learner",
    "## Модель с двойной робастностью (DR-Learner)": "uplift-dr-learner",
    "## Контрольная проверка: согласованность риска и эффекта": "uplift-risk-effect-check",
    "## Сравнение моделей": "uplift-model-comparison",
    "## Оценка бизнес-потенциала": "uplift-business-potential",
    "## Проверка статистической устойчивости": "uplift-stat-stability",
    "## Проверка по известному истинному эффекту": "uplift-true-effect",
    "## Зависимость качества от объема обучающей выборки": "uplift-learning-curve",
    "## Проверка при разных сценариях связи риска и эффекта": "uplift-scenarios",
    "## Итоговые выводы": "uplift-conclusions",
}

SUBSECTION_ANCHORS = {
    "### Подготовка признакового пространства для базовых моделей": "uplift-risk-feature-space",
    "### Выводы по загрузке данных": "uplift-load-summary",
    "### Выводы по производным признакам": "uplift-features-summary",
    "### Выводы по разбиению выборки": "uplift-split-summary",
    "### Выводы по базовым риск-моделям": "uplift-risk-summary",
    "### Выводы по диагностике сопоставимости групп": "uplift-overlap-summary",
    "### Выводы по базовым стратегиям отбора": "uplift-policy-summary",
    "### Выводы по S-Learner": "uplift-s-summary",
    "### Выводы по T-Learner": "uplift-t-summary",
    "### Выводы по X-Learner": "uplift-x-summary",
    "### Выводы по DR-Learner": "uplift-dr-summary",
    "### Выводы по сравнительной таблице моделей": "uplift-comparison-summary",
    "### Выводы по бизнес-потенциалу": "uplift-business-summary",
    "### Выводы по статистической устойчивости": "uplift-stability-summary",
    "### Выводы по проверке на истинном эффекте": "uplift-true-summary",
    "### Выводы по сохранению оценок моделей": "uplift-save-summary",
    "### Выводы по зависимости качества от объема данных": "uplift-learning-summary",
    "### Визуальная проверка сценариев": "uplift-scenario-visual",
    "### Как интерпретировать сценарии": "uplift-scenario-howto",
    "### Выводы по сценарному анализу связи риска и эффекта": "uplift-scenario-summary",
    "### Интерпретация сценарного эксперимента": "uplift-scenario-interpretation",
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
# <h1><center id="uplift-top">Модели индивидуального эффекта коммуникации</center></h1>

### Оглавление
* <a href="#uplift-imports">0. Импорты и воспроизводимость</a>
* <a href="#uplift-load">1. Загрузка данных</a>
* <a href="#uplift-features">2. Производные признаки</a>
* <a href="#uplift-split">3. Train/Test/OOT-разбиение</a>
* <a href="#uplift-risk-baselines">4. Базовые модели риска</a>
* <a href="#uplift-overlap">5. Диагностика сопоставимости групп</a>
* <a href="#uplift-qini-metrics">6. Метрики Qini и AUUC</a>
* <a href="#uplift-basic-policies">7. Базовые стратегии отбора</a>
* <a href="#uplift-s-learner">8. S-Learner</a>
* <a href="#uplift-t-learner">9. T-Learner</a>
* <a href="#uplift-x-learner">10. X-Learner</a>
* <a href="#uplift-dr-learner">11. DR-Learner</a>
* <a href="#uplift-model-comparison">12. Сравнение моделей</a>
* <a href="#uplift-business-potential">13. Бизнес-потенциал</a>
* <a href="#uplift-stat-stability">14. Статистическая устойчивость</a>
* <a href="#uplift-true-effect">15. Проверка по TRUE_UPLIFT</a>
* <a href="#uplift-learning-curve">16. Кривая обучения</a>
* <a href="#uplift-scenarios">17. Сценарный анализ риска и эффекта</a>
* <a href="#uplift-conclusions">18. Итоговые выводы</a>

{box("info", """
<b>Роль ноутбука.</b> Это экспериментальное ядро работы: здесь risk-based стратегии сравниваются с моделями индивидуального эффекта коммуникации и oracle-ориентирами.
""")}

{box("success", """
<b>Главная идея.</b> Хорошая risk-модель отвечает на вопрос «кто рискованнее», а uplift-модель — «чья вероятность дефолта изменится из-за коммуникации». Эти задачи могут совпадать только при определенной связи риска и эффекта.
""")}

{box("warning", """
<b>Граница интерпретации.</b> Oracle и `TRUE_UPLIFT` доступны только потому, что стенд синтетический. Они используются для проверки качества, но не должны попадать в обучение моделей.
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
    if first.startswith("#### "):
        return first, rest
    return None, text.strip()


def format_markdown(text: str, index: int) -> str:
    stripped = text.strip()
    if not stripped:
        return text

    if index == 0:
        # Replaced by top_cell().
        return ""

    heading, rest = heading_from_text(stripped)
    if heading:
        kind = "info"
        if "Выводы" in heading or "Интерпретация сценарного эксперимента" in heading:
            kind = "success"
        if "Итоговые выводы" in heading:
            kind = "success"
        if "Как интерпретировать сценарии" in heading:
            kind = "note"
        return f"{heading}\n\n{box(kind, rest)}" if rest else heading

    return stripped


def inserted_after_original(index: int) -> str | None:
    if index == 24:
        return box(
            "warning",
            "<b>Кандидат на уплотнение вывода.</b> Диагностика overlap/SMD важна методологически, но графический output тяжелый. Для финальной версии можно оставить ключевые SMD и propensity-график, а полный набор вынести в приложение.",
        )
    if index == 55:
        return box(
            "note",
            "<b>Как читать таблицу сравнения.</b> Смотрите не только на абсолютный AUUC, но и на группы моделей: random baseline, risk-based, meta-learners и oracle. Это не одна линейка алгоритмов, а разные уровни доступной информации.",
        )
    if index == 57:
        return box(
            "warning",
            "<b>Кандидат на уплотнение вывода.</b> Большой график сравнения моделей информативен, но для читателя достаточно итоговой диаграммы AUUC и короткого вывода о T-Learner/S-Learner против risk-based стратегий.",
        )
    if index == 67:
        return box(
            "note",
            "<b>Проверка ground truth.</b> Калибровка по децилям важна именно для синтетического стенда: она показывает, восстанавливает ли модель порядок клиентов по истинному эффекту, а не только выигрывает по policy-метрике.",
        )
    if index == 72:
        return box(
            "warning",
            "<b>Кандидат на уплотнение вывода.</b> Learning curve для uplift-моделей тяжелая, но ценная. В компактной версии стоит оставить график и таблицу ключевых точек по S-Learner/T-Learner.",
        )
    if index == 77:
        return box(
            "warning",
            "<b>Кандидат на уплотнение вывода.</b> Визуальная проверка сценариев — самый тяжелый output ноутбука. Для финального текста можно оставить 4 малых графика или одну сводную схему с описанием сценариев.",
        )
    if index == 82:
        return box(
            "note",
            "<b>Итог сценарного блока.</b> Этот график лучше читать как проверку условий применимости: когда риск и эффект расходятся, risk-based стратегия теряет причинный смысл, а uplift-модели становятся более оправданными.",
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
