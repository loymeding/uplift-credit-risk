# Uplift-моделирование в задаче кредитного риска

Репозиторий магистерской диссертации по применению uplift-моделирования в сфере финтеха для оптимизации коммуникаций с заёмщиками.

## О работе

Цель исследования — оценить применимость uplift-моделирования для предсказания гетерогенного эффекта коммуникационных воздействий банка (SMS, автозвонок, звонок оператора) на вероятность дефолта клиента. Uplift-модели сравниваются с классическими подходами: логистической регрессией и градиентным бустингом (CatBoost).

**Ключевые вопросы:**
- Даёт ли uplift-подход прирост качества по сравнению с классическими методами?
- Какой объём данных необходим каждому из подходов?
- Какие преобразования и отбор признаков оптимальны для каждого метода?

## Данные

Исходный датасет — [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) (Kaggle). Поскольку реальных данных о коммуникациях нет, была разработана процедура **синтетической генерации** uplift-слоя, имитирующая реальные CRM-процессы банка:

| Переменная | Описание |
|---|---|
| `BASE_PD` | Базовая вероятность дефолта без коммуникации |
| `CONTACT_PROPENSITY` | Вероятность инициации контакта банком (selection bias) |
| `COMMUNICATION` | Тип воздействия: нет / SMS / робозвонок / оператор |
| `RISK_SEGMENT` | Сегмент риска: low / medium / high |
| `CONTACT_HISTORY` | История контактов (fatigue effect) |
| `PREFERRED_CHANNEL` | Предпочтительный канал клиента |
| `INTERACTION_SCORE` | Взаимодействие признаков клиента и типа коммуникации |
| `DELAY_FLAG` | Флаг отложенной реакции клиента |
| `TRUE_UPLIFT` | Истинный каузальный эффект коммуникации |
| `TARGET_AFTER_CONTACT` | Наблюдаемый результат после воздействия |

Итоговая каузальная структура: `PD_after = BASE_PD + TRUE_UPLIFT`

## Структура репозитория

```
├── data/
│   ├── raw/           # Исходные CSV с Kaggle (в репозиторий не входят)
│   └── processed/     # uplift-dataset.csv (генерируется ноутбуком 1)
├── models/
│   ├── cboost/        # Обученная модель CatBoost
│   └── logreg/        # Обученная модель логистической регрессии
├── features/          # Excel-файлы с важностью признаков
├── docs/              # Текст магистерской диссертации
├── formatted_notebooks/ # Оформленные версии ноутбуков для чтения и защиты
├── prepare_dataset.ipynb            # 1. Подготовка данных и генерация uplift-слоя
├── analytic_dataset.ipynb           # 2. Анализ и валидация датасета
├── gradient_boosting.ipynb          # 3. Модель CatBoost
├── logistic_regression.ipynb        # 4. Логистическая регрессия
├── uplift_models.ipynb              # 5. Uplift-модели
└── policy_evaluation.ipynb          # 6. Бизнес-оценка политик коммуникации
```

## Порядок запуска

Ноутбуки запускаются последовательно из корня репозитория:

```bash
pip install pandas==2.2.2 numpy==2.0.2 catboost==1.2.10 lightgbm optuna==4.8.0 \
    shap matplotlib==3.10.0 seaborn plotly==5.24.1 scipy tqdm scikit-learn optbinning

jupyter lab
```

1. `prepare_dataset.ipynb` — обработка данных, генерация синтетического uplift-слоя, отбор 22 признаков → сохраняет `data/processed/uplift-dataset.csv`
2. `analytic_dataset.ipynb` — разведочный анализ, валидация каузальной структуры
3. `gradient_boosting.ipynb` — обучение CatBoost, Optuna-тюнинг, SHAP-анализ
4. `logistic_regression.ipynb` — логистическая регрессия с WoE-кодированием, Optuna-тюнинг
5. `uplift_models.ipynb` — сравнение risk-based стратегий, S-Learner, T-Learner, X-Learner и DR-Learner
6. `policy_evaluation.ipynb` — перевод скорингов в бизнес-метрики: бюджет, ROI, выбор канала и bootstrap-интервалы

## Оформленные ноутбуки

В директории `formatted_notebooks/` лежат версии тех же ноутбуков, переоформленные для чтения, защиты и последовательного изучения. В них сохранены исходные code-ячейки и outputs, но добавлены оглавления, поясняющие блоки, выводы и пометки о слишком тяжелых выводах.

Порядок запуска и чтения отражен прямо в названиях файлов:

1. `formatted_notebooks/01_prepare_dataset.ipynb`
2. `formatted_notebooks/02_analytic_dataset.ipynb`
3. `formatted_notebooks/03_gradient_boosting.ipynb`
4. `formatted_notebooks/04_logistic_regression.ipynb`
5. `formatted_notebooks/05_uplift_models.ipynb`
6. `formatted_notebooks/06_policy_evaluation.ipynb`

Скрипты в `tools/format_*_notebook.py` воспроизводят оформление и проверяют, что code-ячейки не изменились при генерации оформленных копий.

> **Важно:** исходные CSV из Kaggle нужно положить в `data/raw/` перед запуском первого ноутбука.

## Метрика качества

Нормализованный коэффициент Джини: `Gini = 2 × AUC − 1`

Разбивка данных: 60% train / 20% test / 20% OOT (out-of-time holdout).
