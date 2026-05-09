# -*- coding: utf-8 -*-
"""
Патч uplift_models.ipynb v2:
  1. Feature engineering (5 признаков)
  2. Загрузка/обучение 4 risk-baseline моделей
  3. Обновление ячейки baselines, comparison table, Qini-кривых
  4. Раздел 12: Learning curves (AUUC + Spearman rho vs N)
  5. Переименование выводов в раздел 13
"""
import json, pathlib, copy

NB = pathlib.Path(r"c:\Рабочий стол\Magic_work\uplift_models.ipynb")
nb = json.loads(NB.read_text(encoding='utf-8'))
cells = nb['cells']

# ── Вспомогательные функции ───────────────────────────────────────────────────
def src(text: str):
    lines = text.split('\n')
    return [line + '\n' if i < len(lines) - 1 else line for i, line in enumerate(lines)]

def code_cell(text, cell_id=None):
    c = {'cell_type': 'code', 'execution_count': None, 'metadata': {},
         'outputs': [], 'source': src(text)}
    if cell_id:
        c['id'] = cell_id
    return c

def md_cell(text, cell_id=None):
    c = {'cell_type': 'markdown', 'metadata': {}, 'source': src(text)}
    if cell_id:
        c['id'] = cell_id
    return c

def find_id(cell_id):
    for i, c in enumerate(cells):
        if c.get('id') == cell_id:
            return i
    raise KeyError(f'Cell id not found: {cell_id}')

def insert_after(cell_id, new_cells):
    idx = find_id(cell_id)
    for j, nc in enumerate(new_cells):
        cells.insert(idx + 1 + j, nc)

def replace_cell(cell_id, new_cell):
    idx = find_id(cell_id)
    cells[idx] = new_cell

# ── Назначаем ID ячейкам, у которых их нет (по индексу до вставок) ─────────
_id_map = {4: 'cell-4', 6: 'cell-6', 24: 'cell-24', 25: 'cell-25', 30: 'cell-30'}
for _idx, _cid in _id_map.items():
    if cells[_idx].get('id') is None:
        cells[_idx]['id'] = _cid

# ══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE ENGINEERING — вставляем после ячейки загрузки данных (cell-4)
# ══════════════════════════════════════════════════════════════════════════════
FE_MD = r"""## 1.1. Инженерные признаки

Воспроизводятся те же преобразования, что применялись в `cboost.ipynb` и `logreg.ipynb`,
чтобы uplift-модели и risk-baseline использовали одинаковое признаковое пространство.

| Признак | Формула | Смысл |
|---|---|---|
| `AGE_YEARS` | $-\text{DAYS\_BIRTH}/365$ | Возраст клиента в годах |
| `EMPLOYMENT_YEARS` | $-\text{DAYS\_EMPLOYED}/365$ | Стаж работы в годах |
| `EMPLOYMENT_RATIO` | $\text{EMPLOYMENT\_YEARS}/\text{AGE\_YEARS}$ | Доля жизни в занятости |
| `CREDIT_HISTORY_LENGTH` | $\text{DAYS\_CREDIT\_MAX}-\text{DAYS\_CREDIT\_MIN}$ | Длина кредитной истории (дни) |
| `MONTHS_BALANCE_RANGE` | $\text{MONTHS\_BALANCE\_MIN}-\text{MONTHS\_BALANCE\_MEAN}$ | Разброс глубины кредитных записей |"""

FE_CODE = r"""# Инженерные признаки (идентично cboost.ipynb / logreg.ipynb)
df_full['AGE_YEARS']             = -df_full['DAYS_BIRTH'] / 365
df_full['EMPLOYMENT_YEARS']      = -df_full['DAYS_EMPLOYED'] / 365
df_full['EMPLOYMENT_RATIO']      = df_full['EMPLOYMENT_YEARS'] / df_full['AGE_YEARS']
df_full['CREDIT_HISTORY_LENGTH'] = df_full['DAYS_CREDIT_MAX'] - df_full['DAYS_CREDIT_MIN']
df_full['MONTHS_BALANCE_RANGE']  = df_full['MONTHS_BALANCE_MIN'] - df_full['MONTHS_BALANCE_MEAN']
print('Добавлено 5 инженерных признаков: AGE_YEARS, EMPLOYMENT_YEARS, EMPLOYMENT_RATIO, '
      'CREDIT_HISTORY_LENGTH, MONTHS_BALANCE_RANGE')"""

insert_after('cell-4', [md_cell(FE_MD, 'cell-4a'), code_cell(FE_CODE, 'cell-4b')])

# ══════════════════════════════════════════════════════════════════════════════
# 2. RISK MODELS — загрузка и обучение после ячейки разбивки (cell-6)
# ══════════════════════════════════════════════════════════════════════════════
RISK_MD = r"""## 2.1. Risk-baseline модели

Для корректного сравнения подходов формируются **четыре риск-ориентированных baseline**:

| Модель | Признаки | Target | Источник |
|---|---|---|---|
| **CatBoost (22-отобр., TARGET)** | 22 отобранных (SHAP + IV) + engineering | `TARGET` (ориг.) | загрузка `cboost.cbm` |
| **CatBoost (все feat, TARGET)** | все 139 числовых + 16 категориальных | `TARGET` (ориг.) | обучение здесь |
| **LogReg (8 WoE, TARGET)** | 8 WoE-бинированных | `TARGET` (ориг.) | загрузка `logreg.pkl` |
| **LogReg (все feat, TARGET)** | все 139 числовых, StandardScaler | `TARGET` (ориг.) | обучение здесь |

Все risk-модели обучены на `TARGET` (оригинальная метка дефолта Home Credit) — это воспроизводит
**реальный сценарий**: bank имеет production-модель риска и применяет её для CRM-таргетинга.
Uplift-модели обучаются на `TARGET_AFTER_CONTACT` (синтетический исход с коммуникацией).
Все модели оцениваются по **AUUC на `TARGET_AFTER_CONTACT`**."""

RISK_CODE = r"""import pickle, os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Загрузка сохранённых моделей ──────────────────────────────────────────────
with open('models/cboost/features.json', encoding='utf-8') as f:
    cb_saved_feats = json.load(f)
with open('models/logreg/features.json', encoding='utf-8') as f:
    lr_saved_feats = json.load(f)

cb_saved = CatBoostClassifier()
cb_saved.load_model('models/cboost/cboost.cbm')

lr_saved_bp = pickle.load(open('models/logreg/binning_process.pkl', 'rb'))
lr_saved_lr = pickle.load(open('models/logreg/logreg.pkl', 'rb'))

# Категориальные признаки для cboost.cbm
cb_saved_cat = [f for f in cb_saved_feats if f in df_full.columns and df_full[f].dtype == object]

def _cb_saved_pred(df_split):
    X = df_split[cb_saved_feats].copy().fillna(-999)
    for c in cb_saved_cat:
        X[c] = X[c].fillna('missing').astype(str)
    return cb_saved.predict_proba(X)[:, 1]

def _lr_saved_pred(df_split):
    X = df_split[lr_saved_feats].fillna(-999)
    return lr_saved_lr.predict_proba(lr_saved_bp.transform(X))[:, 1]

cb_saved_score_test = _cb_saved_pred(test)
cb_saved_score_oot  = _cb_saved_pred(oot)
lr_saved_score_test = _lr_saved_pred(test)
lr_saved_score_oot  = _lr_saved_pred(oot)

# ── CatBoost на всех признаках (TARGET) ───────────────────────────────────────
cat_cols_all = [c for c in df_full.columns
                if c not in (set(feature_cols) | {'SK_ID_CURR','TARGET',outcome_col,'COMMUNICATION'}
                             | leakage_cols)
                and df_full[c].dtype == object]

def _make_cb_all_X(df_split):
    return pd.concat([
        df_split[feature_cols].fillna(-999).reset_index(drop=True),
        df_split[cat_cols_all].fillna('missing').astype(str).reset_index(drop=True)
    ], axis=1)

X_train_cba = _make_cb_all_X(train)
X_test_cba  = _make_cb_all_X(test)
X_oot_cba   = _make_cb_all_X(oot)
cat_idx_all = list(range(len(feature_cols), len(feature_cols) + len(cat_cols_all)))

y_train_orig = train['TARGET'].values
y_test_orig  = test['TARGET'].values

cb_all = CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=6,
    cat_features=cat_idx_all,
    random_seed=RANDOM_SEED, verbose=0,
    eval_metric='AUC', early_stopping_rounds=50,
)
cb_all.fit(X_train_cba, y_train_orig, eval_set=(X_test_cba, y_test_orig))

cb_all_score_test = cb_all.predict_proba(X_test_cba)[:, 1]
cb_all_score_oot  = cb_all.predict_proba(X_oot_cba)[:, 1]

# ── LogReg на всех числовых признаках (TARGET) ────────────────────────────────
lr_all = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_SEED)),
])
lr_all.fit(X_train, y_train_orig)
lr_all_score_test = lr_all.predict_proba(X_test)[:, 1]
lr_all_score_oot  = lr_all.predict_proba(X_oot)[:, 1]

# ── Качество risk-моделей на TARGET ───────────────────────────────────────────
print(f'{"Модель":<30} {"Gini (TARGET, test)":>20}')
print('-' * 53)
for name, pred in [
    ('CatBoost (22-отобр., TARGET)', cb_saved_score_test),
    ('CatBoost (все feat, TARGET)',  cb_all_score_test),
    ('LogReg (8 WoE, TARGET)',       lr_saved_score_test),
    ('LogReg (все feat, TARGET)',    lr_all_score_test),
]:
    auc = roc_auc_score(y_test_orig, pred)
    print(f'{name:<30} {2*auc-1:>20.4f}')"""

insert_after('cell-6', [md_cell(RISK_MD, 'cell-6a'), code_cell(RISK_CODE, 'cell-6b')])

# ══════════════════════════════════════════════════════════════════════════════
# 3. ОБНОВЛЕНИЕ ЯЧЕЙКИ BASELINES (54d97867)
# ══════════════════════════════════════════════════════════════════════════════
NEW_BASELINES = r"""rng = np.random.RandomState(RANDOM_SEED)

scores_baselines = {
    'Случайный выбор':                rng.rand(len(y_test)),
    'BASE_PD (синтетич.)':            test['BASE_PD'].values,
    'CatBoost (22-отобр., TARGET)':   cb_saved_score_test,
    'CatBoost (все feat, TARGET)':    cb_all_score_test,
    'LogReg (8 WoE, TARGET)':         lr_saved_score_test,
    'LogReg (все feat, TARGET)':      lr_all_score_test,
    'Oracle (TRUE_UPLIFT)':          -test['TRUE_UPLIFT'].values,
}
scores_baselines_oot = {
    'Случайный выбор':                np.random.RandomState(RANDOM_SEED+1).rand(len(y_oot)),
    'BASE_PD (синтетич.)':            oot['BASE_PD'].values,
    'CatBoost (22-отобр., TARGET)':   cb_saved_score_oot,
    'CatBoost (все feat, TARGET)':    cb_all_score_oot,
    'LogReg (8 WoE, TARGET)':         lr_saved_score_oot,
    'LogReg (все feat, TARGET)':      lr_all_score_oot,
    'Oracle (TRUE_UPLIFT)':          -oot['TRUE_UPLIFT'].values,
}

print(f'{"Стратегия":<35} {"AUUC (test)":>12} {"AUUC (OOT)":>12}')
print('-' * 62)
baseline_auuc = {}
for name in scores_baselines:
    _, _, _, auuc_t = compute_qini(y_test, t_test_bin, scores_baselines[name])
    _, _, _, auuc_o = compute_qini(y_oot,  t_oot_bin,  scores_baselines_oot[name])
    baseline_auuc[name] = auuc_t
    print(f'{name:<35} {auuc_t:>12.6f} {auuc_o:>12.6f}')"""

replace_cell('54d97867', code_cell(NEW_BASELINES, '54d97867'))

# ══════════════════════════════════════════════════════════════════════════════
# 4. ОБНОВЛЕНИЕ ТАБЛИЦЫ СРАВНЕНИЯ (cell-24)
# ══════════════════════════════════════════════════════════════════════════════
NEW_COMPARISON = r"""true_uplift_test = test['TRUE_UPLIFT'].values
results = []

model_configs = [
    ('Случайный выбор',               scores_baselines['Случайный выбор'],
                                      np.zeros(len(y_test)),              'Baseline'),
    ('BASE_PD (синтетич.)',            scores_baselines['BASE_PD (синтетич.)'],
                                      test['BASE_PD'].values,             'Baseline'),
    ('CatBoost (22-отобр., TARGET)',   cb_saved_score_test,
                                      cb_saved_score_test,                'Risk'),
    ('CatBoost (все feat, TARGET)',    cb_all_score_test,
                                      cb_all_score_test,                  'Risk'),
    ('LogReg (8 WoE, TARGET)',         lr_saved_score_test,
                                      lr_saved_score_test,                'Risk'),
    ('LogReg (все feat, TARGET)',      lr_all_score_test,
                                      lr_all_score_test,                  'Risk'),
    ('Oracle (TRUE_UPLIFT)',           scores_baselines['Oracle (TRUE_UPLIFT)'],
                                      -true_uplift_test,                  'Baseline'),
    ('S-Learner',                      score_s_test,  uplift_s_test['best'], 'Uplift'),
    ('T-Learner',                      score_t_test,  uplift_t_test['best'], 'Uplift'),
    ('X-Learner',                      score_x_test,  uplift_x_test,         'Uplift'),
    ('DR-Learner',                     score_dr_test, uplift_dr_test,        'Uplift'),
]

oracle_auuc = baseline_auuc['Oracle (TRUE_UPLIFT)']
for name, score, uplift_pred, mtype in model_configs:
    _, _, _, auuc = compute_qini(y_test, t_test_bin, score)
    sp_rho, sp_p  = stats.spearmanr(uplift_pred, true_uplift_test)
    results.append({'Модель': name, 'AUUC': auuc,
                    '% от Oracle': auuc / oracle_auuc * 100,
                    'Spearman rho': sp_rho, 'p-value': sp_p, 'Тип': mtype})

df_results = pd.DataFrame(results).set_index('Модель')
print(df_results.sort_values('AUUC', ascending=False).to_string(
    float_format=lambda x: f'{x:.4f}'))"""

replace_cell('cell-24', code_cell(NEW_COMPARISON, 'cell-24'))

# ══════════════════════════════════════════════════════════════════════════════
# 5. ОБНОВЛЕНИЕ QINI-КРИВЫХ (cell-25)
# ══════════════════════════════════════════════════════════════════════════════
NEW_QINI = r"""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

plot_styles = [
    ('Oracle (TRUE_UPLIFT)',         'black',      '--', 2.5),
    ('BASE_PD (синтетич.)',          'dimgray',    ':',  1.5),
    ('CatBoost (22-отобр., TARGET)', 'saddlebrown','--', 1.5),
    ('CatBoost (все feat, TARGET)',  'peru',       '-',  1.8),
    ('LogReg (8 WoE, TARGET)',       'slategray',  '--', 1.5),
    ('LogReg (все feat, TARGET)',    'steelblue',  ':',  1.5),
    ('S-Learner',                    'royalblue',  '-',  1.8),
    ('T-Learner',                    'darkorange', '-',  2.0),
    ('X-Learner',                    'green',      '-',  1.5),
    ('DR-Learner',                   'crimson',    '-',  2.0),
    ('Случайный выбор',              'lightgray',  ':',  1.2),
]

score_map_test = {
    **scores_baselines,
    'S-Learner':  score_s_test,
    'T-Learner':  score_t_test,
    'X-Learner':  score_x_test,
    'DR-Learner': score_dr_test,
}
score_map_oot = {
    **scores_baselines_oot,
    'S-Learner':  score_s_oot,
    'T-Learner':  score_t_oot,
    'X-Learner':  score_x_oot,
    'DR-Learner': score_dr_oot,
}

for ax, (y_arr, t_arr, sc_map, suffix) in zip(axes, [
    (y_test, t_test_bin, score_map_test, 'test'),
    (y_oot,  t_oot_bin,  score_map_oot,  'OOT'),
]):
    for label, color, ls, lw in plot_styles:
        if label not in sc_map:
            continue
        fracs, qini, _, auuc = compute_qini(y_arr, t_arr, sc_map[label])
        ax.plot(fracs * 100, qini, color=color, linestyle=ls, linewidth=lw,
                label=f'{label}  ({auuc:.0f})')
    ax.set_xlabel('Доля отобранных клиентов, %')
    ax.set_ylabel('Qini')
    ax.set_title(f'Qini-кривые — {suffix}')
    ax.legend(loc='upper left', fontsize=7)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

plt.suptitle('Сравнение стратегий таргетинга: Qini-кривые', fontsize=13)
plt.tight_layout()
plt.show()"""

replace_cell('cell-25', code_cell(NEW_QINI, 'cell-25'))

# ══════════════════════════════════════════════════════════════════════════════
# 6. ОБНОВЛЕНИЕ СОХРАНЕНИЯ СКОРОВ (da482f41)
# ══════════════════════════════════════════════════════════════════════════════
NEW_SAVE = r"""scores_to_save = pd.DataFrame({
    'split':          ['test'] * len(y_test) + ['oot'] * len(y_oot),
    'y':              np.concatenate([y_test, y_oot]),
    'treatment_bin':  np.concatenate([t_test_bin, t_oot_bin]),
    'COMMUNICATION':  np.concatenate([T_test, T_oot]),
    'BASE_PD':        np.concatenate([test['BASE_PD'].values,  oot['BASE_PD'].values]),
    'TRUE_UPLIFT':    np.concatenate([test['TRUE_UPLIFT'].values, oot['TRUE_UPLIFT'].values]),
    'RISK_SEGMENT':   np.concatenate([test['RISK_SEGMENT'].values, oot['RISK_SEGMENT'].values]),
    # Risk-baseline scores
    'score_cb_saved': np.concatenate([cb_saved_score_test, cb_saved_score_oot]),
    'score_cb_all':   np.concatenate([cb_all_score_test,   cb_all_score_oot]),
    'score_lr_saved': np.concatenate([lr_saved_score_test, lr_saved_score_oot]),
    'score_lr_all':   np.concatenate([lr_all_score_test,   lr_all_score_oot]),
    # Uplift scores
    'score_s':        np.concatenate([score_s_test,  score_s_oot]),
    'score_t':        np.concatenate([score_t_test,  score_t_oot]),
    'score_x':        np.concatenate([score_x_test,  score_x_oot]),
    'score_dr':       np.concatenate([score_dr_test, score_dr_oot]),
    # Per-channel S-Learner
    'uplift_s_sms':      np.concatenate([uplift_s_test['sms'],          uplift_s_oot['sms']]),
    'uplift_s_robot':    np.concatenate([uplift_s_test['robot_call'],   uplift_s_oot['robot_call']]),
    'uplift_s_operator': np.concatenate([uplift_s_test['operator_call'],uplift_s_oot['operator_call']]),
    # Per-channel T-Learner
    'uplift_t_sms':      np.concatenate([uplift_t_test['sms'],          uplift_t_oot['sms']]),
    'uplift_t_robot':    np.concatenate([uplift_t_test['robot_call'],   uplift_t_oot['robot_call']]),
    'uplift_t_operator': np.concatenate([uplift_t_test['operator_call'],uplift_t_oot['operator_call']]),
})
scores_to_save.to_csv('data/processed/uplift_scores.csv', index=False)
print(f"Скоры сохранены: {len(scores_to_save):,} строк × {len(scores_to_save.columns)} колонок")"""

replace_cell('da482f41', code_cell(NEW_SAVE, 'da482f41'))

# ══════════════════════════════════════════════════════════════════════════════
# 7. РАЗДЕЛ 12: LEARNING CURVES — вставляем перед cell-30 (выводы)
# ══════════════════════════════════════════════════════════════════════════════
LC_MD = r"""---

## 12. Зависимость качества от объёма обучающих данных

### Методология

Каждая модель переобучается на **подвыборках** обучающего набора фиксированного размера;
качество оценивается на неизменном тестовом наборе (61 502 наблюдений).
Для uplift-моделей основная метрика — **AUUC** (Qini); дополнительно отслеживается
**Spearman $\rho$** с `TRUE_UPLIFT` как прямое свидетельство каузальной точности.

| Параметр | Значение |
|---|---|
| Доли train | 5%, 10%, 20%, 35%, 50%, 70%, 100% |
| DR-Learner CV | 3-fold (упрощённый, 100 итер. nuisance) |
| Оценочный набор | фиксированный test (61 502) |

**Что ожидаем увидеть:**
- Risk-based модели выходят на плато раньше (нет необходимости в контрфактуальных оценках)
- T-Learner при малых N нестабилен из-за малочисленных SMS-группы (~1.7%)
- DR-Learner требует наибольшего объёма для стабилизации OOF-оценок
- При достаточном N разрыв между лучшим uplift-методом и risk-based может сократиться"""

LC_COMPUTE = r"""fractions_lc = [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0]
MIN_CH_SAMPLES = 150
lc_rows = []

print(f'{"frac":>5} {"N":>7} | {"CB-risk":>8} {"LR-risk":>8} {"S-Lrn":>8} {"T-Lrn":>8} {"DR-Lrn":>8}')
print('-' * 65)

for frac in fractions_lc:
    if frac < 1.0:
        from sklearn.model_selection import train_test_split as _tts
        idx_f, _ = _tts(np.arange(len(X_train)), train_size=frac,
                         stratify=t_train_bin, random_state=RANDOM_SEED)
        idx_f = sorted(idx_f)
    else:
        idx_f = np.arange(len(X_train))

    Xf      = X_train.iloc[idx_f].reset_index(drop=True)
    yf      = y_train[idx_f]
    Tf      = T_train[idx_f]
    tf_bin  = t_train_bin[idx_f]
    y_orig_f = train.iloc[idx_f]['TARGET'].values
    row = {'frac': frac, 'n': len(idx_f)}

    # ── CatBoost-risk (все feat + categoricals) ────────────────────────────
    Xf_cba = pd.concat([
        Xf,
        train.iloc[idx_f][cat_cols_all].fillna('missing').astype(str).reset_index(drop=True)
    ], axis=1)
    cb_lc = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6,
                                cat_features=cat_idx_all,
                                random_seed=RANDOM_SEED, verbose=0)
    cb_lc.fit(Xf_cba, y_orig_f)
    _, _, _, auuc = compute_qini(y_test, t_test_bin, cb_lc.predict_proba(X_test_cba)[:, 1])
    rho, _ = stats.spearmanr(cb_lc.predict_proba(X_test_cba)[:, 1], true_uplift_test)
    row['CatBoost-risk AUUC'] = auuc;  row['CatBoost-risk rho'] = rho

    # ── LogReg-risk (все числовые feat) ────────────────────────────────────
    lr_lc = Pipeline([('scaler', StandardScaler()),
                       ('lr', LogisticRegression(C=0.1, max_iter=500,
                                                  random_state=RANDOM_SEED))])
    lr_lc.fit(Xf, y_orig_f)
    _, _, _, auuc = compute_qini(y_test, t_test_bin, lr_lc.predict_proba(X_test)[:, 1])
    rho, _ = stats.spearmanr(lr_lc.predict_proba(X_test)[:, 1], true_uplift_test)
    row['LogReg-risk AUUC'] = auuc;  row['LogReg-risk rho'] = rho

    # ── S-Learner ─────────────────────────────────────────────────────────
    enc_lc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    enc_lc.fit(Tf.reshape(-1, 1))
    Xf_s = add_treatment_ohe(Xf, Tf, enc_lc)
    s_lc = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6,
                               random_seed=RANDOM_SEED, verbose=0)
    s_lc.fit(Xf_s, yf)
    upl_s_lc = s_learner_uplift_per_channel(s_lc, X_test, enc_lc, channels_list)
    _, _, _, auuc = compute_qini(y_test, t_test_bin, -upl_s_lc['best'])
    rho, _ = stats.spearmanr(upl_s_lc['best'], true_uplift_test)
    row['S-Learner AUUC'] = auuc;  row['S-Learner rho'] = rho

    # ── T-Learner ─────────────────────────────────────────────────────────
    t_lc = {}; ok = True
    for ch in ['control'] + channels_list:
        mask_ch = Tf == ch
        if mask_ch.sum() < MIN_CH_SAMPLES:
            ok = False; break
        m = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=5,
                                random_seed=RANDOM_SEED, verbose=0)
        m.fit(Xf[mask_ch].reset_index(drop=True), yf[mask_ch])
        t_lc[ch] = m
    if ok:
        upl_t_lc = t_learner_uplift(t_lc, X_test, channels_list)
        _, _, _, auuc = compute_qini(y_test, t_test_bin, -upl_t_lc['best'])
        rho, _ = stats.spearmanr(upl_t_lc['best'], true_uplift_test)
        row['T-Learner AUUC'] = auuc;  row['T-Learner rho'] = rho
    else:
        row['T-Learner AUUC'] = float('nan');  row['T-Learner rho'] = float('nan')

    # ── DR-Learner (упрощённый: 3-fold, 100 iter nuisance) ────────────────
    kf3 = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    Xf_arr = Xf.values
    mu1_o = np.zeros(len(Xf_arr)); mu0_o = np.zeros(len(Xf_arr)); e_o = np.zeros(len(Xf_arr))
    for _, (idx_tr, idx_val) in enumerate(kf3.split(Xf_arr)):
        Xk, Xv = Xf_arr[idx_tr], Xf_arr[idx_val]
        yk, tk = yf[idx_tr], tf_bin[idx_tr]
        mu_k = CatBoostClassifier(iterations=100, learning_rate=0.05, depth=4,
                                   random_seed=RANDOM_SEED, verbose=0)
        mu_k.fit(np.column_stack([Xk, tk]), yk)
        mu1_o[idx_val] = mu_k.predict_proba(np.column_stack([Xv, np.ones(len(Xv))]))[:, 1]
        mu0_o[idx_val] = mu_k.predict_proba(np.column_stack([Xv, np.zeros(len(Xv))]))[:, 1]
        e_k = CatBoostClassifier(iterations=50, learning_rate=0.05, depth=3,
                                  random_seed=RANDOM_SEED, verbose=0)
        e_k.fit(Xk, tk)
        e_o[idx_val] = e_k.predict_proba(Xv)[:, 1]
    e_cl = np.clip(e_o, 0.05, 0.95)
    dr_ps = (mu1_o - mu0_o) + (tf_bin - e_cl) / (e_cl * (1 - e_cl)) * (yf - (tf_bin * mu1_o + (1 - tf_bin) * mu0_o))
    dr_lc = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=5,
                               random_seed=RANDOM_SEED, verbose=0)
    dr_lc.fit(Xf_arr, dr_ps)
    upl_dr_lc = dr_lc.predict(X_test.values)
    _, _, _, auuc = compute_qini(y_test, t_test_bin, -upl_dr_lc)
    rho, _ = stats.spearmanr(upl_dr_lc, true_uplift_test)
    row['DR-Learner AUUC'] = auuc;  row['DR-Learner rho'] = rho

    lc_rows.append(row)
    print(f"{frac:5.0%} {len(idx_f):>7,} | "
          f"{row['CatBoost-risk AUUC']:>8.1f} "
          f"{row['LogReg-risk AUUC']:>8.1f} "
          f"{row.get('S-Learner AUUC', float('nan')):>8.1f} "
          f"{row.get('T-Learner AUUC', float('nan')):>8.1f} "
          f"{row.get('DR-Learner AUUC', float('nan')):>8.1f}")

lc_df = pd.DataFrame(lc_rows).set_index('n')
print('\nГотово.')"""

LC_PLOT = r"""fig, axes = plt.subplots(2, 2, figsize=(15, 10))

_styles = {
    'CatBoost-risk': ('saddlebrown', '-',  2.0),
    'LogReg-risk':   ('steelblue',   '--', 1.8),
    'S-Learner':     ('royalblue',   '-',  1.8),
    'T-Learner':     ('darkorange',  '-',  2.2),
    'DR-Learner':    ('crimson',     '-',  2.0),
}
oracle_auuc_lc = baseline_auuc['Oracle (TRUE_UPLIFT)']

for ax_idx, (ax, ylabel, col_suffix, title, log_x) in enumerate([
    (axes[0,0], 'AUUC',        'AUUC', 'AUUC vs объём (линейная шкала)',     False),
    (axes[0,1], 'AUUC',        'AUUC', 'AUUC vs объём (логарифмическая)',     True),
    (axes[1,0], 'Spearman ρ',  'rho',  'Spearman ρ vs объём',                False),
    (axes[1,1], '% от Oracle', 'AUUC', '% от Oracle vs объём (log)',          True),
]):
    if col_suffix == 'AUUC' and ylabel == 'AUUC':
        ax.axhline(oracle_auuc_lc, color='black', ls='--', lw=1.5,
                   label=f'Oracle ({oracle_auuc_lc:.0f})')
    if col_suffix == 'rho':
        ax.axhline(0, color='gray', ls=':', lw=1)

    for model, (color, ls, lw) in _styles.items():
        col = f'{model} {col_suffix}'
        if col not in lc_df.columns:
            continue
        vals = lc_df[col]
        if ylabel == '% от Oracle':
            vals = vals / oracle_auuc_lc * 100
        ax.plot(lc_df.index, vals, f'o{ls}', color=color, lw=lw, label=model)

    if log_x:
        ax.set_xscale('log')
    ax.set_xlabel('Объём обучающей выборки' + (' (log)' if log_x else ''))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

plt.suptitle('Кривые обучения: зависимость от объёма данных', fontsize=13)
plt.tight_layout()
plt.show()

# Итоговая таблица
print('\nИтоговые значения при 100% train:')
full_row = lc_df.iloc[-1]
print(f'{"Модель":<25} {"AUUC":>8} {"% Oracle":>10} {"Spearman rho":>14}')
print('-' * 62)
for model in _styles:
    auuc_v = full_row.get(f'{model} AUUC', float('nan'))
    rho_v  = full_row.get(f'{model} rho', float('nan'))
    pct    = auuc_v / oracle_auuc_lc * 100 if not pd.isna(auuc_v) else float('nan')
    print(f'{model:<25} {auuc_v:>8.1f} {pct:>10.1f}% {rho_v:>14.4f}')"""

# Вставляем раздел 12 перед ячейкой выводов (cell-30)
idx_conclusions = find_id('cell-30')
for j, nc in enumerate([
    md_cell(LC_MD,     'cell-lc-md'),
    code_cell(LC_COMPUTE, 'cell-lc-compute'),
    code_cell(LC_PLOT,    'cell-lc-plot'),
]):
    cells.insert(idx_conclusions + j, nc)

# ══════════════════════════════════════════════════════════════════════════════
# 8. ОБНОВЛЕНИЕ ВЫВОДОВ: раздел 11 → 13
# ══════════════════════════════════════════════════════════════════════════════
idx_conc = find_id('cell-30')
conc_src = ''.join(cells[idx_conc]['source'])
conc_src = conc_src.replace('## 11. Итоговые выводы', '## 13. Итоговые выводы')
cells[idx_conc]['source'] = src(conc_src)

# ── Запись ────────────────────────────────────────────────────────────────────
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(f'Патч применён. Ячеек в ноутбуке: {len(cells)}')
