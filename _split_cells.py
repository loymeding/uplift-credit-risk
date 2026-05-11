# -*- coding: utf-8 -*-
"""Разбивает большие ячейки uplift_models.ipynb на логические части."""
import json, pathlib

NB = pathlib.Path(r"c:\Рабочий стол\Magic_work\uplift_models.ipynb")
nb = json.loads(NB.read_text(encoding='utf-8'))
cells = nb['cells']

def src(text):
    lines = text.split('\n')
    return [line + '\n' if i < len(lines) - 1 else line for i, line in enumerate(lines)]

def code(text, cid=None):
    c = {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': src(text)}
    if cid: c['id'] = cid
    return c

def md(text, cid=None):
    c = {'cell_type': 'markdown', 'metadata': {}, 'source': src(text)}
    if cid: c['id'] = cid
    return c

def find(cid):
    for i, c in enumerate(cells):
        if c.get('id') == cid:
            return i
    raise KeyError(cid)

def replace_with(cid, new_cells):
    idx = find(cid)
    cells[idx:idx+1] = new_cells

# ══════════════════════════════════════════════════════════════════════════════
# 1. CELL-6b (risk models, 91 строк) -> 5 code + 3 md
# ══════════════════════════════════════════════════════════════════════════════
replace_with('cell-6b', [

    code('''from catboost import CatBoostClassifier
from optbinning import BinningProcess
from sklearn.linear_model import LogisticRegression

# Загружаем списки признаков каждой модели
with open('models/cboost/features.json', encoding='utf-8') as f:
    cb_sel_feats = json.load(f)
with open('models/logreg/features.json', encoding='utf-8') as f:
    lr_sel_feats = json.load(f)

# Числовые / категориальные для CatBoost (отобранные)
cb_sel_num = [f for f in cb_sel_feats if f in df_full.columns and df_full[f].dtype != object]
cb_sel_cat = [f for f in cb_sel_feats if f in df_full.columns and df_full[f].dtype == object]
cb_sel_cat_idx = list(range(len(cb_sel_num), len(cb_sel_num) + len(cb_sel_cat)))

# Категориальные колонки для CatBoost (все признаки)
cat_cols_all = [c for c in df_full.columns
                if c not in (set(feature_cols) | {'SK_ID_CURR', 'TARGET', outcome_col, 'COMMUNICATION'} | leakage_cols)
                and df_full[c].dtype == object]
cat_idx_all = list(range(len(feature_cols), len(feature_cols) + len(cat_cols_all)))

# Список всех признаков для LogReg (числовые + категориальные)
lr_sel_cat   = [f for f in lr_sel_feats if df_full[f].dtype == object]
lr_all_feats = feature_cols + cat_cols_all

y_train_orig = train['TARGET'].values
y_test_orig  = test['TARGET'].values

def _make_cb_X(df_split, num_feats, cat_feats):
    """Собирает матрицу для CatBoost: числовые + строковые категориальные."""
    num = df_split[num_feats].fillna(-999).reset_index(drop=True)
    cat = df_split[cat_feats].fillna('missing').astype(str).reset_index(drop=True)
    return pd.concat([num, cat], axis=1)

def _prep_woe(df_split, feats, cat_feats):
    """Приводит DataFrame к типам, ожидаемым BinningProcess."""
    return df_split[feats].fillna(-999).astype({c: str for c in cat_feats})

print(f'cb_sel: {len(cb_sel_num)} числовых + {len(cb_sel_cat)} категориальных')
print(f'cat_cols_all: {len(cat_cols_all)} категориальных для CatBoost-all')
print(f'lr_sel_feats: {len(lr_sel_feats)} признаков  |  lr_all_feats: {len(lr_all_feats)} признаков')''',
    'cell-risk-setup'),

    md('''#### CatBoost (отобранные признаки, TARGET)

Модель обучается на признаковом наборе из `models/cboost/features.json` —
тех же числовых и категориальных признаках, на которых строился `cboost.cbm`
в `Magic_cboost.ipynb`. CatBoost обрабатывает категориальные признаки нативно,
без дополнительного кодирования.'''),

    code('''X_train_cbs = _make_cb_X(train, cb_sel_num, cb_sel_cat)
X_test_cbs  = _make_cb_X(test,  cb_sel_num, cb_sel_cat)
X_oot_cbs   = _make_cb_X(oot,   cb_sel_num, cb_sel_cat)

cb_sel = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                             cat_features=cb_sel_cat_idx,
                             random_seed=RANDOM_SEED, verbose=0,
                             eval_metric='AUC', early_stopping_rounds=50)
cb_sel.fit(X_train_cbs, y_train_orig, eval_set=(X_test_cbs, y_test_orig))

cb_saved_score_test = cb_sel.predict_proba(X_test_cbs)[:, 1]
cb_saved_score_oot  = cb_sel.predict_proba(X_oot_cbs)[:, 1]
gini = 2 * roc_auc_score(y_test_orig, cb_saved_score_test) - 1
print(f'CatBoost (отобр.) Gini на TARGET: {gini:.4f}')''',
    'cell-risk-cb-sel'),

    md('''#### CatBoost (все признаки, TARGET)

Обучается на полном признаковом пространстве: `feature_cols` (139 числовых,
включая engineering) + `cat_cols_all` (все категориальные, не входящие
в leakage). Показывает потолок качества CatBoost на данном датасете.'''),

    code('''X_train_cba = _make_cb_X(train, feature_cols, cat_cols_all)
X_test_cba  = _make_cb_X(test,  feature_cols, cat_cols_all)
X_oot_cba   = _make_cb_X(oot,   feature_cols, cat_cols_all)

cb_all = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                             cat_features=cat_idx_all,
                             random_seed=RANDOM_SEED, verbose=0,
                             eval_metric='AUC', early_stopping_rounds=50)
cb_all.fit(X_train_cba, y_train_orig, eval_set=(X_test_cba, y_test_orig))

cb_all_score_test = cb_all.predict_proba(X_test_cba)[:, 1]
cb_all_score_oot  = cb_all.predict_proba(X_oot_cba)[:, 1]
gini = 2 * roc_auc_score(y_test_orig, cb_all_score_test) - 1
print(f'CatBoost (все feat) Gini на TARGET: {gini:.4f}')''',
    'cell-risk-cb-all'),

    md('''#### Логистическая регрессия с WoE-кодированием (TARGET)

`BinningProcess` из `optbinning` трансформирует каждый признак в его
**Weight of Evidence** — монотонное, масштабированное представление,
устойчивое к выбросам и совместимое с линейной моделью.

`BinningProcess` обучается **только на train** — никакой утечки.
Два варианта: отобранные признаки (тот же набор, что в `logreg.ipynb`)
и все числовые + категориальные через WoE.'''),

    code('''# WoE-бининг обучаем на train, применяем к test и OOT
bp_sel = BinningProcess(variable_names=lr_sel_feats, categorical_variables=lr_sel_cat)
bp_sel.fit(_prep_woe(train, lr_sel_feats, lr_sel_cat), y_train_orig)

bp_all = BinningProcess(variable_names=lr_all_feats, categorical_variables=cat_cols_all)
bp_all.fit(_prep_woe(train, lr_all_feats, cat_cols_all), y_train_orig)

lr_sel = LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_SEED)
lr_sel.fit(bp_sel.transform(_prep_woe(train, lr_sel_feats, lr_sel_cat), metric='woe'), y_train_orig)
lr_saved_score_test = lr_sel.predict_proba(bp_sel.transform(_prep_woe(test, lr_sel_feats, lr_sel_cat), metric='woe'))[:, 1]
lr_saved_score_oot  = lr_sel.predict_proba(bp_sel.transform(_prep_woe(oot,  lr_sel_feats, lr_sel_cat), metric='woe'))[:, 1]
gini = 2 * roc_auc_score(y_test_orig, lr_saved_score_test) - 1
print(f'LogReg (отобр. WoE) Gini на TARGET: {gini:.4f}')

lr_all = LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_SEED)
lr_all.fit(bp_all.transform(_prep_woe(train, lr_all_feats, cat_cols_all), metric='woe'), y_train_orig)
lr_all_score_test = lr_all.predict_proba(bp_all.transform(_prep_woe(test, lr_all_feats, cat_cols_all), metric='woe'))[:, 1]
lr_all_score_oot  = lr_all.predict_proba(bp_all.transform(_prep_woe(oot,  lr_all_feats, cat_cols_all), metric='woe'))[:, 1]
gini = 2 * roc_auc_score(y_test_orig, lr_all_score_test) - 1
print(f'LogReg (все feat WoE) Gini на TARGET: {gini:.4f}')''',
    'cell-risk-lr'),

    code('''# Сводная таблица: качество risk-моделей на исходном TARGET
print(f\'{"Модель":<35} {"Gini (TARGET, test)":>20}\')
print(\'-\' * 58)
for name, pred in [
    (\'CatBoost (22-отобр., TARGET)\', cb_saved_score_test),
    (\'CatBoost (все feat, TARGET)\',  cb_all_score_test),
    (\'LogReg (8 WoE, TARGET)\',       lr_saved_score_test),
    (\'LogReg (все feat, TARGET)\',    lr_all_score_test),
]:
    auc = roc_auc_score(y_test_orig, pred)
    print(f\'{name:<35} {2*auc-1:>20.4f}\')''',
    'cell-risk-table'),
])

# ══════════════════════════════════════════════════════════════════════════════
# 2. S-LEARNER: c097afd8 -> encoder+функция / матрицы+обучение
# ══════════════════════════════════════════════════════════════════════════════
replace_with('c097afd8', [

    code('''from catboost import CatBoostClassifier

# One-hot кодируем treatment: каждый канал -> бинарная колонка COMM_<channel>
# fit только на T_train — encoder не видит test/oot
comm_encoder = OneHotEncoder(sparse_output=False, handle_unknown=\'ignore\')
comm_encoder.fit(T_train.reshape(-1, 1))

def add_treatment_ohe(X_df, T_arr, encoder):
    """Присоединяет one-hot представление treatment к матрице признаков клиента."""
    t_enc = encoder.transform(T_arr.reshape(-1, 1))
    t_df  = pd.DataFrame(t_enc,
                          columns=[f\'COMM_{c}\' for c in encoder.categories_[0]],
                          index=range(len(X_df)))
    return pd.concat([X_df.reset_index(drop=True), t_df], axis=1)

print(\'Treatment categories:\', comm_encoder.categories_[0].tolist())
print(\'OHE columns:\', [f\'COMM_{c}\' for c in comm_encoder.categories_[0]])''',
    'cell-s-encoder'),

    code('''# Добавляем treatment-признаки к матрицам признаков.
# S-Learner видит X и T одновременно — единая модель для всех каналов.
X_train_s = add_treatment_ohe(X_train, T_train, comm_encoder)
X_test_s  = add_treatment_ohe(X_test,  T_test,  comm_encoder)
X_oot_s   = add_treatment_ohe(X_oot,   T_oot,   comm_encoder)

print(f\'Размерность X_train_s: {X_train_s.shape}  \')
print(f\'(добавлено {X_train_s.shape[1] - X_train.shape[1]} OHE-колонки)\')

s_model = CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=6,
    random_seed=RANDOM_SEED, verbose=0,
    eval_metric=\'AUC\', early_stopping_rounds=50,
)
s_model.fit(X_train_s, y_train, eval_set=(X_test_s, y_test))

auc_s = roc_auc_score(y_test, s_model.predict_proba(X_test_s)[:, 1])
print(f\'S-Learner AUC (test): {auc_s:.4f}  |  Gini: {2*auc_s-1:.4f}\')''',
    'cell-s-train'),
])

# ══════════════════════════════════════════════════════════════════════════════
# 3. S-LEARNER: 454af9cb -> функция / предсказание+AUUC
# ══════════════════════════════════════════════════════════════════════════════
replace_with('454af9cb', [

    code('''channels_list = [\'sms\', \'robot_call\', \'operator_call\']

def s_learner_uplift_per_channel(model, X_df, encoder, channels):
    """Вычисляет uplift S-Learner: hat_tau_t(x) = hat_mu(x,t) - hat_mu(x,control).

    Для каждого клиента вычисляется контрфактуальный сценарий:
    - p_ctrl: вероятность дефолта без коммуникации
    - p_ch:   вероятность дефолта при канале ch
    Uplift = p_ch - p_ctrl (отрицательный -> коммуникация помогает).
    'best' = min по каналам (наибольшее снижение PD).
    """
    X_ctrl = add_treatment_ohe(X_df, np.array([\'control\'] * len(X_df)), encoder)
    p_ctrl = model.predict_proba(X_ctrl)[:, 1]
    uplift = {}
    for ch in channels:
        X_ch = add_treatment_ohe(X_df, np.array([ch] * len(X_df)), encoder)
        uplift[ch] = model.predict_proba(X_ch)[:, 1] - p_ctrl
    uplift_matrix = pd.DataFrame({ch: uplift[ch] for ch in channels})
    uplift[\'best\'] = uplift_matrix.min(axis=1).values
    return uplift''',
    'cell-s-func'),

    code('''uplift_s_test = s_learner_uplift_per_channel(s_model, X_test, comm_encoder, channels_list)
uplift_s_oot  = s_learner_uplift_per_channel(s_model, X_oot,  comm_encoder, channels_list)

# score = -uplift[\'best\']: чем больше снижение PD, тем выше приоритет
score_s_test = -uplift_s_test[\'best\']
score_s_oot  = -uplift_s_oot[\'best\']

_, _, _, auuc_s_test = compute_qini(y_test, t_test_bin, score_s_test)
_, _, _, auuc_s_oot  = compute_qini(y_oot,  t_oot_bin,  score_s_oot)
oracle_auuc = baseline_auuc[\'Oracle (TRUE_UPLIFT)\']

print(f\'S-Learner AUUC: test={auuc_s_test:.6f}  |  OOT={auuc_s_oot:.6f}\')
print(f\'Относительная эффективность: {auuc_s_test/oracle_auuc*100:.1f}% от оракула\')
for ch in channels_list:
    print(f\'  Средний uplift ({ch}): {uplift_s_test[ch].mean():.5f}\')''',
    'cell-s-auuc'),
])

# ══════════════════════════════════════════════════════════════════════════════
# 4. T-LEARNER: 544debd2 -> функция / предсказание+AUUC
# ══════════════════════════════════════════════════════════════════════════════
replace_with('544debd2', [

    code('''def t_learner_uplift(models, X_df, channels):
    """Вычисляет uplift T-Learner: hat_tau_t(x) = hat_mu_t(x) - hat_mu_0(x).

    Каждая модель обучалась только на своей группе treatment.
    Uplift = разность предсказаний канальной и контрольной моделей.
    'best' = min по каналам.
    """
    p_ctrl = models[\'control\'].predict_proba(X_df)[:, 1]
    uplift = {}
    for ch in channels:
        uplift[ch] = models[ch].predict_proba(X_df)[:, 1] - p_ctrl
    uplift_matrix = pd.DataFrame({ch: uplift[ch] for ch in channels})
    uplift[\'best\'] = uplift_matrix.min(axis=1).values
    return uplift''',
    'cell-t-func'),

    code('''uplift_t_test = t_learner_uplift(t_models, X_test, channels_list)
uplift_t_oot  = t_learner_uplift(t_models, X_oot,  channels_list)

score_t_test = -uplift_t_test[\'best\']
score_t_oot  = -uplift_t_oot[\'best\']

_, _, _, auuc_t_test = compute_qini(y_test, t_test_bin, score_t_test)
_, _, _, auuc_t_oot  = compute_qini(y_oot,  t_oot_bin,  score_t_oot)
print(f\'T-Learner AUUC: test={auuc_t_test:.6f}  |  OOT={auuc_t_oot:.6f}\')
print(f\'Относительная эффективность: {auuc_t_test/oracle_auuc*100:.1f}% от оракула\')
for ch in channels_list:
    print(f\'  Средний uplift ({ch}): {uplift_t_test[ch].mean():.5f}\')''',
    'cell-t-auuc'),
])

# ══════════════════════════════════════════════════════════════════════════════
# 5. X-LEARNER: e62838c6 -> tau-регрессоры / propensity+функция / предсказание
# ══════════════════════════════════════════════════════════════════════════════
replace_with('e62838c6', [

    code('''# Шаг 3: регрессоры на псевдо-эффектах
# tau1 обучается на treated-группе (d_treat = y - mu0(x)),
# tau0 обучается на контрольной группе (d_ctrl = mu1(x) - y).
# Это "скрещивание" даёт X-Learner его имя.
tau1_model = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=5,
                                random_seed=RANDOM_SEED, verbose=0)
tau0_model = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=5,
                                random_seed=RANDOM_SEED, verbose=0)
tau1_model.fit(X_train[mask_treat], d_treat)
tau0_model.fit(X_train[mask_ctrl],  d_ctrl)
print(f\'tau1 обучен на {mask_treat.sum():,} treated  |  tau0 на {mask_ctrl.sum():,} контрольных\')''',
    'cell-x-tau'),

    code('''# Шаг 4: propensity score g(x) = P(T=1|X=x)
# Взвешивание: hat_tau(x) = g(x)*hat_tau0(x) + (1-g(x))*hat_tau1(x)
# Клиенты с высоким g(x) похожи на treated -> доверяем tau0 больше.
prop_model = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=4,
                                 random_seed=RANDOM_SEED, verbose=0)
prop_model.fit(X_train, t_train_bin)

g_test = prop_model.predict_proba(X_test)[:, 1]
print(f\'Propensity (test): mean={g_test.mean():.4f}  |  \')
print(f\'treated-доля в train: {t_train_bin.mean():.4f}\')

def x_learner_predict(tau1, tau0, prop, X):
    """Предсказывает CATE: propensity-взвешенное смешение tau0 и tau1."""
    g = prop.predict_proba(X)[:, 1]
    return g * tau0.predict(X) + (1 - g) * tau1.predict(X)''',
    'cell-x-prop'),

    code('''uplift_x_test = x_learner_predict(tau1_model, tau0_model, prop_model, X_test)
uplift_x_oot  = x_learner_predict(tau1_model, tau0_model, prop_model, X_oot)

score_x_test = -uplift_x_test
score_x_oot  = -uplift_x_oot

_, _, _, auuc_x_test = compute_qini(y_test, t_test_bin, score_x_test)
_, _, _, auuc_x_oot  = compute_qini(y_oot,  t_oot_bin,  score_x_oot)
print(f\'X-Learner AUUC: test={auuc_x_test:.6f}  |  OOT={auuc_x_oot:.6f}\')
print(f\'Относительная эффективность: {auuc_x_test/oracle_auuc*100:.1f}% от оракула\')
print(f\'Средний uplift (test): {uplift_x_test.mean():.5f}\')''',
    'cell-x-auuc'),
])

# ══════════════════════════════════════════════════════════════════════════════
# 6. DR-LEARNER: ячейка с dr_pseudo -> псевдо-исходы / финальная модель+AUUC
# ══════════════════════════════════════════════════════════════════════════════
for i, c in enumerate(cells):
    if c.get('id') is None and 'dr_pseudo' in ''.join(c['source']) and 'e_clipped' in ''.join(c['source']):
        cells[i:i+1] = [

            code('''# DR псевдо-исходы: doubly robust оценка каузального эффекта
# dr(x) = (mu1(x) - mu0(x))  +  IPW-поправка
# IPW-поправка: (T - e(x)) / (e(x)*(1-e(x))) * (Y - mu_at_obs)
# При правильной хотя бы одной модели (outcome или propensity) оценка несмещена.
e_clipped  = np.clip(e_oof, 0.05, 0.95)   # клиппинг предотвращает деление на ~0
ipw_weight = (t_train_bin - e_clipped) / (e_clipped * (1 - e_clipped))
mu_at_obs  = t_train_bin * mu1_oof + (1 - t_train_bin) * mu0_oof
dr_pseudo  = (mu1_oof - mu0_oof) + ipw_weight * (y_train - mu_at_obs)

print(f\'DR псевдо-исходы: mean={dr_pseudo.mean():.5f}, std={dr_pseudo.std():.5f}\')
print(f\'Доля с отрицательным эффектом (коммуникация помогает): {(dr_pseudo < 0).mean():.1%}\')
print(\'Высокий std -- ожидаем следствие IPW-дисперсии при малом SNR.\')''',
            'cell-dr-pseudo'),

            code('''# Финальная регрессия: CatBoostRegressor учит dr_pseudo -> hat_tau(x)
dr_final = CatBoostRegressor(iterations=400, learning_rate=0.05, depth=6,
                               random_seed=RANDOM_SEED, verbose=0)
dr_final.fit(X_train_arr, dr_pseudo)

uplift_dr_test = dr_final.predict(X_test.values)
uplift_dr_oot  = dr_final.predict(X_oot.values)

score_dr_test = -uplift_dr_test
score_dr_oot  = -uplift_dr_oot

_, _, _, auuc_dr_test = compute_qini(y_test, t_test_bin, score_dr_test)
_, _, _, auuc_dr_oot  = compute_qini(y_oot,  t_oot_bin,  score_dr_oot)
print(f\'DR-Learner AUUC: test={auuc_dr_test:.6f}  |  OOT={auuc_dr_oot:.6f}\')
print(f\'Относительная эффективность: {auuc_dr_test/oracle_auuc*100:.1f}% от оракула\')''',
            'cell-dr-final'),
        ]
        print(f'DR cell split at index {i}')
        break

# ── Запись ─────────────────────────────────────────────────────────────────────
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(f'Готово. Ячеек в ноутбуке: {len(cells)}')
