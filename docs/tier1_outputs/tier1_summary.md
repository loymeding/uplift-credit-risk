# Tier 1 methodology checks: summary

## 1. Two oracle levels

- Test current-policy prevented PD sum: `149.8990`.
- Test counterfactual-oracle prevented PD sum: `1038.9017`.
- Raw effect share current/counterfactual: `14.43%`.
- Net value share current/counterfactual: `14.45%`.
- Current contact rate: `9.75%`; counterfactual oracle contact rate: `97.89%`.

Files:
- `oracle_policy_value_summary.csv`
- `oracle_channel_distribution_test.csv`
- `fig_oracle_channel_distribution.png`

## 2. Overlap and SMD

Top SMD before weighting:

| feature            |   smd_before |   smd_after_ipw |   control_mean |   treated_mean |
|:-------------------|-------------:|----------------:|---------------:|---------------:|
| CONTACT_PROPENSITY |       0.7162 |         -0.0002 |         0.0905 |         0.1721 |
| BASE_PD            |       0.6361 |         -0.0016 |         0.0683 |         0.1861 |
| EXT_SOURCE_1       |      -0.428  |          0.0015 |         0.5123 |         0.4212 |
| EXT_SOURCE_3       |      -0.3829 |         -0.0038 |         0.5186 |         0.4421 |
| AMT_CREDIT         |       0.3373 |          0.004  |    584912      |    727388      |
| AMT_ANNUITY        |       0.3314 |          0.0028 |     26608.9    |     31653.6    |
| EXT_SOURCE_2       |      -0.3104 |         -0.0096 |         0.5206 |         0.4582 |
| SK_DPD_MAX_POS     |       0.2172 |          0.0032 |         9.8971 |        68.683  |

Files:
- `propensity_overlap_summary.csv`
- `smd_balance_before_after_ipw.csv`
- `fig_propensity_overlap.png`
- `fig_smd_love_plot.png`

## 3. AUUC bootstrap CI

| model                |   auuc_original |   ci_2_5 |   ci_97_5 |
|:---------------------|----------------:|---------:|----------:|
| Logged-policy oracle |        435.318  | 405.069  |   459.397 |
| BASE_PD              |        365.925  | 340.214  |   391.888 |
| T-Learner            |        359.846  | 334.674  |   384.472 |
| S-Learner            |        335.533  | 310.13   |   358.746 |
| CatBoost-risk (all)  |        239.56   | 220.923  |   261.234 |
| LogReg-risk (all)    |        236.174  | 217.229  |   257.037 |
| X-Learner            |        153.738  | 131.352  |   172.146 |
| DR-Learner           |        141.514  | 121.789  |   162.491 |
| Случайный выбор      |         62.7321 |  46.0898 |    80.043 |

Paired differences:

| comparison                           |   diff_original |   diff_boot_mean |   ci_2_5 |   ci_97_5 | significant_95pct   |
|:-------------------------------------|----------------:|-----------------:|---------:|----------:|:--------------------|
| T-Learner minus S-Learner            |         24.3131 |          24.1031 |  19.3417 |   29.3189 | True                |
| T-Learner minus BASE_PD              |         -6.0791 |          -6.0086 |  -9.817  |   -2.4804 | True                |
| S-Learner minus BASE_PD              |        -30.3923 |         -30.1117 | -36.4679 |  -24.3646 | True                |
| T-Learner minus CatBoost-risk (all)  |        120.286  |         119.864  | 102.572  |  136.685  | True                |
| S-Learner minus CatBoost-risk (all)  |         95.9727 |          95.761  |  80.0772 |  111.477  | True                |
| T-Learner minus Logged-policy oracle |        -75.4713 |         -72.4819 | -85.0574 |  -58.7722 | True                |
