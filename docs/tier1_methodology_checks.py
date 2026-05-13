"""
Tier 1 methodology checks for the uplift thesis.

This script implements the first block of methodological corrections:
1. Separate logged-policy oracle and counterfactual channel oracle.
2. Estimate current policy vs counterfactual oracle business value.
3. Diagnose propensity overlap and SMD balance.
4. Compute paired bootstrap confidence intervals for AUUC.

Outputs are saved to docs/tier1_outputs/.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANDOM_SEED = 91
N_BOOTSTRAP = 500
N_BINS_QINI = 100

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "docs" / "tier1_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_DATA_PATH = DATA_DIR / "uplift-dataset.csv"
SCORES_PATH = DATA_DIR / "uplift_scores.csv"

COMM_COSTS = {
    "control": 0.0,
    "sms": 5.0,
    "robot_call": 30.0,
    "operator_call": 200.0,
}
LOSS_GIVEN_DEFAULT_RUB = 240_000.0


def read_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def load_scored_dataset() -> pd.DataFrame:
    """Load test/OOT scores and attach oracle/probability columns from full data."""
    scores = pd.read_csv(SCORES_PATH)

    full_header = read_header(FULL_DATA_PATH)
    needed = [
        "COMMUNICATION",
        "ORACLE_COMMUNICATION",
        "TRUE_UPLIFT",
        "ORACLE_TRUE_UPLIFT",
        "BASE_PD",
        "CONTACT_PROPENSITY",
        "TARGET_AFTER_CONTACT",
        "PD_NO_CONTACT",
        "PD_AFTER_CONTACT",
        "PD_SMS",
        "PD_ROBOT_CALL",
        "PD_OPERATOR_CALL",
        "UPLIFT_SMS",
        "UPLIFT_ROBOT_CALL",
        "UPLIFT_OPERATOR_CALL",
        "RISK_SEGMENT",
    ]
    needed = [c for c in needed if c in full_header]
    full = pd.read_csv(FULL_DATA_PATH, usecols=needed)

    n = len(full)
    oot_size = math.ceil(n * 0.2)
    train_size = math.ceil((n - oot_size) * 0.75)
    oot = full.iloc[:oot_size].copy()
    test = full.iloc[oot_size + train_size :].copy()

    oracle_part = pd.concat([test, oot], axis=0).reset_index(drop=True)
    if len(scores) != len(oracle_part):
        raise ValueError(f"Scores rows ({len(scores)}) != reconstructed test+OOT rows ({len(oracle_part)})")

    # Sanity checks: uplift_scores.csv should be test followed by OOT in the same row order.
    if not np.allclose(scores["BASE_PD"].to_numpy(), oracle_part["BASE_PD"].to_numpy(), atol=1e-12):
        raise ValueError("BASE_PD mismatch between uplift_scores.csv and reconstructed split.")
    if not np.allclose(scores["TRUE_UPLIFT"].to_numpy(), oracle_part["TRUE_UPLIFT"].to_numpy(), atol=1e-12):
        raise ValueError("TRUE_UPLIFT mismatch between uplift_scores.csv and reconstructed split.")
    if not (scores["COMMUNICATION"].to_numpy() == oracle_part["COMMUNICATION"].to_numpy()).all():
        raise ValueError("COMMUNICATION mismatch between uplift_scores.csv and reconstructed split.")

    extra_cols = [c for c in oracle_part.columns if c not in scores.columns]
    return pd.concat([scores.reset_index(drop=True), oracle_part[extra_cols].reset_index(drop=True)], axis=1)


def load_full_for_diagnostics() -> pd.DataFrame:
    """Load selected full-data columns for overlap/SMD diagnostics."""
    full_header = read_header(FULL_DATA_PATH)
    requested = [
        "COMMUNICATION",
        "CONTACT_PROPENSITY",
        "BASE_PD",
        "TARGET",
        "TARGET_AFTER_CONTACT",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_INCOME_TOTAL",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "DAYS_LAST_PHONE_CHANGE",
        "REGION_RATING_CLIENT",
        "SK_DPD_MAX_POS",
        "DAYS_CREDIT_MEAN",
    ]
    usecols = [c for c in requested if c in full_header]
    return pd.read_csv(FULL_DATA_PATH, usecols=usecols)


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(w)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(x[mask], weights=w[mask]))


def weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(w)
    if mask.sum() == 0:
        return np.nan
    xm = x[mask]
    wm = w[mask]
    mu = np.average(xm, weights=wm)
    return float(np.average((xm - mu) ** 2, weights=wm))


def smd_for_feature(x: pd.Series, treated: np.ndarray, weights: np.ndarray | None = None) -> float:
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    t = treated.astype(bool)

    if weights is None:
        xt = arr[t]
        xc = arr[~t]
        mt = np.nanmean(xt)
        mc = np.nanmean(xc)
        vt = np.nanvar(xt)
        vc = np.nanvar(xc)
    else:
        w = weights.astype(float)
        mt = weighted_mean(arr[t], w[t])
        mc = weighted_mean(arr[~t], w[~t])
        vt = weighted_var(arr[t], w[t])
        vc = weighted_var(arr[~t], w[~t])

    pooled = np.sqrt((vt + vc) / 2.0)
    if not np.isfinite(pooled) or pooled == 0:
        return np.nan
    return float((mt - mc) / pooled)


def compute_qini_auuc(y: np.ndarray, treatment: np.ndarray, score: np.ndarray, n_bins: int = N_BINS_QINI) -> float:
    """Vectorized version of the notebook Qini/AUUC implementation."""
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    score = np.asarray(score, dtype=float)

    valid = np.isfinite(y) & np.isfinite(treatment) & np.isfinite(score)
    y = y[valid]
    treatment = treatment[valid]
    score = score[valid]

    if len(y) == 0:
        return np.nan

    order = np.argsort(-score, kind="mergesort")
    y = y[order]
    treatment = treatment[order]

    n = len(y)
    n_ctrl = int((1 - treatment).sum())
    if n_ctrl == 0:
        return 0.0

    step = max(1, n // n_bins)
    steps = list(range(step, n + 1, step))
    if steps[-1] != n:
        steps.append(n)

    y_t = y * treatment
    y_c = y * (1 - treatment)
    cum_y_t = np.cumsum(y_t)
    cum_y_c = np.cumsum(y_c)
    cum_t = np.cumsum(treatment)

    fracs = [0.0]
    qini_vals = [0.0]
    for k in steps:
        idx = k - 1
        qini = cum_y_t[idx] - cum_y_c[idx] * (cum_t[idx] / n_ctrl)
        fracs.append(k / n)
        qini_vals.append(float(qini))

    fracs_arr = np.asarray(fracs)
    qini_arr = np.asarray(qini_vals)
    random_arr = fracs_arr * qini_arr[-1]
    return float(np.trapz(qini_arr, fracs_arr) - np.trapz(random_arr, fracs_arr))


def save_oracle_policy_outputs(scored: pd.DataFrame, full_diag: pd.DataFrame) -> None:
    rows = []
    for split_name, df in [
        ("test", scored[scored["split"] == "test"].copy()),
        ("oot", scored[scored["split"] == "oot"].copy()),
        ("test+oot", scored.copy()),
    ]:
        current_prevented = -df["TRUE_UPLIFT"]
        counter_prevented = -df["ORACLE_TRUE_UPLIFT"]
        current_cost = df["COMMUNICATION"].map(COMM_COSTS).fillna(0.0)
        counter_cost = df["ORACLE_COMMUNICATION"].map(COMM_COSTS).fillna(0.0)

        current_value = current_prevented.sum() * LOSS_GIVEN_DEFAULT_RUB
        counter_value = counter_prevented.sum() * LOSS_GIVEN_DEFAULT_RUB
        current_net = current_value - current_cost.sum()
        counter_net = counter_value - counter_cost.sum()

        rows.append(
            {
                "split": split_name,
                "n": len(df),
                "current_prevented_pd_sum": current_prevented.sum(),
                "counterfactual_prevented_pd_sum": counter_prevented.sum(),
                "current_value_mln_rub": current_value / 1_000_000,
                "counterfactual_value_mln_rub": counter_value / 1_000_000,
                "current_cost_mln_rub": current_cost.sum() / 1_000_000,
                "counterfactual_cost_mln_rub": counter_cost.sum() / 1_000_000,
                "current_net_mln_rub": current_net / 1_000_000,
                "counterfactual_net_mln_rub": counter_net / 1_000_000,
                "raw_effect_share_current_vs_counterfactual": current_prevented.sum() / counter_prevented.sum(),
                "net_share_current_vs_counterfactual": current_net / counter_net,
                "current_contact_rate_pct": (df["COMMUNICATION"] != "control").mean() * 100,
                "counterfactual_contact_rate_pct": (df["ORACLE_COMMUNICATION"] != "control").mean() * 100,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "oracle_policy_value_summary.csv", index=False, encoding="utf-8-sig")

    channel_rows = []
    for col, label in [
        ("COMMUNICATION", "current_logged_policy"),
        ("ORACLE_COMMUNICATION", "counterfactual_channel_oracle"),
    ]:
        dist = scored[scored["split"] == "test"][col].value_counts(normalize=True).mul(100)
        for channel, pct in dist.items():
            channel_rows.append({"policy": label, "channel": channel, "pct_test": pct})
    channel_df = pd.DataFrame(channel_rows)
    channel_df.to_csv(OUT_DIR / "oracle_channel_distribution_test.csv", index=False, encoding="utf-8-sig")

    pivot = channel_df.pivot(index="channel", columns="policy", values="pct_test").fillna(0.0)
    pivot = pivot.reindex(["control", "sms", "robot_call", "operator_call"])
    ax = pivot.plot(kind="bar", figsize=(9, 5), color=["#6b7280", "#2563eb"])
    ax.set_title("Текущая политика и контрфактический оракул: распределение каналов")
    ax.set_xlabel("Канал коммуникации")
    ax.set_ylabel("Доля клиентов на test, %")
    ax.legend(["Текущая политика", "Контрфактический оракул"])
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_oracle_channel_distribution.png", dpi=200)
    plt.close()


def save_overlap_smd_outputs(full: pd.DataFrame) -> None:
    df = full.copy()
    df["treated"] = (df["COMMUNICATION"] != "control").astype(int)
    e = df["CONTACT_PROPENSITY"].clip(0.01, 0.99).to_numpy()
    treated = df["treated"].to_numpy(dtype=int)
    weights = np.where(treated == 1, 1.0 / e, 1.0 / (1.0 - e))

    numeric_features = [
        c
        for c in df.columns
        if c not in {"COMMUNICATION", "treated", "TARGET_AFTER_CONTACT", "TARGET"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    rows = []
    for feature in numeric_features:
        rows.append(
            {
                "feature": feature,
                "smd_before": smd_for_feature(df[feature], treated, weights=None),
                "smd_after_ipw": smd_for_feature(df[feature], treated, weights=weights),
                "control_mean": pd.to_numeric(df.loc[treated == 0, feature], errors="coerce").mean(),
                "treated_mean": pd.to_numeric(df.loc[treated == 1, feature], errors="coerce").mean(),
            }
        )
    smd = pd.DataFrame(rows)
    smd["abs_smd_before"] = smd["smd_before"].abs()
    smd["abs_smd_after_ipw"] = smd["smd_after_ipw"].abs()
    smd = smd.sort_values("abs_smd_before", ascending=False)
    smd.to_csv(OUT_DIR / "smd_balance_before_after_ipw.csv", index=False, encoding="utf-8-sig")

    prop_rows = []
    for label, part in [("control", df[df["treated"] == 0]), ("contacted", df[df["treated"] == 1])]:
        q = part["CONTACT_PROPENSITY"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        prop_rows.append(
            {
                "group": label,
                "n": len(part),
                "mean": part["CONTACT_PROPENSITY"].mean(),
                "std": part["CONTACT_PROPENSITY"].std(),
                **{f"q{int(k * 100):02d}": v for k, v in q.items()},
            }
        )
    prop_summary = pd.DataFrame(prop_rows)
    prop_summary.to_csv(OUT_DIR / "propensity_overlap_summary.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(9, 5))
    bins = np.linspace(0, min(1.0, df["CONTACT_PROPENSITY"].quantile(0.995)), 60)
    plt.hist(
        df.loc[df["treated"] == 0, "CONTACT_PROPENSITY"],
        bins=bins,
        alpha=0.65,
        density=True,
        label="Control: без коммуникации",
        color="#6b7280",
    )
    plt.hist(
        df.loc[df["treated"] == 1, "CONTACT_PROPENSITY"],
        bins=bins,
        alpha=0.65,
        density=True,
        label="Contacted: была коммуникация",
        color="#2563eb",
    )
    plt.title("Диагностика overlap: распределение propensity score")
    plt.xlabel("Вероятность контакта CONTACT_PROPENSITY")
    plt.ylabel("Плотность")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_propensity_overlap.png", dpi=200)
    plt.close()

    top = smd.head(12).copy().sort_values("abs_smd_before", ascending=True)
    y_pos = np.arange(len(top))
    plt.figure(figsize=(9, 6))
    plt.scatter(top["abs_smd_before"], y_pos, label="До взвешивания", color="#dc2626")
    plt.scatter(top["abs_smd_after_ipw"], y_pos, label="После IPW", color="#2563eb")
    plt.axvline(0.1, color="black", linestyle="--", linewidth=1, alpha=0.7, label="Порог 0.1")
    plt.axvline(0.2, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="Порог 0.2")
    plt.yticks(y_pos, top["feature"])
    plt.xlabel("|SMD|: стандартизированная разница средних")
    plt.title("Баланс treatment/control по ключевым признакам")
    plt.legend(loc="lower right")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_smd_love_plot.png", dpi=200)
    plt.close()


def strategy_scores(df_test: pd.DataFrame) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(RANDOM_SEED)
    scores = {
        "Случайный выбор": rng.rand(len(df_test)),
        "BASE_PD": df_test["BASE_PD"].to_numpy(),
        "CatBoost-risk (all)": df_test["score_cb_all"].to_numpy(),
        "LogReg-risk (all)": df_test["score_lr_all"].to_numpy(),
        "S-Learner": df_test["score_s"].to_numpy(),
        "T-Learner": df_test["score_t"].to_numpy(),
        "X-Learner": df_test["score_x"].to_numpy(),
        "DR-Learner": df_test["score_dr"].to_numpy(),
        "Logged-policy oracle": (-df_test["TRUE_UPLIFT"]).to_numpy(),
    }
    return scores


def save_bootstrap_auuc_outputs(scored: pd.DataFrame) -> None:
    df_test = scored[scored["split"] == "test"].reset_index(drop=True)
    y = df_test["y"].to_numpy(dtype=float)
    t = df_test["treatment_bin"].to_numpy(dtype=int)
    scores = strategy_scores(df_test)

    original = {name: compute_qini_auuc(y, t, score) for name, score in scores.items()}

    rng = np.random.RandomState(RANDOM_SEED)
    boot = {name: np.empty(N_BOOTSTRAP, dtype=float) for name in scores}
    n = len(df_test)

    for b in range(N_BOOTSTRAP):
        idx = rng.choice(n, size=n, replace=True)
        yb = y[idx]
        tb = t[idx]
        for name, score in scores.items():
            boot[name][b] = compute_qini_auuc(yb, tb, score[idx])

    rows = []
    for name, vals in boot.items():
        rows.append(
            {
                "model": name,
                "auuc_original": original[name],
                "auuc_boot_mean": vals.mean(),
                "ci_2_5": np.percentile(vals, 2.5),
                "ci_97_5": np.percentile(vals, 97.5),
            }
        )
    auuc_summary = pd.DataFrame(rows).sort_values("auuc_original", ascending=False)
    auuc_summary.to_csv(OUT_DIR / "auuc_bootstrap_ci.csv", index=False, encoding="utf-8-sig")

    pairs = [
        ("T-Learner", "S-Learner"),
        ("T-Learner", "BASE_PD"),
        ("S-Learner", "BASE_PD"),
        ("T-Learner", "CatBoost-risk (all)"),
        ("S-Learner", "CatBoost-risk (all)"),
        ("T-Learner", "Logged-policy oracle"),
    ]
    diff_rows = []
    for left, right in pairs:
        diff = boot[left] - boot[right]
        diff_rows.append(
            {
                "comparison": f"{left} minus {right}",
                "diff_original": original[left] - original[right],
                "diff_boot_mean": diff.mean(),
                "ci_2_5": np.percentile(diff, 2.5),
                "ci_97_5": np.percentile(diff, 97.5),
                "significant_95pct": (np.percentile(diff, 2.5) > 0) or (np.percentile(diff, 97.5) < 0),
            }
        )
    diff_summary = pd.DataFrame(diff_rows)
    diff_summary.to_csv(OUT_DIR / "auuc_paired_bootstrap_differences.csv", index=False, encoding="utf-8-sig")

    plot_df = auuc_summary.sort_values("auuc_original", ascending=True)
    y_pos = np.arange(len(plot_df))
    err_low = plot_df["auuc_original"] - plot_df["ci_2_5"]
    err_high = plot_df["ci_97_5"] - plot_df["auuc_original"]
    plt.figure(figsize=(10, 6))
    plt.barh(y_pos, plot_df["auuc_original"], xerr=[err_low, err_high], color="#2563eb", alpha=0.8)
    plt.yticks(y_pos, plot_df["model"])
    plt.xlabel("AUUC на test")
    plt.title(f"Bootstrap 95% CI для AUUC (B={N_BOOTSTRAP})")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_auuc_bootstrap_ci.png", dpi=200)
    plt.close()

    diff_plot = diff_summary.sort_values("diff_original", ascending=True)
    y_pos = np.arange(len(diff_plot))
    err_low = diff_plot["diff_original"] - diff_plot["ci_2_5"]
    err_high = diff_plot["ci_97_5"] - diff_plot["diff_original"]
    colors = np.where(diff_plot["significant_95pct"], "#16a34a", "#9ca3af")
    plt.figure(figsize=(10, 5))
    plt.barh(y_pos, diff_plot["diff_original"], xerr=[err_low, err_high], color=colors, alpha=0.85)
    plt.axvline(0, color="black", linewidth=1)
    plt.yticks(y_pos, diff_plot["comparison"])
    plt.xlabel("Разница AUUC")
    plt.title("Paired bootstrap 95% CI для разницы AUUC")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_auuc_paired_differences.png", dpi=200)
    plt.close()


def write_markdown_summary() -> None:
    oracle = pd.read_csv(OUT_DIR / "oracle_policy_value_summary.csv")
    smd = pd.read_csv(OUT_DIR / "smd_balance_before_after_ipw.csv")
    auuc = pd.read_csv(OUT_DIR / "auuc_bootstrap_ci.csv")
    diffs = pd.read_csv(OUT_DIR / "auuc_paired_bootstrap_differences.csv")

    test_oracle = oracle[oracle["split"] == "test"].iloc[0]
    lines = [
        "# Tier 1 methodology checks: summary",
        "",
        "## 1. Two oracle levels",
        "",
        f"- Test current-policy prevented PD sum: `{test_oracle['current_prevented_pd_sum']:.4f}`.",
        f"- Test counterfactual-oracle prevented PD sum: `{test_oracle['counterfactual_prevented_pd_sum']:.4f}`.",
        f"- Raw effect share current/counterfactual: `{test_oracle['raw_effect_share_current_vs_counterfactual']:.2%}`.",
        f"- Net value share current/counterfactual: `{test_oracle['net_share_current_vs_counterfactual']:.2%}`.",
        f"- Current contact rate: `{test_oracle['current_contact_rate_pct']:.2f}%`; counterfactual oracle contact rate: `{test_oracle['counterfactual_contact_rate_pct']:.2f}%`.",
        "",
        "Files:",
        "- `oracle_policy_value_summary.csv`",
        "- `oracle_channel_distribution_test.csv`",
        "- `fig_oracle_channel_distribution.png`",
        "",
        "## 2. Overlap and SMD",
        "",
        "Top SMD before weighting:",
        "",
        smd[["feature", "smd_before", "smd_after_ipw", "control_mean", "treated_mean"]]
        .head(8)
        .round(4)
        .to_markdown(index=False),
        "",
        "Files:",
        "- `propensity_overlap_summary.csv`",
        "- `smd_balance_before_after_ipw.csv`",
        "- `fig_propensity_overlap.png`",
        "- `fig_smd_love_plot.png`",
        "",
        "## 3. AUUC bootstrap CI",
        "",
        auuc[["model", "auuc_original", "ci_2_5", "ci_97_5"]].round(4).to_markdown(index=False),
        "",
        "Paired differences:",
        "",
        diffs.round(4).to_markdown(index=False),
        "",
    ]
    (OUT_DIR / "tier1_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("Loading scored test/OOT dataset...")
    scored = load_scored_dataset()
    print(f"Scored rows: {len(scored):,}")

    print("Loading full dataset for diagnostics...")
    full_diag = load_full_for_diagnostics()
    print(f"Diagnostic rows: {len(full_diag):,}")

    print("Saving oracle policy outputs...")
    save_oracle_policy_outputs(scored, full_diag)

    print("Saving overlap and SMD diagnostics...")
    save_overlap_smd_outputs(full_diag)

    print(f"Running AUUC paired bootstrap (B={N_BOOTSTRAP})...")
    save_bootstrap_auuc_outputs(scored)

    print("Writing markdown summary...")
    write_markdown_summary()
    print(f"Done. Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
