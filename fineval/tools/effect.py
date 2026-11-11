"""Оценка фактического эффекта: разность разностей с двусторонними фиксированными эффектами.

Модель:  y_it = α_i + γ_t + β · (pilot_i × post_t) + ε_it
  α_i — фиксированный эффект клиента (разный уровень трат),
  γ_t — эффект месяца (сезонность, макро),
  β   — ITT-эффект: изменение метрики на клиента пилотного региона в месяц.
Стандартные ошибки кластеризуются по региону — уровню, на котором назначался пилот.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from fineval.context import EvalContext

METRICS = ["net_income", "interchange_income", "nii_income", "net_cashback_cost", "spend", "avg_balance", "is_active"]
MIN_CLUSTERS_PER_ARM = 10


def _demean_twfe(df: pd.DataFrame, cols: list[str], i: str = "client_id", t: str = "month",
                 tol: float = 1e-8, max_iter: int = 100) -> np.ndarray:
    """Убирает эффекты клиента и месяца (alternating projections; работает и для несбалансированной панели)."""
    X = df[cols].to_numpy(dtype=float).copy()
    gi = df[i].to_numpy()
    gt = df[t].to_numpy()
    _, ii = np.unique(gi, return_inverse=True)
    _, tt = np.unique(gt, return_inverse=True)
    ni, nt = ii.max() + 1, tt.max() + 1
    ci, ct = np.bincount(ii, minlength=ni), np.bincount(tt, minlength=nt)
    for _ in range(max_iter):
        prev = X.copy()
        for j in range(X.shape[1]):
            X[:, j] -= (np.bincount(ii, X[:, j], ni) / ci)[ii]
            X[:, j] -= (np.bincount(tt, X[:, j], nt) / ct)[tt]
        if np.max(np.abs(X - prev)) < tol * (1 + np.max(np.abs(X))):
            break
    return X


def _ols_cluster(y: np.ndarray, X: np.ndarray, clusters: np.ndarray, n_absorbed: int):
    """МНК с кластеризованной ковариацией (CR1, как в Stata/fixest)."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    _, g = np.unique(clusters, return_inverse=True)
    G = g.max() + 1
    scores = np.zeros((G, X.shape[1]))
    np.add.at(scores, g, X * u[:, None])
    meat = scores.T @ scores
    n, k = X.shape
    # при кластеризации по уровню выше FE клиента поправка на поглощённые FE не нужна
    corr = G / (G - 1) * (n - 1) / (n - k)
    V = corr * XtX_inv @ meat @ XtX_inv
    return beta, V, G


def did(df: pd.DataFrame, metric: str, cluster: str = "region_id") -> dict:
    Z = _demean_twfe(df, [metric, "treated_post"])
    y, x = Z[:, 0], Z[:, 1:2]
    beta, V, G = _ols_cluster(y, x, df[cluster].to_numpy(), n_absorbed=0)
    b, se = float(beta[0]), float(np.sqrt(V[0, 0]))
    tcrit = stats.t.ppf(0.975, G - 1)
    p = float(2 * stats.t.sf(abs(b / se), G - 1)) if se > 0 else float("nan")
    treated_post_mean = float(df.loc[df.treated_post == 1, metric].mean())
    cf = treated_post_mean - b
    return {
        "metric": metric,
        "itt": round(b, 4),
        "se": round(se, 4),
        "ci95": [round(b - tcrit * se, 4), round(b + tcrit * se, 4)],
        "p_value": round(p, 4),
        "clusters": int(G),
        "counterfactual_mean": round(cf, 4),
        "relative_effect": round(b / cf, 4) if cf else None,
    }


def check_parallel_trends(ctx: EvalContext, metric: str = "net_income") -> dict:
    """Event study: коэффициенты pilot × (месяц относительно старта), база — месяц −1.

    H0: до пилота нет дифференциального линейного тренда пилот/контроль. Отвергли (p < 0.05) —
    DiD-оценке доверять нельзя. Совместный тест Вальда по всем до-пилотным коэффициентам — справочно. Плюс placebo: «ложный старт» за 6 месяцев до пилота на до-пилотных данных.
    """
    ctx.require("mart")
    df = ctx.mart
    rel = sorted(df.rel_month.unique())
    ks = [k for k in rel if k != -1]
    D = pd.DataFrame({f"k{k}": ((df.rel_month == k) & (df.is_pilot == 1)).astype(float) for k in ks})
    tmp = pd.concat([df[["client_id", "month", metric]].reset_index(drop=True), D], axis=1)
    Z = _demean_twfe(tmp, [metric] + list(D.columns))
    beta, V, G = _ols_cluster(Z[:, 0], Z[:, 1:], df.region_id.to_numpy(), 0)
    se = np.sqrt(np.diag(V))
    pre_idx = [j for j, k in enumerate(ks) if k < -1]
    b_pre, V_pre = beta[pre_idx], V[np.ix_(pre_idx, pre_idx)]
    q = len(pre_idx)
    wald = float(b_pre @ np.linalg.pinv(V_pre) @ b_pre) / q
    p_joint = float(stats.f.sf(wald, q, G - 1))

    # Основной критерий — линейный дифференциальный тренд до пилота (1 ограничение).
    # Совместный тест 11 коэффициентов при 40 кластерах переотвергает H0, поэтому он справочный.
    pre_df = df[df.post == 0].reset_index(drop=True)
    pre_df = pre_df.assign(pilot_trend=(pre_df.is_pilot * pre_df.rel_month).astype(float))
    Zt = _demean_twfe(pre_df, [metric, "pilot_trend"])
    bt, Vt, Gt = _ols_cluster(Zt[:, 0], Zt[:, 1:2], pre_df.region_id.to_numpy(), 0)
    slope, slope_se = float(bt[0]), float(np.sqrt(Vt[0, 0]))
    p_slope = float(2 * stats.t.sf(abs(slope / slope_se), Gt - 1))

    pre = df[df.post == 0].copy()
    fake_start = sorted(pre.rel_month.unique())[-6]
    pre["treated_post"] = ((pre.rel_month >= fake_start) & (pre.is_pilot == 1)).astype(int)
    placebo = did(pre, metric)

    ok = p_slope >= 0.05 and placebo["p_value"] >= 0.05
    res = {
        "metric": metric,
        "event_study": [
            {"rel_month": int(k), "coef": round(float(beta[j]), 3), "se": round(float(se[j]), 3)}
            for j, k in enumerate(ks)
        ],
        "pretrend_slope_per_month": round(slope, 3),
        "pretrend_slope_p_value": round(p_slope, 4),
        "joint_wald_F": round(wald, 3),
        "joint_wald_p_value": round(p_joint, 4),
        "placebo_itt": placebo["itt"],
        "placebo_p_value": placebo["p_value"],
        "parallel_trends_ok": bool(ok),
        "verdict": (
            "Тренды до пилота параллельны, placebo-эффект не значим — DiD применим."
            if ok else
            "Нарушена параллельность трендов или значим placebo — DiD-оценка смещена. "
            "Нужны: синтетический контроль, подбор контрольных регионов или модель с трендами."
        ),
    }
    ctx.results.setdefault("pretrends", {})[metric] = res
    return res


def estimate_effect(ctx: EvalContext, metrics: list[str] | None = None, by_segment: bool = False) -> dict:
    """Фактический эффект пилота (ITT на клиента пилотного региона в месяц) + ATT на подключившего."""
    ctx.require("mart")
    if ctx.results["mart"]["data_quality"]["status"] == "error":
        raise ValueError("В витрине критичные ошибки качества данных — оценка эффекта заблокирована.")
    df = ctx.mart
    metrics = metrics or ["net_income", "interchange_income", "nii_income", "net_cashback_cost", "spend", "avg_balance"]
    bad = [m for m in metrics if m not in METRICS]
    if bad:
        raise ValueError(f"Неизвестные метрики: {bad}. Доступны: {METRICS}")
    n_pilot = df.loc[df.is_pilot == 1, "region_id"].nunique()
    n_ctrl = df.loc[df.is_pilot == 0, "region_id"].nunique()
    warnings = []
    if min(n_pilot, n_ctrl) < MIN_CLUSTERS_PER_ARM:
        warnings.append(f"Мало регионов в группе ({n_pilot}/{n_ctrl}): кластерные ошибки ненадёжны, нужен wild bootstrap.")
    if "net_income" not in ctx.results.get("pretrends", {}):
        warnings.append("Проверка параллельности трендов для net_income не выполнялась.")
    elif not ctx.results["pretrends"]["net_income"]["parallel_trends_ok"]:
        warnings.append("Проверка параллельности трендов НЕ пройдена — оценка может быть смещена.")

    take_up = float(df.loc[(df.is_pilot == 1) & (df.post == 1), "opted_in"].mean())
    effects = {m: did(df, m) for m in metrics}
    main = effects.get("net_income")
    res = {
        "estimand": "ITT: эффект на клиента пилотного региона в месяц (подключил он опцию или нет)",
        "take_up": round(take_up, 4),
        "effects": effects,
        "warnings": warnings,
    }
    if main:
        # Wald/IV: в контроле подключений нет, поэтому ATT = ITT / take_up
        res["att_per_adopter_month"] = {
            "net_income": round(main["itt"] / take_up, 2) if take_up else None,
            "ci95": [round(x / take_up, 2) for x in main["ci95"]] if take_up else None,
        }
    if main:
        pp = df[(df.is_pilot == 1) & (df.post == 1)]
        naive = float(pp.loc[pp.opted_in == 1, "net_income"].mean() - pp.loc[pp.opted_in == 0, "net_income"].mean())
        res["naive_adopters_vs_rest"] = {
            "net_income": round(naive, 2),
            "comment": "Наивное сравнение подключивших с неподключившими: смещено самоотбором "
                       "(подключаются те, кто и так больше тратит). Для решений не использовать.",
        }
    if by_segment:
        res["by_segment"] = {
            s: did(g, "net_income") | {"clients": int(g.client_id.nunique())}
            for s, g in df.groupby("segment")
        }
    ctx.results["effect"] = res
    return res
