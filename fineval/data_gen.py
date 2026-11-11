"""Синтетические «сырые» данные пилота.

Сценарий: банк запускает повышенный кэшбэк в выбранных категориях для
держателей дебетовых карт. Часть повышенного кэшбэка компенсируют партнёры.
Пилот идёт в половине регионов с 2026-01, остальные регионы — контроль.

Экономика для банка (на клиента в месяц):
  чистый доход = interchange * оборот + спред * остаток / 12 − (кэшбэк − компенсация партнёров)

В данные заложен известный истинный эффект: факт и контрфакт считаются на
одних и тех же случайных шумах, поэтому «правильный ответ» известен точно
и методику можно проверить автотестом.

Таблицы (parquet):
  regions        region_id, region_name, is_pilot, pilot_start_month
  clients        client_id, region_id, segment, open_month
  card_activity  client_id, month, spend, category_spend, is_active
  balances       client_id, month, avg_balance
  cashback       client_id, month, cashback_cost, partner_compensation, opted_in
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

MONTHS = [y * 100 + m for y in (2025, 2026) for m in range(1, 13)]  # 202501..202612
PILOT_START = 202601

# Экономические ставки банка (их же использует витрина и паспорт инициативы)
INTERCHANGE_RATE = 0.012  # доход от оборота по карте
FTP_SPREAD = 0.05  # годовой трансфертный спред на остатки
BASE_CASHBACK_RATE = 0.010  # базовый кэшбэк на категорийные траты


@dataclass(frozen=True)
class TrueEffect:
    """Истинные параметры, «зашитые» в данные."""

    take_up: float = 0.35  # доля клиентов пилотных регионов, подключивших опцию
    spend_uplift: float = 0.15  # рост оборота у подключивших
    balance_uplift: float = 0.30  # рост остатков (клиент переносит зарплату / «основной банк»)
    category_share_shift: float = 0.05  # сдвиг доли трат в повышенные категории
    extra_cashback_rate: float = 0.010  # доп. кэшбэк на категорийные траты подключивших
    partner_share: float = 0.50  # доля доп. кэшбэка, компенсируемая партнёрами
    churn_reduction: float = 0.25  # снижение вероятности уйти в неактивность
    pre_trend_violation: float = 0.0  # >0 — ломаем параллельность трендов (для тестов)


SEGMENTS = {
    # сегмент: (доля клиентов, медианный оборот ₽/мес, доля категорийных трат)
    "mass": (0.70, 28_000, 0.30),
    "affluent": (0.25, 75_000, 0.35),
    "premium": (0.05, 210_000, 0.40),
}


def generate(
    out_dir: str | Path,
    n_clients: int = 20_000,
    n_regions: int = 40,
    effect: TrueEffect = TrueEffect(),
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- регионы: пилотные отличаются уровнем (для DiD это допустимо), но не трендом
    regions = pd.DataFrame(
        {
            "region_id": np.arange(1, n_regions + 1),
            "region_name": [f"Регион {i:02d}" for i in range(1, n_regions + 1)],
        }
    )
    pilot_ids = rng.choice(regions.region_id, size=n_regions // 2, replace=False)
    regions["is_pilot"] = regions.region_id.isin(pilot_ids).astype(int)
    regions["pilot_start_month"] = pd.array(
        np.where(regions.is_pilot == 1, PILOT_START, 0), dtype="Int64"
    )
    regions.loc[regions.is_pilot == 0, "pilot_start_month"] = pd.NA
    region_level = rng.normal(0, 0.15, n_regions) + 0.10 * regions.is_pilot.to_numpy()
    region_trend = rng.normal(0.004, 0.0005, n_regions)  # общий рост ~0.4%/мес

    # --- клиенты
    seg_names = list(SEGMENTS)
    seg = rng.choice(seg_names, size=n_clients, p=[SEGMENTS[s][0] for s in seg_names])
    region_of = rng.integers(1, n_regions + 1, n_clients)
    clients = pd.DataFrame(
        {
            "client_id": np.arange(1, n_clients + 1),
            "region_id": region_of,
            "segment": seg,
            "open_month": rng.choice([202301, 202307, 202401, 202407], n_clients),
        }
    )
    base_spend = np.array([SEGMENTS[s][1] for s in seg]) * rng.lognormal(0, 0.45, n_clients)
    base_cat_share = np.clip(
        np.array([SEGMENTS[s][2] for s in seg]) + rng.normal(0, 0.05, n_clients), 0.05, 0.9
    )
    balance_ratio = rng.lognormal(np.log(1.2), 0.5, n_clients)  # остаток ≈ 1.2 месячных трат
    is_pilot_client = regions.set_index("region_id").is_pilot.loc[region_of].to_numpy()
    lvl = region_level[region_of - 1]
    trend = region_trend[region_of - 1]

    # подключение опции: только в пилотных регионах, чаще у активно тратящих (самоотбор!)
    p_opt = effect.take_up * np.clip(base_spend / np.median(base_spend), 0.5, 1.6) / 1.05
    opted = (rng.random(n_clients) < np.clip(p_opt, 0, 0.95)) & (is_pilot_client == 1)

    # --- общие случайные шумы панели
    T = len(MONTHS)
    t_idx = np.arange(T)
    post = np.array([m >= PILOT_START for m in MONTHS])
    season = 1 + 0.06 * np.sin(2 * np.pi * (t_idx % 12 - 2) / 12) + np.where(t_idx % 12 == 11, 0.15, 0)
    extra_trend = effect.pre_trend_violation * is_pilot_client  # «пилотные росли быстрее и до пилота»
    log_spend = (
        np.log(base_spend)[:, None]
        + lvl[:, None]
        + (trend + extra_trend)[:, None] * t_idx[None, :]
        + np.log(season)[None, :]
        + rng.normal(0, 0.18, (n_clients, T))
    )
    bal_noise = rng.lognormal(0, 0.25, (n_clients, T))
    u = rng.random((n_clients, T))
    treat = opted[:, None] & post[None, :]

    def simulate(tm: np.ndarray) -> dict:
        """Факт и контрфакт отличаются только маской воздействия tm."""
        spend = np.exp(log_spend) * np.where(tm, 1 + effect.spend_uplift, 1.0)
        cat_share = np.clip(base_cat_share[:, None] + np.where(tm, effect.category_share_shift, 0), 0, 0.95)
        p_churn = np.where(tm, 0.035 * (1 - effect.churn_reduction), 0.035)
        active = np.ones((n_clients, T), dtype=bool)
        for t in range(1, T):
            active[:, t] = np.where(active[:, t - 1], u[:, t] > p_churn[:, t], u[:, t] < 0.30)
        spend = np.where(active, spend, 0.0)
        category_spend = spend * cat_share
        balance = (
            np.exp(log_spend) * balance_ratio[:, None] * bal_noise
            * np.where(tm, 1 + effect.balance_uplift, 1.0)
            * np.where(active, 1.0, 0.3)  # у неактивных остаток «тает»
        )
        extra_cb = category_spend * np.where(tm, effect.extra_cashback_rate, 0.0)
        cashback_cost = category_spend * BASE_CASHBACK_RATE + extra_cb
        partner = extra_cb * effect.partner_share
        net = spend * INTERCHANGE_RATE + balance * FTP_SPREAD / 12 - (cashback_cost - partner)
        return dict(spend=spend, category_spend=category_spend, active=active, balance=balance,
                    cashback_cost=cashback_cost, partner=partner, net=net)

    f = simulate(treat)
    cf = simulate(np.zeros_like(treat))
    pilot_post = (is_pilot_client == 1)[:, None] & post[None, :]
    true_itt = float((f["net"] - cf["net"])[pilot_post].mean())
    true_att = float((f["net"] - cf["net"])[treat].mean())

    cid = np.repeat(clients.client_id.to_numpy(), T)
    month = np.tile(MONTHS, n_clients)
    tables = {
        "regions": regions,
        "clients": clients,
        "card_activity": pd.DataFrame({
            "client_id": cid, "month": month,
            "spend": f["spend"].ravel().round(2),
            "category_spend": f["category_spend"].ravel().round(2),
            "is_active": f["active"].ravel().astype(int),
        }),
        "balances": pd.DataFrame({
            "client_id": cid, "month": month, "avg_balance": f["balance"].ravel().round(2),
        }),
        "cashback": pd.DataFrame({
            "client_id": cid, "month": month,
            "cashback_cost": f["cashback_cost"].ravel().round(2),
            "partner_compensation": f["partner"].ravel().round(2),
            "opted_in": treat.ravel().astype(int),
        }),
    }
    con = duckdb.connect()
    for name, df in tables.items():
        con.register("df_view", df)
        con.execute(f"COPY df_view TO '{out / (name + '.parquet')}' (FORMAT PARQUET)")
        con.unregister("df_view")
    con.close()

    meta = {
        "effect": asdict(effect),
        "seed": seed,
        "n_clients": n_clients,
        "true_itt_net_income_per_client_month": round(true_itt, 2),
        "true_att_net_income_per_optin_month": round(true_att, 2),
        "note": "Истинный эффект = факт − контрфакт на тех же случайных шумах. "
                "ITT — на клиента пилотного региона, ATT — на подключившего.",
    }
    (out / "_truth.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


if __name__ == "__main__":
    import sys

    print(generate(sys.argv[1] if len(sys.argv) > 1 else "data/raw"))
