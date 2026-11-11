"""Сбор данных: паспорт инициативы, витрина «клиент × месяц», проверки качества, базовые метрики."""
from __future__ import annotations

from pathlib import Path

import duckdb

from fineval.context import EvalContext

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"
SOURCES = ["regions", "clients", "card_activity", "balances", "cashback"]


def render_sql(name: str, params: dict) -> str:
    """Подставляет параметры :name в SQL. Значения — только числа, поэтому инъекции нет."""
    sql = (SQL_DIR / name).read_text()
    for k in sorted(params, key=len, reverse=True):
        v = params[k]
        if not isinstance(v, (int, float)):
            raise TypeError(f"SQL-параметр {k} должен быть числом")
        sql = sql.replace(f":{k}", repr(v))
    return sql


def mart_params(initiative: dict) -> dict:
    eco = initiative["economics"]
    return {
        "pilot_start": int(initiative["pilot"]["start_month"]),
        "interchange": float(eco["interchange_rate"]),
        "ftp_spread": float(eco["ftp_spread"]),
    }


def load_initiative(ctx: EvalContext) -> dict:
    ini = ctx.initiative
    res = {
        "id": ini["id"],
        "name": ini["name"],
        "owner": ini["owner"],
        "description": " ".join(ini["description"].split()),
        "pilot_start_month": ini["pilot"]["start_month"],
        "plan_assumptions": ini["plan_assumptions"],
        "rollout": ini["rollout"],
        "costs": ini["costs"],
        "finance": ini["finance"],
    }
    ctx.results["initiative"] = res
    return res


def build_data_mart(ctx: EvalContext) -> dict:
    con = duckdb.connect()
    for s in SOURCES:
        path = ctx.data_dir / f"{s}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Нет источника {path}. Сгенерируйте данные: python -m fineval.data_gen")
        con.execute(f"CREATE VIEW {s} AS SELECT * FROM read_parquet('{path}')")
    df = con.execute(render_sql("mart_client_month.sql", mart_params(ctx.initiative))).df()
    dq = _dq_checks(con, df)
    con.close()
    ctx.mart = df

    pilot = df.groupby("is_pilot").client_id.nunique()
    res = {
        "rows": int(len(df)),
        "clients": int(df.client_id.nunique()),
        "clients_pilot_regions": int(pilot.get(1, 0)),
        "clients_control_regions": int(pilot.get(0, 0)),
        "regions_pilot": int(df.loc[df.is_pilot == 1, "region_id"].nunique()),
        "regions_control": int(df.loc[df.is_pilot == 0, "region_id"].nunique()),
        "months": [int(df.month.min()), int(df.month.max())],
        "pre_months": int(df.loc[df.post == 0, "month"].nunique()),
        "post_months": int(df.loc[df.post == 1, "month"].nunique()),
        "data_quality": dq,
    }
    ctx.results["mart"] = res
    return res


def _dq_checks(con, df) -> dict:
    """Проверки качества: ошибка — расчёт эффекта блокируется, предупреждение — попадает в отчёт."""
    errors, warnings = [], []
    dup = con.execute(
        "SELECT COUNT(*) FROM (SELECT client_id, month FROM card_activity GROUP BY 1,2 HAVING COUNT(*)>1)"
    ).fetchone()[0]
    if dup:
        errors.append(f"Дубли client_id×month в card_activity: {dup}")
    orphans = con.execute(
        "SELECT COUNT(*) FROM card_activity a LEFT JOIN clients c USING (client_id) WHERE c.client_id IS NULL"
    ).fetchone()[0]
    if orphans:
        errors.append(f"Строки активности без клиента в справочнике: {orphans}")
    for col in ["spend", "category_spend", "avg_balance", "cashback_cost"]:
        n_neg = int((df[col] < 0).sum())
        if n_neg:
            errors.append(f"Отрицательные значения в {col}: {n_neg}")
        n_null = int(df[col].isna().sum())
        if n_null:
            errors.append(f"Пропуски в {col}: {n_null}")
    bad_share = int((df.category_spend > df.spend + 0.01).sum())
    if bad_share:
        errors.append(f"Категорийные траты больше общего оборота: {bad_share} строк")
    leak = int(((df.is_pilot == 0) & (df.opted_in == 1)).sum() + ((df.post == 0) & (df.opted_in == 1)).sum())
    if leak:
        warnings.append(f"Подключения вне пилота (контаминация контроля / до старта): {leak}")
    per_client = df.groupby("client_id").month.count()
    if per_client.nunique() > 1:
        warnings.append(
            f"Несбалансированная панель: у {int((per_client < per_client.max()).sum())} клиентов не все месяцы"
        )
    # выбросы: доля клиент-месяцев с оборотом выше 99.9 перцентиля * 5
    thr = df.spend.quantile(0.999) * 5
    n_out = int((df.spend > thr).sum())
    if n_out:
        warnings.append(f"Экстремальные обороты (> 5×P99.9): {n_out}")
    return {"status": "error" if errors else "ok", "errors": errors, "warnings": warnings}


def baseline_metrics(ctx: EvalContext) -> dict:
    """Средние за до-пилотный период на клиента пилотных регионов в месяц — база для плана."""
    ctx.require("mart")
    df = ctx.mart
    pre = df[(df.post == 0) & (df.is_pilot == 1)]
    eco = ctx.initiative["economics"]
    spend = float(pre.spend.mean())
    res = {
        "period": "до пилота, клиенты пилотных регионов, в среднем на клиента в месяц",
        "avg_spend": round(spend, 2),
        "avg_category_spend": round(float(pre.category_spend.mean()), 2),
        "category_share": round(float(pre.category_spend.sum() / max(pre.spend.sum(), 1)), 4),
        "avg_balance": round(float(pre.avg_balance.mean()), 2),
        "active_share": round(float(pre.is_active.mean()), 4),
        "avg_net_income": round(float(pre.net_income.mean()), 2),
        "rates": {k: eco[k] for k in ("interchange_rate", "ftp_spread", "base_cashback_rate")},
    }
    ctx.results["baseline"] = res
    return res


def plan_unit_economics(ctx: EvalContext, overrides: dict | None = None) -> dict:
    """Ex-ante эффект на клиента в месяц (ITT) из допущений паспорта и базовых метрик.

    ITT = take_up × [ΔInterchange + ΔNII − ΔЧистый кэшбэк]; эффект на удержание не учитываем
    (консервативно).
    """
    ctx.require("baseline")
    b = ctx.results["baseline"]
    a = {**ctx.initiative["plan_assumptions"], **(overrides or {})}
    r = b["rates"]
    S, B, c = b["avg_spend"], b["avg_balance"], b["category_share"]
    cat_after = S * (1 + a["spend_uplift"]) * (c + a["category_share_shift"])
    d_interchange = S * a["spend_uplift"] * r["interchange_rate"]
    d_nii = B * a["balance_uplift"] * r["ftp_spread"] / 12
    d_cashback = r["base_cashback_rate"] * (cat_after - S * c) + a["extra_cashback_rate"] * cat_after * (
        1 - a["partner_share"]
    )
    per_adopter = d_interchange + d_nii - d_cashback
    res = {
        "assumptions": a,
        "per_adopter_month": {
            "interchange": round(d_interchange, 2),
            "nii": round(d_nii, 2),
            "net_cashback_cost": round(-d_cashback, 2),
            "total": round(per_adopter, 2),
        },
        "itt_per_client_month": round(a["take_up"] * per_adopter, 2),
    }
    ctx.results["plan_unit"] = res
    return res


def summarize_mart(df):
    """Средние по группам до/после — для таблицы в отчёте и быстрой «ручной» DiD-проверки."""
    g = df.groupby(["is_pilot", "post"])[["net_income", "spend", "avg_balance", "opted_in"]].mean()
    return {f"{'pilot' if p else 'control'}_{'post' if q else 'pre'}": {k: round(float(v), 2) for k, v in row.items()}
            for (p, q), row in g.iterrows()}


def simple_did(df, metric: str = "net_income") -> float:
    m = df.groupby(["is_pilot", "post"])[metric].mean()
    return float((m[1, 1] - m[1, 0]) - (m[0, 1] - m[0, 0]))


__all__ = [
    "load_initiative", "build_data_mart", "baseline_metrics", "plan_unit_economics",
    "summarize_mart", "simple_did", "render_sql", "mart_params",
]
