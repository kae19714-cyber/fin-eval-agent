"""Финансовая оценка: денежные потоки масштабирования, NPV/IRR/окупаемость, чувствительность, Monte Carlo.

Логика денежного потока по году y (1..H):
  инкрементальная маржа = клиенты_y × 12 × эффект_на_клиента_в_месяц × (1 − затухание)^(y−1)
  EBITDA = маржа − opex − маркетинг
  налог  = (EBITDA − амортизация) × ставка   (отрицательный налог = налоговый щит: у банка есть
                                             прибыль, против которой убыток инициативы зачитывается)
  FCF    = EBITDA − налог;  год 0: −capex
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy import optimize

from fineval.context import EvalContext

SCENARIOS = ("plan", "fact")


# ---------- чистые функции (покрыты unit-тестами) ----------

def npv(rate: float, cfs: list[float] | np.ndarray) -> float:
    cfs = np.asarray(cfs, dtype=float)
    return float(np.sum(cfs / (1 + rate) ** np.arange(len(cfs))))


def irr(cfs: list[float] | np.ndarray) -> tuple[float | None, list[str]]:
    """IRR через поиск корня NPV(r)=0. Предупреждает о неединственности и отсутствии корня."""
    cfs = np.asarray(cfs, dtype=float)
    warnings = []
    signs = np.sign(cfs[cfs != 0])
    changes = int(np.sum(signs[1:] != signs[:-1]))
    if changes == 0:
        return None, ["Все потоки одного знака — IRR не существует."]
    if changes > 1:
        warnings.append(f"Знак потоков меняется {changes} раза — IRR может быть не единственным; опирайтесь на NPV.")
    grid = np.concatenate([np.linspace(-0.95, 1, 400), np.linspace(1, 20, 200)])
    vals = np.array([npv(r, cfs) for r in grid])
    roots = []
    for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if fa == 0:
            roots.append(a)
        elif fa * fb < 0:
            roots.append(optimize.brentq(lambda r: npv(r, cfs), a, b))
    if not roots:
        return None, warnings + ["Корень NPV(r)=0 не найден в диапазоне −95%…2000%."]
    if len(roots) > 1:
        warnings.append(f"Найдено несколько IRR: {[round(r, 4) for r in roots]}")
    return float(roots[0]), warnings


def discounted_payback(rate: float, cfs: list[float] | np.ndarray) -> float | None:
    """Дисконтированный срок окупаемости в годах (линейная интерполяция внутри года)."""
    disc = np.asarray(cfs, dtype=float) / (1 + rate) ** np.arange(len(cfs))
    cum = np.cumsum(disc)
    for t in range(1, len(cum)):
        if cum[t - 1] < 0 <= cum[t]:
            return float(t - 1 + (-cum[t - 1]) / disc[t])
    return None if cum[-1] < 0 else 0.0


def cashflows(unit_effect: float, p: dict) -> dict:
    """Годовые денежные потоки по параметрам p (см. _params)."""
    H = p["horizon_years"]
    years = np.arange(1, H + 1)
    clients = np.asarray(p["eligible_clients"][:H], dtype=float) * p["clients_mult"]
    decay = (1 - p["effect_decay_per_year"]) ** (years - 1)
    margin = clients * 12 * unit_effect * decay
    opex = np.full(H, p["opex_per_year"]) * p["opex_mult"]
    mkt = np.asarray(p["marketing_per_year"][:H], dtype=float)
    capex = p["capex"] * p["capex_mult"]
    dep = np.where(years <= p["depreciation_years"], capex / p["depreciation_years"], 0.0)
    ebitda = margin - opex - mkt
    tax = (ebitda - dep) * p["tax_rate"]
    fcf = np.concatenate([[-capex], ebitda - tax])
    return {"years": list(range(0, H + 1)), "margin": margin, "opex": opex, "marketing": mkt,
            "ebitda": ebitda, "depreciation": dep, "tax": tax, "fcf": fcf}


def _params(ini: dict, overrides: dict | None = None) -> dict:
    p = {
        **ini["rollout"],
        **ini["costs"],
        **ini["finance"],
        "clients_mult": 1.0, "opex_mult": 1.0, "capex_mult": 1.0,
    }
    for k, v in (overrides or {}).items():
        if k not in p:
            raise ValueError(f"Неизвестный параметр «{k}». Допустимые: {sorted(p)}")
        p[k] = v
    return p


def _unit_effect(ctx: EvalContext, scenario: str) -> tuple[float, float]:
    """Эффект на клиента в месяц и его стандартная ошибка для сценария."""
    if scenario == "plan":
        ctx.require("plan_unit")
        u = ctx.results["plan_unit"]["itt_per_client_month"]
        return u, abs(u) * ctx.initiative["uncertainty"]["plan_effect_cv"]
    ctx.require("effect")
    e = ctx.results["effect"]["effects"]["net_income"]
    return e["itt"], e["se"]


# ---------- инструменты агента ----------

def evaluate_investment(ctx: EvalContext, scenario: str, overrides: dict | None = None,
                        unit_effect_override: float | None = None) -> dict:
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario должен быть одним из {SCENARIOS}")
    p = _params(ctx.initiative, overrides)
    unit, _ = _unit_effect(ctx, scenario)
    if unit_effect_override is not None:
        unit = unit_effect_override
    cf = cashflows(unit, p)
    r = p["discount_rate"]
    v_npv = npv(r, cf["fcf"])
    v_irr, irr_warn = irr(cf["fcf"])
    pv_in = npv(r, np.concatenate([[0], cf["fcf"][1:]]))
    res = {
        "scenario": scenario,
        "unit_effect_per_client_month": round(unit, 2),
        "overrides": overrides or {},
        "discount_rate": r,
        "cashflows_mln": {
            k: [round(float(x) / 1e6, 1) for x in (cf[k] if k == "fcf" else np.concatenate([[0], cf[k]]))]
            for k in ("margin", "opex", "marketing", "ebitda", "tax", "fcf")
        },
        "npv_mln": round(v_npv / 1e6, 1),
        "irr": round(v_irr, 4) if v_irr is not None else None,
        "discounted_payback_years": (lambda x: round(x, 2) if x is not None else None)(discounted_payback(r, cf["fcf"])),
        "profitability_index": round(pv_in / (p["capex"] * p["capex_mult"]), 2),
        "breakeven_unit_effect": round(_breakeven(p), 2),
        "warnings": irr_warn,
    }
    if not overrides and unit_effect_override is None:
        ctx.results[f"invest_{scenario}"] = res
    return res


def _breakeven(p: dict) -> float:
    """Эффект на клиента в месяц, при котором NPV = 0 (NPV линеен по эффекту)."""
    r = p["discount_rate"]
    n0 = npv(r, cashflows(0.0, p)["fcf"])
    n1 = npv(r, cashflows(1.0, p)["fcf"])
    return -n0 / (n1 - n0)


def sensitivity_analysis(ctx: EvalContext, scenario: str, n_sims: int = 5000, seed: int = 7) -> dict:
    """Торнадо (±20% по драйверам) и Monte Carlo с неопределённостью эффекта из статистики пилота."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario должен быть одним из {SCENARIOS}")
    ini = ctx.initiative
    base_p = _params(ini)
    unit, se = _unit_effect(ctx, scenario)
    r0 = base_p["discount_rate"]
    base_npv = npv(r0, cashflows(unit, base_p)["fcf"])

    drivers = {
        "Эффект на клиента": lambda p, s: (p, unit * s),
        "Охват клиентов": lambda p, s: ({**p, "clients_mult": s}, unit),
        "CAPEX": lambda p, s: ({**p, "capex_mult": s}, unit),
        "OPEX": lambda p, s: ({**p, "opex_mult": s}, unit),
        "Ставка дисконтирования": lambda p, s: ({**p, "discount_rate": r0 * s}, unit),
        "Затухание эффекта": lambda p, s: ({**p, "effect_decay_per_year": min(0.99, p["effect_decay_per_year"] * s)}, unit),
    }
    tornado = []
    for name, f in drivers.items():
        lo_p, lo_u = f(deepcopy(base_p), 0.8)
        hi_p, hi_u = f(deepcopy(base_p), 1.2)
        lo = npv(lo_p["discount_rate"], cashflows(lo_u, lo_p)["fcf"])
        hi = npv(hi_p["discount_rate"], cashflows(hi_u, hi_p)["fcf"])
        tornado.append({"driver": name, "npv_at_-20%_mln": round(lo / 1e6, 1), "npv_at_+20%_mln": round(hi / 1e6, 1),
                        "swing_mln": round(abs(hi - lo) / 1e6, 1)})
    tornado.sort(key=lambda x: -x["swing_mln"])

    rng = np.random.default_rng(seed)
    unc = ini["uncertainty"]
    tri = lambda k: rng.triangular(*unc[k], n_sims)  # noqa: E731
    u_s = rng.normal(unit, se, n_sims)
    capex_m, opex_m, cl_m = tri("capex"), tri("opex"), tri("eligible_clients")
    decay = rng.triangular(*unc["effect_decay_per_year"], n_sims)
    npvs = np.array([
        npv(r0, cashflows(u_s[i], {**base_p, "capex_mult": capex_m[i], "opex_mult": opex_m[i],
                                   "clients_mult": cl_m[i], "effect_decay_per_year": decay[i]})["fcf"])
        for i in range(n_sims)
    ])
    res = {
        "scenario": scenario,
        "base_npv_mln": round(base_npv / 1e6, 1),
        "tornado": tornado,
        "monte_carlo": {
            "n_sims": n_sims,
            "effect_distribution": f"Normal(mean={unit:.2f}, sd={se:.2f}) ₽ на клиента в месяц",
            "p_npv_positive": round(float(np.mean(npvs > 0)), 3),
            "npv_p5_mln": round(float(np.percentile(npvs, 5)) / 1e6, 1),
            "npv_p50_mln": round(float(np.percentile(npvs, 50)) / 1e6, 1),
            "npv_p95_mln": round(float(np.percentile(npvs, 95)) / 1e6, 1),
            "expected_shortfall_5pct_mln": round(float(npvs[npvs <= np.percentile(npvs, 5)].mean()) / 1e6, 1),
        },
    }
    ctx.results[f"sensitivity_{scenario}"] = res
    return res
