"""Итоговый отчёт: таблицы строятся только из результатов инструментов, текст выводов пишет агент.

Защита от «выдуманных» чисел: каждое число в тексте агента сверяется с числами из
результатов расчётов. Если число не найдено, отчёт не собирается, а агент получает
список подозрительных чисел и должен исправить текст.
"""
from __future__ import annotations

import re
from datetime import date

from fineval.context import EvalContext

RECOMMENDATIONS = (
    "масштабировать",
    "масштабировать с условиями",
    "доработать и повторить пилот",
    "остановить",
)
REQUIRED = ("initiative", "mart", "baseline", "plan_unit", "effect", "invest_plan", "invest_fact", "sensitivity_fact")


def decision_rule(ctx: EvalContext) -> dict:
    """Формальное правило, с которым агент сверяет свою рекомендацию."""
    pt = ctx.results.get("pretrends", {}).get("net_income", {})
    mc = ctx.results["sensitivity_fact"]["monte_carlo"]
    eff = ctx.results["effect"]["effects"]["net_income"]
    if not pt.get("parallel_trends_ok", False):
        rule = "доработать и повторить пилот"
        why = "дизайн оценки не прошёл проверку параллельности трендов"
    elif mc["p_npv_positive"] >= 0.8:
        rule = "масштабировать"
        why = "вероятность NPV>0 не ниже 80%"
    elif eff["p_value"] >= 0.05:
        rule = "доработать и повторить пилот"
        why = "эффект статистически не отличим от нуля"
    elif mc["p_npv_positive"] >= 0.5:
        rule = "масштабировать с условиями"
        why = "эффект статистически значим, но вероятность NPV>0 ниже 80%"
    elif eff["ci95"][1] > ctx.results["invest_fact"]["breakeven_unit_effect"]:
        rule = "доработать и повторить пилот"
        why = ("эффект значим, но вероятность NPV>0 ниже 50%; точка безубыточности внутри доверительного "
               "интервала — экономику можно спасти доработкой условий")
    else:
        rule = "остановить"
        why = "эффект достоверно ниже точки безубыточности"
    return {"recommendation": rule, "reason": why}


# ---------- проверка чисел в тексте ----------

_NUM = re.compile(r"(?<![\w.])[-−]?\d[\d\s ]*(?:[.,]\d+)?")


def _collect_numbers(obj, acc: set[float]) -> set[float]:
    if isinstance(obj, bool):
        return acc
    if isinstance(obj, (int, float)):
        acc.add(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, acc)
    return acc


def _parse(tok: str) -> float | None:
    t = tok.replace("−", "-").replace(" ", "").replace(" ", "").replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def unverified_numbers(text: str, results: dict) -> list[str]:
    """Число принимается, если совпадает с результатом с точностью до округления до показанных знаков."""
    known = _collect_numbers(results, set())
    # производные представления: доли → проценты, рубли → млн/млрд, млн → млрд
    variants = set()
    for k in known:
        variants.update({k, abs(k), k * 100, abs(k) * 100, k / 1e6, abs(k) / 1e6, k / 1e9, abs(k) / 1e9,
                         k / 1e3, abs(k) / 1e3})
    bad = []
    for m in _NUM.finditer(text):
        tok = m.group().strip()
        v = _parse(tok)
        if v is None:
            continue
        av = abs(v)
        if av <= 12 and float(av).is_integer():  # порядковые: «3 года», «5 лет», «12 месяцев»
            continue
        if 2000 <= av <= 2100 or 200001 <= av <= 210012:  # годы и месяцы YYYYMM
            continue
        t = tok.replace("−", "-").replace("\u00a0", "").replace(" ", "").replace(",", ".")
        decimals = len(t.split(".")[1]) if "." in t else 0
        tol = 0.5 * 10 ** (-decimals) + 1e-9
        if not any(abs(av - abs(x)) <= tol for x in variants):
            bad.append(tok)
    return bad


# ---------- сборка отчёта ----------

def build_report(ctx: EvalContext, recommendation: str, summary: str, risks_and_conditions: str) -> dict:
    ctx.require(*REQUIRED)
    if recommendation not in RECOMMENDATIONS:
        raise ValueError(f"recommendation должна быть одной из: {RECOMMENDATIONS}")
    bad = unverified_numbers(summary + "\n" + risks_and_conditions, ctx.results)
    if bad:
        return {
            "status": "rejected",
            "reason": "В тексте есть числа, которых нет в результатах расчётов. Уберите их или возьмите "
                      "значения из результатов инструментов (допустимы округления, проценты, млн/млрд).",
            "unverified_numbers": bad,
        }
    rule = decision_rule(ctx)
    md = _render(ctx, recommendation, summary, risks_and_conditions, rule)
    path = ctx.out_dir / "report.md"
    path.write_text(md)
    from fineval.tools.report_html import render_html
    html_path = ctx.out_dir / "report.html"
    html_path.write_text(render_html(ctx, recommendation, summary, risks_and_conditions, rule))
    ctx.results["report"] = {"path": str(path), "html": str(html_path), "recommendation": recommendation,
                             "decision_rule": rule}
    return {"status": "ok", "path": str(path), "html": str(html_path), "recommendation": recommendation,
            "matches_decision_rule": recommendation == rule["recommendation"], "decision_rule": rule}


def _fmt(x, d=1):
    if x is None:
        return "—"
    return f"{x:,.{d}f}".replace(",", " ")


def _pct(x, d=1):
    return "—" if x is None else f"{x * 100:.{d}f}%"


def _render(ctx: EvalContext, rec: str, summary: str, risks: str, rule: dict) -> str:
    R = ctx.results
    ini, mart, base, plan = R["initiative"], R["mart"], R["baseline"], R["plan_unit"]
    eff, ip, iff, sf = R["effect"], R["invest_plan"], R["invest_fact"], R["sensitivity_fact"]
    sp = R.get("sensitivity_plan")
    pt = R.get("pretrends", {}).get("net_income")
    e = eff["effects"]
    L = []
    a = L.append
    a(f"# Оценка инициативы: {ini['name']}\n")
    a(f"*{ini['owner']} · отчёт сформирован {date.today():%d.%m.%Y} · агент финансовой оценки*\n")
    a(f"## Рекомендация: **{rec}**\n")
    if rec != rule["recommendation"]:
        a(f"> Формальное правило даёт «{rule['recommendation']}» ({rule['reason']}). "
          "Агент отступил от правила — обоснование ниже.\n")
    else:
        a(f"> Совпадает с формальным правилом: {rule['reason']}.\n")
    a(summary.strip() + "\n")

    a("## Ключевые показатели: план и факт\n")
    a("| Показатель | План (бизнес-кейс) | Факт (пилот) |")
    a("|---|---:|---:|")
    a(f"| Подключили опцию | {_pct(plan['assumptions']['take_up'])} | {_pct(eff['take_up'])} |")
    a(f"| Эффект на подключившего, ₽/мес | {_fmt(plan['per_adopter_month']['total'])} | "
      f"{_fmt(eff['att_per_adopter_month']['net_income'])} |")
    a(f"| Эффект на клиента (ITT), ₽/мес | {_fmt(plan['itt_per_client_month'])} | "
      f"{_fmt(e['net_income']['itt'])} (95% ДИ {_fmt(e['net_income']['ci95'][0])}…{_fmt(e['net_income']['ci95'][1])}) |")
    a(f"| NPV при ставке {_pct(ip['discount_rate'], 0)}, млн ₽ | {_fmt(ip['npv_mln'])} | {_fmt(iff['npv_mln'])} |")
    a(f"| IRR | {_pct(ip['irr'])} | {_pct(iff['irr'])} |")
    a(f"| Дисконтированная окупаемость, лет | {_fmt(ip['discounted_payback_years'], 2)} | {_fmt(iff['discounted_payback_years'], 2)} |")
    a(f"| Индекс доходности | {_fmt(ip['profitability_index'], 2)} | {_fmt(iff['profitability_index'], 2)} |")
    if sp:
        a(f"| Вероятность NPV > 0 (Monte Carlo) | {_pct(sp['monte_carlo']['p_npv_positive'], 0)} | "
          f"{_pct(sf['monte_carlo']['p_npv_positive'], 0)} |")
    a(f"\nТочка безубыточности: **{_fmt(iff['breakeven_unit_effect'], 2)} ₽** на клиента в месяц.\n")

    a("## Фактический эффект пилота\n")
    a(f"Метод: разность разностей с фиксированными эффектами клиента и месяца, ошибки кластеризованы "
      f"по региону ({e['net_income']['clusters']} кластеров). Пилот: {mart['regions_pilot']} регионов, "
      f"контроль: {mart['regions_control']}; {mart['clients']} клиентов, {mart['pre_months']} мес. до и "
      f"{mart['post_months']} мес. после старта.\n")
    a("| Метрика (на клиента в месяц) | Эффект ITT | 95% ДИ | p-value | К контрфакту |")
    a("|---|---:|---:|---:|---:|")
    names = {"net_income": "Чистый доход, ₽", "interchange_income": "Interchange, ₽", "nii_income": "Доход на остатки, ₽",
             "net_cashback_cost": "Чистые расходы на кэшбэк, ₽", "spend": "Оборот, ₽", "avg_balance": "Остаток, ₽",
             "is_active": "Доля активных"}
    for k, v in e.items():
        a(f"| {names.get(k, k)} | {_fmt(v['itt'], 2)} | {_fmt(v['ci95'][0], 2)}…{_fmt(v['ci95'][1], 2)} | "
          f"{v['p_value']:.3f} | {_pct(v['relative_effect'])} |")
    if "naive_adopters_vs_rest" in eff:
        a(f"\nДля сравнения: наивная разница «подключившие против остальных» — "
          f"{_fmt(eff['naive_adopters_vs_rest']['net_income'], 2)} ₽/мес. Она завышает эффект в разы из-за самоотбора, "
          f"поэтому в расчётах не используется.\n")
    if "by_segment" in eff:
        a("\n**По сегментам** (чистый доход, ITT ₽/мес):\n")
        a("| Сегмент | Клиентов | Эффект | 95% ДИ | p-value |")
        a("|---|---:|---:|---:|---:|")
        for s, v in eff["by_segment"].items():
            a(f"| {s} | {v['clients']} | {_fmt(v['itt'], 2)} | {_fmt(v['ci95'][0], 2)}…{_fmt(v['ci95'][1], 2)} | {v['p_value']:.3f} |")
    if pt:
        a(f"\n**Проверка дизайна.** Дифференциальный тренд до пилота: {pt['pretrend_slope_per_month']} ₽/мес "
          f"(p = {pt['pretrend_slope_p_value']}); совместный тест Вальда по event study: F = {pt['joint_wald_F']} "
          f"(p = {pt['joint_wald_p_value']}); placebo-эффект {pt['placebo_itt']} ₽ (p = {pt['placebo_p_value']}). "
          f"{pt['verdict']}\n")
    dq = mart["data_quality"]
    a(f"**Качество данных:** {dq['status']}" + (f"; предупреждения: {'; '.join(dq['warnings'])}" if dq["warnings"] else "") + "\n")
    for w in eff["warnings"]:
        a(f"> ⚠ {w}\n")

    a("## Денежные потоки масштабирования (факт), млн ₽\n")
    cf = iff["cashflows_mln"]
    a("| Год | " + " | ".join(str(y) for y in range(len(cf["fcf"]))) + " |")
    a("|---|" + "---:|" * len(cf["fcf"]))
    for k, n in [("margin", "Инкрементальная маржа"), ("opex", "OPEX"), ("marketing", "Маркетинг"),
                 ("ebitda", "EBITDA"), ("tax", "Налог"), ("fcf", "Свободный денежный поток")]:
        a(f"| {n} | " + " | ".join(_fmt(x) for x in cf[k]) + " |")

    a("\n## Чувствительность и риск (факт)\n")
    a("| Драйвер | NPV при −20% | NPV при +20% | Размах |")
    a("|---|---:|---:|---:|")
    for t in sf["tornado"]:
        a(f"| {t['driver']} | {_fmt(t['npv_at_-20%_mln'])} | {_fmt(t['npv_at_+20%_mln'])} | {_fmt(t['swing_mln'])} |")
    mc = sf["monte_carlo"]
    a(f"\nMonte Carlo ({mc['n_sims']} сценариев; эффект ~ {mc['effect_distribution']}, CAPEX/OPEX/охват/затухание — "
      f"треугольные распределения из паспорта): P(NPV>0) = **{_pct(mc['p_npv_positive'], 0)}**, "
      f"медиана {_fmt(mc['npv_p50_mln'])} млн ₽, 90% интервал {_fmt(mc['npv_p5_mln'])}…{_fmt(mc['npv_p95_mln'])} млн ₽, "
      f"средний NPV в худших 5% сценариев {_fmt(mc['expected_shortfall_5pct_mln'])} млн ₽.\n")

    a("## Риски и условия\n")
    a(risks.strip() + "\n")

    a("## Допущения\n")
    a(f"- Ставка дисконтирования {_pct(ini['finance']['discount_rate'], 0)}, налог на прибыль {_pct(ini['finance']['tax_rate'], 0)}; "
      "налоговый щит по убыткам зачитывается в прибыль Группы.")
    a(f"- Охват по годам: {', '.join(_fmt(x, 0) for x in ini['rollout']['eligible_clients'])} клиентов; "
      f"затухание эффекта {_pct(ini['rollout']['effect_decay_per_year'], 0)} в год.")
    a(f"- CAPEX {_fmt(ini['costs']['capex'] / 1e6)} млн ₽, амортизация {ini['costs']['depreciation_years']} лет; "
      f"OPEX {_fmt(ini['costs']['opex_per_year'] / 1e6)} млн ₽ в год.")
    a("- Эффект пилота переносится на сеть как ITT: при масштабировании доля подключений такая же, как в пилоте.")
    a(f"- База плана (до пилота, на клиента в месяц): оборот {_fmt(base['avg_spend'], 0)} ₽, "
      f"остаток {_fmt(base['avg_balance'], 0)} ₽, доля категорийных трат {_pct(base['category_share'])}.")
    return "\n".join(L) + "\n"
