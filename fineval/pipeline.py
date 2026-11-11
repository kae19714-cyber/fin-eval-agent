"""Детерминированный прогон тех же инструментов без LLM.

Нужен для: CI и приёмочных тестов, воспроизводимости, работы без доступа к API.
Текст выводов здесь шаблонный; в режиме агента его пишет модель.
"""
from __future__ import annotations

import json

from fineval.context import EvalContext
from fineval.registry import run_tool
from fineval.tools.report import decision_rule

STEPS = [
    ("load_initiative", {}),
    ("build_data_mart", {}),
    ("baseline_metrics", {}),
    ("plan_unit_economics", {}),
    ("check_parallel_trends", {"metric": "net_income"}),
    ("estimate_effect", {"by_segment": True}),
    ("evaluate_investment", {"scenario": "plan"}),
    ("evaluate_investment", {"scenario": "fact"}),
    ("sensitivity_analysis", {"scenario": "plan"}),
    ("sensitivity_analysis", {"scenario": "fact"}),
]


def run_pipeline(ctx: EvalContext, verbose: bool = True) -> dict:
    for name, args in STEPS:
        out, err = run_tool(ctx, name, args)
        if verbose:
            print(f"  {'✗' if err else '✓'} {name}({json.dumps(args, ensure_ascii=False)})")
        if err:
            raise RuntimeError(f"{name}: {out}")

    R = ctx.results
    rule = decision_rule(ctx)
    summary, risks = auto_narrative(ctx)
    out, err = run_tool(ctx, "build_report",
                        {"recommendation": rule["recommendation"], "summary": summary, "risks_and_conditions": risks})
    if err or json.loads(out).get("status") != "ok":
        raise RuntimeError(f"build_report: {out}")
    if verbose:
        print(f"  ✓ build_report → {R['report']['path']}")
    return R["report"]


def _n(x: float, d: int = 1) -> str:
    """Число по-русски: десятичная запятая, неразрывный пробел в тысячах."""
    s = f"{abs(x):,.{d}f}".replace(",", "\u00a0").replace(".", ",")
    return ("−" if x < 0 else "") + s


def auto_narrative(ctx: EvalContext) -> tuple[str, str]:
    """Шаблонные выводы детерминированного режима. Все числа — из результатов инструментов;
    в режиме агента этот текст пишет модель."""
    R = ctx.results
    eff, plan = R["effect"], R["plan_unit"]
    e = eff["effects"]
    ip, iff, sf = R["invest_plan"], R["invest_fact"], R["sensitivity_fact"]
    mc = sf["monte_carlo"]
    be = iff["breakeven_unit_effect"]
    ratio = plan["itt_per_client_month"] / e["net_income"]["itt"] if e["net_income"]["itt"] else None
    gap = "почти вдвое ниже" if ratio and ratio > 1.6 else "ниже" if ratio and ratio > 1.05 else "на уровне"
    p1 = (
        f"Пилот подтвердил эффект, но он {gap} бизнес-кейса: {_n(e['net_income']['itt'], 2)} ₽ чистого дохода "
        f"на клиента в месяц против {_n(plan['itt_per_client_month'], 2)} ₽. Подключились {_n(eff['take_up'] * 100)}% "
        f"клиентов вместо {_n(plan['assumptions']['take_up'] * 100)}%, а эффект на подключившего — "
        f"{_n(eff['att_per_adopter_month']['net_income'], 2)} ₽ вместо {_n(plan['per_adopter_month']['total'], 2)} ₽. "
        f"Рост оборота и остатков почти целиком уходит на кэшбэк: +{_n(e['interchange_income']['itt'], 2)} ₽ interchange "
        f"и +{_n(e['nii_income']['itt'], 2)} ₽ дохода на остатки против {_n(e['net_cashback_cost']['itt'], 2)} ₽ "
        f"чистых расходов на кэшбэк."
    )
    p2 = (
        f"При масштабировании NPV составит {_n(iff['npv_mln'])} млн ₽ вместо {_n(ip['npv_mln'])} млн ₽ по плану. "
        f"Точка безубыточности {_n(be, 2)} ₽ лежит "
        + ("внутри доверительного интервала эффекта" if e["net_income"]["ci95"][0] < be < e["net_income"]["ci95"][1]
           else "ниже доверительного интервала эффекта" if be <= e["net_income"]["ci95"][0]
           else "выше доверительного интервала эффекта")
        + f", поэтому вероятность NPV > 0 — {_n(mc['p_npv_positive'] * 100, 0)}%."
    )
    seg = eff.get("by_segment", {})
    if seg:
        best = max(seg.items(), key=lambda kv: kv[1]["itt"])
        worst = min((kv for kv in seg.items() if kv[1]["p_value"] < 0.05), key=lambda kv: kv[1]["itt"], default=None)
        if worst and worst[0] != best[0]:
            names = {"mass": "массовом", "affluent": "состоятельном", "premium": "премиальном"}
            p2 += (f" Эффект сосредоточен в {names.get(best[0], best[0])} сегменте: {_n(best[1]['itt'], 2)} ₽ против "
                   f"{_n(worst[1]['itt'], 2)} ₽ в {names.get(worst[0], worst[0])}.")
    risks = "\n".join([
        f"- Экономика на грани: точка безубыточности {_n(be, 2)} ₽, нижняя граница эффекта "
        f"{_n(e['net_income']['ci95'][0], 2)} ₽. В худших 5% сценариев NPV в среднем {_n(mc['expected_shortfall_5pct_mln'])} млн ₽.",
        "- Главный рычаг — расходы на кэшбэк: стоит пересогласовать долю компенсации с партнёрами "
        "и пересчитать NPV как what-if.",
        "- Предложение стоит сфокусировать на сегменте с сильным эффектом и повторить пилот с таргетингом "
        "и заранее заданным критерием успеха.",
        "- Эффект новизны может затухать быстрее заложенного: в повторном пилоте нужен горизонт наблюдения длиннее.",
        "- Выводы детерминированного режима сформированы по шаблону; в режиме агента их пишет модель.",
    ])
    return p1 + "\n\n" + p2, risks
