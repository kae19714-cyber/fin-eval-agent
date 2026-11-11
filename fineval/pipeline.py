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
    e = R["effect"]["effects"]["net_income"]
    summary = (
        f"Фактический эффект пилота — {e['itt']:.2f} ₽ на клиента в месяц против "
        f"{R['plan_unit']['itt_per_client_month']:.2f} ₽ в бизнес-кейсе. NPV масштабирования по факту "
        f"{R['invest_fact']['npv_mln']:.1f} млн ₽ против {R['invest_plan']['npv_mln']:.1f} млн ₽ по плану; "
        f"вероятность NPV > 0 — {R['sensitivity_fact']['monte_carlo']['p_npv_positive'] * 100:.0f}%."
    )
    risks = (
        f"- Точка безубыточности {R['invest_fact']['breakeven_unit_effect']:.2f} ₽ на клиента в месяц.\n"
        "- Шаблонный текст детерминированного режима; содержательные выводы формирует агент."
    )
    out, err = run_tool(ctx, "build_report",
                        {"recommendation": rule["recommendation"], "summary": summary, "risks_and_conditions": risks})
    if err or json.loads(out).get("status") != "ok":
        raise RuntimeError(f"build_report: {out}")
    if verbose:
        print(f"  ✓ build_report → {R['report']['path']}")
    return R["report"]
