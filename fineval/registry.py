"""Реестр инструментов агента: контракт (pydantic-схема), описание для модели, реализация.

Каждый инструмент:
  * принимает только провалидированные входы (ошибка валидации → is_error, модель исправляет вызов);
  * считает детерминированно в коде, а не «в голове» модели;
  * возвращает JSON с результатом и предупреждениями.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fineval.context import EvalContext, PrerequisiteError
from fineval.tools import data, effect, finance, report


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoInput(_In):
    pass


class PlanInput(_In):
    overrides: dict[str, float] | None = Field(
        None, description="Необязательно: переопределить допущения плана (take_up, spend_uplift, balance_uplift, "
                          "category_share_shift, extra_cashback_rate, partner_share) для what-if."
    )


class TrendsInput(_In):
    metric: Literal["net_income", "spend", "avg_balance", "interchange_income", "nii_income", "net_cashback_cost"] = "net_income"


class EffectInput(_In):
    metrics: list[Literal["net_income", "interchange_income", "nii_income", "net_cashback_cost", "spend",
                          "avg_balance", "is_active"]] | None = Field(
        None, description="Метрики для оценки. По умолчанию — чистый доход и его компоненты.")
    by_segment: bool = Field(False, description="Дополнительно оценить эффект на чистый доход по сегментам клиентов.")


class InvestInput(_In):
    scenario: Literal["plan", "fact"] = Field(..., description="plan — эффект из бизнес-кейса, fact — эффект пилота.")
    overrides: dict[str, Any] | None = Field(
        None, description="What-if: переопределить параметры масштабирования. Ключи: discount_rate, tax_rate, "
                          "horizon_years, eligible_clients (список по годам), effect_decay_per_year, capex, opex_per_year, "
                          "marketing_per_year (список), depreciation_years, clients_mult, opex_mult, capex_mult.")
    unit_effect_override: float | None = Field(
        None, description="What-if: свой эффект на клиента в месяц, ₽ (например, эффект одного сегмента).")


class SensInput(_In):
    scenario: Literal["plan", "fact"]


class ReportInput(_In):
    recommendation: Literal[report.RECOMMENDATIONS]  # type: ignore[valid-type]
    summary: str = Field(..., description="Резюме для комитета: 1–3 абзаца. Только числа из результатов инструментов.")
    risks_and_conditions: str = Field(
        ..., description="Риски, ограничения оценки и условия масштабирования — маркированным списком.")


@dataclass
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[..., dict]

    def schema(self) -> dict:
        s = self.input_model.model_json_schema()
        s.pop("title", None)
        return {"name": self.name, "description": self.description, "input_schema": s}


TOOLS: list[Tool] = [
    Tool("load_initiative",
         "Загрузить паспорт инициативы: описание, допущения бизнес-кейса, план масштабирования, затраты, ставку. "
         "Вызывать первым.", NoInput, lambda ctx: data.load_initiative(ctx)),
    Tool("build_data_mart",
         "Собрать витрину «клиент × месяц» из сырых таблиц пилота (SQL) и прогнать проверки качества данных. "
         "Возвращает объём, разбивку на пилот/контроль и статус качества. Если status=error — оценку эффекта не делать.",
         NoInput, lambda ctx: data.build_data_mart(ctx)),
    Tool("baseline_metrics",
         "Базовые метрики до пилота (на клиента пилотного региона в месяц): оборот, остаток, доля категорийных трат, "
         "чистый доход. Нужны для расчёта плана.", NoInput, lambda ctx: data.baseline_metrics(ctx)),
    Tool("plan_unit_economics",
         "Плановый (ex-ante) эффект на клиента в месяц из допущений бизнес-кейса: разложение на interchange, доход на "
         "остатки и расходы на кэшбэк. Можно передать overrides для what-if.",
         PlanInput, lambda ctx, overrides=None: data.plan_unit_economics(ctx, overrides)),
    Tool("check_parallel_trends",
         "Проверить ключевую предпосылку разности разностей: параллельность трендов пилота и контроля до старта "
         "(event study + тест Вальда) и placebo-тест с ложной датой старта. Выполнять ДО интерпретации эффекта.",
         TrendsInput, lambda ctx, metric="net_income": effect.check_parallel_trends(ctx, metric)),
    Tool("estimate_effect",
         "Оценить фактический эффект пилота методом разности разностей (фиксированные эффекты клиента и месяца, "
         "кластеризация по региону). Возвращает ITT на клиента в месяц с 95% ДИ, долю подключений и ATT на подключившего.",
         EffectInput, lambda ctx, metrics=None, by_segment=False: effect.estimate_effect(ctx, metrics, by_segment)),
    Tool("evaluate_investment",
         "Построить денежные потоки масштабирования и посчитать NPV, IRR, дисконтированную окупаемость, индекс "
         "доходности и точку безубыточности по эффекту. scenario=plan — эффект бизнес-кейса, fact — эффект пилота. "
         "Базовые расчёты (без overrides) сохраняются для отчёта; what-if — нет.",
         InvestInput, lambda ctx, scenario, overrides=None, unit_effect_override=None:
             finance.evaluate_investment(ctx, scenario, overrides, unit_effect_override)),
    Tool("sensitivity_analysis",
         "Анализ чувствительности NPV (торнадо ±20% по драйверам) и Monte Carlo: неопределённость эффекта берётся из "
         "стандартной ошибки пилота (fact) или экспертной оценки (plan). Возвращает P(NPV>0) и квантили NPV.",
         SensInput, lambda ctx, scenario: finance.sensitivity_analysis(ctx, scenario)),
    Tool("build_report",
         "Собрать итоговый отчёт для инвестиционного комитета. Таблицы формируются из результатов инструментов "
         "автоматически; вы передаёте рекомендацию, резюме и риски. Числа в тексте сверяются с расчётами: "
         "если найдено число, которого нет в результатах, отчёт отклоняется — исправьте текст и вызовите снова.",
         ReportInput, lambda ctx, **kw: report.build_report(ctx, **kw)),
]
BY_NAME = {t.name: t for t in TOOLS}


def tool_schemas() -> list[dict]:
    return [t.schema() for t in TOOLS]


def run_tool(ctx: EvalContext, name: str, raw_input: dict) -> tuple[str, bool]:
    """Выполнить инструмент. Возвращает (JSON-строка результата, is_error)."""
    tool = BY_NAME.get(name)
    if tool is None:
        return json.dumps({"error": f"Неизвестный инструмент {name}"}, ensure_ascii=False), True
    try:
        args = tool.input_model.model_validate(raw_input or {})
    except ValidationError as e:
        return json.dumps({"error": "Невалидные входные параметры", "details": e.errors(include_url=False)},
                          ensure_ascii=False, default=str), True
    t0 = time.perf_counter()
    try:
        out = tool.fn(ctx, **args.model_dump(exclude_none=True))
    except (PrerequisiteError, ValueError, FileNotFoundError) as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False), True
    out = {**out, "_elapsed_sec": round(time.perf_counter() - t0, 2)}
    return json.dumps(out, ensure_ascii=False, default=str), False
