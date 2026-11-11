"""AI-агент финансовой оценки на Claude: модель планирует шаги и интерпретирует, инструменты считают.

Цикл написан вручную (без SDK tool runner), чтобы был виден каждый шаг и
сохранялся полный журнал вызовов для приёмки и аудита (runs/<id>/trace.jsonl).
"""
from __future__ import annotations

import json
import time

import anthropic

from fineval.context import EvalContext
from fineval.registry import run_tool, tool_schemas

MODEL = "claude-opus-5-5"
MAX_TURNS = 30

SYSTEM_PROMPT = """\
Ты — аналитик дирекции инвестиционной экспертизы банка. Твоя задача — оценить бизнес-инициативу:
сравнить бизнес-кейс (план) с фактическим эффектом пилота и дать рекомендацию инвестиционному комитету.

Как работать:
- Все числа получай только из инструментов. Не считай в уме и не придумывай значения: если нужного числа нет,
  вызови инструмент (в том числе what-if через overrides).
- Порядок по методологии: паспорт → витрина и качество данных → базовые метрики и план → проверка
  параллельности трендов → фактический эффект → NPV плана и факта → чувствительность → отчёт.
- Если качество данных с ошибками или параллельность трендов не подтверждена, прямо скажи, что оценке
  эффекта доверять нельзя, и учти это в рекомендации.
- Различай статистическую значимость эффекта и экономическую: эффект может быть значим, но ниже точки безубыточности.
- Посмотри на эффект по сегментам и, если это меняет решение, посчитай what-if (например, масштабирование
  только на сегмент с сильным эффектом через unit_effect_override и eligible_clients).
- Рекомендацию сверяй с формальным правилом, которое возвращает build_report. Отступать от него можно только
  с явным обоснованием в резюме.
- Пиши для комитета: коротко, по-русски, с выводами и условиями, без описания своих шагов.

Закончи работу вызовом build_report. Если отчёт отклонён из-за непроверенных чисел — исправь текст и вызови снова.
После успешного отчёта ответь одним абзацем: рекомендация и путь к отчёту.
"""


def run_agent(ctx: EvalContext, task: str | None = None, model: str = MODEL, verbose: bool = True) -> dict:
    client = anthropic.Anthropic()
    task = task or (
        f"Оцени инициативу из паспорта {ctx.initiative_path.name}: план против факта пилота, "
        "стоит ли масштабировать и на каких условиях."
    )
    messages: list[dict] = [{"role": "user", "content": task}]
    trace_path = ctx.out_dir / "trace.jsonl"
    trace = trace_path.open("w")
    usage = {"input_tokens": 0, "output_tokens": 0}
    final_text = ""

    def log(event: dict):
        trace.write(json.dumps({"t": round(time.time(), 2), **event}, ensure_ascii=False, default=str) + "\n")
        trace.flush()

    try:
        for turn in range(MAX_TURNS):
            resp = client.beta.messages.create(
                model=model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                tools=tool_schemas(),
                messages=messages,
                output_config={"effort": "high"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            usage["input_tokens"] += resp.usage.input_tokens
            usage["output_tokens"] += resp.usage.output_tokens
            log({"turn": turn, "stop_reason": resp.stop_reason, "model": resp.model,
                 "usage": {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}})

            if resp.stop_reason == "refusal":
                raise RuntimeError(f"Модель отказалась отвечать: {resp.stop_details}")
            messages.append({"role": "assistant", "content": resp.content})

            for b in resp.content:
                if b.type == "text" and b.text.strip():
                    final_text = b.text
                    log({"assistant_text": b.text})
                    if verbose:
                        print(f"\n🤖 {b.text.strip()}\n")

            if resp.stop_reason == "pause_turn":
                continue
            if resp.stop_reason == "max_tokens":
                messages.append({"role": "user", "content": "Ответ обрезан по длине. Продолжи короче."})
                continue
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                break

            results = []
            for tu in tool_uses:
                out, is_err = run_tool(ctx, tu.name, tu.input)
                log({"tool": tu.name, "input": tu.input, "is_error": is_err, "output": json.loads(out)})
                if verbose:
                    mark = "✗" if is_err else "✓"
                    print(f"  {mark} {tu.name}({json.dumps(tu.input, ensure_ascii=False)[:120]})")
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": out, "is_error": is_err})
            # все результаты параллельных вызовов — одним сообщением
            messages.append({"role": "user", "content": results})
        else:
            log({"warning": f"Достигнут лимит {MAX_TURNS} ходов"})
    finally:
        log({"usage_total": usage})
        trace.close()

    rep = ctx.results.get("report")
    return {"final_text": final_text, "report": rep, "trace": str(trace_path), "usage": usage}

