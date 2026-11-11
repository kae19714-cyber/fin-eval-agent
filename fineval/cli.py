"""Точка входа.

  python -m fineval.cli generate                 # сгенерировать данные пилота
  python -m fineval.cli pipeline                 # прогон инструментов без LLM
  python -m fineval.cli agent                    # агент на Claude (нужен ANTHROPIC_API_KEY)
  python -m fineval.cli agent --task "Что будет, если масштабировать только на affluent?"
"""
from __future__ import annotations

import argparse

from fineval.context import EvalContext


def main() -> None:
    ap = argparse.ArgumentParser(prog="fineval")
    ap.add_argument("command", choices=["generate", "pipeline", "agent"])
    ap.add_argument("--initiative", default="initiatives/cashback_pilot.yaml")
    ap.add_argument("--out", default=None, help="Папка для отчёта и журнала (по умолчанию runs/<id>)")
    ap.add_argument("--task", default=None, help="Задание агенту своими словами")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()

    if a.command == "generate":
        from fineval.data_gen import generate
        print(generate("data/raw"))
        return

    ctx = EvalContext.from_initiative(a.initiative, a.out)
    if a.command == "pipeline":
        from fineval.pipeline import run_pipeline
        rep = run_pipeline(ctx)
        print(f"\nРекомендация: {rep['recommendation']}\nОтчёт: {rep['path']}")
    else:
        from fineval.agent import MODEL, run_agent
        res = run_agent(ctx, a.task, model=a.model or MODEL)
        print(f"\nОтчёт: {res['report']['path'] if res['report'] else 'не сформирован'}")
        print(f"Журнал: {res['trace']}\nТокены: {res['usage']}")


if __name__ == "__main__":
    main()
