"""Цикл агента без сети: «модель» заменена сценарием, проверяем обвязку —
параллельные вызовы, возврат ошибок модели, журнал, завершение по build_report."""
import json
from types import SimpleNamespace as NS

import fineval.agent as agent_mod


def _tu(i, name, inp):
    return NS(type="tool_use", id=f"tu_{i}", name=name, input=inp)


def _resp(content, stop="tool_use"):
    return NS(content=content, stop_reason=stop, stop_details=None, model="fake",
              usage=NS(input_tokens=10, output_tokens=5))


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.seen = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.seen.append(json.loads(json.dumps(kw["messages"], default=lambda o: o.__dict__)))
        return self.script.pop(0)


def test_agent_loop_end_to_end(pilot, monkeypatch):
    ctx, _ = pilot
    script = [
        _resp([_tu(1, "load_initiative", {}), _tu(2, "build_data_mart", {})]),
        _resp([_tu(3, "evaluate_investment", {"scenario": "best"})]),  # ошибка валидации
        _resp([_tu(4, "baseline_metrics", {}), _tu(5, "plan_unit_economics", {}),
               _tu(6, "check_parallel_trends", {})]),
        _resp([_tu(7, "estimate_effect", {"by_segment": True})]),
        _resp([_tu(8, "evaluate_investment", {"scenario": "plan"}), _tu(9, "evaluate_investment", {"scenario": "fact"}),
               _tu(10, "sensitivity_analysis", {"scenario": "fact"})]),
        _resp([_tu(11, "build_report", {"recommendation": "масштабировать",
                                        "summary": "Даст 12345 млн ₽.", "risks_and_conditions": "-"})]),
        _resp([_tu(12, "build_report", {"recommendation": "доработать и повторить пилот",
                                        "summary": "Эффект ниже плана.", "risks_and_conditions": "- Партнёры."})]),
        _resp([NS(type="text", text="Рекомендация: доработать и повторить пилот.")], stop="end_turn"),
    ]
    fake = FakeClient(script)
    monkeypatch.setattr(agent_mod.anthropic, "Anthropic", lambda: fake)
    res = agent_mod.run_agent(ctx, verbose=False)

    assert res["report"]["recommendation"] == "доработать и повторить пилот"
    assert res["final_text"].startswith("Рекомендация")
    # параллельные результаты вернулись одним сообщением
    second_call_last = fake.seen[1][-1]["content"]
    assert [r["tool_use_id"] for r in second_call_last] == ["tu_1", "tu_2"]
    # ошибка валидации ушла модели как is_error
    third_call_last = fake.seen[2][-1]["content"][0]
    assert third_call_last["is_error"] is True
    # выдуманное число отклонено, отчёт собран со второй попытки
    rejected = json.loads(fake.seen[6][-1]["content"][0]["content"])
    assert rejected["status"] == "rejected"
    trace = [json.loads(l) for l in open(res["trace"])]
    assert sum(1 for t in trace if "tool" in t) == 12
