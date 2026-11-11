"""Отчёт и защита от выдуманных чисел."""
import json

import pytest

from fineval.pipeline import run_pipeline
from fineval.registry import run_tool
from fineval.tools.report import unverified_numbers


def test_guard_accepts_derived_representations():
    results = {"npv": 215_400_000.0, "p": 0.555, "itt": 14.4631, "irr": 0.3478}
    text = "NPV 215,4 млн ₽, вероятность 55,5%, эффект 14,46 ₽, IRR 34.8%, горизонт 5 лет, с 2026 года."
    assert unverified_numbers(text, results) == []


def test_guard_flags_invented_numbers():
    results = {"npv": 215_400_000.0}
    assert unverified_numbers("NPV составит 1 200 млн ₽", results) == ["1 200"]


@pytest.fixture(scope="module")
def finished(pilot):
    ctx, _ = pilot
    run_pipeline(ctx, verbose=False)
    return ctx


def test_pipeline_end_to_end(finished):
    rep = finished.results["report"]
    md = open(rep["path"]).read()
    assert "## Рекомендация" in md and "Monte Carlo" in md
    assert rep["recommendation"] == rep["decision_rule"]["recommendation"]


def test_report_rejects_hallucinated_numbers(finished):
    out, err = run_tool(finished, "build_report", {
        "recommendation": "масштабировать",
        "summary": "Инициатива принесёт 9 999 млн ₽.",
        "risks_and_conditions": "- нет",
    })
    res = json.loads(out)
    assert not err and res["status"] == "rejected" and "9 999" in res["unverified_numbers"]
