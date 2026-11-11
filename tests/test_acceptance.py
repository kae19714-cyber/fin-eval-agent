"""Приёмочные тесты методики: на данных с известным истинным эффектом.

Это те же критерии, по которым принимается tool у разработчиков (docs/TOOL_REQUIREMENTS.md).
"""
import json

import duckdb
import pytest

from fineval.registry import run_tool
from fineval.tools import data, effect
from fineval.tools.data import SOURCES, mart_params, render_sql


@pytest.fixture(scope="module")
def ready(pilot):
    ctx, truth = pilot
    data.load_initiative(ctx)
    data.build_data_mart(ctx)
    return ctx, truth


def test_mart_row_count_and_formula(ready):
    ctx, _ = ready
    con = duckdb.connect()
    n_src = con.execute(f"SELECT COUNT(*) FROM read_parquet('{ctx.data_dir / 'card_activity.parquet'}')").fetchone()[0]
    assert len(ctx.mart) == n_src  # джойны не размножили и не потеряли строки
    m = ctx.mart
    eco = ctx.initiative["economics"]
    recomputed = m.spend * eco["interchange_rate"] + m.avg_balance * eco["ftp_spread"] / 12 - (
        m.cashback_cost - m.partner_compensation)
    assert (recomputed - m.net_income).abs().max() < 1e-6
    assert set(m.rel_month.unique()) == set(range(-12, 12))


def test_sql_is_portable():
    """Витрина не использует диалектных функций DuckDB, которых нет в Spark SQL."""
    sql = render_sql("mart_client_month.sql", {"pilot_start": 202601, "interchange": 0.012, "ftp_spread": 0.05}).lower()
    for bad in ("read_parquet", "::", "date_diff", "strftime", "qualify", "list_", "struct_pack"):
        assert bad not in sql


def test_dq_ok_on_clean_data(ready):
    ctx, _ = ready
    assert ctx.results["mart"]["data_quality"]["status"] == "ok"


def test_did_recovers_true_effect(ready):
    ctx, truth = ready
    effect.check_parallel_trends(ctx)
    res = effect.estimate_effect(ctx)
    e = res["effects"]["net_income"]
    true = truth["true_itt_net_income_per_client_month"]
    assert e["ci95"][0] <= true <= e["ci95"][1], (e, true)
    assert abs(e["itt"] - true) < 2.5 * e["se"]
    assert res["take_up"] == pytest.approx(truth["effect"]["take_up"], abs=0.05)
    assert res["warnings"] == []


def test_placebo_and_pretrends_pass(ready):
    ctx, _ = ready
    pt = ctx.results["pretrends"]["net_income"]
    assert pt["parallel_trends_ok"], pt


def test_pretrend_violation_detected(broken_pilot):
    ctx, _ = broken_pilot
    data.load_initiative(ctx)
    data.build_data_mart(ctx)
    pt = effect.check_parallel_trends(ctx)
    assert not pt["parallel_trends_ok"]
    res = effect.estimate_effect(ctx)
    assert any("НЕ пройдена" in w for w in res["warnings"])


def test_registry_rejects_bad_input(ready):
    ctx, _ = ready
    out, err = run_tool(ctx, "evaluate_investment", {"scenario": "optimistic"})
    assert err and "Невалидные" in json.loads(out)["error"]
    out, err = run_tool(ctx, "estimate_effect", {"metrics": ["profit"]})
    assert err
    out, err = run_tool(ctx, "build_data_mart", {"unexpected": 1})
    assert err


def test_prerequisites_enforced(tmp_path):
    from tests.conftest import make_ctx
    ctx, _ = make_ctx(tmp_path, n_clients=500)
    out, err = run_tool(ctx, "evaluate_investment", {"scenario": "fact"})
    assert err and "Сначала" in json.loads(out)["error"]


def test_sources_list_matches_generator(ready):
    ctx, _ = ready
    assert all((ctx.data_dir / f"{s}.parquet").exists() for s in SOURCES)
    assert mart_params(ctx.initiative)["pilot_start"] == 202601


def test_method_calibration_over_seeds(tmp_path):
    """Калибровка методики на 10 независимых пилотах: 95% ДИ должны накрывать истинный эффект,
    а тест трендов — редко ошибаться на «чистых» данных."""
    from tests.conftest import make_ctx
    covered, false_alarm = 0, 0
    n = 10
    for seed in range(100, 100 + n):
        ctx, truth = make_ctx(tmp_path / str(seed), n_clients=4000, seed=seed)
        data.load_initiative(ctx)
        data.build_data_mart(ctx)
        false_alarm += not effect.check_parallel_trends(ctx)["parallel_trends_ok"]
        e = effect.estimate_effect(ctx)["effects"]["net_income"]
        covered += e["ci95"][0] <= truth["true_itt_net_income_per_client_month"] <= e["ci95"][1]
    assert covered >= 8, f"покрытие ДИ {covered}/{n}"
    assert false_alarm <= 2, f"ложные срабатывания теста трендов {false_alarm}/{n}"
