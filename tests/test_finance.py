"""Unit-тесты финансовых функций на эталонных значениях."""
import pytest

from fineval.tools.finance import _breakeven, _params, cashflows, discounted_payback, irr, npv
from tests.conftest import BASE


def test_npv_reference():
    # -1000 + 500/1.1 + 600/1.21 = -1000 + 454.545 + 495.868
    assert npv(0.10, [-1000, 500, 600]) == pytest.approx(-49.587, abs=1e-3)


def test_irr_simple():
    r, w = irr([-100, 110])
    assert r == pytest.approx(0.10, abs=1e-9) and w == []


def test_irr_multiple_sign_changes_warns():
    # классический пример с двумя IRR: 10% и 20%
    r, w = irr([-100, 230, -132])
    assert r is not None
    assert any("несколько" in x or "единственным" in x for x in w)


def test_irr_absent():
    r, w = irr([100, 50])
    assert r is None and w


def test_discounted_payback():
    # дисконтированные потоки при 0%: -100, 60, 60 → окупаемость 1 + 40/60
    assert discounted_payback(0.0, [-100, 60, 60]) == pytest.approx(1 + 40 / 60)
    assert discounted_payback(0.1, [-100, 10, 10]) is None


def test_breakeven_gives_zero_npv():
    p = _params(BASE)
    be = _breakeven(p)
    assert npv(p["discount_rate"], cashflows(be, p)["fcf"]) == pytest.approx(0, abs=1)


def test_tax_shield_and_capex_in_year0():
    p = _params(BASE)
    cf = cashflows(0.0, p)
    assert cf["fcf"][0] == -p["capex"]
    assert (cf["tax"] < 0).all()  # без эффекта — только убытки и налоговый щит


def test_unknown_override_rejected():
    with pytest.raises(ValueError):
        _params(BASE, {"wacc": 0.1})
