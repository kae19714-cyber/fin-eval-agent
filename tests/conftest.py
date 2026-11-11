import copy
from pathlib import Path

import pytest
import yaml

from fineval.context import EvalContext
from fineval.data_gen import TrueEffect, generate

ROOT = Path(__file__).resolve().parent.parent
BASE = yaml.safe_load((ROOT / "initiatives/cashback_pilot.yaml").read_text())


def make_ctx(tmp: Path, effect: TrueEffect = TrueEffect(), n_clients: int = 8000, seed: int = 1):
    data_dir = tmp / "raw"
    truth = generate(data_dir, n_clients=n_clients, effect=effect, seed=seed)
    ini = copy.deepcopy(BASE)
    ini["pilot"]["data_dir"] = str(data_dir)
    p = tmp / "initiative.yaml"
    p.write_text(yaml.safe_dump(ini, allow_unicode=True))
    return EvalContext.from_initiative(p, tmp / "out"), truth


@pytest.fixture(scope="session")
def pilot(tmp_path_factory):
    """Чистый пилот с параллельными трендами — ровно тот набор, что в демо (seed 42, 20 000 клиентов).

    Одиночный пилот может не попасть в 95% ДИ просто по случайности (в 1 случае из 20) —
    поэтому калибровка методики проверяется отдельно на 10 сидах (test_method_calibration_over_seeds).
    """
    return make_ctx(tmp_path_factory.mktemp("pilot"), n_clients=20_000, seed=42)


@pytest.fixture(scope="session")
def broken_pilot(tmp_path_factory):
    """Пилотные регионы росли быстрее ещё до пилота — DiD должен это поймать."""
    return make_ctx(tmp_path_factory.mktemp("broken"), TrueEffect(pre_trend_violation=0.006))
