"""Состояние одной оценки: паспорт, витрина, результаты инструментов.

Инструменты складывают результаты в контекст, а отчёт собирается только из
контекста. Числа в итоговых выводах берутся из расчётов, а не из текста модели.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class EvalContext:
    initiative_path: Path
    out_dir: Path
    initiative: dict = field(default_factory=dict)
    mart: pd.DataFrame | None = None
    results: dict[str, Any] = field(default_factory=dict)
    # Данные только для графиков (выборки Monte Carlo и т.п.). В results не кладём:
    # results — это «источник правды» для проверки чисел в тексте агента.
    charts: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_initiative(cls, initiative_path: str | Path, out_dir: str | Path | None = None):
        p = Path(initiative_path)
        if not p.is_absolute():
            p = (ROOT / p) if (ROOT / p).exists() else p.resolve()
        data = yaml.safe_load(p.read_text())
        out = Path(out_dir) if out_dir else ROOT / "runs" / data["id"]
        out.mkdir(parents=True, exist_ok=True)
        return cls(initiative_path=p, out_dir=out, initiative=data)

    @property
    def data_dir(self) -> Path:
        d = Path(self.initiative["pilot"]["data_dir"])
        return d if d.is_absolute() else ROOT / d

    def require(self, *keys: str) -> None:
        missing = [k for k in keys if k not in self.results]
        if missing:
            raise PrerequisiteError(
                f"Сначала нужно выполнить шаги, которые дают: {', '.join(missing)}"
            )


class PrerequisiteError(RuntimeError):
    """Инструмент вызван раньше, чем готовы его входные данные."""
