"""Сверка витрины Spark vs DuckDB (запускается, если установлен pyspark и есть Java)."""
import shutil

import duckdb
import pytest

pyspark = pytest.importorskip("pyspark")
if not shutil.which("java"):
    pytest.skip("нет Java", allow_module_level=True)


def test_spark_mart_matches_duckdb(pilot, tmp_path):
    from spark.build_mart_spark import build
    from fineval.tools import data

    ctx, _ = pilot
    data.load_initiative(ctx)
    data.build_data_mart(ctx)
    n = build(str(ctx.data_dir), str(tmp_path / "mart"))
    assert n == len(ctx.mart)
    con = duckdb.connect()
    spark_sum = con.execute(
        f"SELECT SUM(net_income), SUM(treated_post) FROM read_parquet('{tmp_path}/mart/*/*.parquet', hive_partitioning=1)"
    ).fetchone()
    assert spark_sum[0] == pytest.approx(ctx.mart.net_income.sum(), rel=1e-9)
    assert spark_sum[1] == ctx.mart.treated_post.sum()
