"""Та же витрина на PySpark — для прод-объёмов (миллионы клиентов × месяцы).

SQL берётся из fineval/sql/mart_client_month.sql без изменений: локальная
версия на DuckDB и Spark-версия не расходятся по логике.

  pip install pyspark   # нужна Java 17+
  python spark/build_mart_spark.py data/raw data/mart_spark
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fineval.tools.data import SOURCES, mart_params, render_sql  # noqa: E402


def build(raw_dir: str, out_dir: str, initiative: str = "initiatives/cashback_pilot.yaml") -> int:
    spark = (SparkSession.builder.appName("fineval-mart")
             .config("spark.sql.session.timeZone", "UTC").getOrCreate())
    for s in SOURCES:
        spark.read.parquet(str(Path(raw_dir) / f"{s}.parquet")).createOrReplaceTempView(s)
    ini = yaml.safe_load((ROOT / initiative).read_text())
    mart = spark.sql(render_sql("mart_client_month.sql", mart_params(ini)))
    mart.write.mode("overwrite").partitionBy("month").parquet(out_dir)
    n = mart.count()
    spark.stop()
    return n


if __name__ == "__main__":
    raw, out = (sys.argv[1:3] + ["data/raw", "data/mart_spark"][len(sys.argv[1:3]):])
    print(f"Строк в витрине: {build(raw, out)}")
