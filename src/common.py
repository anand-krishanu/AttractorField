"""Shared helpers for every Spark stage (local Docker or Dataproc)."""
import argparse

from pyspark.sql import SparkSession


def get_args(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--base", default="data", help="Base path: local dir (data) or gs://<bucket>")
    p.add_argument("--dataset", default="ml-latest-small", help="ml-latest-small | ml-25m")
    return p.parse_args()


def get_spark(app):
    spark = SparkSession.builder.appName(app).config("spark.sql.adaptive.enabled", "true").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def path(base, *parts):
    return "/".join([base.rstrip("/"), *parts])


def log_df(name, df):
    """Print row count + schema so every stage leaves an audit trail in the job log."""
    n = df.count()
    print(f"[{name}] rows={n:,}")
    df.printSchema()
    return n
