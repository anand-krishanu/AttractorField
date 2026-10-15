"""Step 3 - Bronze layer: raw CSV -> Parquet with explicit schemas and ingestion metadata.
Data is kept as close to the source as possible (no cleaning here)."""
from pyspark.sql import functions as F

from common import get_args, get_spark, log_df, path

SCHEMAS = {
    "ratings": "userId INT, movieId INT, rating DOUBLE, timestamp LONG",
    "movies": "movieId INT, title STRING, genres STRING",
    "tags": "userId INT, movieId INT, tag STRING, timestamp LONG",
    "links": "movieId INT, imdbId STRING, tmdbId INT",
}


def main():
    a = get_args("Bronze layer")
    spark = get_spark("movie-reco-bronze")
    for name, schema in SCHEMAS.items():
        df = (spark.read.option("header", True).option("escape", '"').schema(schema)
              .csv(path(a.base, "raw", a.dataset, f"{name}.csv"))
              .withColumn("ingest_ts", F.current_timestamp())
              .withColumn("source_file", F.input_file_name())
              .withColumn("dataset_version", F.lit(a.dataset)))
        df.write.mode("overwrite").parquet(path(a.base, "bronze", name))
        log_df(f"bronze.{name}", spark.read.parquet(path(a.base, "bronze", name)))
    spark.stop()


if __name__ == "__main__":
    main()
