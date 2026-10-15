"""Step 4 - Silver layer: clean and conform the Bronze data.
Handles nulls, out-of-range values, duplicates, types, inconsistent text and orphan keys,
and writes a data-quality report (silver/dq_report)."""
from pyspark.sql import Window
from pyspark.sql import functions as F

from common import get_args, get_spark, path


def main():
    a = get_args("Silver layer")
    spark = get_spark("movie-reco-silver")
    bronze = lambda t: spark.read.parquet(path(a.base, "bronze", t))
    dq = []

    def track(table, check, before, after):
        dq.append((table, check, before, after, before - after))
        print(f"[dq] {table:<8} {check:<28} {before:>12,} -> {after:>12,} (removed {before - after:,})")
        return after

    # ---- movies + links -------------------------------------------------
    movies = bronze("movies")
    n = movies.count()
    movies = movies.dropna(subset=["movieId", "title"])
    n = track("movies", "null id/title", n, movies.count())
    movies = movies.dropDuplicates(["movieId"])
    track("movies", "duplicate movieId", n, movies.count())
    links = bronze("links").select("movieId", "imdbId", "tmdbId").dropDuplicates(["movieId"])
    movies = (movies
              .withColumn("title", F.trim("title"))
              .withColumn("release_year", F.regexp_extract("title", r"\((\d{4})\)\s*$", 1).cast("int"))
              .withColumn("clean_title", F.trim(F.regexp_replace("title", r"\s*\(\d{4}\)\s*$", "")))
              .withColumn("genres", F.when(F.col("genres") == "(no genres listed)", None)
                          .otherwise(F.split("genres", r"\|")))
              .join(links, "movieId", "left")
              .select("movieId", "title", "clean_title", "release_year", "genres", "imdbId", "tmdbId"))
    movies.write.mode("overwrite").parquet(path(a.base, "silver", "movies"))

    # ---- ratings --------------------------------------------------------
    ratings = bronze("ratings")
    n = ratings.count()
    ratings = ratings.dropna(subset=["userId", "movieId", "rating", "timestamp"])
    n = track("ratings", "null fields", n, ratings.count())
    ratings = ratings.filter(F.col("rating").between(0.5, 5.0) & (F.col("timestamp") > 0))
    n = track("ratings", "invalid rating/timestamp", n, ratings.count())
    # keep the latest rating when a user rated the same movie more than once
    w = Window.partitionBy("userId", "movieId").orderBy(F.col("timestamp").desc())
    ratings = ratings.withColumn("rn", F.row_number().over(w)).filter("rn = 1").drop("rn")
    n = track("ratings", "duplicate user-movie", n, ratings.count())
    ratings = ratings.join(movies.select("movieId"), "movieId", "left_semi")
    track("ratings", "orphan movieId", n, ratings.count())
    ratings = (ratings
               .withColumn("rated_at", F.timestamp_seconds("timestamp"))
               .withColumn("rating_year", F.year("rated_at"))
               .select("userId", "movieId", "rating", "rated_at", "rating_year"))
    ratings.write.mode("overwrite").partitionBy("rating_year").parquet(path(a.base, "silver", "ratings"))

    # ---- tags -----------------------------------------------------------
    tags = bronze("tags")
    n = tags.count()
    tags = (tags.withColumn("tag", F.lower(F.trim("tag")))
            .filter(F.col("tag").isNotNull() & (F.length("tag") > 0)))
    n = track("tags", "null/empty tag", n, tags.count())
    tags = tags.dropDuplicates(["userId", "movieId", "tag"])
    track("tags", "duplicate tag", n, tags.count())
    (tags.withColumn("tagged_at", F.timestamp_seconds("timestamp"))
     .select("userId", "movieId", "tag", "tagged_at")
     .write.mode("overwrite").parquet(path(a.base, "silver", "tags")))

    (spark.createDataFrame(dq, "table STRING, check STRING, rows_before LONG, rows_after LONG, removed LONG")
     .coalesce(1).write.mode("overwrite").parquet(path(a.base, "silver", "dq_report")))
    spark.stop()


if __name__ == "__main__":
    main()
