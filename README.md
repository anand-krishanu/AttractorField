# Movie Recommendation System using Big Data Analytics

An end-to-end Big Data pipeline on **MovieLens 25M**. It uses PySpark with a Bronze → Silver → Gold medallion design, Spark SQL and window functions, a **Spark MLlib ALS** recommender, **Google Cloud Dataproc + GCS**, an **Airflow** DAG, and a **Streamlit** dashboard.

## 1. Problem statement
Streaming catalogues have tens of thousands of titles, and users struggle to find movies they'll enjoy. Content teams also lack visibility into how tastes change over time.
**Goal:** process 25 million historical ratings at scale, then:
1. Clean the rating data and analyse rating behaviour (popularity, genre trends, user segments).
2. Train a collaborative-filtering model that predicts a user's rating for unseen movies.
3. Serve a personalised Top-10 list for every user, plus "similar movies" for popular titles.

**Target users:** viewers, who get recommendations, and product and content teams, who get KPIs and trends.
**Expected outcomes:** analysis-ready Gold tables, an ALS model that beats a global-mean baseline on RMSE, Top-10 recommendations for every user, and a dashboard built on these outputs.

## 2. Dataset
| | |
|---|---|
| Name | MovieLens 25M (`ml-25m`), plus `ml-latest-small` for local development |
| Source | GroupLens Research, University of Minnesota: https://grouplens.org/datasets/movielens/25m/ |
| Size | 25,000,095 ratings · 62,423 movies · 162,541 users · 1,093,360 tags (1995–2019) |
| Files | `ratings.csv` (userId, movieId, rating 0.5–5, timestamp), `movies.csv` (title, pipe-separated genres), `tags.csv`, `links.csv` (IMDb/TMDb ids) |

**Citation:** F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19. https://doi.org/10.1145/2827872

## 3. Architecture
```
GroupLens (zip) ─► ingest.py ─► GCS raw/ ─► Bronze (Parquet + ingest metadata)
                                              │
                                              ▼
                          Silver (cleaned, deduped, typed, partitioned by year) + DQ report
                                              │
                         ┌────────────────────┴───────────────────┐
                         ▼                                        ▼
            Gold analytics tables (Spark SQL,          MLlib ALS (tuning, RMSE/MAE,
            windows, aggregations)                    precision@10, NDCG@10)
                         │                                        │
                         └──────────► Streamlit dashboard ◄───────┘
                                    (KPIs, trends, recs, model)
Orchestration: run_pipeline.py (local Docker) · scripts/dataproc_run.ps1 (GCP) · dags/movie_reco_dag.py (Airflow)
```

| Step | Implementation |
|---|---|
| Ingest | `src/ingest.py`: downloads the release, writes `_manifest.json` (source, timestamp, file sizes) and uploads to GCS |
| Bronze | `src/bronze.py`: explicit schemas, plus `ingest_ts`, `source_file` and `dataset_version` |
| Silver | `src/silver.py`: nulls, out-of-range ratings, duplicate (user, movie) ratings (keeps the latest, using `row_number`), orphan movieIds, year parsed from the title, genre arrays, normalised tags, links joined. Ratings are partitioned by `rating_year`. Writes `dq_report` |
| Gold | `src/gold.py`: `movie_stats` (Bayesian weighted rating), `top_movies_by_genre` (`rank()`), `genre_trends` (Spark SQL + `LAG` YoY growth), `user_activity` (segments), `user_top_rated`, `rating_distribution`, `top_tags`, `yearly_kpis` (running total) |
| ML | `src/train_als.py`: trains only on movies with at least 50 ratings (avoids obscure-title noise), 70/10/20 split, grid over rank {10, 20} × regParam {0.05, 0.1}, refit, test RMSE/MAE against a baseline, precision@10 and NDCG@10. Outputs `user_recs_top10` (unseen only), `similar_movies` (cosine on item factors), `als_model` and `metrics` |
| Dashboard | `dashboard/app.py`: Trends, Movies & Genres, Users, Recommendations, Model and Data Quality tabs |

## 4. How to run
**Local (Docker; Spark 3.5 / Java 17, mirroring Dataproc 2.2):**
```bash
pip install -r requirements.txt
python run_pipeline.py                     # ml-latest-small, about 2 minutes
streamlit run dashboard/app.py
```

**Google Cloud Dataproc (full 25M):**
```powershell
gcloud auth login
.\scripts\dataproc_run.ps1 -Project <your-project-id>   # creates bucket + cluster, runs all jobs, downloads results, deletes cluster
streamlit run dashboard/app.py
```

**Airflow:** deploy `dags/movie_reco_dag.py` to Cloud Composer or Airflow 2.x with `apache-airflow-providers-google`, and set the Variables `gcp_project`, `gcp_region` and `gcs_bucket`.

## 5. Results
Results from the full **MovieLens 25M** run on Dataproc:

| Metric | Value |
|---|---|
| Ratings / users / movies | 25,000,095 / 162,541 / 59,047 |
| Movies in model (at least 50 ratings) | 13,176 |
| Best ALS params | rank 20, regParam 0.05 |
| **Test RMSE (ALS)** | **0.787** |
| Test RMSE (global-mean baseline) | 1.058 |
| Test MAE | 0.61 |
| Precision@10 / NDCG@10 | 0.024 / 0.031 |
| Users with a Top-10 list | 162,540 |

ALS cuts RMSE by about 26% compared with the baseline. The full write-up, with the data-quality numbers, charts and cloud evidence, is in [`docs/report.pdf`](docs/report.pdf) (built by `python docs/make_report.py`).

## Cloud execution evidence
The full 25M pipeline ran on **Google Cloud Dataproc 2.2** (project `movie-reco-bigdata-2026`, region `asia-south1`, 1 × e2-standard-2 master and 2 × e2-highmem-4 workers). The proof is in [`docs/cloud_run_evidence/`](docs/cloud_run_evidence/), regenerated with `python scripts/export_cloud_evidence.py --project <id> --account <email>`:
- `jobs.txt`: Dataproc job history (job IDs, state, start and end times, durations)
- `run_full_pipeline.log` and `run_als_rerun.log`: cleaned console logs of the runs
- `bucket_listing.txt`: GCS bucket size per layer (raw, bronze, silver, gold, output) and the full object listing
- `train_als_driver_output.txt`: Spark driver output of the final ALS job (tuning grid and metrics)
- `screenshots/`: GCP console screenshots (Dataproc jobs, bucket, billing)

| Dataproc job | Duration |
|---|---|
| ingest (GroupLens to GCS) | ~70 s |
| bronze | ~190 s |
| silver | ~260 s |
| gold | ~245 s |
| train_als (grid of 4 + refit + recommendations) | ~19 min |

## 6. Tech stack
PySpark 3.5 · Spark SQL · Spark MLlib · Google Cloud Storage · Google Cloud Dataproc 2.2 · Apache Airflow · Streamlit · Plotly · Docker

## 7. References
- Harper & Konstan (2015), *The MovieLens Datasets: History and Context*, ACM TiiS.
- Koren, Bell & Volinsky (2009), *Matrix Factorization Techniques for Recommender Systems*, IEEE Computer.
- Hu, Koren & Volinsky (2008), *Collaborative Filtering for Implicit Feedback Datasets*, ICDM (the basis of ALS in Spark).
- Apache Spark MLlib collaborative filtering docs: https://spark.apache.org/docs/3.5.3/ml-collaborative-filtering.html
- Google Cloud Dataproc docs: https://cloud.google.com/dataproc/docs
