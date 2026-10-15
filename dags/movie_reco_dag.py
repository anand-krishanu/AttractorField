"""Airflow DAG: end-to-end movie recommendation pipeline on Google Cloud Dataproc.

create_cluster -> ingest -> bronze -> silver -> gold -> train_als -> delete_cluster

Deploy to Cloud Composer / any Airflow 2.x with apache-airflow-providers-google, then set
Airflow Variables: gcp_project, gcp_region, gcs_bucket (code must be uploaded to gs://<bucket>/code/).
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.providers.google.cloud.operators.dataproc import (
    DataprocCreateClusterOperator,
    DataprocDeleteClusterOperator,
    DataprocSubmitJobOperator,
)
from airflow.utils.trigger_rule import TriggerRule

PROJECT = Variable.get("gcp_project", "my-gcp-project")
REGION = Variable.get("gcp_region", "asia-south1")
BUCKET = Variable.get("gcs_bucket", f"{PROJECT}-movie-reco")
DATASET = "ml-25m"
CLUSTER = "movie-reco-{{ ds_nodash }}"
CODE = f"gs://{BUCKET}/code"

CLUSTER_CONFIG = {
    "master_config": {"num_instances": 1, "machine_type_uri": "e2-standard-2",
                      "disk_config": {"boot_disk_size_gb": 100}},
    "worker_config": {"num_instances": 2, "machine_type_uri": "e2-highmem-4",
                      "disk_config": {"boot_disk_size_gb": 100}},
    "software_config": {"image_version": "2.2-debian12"},
    "lifecycle_config": {"idle_delete_ttl": {"seconds": 1800}},
}


def pyspark_job(script, args):
    return {
        "reference": {"project_id": PROJECT},
        "placement": {"cluster_name": CLUSTER},
        "pyspark_job": {"main_python_file_uri": f"{CODE}/{script}",
                        "python_file_uris": [f"{CODE}/common.py"], "args": args},
    }


with DAG(
    dag_id="movie_recommendation_pipeline",
    start_date=datetime(2026, 10, 1),
    schedule="@weekly",  # MovieLens snapshots change rarely; weekly retrain is plenty
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["bigdata", "movielens", "dataproc"],
) as dag:
    create = DataprocCreateClusterOperator(task_id="create_cluster", project_id=PROJECT, region=REGION,
                                           cluster_name=CLUSTER, cluster_config=CLUSTER_CONFIG)

    ingest = DataprocSubmitJobOperator(
        task_id="ingest", project_id=PROJECT, region=REGION,
        job=pyspark_job("ingest.py", ["--dataset", DATASET, "--out", "/tmp/raw", "--gcs-bucket", BUCKET]))

    stages = [DataprocSubmitJobOperator(
        task_id=stage, project_id=PROJECT, region=REGION,
        job=pyspark_job(f"{stage}.py", ["--base", f"gs://{BUCKET}", "--dataset", DATASET]))
        for stage in ("bronze", "silver", "gold", "train_als")]

    delete = DataprocDeleteClusterOperator(task_id="delete_cluster", project_id=PROJECT, region=REGION,
                                           cluster_name=CLUSTER, trigger_rule=TriggerRule.ALL_DONE)

    create >> ingest >> stages[0] >> stages[1] >> stages[2] >> stages[3] >> delete
