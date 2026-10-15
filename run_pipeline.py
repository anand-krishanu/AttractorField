"""Run the full pipeline locally: ingest (host) -> bronze -> silver -> gold -> ALS (Spark in Docker).

    python run_pipeline.py                       # ml-latest-small (fast dev run)
    python run_pipeline.py --dataset ml-25m --driver-memory 10g
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

IMAGE = "movie-reco-spark:3.5"
STAGES = ["bronze", "silver", "gold", "train_als"]
ROOT = Path(__file__).resolve().parent


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ml-latest-small")
    p.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    p.add_argument("--driver-memory", default="4g")
    p.add_argument("--skip-ingest", action="store_true")
    a = p.parse_args()

    if not a.skip_ingest:
        run([sys.executable, str(ROOT / "src" / "ingest.py"), "--dataset", a.dataset])
    run(["docker", "build", "-q", "-t", IMAGE, str(ROOT / "docker")])
    for stage in a.stages:
        t = time.time()
        run(["docker", "run", "--rm", "-v", f"{ROOT}:/app", "-w", "/app", IMAGE,
             "/opt/spark/bin/spark-submit", "--driver-memory", a.driver_memory,
             "--py-files", "src/common.py", f"src/{stage}.py", "--base", "data", "--dataset", a.dataset])
        print(f"== {stage} finished in {time.time() - t:.0f}s ==")


if __name__ == "__main__":
    main()
