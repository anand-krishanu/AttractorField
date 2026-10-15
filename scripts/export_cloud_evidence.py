"""Export proof of the Google Cloud Dataproc run into docs/cloud_run_evidence/ (data/ is git-ignored).

    python scripts/export_cloud_evidence.py --project <id> --account <email>

Writes: cleaned run logs, Dataproc job history (jobs.txt), bucket listing and the final ALS driver output.
"""
import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "cloud_run_evidence"
NOISE = re.compile(r"^\s*(\+ |At line|CategoryInfo|FullyQualifiedErrorId|~+\s*$|\.+\s*$|% Total|Dload|\d+\s+\d+[kMG]?\s+\d+)")


def gcloud(*args):
    r = subprocess.run(["gcloud", *args], capture_output=True, text=True, encoding="utf-8",
                       shell=os.name == "nt")  # gcloud is a .cmd on Windows
    if r.returncode:
        raise SystemExit(r.stderr)
    return r.stdout


def clean_log(src, dst):
    text = src.read_bytes().replace(b"\x00", b"").decode("utf-8", "ignore").lstrip("﻿")
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "<account>", text)  # keep e-mail addresses out of the repo
    lines = [re.sub(r"^powershell : ", "", l.rstrip()) for l in text.splitlines()]
    dst.write_text("\n".join(l for l in lines if l.strip() and not NOISE.match(l)) + "\n", encoding="utf-8")


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")[:26] + "+00:00") if s else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--account", required=True)
    p.add_argument("--region", default="asia-south1")
    a = p.parse_args()
    os.environ["CLOUDSDK_CORE_PROJECT"], os.environ["CLOUDSDK_CORE_ACCOUNT"] = a.project, a.account
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screenshots").mkdir(exist_ok=True)
    bucket = f"gs://{a.project}-movie-reco"

    for name, dst in (("dataproc_run_full.log", "run_full_pipeline.log"), ("dataproc_run.log", "run_als_rerun.log")):
        if (ROOT / "data" / name).exists():
            clean_log(ROOT / "data" / name, OUT / dst)

    jobs = json.loads(gcloud("dataproc", "jobs", "list", f"--region={a.region}", "--format=json"))
    rows = []
    for j in sorted(jobs, key=lambda j: j["statusHistory"][0]["stateStartTime"] if j.get("statusHistory") else ""):
        start = ts(j["statusHistory"][0]["stateStartTime"]) if j.get("statusHistory") else None
        end = ts(j["status"]["stateStartTime"])
        rows.append((j["reference"]["jobId"], j["pysparkJob"]["mainPythonFileUri"].rsplit("/", 1)[-1],
                     " ".join(j["pysparkJob"].get("args", [])), j["status"]["state"],
                     f"{start:%Y-%m-%d %H:%M:%S}" if start else "", f"{end:%H:%M:%S}",
                     f"{(end - start).total_seconds():.0f}" if start else ""))
    header = ("job_id", "script", "args", "state", "start_utc", "end_utc", "secs")
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    fmt = lambda r: "  ".join(str(v).ljust(w) for v, w in zip(r, widths))
    (OUT / "jobs.txt").write_text(
        f"Dataproc job history - project {a.project}, region {a.region}, cluster movie-reco-cluster\n"
        "(the ERROR ingest job was the first attempt: Dataproc 2.2 nodes have internal IPs only, so the download "
        "from GroupLens failed; fixed by adding Cloud NAT. The second train_als run is the re-run with the "
        ">= 50 ratings-per-movie filter.)\n\n" + "\n".join(fmt(r) for r in [header, *rows]) + "\n", encoding="utf-8")

    du = gcloud("storage", "du", "-s", "-r", *[f"{bucket}/{d}" for d in ("raw", "bronze", "silver", "gold", "output", "code")])
    (OUT / "bucket_listing.txt").write_text(
        f"Bucket {bucket} - size per layer (bytes)\n\n{du}\n\nFull listing\n\n"
        + gcloud("storage", "ls", "-r", "-l", *[f"{bucket}/{d}" for d in ("raw", "bronze", "silver", "gold", "output", "code")]),
        encoding="utf-8")

    als = [j for j in jobs if j["pysparkJob"]["mainPythonFileUri"].endswith("train_als.py") and j["status"]["state"] == "DONE"]
    if als:
        last = max(als, key=lambda j: j["status"]["stateStartTime"])
        driver = gcloud("storage", "cat", last["driverOutputResourceUri"] + "*")
        (OUT / "train_als_driver_output.txt").write_text(
            "\n".join(l for l in driver.splitlines() if " WARN " not in l and " INFO " not in l) + "\n", encoding="utf-8")
    print(f"Evidence written to {OUT}")


if __name__ == "__main__":
    main()
