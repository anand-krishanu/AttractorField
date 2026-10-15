"""Step 2 - Ingest: download a MovieLens release from GroupLens, unzip it into
data/raw/<dataset>/ and write an ingestion manifest. Optionally upload to GCS.

Source: GroupLens Research, University of Minnesota - https://grouplens.org/datasets/movielens/
"""
import argparse
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

URLS = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ml-latest-small", choices=URLS)
    p.add_argument("--out", default="data/raw")
    p.add_argument("--gcs-bucket", help="If set, upload raw files to gs://<bucket>/raw/<dataset>/")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / f"{a.dataset}.zip"
    target = out / a.dataset
    if not target.exists():
        if not zip_path.exists():
            print(f"Downloading {URLS[a.dataset]} ...")
            # curl uses the OS certificate store (avoids Python cert issues on Windows)
            subprocess.run(["curl", "-fL", "-o", str(zip_path), URLS[a.dataset]], check=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out)
    else:
        print(f"{target} already present, skipping download")

    files = sorted(f for f in target.glob("*.csv"))
    manifest = {
        "dataset": a.dataset,
        "source_url": URLS[a.dataset],
        "source": "GroupLens Research, University of Minnesota",
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [{"name": f.name, "bytes": f.stat().st_size} for f in files],
    }
    (target / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    for f in manifest["files"]:
        print(f"  {f['name']:<15} {f['bytes'] / 1e6:8.1f} MB")

    if a.gcs_bucket:
        dest = f"gs://{a.gcs_bucket}/raw/{a.dataset}/"
        subprocess.run(["gsutil", "-m", "cp", *map(str, files), str(target / "_manifest.json"), dest],
                       check=True, shell=True)
        print(f"Uploaded to {dest}")


if __name__ == "__main__":
    main()
