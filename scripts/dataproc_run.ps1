# Run the whole pipeline on Google Cloud Dataproc and pull the results back for the dashboard.
#   .\scripts\dataproc_run.ps1 -Project <gcp-project-id>
# Steps: bucket -> cluster -> ingest (on cluster, straight into GCS) -> bronze -> silver -> gold -> ALS
#        -> download Gold/outputs to .\data -> delete cluster
param(
    [Parameter(Mandatory = $true)][string]$Project,
    [string]$Region = "asia-south1",
    [string]$Bucket = "$Project-movie-reco",
    [string]$Dataset = "ml-25m",
    [string]$Cluster = "movie-reco-cluster",
    [string]$MasterType = "e2-standard-2",
    [string]$WorkerType = "e2-highmem-4",
    [string]$Account,
    [string[]]$Stages = @("ingest", "bronze", "silver", "gold", "train_als"),
    [switch]$KeepCluster
)
$ErrorActionPreference = "Continue"  # native gcloud stderr must not abort; failures are caught by Check
# Scope account/project to this run only (does not change the global gcloud config)
$env:CLOUDSDK_CORE_PROJECT = $Project
if ($Account) { $env:CLOUDSDK_CORE_ACCOUNT = $Account }
$root = Split-Path $PSScriptRoot -Parent
$gs = "gs://$Bucket"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Check { if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE)" } }

Step "Project + APIs"
Write-Host "Account: $(gcloud config get-value account)  Project: $Project"
gcloud services enable dataproc.googleapis.com storage.googleapis.com compute.googleapis.com; Check

Step "Bucket $gs"
gcloud storage buckets describe $gs 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { gcloud storage buckets create $gs --location=$Region; Check }
gcloud storage cp "$root\src\*.py" "$gs/code/"; Check

Step "Cloud NAT (Dataproc 2.2 nodes have internal IPs only; needed to download from GroupLens)"
gcloud compute routers describe movie-reco-router --region=$Region 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    gcloud compute routers create movie-reco-router --network=default --region=$Region; Check
    gcloud compute routers nats create movie-reco-nat --router=movie-reco-router --region=$Region `
        --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges; Check
}

Step "Cluster $Cluster (1 x $MasterType + 2 x $WorkerType, auto-delete after 30 min idle)"
gcloud dataproc clusters describe $Cluster --region=$Region 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    gcloud dataproc clusters create $Cluster --region=$Region --image-version=2.2-debian12 `
        --master-machine-type=$MasterType --master-boot-disk-size=100 `
        --num-workers=2 --worker-machine-type=$WorkerType --worker-boot-disk-size=100 `
        --max-idle=30m --bucket=$Bucket; Check
}

function Submit($script, $extra) {
    Step "Job: $script"
    $t = Get-Date
    gcloud dataproc jobs submit pyspark "$gs/code/$script" --cluster=$Cluster --region=$Region `
        --py-files="$gs/code/common.py" "--" @extra; Check
    Write-Host ("{0} finished in {1:N0}s" -f $script, ((Get-Date) - $t).TotalSeconds)
}

# Ingest runs on the cluster driver: GroupLens -> master node -> GCS (fast, no local upload)
foreach ($stage in $Stages) {
    if ($stage -eq "ingest") { Submit "ingest.py" @("--dataset", $Dataset, "--out", "/tmp/raw", "--gcs-bucket", $Bucket) }
    else { Submit "$stage.py" @("--base", $gs, "--dataset", $Dataset) }
}

Step "Download results to .\data for the dashboard"
foreach ($d in "gold", "silver/dq_report", "output/metrics", "output/user_recs_top10", "output/similar_movies", "output/test_predictions_sample") {
    $dest = Join-Path $root "data\$($d -replace '/', '\')"
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    New-Item -ItemType Directory -Force (Split-Path $dest -Parent) | Out-Null
    gcloud storage cp -r "$gs/$d" (Split-Path $dest -Parent); Check
}

if (-not $KeepCluster) {
    Step "Deleting cluster"
    gcloud dataproc clusters delete $Cluster --region=$Region --quiet; Check
    gcloud compute routers delete movie-reco-router --region=$Region --quiet; Check
}
Write-Host "`nDone. Results in $gs and .\data" -ForegroundColor Green
