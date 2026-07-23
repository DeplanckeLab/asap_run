#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

if ! docker image inspect fabdavid/asap_run:v6.1 >/dev/null 2>&1; then
  echo "Missing base image fabdavid/asap_run:v6.1"
  echo "Pull it first: docker pull fabdavid/asap_run:v6.1"
  exit 1
fi

echo "Building v6 from v6.1 package base + latest scripts..."
docker build -f Dockerfile.v6.2 -t fabdavid/asap_run:v6 .
docker tag fabdavid/asap_run:v6 fabdavid/asap_run:v6.2

echo "Verifying Seurat version..."
docker run --rm fabdavid/asap_run:v6 Rscript -e 'cat("Seurat:", as.character(packageVersion("Seurat")), "\n"); cat("SeuratObject:", as.character(packageVersion("SeuratObject")), "\n")'

echo "Done. Tagged fabdavid/asap_run:v6 and fabdavid/asap_run:v6.2"
