#!/usr/bin/env bash
# Download the released amyloid-plaque weights (114 MB, too large to commit to git)
# from the GitHub release and check the checksum.
set -euo pipefail
cd "$(dirname "$0")"
URL="https://github.com/Center-for-ComputationalNeuropathology/NeuroSpot-YOLO/releases/download/amyloid-plaque-v1.0/amyloid_plaque_yolo11x.pt"
curl -L --fail -o amyloid_plaque_yolo11x.pt "$URL"
sha256sum -c SHA256SUMS
