#!/usr/bin/env bash
# One-time GPU setup for G4 (RTX PRO 6000 Blackwell) on Debian 12 GCP VM.
set -euo pipefail

echo "==> [1/3] Install kernel headers"
sudo apt-get update -qq
sudo apt-get install -y -qq linux-headers-$(uname -r) dkms build-essential

echo "==> [2/3] Install NVIDIA R580 GRID driver (required for G4 / Blackwell)"
cd /tmp
gcloud storage cp gs://gce-nvidia-gpu-drivers/NVIDIA-Linux-x86_64-580.126.09-grid-gcp.run .
chmod +x NVIDIA-Linux-x86_64-580.126.09-grid-gcp.run
sudo ./NVIDIA-Linux-x86_64-580.126.09-grid-gcp.run -s --no-cc-version-check

echo "==> [3/3] Verify GPU"
nvidia-smi

echo ""
echo "Next: upgrade PyTorch for Blackwell (sm_120):"
echo "  source .venv/bin/activate"
echo "  pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128"
