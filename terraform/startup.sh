#!/usr/bin/env bash
set -euo pipefail

MARKER="/var/log/assetopsbench-setup-done"
if [ -f "$MARKER" ]; then
  echo "Setup already completed, skipping."
  exit 0
fi

export DEBIAN_FRONTEND=noninteractive
WORK_DIR="/opt/assetopsbench"

echo "==> Installing base packages..."
apt-get update -qq
apt-get install -y -qq curl git jq apt-transport-https ca-certificates gnupg lsb-release

# Docker CE
echo "==> Installing Docker CE..."
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin

systemctl enable --now docker

# Let non-root users use docker
for u in $(ls /home); do
  usermod -aG docker "$u" 2>/dev/null || true
done

# python 3.12 + uv
echo "==> Installing pip + uv..."
apt-get install -y -qq python3-pip
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

# Make uv available for all users
cp /root/.local/bin/uv /usr/local/bin/uv || true
cp /root/.local/bin/uvx /usr/local/bin/uvx 2>/dev/null || true

# clone repo
echo "==> Cloning AssetOpsBench..."
git clone https://github.com/IBM/AssetOpsBench.git "$WORK_DIR"
cd "$WORK_DIR"

# fetch secrets from Secret Manager into .env
echo "==> Building .env from Secret Manager..."
cp .env.public .env

fetch_secret() {
  local secret_id="$1"
  gcloud secrets versions access latest --secret="$secret_id" --project="${project_id}" 2>/dev/null || echo ""
}

%{ for key in active_secret_keys ~}
SECRET_VAL=$(fetch_secret "${vm_name}-${replace(key, "_", "-")}")
if [ -n "$SECRET_VAL" ]; then
  sed -i "s|^${upper(key)}=.*|${upper(key)}=$SECRET_VAL|" .env
fi
%{ endfor ~}

chmod 600 .env

# start CouchDB
echo "==> Starting CouchDB via Docker Compose..."
docker compose -f src/couchdb/docker-compose.yaml up -d

echo "==> Waiting for CouchDB to be healthy..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:5984/ >/dev/null 2>&1; then
    echo "CouchDB is ready."
    break
  fi
  sleep 5
done

# install python deps

echo "==> Running uv sync..."
cd "$WORK_DIR"
uv sync

echo "==> Verifying plan-execute CLI..."
uv run plan-execute --help || echo "warning: plan-execute --help failed, check logs."

# GCS helper alias
cat >> /etc/profile.d/assetopsbench.sh <<'PROFILE'
export ASSETOPSBENCH_DIR="/opt/assetopsbench"
alias aob-cd='cd /opt/assetopsbench'
alias aob-upload-results='gsutil -m cp -r /opt/assetopsbench/benchmark/cods_track*/track*_result/ gs://${bucket_name}/'
# Put the project's uv venv on PATH so huggingface-cli, wandb, pytest, vllm etc.
# are directly callable without `uv run` (which would re-sync the venv).
case ":$PATH:" in
  *":/opt/assetopsbench/.venv/bin:"*) ;;
  *) export PATH="/opt/assetopsbench/.venv/bin:$PATH" ;;
esac
PROFILE

# permissions (OS Login users are non-root)
chmod -R a+rwX "$WORK_DIR"

# GPU stack (vLLM + llm-compressor), only when GPU is attached
%{ if install_gpu_stack ~}
echo "==> GPU detected (install_gpu_stack=true); preparing vLLM..."

# Verify NVIDIA driver. The Deep Learning VM image ships drivers + CUDA.
# plain Ubuntu needs the install-driver script (best-effort, may need a reboot).
if ! command -v nvidia-smi >/dev/null 2>&1; then
  if [ -x /opt/deeplearning/install-driver.sh ]; then
    /opt/deeplearning/install-driver.sh || echo "WARN: driver install failed; install manually."
  else
    echo "WARN: nvidia-smi missing and no DL VM helper. Install drivers manually before running vLLM."
  fi
fi
nvidia-smi || true

echo "==> Installing vLLM 0.19.0 + llmcompressor 0.10.0.2 into the project venv..."
cd "$WORK_DIR"
# uv will auto-fetch Python 3.12 and create .venv on the first `uv sync`, which
# already ran above. `uv pip install` targets that venv so vllm + friends share
# the same Python runtime as the MCP servers.
/usr/local/bin/uv pip install \
    --torch-backend=cu129 \
    "vllm==0.19.0" \
    "llmcompressor==0.10.0.2" \
    "compressed-tensors==0.14.0.1" \
    "transformers==4.57.6" \
    "accelerate" \
    "datasets>=2.14" \
    "huggingface-hub>=0.25" \
    "wandb" \
    "openai>=1.40" \
    "pillow>=10.0" || echo "WARN: GPU pip install failed; finish manually inside SSH."

# Convenience: expose vLLM port-forward command.
cat >> /etc/profile.d/assetopsbench.sh <<'PROFILE'
alias vllm-serve='bash /opt/assetopsbench/scripts/serve_vllm.sh'
export VLM_BASE_URL="http://localhost:${vllm_port}/v1"
PROFILE
%{ endif ~}

# done
touch "$MARKER"
echo "==> AssetOpsBench setup complete!"
echo "==> SSH in and run: cd /opt/assetopsbench && uv run plan-execute '...'"
