#!/usr/bin/env bash
# Spin up / tear down the course L4 VM (IAP SSH). Python+vLLM you install after SSH.

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-high-perf-ml-487201}"
# Space-separated fallback list. create tries each zone in order until one
# has capacity (L4 + g2-standard-8 frequently hit ZONE_RESOURCE_POOL_EXHAUSTED).
# Other subcommands ignore this and discover the VM's actual zone via gcloud.
ZONES="${ZONES:-us-central1-a us-central1-b us-central1-c}"
VM_NAME="${VM_NAME:-benchmarking}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
GPU_TYPE="${GPU_TYPE:-nvidia-l4}"
GPU_COUNT="${GPU_COUNT:-1}"
BOOT_DISK_GB="${BOOT_DISK_GB:-200}"
VLLM_PORT="${VLLM_PORT:-8000}"
PREEMPTIBLE="${PREEMPTIBLE:-0}"

# CUDA 12.9 + Ubuntu 22.04 + NVIDIA driver 580. Saves ~30 min of driver install.
# Family resolves to the latest versioned image (e.g. *-v20260430).
IMAGE_FAMILY_PROJECT="deeplearning-platform-release"
IMAGE_FAMILY="common-cu129-ubuntu-2204-nvidia-580"

NETWORK="default"
IAP_SOURCE="35.235.240.0/20"
SSH_FW="${VM_NAME}-allow-iap-ssh"
VLLM_FW="${VM_NAME}-allow-iap-vllm"

cmd="${1:-create}"

require_gcloud() {
    command -v gcloud >/dev/null 2>&1 || { echo "ERROR: gcloud CLI not found"; exit 1; }
    gcloud config set project "$PROJECT_ID" >/dev/null
}

# Find which zone the VM currently lives in (for ssh / stop / destroy).
discover_zone() {
    local z
    z=$(gcloud compute instances list \
            --filter="name=$VM_NAME" \
            --format="value(zone.basename())" \
            --project "$PROJECT_ID" 2>/dev/null | head -1)
    if [ -z "$z" ]; then
        echo "ERROR: VM '$VM_NAME' not found in project $PROJECT_ID" >&2
        return 1
    fi
    echo "$z"
}

ensure_apis() {
    echo "==> Enabling required APIs (idempotent)"
    gcloud services enable compute.googleapis.com iap.googleapis.com --project "$PROJECT_ID"
}

ensure_firewall() {
    if ! gcloud compute firewall-rules describe "$SSH_FW" --project "$PROJECT_ID" >/dev/null 2>&1; then
        echo "==> Creating IAP SSH firewall ($SSH_FW)"
        gcloud compute firewall-rules create "$SSH_FW" \
            --project "$PROJECT_ID" \
            --network "$NETWORK" \
            --direction INGRESS \
            --action ALLOW \
            --rules tcp:22 \
            --source-ranges "$IAP_SOURCE"
    else
        echo "==> SSH firewall already exists, skipping"
    fi

    if ! gcloud compute firewall-rules describe "$VLLM_FW" --project "$PROJECT_ID" >/dev/null 2>&1; then
        echo "==> Creating IAP vLLM firewall ($VLLM_FW, port $VLLM_PORT, target tag 'assetopsbench')"
        gcloud compute firewall-rules create "$VLLM_FW" \
            --project "$PROJECT_ID" \
            --network "$NETWORK" \
            --direction INGRESS \
            --action ALLOW \
            --rules "tcp:$VLLM_PORT" \
            --source-ranges "$IAP_SOURCE" \
            --target-tags assetopsbench
    else
        echo "==> vLLM firewall already exists, skipping"
    fi
}

create_vm() {
    local existing_zone
    existing_zone=$(gcloud compute instances list \
            --filter="name=$VM_NAME" \
            --format="value(zone.basename())" \
            --project "$PROJECT_ID" 2>/dev/null | head -1)
    if [ -n "$existing_zone" ]; then
        echo "==> VM '$VM_NAME' already exists in $existing_zone; nothing to create"
        return 0
    fi

    local sched_args=(
        "--maintenance-policy=TERMINATE"   # GPU VMs cannot live-migrate
        "--no-restart-on-failure"
    )
    if [ "$PREEMPTIBLE" = "1" ]; then
        sched_args+=("--provisioning-model=SPOT" "--instance-termination-action=STOP")
    else
        sched_args+=("--provisioning-model=STANDARD")
    fi

    local z
    for z in $ZONES; do
        echo "==> Trying $z: $MACHINE_TYPE + ${GPU_COUNT}x $GPU_TYPE"
        if gcloud compute instances create "$VM_NAME" \
            --project "$PROJECT_ID" \
            --zone "$z" \
            --machine-type "$MACHINE_TYPE" \
            --image-project "$IMAGE_FAMILY_PROJECT" \
            --image-family "$IMAGE_FAMILY" \
            --boot-disk-size "${BOOT_DISK_GB}GB" \
            --boot-disk-type pd-ssd \
            --accelerator "type=$GPU_TYPE,count=$GPU_COUNT" \
            --metadata enable-oslogin=TRUE,install-nvidia-driver=True \
            --network "$NETWORK" \
            --no-address \
            --tags assetopsbench \
            --scopes cloud-platform \
            "${sched_args[@]}" 2>&1 | tee /tmp/setup_vm_$$.log; then
            rm -f /tmp/setup_vm_$$.log
            CREATED_ZONE="$z"
            return 0
        fi
        # If it failed for a non-capacity reason, stop trying other zones.
        if ! grep -q "ZONE_RESOURCE_POOL_EXHAUSTED" /tmp/setup_vm_$$.log; then
            rm -f /tmp/setup_vm_$$.log
            echo "==> Failure was not a capacity issue; not trying other zones."
            return 1
        fi
        rm -f /tmp/setup_vm_$$.log
        echo "==> $z exhausted, trying next zone..."
    done
    echo "ERROR: All zones in '$ZONES' returned ZONE_RESOURCE_POOL_EXHAUSTED."
    echo "       Try again later, set PREEMPTIBLE=1 (often more capacity), or"
    echo "       widen ZONES (e.g. ZONES='us-central1-a us-east1-c us-west1-b')."
    return 1
}

case "$cmd" in
    create)
        require_gcloud
        ensure_apis
        ensure_firewall
        CREATED_ZONE=""
        create_vm
        echo
        echo "==================================================================="
        echo "  VM '$VM_NAME' is up in zone $CREATED_ZONE."
        echo "  Driver install runs on first boot (~3-5 min)."
        echo "==================================================================="
        echo "  SSH:        bash $0 ssh"
        echo "  vLLM tunnel: bash $0 tunnel    (forward localhost:$VLLM_PORT)"
        echo "  Stop (cheap): bash $0 stop"
        echo "  Destroy:    bash $0 destroy"
        echo
        echo "  Verify GPU once SSH'd in:  nvidia-smi"
        ;;
    ssh)
        require_gcloud
        z=$(discover_zone)
        exec gcloud compute ssh "$VM_NAME" \
            --zone "$z" --project "$PROJECT_ID" --tunnel-through-iap
        ;;
    tunnel)
        require_gcloud
        z=$(discover_zone)
        echo "==> Forwarding localhost:$VLLM_PORT -> $VM_NAME:$VLLM_PORT (zone $z, Ctrl-C to stop)"
        exec gcloud compute start-iap-tunnel "$VM_NAME" "$VLLM_PORT" \
            --local-host-port="localhost:$VLLM_PORT" \
            --zone "$z" --project "$PROJECT_ID"
        ;;
    stop)
        require_gcloud
        z=$(discover_zone)
        echo "==> Stopping $VM_NAME in $z (boot disk preserved)"
        gcloud compute instances stop "$VM_NAME" --zone "$z" --project "$PROJECT_ID"
        ;;
    start)
        require_gcloud
        z=$(discover_zone)
        echo "==> Starting $VM_NAME in $z"
        gcloud compute instances start "$VM_NAME" --zone "$z" --project "$PROJECT_ID"
        ;;
    destroy)
        require_gcloud
        z=$(discover_zone)
        echo "==> Deleting $VM_NAME in $z (boot disk will be removed)"
        gcloud compute instances delete "$VM_NAME" --zone "$z" --project "$PROJECT_ID" --quiet
        ;;
    *)
        echo "Usage: $0 {create|ssh|tunnel|stop|start|destroy}"
        exit 2
        ;;
esac
