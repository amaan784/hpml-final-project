resource "google_service_account" "vm" {
  account_id   = "${var.vm_name}-sa"
  display_name = "AssetOpsBench VM Service Account"

  depends_on = [google_project_service.apis]
}

resource "google_project_iam_member" "vm_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.vm.email}"
}

resource "google_project_iam_member" "vm_metric_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.vm.email}"
}

resource "google_compute_instance" "vm" {
  name         = var.vm_name
  machine_type = var.machine_type
  zone         = var.zone

  tags = ["assetopsbench"]

  boot_disk {
    initialize_params {
      image = var.use_dl_image ? "deeplearning-platform-release/common-cu129-ubuntu-2204-nvidia-580" : "ubuntu-os-cloud/ubuntu-2404-lts-amd64"
      size  = var.boot_disk_size_gb
      type  = "pd-ssd"
    }
  }

  network_interface {
    network = data.google_compute_network.default.name
    # No access_config = no external IP (Columbia org policy compatible)
  }

  service_account {
    email  = google_service_account.vm.email
    scopes = ["cloud-platform"]
  }

  # GPU attachment (HPML: vLLM serving). Skipped entirely when gpu_count == 0.
  dynamic "guest_accelerator" {
    for_each = var.gpu_count > 0 ? [1] : []
    content {
      type  = var.gpu_type
      count = var.gpu_count
    }
  }

  scheduling {
    preemptible       = var.preemptible
    automatic_restart = var.preemptible ? false : true
    # GPU VMs cannot be live-migrated. must terminate on host maintenance.
    on_host_maintenance = var.gpu_count > 0 ? "TERMINATE" : "MIGRATE"
  }

  metadata_startup_script = templatefile("${path.module}/startup.sh", {
    project_id         = var.project_id
    vm_name            = var.vm_name
    bucket_name        = local.bucket_name
    active_secret_keys = local.active_secret_keys
    install_gpu_stack  = var.gpu_count > 0
    vllm_port          = var.vllm_port
  })

  metadata = {
    enable-oslogin = "TRUE"
  }

  depends_on = [
    google_project_service.apis,
    google_secret_manager_secret_version.versions,
  ]
}
