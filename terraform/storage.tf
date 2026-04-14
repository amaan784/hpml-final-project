locals {
  bucket_name = var.results_bucket_name != "" ? var.results_bucket_name : "${var.project_id}-${var.vm_name}-results"
}

resource "google_storage_bucket" "results" {
  name     = local.bucket_name
  location = var.region

  uniform_bucket_level_access = true
  force_destroy               = true

  dynamic "lifecycle_rule" {
    for_each = var.results_retention_days > 0 ? [1] : []
    content {
      action {
        type = "Delete"
      }
      condition {
        age = var.results_retention_days
      }
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_storage_bucket_iam_member" "vm_writer" {
  bucket = google_storage_bucket.results.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.vm.email}"
}
