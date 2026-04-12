locals {
  secret_values = {
    couchdb_password   = var.couchdb_password
    watsonx_apikey     = var.watsonx_apikey
    watsonx_project_id = var.watsonx_project_id
    watsonx_url        = var.watsonx_url
    litellm_api_key    = var.litellm_api_key
    litellm_base_url   = var.litellm_base_url
  }

  # Non-sensitive set of keys where the value is non-empty.
  # nonsensitive() is safe here: we only reveal *which* keys are provided, not their values.
  active_secret_keys = toset([
    for k, v in local.secret_values : nonsensitive(k)
    if nonsensitive(v != "")
  ])
}

resource "google_secret_manager_secret" "secrets" {
  for_each  = local.active_secret_keys
  secret_id = "${var.vm_name}-${replace(each.key, "_", "-")}"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "versions" {
  for_each    = local.active_secret_keys
  secret      = google_secret_manager_secret.secrets[each.key].id
  secret_data = local.secret_values[each.key]
}

resource "google_secret_manager_secret_iam_member" "vm_accessor" {
  for_each  = local.active_secret_keys
  secret_id = google_secret_manager_secret.secrets[each.key].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.vm.email}"
}
