locals {
  account_flag = var.gcp_account != "" ? " --account=${var.gcp_account}" : ""
}

output "vm_name" {
  description = "GCE instance name"
  value       = google_compute_instance.vm.name
}

output "vm_zone" {
  description = "Zone the VM is running in"
  value       = google_compute_instance.vm.zone
}

output "ssh_command" {
  description = "SSH into the VM via IAP tunnel (no external IP)"
  value       = "gcloud compute ssh ${google_compute_instance.vm.name} --tunnel-through-iap --project=${var.project_id} --zone=${var.zone}${local.account_flag}"
}

output "vllm_tunnel_command" {
  description = "Open IAP SSH session that also forwards localhost:vllm_port -> vm:vllm_port"
  value       = "gcloud compute ssh ${google_compute_instance.vm.name} --tunnel-through-iap --project=${var.project_id} --zone=${var.zone}${local.account_flag} -- -L ${var.vllm_port}:localhost:${var.vllm_port} -N"
}

output "ssh_config_block" {
  description = "Add to ~/.ssh/config for VS Code Remote-SSH (replace YOUR_USERNAME with your gcloud OS Login username)"
  value       = <<-EOT

Host ${google_compute_instance.vm.name}
    HostName ${google_compute_instance.vm.name}
    IdentityFile ~/.ssh/google_compute_engine
    User YOUR_USERNAME
    ProxyCommand gcloud compute start-iap-tunnel %h %p --listen-on-stdin --project=${var.project_id} --zone=${var.zone}${local.account_flag}
  EOT
}

output "results_bucket" {
  description = "GCS bucket for benchmark results"
  value       = google_storage_bucket.results.name
}

output "quickstart" {
  description = "Quick-start instructions"
  value       = <<-EOT

AssetOpsBench is provisioned!

1. SSH in (via IAP tunnel, no external IP):
   gcloud compute ssh ${google_compute_instance.vm.name} --tunnel-through-iap --project=${var.project_id} --zone=${var.zone}${local.account_flag}

2. Wait for startup to finish (first boot ~3-5 min):
   sudo journalctl -f -u google-startup-scripts

3. Run the agent:
   cd /opt/assetopsbench
   uv run plan-execute "What assets are at site MAIN?"

4. Upload results to GCS:
   aob-upload-results

  EOT
}
