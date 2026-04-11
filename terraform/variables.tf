variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "gcp_account" {
  description = "GCP account email to use for gcloud commands (e.g. yc4670@columbia.edu)"
  type        = string
  default     = ""
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "machine_type" {
  description = "GCE machine type (e2-standard-4 = 4 vCPU / 16 GB)"
  type        = string
  default     = "e2-standard-4"
}

variable "boot_disk_size_gb" {
  description = "Boot disk size in GB"
  type        = number
  default     = 50
}

variable "vm_name" {
  description = "Name for the GCE instance"
  type        = string
  default     = "assetopsbench"
}

# networking

variable "create_nat" {
  description = "Create a new Cloud Router + NAT. Set false if one already exists (e.g. from HW1)."
  type        = bool
  default     = true
}

variable "create_iap_firewall" {
  description = "Create IAP SSH firewall rule. Set false if one already exists."
  type        = bool
  default     = true
}

# secrets (LLM backends)

variable "couchdb_password" {
  description = "CouchDB admin password"
  type        = string
  default     = "password"
  sensitive   = true
}

variable "watsonx_apikey" {
  description = "IBM WatsonX API key (leave empty to skip)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "watsonx_project_id" {
  description = "IBM WatsonX project ID"
  type        = string
  default     = ""
}

variable "watsonx_url" {
  description = "IBM WatsonX endpoint URL"
  type        = string
  default     = "https://us-south.ml.cloud.ibm.com"
}

variable "litellm_api_key" {
  description = "LiteLLM proxy API key (OpenAI / Anthropic / etc.)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "litellm_base_url" {
  description = "LiteLLM proxy base URL"
  type        = string
  default     = ""
}

# storage

variable "results_bucket_name" {
  description = "GCS bucket name for benchmark results (auto-generated if empty)"
  type        = string
  default     = ""
}

variable "results_retention_days" {
  description = "Auto-delete objects in the results bucket after this many days (0 = disabled)"
  type        = number
  default     = 90
}

# preemptibility

variable "preemptible" {
  description = "Use a preemptible (spot) VM to save ~70% cost"
  type        = bool
  default     = false
}

# GPU (vLLM serving)

variable "gpu_type" {
  description = "GPU model attached to the VM. Empty = no GPU. Examples: 'nvidia-l4', 'nvidia-l40s'."
  type        = string
  default     = ""
}

variable "gpu_count" {
  description = "Number of accelerators of gpu_type to attach. 0 = no GPU."
  type        = number
  default     = 0
}

variable "vllm_port" {
  description = "TCP port vLLM listens on inside the VM (also opened to IAP source range)."
  type        = number
  default     = 8000
}

variable "use_dl_image" {
  description = "Use the GCP Deep Learning VM image (CUDA 12.4 preinstalled) instead of plain Ubuntu 24.04."
  type        = bool
  default     = false
}
