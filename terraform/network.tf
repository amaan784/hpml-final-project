# Use the default VPC - Columbia org policy already exists on it
data "google_compute_network" "default" {
  name = "default"

  depends_on = [google_project_service.apis]
}

# firewall

resource "google_compute_firewall" "allow_iap_ssh" {
  count      = var.create_iap_firewall ? 1 : 0
  depends_on = [google_project_service.apis]
  name       = "${var.vm_name}-allow-iap-ssh"
  network    = data.google_compute_network.default.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  # IAP uses this IP range to proxy SSH connections
  source_ranges = ["35.235.240.0/20"]
}

resource "google_compute_firewall" "allow_internal" {
  depends_on = [google_project_service.apis]
  name       = "${var.vm_name}-allow-internal"
  network    = data.google_compute_network.default.name

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = ["10.128.0.0/9"]
}

# Cloud Router + NAT for outbound internet (no external IP).
# Set create_nat = false if a NAT already exists (e.g. from HW1).

resource "google_compute_router" "router" {
  count      = var.create_nat ? 1 : 0
  depends_on = [google_project_service.apis]
  name       = "${var.vm_name}-router"
  network    = data.google_compute_network.default.name
  region     = var.region

  bgp {
    asn = 64514
  }
}

resource "google_compute_router_nat" "nat" {
  count                              = var.create_nat ? 1 : 0
  name                               = "${var.vm_name}-nat"
  router                             = google_compute_router.router[0].name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

# vLLM port, IAP source range only (no public exposure).
resource "google_compute_firewall" "allow_iap_vllm" {
  count      = var.gpu_count > 0 && var.create_iap_firewall ? 1 : 0
  depends_on = [google_project_service.apis]
  name       = "${var.vm_name}-allow-iap-vllm"
  network    = data.google_compute_network.default.name

  allow {
    protocol = "tcp"
    ports    = [tostring(var.vllm_port)]
  }

  source_ranges = ["35.235.240.0/20"]
  target_tags   = ["assetopsbench"]
}
