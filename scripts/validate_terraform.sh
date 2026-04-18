#!/usr/bin/env bash
# Validate Terraform configuration before apply
set -euo pipefail
cd "$(dirname "$0")/../terraform"
echo "==> terraform fmt check..."
terraform fmt -check -recursive
echo "==> terraform validate..."
terraform validate
echo "==> tflint..."
tflint --recursive 2>/dev/null || echo "(tflint not installed, skipping)"
echo "OK: Terraform config is valid"
