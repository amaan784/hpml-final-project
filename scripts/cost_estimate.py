#!/usr/bin/env python3
# Rough GCP cost estimate for a full benchmark sweep.

import argparse

# rough GCP $/hr numbers for back-of-napkin estimates
PRICES = {
    "a2-highgpu-1g": 3.67,   # A100 40GB, $/hr
    "a2-highgpu-2g": 7.35,
    "a2-megagpu-16g": 58.72, # 16xA100
    "n1-standard-8": 0.38,
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--machine-type", default="a2-highgpu-1g")
    p.add_argument("--hours", type=float, default=24.0)
    p.add_argument("--variants", type=int, default=10)
    args = p.parse_args()
    rate = PRICES.get(args.machine_type, 1.0)
    cost = rate * args.hours
    print(f"Estimated cost: ${cost:.2f} for {args.hours}h on {args.machine_type}")
    print(f"Per variant: ${cost/args.variants:.2f}")
if __name__ == "__main__":
    main()

