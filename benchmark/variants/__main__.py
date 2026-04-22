# CLI for the variant registry.
# Usage:
#   python -m benchmark.variants list [--family L2]
#   python -m benchmark.variants show <name>
#   python -m benchmark.variants serve <name>   # prints bash command
#   python -m benchmark.variants env <name>     # prints export lines

import argparse
import shlex
import sys

from . import REGISTRY, all_variants, by_family, get, resolve_model_id


def _print_list(family):
    # CLI/helper entry for `print_list`.
    items = by_family(family) if family else all_variants()
    width = max(len(v.name) for v in items)
    for v in items:
        print(f"{v.name:<{width}}  [{v.short:<8}]  {v.description}")


def _print_show(name):
    v = get(name)
    print(f"name:         {v.name}")
    print(f"short:        {v.short}")
    print(f"family:       {v.family}")
    print(f"description:  {v.description}")
    print(f"model_id:     {v.model_id}  -> {resolve_model_id(v.model_id)}")
    print(f"vllm_extra:   {' '.join(v.vllm_extra) if v.vllm_extra else '(none)'}")
    print(f"image_max:    {v.image_max_side}")

    # runs when v.env
    if v.env:
        print("env:")
        # each pass handles the next item in the sequence
        for k, val in v.env.items():
            print(f"  {k}={val}")
    print("requires:")
    for r in v.requires:
        print(f"  - {r}")
    print("\n# How it works\n")
    print(v.how_it_works)
    print("\n# How to test\n")
    print(v.howto_test)


def _print_serve(name):
    v = get(name)
    extra = " ".join(shlex.quote(x) for x in v.vllm_extra)
    model = resolve_model_id(v.model_id)
    print(f"MODEL={shlex.quote(model)} EXTRA={shlex.quote(extra)} bash scripts/serve_vllm.sh")


def _print_env(name):
    # emit `export FOO=bar` lines so callers can `eval $(...)`
    v = get(name)
    print(f"export VLM_MODEL={shlex.quote(resolve_model_id(v.model_id))}")
    print(f"export VLM_IMAGE_MAX_SIDE={v.image_max_side}")
    for k, val in v.env.items():
        print(f"export {k}={shlex.quote(val)}")


def main():
    # CLI/helper entry for `main`.
    p = argparse.ArgumentParser(prog="python -m benchmark.variants")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="List all registered variants.")
    pl.add_argument("--family", help="Filter by L0 / L1 / L2 / L3 / ...")

    ps = sub.add_parser("show", help="Detailed variant info (how it works + howto test).")
    ps.add_argument("name")

    pv = sub.add_parser("serve", help="Print the bash command to start vLLM for this variant.")
    pv.add_argument("name")

    pe = sub.add_parser("env", help="Print export lines for VLM_MODEL/VLM_IMAGE_MAX_SIDE/...")
    pe.add_argument("name")

    args = p.parse_args()

    # runs when args.cmd == 'list'
    if args.cmd == "list":
        _print_list(args.family)
    elif args.cmd == "show":
        _print_show(args.name)
    elif args.cmd == "serve":
        _print_serve(args.name)
    elif args.cmd == "env":
        _print_env(args.name)
    return 0

if __name__ == "__main__":
    sys.exit(main())

