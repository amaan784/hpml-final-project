"""SHA256 manifest for a quantized checkpoint.

The quantization scripts in ``scripts/`` are seeded (see ``benchmark/seeding``),
but the GPTQ / SmoothQuant kernels still drift across hardware, driver, and
library versions. Recording the SHA256 of every weight shard lets us pin a
specific checkpoint as the canonical artifact and detect silent drift between
the quant run and the bench run.

Pair with ``scripts/verify_checkpoint.py`` (which validates *structure* --
config layout, packed-INT4 key naming for vLLM 0.19). Hash + structure
together cover both "is this the same artifact?" and "is it a usable
artifact?".

Usage:
  # After quantization: record the manifest in-place.
  python scripts/hash_checkpoint.py write models/qwen2.5-vl-7b-awq-domain

  # Before bench: verify the manifest matches reality.
  python scripts/hash_checkpoint.py verify models/qwen2.5-vl-7b-awq-domain
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

MANIFEST_NAME = "SHA256SUMS"

# Files we hash. Tokenizer JSONs are included so a swapped tokenizer is
# treated as a different artifact -- the chat template materially changes
# benchmark behaviour. "*.index.json" covers the sharded-checkpoint index
# (model.safetensors.index.json), which "*.safetensors" does NOT match;
# recipe.yaml is llmcompressor's quantization recipe; vocab.json/merges.txt
# are the Qwen BPE tokenizer files.
_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.index.json", "config.json",
             "tokenizer*.json", "tokenizer*.model", "special_tokens_map.json",
             "vocab.json", "merges.txt", "added_tokens.json", "recipe.yaml",
             "preprocessor_config.json", "processor_config.json",
             "chat_template.json", "generation_config.json")


def _iter_files(ckpt: Path) -> list[Path]:
    seen: set[Path] = set()
    for pat in _PATTERNS:
        for p in ckpt.glob(pat):
            if p.is_file() and p.name != MANIFEST_NAME:
                seen.add(p.resolve())
    return sorted(seen)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _write_manifest(ckpt: Path) -> int:
    files = _iter_files(ckpt)
    if not files:
        print(f"FAIL: no hashable files under {ckpt}", file=sys.stderr)
        return 1
    lines: list[str] = []
    for p in files:
        digest = _sha256(p)
        rel = p.relative_to(ckpt).as_posix()
        lines.append(f"{digest}  {rel}")
        print(f"  {digest[:12]}...  {rel}")
    (ckpt / MANIFEST_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"==> wrote {len(lines)} entries to {ckpt / MANIFEST_NAME}")
    return 0


def _verify_manifest(ckpt: Path) -> int:
    manifest = ckpt / MANIFEST_NAME
    if not manifest.exists():
        print(f"FAIL: {manifest} not present -- run `hash_checkpoint.py write` first",
              file=sys.stderr)
        return 2
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        if not digest or not rel:
            print(f"FAIL: malformed line in manifest: {line!r}", file=sys.stderr)
            return 2
        expected[rel] = digest

    mismatched: list[str] = []
    missing: list[str] = []
    for rel, want in expected.items():
        p = ckpt / rel
        if not p.exists():
            missing.append(rel)
            continue
        got = _sha256(p)
        marker = "OK  " if got == want else "BAD "
        print(f"  {marker}{got[:12]}...  {rel}")
        if got != want:
            mismatched.append(rel)

    extra = sorted(
        p.relative_to(ckpt).as_posix() for p in _iter_files(ckpt)
        if p.relative_to(ckpt).as_posix() not in expected
    )

    if missing or mismatched or extra:
        if missing:
            print(f"FAIL: {len(missing)} files missing: {missing[:5]}", file=sys.stderr)
        if mismatched:
            print(f"FAIL: {len(mismatched)} files mismatched: {mismatched[:5]}",
                  file=sys.stderr)
        if extra:
            print(f"WARN: {len(extra)} files not in manifest: {extra[:5]}",
                  file=sys.stderr)
        return 1 if (missing or mismatched) else 0
    print(f"==> OK ({len(expected)} files match manifest)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mode", choices=["write", "verify"])
    p.add_argument("ckpt_dir", help="Path to checkpoint directory")
    args = p.parse_args()
    ckpt = Path(args.ckpt_dir)
    if not ckpt.is_dir():
        print(f"FAIL: {ckpt} is not a directory", file=sys.stderr)
        return 2
    return _write_manifest(ckpt) if args.mode == "write" else _verify_manifest(ckpt)


if __name__ == "__main__":
    sys.exit(main())
