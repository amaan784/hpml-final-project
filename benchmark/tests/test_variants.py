# Tests for the optimization-variant registry.
# Locked to the 10-variant final sweep (5 Qwen + 5 Llama).
# Cut variants (W8A8, ablations, extra L3 resolutions) are parked in _unused/.

import pytest

from benchmark import variants


# 10 active variants: 5 per family
KNOWN_NAMES = {
    # Qwen track
    "L0_baseline",
    "L1_awq_w4a16_domain",
    "L1_awq_w4a16_generic",
    "L2_full_bundle",
    "L3_image_512",
    # Llama track
    "L0_llama_baseline",
    "L1_llama_awq_w4a16_domain",
    "L1_llama_awq_w4a16_generic",
    "L2_llama_full_bundle",
    "L3_llama_image_512",
}


class TestRegistry:
    def test_all_known_variants_register(self):
        # `test_all_known_variants_register` lives here so benchmark steps read top-down.
        registered = {v.name for v in variants.all_variants()}
        missing = KNOWN_NAMES - registered
        assert not missing, f"Missing variants: {missing}"

    def test_only_active_variants_registered(self):
        # `test_only_active_variants_registered` lives here so benchmark steps read top-down.
        registered = {v.name for v in variants.all_variants()}
        extra = registered - KNOWN_NAMES
        assert not extra, (
            f"Extra variants registered (move to _unused/variants/?): {extra}"
        )

    def test_get_unknown_raises(self):
        # `test_get_unknown_raises` lives here so benchmark steps read top-down.
        with pytest.raises(KeyError, match="Unknown variant"):
            variants.get("does_not_exist")

    def test_short_tags_unique(self):
        # `test_short_tags_unique` lives here so benchmark steps read top-down.
        shorts = [v.short for v in variants.all_variants()]
        assert len(shorts) == len(set(shorts)), \
            f"Duplicate short tags: {shorts}"

    def test_every_variant_has_docs(self):
        for v in variants.all_variants():
            assert v.description, f"{v.name}: missing description"
            assert v.how_it_works, f"{v.name}: missing how_it_works"
            assert v.howto_test, f"{v.name}: missing howto_test"

    def test_by_family(self):
        # Shared `test_by_family` logic reused by multiple benchmark paths.
        for family in ("L0", "L1", "L2", "L3"):
            items = variants.by_family(family)
            assert items, f"family {family} has no variants"
            assert all(v.family == family for v in items)


class TestPlaceholders:
    def test_resolves_known_placeholder(self):
        # default values when env unset (W8A8 placeholders dropped with cut variants)
        assert variants.resolve_model_id("$AWQ_DOMAIN").endswith("awq-domain")
        assert variants.resolve_model_id("$AWQ_GENERIC").endswith("awq-generic")
        assert variants.resolve_model_id("$AWQ_LLAMA_DOMAIN").endswith("awq-domain-real")
        assert variants.resolve_model_id("$AWQ_LLAMA_GENERIC").endswith("awq-generic-real")

    def test_passthrough_concrete_id(self):
        # Helper for `test_passthrough_concrete_id`.
        assert variants.resolve_model_id("Qwen/Qwen2.5-VL-7B-Instruct") == "Qwen/Qwen2.5-VL-7B-Instruct"

    def test_env_override(self, monkeypatch):
        # `test_env_override` lives here so benchmark steps read top-down.
        monkeypatch.setenv("AWQ_DOMAIN", "/custom/path")
        assert variants.resolve_model_id("$AWQ_DOMAIN") == "/custom/path"


class TestVariantShape:
    def test_l0_is_fp16(self):
        # CLI/helper entry for `test_l0_is_fp16`.
        v = variants.get("L0_baseline")
        assert v.vllm_extra == ()
        assert v.image_max_side == 1024
        assert "Qwen/Qwen2.5-VL-7B-Instruct" in v.model_id

    def test_l1_w4a16_domain_uses_placeholder(self):
        # Helper for `test_l1_w4a16_domain_uses_placeholder`.
        v = variants.get("L1_awq_w4a16_domain")
        assert v.model_id == "$AWQ_DOMAIN"

    def test_l1_w4a16_generic_uses_placeholder(self):
        v = variants.get("L1_awq_w4a16_generic")
        assert v.model_id == "$AWQ_GENERIC"

    def test_l1_llama_w4a16_generic_uses_placeholder(self):
        v = variants.get("L1_llama_awq_w4a16_generic")
        assert v.model_id == "$AWQ_LLAMA_GENERIC"

    def test_l2_full_bundle_includes_all_flags(self):
        # Helper for `test_l2_full_bundle_includes_all_flags`.
        v = variants.get("L2_full_bundle")
        flags = " ".join(v.vllm_extra)
        # process records in deterministic order
        for flag in ("--enable-prefix-caching", "--enable-chunked-prefill",
                     "--kv-cache-dtype", "--gpu-memory-utilization"):
            assert flag in flags, f"L2 bundle missing {flag}"

    def test_llama_l2_full_bundle_includes_all_flags(self):
        v = variants.get("L2_llama_full_bundle")
        flags = " ".join(v.vllm_extra)
        # each pass handles the next item in the sequence
        for flag in ("--enable-prefix-caching", "--enable-chunked-prefill",
                     "--kv-cache-dtype", "--gpu-memory-utilization"):
            assert flag in flags, f"L2_llama_full_bundle missing {flag}"

    def test_l3_only_512_kept(self):
        # only 512 is kept in the active set (768/1024/1536 parked in _unused)
        sides = {v.image_max_side for v in variants.by_family("L3")}
        assert sides == {512}

    def test_per_family_variant_count(self):
        qwen = [v for v in variants.all_variants() if "llama" not in v.name]
        llama = [v for v in variants.all_variants() if "llama" in v.name]
        assert len(qwen) == 5, f"Expected 5 Qwen variants, got {len(qwen)}"
        assert len(llama) == 5, f"Expected 5 Llama variants, got {len(llama)}"
