# Vision MCP tests: stub VLM + HF. integration tests need VLM_BASE_URL.

import pytest

from servers.vision import image_loader
from servers.vision.image_loader import DatasetSpec, register_dataset
from servers.vision.main import (
    _normalize_equipment,
    _parse_condition,
    _parse_defects,
    mcp,
)

from .conftest import call_tool, requires_vlm


SUBSTATION_TAXONOMY = image_loader.get_spec("substation").prompt_taxonomy


# Pure-function parsing helpers
class TestNormalizeEquipment:
    def test_exact_class(self):
        # does `test_exact_class`, split out so we can reuse it from a few call sites.
        assert _normalize_equipment(
            "transformer\nCooling fins visible.", SUBSTATION_TAXONOMY
        ) == "transformer"

    def test_prefixed_label(self):
        # `test_prefixed_label` lives here. was getting too cramped inline.
        assert _normalize_equipment(
            "Equipment: circuit breaker", SUBSTATION_TAXONOMY
        ) == "circuit breaker"

    def test_substring_match(self):
        assert _normalize_equipment(
            "This appears to be a busbar.", SUBSTATION_TAXONOMY
        ) == "busbar"

    def test_longest_match_wins(self):
        # 'current transformer' must beat 'transformer'.
        assert _normalize_equipment(
            "current transformer", SUBSTATION_TAXONOMY
        ) == "current transformer"

    def test_unknown_passes_through(self):
        # does `test_unknown_passes_through`, split out so we can reuse it from a few call sites.
        assert _normalize_equipment("solar panel", SUBSTATION_TAXONOMY) == "solar panel"

    def test_empty(self):
        # walk through `test_empty`, kept separate so the main flow stays readable.
        assert _normalize_equipment("", SUBSTATION_TAXONOMY) == "unknown"

    def test_open_set_no_taxonomy(self):
        # No taxonomy registered (e.g. unknown teammate dataset) -> free text.
        assert _normalize_equipment("rotor bar fault", ()) == "rotor bar fault"

    def test_all_substation_classes_self_match(self):
        # walk through `test_all_substation_classes_self_match`, kept separate so the main flow stays readable.
        for cls in SUBSTATION_TAXONOMY:
            assert _normalize_equipment(cls, SUBSTATION_TAXONOMY) == cls

    def test_motor_taxonomy(self):
        # Demonstrates teammate-domain plugability without code changes.
        motor_taxonomy = (
            "healthy", "inner race fault", "outer race fault",
            "ball fault", "rotor bar fault", "stator winding fault",
        )
        assert _normalize_equipment("inner race fault", motor_taxonomy) == "inner race fault"
        assert _normalize_equipment(
            "I think this is an outer race fault.", motor_taxonomy
        ) == "outer race fault"


class TestParseCondition:
    def test_well_formed(self):
        cond, conf = _parse_condition(
            "condition: degraded; confidence: high; reason: visible corrosion"
        )
        assert cond == "degraded"
        assert conf == "high"

    def test_loose_format(self):
        # walk through `test_loose_format`, kept separate so the main flow stays readable.
        cond, conf = _parse_condition("condition: GOOD\nconfidence: medium\nreason: ok")
        assert cond == "good"
        assert conf == "medium"

    def test_missing_fields(self):
        # does `test_missing_fields`, split out so we can reuse it from a few call sites.
        cond, conf = _parse_condition("the equipment is fine")
        assert cond == "unknown"
        assert conf == "low"


class TestParseDefects:
    def test_none(self):
        assert _parse_defects("none") == []

    def test_intact(self):
        # does `test_intact`, split out so we can reuse it from a few call sites.
        assert _parse_defects("intact") == []

    def test_list(self):
        # `test_list` lives here. was getting too cramped inline.
        assert _parse_defects("corrosion, oil leak, broken insulator") == [
            "corrosion", "oil leak", "broken insulator",
        ]

    def test_multiline_takes_first(self):
        out = _parse_defects("rust, dent\nadditional notes here")
        assert out == ["rust", "dent"]


# Dataset registry plug-and-play
class TestDatasetRegistry:
    def test_substation_registered(self):
        # `test_substation_registered` lives here. was getting too cramped inline.
        spec = image_loader.get_spec("substation")
        assert spec.repo == "AndrzejDD/15-class-Substation-Equipment"
        assert "substation" in spec.domain_hint

    def test_transformer_alias(self):
        # walk through `test_transformer_alias`, kept separate so the main flow stays readable.
        spec = image_loader.get_spec("transformer")
        assert spec.repo == "AndrzejDD/15-class-Substation-Equipment"

    def test_unknown_alias_raises(self):
        with pytest.raises(KeyError, match="Unknown dataset alias"):
            image_loader.get_spec("not_registered")

    def test_register_new_dataset(self):
        # Simulates a teammate plugging in their dataset.
        register_dataset(DatasetSpec(
            alias="motor_test",
            repo="example/motor",
            domain_hint="electric motor health",
            prompt_taxonomy=("healthy", "inner race fault"),
        ))
        spec = image_loader.get_spec("motor_test")
        assert spec.alias == "motor_test"
        assert "motor_test" in image_loader.list_aliases()

    def test_domain_hint_for_path_returns_none(self):
        # `test_domain_hint_for_path_returns_none` lives here. was getting too cramped inline.
        assert image_loader.domain_hint_for("/tmp/x.jpg") is None

    def test_taxonomy_for_path_returns_empty(self):
        assert image_loader.taxonomy_for("/tmp/x.jpg") == ()


# Tool integration with mocks
class TestAnalyzeImage:
    @pytest.mark.anyio
    async def test_free_form(self, fake_image, mock_vlm):
        mock_vlm.append("The motor shows clear signs of bearing wear.")
        out = await call_tool(
            mcp,
            "analyze_image",
            {
                "image_ref": "hf://transformer/0",
                "question": "Are there bearing-wear signs?",
            },
        )
        assert out["answer"].startswith("The motor shows")

    @pytest.mark.anyio
    async def test_vlm_unreachable(self, fake_image, vlm_error):
        # Async bit: does `test_vlm_unreachable`, split out so we can reuse it from a few call sites.
        out = await call_tool(
            mcp,
            "analyze_image",
            {"image_ref": "hf://transformer/0", "question": "?"},
        )
        assert "error" in out


class TestClassifyEquipment:
    @pytest.mark.anyio
    async def test_substation_taxonomy_normalizes(self, fake_image, mock_vlm):
        # Async bit: walk through `test_substation_taxonomy_normalizes`, kept separate so the main flow stays readable.
        mock_vlm.append("transformer\nVisible cooling radiators.")
        out = await call_tool(
            mcp, "classify_equipment", {"image_ref": "hf://transformer/train/0"}
        )
        assert out["equipment_type"] == "transformer"
        assert "substation" in out["domain"]

    @pytest.mark.anyio
    async def test_substring_normalization(self, fake_image, mock_vlm):
        mock_vlm.append("This is clearly a circuit breaker housing.")
        out = await call_tool(
            mcp, "classify_equipment", {"image_ref": "hf://transformer/0"}
        )
        assert out["equipment_type"] == "circuit breaker"

    @pytest.mark.anyio
    async def test_open_set_dataset(self, fake_image, mock_vlm, monkeypatch):
        # Register a teammate dataset with no taxonomy -> open-set normalization.
        register_dataset(DatasetSpec(
            alias="open_demo",
            repo="example/open",
            domain_hint="generic plant equipment",
        ))
        mock_vlm.append("hydraulic pump\nReason: visible piston rod.")
        out = await call_tool(
            mcp, "classify_equipment", {"image_ref": "hf://open_demo/0"}
        )
        assert out["equipment_type"] == "hydraulic pump"
        assert "plant equipment" in out["domain"]

    @pytest.mark.anyio
    async def test_vlm_unreachable(self, fake_image, vlm_error):
        # Async bit: does `test_vlm_unreachable`, split out so we can reuse it from a few call sites.
        out = await call_tool(
            mcp, "classify_equipment", {"image_ref": "hf://transformer/0"}
        )
        assert "error" in out
        assert "vlm_call_failed" in out["error"]

    @pytest.mark.anyio
    async def test_bad_image_ref(self, mock_vlm):
        # Async bit: does `test_bad_image_ref`, split out so we can reuse it from a few call sites.
        out = await call_tool(
            mcp, "classify_equipment", {"image_ref": "/nonexistent/path.jpg"}
        )
        assert "error" in out
        assert "image_load_failed" in out["error"]


class TestDetectVisualDefects:
    @pytest.mark.anyio
    async def test_no_defects(self, fake_image, mock_vlm):
        mock_vlm.append("none")
        out = await call_tool(
            mcp, "detect_visual_defects", {"image_ref": "hf://transformer/0"}
        )
        assert out["defects"] == []

    @pytest.mark.anyio
    async def test_some_defects(self, fake_image, mock_vlm):
        mock_vlm.append("corrosion, oil leak")
        out = await call_tool(
            mcp, "detect_visual_defects", {"image_ref": "hf://transformer/0"}
        )
        assert out["defects"] == ["corrosion", "oil leak"]


class TestAssessCondition:
    @pytest.mark.anyio
    async def test_well_formed(self, fake_image, mock_vlm):
        mock_vlm.append(
            "condition: critical; confidence: high; reason: severe arcing damage"
        )
        out = await call_tool(
            mcp, "assess_condition", {"image_ref": "hf://transformer/0"}
        )
        assert out["condition"] == "critical"
        assert out["confidence"] == "high"


class TestReadGauge:
    @pytest.mark.anyio
    async def test_reading(self, fake_image, mock_vlm):
        # Async bit: does `test_reading`, split out so we can reuse it from a few call sites.
        mock_vlm.append("42.5 kV")
        out = await call_tool(mcp, "read_gauge", {"image_ref": "hf://transformer/0"})
        assert out["reading"] == "42.5 kV"


# Live integration (requires VLM_BASE_URL)
@requires_vlm
@pytest.mark.anyio
async def test_classify_live():
    out = await call_tool(
        mcp, "classify_equipment", {"image_ref": "hf://transformer/train/0"}
    )
    assert "equipment_type" in out
