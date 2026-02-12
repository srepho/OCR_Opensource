"""Smoke tests for adapter registry: validate import paths and class structure."""

import importlib

import pytest
import yaml


@pytest.fixture(scope="module")
def model_registry():
    with open("config/model_registry.yaml") as f:
        return yaml.safe_load(f)


class TestAdapterRegistry:
    """Validate that all adapter classes can be imported from the registry."""

    def test_all_adapters_importable(self, model_registry):
        """Every adapter_class in model_registry.yaml must be importable."""
        errors = []
        for model_key, info in model_registry["models"].items():
            adapter_path = info["adapter_class"]
            parts = adapter_path.rsplit(".", 1)
            module_path = parts[0]
            class_name = parts[1]

            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                errors.append(f"{model_key}: {adapter_path} -> {e}")

        assert not errors, "Failed to import adapters:\n" + "\n".join(errors)

    def test_all_adapters_have_required_methods(self, model_registry):
        """All adapters must implement load_model and ocr_page."""
        for model_key, info in model_registry["models"].items():
            adapter_path = info["adapter_class"]
            parts = adapter_path.rsplit(".", 1)
            module_path = parts[0]
            class_name = parts[1]

            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)

            assert hasattr(cls, "load_model"), f"{model_key}: missing load_model"
            assert hasattr(cls, "ocr_page"), f"{model_key}: missing ocr_page"
            assert hasattr(cls, "unload_model"), f"{model_key}: missing unload_model"
            assert hasattr(cls, "ocr_page_profiled"), f"{model_key}: missing ocr_page_profiled"

    def test_all_adapters_instantiable(self, model_registry):
        """All adapters must be instantiable with default args."""
        for model_key, info in model_registry["models"].items():
            adapter_path = info["adapter_class"]
            parts = adapter_path.rsplit(".", 1)
            module_path = parts[0]
            class_name = parts[1]

            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)

            instance = cls()
            assert not instance.is_loaded
            assert instance.model_name

    def test_registry_required_fields(self, model_registry):
        """All registry entries must have required metadata."""
        required_fields = {"name", "adapter_class", "tier", "vram_gb", "api_pattern", "output_format"}
        for model_key, info in model_registry["models"].items():
            missing = required_fields - set(info.keys())
            assert not missing, f"{model_key}: missing fields {missing}"

    def test_registry_tier_values(self, model_registry):
        """Tier values must be valid."""
        valid_tiers = {1, 2, "cpu"}
        for model_key, info in model_registry["models"].items():
            assert info["tier"] in valid_tiers, (
                f"{model_key}: invalid tier '{info['tier']}', must be one of {valid_tiers}"
            )

    def test_no_duplicate_adapter_classes(self, model_registry):
        """No two models should share the same adapter class."""
        classes = [info["adapter_class"] for info in model_registry["models"].values()]
        assert len(classes) == len(set(classes)), "Duplicate adapter classes found"


class TestBaseAdapter:
    """Test the base adapter ABC and helpers."""

    def test_ocr_result_defaults(self):
        from src.adapters.base import OCRResult
        result = OCRResult(text="hello")
        assert result.text == "hello"
        assert result.tables == []
        assert result.format == "text"
        assert result.text_length() == 5
        assert not result.has_tables()

    def test_extract_tables_from_markdown(self):
        from src.adapters.base import extract_tables_from_markdown
        md = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
        tables = extract_tables_from_markdown(md)
        assert len(tables) == 1
        assert "<table>" in tables[0]
        assert "<th>A</th>" in tables[0]

    def test_extract_tables_from_html(self):
        from src.adapters.base import extract_tables_from_html
        html = "text <table><tr><td>x</td></tr></table> more text"
        tables = extract_tables_from_html(html)
        assert len(tables) == 1

    def test_extract_no_tables(self):
        from src.adapters.base import extract_tables_from_markdown, extract_tables_from_html
        assert extract_tables_from_markdown("just plain text") == []
        assert extract_tables_from_html("just plain text") == []

    def test_inference_profile_defaults(self):
        from src.adapters.base import InferenceProfile
        profile = InferenceProfile()
        assert profile.wall_time_seconds == 0.0
        assert profile.gpu_memory_peak_mb == 0.0
