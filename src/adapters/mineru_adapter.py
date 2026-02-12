"""MinerU adapter scaffold.

This adapter intentionally fails fast until MinerU's APIs/CLI integration is added.
"""

import importlib.util

from PIL import Image

from src.adapters.base import OCRAdapter, OCRResult


class MinerUAdapter(OCRAdapter):
    """Scaffold adapter for MinerU pipeline."""

    def __init__(self, model_name: str = "MinerU", device: str = "cpu"):
        super().__init__(model_name, device)

    def load_model(self) -> None:
        # MinerU distribution names can vary; detect common module names.
        has_mineru = importlib.util.find_spec("mineru") is not None
        has_magic_pdf = importlib.util.find_spec("magic_pdf") is not None
        if not (has_mineru or has_magic_pdf):
            raise ImportError(
                "MinerU adapter scaffold is enabled, but MinerU is not installed. "
                "Install MinerU, then wire API/CLI calls in src/adapters/mineru_adapter.py."
            )
        self._loaded = True

    def ocr_page(self, image: Image.Image) -> OCRResult:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        raise RuntimeError(
            "MinerU adapter is a scaffold only. "
            "Implement concrete MinerU integration in ocr_page()."
        )
