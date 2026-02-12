"""Surya adapter scaffold.

This adapter intentionally fails fast until Surya's Python APIs are wired in.
"""

import importlib.util

from PIL import Image

from src.adapters.base import OCRAdapter, OCRResult


class SuryaAdapter(OCRAdapter):
    """Scaffold adapter for Surya OCR toolkit."""

    def __init__(self, model_name: str = "Surya OCR Toolkit", device: str = "cpu"):
        super().__init__(model_name, device)

    def load_model(self) -> None:
        # Surya package naming can vary between builds; check common module names.
        has_surya = importlib.util.find_spec("surya") is not None
        has_surya_ocr = importlib.util.find_spec("surya_ocr") is not None
        if not (has_surya or has_surya_ocr):
            raise ImportError(
                "Surya adapter scaffold is enabled, but Surya is not installed. "
                "Install Surya, then implement API calls in src/adapters/surya_adapter.py."
            )
        self._loaded = True

    def ocr_page(self, image: Image.Image) -> OCRResult:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        raise RuntimeError(
            "Surya adapter is a scaffold only. "
            "Implement concrete Surya API integration in ocr_page()."
        )
