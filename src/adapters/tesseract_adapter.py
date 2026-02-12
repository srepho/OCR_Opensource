"""Tesseract adapter: classic OCR baseline via pytesseract."""

import shutil

from PIL import Image

from src.adapters.base import OCRAdapter, OCRResult


class TesseractAdapter(OCRAdapter):
    """Adapter for Tesseract OCR.

    Requires:
    - Python package: pytesseract
    - System binary: tesseract
    """

    def __init__(
        self,
        model_name: str = "Tesseract OCR",
        device: str = "cpu",
        lang: str = "eng",
        config: str = "",
    ):
        super().__init__(model_name, device)
        self.lang = lang
        self.config = config
        self._pytesseract = None

    def load_model(self) -> None:
        try:
            import pytesseract
        except ImportError:
            raise ImportError(
                "pytesseract is required for TesseractAdapter. "
                "Install with: pip install pytesseract"
            )

        if shutil.which("tesseract") is None:
            raise RuntimeError(
                "Tesseract binary not found in PATH. "
                "Install the system package (e.g., `brew install tesseract`)."
            )

        self._pytesseract = pytesseract
        self._loaded = True

    def ocr_page(self, image: Image.Image) -> OCRResult:
        if not self._loaded or self._pytesseract is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        text = self._pytesseract.image_to_string(
            image,
            lang=self.lang,
            config=self.config,
        )

        return OCRResult(
            text=text,
            tables=[],
            raw_output=text,
            format="text",
            metadata={"engine": "tesseract", "lang": self.lang},
        )

    def unload_model(self) -> None:
        self._pytesseract = None
        super().unload_model()
