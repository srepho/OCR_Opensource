"""GOT-OCR2 adapter: smallest VLM with custom chat method."""

from PIL import Image

from src.adapters.base import OCRAdapter, OCRResult


class GOTOCR2Adapter(OCRAdapter):
    """Adapter for GOT-OCR2_0.

    Uses custom model.chat() method for inference.
    Supports format_text output for better layout.
    """

    HF_ID = "stepfun-ai/GOT-OCR2_0"

    def __init__(self, model_name: str = "GOT-OCR2", device: str = "auto"):
        super().__init__(model_name, device)
        self._tokenizer = None

    def load_model(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = self._resolve_device()
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.HF_ID,
            trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            self.HF_ID,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
        self._model.eval()
        self._loaded = True

    def ocr_page(self, image: Image.Image) -> OCRResult:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # GOT-OCR2 requires saving image to temp file for its chat method
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        try:
            # Use format_text for structured output
            text = self._model.chat(
                self._tokenizer,
                tmp_path,
                ocr_type="format",
            )
        finally:
            os.unlink(tmp_path)

        tables = self._extract_tables(text, "markdown")

        return OCRResult(
            text=text,
            tables=tables,
            raw_output=text,
            format="markdown",
            metadata={"model": self.HF_ID, "ocr_type": "format"},
        )

    def unload_model(self) -> None:
        self._tokenizer = None
        super().unload_model()

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
