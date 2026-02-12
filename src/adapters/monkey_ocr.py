"""MonkeyOCR-1.2B adapter: specialized document OCR model."""

from PIL import Image

from src.adapters.base import OCRAdapter, OCRResult


class MonkeyOCRAdapter(OCRAdapter):
    """Adapter for MonkeyOCR.

    Custom method document OCR model.
    """

    HF_ID = "echo840/MonkeyOCR"

    def __init__(self, model_name: str = "MonkeyOCR-1.2B", device: str = "auto"):
        super().__init__(model_name, device)
        self._tokenizer = None

    def load_model(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = self._resolve_device()
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.HF_ID, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.HF_ID,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        self._model.eval()
        self._loaded = True

    def ocr_page(self, image: Image.Image) -> OCRResult:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import torch

        device = self._resolve_device()

        # MonkeyOCR may use a chat-style or direct method
        # Try chat method first, then fall back
        try:
            text = self._model.chat(
                self._tokenizer,
                image,
                "Extract all text from this document image with layout preserved.",
            )
        except (AttributeError, TypeError):
            # Fallback: use processor-based approach
            inputs = self._tokenizer(
                "Extract all text from this document image.",
                return_tensors="pt",
            ).to(device)

            # If model has image processing
            try:
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(self.HF_ID, trust_remote_code=True)
                inputs = processor(
                    images=image,
                    text="Extract all text from this document image.",
                    return_tensors="pt",
                ).to(device)
            except Exception:
                pass

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=False,
                )

            text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        tables = self._extract_tables(text, "markdown")

        return OCRResult(
            text=text,
            tables=tables,
            raw_output=text,
            format="markdown",
            metadata={"model": self.HF_ID},
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
