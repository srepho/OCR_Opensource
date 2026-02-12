"""DeepSeek-OCR adapter: DeepSeek's dedicated OCR model."""

from PIL import Image

from src.adapters.base import OCRAdapter, OCRResult


class DeepSeekOCRAdapter(OCRAdapter):
    """Adapter for DeepSeek-OCR.

    Custom method VLM for document OCR.
    """

    HF_ID = "DeepSeek-OCR/DeepSeek-OCR"

    def __init__(self, model_name: str = "DeepSeek-OCR", device: str = "auto"):
        super().__init__(model_name, device)
        self._tokenizer = None

    def load_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

        device = self._resolve_device()
        dtype = torch.float16 if device == "cuda" else torch.float32

        self._processor = AutoProcessor.from_pretrained(self.HF_ID, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self.HF_ID, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
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

        # DeepSeek-OCR chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "OCR this document page. Output the full text with formatting preserved."},
                ],
            }
        ]

        # Try chat template first, fall back to direct processing
        try:
            prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self._processor(
                text=prompt,
                images=[image],
                return_tensors="pt",
            ).to(device)
        except (AttributeError, TypeError):
            inputs = self._processor(
                images=image,
                text="OCR this document page. Output the full text with formatting preserved.",
                return_tensors="pt",
            ).to(device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)

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
