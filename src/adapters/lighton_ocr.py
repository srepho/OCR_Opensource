"""LightOnOCR-2-1B adapter: top-performing 1B parameter OCR model."""

from PIL import Image

from src.adapters.base import OCRAdapter, OCRResult


class LightOnOCRAdapter(OCRAdapter):
    """Adapter for LightOnOCR-2-1B.

    Uses standard transformers chat template API.
    Outputs markdown with tables.
    """

    HF_ID = "lightonai/LightOnOCR-2-1B"

    def __init__(self, model_name: str = "LightOnOCR-2-1B", device: str = "auto"):
        super().__init__(model_name, device)
        self._tokenizer = None

    def load_model(self) -> None:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        device = self._resolve_device()
        dtype = torch.float16 if device == "cuda" else torch.float32

        self._processor = AutoProcessor.from_pretrained(self.HF_ID, trust_remote_code=True)
        self._model = AutoModelForVision2Seq.from_pretrained(
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

        # Build chat message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract all text from this document image. Preserve the layout, tables, and formatting."},
                ],
            }
        ]

        # Apply chat template
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
            )

        # Decode only generated tokens
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = self._processor.decode(generated, skip_special_tokens=True)

        tables = self._extract_tables(text, "markdown")

        return OCRResult(
            text=text,
            tables=tables,
            raw_output=text,
            format="markdown",
            metadata={"model": self.HF_ID},
        )

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
