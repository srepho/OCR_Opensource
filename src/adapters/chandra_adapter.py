"""Chandra adapter: document-specialized VLM."""

from PIL import Image

from src.adapters.base import OCRAdapter, OCRResult


class ChandraAdapter(OCRAdapter):
    """Adapter for Chandra document VLM.

    Uses chat template API. Requires A100 (Tier 2).
    """

    HF_ID = "adarshxs/Chandra"

    def __init__(self, model_name: str = "Chandra", device: str = "auto"):
        super().__init__(model_name, device)

    def load_model(self) -> None:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        device = self._resolve_device()
        if not device.startswith("cuda"):
            raise RuntimeError(
                f"{self.model_name} requires CUDA GPU (Tier 2 model, ~16GB VRAM). "
                f"No CUDA device available."
            )

        self._processor = AutoProcessor.from_pretrained(self.HF_ID, trust_remote_code=True)
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.HF_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        self._loaded = True

    def ocr_page(self, image: Image.Image) -> OCRResult:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract all text from this document image. Preserve layout, tables, and formatting."},
                ],
            }
        ]

        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
            )

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
