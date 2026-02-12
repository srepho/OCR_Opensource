"""dots.ocr adapter: document-specific small model."""

from PIL import Image

from src.adapters.base import OCRAdapter, OCRResult


class DotsOCRAdapter(OCRAdapter):
    """Adapter for dots.ocr (dots.llm1.base).

    Custom inference method for document OCR.
    """

    HF_ID = "NexaAIDev/dots.llm1.base"

    def __init__(self, model_name: str = "dots.ocr", device: str = "auto"):
        super().__init__(model_name, device)
        self._tokenizer = None

    def load_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

        device = self._resolve_device()
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

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

        # dots.ocr uses a specific prompt format
        prompt = self._get_instruction("Extract all text from this document image.")
        inputs = self._processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **self._get_generation_kwargs(),
            )

        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from output if echoed
        if text.startswith(prompt):
            text = text[len(prompt):].strip()

        return OCRResult(
            text=text,
            tables=[],
            raw_output=text,
            format="text",
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
