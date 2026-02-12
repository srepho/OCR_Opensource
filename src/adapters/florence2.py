"""Florence-2 adapter: Microsoft's vision-language model with OCR tasks."""

from PIL import Image

from src.adapters.base import OCRAdapter, OCRResult


class Florence2Adapter(OCRAdapter):
    """Adapter for Microsoft Florence-2-large.

    Uses task-specific prompts with Florence's custom architecture.
    Supports <OCR> and <OCR_WITH_REGION> tasks.
    """

    HF_ID = "microsoft/Florence-2-large"

    def __init__(self, model_name: str = "Florence-2", device: str = "auto"):
        super().__init__(model_name, device)

    def load_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        device = self._resolve_device()
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

        self._processor = AutoProcessor.from_pretrained(
            self.HF_ID, trust_remote_code=True
        )
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

        # Florence-2 uses task-specific prompts
        task_prompt = "<OCR>"

        inputs = self._processor(
            text=task_prompt,
            images=image,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                **self._get_generation_kwargs(num_beams=3),
            )

        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        # Post-process Florence output
        text = self._processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )

        # The result is typically a dict with the task key
        if isinstance(text, dict):
            text = text.get("<OCR>", str(text))

        if not isinstance(text, str):
            text = str(text)

        return OCRResult(
            text=text,
            tables=[],
            raw_output=text,
            format="text",
            metadata={"model": self.HF_ID, "task": "OCR"},
        )

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
