"""Qwen2.5-VL-7B adapter: general VLM with strong OCR capability."""

from PIL import Image

from src.adapters.base import OCRAdapter, OCRResult


class Qwen25VLAdapter(OCRAdapter):
    """Adapter for Qwen2.5-VL-7B-Instruct.

    Uses chat template with Qwen2.5-VL specific image handling.
    Requires A100 (Tier 2).
    """

    HF_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

    def __init__(self, model_name: str = "Qwen2.5-VL-7B", device: str = "auto"):
        super().__init__(model_name, device)

    def load_model(self) -> None:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        device = self._resolve_device()
        if not device.startswith("cuda"):
            raise RuntimeError(
                f"{self.model_name} requires CUDA GPU (Tier 2 model, ~16GB VRAM). "
                f"No CUDA device available."
            )

        self._processor = AutoProcessor.from_pretrained(self.HF_ID, trust_remote_code=True)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
                    {"type": "text", "text": self._get_instruction(
                        "Extract all text from this document image. Preserve the layout, tables, and formatting as markdown."
                    )},
                ],
            }
        ]

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process with Qwen2.5-VL processor
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
        except ImportError:
            image_inputs = [image]
            video_inputs = None

        inputs = self._processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **self._get_generation_kwargs(),
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
