"""RolmOCR adapter: Reducto's refined olmOCR."""

from PIL import Image

from src.adapters.base import OCRAdapter, OCRResult


class RolmOCRAdapter(OCRAdapter):
    """Adapter for RolmOCR.

    Reducto's refinement of olmOCR, same Qwen2.5-VL backbone.
    Requires A100 (Tier 2).
    """

    HF_ID = "reducto/RolmOCR"

    def __init__(self, model_name: str = "RolmOCR", device: str = "auto"):
        super().__init__(model_name, device)

    def load_model(self) -> None:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        if not torch.cuda.is_available():
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
                    {"type": "text", "text": (
                        "Return the full text content of this document page. "
                        "Preserve reading order, tables, and formatting in markdown."
                    )},
                ],
            }
        ]

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

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
