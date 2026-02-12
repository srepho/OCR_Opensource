"""Granite-Docling-258M adapter scaffold with generic VLM inference."""

from PIL import Image

from src.adapters.base import OCRAdapter, OCRResult


class GraniteDocling258MAdapter(OCRAdapter):
    """Adapter for Granite-Docling-258M.

    This is a scaffold adapter. It uses common transformers APIs and may require
    model-specific prompt or post-processing refinement for best quality.
    """

    HF_ID = "ibm-granite/granite-docling-258M"

    def __init__(self, model_name: str = "Granite-Docling-258M", device: str = "auto"):
        super().__init__(model_name, device)

    def load_model(self) -> None:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForVision2Seq,
            AutoProcessor,
        )

        device = self._resolve_device()
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

        self._processor = AutoProcessor.from_pretrained(self.HF_ID, trust_remote_code=True)

        self._model = None
        loaders = [AutoModelForVision2Seq, AutoModelForCausalLM]
        last_error = None

        for loader in loaders:
            try:
                model = loader.from_pretrained(
                    self.HF_ID,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                )
                self._model = model.to(device)
                break
            except Exception as e:  # noqa: BLE001 - surface last loader error below
                last_error = e

        if self._model is None:
            raise RuntimeError(
                f"Failed to load {self.HF_ID} with generic loaders. "
                f"Model likely needs custom API integration. Last error: {last_error}"
            )

        self._model.eval()
        self._loaded = True

    def ocr_page(self, image: Image.Image) -> OCRResult:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import torch

        prompt = (
            "Convert this document page into markdown. "
            "Preserve heading hierarchy, lists, and tables."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            rendered_prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[rendered_prompt],
                images=[image],
                return_tensors="pt",
            ).to(self._model.device)
        except Exception:
            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        generated = outputs[0][input_len:]

        text = None
        for decode_fn in ("decode", "batch_decode"):
            fn = getattr(self._processor, decode_fn, None)
            if fn is None:
                continue
            try:
                if decode_fn == "decode":
                    text = fn(generated, skip_special_tokens=True)
                else:
                    text = fn([generated], skip_special_tokens=True)[0]
                break
            except Exception:
                continue

        if text is None:
            text = str(generated.tolist())

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
