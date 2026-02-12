"""DocTR adapter: detection + recognition OCR pipeline."""

import numpy as np
from PIL import Image

from src.adapters.base import OCRAdapter, OCRResult


class DocTRAdapter(OCRAdapter):
    """Adapter for DocTR (Document Text Recognition).

    Traditional detection + recognition pipeline.
    Works on CPU or GPU, good baseline model.
    """

    def __init__(self, model_name: str = "DocTR", device: str = "auto"):
        super().__init__(model_name, device)

    def load_model(self) -> None:
        from doctr.models import ocr_predictor

        use_gpu = self._resolve_device() != "cpu"
        self._model = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_vgg16_bn",
            pretrained=True,
            assume_straight_pages=True,
        )
        if use_gpu:
            self._model = self._model.cuda()

        self._loaded = True

    def ocr_page(self, image: Image.Image) -> OCRResult:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # DocTR expects numpy array
        img_array = np.array(image)

        # Run prediction
        result = self._model([img_array])

        # Extract text with layout preservation
        text_blocks = []
        for page in result.pages:
            for block in page.blocks:
                block_lines = []
                for line in block.lines:
                    words = [word.value for word in line.words]
                    block_lines.append(" ".join(words))
                text_blocks.append("\n".join(block_lines))

        full_text = "\n\n".join(text_blocks)

        return OCRResult(
            text=full_text,
            tables=[],
            raw_output=full_text,
            format="text",
            metadata={"engine": "doctr"},
        )

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
