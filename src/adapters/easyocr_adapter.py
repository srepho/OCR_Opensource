"""EasyOCR adapter: simple OCR library supporting 80+ languages."""

import numpy as np
from PIL import Image

from src.adapters.base import OCRAdapter, OCRResult


class EasyOCRAdapter(OCRAdapter):
    """Adapter for EasyOCR.

    Simple detection + recognition pipeline.
    CPU-friendly, good baseline.
    """

    def __init__(self, model_name: str = "EasyOCR", device: str = "cpu"):
        super().__init__(model_name, device)

    def load_model(self) -> None:
        import easyocr

        use_gpu = self._resolve_device() == "cuda"
        self._model = easyocr.Reader(
            ["en"],
            gpu=use_gpu,
            verbose=False,
        )
        self._loaded = True

    def ocr_page(self, image: Image.Image) -> OCRResult:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        img_array = np.array(image)
        results = self._model.readtext(img_array)

        if not results:
            return OCRResult(text="", format="text", metadata={"engine": "easyocr"})

        # results: list of (bbox, text, confidence)
        # Sort by vertical position, then horizontal
        detections = []
        for bbox, text, confidence in results:
            # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            top_y = min(p[1] for p in bbox)
            left_x = min(p[0] for p in bbox)
            detections.append((top_y, left_x, text, confidence))

        detections.sort(key=lambda x: (x[0], x[1]))
        grouped = self._group_into_lines(detections)

        full_text = "\n".join(grouped)

        return OCRResult(
            text=full_text,
            tables=[],
            raw_output=full_text,
            format="text",
            metadata={"engine": "easyocr"},
        )

    def _group_into_lines(
        self, detections: list[tuple], y_threshold: float = 15.0
    ) -> list[str]:
        """Group detected text blocks into lines based on Y proximity."""
        if not detections:
            return []

        lines = []
        current_line = [detections[0]]

        for det in detections[1:]:
            if abs(det[0] - current_line[-1][0]) <= y_threshold:
                current_line.append(det)
            else:
                current_line.sort(key=lambda x: x[1])
                line_text = " ".join(d[2] for d in current_line)
                lines.append(line_text)
                current_line = [det]

        if current_line:
            current_line.sort(key=lambda x: x[1])
            line_text = " ".join(d[2] for d in current_line)
            lines.append(line_text)

        return lines

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cpu"
