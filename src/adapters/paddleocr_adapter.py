"""PaddleOCR adapter: Baidu's detection + recognition toolkit."""

import numpy as np
from PIL import Image

from src.adapters.base import OCRAdapter, OCRResult


class PaddleOCRAdapter(OCRAdapter):
    """Adapter for PaddleOCR.

    Traditional detection + recognition pipeline.
    CPU-friendly, no GPU required.
    """

    def __init__(self, model_name: str = "PaddleOCR", device: str = "cpu"):
        super().__init__(model_name, device)

    def load_model(self) -> None:
        from paddleocr import PaddleOCR

        use_gpu = self._resolve_device() == "cuda"
        self._model = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=use_gpu,
            show_log=False,
        )
        self._loaded = True

    def ocr_page(self, image: Image.Image) -> OCRResult:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        img_array = np.array(image)
        result = self._model.ocr(img_array, cls=True)

        if not result or not result[0]:
            return OCRResult(text="", format="text", metadata={"engine": "paddleocr"})

        # Sort by vertical position (top of bounding box), then horizontal
        lines = []
        for detection in result[0]:
            bbox = detection[0]
            text = detection[1][0]
            confidence = detection[1][1]

            # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            top_y = min(p[1] for p in bbox)
            left_x = min(p[0] for p in bbox)
            lines.append((top_y, left_x, text, confidence))

        # Group into lines by Y proximity
        lines.sort(key=lambda x: (x[0], x[1]))
        grouped = self._group_into_lines(lines)

        full_text = "\n".join(grouped)

        return OCRResult(
            text=full_text,
            tables=[],
            raw_output=full_text,
            format="text",
            metadata={"engine": "paddleocr"},
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
                # Sort current line by X position and join
                current_line.sort(key=lambda x: x[1])
                line_text = " ".join(d[2] for d in current_line)
                lines.append(line_text)
                current_line = [det]

        # Don't forget the last line
        if current_line:
            current_line.sort(key=lambda x: x[1])
            line_text = " ".join(d[2] for d in current_line)
            lines.append(line_text)

        return lines

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cpu"  # PaddleOCR defaults to CPU
