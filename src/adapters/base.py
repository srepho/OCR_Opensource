"""Base classes for OCR model adapters."""

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image


@dataclass
class OCRResult:
    """Result from an OCR model for a single page."""

    text: str
    tables: list[str] = field(default_factory=list)  # HTML table strings
    raw_output: str = ""
    format: str = "text"  # 'text', 'markdown', 'html'
    metadata: dict = field(default_factory=dict)

    def has_tables(self) -> bool:
        return len(self.tables) > 0

    def text_length(self) -> int:
        return len(self.text)


@dataclass
class InferenceProfile:
    """Performance profile for a single inference call."""

    wall_time_seconds: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    tokens_generated: int = 0


def extract_tables_from_markdown(text: str) -> list[str]:
    """Extract markdown tables and convert to simple HTML.

    Finds pipe-delimited tables in markdown text and converts them.
    """
    tables = []
    lines = text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        # Look for start of a markdown table (line with pipes)
        if "|" in line and line.startswith("|"):
            table_lines = [line]
            i += 1
            while i < len(lines) and "|" in lines[i].strip():
                table_lines.append(lines[i].strip())
                i += 1

            if len(table_lines) >= 2:
                html = _markdown_table_to_html(table_lines)
                if html:
                    tables.append(html)
        else:
            i += 1

    return tables


def extract_tables_from_html(text: str) -> list[str]:
    """Extract HTML <table> elements from text."""
    pattern = re.compile(r"<table.*?</table>", re.DOTALL | re.IGNORECASE)
    return pattern.findall(text)


def _markdown_table_to_html(lines: list[str]) -> str | None:
    """Convert markdown table lines to HTML table string."""
    if len(lines) < 2:
        return None

    def parse_row(line: str) -> list[str]:
        cells = line.split("|")
        # Strip outer empty cells from leading/trailing pipes
        if cells and cells[0].strip() == "":
            cells = cells[1:]
        if cells and cells[-1].strip() == "":
            cells = cells[:-1]
        return [c.strip() for c in cells]

    rows = []
    separator_idx = None

    for idx, line in enumerate(lines):
        # Detect separator line (e.g., |---|---|)
        stripped = line.replace("|", "").replace("-", "").replace(":", "").strip()
        if not stripped and idx > 0:
            separator_idx = idx
            continue
        rows.append(parse_row(line))

    if not rows:
        return None

    html_parts = ["<table>"]

    for idx, row in enumerate(rows):
        html_parts.append("  <tr>")
        tag = "th" if idx == 0 and separator_idx is not None else "td"
        for cell in row:
            html_parts.append(f"    <{tag}>{cell}</{tag}>")
        html_parts.append("  </tr>")

    html_parts.append("</table>")
    return "\n".join(html_parts)


class OCRAdapter(ABC):
    """Abstract base class for OCR model adapters.

    All OCR models must implement this interface.
    """

    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
        self._loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and processor into memory."""

    @abstractmethod
    def ocr_page(self, image: Image.Image) -> OCRResult:
        """Run OCR on a single page image.

        Args:
            image: PIL Image of the page

        Returns:
            OCRResult with extracted text and tables
        """

    def unload_model(self) -> None:
        """Unload model from memory to free GPU/RAM."""
        self._model = None
        self._processor = None
        self._loaded = False

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        import gc
        gc.collect()

    def ocr_page_profiled(self, image: Image.Image) -> tuple[OCRResult, InferenceProfile]:
        """Run OCR with performance profiling.

        Returns:
            Tuple of (OCRResult, InferenceProfile)
        """
        profile = InferenceProfile()

        # Track GPU memory if available
        gpu_tracking = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                gpu_tracking = True
        except ImportError:
            pass

        start_time = time.perf_counter()
        result = self.ocr_page(image)
        profile.wall_time_seconds = time.perf_counter() - start_time

        if gpu_tracking:
            import torch
            profile.gpu_memory_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        profile.tokens_generated = len(result.text.split())

        return result, profile

    def ocr_file(self, image_path: str | Path) -> OCRResult:
        """Run OCR on an image file."""
        image = Image.open(str(image_path)).convert("RGB")
        return self.ocr_page(image)

    def _extract_tables(self, text: str, fmt: str) -> list[str]:
        """Extract tables from model output based on format."""
        if fmt == "html":
            return extract_tables_from_html(text)
        elif fmt == "markdown":
            tables = extract_tables_from_markdown(text)
            # Also check for embedded HTML tables in markdown
            tables.extend(extract_tables_from_html(text))
            return tables
        return []

    def _get_instruction(self, default: str = "") -> str:
        """Return the benchmark canonical instruction if set, else the default."""
        benchmark = getattr(self, "benchmark_instruction", "")
        return benchmark if benchmark else default

    # Keys from benchmark_decoding that are safe to forward to any HF generate().
    # Adapter-specific defaults bypass this filter (adapter authors know their model).
    _SAFE_GENERATE_KEYS = frozenset({
        "max_new_tokens", "max_length", "do_sample",
        "temperature", "top_p", "top_k",
        "num_beams", "repetition_penalty",
        "length_penalty", "early_stopping",
        "no_repeat_ngram_size",
    })

    def _get_generation_kwargs(self, **adapter_defaults) -> dict:
        """Merge benchmark decoding config with adapter-specific defaults.

        Adapter defaults are used as the base; benchmark_decoding overrides them.
        This lets adapters keep model-specific params (like num_beams for Florence)
        while still respecting the benchmark protocol.

        Benchmark keys are filtered to ``_SAFE_GENERATE_KEYS`` to avoid passing
        unknown parameters to models with strict generate() signatures.
        """
        base = {"max_new_tokens": 4096, "do_sample": False}
        base.update(adapter_defaults)
        benchmark = getattr(self, "benchmark_decoding", {})
        if benchmark:
            base.update(
                {k: v for k, v in benchmark.items() if k in self._SAFE_GENERATE_KEYS}
            )
        return base

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}', loaded={self._loaded})"
