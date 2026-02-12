"""Render PDF pages to PNG images using pypdfium2."""

from pathlib import Path

import pypdfium2 as pdfium
import yaml
from tqdm import tqdm


def load_config(config_path: str = "config/benchmark_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def render_pdf(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 200,
    max_pages: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Render all pages of a PDF to PNG images.

    Images saved as: output_dir/{pdf_stem}/page_001.png
    Returns list of rendered image paths.
    """
    stem = pdf_path.stem
    page_dir = output_dir / stem
    page_dir.mkdir(parents=True, exist_ok=True)

    doc = pdfium.PdfDocument(str(pdf_path))
    n_pages = len(doc)
    if max_pages is not None:
        n_pages = min(n_pages, max_pages)

    rendered = []
    scale = dpi / 72  # pypdfium2 uses 72 DPI as base

    for i in range(n_pages):
        out_path = page_dir / f"page_{i + 1:03d}.png"
        if out_path.exists() and not overwrite:
            rendered.append(out_path)
            continue

        page = doc[i]
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        image.save(str(out_path))
        rendered.append(out_path)

    doc.close()
    return rendered


def render_all_pdfs(
    pdf_dir: str | Path,
    image_dir: str | Path,
    dpi: int = 200,
    max_pages: int | None = None,
    overwrite: bool = False,
) -> dict[str, list[Path]]:
    """Render all PDFs in a directory to images.

    Returns dict mapping PDF stem to list of image paths.
    """
    pdf_dir = Path(pdf_dir)
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs to render")

    results = {}
    for pdf_path in tqdm(pdfs, desc="Rendering PDFs"):
        images = render_pdf(pdf_path, image_dir, dpi, max_pages, overwrite)
        results[pdf_path.stem] = images

    total_pages = sum(len(v) for v in results.values())
    print(f"Rendered {total_pages} pages from {len(results)} PDFs")
    return results


def render_sample_set(
    sample_set_path: str | Path,
    pdf_dir: str | Path,
    image_dir: str | Path,
    dpi: int = 200,
    overwrite: bool = False,
) -> list[Path]:
    """Render only pages specified in a sample set JSON file."""
    import json

    with open(sample_set_path) as f:
        sample_set = json.load(f)

    pdf_dir = Path(pdf_dir)
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    rendered = []
    for entry in tqdm(sample_set["pages"], desc="Rendering sample set"):
        pdf_stem = entry["pdf_stem"]
        page_num = entry["page_num"]
        pdf_path = pdf_dir / f"{pdf_stem}.pdf"

        if not pdf_path.exists():
            print(f"Warning: {pdf_path} not found, skipping")
            continue

        page_dir = image_dir / pdf_stem
        page_dir.mkdir(parents=True, exist_ok=True)
        out_path = page_dir / f"page_{page_num:03d}.png"

        if out_path.exists() and not overwrite:
            rendered.append(out_path)
            continue

        doc = pdfium.PdfDocument(str(pdf_path))
        if page_num > len(doc):
            print(f"Warning: page {page_num} > {len(doc)} in {pdf_stem}")
            doc.close()
            continue

        scale = dpi / 72
        page = doc[page_num - 1]  # 0-indexed
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        image.save(str(out_path))
        rendered.append(out_path)
        doc.close()

    print(f"Rendered {len(rendered)} sample pages")
    return rendered


def main():
    config = load_config()
    render_all_pdfs(
        pdf_dir=config["paths"]["pdf_dir"],
        image_dir=config["paths"]["image_dir"],
        dpi=config["rendering"]["dpi"],
        max_pages=config["rendering"].get("max_pages_per_pdf"),
    )


if __name__ == "__main__":
    main()
