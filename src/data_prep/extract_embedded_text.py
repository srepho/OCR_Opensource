"""Extract embedded text from native PDFs using pdfplumber and PyMuPDF.

Dual extraction with best-of selection per page:
- Low discrepancy (NED <= threshold): use pdfplumber (both agree)
- High discrepancy + similar char count: use PyMuPDF (better reading order for multi-column)
- PyMuPDF returns nothing: use pdfplumber
- pdfplumber returns artifacts (doubled chars): use PyMuPDF

Supports multiprocessing for faster extraction.
"""

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
import yaml
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm


def load_config(config_path: str = "config/benchmark_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_with_pdfplumber(pdf_path: Path) -> list[str]:
    """Extract text from all pages using pdfplumber."""
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return pages


def extract_with_pymupdf(pdf_path: Path) -> list[str]:
    """Extract text from all pages using PyMuPDF."""
    pages = []
    doc = fitz.open(str(pdf_path))
    for page in doc:
        text = page.get_text("text") or ""
        pages.append(text)
    doc.close()
    return pages


def compute_ned(text_a: str, text_b: str) -> float:
    """Compute Normalized Edit Distance between two strings."""
    if not text_a and not text_b:
        return 0.0
    max_len = max(len(text_a), len(text_b))
    if max_len == 0:
        return 0.0
    dist = Levenshtein.distance(text_a, text_b)
    return dist / max_len


_DOUBLED_PAIRS_RE = re.compile(r"(.)\1")


def _has_doubled_chars(text: str) -> bool:
    """Detect InDesign production artifacts like '553388118800'.

    These artifacts consist of nearly every character being doubled in pairs
    (e.g., '55' '33' '88'), so consecutive pair matches cover most of the text.
    """
    if len(text) < 20:
        return False
    pair_count = len(_DOUBLED_PAIRS_RE.findall(text))
    # Each pair covers 2 chars; if pairs cover >40% of text it's an artifact
    return (pair_count * 2) > len(text) * 0.4


def select_best_text(
    plumber_text: str,
    pymupdf_text: str,
    ned: float,
    discrepancy_threshold: float,
) -> tuple[str, str]:
    """Select the best ground truth text from two extractors.

    Returns (best_text, source) where source is 'pdfplumber', 'pymupdf', or 'both_agree'.
    """
    plumber_stripped = plumber_text.strip()
    pymupdf_stripped = pymupdf_text.strip()

    # Both empty
    if not plumber_stripped and not pymupdf_stripped:
        return "", "both_empty"

    # Only one has text
    if not pymupdf_stripped:
        # Check if pdfplumber text is an artifact
        if _has_doubled_chars(plumber_stripped):
            return "", "both_empty"
        return plumber_text, "pdfplumber"
    if not plumber_stripped:
        return pymupdf_text, "pymupdf"

    # Low discrepancy — extractors agree, use pdfplumber
    if ned <= discrepancy_threshold:
        return plumber_text, "both_agree"

    # pdfplumber returned artifact text
    if _has_doubled_chars(plumber_stripped):
        return pymupdf_text, "pymupdf"

    # High discrepancy with similar char count = reading order difference
    # PyMuPDF reads column-by-column which is better for multi-column layouts
    cp, cm = len(plumber_stripped), len(pymupdf_stripped)
    if cp > 0 and cm > 0:
        ratio = min(cp, cm) / max(cp, cm)
        if ratio > 0.85:
            # Similar amount of text, different ordering — prefer PyMuPDF
            return pymupdf_text, "pymupdf"

    # pdfplumber has substantially more text — it may be picking up
    # headers/footers/extra content. Use pdfplumber.
    if cp > cm * 1.15:
        return plumber_text, "pdfplumber"

    # PyMuPDF has more text
    return pymupdf_text, "pymupdf"


def extract_pdf_text(
    pdf_path: Path,
    output_dir: Path,
    discrepancy_threshold: float = 0.05,
    overwrite: bool = False,
) -> dict:
    """Extract embedded text from a PDF using dual extractors with best-of selection.

    Saves per-page text files and returns metadata including discrepancy flags.
    """
    stem = pdf_path.stem
    page_dir = output_dir / stem
    page_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    meta_path = page_dir / "_extraction_meta.json"
    if meta_path.exists() and not overwrite:
        with open(meta_path) as f:
            return json.load(f)

    # Dual extraction
    plumber_pages = extract_with_pdfplumber(pdf_path)
    pymupdf_pages = extract_with_pymupdf(pdf_path)

    n_pages = max(len(plumber_pages), len(pymupdf_pages))
    page_meta = []

    source_counts = {"both_agree": 0, "pdfplumber": 0, "pymupdf": 0, "both_empty": 0}

    for i in range(n_pages):
        plumber_text = plumber_pages[i] if i < len(plumber_pages) else ""
        pymupdf_text = pymupdf_pages[i] if i < len(pymupdf_pages) else ""

        ned = compute_ned(plumber_text, pymupdf_text)
        has_discrepancy = ned > discrepancy_threshold

        # Best-of selection
        best_text, source = select_best_text(
            plumber_text, pymupdf_text, ned, discrepancy_threshold
        )

        source_counts[source] += 1

        # Save primary (best) text
        page_file = page_dir / f"page_{i + 1:03d}.txt"
        page_file.write_text(best_text, encoding="utf-8")

        # Save both originals if discrepancy, for auditing
        if has_discrepancy:
            alt_plumber = page_dir / f"page_{i + 1:03d}_pdfplumber.txt"
            alt_plumber.write_text(plumber_text, encoding="utf-8")
            alt_pymupdf = page_dir / f"page_{i + 1:03d}_pymupdf.txt"
            alt_pymupdf.write_text(pymupdf_text, encoding="utf-8")

        page_meta.append({
            "page_num": i + 1,
            "char_count_pdfplumber": len(plumber_text),
            "char_count_pymupdf": len(pymupdf_text),
            "char_count_best": len(best_text),
            "ned_between_extractors": round(ned, 4),
            "has_discrepancy": has_discrepancy,
            "selected_source": source,
            "is_empty": len(best_text.strip()) == 0,
        })

    metadata = {
        "pdf_stem": stem,
        "total_pages": n_pages,
        "pages_with_discrepancy": sum(1 for p in page_meta if p["has_discrepancy"]),
        "empty_pages": sum(1 for p in page_meta if p["is_empty"]),
        "source_counts": source_counts,
        "pages": page_meta,
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def _extract_single_pdf(args: tuple) -> dict:
    """Worker function for multiprocessing. Takes a tuple to be picklable."""
    pdf_path, output_dir, discrepancy_threshold, overwrite = args
    return extract_pdf_text(
        Path(pdf_path), Path(output_dir), discrepancy_threshold, overwrite
    )


def extract_all_pdfs(
    pdf_dir: str | Path,
    output_dir: str | Path,
    discrepancy_threshold: float = 0.05,
    overwrite: bool = False,
    max_workers: int | None = None,
) -> list[dict]:
    """Extract embedded text from all PDFs using multiprocessing.

    Returns list of per-PDF metadata dicts.
    """
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs for text extraction")

    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, len(pdfs))
    print(f"Using {max_workers} worker processes")

    # Build args for each PDF
    work_args = [
        (str(pdf_path), str(output_dir), discrepancy_threshold, overwrite)
        for pdf_path in pdfs
    ]

    all_meta = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_extract_single_pdf, args): args[0]
            for args in work_args
        }
        with tqdm(total=len(futures), desc="Extracting embedded text") as pbar:
            for future in as_completed(futures):
                try:
                    meta = future.result()
                    all_meta.append(meta)
                except Exception as e:
                    pdf_name = Path(futures[future]).name
                    print(f"\nError processing {pdf_name}: {e}")
                pbar.update(1)

    # Sort by PDF stem for consistent ordering
    all_meta.sort(key=lambda m: m["pdf_stem"])

    # Aggregate stats
    total_pages = sum(m["total_pages"] for m in all_meta)
    discrepancy_pages = sum(m["pages_with_discrepancy"] for m in all_meta)
    empty_pages = sum(m["empty_pages"] for m in all_meta)

    # Aggregate source selection counts
    total_sources = {"both_agree": 0, "pdfplumber": 0, "pymupdf": 0, "both_empty": 0}
    for m in all_meta:
        for k, v in m.get("source_counts", {}).items():
            total_sources[k] = total_sources.get(k, 0) + v

    global_meta = {
        "total_pdfs": len(all_meta),
        "total_pages": total_pages,
        "pages_with_discrepancy": discrepancy_pages,
        "empty_pages": empty_pages,
        "source_selection": total_sources,
        "per_pdf": [{
            "pdf_stem": m["pdf_stem"],
            "total_pages": m["total_pages"],
            "pages_with_discrepancy": m["pages_with_discrepancy"],
            "empty_pages": m["empty_pages"],
            "source_counts": m.get("source_counts", {}),
        } for m in all_meta],
    }

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(global_meta, f, indent=2)

    print(f"\nExtracted text from {total_pages} pages across {len(all_meta)} PDFs")
    print(f"  Discrepancy pages: {discrepancy_pages}")
    print(f"  Empty pages: {empty_pages}")
    print(f"  Source selection: {total_sources}")

    return all_meta


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract embedded text from PDFs")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-extract even if output already exists",
    )
    args = parser.parse_args()

    config = load_config()
    extract_all_pdfs(
        pdf_dir=config["paths"]["pdf_dir"],
        output_dir=config["paths"]["embedded_text_dir"],
        discrepancy_threshold=config["ground_truth"]["discrepancy_threshold"],
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
