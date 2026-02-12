"""Build stratified sample sets from the corpus for benchmarking."""

import json
import random
from pathlib import Path

import yaml
from tqdm import tqdm


def load_config(config_path: str = "config/benchmark_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_corpus_metadata(embedded_text_dir: str | Path) -> dict:
    """Load the global metadata.json from embedded text extraction."""
    meta_path = Path(embedded_text_dir) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found at {meta_path}. Run extract_embedded_text.py first."
        )
    with open(meta_path) as f:
        return json.load(f)


def classify_page_content(
    text_path: Path,
) -> str:
    """Heuristic classification of page content type.

    Returns one of: 'table', 'multi_column', 'list_heavy', 'text_only', 'sparse'
    """
    if not text_path.exists():
        return "sparse"

    text = text_path.read_text(encoding="utf-8")

    if len(text.strip()) < 50:
        return "sparse"

    lines = text.strip().split("\n")
    non_empty_lines = [l for l in lines if l.strip()]

    # Table heuristics: multiple lines with consistent delimiters or aligned columns
    pipe_lines = sum(1 for l in non_empty_lines if "|" in l)
    tab_lines = sum(1 for l in non_empty_lines if "\t" in l)
    if pipe_lines > 3 or tab_lines > 3:
        return "table"

    # Look for tabular patterns: lines with multiple number clusters
    number_heavy = sum(
        1 for l in non_empty_lines
        if sum(1 for c in l if c.isdigit()) > len(l) * 0.2 and len(l) > 20
    )
    if number_heavy > 5:
        return "table"

    # Bullet/list heuristics
    bullet_chars = {"•", "-", "–", "►", "■", "▪", "○"}
    bullet_lines = sum(
        1 for l in non_empty_lines
        if l.strip() and l.strip()[0] in bullet_chars
    )
    numbered_list = sum(
        1 for l in non_empty_lines
        if l.strip() and len(l.strip()) > 2 and l.strip()[0].isdigit() and l.strip()[1] in ".)"
    )
    if (bullet_lines + numbered_list) > len(non_empty_lines) * 0.3:
        return "list_heavy"

    # Multi-column heuristic: large whitespace gaps in the middle of lines
    wide_gap_lines = sum(
        1 for l in non_empty_lines
        if "   " in l.strip() and len(l) > 40
    )
    if wide_gap_lines > len(non_empty_lines) * 0.3:
        return "multi_column"

    return "text_only"


def build_page_index(
    embedded_text_dir: str | Path,
    image_dir: str | Path,
) -> list[dict]:
    """Build an index of all available pages with content classification."""
    embedded_text_dir = Path(embedded_text_dir)
    image_dir = Path(image_dir)

    pages = []
    for pdf_meta_path in sorted(embedded_text_dir.glob("*/_extraction_meta.json")):
        with open(pdf_meta_path) as f:
            pdf_meta = json.load(f)

        stem = pdf_meta["pdf_stem"]
        for page_info in pdf_meta["pages"]:
            page_num = page_info["page_num"]
            total_pages = pdf_meta["total_pages"]
            text_path = embedded_text_dir / stem / f"page_{page_num:03d}.txt"
            image_path = image_dir / stem / f"page_{page_num:03d}.png"

            # Page position category
            if page_num <= 2:
                position = "front"
            elif page_num >= total_pages - 1:
                position = "back"
            elif page_num <= total_pages * 0.25:
                position = "early"
            elif page_num >= total_pages * 0.75:
                position = "late"
            else:
                position = "middle"

            content_type = classify_page_content(text_path)

            pages.append({
                "pdf_stem": stem,
                "page_num": page_num,
                "total_pages_in_doc": total_pages,
                "position": position,
                "content_type": content_type,
                "char_count": page_info["char_count_pdfplumber"],
                "is_empty": page_info["is_empty"],
                "has_discrepancy": page_info["has_discrepancy"],
                "has_image": image_path.exists(),
            })

    return pages


def build_quick_dev(pages: list[dict], n: int = 20, seed: int = 42) -> list[dict]:
    """Build a small development set with diverse content."""
    rng = random.Random(seed)

    # Filter to non-empty pages with images
    candidates = [p for p in pages if not p["is_empty"] and p["has_image"]]

    if not candidates:
        print("Warning: no candidate pages for quick_dev set")
        return []

    # Try to get diverse content types
    by_type = {}
    for p in candidates:
        by_type.setdefault(p["content_type"], []).append(p)

    selected = []
    types = list(by_type.keys())
    per_type = max(1, n // len(types)) if types else n

    for t in types:
        pool = by_type[t]
        rng.shuffle(pool)
        selected.extend(pool[:per_type])

    # Fill remaining
    rng.shuffle(candidates)
    for p in candidates:
        if len(selected) >= n:
            break
        if p not in selected:
            selected.append(p)

    return selected[:n]


def build_stratified(
    pages: list[dict],
    n_pages: int = 100,
    n_docs: int = 20,
    pages_per_doc: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Build a stratified sample set.

    Selects n_docs documents, then pages_per_doc pages per document,
    stratified by content type and page position.
    """
    rng = random.Random(seed)

    # Filter to non-empty pages with images
    candidates = [p for p in pages if not p["is_empty"] and p["has_image"]]

    # Group by document
    by_doc = {}
    for p in candidates:
        by_doc.setdefault(p["pdf_stem"], []).append(p)

    # Select documents (spread across different stems)
    doc_stems = list(by_doc.keys())
    rng.shuffle(doc_stems)
    selected_docs = doc_stems[:n_docs]

    selected = []
    for stem in selected_docs:
        doc_pages = by_doc[stem]

        # Stratify within document by content type
        by_type = {}
        for p in doc_pages:
            by_type.setdefault(p["content_type"], []).append(p)

        doc_selected = []
        types = list(by_type.keys())

        # Round-robin through content types
        type_idx = 0
        while len(doc_selected) < pages_per_doc and any(by_type.values()):
            t = types[type_idx % len(types)]
            if by_type[t]:
                page = rng.choice(by_type[t])
                by_type[t].remove(page)
                doc_selected.append(page)
            type_idx += 1
            # Avoid infinite loop if all pools empty for some types
            if type_idx > pages_per_doc * len(types):
                break

        selected.extend(doc_selected)

    return selected[:n_pages]


def build_table_focus(pages: list[dict], n: int = 50, seed: int = 42) -> list[dict]:
    """Build a sample set focused on pages with tables."""
    rng = random.Random(seed)

    table_pages = [
        p for p in pages
        if p["content_type"] == "table" and not p["is_empty"] and p["has_image"]
    ]

    if len(table_pages) < n:
        # Include pages with number-heavy content too
        multi_col = [
            p for p in pages
            if p["content_type"] == "multi_column" and not p["is_empty"] and p["has_image"]
            and p not in table_pages
        ]
        table_pages.extend(multi_col)

    rng.shuffle(table_pages)
    return table_pages[:n]


def build_full_benchmark(pages: list[dict], n: int = 300, seed: int = 42) -> list[dict]:
    """Build the full benchmark set with broad coverage."""
    rng = random.Random(seed)

    candidates = [p for p in pages if not p["is_empty"] and p["has_image"]]

    if not candidates:
        print("Warning: no candidate pages for full_benchmark set")
        return []

    # Stratify by content type
    by_type = {}
    for p in candidates:
        by_type.setdefault(p["content_type"], []).append(p)

    selected = []
    types = list(by_type.keys())
    per_type = n // len(types) if types else n

    for t in types:
        pool = by_type[t]
        rng.shuffle(pool)
        selected.extend(pool[:per_type])

    # Fill remaining
    rng.shuffle(candidates)
    for p in candidates:
        if len(selected) >= n:
            break
        if p not in selected:
            selected.append(p)

    return selected[:n]


def save_sample_set(
    pages: list[dict],
    output_path: str | Path,
    name: str,
    description: str,
):
    """Save a sample set as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Summarize content type distribution
    type_dist = {}
    for p in pages:
        type_dist[p["content_type"]] = type_dist.get(p["content_type"], 0) + 1

    sample_set = {
        "name": name,
        "description": description,
        "total_pages": len(pages),
        "unique_documents": len(set(p["pdf_stem"] for p in pages)),
        "content_type_distribution": type_dist,
        "pages": [
            {
                "pdf_stem": p["pdf_stem"],
                "page_num": p["page_num"],
                "content_type": p["content_type"],
                "position": p["position"],
            }
            for p in pages
        ],
    }

    with open(output_path, "w") as f:
        json.dump(sample_set, f, indent=2)

    print(f"Saved {name}: {len(pages)} pages from {sample_set['unique_documents']} docs")
    print(f"  Content types: {type_dist}")


def build_all_sample_sets(
    embedded_text_dir: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
):
    """Build all sample sets from the corpus."""
    print("Building page index...")
    pages = build_page_index(embedded_text_dir, image_dir)
    print(f"Indexed {len(pages)} pages")

    output_dir = Path(output_dir)

    # Quick dev set
    quick = build_quick_dev(pages, n=20, seed=seed)
    save_sample_set(quick, output_dir / "quick_dev.json", "quick_dev",
                    "Adapter development and debugging")

    # Stratified 100
    strat = build_stratified(pages, n_pages=100, n_docs=20, pages_per_doc=5, seed=seed)
    save_sample_set(strat, output_dir / "stratified_100.json", "stratified_100",
                    "Primary benchmark set")

    # Table focus
    tables = build_table_focus(pages, n=50, seed=seed)
    save_sample_set(tables, output_dir / "table_focus.json", "table_focus",
                    "Pages with tables for TEDS evaluation")

    # Full benchmark
    full = build_full_benchmark(pages, n=300, seed=seed)
    save_sample_set(full, output_dir / "full_benchmark.json", "full_benchmark",
                    "Comprehensive comparison set")


def main():
    config = load_config()
    build_all_sample_sets(
        embedded_text_dir=config["paths"]["embedded_text_dir"],
        image_dir=config["paths"]["image_dir"],
        output_dir=config["paths"]["sample_sets_dir"],
        seed=config["sampling"]["random_seed"],
    )


if __name__ == "__main__":
    main()
