"""Qualitative comparison report: side-by-side HTML viewer for OCR outputs.

Generates a self-contained HTML page that shows, for each sample page:
  - The original page image
  - The ground truth text
  - Each model's OCR output in columns
  - Color-highlighted differences (using difflib)
  - Keyboard navigation between pages

Usage:
    from src.reporting.qualitative import generate_comparison_report
    generate_comparison_report(
        sample_set_path="data/sample_sets/quick_dev.json",
        model_keys=["doctr", "lighton_ocr", "got_ocr2"],
        raw_outputs_dir="results/raw_outputs",
        gt_text_dir="data/embedded_text",
        image_dir="data/page_images",
        output_path="results/qualitative/comparison.html",
    )
"""

import base64
import difflib
import html as html_mod
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Diff highlighting
# ---------------------------------------------------------------------------

def _diff_to_html(reference: str, candidate: str) -> str:
    """Produce an HTML fragment highlighting differences between *reference* and *candidate*.

    Uses ``difflib.SequenceMatcher`` at the line level.  Lines present only in the
    reference are shown with a red strikethrough; lines present only in the
    candidate are shown with a green background; changed lines show the old text
    struck through in red followed by the new text in green.
    """
    ref_lines = reference.splitlines(keepends=True)
    cand_lines = candidate.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, ref_lines, cand_lines)

    parts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for line in ref_lines[i1:i2]:
                parts.append(html_mod.escape(line))
        elif tag == "replace":
            for line in ref_lines[i1:i2]:
                parts.append(
                    f'<span class="diff-del">{html_mod.escape(line)}</span>'
                )
            for line in cand_lines[j1:j2]:
                parts.append(
                    f'<span class="diff-add">{html_mod.escape(line)}</span>'
                )
        elif tag == "delete":
            for line in ref_lines[i1:i2]:
                parts.append(
                    f'<span class="diff-del">{html_mod.escape(line)}</span>'
                )
        elif tag == "insert":
            for line in cand_lines[j1:j2]:
                parts.append(
                    f'<span class="diff-add">{html_mod.escape(line)}</span>'
                )

    return "".join(parts)


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def _encode_image_base64(image_path: Path) -> str | None:
    """Read an image file and return a data-URI string, or None if missing."""
    if not image_path.exists():
        return None
    data = image_path.read_bytes()
    suffix = image_path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".bmp": "image/bmp",
    }.get(suffix, "image/png")
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


# ---------------------------------------------------------------------------
# Per-page HTML generation
# ---------------------------------------------------------------------------

def generate_page_comparison(
    pdf_stem: str,
    page_num: int,
    model_keys: list[str],
    raw_outputs_dir: str | Path,
    gt_text_dir: str | Path,
    image_dir: str | Path,
) -> str:
    """Return an HTML section for one page showing image, GT, and model outputs.

    Args:
        pdf_stem: PDF document stem name.
        page_num: 1-based page number.
        model_keys: List of model identifiers.
        raw_outputs_dir: Base directory with ``{model}/{pdf_stem}/page_NNN.md``.
        gt_text_dir: Directory with ``{pdf_stem}/page_NNN.txt`` ground truth.
        image_dir: Directory with ``{pdf_stem}/page_NNN.png`` page images.

    Returns:
        An HTML string (a ``<section>`` element) for this page comparison.
    """
    raw_outputs_dir = Path(raw_outputs_dir)
    gt_text_dir = Path(gt_text_dir)
    image_dir = Path(image_dir)

    page_tag = f"page_{page_num:03d}"
    page_id = f"{pdf_stem}_{page_tag}"

    # Load ground truth
    gt_path = gt_text_dir / pdf_stem / f"{page_tag}.txt"
    gt_text = gt_path.read_text(encoding="utf-8") if gt_path.exists() else "[Ground truth not available]"

    # Load image
    image_path = image_dir / pdf_stem / f"{page_tag}.png"
    image_data_uri = _encode_image_base64(image_path)
    if image_data_uri:
        image_html = f'<img src="{image_data_uri}" alt="{page_id}" class="page-img" />'
    else:
        image_html = '<div class="no-image">Image not available</div>'

    # Load model outputs and generate diff highlights
    model_columns: list[str] = []
    for model_key in model_keys:
        pred_path = raw_outputs_dir / model_key / pdf_stem / f"{page_tag}.md"
        if pred_path.exists():
            pred_text = pred_path.read_text(encoding="utf-8")
            diff_html = _diff_to_html(gt_text, pred_text)
        else:
            pred_text = ""
            diff_html = '<span class="missing">Output not available</span>'

        col = (
            f'<div class="model-col">\n'
            f'  <h4>{html_mod.escape(model_key)}</h4>\n'
            f'  <div class="toggle-bar">\n'
            f'    <button onclick="toggleView(this, \'raw\')">Raw</button>\n'
            f'    <button class="active" onclick="toggleView(this, \'diff\')">Diff</button>\n'
            f'  </div>\n'
            f'  <pre class="model-output diff-view">{diff_html}</pre>\n'
            f'  <pre class="model-output raw-view" style="display:none">'
            f'{html_mod.escape(pred_text)}</pre>\n'
            f'</div>'
        )
        model_columns.append(col)

    models_html = "\n".join(model_columns)

    # Determine grid column count:  image + GT + N models
    total_cols = 2 + len(model_keys)
    col_template = f"300px 1fr {'1fr ' * len(model_keys)}".strip()

    return f"""
<section class="page-section" id="{html_mod.escape(page_id)}">
    <h3 class="page-title">{html_mod.escape(pdf_stem)} &mdash; Page {page_num}</h3>
    <div class="page-grid" style="grid-template-columns: {col_template};">
        <div class="image-col">
            <h4>Page Image</h4>
            {image_html}
        </div>
        <div class="gt-col">
            <h4>Ground Truth</h4>
            <pre class="gt-text">{html_mod.escape(gt_text)}</pre>
        </div>
        {models_html}
    </div>
</section>
"""


# ---------------------------------------------------------------------------
# Full report generation
# ---------------------------------------------------------------------------

def generate_comparison_report(
    sample_set_path: str | Path,
    model_keys: list[str],
    raw_outputs_dir: str | Path,
    gt_text_dir: str | Path,
    image_dir: str | Path,
    output_path: str | Path,
    max_pages: int = 20,
) -> None:
    """Generate a full self-contained HTML comparison report.

    Args:
        sample_set_path: Path to a sample set JSON file (as produced by
            ``build_sample_sets``).
        model_keys: List of model identifiers to include.
        raw_outputs_dir: Base directory containing model output subdirectories.
        gt_text_dir: Directory containing ground truth text files.
        image_dir: Directory containing page image files.
        output_path: File path where the HTML report will be written.
        max_pages: Maximum number of pages to include in the report.
    """
    sample_set_path = Path(sample_set_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(sample_set_path) as f:
        sample_set = json.load(f)

    pages = sample_set.get("pages", [])[:max_pages]

    print(f"Generating qualitative comparison for {len(pages)} pages, "
          f"{len(model_keys)} models...")

    # Build page sections
    page_sections: list[str] = []
    page_nav_items: list[str] = []

    for idx, page_entry in enumerate(pages):
        pdf_stem = page_entry["pdf_stem"]
        page_num = page_entry["page_num"]
        content_type = page_entry.get("content_type", "unknown")
        page_id = f"{pdf_stem}_page_{page_num:03d}"

        section_html = generate_page_comparison(
            pdf_stem=pdf_stem,
            page_num=page_num,
            model_keys=model_keys,
            raw_outputs_dir=raw_outputs_dir,
            gt_text_dir=gt_text_dir,
            image_dir=image_dir,
        )
        page_sections.append(section_html)

        label = f"{pdf_stem} p{page_num} ({content_type})"
        page_nav_items.append(
            f'<a href="#{html_mod.escape(page_id)}" class="nav-item" '
            f'data-index="{idx}">{html_mod.escape(label)}</a>'
        )

    sections_html = "\n".join(page_sections)
    nav_html = "\n            ".join(page_nav_items)
    models_header = ", ".join(model_keys)

    full_html = _build_report_html(
        sections_html=sections_html,
        nav_html=nav_html,
        models_header=models_header,
        n_pages=len(pages),
        n_models=len(model_keys),
        sample_set_name=sample_set.get("name", "unknown"),
    )

    output_path.write_text(full_html, encoding="utf-8")
    print(f"Saved qualitative report -> {output_path}")


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def _build_report_html(
    sections_html: str,
    nav_html: str,
    models_header: str,
    n_pages: int,
    n_models: int,
    sample_set_name: str,
) -> str:
    """Assemble the complete self-contained HTML report."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR Qualitative Comparison - {html_mod.escape(sample_set_name)}</title>
    <style>
        /* ---- Reset & Base ---- */
        *, *::before, *::after {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 0;
            background: #f0f2f5; color: #1a1a2e;
        }}

        /* ---- Header ---- */
        header {{
            background: #2c3e50; color: white;
            padding: 18px 30px;
            position: sticky; top: 0; z-index: 100;
            display: flex; align-items: center; justify-content: space-between;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        header h1 {{ margin: 0; font-size: 1.3em; }}
        header .meta {{ font-size: 0.85em; opacity: 0.8; }}

        /* ---- Navigation Sidebar ---- */
        .layout {{
            display: flex; min-height: calc(100vh - 60px);
        }}
        nav.sidebar {{
            width: 260px; min-width: 260px;
            background: #34495e; color: white;
            padding: 15px 10px;
            overflow-y: auto;
            position: sticky; top: 60px;
            height: calc(100vh - 60px);
        }}
        nav.sidebar h3 {{
            margin: 0 0 12px 5px; font-size: 0.9em;
            text-transform: uppercase; opacity: 0.7;
        }}
        nav.sidebar .nav-item {{
            display: block; padding: 8px 12px; margin: 3px 0;
            color: #bdc3c7; text-decoration: none;
            border-radius: 5px; font-size: 0.82em;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            transition: background 0.15s;
        }}
        nav.sidebar .nav-item:hover {{ background: #3d566e; color: white; }}
        nav.sidebar .nav-item.active {{ background: #3498db; color: white; }}

        /* ---- Main Content ---- */
        main {{
            flex: 1; padding: 20px 25px;
            overflow-x: auto;
        }}

        /* ---- Page Section ---- */
        .page-section {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            margin-bottom: 30px;
            padding: 20px;
        }}
        .page-title {{
            margin: 0 0 15px 0;
            font-size: 1.1em;
            color: #2c3e50;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}

        /* ---- Grid ---- */
        .page-grid {{
            display: grid;
            gap: 15px;
            align-items: start;
        }}

        .page-grid h4 {{
            margin: 0 0 8px 0; font-size: 0.85em;
            text-transform: uppercase; color: #7f8c8d;
        }}

        .image-col {{ position: sticky; top: 80px; }}
        .page-img {{
            width: 100%; max-width: 300px;
            border: 1px solid #ddd; border-radius: 4px;
        }}
        .no-image {{
            width: 280px; height: 360px;
            background: #ecf0f1; display: flex;
            align-items: center; justify-content: center;
            color: #95a5a6; border-radius: 4px;
        }}

        /* ---- Text Panels ---- */
        pre {{
            white-space: pre-wrap; word-wrap: break-word;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 0.78em; line-height: 1.45;
            margin: 0; padding: 12px;
            background: #fafbfc; border: 1px solid #e1e4e8;
            border-radius: 6px;
            max-height: 600px; overflow-y: auto;
        }}
        .gt-text {{ background: #f0faf0; border-color: #c3e6cb; }}

        /* ---- Diff Highlighting ---- */
        .diff-del {{
            background: #fdd; color: #900;
            text-decoration: line-through;
        }}
        .diff-add {{
            background: #dfd; color: #060;
        }}
        .missing {{
            color: #95a5a6; font-style: italic;
        }}

        /* ---- Toggle Buttons ---- */
        .toggle-bar {{
            margin-bottom: 6px;
        }}
        .toggle-bar button {{
            font-size: 0.72em; padding: 3px 10px;
            border: 1px solid #bdc3c7; border-radius: 3px;
            background: #ecf0f1; color: #555; cursor: pointer;
            margin-right: 4px;
        }}
        .toggle-bar button.active {{
            background: #3498db; color: white; border-color: #3498db;
        }}

        /* ---- Keyboard Shortcut Hint ---- */
        .kbd-hint {{
            text-align: center; padding: 10px;
            font-size: 0.8em; color: #95a5a6;
        }}
        kbd {{
            background: #eee; border: 1px solid #ccc;
            border-radius: 3px; padding: 2px 6px;
            font-family: monospace; font-size: 0.9em;
        }}

        /* ---- Footer ---- */
        footer {{
            text-align: center; padding: 20px;
            color: #95a5a6; font-size: 0.8em;
        }}
    </style>
</head>
<body>

<header>
    <h1>OCR Qualitative Comparison</h1>
    <div class="meta">
        Sample set: <strong>{html_mod.escape(sample_set_name)}</strong>
        &nbsp;|&nbsp; {n_pages} pages &nbsp;|&nbsp; {n_models} models ({html_mod.escape(models_header)})
    </div>
</header>

<div class="layout">
    <nav class="sidebar" id="sidebar">
        <h3>Pages</h3>
        {nav_html}
    </nav>

    <main id="main-content">
        <div class="kbd-hint">
            Navigate: <kbd>&larr;</kbd> Previous &nbsp; <kbd>&rarr;</kbd> Next
        </div>

        {sections_html}

        <footer>Generated by OCR Benchmark Framework</footer>
    </main>
</div>

<script>
(function() {{
    // --- Keyboard navigation ---
    const sections = document.querySelectorAll('.page-section');
    const navItems = document.querySelectorAll('.nav-item');
    let currentIdx = 0;

    function scrollToSection(idx) {{
        if (idx < 0 || idx >= sections.length) return;
        currentIdx = idx;
        sections[idx].scrollIntoView({{ behavior: 'smooth', block: 'start' }});
        navItems.forEach(n => n.classList.remove('active'));
        if (navItems[idx]) navItems[idx].classList.add('active');
    }}

    document.addEventListener('keydown', function(e) {{
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {{
            e.preventDefault();
            scrollToSection(currentIdx + 1);
        }} else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {{
            e.preventDefault();
            scrollToSection(currentIdx - 1);
        }}
    }});

    // Highlight active nav item on scroll
    const observer = new IntersectionObserver(entries => {{
        entries.forEach(entry => {{
            if (entry.isIntersecting) {{
                const idx = Array.from(sections).indexOf(entry.target);
                if (idx >= 0) {{
                    currentIdx = idx;
                    navItems.forEach(n => n.classList.remove('active'));
                    if (navItems[idx]) navItems[idx].classList.add('active');
                }}
            }}
        }});
    }}, {{ rootMargin: '-80px 0px -60% 0px' }});

    sections.forEach(s => observer.observe(s));

    // Initialize first item as active
    if (navItems[0]) navItems[0].classList.add('active');
}})();

// --- Toggle raw / diff view ---
function toggleView(btn, mode) {{
    const col = btn.closest('.model-col');
    const diffPre = col.querySelector('.diff-view');
    const rawPre = col.querySelector('.raw-view');
    const buttons = col.querySelectorAll('.toggle-bar button');

    buttons.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    if (mode === 'raw') {{
        diffPre.style.display = 'none';
        rawPre.style.display = 'block';
    }} else {{
        diffPre.style.display = 'block';
        rawPre.style.display = 'none';
    }}
}}
</script>

</body>
</html>
"""
