# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OCR benchmark framework comparing 22 OCR models against 97 Australian insurance PDS PDFs (~7,566 pages). The PDFs are native (embedded text), providing free ground truth for text accuracy evaluation.

## Environment Setup

```bash
conda activate ocr_benchmark  # or your project-specific env
pip install -e .               # core deps (PDF, eval libraries)
pip install -e ".[viz]"        # add plotly/matplotlib for reporting
pip install -e ".[traditional]" # add DocTR, PaddleOCR, EasyOCR
pip install -e ".[toolkits]"   # add Surya + MinerU adapters (optional)
```

For Colab GPU notebooks: `pip install -r requirements/colab_tier1.txt` (T4) or `requirements/colab_tier2.txt` (A100).

Toolkit-focused setup: `pip install -r requirements/toolkits.txt`.

Run tests with: `pytest tests/`.

Adapter smoke check:

```bash
python scripts/smoke_adapters.py              # instantiate adapters only
python scripts/smoke_adapters.py --load-model # instantiate + load/unload
```

## Architecture

### Data Flow

```
PDF zips → [data_prep] → images + ground truth text
                              ↓
         [build_sample_sets] → sample set JSONs (page manifests)
                              ↓
         [adapters via runner] → raw OCR outputs per model per page
                              ↓
         [evaluator] → metrics JSONs + comparison CSV
                              ↓
         [reporting] → Plotly dashboard + qualitative HTML viewer
```

### Adapter Pattern (central abstraction)

All models in `config/model_registry.yaml` implement `OCRAdapter` in `src/adapters/base.py` (some are working scaffolds pending full integration):
- `load_model()` / `unload_model()` — lazy loading for VRAM management; models run sequentially with explicit unload between each
- `ocr_page(PIL.Image) -> OCRResult` — single-page inference returning text + extracted tables
- `ocr_page_profiled()` — wraps `ocr_page` with GPU memory and wall-time tracking

Models are registered in `config/model_registry.yaml` with their HF ID, adapter class path, tier, VRAM requirement, and API pattern. The runner dynamically imports adapters via `importlib`.

**API patterns across adapters:**
- `chat_template` — standard `processor.apply_chat_template()` (LightOn, Nanonets, Granite, Qwen, Chandra)
- `custom_method` — model-specific `.chat()` or task prompts (GOT-OCR2, Florence-2, MonkeyOCR, dots.ocr, DeepSeek)
- `toolkit_wrapped` — Qwen2.5-VL backbone with custom prompting or toolkit integration (olmOCR, RolmOCR, OCRFlux, MinerU scaffold)
- `traditional` — detection + recognition pipeline or classic OCR engines (DocTR, PaddleOCR, EasyOCR, Tesseract, Surya scaffold)

### Configuration

`config/benchmark_config.yaml` drives paths, rendering DPI, sampling params, eval metrics, and normalization settings. `config/model_registry.yaml` maps model keys to adapter classes and metadata. Both are loaded by most modules via `load_config()`.

### Pipeline Orchestration

`src/pipeline/runner.py` is the main execution engine:
- `run_model_on_sample_set()` loads one adapter, iterates a sample set JSON, saves per-page `.md` + `_meta.json` + `_tables.json`, then unloads
- `run_all_models()` runs models sequentially with VRAM cleanup between each
- Supports `skip_existing=True` for resumable runs

### Evaluation

`src/pipeline/evaluator.py` loads raw outputs + ground truth, applies text normalization (`src/evaluation/normalize.py`), then computes:
- Text metrics: NED (rapidfuzz), CER/WER (jiwer), BLEU (sacrebleu), fuzzy ratio
- Table metrics: TEDS via `table-recognition-metric` with greedy table matching
- Composite score: weighted average with re-normalization when components are missing

### Ground Truth

Dual extraction from embedded PDF text using pdfplumber + PyMuPDF with **best-of selection** per page (multiprocessed across 8 cores):
- **Both agree** (NED <= 5%): use pdfplumber (21% of pages)
- **Reading order difference** (high NED, similar char count): use PyMuPDF — it reads column-by-column which is correct for multi-column insurance layouts (71% of pages)
- **pdfplumber has more text**: use pdfplumber (8% of pages)
- **Artifact detection**: pdfplumber sometimes extracts InDesign production metadata (doubled chars); these are detected and PyMuPDF is used instead
- Both original extractions are saved alongside for auditing on discrepancy pages

The extraction script (`src/data_prep/extract_embedded_text.py`) uses `concurrent.futures.ProcessPoolExecutor` for parallel processing.

## Adding a New Model

1. Add entry to `config/model_registry.yaml` with `hf_id`, `adapter_class`, `tier`, `vram_gb`, `api_pattern`
2. Create adapter file in `src/adapters/` extending `OCRAdapter`
3. Implement `load_model()` and `ocr_page()` — use `_resolve_device()` for CUDA auto-detection, `_extract_tables()` for format-aware table parsing
4. The runner will pick it up by model key automatically

## Notebook Workflow (sequential)

| # | Notebook | Where | Purpose |
|---|----------|-------|---------|
| 00 | data_prep | Local | Unzip PDFs, render PNGs, extract GT text |
| 01 | build_sample_sets | Local | Create stratified sample set JSONs |
| 02 | tier1_benchmark | Colab T4 | Run Tier 1 models |
| 03 | tier2_benchmark | Colab A100 | Run 4 Tier 2 models (7B) |
| 04 | traditional_baselines | CPU | Run CPU/toolkit baselines |
| 05 | evaluation | Local | Compute all metrics vs GT |
| 06 | results_dashboard | Local | Generate comparison charts |

## Data Prep Status (completed 2026-02-12)

| Step | Result |
|------|--------|
| PDF extraction | 97 PDFs from 5 zip batches |
| Page rendering | 7,390 pages at 200 DPI PNG |
| Ground truth | 7,390 pages (best-of selection: 71% PyMuPDF, 21% both agree, 8% pdfplumber, 29 empty) |
| Sample sets | 4 sets built (see below) |

### Sample Sets

| Set | Pages | Docs | Content Distribution |
|-----|-------|------|----------------------|
| `quick_dev` | 20 | 16 | 7 text, 4 table, 4 list, 4 sparse, 1 multi_column |
| `stratified_100` | 100 | 20 | 52 text, 37 list, 6 table, 5 sparse |
| `table_focus` | 50 | 13 | 50 table |
| `full_benchmark` | 300 | 84 | 137 text, 63 table, 62 list, 37 sparse, 1 multi_column |

## Key Data Structures

- **Sample sets** (`data/sample_sets/*.json`): JSON with `pages` array of `{pdf_stem, page_num, content_type, position}`
- **Raw outputs** (`results/raw_outputs/<model>/<stem>/page_NNN.md`): OCR text per page
- **Metrics** (`results/metrics/<model>/text_metrics.json`): per-page and aggregate scores
- **GT metadata** (`data/ground_truth/embedded_text/metadata.json`): global stats + per-PDF source selection counts
- **Per-PDF GT metadata** (`data/ground_truth/embedded_text/<stem>/_extraction_meta.json`): per-page NED, selected source, char counts
