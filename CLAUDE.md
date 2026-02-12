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

### Canonical Inference Protocol

`benchmark_config.yaml` defines an `inference_protocol` section with a canonical instruction and deterministic decoding params. The runner (`runner.py:198-199`) sets `benchmark_instruction` and `benchmark_decoding` on each adapter via `setattr` before `load_model()`.

**Base class helpers** in `src/adapters/base.py`:
- `_get_instruction(default)` — returns `benchmark_instruction` if set, else the adapter's hardcoded default
- `_get_generation_kwargs(**adapter_defaults)` — merges base defaults, adapter-specific params, and benchmark decoding config. Benchmark keys are filtered through `_SAFE_GENERATE_KEYS` allowlist to avoid TypeError on strict generate() signatures.

15 VLM adapters use both helpers. Florence-2 uses `_get_generation_kwargs(num_beams=3)` only (keeps `<OCR>` task token). GOT-OCR2, traditional adapters, and scaffolds are unchanged (no text prompt or HF generate).

### Configuration

`config/benchmark_config.yaml` drives paths, rendering DPI, sampling params, eval metrics, normalization settings, inference protocol, robustness slices, and ranking/uncertainty config. `config/model_registry.yaml` maps model keys to adapter classes and metadata. Both are loaded by most modules via `load_config()`.

### Pipeline Orchestration

`src/pipeline/runner.py` is the main execution engine:
- `run_model_on_sample_set()` loads one adapter, sets benchmark protocol attrs, iterates a sample set JSON, saves per-page `.md` + `_meta.json` + `_tables.json`, then unloads
- `run_all_models()` runs models sequentially with VRAM cleanup between each
- `run_model_robustness_suite()` runs a model across all enabled robustness slices (image transforms: rotation, blur, JPEG compression, downscale)
- `_set_global_determinism(seed)` seeds Python, NumPy, and PyTorch for reproducible runs
- `_apply_robustness_transform(image, transform)` applies configured image perturbations
- Supports `skip_existing=True` for resumable runs; writes `run_manifest.json` per model

### Evaluation

`src/pipeline/evaluator.py` loads raw outputs + ground truth, applies text normalization (`src/evaluation/normalize.py`), then computes:
- Text metrics: NED (rapidfuzz), CER/WER (jiwer), BLEU (sacrebleu), fuzzy ratio
- Table metrics: TEDS via `table-recognition-metric` with greedy table matching
- Composite score: weighted average with re-normalization when components are missing
- Coverage rate: % of pages that produced output (missing pages scored as worst-case)
- Bootstrap confidence intervals: `_bootstrap_mean_ci()` for all aggregate metrics
- Ranking score: `rank_score = (w_text * text_accuracy + w_table * table_accuracy) * coverage_rate`

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
3. Implement `load_model()` and `ocr_page()`:
   - Use `_resolve_device()` for CUDA auto-detection
   - Use `_extract_tables()` for format-aware table parsing
   - Use `self._get_instruction("your default prompt")` instead of hardcoding the prompt
   - Use `**self._get_generation_kwargs()` (or with adapter-specific defaults like `num_beams=3`) instead of hardcoding `max_new_tokens`/`do_sample`
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
- **Run manifest** (`results/raw_outputs/<model>/run_manifest.json`): records canonical instruction, decoding config, sample set, device, timing

## Implementation Status (as of 2026-02-13)

| Component | Status | Notes |
|-----------|--------|-------|
| Data prep (PDFs, images, GT) | Done | 97 PDFs, 7,390 pages, 4 sample sets |
| Adapter framework (22 models) | Done | All importable and instantiable |
| Canonical inference protocol | Done | 15 VLM adapters wired, Florence-2 decoding-only |
| Runner (clean + robustness) | Done | Deterministic seeding, image transforms, manifests |
| Evaluator (metrics + CIs) | Done | Coverage, bootstrap CIs, ranking score |
| Reporting dashboard | Done | Plotly charts + HTML viewer |
| **Actual GPU benchmark runs** | **Not started** | Need Colab T4/A100 to run notebooks 02-04 |
| Evaluation on real outputs | Not started | Blocked on benchmark runs |

### Next Steps

1. **Run Tier 1 models** on Colab T4 (`notebooks/02_tier1_benchmark.ipynb`) with `quick_dev` sample set first
2. **Run Tier 2 models** on Colab A100 (`notebooks/03_tier2_benchmark.ipynb`)
3. **Run traditional baselines** on CPU (`notebooks/04_traditional_baselines.ipynb`)
4. **Run evaluation** (`notebooks/05_evaluation.ipynb`) once raw outputs exist
5. **Generate dashboard** (`notebooks/06_results_dashboard.ipynb`)
6. Optionally run robustness suite after clean runs complete

### Environment Note

The `ocr_benchmark` conda environment does not exist yet on the local machine. Tests were run using the `allytest` env (`~/anaconda3/envs/allytest/bin/python`). Create the env before running GPU workloads:
```bash
conda create -n ocr_benchmark python=3.10
conda activate ocr_benchmark
pip install -e ".[viz,traditional]"
```
