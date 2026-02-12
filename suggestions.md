# OCR Benchmark Project Review - Updated Notes

This file tracks review findings and implementation status after the follow-up fixes.

## Status Summary (Updated: 2026-02-12)

- Core review issues were implemented and re-checked.
- Model registry expanded from 16 to 22 models.
- New docs/setup notes were added for toolkits and adapter smoke-checking.
- Remaining planned work is concentrated in Surya/MinerU concrete adapter integration.

## Completed From Review

1. Composite aggregation now preserves valid zero values.
   - Updated: `src/evaluation/composite_score.py`
   - Change: missing values represented as `None`; aggregation excludes only `None`.

2. Table metric error handling narrowed and logged.
   - Updated: `src/evaluation/table_metrics.py`
   - Change: broad catch removed; expected per-table errors now logged with context.

3. CSV comparison export made robust to mixed row schemas.
   - Updated: `src/pipeline/evaluator.py`
   - Change: fieldnames built from union of row keys.

4. Sample set generation edge case guards added.
   - Updated: `src/data_prep/build_sample_sets.py`
   - Change: safe behavior when candidate pools are empty.

5. Tier-2 CUDA requirements made explicit and safer.
   - Updated: `src/adapters/qwen25_vl.py`
   - Updated: `src/adapters/chandra_adapter.py`
   - Updated: `src/adapters/olmocr.py`
   - Updated: `src/adapters/rolmocr.py`
   - Change: fail-fast on non-CUDA environments; `cuda:0`-style devices now accepted where applicable.

6. Configured composite weights are now used at evaluation time.
   - Updated: `src/pipeline/evaluator.py`

7. Unicode normalization form is now honored.
   - Updated: `src/evaluation/normalize.py`

8. ZIP extraction safety and collision handling improved.
   - Updated: `src/data_prep/extract_pdfs.py`
   - Change: member path validation, duplicate filename collision warnings, no raw `extract()` for PDFs.

9. Dependency alignment improved.
   - Updated: `pyproject.toml`
   - Added extras for toolkits and missing adapter dependency (`pytesseract`).

10. Model expansion and scaffolding implemented.
    - Updated: `config/model_registry.yaml` (now 22 models)
    - Added adapters:
      - `src/adapters/qwen25_vl_3b.py`
      - `src/adapters/paddleocr_vl15.py`
      - `src/adapters/granite_docling_258m.py`
      - `src/adapters/tesseract_adapter.py`
      - `src/adapters/surya_adapter.py` (scaffold)
      - `src/adapters/mineru_adapter.py` (scaffold)

11. Operational tooling/documentation added.
    - Added: `scripts/smoke_adapters.py`
    - Added: `requirements/toolkits.txt`
    - Added: `README.md`
    - Updated: `CLAUDE.md`

## Data Preparation (completed 2026-02-12)

12. PDF extraction, page rendering, and ground truth extraction completed.
    - 97 PDFs extracted from 5 zip batches
    - 7,390 pages rendered at 200 DPI PNG
    - 7,390 pages of ground truth text extracted

13. Ground truth extraction upgraded to best-of selection with multiprocessing.
    - Updated: `src/data_prep/extract_embedded_text.py`
    - Change: dual extraction now uses `select_best_text()` to pick the better extractor per page:
      - PyMuPDF selected for 71% of pages (better reading order for multi-column layouts)
      - Both agree on 21% (low NED, pdfplumber used)
      - pdfplumber selected for 8% (more text captured)
      - InDesign artifact detection via `_has_doubled_chars()` regex
    - Change: `ProcessPoolExecutor` with 8 workers (2.8x speedup: 13:47 → 4:50)
    - Change: both original extractions saved alongside best text for auditing
    - Change: per-page metadata now includes `selected_source` and `char_count_best`
    - Fixed: `pyproject.toml` build-backend changed from `setuptools.backends._legacy:_Backend` to `setuptools.build_meta`

14. Sample sets built with improved content classification.
    - 4 sample sets: quick_dev (20), stratified_100 (100), table_focus (50), full_benchmark (300)
    - Table detection improved from 13 → 50 pages after GT fix (PyMuPDF preserves table structure in text)

## Open Items

1. Surya and MinerU adapters are intentionally scaffolds.
   - Files:
     - `src/adapters/surya_adapter.py`
     - `src/adapters/mineru_adapter.py`
   - Next step: implement real OCR execution paths (API/CLI integration + output normalization).

2. Environment setup still required before smoke checks can pass in a fresh shell.
   - Example failure seen: missing `yaml` dependency when running smoke script without `pip install -e .`.

3. Next steps: run OCR benchmarks.
   - Test pipeline end-to-end with Tesseract on `quick_dev` (CPU, local)
   - Run Tier 1 models on Colab T4
   - Run Tier 2 models on Colab A100
   - Run traditional baselines on CPU
   - Evaluate all models and generate dashboard

## Validation Notes

- Syntax checks passed:
  - `python3 -m compileall -q src`
  - `python3 -m compileall -q src scripts`
- Test suite: 60/60 tests passing (`pytest tests/`)
- Full runtime model loads were not executed in this environment due to missing optional dependencies and model weight download constraints.
