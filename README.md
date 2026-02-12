# OCR Benchmark Framework

Benchmark framework for comparing OCR models on Australian insurance PDS PDFs.

## Quick Start

```bash
conda activate ocr_benchmark  # optional
pip install -e .
pip install -e ".[viz]"
pip install -e ".[traditional]"
pip install -e ".[toolkits]"  # optional: Surya + MinerU
```

## Adapter Smoke Testing

Use the smoke script to verify model registry entries and adapter wiring before running full benchmarks.

```bash
# Fast check: import + instantiate adapters from config/model_registry.yaml
python scripts/smoke_adapters.py

# Check a subset
python scripts/smoke_adapters.py --models tesseract qwen25_vl_3b

# Deep check: instantiate + load/unload model
python scripts/smoke_adapters.py --load-model
```

## Expected Failure Modes

- `Import error: No module named 'yaml'`
  - Cause: core project dependencies not installed.
  - Fix: `pip install -e .`

- `load_model failed` for GPU-heavy models
  - Cause: no CUDA GPU available or insufficient VRAM.
  - Fix: run with supported hardware or test CPU-capable models only.

- `Tesseract binary not found in PATH`
  - Cause: `pytesseract` Python package installed, but system `tesseract` executable is missing.
  - Fix (macOS): `brew install tesseract`

- `Surya adapter is a scaffold only` / `MinerU adapter is a scaffold only`
  - Cause: those adapters are currently fail-fast scaffolds.
  - Fix: implement concrete API calls in:
    - `src/adapters/surya_adapter.py`
    - `src/adapters/mineru_adapter.py`
