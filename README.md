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

## Benchmark Protocol

- Deterministic inference is controlled in `config/benchmark_config.yaml` under `inference_protocol`.
- Evaluation now penalizes missing predictions/GT pages with worst-case text metrics and reports:
  - `coverage_rate`
  - `failure_rate`
  - `rank_score` (text/table quality weighted by coverage)
- Bootstrap confidence intervals for aggregate metrics are enabled via:
  - `evaluation.uncertainty.bootstrap_resamples`
  - `evaluation.uncertainty.confidence_level`

## Robustness Slices

Robustness perturbations are configured under `robustness_slices` in `config/benchmark_config.yaml`.

To run a model on one slice using pipeline APIs:

```python
from src.pipeline.runner import run_model_on_sample_set, load_model_registry, load_config

config = load_config("config/benchmark_config.yaml")
registry = load_model_registry("config/model_registry.yaml")
slice_cfg = config["robustness_slices"]["rotate_2deg"]

run_model_on_sample_set(
    model_key="tesseract",
    sample_set_path="data/sample_sets/quick_dev.json",
    image_dir=config["paths"]["image_dir"],
    output_dir=config["paths"]["raw_outputs_dir"],
    registry=registry,
    device="cpu",
    config=config,
    run_label="rotate_2deg",
    robustness_transform=slice_cfg,
)
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
