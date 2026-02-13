# OCR Benchmark: Open-Weight Models on Australian Insurance Documents

A benchmarking framework comparing **22 open-weight OCR models** against 97 Australian insurance Product Disclosure Statement (PDS) PDFs (~7,500 pages). The PDFs contain native embedded text, providing free ground truth for measuring text extraction accuracy without manual annotation.

## Why This Benchmark?

Most OCR benchmarks use academic datasets (receipts, handwriting, scene text). Real-world document processing — especially in regulated industries like insurance — involves **dense multi-column layouts, nested tables, legal boilerplate, and mixed formatting**. This benchmark fills that gap by evaluating models on genuine insurance documents that represent the kind of content organisations actually need to extract.

### What We Measure

- **Text accuracy**: Normalized Edit Distance, Character/Word Error Rate, BLEU, fuzzy matching
- **Table extraction**: Tree Edit Distance Similarity (TEDS) with greedy table matching
- **Coverage**: percentage of pages a model successfully processes (failures scored as worst-case)
- **Robustness**: accuracy under image perturbations (rotation, blur, JPEG compression, downscaling)
- **Confidence intervals**: bootstrap CIs on all aggregate metrics

## Models (22 total)

### Tier 1 — T4-compatible (≤8 GB VRAM)

| Model | Parameters | Source | Notes |
|-------|-----------|--------|-------|
| [LightOn OCR 2](https://huggingface.co/lightonai/LightOnOCR-2-1B) | 1B | LightOn | Strong benchmark leader, clean API |
| [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR2_0) | 0.6B | StepFun | Smallest VLM in the set |
| [dots.ocr](https://huggingface.co/NexaAIDev/dots.llm1.base) | 1.6B | NexaAI | Document-specific small model |
| [DeepSeek-OCR](https://huggingface.co/DeepSeek-OCR/DeepSeek-OCR) | 2B | DeepSeek | Dedicated OCR model |
| [Nanonets-OCR-s](https://huggingface.co/nanonets/Nanonets-OCR-s) | 3B | Nanonets | Markdown output |
| [OCRFlux-3B](https://huggingface.co/Carkham/OCRFlux-3B) | 3B | Carkham | Qwen2.5-VL architecture |
| [Florence-2](https://huggingface.co/microsoft/Florence-2-large) | 0.77B | Microsoft | Vision-language model with OCR tasks |
| [Granite Vision 3.3](https://huggingface.co/ibm-granite/granite-vision-3.3-2b) | 2B | IBM | General vision model |
| [Granite-Docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M) | 258M | IBM | Small document conversion model |
| [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR) | 1.2B | — | Specialized document OCR |
| [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) | 0.9B | PaddlePaddle | Strong layout/table parsing |
| [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | 3B | Alibaba | Smaller Qwen2.5-VL variant |
| [DocTR](https://github.com/mindee/doctr) | ~25M | Mindee | Detection + recognition pipeline |

### Tier 2 — A100 required (>8 GB VRAM)

| Model | Parameters | Source | Notes |
|-------|-----------|--------|-------|
| [olmOCR](https://huggingface.co/allenai/olmOCR-7B-0225-preview) | 7B | Allen AI | Uses custom toolkit |
| [RolmOCR](https://huggingface.co/reducto/RolmOCR) | 7B | Reducto | Refined olmOCR |
| [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 7B | Alibaba | General VLM, strong OCR capability |
| [Chandra](https://huggingface.co/adarshxs/Chandra) | 7B | — | Document-specialized VLM |

### Traditional / CPU Baselines

| Model | Notes |
|-------|-------|
| [Tesseract](https://github.com/tesseract-ocr/tesseract) | Classic OCR baseline |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Baidu's detection + recognition toolkit |
| [EasyOCR](https://github.com/JaidedAI/EasyOCR) | 80+ language support |
| [Surya](https://github.com/VikParuchuri/surya) | Scaffold — integration pending |
| [MinerU](https://github.com/opendatalab/MinerU) | Scaffold — integration pending |

## Architecture

```
PDF corpus  ──►  Data Prep (render PNGs + extract ground truth)
                        │
                        ▼
              Sample Sets (stratified page manifests)
                        │
                        ▼
              Runner (adapters load/infer/unload sequentially)
                        │
                        ▼
              Evaluator (metrics + bootstrap CIs + ranking)
                        │
                        ▼
              Dashboard (Plotly charts + qualitative HTML viewer)
```

All models implement a common `OCRAdapter` interface with `load_model()` / `ocr_page()` / `unload_model()`. The runner loads one model at a time, runs inference across a sample set, saves per-page markdown outputs, then unloads before the next model. A canonical inference protocol (deterministic decoding, shared prompt) ensures fair comparison.

## Ground Truth

Ground truth is extracted from the PDFs' embedded text layer using **pdfplumber** and **PyMuPDF** with best-of selection per page:

- **Both agree** (71% of pages where PyMuPDF's column-by-column reading order is preferred)
- **pdfplumber preferred** when it captures more text (8% of pages)
- **Both identical** (21% of pages)

This dual-extraction approach handles the multi-column layouts common in insurance documents, where single-extractor approaches often scramble reading order.

## Sample Sets

| Set | Pages | Documents | Purpose |
|-----|-------|-----------|---------|
| `quick_dev` | 20 | 16 | Fast adapter development and debugging |
| `stratified_100` | 100 | 20 | Primary benchmark set |
| `table_focus` | 50 | 13 | Table extraction evaluation |
| `full_benchmark` | 300 | 84 | Comprehensive comparison |

## Getting Started

### Installation

```bash
# Core (PDF processing + evaluation metrics)
pip install -e .

# Add visualisation
pip install -e ".[viz]"

# Add traditional OCR baselines (DocTR, PaddleOCR, EasyOCR, Tesseract)
pip install -e ".[traditional]"

# For Colab GPU runs
pip install -r requirements/colab_tier1.txt   # T4
pip install -r requirements/colab_tier2.txt   # A100
```

### Verify Adapters

```bash
# Quick: import + instantiate all adapters
python scripts/smoke_adapters.py

# Test specific models
python scripts/smoke_adapters.py --models tesseract qwen25_vl_3b

# Deep: instantiate + load/unload (needs GPU for VLMs)
python scripts/smoke_adapters.py --load-model
```

### Run Tests

```bash
pytest tests/
```

## Notebook Workflow

The benchmark is structured as a sequence of notebooks, designed to run data prep locally and GPU inference on Colab:

| # | Notebook | Environment | Purpose |
|---|----------|-------------|---------|
| 00 | `data_prep` | Local | Unzip PDFs, render page images, extract ground truth |
| 01 | `build_sample_sets` | Local | Create stratified sample set manifests |
| 02 | `tier1_benchmark` | Colab T4 | Run Tier 1 models (≤8 GB) |
| 03 | `tier2_benchmark` | Colab A100 | Run Tier 2 models (7B) |
| 04 | `traditional_baselines` | CPU | Run CPU/toolkit baselines |
| 05 | `evaluation` | Local | Compute all metrics vs ground truth |
| 06 | `results_dashboard` | Local | Generate comparison charts and viewer |

## Adding a New Model

1. Add an entry to `config/model_registry.yaml`
2. Create an adapter in `src/adapters/` extending `OCRAdapter`
3. Implement `load_model()` and `ocr_page()` using the base class helpers (`_get_instruction()`, `_get_generation_kwargs()`, `_resolve_device()`, `_extract_tables()`)
4. The runner picks it up by model key automatically

## Project Status

- Data preparation (PDFs, images, ground truth, sample sets): **complete**
- Adapter framework (22 models): **complete**
- Runner, evaluator, and dashboard: **complete**
- GPU benchmark runs: **not yet started** — needs Colab T4/A100

## License

This project is open source. See [LICENSE](LICENSE) for details.
