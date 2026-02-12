"""Pipeline runner: orchestrates OCR inference across sample sets."""

import importlib
import json
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

from src.adapters.base import OCRAdapter, OCRResult, InferenceProfile
from src.pipeline.profiler import ModelProfile


def load_config(config_path: str = "config/benchmark_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_registry(registry_path: str = "config/model_registry.yaml") -> dict:
    with open(registry_path) as f:
        return yaml.safe_load(f)


def load_sample_set(sample_set_path: str | Path) -> dict:
    with open(sample_set_path) as f:
        return json.load(f)


def instantiate_adapter(model_key: str, registry: dict, device: str = "auto") -> OCRAdapter:
    """Instantiate an OCR adapter from the model registry.

    Args:
        model_key: Key in model_registry.yaml (e.g., 'lighton_ocr')
        registry: Loaded model registry dict
        device: Device string ('auto', 'cuda', 'cpu')

    Returns:
        OCRAdapter instance (not yet loaded)
    """
    model_info = registry["models"][model_key]
    adapter_path = model_info["adapter_class"]

    # Parse module and class name
    parts = adapter_path.rsplit(".", 1)
    module_path = parts[0]
    class_name = parts[1]

    module = importlib.import_module(module_path)
    adapter_class = getattr(module, class_name)

    return adapter_class(
        model_name=model_info["name"],
        device=device,
    )


def run_model_on_sample_set(
    model_key: str,
    sample_set_path: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    registry: dict | None = None,
    device: str = "auto",
    skip_existing: bool = True,
) -> ModelProfile:
    """Run a single model on a sample set.

    Args:
        model_key: Key in model_registry.yaml
        sample_set_path: Path to sample set JSON
        image_dir: Base directory for page images
        output_dir: Base directory for raw outputs
        registry: Model registry dict (loaded if None)
        device: Device string
        skip_existing: Skip pages that already have output files

    Returns:
        ModelProfile with timing/memory stats
    """
    if registry is None:
        registry = load_model_registry()

    sample_set = load_sample_set(sample_set_path)
    image_dir = Path(image_dir)
    model_output_dir = Path(output_dir) / model_key
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Check what's already done
    pages_to_process = []
    for page_entry in sample_set["pages"]:
        stem = page_entry["pdf_stem"]
        page_num = page_entry["page_num"]
        out_path = model_output_dir / stem / f"page_{page_num:03d}.md"
        if skip_existing and out_path.exists():
            continue
        pages_to_process.append(page_entry)

    if not pages_to_process:
        print(f"{model_key}: All {len(sample_set['pages'])} pages already processed, skipping")
        return ModelProfile(model_name=model_key)

    print(f"{model_key}: Processing {len(pages_to_process)} pages "
          f"({len(sample_set['pages']) - len(pages_to_process)} already done)")

    # Load model
    adapter = instantiate_adapter(model_key, registry, device)
    adapter.load_model()

    profile = ModelProfile(model_name=model_key)

    try:
        for page_entry in tqdm(pages_to_process, desc=f"OCR: {model_key}"):
            stem = page_entry["pdf_stem"]
            page_num = page_entry["page_num"]

            image_path = image_dir / stem / f"page_{page_num:03d}.png"
            if not image_path.exists():
                print(f"Warning: {image_path} not found, skipping")
                continue

            # Run OCR with profiling
            image = Image.open(str(image_path)).convert("RGB")
            result, page_profile = adapter.ocr_page_profiled(image)
            profile.add_page_profile(page_profile)

            # Save output
            page_out_dir = model_output_dir / stem
            page_out_dir.mkdir(parents=True, exist_ok=True)

            out_path = page_out_dir / f"page_{page_num:03d}.md"
            out_path.write_text(result.text, encoding="utf-8")

            # Save metadata
            meta_path = page_out_dir / f"page_{page_num:03d}_meta.json"
            meta = {
                "model": model_key,
                "pdf_stem": stem,
                "page_num": page_num,
                "format": result.format,
                "text_length": result.text_length(),
                "num_tables": len(result.tables),
                "wall_time_seconds": round(page_profile.wall_time_seconds, 3),
                "gpu_memory_peak_mb": round(page_profile.gpu_memory_peak_mb, 1),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            # Save tables if any
            if result.tables:
                tables_path = page_out_dir / f"page_{page_num:03d}_tables.json"
                with open(tables_path, "w") as f:
                    json.dump(result.tables, f, indent=2)

    finally:
        adapter.unload_model()

    # Save model profile
    profile_path = model_output_dir / "profile.json"
    with open(profile_path, "w") as f:
        json.dump(profile.to_dict(), f, indent=2)

    print(f"{model_key}: Done. Avg {profile.avg_time_per_page:.2f}s/page, "
          f"Peak GPU {profile.peak_gpu_memory_mb:.0f}MB")

    return profile


def run_all_models(
    model_keys: list[str],
    sample_set_path: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    skip_existing: bool = True,
) -> dict[str, ModelProfile]:
    """Run multiple models sequentially on a sample set.

    Unloads each model before loading the next to manage VRAM.
    """
    registry = load_model_registry()
    profiles = {}

    for model_key in model_keys:
        print(f"\n{'='*60}")
        print(f"Running {model_key}")
        print(f"{'='*60}")

        try:
            profile = run_model_on_sample_set(
                model_key=model_key,
                sample_set_path=sample_set_path,
                image_dir=image_dir,
                output_dir=output_dir,
                registry=registry,
                device=device,
                skip_existing=skip_existing,
            )
            profiles[model_key] = profile
        except Exception as e:
            print(f"ERROR running {model_key}: {e}")
            profiles[model_key] = ModelProfile(model_name=model_key)

    return profiles
