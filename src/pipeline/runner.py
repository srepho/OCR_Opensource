"""Pipeline runner: orchestrates OCR inference across sample sets."""

import importlib
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageFilter
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    def tqdm(iterable, **kwargs):
        return iterable

from src.adapters.base import OCRAdapter, OCRResult, InferenceProfile
from src.pipeline.profiler import ModelProfile


def load_config(config_path: str = "config/benchmark_config.yaml") -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_registry(registry_path: str = "config/model_registry.yaml") -> dict:
    import yaml
    with open(registry_path) as f:
        return yaml.safe_load(f)


def load_sample_set(sample_set_path: str | Path) -> dict:
    with open(sample_set_path) as f:
        return json.load(f)


def _set_global_determinism(seed: int) -> None:
    """Set deterministic seeds for reproducible benchmark runs."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _apply_robustness_transform(image: Image.Image, transform: dict | None) -> Image.Image:
    """Apply an optional robustness transform to a page image."""
    if not transform:
        return image

    out = image.convert("RGB")

    rotate_degrees = transform.get("rotate_degrees")
    if rotate_degrees:
        out = out.rotate(float(rotate_degrees), expand=True, fillcolor="white")

    blur_radius = transform.get("gaussian_blur_radius")
    if blur_radius:
        out = out.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))

    downscale_factor = transform.get("downscale_factor")
    if downscale_factor:
        factor = float(downscale_factor)
        if 0 < factor < 1.0:
            w, h = out.size
            dw = max(1, int(w * factor))
            dh = max(1, int(h * factor))
            out = out.resize((dw, dh), Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BILINEAR)

    jpeg_quality = transform.get("jpeg_quality")
    if jpeg_quality:
        q = int(jpeg_quality)
        q = max(1, min(95, q))
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        with Image.open(buf) as compressed:
            out = compressed.convert("RGB")

    return out


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
    config: dict | None = None,
    run_label: str = "clean",
    robustness_transform: dict | None = None,
    deterministic: bool | None = None,
    seed: int | None = None,
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
        config: Benchmark config (loaded if None)
        run_label: Label for this run (e.g. 'clean', 'rotate_2deg')
        robustness_transform: Optional transform config applied to each input image
        deterministic: Override for deterministic execution (uses config if None)
        seed: Override random seed (uses config if None)

    Returns:
        ModelProfile with timing/memory stats
    """
    if registry is None:
        registry = load_model_registry()
    if config is None:
        config = load_config()

    sample_set = load_sample_set(sample_set_path)
    image_dir = Path(image_dir)
    output_root = Path(output_dir)
    if run_label and run_label != "clean":
        output_root = output_root / run_label
    model_output_dir = output_root / model_key
    model_output_dir.mkdir(parents=True, exist_ok=True)

    protocol = config.get("inference_protocol", {})
    protocol_deterministic = protocol.get("deterministic", True) if deterministic is None else deterministic
    protocol_seed = int(protocol.get("global_seed", 42) if seed is None else seed)
    decoding_cfg = protocol.get("decoding", {})
    canonical_instruction = protocol.get("canonical_instruction", "")

    if protocol_deterministic:
        _set_global_determinism(protocol_seed)

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
    # Adapters can opt-in to consuming these shared benchmark settings.
    setattr(adapter, "benchmark_instruction", canonical_instruction)
    setattr(adapter, "benchmark_decoding", decoding_cfg)
    adapter.load_model()

    profile = ModelProfile(model_name=model_key)

    manifest_path = model_output_dir / "run_manifest.json"
    manifest = {
        "model": model_key,
        "sample_set_path": str(sample_set_path),
        "run_label": run_label,
        "device": device,
        "skip_existing": skip_existing,
        "deterministic": protocol_deterministic,
        "seed": protocol_seed,
        "inference_protocol": {
            "canonical_instruction": canonical_instruction,
            "decoding": decoding_cfg,
        },
        "robustness_transform": robustness_transform or {},
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

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
            image = _apply_robustness_transform(image, robustness_transform)
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
                "run_label": run_label,
                "pdf_stem": stem,
                "page_num": page_num,
                "format": result.format,
                "text_length": result.text_length(),
                "num_tables": len(result.tables),
                "wall_time_seconds": round(page_profile.wall_time_seconds, 3),
                "gpu_memory_peak_mb": round(page_profile.gpu_memory_peak_mb, 1),
                "deterministic": protocol_deterministic,
                "seed": protocol_seed,
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


def run_model_robustness_suite(
    model_key: str,
    sample_set_path: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    slice_configs: dict[str, dict],
    registry: dict | None = None,
    device: str = "auto",
    skip_existing: bool = True,
    config: dict | None = None,
) -> dict[str, ModelProfile]:
    """Run one model across configured robustness slices."""
    if registry is None:
        registry = load_model_registry()
    if config is None:
        config = load_config()

    profiles: dict[str, ModelProfile] = {}
    for slice_name, slice_cfg in slice_configs.items():
        if not slice_cfg.get("enabled", True):
            continue
        print(f"\n[{model_key}] Robustness slice: {slice_name}")
        profile = run_model_on_sample_set(
            model_key=model_key,
            sample_set_path=sample_set_path,
            image_dir=image_dir,
            output_dir=output_dir,
            registry=registry,
            device=device,
            skip_existing=skip_existing,
            config=config,
            run_label=slice_name,
            robustness_transform=slice_cfg if slice_name != "clean" else None,
        )
        profiles[slice_name] = profile

    return profiles


def run_all_models(
    model_keys: list[str],
    sample_set_path: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    skip_existing: bool = True,
    config: dict | None = None,
    run_label: str = "clean",
    robustness_transform: dict | None = None,
) -> dict[str, ModelProfile]:
    """Run multiple models sequentially on a sample set.

    Unloads each model before loading the next to manage VRAM.
    """
    registry = load_model_registry()
    if config is None:
        config = load_config()
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
                config=config,
                run_label=run_label,
                robustness_transform=robustness_transform,
            )
            profiles[model_key] = profile
        except Exception as e:
            print(f"ERROR running {model_key}: {e}")
            profiles[model_key] = ModelProfile(model_name=model_key)

    return profiles
