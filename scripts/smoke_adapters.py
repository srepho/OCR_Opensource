#!/usr/bin/env python3
"""Smoke-check adapter registry entries.

Default behavior only verifies that adapters can be instantiated from
`config/model_registry.yaml` (fast, no model downloads).

Use `--load-model` to call `load_model()` as well (slow; may download weights).
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
import sys

# Make `src/` importable when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check OCR adapters")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("config/model_registry.yaml"),
        help="Path to model registry YAML",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Specific model keys to test (default: all)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device passed to adapter instantiation (default: auto)",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Also run adapter.load_model() and unload_model()",
    )
    return parser.parse_args()


def main() -> int:
    try:
        from src.pipeline.runner import instantiate_adapter, load_model_registry
    except Exception as e:  # noqa: BLE001
        print(
            "Failed to import pipeline modules required for smoke checks. "
            "Install project deps first (e.g. `pip install -e .`)."
        )
        print(f"Import error: {e}")
        return 1

    args = parse_args()
    registry = load_model_registry(str(args.registry))

    all_model_keys = sorted(registry.get("models", {}).keys())
    model_keys = args.models if args.models else all_model_keys

    if not model_keys:
        print("No models found in registry.")
        return 1

    failures: list[tuple[str, str]] = []

    for model_key in model_keys:
        if model_key not in registry["models"]:
            msg = f"Model key not in registry: {model_key}"
            print(f"[FAIL] {model_key}: {msg}")
            failures.append((model_key, msg))
            continue

        print(f"[CHECK] {model_key}")
        adapter = None

        try:
            adapter = instantiate_adapter(model_key, registry, device=args.device)
            print(f"  [OK] instantiate -> {adapter.__class__.__name__}")
        except Exception as e:  # noqa: BLE001
            msg = f"instantiate failed: {e}"
            print(f"  [FAIL] {msg}")
            failures.append((model_key, msg))
            traceback.print_exc(limit=1)
            continue

        if args.load_model:
            try:
                adapter.load_model()
                print("  [OK] load_model")
            except Exception as e:  # noqa: BLE001
                msg = f"load_model failed: {e}"
                print(f"  [FAIL] {msg}")
                failures.append((model_key, msg))
                traceback.print_exc(limit=1)
            finally:
                try:
                    adapter.unload_model()
                except Exception:
                    pass

    print()
    print(f"Checked {len(model_keys)} model(s).")
    if failures:
        print(f"Failures: {len(failures)}")
        for model_key, err in failures:
            print(f"- {model_key}: {err}")
        return 1

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
