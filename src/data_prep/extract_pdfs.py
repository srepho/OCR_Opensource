"""Extract PDFs from zip batches into a flat directory."""

import zipfile
from pathlib import Path

import yaml
from tqdm import tqdm


def load_config(config_path: str = "config/benchmark_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_all_batches(
    zip_paths: list[str | Path],
    output_dir: str | Path,
    overwrite: bool = False,
) -> list[Path]:
    """Extract all PDFs from zip batches into a flat output directory.

    Returns list of extracted PDF paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted = []
    seen_filenames: dict[str, str] = {}  # filename -> source zip/member for collision detection

    for zip_path in tqdm(zip_paths, desc="Extracting batches"):
        zip_path = Path(zip_path)
        if not zip_path.exists():
            print(f"Warning: {zip_path} not found, skipping")
            continue

        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if not member.lower().endswith(".pdf"):
                    continue
                # Flatten: use just the filename, not nested dirs
                filename = Path(member).name
                if filename.startswith(".") or filename.startswith("__"):
                    continue

                # Validate: reject path traversal in the raw zip member name
                if Path(member).is_absolute() or ".." in Path(member).parts:
                    print(f"Warning: skipping unsafe path in zip: {member}")
                    continue

                # Detect filename collisions across different zip paths/folders
                source_key = f"{zip_path.name}/{member}"
                if filename in seen_filenames and seen_filenames[filename] != source_key:
                    print(
                        f"Warning: filename collision for '{filename}': "
                        f"already from '{seen_filenames[filename]}', "
                        f"skipping duplicate from '{source_key}'"
                    )
                    continue
                seen_filenames[filename] = source_key

                dest = output_dir / filename
                if dest.exists() and not overwrite:
                    extracted.append(dest)
                    continue
                # Read bytes directly instead of extract() to avoid writing
                # to an attacker-controlled path
                dest.write_bytes(zf.read(member))
                extracted.append(dest)

    print(f"Extracted {len(extracted)} PDFs to {output_dir}")
    return sorted(extracted)


def main():
    config = load_config()
    zip_paths = config.get("zip_batches", [])
    pdf_dir = config["paths"]["pdf_dir"]
    extract_all_batches(zip_paths, pdf_dir)


if __name__ == "__main__":
    main()
