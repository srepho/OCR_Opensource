"""Evaluator: compute metrics by comparing OCR outputs against ground truth."""

import json
from pathlib import Path

import yaml
from tqdm import tqdm

from src.evaluation.normalize import normalize_text
from src.evaluation.text_metrics import compute_all_metrics, TextMetrics
from src.evaluation.table_metrics import compute_page_table_metrics, TableMetrics
from src.evaluation.composite_score import compute_composite, aggregate_scores, CompositeScore


def load_config(config_path: str = "config/benchmark_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_sample_set(sample_set_path: str | Path) -> dict:
    with open(sample_set_path) as f:
        return json.load(f)


def evaluate_model(
    model_key: str,
    sample_set_path: str | Path,
    raw_outputs_dir: str | Path,
    gt_text_dir: str | Path,
    gt_tables_dir: str | Path | None = None,
    metrics_dir: str | Path | None = None,
    config: dict | None = None,
) -> dict:
    """Evaluate a single model's outputs against ground truth.

    Args:
        model_key: Model identifier
        sample_set_path: Path to sample set JSON
        raw_outputs_dir: Base dir for raw model outputs
        gt_text_dir: Dir with ground truth embedded text
        gt_tables_dir: Dir with ground truth table HTML (optional)
        metrics_dir: Dir to save metric results (optional)
        config: Benchmark config dict

    Returns:
        Dict with per-page and aggregate metrics
    """
    if config is None:
        config = load_config()

    sample_set = load_sample_set(sample_set_path)
    raw_outputs_dir = Path(raw_outputs_dir)
    gt_text_dir = Path(gt_text_dir)
    model_output_dir = raw_outputs_dir / model_key

    norm_config = config.get("normalization", {})

    page_results = []

    for page_entry in tqdm(sample_set["pages"], desc=f"Evaluating {model_key}"):
        stem = page_entry["pdf_stem"]
        page_num = page_entry["page_num"]

        # Load model output
        pred_path = model_output_dir / stem / f"page_{page_num:03d}.md"
        if not pred_path.exists():
            continue
        pred_text = pred_path.read_text(encoding="utf-8")

        # Load ground truth
        gt_path = gt_text_dir / stem / f"page_{page_num:03d}.txt"
        if not gt_path.exists():
            continue
        gt_text = gt_path.read_text(encoding="utf-8")

        # Normalize both
        pred_norm = normalize_text(
            pred_text,
            unicode_form=norm_config.get("unicode", "NFKC"),
            do_strip_markdown=norm_config.get("strip_markdown", True),
            do_collapse_whitespace=norm_config.get("collapse_whitespace", True),
            do_remove_page_numbers=norm_config.get("remove_page_numbers", True),
            do_lowercase=norm_config.get("lowercase", False),
        )
        gt_norm = normalize_text(
            gt_text,
            unicode_form=norm_config.get("unicode", "NFKC"),
            do_strip_markdown=False,  # GT is plain text
            do_collapse_whitespace=norm_config.get("collapse_whitespace", True),
            do_remove_page_numbers=norm_config.get("remove_page_numbers", True),
            do_lowercase=norm_config.get("lowercase", False),
        )

        # Text metrics
        text_metrics = compute_all_metrics(pred_norm, gt_norm)

        # Table metrics (if GT tables available)
        table_metrics = None
        if gt_tables_dir:
            gt_tables_path = Path(gt_tables_dir) / stem / f"page_{page_num:03d}.json"
            if gt_tables_path.exists():
                with open(gt_tables_path) as f:
                    ref_tables = json.load(f)
                # Load predicted tables
                pred_tables_path = model_output_dir / stem / f"page_{page_num:03d}_tables.json"
                pred_tables = []
                if pred_tables_path.exists():
                    with open(pred_tables_path) as f:
                        pred_tables = json.load(f)
                table_metrics = compute_page_table_metrics(pred_tables, ref_tables)

        # Composite score (use configured weights)
        eval_config = config.get("evaluation", {})
        comp_weights = eval_config.get("composite_weights", {})
        composite = compute_composite(
            text_metrics=text_metrics,
            table_metrics=table_metrics,
            weight_text=comp_weights.get("text_accuracy", 0.333),
            weight_table=comp_weights.get("table_accuracy", 0.333),
            weight_layout=comp_weights.get("layout_score", 0.333),
        )

        page_result = {
            "pdf_stem": stem,
            "page_num": page_num,
            "content_type": page_entry.get("content_type", "unknown"),
            "text_metrics": text_metrics.to_dict(),
            "composite": composite.to_dict(),
        }
        if table_metrics:
            page_result["table_metrics"] = table_metrics.to_dict()

        page_results.append(page_result)

    # Aggregate metrics
    composites = [
        CompositeScore(
            text_accuracy=r["composite"].get("text_accuracy"),
            table_accuracy=r["composite"].get("table_accuracy"),
            layout_score=r["composite"].get("layout_score"),
            composite=r["composite"].get("composite", 0.0),
        )
        for r in page_results
    ]
    aggregated = aggregate_scores(composites)

    # Aggregate text metrics
    text_fields = ["ned", "cer", "wer", "bleu", "fuzzy_ratio"]
    import statistics
    text_agg = {}
    for field in text_fields:
        values = [r["text_metrics"][field] for r in page_results]
        if values:
            text_agg[field] = {
                "mean": round(statistics.mean(values), 4),
                "median": round(statistics.median(values), 4),
                "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            }

    evaluation = {
        "model": model_key,
        "sample_set": sample_set.get("name", "unknown"),
        "total_pages_evaluated": len(page_results),
        "aggregate": {
            "text": text_agg,
            "composite": aggregated,
        },
        "per_page": page_results,
    }

    # Save if metrics_dir specified
    if metrics_dir:
        metrics_dir = Path(metrics_dir) / model_key
        metrics_dir.mkdir(parents=True, exist_ok=True)

        with open(metrics_dir / "text_metrics.json", "w") as f:
            json.dump(evaluation, f, indent=2)

        print(f"{model_key}: Evaluated {len(page_results)} pages")
        if text_agg:
            print(f"  NED: {text_agg['ned']['mean']:.4f} (mean)")
            print(f"  CER: {text_agg['cer']['mean']:.4f} (mean)")
            print(f"  WER: {text_agg['wer']['mean']:.4f} (mean)")
            print(f"  BLEU: {text_agg['bleu']['mean']:.2f} (mean)")

    return evaluation


def evaluate_all_models(
    model_keys: list[str],
    sample_set_path: str | Path,
    raw_outputs_dir: str | Path,
    gt_text_dir: str | Path,
    gt_tables_dir: str | Path | None = None,
    metrics_dir: str | Path | None = None,
    config: dict | None = None,
) -> dict[str, dict]:
    """Evaluate all models and produce comparison summary."""
    results = {}

    for model_key in model_keys:
        try:
            result = evaluate_model(
                model_key=model_key,
                sample_set_path=sample_set_path,
                raw_outputs_dir=raw_outputs_dir,
                gt_text_dir=gt_text_dir,
                gt_tables_dir=gt_tables_dir,
                metrics_dir=metrics_dir,
                config=config,
            )
            results[model_key] = result
        except Exception as e:
            print(f"ERROR evaluating {model_key}: {e}")

    # Build comparison summary
    if metrics_dir:
        _save_comparison_summary(results, Path(metrics_dir).parent / "comparison")

    return results


def _save_comparison_summary(results: dict[str, dict], comparison_dir: Path) -> None:
    """Save a CSV comparison summary of all models."""
    import csv

    comparison_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_key, result in results.items():
        agg = result.get("aggregate", {})
        text_agg = agg.get("text", {})
        comp_agg = agg.get("composite", {})

        row = {
            "model": model_key,
            "pages_evaluated": result.get("total_pages_evaluated", 0),
        }
        for metric in ["ned", "cer", "wer", "bleu", "fuzzy_ratio"]:
            if metric in text_agg:
                row[f"{metric}_mean"] = text_agg[metric]["mean"]
                row[f"{metric}_median"] = text_agg[metric]["median"]

        for comp in ["text_accuracy", "composite"]:
            if comp in comp_agg:
                row[f"{comp}_mean"] = comp_agg[comp]["mean"]

        rows.append(row)

    if rows:
        csv_path = comparison_dir / "summary_table.csv"
        # Build fieldnames from union of all row keys to handle heterogeneous schemas
        all_keys: dict[str, None] = {}
        for row in rows:
            for key in row:
                all_keys.setdefault(key, None)
        fieldnames = list(all_keys.keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved comparison summary to {csv_path}")
