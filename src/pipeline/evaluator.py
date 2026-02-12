"""Evaluator: compute metrics by comparing OCR outputs against ground truth."""

import json
import random
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    def tqdm(iterable, **kwargs):
        return iterable

from src.evaluation.composite_score import CompositeScore, aggregate_scores, compute_composite
from src.evaluation.normalize import normalize_text
from src.evaluation.table_metrics import compute_page_table_metrics
from src.evaluation.text_metrics import TextMetrics, compute_all_metrics


def load_config(config_path: str = "config/benchmark_config.yaml") -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_sample_set(sample_set_path: str | Path) -> dict:
    with open(sample_set_path) as f:
        return json.load(f)


def _bootstrap_mean_ci(
    values: list[float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float] | None:
    """Bootstrap confidence interval for a sample mean."""
    if not values:
        return None
    if n_resamples < 1:
        return None
    if len(values) == 1:
        return values[0], values[0]

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()

    alpha = max(0.0, min(1.0, 1.0 - confidence_level))
    lo_idx = int((alpha / 2.0) * (n_resamples - 1))
    hi_idx = int((1.0 - alpha / 2.0) * (n_resamples - 1))
    lo_idx = max(0, min(lo_idx, n_resamples - 1))
    hi_idx = max(0, min(hi_idx, n_resamples - 1))
    return means[lo_idx], means[hi_idx]


def _failed_text_metrics() -> TextMetrics:
    """Worst-case text metrics used for failed/unscored pages."""
    return TextMetrics(
        ned=1.0,
        cer=1.0,
        wer=1.0,
        bleu=0.0,
        fuzzy_ratio=0.0,
    )


def evaluate_model(
    model_key: str,
    sample_set_path: str | Path,
    raw_outputs_dir: str | Path,
    gt_text_dir: str | Path,
    gt_tables_dir: str | Path | None = None,
    metrics_dir: str | Path | None = None,
    config: dict | None = None,
    run_label: str = "clean",
) -> dict:
    """Evaluate a single model's outputs against ground truth.

    Missing predictions/ground truth are counted as failed pages and assigned
    worst-case text scores so coverage impacts final rankings.
    """
    if config is None:
        config = load_config()

    sample_set = load_sample_set(sample_set_path)
    raw_outputs_dir = Path(raw_outputs_dir)
    gt_text_dir = Path(gt_text_dir)

    model_output_dir = raw_outputs_dir / model_key
    if run_label and run_label != "clean":
        model_output_dir = raw_outputs_dir / run_label / model_key

    norm_config = config.get("normalization", {})
    eval_config = config.get("evaluation", {})
    comp_weights = eval_config.get("composite_weights", {})

    uncertainty_cfg = eval_config.get("uncertainty", {})
    n_bootstrap = int(uncertainty_cfg.get("bootstrap_resamples", 1000))
    confidence = float(uncertainty_cfg.get("confidence_level", 0.95))
    bootstrap_seed = int(uncertainty_cfg.get("random_seed", 42))

    ranking_cfg = eval_config.get("ranking", {})
    rank_weights = ranking_cfg.get("weights", {})
    rank_w_text = float(rank_weights.get("text_accuracy", 0.7))
    rank_w_table = float(rank_weights.get("table_accuracy", 0.3))

    page_results = []
    missing_pred_pages = 0
    missing_gt_pages = 0
    failed_pages = 0

    for page_entry in tqdm(sample_set["pages"], desc=f"Evaluating {model_key}:{run_label}"):
        stem = page_entry["pdf_stem"]
        page_num = page_entry["page_num"]

        pred_path = model_output_dir / stem / f"page_{page_num:03d}.md"
        gt_path = gt_text_dir / stem / f"page_{page_num:03d}.txt"

        pred_exists = pred_path.exists()
        gt_exists = gt_path.exists()

        if not pred_exists or not gt_exists:
            failed_pages += 1
            reasons = []
            if not pred_exists:
                missing_pred_pages += 1
                reasons.append("missing_prediction")
            if not gt_exists:
                missing_gt_pages += 1
                reasons.append("missing_ground_truth")

            text_metrics = _failed_text_metrics()
            composite = compute_composite(
                text_metrics=text_metrics,
                weight_text=comp_weights.get("text_accuracy", 0.333),
                weight_table=comp_weights.get("table_accuracy", 0.333),
                weight_layout=comp_weights.get("layout_score", 0.333),
            )

            page_results.append({
                "pdf_stem": stem,
                "page_num": page_num,
                "content_type": page_entry.get("content_type", "unknown"),
                "status": "failed",
                "failure_reason": ",".join(reasons),
                "prediction_exists": pred_exists,
                "ground_truth_exists": gt_exists,
                "text_metrics": text_metrics.to_dict(),
                "composite": composite.to_dict(),
            })
            continue

        pred_text = pred_path.read_text(encoding="utf-8")
        gt_text = gt_path.read_text(encoding="utf-8")

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
            do_strip_markdown=False,
            do_collapse_whitespace=norm_config.get("collapse_whitespace", True),
            do_remove_page_numbers=norm_config.get("remove_page_numbers", True),
            do_lowercase=norm_config.get("lowercase", False),
        )

        text_metrics = compute_all_metrics(pred_norm, gt_norm)

        table_metrics = None
        if gt_tables_dir:
            gt_tables_path = Path(gt_tables_dir) / stem / f"page_{page_num:03d}.json"
            if gt_tables_path.exists():
                with open(gt_tables_path) as f:
                    ref_tables = json.load(f)
                pred_tables_path = model_output_dir / stem / f"page_{page_num:03d}_tables.json"
                pred_tables = []
                if pred_tables_path.exists():
                    with open(pred_tables_path) as f:
                        pred_tables = json.load(f)
                table_metrics = compute_page_table_metrics(pred_tables, ref_tables)

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
            "status": "ok",
            "failure_reason": "",
            "prediction_exists": True,
            "ground_truth_exists": True,
            "text_metrics": text_metrics.to_dict(),
            "composite": composite.to_dict(),
        }
        if table_metrics:
            page_result["table_metrics"] = table_metrics.to_dict()

        page_results.append(page_result)

    total_target = len(sample_set["pages"])
    pages_successful = total_target - failed_pages
    coverage_rate = (pages_successful / total_target * 100.0) if total_target else 0.0
    failure_rate = (failed_pages / total_target * 100.0) if total_target else 0.0

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

    import statistics
    text_fields = ["ned", "cer", "wer", "bleu", "fuzzy_ratio"]
    text_agg = {}
    for field in text_fields:
        values = [r["text_metrics"][field] for r in page_results]
        if values:
            text_agg[field] = {
                "mean": round(statistics.mean(values), 4),
                "median": round(statistics.median(values), 4),
                "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            }
            ci = _bootstrap_mean_ci(values, n_bootstrap, confidence, bootstrap_seed)
            if ci:
                text_agg[field]["ci_low"] = round(ci[0], 4)
                text_agg[field]["ci_high"] = round(ci[1], 4)

    table_values = []
    for r in page_results:
        tm = r.get("table_metrics")
        if tm and tm.get("tables_reference", 0) > 0:
            table_values.append(tm.get("teds", 0.0))
    table_agg = {}
    if table_values:
        table_agg = {
            "teds": {
                "mean": round(statistics.mean(table_values), 2),
                "median": round(statistics.median(table_values), 2),
                "std": round(statistics.stdev(table_values), 2) if len(table_values) > 1 else 0.0,
                "count": len(table_values),
            }
        }
        ci = _bootstrap_mean_ci(table_values, n_bootstrap, confidence, bootstrap_seed)
        if ci:
            table_agg["teds"]["ci_low"] = round(ci[0], 2)
            table_agg["teds"]["ci_high"] = round(ci[1], 2)

    for comp_field in ["text_accuracy", "table_accuracy", "layout_score", "composite"]:
        comp_values = [getattr(c, comp_field) for c in composites if getattr(c, comp_field) is not None]
        ci = _bootstrap_mean_ci(comp_values, n_bootstrap, confidence, bootstrap_seed)
        if ci and comp_field in aggregated:
            aggregated[comp_field]["ci_low"] = round(ci[0], 2)
            aggregated[comp_field]["ci_high"] = round(ci[1], 2)

    text_acc_mean = aggregated.get("text_accuracy", {}).get("mean", 0.0)
    table_count = aggregated.get("table_accuracy", {}).get("count", 0)
    table_acc_mean = aggregated.get("table_accuracy", {}).get("mean", 0.0) if table_count else None
    table_for_rank = text_acc_mean if table_acc_mean is None else table_acc_mean
    rank_base = (rank_w_text * text_acc_mean) + (rank_w_table * table_for_rank)
    rank_score = rank_base * (coverage_rate / 100.0)

    manifest_path = model_output_dir / "run_manifest.json"
    run_manifest = None
    if manifest_path.exists():
        with open(manifest_path) as f:
            run_manifest = json.load(f)

    evaluation = {
        "model": model_key,
        "run_label": run_label,
        "sample_set": sample_set.get("name", "unknown"),
        "total_pages_target": total_target,
        "total_pages_evaluated": len(page_results),
        "pages_successful": pages_successful,
        "pages_failed": failed_pages,
        "coverage_rate": round(coverage_rate, 2),
        "failure_rate": round(failure_rate, 2),
        "failure_breakdown": {
            "missing_prediction": missing_pred_pages,
            "missing_ground_truth": missing_gt_pages,
        },
        "ranking": {
            "text_accuracy_mean": round(text_acc_mean, 2),
            "table_accuracy_mean": round(table_acc_mean, 2) if table_acc_mean is not None else None,
            "coverage_rate": round(coverage_rate, 2),
            "rank_score": round(rank_score, 2),
        },
        "aggregate": {
            "text": text_agg,
            "table": table_agg,
            "composite": aggregated,
        },
        "per_page": page_results,
    }
    if run_manifest is not None:
        evaluation["run_manifest"] = run_manifest

    if metrics_dir:
        base_metrics_dir = Path(metrics_dir)
        if run_label and run_label != "clean":
            base_metrics_dir = base_metrics_dir / run_label
        model_metrics_dir = base_metrics_dir / model_key
        model_metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(model_metrics_dir / "text_metrics.json", "w") as f:
            json.dump(evaluation, f, indent=2)

        print(f"{model_key} [{run_label}]: Evaluated {len(page_results)} pages")
        print(f"  Coverage: {coverage_rate:.2f}% ({pages_successful}/{total_target})")
        print(f"  Failed pages: {failed_pages}")
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
    run_label: str = "clean",
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
                run_label=run_label,
            )
            results[model_key] = result
        except Exception as e:
            print(f"ERROR evaluating {model_key}: {e}")

    if metrics_dir:
        comparison_dir = Path(metrics_dir).parent / "comparison"
        if run_label and run_label != "clean":
            comparison_dir = comparison_dir / run_label
        _save_comparison_summary(results, comparison_dir)

    return results


def evaluate_model_robustness(
    model_key: str,
    sample_set_path: str | Path,
    raw_outputs_dir: str | Path,
    gt_text_dir: str | Path,
    slice_names: list[str],
    gt_tables_dir: str | Path | None = None,
    metrics_dir: str | Path | None = None,
    config: dict | None = None,
) -> dict[str, dict]:
    """Evaluate one model across multiple robustness slices."""
    by_slice = {}
    for slice_name in slice_names:
        by_slice[slice_name] = evaluate_model(
            model_key=model_key,
            sample_set_path=sample_set_path,
            raw_outputs_dir=raw_outputs_dir,
            gt_text_dir=gt_text_dir,
            gt_tables_dir=gt_tables_dir,
            metrics_dir=metrics_dir,
            config=config,
            run_label=slice_name,
        )
    return by_slice


def _save_comparison_summary(results: dict[str, dict], comparison_dir: Path) -> None:
    """Save a CSV comparison summary of all models."""
    import csv

    comparison_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_key, result in results.items():
        agg = result.get("aggregate", {})
        text_agg = agg.get("text", {})
        comp_agg = agg.get("composite", {})
        ranking = result.get("ranking", {})

        row = {
            "model": model_key,
            "run_label": result.get("run_label", "clean"),
            "pages_target": result.get("total_pages_target", 0),
            "pages_evaluated": result.get("total_pages_evaluated", 0),
            "pages_successful": result.get("pages_successful", 0),
            "pages_failed": result.get("pages_failed", 0),
            "coverage_rate": result.get("coverage_rate", 0.0),
            "failure_rate": result.get("failure_rate", 0.0),
            "rank_score": ranking.get("rank_score", 0.0),
            "text_accuracy_mean": ranking.get("text_accuracy_mean", 0.0),
            "table_accuracy_mean": ranking.get("table_accuracy_mean"),
        }

        for metric in ["ned", "cer", "wer", "bleu", "fuzzy_ratio"]:
            if metric in text_agg:
                row[f"{metric}_mean"] = text_agg[metric].get("mean")
                row[f"{metric}_median"] = text_agg[metric].get("median")
                row[f"{metric}_ci_low"] = text_agg[metric].get("ci_low")
                row[f"{metric}_ci_high"] = text_agg[metric].get("ci_high")

        for comp in ["text_accuracy", "table_accuracy", "composite"]:
            if comp in comp_agg:
                row[f"{comp}_mean"] = comp_agg[comp].get("mean")
                row[f"{comp}_ci_low"] = comp_agg[comp].get("ci_low")
                row[f"{comp}_ci_high"] = comp_agg[comp].get("ci_high")

        rows.append(row)

    if rows:
        csv_path = comparison_dir / "summary_table.csv"
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
