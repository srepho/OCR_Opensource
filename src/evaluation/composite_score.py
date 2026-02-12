"""Composite scoring for overall OCR model comparison."""

from dataclasses import dataclass

from .text_metrics import TextMetrics
from .table_metrics import TableMetrics


@dataclass
class CompositeScore:
    """Weighted composite score combining text, table, and layout metrics."""

    text_accuracy: float | None  # (1-NED)*100, range 0-100; None if not evaluated
    table_accuracy: float | None  # TEDS, range 0-100; None if no table GT
    layout_score: float | None  # Layout similarity, range 0-100; None if not evaluated
    composite: float  # Weighted average, range 0-100

    # Weights used
    weight_text: float = 0.333
    weight_table: float = 0.333
    weight_layout: float = 0.333

    def to_dict(self) -> dict:
        return {
            "text_accuracy": round(self.text_accuracy, 2) if self.text_accuracy is not None else None,
            "table_accuracy": round(self.table_accuracy, 2) if self.table_accuracy is not None else None,
            "layout_score": round(self.layout_score, 2) if self.layout_score is not None else None,
            "composite": round(self.composite, 2),
        }


def compute_composite(
    text_metrics: TextMetrics | None = None,
    table_metrics: TableMetrics | None = None,
    layout_score: float | None = None,
    weight_text: float = 0.333,
    weight_table: float = 0.333,
    weight_layout: float = 0.333,
) -> CompositeScore:
    """Compute weighted composite score.

    Missing components are excluded and weights re-normalized.

    Args:
        text_metrics: Text comparison metrics (provides NED-based accuracy)
        table_metrics: Table comparison metrics (provides TEDS)
        layout_score: Layout similarity score (0-100), from manual annotation
        weight_text: Weight for text accuracy component
        weight_table: Weight for table accuracy component
        weight_layout: Weight for layout similarity component

    Returns:
        CompositeScore with individual and combined scores
    """
    components = []
    weights = []

    text_acc: float | None = None
    table_acc: float | None = None
    layout_sc: float | None = None

    if text_metrics is not None:
        text_acc = text_metrics.text_accuracy
        components.append(text_acc)
        weights.append(weight_text)

    if table_metrics is not None and table_metrics.tables_reference > 0:
        table_acc = table_metrics.teds
        components.append(table_acc)
        weights.append(weight_table)

    if layout_score is not None:
        layout_sc = layout_score
        components.append(layout_sc)
        weights.append(weight_layout)

    # Re-normalize weights
    if weights:
        total_weight = sum(weights)
        composite = sum(c * w for c, w in zip(components, weights)) / total_weight
    else:
        composite = 0.0

    return CompositeScore(
        text_accuracy=text_acc,
        table_accuracy=table_acc,
        layout_score=layout_sc,
        composite=composite,
        weight_text=weight_text,
        weight_table=weight_table,
        weight_layout=weight_layout,
    )


def aggregate_scores(scores: list[CompositeScore]) -> dict:
    """Aggregate composite scores across multiple pages.

    Returns dict with mean, median, min, max for each component.
    """
    import statistics

    if not scores:
        return {}

    fields = ["text_accuracy", "table_accuracy", "layout_score", "composite"]
    result = {}

    for field in fields:
        values = [getattr(s, field) for s in scores]
        present = [v for v in values if v is not None]  # Exclude missing (None), keep 0.0

        if present:
            result[field] = {
                "mean": round(statistics.mean(present), 2),
                "median": round(statistics.median(present), 2),
                "min": round(min(present), 2),
                "max": round(max(present), 2),
                "std": round(statistics.stdev(present), 2) if len(present) > 1 else 0.0,
                "count": len(present),
            }
        else:
            result[field] = {
                "mean": 0.0, "median": 0.0, "min": 0.0,
                "max": 0.0, "std": 0.0, "count": 0,
            }

    return result
