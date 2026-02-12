"""Table evaluation metrics using Tree Edit Distance based Similarity (TEDS)."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TableMetrics:
    """Table evaluation metrics for a single page."""

    teds: float  # Tree Edit Distance Similarity (0-100)
    tables_predicted: int
    tables_reference: int
    tables_matched: int

    def to_dict(self) -> dict:
        return {
            "teds": round(self.teds, 2),
            "tables_predicted": self.tables_predicted,
            "tables_reference": self.tables_reference,
            "tables_matched": self.tables_matched,
        }


def compute_teds(pred_html: str, ref_html: str) -> float:
    """Compute TEDS between predicted and reference HTML tables.

    Uses the table-recognition-metric package.
    Returns similarity score from 0 to 100.
    """
    try:
        from table_recognition_metric import TEDS as TEDSCalculator
    except ImportError:
        raise ImportError(
            "table-recognition-metric is required for TEDS. "
            "Install with: pip install table-recognition-metric"
        )

    teds_calc = TEDSCalculator(structure_only=False)
    score = teds_calc.evaluate(pred_html, ref_html)
    return score * 100  # Convert to 0-100 scale


def compute_teds_structure_only(pred_html: str, ref_html: str) -> float:
    """Compute TEDS considering only table structure (ignoring cell text).

    Useful for evaluating layout understanding independent of OCR accuracy.
    """
    try:
        from table_recognition_metric import TEDS as TEDSCalculator
    except ImportError:
        raise ImportError(
            "table-recognition-metric is required for TEDS. "
            "Install with: pip install table-recognition-metric"
        )

    teds_calc = TEDSCalculator(structure_only=True)
    score = teds_calc.evaluate(pred_html, ref_html)
    return score * 100


def match_tables(
    pred_tables: list[str],
    ref_tables: list[str],
) -> list[tuple[int, int, float]]:
    """Match predicted tables to reference tables by best TEDS score.

    Returns list of (pred_idx, ref_idx, teds_score) tuples.
    Uses greedy matching by highest score.
    """
    if not pred_tables or not ref_tables:
        return []

    # Compute pairwise TEDS scores
    scores = {}
    for i, pred in enumerate(pred_tables):
        for j, ref in enumerate(ref_tables):
            try:
                scores[(i, j)] = compute_teds(pred, ref)
            except (ValueError, KeyError, IndexError) as e:
                logger.warning(
                    "TEDS computation failed for pred_table[%d] vs ref_table[%d]: %s",
                    i, j, e,
                )
                scores[(i, j)] = 0.0

    # Greedy matching
    matched = []
    used_pred = set()
    used_ref = set()

    # Sort by score descending
    sorted_pairs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    for (i, j), score in sorted_pairs:
        if i in used_pred or j in used_ref:
            continue
        matched.append((i, j, score))
        used_pred.add(i)
        used_ref.add(j)

    return matched


def compute_page_table_metrics(
    pred_tables: list[str],
    ref_tables: list[str],
) -> TableMetrics:
    """Compute table metrics for a single page.

    Args:
        pred_tables: List of predicted HTML table strings
        ref_tables: List of reference HTML table strings

    Returns:
        TableMetrics with TEDS and matching info
    """
    if not ref_tables:
        return TableMetrics(
            teds=100.0 if not pred_tables else 0.0,
            tables_predicted=len(pred_tables),
            tables_reference=0,
            tables_matched=0,
        )

    if not pred_tables:
        return TableMetrics(
            teds=0.0,
            tables_predicted=0,
            tables_reference=len(ref_tables),
            tables_matched=0,
        )

    matches = match_tables(pred_tables, ref_tables)
    avg_teds = sum(s for _, _, s in matches) / len(ref_tables) if ref_tables else 0.0

    return TableMetrics(
        teds=avg_teds,
        tables_predicted=len(pred_tables),
        tables_reference=len(ref_tables),
        tables_matched=len(matches),
    )
