"""Text-level evaluation metrics for OCR output comparison."""

from dataclasses import dataclass

import jiwer
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sacrebleu.metrics import BLEU


@dataclass
class TextMetrics:
    """Collection of text comparison metrics."""

    ned: float  # Normalized Edit Distance (0=identical, 1=completely different)
    cer: float  # Character Error Rate
    wer: float  # Word Error Rate
    bleu: float  # BLEU score (0-100)
    fuzzy_ratio: float  # Token sort ratio (0-100)

    def to_dict(self) -> dict:
        return {
            "ned": round(self.ned, 4),
            "cer": round(self.cer, 4),
            "wer": round(self.wer, 4),
            "bleu": round(self.bleu, 2),
            "fuzzy_ratio": round(self.fuzzy_ratio, 2),
        }

    @property
    def text_accuracy(self) -> float:
        """Overall text accuracy score (0-100), higher is better."""
        return (1.0 - self.ned) * 100


def compute_ned(prediction: str, reference: str) -> float:
    """Compute Normalized Edit Distance.

    Returns 0.0 for identical strings, 1.0 for completely different.
    """
    if not prediction and not reference:
        return 0.0
    max_len = max(len(prediction), len(reference))
    if max_len == 0:
        return 0.0
    dist = Levenshtein.distance(prediction, reference)
    return dist / max_len


def compute_cer(prediction: str, reference: str) -> float:
    """Compute Character Error Rate using jiwer.

    CER = (S + D + I) / N where N is reference length in characters.
    """
    if not reference:
        return 0.0 if not prediction else 1.0

    try:
        cer = jiwer.cer(reference, prediction)
    except ValueError:
        return 1.0

    return min(cer, 1.0)  # Cap at 1.0


def compute_wer(prediction: str, reference: str) -> float:
    """Compute Word Error Rate using jiwer.

    WER = (S + D + I) / N where N is reference length in words.
    """
    if not reference.strip():
        return 0.0 if not prediction.strip() else 1.0

    try:
        wer = jiwer.wer(reference, prediction)
    except ValueError:
        return 1.0

    return min(wer, 1.0)  # Cap at 1.0


def compute_bleu(prediction: str, reference: str) -> float:
    """Compute BLEU score using sacrebleu.

    Returns score from 0 to 100.
    """
    if not reference.strip() or not prediction.strip():
        return 0.0

    bleu = BLEU(effective_order=True)
    result = bleu.corpus_score([prediction], [[reference]])
    return result.score


def compute_fuzzy_ratio(prediction: str, reference: str) -> float:
    """Compute fuzzy token sort ratio using rapidfuzz.

    Returns similarity score from 0 to 100.
    """
    if not prediction and not reference:
        return 100.0
    return fuzz.token_sort_ratio(prediction, reference)


def compute_all_metrics(prediction: str, reference: str) -> TextMetrics:
    """Compute all text metrics between prediction and reference.

    Args:
        prediction: OCR model output text (normalized)
        reference: Ground truth text (normalized)

    Returns:
        TextMetrics dataclass with all scores
    """
    return TextMetrics(
        ned=compute_ned(prediction, reference),
        cer=compute_cer(prediction, reference),
        wer=compute_wer(prediction, reference),
        bleu=compute_bleu(prediction, reference),
        fuzzy_ratio=compute_fuzzy_ratio(prediction, reference),
    )
