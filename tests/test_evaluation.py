"""Tests for evaluation modules: edge cases around zero scores, empty inputs, normalization."""

import pytest

from src.evaluation.normalize import normalize_text, normalize_unicode
from src.evaluation.text_metrics import (
    TextMetrics,
    compute_all_metrics,
    compute_ned,
    compute_cer,
    compute_wer,
    compute_bleu,
    compute_fuzzy_ratio,
)
from src.evaluation.composite_score import (
    CompositeScore,
    compute_composite,
    aggregate_scores,
)
from src.evaluation.table_metrics import (
    TableMetrics,
    compute_page_table_metrics,
    match_tables,
)


# ── Text Metrics ──────────────────────────────────────────────────


class TestNED:
    def test_identical_strings(self):
        assert compute_ned("hello", "hello") == 0.0

    def test_completely_different(self):
        assert compute_ned("abc", "xyz") == 1.0

    def test_empty_both(self):
        assert compute_ned("", "") == 0.0

    def test_one_empty(self):
        assert compute_ned("abc", "") == 1.0
        assert compute_ned("", "abc") == 1.0

    def test_partial_match(self):
        ned = compute_ned("hello", "helo")
        assert 0.0 < ned < 1.0


class TestCER:
    def test_identical(self):
        assert compute_cer("hello world", "hello world") == 0.0

    def test_empty_reference_with_prediction(self):
        # No reference text means we can't compute meaningful CER
        # jiwer returns 1.0 for empty reference with non-empty prediction
        assert compute_cer("some text", "") == 1.0

    def test_empty_prediction_nonempty_reference(self):
        assert compute_cer("", "hello") == 1.0

    def test_both_empty(self):
        assert compute_cer("", "") == 0.0


class TestWER:
    def test_identical(self):
        assert compute_wer("hello world", "hello world") == 0.0

    def test_empty_reference_with_prediction(self):
        # No reference words means we can't compute meaningful WER
        assert compute_wer("text", "") == 1.0

    def test_empty_prediction(self):
        assert compute_wer("", "hello world") == 1.0

    def test_whitespace_only(self):
        assert compute_wer("", "   ") == 0.0


class TestBLEU:
    def test_identical(self):
        score = compute_bleu("the cat sat on the mat", "the cat sat on the mat")
        assert score > 90.0  # should be near 100

    def test_empty_strings(self):
        assert compute_bleu("", "") == 0.0
        assert compute_bleu("hello", "") == 0.0
        assert compute_bleu("", "hello") == 0.0


class TestFuzzyRatio:
    def test_identical(self):
        assert compute_fuzzy_ratio("hello world", "hello world") == 100.0

    def test_both_empty(self):
        assert compute_fuzzy_ratio("", "") == 100.0

    def test_completely_different(self):
        score = compute_fuzzy_ratio("abc", "xyz")
        assert score < 50.0


class TestComputeAllMetrics:
    def test_identical_text(self):
        metrics = compute_all_metrics("hello world", "hello world")
        assert metrics.ned == 0.0
        assert metrics.cer == 0.0
        assert metrics.wer == 0.0
        assert metrics.text_accuracy == 100.0

    def test_empty_both(self):
        metrics = compute_all_metrics("", "")
        assert metrics.ned == 0.0

    def test_returns_text_metrics(self):
        metrics = compute_all_metrics("foo", "bar")
        assert isinstance(metrics, TextMetrics)
        d = metrics.to_dict()
        assert all(k in d for k in ["ned", "cer", "wer", "bleu", "fuzzy_ratio"])

    def test_zero_score_legitimate(self):
        """A prediction that is completely wrong should give 0.0 accuracy, not None."""
        metrics = compute_all_metrics("zzzzz", "hello world this is a test")
        assert metrics.text_accuracy == pytest.approx(0.0, abs=5.0)  # near 0


# ── Normalization ─────────────────────────────────────────────────


class TestNormalization:
    def test_unicode_form_respected(self):
        # U+2160 ROMAN NUMERAL ONE is compatibility-equivalent to "I"
        # NFKC decomposes it, NFC does not
        text = "\u2160"  # Ⅰ (Roman numeral one)
        nfkc = normalize_unicode(text, "NFKC")
        nfc = normalize_unicode(text, "NFC")
        assert nfkc == "I"  # NFKC decomposes compatibility characters
        assert nfc == "\u2160"  # NFC preserves compatibility characters

    def test_normalize_text_uses_form(self):
        text = "\u2160"  # Ⅰ (Roman numeral one)
        result_nfkc = normalize_text(text, unicode_form="NFKC", do_strip_markdown=False,
                                     do_collapse_whitespace=False, do_remove_page_numbers=False)
        result_nfc = normalize_text(text, unicode_form="NFC", do_strip_markdown=False,
                                    do_collapse_whitespace=False, do_remove_page_numbers=False)
        assert result_nfkc == "I"
        assert result_nfc == "\u2160"

    def test_strip_markdown_headers(self):
        text = "## Section Title\nSome text"
        result = normalize_text(text, do_strip_markdown=True, do_collapse_whitespace=False,
                                do_remove_page_numbers=False)
        assert "##" not in result
        assert "Section Title" in result

    def test_empty_text(self):
        assert normalize_text("") == ""
        assert normalize_text(None) == ""  # type: ignore

    def test_page_number_removal(self):
        text = "Some content\nPage 5 of 20\nMore content"
        result = normalize_text(text, do_strip_markdown=False, do_remove_page_numbers=True)
        assert "Page 5" not in result


# ── Composite Score ───────────────────────────────────────────────


class TestCompositeScore:
    def test_text_only(self):
        metrics = TextMetrics(ned=0.1, cer=0.1, wer=0.2, bleu=80.0, fuzzy_ratio=90.0)
        score = compute_composite(text_metrics=metrics)
        assert score.text_accuracy == pytest.approx(90.0)
        assert score.table_accuracy is None
        assert score.layout_score is None
        assert score.composite == pytest.approx(90.0)  # only text, so composite = text

    def test_zero_text_accuracy_preserved(self):
        """Finding 1: zero scores must not be treated as missing."""
        metrics = TextMetrics(ned=1.0, cer=1.0, wer=1.0, bleu=0.0, fuzzy_ratio=0.0)
        score = compute_composite(text_metrics=metrics)
        assert score.text_accuracy == 0.0  # must be 0.0, NOT None
        assert score.composite == 0.0

    def test_missing_components_are_none(self):
        score = compute_composite()  # nothing provided
        assert score.text_accuracy is None
        assert score.table_accuracy is None
        assert score.layout_score is None
        assert score.composite == 0.0

    def test_to_dict_none_handling(self):
        score = CompositeScore(
            text_accuracy=None, table_accuracy=None,
            layout_score=None, composite=0.0,
        )
        d = score.to_dict()
        assert d["text_accuracy"] is None
        assert d["composite"] == 0.0


class TestAggregateScores:
    def test_zero_scores_included(self):
        """Finding 1: aggregate must include legitimate 0.0 scores."""
        scores = [
            CompositeScore(text_accuracy=0.0, table_accuracy=None, layout_score=None, composite=0.0),
            CompositeScore(text_accuracy=100.0, table_accuracy=None, layout_score=None, composite=100.0),
        ]
        result = aggregate_scores(scores)
        # text_accuracy should average 0.0 and 100.0 = 50.0
        assert result["text_accuracy"]["mean"] == 50.0
        assert result["text_accuracy"]["count"] == 2

    def test_none_scores_excluded(self):
        """None (missing) should be excluded, not 0.0."""
        scores = [
            CompositeScore(text_accuracy=80.0, table_accuracy=None, layout_score=None, composite=80.0),
            CompositeScore(text_accuracy=60.0, table_accuracy=None, layout_score=None, composite=60.0),
        ]
        result = aggregate_scores(scores)
        assert result["table_accuracy"]["count"] == 0
        assert result["text_accuracy"]["count"] == 2

    def test_empty_list(self):
        assert aggregate_scores([]) == {}

    def test_all_none(self):
        scores = [
            CompositeScore(text_accuracy=None, table_accuracy=None, layout_score=None, composite=0.0),
        ]
        result = aggregate_scores(scores)
        assert result["text_accuracy"]["count"] == 0
        assert result["composite"]["count"] == 1


# ── Table Metrics ─────────────────────────────────────────────────


class TestTableMetrics:
    def test_no_reference_no_prediction(self):
        metrics = compute_page_table_metrics([], [])
        assert metrics.teds == 100.0  # no tables expected, none found = perfect

    def test_no_reference_with_prediction(self):
        metrics = compute_page_table_metrics(["<table><tr><td>x</td></tr></table>"], [])
        assert metrics.teds == 0.0  # spurious table predicted

    def test_no_prediction_with_reference(self):
        metrics = compute_page_table_metrics([], ["<table><tr><td>x</td></tr></table>"])
        assert metrics.teds == 0.0
        assert metrics.tables_matched == 0

    def test_match_tables_empty(self):
        assert match_tables([], []) == []
        assert match_tables(["<table></table>"], []) == []
        assert match_tables([], ["<table></table>"]) == []


# ── CSV Summary Edge Case ─────────────────────────────────────────


class TestCSVSummary:
    def test_heterogeneous_rows(self, tmp_path):
        """Finding 3: CSV writer should handle rows with different key sets."""
        import csv
        from src.pipeline.evaluator import _save_comparison_summary

        results = {
            "model_a": {
                "total_pages_evaluated": 10,
                "aggregate": {
                    "text": {"ned": {"mean": 0.1, "median": 0.1}},
                    "composite": {"text_accuracy": {"mean": 90.0}},
                },
            },
            "model_b": {
                "total_pages_evaluated": 5,
                "aggregate": {
                    "text": {},  # No text metrics (failed)
                    "composite": {},
                },
            },
        }

        _save_comparison_summary(results, tmp_path)
        csv_path = tmp_path / "summary_table.csv"
        assert csv_path.exists()

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
