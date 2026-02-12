"""Tests for data preparation modules."""

import pytest

from src.data_prep.build_sample_sets import (
    build_quick_dev,
    build_full_benchmark,
    build_table_focus,
    classify_page_content,
)


class TestBuildSampleSets:
    def test_quick_dev_empty_pages(self):
        """Finding 4: should not crash with no candidate pages."""
        result = build_quick_dev([], n=20)
        assert result == []

    def test_quick_dev_all_empty(self):
        """Pages that are all empty should return empty set."""
        pages = [{"is_empty": True, "has_image": True, "content_type": "text_only"}]
        result = build_quick_dev(pages, n=5)
        assert result == []

    def test_quick_dev_respects_n(self):
        pages = [
            {"is_empty": False, "has_image": True, "content_type": "text_only",
             "pdf_stem": f"doc_{i}", "page_num": 1}
            for i in range(50)
        ]
        result = build_quick_dev(pages, n=10)
        assert len(result) == 10

    def test_full_benchmark_empty_pages(self):
        result = build_full_benchmark([], n=100)
        assert result == []

    def test_table_focus_no_tables(self):
        pages = [
            {"is_empty": False, "has_image": True, "content_type": "text_only",
             "pdf_stem": "doc_1", "page_num": 1}
        ]
        result = build_table_focus(pages, n=10)
        assert result == []  # no table or multi_column pages


class TestClassifyPageContent:
    def test_sparse_for_missing_file(self, tmp_path):
        fake_path = tmp_path / "nonexistent.txt"
        assert classify_page_content(fake_path) == "sparse"

    def test_sparse_for_short_text(self, tmp_path):
        f = tmp_path / "short.txt"
        f.write_text("hi")
        assert classify_page_content(f) == "sparse"

    def test_table_detection(self, tmp_path):
        f = tmp_path / "table.txt"
        # Must be > 50 chars to avoid "sparse" classification, and have >3 pipe lines
        f.write_text(
            "Column A Description | Column B Description\n"
            "---|---\n"
            "Value 1 | Value 2\n"
            "Value 3 | Value 4\n"
            "Value 5 | Value 6\n"
        )
        assert classify_page_content(f) == "table"

    def test_text_only(self, tmp_path):
        f = tmp_path / "text.txt"
        f.write_text("This is a normal paragraph of text that goes on for a while. " * 5)
        assert classify_page_content(f) == "text_only"
