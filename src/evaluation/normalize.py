"""Text normalization for fair OCR comparison."""

import re
import unicodedata


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Apply unicode normalization with the specified form."""
    return unicodedata.normalize(form, text)


def strip_markdown(text: str) -> str:
    """Remove markdown formatting, preserving text content."""
    # Remove headers (# ## ### etc.)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)

    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Remove links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove images ![alt](url) -> alt
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Remove blockquote markers
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)

    # Remove list markers (- * + and numbered)
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)

    return text


def collapse_whitespace(text: str) -> str:
    """Collapse multiple whitespace to single spaces, normalize line breaks."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple blank lines to single
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces to single
    text = re.sub(r"[ \t]+", " ", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def remove_page_numbers(text: str) -> str:
    """Remove common page number patterns."""
    # "Page X of Y" or "Page X"
    text = re.sub(r"(?i)page\s+\d+\s*(of\s+\d+)?", "", text)
    # Standalone numbers at start/end of text that look like page numbers
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)
    # "- X -" or "— X —" page number format
    text = re.sub(r"^[\s]*[-–—]\s*\d{1,3}\s*[-–—][\s]*$", "", text, flags=re.MULTILINE)
    return text


def remove_headers_footers(text: str) -> str:
    """Remove common repeated header/footer patterns.

    This is a best-effort heuristic for insurance PDS documents.
    """
    lines = text.split("\n")
    if len(lines) < 5:
        return text

    # Remove very short first/last lines that look like headers/footers
    cleaned = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip very short lines at start (likely header) or end (likely footer)
        if i < 2 or i >= len(lines) - 2:
            if len(stripped) < 5 and stripped.isdigit():
                continue
        cleaned.append(line)

    return "\n".join(cleaned)


def normalize_text(
    text: str,
    unicode_form: str = "NFKC",
    do_strip_markdown: bool = True,
    do_collapse_whitespace: bool = True,
    do_remove_page_numbers: bool = True,
    do_lowercase: bool = False,
) -> str:
    """Apply full normalization pipeline to text.

    Args:
        text: Raw text to normalize
        unicode_form: Unicode normalization form (NFKC recommended)
        do_strip_markdown: Remove markdown formatting
        do_collapse_whitespace: Collapse whitespace
        do_remove_page_numbers: Remove page number patterns
        do_lowercase: Convert to lowercase

    Returns:
        Normalized text
    """
    if not text:
        return ""

    text = normalize_unicode(text, form=unicode_form)

    if do_strip_markdown:
        text = strip_markdown(text)

    if do_remove_page_numbers:
        text = remove_page_numbers(text)

    if do_collapse_whitespace:
        text = collapse_whitespace(text)

    if do_lowercase:
        text = text.lower()

    return text
