"""
Text processing and typography normalization utilities for the Ghaemieh OCR pipeline.
Provides sentence unwrapping, smart paragraph reflow, and structural element preservation.
"""

import re
from typing import Optional


class TextProcessor:
    """Utility class for normalizing OCR raw text outputs, reflowing paragraphs and preserving document layouts."""

    @staticmethod
    def is_structural_line(line: str) -> bool:
        """
        Detects whether a given line represents a structural element that should not be merged
        with adjacent lines (such as headings, poetry, lists, footnote numbers, or dividers).
        """
        s = line.strip()
        if not s:
            return False

        # Markdown headings (e.g. # Title, ## Section)
        if s.startswith("#"):
            return True

        # Headings in brackets commonly found in classical Arabic/Persian literature (e.g. [إشارة المغيرة...])
        if s.startswith("[") and s.endswith("]"):
            return True

        # Book section indicators (e.g. فصل اول, باب دوم, مقدمه, مسأله...)
        if re.match(r"^(فصل|بخش|باب|مقدمه|اصل|ماده|گفتار|درس|مسأله|تنبيه|خاتمه|کتاب)\s*", s):
            return True

        # Numbered list or footnote references: e.g. (1), (۱), [1], [۱], 1., ۱-
        if re.match(r"^[\[\(]?(\d+|[۰-۹]+|[a-zA-Z])[\.\-\)\]]\s*", s):
            return True

        # Bullet points
        if re.match(r"^[-*•▪▫–—]\s+", s):
            return True

        # Dividers and horizontal rules
        if s.startswith("---") or s.startswith("___") or s.startswith("***"):
            return True

        # Poetry verses with asterisks or classical dot-leaders
        if "***" in s or "  *  " in s or "......." in s:
            return True

        return False

    @classmethod
    def reflow_paragraphs(cls, text: str) -> str:
        """
        Unwraps hard line breaks in raw OCR text to form natural, continuous paragraphs.
        
        - Preserves explicit paragraph breaks (empty lines / double newlines).
        - Preserves headings, poetry lines, lists, and footnotes on separate lines.
        - Unwraps mid-sentence line breaks with proper word spacing.
        - Resolves English hyphenation at line breaks (e.g., 'repre- \\n sentation' -> 'representation').
        """
        if not text or not text.strip():
            return text or ""

        # Normalize line endings
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")

        # Split into blocks by explicit paragraph separators (two or more consecutive newlines)
        blocks = re.split(r"\n\s*\n", normalized)
        processed_blocks = []

        for block in blocks:
            raw_lines = [l.strip() for l in block.split("\n") if l.strip()]
            if not raw_lines:
                continue

            merged_lines = []
            i = 0
            while i < len(raw_lines):
                current_line = raw_lines[i]

                # Structural line: preserve on its own line
                if cls.is_structural_line(current_line):
                    merged_lines.append(current_line)
                    i += 1
                    continue

                # Normal prose: check if subsequent lines should be joined into the same paragraph
                while i + 1 < len(raw_lines):
                    next_line = raw_lines[i + 1]

                    # Stop merging if next line is structural
                    if cls.is_structural_line(next_line):
                        break

                    # Stop merging if current line ends with a colon or semicolon introducing a list/quote
                    if current_line.endswith(":") or current_line.endswith("؛"):
                        break

                    # Resolve hyphenation at line breaks in Latin scripts
                    if current_line.endswith("-") and not re.match(r"^[-*•]", current_line):
                        current_line = current_line[:-1] + next_line
                    else:
                        current_line = current_line + " " + next_line

                    i += 1

                merged_lines.append(current_line)
                i += 1

            processed_blocks.append("\n".join(merged_lines))

        return "\n\n".join(processed_blocks)
