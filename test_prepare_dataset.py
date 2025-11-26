"""Unit tests for prepare_dataset.py functions."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from prepare_dataset import Example, build_text, load_rows


class TestBuildText:
    """Test build_text function."""

    def test_with_input(self):
        """Test with non-empty input."""
        result = build_text("Summarize this", "Long text here", "Short summary")
        expected = "### Instruction\nSummarize this\n\n### Input\nLong text here\n\n### Response\nShort summary"
        assert result == expected

    def test_without_input(self):
        """Test with empty input."""
        result = build_text("Hello", "", "Hi there")
        expected = "### Instruction\nHello\n\n### Response\nHi there"
        assert result == expected

    def test_stripped_whitespace(self):
        """Test that whitespace is stripped."""
        result = build_text("  Summarize  ", "  text  ", "  summary  ")
        expected = "### Instruction\nSummarize\n\n### Input\ntext\n\n### Response\nsummary"
        assert result == expected


class TestLoadRows:
    """Test load_rows function."""

    def test_csv_valid(self):
        """Test loading valid CSV."""
        csv_content = "instruction,input,output\nSummarize,Text,Summary\nHello,,Hi\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            path = Path(f.name)

        try:
            rows = load_rows(path)
            assert len(rows) == 2
            assert rows[0] == {"instruction": "Summarize", "input": "Text", "output": "Summary"}
            assert rows[1] == {"instruction": "Hello", "input": "", "output": "Hi"}
        finally:
            path.unlink()

    def test_csv_missing_instruction(self):
        """Test CSV with missing instruction."""
        csv_content = "instruction,input,output\n,Text,Summary\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(SystemExit, match="Validation error.*Row 2.*instruction.*empty"):
                load_rows(path)
        finally:
            path.unlink()

    def test_csv_missing_output(self):
        """Test CSV with missing output."""
        csv_content = "instruction,input,output\nSummarize,Text,\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(SystemExit, match="Validation error.*Row 2.*output.*empty"):
                load_rows(path)
        finally:
            path.unlink()

    def test_jsonl_valid(self):
        """Test loading valid JSONL."""
        jsonl_content = '{"instruction": "Summarize", "input": "Text", "output": "Summary"}\n{"instruction": "Hello", "output": "Hi"}\n'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(jsonl_content)
            f.flush()
            path = Path(f.name)

        try:
            rows = load_rows(path)
            assert len(rows) == 2
            assert rows[0] == {"instruction": "Summarize", "input": "Text", "output": "Summary"}
            assert rows[1] == {"instruction": "Hello", "input": "", "output": "Hi"}
        finally:
            path.unlink()

    def test_jsonl_invalid_json(self):
        """Test JSONL with invalid JSON."""
        jsonl_content = '{"instruction": "Summarize"}\ninvalid\n'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(jsonl_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(SystemExit, match="Validation error.*line 2"):
                load_rows(path)
        finally:
            path.unlink()

    def test_jsonl_missing_instruction(self):
        """Test JSONL with missing instruction."""
        jsonl_content = '{"output": "Summary"}\n'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(jsonl_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(SystemExit, match="Validation error.*Line 1.*instruction.*empty"):
                load_rows(path)
        finally:
            path.unlink()

    def test_json_valid_list(self):
        """Test loading valid JSON list."""
        json_content = '[{"instruction": "Summarize", "input": "Text", "output": "Summary"}, {"instruction": "Hello", "output": "Hi"}]'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            f.flush()
            path = Path(f.name)

        try:
            rows = load_rows(path)
            assert len(rows) == 2
            assert rows[0] == {"instruction": "Summarize", "input": "Text", "output": "Summary"}
            assert rows[1] == {"instruction": "Hello", "input": "", "output": "Hi"}
        finally:
            path.unlink()

    def test_json_valid_dict_with_data(self):
        """Test loading valid JSON dict with 'data' key."""
        json_content = '{"data": [{"instruction": "Summarize", "output": "Summary"}]}'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            f.flush()
            path = Path(f.name)

        try:
            rows = load_rows(path)
            assert len(rows) == 1
            assert rows[0] == {"instruction": "Summarize", "input": "", "output": "Summary"}
        finally:
            path.unlink()

    def test_json_invalid_structure(self):
        """Test JSON with invalid structure."""
        json_content = '{"invalid": "structure"}'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(SystemExit, match="JSON must be a list"):
                load_rows(path)
        finally:
            path.unlink()

    def test_json_invalid_json(self):
        """Test invalid JSON."""
        json_content = '{"invalid": json}'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(SystemExit, match="Invalid JSON"):
                load_rows(path)
        finally:
            path.unlink()

    def test_file_not_exist(self):
        """Test non-existent file."""
        path = Path("nonexistent.csv")
        with pytest.raises(SystemExit, match="Input file does not exist"):
            load_rows(path)

    def test_unsupported_format(self):
        """Test unsupported file format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("content")
            f.flush()
            path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                load_rows(path)
        finally:
            path.unlink()