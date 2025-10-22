"""
Output Formatters for Multi-Format Report Generation

Supports Markdown, JSON, and CSV output formats.
"""

from typing import Dict, Any, List
from abc import ABC, abstractmethod
import json
import csv
from io import StringIO


class OutputFormatter(ABC):
    """Base class for output formatters"""

    @abstractmethod
    def format(self, data: Dict[str, Any]) -> str:
        """
        Format data to string output.

        Args:
            data: Dictionary containing report data

        Returns:
            Formatted string
        """
        pass


class MarkdownFormatter(OutputFormatter):
    """Format reports as Markdown"""

    def format(self, data: Dict[str, Any]) -> str:
        """Format data as Markdown"""
        lines = []

        # Title
        if "title" in data:
            lines.append(f"# {data['title']}\n")

        # Metadata
        if "metadata" in data:
            lines.append("## Metadata\n")
            for key, value in data["metadata"].items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        # Summary
        if "summary" in data:
            lines.append("## Summary\n")
            lines.append(data["summary"])
            lines.append("")

        # Sections
        if "sections" in data:
            for section in data["sections"]:
                lines.append(f"## {section['title']}\n")
                if "content" in section:
                    lines.append(section["content"])
                if "items" in section:
                    for item in section["items"]:
                        lines.append(f"- {item}")
                lines.append("")

        # Tables
        if "tables" in data:
            for table in data["tables"]:
                lines.append(f"### {table.get('title', 'Table')}\n")
                lines.append(self._format_table(table))
                lines.append("")

        return "\n".join(lines)

    def _format_table(self, table: Dict[str, Any]) -> str:
        """Format table as Markdown"""
        if "headers" not in table or "rows" not in table:
            return ""

        lines = []
        headers = table["headers"]
        rows = table["rows"]

        # Header row
        lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        # Separator
        lines.append("|" + "|".join([" --- " for _ in headers]) + "|")
        # Data rows
        for row in rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

        return "\n".join(lines)


class JSONFormatter(OutputFormatter):
    """Format reports as JSON"""

    def __init__(self, indent: int = 2):
        """
        Args:
            indent: Indentation level for pretty printing
        """
        self.indent = indent

    def format(self, data: Dict[str, Any]) -> str:
        """Format data as JSON"""
        return json.dumps(data, indent=self.indent, default=str)


class CSVFormatter(OutputFormatter):
    """Format reports as CSV"""

    def format(self, data: Dict[str, Any]) -> str:
        """
        Format data as CSV.

        Expects data to have a 'table' key with 'headers' and 'rows'.
        """
        if "table" not in data:
            # If no table, try to create one from flat data
            return self._format_dict_as_csv(data)

        table = data["table"]
        if "headers" not in table or "rows" not in table:
            return ""

        output = StringIO()
        writer = csv.writer(output)

        # Write headers
        writer.writerow(table["headers"])

        # Write rows
        for row in table["rows"]:
            writer.writerow(row)

        return output.getvalue()

    def _format_dict_as_csv(self, data: Dict[str, Any]) -> str:
        """Convert dictionary to CSV format"""
        output = StringIO()
        writer = csv.writer(output)

        # Headers
        writer.writerow(["Key", "Value"])

        # Data
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            writer.writerow([key, value])

        return output.getvalue()


class FormatterFactory:
    """Factory for creating formatters"""

    _formatters = {
        "markdown": MarkdownFormatter,
        "md": MarkdownFormatter,
        "json": JSONFormatter,
        "csv": CSVFormatter
    }

    @classmethod
    def create(cls, format: str, **kwargs) -> OutputFormatter:
        """
        Create formatter instance.

        Args:
            format: Format name (markdown, json, csv)
            **kwargs: Additional arguments for formatter

        Returns:
            OutputFormatter instance

        Raises:
            ValueError: If format not supported
        """
        format_lower = format.lower()
        if format_lower not in cls._formatters:
            raise ValueError(
                f"Unsupported format: {format}. "
                f"Supported: {list(cls._formatters.keys())}"
            )

        return cls._formatters[format_lower](**kwargs)

    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """Get list of supported formats"""
        return list(cls._formatters.keys())
