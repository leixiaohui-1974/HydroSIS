"""Minimal response classes used by the shimmed FastAPI implementation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HTMLResponse:
    """Simple wrapper carrying HTML content."""

    content: str

    def __iter__(self):  # pragma: no cover - unused in tests
        yield self.content


__all__ = ["HTMLResponse"]
