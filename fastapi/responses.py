"""Minimal response classes used by the shimmed FastAPI implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass
class HTMLResponse:
    """Simple wrapper carrying HTML content."""

    content: str

    def __iter__(self):  # pragma: no cover - unused in tests
        yield self.content


@dataclass
class StreamingResponse:
    """Very small streaming response shim used for server-sent events in tests."""

    body: Iterable[str]
    media_type: str = "text/plain"

    def __iter__(self) -> Iterator[str]:
        yield from self.body


__all__ = ["HTMLResponse", "StreamingResponse"]
