"""Static file mount placeholder used to mirror FastAPI's API."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class StaticFiles:
    """Store the directory path for mounted static assets."""

    directory: str

    def __post_init__(self) -> None:  # pragma: no cover - defensive
        Path(self.directory).resolve()


__all__ = ["StaticFiles"]
