"""Minimal FastAPI-compatible shim for testing without external dependency."""
from __future__ import annotations

from .app import FastAPI, HTTPException, Response

__all__ = ["FastAPI", "HTTPException", "Response"]
