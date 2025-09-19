"""Tiny synchronous test client emulating fastapi.testclient behaviour."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .app import FastAPI


@dataclass
class _ClientResponse:
    status_code: int
    data: Any

    def json(self) -> Any:
        return self.data

    @property
    def text(self) -> str:
        if isinstance(self.data, str):
            return self.data
        return json.dumps(self.data, ensure_ascii=False)


class TestClient:
    """Provide a minimal requests-like interface for tests."""

    __test__ = False

    def __init__(self, app: FastAPI) -> None:
        self._app = app

    def get(self, path: str) -> _ClientResponse:
        response = self._app.handle_request("GET", path)
        return _ClientResponse(status_code=response.status_code, data=response.json())

    def post(self, path: str, json: Optional[Mapping[str, Any]] = None) -> _ClientResponse:
        response = self._app.handle_request("POST", path, body=dict(json or {}))
        return _ClientResponse(status_code=response.status_code, data=response.json())

    def put(self, path: str, json: Optional[Mapping[str, Any]] = None) -> _ClientResponse:
        response = self._app.handle_request("PUT", path, body=dict(json or {}))
        return _ClientResponse(status_code=response.status_code, data=response.json())

    def delete(self, path: str) -> _ClientResponse:
        response = self._app.handle_request("DELETE", path)
        return _ClientResponse(status_code=response.status_code, data=response.json())


__all__ = ["TestClient"]
